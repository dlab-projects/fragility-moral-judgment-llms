#!/usr/bin/env python3
"""
Explanation-First Evaluation Script via OpenRouter

Tests whether verdict-sympathy correlation persists when explanation comes BEFORE verdict.
Standard approach: Verdict → Explanation
This approach: Explanation → Verdict

Uses stratified sampling across all perturbation types to ensure balanced representation.

Supported models via OpenRouter:
- openai/gpt-4.1
- anthropic/claude-3.7-sonnet
- google/gemini-2.5-flash
- deepseek/deepseek-chat
- qwen/qwen-2.5-72b-instruct

Usage:
    # Dry run with cost estimation
    python evaluate_explanation_first.py --sample-size 100 --dry-run

    # Run for all 4 models with 100 samples per perturbation (excluding Gemini)
    python evaluate_explanation_first.py --sample-size 100

    # Run for specific model only
    python evaluate_explanation_first.py --model openai/gpt-4.1 --sample-size 50
"""

import argparse
import os
import sys
import json
import time
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from llm_evaluations_everyday_dilemmas.config_explanation_first import EXPLANATION_FIRST_TEMPLATES
from llm_evaluations_everyday_dilemmas.evaluation_types import standardize_judgment

# OpenAI client for OpenRouter
try:
    from openai import OpenAI
except ImportError:
    print("Error: OpenAI library not found. Install with: pip install openai")
    sys.exit(1)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Model configurations (excluding Gemini - not in main study)
MODELS = {
    'openai/gpt-4.1': {'name': 'gpt41', 'input_price': 2.5, 'output_price': 10.0},
    'anthropic/claude-3.7-sonnet': {'name': 'claude37', 'input_price': 3.0, 'output_price': 15.0},
    'deepseek/deepseek-chat': {'name': 'deepseek', 'input_price': 0.14, 'output_price': 0.28},
    'qwen/qwen-2.5-72b-instruct': {'name': 'qwen25', 'input_price': 0.35, 'output_price': 0.35}
    # 'google/gemini-2.5-flash': excluded - not in main study
}

# Aliases for common short names (maps to full OpenRouter paths)
MODEL_ALIASES = {
    'gpt-4.1': 'openai/gpt-4.1',
    'gpt41': 'openai/gpt-4.1',
    'claude-3.7-sonnet': 'anthropic/claude-3.7-sonnet',
    'claude37': 'anthropic/claude-3.7-sonnet',
    'deepseek-chat': 'deepseek/deepseek-chat',
    'deepseek': 'deepseek/deepseek-chat',
    'qwen-2.5-72b-instruct': 'qwen/qwen-2.5-72b-instruct',
    'qwen25': 'qwen/qwen-2.5-72b-instruct',
}

def resolve_model_name(model: str) -> str:
    """Resolve short model name to full OpenRouter path."""
    if model in MODELS:
        return model
    if model in MODEL_ALIASES:
        return MODEL_ALIASES[model]
    return model  # Return as-is, validation will catch invalid names

# Perturbation categories for stratification
PERTURBATION_TYPES = [
    'none',
    # Robustness
    'add_extraneous_detail', 'change_trivial_detail', 'remove_sentence',
    # Presentation
    'firstperson_atfault', 'thirdperson',
    # Psychological
    'push_nta_self_justifying', 'push_nta_social_proof', 'push_nta_victim_pattern',
    'push_yta_pattern_admission', 'push_yta_self_condemning', 'push_yta_social_proof'
]


class ExplanationFirstEvaluator:
    """Evaluator for explanation-first approach across multiple models."""

    def __init__(self, api_key: str, model: str, temperature: float = 0.4):
        self.api_key = api_key
        self.model = model
        self.model_name = MODELS[model]['name']
        self.temperature = temperature

        # OpenRouter client
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            default_headers={
                "HTTP-Referer": "https://github.com/llm-evaluations",
                "X-Title": "AITA Explanation-First Evaluation"
            }
        )

        # Rate limiting and concurrency
        self.last_request_time = 0
        self.min_request_interval = 0.1  # 100ms between requests
        self.rate_limit_lock = Lock()
        self.save_lock = Lock()
        self.max_workers = 10

        logger.info(f"Initialized evaluator for {model} (alias: {self.model_name})")

    def build_explanation_first_prompt(self, dilemma_text: str, perturbation_type: str) -> str:
        """Build explanation-first prompt based on perturbation type."""
        # Determine which template to use
        if perturbation_type in ['firstperson_atfault', 'firstperson']:
            template_type = 'firstperson'
        elif perturbation_type == 'thirdperson':
            template_type = 'thirdperson'
        else:
            # Default to AITA template for baseline and other perturbations
            template_type = 'aita'

        template = EXPLANATION_FIRST_TEMPLATES[template_type]

        # Add JSON instruction
        json_instruction = """
Please format your response as a JSON object with exactly two fields:
{
  "explanation": "Your detailed one-paragraph analysis here",
  "judgment": "Your categorical verdict here"
}
""".strip()

        return f"{template.format(dilemma_text=dilemma_text)}\n\n{json_instruction}"

    def evaluate_single_scenario(
        self,
        row: pd.Series,
        index: int
    ) -> Dict[str, Any]:
        """Evaluate a single scenario with explanation-first approach."""

        # Extract metadata
        metadata = {
            'id': row.get('id'),
            'title': row.get('title'),
            'perturbation_type': row.get('perturbation_type'),
            'perturbed_text': row.get('perturbed_text'),
            'base_verdict': row.get('base_verdict'),
            'model': self.model_name,
            'model_fullname': self.model,
            'scenario_index': index,
            'evaluation_mode': 'explanation_first'
        }

        # Build prompt
        prompt = self.build_explanation_first_prompt(
            str(row['perturbed_text']),
            str(row['perturbation_type'])
        )

        # Rate limiting
        with self.rate_limit_lock:
            time_since_last = time.time() - self.last_request_time
            if time_since_last < self.min_request_interval:
                time.sleep(self.min_request_interval - time_since_last)
            self.last_request_time = time.time()

        # API call with retries
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=800,
                    response_format={"type": "json_object"}
                )

                # Check for truncation
                finish_reason = response.choices[0].finish_reason
                if finish_reason == 'length':
                    logger.warning(f"Response truncated (finish_reason=length), retry {attempt + 1}/{max_retries}")
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)
                        continue

                # Parse response
                content = response.choices[0].message.content
                result = json.loads(content)

                # Validate
                if 'explanation' not in result or 'judgment' not in result:
                    if attempt < max_retries - 1:
                        logger.warning(f"Missing fields, retry {attempt + 1}/{max_retries}")
                        time.sleep(2 ** attempt)
                        continue
                    else:
                        return {
                            **metadata,
                            'explanation': '',
                            'judgment': 'ERROR',
                            'standardized_judgment': 'Error',
                            'error': f'Missing fields: {list(result.keys())}',
                            'raw_response': content
                        }

                # Standardize judgment
                raw_judgment = result['judgment'].strip()
                standardized = standardize_judgment(raw_judgment)

                return {
                    **metadata,
                    'explanation': result['explanation'].strip(),
                    'judgment': raw_judgment,
                    'standardized_judgment': standardized,
                    'error': None,
                    'raw_response': content
                }

            except json.JSONDecodeError as e:
                if attempt < max_retries - 1:
                    logger.warning(f"JSON decode error, retry {attempt + 1}/{max_retries}")
                    time.sleep(2 ** attempt)
                else:
                    return {
                        **metadata,
                        'explanation': '',
                        'judgment': 'ERROR',
                        'standardized_judgment': 'Error',
                        'error': f'JSON decode error: {str(e)}',
                        'raw_response': ''
                    }

            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"API error, retry {attempt + 1}/{max_retries}: {str(e)[:100]}")
                    time.sleep(2 ** attempt)
                else:
                    return {
                        **metadata,
                        'explanation': '',
                        'judgment': 'ERROR',
                        'standardized_judgment': 'Error',
                        'error': str(e),
                        'raw_response': ''
                    }

    def run_evaluation(
        self,
        df: pd.DataFrame,
        output_file: str,
        resume: bool = True
    ) -> List[Dict[str, Any]]:
        """Run parallel evaluation with progress tracking."""

        # Setup resume
        results = [None] * len(df)
        start_idx = 0

        if resume and Path(output_file).exists():
            existing_df = pd.read_csv(output_file)
            for idx, row in existing_df.iterrows():
                if pd.notna(row.get('judgment')):
                    results[idx] = row.to_dict()
                    start_idx = idx + 1
            if start_idx > 0:
                logger.info(f"Resuming from row {start_idx}")

        errors = 0
        last_save = time.time()

        # Parallel execution
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures_to_idx = {}
            for idx in range(start_idx, len(df)):
                if results[idx] is not None:
                    continue

                row = df.iloc[idx]
                future = executor.submit(self.evaluate_single_scenario, row, idx)
                futures_to_idx[future] = idx

            # Process results
            with tqdm(total=len(df), initial=start_idx, desc=f"{self.model_name}") as pbar:
                for future in as_completed(futures_to_idx):
                    idx = futures_to_idx[future]

                    try:
                        result = future.result()
                        results[idx] = result

                        if result.get('error'):
                            errors += 1

                        pbar.update(1)

                        # Periodic save
                        if (idx - start_idx + 1) % 50 == 0 or (time.time() - last_save) > 120:
                            with self.save_lock:
                                self._save_results(results, output_file)
                                last_save = time.time()

                    except Exception as e:
                        logger.error(f"Worker failed for scenario {idx}: {e}")
                        results[idx] = {
                            'scenario_index': idx,
                            'model': self.model_name,
                            'error': str(e),
                            'judgment': 'ERROR',
                            'standardized_judgment': 'Error',
                            'explanation': ''
                        }
                        errors += 1
                        pbar.update(1)

        # Final save
        with self.save_lock:
            self._save_results(results, output_file)

        logger.info(f"Completed {len(results)} evaluations ({errors} errors)")
        return results

    def _save_results(self, results: List[Dict], output_file: str):
        """Save results to CSV."""
        valid_results = [r for r in results if r is not None]
        if valid_results:
            df = pd.DataFrame(valid_results)
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_file, index=False)


def stratified_sample(
    master_file: str,
    sample_size_per_perturbation: int,
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Create stratified sample across all perturbation types.

    Args:
        master_file: Path to master dataset
        sample_size_per_perturbation: Number of samples per perturbation type
        random_seed: Random seed for reproducibility

    Returns:
        Stratified sample DataFrame
    """
    logger.info("=" * 80)
    logger.info("STRATIFIED SAMPLING")
    logger.info("=" * 80)

    # Load master dataset
    if master_file.endswith('.parquet'):
        df = pd.read_parquet(master_file)
    else:
        df = pd.read_csv(master_file)

    logger.info(f"Loaded master dataset: {len(df):,} rows")

    # Filter to single model to avoid duplicates (we just need the scenarios)
    df_single_model = df[df['model'] == 'gpt41'].copy()
    logger.info(f"Filtered to GPT-4.1 scenarios: {len(df_single_model):,} rows")

    # Stratified sampling
    samples = []
    np.random.seed(random_seed)

    for pert_type in PERTURBATION_TYPES:
        subset = df_single_model[df_single_model['perturbation_type'] == pert_type]

        if len(subset) == 0:
            logger.warning(f"  ⚠️  No data for {pert_type}")
            continue

        # Sample
        n_sample = min(sample_size_per_perturbation, len(subset))
        sampled = subset.sample(n=n_sample, random_state=random_seed)

        logger.info(f"  {pert_type}: {n_sample:,} samples (from {len(subset):,} available)")
        samples.append(sampled)

    # Combine
    df_sampled = pd.concat(samples, ignore_index=True)

    logger.info(f"\nTotal stratified sample: {len(df_sampled):,} scenarios")
    logger.info(f"Perturbation types: {df_sampled['perturbation_type'].nunique()}")
    logger.info("=" * 80)

    return df_sampled


def estimate_cost(
    num_scenarios: int,
    models: List[str]
) -> Dict[str, Any]:
    """Estimate cost for explanation-first evaluation."""

    logger.info("\n" + "=" * 80)
    logger.info("COST ESTIMATION")
    logger.info("=" * 80)

    # Token estimates
    avg_input_tokens = 900  # dilemma + explanation-first prompt
    avg_output_tokens = 350  # explanation + judgment (slightly more than standard)

    total_cost = 0
    cost_breakdown = {}

    for model in models:
        config = MODELS[model]
        input_price_per_m = config['input_price']
        output_price_per_m = config['output_price']

        total_input_tokens = num_scenarios * avg_input_tokens
        total_output_tokens = num_scenarios * avg_output_tokens

        input_cost = (total_input_tokens / 1_000_000) * input_price_per_m
        output_cost = (total_output_tokens / 1_000_000) * output_price_per_m
        model_cost = input_cost + output_cost

        cost_breakdown[model] = {
            'scenarios': num_scenarios,
            'input_tokens': total_input_tokens,
            'output_tokens': total_output_tokens,
            'input_cost': input_cost,
            'output_cost': output_cost,
            'total_cost': model_cost
        }

        total_cost += model_cost

        logger.info(f"\n{model} ({config['name']}):")
        logger.info(f"  Input:  {total_input_tokens:,} tokens @ ${input_price_per_m}/M = ${input_cost:.2f}")
        logger.info(f"  Output: {total_output_tokens:,} tokens @ ${output_price_per_m}/M = ${output_cost:.2f}")
        logger.info(f"  Subtotal: ${model_cost:.2f}")

    logger.info(f"\n{'=' * 80}")
    logger.info(f"TOTAL ESTIMATED COST: ${total_cost:.2f}")
    logger.info(f"Total scenarios: {num_scenarios:,}")
    logger.info(f"Total evaluations: {num_scenarios * len(models):,}")
    logger.info("=" * 80)

    return {
        'total_cost': total_cost,
        'breakdown': cost_breakdown,
        'num_scenarios': num_scenarios,
        'num_models': len(models)
    }


def main():
    parser = argparse.ArgumentParser(
        description='Explanation-First Evaluation via OpenRouter',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run with 100 samples per perturbation
  python evaluate_explanation_first.py --sample-size 100 --dry-run

  # Run all 4 models with 100 samples each (excluding Gemini)
  python evaluate_explanation_first.py --sample-size 100

  # Run single model only
  python evaluate_explanation_first.py --model openai/gpt-4.1 --sample-size 50

  # Custom master file and output directory
  python evaluate_explanation_first.py \
      --master custom_master.parquet \
      --output-dir custom_results/ \
      --sample-size 100
        """
    )

    parser.add_argument('input_file', nargs='?', default=None,
                       help='Input CSV file with scenarios (skips stratified sampling)')
    parser.add_argument('--master',
                       default='final_results/parquet/master_final.parquet',
                       help='Master dataset file for stratified sampling (ignored if input_file provided)')
    parser.add_argument('--sample-size', type=int, default=None,
                       help='Number of samples per perturbation type (required unless input_file provided)')
    parser.add_argument('--model', type=str,
                       help='Run single model only (e.g., openai/gpt-4.1)')
    parser.add_argument('--output', '--output-file',
                       help='Output CSV file (for input_file mode)')
    parser.add_argument('--output-dir',
                       default='results/explanation_first/',
                       help='Output directory for results (for stratified sampling mode)')
    parser.add_argument('--temperature', type=float, default=0.4,
                       help='Model temperature (default: 0.4)')
    parser.add_argument('--workers', type=int, default=10,
                       help='Number of parallel workers (default: 10)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Cost estimation only, do not run evaluations')
    parser.add_argument('--random-seed', type=int, default=42,
                       help='Random seed for stratified sampling')

    args = parser.parse_args()

    # Check API key
    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY not set")
        print("Set it with: export OPENROUTER_API_KEY='your-key-here'")
        return 1

    # Determine mode: input_file (direct) or master (stratified sampling)
    if args.input_file:
        # Direct input file mode
        if not Path(args.input_file).exists():
            print(f"ERROR: Input file not found: {args.input_file}")
            return 1

        logger.info(f"Loading input file: {args.input_file}")
        df_sample = pd.read_csv(args.input_file)

        # Apply sample-size limit if specified
        if args.sample_size and args.sample_size < len(df_sample):
            df_sample = df_sample.head(args.sample_size)
            logger.info(f"Limited to {args.sample_size} samples")

        logger.info(f"Loaded {len(df_sample)} scenarios from input file")
    else:
        # Stratified sampling mode - requires sample-size
        if args.sample_size is None:
            print("ERROR: --sample-size required when not using input_file")
            return 1

        if not Path(args.master).exists():
            print(f"ERROR: Master file not found: {args.master}")
            return 1

        # Create stratified sample
        df_sample = stratified_sample(
            args.master,
            args.sample_size,
            args.random_seed
        )

    # Determine which models to run
    if args.model:
        resolved_model = resolve_model_name(args.model)
        if resolved_model not in MODELS:
            print(f"ERROR: Unknown model {args.model}")
            print(f"Available: {', '.join(MODELS.keys())}")
            print(f"Aliases: {', '.join(MODEL_ALIASES.keys())}")
            return 1
        models_to_run = [resolved_model]
    else:
        models_to_run = list(MODELS.keys())

    logger.info(f"Models to evaluate: {', '.join([MODELS[m]['name'] for m in models_to_run])}")

    # Estimate cost
    cost_info = estimate_cost(len(df_sample), models_to_run)

    if args.dry_run:
        logger.info("\n🔍 DRY RUN MODE - Exiting without evaluation")
        return 0

    # Confirm (skip if using input_file - assumed to be intentional)
    if not args.input_file:
        print("\n" + "=" * 80)
        response = input("Proceed with evaluation? (yes/no): ")
        if response.lower() not in ['yes', 'y']:
            print("Aborted")
            return 0

    # Run evaluations
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for model in models_to_run:
        model_name = MODELS[model]['name']

        # Determine output file
        if args.output:
            output_file = Path(args.output)
        else:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_dir / f"explanation_first_{model_name}_{timestamp}.csv"

        output_file.parent.mkdir(parents=True, exist_ok=True)

        logger.info("\n" + "=" * 80)
        logger.info(f"EVALUATING: {model} ({model_name})")
        logger.info("=" * 80)

        evaluator = ExplanationFirstEvaluator(
            api_key=api_key,
            model=model,
            temperature=args.temperature
        )
        evaluator.max_workers = args.workers

        results = evaluator.run_evaluation(
            df=df_sample,
            output_file=str(output_file),
            resume=True
        )

        logger.info(f"✅ Saved: {output_file}")

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("ALL EVALUATIONS COMPLETE!")
    logger.info("=" * 80)
    if args.output:
        logger.info(f"Results saved to: {args.output}")
    else:
        logger.info(f"Results saved to: {args.output_dir}/")
    logger.info(f"Total evaluations: {len(df_sample) * len(models_to_run):,}")
    logger.info(f"Total cost: ~${cost_info['total_cost']:.2f}")

    return 0


if __name__ == "__main__":
    exit(main())
