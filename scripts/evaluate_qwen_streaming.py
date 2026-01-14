#!/usr/bin/env python3
"""
Qwen 2.5 72B streaming evaluation script via OpenRouter.

This script provides structured streaming evaluation with proper rate limiting,
cost estimation, and metadata preservation using the OpenRouter API.

Key features:
- OpenRouter API integration with OpenAI-compatible client
- Structured JSON response parsing
- Incremental saving with resume capability
- Cost estimation and validation
- Enhanced metadata preservation
- Multiple prompt templates (AITA, firstperson, thirdperson)

Usage:
    python evaluate_qwen_streaming.py input.csv --output results.csv
    python evaluate_qwen_streaming.py input.csv --dry-run  # Cost estimation only
    python evaluate_qwen_streaming.py input.csv --perturbation-type none --sample-size 10
"""

import argparse
import os
import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
import logging
from datetime import datetime
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from llm_evaluations_everyday_dilemmas.evaluation_base import BaseEvaluator, EvaluationConfig, suggest_output_filename
from llm_evaluations_everyday_dilemmas.cost_calculator import CostCalculator
from llm_evaluations_everyday_dilemmas.result_matcher import EnhancedResultMatcher
from llm_evaluations_everyday_dilemmas.evaluation_types import standardize_judgment

# OpenAI client for OpenRouter
try:
    from openai import OpenAI
except ImportError:
    print("Error: OpenAI library not found. Install with: pip install openai")
    sys.exit(1)


class QwenStreamingEvaluator(BaseEvaluator):
    """
    Qwen 2.5 72B streaming evaluator via OpenRouter with structured output.

    Processes scenarios one at a time with comprehensive error handling,
    incremental saving, and resume capability.
    """

    def __init__(self, config: EvaluationConfig, api_key: str):
        super().__init__(config, api_key)

        # Configure OpenRouter client
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            default_headers={
                "HTTP-Referer": "https://github.com/your-username/llm_evaluations_everyday_dilemmas",
                "X-Title": "AITA Moral Dilemmas Evaluation"
            }
        )

        self.cost_calculator = CostCalculator()
        self.result_matcher = EnhancedResultMatcher()

        # Track rate limiting (OpenRouter has generous limits but we'll be conservative)
        self.last_request_time = 0
        self.min_request_interval = 0.1  # 100ms between requests

        # Thread-safety for parallelization
        self.save_lock = Lock()           # Protect CSV writes
        self.rate_limit_lock = Lock()     # Protect rate limiting
        self.max_workers = 10             # Concurrent API calls

    def estimate_cost(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Estimate cost for Qwen streaming processing."""
        num_scenarios = len(df)
        total_requests = num_scenarios * 1  # Single run per scenario

        # Get perturbation type from data
        perturbation_types = df['perturbation_type'].unique()
        if len(perturbation_types) > 1:
            self.logger.warning(f"Multiple perturbation types found: {perturbation_types}")
            self.logger.warning("This script expects files with a single perturbation type")

        # Sample some prompts for better token estimation
        sample_texts = []
        if not df.empty:
            sample_rows = df.head(min(5, len(df)))
            for _, row in sample_rows.iterrows():
                prompt = self.build_evaluation_prompt(row)
                sample_texts.append(prompt)

        cost_info = self.cost_calculator.estimate_dataset_cost(
            df=df,
            model=self.config.model,
            num_runs=1,
            is_batch=False,
            sample_texts=sample_texts
        )

        # Add time estimation
        estimated_time_per_request = 3  # seconds per request
        total_time = (total_requests * estimated_time_per_request) / 60  # minutes
        cost_info['estimated_duration_minutes'] = total_time

        self.cost_calculator.print_cost_summary(cost_info)
        print(f"Estimated duration: {total_time:.1f} minutes")

        # Add perturbation info
        print(f"\nPerturbation type: {perturbation_types[0] if len(perturbation_types) == 1 else 'MIXED'}")
        print(f"Scenarios: {num_scenarios}")

        return cost_info

    def evaluate_single_scenario(self, row: pd.Series, scenario_idx: int, run_number: int = 1) -> Dict[str, Any]:
        """
        Evaluate a single scenario with error handling and retries.

        Args:
            row: DataFrame row containing scenario data
            scenario_idx: Index of scenario for progress tracking
            run_number: Run number for 3-fold evaluations

        Returns:
            Result dictionary with evaluation and metadata
        """
        # Create scenario metadata
        metadata = self.create_scenario_metadata(row)
        metadata['scenario_index'] = scenario_idx
        metadata['run_number'] = run_number

        # Build prompt using the framework's prompt builder
        prompt = self.build_evaluation_prompt(row)

        max_retries = self.config.max_retries
        for attempt in range(max_retries):
            try:
                # Thread-safe rate limiting
                with self.rate_limit_lock:
                    time_since_last = time.time() - self.last_request_time
                    if time_since_last < self.min_request_interval:
                        time.sleep(self.min_request_interval - time_since_last)
                    self.last_request_time = time.time()  # Update timestamp inside lock

                # Make API call with JSON mode
                response = self.client.chat.completions.create(
                    model=self.config.model,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.config.temperature,
                    max_tokens=500,
                    response_format={"type": "json_object"}
                )

                # Extract response
                raw_response = response.choices[0].message.content

                # Parse JSON response
                evaluation_data = self._parse_json_response(raw_response)

                # Standardize judgment
                standardized = standardize_judgment(evaluation_data.get('judgment', 'ERROR'))

                # Combine metadata with evaluation results
                result = {
                    **metadata,
                    f'judgment_qwen25_run_{run_number}': evaluation_data.get('judgment', 'ERROR'),
                    f'explanation_qwen25_run_{run_number}': evaluation_data.get('explanation', ''),
                    f'standardized_judgment_qwen25_run_{run_number}': standardized,
                    'temperature_used': self.config.temperature,
                    'raw_response': raw_response,
                    'error': None
                }

                return result

            except Exception as e:
                self.logger.warning(f"Attempt {attempt + 1}/{max_retries} failed: {str(e)[:100]}")
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    time.sleep(wait_time)
                else:
                    # Final attempt failed - return error result
                    result = {
                        **metadata,
                        f'judgment_qwen25_run_{run_number}': 'ERROR',
                        f'explanation_qwen25_run_{run_number}': '',
                        f'standardized_judgment_qwen25_run_{run_number}': 'Error',
                        'temperature_used': self.config.temperature,
                        'raw_response': '',
                        'error': str(e)
                    }
                    return result

        # Should never reach here, but return error just in case
        return {
            **metadata,
            f'judgment_qwen25_run_{run_number}': 'ERROR',
            f'explanation_qwen25_run_{run_number}': '',
            f'standardized_judgment_qwen25_run_{run_number}': 'Error',
            'temperature_used': self.config.temperature,
            'error': 'Max retries exceeded'
        }

    def _parse_json_response(self, response_text: str) -> Dict[str, str]:
        """Parse JSON response from Qwen, with fallback to text extraction."""
        import re

        text = response_text.strip()

        # Remove markdown code blocks if present
        text = re.sub(r'```json\s*', '', text)
        text = re.sub(r'```\s*', '', text)

        # Try to find and parse JSON object
        start = text.find('{')
        end = text.rfind('}') + 1

        if start != -1 and end > start:
            json_str = text[start:end]
            try:
                parsed = json.loads(json_str)
                if 'judgment' in parsed and 'explanation' in parsed:
                    # Clean up explanation to remove judgment prefix
                    explanation = self._clean_explanation(parsed['explanation'], parsed['judgment'])
                    return {
                        'judgment': parsed['judgment'],
                        'explanation': explanation
                    }
            except json.JSONDecodeError:
                pass

        # Fallback: try to extract judgment from text
        judgment_patterns = [
            r'^(YTA|NTA|ESH|NAH|INFO)\b',
            r'\b(YTA|NTA|ESH|NAH|INFO)\b',
        ]

        judgment = 'ERROR'
        for pattern in judgment_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                judgment = match.group(1).upper()
                break

        # Clean the explanation
        explanation = self._clean_explanation(text, judgment)

        return {
            'judgment': judgment,
            'explanation': explanation
        }

    def _clean_explanation(self, explanation: str, judgment: str) -> str:
        """Remove judgment label from the beginning of explanation."""
        import re

        # Don't clean "Both" as it's a legitimate English word
        if judgment.upper() == 'BOTH':
            return explanation.strip()

        # Remove patterns like "**YTA**", "**NTA - Not the Asshole**", "YTA:", etc.
        patterns = [
            r'^\*\*' + judgment + r'(\s*[-–—]\s*[^*]+)?\*\*\s*',  # **YTA** or **NTA - Not the Asshole**
            r'^\*\*' + judgment + r'\*\*\s*',  # **YTA**
            r'^' + judgment + r'\s*[-–—:]\s*',  # YTA: or NTA -
            r'^' + judgment + r'\s+',  # YTA followed by space
        ]

        cleaned = explanation
        for pattern in patterns:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
            if cleaned != explanation:
                break  # Stop after first match

        return cleaned.strip()

    def run_evaluation(self, df: pd.DataFrame, run_number: int = 1) -> List[Dict[str, Any]]:
        """
        Run parallel evaluation on dataset with progress tracking and incremental saving.

        Args:
            df: DataFrame with scenarios to evaluate
            run_number: Run number for 3-fold evaluations

        Returns:
            List of evaluation results
        """
        total = len(df)
        output_file = Path(self.config.output_file)

        # Pre-allocate results list for ordered storage
        results = [None] * total
        start_idx = 0
        completed_count = 0

        # Check for existing results to resume from
        if self.config.resume and output_file.exists():
            try:
                existing_df = pd.read_csv(output_file)
                judgment_col = f'judgment_qwen25_run_{run_number}'

                if judgment_col in existing_df.columns:
                    # Load existing results
                    for idx, row in existing_df.iterrows():
                        if pd.notna(row.get(judgment_col)):
                            results[idx] = row.to_dict()
                            completed_count += 1

                    if completed_count > 0:
                        self.logger.info(f"Resuming from existing file with {completed_count} completed evaluations")
                        start_idx = completed_count
            except Exception as e:
                self.logger.warning(f"Could not resume from existing file: {e}")

        errors = 0
        last_save = time.time()

        self.logger.info(f"Starting parallel evaluation with {self.max_workers} workers")
        self.logger.info(f"Processing from index {start_idx}/{total}")

        # Create thread pool executor
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit tasks for scenarios that need evaluation
            futures_to_idx = {}
            for idx in range(start_idx, total):
                # Skip if already completed
                if results[idx] is not None:
                    continue

                row = df.iloc[idx]
                future = executor.submit(
                    self.evaluate_single_scenario,
                    row, idx, run_number
                )
                futures_to_idx[future] = idx

            # Process results as they complete
            with tqdm(total=total, initial=completed_count, desc=f"Run {run_number}") as pbar:
                for future in as_completed(futures_to_idx):
                    idx = futures_to_idx[future]

                    try:
                        result = future.result()
                        results[idx] = result

                        if result.get('error'):
                            errors += 1

                        completed_count += 1
                        pbar.update(1)

                        # Periodic saving (thread-safe)
                        if completed_count % self.config.save_interval == 0 or \
                           (time.time() - last_save) > 120:
                            with self.save_lock:
                                self._save_results(results, output_file)
                                last_save = time.time()

                    except Exception as e:
                        # Worker exception handling
                        self.logger.error(f"Worker failed for scenario {idx}: {e}")
                        results[idx] = {
                            'scenario_id': df.iloc[idx].get('scenario_id', f'unknown_{idx}'),
                            f'judgment_qwen25_run_{run_number}': 'ERROR',
                            f'explanation_qwen25_run_{run_number}': '',
                            f'standardized_judgment_qwen25_run_{run_number}': 'Error',
                            'error': str(e)
                        }
                        errors += 1
                        completed_count += 1
                        pbar.update(1)

        # Final save (thread-safe)
        with self.save_lock:
            self._save_results(results, output_file)

        self.logger.info(f"\nCompleted {total} evaluations")
        self.logger.info(f"Errors: {errors} ({errors/total*100:.1f}%)")
        self.logger.info(f"Results saved to: {output_file}")

        return results

    def _save_results(self, results: List[Dict[str, Any]], output_file: Path):
        """Save results to CSV, filtering out None values and maintaining order."""
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Filter out None values (incomplete results during parallel execution)
        completed_results = [r for r in results if r is not None]

        if not completed_results:
            return  # Nothing to save yet

        # Convert to DataFrame and sort by scenario_id to maintain consistent order
        results_df = pd.DataFrame(completed_results)

        # Sort by scenario_id if available, otherwise by index
        if 'scenario_id' in results_df.columns:
            results_df = results_df.sort_values('scenario_id')
        elif 'scenario_index' in results_df.columns:
            results_df = results_df.sort_values('scenario_index')

        results_df.to_csv(output_file, index=False)

    def execute(self, input_file: str, run_number: int = 1) -> List[Dict[str, Any]]:
        """
        Main execution method.

        Args:
            input_file: Path to input CSV
            run_number: Run number for 3-fold evaluations

        Returns:
            List of evaluation results
        """
        # Load and validate data
        df = self.load_and_validate_data(input_file)

        # Estimate cost
        if self.config.dry_run:
            self.estimate_cost(df)
            return []

        # Run evaluation
        results = self.run_evaluation(df, run_number)

        return results


def create_argument_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Qwen 2.5 72B streaming evaluation via OpenRouter",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic evaluation
  python evaluate_qwen_streaming.py input.csv --output results.csv

  # Dry run for cost estimation
  python evaluate_qwen_streaming.py input.csv --dry-run

  # Filter to specific perturbation type
  python evaluate_qwen_streaming.py input.csv --perturbation-type none

  # 3-fold evaluation (run 3 times)
  python evaluate_qwen_streaming.py input.csv --runs 3
        """
    )

    # Input/output options
    parser.add_argument("input_file", help="Input CSV file with perturbed scenarios")
    parser.add_argument("--output", "--output-file",
                       help="Output CSV file for results (auto-generated if not specified)")

    # Model options
    parser.add_argument("--model", default="qwen/qwen-2.5-72b-instruct",
                       help="OpenRouter model to use (default: qwen/qwen-2.5-72b-instruct)")
    parser.add_argument("--temperature", type=float, default=0.4,
                       help="Model temperature (default: 0.4)")

    # Evaluation options
    parser.add_argument("--runs", type=int, default=1,
                       help="Number of evaluation runs (default: 1, use 3 for base)")

    # Filtering options
    parser.add_argument("--perturbation-type",
                       help="Filter to specific perturbation type")
    parser.add_argument("--sample-size", type=int,
                       help="Maximum number of scenarios to process")
    parser.add_argument("--offset", type=int, default=0,
                       help="Number of scenarios to skip (default: 0)")

    # Control options
    parser.add_argument("--dry-run", action="store_true",
                       help="Show cost estimation only, don't run evaluation")
    parser.add_argument("--no-resume", action="store_true",
                       help="Don't resume from existing results")
    parser.add_argument("--save-interval", type=int, default=10,
                       help="Save results every N scenarios (default: 10)")
    parser.add_argument("--workers", type=int, default=10,
                       help="Number of concurrent workers for parallel evaluation (default: 10)")

    return parser


def main():
    parser = create_argument_parser()
    args = parser.parse_args()

    # Check for API key
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("Error: Set OPENROUTER_API_KEY environment variable")
        print("Add to ~/.zshrc: export OPENROUTER_API_KEY='your-key-here'")
        return 1

    # Validate input file
    if not Path(args.input_file).exists():
        print(f"Error: Input file {args.input_file} not found")
        return 1

    # Create output file path if not specified
    if not args.output:
        args.output = suggest_output_filename(args.input_file, "qwen", args.model)

    # Create configuration
    config = EvaluationConfig(
        provider="qwen",
        model=args.model,
        temperature=args.temperature,
        perturbation_type=args.perturbation_type,
        sample_size=args.sample_size,
        offset=args.offset,
        output_file=args.output,
        dry_run=args.dry_run,
        resume=not args.no_resume,
        save_interval=args.save_interval,
        input_file=args.input_file
    )

    # Create evaluator
    evaluator = QwenStreamingEvaluator(config, api_key)
    evaluator.max_workers = args.workers  # Set number of concurrent workers

    try:
        # Run evaluations (potentially multiple runs for 3-fold)
        for run_num in range(1, args.runs + 1):
            if args.runs > 1:
                print(f"\n{'=' * 80}")
                print(f"RUN {run_num} of {args.runs}")
                print(f"{'=' * 80}\n")

                # For multiple runs, adjust output filename
                if run_num > 1:
                    base_output = Path(args.output)
                    config.output_file = str(base_output.parent / f"{base_output.stem}_run{run_num}{base_output.suffix}")

            results = evaluator.execute(args.input_file, run_number=run_num)

            if not args.dry_run and results:
                print(f"\n✓ Run {run_num} completed successfully!")

        return 0

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
