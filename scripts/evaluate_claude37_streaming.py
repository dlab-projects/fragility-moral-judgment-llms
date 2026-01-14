#!/usr/bin/env python3
"""
Claude 3.7 Sonnet streaming evaluation script via OpenRouter.

Pricing: $3/M input, $15/M output tokens
Estimated cost for 6,384 scenarios: ~$46

Usage:
    python evaluate_claude37_streaming.py input.csv --output results.csv
    python evaluate_claude37_streaming.py input.csv --dry-run  # Cost estimation only
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


class Claude37StreamingEvaluator(BaseEvaluator):
    """Claude 3.7 Sonnet streaming evaluator via OpenRouter with structured output."""

    def __init__(self, config: EvaluationConfig, api_key: str):
        super().__init__(config, api_key)

        # Configure OpenRouter client
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            default_headers={
                "HTTP-Referer": "https://github.com/llm-evaluations",
                "X-Title": "AITA Moral Dilemmas Evaluation"
            }
        )

        self.cost_calculator = CostCalculator()
        self.result_matcher = EnhancedResultMatcher()

        # Track rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.1  # 100ms between requests

        # Thread-safety for parallelization
        self.save_lock = Lock()
        self.rate_limit_lock = Lock()
        self.max_workers = 10  # Concurrent API calls

    def estimate_cost(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Estimate cost for Claude 3.7 Sonnet streaming processing."""
        num_scenarios = len(df)

        # Token estimates
        avg_scenario_tokens = 600
        prompt_tokens = 300
        output_tokens = 300

        total_input_tokens = (avg_scenario_tokens + prompt_tokens) * num_scenarios
        total_output_tokens = output_tokens * num_scenarios

        # Pricing (OpenRouter)
        input_cost = (total_input_tokens / 1_000_000) * 3.0
        output_cost = (total_output_tokens / 1_000_000) * 15.0
        total_cost = input_cost + output_cost

        return {
            'num_scenarios': num_scenarios,
            'total_requests': num_scenarios,
            'estimated_input_tokens': total_input_tokens,
            'estimated_output_tokens': total_output_tokens,
            'estimated_input_cost': input_cost,
            'estimated_output_cost': output_cost,
            'estimated_cost': total_cost,
            'estimated_total_cost': total_cost,
            'model': self.config.model
        }

    def evaluate_single_scenario(
        self,
        row: pd.Series,
        index: int,
        run_number: int = 1
    ) -> Dict[str, Any]:
        """Evaluate a single scenario using Claude 3.7 Sonnet."""

        # Create scenario metadata
        metadata = self.create_scenario_metadata(row)
        metadata['scenario_index'] = index
        metadata['run_number'] = run_number

        # Get perturbation-specific prompt
        prompt = self.build_evaluation_prompt(row)

        # Rate limiting (thread-safe)
        with self.rate_limit_lock:
            time_since_last = time.time() - self.last_request_time
            if time_since_last < self.min_request_interval:
                time.sleep(self.min_request_interval - time_since_last)
            self.last_request_time = time.time()

        try:
            response = self.client.chat.completions.create(
                model="anthropic/claude-3.7-sonnet",
                messages=[{"role": "user", "content": prompt}],
                temperature=self.config.temperature,
                max_tokens=600,
                response_format={"type": "json_object"}
            )

            # Parse response
            content = response.choices[0].message.content
            result = json.loads(content)

            # Validate required fields
            if 'judgment' not in result or 'explanation' not in result:
                return {
                    **metadata,
                    'judgment': 'ERROR',
                    'standardized_judgment': 'Error',
                    'explanation': '',
                    'error': f"Missing required fields in response: {list(result.keys())}",
                    'raw_response': content
                }

            # Standardize judgment
            raw_judgment = result['judgment']
            standardized = standardize_judgment(raw_judgment)

            # Create result
            return {
                **metadata,
                'judgment': raw_judgment,
                'standardized_judgment': standardized,
                'explanation': result['explanation'],
                'error': None,
                'raw_response': content
            }

        except json.JSONDecodeError as e:
            return {
                **metadata,
                'judgment': 'ERROR',
                'standardized_judgment': 'Error',
                'explanation': '',
                'error': f"JSON parse error: {str(e)}",
                'raw_response': ''
            }
        except Exception as e:
            return {
                **metadata,
                'judgment': 'ERROR',
                'standardized_judgment': 'Error',
                'explanation': '',
                'error': str(e),
                'raw_response': ''
            }

    def run_evaluation(self, df: pd.DataFrame, run_number: int = 1) -> List[Dict[str, Any]]:
        """Run evaluation with parallel processing."""

        # Pre-allocate results list for ordered storage
        results = [None] * len(df)
        errors = 0
        completed_count = 0
        last_save = time.time()

        # Check for resume
        start_idx = 0
        if self.config.output_file and Path(self.config.output_file).exists():
            existing_df = pd.read_csv(self.config.output_file)
            completed_count = len(existing_df)
            self.logger.info(f"Resuming from row {completed_count}")
            start_idx = completed_count

            # Load existing results
            for idx in range(completed_count):
                if idx < len(existing_df):
                    results[idx] = existing_df.iloc[idx].to_dict()

        # Create thread pool executor
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            futures_to_idx = {}
            for idx in range(start_idx, len(df)):
                if results[idx] is not None:
                    completed_count += 1
                    continue

                row = df.iloc[idx]
                future = executor.submit(
                    self.evaluate_single_scenario,
                    row, idx, run_number
                )
                futures_to_idx[future] = idx

            # Process results as they complete
            with tqdm(total=len(df), initial=completed_count,
                     desc=f"Claude 3.7 Run {run_number}") as pbar:
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
                                self._save_results(results, self.config.output_file)
                                last_save = time.time()

                    except Exception as e:
                        self.logger.error(f"Worker failed for scenario {idx}: {e}")
                        # Create error result
                        row = df.iloc[idx]
                        metadata = self.create_scenario_metadata(row)
                        results[idx] = {
                            **metadata,
                            'scenario_index': idx,
                            'run_number': run_number,
                            'judgment': 'ERROR',
                            'standardized_judgment': 'Error',
                            'explanation': '',
                            'error': str(e),
                            'raw_response': ''
                        }
                        errors += 1
                        pbar.update(1)

        # Final save
        with self.save_lock:
            self._save_results(results, self.config.output_file)

        self.logger.info(f"Completed {len(results)} evaluations ({errors} errors)")
        return results

    def _save_results(self, results: List[Dict], output_file: str):
        """Save results to CSV (thread-safe)."""
        valid_results = [r for r in results if r is not None]
        if valid_results:
            df_results = pd.DataFrame(valid_results)
            df_results.to_csv(output_file, index=False)


def main():
    parser = argparse.ArgumentParser(description="Evaluate scenarios with Claude 3.7 Sonnet")
    parser.add_argument('input_file', help='Input CSV file with scenarios')
    parser.add_argument('--output', help='Output CSV file (auto-generated if not specified)')
    parser.add_argument('--dry-run', action='store_true', help='Estimate cost without running')
    parser.add_argument('--temperature', type=float, default=0.4, help='Sampling temperature')
    parser.add_argument('--save-interval', type=int, default=100, help='Save every N evaluations')
    parser.add_argument('--workers', type=int, default=10, help='Number of parallel workers')
    parser.add_argument('--sample-size', type=int, help='Limit to first N scenarios (for testing)')

    args = parser.parse_args()

    # Check API key
    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY not set")
        print("Set it with: export OPENROUTER_API_KEY='your-key-here'")
        return 1

    # Load input
    df = pd.read_csv(args.input_file)
    print(f"Loaded {len(df)} scenarios from {args.input_file}")

    if args.sample_size:
        df = df.head(args.sample_size)
        print(f"Limited to first {len(df)} scenarios")

    # Create config
    if not args.output:
        args.output = suggest_output_filename(args.input_file, 'anthropic', 'claude37')

    config = EvaluationConfig(
        provider='anthropic',
        model='anthropic/claude-3.7-sonnet',
        temperature=args.temperature,
        save_interval=args.save_interval,
        output_file=args.output,
        dry_run=args.dry_run,
        resume=True,
        input_file=args.input_file
    )

    # Initialize evaluator
    evaluator = Claude37StreamingEvaluator(config, api_key)
    evaluator.max_workers = args.workers

    # Estimate cost
    cost_info = evaluator.estimate_cost(df)
    print("\n" + "="*70)
    print("COST ESTIMATION")
    print("="*70)
    print(f"Scenarios: {cost_info['num_scenarios']:,}")
    print(f"Input tokens:  {cost_info['estimated_input_tokens']:,} @ $3/M = ${cost_info['estimated_input_cost']:.2f}")
    print(f"Output tokens: {cost_info['estimated_output_tokens']:,} @ $15/M = ${cost_info['estimated_output_cost']:.2f}")
    print(f"\nTOTAL ESTIMATED COST: ${cost_info['estimated_total_cost']:.2f}")
    print("="*70)

    if args.dry_run:
        print("\nDry run mode - exiting without evaluation")
        return 0

    # Confirm
    response = input("\nProceed with evaluation? (yes/no): ")
    if response.lower() not in ['yes', 'y']:
        print("Aborted")
        return 0

    # Run evaluation
    print(f"\nStarting evaluation...")
    print(f"Output: {args.output}")

    results = evaluator.run_evaluation(df, run_number=1)

    print(f"\n✅ Complete! Results saved to: {args.output}")
    return 0


if __name__ == "__main__":
    exit(main())
