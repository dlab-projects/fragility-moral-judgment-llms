#!/usr/bin/env python3
"""
Google Gemini streaming evaluation script for ethical dilemmas.

This script provides structured streaming evaluation with proper rate limiting,
cost estimation, and metadata preservation. Unlike batch APIs, this processes
scenarios one at a time with comprehensive error handling and resume capability.

Key features:
- Intelligent rate limiting for Google API
- Structured JSON response parsing
- Incremental saving with resume capability
- Cost estimation and validation
- Enhanced metadata preservation

Usage:
    python evaluate_google_streaming.py input.csv --output results.csv --model gemini-2.5-flash
    python evaluate_google_streaming.py input.csv --dry-run  # Cost estimation only
    python evaluate_google_streaming.py input.csv --perturbation-type none --sample-size 10
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
import threading
from collections import deque
from pydantic import BaseModel

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from llm_evaluations_everyday_dilemmas.evaluation_base import BaseEvaluator, EvaluationConfig, suggest_output_filename
from llm_evaluations_everyday_dilemmas.cost_calculator import CostCalculator
from llm_evaluations_everyday_dilemmas.result_matcher import EnhancedResultMatcher

# Google imports
try:
    from google import genai
except ImportError:
    print("Error: Google GenAI library not found. Install with: pip install google-genai")
    sys.exit(1)


class EvaluationResponseModel(BaseModel):
    """Structured response model for Google evaluation."""
    judgment: str
    explanation: str


class GoogleRateLimiter:
    """
    Intelligent rate limiter specifically for Google Gemini API.
    
    Handles both requests per minute (RPM) and tokens per minute (TPM) limits
    with adaptive backoff for rate limit errors.
    """
    
    def __init__(self, model: str, rpm_limit: int = 60, tpm_limit: int = 60000):
        self.model = model
        self.rpm_limit = rpm_limit
        self.tpm_limit = tpm_limit
        
        # Apply safety factors
        self.effective_rpm = int(rpm_limit * 0.8)  # 80% of limit for safety
        self.effective_tpm = int(tpm_limit * 0.8)
        
        # Track requests and tokens
        self.request_times = deque()
        self.token_usage = deque()  # (timestamp, tokens) tuples
        self.lock = threading.Lock()
        
        logging.info(f"Google rate limiter: {self.effective_rpm} RPM, {self.effective_tpm:,} TPM")
    
    def wait_if_needed(self, estimated_tokens: int = 1000):
        """Wait if necessary to stay within rate limits."""
        with self.lock:
            now = time.time()
            
            # Clean old entries (older than 1 minute)
            while self.request_times and now - self.request_times[0] > 60:
                self.request_times.popleft()
            
            while self.token_usage and now - self.token_usage[0][0] > 60:
                self.token_usage.popleft()
            
            # Check RPM limit
            if len(self.request_times) >= self.effective_rpm:
                wait_time = 60 - (now - self.request_times[0])
                if wait_time > 0:
                    logging.info(f"RPM limit reached, waiting {wait_time:.1f}s")
                    time.sleep(wait_time)
            
            # Check TPM limit
            current_tokens = sum(tokens for _, tokens in self.token_usage)
            if current_tokens + estimated_tokens > self.effective_tpm:
                wait_time = 60 - (now - self.token_usage[0][0])
                if wait_time > 0:
                    logging.info(f"TPM limit would be exceeded, waiting {wait_time:.1f}s")
                    time.sleep(wait_time)
            
            # Record this request
            self.request_times.append(now)
            self.token_usage.append((now, estimated_tokens))


class GoogleStreamingEvaluator(BaseEvaluator):
    """
    Google Gemini streaming evaluator with structured output and rate limiting.
    
    Processes scenarios one at a time with comprehensive error handling,
    incremental saving, and resume capability.
    """
    
    def __init__(self, config: EvaluationConfig, api_key: str):
        super().__init__(config, api_key)
        
        # Configure Google GenAI client
        self.client = genai.Client(api_key=api_key)
        
        self.cost_calculator = CostCalculator()
        self.result_matcher = EnhancedResultMatcher()
        
        # Setup rate limiter
        self.rate_limiter = GoogleRateLimiter(
            model=config.model,
            rpm_limit=60,  # Conservative default
            tpm_limit=60000
        )
        
        # Generation config for structured responses
        self.generation_config = {
            "temperature": config.temperature,
            "max_output_tokens": 3000,  # Increased to handle verbose responses with structured output
            "response_mime_type": "application/json",
            "response_schema": EvaluationResponseModel
        }
    
    def estimate_cost(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Estimate cost for Google streaming processing."""
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
        estimated_time_per_request = 3  # seconds per request including rate limiting
        total_time = (total_requests * estimated_time_per_request) / 60  # minutes
        cost_info['estimated_duration_minutes'] = total_time
        
        self.cost_calculator.print_cost_summary(cost_info)
        print(f"Estimated duration: {total_time:.1f} minutes")
        
        # Add perturbation info
        print(f"\nPerturbation type: {perturbation_types[0] if len(perturbation_types) == 1 else 'MIXED'}")
        print(f"Scenarios: {num_scenarios}")
        
        return cost_info
    
    def evaluate_single_scenario(self, row: pd.Series, scenario_idx: int) -> Dict[str, Any]:
        """
        Evaluate a single scenario with error handling and retries.
        
        Args:
            row: DataFrame row containing scenario data
            scenario_idx: Index of scenario for progress tracking
            
        Returns:
            Result dictionary with evaluation and metadata
        """
        # Create scenario metadata
        metadata = self.create_scenario_metadata(row)
        metadata['scenario_index'] = scenario_idx
        
        # Build prompt
        prompt = self.build_evaluation_prompt(row)
        
        # Estimate tokens for rate limiting
        estimated_tokens = self.cost_calculator.estimate_tokens(prompt)
        
        max_retries = self.config.max_retries
        for attempt in range(max_retries):
            try:
                # Apply rate limiting
                self.rate_limiter.wait_if_needed(estimated_tokens)
                
                # Make API call
                response = self.client.models.generate_content(
                    model=self.config.model,
                    contents=prompt,
                    config=self.generation_config
                )
                
                # Debug: log response structure
                self.logger.debug(f"Response type: {type(response)}")
                if hasattr(response, 'parsed'):
                    self.logger.debug(f"Has parsed: {response.parsed is not None}")
                if hasattr(response, 'text'):
                    self.logger.debug(f"Has text: {response.text is not None}")
                    if response.text:
                        self.logger.debug(f"Text length: {len(response.text)} chars")
                # Check for MAX_TOKENS finish reason
                if hasattr(response, 'candidates') and response.candidates:
                    for candidate in response.candidates:
                        if hasattr(candidate, 'finish_reason') and str(candidate.finish_reason) == 'MAX_TOKENS':
                            self.logger.warning("Response hit MAX_TOKENS limit - increasing max_output_tokens may help")
                
                # Parse response
                if response and response.parsed:
                    # The parsed response should be our Pydantic model
                    try:
                        parsed = response.parsed
                        evaluation_data = {
                            'judgment': parsed.judgment,
                            'explanation': parsed.explanation
                        }
                    except AttributeError as e:
                        self.logger.error(f"Failed to access parsed attributes: {e}")
                        self.logger.error(f"Parsed object type: {type(response.parsed)}")
                        self.logger.error(f"Parsed object: {response.parsed}")
                        raise
                elif response and hasattr(response, 'text') and response.text:
                    # Fallback to text parsing
                    self.logger.debug(f"Falling back to text parsing. Response text: {response.text[:200]}")
                    try:
                        evaluation_data = json.loads(response.text)
                        
                        # Validate required fields
                        if 'judgment' not in evaluation_data:
                            evaluation_data['judgment'] = 'ERROR'
                            evaluation_data['explanation'] = 'Missing judgment in response'
                        
                        if 'explanation' not in evaluation_data:
                            evaluation_data['explanation'] = 'No explanation provided'
                        
                        # If we successfully parsed JSON from text, log this
                        self.logger.debug("Successfully parsed JSON from text response")
                            
                    except json.JSONDecodeError as e:
                        # Check if it looks like truncated JSON
                        if response.text and '"explanation"' in response.text and not response.text.rstrip().endswith('}'):
                            self.logger.warning("JSON appears truncated - likely hit token limit")
                            # Try to extract what we can
                            import re
                            judgment_match = re.search(r'"judgment":\s*"([^"]+)"', response.text)
                            if judgment_match:
                                evaluation_data = {
                                    'judgment': judgment_match.group(1),
                                    'explanation': 'Response truncated due to token limit',
                                    'raw_response': response.text
                                }
                            else:
                                evaluation_data = {
                                    'judgment': 'ERROR',
                                    'explanation': f'Truncated JSON response. Error: {str(e)}',
                                    'raw_response': response.text
                                }
                        else:
                            evaluation_data = {
                                'judgment': 'ERROR',
                                'explanation': f'JSON decode error: {str(e)}. Response text: {response.text[:200]}...',
                                'raw_response': response.text
                            }
                else:
                    # Log more details about what we got
                    self.logger.error(f"No valid response from API. Response: {response}")
                    # Raise exception to trigger retry
                    raise ValueError("No valid response from API - no parsed data or text content")
                
                # Process response
                result = self.process_evaluation_response(evaluation_data, metadata)
                result['attempt_number'] = attempt + 1
                
                return result
                
            except Exception as e:
                self.logger.warning(f"Attempt {attempt + 1}/{max_retries} failed for scenario {scenario_idx}: {e}")
                
                if attempt == max_retries - 1:
                    # Final attempt failed
                    error_result = metadata.copy()
                    error_result.update({
                        'judgment': 'ERROR',
                        'explanation': f'All {max_retries} attempts failed. Last error: {str(e)}',
                        'error_type': type(e).__name__,
                        'attempt_number': max_retries,
                    })
                    return self.process_evaluation_response(error_result, metadata)
                
                # Wait before retry
                wait_time = (attempt + 1) * 2  # Exponential backoff
                self.logger.info(f"Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
    
    def run_evaluation(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Run streaming evaluation with incremental saving and resume capability.
        
        Args:
            df: DataFrame with perturbed scenarios
            
        Returns:
            List of evaluation results
        """
        # Setup resume capability
        existing_results, start_idx = self.setup_resume_capability(self.config.output_file)
        all_results = existing_results.copy()
        
        # Track scenarios to process
        scenarios_to_process = df.iloc[start_idx:] if start_idx > 0 else df
        total_scenarios = len(df)
        
        if start_idx > 0:
            self.logger.info(f"Resuming from scenario {start_idx + 1}/{total_scenarios}")
        
        # Create progress bar
        progress_bar = tqdm(
            scenarios_to_process.iterrows(),
            total=total_scenarios,
            desc="Evaluating scenarios",
            initial=start_idx
        )
        
        processed_count = 0
        error_count = 0
        
        try:
            for row_idx, row in progress_bar:
                scenario_idx = start_idx + processed_count
                
                # Update progress bar description
                progress_bar.set_description(f"Evaluating scenario {scenario_idx + 1}/{total_scenarios}")
                
                # Evaluate scenario
                result = self.evaluate_single_scenario(row, scenario_idx)
                all_results.append(result)
                
                # Track errors
                if result.get('judgment') == 'ERROR':
                    error_count += 1
                
                processed_count += 1
                
                # Incremental save every N scenarios
                if processed_count % self.config.save_interval == 0:
                    self.save_results_incrementally(all_results, self.config.output_file, is_final=False)
                    
                    # Update progress with error rate
                    error_rate = error_count / processed_count * 100
                    progress_bar.set_postfix({
                        'errors': f'{error_count}/{processed_count} ({error_rate:.1f}%)',
                        'saved': f'{len(all_results)} results'
                    })
        
        except KeyboardInterrupt:
            self.logger.info("Evaluation interrupted by user")
            # Save current progress
            self.save_results_incrementally(all_results, self.config.output_file, is_final=False)
            raise
        
        finally:
            progress_bar.close()
        
        # Final save
        self.save_results_incrementally(all_results, self.config.output_file, is_final=True)
        
        # Summary statistics
        self.logger.info(f"Evaluation complete: {len(all_results)} total results")
        self.logger.info(f"New evaluations: {processed_count}")
        self.logger.info(f"Error rate: {error_count}/{processed_count} ({error_count/processed_count*100:.1f}%)")
        
        return all_results


def create_argument_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Google Gemini streaming evaluation with structured output",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run streaming evaluation with cost estimation
  python evaluate_google_streaming.py input.csv --output results.csv --model gemini-2.0-flash
  
  # Dry run to see cost estimate only
  python evaluate_google_streaming.py input.csv --output results.csv --dry-run
  
  # Filter to specific perturbation type
  python evaluate_google_streaming.py input.csv --output results.csv --perturbation-type none
  
  # Resume interrupted evaluation
  python evaluate_google_streaming.py input.csv --output results.csv  # Automatically resumes
  
  # Process subset with custom save interval
  python evaluate_google_streaming.py input.csv --output results.csv --sample-size 100 --save-interval 5
        """
    )
    
    # Input/output options
    parser.add_argument("input_file", help="Input CSV file with perturbed scenarios")
    parser.add_argument("--output", "--output-file", required=False, 
                       help="Output CSV file for results (auto-generated if not specified)")
    
    # Model options
    parser.add_argument("--model", default="gemini-2.5-flash", 
                       help="Google model to use (default: gemini-2.5-flash)")
    parser.add_argument("--temperature", type=float, default=0.4,
                       help="Model temperature (default: 0.4)")
    parser.add_argument("--max-retries", type=int, default=3,
                       help="Maximum retries per scenario (default: 3)")
    
    # Filtering options
    parser.add_argument("--perturbation-type", 
                       help="Filter to specific perturbation type (e.g., none, firstperson)")
    parser.add_argument("--sample-size", type=int,
                       help="Maximum number of scenarios to process")
    parser.add_argument("--offset", type=int, default=0,
                       help="Number of scenarios to skip (default: 0)")
    
    # Processing options
    parser.add_argument("--save-interval", type=int, default=10,
                       help="Save results every N scenarios (default: 10)")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show cost estimation only, don't run evaluation")
    parser.add_argument("--no-resume", action="store_true",
                       help="Don't resume from existing temp files")
    
    return parser


def main():
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Check for API key
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Error: Set GOOGLE_API_KEY environment variable")
        return
    
    # Validate input file
    if not Path(args.input_file).exists():
        print(f"Error: Input file {args.input_file} not found")
        return
    
    # Create output file path if not specified
    if not args.output:
        args.output = suggest_output_filename(args.input_file, "google", args.model)
    
    # Create configuration
    config = EvaluationConfig(
        provider="google",
        model=args.model,
        temperature=args.temperature,
        max_retries=args.max_retries,
        perturbation_type=args.perturbation_type,
        sample_size=args.sample_size,
        offset=args.offset,
        output_file=args.output,
        dry_run=args.dry_run,
        resume=not args.no_resume,
        save_interval=args.save_interval
    )
    
    # Create evaluator and run
    evaluator = GoogleStreamingEvaluator(config, api_key)
    
    try:
        results = evaluator.execute(args.input_file)
        
        if not args.dry_run:
            print(f"\n✅ Google streaming evaluation completed!")
            print(f"Results saved to: {args.output}")
            
            if results:
                # Summary statistics
                total_results = len(results)
                error_count = sum(1 for r in results if r.get('judgment') == 'ERROR')
                success_rate = (total_results - error_count) / total_results * 100
                
                print(f"Total results: {total_results}")
                print(f"Success rate: {success_rate:.1f}% ({total_results - error_count}/{total_results})")
                
                # Judgment distribution (excluding errors)
                successful_results = [r for r in results if r.get('judgment') != 'ERROR']
                if successful_results:
                    judgments = [r['judgment'] for r in successful_results]
                    judgment_counts = pd.Series(judgments).value_counts()
                    print("\nJudgment distribution:")
                    for judgment, count in judgment_counts.items():
                        percentage = count / len(successful_results) * 100
                        print(f"  {judgment}: {count} ({percentage:.1f}%)")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()