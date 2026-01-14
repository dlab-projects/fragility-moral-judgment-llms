#!/usr/bin/env python3
"""
Anthropic Batch API evaluation script.

This script specifically addresses the perturbation filtering issue where 
Anthropic batch included unexpected perturbation types (add_emotional_language 
when only 'none' was specified), causing evaluation contamination.

- Strict perturbation type validation before batch submission
- Enhanced custom_id format with content hashing for robust matching
- Proper Anthropic batch JSON formatting
- Mandatory cost estimation and user confirmation

Usage:
    python scripts/evaluation/evaluate_anthropic_batch.py data/perturbed_split/change_trivial_detail.csv --model claude-3-7-sonnet-latest
    python scripts/evaluation/evaluate_anthropic_batch.py input.csv --dry-run  # Cost estimation only
    python scripts/evaluation/evaluate_anthropic_batch.py input.csv --perturbation-type none --sample-size 10
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

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from llm_evaluations_everyday_dilemmas.evaluation_base import BaseEvaluator, EvaluationConfig, suggest_output_filename
from llm_evaluations_everyday_dilemmas.cost_calculator import CostCalculator
from llm_evaluations_everyday_dilemmas.result_matcher import EnhancedResultMatcher, create_enhanced_batch_metadata

# Anthropic imports
try:
    import anthropic
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request
    from anthropic import RateLimitError as AnthropicRateLimitError
except ImportError:
    print("Error: Anthropic library not found. Install with: pip install anthropic")
    sys.exit(1)


class AnthropicBatchEvaluator(BaseEvaluator):
    """
    Anthropic Batch API evaluator with strict perturbation filtering.
    
    Solves the critical issue where unexpected perturbation types were
    included in batch processing, contaminating evaluation results.
    """
    
    def __init__(self, config: EvaluationConfig, api_key: str):
        super().__init__(config, api_key)
        self.client = anthropic.Anthropic(api_key=api_key)
        self.cost_calculator = CostCalculator()
        self.result_matcher = EnhancedResultMatcher()
        
        # Track batch processing
        self.batch_metadata: List[Dict[str, Any]] = []
        self.submitted_batch_id: Optional[str] = None
    
    def validate_single_perturbation(self, df: pd.DataFrame) -> bool:
        """
        CRITICAL: Validate that only one perturbation type exists in the data.
        
        This validation prevents the costly mistake where wrong perturbation 
        types are included in the batch, contaminating results.
        """
        actual_types = list(df['perturbation_type'].unique())
        
        if len(actual_types) == 1:
            self.logger.info(f"✅ Single perturbation validation passed: '{actual_types[0]}'")
            return True
        else:
            self.logger.error(
                f"🚨 CRITICAL VALIDATION FAILED: Multiple perturbation types found: {sorted(actual_types)}"
            )
            self.logger.error(
                "This would contaminate your batch with mixed perturbation types!"
            )
            self.logger.error(
                "Please use separate CSV files for each perturbation type."
            )
            return False
    
    def estimate_cost(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Estimate cost for Anthropic batch processing."""
        num_scenarios = len(df)
        total_requests = num_scenarios * 1  # Single run per scenario for batch
        
        # Get perturbation type from data
        perturbation_types = df['perturbation_type'].unique()
        
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
            is_batch=True,
            sample_texts=sample_texts
        )
        
        self.cost_calculator.print_cost_summary(cost_info)
        
        # Add perturbation info
        print(f"\nPerturbation type: {perturbation_types[0] if len(perturbation_types) == 1 else 'MIXED - ERROR'}")
        print(f"Scenarios: {num_scenarios}")
        
        return cost_info
    
    def create_batch_request(self, df: pd.DataFrame) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Create Anthropic batch request with enhanced metadata preservation.
        
        Returns:
            Tuple of (batch_requests, batch_metadata)
        """
        batch_requests = []
        self.batch_metadata = create_enhanced_batch_metadata(
            df=df,
            exp_prefix=f"a{datetime.now().strftime('%m%d%H%M')}_",
            num_runs=1
        )
        
        for metadata in self.batch_metadata:
            row_data = metadata['row_data']
            custom_id = metadata['custom_id']
            
            # Build evaluation prompt
            prompt = self.build_evaluation_prompt(pd.Series(row_data))
            
            # Create Anthropic batch request format
            batch_request = {
                "custom_id": custom_id,
                "params": {
                    "model": self.config.model,
                    "max_tokens": 500,
                    "temperature": self.config.temperature,
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                }
            }
            
            batch_requests.append(batch_request)
        
        self.logger.info(f"Created {len(batch_requests)} Anthropic batch requests with enhanced metadata")
        return batch_requests, self.batch_metadata
    
    def submit_batch(self, batch_requests: List[Dict[str, Any]]) -> str:
        """
        Submit batch to Anthropic and return batch ID.
        
        Args:
            batch_requests: List of Anthropic batch request dictionaries
            
        Returns:
            Batch ID string
        """
        self.logger.info("Submitting batch to Anthropic...")
        
        # Convert to Anthropic Request format
        anthropic_requests = []
        for request in batch_requests:
            anthropic_request = Request(
                custom_id=request["custom_id"],
                params=MessageCreateParamsNonStreaming(**request["params"])
            )
            anthropic_requests.append(anthropic_request)
        
        # Create batch
        batch = self.client.messages.batches.create(
            requests=anthropic_requests
        )
        
        self.submitted_batch_id = batch.id
        self.logger.info(f"Batch submitted successfully! Batch ID: {batch.id}")
        
        return batch.id
    
    def monitor_batch_status(self, batch_id: str) -> Dict[str, Any]:
        """
        Monitor batch status and provide updates.
        
        Args:
            batch_id: Anthropic batch ID
            
        Returns:
            Final batch status information
        """
        self.logger.info(f"Monitoring Anthropic batch {batch_id}...")
        
        start_time = time.time()
        last_status = None
        
        while True:
            try:
                batch = self.client.messages.batches.retrieve(batch_id)
                current_status = batch.processing_status
                
                if current_status != last_status:
                    elapsed = time.time() - start_time
                    self.logger.info(f"Batch status: {current_status} (elapsed: {elapsed:.1f}s)")
                    
                    if hasattr(batch, 'request_counts') and batch.request_counts:
                        counts = batch.request_counts
                        total = counts.processing + counts.succeeded + counts.errored + counts.canceled + counts.expired
                        completed = counts.succeeded + counts.errored
                        
                        if total > 0:
                            progress = completed / total * 100
                            self.logger.info(f"  Progress: {completed}/{total} completed ({progress:.1f}%)")
                            self.logger.info(f"  Breakdown: {counts.succeeded} succeeded, {counts.errored} errored, {counts.processing} processing")
                    
                    last_status = current_status
                
                if current_status in ['ended']:
                    return {
                        'batch_id': batch_id,
                        'status': current_status,
                        'batch_object': batch,
                        'elapsed_time': time.time() - start_time
                    }
                
                # Wait before next check
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Error checking batch status: {e}")
                time.sleep(60)  # Wait longer on error
    
    def download_and_process_results(self, batch_id: str, original_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Download batch results and match them to original scenarios.
        
        Args:
            batch_id: Anthropic batch ID
            original_df: Original DataFrame with perturbed scenarios
            
        Returns:
            List of matched results with preserved metadata
        """
        # Get batch object
        batch = self.client.messages.batches.retrieve(batch_id)
        
        if batch.processing_status != 'ended':
            raise ValueError(f"Batch not completed. Status: {batch.processing_status}")
        
        # Get batch results
        self.logger.info("Downloading batch results...")
        results_response = self.client.messages.batches.results(batch_id)
        
        # Parse results
        batch_results = []
        for result in results_response:
            result_data = {
                'custom_id': result.custom_id,
                'result_type': result.result.type
            }
            
            if result.result.type == 'succeeded':
                message = result.result.message
                content = message.content[0].text if message.content else ""
                
                # Parse JSON response if possible, otherwise parse as plain text
                try:
                    parsed_content = json.loads(content)
                    result_data.update(parsed_content)
                except json.JSONDecodeError:
                    # Try to repair malformed JSON before falling back to text parsing
                    repaired_json = self._attempt_json_repair(content)
                    if repaired_json:
                        result_data.update(repaired_json)
                    else:
                        # Parse plain text response
                        parsed_text = self._parse_anthropic_text_response(content)
                        result_data.update(parsed_text)
            else:
                # Handle error case
                error_info = result.result.error if hasattr(result.result, 'error') else {}
                result_data.update({
                    'judgment': 'ERROR',
                    'explanation': f'Batch request failed: {error_info}',
                    'error_type': error_info.get('type', 'unknown'),
                    'error_message': error_info.get('message', 'No error message')
                })
            
            # Add batch metadata
            result_data.update({
                'batch_custom_id': result.custom_id,
                'batch_status': 'completed',
                'anthropic_batch_id': batch_id
            })
            
            batch_results.append(result_data)
        
        self.logger.info(f"Downloaded {len(batch_results)} batch results")
        
        # Match results to original scenarios using enhanced matcher
        matched_results, unmatched_results = self.result_matcher.process_batch_results(
            batch_results=batch_results,
            original_df=original_df
        )
        
        # Save unmatched results for debugging
        if unmatched_results:
            unmatched_file = Path(self.config.output_file).with_suffix('.unmatched.json')
            self.result_matcher.save_unmatched_results(unmatched_results, str(unmatched_file))
        
        # Validate matching completeness
        validation_report = self.result_matcher.validate_matching_completeness(
            original_df=original_df,
            matched_results=matched_results,
            expected_runs_per_scenario=1
        )
        
        self.result_matcher.print_validation_report(validation_report)
        
        if not validation_report['is_complete']:
            self.logger.warning("Batch result matching was incomplete!")
        
        return matched_results
    
    def _attempt_json_repair(self, response_text: str) -> Optional[Dict[str, Any]]:
        """
        Attempt to repair malformed JSON responses from Anthropic.
        
        Specifically handles cases where the response is valid JSON but missing
        the closing curly bracket.
        
        Args:
            response_text: The raw response text from Anthropic
            
        Returns:
            Parsed JSON dict if repair successful, None otherwise
        """
        text = response_text.strip()
        
        # Check if this looks like incomplete JSON (starts with { but fails to parse)
        if not text.startswith('{'):
            return None
            
        # Try adding missing closing bracket
        if not text.endswith('}'):
            repaired_text = text + '}'
            try:
                parsed = json.loads(repaired_text)
                # Validate that it has the expected structure
                if isinstance(parsed, dict) and 'judgment' in parsed and 'explanation' in parsed:
                    self.logger.info("Successfully repaired JSON with missing closing bracket")
                    return parsed
            except json.JSONDecodeError:
                pass
        
        # Try other common JSON repair patterns if needed
        # For now, just handle the missing bracket case
        
        return None
    
    def _parse_anthropic_text_response(self, response_text: str) -> Dict[str, str]:
        """Parse plain text response from Anthropic to extract judgment and explanation."""
        import re
        
        # Look for common judgment patterns at the beginning
        judgment_patterns = [
            r'^(YTA|NTA|ESH|NAH|INFO)\b',  # Standard AITA judgments at start
            r'\b(YTA|NTA|ESH|NAH|INFO)\b',  # Judgment anywhere in text
        ]
        
        judgment = 'ERROR'
        explanation = response_text.strip()
        
        # Try to extract judgment
        for pattern in judgment_patterns:
            match = re.search(pattern, response_text.strip(), re.IGNORECASE)
            if match:
                judgment = match.group(1).upper()
                explanation = response_text.strip()
                break
        
        # If no clear judgment found but text starts with judgment-like words
        if judgment == 'ERROR':
            first_word = response_text.strip().split()[0] if response_text.strip() else ''
            if first_word.upper() in ['YTA', 'NTA', 'ESH', 'NAH', 'INFO']:
                judgment = first_word.upper()
                explanation = response_text.strip()
        
        return {
            'judgment': judgment,
            'explanation': explanation,
            'raw_response': response_text.strip()
        }
    
    def run_evaluation(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Run Anthropic batch evaluation with strict perturbation filtering.
        
        Args:
            df: DataFrame with perturbed scenarios
            
        Returns:
            List of evaluation results with preserved metadata
        """
        # CRITICAL: Validate single perturbation type before batch submission
        if not self.validate_single_perturbation(df):
            raise ValueError(
                "🚨 CRITICAL: Multiple perturbation types detected! "
                "This would contaminate your batch with mixed perturbation types. "
                "Please use separate CSV files for each perturbation type."
            )
        
        # Create batch requests with enhanced metadata
        batch_requests, batch_metadata = self.create_batch_request(df)
        
        # Submit batch
        batch_id = self.submit_batch(batch_requests)
        
        # Save batch metadata for recovery
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        metadata_file = Path(self.config.output_file).with_suffix('.batch_metadata.json')
        with open(metadata_file, 'w') as f:
            json.dump({
                'batch_id': batch_id,
                'metadata_file': str(metadata_file),
                'config': self.config.model_dump(),
                'batch_metadata': batch_metadata,
                'submitted_at': datetime.now().isoformat(),
                'perturbation_validation': {
                    'perturbation_type_filter': self.config.perturbation_type,
                    'actual_types_in_batch': list(df['perturbation_type'].unique()),
                    'validation_passed': True
                }
            }, f, indent=2, default=str)
        
        self.logger.info(f"Batch metadata saved to {metadata_file}")
        
        print(f"\n🚀 ANTHROPIC BATCH SUBMITTED SUCCESSFULLY!")
        print(f"Batch ID: {batch_id}")
        print(f"Metadata file: {metadata_file}")
        print(f"✅ Perturbation validation: PASSED")
        if self.config.perturbation_type:
            print(f"   Only '{self.config.perturbation_type}' perturbations included")
        print(f"\nTo monitor batch status:")
        print(f"  python {__file__} --monitor-batch {batch_id}")
        print(f"\nTo download results when complete:")
        print(f"  python {__file__} --download-batch {batch_id} --metadata-file {metadata_file}")
        
        return []  # Batch mode returns empty list initially


def create_argument_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Anthropic Batch API evaluation with strict perturbation filtering",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Submit batch with cost estimation
  python evaluate_anthropic_batch.py input.csv --output results.csv --model claude-3-7-sonnet-latest
  
  # Dry run to see cost estimate only
  python evaluate_anthropic_batch.py input.csv --output results.csv --dry-run
  
  # Filter to specific perturbation type (CRITICAL for preventing contamination)
  python evaluate_anthropic_batch.py input.csv --output results.csv --perturbation-type none
  
  # Monitor existing batch
  python evaluate_anthropic_batch.py --monitor-batch batch_12345
  
  # Download completed batch results
  python evaluate_anthropic_batch.py --download-batch batch_12345 --metadata-file batch.metadata.json
        """
    )
    
    # Input/output options
    parser.add_argument("input_file", nargs="?", help="Input CSV file with perturbed scenarios")
    parser.add_argument("--output", "--output-file", required=False, 
                       help="Output CSV file for results (auto-generated if not specified)")
    
    # Model options
    parser.add_argument("--model", default="claude-3-7-sonnet-latest", 
                       help="Anthropic model to use (default: claude-3-7-sonnet-latest)")
    parser.add_argument("--temperature", type=float, default=0.4,
                       help="Model temperature (default: 0.4)")
    
    # Filtering options
    parser.add_argument("--perturbation-type", 
                       help="Filter to specific perturbation type (CRITICAL: prevents contamination)")
    parser.add_argument("--sample-size", type=int,
                       help="Maximum number of scenarios to process")
    parser.add_argument("--offset", type=int, default=0,
                       help="Number of scenarios to skip (default: 0)")
    
    # Batch management options
    parser.add_argument("--monitor-batch", 
                       help="Monitor status of existing batch ID")
    parser.add_argument("--download-batch",
                       help="Download results for completed batch ID")
    parser.add_argument("--metadata-file", 
                       help="Batch metadata file (required for --download-batch)")
    
    # Control options
    parser.add_argument("--dry-run", action="store_true",
                       help="Show cost estimation only, don't submit batch")
    parser.add_argument("--no-resume", action="store_true",
                       help="Don't resume from existing temp files")
    
    return parser


def monitor_batch_command(batch_id: str, api_key: str) -> None:
    """Monitor batch status command."""
    client = anthropic.Anthropic(api_key=api_key)
    
    print(f"Monitoring Anthropic batch: {batch_id}")
    start_time = time.time()
    last_status = None
    
    while True:
        try:
            batch = client.messages.batches.retrieve(batch_id)
            current_status = batch.processing_status
            
            if current_status != last_status:
                elapsed = time.time() - start_time
                print(f"Status: {current_status} (elapsed: {elapsed:.1f}s)")
                
                if hasattr(batch, 'request_counts') and batch.request_counts:
                    counts = batch.request_counts
                    total = counts.processing + counts.succeeded + counts.errored + counts.canceled + counts.expired
                    completed = counts.succeeded + counts.errored
                    
                    if total > 0:
                        progress = completed / total * 100
                        print(f"Progress: {completed}/{total} completed ({progress:.1f}%)")
                        print(f"Breakdown: {counts.succeeded} succeeded, {counts.errored} errored, {counts.processing} processing")
                
                last_status = current_status
            
            if current_status == 'ended':
                print(f"\n✅ Batch completed successfully!")
                print("Use --download-batch to get results.")
                break
            
            time.sleep(30)
            
        except KeyboardInterrupt:
            print("\nMonitoring interrupted by user")
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(60)


def download_batch_command(batch_id: str, metadata_file: str, api_key: str) -> None:
    """Download batch results command."""
    if not Path(metadata_file).exists():
        print(f"Error: Metadata file {metadata_file} not found")
        return
    
    # Load batch metadata
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    # Recreate config from metadata
    config_data = metadata['config']
    config = EvaluationConfig(**config_data)
    
    # Create evaluator
    evaluator = AnthropicBatchEvaluator(config, api_key)
    
    # Load original dataset for matching
    # Try to find the original input file from metadata or infer from current directory
    possible_input_files = [
        metadata.get('input_file'),  # First check if stored in metadata
        config_data.get('input_file'),
        'data/perturbed_split/none.csv',  # Common split file location
        'data/candidates_perturbed.csv',
        'candidates_perturbed.csv'
    ]
    
    # Also try to infer from output filename
    output_path = Path(config.output_file)
    if 'none' in output_path.stem:
        possible_input_files.insert(1, 'data/perturbed_split/none.csv')
    elif 'add_emotional_language' in output_path.stem:
        possible_input_files.insert(1, 'data/perturbed_split/add_emotional_language.csv')
    
    input_file = None
    for file_path in possible_input_files:
        if file_path and Path(file_path).exists():
            input_file = file_path
            break
    
    if not input_file:
        print("Error: Could not find original input file for result matching")
        print("Please ensure the original perturbed scenarios CSV is available")
        return
    
    print(f"Loading original data from: {input_file}")
    original_df = pd.read_csv(input_file)
    
    # Apply same filtering as original batch
    if config.perturbation_type:
        original_df = original_df[original_df['perturbation_type'] == config.perturbation_type]
        print(f"Filtered to perturbation_type='{config.perturbation_type}': {len(original_df)} scenarios")
    
    if config.sample_size:
        original_df = original_df.iloc[config.offset:config.offset + config.sample_size]
    
    # Download and process results
    try:
        results = evaluator.download_and_process_results(batch_id, original_df)
        
        if results:
            # Save results
            output_file = config.output_file
            results_df = pd.DataFrame(results)
            results_df.to_csv(output_file, index=False)
            
            print(f"\n✅ Successfully downloaded and processed {len(results)} results")
            print(f"Results saved to: {output_file}")
            
            # Validate perturbation types in results
            if 'perturbation_type' in results_df.columns:
                result_types = set(results_df['perturbation_type'].unique())
                expected_types = {config.perturbation_type} if config.perturbation_type else result_types
                
                if result_types == expected_types:
                    print(f"✅ Perturbation validation: Results contain only expected types: {result_types}")
                else:
                    print(f"⚠️  Perturbation validation: Unexpected types found!")
                    print(f"   Expected: {expected_types}")
                    print(f"   Found: {result_types}")
            
            # Summary statistics
            if 'judgment' in results_df.columns:
                judgment_counts = results_df['judgment'].value_counts()
                print("\nJudgment distribution:")
                for judgment, count in judgment_counts.items():
                    print(f"  {judgment}: {count}")
        else:
            print("❌ No results were successfully matched")
            
    except Exception as e:
        print(f"Error downloading batch results: {e}")


def main():
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Check for API key
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: Set ANTHROPIC_API_KEY environment variable")
        return
    
    # Handle batch management commands
    if args.monitor_batch:
        monitor_batch_command(args.monitor_batch, api_key)
        return
    
    if args.download_batch:
        if not args.metadata_file:
            print("Error: --metadata-file required with --download-batch")
            return
        download_batch_command(args.download_batch, args.metadata_file, api_key)
        return
    
    # Regular batch submission
    if not args.input_file:
        print("Error: Input file required for batch submission")
        parser.print_help()
        return
    
    # Validate input file
    if not Path(args.input_file).exists():
        print(f"Error: Input file {args.input_file} not found")
        return
    
    # Create output file path if not specified
    if not args.output:
        args.output = suggest_output_filename(args.input_file, "anthropic", args.model)
    
    # Create configuration
    config = EvaluationConfig(
        provider="anthropic",
        model=args.model,
        temperature=args.temperature,
        perturbation_type=args.perturbation_type,
        sample_size=args.sample_size,
        offset=args.offset,
        output_file=args.output,
        dry_run=args.dry_run,
        resume=not args.no_resume,
        input_file=args.input_file
    )
    
    # Create evaluator and run
    evaluator = AnthropicBatchEvaluator(config, api_key)
    
    try:
        results = evaluator.execute(args.input_file)
        
        if not args.dry_run and not results:
            print("\n✅ Anthropic batch submitted successfully!")
            print("✅ Perturbation validation passed - no contamination risk!")
            print("Use the monitoring commands shown above to track progress.")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()