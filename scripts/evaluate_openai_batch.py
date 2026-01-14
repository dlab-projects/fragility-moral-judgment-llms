#!/usr/bin/env python3
"""
OpenAI Batch API evaluation script for ethical dilemmas.

- Custom_id format with content hashing for robust matching
- Comprehensive metadata preservation in batch requests
- Mandatory cost estimation and user confirmation
- Proper perturbation type filtering validation

Usage:
    python evaluate_openai_batch.py input.csv --output results.csv --model gpt-4.1
    python evaluate_openai_batch.py input.csv --dry-run  # Cost estimation only
    python evaluate_openai_batch.py input.csv --perturbation-type none --sample-size 10
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

# OpenAI imports
try:
    from openai import OpenAI
    from openai import RateLimitError
except ImportError:
    print("Error: OpenAI library not found. Install with: pip install openai")
    sys.exit(1)


class OpenAIBatchEvaluator(BaseEvaluator):
    """
    OpenAI Batch API evaluator with enhanced metadata preservation.
    
    Solves the critical issue where batch results couldn't be matched
    back to original scenarios due to insufficient custom_id metadata.
    """
    
    def __init__(self, config: EvaluationConfig, api_key: str):
        super().__init__(config, api_key)
        self.client = OpenAI(api_key=api_key)
        self.cost_calculator = CostCalculator()
        self.result_matcher = EnhancedResultMatcher()
        
        # Track batch processing
        self.batch_metadata: List[Dict[str, Any]] = []
        self.submitted_batch_id: Optional[str] = None
    
    def estimate_cost(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Estimate cost for OpenAI batch processing."""
        num_scenarios = len(df)
        total_requests = num_scenarios * 1  # Single run per scenario for batch
        
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
            is_batch=True,
            sample_texts=sample_texts
        )
        
        self.cost_calculator.print_cost_summary(cost_info)
        
        # Add perturbation info
        print(f"\nPerturbation type: {perturbation_types[0] if len(perturbation_types) == 1 else 'MIXED'}")
        print(f"Scenarios: {num_scenarios}")
        
        return cost_info
    
    def create_batch_request(self, df: pd.DataFrame) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Create batch request with enhanced metadata preservation.
        
        Returns:
            Tuple of (batch_requests, batch_metadata)
        """
        batch_requests = []
        self.batch_metadata = create_enhanced_batch_metadata(
            df=df,
            exp_prefix=f"o{datetime.now().strftime('%m%d%H%M')}_",
            num_runs=1
        )
        
        for metadata in self.batch_metadata:
            row_data = metadata['row_data']
            custom_id = metadata['custom_id']
            
            # Build evaluation prompt
            prompt = self.build_evaluation_prompt(pd.Series(row_data))
            
            # Create OpenAI batch request format
            batch_request = {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": self.config.model,
                    "messages": [
#                        {
#                            "role": "system",
#                            "content": "You are an expert evaluator of ethical dilemmas. Provide clear, structured responses in JSON format."
#                        },
                        {
                            "role": "user", 
                            "content": prompt
                        }
                    ],
                    "temperature": self.config.temperature,
                    "response_format": {"type": "json_object"},
                    "max_tokens": 500
                }
            }
            
            batch_requests.append(batch_request)
        
        self.logger.info(f"Created {len(batch_requests)} batch requests with enhanced metadata")
        return batch_requests, self.batch_metadata
    
    def submit_batch(self, batch_requests: List[Dict[str, Any]], batch_file_path: str) -> str:
        """
        Submit batch to OpenAI and return batch ID.
        
        Args:
            batch_requests: List of batch request dictionaries
            batch_file_path: Path to save the batch file
            
        Returns:
            Batch ID string
        """
        # Save batch requests to JSONL file
        with open(batch_file_path, 'w') as f:
            for request in batch_requests:
                f.write(json.dumps(request) + '\n')
        
        self.logger.info(f"Saved batch requests to {batch_file_path}")
        
        # Upload batch file to OpenAI
        self.logger.info("Uploading batch file to OpenAI...")
        with open(batch_file_path, 'rb') as f:
            file_upload = self.client.files.create(
                file=f,
                purpose="batch"
            )
        
        self.logger.info(f"File uploaded with ID: {file_upload.id}")
        
        # Create batch
        self.logger.info("Creating batch...")
        batch = self.client.batches.create(
            input_file_id=file_upload.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={
                "description": f"Ethical dilemma evaluation - {self.config.perturbation_type or 'all_types'}",
                "created_by": "evaluate_openai_batch.py",
                "model": self.config.model,
                "scenarios": str(len(batch_requests))
            }
        )
        
        self.submitted_batch_id = batch.id
        self.logger.info(f"Batch submitted successfully! Batch ID: {batch.id}")
        
        return batch.id
    
    def monitor_batch_status(self, batch_id: str) -> Dict[str, Any]:
        """
        Monitor batch status and provide updates.
        
        Args:
            batch_id: OpenAI batch ID
            
        Returns:
            Final batch status information
        """
        self.logger.info(f"Monitoring batch {batch_id}...")
        
        start_time = time.time()
        last_status = None
        
        while True:
            try:
                batch = self.client.batches.retrieve(batch_id)
                current_status = batch.status
                
                if current_status != last_status:
                    elapsed = time.time() - start_time
                    self.logger.info(f"Batch status: {current_status} (elapsed: {elapsed:.1f}s)")
                    
                    if hasattr(batch, 'request_counts') and batch.request_counts:
                        counts = batch.request_counts
                        total = counts.total if hasattr(counts, 'total') else 0
                        completed = counts.completed if hasattr(counts, 'completed') else 0
                        failed = counts.failed if hasattr(counts, 'failed') else 0
                        
                        if total > 0:
                            progress = (completed + failed) / total * 100
                            self.logger.info(f"  Progress: {completed}/{total} completed ({progress:.1f}%), {failed} failed")
                    
                    last_status = current_status
                
                if current_status in ['completed', 'failed', 'expired', 'cancelled']:
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
            batch_id: OpenAI batch ID
            original_df: Original DataFrame with perturbed scenarios
            
        Returns:
            List of matched results with preserved metadata
        """
        # Get batch object
        batch = self.client.batches.retrieve(batch_id)
        
        if batch.status != 'completed':
            raise ValueError(f"Batch not completed. Status: {batch.status}")
        
        if not batch.output_file_id:
            raise ValueError("No output file available for completed batch")
        
        # Download results file
        self.logger.info("Downloading batch results...")
        output_content = self.client.files.content(batch.output_file_id)
        
        # Parse JSONL results
        batch_results = []
        for line in output_content.text.strip().split('\n'):
            if line.strip():
                result = json.loads(line)
                batch_results.append(result)
        
        self.logger.info(f"Downloaded {len(batch_results)} batch results")
        
        # Process results with enhanced matching
        processed_results = []
        for result in batch_results:
            # Extract response data
            response_data = {}
            
            if result.get('response') and result['response'].get('body'):
                body = result['response']['body']
                if body.get('choices') and len(body['choices']) > 0:
                    choice = body['choices'][0]
                    if choice.get('message') and choice['message'].get('content'):
                        content = choice['message']['content']
                        
                        # Parse JSON response
                        try:
                            parsed_content = json.loads(content)
                            response_data = parsed_content
                        except json.JSONDecodeError:
                            response_data = {
                                'judgment': 'ERROR',
                                'explanation': f'Failed to parse JSON: {content[:100]}...',
                                'raw_response': content
                            }
            
            # Add batch metadata
            response_data.update({
                'batch_custom_id': result.get('custom_id', ''),
                'batch_status': 'completed',
                'openai_batch_id': batch_id
            })
            
            processed_results.append(response_data)
        
        # Match results to original scenarios using enhanced matcher
        matched_results, unmatched_results = self.result_matcher.process_batch_results(
            batch_results=processed_results,
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
    
    def run_evaluation(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Run OpenAI batch evaluation with enhanced metadata preservation.
        
        Args:
            df: DataFrame with perturbed scenarios
            
        Returns:
            List of evaluation results with preserved metadata
        """
        # Create batch requests with enhanced metadata
        batch_requests, batch_metadata = self.create_batch_request(df)
        
        # Create batch file path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_file = Path(self.config.output_file).parent / f"openai_batch_{timestamp}.jsonl"
        
        # Submit batch
        batch_id = self.submit_batch(batch_requests, str(batch_file))
        
        # Save batch metadata for recovery
        metadata_file = Path(self.config.output_file).with_suffix('.batch_metadata.json')
        with open(metadata_file, 'w') as f:
            json.dump({
                'batch_id': batch_id,
                'batch_file': str(batch_file),
                'metadata_file': str(metadata_file),
                'config': self.config.model_dump(),
                'batch_metadata': batch_metadata,
                'submitted_at': datetime.now().isoformat(),
                'input_file': getattr(self.config, 'input_file', None)  # Store input file path
            }, f, indent=2, default=str)
        
        self.logger.info(f"Batch metadata saved to {metadata_file}")
        
        print(f"\n🚀 BATCH SUBMITTED SUCCESSFULLY!")
        print(f"Batch ID: {batch_id}")
        print(f"Batch file: {batch_file}")
        print(f"Metadata file: {metadata_file}")
        print(f"\nTo monitor batch status:")
        print(f"  python {__file__} --monitor-batch {batch_id}")
        print(f"\nTo download results when complete:")
        print(f"  python {__file__} --download-batch {batch_id} --metadata-file {metadata_file}")
        
        return []  # Batch mode returns empty list initially


def create_argument_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="OpenAI Batch API evaluation with enhanced metadata preservation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Submit batch with cost estimation
  python evaluate_openai_batch.py input.csv --output results.csv --model gpt-4.1-mini
  
  # Dry run to see cost estimate only
  python evaluate_openai_batch.py input.csv --output results.csv --dry-run
  
  # Filter to specific perturbation type
  python evaluate_openai_batch.py input.csv --output results.csv --perturbation-type none
  
  # Monitor existing batch
  python evaluate_openai_batch.py --monitor-batch batch_12345
  
  # Download completed batch results
  python evaluate_openai_batch.py --download-batch batch_12345 --metadata-file batch.metadata.json
        """
    )
    
    # Input/output options
    parser.add_argument("input_file", nargs="?", help="Input CSV file with perturbed scenarios")
    parser.add_argument("--output", "--output-file", required=False, 
                       help="Output CSV file for results (auto-generated if not specified)")
    
    # Model options
    parser.add_argument("--model", default="gpt-4.1", 
                       help="OpenAI model to use (default: gpt-4.1)")
    parser.add_argument("--temperature", type=float, default=0.4,
                       help="Model temperature (default: 0.4)")
    
    # Filtering options
    parser.add_argument("--perturbation-type", 
                       help="Filter to specific perturbation type (e.g., none, firstperson)")
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
    client = OpenAI(api_key=api_key)
    
    print(f"Monitoring batch: {batch_id}")
    start_time = time.time()
    last_status = None
    
    while True:
        try:
            batch = client.batches.retrieve(batch_id)
            current_status = batch.status
            
            if current_status != last_status:
                elapsed = time.time() - start_time
                print(f"Status: {current_status} (elapsed: {elapsed:.1f}s)")
                
                if hasattr(batch, 'request_counts') and batch.request_counts:
                    counts = batch.request_counts
                    total = getattr(counts, 'total', 0)
                    completed = getattr(counts, 'completed', 0) 
                    failed = getattr(counts, 'failed', 0)
                    
                    if total > 0:
                        progress = (completed + failed) / total * 100
                        print(f"Progress: {completed}/{total} completed ({progress:.1f}%), {failed} failed")
                
                last_status = current_status
            
            if current_status in ['completed', 'failed', 'expired', 'cancelled']:
                print(f"\nBatch finished with status: {current_status}")
                if current_status == 'completed':
                    print("✅ Batch completed successfully! Use --download-batch to get results.")
                else:
                    print(f"❌ Batch finished with non-successful status: {current_status}")
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
    evaluator = OpenAIBatchEvaluator(config, api_key)
    
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
        print(f"Looked for: {[f for f in possible_input_files if f]}")
        return
    
    print(f"Loading original data from: {input_file}")
    original_df = pd.read_csv(input_file)
    
    # Apply same filtering as original batch
    if config.perturbation_type:
        original_df = original_df[original_df['perturbation_type'] == config.perturbation_type]
    
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
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: Set OPENAI_API_KEY environment variable")
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
        args.output = suggest_output_filename(args.input_file, "openai", args.model)
    
    # Create configuration
    config = EvaluationConfig(
        provider="openai",
        model=args.model,
        temperature=args.temperature,
        perturbation_type=args.perturbation_type,
        sample_size=args.sample_size,
        offset=args.offset,
        output_file=args.output,
        dry_run=args.dry_run,
        resume=not args.no_resume,
        input_file=args.input_file  # Store input file path
    )
    
    # Create evaluator and run
    evaluator = OpenAIBatchEvaluator(config, api_key)
    
    try:
        results = evaluator.execute(args.input_file)
        
        if not args.dry_run and not results:
            print("\n✅ Batch submitted successfully!")
            print("Use the monitoring commands shown above to track progress.")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()