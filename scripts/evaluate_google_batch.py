#!/usr/bin/env python3
"""
Google Gemini Batch API evaluation script for ethical dilemmas.

Uses Google's new batch mode for efficient large-scale evaluation with:
- 50% cost reduction compared to streaming API
- 24-hour target turnaround time
- Same models and capabilities as interactive API
- Enhanced metadata preservation for result matching
- JSONL file-based batch processing

Usage:
    python evaluate_google_batch.py input.csv --output results.csv --model gemini-2.5-flash
    python evaluate_google_batch.py input.csv --dry-run  # Cost estimation only
    python evaluate_google_batch.py input.csv --perturbation-type none --sample-size 10
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

# Google imports - using the new google.genai package
try:
    from google import genai
    from google.genai import types
    from pydantic import BaseModel
except ImportError:
    print("Error: Google GenAI library not found. Install with: pip install google-genai")
    sys.exit(1)


class EthicalJudgment(BaseModel):
    """Pydantic model for structured ethical judgment responses."""
    judgment: str  # YTA, NTA, ESH, NAH, INFO
    explanation: str


class GoogleBatchEvaluator(BaseEvaluator):
    """
    Google Gemini Batch API evaluator with enhanced metadata preservation.
    
    Uses Google's new batch mode for cost-effective large-scale evaluation
    while maintaining full compatibility with existing result processing.
    """
    
    def __init__(self, config: EvaluationConfig, api_key: str):
        super().__init__(config, api_key)
        self.client = genai.Client(api_key=api_key)
        self.cost_calculator = CostCalculator()
        self.result_matcher = EnhancedResultMatcher()
        
        # Track batch processing
        self.batch_metadata: List[Dict[str, Any]] = []
        self.submitted_batch_name: Optional[str] = None
        self.batch_job = None
        
        # Add retry configuration
        self.max_retries = getattr(config, 'max_retries', 3)
        self.base_wait_time = 2
    
    def estimate_cost(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Estimate cost for Google batch processing (50% reduction vs streaming)."""
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
            is_batch=True,  # This applies 50% cost reduction
            sample_texts=sample_texts
        )
        
        self.cost_calculator.print_cost_summary(cost_info)
        
        # Add batch-specific info
        print(f"\n📦 BATCH MODE BENEFITS:")
        print(f"   Cost reduction: 50% vs streaming API")
        print(f"   Target turnaround: 24 hours")
        print(f"   Batch size: {num_scenarios:,} requests")
        
        # Add perturbation info
        print(f"\nPerturbation type: {perturbation_types[0] if len(perturbation_types) == 1 else 'MIXED'}")
        print(f"Scenarios: {num_scenarios}")
        
        return cost_info
    
    def create_batch_jsonl(self, df: pd.DataFrame) -> tuple[str, List[Dict[str, Any]]]:
        """
        Create JSONL file for Google batch request with enhanced metadata preservation.
        
        Returns:
            Tuple of (jsonl_file_path, batch_metadata)
        """
        # Create batch metadata
        self.batch_metadata = create_enhanced_batch_metadata(
            df=df,
            exp_prefix=f"g{datetime.now().strftime('%m%d%H%M')}_",
            num_runs=1
        )
        
        # Create JSONL file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        jsonl_file = Path(self.config.output_file).parent / f"google_batch_{timestamp}.jsonl"
        
        with open(jsonl_file, 'w') as f:
            for metadata in self.batch_metadata:
                row_data = metadata['row_data']
                custom_id = metadata['custom_id']
                
                # Build evaluation prompt
                prompt = self.build_evaluation_prompt(pd.Series(row_data))
                
                # Create request object for JSONL in Google's required format with structured output
                request = {
                    "key": custom_id,  # User-defined key for tracking
                    "request": {
                        "contents": [
                            {
                                "parts": [
                                    {"text": prompt}
                                ]
                            }
                        ],
                        "generationConfig": {
                            "temperature": self.config.temperature,
                            "maxOutputTokens": 3000,
                            "response_mime_type": "application/json",
                            "response_json_schema": EthicalJudgment.model_json_schema()
                        }
                    }
                }
                
                # Write as single line JSON
                f.write(json.dumps(request) + '\n')
        
        self.logger.info(f"Created JSONL file with {len(self.batch_metadata)} requests: {jsonl_file}")
        return str(jsonl_file), self.batch_metadata
    
    def submit_batch(self, jsonl_file: str) -> str:
        """
        Submit batch to Google Batch API using file-based approach.
        
        File-based batching is the only reliable method that supports:
        - JSON schema validation (responseMimeType + responseSchema)
        - Full GenerateContentRequest structure
        - Robust batch processing
        
        Returns:
            Batch job name/ID
        """
        # Check file size for logging
        file_size = Path(jsonl_file).stat().st_size
        file_size_mb = file_size / (1024 * 1024)
        
        self.logger.info(f"Batch file size: {file_size_mb:.2f}MB")
        self.logger.info("Using file-based batch submission for full feature support")
        
        return self._submit_batch_file(jsonl_file)
    
    def _submit_batch_inline(self, jsonl_file: str) -> str:
        """Submit batch using inline requests (for files < 20MB)."""
        self.logger.info("Using inline requests for batch submission")
        
        # Read and parse JSONL requests and convert to inline format
        inline_requests = []
        with open(jsonl_file, 'r') as f:
            for line in f:
                if line.strip():
                    request_data = json.loads(line)
                    
                    # Clean up generationConfig for inline requests - remove JSON schema constraints
                    generation_config = request_data['request']['generationConfig'].copy()
                    # Remove responseMimeType and responseSchema as they're not allowed in inline requests
                    generation_config.pop('responseMimeType', None)
                    generation_config.pop('responseSchema', None)
                    
                    # Convert to simple inline request format (plain dictionary)
                    # Note: No 'key' field - that's only for file-based batching
                    inline_request = {
                        'contents': request_data['request']['contents'],
                        'generationConfig': generation_config
                    }
                    inline_requests.append(inline_request)
        
        model_name = f"models/{self.config.model}"
        
        self.logger.info(f"Submitting batch job with {len(inline_requests)} inline requests")
        self.logger.info(f"Model: {model_name}")
        
        # Create batch job with inline requests - use exact syntax from documentation
        self.batch_job = self.client.batches.create(
            model=model_name,
            src=inline_requests,
            config={
                'display_name': f"ethical_dilemmas_{self.config.perturbation_type or 'mixed'}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            }
        )
        
        self.submitted_batch_name = self.batch_job.name
        self.logger.info(f"✅ Batch submitted successfully with inline requests!")
        self.logger.info(f"Batch name: {self.submitted_batch_name}")
        self.logger.info(f"State: {self.batch_job.state}")
        
        return self.submitted_batch_name
    
    def _submit_batch_file(self, jsonl_file: str) -> str:
        """Submit batch using file upload (for larger files)."""
        self.logger.info("Using file upload for batch submission")
        
        # Upload the JSONL file
        uploaded_file = self.client.files.upload(
            file=jsonl_file,
            config=types.UploadFileConfig(
                display_name=f"batch_input_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                mime_type='text/plain'  # Try text/plain instead of application/jsonl
            )
        )
        
        self.logger.info(f"File uploaded: {uploaded_file.name}")
        self.logger.info(f"File state: {getattr(uploaded_file, 'state', 'unknown')}")
        
        # Wait briefly for file to be processed
        time.sleep(2)
        
        # Create batch job - try without models/ prefix
        model_name = self.config.model
        
        self.logger.info(f"Submitting batch job with model: {model_name}")
        self.logger.info(f"Using uploaded file: {uploaded_file.name}")
        
        # Create the batch job using simple config dict
        try:
            self.batch_job = self.client.batches.create(
                model=model_name,
                src=uploaded_file.name,
                config={
                    'display_name': f"ethical_dilemmas_{self.config.perturbation_type or 'mixed'}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                }
            )
        except ValueError as e:
            # If that fails, try without the models/ prefix
            self.logger.warning(f"Failed with models/ prefix: {e}, trying without prefix")
            self.batch_job = self.client.batches.create(
                model=self.config.model,
                src=uploaded_file.name,
                config={
                    'display_name': f"ethical_dilemmas_{self.config.perturbation_type or 'mixed'}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                }
            )
        
        self.submitted_batch_name = self.batch_job.name
        self.logger.info(f"✅ Batch submitted successfully!")
        self.logger.info(f"Batch name: {self.submitted_batch_name}")
        self.logger.info(f"State: {self.batch_job.state}")
        
        return self.submitted_batch_name
    
    def monitor_batch(self, batch_name: str) -> Dict[str, Any]:
        """Monitor batch job status."""
        try:
            batch_job = self.client.batches.get(name=batch_name)
            
            status_info = {
                'name': getattr(batch_job, 'name', batch_name),
                'state': getattr(batch_job, 'state', 'UNKNOWN'),
                'display_name': getattr(batch_job, 'display_name', 'N/A'),
                'create_time': getattr(batch_job, 'create_time', 'N/A'),
                'update_time': getattr(batch_job, 'update_time', 'N/A'),
                'request_count': getattr(batch_job, 'request_count', 0),
                'error_message': getattr(batch_job, 'error_message', None)
            }
            
            # Add batch stats if available
            if hasattr(batch_job, 'batch_stats'):
                batch_stats = batch_job.batch_stats
                status_info['completed_count'] = getattr(batch_stats, 'completed_request_count', 0)
                status_info['failed_count'] = getattr(batch_stats, 'failed_request_count', 0)
                if status_info['request_count'] > 0:
                    status_info['progress'] = status_info['completed_count'] / status_info['request_count'] * 100
            
            return status_info
            
        except Exception as e:
            # Check if this is the Vertex AI error
            if "This method is only supported in the Vertex AI client" in str(e):
                return {
                    'error': 'Batch monitoring not supported with current client configuration',
                    'suggestion': 'Check batch status manually at https://aistudio.google.com/app/batches',
                    'batch_name': batch_name
                }
            else:
                self.logger.error(f"Failed to get batch status: {e}")
                return {'error': str(e)}
    
    def download_batch_results(self, batch_name: str, metadata_file: Optional[str] = None) -> str:
        """
        Download and process batch results.
        
        Returns:
            Path to the processed results CSV file
        """
        self.logger.info(f"Checking batch status: {batch_name}")
        
        # Get batch job
        batch_job = self.client.batches.get(name=batch_name)
        
        if batch_job.state != "JOB_STATE_SUCCEEDED":
            raise ValueError(f"Batch is not complete. Current state: {batch_job.state}")
        
        # Get output file info based on batch type
        if hasattr(batch_job, 'dest') and hasattr(batch_job.dest, 'file_name') and batch_job.dest.file_name:
            # File-based batch - download from file
            output_file_name = batch_job.dest.file_name
            self.logger.info(f"Downloading file-based results from: {output_file_name}")
            
            # Create local output file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_jsonl = Path(self.config.output_file).parent / f"google_batch_results_{timestamp}.jsonl"
            
            # Download content using the correct API
            with open(results_jsonl, 'wb') as f:
                content = self.client.files.download(file=output_file_name)
                f.write(content)
                
        elif hasattr(batch_job, 'dest') and hasattr(batch_job.dest, 'inlined_responses'):
            # Inline batch - process responses directly
            self.logger.info("Processing inline batch responses")
            
            # Create local output file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_jsonl = Path(self.config.output_file).parent / f"google_batch_results_{timestamp}.jsonl"
            
            # Convert inline responses to JSONL format
            import json
            with open(results_jsonl, 'w') as f:
                for i, response in enumerate(batch_job.dest.inlined_responses):
                    # Create a result entry matching the file-based format
                    result_entry = {
                        'custom_id': f'inline_request_{i}',  # Will need to match with metadata
                        'response': response
                    }
                    f.write(json.dumps(result_entry) + '\n')
        else:
            raise ValueError("Batch job has no output file or inline responses")
        
        self.logger.info(f"Downloaded results to: {results_jsonl}")
        
        # Load metadata if provided
        if metadata_file and Path(metadata_file).exists():
            import json
            with open(metadata_file, 'r') as f:
                saved_metadata = json.load(f)
                self.batch_metadata = saved_metadata.get('batch_metadata', [])
                self.config = EvaluationConfig(**saved_metadata.get('config', {}))
                # Store the original input file path for consistent ordering
                self.original_input_file = saved_metadata.get('input_file')
        
        # Process results
        return self.process_batch_results(results_jsonl)
    
    def process_batch_results(self, results_jsonl: str) -> str:
        """
        Process JSONL results and create final CSV output.
        
        Returns:
            Path to the processed CSV file
        """
        results = []
        
        # Read JSONL results
        with open(results_jsonl, 'r') as f:
            for line in f:
                if line.strip():
                    result = json.loads(line)
                    
                    # Convert Google 'key' to 'custom_id' for compatibility with result matcher
                    if 'key' in result:
                        result['custom_id'] = result['key']
                    
                    # Extract response from Google batch format
                    if 'response' in result and 'candidates' in result['response']:
                        candidates = result['response']['candidates']
                        if candidates and len(candidates) > 0:
                            content = candidates[0].get('content', {})
                            parts = content.get('parts', [])
                            if parts and len(parts) > 0:
                                text = parts[0].get('text', '')
                                try:
                                    # Try to parse as JSON first (if structured output works)
                                    parsed_json = json.loads(text)
                                    
                                    # Validate the response structure
                                    judgment = parsed_json.get('judgment', 'ERROR')
                                    explanation = parsed_json.get('explanation', 'No explanation provided')
                                    
                                    # Validate judgment is one of expected values
                                    valid_judgments = ['YTA', 'NTA', 'ESH', 'NAH', 'INFO']
                                    if judgment not in valid_judgments:
                                        self.logger.warning(f"Invalid judgment '{judgment}' for {result.get('custom_id', 'unknown')}")
                                        judgment = 'ERROR'
                                        explanation = f"Invalid judgment in response: {text[:100]}"
                                    
                                    result['response'] = {
                                        'judgment': judgment,
                                        'explanation': explanation
                                    }
                                    
                                except (json.JSONDecodeError, ValueError):
                                    # Fallback to text parsing if JSON parsing fails
                                    judgment = 'ERROR'
                                    explanation = text.strip()
                                    
                                    # Improved judgment extraction
                                    lines = text.strip().split('\n')
                                    first_line = lines[0] if lines else ''
                                    
                                    # Check first line for judgment
                                    if first_line.upper() in ['YTA', 'NTA', 'ESH', 'NAH', 'INFO']:
                                        judgment = first_line.upper()
                                    else:
                                        # Check if judgment appears at start of first line
                                        for valid_judgment in ['YTA', 'NTA', 'ESH', 'NAH', 'INFO']:
                                            if first_line.upper().startswith(valid_judgment):
                                                judgment = valid_judgment
                                                break
                                    
                                    result['response'] = {
                                        'judgment': judgment,
                                        'explanation': explanation
                                    }
                            else:
                                self.logger.warning(f"No text content found for {result.get('custom_id', 'unknown')}")
                                result['response'] = {'judgment': 'ERROR', 'explanation': 'No response content'}
                    else:
                        self.logger.warning(f"No valid response found for {result.get('custom_id', 'unknown')}")
                        result['response'] = {'judgment': 'ERROR', 'explanation': 'No response found'}
                    
                    results.append(result)
        
        self.logger.info(f"Processed {len(results)} results")
        
        # Match results with metadata using enhanced result matcher
        # Load original CSV file to preserve original ordering (like OpenAI/Anthropic)
        import pandas as pd
        from pathlib import Path
        if hasattr(self, 'original_input_file') and self.original_input_file and Path(self.original_input_file).exists():
            # Load from original CSV file to preserve original ordering
            original_df = pd.read_csv(self.original_input_file)
            self.logger.info(f"Loaded original CSV file for consistent ordering: {self.original_input_file}")
        else:
            # Fallback to batch metadata (original behavior)
            original_df = pd.DataFrame([metadata['row_data'] for metadata in self.batch_metadata])
            self.logger.warning("Using batch metadata for ordering - results may not match original file order")
        
        # Process results with the enhanced result matcher
        matched_results, unmatched_results = self.result_matcher.process_batch_results(
            batch_results=results,
            original_df=original_df
        )
        
        # Convert to DataFrame
        df_results = pd.DataFrame(matched_results)
        
        # Check if standard columns already exist (added by result_matcher)
        has_judgment = 'judgment' in df_results.columns
        has_explanation = 'explanation' in df_results.columns
        
        # Only add versioned columns if we don't have standard columns, or if this is a multi-run scenario
        run_num = 1
        
        if not has_judgment or not has_explanation:
            # Standard columns don't exist, create them from versioned columns if needed
            judgment_col = f'judgment_{self.config.model.replace("-", "_").replace(".", "_")}_run_{run_num}'
            explanation_col = f'explanation_{self.config.model.replace("-", "_").replace(".", "_")}_run_{run_num}'
            
            if not has_judgment:
                df_results['judgment'] = df_results.get('judgment', 'ERROR')
            if not has_explanation:
                df_results['explanation'] = df_results.get('explanation', '')
                
        # Note: We removed the duplicate versioned column creation since result_matcher already provides standard columns
        
        # Drop processing columns
        columns_to_drop = ['custom_id', 'response', 'request_id', 'key']
        df_results = df_results.drop(columns=[col for col in columns_to_drop if col in df_results.columns])
        
        # Save final results
        df_results.to_csv(self.config.output_file, index=False)
        self.logger.info(f"✅ Saved final results to: {self.config.output_file}")
        
        # Print summary
        print(f"\n📊 EVALUATION SUMMARY:")
        print(f"Total scenarios evaluated: {len(df_results)}")
        print(f"Output file: {self.config.output_file}")
        
        # Show judgment distribution
        print(f"\nJudgment distribution:")
        if 'judgment' in df_results.columns:
            judgment_dist = df_results['judgment'].value_counts()
            for judgment, count in judgment_dist.items():
                print(f"  {judgment}: {count} ({count/len(df_results)*100:.1f}%)")
        else:
            print("  No judgment column found")
        
        return self.config.output_file
    
    def run_evaluation(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Run the evaluation on the given dataset.
        
        This method is required by BaseEvaluator abstract class.
        For batch mode, it creates and submits the batch job.
        
        Returns:
            Empty list (results are saved to file after batch completion)
        """
        # Create JSONL file
        jsonl_file, batch_metadata = self.create_batch_jsonl(df)
        
        # Submit batch
        batch_name = self.submit_batch(jsonl_file)
        
        # Save batch metadata for recovery
        metadata_file = Path(self.config.output_file).with_suffix('.batch_metadata.json')
        with open(metadata_file, 'w') as f:
            json.dump({
                'batch_name': batch_name,
                'batch_file': jsonl_file,
                'metadata_file': str(metadata_file),
                'config': self.config.model_dump(),
                'batch_metadata': batch_metadata,
                'submitted_at': datetime.now().isoformat(),
                'input_file': getattr(self.config, 'input_file', None)
            }, f, indent=2, default=str)
        
        self.logger.info(f"Batch metadata saved to {metadata_file}")
        
        print(f"\n🚀 BATCH SUBMITTED SUCCESSFULLY!")
        print(f"Batch name: {batch_name}")
        print(f"Batch file: {jsonl_file}")
        print(f"Metadata file: {metadata_file}")
        print(f"\nTo monitor batch status:")
        print(f"  python {__file__} --monitor-batch {batch_name}")
        print(f"\nTo download results when complete:")
        print(f"  python {__file__} --download-batch {batch_name} --metadata-file {metadata_file}")
        
        return []  # Batch mode returns empty list initially
    
    def execute(self, input_file: str) -> List[Dict[str, Any]]:
        """
        Execute batch evaluation from file path.
        
        This method loads data and calls run_evaluation.
        
        Returns:
            Empty list (results are saved to file after batch completion)
        """
        # Load data
        df = self.load_and_validate_data(input_file)
        
        # Run evaluation
        return self.run_evaluation(df)


def create_argument_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Google Gemini Batch API evaluation with 50% cost reduction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Submit batch with cost estimation
  python evaluate_google_batch.py input.csv --output results.csv --model gemini-2.5-flash
  
  # Cost estimation only
  python evaluate_google_batch.py input.csv --dry-run
  
  # Monitor batch status
  python evaluate_google_batch.py --monitor-batch batch_name
  
  # Download completed batch results
  python evaluate_google_batch.py --download-batch batch_name --metadata-file results.batch_metadata.json
  
  # Process specific perturbation type
  python evaluate_google_batch.py input.csv --perturbation-type add_emotion --output emotion_results.csv
"""
    )
    
    # Input file (optional for monitoring/downloading)
    parser.add_argument("input_file", nargs='?', help="Input CSV file with scenarios to evaluate")
    
    # Model and output options
    parser.add_argument("--model", default="gemini-2.5-flash",
                       help="Model to use (default: gemini-2.5-flash)")
    parser.add_argument("--output", "--output-file", dest="output_file",
                       help="Output CSV file for results")
    
    # Processing options
    parser.add_argument("--perturbation-type",
                       help="Filter to specific perturbation type")
    parser.add_argument("--sample-size", type=int,
                       help="Maximum number of scenarios to process")
    parser.add_argument("--offset", type=int, default=0,
                       help="Number of rows to skip from start")
    parser.add_argument("--temperature", type=float, default=0.4,
                       help="Model temperature (default: 0.4)")
    
    # Batch operations
    parser.add_argument("--monitor-batch",
                       help="Monitor status of a batch job")
    parser.add_argument("--download-batch",
                       help="Download results from completed batch")
    parser.add_argument("--metadata-file",
                       help="Metadata file for batch recovery")
    
    # Control options
    parser.add_argument("--dry-run", action="store_true",
                       help="Show cost estimation without running evaluation")
    parser.add_argument("--no-resume", action="store_true",
                       help="Don't resume from existing results")
    
    return parser


def main():
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Get API key
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Error: GOOGLE_API_KEY environment variable not set")
        print("Get your API key from: https://makersuite.google.com/app/apikey")
        return
    
    # Handle batch monitoring
    if args.monitor_batch:
        config = EvaluationConfig(
            provider="google",
            model="gemini-2.5-flash",
            temperature=0.4,
            output_file="temp.csv"
        )
        evaluator = GoogleBatchEvaluator(config, api_key)
        
        print(f"\n📊 BATCH STATUS")
        print("=" * 60)
        
        status = evaluator.monitor_batch(args.monitor_batch)
        if 'error' in status:
            print(f"❌ Error: {status['error']}")
        else:
            print(f"Batch name: {status['name']}")
            print(f"State: {status['state']}")
            print(f"Display name: {status['display_name']}")
            print(f"Created: {status['create_time']}")
            print(f"Updated: {status['update_time']}")
            print(f"Request count: {status['request_count']}")
            
            if 'progress' in status:
                print(f"Progress: {status['completed_count']}/{status['request_count']} ({status['progress']:.1f}%)")
            
            if status['error_message']:
                print(f"Error: {status['error_message']}")
        
        return
    
    # Handle batch download
    if args.download_batch:
        if not args.metadata_file:
            print("Error: --metadata-file required for downloading batch results")
            return
        
        # Load config from metadata
        with open(args.metadata_file, 'r') as f:
            metadata = json.load(f)
        
        config = EvaluationConfig(**metadata['config'])
        evaluator = GoogleBatchEvaluator(config, api_key)
        
        try:
            output_file = evaluator.download_batch_results(args.download_batch, args.metadata_file)
            print(f"\n✅ Results successfully downloaded and processed!")
            print(f"Output file: {output_file}")
        except Exception as e:
            print(f"\n❌ Error downloading results: {e}")
            import traceback
            traceback.print_exc()
        
        return
    
    # Regular evaluation mode
    if not args.input_file:
        parser.error("Input file required for evaluation mode")
    
    # Validate input file
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: Input file not found: {args.input_file}")
        return
    
    # Debug: Check what model value we have
    print(f"Debug: args.model = {args.model}")
    print(f"Debug: type(args.model) = {type(args.model)}")
    
    # Ensure model has a default value
    if not args.model:
        args.model = "gemini-2.5-flash"
        print(f"Using default model: {args.model}")
    
    # Create output filename if not specified
    if not args.output_file:
        print(f"Debug: About to call suggest_output_filename")
        print(f"  input_file={args.input_file}")
        print(f"  model={args.model}")
        print(f"  perturbation_type={args.perturbation_type}")
        args.output_file = suggest_output_filename(
            input_file=args.input_file,
            provider="google",
            model=args.model
        )
    
    # Create output directory
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create configuration
    config = EvaluationConfig(
        provider="google",
        model=args.model,
        temperature=args.temperature,
        perturbation_type=args.perturbation_type,
        sample_size=args.sample_size,
        offset=args.offset,
        output_file=args.output_file,
        dry_run=args.dry_run,
        resume=not args.no_resume
    )
    
    # Create evaluator
    evaluator = GoogleBatchEvaluator(config, api_key)
    
    try:
        # Estimate cost
        df = evaluator.load_and_validate_data(args.input_file)
        cost_info = evaluator.estimate_cost(df)
        
        if args.dry_run:
            print("\n✅ Dry run complete - no evaluation performed")
            return
        
        # Get user confirmation
        print(f"\n{'='*60}")
        print("CONFIRM EVALUATION")
        print(f"{'='*60}")
        print(f"Model: {args.model}")
        print(f"Temperature: {args.temperature}")
        print(f"Scenarios: {len(df)}")
        print(f"Output: {args.output_file}")
        
        response = input("\nProceed with batch submission? (yes/no): ")
        if response.lower() not in ['yes', 'y']:
            print("Evaluation cancelled")
            return
        
        # Execute evaluation
        print("\n" + "="*60)
        print("SUBMITTING BATCH")
        print("="*60)
        
        evaluator.execute(args.input_file)
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Evaluation interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()