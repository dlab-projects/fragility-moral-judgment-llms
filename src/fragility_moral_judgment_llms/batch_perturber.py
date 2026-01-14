"""
Google Batch API perturbation implementation for efficient large-scale perturbation generation.

Uses Google's batch mode for cost-effective perturbation generation with:
- 50% cost reduction compared to streaming API
- Structured JSON output for reliable parsing
- Enhanced metadata preservation for result matching
- JSONL file-based batch processing
"""

import json
import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
from datetime import datetime

# Google imports - using the new google.genai package
try:
    from google import genai
    from google.genai import types
except ImportError:
    print("Error: Google GenAI library not found. Install with: pip install google-genai")
    raise

from .perturber import PerturbationResult, PerturbationConfig
from .config import PRESENTATION_TEMPLATES, VARIATION_TEMPLATES
from .cost_calculator import CostCalculator
from .result_matcher import EnhancedResultMatcher, create_enhanced_batch_metadata


class GoogleBatchPerturber:
    """
    Google Gemini Batch API perturber with enhanced metadata preservation.
    
    Uses Google's batch mode for cost-effective large-scale perturbation generation
    while maintaining full compatibility with existing result processing.
    """
    
    def __init__(self, api_key: str, temperature: float = 0.4):
        self.client = genai.Client(api_key=api_key)
        self.temperature = temperature
        self.cost_calculator = CostCalculator()
        self.result_matcher = EnhancedResultMatcher()
        self.logger = logging.getLogger(__name__)
        
        # Track batch processing
        self.batch_metadata: List[Dict[str, Any]] = []
        self.submitted_batch_name: Optional[str] = None
        self.batch_job = None
    
    def build_perturbation_prompt(self, scenario_text: str, perturbation_config: PerturbationConfig) -> str:
        """Build the perturbation prompt for a given scenario and perturbation type."""
        perturbation_type = perturbation_config.perturbation_type
        
        # Handle 'none' perturbation (no changes)
        if perturbation_type == 'none':
            return f"""You are tasked with returning the original text without any changes.

Original scenario:
{scenario_text}

Return the exact original text in the following JSON format:
{{
    "perturbed_text": "[exact original text]",
    "success": true,
    "perturbation_description": "No perturbation applied - original text preserved",
    "perturbation_degree": 0
}}"""
        
        # Get perturbation template
        if perturbation_config.is_format_perturbation:
            template = PRESENTATION_TEMPLATES.get(perturbation_type)
        else:
            template = VARIATION_TEMPLATES.get(perturbation_type)
        
        if not template:
            raise ValueError(f"Unknown perturbation type: {perturbation_type}")
        
        # Build the prompt with structured output instructions
        prompt = f"""You are tasked with applying the following perturbation to an ethical dilemma scenario.

Perturbation type: {perturbation_type}
Instructions: {template}

Original scenario:
{scenario_text}

Please apply the perturbation and respond in the following JSON format.
IMPORTANT: Use normal spaces and line breaks in your text - do not use excessive tabs or whitespace characters.

{{
    "perturbed_text": "[the modified scenario text]",
    "success": true,
    "perturbation_description": "[brief description of what changes were made]",
    "perturbation_degree": [0-3 scale where 0=no change, 1=minor, 2=moderate, 3=major changes]
}}

If you cannot apply the perturbation for any reason, respond with:
{{
    "perturbed_text": "{scenario_text}",
    "success": false,
    "perturbation_description": "[explanation of why perturbation failed]",
    "perturbation_degree": 0,
    "error": "[error description]"
}}

Important: 
- Maintain the ethical dilemma's core structure and meaning
- Only make changes that align with the specific perturbation instructions
- Rate the degree of change realistically (most perturbations should be 1-2)
- Ensure the perturbed text is still a coherent ethical dilemma
"""
        
        return prompt
    
    def _validate_input_data(self, df: pd.DataFrame) -> None:
        """Validate that input data is appropriate for perturbation."""
        # Check for required text columns
        required_columns = ['text', 'selftext_cleaned']
        has_text_column = any(col in df.columns for col in required_columns)
        
        if not has_text_column:
            available_cols = list(df.columns)
            raise ValueError(
                f"❌ ERROR: Input file must contain original text data in 'text' or 'selftext_cleaned' columns.\n"
                f"Found columns: {available_cols}\n"
                f"This appears to be a processed/perturbation file. Use candidates.csv or similar original data files."
            )
        
        # Check if this looks like already-processed perturbation data
        if 'perturbed_text' in df.columns and 'perturbation_type' in df.columns:
            print("⚠️  WARNING: This appears to be already-processed perturbation data.")
            print("   The batch perturber should be used with original scenario data (e.g., candidates.csv).")
            print("   Re-perturbing already-processed data may produce unexpected results.")
            raise ValueError(
                "❌ ERROR: Use original scenario data instead of processed perturbation files.\n"
                "   Expected: candidates.csv with 'selftext_cleaned' column\n"
                f"   Found: File with 'perturbed_text' and 'perturbation_type' columns"
            )
    
    def estimate_cost(self, df: pd.DataFrame, perturbation_configs: List[PerturbationConfig]) -> Dict[str, Any]:
        """Estimate cost for Google batch perturbation processing (50% reduction vs streaming)."""
        # Validate input file has proper text columns (do this early, even for dry runs)
        self._validate_input_data(df)
        
        num_scenarios = len(df)
        total_requests = num_scenarios * len(perturbation_configs)
        
        # Sample some prompts for better token estimation
        sample_texts = []
        if not df.empty and perturbation_configs:
            sample_rows = df.head(min(3, len(df)))
            for _, row in sample_rows.iterrows():
                for config in perturbation_configs[:2]:  # Sample first 2 perturbations
                    # Use 'selftext_cleaned' from candidates.csv or 'text' column only
                    text_content = row.get('text', row.get('selftext_cleaned', ''))
                    prompt = self.build_perturbation_prompt(text_content, config)
                    sample_texts.append(prompt)
        
        # Get primary model from configs
        primary_model = perturbation_configs[0].model if perturbation_configs else "gemini-2.5-flash"
        
        cost_info = self.cost_calculator.estimate_dataset_cost(
            df=df,
            model=primary_model,
            num_runs=len(perturbation_configs),
            is_batch=True,  # This applies 50% cost reduction
            sample_texts=sample_texts
        )
        
        self.cost_calculator.print_cost_summary(cost_info)
        
        # Add batch-specific info
        print(f"\n📦 BATCH MODE BENEFITS:")
        print(f"   Cost reduction: 50% vs streaming API")
        print(f"   Target turnaround: 24 hours")
        print(f"   Batch size: {total_requests:,} requests")
        
        # Add perturbation info
        perturbation_types = [config.perturbation_type for config in perturbation_configs]
        print(f"\nPerturbation types: {', '.join(perturbation_types)}")
        print(f"Scenarios: {num_scenarios}")
        print(f"Total perturbations: {total_requests}")
        
        return cost_info
    
    def create_batch_jsonl(self, df: pd.DataFrame, perturbation_configs: List[PerturbationConfig], 
                          metadata_columns: List[str], output_file: str) -> tuple[str, List[Dict[str, Any]]]:
        """
        Create JSONL file for Google batch perturbation request with enhanced metadata preservation.
        
        Returns:
            Tuple of (jsonl_file_path, batch_metadata)
        """
        # Input validation is done in estimate_cost() method
        # Create enhanced batch metadata for perturbations
        self.batch_metadata = []
        
        # Create entries for each scenario + perturbation combination
        for idx, (_, row) in enumerate(df.iterrows()):
            for config in perturbation_configs:
                # Create unique identifier
                scenario_id = f"s{idx}"
                perturbation_abbrev = config.perturbation_type[:8]  # First 8 chars
                model_abbrev = config.model.split('-')[-1][:4]  # Last part, first 4 chars
                
                # Use 'selftext_cleaned' from candidates.csv or 'text' column only
                text_content = row.get('text', row.get('selftext_cleaned', ''))
                
                # Create hash for uniqueness
                content_hash = str(hash(f"{text_content}_{config.perturbation_type}_{config.model}"))[-6:]
                custom_id = f"p_{scenario_id}_{perturbation_abbrev}_{model_abbrev}_{content_hash}"
                
                # Prepare metadata with missing fields for compatibility
                metadata = {
                    'custom_id': custom_id,
                    'row_data': {
                        'text_index': idx,  # Add missing text_index field
                        'scenario_name': row.get('scenario_name', ''),  # Add missing scenario_name field
                        'perturbation_type': config.perturbation_type,
                        'text': text_content,  # Add the text content that was extracted above
                        **{col: row.get(col, '') for col in metadata_columns if col in row},
                        'model': config.model,
                        'perturbed_text': '',  # Will be filled by results
                        'perturbation_description': '',  # Will be filled by results
                        'perturbation_degree': '',  # Will be filled by results
                        'gender_swap_makes_sense': row.get('gender_swap_makes_sense', ''),  # Add missing field
                        'success': '',  # Will be filled by results
                        'error': '',  # Will be filled by results
                        'processing_time': '',  # Will be filled by results
                        'scenario_id': scenario_id
                    },
                    'perturbation_config': config.model_dump(),
                    'start_time': time.time()  # Track processing time
                }
                
                self.batch_metadata.append(metadata)
        
        # Create JSONL file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        jsonl_file = Path(output_file).parent / f"google_batch_perturbations_{timestamp}.jsonl"
        
        with open(jsonl_file, 'w') as f:
            for metadata in self.batch_metadata:
                row_data = metadata['row_data']
                custom_id = metadata['custom_id']
                config = PerturbationConfig(**metadata['perturbation_config'])
                
                # Build perturbation prompt
                prompt = self.build_perturbation_prompt(row_data['text'], config)
                
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
                            "temperature": self.temperature,
                            "maxOutputTokens": 8000,  # Increased for longer scenarios
                            "response_mime_type": "application/json",
                            "response_json_schema": PerturbationResult.model_json_schema()
                        }
                    }
                }
                
                # Write as single line JSON
                f.write(json.dumps(request) + '\n')
        
        self.logger.info(f"Created JSONL file with {len(self.batch_metadata)} requests: {jsonl_file}")
        return str(jsonl_file), self.batch_metadata
    
    def submit_batch(self, jsonl_file: str, perturbation_types: List[str]) -> str:
        """
        Submit batch to Google Batch API using file-based approach.
        
        Returns:
            Batch job name/ID
        """
        # Check file size for logging
        file_size = Path(jsonl_file).stat().st_size
        file_size_mb = file_size / (1024 * 1024)
        
        self.logger.info(f"Batch file size: {file_size_mb:.2f}MB")
        self.logger.info("Using file-based batch submission for structured output support")
        
        return self._submit_batch_file(jsonl_file, perturbation_types)
    
    def _submit_batch_file(self, jsonl_file: str, perturbation_types: List[str]) -> str:
        """Submit batch using file upload."""
        self.logger.info("Using file upload for batch submission")
        
        # Upload the JSONL file
        uploaded_file = self.client.files.upload(
            file=jsonl_file,
            config=types.UploadFileConfig(
                display_name=f"batch_perturbations_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                mime_type='text/plain'
            )
        )
        
        self.logger.info(f"File uploaded: {uploaded_file.name}")
        
        # Wait briefly for file to be processed
        time.sleep(2)
        
        # Create batch job - use same approach as working evaluate_google_batch.py
        model_name = "gemini-2.5-flash"  # Don't use models/ prefix (matches working evaluator)
        
        self.logger.info(f"Submitting batch job with model: {model_name}")
        
        try:
            self.batch_job = self.client.batches.create(
                model=model_name,
                src=uploaded_file.name,
                config={
                    'display_name': f"perturbations_{'_'.join(perturbation_types[:3])}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                }
            )
        except ValueError as e:
            # If that fails, try the same model again (keeping same approach as working evaluator)
            self.logger.warning(f"Failed with first attempt: {e}, retrying...")
            self.batch_job = self.client.batches.create(
                model="gemini-2.5-flash",
                src=uploaded_file.name,
                config={
                    'display_name': f"perturbations_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
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
        
        # Download results based on batch type
        if hasattr(batch_job, 'dest') and hasattr(batch_job.dest, 'file_name') and batch_job.dest.file_name:
            # File-based batch - download from file
            output_file_name = batch_job.dest.file_name
            self.logger.info(f"Downloading file-based results from: {output_file_name}")
            
            # Create local output file in data/ directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            data_dir = Path.cwd() / "data"
            data_dir.mkdir(exist_ok=True)
            results_jsonl = data_dir / f"google_batch_perturbation_results_{timestamp}.jsonl"
            
            # Download content
            with open(results_jsonl, 'wb') as f:
                content = self.client.files.download(file=output_file_name)
                f.write(content)
        else:
            raise ValueError("Batch job has no output file")
        
        self.logger.info(f"Downloaded results to: {results_jsonl}")
        
        # Load metadata if provided
        if metadata_file and Path(metadata_file).exists():
            with open(metadata_file, 'r') as f:
                saved_metadata = json.load(f)
                self.batch_metadata = saved_metadata.get('batch_metadata', [])
        
        return str(results_jsonl)
    
    def process_batch_results(self, results_jsonl: str, output_file: str) -> str:
        """
        Process JSONL results and create final CSV output.
        
        Returns:
            Path to the processed CSV file
        """
        results = []
        failed_results = []  # Track failed perturbations
        
        # Read JSONL results
        with open(results_jsonl, 'r') as f:
            for line_num, line in enumerate(f, 1):
                if line.strip():
                    result = json.loads(line)
                    
                    # Convert Google 'key' to 'custom_id' for compatibility
                    if 'key' in result:
                        result['custom_id'] = result['key']
                    
                    # Check for API errors first
                    if 'error' in result:
                        self.logger.error(f"API error for {result.get('custom_id', 'unknown')}: {result['error']}")
                        result['perturbation_data'] = {
                            'perturbed_text': '',
                            'success': False,
                            'perturbation_description': 'API request failed',
                            'perturbation_degree': 0,
                            'error': f"API error: {result['error']}"
                        }
                        failed_results.append(result.get('custom_id', f'line_{line_num}'))
                        results.append(result)
                        continue
                    
                    # Check for content filter blocks
                    if 'response' in result and 'promptFeedback' in result['response']:
                        prompt_feedback = result['response']['promptFeedback']
                        if 'blockReason' in prompt_feedback:
                            block_reason = prompt_feedback['blockReason']
                            self.logger.warning(f"Content blocked for {result.get('custom_id', 'unknown')}: {block_reason}")
                            result['perturbation_data'] = {
                                'perturbed_text': '',
                                'success': False,
                                'perturbation_description': f'Content blocked by API: {block_reason}',
                                'perturbation_degree': 0,
                                'error': f"Content filter: {block_reason}"
                            }
                            failed_results.append(result.get('custom_id', f'line_{line_num}'))
                            results.append(result)
                            continue
                    
                    # Extract response from Google batch format
                    if 'response' in result and 'candidates' in result['response']:
                        candidates = result['response']['candidates']
                        if candidates and len(candidates) > 0:
                            content = candidates[0].get('content', {})
                            parts = content.get('parts', [])
                            if parts and len(parts) > 0:
                                text = parts[0].get('text', '')
                                
                                # Check for empty response
                                if not text or text.strip() == '':
                                    self.logger.warning(f"Empty response for {result.get('custom_id', 'unknown')}")
                                    result['perturbation_data'] = {
                                        'perturbed_text': '',
                                        'success': False,
                                        'perturbation_description': 'Empty API response',
                                        'perturbation_degree': 0,
                                        'error': 'API returned empty response'
                                    }
                                    failed_results.append(result.get('custom_id', f'line_{line_num}'))
                                    results.append(result)
                                    continue
                                
                                # Try to parse as JSON first, then fall back to text parsing
                                try:
                                    # Clean the text before JSON parsing (remove markdown code blocks if present)
                                    clean_text = text.strip()
                                    if clean_text.startswith('```json'):
                                        clean_text = clean_text[7:]  # Remove ```json
                                    if clean_text.endswith('```'):
                                        clean_text = clean_text[:-3]  # Remove trailing ```
                                    clean_text = clean_text.strip()
                                    
                                    parsed_result = json.loads(clean_text)
                                    
                                    # If JSON parsing succeeds, extract data
                                    result['perturbation_data'] = {
                                        'perturbed_text': parsed_result.get('perturbed_text', ''),
                                        'success': parsed_result.get('success', True),
                                        'perturbation_description': parsed_result.get('perturbation_description', ''),
                                        'perturbation_degree': parsed_result.get('perturbation_degree', 1),
                                        'error': parsed_result.get('error')
                                    }
                                    self.logger.debug(f"Successfully parsed JSON for {result.get('custom_id', 'unknown')}")
                                except (json.JSONDecodeError, ValueError) as e:
                                    # If JSON parsing fails, check if the response looks like JSON but is malformed
                                    if text.strip().startswith('{') and 'perturbed_text' in text:
                                        # Check if response was likely truncated
                                        is_truncated = not text.strip().endswith('}') or text.count('{') != text.count('}')
                                        
                                        if is_truncated:
                                            self.logger.warning(f"Response appears truncated (likely hit token limit) for {result.get('custom_id', 'unknown')}")
                                        else:
                                            self.logger.warning(f"JSON parsing failed for {result.get('custom_id', 'unknown')}, but response looks like JSON: {str(e)[:100]}")
                                        
                                        # Try to extract perturbed_text using regex as fallback
                                        import re
                                        match = re.search(r'"perturbed_text":\s*"((?:[^"\\]|\\.)*)"', text, re.DOTALL)
                                        if match:
                                            extracted_text = match.group(1)
                                            # Properly decode JSON escape sequences
                                            extracted_text = extracted_text.replace('\\"', '"')
                                            extracted_text = extracted_text.replace('\\n', '\n')
                                            extracted_text = extracted_text.replace('\\r', '\r')
                                            extracted_text = extracted_text.replace('\\\\', '\\')
                                            
                                            # Fix excessive escaped tabs that Gemini sometimes generates
                                            # Replace literal \t (two characters) with a space if it appears excessively
                                            if extracted_text.count('\\t') > 10:  # If there are many \t sequences
                                                extracted_text = re.sub(r'(\\t)+', ' ', extracted_text)
                                            else:
                                                extracted_text = extracted_text.replace('\\t', '\t')
                                            
                                            # Fix common UTF-8 encoding issues
                                            extracted_text = extracted_text.replace('‚Äô', "'")  # Smart apostrophe
                                            extracted_text = extracted_text.replace('‚Äú', '"')  # Smart quote open
                                            extracted_text = extracted_text.replace('‚Äù', '"')  # Smart quote close
                                            extracted_text = extracted_text.replace('‚Äì', '—')  # Em dash
                                            
                                            # Fix excessive tab/whitespace issues (Gemini JSON generation bug)
                                            extracted_text = re.sub(r'\t{3,}', ' ', extracted_text)      # Replace 3+ tabs with single space
                                            extracted_text = re.sub(r'\n\t+', '\n', extracted_text)      # Remove tabs after newlines
                                            extracted_text = re.sub(r'\s{4,}', ' ', extracted_text)      # Replace 4+ spaces with single space
                                            extracted_text = re.sub(r'([a-zA-Z])\t+([a-zA-Z])', r'\1 \2', extracted_text)  # Replace tabs between words with spaces
                                            
                                            result['perturbation_data'] = {
                                                'perturbed_text': extracted_text,
                                                'success': True,
                                                'perturbation_description': 'Extracted from truncated/malformed JSON',
                                                'perturbation_degree': 1,
                                                'error': f'JSON parse error (possibly truncated): {str(e)}'
                                            }
                                        else:
                                            # Complete fallback - treat as raw text
                                            result['perturbation_data'] = {
                                                'perturbed_text': text.strip(),
                                                'success': False,
                                                'perturbation_description': 'Raw response (JSON parse failed)',
                                                'perturbation_degree': 1,
                                                'error': f'JSON parse error: {str(e)}'
                                            }
                                    else:
                                        # Parse text response (model responded with plain text instead of JSON)
                                        result['perturbation_data'] = {
                                            'perturbed_text': text.strip(),
                                            'success': True,  # Assume success if we got a response
                                            'perturbation_description': 'Perturbation applied (text response)',
                                            'perturbation_degree': 1,
                                            'error': None
                                        }
                            else:
                                # No parts in content
                                self.logger.warning(f"No parts in content for {result.get('custom_id', 'unknown')}")
                                result['perturbation_data'] = {
                                    'perturbed_text': '',
                                    'success': False,
                                    'perturbation_description': 'No content parts in API response',
                                    'perturbation_degree': 0,
                                    'error': 'API response missing content parts'
                                }
                                failed_results.append(result.get('custom_id', f'line_{line_num}'))
                        else:
                            # No candidates in response
                            self.logger.warning(f"No candidates in response for {result.get('custom_id', 'unknown')}")
                            result['perturbation_data'] = {
                                'perturbed_text': '',
                                'success': False,
                                'perturbation_description': 'No candidates in API response',
                                'perturbation_degree': 0,
                                'error': 'API response missing candidates'
                            }
                            failed_results.append(result.get('custom_id', f'line_{line_num}'))
                    else:
                        # No response field at all
                        self.logger.warning(f"No response field for {result.get('custom_id', 'unknown')}")
                        result['perturbation_data'] = {
                            'perturbed_text': '',
                            'success': False,
                            'perturbation_description': 'No response from API',
                            'perturbation_degree': 0,
                            'error': 'API returned no response field'
                        }
                        failed_results.append(result.get('custom_id', f'line_{line_num}'))
                    
                    results.append(result)
        
        self.logger.info(f"Processed {len(results)} results")
        
        # Match results with metadata
        matched_results = []
        for result in results:
            custom_id = result.get('custom_id', '')
            
            # Find matching metadata
            matching_metadata = None
            for metadata in self.batch_metadata:
                if metadata['custom_id'] == custom_id:
                    matching_metadata = metadata
                    break
            
            if matching_metadata:
                # Calculate processing time (approximate for batch)
                start_time = matching_metadata.get('start_time', time.time())
                processing_time = time.time() - start_time
                
                # Combine result with original metadata and add missing fields
                combined_result = {
                    **matching_metadata['row_data'],
                    **result.get('perturbation_data', {}),
                    'processing_time': processing_time,
                    'custom_id': custom_id
                }
                
                # Add gender_swap_makes_sense logic for gender_swap perturbations
                if combined_result.get('perturbation_type') == 'gender_swap':
                    # This should be determined by the perturbation logic, but for now set to False by default
                    if not combined_result.get('gender_swap_makes_sense'):
                        combined_result['gender_swap_makes_sense'] = False
                
                matched_results.append(combined_result)
            else:
                self.logger.warning(f"No metadata found for custom_id: {custom_id}")
        
        # Convert to DataFrame
        df_results = pd.DataFrame(matched_results)
        
        # Clean up columns
        columns_to_drop = ['custom_id']
        df_results = df_results.drop(columns=[col for col in columns_to_drop if col in df_results.columns])
        
        # Reorder columns to match original perturber output
        # Original order: ['text_index', 'scenario_name', 'perturbation_type', 'old_index', 'id',
        #                  'title', 'author', 'created_utc', 'score', 'url', 'permalink',
        #                  'n_comments', 'n_verdicts', 'model', 'perturbed_text',
        #                  'perturbation_description', 'perturbation_degree',
        #                  'gender_swap_makes_sense', 'success', 'error', 'processing_time']
        preferred_order = [
            'text_index', 'scenario_name', 'perturbation_type', 'old_index', 'id',
            'title', 'author', 'created_utc', 'score', 'url', 'permalink',
            'n_comments', 'n_verdicts', 'model', 'perturbed_text',
            'perturbation_description', 'perturbation_degree',
            'gender_swap_makes_sense', 'success', 'error', 'processing_time'
        ]
        
        # Reorder columns, keeping any extra columns at the end
        existing_columns = list(df_results.columns)
        ordered_columns = []
        
        # Add columns in preferred order if they exist
        for col in preferred_order:
            if col in existing_columns:
                ordered_columns.append(col)
                existing_columns.remove(col)
        
        # Add any remaining columns
        ordered_columns.extend(existing_columns)
        
        # Reorder DataFrame
        df_results = df_results[ordered_columns]
        
        # Save final results
        # Clean text data before saving to prevent encoding issues
        for col in df_results.columns:
            if df_results[col].dtype == 'object':  # String columns
                df_results[col] = df_results[col].astype(str).replace({
                    '‚Äô': "'",  # Smart apostrophe
                    '‚Äú': '"',  # Smart quote open  
                    '‚Äù': '"',  # Smart quote close
                    '‚Äì': '—'   # Em dash
                })
                # Fix excessive whitespace in string columns (Gemini JSON generation bug)
                df_results[col] = df_results[col].str.replace(r'\t{3,}', ' ', regex=True)      # Replace 3+ tabs with space
                df_results[col] = df_results[col].str.replace(r'\n\t+', '\n', regex=True)      # Remove tabs after newlines
                df_results[col] = df_results[col].str.replace(r'\s{4,}', ' ', regex=True)      # Replace 4+ spaces with single space
                df_results[col] = df_results[col].str.replace(r'([a-zA-Z])\t+([a-zA-Z])', r'\1 \2', regex=True)  # Replace tabs between words
        
        df_results.to_csv(output_file, index=False, encoding='utf-8')
        self.logger.info(f"✅ Saved final results to: {output_file}")
        
        # Print summary
        print(f"\n📊 PERTURBATION SUMMARY:")
        print(f"Total perturbations generated: {len(df_results)}")
        print(f"Output file: {output_file}")
        
        # Show success rate
        if 'success' in df_results.columns:
            # Convert success column to boolean then to int for calculation
            success_rate = df_results['success'].astype(bool).astype(int).mean() * 100
            failed_count = len(df_results[df_results['success'] == False])
            print(f"Success rate: {success_rate:.1f}%")
            if failed_count > 0:
                print(f"⚠️  Failed perturbations: {failed_count}")
        
        # Show perturbation type distribution
        if 'perturbation_type' in df_results.columns:
            print(f"\nPerturbation type distribution:")
            type_dist = df_results['perturbation_type'].value_counts()
            for ptype, count in type_dist.items():
                print(f"  {ptype}: {count}")
        
        # If there were failures, report them with categorization
        if failed_results:
            print(f"\n⚠️  WARNING: {len(failed_results)} perturbations failed!")
            
            # Categorize failures by error type
            if 'error' in df_results.columns:
                error_types = df_results[df_results['success'] == False]['error'].value_counts()
                if not error_types.empty:
                    print(f"\nFailure breakdown:")
                    for error_type, count in error_types.items():
                        if 'Content filter' in str(error_type):
                            print(f"  🚫 Content blocked: {count}")
                        elif 'empty response' in str(error_type).lower():
                            print(f"  ⚠️  Empty responses: {count}")
                        else:
                            print(f"  ❌ {error_type}: {count}")
            
            print(f"\nFailed custom IDs (first 10):")
            for failed_id in failed_results[:10]:
                print(f"  - {failed_id}")
            if len(failed_results) > 10:
                print(f"  ... and {len(failed_results) - 10} more")
            
            # Save failed IDs to a file for potential retry
            failed_file = Path(output_file).with_suffix('.failed_ids.txt')
            with open(failed_file, 'w') as f:
                for failed_id in failed_results:
                    f.write(f"{failed_id}\n")
            print(f"\nFailed IDs saved to: {failed_file}")
            print(f"You can use these IDs to create a retry batch for failed perturbations.")
            
            # Special note for content filter blocks
            content_blocked = [fid for fid in failed_results if 'PROHIBITED' in str(fid) or any(
                'Content filter' in str(err) for _, err in 
                df_results[df_results['success'] == False][['error']].iterrows() if err.name in failed_results
            )]
            if content_blocked:
                print(f"\n💡 Note: Some content was blocked by Google's safety filters.")
                print(f"   This typically happens with explicit language in certain perturbation types.")
        
        return output_file