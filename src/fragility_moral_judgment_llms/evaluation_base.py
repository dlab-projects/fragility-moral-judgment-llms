"""
Base classes and utilities for LLM evaluation scripts.

This module provides common functionality shared across all provider-specific
evaluation scripts, including data validation, prompt building, and result processing.
"""

import pandas as pd
import logging
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
from abc import ABC, abstractmethod
import hashlib
import json
from datetime import datetime
from pydantic import BaseModel

from .prompt_builder import PromptBuilder
from .evaluation_types import EvaluationResponse, standardize_judgment


def suggest_output_filename(input_file: str, provider: str, model: str, base_output_dir: str = "results") -> str:
    """
    Suggest an appropriate output filename based on input file and provider.
    
    For single-perturbation files, includes the perturbation type in the output name.
    
    Args:
        input_file: Path to input CSV file
        provider: Provider name (openai, anthropic, google)
        model: Model name
        base_output_dir: Base directory for outputs
        
    Returns:
        Suggested output file path
    """
    input_path = Path(input_file)
    input_stem = input_path.stem
    
    # Create output directory
    output_dir = Path(base_output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Clean model name for filename
    clean_model = model.replace("-", "_").replace(".", "_")
    
    # Create filename - include input stem which likely contains perturbation info
    filename = f"{provider}_{clean_model}_{input_stem}_{timestamp}.csv"
    
    return str(output_dir / filename)


class EvaluationConfig(BaseModel):
    """Configuration for evaluation runs."""
    provider: str
    model: str
    temperature: float = 0.4
    max_retries: int = 3
    batch_size: int = 1
    dry_run: bool = False
    
    # Filtering options
    perturbation_type: Optional[str] = None
    sample_size: Optional[int] = None
    offset: int = 0
    
    # Output options
    output_file: str
    resume: bool = True
    save_interval: int = 10
    
    # Input tracking
    input_file: Optional[str] = None


class BaseEvaluator(ABC):
    """
    Abstract base class for provider-specific evaluators.
    
    Provides common functionality for data loading, validation, prompt building,
    and result processing that's shared across all providers.
    """
    
    def __init__(self, config: EvaluationConfig, api_key: str):
        self.config = config
        self.api_key = api_key
        self.prompt_builder = PromptBuilder(provider=self.config.provider)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        
        # Result tracking
        self.completed_results: List[Dict[str, Any]] = []
        self.failed_results: List[Dict[str, Any]] = []
        
    def load_and_validate_data(self, input_file: str) -> pd.DataFrame:
        """
        Load and validate the input perturbed dataset.
        
        Args:
            input_file: Path to the perturbed scenarios CSV
            
        Returns:
            Validated DataFrame
            
        Raises:
            ValueError: If data validation fails
        """
        try:
            df = pd.read_csv(input_file)
            self.logger.info(f"Loaded {len(df):,} rows from {input_file}")
        except Exception as e:
            raise ValueError(f"Failed to load input file {input_file}: {e}")
        
        if df.empty:
            raise ValueError("Input file is empty")
            
        # Validate required columns
        required_cols = {'perturbation_type', 'perturbed_text'}
        if missing := required_cols - set(df.columns):
            raise ValueError(f"Missing required columns: {missing}")
        
        # Check perturbation types in the file
        available_types = list(df['perturbation_type'].unique())
        
        # For single-perturbation files (expected use case)
        if len(available_types) == 1:
            perturbation_type = available_types[0]
            self.logger.info(f"Processing file with perturbation type: '{perturbation_type}'")
            self.logger.info(f"Scenarios: {len(df):,}")
        else:
            # Multiple types - warn but continue
            self.logger.warning(f"Multiple perturbation types found: {sorted(available_types)}")
            self.logger.warning("This script is designed for single-perturbation CSV files")
            self.logger.warning("Consider splitting your data by perturbation type first")
        
        # Apply filters
        original_count = len(df)
        
        # Optional filtering by perturbation type if specified
        if self.config.perturbation_type:
            df = df[df['perturbation_type'] == self.config.perturbation_type]
            filtered_count = len(df)
            
            if filtered_count == 0:
                raise ValueError(
                    f"No data found for perturbation type '{self.config.perturbation_type}'. "
                    f"Available types: {', '.join(sorted(available_types))}"
                )
            
            if filtered_count < original_count:
                self.logger.info(
                    f"Filtered to {filtered_count:,} rows with perturbation_type='{self.config.perturbation_type}'"
                )
        
        # Apply sampling
        if self.config.sample_size or self.config.offset > 0:
            df = self._apply_sampling(df)
        
        self.logger.info(f"Final dataset: {len(df):,} rows to process")
        return df
    
    def _apply_sampling(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply offset and sample size filtering to the dataset."""
        if 'scenario_name' in df.columns:
            # Sample by unique scenarios, not rows
            unique_scenarios = df['scenario_name'].unique()
            total_scenarios = len(unique_scenarios)
            
            if self.config.offset >= total_scenarios:
                raise ValueError(f"Offset {self.config.offset} >= total scenarios {total_scenarios}")
            
            start_idx = self.config.offset
            if self.config.sample_size:
                end_idx = min(self.config.offset + self.config.sample_size, total_scenarios)
                selected_scenarios = unique_scenarios[start_idx:end_idx]
                self.logger.info(f"Sampling scenarios {start_idx+1}-{end_idx} ({len(selected_scenarios)} scenarios)")
            else:
                selected_scenarios = unique_scenarios[start_idx:]
                self.logger.info(f"Processing scenarios {start_idx+1}-{total_scenarios} ({len(selected_scenarios)} scenarios)")
            
            df = df[df['scenario_name'].isin(selected_scenarios)]
        else:
            # Fallback to row-based sampling
            if self.config.sample_size:
                end_idx = min(self.config.offset + self.config.sample_size, len(df))
                df = df.iloc[self.config.offset:end_idx]
                self.logger.info(f"Sampling rows {self.config.offset}-{end_idx-1}")
        
        return df
    
    def build_evaluation_prompt(self, row: pd.Series, persona: Optional[str] = None) -> str:
        """
        Build evaluation prompt for a scenario row.
        
        Args:
            row: DataFrame row containing scenario data
            persona: Optional persona for evaluation
            
        Returns:
            Complete evaluation prompt
        """
        # Determine framework type based on perturbation type
        perturbation_type = row.get('perturbation_type', 'none')
        if perturbation_type in ['firstperson', 'firstperson_inthewrong', 'firstperson_atfault', 'thirdperson']:
            framework_type = perturbation_type
        else:
            framework_type = row.get('format_type', 'aita')
        
        return self.prompt_builder.build_prompt(
            dilemma_text=str(row['perturbed_text']),
            framework_type=framework_type,
            persona=persona
        )
    
    def create_content_hash(self, text: str) -> str:
        """Create a hash of text content for deduplication and matching."""
        normalized = text.strip().lower()
        return hashlib.md5(normalized.encode('utf-8')).hexdigest()[:16]
    
    def create_scenario_metadata(self, row: pd.Series) -> Dict[str, Any]:
        """Extract and preserve metadata from a scenario row."""
        metadata = {}
        
        # Always preserve these key fields
        preserve_fields = [
            'text_index', 'scenario_name', 'perturbation_type', 'perturbed_text',
            'perturbation_description', 'old_index', 'id', 'title', 'author',
            'created_utc', 'score', 'url', 'permalink', 'n_comments'
        ]
        
        for field in preserve_fields:
            if field in row.index:
                metadata[field] = row[field]
        
        # Add content hash for robust matching
        metadata['content_hash'] = self.create_content_hash(str(row.get('perturbed_text', '')))
        
        return metadata
    
    def setup_resume_capability(self, output_file: str) -> tuple[List[Dict[str, Any]], int]:
        """
        Setup resume capability by loading existing results.
        
        Args:
            output_file: Path to the output file
            
        Returns:
            Tuple of (existing_results, start_index)
        """
        temp_file = Path(output_file).with_suffix('.temp.csv')
        existing_results = []
        start_idx = 0
        
        if self.config.resume and temp_file.exists():
            try:
                existing_df = pd.read_csv(temp_file)
                if not existing_df.empty:
                    existing_results = existing_df.to_dict('records')
                    start_idx = len(existing_results)
                    self.logger.info(f"Resuming: Found {len(existing_results)} existing results")
            except Exception as e:
                self.logger.warning(f"Could not resume from temp file: {e}. Starting fresh.")
        
        return existing_results, start_idx
    
    def save_results_incrementally(self, results: List[Dict[str, Any]], output_file: str, is_final: bool = False):
        """Save results incrementally to prevent data loss."""
        if not results:
            return
            
        df = pd.DataFrame(results)
        
        if is_final:
            # Final save to the actual output file
            df.to_csv(output_file, index=False)
            
            # Clean up temp file
            temp_file = Path(output_file).with_suffix('.temp.csv')
            if temp_file.exists():
                temp_file.unlink()
                
            self.logger.info(f"Final results saved to {output_file}")
        else:
            # Incremental save to temp file
            temp_file = Path(output_file).with_suffix('.temp.csv')
            df.to_csv(temp_file, index=False)
    
    def validate_perturbation_types(self, df: pd.DataFrame) -> bool:
        """
        Validate that the dataset contains a single perturbation type.
        
        This is critical for preventing costly mistakes in batch processing.
        """
        actual_types = list(df['perturbation_type'].unique())
        
        if len(actual_types) == 1:
            self.logger.info(f"✅ Validation passed: Single perturbation type '{actual_types[0]}'")
            return True
        else:
            self.logger.error(
                f"🚨 VALIDATION FAILED: Multiple perturbation types found: {sorted(actual_types)}"
            )
            self.logger.error(
                "This script expects CSV files with a single perturbation type."
            )
            self.logger.error(
                "Please split your data by perturbation type before running evaluations."
            )
            return False
    
    def process_evaluation_response(self, raw_response: Union[Dict[str, Any], EvaluationResponse], 
                                   metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process and standardize an evaluation response.
        
        Args:
            raw_response: Raw response from the LLM
            metadata: Scenario metadata to preserve
            
        Returns:
            Standardized result dictionary
        """
        result = metadata.copy()
        
        if isinstance(raw_response, EvaluationResponse):
            result.update(raw_response.model_dump())
        elif isinstance(raw_response, dict):
            result.update(raw_response)
        else:
            result.update({
                'judgment': 'ERROR',
                'explanation': f'Invalid response type: {type(raw_response)}'
            })
        
        # Add standardized judgment
        result['standardized_judgment'] = standardize_judgment(result.get('judgment', 'ERROR'))
        result['model_used'] = self.config.model
        result['temperature_used'] = self.config.temperature
        
        return result
    
    @abstractmethod
    def estimate_cost(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Estimate the cost of processing the given dataset."""
        pass
    
    @abstractmethod
    def run_evaluation(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Run the evaluation on the given dataset."""
        pass
    
    def execute(self, input_file: str) -> List[Dict[str, Any]]:
        """
        Execute the complete evaluation pipeline.
        
        Args:
            input_file: Path to input perturbed scenarios CSV
            
        Returns:
            List of evaluation results
        """
        self.logger.info(f"Starting {self.config.provider} evaluation")
        self.logger.info(f"Model: {self.config.model}")
        self.logger.info(f"Temperature: {self.config.temperature}")
        
        # Load and validate data
        df = self.load_and_validate_data(input_file)
        
        # Validate perturbation types (critical for batch processing)
        if not self.validate_perturbation_types(df):
            raise ValueError("Perturbation type validation failed")
        
        # Estimate cost and get user confirmation
        cost_info = self.estimate_cost(df)
        if not self.config.dry_run:
            self._confirm_cost(cost_info)
        
        # Run evaluation
        if self.config.dry_run:
            self.logger.info("DRY RUN MODE: Skipping actual evaluation")
            return []
        
        results = self.run_evaluation(df)
        
        # Save final results
        self.save_results_incrementally(results, self.config.output_file, is_final=True)
        
        self.logger.info(f"Evaluation complete: {len(results)} results")
        return results
    
    def _confirm_cost(self, cost_info: Dict[str, Any]):
        """Ask user to confirm the estimated cost before proceeding."""
        print("\n" + "="*60)
        print("🔥 COST ESTIMATION & CONFIRMATION")
        print("="*60)
        
        print(f"Provider: {self.config.provider}")
        print(f"Model: {self.config.model}")
        print(f"Total requests: {cost_info.get('total_requests', 'Unknown'):,}")
        
        if 'estimated_cost' in cost_info:
            print(f"Estimated cost: ${cost_info['estimated_cost']:.2f}")
            
            if cost_info.get('has_batch_discount'):
                print(f"Includes batch discount: {cost_info.get('batch_discount', 50)}%")
        
        print(f"\nPerturbation type filter: {self.config.perturbation_type or 'None (all types)'}")
        print(f"Sample size: {self.config.sample_size or 'All scenarios'}")
        
        print("\n⚠️  This will incur real API costs!")
        
        response = input("\nProceed with evaluation? (type 'yes' to confirm): ").strip().lower()
        if response != 'yes':
            print("Evaluation cancelled by user")
            exit(0)
        
        print("✅ User confirmed. Starting evaluation...")
        print("="*60)