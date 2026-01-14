"""
Result matching utilities for batch evaluation processing.

This module handles matching batch evaluation results back to their original perturbed scenarios.
"""

import pandas as pd
import json
import hashlib
import re
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
import logging

from .evaluation_types import standardize_judgment


class EnhancedResultMatcher:
    """
    Enhanced result matcher that preserves metadata and enables robust
    matching between batch results and original perturbed scenarios.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def create_enhanced_custom_id(self, 
                                 scenario_id: str,
                                 dilemma_idx: int, 
                                 run_num: int,
                                 content_hash: str,
                                 perturbation_type: str,
                                 exp_prefix: str = "") -> str:
        """
        Create enhanced custom ID that includes content hash for robust matching.
        
        Args:
            scenario_id: Unique scenario ID (from id column, e.g., Reddit post ID)
            dilemma_idx: Index in current batch
            run_num: Run number (1-based)
            content_hash: Hash of the perturbed text content
            perturbation_type: Type of perturbation applied
            exp_prefix: Experiment prefix
            
        Returns:
            Enhanced custom ID string
        """
        # Shorten custom_id to stay under 64 character limit
        # Use abbreviated perturbation type and shorter hash
        perturbation_mapping = {
            # Original perturbation types
            'remove_sentence': 'rmvsentc',
            'change_trivial_detail': 'chgtrivl',
            'add_emotional_language': 'addemotl',
            'firstperson': 'firstper',
            'thirdperson': 'thirdper',
            # New perturbation types - ensure unique abbreviations
            'gender_swap': 'gswap',
            'increase_status_narrator': 'incstatnar',
            'increase_status_other_party': 'incstatoth',
            'remove_emotional_language': 'rmemotl',
            'add_certainty_markers': 'addcert',
            'add_uncertainty_markers': 'adduncert',
            'change_relationship_type': 'chgreltyp',
            'escalation_narrator': 'escnar',
            'escalation_other_party': 'escoth',
            'perspective_switch': 'perspswt',
            'increase_intentionality_narrator': 'incintnar',
            'decrease_intentionality_narrator': 'decintnar',
            'assertive_user_framing': 'assertfr',
            'none': 'none',
            # Psychology perturbation types (push_*)
            'push_yta_self_condemning': 'pushytasc',
            'push_yta_social_proof': 'pushytasp',
            'push_yta_pattern_admission': 'pushytapa',
            'push_nta_self_justifying': 'pushntasj',
            'push_nta_social_proof': 'pushntasp',
            'push_nta_victim_pattern': 'pushntavp'
        }
        
        if perturbation_type in perturbation_mapping:
            pert_abbrev = perturbation_mapping[perturbation_type]
        elif len(perturbation_type) > 8:
            # Fallback for unknown types - ensure uniqueness by using hash suffix
            pert_abbrev = perturbation_type[:6] + hashlib.md5(perturbation_type.encode()).hexdigest()[:2]
        else:
            pert_abbrev = perturbation_type
        return f"{exp_prefix}{scenario_id}_{dilemma_idx}_r{run_num}_{pert_abbrev}_{content_hash[:4]}"
    
    def _expand_perturbation_type(self, pert_abbrev: str) -> str:
        """Expand truncated perturbation type back to full name."""
        expansion_map = {
            # New clean abbreviations (no underscores)
            'rmvsentc': 'remove_sentence',
            'chgtrivl': 'change_trivial_detail', 
            'addemotl': 'add_emotional_language',
            'firstper': 'firstperson',
            'thirdper': 'thirdperson',
            # New perturbation types - matching the mapping above
            'gswap': 'gender_swap',
            'incstatnar': 'increase_status_narrator',
            'incstatoth': 'increase_status_other_party',
            'rmemotl': 'remove_emotional_language',
            'addcert': 'add_certainty_markers',
            'adduncert': 'add_uncertainty_markers',
            'chgreltyp': 'change_relationship_type',
            'escnar': 'escalation_narrator',
            'escoth': 'escalation_other_party',
            'perspswt': 'perspective_switch',
            'incintnar': 'increase_intentionality_narrator',
            'decintnar': 'decrease_intentionality_narrator',
            'assertfr': 'assertive_user_framing',
            # Old problematic abbreviations (for backward compatibility)
            'remove_s': 'remove_sentence',
            'change_t': 'change_trivial_detail',
            'add_emot': 'add_emotional_language',
            'assert78': 'assertive_user_framing',  # Backward compatibility for existing batch
            'none': 'none',
            # Psychology perturbation types (push_*)
            'pushytasc': 'push_yta_self_condemning',
            'pushytasp': 'push_yta_social_proof',
            'pushytapa': 'push_yta_pattern_admission',
            'pushntasj': 'push_nta_self_justifying',
            'pushntasp': 'push_nta_social_proof',
            'pushntavp': 'push_nta_victim_pattern'
        }
        return expansion_map.get(pert_abbrev, pert_abbrev)

    def parse_enhanced_custom_id(self, custom_id: str) -> Dict[str, Any]:
        """
        Parse enhanced custom ID to extract metadata.
        
        Args:
            custom_id: Enhanced custom ID string
            
        Returns:
            Dictionary with parsed metadata
        """
        # Handle different custom ID formats for backward compatibility
        
        # Broken underscore format: a07071928_1hy6esj_3184_r1_remove_s_7d60
        # This happens when perturbation type gets truncated with underscore
        broken_underscore_pattern = r'([^_]+)_([^_]+)_(\d+)_r(\d+)_([^_]+)_([^_]+)_([a-f0-9]{4})'
        match = re.match(broken_underscore_pattern, custom_id)
        
        if match:
            exp_prefix, scenario_id, dilemma_idx, run_num, pert_part1, pert_part2, content_hash = match.groups()
            # Reconstruct the perturbation type
            perturbation_type = f"{pert_part1}_{pert_part2}"
            # Expand truncated perturbation type
            full_perturbation_type = self._expand_perturbation_type(perturbation_type)
            return {
                'scenario_id': scenario_id,
                'dilemma_idx': int(dilemma_idx), 
                'run_num': int(run_num),
                'perturbation_type': full_perturbation_type,
                'content_hash': content_hash,
                'exp_prefix': exp_prefix,
                'format': 'broken_underscore'
            }
        
        # Current shortened format: a07061611_1iudlf2_0_r1_firstper_63d9
        shortened_pattern = r'([^_]+)_([^_]+)_(\d+)_r(\d+)_([^_]+)_([a-f0-9]{4})'
        match = re.match(shortened_pattern, custom_id)
        
        if match:
            exp_prefix, scenario_id, dilemma_idx, run_num, perturbation_type, content_hash = match.groups()
            # Expand truncated perturbation type
            full_perturbation_type = self._expand_perturbation_type(perturbation_type)
            return {
                'scenario_id': scenario_id,
                'dilemma_idx': int(dilemma_idx), 
                'run_num': int(run_num),
                'perturbation_type': full_perturbation_type,
                'content_hash': content_hash,
                'exp_prefix': exp_prefix,
                'format': 'shortened'
            }
        
        # Old OpenAI format: openai_20250706_230415_1igyroj_3185_r1_thirdper_7190
        old_openai_pattern = r'openai_\d{8}_\d{6}_([^_]+)_(\d+)_r(\d+)_([^_]+)_([a-f0-9]{4})'
        match = re.match(old_openai_pattern, custom_id)
        
        if match:
            scenario_id, dilemma_idx, run_num, perturbation_type, content_hash = match.groups()
            # Expand truncated perturbation type
            full_perturbation_type = self._expand_perturbation_type(perturbation_type)
            return {
                'scenario_id': scenario_id,
                'dilemma_idx': int(dilemma_idx), 
                'run_num': int(run_num),
                'perturbation_type': full_perturbation_type,
                'content_hash': content_hash,
                'exp_prefix': 'openai',
                'format': 'old_openai'
            }
        
        # Legacy enhanced format: exp_eval_1iudlf2_0_r1_none_abc12345
        enhanced_pattern = r'(?:(.+?)_)?eval_([^_]+)_(\d+)_r(\d+)_([^_]+)_([a-f0-9]{8})'
        match = re.match(enhanced_pattern, custom_id)
        
        if match:
            exp_prefix, scenario_id, dilemma_idx, run_num, perturbation_type, content_hash = match.groups()
            
            # For backward compatibility, if scenario_id is numeric, treat it as old text_index format
            if scenario_id.isdigit():
                # This is the old format - scenario_id is actually text_index (row number)
                # We need to map it back to the actual scenario ID
                scenario_id = f"text_index_{scenario_id}"  # Use a temporary key for lookup
            
            return {
                'scenario_id': scenario_id,
                'dilemma_idx': int(dilemma_idx), 
                'run_num': int(run_num),
                'perturbation_type': perturbation_type,
                'content_hash': content_hash,
                'exp_prefix': exp_prefix or '',
                'format': 'enhanced'
            }
        
        # Legacy format: expeval_batch_dilemma_123_idx_456_run_1
        legacy_pattern = r'(?:(.+?)_)?dilemma_(\d+)(?:_idx_(\d+))?_run_(\d+)'
        match = re.match(legacy_pattern, custom_id)
        
        if match:
            exp_prefix, text_index, dilemma_idx, run_num = match.groups()
            return {
                'scenario_id': str(text_index),  # Convert to string for consistency
                'dilemma_idx': int(dilemma_idx) if dilemma_idx else 0,
                'run_num': int(run_num),
                'perturbation_type': None,  # Unknown in legacy format
                'content_hash': None,
                'exp_prefix': exp_prefix or '',
                'format': 'legacy'
            }
        
        # Fallback for unknown formats
        self.logger.warning(f"Could not parse custom_id: {custom_id}")
        return {
            'scenario_id': None,
            'dilemma_idx': None,
            'run_num': None,
            'perturbation_type': None,
            'content_hash': None,
            'exp_prefix': '',
            'format': 'unknown'
        }
    
    def create_content_hash(self, text: str) -> str:
        """Create a hash of text content for matching."""
        normalized = str(text).strip().lower()
        return hashlib.md5(normalized.encode('utf-8')).hexdigest()
    
    def build_scenario_lookup(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """
        Build lookup tables for matching batch results to original scenarios.
        
        Args:
            df: DataFrame with original perturbed scenarios
            
        Returns:
            Dictionary with multiple lookup strategies
        """
        lookup = {
            'by_scenario_id': {},
            'by_content_hash': {},
            'by_combined_key': {},
            'metadata_by_index': {}
        }
        
        for idx, row in df.iterrows():
            scenario_id = str(row.get('id', f'scenario_{idx}'))  # Use id column, fallback to row index
            content_hash = self.create_content_hash(str(row.get('perturbed_text', '')))
            perturbation_type = row.get('perturbation_type', 'unknown')
            
            # Create comprehensive metadata record
            metadata = {
                'perturbation_type': perturbation_type,
                'perturbed_text': str(row.get('perturbed_text', ''))
            }
            
            # Preserve all other columns except unwanted ones
            excluded_columns = {'model', 'success', 'error', 'processing_time'}
            for col in df.columns:
                if col not in metadata and col not in excluded_columns:
                    metadata[col] = row[col]
            
            # Multiple lookup strategies for robustness
            # IMPORTANT: Don't overwrite by_scenario_id for mixed perturbations - only use for single perturbation datasets
            if scenario_id not in lookup['by_scenario_id']:
                lookup['by_scenario_id'][scenario_id] = metadata
            
            lookup['by_content_hash'][content_hash] = metadata
            lookup['by_combined_key'][f"{scenario_id}_{perturbation_type}"] = metadata
            lookup['metadata_by_index'][idx] = metadata
            
            # BACKWARD COMPATIBILITY: Add lookup by old text_index format
            text_index_key = f"text_index_{idx}"
            if text_index_key not in lookup['by_scenario_id']:
                lookup['by_scenario_id'][text_index_key] = metadata
            lookup['by_combined_key'][f"{text_index_key}_{perturbation_type}"] = metadata
        
        self.logger.info(f"Built lookup tables for {len(df)} scenarios")
        return lookup
    
    def match_batch_result(self, 
                          batch_result: Dict[str, Any],
                          scenario_lookup: Dict[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Match a single batch result to its original scenario.
        
        Args:
            batch_result: Individual batch result with custom_id
            scenario_lookup: Lookup tables from build_scenario_lookup
            
        Returns:
            Matched scenario metadata or None if no match found
        """
        # Try both field names for compatibility
        custom_id = batch_result.get('custom_id') or batch_result.get('batch_custom_id', '')
        if not custom_id:
            self.logger.warning("Batch result missing custom_id or batch_custom_id")
            return None
        
        # Parse custom ID
        parsed = self.parse_enhanced_custom_id(custom_id)
        
        if parsed['format'] == 'unknown':
            return None
        
        # Try different matching strategies in order of preference
        
        # Strategy 1: Enhanced matching with content hash (most robust)
        if parsed['content_hash'] and parsed['content_hash'] in scenario_lookup['by_content_hash']:
            match = scenario_lookup['by_content_hash'][parsed['content_hash']]
            self.logger.info(f"✅ Matched by content hash: {custom_id}")
            return match
        
        # Strategy 2: Combined scenario_id + perturbation_type
        if parsed['scenario_id'] is not None and parsed['perturbation_type']:
            combined_key = f"{parsed['scenario_id']}_{parsed['perturbation_type']}"
            if combined_key in scenario_lookup['by_combined_key']:
                match = scenario_lookup['by_combined_key'][combined_key]
                self.logger.info(f"✅ Matched by combined key: {custom_id} -> {combined_key}")
                return match
            else:
                self.logger.warning(f"❌ Combined key not found: {combined_key} for {custom_id}")
        
        # Strategy 3: Scenario ID only (less robust, legacy compatibility)
        if parsed['scenario_id'] is not None and parsed['scenario_id'] in scenario_lookup['by_scenario_id']:
            match = scenario_lookup['by_scenario_id'][parsed['scenario_id']]
            self.logger.warning(f"⚠️ Matched by scenario_id only (fallback): {custom_id} -> {parsed['scenario_id']}")
            return match
        
        self.logger.warning(f"No match found for custom_id: {custom_id}")
        return None
    
    def process_batch_results(self,
                             batch_results: List[Dict[str, Any]], 
                             original_df: pd.DataFrame) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Process complete batch results and match them to original scenarios.
        
        Args:
            batch_results: List of batch result dictionaries
            original_df: Original DataFrame with perturbed scenarios
            
        Returns:
            Tuple of (matched_results, unmatched_results)
        """
        scenario_lookup = self.build_scenario_lookup(original_df)
        
        matched_results = []
        unmatched_results = []
        
        for batch_result in batch_results:
            matched_metadata = self.match_batch_result(batch_result, scenario_lookup)
            
            if matched_metadata:
                # Combine batch result with original metadata
                combined_result = matched_metadata.copy()
                
                # Add evaluation results (judgment, explanation, etc.)
                # Handle different response formats
                if 'response' in batch_result:
                    response_data = batch_result['response']
                    if isinstance(response_data, dict):
                        combined_result.update(response_data)
                    elif isinstance(response_data, str):
                        # Try to parse JSON response
                        try:
                            parsed_response = json.loads(response_data)
                            combined_result.update(parsed_response)
                        except json.JSONDecodeError:
                            # Handle plain text responses (common for Anthropic)
                            parsed_text = self._parse_text_response(response_data)
                            if parsed_text['judgment'] != 'ERROR':
                                combined_result.update(parsed_text)
                            else:
                                combined_result['raw_response'] = response_data
                                combined_result['judgment'] = 'ERROR'
                                combined_result['explanation'] = f'Failed to parse response: {response_data[:100]}'
                else:
                    # Handle already-processed results (OpenAI format)
                    # Copy all fields except metadata fields
                    for key, value in batch_result.items():
                        if key not in ['batch_custom_id', 'batch_status', 'openai_batch_id']:
                            combined_result[key] = value
                
                # Add batch metadata
                batch_custom_id = batch_result.get('custom_id') or batch_result.get('batch_custom_id', '')
                combined_result['batch_custom_id'] = batch_custom_id
                combined_result['batch_status'] = batch_result.get('status') or batch_result.get('batch_status', 'unknown')
                
                # Parse run number from custom_id
                parsed_id = self.parse_enhanced_custom_id(batch_custom_id)
                combined_result['run_number'] = parsed_id.get('run_num', 1)
                
                # Add standardized judgment
                raw_judgment = combined_result.get('judgment', '')
                standardized_judgment = standardize_judgment(raw_judgment)
                
                # Reorder to put standardized_judgment right after judgment
                if 'judgment' in combined_result:
                    ordered_result = {}
                    for key, value in combined_result.items():
                        ordered_result[key] = value
                        if key == 'judgment':
                            ordered_result['standardized_judgment'] = standardized_judgment
                    combined_result = ordered_result
                else:
                    combined_result['standardized_judgment'] = standardized_judgment
                
                matched_results.append(combined_result)
            else:
                unmatched_results.append(batch_result)
        
        self.logger.info(f"Matched {len(matched_results)} results, {len(unmatched_results)} unmatched")
        
        if unmatched_results:
            self.logger.warning(f"Failed to match {len(unmatched_results)} batch results")
            
        return matched_results, unmatched_results
    
    def _parse_text_response(self, response_text: str) -> Dict[str, str]:
        """Parse plain text response to extract judgment and explanation."""
        import re
        
        # Look for common judgment patterns at the beginning
        judgment_patterns = [
            r'^(YTA|NTA|ESH|NAH|INFO)\b',  # Standard AITA judgments
            r'\b(YTA|NTA|ESH|NAH|INFO)\b.*?(?:\.|$)',  # Judgment anywhere in first sentence
        ]
        
        judgment = 'ERROR'
        explanation = response_text.strip()
        
        # Try to extract judgment
        for pattern in judgment_patterns:
            match = re.search(pattern, response_text.strip(), re.IGNORECASE)
            if match:
                judgment = match.group(1).upper()
                # Use the full text as explanation, but clean it up
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
            'explanation': explanation
        }
    
    def save_unmatched_results(self, unmatched_results: List[Dict[str, Any]], output_file: str):
        """Save unmatched results for debugging."""
        if not unmatched_results:
            return
            
        unmatched_file = Path(output_file).with_suffix('.unmatched.json')
        with open(unmatched_file, 'w') as f:
            json.dump(unmatched_results, f, indent=2, default=str)
        
        self.logger.warning(f"Saved {len(unmatched_results)} unmatched results to {unmatched_file}")
    
    def validate_matching_completeness(self, 
                                     original_df: pd.DataFrame,
                                     matched_results: List[Dict[str, Any]],
                                     expected_runs_per_scenario: int = 1) -> Dict[str, Any]:
        """
        Validate that batch result matching is complete and accurate.
        
        Args:
            original_df: Original perturbed scenarios DataFrame
            matched_results: List of matched result dictionaries
            expected_runs_per_scenario: Expected number of runs per scenario
            
        Returns:
            Validation report dictionary
        """
        expected_total = len(original_df) * expected_runs_per_scenario
        actual_total = len(matched_results)
        
        # Count results by scenario
        scenario_counts = {}
        for result in matched_results:
            scenario_key = f"{result.get('id')}_{result.get('perturbation_type')}"
            scenario_counts[scenario_key] = scenario_counts.get(scenario_key, 0) + 1
        
        # Find missing scenarios
        missing_scenarios = []
        for _, row in original_df.iterrows():
            scenario_id = str(row.get('id', f'scenario_{row.name}'))
            scenario_key = f"{scenario_id}_{row.get('perturbation_type')}"
            if scenario_key not in scenario_counts:
                missing_scenarios.append(scenario_key)
        
        # Find scenarios with wrong count
        wrong_count_scenarios = []
        for scenario_key, count in scenario_counts.items():
            if count != expected_runs_per_scenario:
                wrong_count_scenarios.append((scenario_key, count))
        
        validation_report = {
            'expected_total': expected_total,
            'actual_total': actual_total,
            'match_rate': actual_total / expected_total if expected_total > 0 else 0,
            'missing_scenarios': missing_scenarios,
            'wrong_count_scenarios': wrong_count_scenarios,
            'is_complete': len(missing_scenarios) == 0 and len(wrong_count_scenarios) == 0
        }
        
        return validation_report
    
    def print_validation_report(self, validation_report: Dict[str, Any]):
        """Print a formatted validation report."""
        print("\n" + "="*50)
        print("📊 BATCH RESULT MATCHING VALIDATION")
        print("="*50)
        
        print(f"Expected results: {validation_report['expected_total']:,}")
        print(f"Matched results: {validation_report['actual_total']:,}")
        print(f"Match rate: {validation_report['match_rate']*100:.1f}%")
        
        if validation_report['is_complete']:
            print("✅ VALIDATION PASSED: All results matched successfully")
        else:
            print("❌ VALIDATION FAILED:")
            
            if validation_report['missing_scenarios']:
                print(f"  Missing scenarios: {len(validation_report['missing_scenarios'])}")
                for scenario in validation_report['missing_scenarios'][:5]:
                    print(f"    - {scenario}")
                if len(validation_report['missing_scenarios']) > 5:
                    print(f"    ... and {len(validation_report['missing_scenarios']) - 5} more")
            
            if validation_report['wrong_count_scenarios']:
                print(f"  Scenarios with wrong result count:")
                for scenario, count in validation_report['wrong_count_scenarios'][:5]:
                    print(f"    - {scenario}: {count} results")
                if len(validation_report['wrong_count_scenarios']) > 5:
                    print(f"    ... and {len(validation_report['wrong_count_scenarios']) - 5} more")
        
        print("="*50)


# Convenience functions for direct use
def create_enhanced_batch_metadata(df: pd.DataFrame, 
                                  exp_prefix: str = "",
                                  num_runs: int = 1) -> List[Dict[str, Any]]:
    """
    Create enhanced metadata for batch requests.
    
    Args:
        df: DataFrame with perturbed scenarios
        exp_prefix: Experiment prefix for custom IDs
        num_runs: Number of runs per scenario
        
    Returns:
        List of metadata dictionaries for batch requests
    """
    matcher = EnhancedResultMatcher()
    batch_metadata = []
    
    for run_num in range(num_runs):
        for dilemma_idx, (_, row) in enumerate(df.iterrows()):
            scenario_id = str(row.get('id', f'scenario_{dilemma_idx}'))
            content_hash = matcher.create_content_hash(str(row.get('perturbed_text', '')))
            perturbation_type = row.get('perturbation_type', 'unknown')
            
            custom_id = matcher.create_enhanced_custom_id(
                scenario_id=scenario_id,
                dilemma_idx=dilemma_idx,
                run_num=run_num + 1,
                content_hash=content_hash,
                perturbation_type=perturbation_type,
                exp_prefix=exp_prefix
            )
            
            metadata = {
                'custom_id': custom_id,
                'scenario_id': scenario_id,
                'dilemma_idx': dilemma_idx,
                'run_num': run_num + 1,
                'content_hash': content_hash,
                'perturbation_type': perturbation_type,
                'row_data': row.to_dict()
            }
            
            batch_metadata.append(metadata)
    
    return batch_metadata