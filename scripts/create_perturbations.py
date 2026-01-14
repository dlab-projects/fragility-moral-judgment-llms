#!/usr/bin/env python3
"""
Streaming API perturbation script for ethical dilemmas with metadata preservation.

Uses Google's streaming API for real-time perturbation generation with:
- Real-time processing and progress tracking
- Enhanced metadata preservation for result matching
- Parallel processing support for faster execution
- Resume capability for interrupted sessions

This script preserves identifying columns from the original dataset including:
- old_index, id, title, author, created_utc, score
- url, permalink, n_comments, n_verdicts
- and any other specified metadata columns

Usage Examples:
    # Basic perturbation with single type
    python scripts/perturbations/create_perturbations.py --perturbations add_extraneous_detail:flash:1

    # Multiple perturbations with different models and priorities
    python scripts/perturbations/create_perturbations.py --perturbations add_emotion:flash:1 change_trivial:pro:2 firstperson:flash:1

    # Process subset with custom metadata preservation
    python scripts/perturbations/create_perturbations.py --perturbations none:flash:1 --sample-size 100 --metadata-columns id title author score

    # Parallel processing for faster execution
    python scripts/perturbations/create_perturbations.py --perturbations gender_swap:flash:1 --parallel --max-workers 8

    # Resume interrupted session
    python scripts/perturbations/create_perturbations.py --perturbations add_emotion:flash:1 --cache-dir cache/

    # Verbose logging for debugging
    python scripts/perturbations/create_perturbations.py --perturbations firstperson:flash:1 --verbose

Perturbation Format:
    type:model:priority
    - type: Perturbation type (e.g., add_emotion, firstperson, gender_swap)
    - model: Model short name (flash, pro, o1, claude35, claude37)
    - priority: Integer priority (1=highest, higher numbers = lower priority)

Available Models:
    - flash: gemini-2.5-flash (fast, cost-effective)
    - pro: gemini-2.5-pro (higher quality)
    - o1: gpt-4o (OpenAI)
    - claude35: claude-3-5-sonnet-20241022
    - claude37: claude-3-7-haiku-20241022

Common Perturbation Types:
    Content Variations:
    - add_extraneous_detail: Add morally irrelevant details
    - add_emotional_language: Add emotional descriptors
    - change_trivial_detail: Modify unimportant specifics
    - remove_sentence: Remove one sentence
    
    Perspective Changes:
    - firstperson: Convert to first-person narration
    - thirdperson: Convert to third-person narration
    - gender_swap: Swap gender pronouns
    
    Special:
    - none: No perturbation (baseline/control)

Output:
    - CSV file with perturbed scenarios and preserved metadata
    - Automatic timestamp in filename if not specified
    - Progress tracking and error reporting
    - Resume capability via cache files
"""

import os
import sys
from pathlib import Path
import argparse
from datetime import datetime
from typing import List
import pandas as pd

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from llm_evaluations_everyday_dilemmas.perturber import (
    create_perturbed_dataset,
    PerturbationConfig,
    DilemmaPerturber
)
from llm_evaluations_everyday_dilemmas.config import PRESENTATION_TEMPLATES, VARIATION_TEMPLATES


def parse_perturbation_spec(spec: str) -> PerturbationConfig:
    """Parse perturbation specification string."""
    parts = spec.split(':')
    
    # Special case: allow bare 'none' without model specification
    if len(parts) == 1 and parts[0] == 'none':
        perturbation_type = 'none'
        model = 'flash'
        priority = 1
    elif len(parts) < 2:
        raise ValueError(f"Invalid perturbation spec: {spec}. Format: 'type:model' or 'type:model:priority'")
    else:
        perturbation_type = parts[0]
        model = parts[1]
        priority = 1
    
    # Handle priority if specified
    if len(parts) == 3:
        try:
            priority = int(parts[2])
        except ValueError:
            raise ValueError(f"Invalid priority in spec: {spec}. Must be integer.")
    
    # Auto-detect perturbation type
    is_format_perturbation = False
    if perturbation_type in PRESENTATION_TEMPLATES:
        is_format_perturbation = True
    elif perturbation_type in VARIATION_TEMPLATES:
        is_format_perturbation = False
    elif perturbation_type == 'none':
        is_format_perturbation = False
    else:
        all_perturbations = list(VARIATION_TEMPLATES.keys()) + list(PRESENTATION_TEMPLATES.keys())
        raise ValueError(f"Unknown perturbation: {perturbation_type}. Available: {all_perturbations[:10]}...")
    
    # Validate model
    available_models = DilemmaPerturber.AVAILABLE_MODELS
    if model not in available_models:
        raise ValueError(f"Unknown model: {model}. Available: {list(available_models.keys())}")
    
    model_name = available_models[model]
    
    return PerturbationConfig(
        perturbation_type=perturbation_type,
        model=model_name,
        is_format_perturbation=is_format_perturbation,
        priority=priority
    )


def main():
    parser = argparse.ArgumentParser(
        description="Streaming API perturbation script with metadata preservation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic single perturbation
  python create_perturbations.py --perturbations add_extraneous_detail:flash:1

  # Multiple perturbations with different models
  python create_perturbations.py --perturbations add_emotion:flash:1 change_trivial:pro:2 firstperson:flash:1

  # Process subset with parallel processing
  python create_perturbations.py --perturbations gender_swap:flash:1 --sample-size 100 --parallel --max-workers 8

  # Custom metadata preservation
  python create_perturbations.py --perturbations none:flash:1 --metadata-columns id title author score

  # Resume interrupted session with verbose logging
  python create_perturbations.py --perturbations add_emotion:flash:1 --cache-dir cache/ --verbose

Perturbation Format: type:model:priority
  Available models: flash, pro, o1, claude35, claude37
  Common types: add_extraneous_detail, add_emotional_language, firstperson, thirdperson, gender_swap, none
"""
    )
    
    # Input/Output options
    parser.add_argument("--input-file", default="candidates.csv", 
                       help="Input CSV file (default: candidates.csv)")
    parser.add_argument("--output-file", 
                       help="Output CSV file (default: auto-generated with timestamp)")
    
    # Perturbation configuration
    parser.add_argument("--perturbations", nargs="+", required=True,
                       help="Perturbation specifications in format 'type:model:priority'")
    
    # Metadata columns to preserve
    parser.add_argument("--metadata-columns", nargs="+",
                       default=["old_index", "id", "title", "author", "created_utc", "score", 
                               "url", "permalink", "n_comments", "n_verdicts"],
                       help="Columns from input to preserve in output (default: key identifiers)")
    
    # Processing options
    parser.add_argument("--sample-size", type=int, 
                       help="Number of dilemmas to process (default: all)")
    parser.add_argument("--temperature", type=float, default=0.4,
                       help="Model temperature (default: .4)")
    parser.add_argument("--cache-dir", 
                       help="Directory for result caching")
    parser.add_argument("--no-resume", action="store_true",
                       help="Don't resume from existing temp file")
    parser.add_argument("--clear-cache", action="store_true",
                       help="Clear all cached results before processing")
    parser.add_argument("--parallel", action="store_true",
                       help="Run perturbations in parallel for faster processing")
    parser.add_argument("--max-workers", type=int, default=4,
                       help="Maximum concurrent workers for parallel processing")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Show verbose progress")
    
    args = parser.parse_args()

    # Check API key
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Error: Set GOOGLE_API_KEY environment variable")
        return

    # Parse perturbation configurations
    try:
        perturbation_configs = []
        for spec in args.perturbations:
            config = parse_perturbation_spec(spec)
            perturbation_configs.append(config)
        
        # Sort by priority
        perturbation_configs.sort(key=lambda x: x.priority)
        
    except ValueError as e:
        print(f"Error parsing perturbations: {e}")
        return

    # Setup paths
    data_dir = Path("data")
    input_file = data_dir / args.input_file
    
    if not input_file.exists():
        print(f"Error: {input_file} not found")
        return

    # Create output file with timestamp if not specified
    if args.output_file:
        output_file = data_dir / args.output_file
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = data_dir / f"perturbed_with_metadata_{timestamp}.csv"

    # Display configuration
    print("🚀 DilemmaPerturber with Metadata Preservation")
    print("=" * 70)
    print(f"Input: {input_file}")
    print(f"Output: {output_file}")
    print(f"Sample size: {args.sample_size or 'All dilemmas'}")
    print(f"Temperature: {args.temperature}")
    print(f"Metadata columns to preserve: {', '.join(args.metadata_columns)}")
    
    print(f"\n📋 Perturbation Configuration:")
    for i, config in enumerate(perturbation_configs, 1):
        model_short = config.model.split('-')[-1] if '-' in config.model else config.model
        print(f"  {i}. {config.perturbation_type} → {model_short} [priority: {config.priority}]")
    
    if args.parallel:
        print(f"\n⚡ Concurrent Processing: {args.max_workers} workers")
    
    print("\n" + "=" * 70)

    # Set logging level for verbose output
    if args.verbose:
        import logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Run enhanced perturbation with metadata preservation
    try:
        result_df, metrics_report = create_perturbed_dataset(
            input_file=str(input_file),
            output_file=str(output_file),
            api_key=api_key,
            perturbation_configs=perturbation_configs,
            metadata_columns=args.metadata_columns,
            max_samples=args.sample_size,
            cache_dir=args.cache_dir,
            temperature=args.temperature,
            resume=not args.no_resume,
            clear_cache=args.clear_cache,
            parallel=args.parallel,
            max_workers=args.max_workers
        )

        if not result_df.empty:
            success_rate = result_df['success'].mean() * 100
            
            print(f"\n✅ STREAMING COMPLETE")
            print(f"Generated {len(result_df)} perturbations")
            print(f"Success rate: {success_rate:.1f}%")
            print(f"Saved to: {output_file}")
            
            # Show preserved metadata columns
            preserved_cols = [col for col in args.metadata_columns if col in result_df.columns]
            print(f"\n📋 Preserved metadata columns: {', '.join(preserved_cols)}")
            
            # Show sample results with metadata
            print(f"\n📝 SAMPLE RESULTS WITH METADATA")
            sample_cols = ['text_index', 'scenario_name', 'perturbation_type'] + preserved_cols[:3]
            print(result_df[sample_cols].head(5).to_string())
            
        else:
            print("❌ No perturbations generated")
            
    except KeyboardInterrupt:
        print(f"\n⚠️  Generation interrupted by user")
        
    except Exception as e:
        print(f"\n❌ Error during generation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()