#!/usr/bin/env python3
"""
Google Gemini Batch API perturbation script for ethical dilemmas.

Uses Google's batch mode for efficient large-scale perturbation generation with:
- 50% cost reduction compared to streaming API
- Structured JSON output for reliable parsing
- Enhanced metadata preservation for result matching
- JSONL file-based batch processing

Usage:
    python scripts/perturbations/create_perturbations_batch.py --input candidates.csv --perturbations add_extraneous_detail:flash
    python scripts/perturbations/create_perturbations_batch.py --input candidates.csv --perturbations add_emotion:flash change_trivial:flash --output perturbed_batch.csv
    python scripts/perturbations/create_perturbations_batch.py --input candidates.csv --perturbations none:flash --dry-run  # Cost estimation only
    python scripts/perturbations/create_perturbations_batch.py --monitor-batch batch_name
    python scripts/perturbations/create_perturbations_batch.py --download-batch batch_name --metadata-file batch_metadata.json
"""

import argparse
import os
import sys
import json
import logging
from pathlib import Path
from typing import List
import pandas as pd
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from llm_evaluations_everyday_dilemmas.perturber import PerturbationConfig, DilemmaPerturber
from llm_evaluations_everyday_dilemmas.config import PRESENTATION_TEMPLATES, VARIATION_TEMPLATES
from llm_evaluations_everyday_dilemmas.batch_perturber import GoogleBatchPerturber


def parse_perturbation_spec(spec: str) -> PerturbationConfig:
    """Parse perturbation specification string."""
    parts = spec.split(':')
    
    # Special case: allow bare 'none' without model specification
    if len(parts) == 1 and parts[0] == 'none':
        perturbation_type = 'none'
        model = 'gemini-2.5-flash'
        priority = 1
    elif len(parts) < 2:
        raise ValueError(f"Invalid perturbation spec: {spec}. Format: 'type:model' or 'type:model:priority'")
    else:
        perturbation_type = parts[0]
        model_short = parts[1]
        priority = 1
    
    # Handle priority if specified
    if len(parts) == 3:
        try:
            priority = int(parts[2])
        except ValueError:
            raise ValueError(f"Invalid priority in spec: {spec}. Must be integer.")
    
    # Validate model using centralized validation (like original script)
    available_models = DilemmaPerturber.AVAILABLE_MODELS
    if model_short not in available_models:
        raise ValueError(f"Unknown model: {model_short}. Available: {list(available_models.keys())}")
    
    model = available_models[model_short]
    
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
    
    return PerturbationConfig(
        perturbation_type=perturbation_type,
        model=model,
        is_format_perturbation=is_format_perturbation,
        priority=priority
    )


def create_argument_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Google Gemini Batch API perturbation with 50% cost reduction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Submit batch with cost estimation
  python create_perturbations_batch.py --input candidates.csv --perturbations add_emotion:flash change_trivial:flash --output perturbed_batch.csv
  
  # Cost estimation only
  python create_perturbations_batch.py --input candidates.csv --perturbations add_emotion:flash --dry-run
  
  # Monitor batch status
  python create_perturbations_batch.py --monitor-batch batch_name
  
  # Download completed batch results
  python create_perturbations_batch.py --download-batch batch_name --metadata-file perturbed_batch.batch_metadata.json
  
  # Process specific number of scenarios
  python create_perturbations_batch.py --input candidates.csv --perturbations firstperson:flash thirdperson:flash --sample-size 100
  
  # Process only chunk 2 of a large dataset (for manual pacing)
  python create_perturbations_batch.py --input candidates.csv --perturbations add_emotion:flash --chunk-index 2
  
  # Auto-chunk large datasets (automatically skips existing chunks)
  python create_perturbations_batch.py --input candidates.csv --perturbations add_emotion:flash --auto-chunk
"""
    )
    
    # Input file (optional for monitoring/downloading)
    parser.add_argument("--input", "--input-file", dest="input_file", default="candidates.csv", 
                       help="Input CSV file with scenarios to perturb")
    
    # Perturbation configuration
    parser.add_argument("--perturbations", nargs="+",
                       help="Perturbation specifications in format 'type:model' or 'type:model:priority'")
    
    # Output options
    parser.add_argument("--output", "--output-file", dest="output_file",
                       help="Output CSV file for results")
    
    # Metadata columns to preserve
    parser.add_argument("--metadata-columns", nargs="+",
                       default=["old_index", "id", "title", "author", "created_utc", "score", 
                               "url", "permalink", "n_comments", "n_verdicts"],
                       help="Columns from input to preserve in output (default: key identifiers)")
    
    # Processing options
    parser.add_argument("--sample-size", type=int,
                       help="Maximum number of scenarios to process")
    parser.add_argument("--chunk-size", type=int, default=1000,
                       help="Process in chunks of this size (default: 1000, max recommended: 2000)")
    parser.add_argument("--chunk-index", type=int,
                       help="Process only a specific chunk (1-based index)")
    parser.add_argument("--auto-chunk", action="store_true",
                       help="Automatically chunk large datasets without prompting")
    parser.add_argument("--offset", type=int, default=0,
                       help="Number of rows to skip from start")
    parser.add_argument("--temperature", type=float, default=0.4,
                       help="Model temperature (default: 0.4)")
    parser.add_argument("--cache-dir", 
                       help="Directory for result caching")
    parser.add_argument("--no-resume", action="store_true",
                       help="Don't resume from existing temp files")
    parser.add_argument("--clear-cache", action="store_true",
                       help="Clear all cached results before processing")
    parser.add_argument("--parallel", action="store_true",
                       help="Run perturbations in parallel for faster processing")
    parser.add_argument("--max-workers", type=int, default=4,
                       help="Maximum concurrent workers for parallel processing")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Show verbose progress")
    
    # Batch operations
    parser.add_argument("--monitor-batch",
                       help="Monitor status of a batch job")
    parser.add_argument("--download-batch",
                       help="Download results from completed batch")
    parser.add_argument("--metadata-file",
                       help="Metadata file for batch recovery")
    
    # Control options
    parser.add_argument("--dry-run", action="store_true",
                       help="Show cost estimation without running perturbation generation")
    
    return parser


def main():
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Setup logging with verbose option (like original script)
    if args.verbose:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    else:
        logging.basicConfig(
            level=logging.WARNING,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    # Get API key
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Error: GOOGLE_API_KEY environment variable not set")
        print("Get your API key from: https://makersuite.google.com/app/apikey")
        return
    
    # Create perturber
    perturber = GoogleBatchPerturber(api_key, temperature=args.temperature)
    
    # Handle batch monitoring
    if args.monitor_batch:
        print(f"\n📊 BATCH STATUS")
        print("=" * 60)
        
        status = perturber.monitor_batch(args.monitor_batch)
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
        
        # Load metadata file to get output path
        with open(args.metadata_file, 'r') as f:
            metadata = json.load(f)
        
        output_file = metadata.get('output_file', 'batch_perturbation_results.csv')
        
        try:
            results_jsonl = perturber.download_batch_results(args.download_batch, args.metadata_file)
            final_output = perturber.process_batch_results(results_jsonl, output_file)
            print(f"\n✅ Results successfully downloaded and processed!")
            print(f"Output file: {final_output}")
        except Exception as e:
            print(f"\n❌ Error downloading results: {e}")
            import traceback
            traceback.print_exc()
        
        return
    
    # Regular perturbation mode
    if not args.input_file:
        parser.error("--input required for perturbation mode")
    
    if not args.perturbations:
        parser.error("--perturbations required for perturbation mode")
    
    # Setup paths with automatic data/ directory prefixing (like original script)
    data_dir = Path("data")
    
    # Better error handling for data directory (like original script)
    if not data_dir.exists():
        print(f"Error: Data directory '{data_dir}' not found")
        print("Please ensure you're running from the project root directory")
        return
    
    input_path = data_dir / args.input_file
    
    if not input_path.exists():
        print(f"Error: {input_path} not found")
        print(f"Available files in {data_dir}:")
        try:
            csv_files = list(data_dir.glob("*.csv"))
            if csv_files:
                for csv_file in csv_files[:5]:  # Show first 5 CSV files
                    print(f"  - {csv_file.name}")
                if len(csv_files) > 5:
                    print(f"  ... and {len(csv_files) - 5} more")
            else:
                print("  No CSV files found")
        except Exception:
            pass
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
    
    # Create output filename if not specified (save in data/ directory like original script)
    if not args.output_file:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        perturbation_names = "_".join([config.perturbation_type for config in perturbation_configs[:3]])
        args.output_file = f"perturbed_batch_{perturbation_names}_{timestamp}.csv"
    
    # Ensure output is in data/ directory (like original script)
    output_path = data_dir / args.output_file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Update args.output_file to use the full path
    args.output_file = str(output_path)
    
    try:
        # Load and validate data
        print(f"Loading data from: {input_path}")
        df = pd.read_csv(input_path)
        
        # Apply sample size and offset
        if args.offset > 0:
            df = df.iloc[args.offset:]
        if args.sample_size:
            df = df.head(args.sample_size)
        
        # Check if we need to chunk the data for large datasets or if specific chunk requested
        total_scenarios = len(df)
        
        # Handle specific chunk index request
        if args.chunk_index:
            total_chunks = (total_scenarios + args.chunk_size - 1) // args.chunk_size
            if args.chunk_index < 1 or args.chunk_index > total_chunks:
                print(f"❌ Invalid chunk index: {args.chunk_index}")
                print(f"   Valid range: 1-{total_chunks} (with chunk size {args.chunk_size})")
                return
            
            print(f"\n🎯 SINGLE CHUNK MODE")
            print(f"Processing chunk {args.chunk_index}/{total_chunks} (chunk size: {args.chunk_size})")
            
            # Calculate chunk boundaries
            chunk_start = (args.chunk_index - 1) * args.chunk_size
            chunk_end = min(chunk_start + args.chunk_size, total_scenarios)
            chunk_df = df.iloc[chunk_start:chunk_end].copy()
            
            print(f"Scenarios {chunk_start+1}-{chunk_end} ({len(chunk_df)} scenarios)")
            
            # Create chunk-specific output file
            base_output = Path(args.output_file)
            chunk_output = base_output.parent / f"{base_output.stem}_chunk{args.chunk_index:03d}{base_output.suffix}"
            
            # Check if this chunk was already processed
            chunk_metadata_file = chunk_output.with_suffix('.batch_metadata.json')
            if chunk_metadata_file.exists():
                print(f"⚠️  Chunk {args.chunk_index} already processed!")
                print(f"   Metadata file exists: {chunk_metadata_file}")
                proceed = input("Process anyway? (yes/no): ")
                if proceed.lower() not in ['yes', 'y']:
                    print("✅ Skipped - use different chunk index or remove metadata file")
                    return
            
            # Process this single chunk
            chunk_perturber = GoogleBatchPerturber(api_key, temperature=args.temperature)
            chunk_cost = chunk_perturber.estimate_cost(chunk_df, perturbation_configs)
            
            print(f"Chunk output: {chunk_output}")
            proceed = input(f"Process chunk {args.chunk_index}? (yes/no): ")
            if proceed.lower() in ['yes', 'y']:
                jsonl_file, batch_metadata = chunk_perturber.create_batch_jsonl(
                    chunk_df, perturbation_configs, args.metadata_columns, str(chunk_output)
                )
                
                perturbation_types = [config.perturbation_type for config in perturbation_configs]
                batch_name = chunk_perturber.submit_batch(jsonl_file, perturbation_types)
                
                # Save chunk metadata
                with open(chunk_metadata_file, 'w') as f:
                    json.dump({
                        'batch_name': batch_name,
                        'batch_file': jsonl_file,
                        'metadata_file': str(chunk_metadata_file),
                        'output_file': str(chunk_output),
                        'perturbation_configs': [config.model_dump() for config in perturbation_configs],
                        'batch_metadata': batch_metadata,
                        'submitted_at': datetime.now().isoformat(),
                        'input_file': str(input_path),
                        'metadata_columns': args.metadata_columns,
                        'chunk_info': {
                            'chunk_num': args.chunk_index,
                            'total_chunks': total_chunks,
                            'chunk_start': chunk_start,
                            'chunk_end': chunk_end
                        }
                    }, f, indent=2, default=str)
                
                print(f"✅ Chunk {args.chunk_index} submitted: {batch_name}")
                print(f"Metadata: {chunk_metadata_file}")
                print(f"\n📥 TO DOWNLOAD WHEN COMPLETE (~24 hours):")
                print(f"   python {__file__} --monitor-batch {batch_name}")
                print(f"   python {__file__} --download-batch {batch_name} --metadata-file {chunk_metadata_file}")
            else:
                print("❌ Aborted")
            return
        
        if total_scenarios > args.chunk_size and not args.sample_size:
            print(f"⚠️  Large dataset detected: {total_scenarios} scenarios")
            print(f"   Recommended: Use --chunk-size {args.chunk_size} or --sample-size for testing")
            print(f"   Large batches (>{args.chunk_size}) may fail due to API limits")
            print(f"   💡 Or use --chunk-index N to process one chunk at a time")
            
            if args.auto_chunk:
                response = 'chunk'
                print(f"🤖 Auto-chunking enabled, processing in chunks of {args.chunk_size}")
            else:
                response = input(f"\nProcess all {total_scenarios} scenarios in one batch? (yes/no/chunk): ")
            
            if response.lower() == 'chunk':
                print(f"\n📦 CHUNKING MODE")
                print(f"Processing {total_scenarios} scenarios in chunks of {args.chunk_size}")
                
                # Check for existing chunks first
                total_chunks = (total_scenarios + args.chunk_size - 1) // args.chunk_size
                existing_chunks = []
                base_output = Path(args.output_file)
                for i in range(1, total_chunks + 1):
                    chunk_metadata_file = base_output.parent / f"{base_output.stem}_chunk{i:03d}.batch_metadata.json"
                    if chunk_metadata_file.exists():
                        existing_chunks.append(i)
                
                if existing_chunks:
                    print(f"\n⚠️  Found existing chunks: {existing_chunks}")
                    print(f"   These will be skipped automatically")
                
                # Process in chunks
                for chunk_start in range(0, total_scenarios, args.chunk_size):
                    chunk_end = min(chunk_start + args.chunk_size, total_scenarios)
                    chunk_df = df.iloc[chunk_start:chunk_end].copy()
                    chunk_num = (chunk_start // args.chunk_size) + 1
                    total_chunks = (total_scenarios + args.chunk_size - 1) // args.chunk_size
                    
                    print(f"\n{'='*60}")
                    print(f"PROCESSING CHUNK {chunk_num}/{total_chunks}")
                    print(f"Scenarios {chunk_start+1}-{chunk_end} ({len(chunk_df)} scenarios)")
                    print(f"{'='*60}")
                    
                    # Create chunk-specific output file
                    base_output = Path(args.output_file)
                    chunk_output = base_output.parent / f"{base_output.stem}_chunk{chunk_num:03d}{base_output.suffix}"
                    
                    # Check if this chunk already exists
                    chunk_metadata_file = chunk_output.with_suffix('.batch_metadata.json')
                    if chunk_metadata_file.exists():
                        print(f"⏭️  Chunk {chunk_num} already processed (metadata exists)")
                        print(f"   Metadata: {chunk_metadata_file}")
                        continue
                    
                    # Process this chunk
                    chunk_perturber = GoogleBatchPerturber(api_key, temperature=args.temperature)
                    chunk_cost = chunk_perturber.estimate_cost(chunk_df, perturbation_configs)
                    
                    print(f"Chunk output: {chunk_output}")
                    proceed = input(f"Process chunk {chunk_num}? (yes/no/skip): ")
                    if proceed.lower() in ['yes', 'y']:
                        jsonl_file, batch_metadata = chunk_perturber.create_batch_jsonl(
                            chunk_df, perturbation_configs, args.metadata_columns, str(chunk_output)
                        )
                        
                        perturbation_types = [config.perturbation_type for config in perturbation_configs]
                        batch_name = chunk_perturber.submit_batch(jsonl_file, perturbation_types)
                        
                        # Save chunk metadata
                        chunk_metadata_file = chunk_output.with_suffix('.batch_metadata.json')
                        with open(chunk_metadata_file, 'w') as f:
                            json.dump({
                                'batch_name': batch_name,
                                'batch_file': jsonl_file,
                                'metadata_file': str(chunk_metadata_file),
                                'output_file': str(chunk_output),
                                'perturbation_configs': [config.model_dump() for config in perturbation_configs],
                                'batch_metadata': batch_metadata,
                                'submitted_at': datetime.now().isoformat(),
                                'input_file': str(input_path),
                                'metadata_columns': args.metadata_columns,
                                'chunk_info': {
                                    'chunk_num': chunk_num,
                                    'total_chunks': total_chunks,
                                    'chunk_start': chunk_start,
                                    'chunk_end': chunk_end
                                }
                            }, f, indent=2, default=str)
                        
                        print(f"✅ Chunk {chunk_num} submitted: {batch_name}")
                        print(f"Metadata: {chunk_metadata_file}")
                        print(f"\n📥 TO DOWNLOAD WHEN COMPLETE (~24 hours):")
                        print(f"   python {__file__} --monitor-batch {batch_name}")
                        print(f"   python {__file__} --download-batch {batch_name} --metadata-file {chunk_metadata_file}")
                    elif proceed.lower() == 'skip':
                        print(f"⏭️  Skipped chunk {chunk_num}")
                        continue
                    else:
                        print("❌ Aborted chunking process")
                        return
                
                print(f"\n🎉 All chunks submitted! Monitor each batch separately.")
                return
            
            elif response.lower() not in ['yes', 'y']:
                print("❌ Aborted due to large dataset size")
                print(f"💡 Try: --sample-size 100 for testing or --chunk-size {args.chunk_size} for processing")
                return
        
        print(f"Processing {len(df)} scenarios")
        
        # Estimate cost
        cost_info = perturber.estimate_cost(df, perturbation_configs)
        
        # Display comprehensive configuration (like original script)
        print("\n🚀 Google Batch Perturber with Metadata Preservation")
        print("=" * 70)
        print(f"Input: {input_path}")
        print(f"Output: {output_path}")
        print(f"Sample size: {args.sample_size or 'All dilemmas'}")
        print(f"Temperature: {args.temperature}")
        print(f"Metadata columns to preserve: {', '.join(args.metadata_columns)}")
        
        print(f"\n📋 Perturbation Configuration:")
        for i, config in enumerate(perturbation_configs, 1):
            model_short = config.model.split('-')[-1] if '-' in config.model else config.model
            print(f"  {i}. {config.perturbation_type} → {model_short} [priority: {config.priority}]")
        
        if args.parallel:
            print(f"\n⚡ Concurrent Processing: {args.max_workers} workers")
        
        if args.verbose:
            print(f"\n🔍 Verbose logging enabled")
        
        print("\n" + "=" * 70)
        
        if args.dry_run:
            print("\n✅ Dry run complete - no perturbation performed")
            return
        
        # Get user confirmation
        print(f"\n{'='*60}")
        print("CONFIRM PERTURBATION BATCH")
        print(f"{'='*60}")
        print(f"Input: {input_path}")
        print(f"Perturbations: {[config.perturbation_type for config in perturbation_configs]}")
        print(f"Temperature: {args.temperature}")
        print(f"Scenarios: {len(df)}")
        print(f"Total perturbations: {len(df) * len(perturbation_configs)}")
        print(f"Output: {output_path}")
        
        response = input("\nProceed with batch submission? (yes/no): ")
        if response.lower() not in ['yes', 'y']:
            print("Perturbation cancelled")
            return
        
        # Execute perturbation batch
        print("\n" + "="*60)
        print("SUBMITTING PERTURBATION BATCH")
        print("="*60)
        
        # Create batch JSONL
        jsonl_file, batch_metadata = perturber.create_batch_jsonl(
            df, perturbation_configs, args.metadata_columns, str(output_path)
        )
        
        # Submit batch
        perturbation_types = [config.perturbation_type for config in perturbation_configs]
        batch_name = perturber.submit_batch(jsonl_file, perturbation_types)
        
        # Save batch metadata for recovery (also in data/ directory)
        metadata_file = output_path.with_suffix('.batch_metadata.json')
        with open(metadata_file, 'w') as f:
            json.dump({
                'batch_name': batch_name,
                'batch_file': jsonl_file,
                'metadata_file': str(metadata_file),
                'output_file': str(output_path),
                'perturbation_configs': [config.model_dump() for config in perturbation_configs],
                'batch_metadata': batch_metadata,
                'submitted_at': datetime.now().isoformat(),
                'input_file': str(input_path),
                'metadata_columns': args.metadata_columns
            }, f, indent=2, default=str)
        
        print(f"\n🚀 BATCH SUBMITTED SUCCESSFULLY!")
        print(f"Generated {len(df) * len(perturbation_configs)} perturbation requests")
        print(f"Scenarios processed: {len(df)}")
        print(f"Perturbation types: {len(perturbation_configs)}")
        print(f"Batch name: {batch_name}")
        print(f"Batch file: {jsonl_file}")
        print(f"Metadata file: {metadata_file}")
        
        # Show preserved metadata columns (like original script)
        print(f"\n📋 Preserved metadata columns: {', '.join(args.metadata_columns)}")
        
        print(f"\n📋 Submitted Perturbations:")
        for i, config in enumerate(perturbation_configs, 1):
            model_short = config.model.split('-')[-1] if '-' in config.model else config.model
            print(f"  {i}. {config.perturbation_type} → {model_short}")
        
        print(f"\nTo monitor batch status:")
        print(f"  python {__file__} --monitor-batch {batch_name}")
        print(f"\nTo download results when complete:")
        print(f"  python {__file__} --download-batch {batch_name} --metadata-file {metadata_file}")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Perturbation interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()