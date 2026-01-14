#!/usr/bin/env python3
"""
Evaluate reasoning models across protocol variations.

Research Question: Does explicit reasoning protect against protocol sensitivity?

Design:
- Sample: 1,200 scenarios (existing protocol test sample)
- Protocols: 3 (verdict_first, explanation_first, system_prompt)
- Models:
  - OpenAI o3-mini (reasoning hidden, compared to GPT-4.1 baseline)
  - DeepSeek R1 (reasoning visible, compared to DeepSeek V3 baseline)

Usage:
    # Tiny pilot (3 scenarios, o3-mini medium, verdict_first only)
    python scripts/evaluation/evaluate_reasoning_protocols.py --pilot

    # Single model + protocol
    python scripts/evaluation/evaluate_reasoning_protocols.py --model deepseek-r1 --protocol verdict_first

    # Full run (2 models × 3 protocols = 6 conditions)
    python scripts/evaluation/evaluate_reasoning_protocols.py --full
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

import pandas as pd
import requests
from tqdm import tqdm

# Add src to path for config imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from llm_evaluations_everyday_dilemmas.config import (
    EVALUATION_TEMPLATES,
    JSON_INSTRUCTIONS,
    PERTURBATION_TO_FORMAT,
    SYSTEM_PROMPT_TOOL,
)
from llm_evaluations_everyday_dilemmas.config_explanation_first import (
    EXPLANATION_FIRST_TEMPLATES,
)

# =============================================================================
# Configuration
# =============================================================================

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

MODELS = {
    "o3-mini": {
        "id": "openai/o3-mini",
        "name": "OpenAI o3-mini",
        "input_cost_per_m": 1.10,
        "output_cost_per_m": 4.40,
        "thinking_format": "openai",
        "max_tokens": 8192,
        "supports_effort": True,  # Uses reasoning_effort parameter
    },
    "deepseek-r1": {
        "id": "deepseek/deepseek-r1-0528",
        "name": "DeepSeek R1",
        "input_cost_per_m": 0.70,
        "output_cost_per_m": 2.40,
        "thinking_format": "deepseek",
        "max_tokens": 16384,
        "supports_effort": False,  # No effort parameter, uses default reasoning
    },
    "claude-thinking": {
        "id": "anthropic/claude-3.7-sonnet:thinking",
        "name": "Claude 3.7 Sonnet (thinking)",
        "input_cost_per_m": 3.00,
        "output_cost_per_m": 15.00,
        "thinking_format": "anthropic",
        "max_tokens": 16384,
        "supports_effort": False,  # No effort parameter
    },
    "qwq-32b": {
        "id": "qwen/qwq-32b",
        "name": "Qwen QwQ-32B",
        "input_cost_per_m": 0.15,
        "output_cost_per_m": 0.40,
        "thinking_format": "qwen",
        "max_tokens": 16384,
        "supports_effort": False,
    },
}

# Parallel execution settings
MAX_WORKERS = 10
MIN_REQUEST_INTERVAL = 0.1  # 100ms between requests

# Reasoning effort levels for o3-mini
REASONING_EFFORTS = ["low", "medium", "high"]

# Sample file
SAMPLE_FILE = "data/protocol_test_samples/protocol_sample_n1200_seed42_20260108_200337.csv"

# Output directory
OUTPUT_DIR = Path("results/reasoning_protocols")


# =============================================================================
# Protocol Prompt Builders
# =============================================================================

def get_explanation_first_prompt(perturbation_type: str, dilemma_text: str) -> dict:
    """
    Explanation-first protocol: reasoning before verdict.
    Uses exact same templates as original evaluate_explanation_first.py.
    Returns dict with 'messages' for API call.
    """
    # Determine which template to use (matches original evaluate_explanation_first.py)
    if perturbation_type in ['firstperson_atfault', 'firstperson']:
        template_type = 'firstperson'
    elif perturbation_type == 'thirdperson':
        template_type = 'thirdperson'
    else:
        # Default to AITA template for baseline and other perturbations
        template_type = 'aita'

    # Get template from config (same as original)
    template = EXPLANATION_FIRST_TEMPLATES[template_type]

    # Format with dilemma text
    prompt = template.format(dilemma_text=dilemma_text)

    # Add JSON instruction (exact match to original evaluate_explanation_first.py)
    json_instruction = """
Please format your response as a JSON object with exactly two fields:
{
  "explanation": "Your detailed one-paragraph analysis here",
  "judgment": "Your categorical verdict here"
}""".strip()

    full_prompt = f"{prompt}\n\n{json_instruction}"

    return {"messages": [{"role": "user", "content": full_prompt}]}


def get_system_prompt_protocol(perturbation_type: str, dilemma_text: str) -> dict:
    """
    System-prompt protocol: instructions in system message, dilemma in user message.
    Returns dict with 'messages' for API call.
    """
    format_type = PERTURBATION_TO_FORMAT.get(perturbation_type, "aita")
    system_prompt = SYSTEM_PROMPT_TOOL[format_type]

    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"<dilemma>\n{dilemma_text}\n</dilemma>"}
        ]
    }


def get_verdict_first_prompt(perturbation_type: str, dilemma_text: str) -> dict:
    """
    Verdict-first protocol: the original study approach.
    Uses EVALUATION_TEMPLATES + JSON_INSTRUCTIONS from config.py.
    This is the standard evaluation where verdict comes before explanation.
    Returns dict with 'messages' for API call.
    """
    format_type = PERTURBATION_TO_FORMAT.get(perturbation_type, "aita")

    # Get base template (same as original evaluations)
    base_template = EVALUATION_TEMPLATES[format_type]

    # Add JSON instructions (same as original evaluations for OpenAI/Anthropic/etc)
    if format_type in JSON_INSTRUCTIONS:
        full_template = base_template + JSON_INSTRUCTIONS[format_type]
    else:
        full_template = base_template

    # Format with dilemma text (persona_prompt is empty for standard evals)
    prompt = full_template.format(dilemma_text=dilemma_text, persona_prompt="")

    return {"messages": [{"role": "user", "content": prompt}]}


PROTOCOL_BUILDERS = {
    "verdict_first": get_verdict_first_prompt,
    "explanation_first": get_explanation_first_prompt,
    "system_prompt": get_system_prompt_protocol,
}


# =============================================================================
# API Functions
# =============================================================================

def call_openrouter(
    model_id: str,
    messages: list,
    max_tokens: int = 4096,
    temperature: float = 0.4,
    reasoning_effort: str = None,
) -> dict:
    """Call OpenRouter API with optional reasoning effort level."""
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/llm-evaluations",
    }

    payload = {
        "model": model_id,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    # Add reasoning effort for o3-mini (low/medium/high)
    if reasoning_effort is not None:
        payload["reasoning"] = {"effort": reasoning_effort}

    response = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=300)
    response.raise_for_status()
    return response.json()


def parse_response(response: dict, thinking_format: str) -> dict:
    """Parse API response, extracting thinking and final answer."""
    result = {
        "raw_response": response,
        "thinking": None,
        "final_response": None,
        "judgment": None,
        "explanation": None,
        "input_tokens": 0,
        "output_tokens": 0,
        "error": None,
    }

    try:
        # Extract token usage
        usage = response.get("usage", {})
        result["input_tokens"] = usage.get("prompt_tokens", 0)
        result["output_tokens"] = usage.get("completion_tokens", 0)

        # Get content
        choices = response.get("choices", [])
        if not choices:
            result["error"] = "No choices in response"
            return result

        message = choices[0].get("message", {})
        content = message.get("content", "")

        # Check for reasoning field (OpenRouter unified format)
        reasoning = message.get("reasoning", None)

        if thinking_format in ["deepseek", "qwen"]:
            # OpenRouter returns reasoning in separate field
            if reasoning:
                result["thinking"] = reasoning
                result["final_response"] = content
            # Fallback: check for <think> tags in content
            elif "<think>" in content and "</think>" in content:
                think_start = content.find("<think>") + len("<think>")
                think_end = content.find("</think>")
                result["thinking"] = content[think_start:think_end].strip()
                result["final_response"] = content[think_end + len("</think>"):].strip()
            else:
                result["final_response"] = content
        elif thinking_format == "anthropic":
            # Anthropic extended thinking - check for reasoning field or <thinking> tags
            if reasoning:
                result["thinking"] = reasoning
                result["final_response"] = content
            elif "<thinking>" in content and "</thinking>" in content:
                think_start = content.find("<thinking>") + len("<thinking>")
                think_end = content.find("</thinking>")
                result["thinking"] = content[think_start:think_end].strip()
                result["final_response"] = content[think_end + len("</thinking>"):].strip()
            else:
                result["final_response"] = content
        else:
            if reasoning:
                result["thinking"] = reasoning
            result["final_response"] = content

        # Parse JSON judgment from final response
        final = result["final_response"] or content

        # Try to extract JSON
        json_start = final.find("{")
        json_end = final.rfind("}") + 1
        if json_start >= 0 and json_end > json_start:
            json_str = final[json_start:json_end]
            parsed = json.loads(json_str)
            result["judgment"] = parsed.get("judgment")
            result["explanation"] = parsed.get("explanation")
        else:
            # For unstructured, there may be no JSON - store raw response
            result["error"] = "No JSON found in response"
            result["explanation"] = final  # Store full response for later classification

    except json.JSONDecodeError as e:
        result["error"] = f"JSON parse error: {e}"
    except Exception as e:
        result["error"] = f"Parse error: {e}"

    return result


# =============================================================================
# Evaluation Functions
# =============================================================================

def load_sample(sample_file: str, pilot: bool = False, pilot_size: int = 3) -> pd.DataFrame:
    """Load evaluation sample."""
    df = pd.read_csv(sample_file)

    if pilot:
        # Simple random sample for tiny pilot
        pilot_df = df.sample(n=min(pilot_size, len(df)), random_state=42)
        print(f"Pilot sample: {len(pilot_df)} scenarios")
        return pilot_df

    return df


def evaluate_single_scenario(
    row: pd.Series,
    model_key: str,
    model_config: dict,
    reasoning_effort: str,
    protocol: str,
    condition: str,
    protocol_builder,
    rate_limit_lock: Lock,
    last_request_time: list,  # Use list to allow mutation in closure
) -> dict:
    """Evaluate a single scenario. Thread-safe."""
    scenario_id = f"{row['id']}_{row['perturbation_type']}"

    # Build prompt for this protocol
    prompt_config = protocol_builder(
        perturbation_type=row["perturbation_type"],
        dilemma_text=row["perturbed_text"]
    )

    # Rate limiting (thread-safe)
    with rate_limit_lock:
        time_since_last = time.time() - last_request_time[0]
        if time_since_last < MIN_REQUEST_INTERVAL:
            time.sleep(MIN_REQUEST_INTERVAL - time_since_last)
        last_request_time[0] = time.time()

    try:
        # Call API
        response = call_openrouter(
            model_id=model_config["id"],
            messages=prompt_config["messages"],
            max_tokens=model_config["max_tokens"],
            reasoning_effort=reasoning_effort if model_config.get("supports_effort") else None,
        )
        parsed = parse_response(response, model_config["thinking_format"])

        # Check for truncation
        finish_reason = response.get("choices", [{}])[0].get("finish_reason", "")
        if finish_reason == "length":
            parsed["error"] = f"Response truncated (finish_reason=length)"

        # Store result
        return {
            "scenario_id": scenario_id,
            "id": row["id"],
            "perturbation_type": row["perturbation_type"],
            "model": model_key,
            "model_id": model_config["id"],
            "reasoning_effort": reasoning_effort,
            "protocol": protocol,
            "condition": condition,
            "judgment": parsed["judgment"],
            "explanation": parsed["explanation"],
            "thinking": parsed["thinking"],
            "thinking_length": len(parsed["thinking"]) if parsed["thinking"] else 0,
            "final_response": parsed["final_response"],
            "raw_response": json.dumps(parsed["raw_response"]),
            "input_tokens": parsed["input_tokens"],
            "output_tokens": parsed["output_tokens"],
            "error": parsed["error"],
            "timestamp": datetime.now().isoformat(),
        }

    except requests.exceptions.RequestException as e:
        time.sleep(2)  # Brief backoff on API errors
        return {
            "scenario_id": scenario_id,
            "id": row["id"],
            "perturbation_type": row["perturbation_type"],
            "model": model_key,
            "model_id": model_config["id"],
            "protocol": protocol,
            "condition": condition,
            "reasoning_effort": reasoning_effort,
            "error": f"API error: {str(e)}",
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        return {
            "scenario_id": scenario_id,
            "id": row["id"],
            "perturbation_type": row["perturbation_type"],
            "model": model_key,
            "model_id": model_config["id"],
            "protocol": protocol,
            "condition": condition,
            "reasoning_effort": reasoning_effort,
            "error": f"Error: {str(e)}",
            "timestamp": datetime.now().isoformat(),
        }


def run_evaluation(
    model_key: str,
    reasoning_effort: str,
    protocol: str,
    sample_df: pd.DataFrame,
    output_dir: Path,
    checkpoint_every: int = 50,
    max_workers: int = MAX_WORKERS,
) -> pd.DataFrame:
    """Run parallel evaluation for a single model/effort/protocol combination."""
    model_config = MODELS[model_key]

    # Create condition identifier
    condition = f"{model_key}_{reasoning_effort}_{protocol}"

    print(f"\n{'='*60}")
    print(f"Evaluating: {model_config['name']}")
    print(f"Reasoning effort: {reasoning_effort}")
    print(f"Protocol: {protocol}")
    print(f"Scenarios: {len(sample_df)}")
    print(f"Workers: {max_workers}")
    print(f"{'='*60}\n")

    # Output file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"reasoning_protocol_{condition}_{timestamp}.parquet"
    checkpoint_file = output_dir / f"reasoning_protocol_{condition}_checkpoint.parquet"

    # Check for existing checkpoint
    results = []
    completed_ids = set()

    if checkpoint_file.exists():
        checkpoint_df = pd.read_parquet(checkpoint_file)
        results = checkpoint_df.to_dict("records")
        completed_ids = set(checkpoint_df["scenario_id"].unique())
        print(f"Resuming from checkpoint: {len(completed_ids)} already completed")

    # Thread-safe rate limiting
    rate_limit_lock = Lock()
    save_lock = Lock()
    last_request_time = [0]  # Use list to allow mutation

    # Get protocol builder
    protocol_builder = PROTOCOL_BUILDERS[protocol]

    # Track progress
    errors = 0
    last_save = time.time()

    # Filter to scenarios not yet completed
    pending_rows = [
        (idx, row) for idx, row in sample_df.iterrows()
        if f"{row['id']}_{row['perturbation_type']}" not in completed_ids
    ]

    # Parallel execution
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures_to_idx = {}
        for idx, row in pending_rows:
            future = executor.submit(
                evaluate_single_scenario,
                row,
                model_key,
                model_config,
                reasoning_effort,
                protocol,
                condition,
                protocol_builder,
                rate_limit_lock,
                last_request_time,
            )
            futures_to_idx[future] = idx

        # Process results with progress bar
        with tqdm(total=len(sample_df), initial=len(completed_ids), desc=condition) as pbar:
            for future in as_completed(futures_to_idx):
                try:
                    result = future.result()
                    results.append(result)

                    if result.get("error"):
                        errors += 1

                    pbar.update(1)

                    # Periodic checkpoint (thread-safe)
                    if len(results) % checkpoint_every == 0 or (time.time() - last_save) > 120:
                        with save_lock:
                            checkpoint_df = pd.DataFrame(results)
                            checkpoint_df.to_parquet(checkpoint_file, index=False)
                            last_save = time.time()

                except Exception as e:
                    print(f"\nWorker failed: {e}")
                    errors += 1
                    pbar.update(1)

    # Save final results
    results_df = pd.DataFrame(results)
    results_df.to_parquet(output_file, index=False)

    # Remove checkpoint
    if checkpoint_file.exists():
        checkpoint_file.unlink()

    # Cost summary
    total_input_tokens = results_df["input_tokens"].sum() if "input_tokens" in results_df.columns else 0
    total_output_tokens = results_df["output_tokens"].sum() if "output_tokens" in results_df.columns else 0
    input_cost = total_input_tokens / 1_000_000 * model_config["input_cost_per_m"]
    output_cost = total_output_tokens / 1_000_000 * model_config["output_cost_per_m"]
    total_cost = input_cost + output_cost

    print(f"\n{'='*60}")
    print(f"COMPLETED: {condition}")
    print(f"{'='*60}")
    print(f"Results saved: {output_file}")
    print(f"Total scenarios: {len(results_df)}")
    print(f"Successful: {results_df['judgment'].notna().sum() if 'judgment' in results_df.columns else 0}")
    print(f"Errors: {errors}")
    print(f"\nToken usage:")
    print(f"  Input:  {total_input_tokens:,} tokens (${input_cost:.2f})")
    print(f"  Output: {total_output_tokens:,} tokens (${output_cost:.2f})")
    print(f"  TOTAL COST: ${total_cost:.2f}")
    print(f"{'='*60}\n")

    return results_df


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate reasoning models across protocols")
    parser.add_argument("--pilot", action="store_true", help="Run tiny pilot (3 scenarios, medium effort, verdict_first only)")
    parser.add_argument("--model", type=str, choices=list(MODELS.keys()), default="o3-mini", help="Model to use")
    parser.add_argument("--effort", type=str, choices=REASONING_EFFORTS, help="Reasoning effort (low/medium/high)")
    parser.add_argument("--protocol", type=str, choices=list(PROTOCOL_BUILDERS.keys()), help="Single protocol")
    parser.add_argument("--full", action="store_true", help="Run all effort levels and protocols")
    parser.add_argument("--sample-file", type=str, default=SAMPLE_FILE, help="Sample file path")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS, help=f"Parallel workers (default: {MAX_WORKERS})")
    parser.add_argument("--pilot-size", type=int, default=None, help="Number of scenarios (default: full sample)")
    args = parser.parse_args()

    # Check API key
    if not OPENROUTER_API_KEY:
        print("ERROR: OPENROUTER_API_KEY environment variable not set")
        return

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load sample - only use pilot mode if --pilot flag or explicit --pilot-size
    use_pilot = args.pilot or (args.pilot_size is not None)
    pilot_size = args.pilot_size if args.pilot_size is not None else 3
    sample_df = load_sample(args.sample_file, pilot=use_pilot, pilot_size=pilot_size)
    print(f"Loaded {len(sample_df)} scenarios from {args.sample_file}")

    # Determine what to run
    if args.pilot:
        # Tiny pilot: o3-mini @ medium, verdict_first only
        runs = [
            ("o3-mini", "medium", "verdict_first"),
        ]
    elif args.full:
        # Full run: both models × all protocols
        runs = []
        # o3-mini with medium effort × 3 protocols
        for protocol in PROTOCOL_BUILDERS.keys():
            runs.append(("o3-mini", "medium", protocol))
        # DeepSeek R1 (no effort param) × 3 protocols
        for protocol in PROTOCOL_BUILDERS.keys():
            runs.append(("deepseek-r1", None, protocol))
    elif args.model or args.protocol:
        # Custom run with specified model and/or protocol
        model = args.model or "o3-mini"
        protocols = [args.protocol] if args.protocol else list(PROTOCOL_BUILDERS.keys())

        # Handle effort: DeepSeek R1 doesn't support effort, o3-mini defaults to medium
        if model == "deepseek-r1":
            efforts = [None]  # DeepSeek R1 doesn't use effort parameter
        elif args.effort:
            efforts = [args.effort]
        else:
            efforts = ["medium"]  # Default to medium for o3-mini

        runs = [(model, e, p) for e in efforts for p in protocols]
    else:
        print("Specify --pilot, --full, or --effort/--protocol")
        return

    # Estimate costs
    print("\n" + "="*60)
    print("EVALUATION PLAN")
    print("="*60)
    print(f"\nRuns to execute: {len(runs)}")
    print(f"Parallel workers: {args.workers}")
    for model, effort, protocol in runs:
        print(f"  - {model} (effort={effort}) × {protocol}")

    # Rough cost estimate (o3-mini output varies by effort level)
    avg_input_tokens = 800
    effort_output_tokens = {"low": 600, "medium": 1200, "high": 2000}
    total_calls = len(runs) * len(sample_df)

    total_cost_estimate = 0
    for model, effort, protocol in runs:
        config = MODELS[model]
        calls = len(sample_df)
        avg_output = effort_output_tokens.get(effort, 1200)
        input_cost = calls * avg_input_tokens / 1_000_000 * config["input_cost_per_m"]
        output_cost = calls * avg_output / 1_000_000 * config["output_cost_per_m"]
        total_cost_estimate += input_cost + output_cost

    print(f"\nTotal API calls: {total_calls:,}")
    print(f"Estimated cost: ~${total_cost_estimate:.2f}")
    print("="*60 + "\n")

    # Confirm (skip for small tests under 50 calls)
    if total_calls > 50:
        response = input("Proceed? (y/n): ")
        if response.lower() != "y":
            print("Aborted")
            return

    # Run evaluations
    all_results = []
    for model, effort, protocol in runs:
        results_df = run_evaluation(
            model_key=model,
            reasoning_effort=effort,
            protocol=protocol,
            sample_df=sample_df,
            output_dir=OUTPUT_DIR,
            max_workers=args.workers,
        )
        all_results.append(results_df)

    # Combine results
    if len(all_results) > 1:
        combined = pd.concat(all_results, ignore_index=True)
        combined_file = OUTPUT_DIR / f"reasoning_protocol_combined_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
        combined.to_parquet(combined_file, index=False)
        print(f"\nCombined results: {combined_file}")

    print("\nDone!")


if __name__ == "__main__":
    main()
