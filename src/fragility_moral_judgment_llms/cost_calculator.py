"""
Cost calculation utilities for LLM evaluation scripts.

Provides accurate cost estimation for different providers and models
to prevent unexpected charges during batch processing.
"""

from typing import Dict, Any, Optional
import pandas as pd


class CostCalculator:
    """Calculate estimated costs for LLM evaluations across different providers."""
    
    # Pricing data (per million tokens) - updated june 2025
    PRICING = {
        # Google models
        "gemini-2.0-flash": {"input": 0.10, "output": 0.40, "batch_discount": 0},
        "gemini-2.5-flash": {"input": 0.30, "output": 2.50, "batch_discount": 0}, 
        "gemini-2.5-pro": {"input": 1.25, "output": 10.00, "batch_discount": 0},
        
        # Anthropic models
        "claude-3-5-haiku-latest": {"input": 0.80, "output": 4.00, "batch_discount": 0.5},
        "claude-3-5-sonnet-latest": {"input": 3.00, "output": 15.00, "batch_discount": 0.5},
        "claude-3-7-sonnet-latest": {"input": 3.00, "output": 15.00, "batch_discount": 0.5},
        "claude-sonnet-4-0": {"input": 3.00, "output": 15.00, "batch_discount": 0.5},
        "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25, "batch_discount": 0.5},
        
        # OpenAI models
        "gpt-4.1": {"input": 2.00, "output": 8.00, "batch_discount": 0.5},
        "gpt-4.1-mini": {"input": 0.40, "output": 1.60, "batch_discount": 0.5},
        "gpt-4.1-nano": {"input": 0.10, "output": 0.40, "batch_discount": 0.5},
        "o3": {"input": 2.00, "output": 8.00, "batch_discount": 0.5},
        "o3-mini": {"input": 1.10, "output": 4.40, "batch_discount": 0.5}
    }
    
    # Token estimation constants
    DEFAULT_INPUT_TOKENS = 800   # Typical evaluation prompt
    DEFAULT_OUTPUT_TOKENS = 150  # Typical judgment + explanation
    
    @classmethod
    def estimate_tokens(cls, text: str) -> int:
        """Estimate token count for text using simple heuristic."""
        # Rough estimation: 1 token ≈ 0.75 words for English text
        word_count = len(text.split())
        return int(word_count / 0.75)
    
    @classmethod
    def calculate_cost(cls, 
                      model: str, 
                      num_requests: int,
                      avg_input_tokens: Optional[int] = None,
                      avg_output_tokens: Optional[int] = None,
                      is_batch: bool = False) -> Dict[str, Any]:
        """
        Calculate estimated cost for a set of requests.
        
        Args:
            model: Model name to use for pricing
            num_requests: Number of requests to process
            avg_input_tokens: Average input tokens per request
            avg_output_tokens: Average output tokens per request  
            is_batch: Whether batch processing (for discount calculation)
            
        Returns:
            Dictionary with cost breakdown
        """
        if model not in cls.PRICING:
            return {
                "model": model,
                "supported": False,
                "error": f"Pricing not available for model '{model}'"
            }
        
        pricing = cls.PRICING[model]
        input_tokens = avg_input_tokens or cls.DEFAULT_INPUT_TOKENS
        output_tokens = avg_output_tokens or cls.DEFAULT_OUTPUT_TOKENS
        
        # Calculate base costs
        total_input_tokens = num_requests * input_tokens
        total_output_tokens = num_requests * output_tokens
        
        input_cost = (total_input_tokens * pricing["input"]) / 1_000_000
        output_cost = (total_output_tokens * pricing["output"]) / 1_000_000
        total_cost = input_cost + output_cost
        
        # Apply batch discount if applicable
        batch_discount = pricing.get("batch_discount", 0)
        has_batch_discount = is_batch and batch_discount > 0
        discounted_cost = total_cost * (1 - batch_discount) if has_batch_discount else total_cost
        
        return {
            "model": model,
            "supported": True,
            "num_requests": num_requests,
            "input_tokens_per_request": input_tokens,
            "output_tokens_per_request": output_tokens,
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": total_cost,
            "batch_discount": batch_discount,
            "has_batch_discount": has_batch_discount,
            "discounted_cost": discounted_cost,
            "estimated_cost": discounted_cost,
            "pricing_per_million": pricing
        }
    
    @classmethod
    def estimate_dataset_cost(cls,
                             df: pd.DataFrame,
                             model: str,
                             num_runs: int = 1,
                             is_batch: bool = False,
                             sample_texts: Optional[list] = None) -> Dict[str, Any]:
        """
        Estimate cost for evaluating a complete dataset.
        
        Args:
            df: DataFrame with perturbed scenarios
            model: Model name to use for pricing
            num_runs: Number of evaluation runs per scenario
            is_batch: Whether using batch processing
            sample_texts: Optional sample texts for better token estimation
            
        Returns:
            Detailed cost breakdown
        """
        num_scenarios = len(df)
        total_requests = num_scenarios * num_runs
        
        # Estimate tokens from sample texts if provided
        if sample_texts and len(sample_texts) > 0:
            # Calculate average from sample
            sample_input_tokens = [cls.estimate_tokens(text) for text in sample_texts[:10]]
            avg_input_tokens = int(sum(sample_input_tokens) / len(sample_input_tokens))
        else:
            # Use default estimation
            avg_input_tokens = cls.DEFAULT_INPUT_TOKENS
        
        cost_info = cls.calculate_cost(
            model=model,
            num_requests=total_requests,
            avg_input_tokens=avg_input_tokens,
            is_batch=is_batch
        )
        
        # Add dataset-specific information
        cost_info.update({
            "num_scenarios": num_scenarios,
            "num_runs": num_runs,
            "total_requests": total_requests,
            "is_batch": is_batch
        })
        
        return cost_info
    
    @classmethod
    def print_cost_summary(cls, cost_info: Dict[str, Any]):
        """Print a formatted cost summary."""
        if not cost_info.get("supported", False):
            print(f"❌ Cost estimation not available: {cost_info.get('error', 'Unknown error')}")
            return
        
        print(f"\n💰 COST ESTIMATION")
        print(f"-" * 40)
        print(f"Model: {cost_info['model']}")
        print(f"Scenarios: {cost_info.get('num_scenarios', 'N/A'):,}")
        print(f"Runs per scenario: {cost_info.get('num_runs', 1)}")
        print(f"Total requests: {cost_info['total_requests']:,}")
        print(f"")
        print(f"Token estimation:")
        print(f"  Input tokens/request: ~{cost_info['input_tokens_per_request']:,}")
        print(f"  Output tokens/request: ~{cost_info['output_tokens_per_request']:,}")
        print(f"  Total input tokens: {cost_info['total_input_tokens']:,}")
        print(f"  Total output tokens: {cost_info['total_output_tokens']:,}")
        print(f"")
        print(f"Cost breakdown:")
        print(f"  Input cost (${cost_info['pricing_per_million']['input']}/M): ${cost_info['input_cost']:.2f}")
        print(f"  Output cost (${cost_info['pricing_per_million']['output']}/M): ${cost_info['output_cost']:.2f}")
        print(f"  Base total: ${cost_info['total_cost']:.2f}")
        
        if cost_info.get('has_batch_discount'):
            discount_pct = cost_info['batch_discount'] * 100
            print(f"  Batch discount ({discount_pct}%): -${cost_info['total_cost'] - cost_info['discounted_cost']:.2f}")
            print(f"  Final cost: ${cost_info['discounted_cost']:.2f}")
        else:
            print(f"  Final cost: ${cost_info['total_cost']:.2f}")
        
        print(f"")
        
        # Cost warnings
        if cost_info['estimated_cost'] > 10:
            print(f"⚠️  HIGH COST WARNING: This will cost over $10")
        elif cost_info['estimated_cost'] > 50:
            print(f"🚨 VERY HIGH COST WARNING: This will cost over $50")
    
    @classmethod
    def get_supported_models(cls) -> Dict[str, list]:
        """Get list of supported models by provider."""
        models_by_provider = {
            "google": [],
            "anthropic": [],
            "openai": []
        }
        
        for model in cls.PRICING.keys():
            if model.startswith("gemini"):
                models_by_provider["google"].append(model)
            elif model.startswith("claude"):
                models_by_provider["anthropic"].append(model) 
            elif model.startswith(("gpt", "o3")):
                models_by_provider["openai"].append(model)
        
        return models_by_provider
    
    @classmethod
    def validate_model_pricing(cls, provider: str, model: str) -> bool:
        """Check if pricing is available for a given model."""
        return model in cls.PRICING
    
    @classmethod
    def get_cheapest_models(cls, provider: str) -> Dict[str, str]:
        """Get the cheapest models for a given provider."""
        supported_models = cls.get_supported_models()
        
        if provider not in supported_models:
            return {}
        
        provider_models = supported_models[provider]
        if not provider_models:
            return {}
        
        # Find model with lowest combined input + output cost (rough approximation)
        cheapest = None
        lowest_cost = float('inf')
        
        for model in provider_models:
            pricing = cls.PRICING[model]
            # Use weighted average (more input than output typically)
            combined_cost = pricing["input"] * 0.8 + pricing["output"] * 0.2
            
            if combined_cost < lowest_cost:
                lowest_cost = combined_cost
                cheapest = model
        
        return {"model": cheapest, "cost_score": lowest_cost} if cheapest else {}


# Convenience functions for direct use
def estimate_cost(model: str, num_requests: int, is_batch: bool = False) -> Dict[str, Any]:
    """Convenience function for quick cost estimation."""
    return CostCalculator.calculate_cost(model, num_requests, is_batch=is_batch)


def print_cost_estimate(model: str, num_requests: int, is_batch: bool = False):
    """Convenience function to print cost estimate."""
    cost_info = estimate_cost(model, num_requests, is_batch)
    CostCalculator.print_cost_summary(cost_info)