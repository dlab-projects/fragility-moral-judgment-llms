"""
Streaming dilemma perturber with fine-grained control over perturbations.

1. Stream-based processing with real-time results
2. Control perturbations per dilemma
3. Model selection per perturbation type
4. Result caching 
5. Aim tracking

"""

import logging
import hashlib
import json
import time
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import pandas as pd
from pydantic import BaseModel
from google import genai
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_not_exception_type
from tqdm import tqdm
import difflib
import re
import signal
import sys
import numpy as np
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
from datetime import datetime
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from collections import deque

class ContentFilteredException(Exception):
    """Exception for content that's blocked by safety filters."""
    pass

from .config import (
    PRESENTATION_TEMPLATES,
    VARIATION_TEMPLATES,
    MODEL_CONFIG,
    RATE_LIMITING_CONFIG,
    GENDER_SWAP_CLASSIFICATION_CONFIG
)

class GeminiRateLimiter:
    """
    Rate limiter for Gemini API based on official rate limits.
    
    Gemini 2.5 Flash rate limits by tier:
    - Free: 10 RPM, 250K TPM
    - Tier 1: 1K RPM, 1M TPM  
    - Tier 2: 2K RPM, 3M TPM
    - Tier 3: 10K RPM, 8M TPM
    
    Gemini 2.0 Flash rate limits by tier:
    - Free: 15 RPM, 1M TPM
    - Tier 1: 2K RPM, 4M TPM
    - Tier 2: 10K RPM, 10M TPM  
    - Tier 3: 30K RPM, 30M TPM
    """
    
    def __init__(self, model_name: str = "gemini-2.5-flash", tier: str = "free"):
        self.model_name = model_name
        self.tier = tier.lower()
        
        # Use centralized rate limits configuration
        rate_limits = RATE_LIMITING_CONFIG["rate_limits"]
        
        # Get current limits
        model_key = "gemini-2.5-flash" if "2.5" in model_name else "gemini-2.0-flash"
        limits = rate_limits.get(model_key, rate_limits["gemini-2.5-flash"]).get(tier, rate_limits["gemini-2.5-flash"]["free"])
        
        self.rpm_limit = limits["rpm"]
        self.tpm_limit = limits["tpm"]
        
        # Request tracking
        self.request_times = deque()
        self.token_usage = deque()
        self.lock = threading.Lock()
        
        # Use centralized safety factor configuration
        self.rpm_safety_factor = RATE_LIMITING_CONFIG["safety_factor_rpm"]
        self.tpm_safety_factor = RATE_LIMITING_CONFIG["safety_factor_tpm"]
        
        logging.info(f"Initialized rate limiter for {model_name} ({tier}): {self.rpm_limit} RPM, {self.tpm_limit:,} TPM")
    
    def wait_if_needed(self, estimated_tokens: int = 1000):
        """Wait if necessary to stay within rate limits."""
        logging.debug(f"Rate limiter called with {estimated_tokens} tokens")
        with self.lock:
            now = time.time()
            
            # Clean old entries (older than 1 minute)
            while self.request_times and now - self.request_times[0] > 60:
                self.request_times.popleft()
            while self.token_usage and now - self.token_usage[0][0] > 60:
                self.token_usage.popleft()
            
            # Check RPM limit
            effective_rpm_limit = int(self.rpm_limit * self.rpm_safety_factor)
            logging.debug(f"Current requests in window: {len(self.request_times)}/{effective_rpm_limit}")
            
            if len(self.request_times) >= effective_rpm_limit:
                wait_time = 60 - (now - self.request_times[0]) + 1  # +1 second buffer
                if wait_time > 0:
                    logging.info(f"Rate limit: waiting {wait_time:.1f}s for RPM limit ({len(self.request_times)}/{effective_rpm_limit})")
                    time.sleep(wait_time)
                    # Clean again after waiting
                    now = time.time()
                    while self.request_times and now - self.request_times[0] > 60:
                        self.request_times.popleft()
            
            # Check TPM limit  
            effective_tpm_limit = int(self.tpm_limit * self.tpm_safety_factor)
            current_tokens = sum(tokens for _, tokens in self.token_usage)
            if current_tokens + estimated_tokens > effective_tpm_limit:
                # Wait until oldest token usage expires
                if self.token_usage:
                    wait_time = 60 - (now - self.token_usage[0][0]) + 1  # +1 second buffer
                    if wait_time > 0:
                        logging.info(f"Rate limit: waiting {wait_time:.1f}s for TPM limit ({current_tokens + estimated_tokens:,}/{effective_tpm_limit:,})")
                        time.sleep(wait_time)
            
            # Record this request
            self.request_times.append(now)
            self.token_usage.append((now, estimated_tokens))
    
    def update_actual_tokens(self, actual_tokens: int):
        """Update the last request with actual token usage."""
        with self.lock:
            if self.token_usage:
                timestamp, _ = self.token_usage.pop()
                self.token_usage.append((timestamp, actual_tokens))


class PerturbationResponse(BaseModel):
    """Response schema for perturbation generation."""
    perturbed_text: str
    description_of_changes: str
    scenario_name: Optional[str] = None


class PerturbationResult(BaseModel):
    """Pydantic model for structured batch perturbation outputs."""
    perturbed_text: str
    success: bool
    perturbation_description: str
    perturbation_degree: int  # 0-3 scale
    error: Optional[str] = None


class PerturbationConfig(BaseModel):
    """Configuration for a specific perturbation."""
    perturbation_type: str
    model: str
    is_format_perturbation: bool = False
    priority: int = 1  # For ordering when multiple perturbations requested


class StreamingMetrics(BaseModel):
    """Metrics for streaming perturbation processing."""
    total_requests: int = 0
    successful_requests: int = 0
    cached_results: int = 0
    failed_requests: int = 0
    total_processing_time: float = 0.0
    model_usage: Dict[str, int] = {}
    cache_hit_rate: float = 0.0
    perturbation_type_stats: Dict[str, Dict[str, int]] = {}
    
    def add_request(self, perturbation_type: str, model: str, success: bool, cached: bool, processing_time: float):
        """Add metrics for a completed request."""
        self.total_requests += 1
        self.total_processing_time += processing_time
        
        if cached:
            self.cached_results += 1
        elif success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1
        
        # Update model usage
        if not cached:
            self.model_usage[model] = self.model_usage.get(model, 0) + 1
        
        # Update perturbation type stats
        if perturbation_type not in self.perturbation_type_stats:
            self.perturbation_type_stats[perturbation_type] = {"success": 0, "failed": 0, "cached": 0}
        
        if cached:
            self.perturbation_type_stats[perturbation_type]["cached"] += 1
        elif success:
            self.perturbation_type_stats[perturbation_type]["success"] += 1
        else:
            self.perturbation_type_stats[perturbation_type]["failed"] += 1
        
        # Update cache hit rate
        self.cache_hit_rate = self.cached_results / max(self.total_requests, 1)


class DilemmaPerturber:
    """
    Streaming dilemma perturber with fine-grained control and real-time processing.
    
    Features:
    - Stream results as they complete
    - Choose format OR content perturbations (no forced matrix)
    - Model selection per perturbation
    - Real-time metrics and progress
    """
    
    CACHE_VERSION = "v2.0"
    
    # Available models
    AVAILABLE_MODELS = {
        "flash": "gemini-2.5-flash",
        "pro": "gemini-2.5-pro", 
        "flash-2": "gemini-2.0-flash",
        "default": "gemini-2.5-flash"
    }
    
    def __init__(self, api_key: str, 
                 cache_dir: Optional[str] = None,
                 temperature: float = .4,
                 clear_cache: bool = False,
                 enable_semantic_similarity: bool = True,
                 rate_limit_tier: Optional[str] = None,
                 disable_rate_limiting: Optional[bool] = None):
        """
        Initialize streaming perturber.
        
        Args:
            api_key: Gemini API key
            cache_dir: Directory for result caching
            temperature: Model temperature for responses
            rate_limit_tier: Gemini API tier ("free", "tier1", "tier2", "tier3") - uses config default if None
            clear_cache: If True, clear all cached results on initialization
            enable_semantic_similarity: If True, compute semantic similarity for perturbation degrees
            disable_rate_limiting: If True, disable Gemini API rate limiting for testing - uses config default if None
        """
        self.api_key = api_key
        self.temperature = temperature
        self.enable_semantic_similarity = enable_semantic_similarity
        
        # Use centralized rate limiting configuration for defaults - now disabled by default
        self.rate_limit_tier = rate_limit_tier or RATE_LIMITING_CONFIG["default_tier"]
        self.disable_rate_limiting = disable_rate_limiting if disable_rate_limiting is not None else True  # Disable by default, use exponential backoff
        
        # Initialize Gemini client
        self.client = genai.Client(api_key=api_key)
        
        # Initialize rate limiters (will be created per model as needed)
        self.rate_limiters = {}
                
        # Setup caching
        if cache_dir is None:
            cache_dir = Path.home() / ".cache" / "perturber"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.result_cache = {}
        
        # Clear cache if requested
        if clear_cache:
            self.clear_cache()
            logging.info("Cache cleared on initialization")
        else:
            self._load_cache()
        
        # Initialize sentence transformer for semantic similarity (only if enabled and available)
        if self.enable_semantic_similarity and SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                cache_path = '/Users/tomvannuenen/.cache/torch/sentence_transformers/sentence-transformers_all-MiniLM-L6-v2'
                if os.path.exists(cache_path):
                    self.sentence_model = SentenceTransformer(cache_path)
                else:
                    self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
                logging.info("Loaded sentence embedding model: all-MiniLM-L6-v2")
            except Exception as e:
                logging.warning(f"Failed to load sentence embedding model: {e}. Using lexical similarity only.")
                self.sentence_model = None
        else:
            self.sentence_model = None
            if not SENTENCE_TRANSFORMERS_AVAILABLE:
                logging.info("sentence_transformers not available - using lexical similarity only")
            else:
                logging.info("Semantic similarity disabled - using lexical similarity only")
        
        # Initialize metrics
        self.metrics = StreamingMetrics()
        
        # Cache for semantic embeddings to avoid recomputation
        self.embedding_cache = {}
        
        # Track cache changes for efficient saving
        self._cache_dirty = False
        self._last_cache_save = time.time()
        self._cache_save_interval = 30  # Save cache every 30 seconds at most
        
        logging.info(f"Initialized DilemmaPerturber with temperature: {temperature}")

    def _get_cache_key(self, text: str, perturbation_type: str, model: str) -> str:
        """Generate cache key for text+perturbation+model combination."""
        content = f"{text}|{perturbation_type}|{model}|{self.temperature}"
        return hashlib.sha256(content.encode()).hexdigest()
    
    def _load_cache(self):
        """Load cached results from disk."""
        cache_file = self.cache_dir / f"result_cache_{self.CACHE_VERSION}.json"
        
        # Clean up old cache files from different versions
        for old_cache in self.cache_dir.glob("result_cache_*.json"):
            if old_cache.name != f"result_cache_{self.CACHE_VERSION}.json":
                try:
                    old_cache.unlink()
                except Exception as e:
                    logging.warning(f"Failed to remove old cache {old_cache.name}: {e}")
        
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    self.result_cache = json.load(f)
                logging.info(f"Loaded {len(self.result_cache)} cached results")
            except Exception as e:
                logging.warning(f"Failed to load cache: {e}")
                self.result_cache = {}
        else:
            self.result_cache = {}
    
    def _get_rate_limiter(self, model_name: str) -> GeminiRateLimiter:
        """Get or create a rate limiter for the specified model."""
        if model_name not in self.rate_limiters:
            self.rate_limiters[model_name] = GeminiRateLimiter(model_name, self.rate_limit_tier)
        return self.rate_limiters[model_name]
    
    def _save_cache(self, force: bool = False):
        """Save cached results to disk with throttling."""
        current_time = time.time()
        
        # Only save if cache is dirty and enough time has passed, or if forced
        if not force and (not self._cache_dirty or current_time - self._last_cache_save < self._cache_save_interval):
            return
            
        try:
            cache_file = self.cache_dir / f"result_cache_{self.CACHE_VERSION}.json"
            with open(cache_file, 'w') as f:
                json.dump(self.result_cache, f, separators=(',', ':'))  # Compact format
            self._cache_dirty = False
            self._last_cache_save = current_time
        except Exception as e:
            logging.warning(f"Failed to save cache: {e}")

    def clear_cache(self):
        """Clear all cached results from memory and disk."""
        # Clear in-memory cache
        self.result_cache = {}
        
        # Remove all cache files from disk
        try:
            for cache_file in self.cache_dir.glob("result_cache_*.json"):
                cache_file.unlink()
                logging.info(f"Removed cache file: {cache_file}")
        except Exception as e:
            logging.warning(f"Failed to clear cache files: {e}")
        
        logging.info("Cache cleared successfully")

    def _compute_perturbation_degree(self, original: str, perturbed: str) -> int:
        """Compute perturbation degree based on lexical, length, and semantic similarity metrics."""
        if original == perturbed:
            return 0
        
        # Clean texts for comparison
        original_clean = re.sub(r'\s+', ' ', original.strip().lower())
        perturbed_clean = re.sub(r'\s+', ' ', perturbed.strip().lower())
        
        # 1. Lexical similarity using difflib
        lexical_similarity = difflib.SequenceMatcher(None, original_clean, perturbed_clean).ratio()
        
        # 2. Word and character count differences
        original_words = original_clean.split()
        perturbed_words = perturbed_clean.split()
        word_count_ratio = abs(len(original_words) - len(perturbed_words)) / max(len(original_words), 1)
        char_count_ratio = abs(len(original_clean) - len(perturbed_clean)) / max(len(original_clean), 1)
        
        # 3. Semantic similarity using sentence embeddings (with caching, if enabled)
        semantic_similarity = 0.0  # Default if model unavailable
        if self.enable_semantic_similarity and self.sentence_model is not None:
            try:
                # Use caching for embeddings to avoid recomputation
                original_key = hashlib.sha256(original.strip().encode()).hexdigest()[:16]
                perturbed_key = hashlib.sha256(perturbed.strip().encode()).hexdigest()[:16]
                
                # Get or compute original embedding
                if original_key not in self.embedding_cache:
                    self.embedding_cache[original_key] = self.sentence_model.encode([original.strip()])[0]
                original_embedding = self.embedding_cache[original_key]
                
                # Get or compute perturbed embedding
                if perturbed_key not in self.embedding_cache:
                    self.embedding_cache[perturbed_key] = self.sentence_model.encode([perturbed.strip()])[0]
                perturbed_embedding = self.embedding_cache[perturbed_key]
                
                # Cosine similarity between embeddings
                cos_sim = np.dot(original_embedding, perturbed_embedding) / (
                    np.linalg.norm(original_embedding) * np.linalg.norm(perturbed_embedding)
                )
                semantic_similarity = float(cos_sim)
                
                # Limit cache size to prevent memory issues
                if len(self.embedding_cache) > 1000:
                    # Remove oldest 100 entries (simple eviction)
                    keys_to_remove = list(self.embedding_cache.keys())[:100]
                    for key in keys_to_remove:
                        del self.embedding_cache[key]
                        
            except Exception as e:
                logging.warning(f"Semantic similarity computation failed: {e}")
                semantic_similarity = lexical_similarity  # Fallback to lexical
        else:
            semantic_similarity = lexical_similarity  # Fallback if no model or disabled
        
        # Combine metrics for degree classification
        lexical_dissimilarity = 1 - lexical_similarity
        semantic_dissimilarity = 1 - semantic_similarity
        
        # Weight the different metrics (semantic is most important for perturbation degree)
        combined_dissimilarity = (
            0.5 * semantic_dissimilarity +  # Semantic changes matter most
            0.3 * lexical_dissimilarity +   # Surface-level changes 
            0.2 * max(word_count_ratio, char_count_ratio)  # Length changes
        )
        
        # Classify into 5 degrees based on combined dissimilarity
        if combined_dissimilarity <= 0.1:
            return 1  # Minimal change (mostly cosmetic)
        elif combined_dissimilarity <= 0.25:
            return 2  # Small change (minor semantic shift)
        elif combined_dissimilarity <= 0.45:
            return 3  # Moderate change (noticeable semantic difference)
        elif combined_dissimilarity <= 0.65:
            return 4  # Significant change (substantial semantic difference)
        else:
            return 5  # Extensive change (major semantic transformation)

    @retry(
        stop=stop_after_attempt(5), 
        wait=wait_exponential(multiplier=2, min=1, max=60),
        retry=retry_if_not_exception_type(ContentFilteredException)
    )

    def _classify_gender_swap(self, original_text: str, perturbed_text: str, model: str) -> bool:
        """
        Classify whether a gender swap perturbation makes sense.
        
        Args:
            original_text: The original text before perturbation
            perturbed_text: The text after gender swap perturbation
            model: The model to use for classification
            
        Returns:
            True if gender swap makes sense, False otherwise
        """
        # Check if gender swap classification is enabled
        if not GENDER_SWAP_CLASSIFICATION_CONFIG.get("enabled", True):
            return None
            
        prompt = GENDER_SWAP_CLASSIFICATION_CONFIG["prompt_template"].format(
            original_text=original_text,
            perturbed_text=perturbed_text
        )
        
        try:
            # Apply rate limiting before making the request (if enabled)
            if not self.disable_rate_limiting:
                rate_limiter = self._get_rate_limiter(model)
                estimated_tokens = len(prompt.split()) * 1.3  # Rough token estimation
                rate_limiter.wait_if_needed(int(estimated_tokens))
            
            response = self.client.models.generate_content(
                model=model,
                contents=prompt,
                config={
                    "temperature": GENDER_SWAP_CLASSIFICATION_CONFIG.get("temperature", 0.1),
                    "max_output_tokens": GENDER_SWAP_CLASSIFICATION_CONFIG.get("max_tokens", 10)
                }
            )
            
            # Update rate limiter with actual usage if available (if rate limiting enabled)
            if not self.disable_rate_limiting:
                try:
                    if hasattr(response, 'usage_metadata') and response.usage_metadata:
                        actual_tokens = getattr(response.usage_metadata, 'total_token_count', int(estimated_tokens))
                        rate_limiter.update_actual_tokens(actual_tokens)
                except Exception:
                    pass  # Ignore usage tracking errors
            
            if response and response.text:
                result = response.text.strip().lower()
                return result == "true"
            else:
                logging.warning("No response from gender swap classification API")
                return False
                
        except Exception as e:
            logging.error(f"Error in gender swap classification: {e}")
            return False

    def _call_api_structured(self, prompt: str, model: str) -> PerturbationResponse:
        """Make structured API call with retry logic."""
        try:
            # Apply rate limiting before making the request (if enabled)
            if not self.disable_rate_limiting:
                rate_limiter = self._get_rate_limiter(model)
                estimated_tokens = len(prompt.split()) * 1.3  # Rough token estimation
                rate_limiter.wait_if_needed(int(estimated_tokens))
            
            response = self.client.models.generate_content(
                model=model,
                contents=prompt,
                config={
                    "response_mime_type": "application/json",
                    "response_schema": PerturbationResponse,
                    "temperature": self.temperature,
                    "max_output_tokens": 8192  # Prevent runaway generation
                }
            )
            
            # Update rate limiter with actual usage if available (if rate limiting enabled)
            if not self.disable_rate_limiting:
                try:
                    if hasattr(response, 'usage_metadata') and response.usage_metadata:
                        actual_tokens = getattr(response.usage_metadata, 'total_token_count', int(estimated_tokens))
                        rate_limiter.update_actual_tokens(actual_tokens)
                except Exception:
                    pass  # Ignore usage tracking errors
            
            if response and response.parsed:
                return response.parsed
            else:
                # Check for content filtering
                if response and hasattr(response, 'prompt_feedback') and response.prompt_feedback:
                    if hasattr(response.prompt_feedback, 'block_reason') and response.prompt_feedback.block_reason:
                        if 'PROHIBITED_CONTENT' in str(response.prompt_feedback.block_reason):
                            logging.warning(f"Content blocked by safety filter for model {model}")
                            raise ContentFilteredException("Content blocked by safety filter")
                
                # Check candidates for content filtering
                if response and hasattr(response, 'candidates') and response.candidates:
                    for candidate in response.candidates:
                        if hasattr(candidate, 'finish_reason') and candidate.finish_reason:
                            if 'PROHIBITED_CONTENT' in str(candidate.finish_reason):
                                logging.warning(f"Content blocked by safety filter for model {model}")
                                raise ContentFilteredException("Content blocked by safety filter")
                            elif 'MAX_TOKENS' in str(candidate.finish_reason):
                                logging.warning(f"Response hit max tokens limit for model {model}")
                                raise ContentFilteredException("Response hit max tokens limit")
                
                logging.error(f"Empty structured response from API for model {model}")
                logging.error(f"Response object: {response}")
                if response and hasattr(response, 'prompt_feedback'):
                    logging.error(f"Prompt feedback: {response.prompt_feedback}")
                raise Exception("Empty structured response from API")
        except Exception as e:
            error_msg = str(e)
            if "timeout" in error_msg.lower():
                logging.warning(f"API timeout after 120s for model {model} - will retry")
            elif "quota" in error_msg.lower() or "rate" in error_msg.lower():
                logging.warning(f"Rate limit/quota error for model {model} - will retry after exponential backoff")
            else:
                logging.error(f"Structured API call failed for model {model}: {error_msg}")
            raise

    @retry(
        stop=stop_after_attempt(5), 
        wait=wait_exponential(multiplier=2, min=1, max=60),
        retry=retry_if_not_exception_type(ContentFilteredException)
    )

    def _call_api_simple(self, prompt: str, model: str) -> str:
        """Make simple API call for format changes that don't need descriptions."""
        try:
            # Apply rate limiting before making the request (if enabled)
            if not self.disable_rate_limiting:
                rate_limiter = self._get_rate_limiter(model)
                estimated_tokens = len(prompt.split()) * 1.3  # Rough token estimation
                rate_limiter.wait_if_needed(int(estimated_tokens))
            
            response = self.client.models.generate_content(
                model=model,
                contents=prompt,
                config={
                    "temperature": self.temperature,
                    "max_output_tokens": 8192  # Prevent runaway generation
                }
            )
            
            # Update rate limiter with actual usage if available (if rate limiting enabled)
            if not self.disable_rate_limiting:
                try:
                    if hasattr(response, 'usage_metadata') and response.usage_metadata:
                        actual_tokens = getattr(response.usage_metadata, 'total_token_count', int(estimated_tokens))
                        rate_limiter.update_actual_tokens(actual_tokens)
                except Exception:
                    pass  # Ignore usage tracking errors
            
            if response and response.text:
                return response.text.strip()
            else:
                # Check for content filtering
                if response and hasattr(response, 'prompt_feedback') and response.prompt_feedback:
                    if hasattr(response.prompt_feedback, 'block_reason') and response.prompt_feedback.block_reason:
                        if 'PROHIBITED_CONTENT' in str(response.prompt_feedback.block_reason):
                            logging.warning(f"Content blocked by safety filter for model {model}")
                            raise ContentFilteredException("Content blocked by safety filter")
                
                # Check candidates for content filtering
                if response and hasattr(response, 'candidates') and response.candidates:
                    for candidate in response.candidates:
                        if hasattr(candidate, 'finish_reason') and candidate.finish_reason:
                            if 'PROHIBITED_CONTENT' in str(candidate.finish_reason):
                                logging.warning(f"Content blocked by safety filter for model {model}")
                                raise ContentFilteredException("Content blocked by safety filter")
                            elif 'MAX_TOKENS' in str(candidate.finish_reason):
                                logging.warning(f"Response hit max tokens limit for model {model}")
                                raise ContentFilteredException("Response hit max tokens limit")
                
                logging.error(f"Empty response from API for model {model}")
                logging.error(f"Response object: {response}")
                if response and hasattr(response, 'prompt_feedback'):
                    logging.error(f"Prompt feedback: {response.prompt_feedback}")
                raise Exception("Empty response from API")
        except Exception as e:
            error_msg = str(e)
            print(error_msg)
            if "timeout" in error_msg.lower():
                logging.warning(f"API timeout after 120s for model {model} - will retry")
            elif "quota" in error_msg.lower() or "rate" in error_msg.lower():
                logging.warning(f"Rate limit/quota error for model {model} - will retry after exponential backoff")
            else:
                logging.error(f"Simple API call failed for model {model}: {error_msg}")
            raise

    def _create_structured_prompt(self, base_template: str, text: str, perturbation_type: str) -> str:
        """Create a structured prompt that asks for perturbation, description, and scenario name."""
        
        # Extract the main instruction from the template (remove {text} placeholder)
        instruction = base_template.replace("Text to modify:\n{text}", "").replace("{text}", "").strip()
        
        structured_prompt = f"""
{instruction}

Additionally, provide a one-sentence description of exactly what you changed.

Text to modify:
{text}
"""
        return structured_prompt

    def _generate_scenario_name(self, text: str, model: str = None) -> str:
        """Generate a scenario name for baseline text."""
        if model is None:
            model = self.AVAILABLE_MODELS["flash"]  # Use fast model for scenario naming
        
        prompt = f"""
Please provide a 2-word scenario name (e.g., "wedding_invite", "pizza_roommate", "work_conflict") that captures the core situation in this ethical dilemma for quick reference.

Return only the 2-word scenario name, nothing else.

Text:
{text}
"""
        try:
            scenario_name = self._call_api_simple(prompt, model).strip()
            # Clean up the response - only use fallback for empty responses
            if not scenario_name.strip():
                scenario_name = "unknown_scenario"
            return scenario_name
        except Exception as e:
            logging.warning(f"Failed to generate scenario name: {e}")
            return "unknown_scenario"
    
    def _generate_scenario_names_batch(self, texts: List[str], model: str = None, progress_callback: Optional[callable] = None) -> List[str]:
        """Generate scenario names for multiple texts in batch to reduce API calls."""
        if model is None:
            model = self.AVAILABLE_MODELS["flash"]
        
        logging.info(f"Starting scenario name generation for {len(texts)} texts using model {model}")
        scenario_names = []
        
        # Process in batches to avoid overly long prompts
        batch_size = 5
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            logging.info(f"Processing scenario name batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}: {len(batch_texts)} texts")
            
            prompt = "Please provide 2-word scenario names for these ethical dilemmas. Format your response as a numbered list with just the scenario names:\n\n"
            
            for idx, text in enumerate(batch_texts, 1):
                prompt += f"{idx}. Text: {text[:200]}...\n\n"  # Truncate long texts
            
            prompt += "Return only the numbered list of 2-word scenario names (e.g., '1. wedding_invite', '2. pizza_roommate'):"
            
            try:
                logging.info(f"Making API call for scenario names batch {i//batch_size + 1}")
                response = self._call_api_simple(prompt, model).strip()
                logging.info(f"Received response for scenario names batch {i//batch_size + 1}")
                
                # Update progress
                if progress_callback:
                    progress_callback()
                
                # Parse the response
                batch_names = []
                for line in response.split('\n'):
                    line = line.strip()
                    if line and any(char.isdigit() for char in line[:3]):
                        # Extract scenario name after number
                        parts = line.split('.', 1)
                        if len(parts) > 1:
                            name = parts[1].strip()
                            # Accept any non-empty scenario name
                            if name.strip():
                                batch_names.append(name)
                            else:
                                batch_names.append("unknown_scenario")
                        else:
                            batch_names.append("unknown_scenario")
                
                # Ensure we have the right number of names
                while len(batch_names) < len(batch_texts):
                    batch_names.append("unknown_scenario")
                
                scenario_names.extend(batch_names[:len(batch_texts)])
                
                # Add small delay between batches to avoid rate limits
                if i + batch_size < len(texts):  # Don't delay after the last batch
                    time.sleep(0.5)  # 500ms delay between batches
                
            except Exception as e:
                logging.warning(f"Failed to generate batch scenario names: {e}")
                # Fallback to individual generation for this batch
                for text in batch_texts:
                    scenario_names.append(self._generate_scenario_name(text, model))
        
        return scenario_names

    def apply_perturbation(
        self, 
        text: str, 
        perturbation_config: PerturbationConfig,
        original_text: Optional[str] = None,
        scenario_name: Optional[str] = None,
        generate_scenario_name: bool = False
    ) -> Dict[str, Any]:
        """
        Apply a single perturbation to text with streaming result.
        
        Args:
            text: Text to perturb (could be original or already perturbed)
            perturbation_config: Configuration for the perturbation
            original_text: Original text for degree calculation (if text is already perturbed)
            scenario_name: Pre-generated scenario name (if None, will use fallback)
            generate_scenario_name: If True, generate scenario name as part of this perturbation
            
        Returns:
            Dictionary with perturbation result and metadata
        """
        start_time = time.time()
        perturbation_type = perturbation_config.perturbation_type
        model = perturbation_config.model
        is_format = perturbation_config.is_format_perturbation
        
        # For non-baseline perturbations, ensure scenario_name is provided
        if scenario_name is None and perturbation_type != 'none' and not generate_scenario_name:
            scenario_name = "unknown_scenario"
            logging.warning(f"scenario_name not provided for {perturbation_type} - using fallback")
        
        # Check cache first
        cache_key = self._get_cache_key(text, perturbation_type, model)
        if cache_key in self.result_cache:
            cached_result = self.result_cache[cache_key]
            processing_time = time.time() - start_time
            self.metrics.add_request(perturbation_type, model, True, True, processing_time)
            return cached_result
        
        # Initialize gender swap classification result
        gender_swap_makes_sense = None
        
        try:
            # Handle baseline case
            if perturbation_type == 'none':
                if generate_scenario_name:
                    # Generate scenario name using a simple API call
                    prompt = f"""
Please provide a 2-word scenario name (e.g., "wedding_invite", "pizza_roommate", "work_conflict") that captures the core situation in this ethical dilemma for quick reference.

Return only the 2-word scenario name, nothing else.

Text:
{text}
"""
                    try:
                        generated_name = self._call_api_simple(prompt, model).strip()
                        logging.info(f"Generated scenario name raw response: '{generated_name}'")
                        # Clean up the response - only use fallback for empty responses
                        if generated_name.strip():
                            scenario_name = generated_name
                            logging.info(f"Using generated scenario name: '{scenario_name}'")
                        else:
                            logging.warning(f"Empty scenario name response, using fallback")
                            scenario_name = "unknown_scenario"
                    except Exception as e:
                        logging.warning(f"Failed to generate scenario name: {e}")
                        scenario_name = "unknown_scenario"
                else:
                    if scenario_name is None:
                        scenario_name = "unknown_scenario"
                        logging.warning("scenario_name not provided for baseline case")
                
                result = {
                    'scenario_name': scenario_name,
                    'perturbation_type': perturbation_type,
                    'model': model,
                    'perturbed_text': text,
                    'perturbation_description': "No changes made (baseline)",
                    'perturbation_degree': 0,
                    'gender_swap_makes_sense': gender_swap_makes_sense,
                    'success': True,
                    'error': None,
                    'processing_time': time.time() - start_time
                }
            else:
                # Get the appropriate template
                if is_format:
                    if perturbation_type not in PRESENTATION_TEMPLATES:
                        raise ValueError(f"Unknown format perturbation: {perturbation_type}")
                    template = PRESENTATION_TEMPLATES[perturbation_type]
                    
                    # For format perturbations, use simple API call
                    if perturbation_type == 'aita':
                        # AITA is pass-through
                        perturbed_text = text
                        description = "Applied AITA format (pass-through)"
                        # scenario_name should always be provided by caller to prevent race conditions
                        if scenario_name is None:
                            scenario_name = "unknown_scenario"
                            logging.warning("scenario_name not provided for AITA format")
                    else:
                        format_prompt = template.format(text=text)
                        perturbed_text = self._call_api_simple(format_prompt, model)
                        description = f"Applied {perturbation_type} format transformation"
                        # scenario_name should always be provided by caller to prevent race conditions
                        if scenario_name is None:
                            scenario_name = "unknown_scenario"
                            logging.warning(f"scenario_name not provided for {perturbation_type} format")
                else:
                    if perturbation_type not in VARIATION_TEMPLATES:
                        raise ValueError(f"Unknown content perturbation: {perturbation_type}")
                    template = VARIATION_TEMPLATES[perturbation_type]
                    
                    # For content perturbations, use structured API call
                    enhanced_prompt = self._create_structured_prompt(template, text, perturbation_type)
                    response = self._call_api_structured(enhanced_prompt, model)
                    perturbed_text = response.perturbed_text
                    description = response.description_of_changes
                    
                    # Add gender swap classification if this is a gender swap perturbation
                    if perturbation_type == 'gender_swap':
                        gender_swap_makes_sense = self._classify_gender_swap(text, perturbed_text, model)
                
                # Compute perturbation degree
                comparison_text = original_text if original_text else text
                perturbation_degree = self._compute_perturbation_degree(comparison_text, perturbed_text)
                
                result = {
                    'scenario_name': scenario_name,
                    'perturbation_type': perturbation_type,
                    'model': model,
                    'perturbed_text': perturbed_text,
                    'perturbation_description': description,
                    'perturbation_degree': perturbation_degree,
                    'gender_swap_makes_sense': gender_swap_makes_sense,
                    'success': True,
                    'error': None,
                    'processing_time': time.time() - start_time
                }
            
            # Cache the result
            self.result_cache[cache_key] = result
            self._cache_dirty = True
            
            # Update metrics
            processing_time = time.time() - start_time
            self.metrics.add_request(perturbation_type, model, True, False, processing_time)
            
            return result
            
        except ContentFilteredException as e:
            error_msg = str(e)
            logging.warning(f"Content filtered for perturbation {perturbation_type}: {error_msg}")
            
            # Create filtered result - return original text with success=False but no retry
            result = {
                'scenario_name': scenario_name if scenario_name else 'filtered_scenario',
                'perturbation_type': perturbation_type,
                'model': model,
                'perturbed_text': text,  # Return original text when filtered
                'perturbation_description': f"CONTENT_FILTERED: {error_msg}",
                'perturbation_degree': 0,  # No change since original text returned
                'gender_swap_makes_sense': gender_swap_makes_sense,
                'success': False,
                'error': f"CONTENT_FILTERED: {error_msg}",
                'processing_time': time.time() - start_time
            }
            
            # Update metrics for filtered content
            processing_time = time.time() - start_time
            self.metrics.add_request(perturbation_type, model, False, True, processing_time)  # True for cache_hit to distinguish from real errors
            
            return result
            
        except Exception as e:
            error_msg = str(e)
            logging.error(f"Failed to apply perturbation {perturbation_type}: {error_msg}")
            
            # Create error result
            result = {
                'scenario_name': scenario_name if scenario_name else 'error_scenario',
                'perturbation_type': perturbation_type,
                'model': model,
                #'is_format_perturbation': is_format,
                'perturbed_text': text,  # Return original text on error
                'perturbation_description': f"ERROR: {error_msg}",
                'perturbation_degree': -1,
                'gender_swap_makes_sense': gender_swap_makes_sense,
                'success': False,
                'error': error_msg,
                'processing_time': time.time() - start_time
            }
            
            # Update metrics
            processing_time = time.time() - start_time
            self.metrics.add_request(perturbation_type, model, False, False, processing_time)
            
            return result

    def stream_perturbations(
        self,
        texts: List[str],
        perturbation_configs: List[PerturbationConfig],
        progress_callback: Optional[callable] = None,
        parallel: bool = True,
        max_workers: int = 8
    ):
        """
        Stream perturbations as they complete.
        
        Args:
            texts: List of texts to process
            perturbation_configs: List of perturbation configurations to apply
            progress_callback: Optional callback for progress updates
            parallel: If True, run perturbations concurrently within each text
            max_workers: Maximum number of concurrent workers (only used if parallel=True)
            
        Yields:
            Tuple of (text_index, perturbation_result)
        """
        total_operations = len(texts) * len(perturbation_configs)
        
        # Scenario names will be generated during the first perturbation for each text
        
        for text_idx, text in enumerate(texts):
            # Sort perturbations by priority
            sorted_configs = sorted(perturbation_configs, key=lambda x: x.priority)
            
            original_text = text  # Keep reference to original
            
            # Scenario names will be generated during the first perturbation (baseline)
            scenario_name = None  # Will be set by the first perturbation
            
            if parallel and len(sorted_configs) > 1:
                # For parallel processing, we need to ensure the baseline (none) perturbation runs first
                # to generate the scenario name that other perturbations will use
                captured_text_idx = text_idx
                captured_original_text = original_text
                
                baseline_config = None
                other_configs = []
                
                for config in sorted_configs:
                    if config.perturbation_type == 'none':
                        baseline_config = config
                    else:
                        other_configs.append(config)
                
                # Process baseline first to get scenario name
                if baseline_config:
                    baseline_result = self.apply_perturbation(captured_original_text, baseline_config, captured_original_text, None, generate_scenario_name=True)
                    baseline_result['text_index'] = captured_text_idx
                    captured_scenario_name = baseline_result.get('scenario_name', 'unknown_scenario')
                    yield text_idx, baseline_result
                    if progress_callback:
                        progress_callback(1)
                else:
                    captured_scenario_name = "unknown_scenario"
                
                # Now process other perturbations with the scenario name
                def apply_perturbation_with_metadata(config):
                    result = self.apply_perturbation(captured_original_text, config, captured_original_text, captured_scenario_name)
                    result['text_index'] = captured_text_idx
                    return result
                
                # Use ThreadPoolExecutor for remaining perturbations
                if other_configs:
                    with ThreadPoolExecutor(max_workers=min(max_workers, len(other_configs))) as executor:
                        # Submit remaining perturbations
                        future_to_config = {
                            executor.submit(apply_perturbation_with_metadata, config): config 
                            for config in other_configs
                        }
                        
                        # Yield results as they complete
                        for future in as_completed(future_to_config):
                            result = future.result()
                            # Validation: ensure result has correct text_index and scenario_name
                            if result.get('text_index') != captured_text_idx:
                                logging.error(f"CRITICAL: text_index mismatch in parallel result! Expected {captured_text_idx}, got {result.get('text_index')}")
                            if result.get('scenario_name') != captured_scenario_name:
                                logging.error(f"CRITICAL: scenario_name mismatch! Expected '{captured_scenario_name}', got '{result.get('scenario_name')}'")
                            yield text_idx, result
                            
                            # Update progress
                            if progress_callback:
                                progress_callback(1)
            else:
                # Sequential processing (default behavior)
                for i, config in enumerate(sorted_configs):
                    # For the first perturbation, generate scenario name if not set yet
                    generate_scenario = (i == 0 and scenario_name is None)
                    
                    # Apply perturbation to original text
                    result = self.apply_perturbation(original_text, config, original_text, scenario_name, generate_scenario_name=generate_scenario)
                    
                    # Update scenario_name for subsequent perturbations
                    if scenario_name is None and result.get('scenario_name'):
                        scenario_name = result['scenario_name']
                    
                    # Add text metadata
                    result['text_index'] = text_idx
                    
                    # Validation: ensure result has correct metadata
                    if result.get('text_index') != text_idx:
                        logging.error(f"CRITICAL: text_index mismatch in sequential result! Expected {text_idx}, got {result.get('text_index')}")
                    
                    # Yield result immediately (streaming)
                    yield text_idx, result
                    
                    # Update progress
                    if progress_callback:
                        progress_callback(1)
        
        # Save cache after processing (force save)
        self._save_cache(force=True)

    def get_metrics_report(self) -> Dict[str, Any]:
        """Get comprehensive metrics report."""
        return {
            "summary": {
                "total_requests": self.metrics.total_requests,
                "successful_requests": self.metrics.successful_requests,
                "cached_results": self.metrics.cached_results,
                "failed_requests": self.metrics.failed_requests,
                "cache_hit_rate": f"{self.metrics.cache_hit_rate:.2%}",
                "total_processing_time": f"{self.metrics.total_processing_time:.2f}s",
                "average_request_time": f"{self.metrics.total_processing_time / max(self.metrics.total_requests, 1):.2f}s"
            },
            "model_usage": self.metrics.model_usage,
            "perturbation_type_stats": self.metrics.perturbation_type_stats
        }

    @classmethod
    def get_available_perturbations(cls) -> Dict[str, List[str]]:
        """Get lists of available perturbations by type."""
        return {
            "content_perturbations": list(VARIATION_TEMPLATES.keys()),
            "format_perturbations": list(PRESENTATION_TEMPLATES.keys()),
            "available_models": list(cls.AVAILABLE_MODELS.keys())
        }


def create_perturbed_dataset(
    input_file: str,
    output_file: str,
    api_key: str,
    perturbation_configs: List[PerturbationConfig],
    metadata_columns: List[str],
    max_samples: Optional[int] = None,
    text_column: str = 'selftext_cleaned',
    cache_dir: Optional[str] = None,
    temperature: float = 0.4,
    resume: bool = True,
    clear_cache: bool = False,
    parallel: bool = True,
    max_workers: int = 8,
    enable_semantic_similarity: bool = True,
    progress_save_interval: int = 10,
    disable_rate_limiting: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Create perturbed dataset with metadata preservation.
    
    ⚠️  METADATA IS MANDATORY: This function always preserves metadata to prevent data loss.
    
    Args:
        input_file: Path to CSV with original dilemmas
        output_file: Path to save perturbed dataset
        api_key: Gemini API key
        perturbation_configs: List of perturbation configurations to apply
        metadata_columns: List of column names to preserve from input (REQUIRED)
        max_samples: Max number of original texts to process
        text_column: Column name with dilemma texts
        cache_dir: Directory for result caching
        temperature: Model temperature
        resume: If True, resume from existing temp file if it exists
        clear_cache: If True, clear all cached results before processing
        parallel: If True, run perturbations concurrently for faster processing
        max_workers: Maximum number of concurrent workers
        enable_semantic_similarity: If True, compute semantic similarity
        progress_save_interval: Save progress every N results
        disable_rate_limiting: If True, disable Gemini API rate limiting
        
    Returns:
        Tuple of (final DataFrame with metadata, metrics report)
    """
    # Setup temp file path
    temp_file = Path(output_file).with_suffix('.temp.csv')
    
    # Handle resume logic
    all_results = []
    start_idx = 0
    
    if resume and temp_file.exists():
        try:
            existing_df = pd.read_csv(temp_file)
            if not existing_df.empty:
                all_results = existing_df.to_dict('records')
                
                # Resume from last text_index
                if 'text_index' in existing_df.columns:
                    start_idx = existing_df['text_index'].max() + 1
                else:
                    # Fallback
                    expected_per_text = len(perturbation_configs)
                    if expected_per_text > 0:
                        start_idx = len(all_results) // expected_per_text
                
                logging.info(f"Resuming from temp file: {len(all_results)} existing results, starting from text {start_idx}")
        except Exception as e:
            logging.warning(f"Could not resume from temp file: {e}. Starting fresh.")
            all_results = []
            start_idx = 0
    
    # Read input
    try:
        df = pd.read_csv(input_file)
        logging.info(f"Loaded {len(df)} rows from {input_file}")
    except Exception as e:
        logging.error(f"Failed to read {input_file}: {e}")
        return pd.DataFrame(), {}

    if text_column not in df.columns:
        logging.error(f"Column '{text_column}' not found in input file")
        return pd.DataFrame(), {}

    # Identify available metadata columns
    available_metadata = [col for col in metadata_columns if col in df.columns]
    missing_metadata = [col for col in metadata_columns if col not in df.columns]
    
    if missing_metadata:
        logging.warning(f"Requested metadata columns not found: {missing_metadata}")
    
    logging.info(f"Preserving metadata columns: {available_metadata}")

    # Sample if requested
    if max_samples:
        df = df.head(max_samples)
        logging.info(f"Using {len(df)} samples")

    # Create streaming perturber
    perturber = DilemmaPerturber(
        api_key=api_key,
        cache_dir=cache_dir,
        temperature=temperature,
        clear_cache=clear_cache,
        enable_semantic_similarity=enable_semantic_similarity,
        disable_rate_limiting=disable_rate_limiting
    )
    
    # Extract texts to process (skip already processed)
    texts = df[text_column].tolist()[start_idx:]
    
    # Calculate total operations for progress tracking
    total_operations = len(texts) * len(perturbation_configs)
    
    # Setup signal handler for graceful interruption
    def signal_handler(signum, frame):
        logging.info("Interrupt received, saving progress...")
        if all_results:
            temp_df = pd.DataFrame(all_results)
            temp_df.to_csv(temp_file, index=False)
            logging.info(f"Progress saved to {temp_file}")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Stream process with progress bar
    with tqdm(total=total_operations, desc="Streaming perturbations", unit="operation") as pbar:
        for text_idx, result in perturber.stream_perturbations(
            texts, 
            perturbation_configs, 
            progress_callback=pbar.update,
            parallel=parallel,
            max_workers=max_workers
        ):
            # Adjust text index for resume
            actual_idx = text_idx + start_idx
            result['text_index'] = actual_idx
            
            # Add metadata from original row
            original_row = df.iloc[actual_idx]
            for col in available_metadata:
                result[col] = original_row[col]
            
            # Add to results
            all_results.append(result)
            
            # Save progress periodically
            if len(all_results) % progress_save_interval == 0:
                temp_df = pd.DataFrame(all_results)
                temp_df.to_csv(temp_file, index=False)

    # Get metrics report
    metrics_report = perturber.get_metrics_report()
    
    # Save final results
    if all_results:
        final_df = pd.DataFrame(all_results)
        
        # Validation
        print("\n=== FINAL VALIDATION ===")
        scenario_text_index_check = final_df.groupby('scenario_name')['text_index'].nunique()
        problematic_scenarios = scenario_text_index_check[scenario_text_index_check > 1]
        
        if len(problematic_scenarios) > 0:
            logging.error(f"❌ VALIDATION FAILED: {len(problematic_scenarios)} scenarios have inconsistent text_index!")
            print(f"❌ VALIDATION FAILED: {len(problematic_scenarios)} scenarios have inconsistent text_index!")
        else:
            print(f"✅ VALIDATION PASSED: All scenarios have consistent text_index")
            print(f"✅ Total scenarios: {final_df['scenario_name'].nunique()}")
            print(f"✅ Total perturbations: {len(final_df)}")
            print(f"✅ Preserved metadata columns: {', '.join(available_metadata)}")
        
        # Reorder columns to put metadata first
        column_order = ['text_index', 'scenario_name', 'perturbation_type'] + available_metadata
        other_cols = [col for col in final_df.columns if col not in column_order]
        final_df = final_df[column_order + other_cols]
        
        final_df.to_csv(output_file, index=False)
        
        # Clean up temp file on successful completion
        if temp_file.exists():
            temp_file.unlink()
            logging.info(f"Removed temp file {temp_file}")
        
        # Log stats
        success_rate = final_df['success'].mean() * 100
        logging.info(f"Saved {len(final_df)} perturbations to {output_file}")
        logging.info(f"Success rate: {success_rate:.1f}%")
        logging.info(f"Metadata preserved: {available_metadata}")
        
        return final_df, metrics_report
    else:
        logging.error("No perturbations were generated")
        return pd.DataFrame(), metrics_report


# Legacy aliases for backward compatibility
PerturbationMetrics = StreamingMetrics
create_perturbed_dataset_with_metadata = create_perturbed_dataset  # Metadata is now mandatory