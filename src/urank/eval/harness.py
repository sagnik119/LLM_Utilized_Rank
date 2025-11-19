"""
Wrapper around EleutherAI lm-evaluation-harness for comprehensive evaluation.

This module provides integration with lm-eval-harness to run multiple tasks
and return metrics including perplexity where supported.

Prerequisites:
    pip install lm-eval

Example:
    >>> from urank.eval.harness import run_lm_eval
    >>> results = run_lm_eval(
    ...     "ckpts/gpt2_compressed",
    ...     tasks=["wikitext", "arc_easy", "hellaswag"],
    ...     batch_size=8
    ... )
    >>> print(results["results"]["wikitext"]["perplexity"])
"""

from typing import Dict, List, Optional
import warnings
import torch
import numpy as np


def sanitize_json(o):
    """Convert numpy types → Python native JSON-safe types."""
    if isinstance(o, (np.generic,)):
        return o.item()
    if isinstance(o, dict):
        return {k: sanitize_json(v) for k, v in o.items()}
    if isinstance(o, list):
        return [sanitize_json(v) for v in o]
    if isinstance(o, tuple):
        return tuple(sanitize_json(v) for v in o)
    return o

def run_lm_eval(
    model_name_or_path: str,
    tasks: List[str],
    batch_size: int = 8,
    limit: Optional[int] = None,
    device: Optional[str] = None,
    trust_remote_code: bool = True,
    output_path: Optional[str] = None,
) -> Dict:
    """
    Run lm-evaluation-harness on specified tasks.
    
    Args:
        model_name_or_path: HuggingFace model path or local checkpoint directory
        tasks: List of task names (e.g., ["wikitext", "arc_easy", "hellaswag"])
        batch_size: Batch size for evaluation (default: 8)
        limit: Optional limit on number of examples per task
        device: Device to use (default: auto-detect)
        trust_remote_code: Whether to trust remote code for custom models
        output_path: Optional path to save results JSON
        
    Returns:
        Dictionary containing evaluation results with structure:
        {
            "results": {
                "task_name": {
                    "acc": 0.xx,
                    "perplexity": xx.xx,  # if applicable
                    ...
                }
            },
            "config": {...}
        }
        
    Example:
        >>> results = run_lm_eval(
        ...     "gpt2",
        ...     tasks=["wikitext", "arc_easy"],
        ...     batch_size=16
        ... )
        >>> print(f"WikiText PPL: {results['results']['wikitext']['word_perplexity']}")
    """
    try:
        # Import lazily to avoid hard dependency when not needed
        from lm_eval import evaluator
        from lm_eval.models.huggingface import HFLM
    except ImportError as e:
        raise ImportError(
            "lm-eval not installed. Install with: pip install lm-eval"
        ) from e
    
    # Normalize device to string (lm-eval requires a str)
    
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    elif isinstance(device, int):
        device = f"cuda:{device}"
    elif not isinstance(device, str):
        raise ValueError(f"Device must be a string, not: {device}")

    # Pre-load custom architecture if needed (workaround for lm-eval trust_remote_code issue)
    import os
    if trust_remote_code and os.path.exists(model_name_or_path):
        modeling_file = os.path.join(model_name_or_path, "modeling_gpt2_compressed.py")
        if os.path.exists(modeling_file):
            # Pre-register custom config and model before lm-eval loads it
            import sys
            sys.path.insert(0, model_name_or_path)
            try:
                from modeling_gpt2_compressed import GPT2CompressedConfig, GPT2CompressedLMHeadModel
                from transformers import AutoConfig, AutoModelForCausalLM
                AutoConfig.register("gpt2_compressed", GPT2CompressedConfig)
                AutoModelForCausalLM.register(GPT2CompressedConfig, GPT2CompressedLMHeadModel)
                print(f"Pre-registered custom architecture from {model_name_or_path}")
            except Exception as reg_error:
                warnings.warn(f"Could not pre-register custom architecture: {reg_error}")
            finally:
                sys.path.pop(0)

    # Create model wrapper
    try:
        model = HFLM(
            pretrained=model_name_or_path,
            batch_size=batch_size,
            device=device,
            trust_remote_code=trust_remote_code,
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to load model {model_name_or_path}: {e}"
        ) from e
    
    # Run evaluation
    results = evaluator.simple_evaluate(
        model=model,
        tasks=tasks,
        batch_size=batch_size,
        limit=limit,
    )
    
    # Optionally save results
    if output_path:
        import json
        with open(output_path, 'w') as f:
            json.dump(sanitize_json(results), f, indent=2)
    
    return results


def extract_metrics(results: Dict, tasks: Optional[List[str]] = None) -> Dict[str, Dict[str, float]]:
    """
    Extract key metrics from lm-eval results.
    
    Args:
        results: Results dictionary from run_lm_eval
        tasks: Optional list of tasks to extract (default: all)
        
    Returns:
        Dictionary mapping task names to metric dictionaries
        
    Example:
        >>> results = run_lm_eval("gpt2", ["arc_easy", "wikitext"])
        >>> metrics = extract_metrics(results)
        >>> print(metrics["arc_easy"]["acc"])
    """
    if "results" not in results:
        warnings.warn("Results dictionary missing 'results' key")
        return {}
    
    task_results = results["results"]
    
    if tasks is None:
        tasks = list(task_results.keys())
    
    extracted = {}
    for task in tasks:
        if task not in task_results:
            warnings.warn(f"Task {task} not found in results")
            continue
        
        task_metrics = task_results[task]
        
        # Extract common metrics
        metrics = {}
        for key in ["acc", "acc_norm", "perplexity", "word_perplexity", 
                    "byte_perplexity", "bits_per_byte", "exact_match"]:
            if key in task_metrics:
                metrics[key] = task_metrics[key]
        
        extracted[task] = metrics
    
    return extracted