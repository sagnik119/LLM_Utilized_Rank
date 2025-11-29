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


def sanitize_json(obj):
    """Recursively convert all NumPy and torch types to JSON-safe Python types."""
    import torch
    
    # Base simple types
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    
    # NumPy scalar -> Python scalar
    if isinstance(obj, np.generic):
        return obj.item()
    
    # NumPy array -> list (recursively)
    if isinstance(obj, np.ndarray):
        return sanitize_json(obj.tolist())
    
    # Torch tensor -> list
    if isinstance(obj, torch.Tensor):
        return sanitize_json(obj.tolist())
    
    # Dict -> recursively sanitize
    if isinstance(obj, dict):
        return {sanitize_json(k): sanitize_json(v) for k, v in obj.items()}
    
    # List/tuple -> recursively sanitize
    if isinstance(obj, (list, tuple)):
        return [sanitize_json(x) for x in obj]
    
    # Fallback -> convert to string (handles dtype, etc.)
    return str(obj)

def run_lm_eval(
    model_name_or_path: str,
    tasks: List[str],
    batch_size: int = 8,
    limit: Optional[int] = None,
    device: Optional[str] = None,
    trust_remote_code: bool = True,
    output_path: Optional[str] = None,
    tokenizer = None,
    dtype: str = "float16",
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
        tokenizer: Optional pre-loaded tokenizer (for compressed models)
        dtype: Model dtype (default: "float16" for speed)
        
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

    # Create model wrapper
    try:
        # Build kwargs for HFLM
        hflm_kwargs = {
            "pretrained": model_name_or_path,
            "batch_size": batch_size,
            "device": device,
            "device_map": "auto",
            "trust_remote_code": trust_remote_code,
        }
        
        # Pass tokenizer if provided (important for compressed models)
        if tokenizer is not None:
            hflm_kwargs["tokenizer"] = tokenizer
        
        # Set dtype for faster evaluation
        if dtype == "float16":
            hflm_kwargs["dtype"] = "float16"
        elif dtype == "bfloat16":
            hflm_kwargs["dtype"] = "bfloat16"
        elif dtype == "float32":
            hflm_kwargs["dtype"] = "float32"
        
        model = HFLM(**hflm_kwargs)
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