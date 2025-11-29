#!/usr/bin/env python3
"""
Evaluate a model using lm-evaluation-harness.
Fully compatible with compressed GPT-2 and LLaMA models.

This script provides a CLI for running comprehensive evaluations using
the EleutherAI lm-eval-harness, including perplexity and zero/few-shot tasks.

Usage:
    python scripts/eval_lm_eval.py --model ckpts/llama2_compressed \
        --tasks wikitext arc_easy hellaswag winogrande piqa \
        --batch 4 --trust-remote-code --out eval_results.json
"""

import argparse
import json
from transformers import AutoTokenizer, AutoConfig
from urank.eval.harness import run_lm_eval, extract_metrics


def must_trust_remote_code(model_path):
    """
    Detect whether the model requires trust_remote_code=True.
    This is true for any compressed checkpoint.
    
    Args:
        model_path: Path to model checkpoint
        
    Returns:
        bool: True if model requires trust_remote_code
    """
    try:
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        return getattr(config, "is_compressed", False)
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser(description="Evaluate model with lm-eval-harness")
    parser.add_argument("--model", required=True, help="HF model path or local checkpoint")
    parser.add_argument(
        "--tasks", 
        nargs="+", 
        required=True,
        help="Tasks to evaluate (e.g., wikitext arc_easy hellaswag winogrande piqa)"
    )
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--limit", type=int, default=None, help="Limit examples per task")
    parser.add_argument("--device", default=None, help="Device (cuda/cpu)")
    parser.add_argument("--out", default=None, help="Output JSON file")
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        default=False,
        help="Force trust_remote_code=True (recommended for compressed models)"
    )
    args = parser.parse_args()
    
    model_path = args.model
    
    # Detect compressed model and automatically enable trust_remote_code
    auto_trust = must_trust_remote_code(model_path)
    trust_rc = args.trust_remote_code or auto_trust
    
    if trust_rc:
        print("✓ Using trust_remote_code=True (compressed model detected)")
    else:
        print("ℹ Using trust_remote_code=False")
    
    print(f"Evaluating model: {model_path}")
    print(f"Tasks: {', '.join(args.tasks)}")
    print(f"Batch size: {args.batch}")
    
    # Load tokenizer explicitly (important for custom architectures)
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=trust_rc
    )
    
    # Run evaluation
    results = run_lm_eval(
        model_name_or_path=model_path,
        tokenizer=tokenizer,
        tasks=args.tasks,
        batch_size=args.batch,
        limit=args.limit,
        device=args.device,
        trust_remote_code=trust_rc,
        output_path=args.out,
    )
    
    # Extract and display key metrics
    metrics = extract_metrics(results, args.tasks)
    
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    for task, task_metrics in metrics.items():
        print(f"\n{task}:")
        for metric_name, metric_value in task_metrics.items():
            if isinstance(metric_value, float):
                print(f"  {metric_name}: {metric_value:.4f}")
            else:
                print(f"  {metric_name}: {metric_value}")
    
    # Save if not already saved
    if not args.out:
        print("\n" + "="*60)
        print("Full results:")
        print(json.dumps(results, indent=2))
    else:
        print(f"\nFull results saved to: {args.out}")


if __name__ == "__main__":
    main()