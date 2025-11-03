#!/usr/bin/env python3
"""
Evaluate model perplexity on a test dataset.

This script loads a model and evaluates its perplexity on a specified
dataset split, useful for comparing original vs compressed models.

Usage:
    python scripts/eval_perplexity.py --model ckpts/gpt2_compressed \\
        --dataset wikitext --split test --samples 5000
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from urank.eval.perplexity import PerplexityEvaluator


def main():
    parser = argparse.ArgumentParser(description="Evaluate model perplexity")
    parser.add_argument("--model", type=str, required=True, help="Model name or path")
    parser.add_argument("--dataset", type=str, default="wikitext", help="Dataset name")
    parser.add_argument("--config", type=str, default="wikitext-2-raw-v1", help="Dataset config")
    parser.add_argument("--split", type=str, default="test", help="Dataset split")
    parser.add_argument("--samples", type=int, default=5000, help="Number of samples")
    parser.add_argument("--seq-len", type=int, default=512, help="Maximum sequence length")
    args = parser.parse_args()
    
    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load model and tokenizer
    print(f"Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(args.model).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # Setup evaluator
    print(f"\nEvaluating on {args.dataset} ({args.split})...")
    evaluator = PerplexityEvaluator(
        model,
        tokenizer,
        dataset_name=args.dataset,
        config=args.config,
        split=args.split,
        samples=args.samples,
        seq_len=args.seq_len,
    )
    
    # Evaluate
    ppl = evaluator()
    
    # Print results
    print(f"\nResults:")
    print(f"  Perplexity: {ppl:.2f}")
    print(f"  Parameters: {total_params:,}")
    
    # Also output as JSON for easy parsing
    import json
    results = {
        "perplexity": float(ppl),
        "parameters": total_params,
        "model": args.model,
        "dataset": f"{args.dataset}/{args.config}",
        "split": args.split,
        "samples": args.samples,
    }
    print(f"\nJSON: {json.dumps(results)}")


if __name__ == "__main__":
    main()