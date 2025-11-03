#!/usr/bin/env python3
"""
Search for optimal ranks using binary search with validation constraint.

This script loads a model and activation statistics, then searches for the
lowest energy thresholds (e_S, e_T) that maintain validation metric within
epsilon percent of baseline.

Usage:
    python scripts/search_ranks.py --model gpt2 --stats stats/gpt2_xtx.pt \\
        --epsilon 0.1 --val-dataset wikitext --val-samples 2000 --out ranks.json
"""

import argparse
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from urank.rank_search import RankSearcher
from urank.eval.perplexity import PerplexityEvaluator


def main():
    parser = argparse.ArgumentParser(description="Search optimal ranks")
    parser.add_argument("--model", type=str, required=True, help="Model name or path")
    parser.add_argument("--stats", type=str, required=True, help="Path to X^T X statistics")
    parser.add_argument("--epsilon", type=float, default=0.1, help="Max validation drop (%)")
    parser.add_argument("--val-dataset", type=str, default="wikitext", help="Validation dataset")
    parser.add_argument("--val-config", type=str, default="wikitext-2-raw-v1", help="Dataset config")
    parser.add_argument("--val-samples", type=int, default=2000, help="Validation samples")
    parser.add_argument("--out", type=str, required=True, help="Output JSON file")
    parser.add_argument("--energy-min", type=float, default=0.8, help="Min energy threshold")
    parser.add_argument("--energy-max", type=float, default=0.9999, help="Max energy threshold")
    parser.add_argument("--filter", type=str, default=None, help="Regex to filter layer names")
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
    
    # Load statistics
    print(f"Loading statistics from: {args.stats}")
    xtx_map = torch.load(args.stats)
    print(f"Loaded statistics for {len(xtx_map)} layers")
    
    # Setup evaluator
    print(f"Setting up evaluator on {args.val_dataset}")
    eval_fn = PerplexityEvaluator(
        model, 
        tokenizer,
        dataset_name=args.val_dataset,
        config=args.val_config,
        split="validation",
        samples=args.val_samples
    )
    
    # Baseline perplexity
    print("Computing baseline perplexity...")
    baseline_ppl = eval_fn(model)
    print(f"Baseline perplexity: {baseline_ppl:.2f}")
    
    # Setup searcher
    searcher = RankSearcher(
        model,
        xtx_map,
        eval_fn,
        epsilon_percent=args.epsilon,
        energy_min=args.energy_min,
        energy_max=args.energy_max
    )
    
    # Optionally filter layers
    layer_filter = None
    if args.filter:
        import re
        pattern = re.compile(args.filter)
        layer_filter = lambda name: pattern.search(name) is not None
    
    # Search all layers
    print(f"\nSearching ranks with epsilon={args.epsilon}%...")
    results = searcher.search_all_layers(layer_filter=layer_filter)
    
    # Convert to JSON-serializable format
    results_json = {}
    for name, result in results.items():
        results_json[name] = {
            "k_S": result.k_S,
            "k_T": result.k_T,
            "r": result.r,
            "e_S": result.e_S,
            "e_T": result.e_T,
        }
    
    # Save results
    print(f"\nSaving results for {len(results_json)} layers to {args.out}")
    with open(args.out, "w") as f:
        json.dump(results_json, f, indent=2)
    
    # Print summary
    total_r = sum(r["r"] for r in results_json.values())
    avg_r = total_r / len(results_json) if results_json else 0
    print(f"\nSummary:")
    print(f"  Total layers: {len(results_json)}")
    print(f"  Average utilized rank: {avg_r:.1f}")
    print("Done!")


if __name__ == "__main__":
    main()