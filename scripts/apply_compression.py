#!/usr/bin/env python3
"""
Apply compression to a model based on rank search results.

This script loads a model, activation statistics, and rank search results,
then applies the compression by transforming weights and optionally factorizing.

Usage:
    python scripts/apply_compression.py --model gpt2 --ranks ranks.json \\
        --stats stats/gpt2_xtx.pt --out ckpts/gpt2_compressed
"""

import argparse
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from urank.compression import apply_compression, compute_compression_stats


def main():
    parser = argparse.ArgumentParser(description="Apply compression to model")
    parser.add_argument("--model", type=str, required=True, help="Model name or path")
    parser.add_argument("--ranks", type=str, required=True, help="Rank search results JSON")
    parser.add_argument("--stats", type=str, required=True, help="Path to X^T X statistics")
    parser.add_argument("--out", type=str, required=True, help="Output directory")
    parser.add_argument("--factorize-threshold", type=float, default=None, 
                       help="Custom factorization threshold")
    args = parser.parse_args()
    
    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load model and tokenizer
    print(f"Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(args.model).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    # Load rank search results
    print(f"Loading ranks from: {args.ranks}")
    with open(args.ranks, "r") as f:
        ranks = json.load(f)
    print(f"Loaded ranks for {len(ranks)} layers")
    
    # Load statistics
    print(f"Loading statistics from: {args.stats}")
    xtx_map = torch.load(args.stats)
    
    # Count original parameters
    original_params = sum(p.numel() for p in model.parameters())
    print(f"Original parameters: {original_params:,}")
    
    # Apply compression
    print("\nApplying compression...")
    apply_compression(
        model,
        xtx_map,
        ranks,
        factorize_threshold=args.factorize_threshold
    )
    
    # Count compressed parameters
    compressed_params = sum(p.numel() for p in model.parameters())
    reduction = 100 * (1 - compressed_params / original_params)
    
    print(f"\nCompression summary:")
    print(f"  Original parameters:   {original_params:,}")
    print(f"  Compressed parameters: {compressed_params:,}")
    print(f"  Reduction:             {reduction:.2f}%")
    
    # Save compressed model
    print(f"\nSaving compressed model to: {args.out}")
    model.save_pretrained(args.out, safe_serialization=True)
    tokenizer.save_pretrained(args.out)
    
    # Also save the ranks config for reference
    with open(f"{args.out}/ranks.json", "w") as f:
        json.dump(ranks, f, indent=2)
    
    print("Done!")


if __name__ == "__main__":
    main()