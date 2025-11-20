#!/usr/bin/env python3
"""
Apply compression to a model using output-activation covariance (Y^T Y).

This script:
  • Loads a model
  • Loads Y^T Y statistics
  • Loads rank assignments (from search_ranks_energy.py)
  • Reconstructs each layer from its top-r output eigenvectors
  • Optionally factorizes compressed layers
  • Saves the compressed checkpoint

Usage:
    python scripts/apply_compression.py \
        --model gpt2 \
        --ranks ranks/gpt2_energy.json \
        --stats stats/gpt2_yty.pt \
        --out ckpts/gpt2_compressed
"""

import argparse
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from urank.compression import apply_compression


def main():
    parser = argparse.ArgumentParser(description="Apply Y^T Y-based compression")
    parser.add_argument("--model", type=str, required=True, help="Model name or path")
    parser.add_argument("--ranks", type=str, required=True,
                        help="JSON file produced by search_ranks_energy.py")
    parser.add_argument("--stats", type=str, required=True,
                        help="Path to Y^T Y statistics (.pt)")
    parser.add_argument("--out", type=str, required=True,
                        help="Directory to save compressed model")
    parser.add_argument("--factorize-threshold", type=float, default=None,
                        help="Optional parameter budget threshold for factorization")
    args = parser.parse_args()

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load model + tokenizer
    print(f"Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Load ranks
    print(f"Loading ranks from: {args.ranks}")
    with open(args.ranks, "r") as f:
        ranks = json.load(f)
    print(f"Loaded ranks for {len(ranks)} layers")

    # Load Y^T Y statistics
    print(f"Loading Y^T Y statistics from: {args.stats}")
    yty_map = torch.load(args.stats, map_location="cpu")

    # Original param count
    orig_params = sum(p.numel() for p in model.parameters())
    print(f"Original parameters: {orig_params:,}")

    # Apply compression
    print("\nApplying compression...")
    apply_compression(
        model=model,
        yty_map=yty_map,
        ranks=ranks,
        factorize_threshold=args.factorize_threshold,
    )

    # Compressed param count
    comp_params = sum(p.numel() for p in model.parameters())
    reduction = 100 * (1 - comp_params / orig_params)

    print("\nCompression summary:")
    print(f"Original parameters:   {orig_params:,}")
    print(f"Compressed parameters: {comp_params:,}")
    print(f"Reduction:             {reduction:.2f}%")

    # Save
    print(f"\nSaving compressed model to: {args.out}")
    model.save_pretrained(args.out, safe_serialization=True)
    tokenizer.save_pretrained(args.out)

    with open(f"{args.out}/ranks.json", "w") as f:
        json.dump(ranks, f, indent=2)

    print("Done!")


if __name__ == "__main__":
    main()