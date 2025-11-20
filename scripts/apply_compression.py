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
import shutil
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

from urank.compression import apply_compression


def main():
    parser = argparse.ArgumentParser(description="Apply compression (utilized rank or weight SVD)")
    parser.add_argument("--model", type=str, required=True, help="Model name or path")
    parser.add_argument("--ranks", type=str, required=True,
                        help="JSON file with rank assignments")
    parser.add_argument("--stats", type=str, default=None,
                        help="Path to Y^T Y statistics (.pt) - required for utilized rank mode")
    parser.add_argument("--out", type=str, required=True,
                        help="Directory to save compressed model")
    parser.add_argument("--factorize-threshold", type=float, default=None,
                        help="Optional parameter budget threshold for factorization")
    parser.add_argument("--mode", type=str, default="utilized",
                        choices=["utilized", "weight_svd"],
                        help="Compression mode: 'utilized' uses Y^T Y stats, 'weight_svd' uses naive SVD on weights")
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

    # Original param count
    orig_params = sum(p.numel() for p in model.parameters())
    print(f"Original parameters: {orig_params:,}")

    # Apply compression based on mode
    if args.mode == "utilized":
        # Utilized rank mode: requires Y^T Y statistics
        if args.stats is None:
            raise ValueError("--stats is required for utilized rank mode")
        
        print(f"Loading Y^T Y statistics from: {args.stats}")
        yty_map = torch.load(args.stats, map_location="cpu")
        
        print("\nApplying utilized rank compression...")
        apply_compression(
            model=model,
            yty_map=yty_map,
            ranks=ranks,
            factorize_threshold=args.factorize_threshold,
        )
    
    elif args.mode == "weight_svd":
        # Weight SVD mode: no activation data needed
        print("\nApplying weight-SVD compression...")
        from urank.compression import apply_weight_svd_compression
        apply_weight_svd_compression(
            model=model,
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

    # Save model and tokenizer
    print(f"\nSaving compressed model to: {args.out}")
    out_path = Path(args.out)
    out_path.mkdir(parents=True, exist_ok=True)
    
    model.save_pretrained(args.out, safe_serialization=True)
    tokenizer.save_pretrained(args.out)

    with open(out_path / "ranks.json", "w") as f:
        json.dump(ranks, f, indent=2)

    # Copy custom architecture files for trust_remote_code
    print("\nCopying custom architecture files...")
    src_root = Path(__file__).parent.parent / "src" / "urank"
    
    files_to_copy = [
        "modeling_gpt2_compressed.py",
        "configuration_gpt2_compressed.py"
    ]
    
    for fname in files_to_copy:
        src_file = src_root / fname
        if src_file.exists():
            shutil.copy2(src_file, out_path / fname)
            print(f"  ✓ Copied {fname}")
        else:
            print(f"  ⚠ Warning: {fname} not found at {src_file}")
    
    # Write __init__.py
    (out_path / "__init__.py").write_text(
        "from .modeling_gpt2_compressed import GPT2CompressedLMHeadModel\n"
        "from .configuration_gpt2_compressed import GPT2CompressedConfig\n"
        "\n"
        "__all__ = ['GPT2CompressedLMHeadModel', 'GPT2CompressedConfig']\n"
    )
    print("  ✓ Created __init__.py")
    
    # Update config.json with custom architecture
    config_path = out_path / "config.json"
    with open(config_path) as f:
        config = json.load(f)
    
    config.update({
        "model_type": "gpt2_compressed",
        "architectures": ["GPT2CompressedLMHeadModel"],
        "auto_map": {
            "AutoConfig": "configuration_gpt2_compressed.GPT2CompressedConfig",
            "AutoModelForCausalLM": "modeling_gpt2_compressed.GPT2CompressedLMHeadModel",
        },
    })
    
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print("  ✓ Updated config.json")

    print("\n✓ Compression complete!")
    print(f"\nTo load the compressed model:")
    print(f"  from transformers import AutoModelForCausalLM")
    print(f"  model = AutoModelForCausalLM.from_pretrained('{args.out}', trust_remote_code=True)")


if __name__ == "__main__":
    main()