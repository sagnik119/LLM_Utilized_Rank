#!/usr/bin/env python3
"""
Search ranks based on spectral energy preservation of output activations.

This script computes the utilized rank for each layer by finding the minimum
rank that preserves a target percentage (default 99%) of the spectral energy
in the output activation space (Y^T Y).

Unlike search_ranks.py (which uses validation perplexity), this approach is:
- Fast (no forward passes needed)
- Deterministic (no validation set dependency)
- Theoretically grounded (preserves spectral properties)

Usage:
    python scripts/search_ranks_energy.py --stats stats/gpt2_yty.pt \
        --energy 0.99 --out ranks/gpt2_energy.json
"""

import argparse
import json
import torch
from pathlib import Path


def compute_rank_from_energy(C, energy=0.99):
    """
    Compute the minimum rank that preserves a target energy fraction.
    
    Args:
        C: Covariance matrix (d_out, d_out), assumed symmetric PSD
        energy: Target energy fraction (default 0.99 = 99%)
    
    Returns:
        r: Minimum rank preserving target energy
        eigvals: All eigenvalues (descending order)
        cum: Cumulative energy fractions
    """
    # Compute eigenvalues (eigvalsh returns ascending order)
    eigvals = torch.linalg.eigvalsh(C)
    
    # Sort descending
    eigvals = torch.sort(eigvals, descending=True).values
    
    # Clamp negative eigenvalues (numerical precision issues)
    eigvals = torch.clamp(eigvals, min=0.0)
    
    # Compute cumulative energy
    total = eigvals.sum()
    if total < 1e-12:
        # Degenerate case: all eigenvalues ~0
        return 1, eigvals, torch.zeros_like(eigvals)
    
    cum = torch.cumsum(eigvals, dim=0) / total
    
    # Find minimum rank where cumulative energy >= target
    idx = (cum >= energy).nonzero(as_tuple=True)[0]
    if len(idx) == 0:
        # Energy target not achievable (shouldn't happen with 0.99)
        r = len(eigvals)
    else:
        r = int(idx[0].item()) + 1
    
    return r, eigvals, cum


def main():
    parser = argparse.ArgumentParser(
        description="Compute utilized ranks from output activation energy"
    )
    parser.add_argument(
        "--stats", 
        type=str, 
        required=True,
        help="Path to Y^T Y statistics file (.pt)"
    )
    parser.add_argument(
        "--energy", 
        type=float, 
        default=0.99,
        help="Target energy fraction (default: 0.99 = 99%%)"
    )
    parser.add_argument(
        "--out", 
        type=str, 
        required=True,
        help="Output JSON file path"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed eigenvalue information"
    )
    args = parser.parse_args()
    
    # Validate arguments
    if not (0.0 < args.energy <= 1.0):
        raise ValueError(f"Energy must be in (0, 1], got {args.energy}")
    
    stats_path = Path(args.stats)
    if not stats_path.exists():
        raise FileNotFoundError(f"Stats file not found: {stats_path}")
    
    # Load Y^T Y statistics
    print(f"Loading output covariance matrices from: {stats_path}")
    yty_map = torch.load(args.stats, map_location="cpu")
    print(f"  Loaded statistics for {len(yty_map)} layers\n")
    
    # Compute ranks for each layer
    results = {}
    total_params_original = 0
    total_params_compressed = 0
    
    print(f"Computing utilized ranks (target energy: {args.energy*100:.1f}%)...")
    print("-" * 80)
    
    for name, C in sorted(yty_map.items()):
        # Convert to float32 for eigendecomposition (eigvalsh doesn't support bf16)
        C = C.float()
        
        r, eigvals, cum = compute_rank_from_energy(C, args.energy)
        
        d_out = C.shape[0]
        achieved_energy = float(cum[r-1].item()) if r > 0 else 0.0
        
        # Store results
        results[name] = {
            "r": int(r),
            "d_out": int(d_out),
            "energy": achieved_energy,
            "compression_ratio": float(r / d_out) if d_out > 0 else 1.0
        }
        
        # Print summary
        print(f"{name}")
        print(f"  d_out: {d_out}, r: {r} ({r/d_out*100:.1f}%), energy: {achieved_energy:.4f}")
        
        if args.verbose:
            # Print top eigenvalues
            top_k = min(10, len(eigvals))
            print(f"  Top {top_k} eigenvalues: {eigvals[:top_k].tolist()}")
            print(f"  Cumulative energy at r={r}: {cum[r-1]:.4f}")
    
    print("-" * 80)
    
    # Print summary statistics
    ranks = [res["r"] for res in results.values()]
    d_outs = [res["d_out"] for res in results.values()]
    compression_ratios = [res["compression_ratio"] for res in results.values()]
    
    print(f"\nSummary Statistics:")
    print(f"  Total layers: {len(results)}")
    print(f"  Rank range: [{min(ranks)}, {max(ranks)}]")
    print(f"  Average rank: {sum(ranks)/len(ranks):.1f}")
    print(f"  Average compression ratio: {sum(compression_ratios)/len(compression_ratios):.3f}")
    print(f"  Output dimension range: [{min(d_outs)}, {max(d_outs)}]")
    
    # Save results
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Saved ranks for {len(results)} layers to {output_path}")
    print(f"\nNext steps:")
    print(f"  1. Apply compression:")
    print(f"     python scripts/apply_compression_energy.py --model gpt2 \\")
    print(f"       --ranks {output_path} --stats {args.stats} --out ckpts/gpt2_energy")
    print(f"  2. Evaluate perplexity:")
    print(f"     python scripts/eval_perplexity.py --model ckpts/gpt2_energy \\")
    print(f"       --dataset wikitext --split test")


if __name__ == "__main__":
    main()