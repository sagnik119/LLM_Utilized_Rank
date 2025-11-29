#!/usr/bin/env python3
"""
Search ranks based on naive weight SVD (baseline method).

This script computes ranks for each layer by performing SVD on the weight
matrix W itself (no activation data required) and finding the minimum rank
that preserves a target percentage of spectral energy.

This is a baseline comparison to the data-driven utilized rank method.

Usage:
    python scripts/search_ranks_weight_svd.py --model gpt2 \
        --energy 0.9995 --out ranks/gpt2_weight_svd.json
"""

import argparse
import json
import torch
import torch.nn as nn
from pathlib import Path
from transformers import AutoModelForCausalLM


# GPT-2 Conv1D support
try:
    from transformers.models.gpt2.modeling_gpt2 import Conv1D
    HAS_CONV1D = True
except Exception:
    Conv1D = None
    HAS_CONV1D = False


def compute_weight_svd_ranks(
    model: nn.Module,
    target_energy: float = 0.9995,
    layer_patterns: list = None
):
    """
    Compute ranks from SVD of weight matrices (no activation data).
    
    Args:
        model: PyTorch model
        target_energy: Target spectral energy fraction (default: 0.9995)
        layer_patterns: Optional list of layer name patterns to include
    
    Returns:
        dict: {layer_name: {"r": rank, "d_out": m, "d_in": d, "energy": float}}
    """
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from src.urank.instrumentation import iter_candidate_linear_modules
    
    ranks = {}
    
    print(f"Computing weight SVD ranks (target energy: {target_energy*100:.4f}%)...")
    print("-" * 80)
    
    with torch.no_grad():
        for name, module in iter_candidate_linear_modules(model):
            # Check Conv1D type
            is_conv1d = HAS_CONV1D and isinstance(module, Conv1D)
            
            # Optional manual filtering if layer_patterns provided
            if layer_patterns is not None and not any(pattern in name for pattern in layer_patterns):
                continue
            
            # Extract weight matrix as (d_out, d_in)
            if is_conv1d:
                W = module.weight.data.T.float()  # Conv1D stores transposed
            else:
                W = module.weight.data.float()
            
            m, d = W.shape  # d_out, d_in
            
            # Perform full SVD on weight matrix
            # For GPT-2 small sizes this is fast
            U, S, Vh = torch.linalg.svd(W, full_matrices=False)
            
            # Compute spectral energy from singular values
            S2 = S ** 2
            total = S2.sum()
            
            if total < 1e-12:
                # Degenerate case: weight matrix is ~0
                r = 1
                energy_kept = 0.0
            else:
                cum = torch.cumsum(S2, dim=0) / total
                
                # Find smallest r where cumulative energy >= target
                idx = (cum >= target_energy).nonzero(as_tuple=True)[0]
                if len(idx) == 0:
                    # Target not achievable (shouldn't happen with 0.9995)
                    r = len(S)
                else:
                    r = int(idx[0].item()) + 1
                
                energy_kept = float(cum[r-1].item())
            
            # Store results
            ranks[name] = {
                "r": int(r),
                "d_out": int(m),
                "d_in": int(d),
                "energy": energy_kept,
                "compression_ratio": float(r / min(m, d))
            }
            
            # Print summary
            print(f"{name}")
            print(f"  Shape: ({m}, {d}), r: {r} ({r/min(m,d)*100:.1f}%), energy: {energy_kept:.6f}")
    
    print("-" * 80)
    
    return ranks


def main():
    parser = argparse.ArgumentParser(
        description="Compute ranks from naive weight SVD (baseline method)"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name or path"
    )
    parser.add_argument(
        "--energy",
        type=float,
        default=0.9995,
        help="Target energy fraction (default: 0.9995 = 99.95%%)"
    )
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output JSON file path"
    )
    parser.add_argument(
        "--layer-patterns",
        type=str,
        nargs="+",
        default=None,
        help="Layer name patterns to include (default: attn.c_attn, attn.c_proj, mlp.c_fc, mlp.c_proj)"
    )
    args = parser.parse_args()
    
    # Validate arguments
    if not (0.0 < args.energy <= 1.0):
        raise ValueError(f"Energy must be in (0, 1], got {args.energy}")
    
    # Load model
    print(f"Loading model: {args.model}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32).to(device)
    print(f"  Model loaded on {device}\n")
    
    # Compute ranks
    results = compute_weight_svd_ranks(
        model=model,
        target_energy=args.energy,
        layer_patterns=args.layer_patterns
    )
    
    # Print summary statistics
    if results:
        ranks = [res["r"] for res in results.values()]
        d_outs = [res["d_out"] for res in results.values()]
        d_ins = [res["d_in"] for res in results.values()]
        compression_ratios = [res["compression_ratio"] for res in results.values()]
        
        print(f"\nSummary Statistics:")
        print(f"  Total layers: {len(results)}")
        print(f"  Rank range: [{min(ranks)}, {max(ranks)}]")
        print(f"  Average rank: {sum(ranks)/len(ranks):.1f}")
        print(f"  Average compression ratio: {sum(compression_ratios)/len(compression_ratios):.3f}")
        print(f"  Output dimension range: [{min(d_outs)}, {max(d_outs)}]")
        print(f"  Input dimension range: [{min(d_ins)}, {max(d_ins)}]")
    else:
        print("\n⚠ Warning: No layers matched the inclusion criteria")
    
    # Save results
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Saved ranks for {len(results)} layers to {output_path}")
    print(f"\nNext steps:")
    print(f"  1. Apply compression:")
    print(f"     python scripts/apply_compression_weight_svd.py --model {args.model} \\")
    print(f"       --ranks {output_path} --out ckpts/{Path(args.model).name}_weight_svd")
    print(f"  2. Fine-tune:")
    print(f"     python scripts/finetune_compressed.py --preset full \\")
    print(f"       --model ckpts/{Path(args.model).name}_weight_svd --data redpajama \\")
    print(f"       --out outputs/{Path(args.model).name}_weight_svd_fullft --max-train-tokens 150000000")
    print(f"  3. Evaluate:")
    print(f"     python scripts/eval_lm_eval.py \\")
    print(f"       --model outputs/{Path(args.model).name}_weight_svd_fullft \\")
    print(f"       --tasks wikitext arc_easy hellaswag --batch 8 --out results_weight_svd.json")


if __name__ == "__main__":
    main()