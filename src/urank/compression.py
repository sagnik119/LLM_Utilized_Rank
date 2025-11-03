"""
Model compression utilities for applying utilized rank transformations.

This module provides utilities for:
- Applying compression to all layers based on rank search results
- Deciding whether to factorize or keep transformed weights
- Replacing layers with low-rank factorizations
"""

from typing import Dict, Optional
import torch
import torch.nn as nn

from .transform import transform_weight_ps_pt, factorize_rank, should_factorize
from .subspace import topk_from_xtx, projector_from_vecs
from .instrumentation import get_parent_and_attr


def apply_compression(
    model: nn.Module,
    xtx_map: Dict[str, torch.Tensor],
    ranks: Dict[str, Dict],
    factorize_threshold: Optional[float] = None
):
    """
    Apply compression to model based on rank search results.
    
    For each layer in ranks:
    1. Compute projectors P_S, P_T from energy thresholds
    2. Transform weight: W' = P_T @ W @ P_S
    3. If factorization saves parameters, replace with two Linear layers
    4. Otherwise, update weight in-place with W'
    
    Args:
        model: PyTorch model to compress
        xtx_map: Dictionary mapping layer names to X^T X matrices
        ranks: Dictionary mapping layer names to rank info dicts with keys:
               - k_S, k_T, r: Subspace dimensions and utilized rank
               - e_S, e_T: Energy thresholds (if available)
        factorize_threshold: Optional ratio threshold for factorization decision
                            If None, uses r * (m + d) < m * d criterion
    
    Example:
        >>> with open("ranks.json") as f:
        ...     ranks = json.load(f)
        >>> xtx_map = torch.load("stats.pt")
        >>> apply_compression(model, xtx_map, ranks)
        >>> model.save_pretrained("compressed_model")
    """
    # Build name -> module mapping for Linear layers
    name_to_module = {
        n: m for n, m in model.named_modules() 
        if isinstance(m, nn.Linear)
    }
    
    for name, cfg in ranks.items():
        if name not in name_to_module:
            print(f"Warning: Layer {name} not found in model")
            continue
        
        if name not in xtx_map:
            print(f"Warning: No statistics for layer {name}")
            continue
        
        lin: nn.Linear = name_to_module[name]
        xtx = xtx_map[name]
        
        # Compute input subspace projector
        if cfg.get("e_S") is not None:
            V_S, _ = topk_from_xtx(xtx, cfg["e_S"])
        else:
            # Use k_S directly if energy not stored
            evals, evecs = torch.linalg.eigh(xtx)
            idx = torch.argsort(evals, descending=True)
            V_S = evecs[:, idx][:, :cfg["k_S"]]
        
        P_S = projector_from_vecs(V_S)
        
        # Compute output subspace projector
        W = lin.weight.data
        ttx = W @ xtx.to(W.device) @ W.T
        
        if cfg.get("e_T") is not None:
            V_T, _ = topk_from_xtx(ttx.cpu(), cfg["e_T"])
        else:
            # Use k_T directly
            evals, evecs = torch.linalg.eigh(ttx.cpu())
            idx = torch.argsort(evals, descending=True)
            V_T = evecs[:, idx][:, :cfg["k_T"]]
        
        P_T = projector_from_vecs(V_T)
        
        # Transform weight
        W_prime = transform_weight_ps_pt(lin, P_S, P_T)
        
        # Decide whether to factorize
        m, d = W_prime.shape
        r = cfg["r"]
        
        # Check if factorization is beneficial
        do_factorize = False
        if factorize_threshold is not None:
            # Use custom threshold
            do_factorize = (r * (m + d)) / (m * d) < factorize_threshold
        else:
            # Use standard criterion
            do_factorize = should_factorize(m, d, r)
        
        if do_factorize and r < min(m, d):
            # Factorize and replace layer
            left, right = factorize_rank(W_prime, r)
            
            # Create new Linear layers (no bias for simplicity)
            lin_left = nn.Linear(r, m, bias=False)
            lin_right = nn.Linear(d, r, bias=False)
            lin_left.weight.data.copy_(left)
            lin_right.weight.data.copy_(right)
            
            # Handle bias if present
            if lin.bias is not None:
                lin_left.bias = lin.bias
            
            # Replace in parent module
            parent, attr = get_parent_and_attr(model, name)
            setattr(parent, attr, nn.Sequential(lin_right, lin_left))
            
            param_ratio = (r * (m + d)) / (m * d)
            print(f"Factorized {name}: ({m}, {d}) -> ({m}, {r}) @ ({r}, {d}) "
                  f"[{param_ratio:.2%} params]")
        else:
            # Just update weight with transformed version
            lin.weight.data.copy_(W_prime)
            print(f"Transformed {name}: ({m}, {d}) with r={r} (kept as single layer)")


def compute_compression_stats(
    original_model: nn.Module,
    compressed_model: nn.Module
) -> Dict[str, float]:
    """
    Compute compression statistics comparing original and compressed models.
    
    Args:
        original_model: Original uncompressed model
        compressed_model: Compressed model
        
    Returns:
        Dictionary with compression statistics:
        - original_params: Total parameters in original
        - compressed_params: Total parameters in compressed
        - compression_ratio: Ratio of compressed to original
        - params_saved: Number of parameters saved
    """
    orig_params = sum(p.numel() for p in original_model.parameters())
    comp_params = sum(p.numel() for p in compressed_model.parameters())
    
    return {
        "original_params": orig_params,
        "compressed_params": comp_params,
        "compression_ratio": comp_params / orig_params if orig_params > 0 else 0,
        "params_saved": orig_params - comp_params,
        "reduction_percent": 100 * (1 - comp_params / orig_params) if orig_params > 0 else 0,
    }