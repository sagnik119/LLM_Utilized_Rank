"""
Output-activation–based model compression utilities.

Given:
  • Y^T Y for each layer
  • rank r for each layer (from search_ranks_energy.py)

We reconstruct:
    W' = U_r (U_r^T W)
where U_r are the top r eigenvectors of Y^T Y.

Optional: replace layer with factorized Sequential(B → A).

Supports nn.Linear and GPT-2 Conv1D.
"""

from typing import Dict, Optional
import torch
import torch.nn as nn
from .instrumentation import get_parent_and_attr

# GPT-2 Conv1D
try:
    from transformers.models.gpt2.modeling_gpt2 import Conv1D
    HAS_CONV1D = True
except Exception:
    Conv1D = None
    HAS_CONV1D = False


def should_factorize(m, d, r, threshold=None):
    """
    Decide whether factorization is parameter-efficient.

    Default criterion:
        r (m + d) < m d
    """
    if threshold is not None:
        return (r * (m + d)) / (m * d) < threshold

    return r * (m + d) < m * d


def apply_compression(
    model: nn.Module,
    yty_map: Dict[str, torch.Tensor],
    ranks: Dict[str, Dict],
    factorize_threshold: Optional[float] = None,
):
    """
    Apply Y^T Y-based spectral compression to each layer.

    Args:
        model: GPT-family model (Linear/Conv1D weights)
        yty_map: dict: layer_name → Y^T Y matrix (d_out × d_out)
        ranks: dict: layer_name → {"r": int, "d_out": int, "energy": float}
        factorize_threshold: optional compression ratio threshold
    """

    # Map module names to Linear/Conv1D modules
    name_to_module = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) or (HAS_CONV1D and isinstance(module, Conv1D)):
            name_to_module[name] = module

    # Process each ranked layer
    for name, cfg in ranks.items():

        if name not in name_to_module:
            print(f"[WARN] Layer not found in model: {name}")
            continue
        if name not in yty_map:
            print(f"[WARN] No Y^T Y stats for layer: {name}")
            continue

        module = name_to_module[name]
        r = cfg["r"]

        # Extract weight matrix W as (d_out, d_in)
        if HAS_CONV1D and isinstance(module, Conv1D):
            W = module.weight.data.T.float()
        else:
            W = module.weight.data.float()
        
        # Load C and move to same device as W
        C = yty_map[name].float().to(W.device)  # Y^T Y

        d_out, d_in = W.shape
        assert C.shape[0] == d_out, f"Y^T Y wrong size for {name}"

        # Eigen-decomposition of Y^T Y
        eigvals, eigvecs = torch.linalg.eigh(C)
        idx = torch.argsort(eigvals, descending=True)
        U = eigvecs[:, idx][:, :r].to(W.device)  # (d_out × r)

        # Compute A, B but DO NOT apply projection unless factorizing
        A = U                           # (d_out × r)
        B = (U.T @ W)                   # (r × d_in)

        # Check parameter efficiency
        do_factorize = should_factorize(d_out, d_in, r, threshold=factorize_threshold)

        if do_factorize:
            # Use factorized form
            from .modeling_gpt2_compressed import FactorizedLinear
            
            bias = module.bias.data if module.bias is not None else None
            factorized = FactorizedLinear(A, B, bias)

            parent, attr = get_parent_and_attr(model, name)
            setattr(parent, attr, factorized)

            print(f"[FACTORIZED] {name}: ({d_out},{d_in}) → r={r}")
            print(f"[SAVE] Factorized {name} → A={A.shape}, B={B.shape}")
        else:
            # DO NOT MODIFY WEIGHT AT ALL
            # Keep original W exactly as is
            print(f"[SKIPPED] {name}: r={r} not parameter-efficient; layer left unchanged")


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