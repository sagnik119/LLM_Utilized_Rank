"""
Binary search for optimal ranks with validation constraints.

This module provides utilities for:
- Searching for optimal subspace dimensions via binary search
- Validating rank choices against perplexity/accuracy constraints
- Computing utilized ranks from subspace dimensions
- Supports both nn.Linear and GPT-2 Conv1D layers
"""

from dataclasses import dataclass
from typing import Dict, Callable, Optional, Union
import torch
import torch.nn as nn

from .subspace import topk_from_xtx, projector_from_vecs
from .transform import transform_weight_ps_pt
from .utils import eval_no_grad

# Import Conv1D for GPT-2 compatibility
try:
    from transformers.models.gpt2.modeling_gpt2 import Conv1D
    HAS_CONV1D = True
except ImportError:
    Conv1D = None
    HAS_CONV1D = False


@dataclass
class RankResult:
    """
    Result of rank search for a single layer.
    
    Attributes:
        k_S: Input subspace dimension
        k_T: Output subspace dimension
        r: Utilized rank (min of k_S and k_T)
        e_S: Energy threshold used for input subspace
        e_T: Energy threshold used for output subspace
    """
    k_S: int
    k_T: int
    r: int
    e_S: Optional[float] = None
    e_T: Optional[float] = None


class RankSearcher:
    """
    Binary search for optimal ranks per layer.
    
    Searches for the lowest energy thresholds (e_S, e_T) such that
    transforming a layer's weights with the corresponding projectors
    maintains validation metric within epsilon percent of baseline.
    
    Args:
        model: PyTorch model to search
        xtx_map: Dictionary mapping layer names to X^T X matrices
        eval_fn: Evaluation function that takes model and returns metric
                 (lower is better, e.g., perplexity)
        epsilon_percent: Maximum allowed relative metric increase (default: 0.1%)
        energy_min: Minimum energy threshold to try (default: 0.8)
        energy_max: Maximum energy threshold to try (default: 0.9999)
        
    Example:
        >>> eval_fn = PerplexityEvaluator(model, tokenizer)
        >>> searcher = RankSearcher(model, xtx_map, eval_fn, epsilon_percent=0.1)
        >>> result = searcher.search_layer("transformer.h.0.attn.c_attn", layer)
        >>> print(f"k_S={result.k_S}, k_T={result.k_T}, r={result.r}")
    """
    
    def __init__(
        self,
        model: nn.Module,
        xtx_map: Dict[str, torch.Tensor],
        eval_fn: Callable[[nn.Module], float],
        epsilon_percent: float = 0.1,
        energy_min: float = 0.8,
        energy_max: float = 0.9999,
    ):
        self.model = model
        self.xtx_map = xtx_map
        self.eval_fn = eval_fn
        self.eps = epsilon_percent
        self.energy_min = energy_min
        self.energy_max = energy_max

    def _metric_ok(self, base_metric: float, new_metric: float) -> bool:
        """
        Check if new metric is within epsilon percent of baseline.
        
        Args:
            base_metric: Baseline metric value
            new_metric: New metric value to check
            
        Returns:
            True if relative change <= epsilon_percent
        """
        delta = 100.0 * (new_metric - base_metric) / max(base_metric, 1e-6)
        return delta <= self.eps

    def search_layer(self, name: str, linear: nn.Module, max_iters: int = 12) -> RankResult:
        """
        Binary search for optimal energy thresholds for a single layer.
        
        Supports both nn.Linear and GPT-2 Conv1D layers.
        
        Args:
            name: Qualified name of the layer
            linear: Linear or Conv1D module to search
            max_iters: Maximum binary search iterations (default: 12, ~1e-4 resolution)
            
        Returns:
            RankResult with optimal dimensions and utilized rank
        """
        # Baseline metric
        with eval_no_grad(self.model):
            base = self.eval_fn(self.model)
        
        if name not in self.xtx_map:
            # Layer not in statistics - return full rank
            d = getattr(linear, "in_features", linear.weight.shape[1])
            m = getattr(linear, "out_features", linear.weight.shape[0])
            return RankResult(k_S=d, k_T=m, r=min(d, m))
        
        xtx = self.xtx_map[name]
        
        # Check if this is a Conv1D layer (GPT-2)
        is_conv1d = HAS_CONV1D and Conv1D is not None and isinstance(linear, Conv1D)
        
        # Get weight as (m, d) regardless of module type
        if is_conv1d:
            # Conv1D stores as (in, out) => transpose to (out, in)
            W = linear.weight.data.T.contiguous()
        else:
            W = linear.weight.data
        m, d = W.shape
        
        # Binary search on energy threshold
        lo, hi = self.energy_min, self.energy_max
        best = None
        best_energy = None
        
        for _ in range(max_iters):
            # Use same energy for both subspaces
            e_S = e_T = (lo + hi) / 2
            
            # Compute input subspace (stays on CUDA if xtx is on CUDA)
            V_S, k_S = topk_from_xtx(xtx.to(W.device), e_S)
            P_S = projector_from_vecs(V_S)
            
            # Infer output subspace: T^T T = W · (X^T X) · W^T
            ttx = W @ xtx.to(W.device) @ W.T
            V_T, k_T = topk_from_xtx(ttx, e_T)  # No .cpu() - stay on device
            P_T = projector_from_vecs(V_T)
            
            # Save original weight in native format
            if is_conv1d:
                W_native_save = linear.weight.data.clone()
            else:
                W_native_save = linear.weight.data.clone()
            
            # Transform weight
            W_prime = P_T.to(W) @ W @ P_S.to(W)
            
            # Apply transformed weight back to module in native format
            if is_conv1d:
                # Transpose back for Conv1D storage
                linear.weight.data.copy_(W_prime.T)
            else:
                linear.weight.data.copy_(W_prime)
            
            # Evaluate
            with eval_no_grad(self.model):
                m_new = self.eval_fn(self.model)
            
            ok = self._metric_ok(base, m_new)
            
            # Restore original weight
            linear.weight.data.copy_(W_native_save)
            
            if ok:
                # Transformation acceptable - try more aggressive (lower energy)
                best = (k_S, k_T)
                best_energy = (e_S, e_T)
                hi = e_S
            else:
                # Too aggressive - increase energy
                lo = e_S
        
        if best is None:
            # No acceptable transformation found - use full rank
            return RankResult(k_S=d, k_T=m, r=min(d, m))
        
        k_S, k_T = best
        r = min(k_S, k_T)
        e_S, e_T = best_energy if best_energy else (None, None)
        
        return RankResult(k_S=k_S, k_T=k_T, r=r, e_S=e_S, e_T=e_T)
    
    def search_all_layers(self, layer_filter: Optional[Callable[[str], bool]] = None) -> Dict[str, RankResult]:
        """
        Search optimal ranks for all eligible Linear/Conv1D layers.
        
        Args:
            layer_filter: Optional function to filter layer names (default: all in xtx_map)
            
        Returns:
            Dictionary mapping layer names to RankResult objects
        """
        results = {}
        
        for name, module in self.model.named_modules():
            # Check for both nn.Linear and Conv1D (GPT-2)
            is_target = isinstance(module, nn.Linear)
            if HAS_CONV1D and Conv1D is not None:
                is_target = is_target or isinstance(module, Conv1D)
            
            if not is_target:
                continue
            if name not in self.xtx_map:
                continue
            if layer_filter and not layer_filter(name):
                continue
            
            print(f"Searching layer: {name}")
            try:
                result = self.search_layer(name, module)
                results[name] = result
                print(f"  -> k_S={result.k_S}, k_T={result.k_T}, r={result.r}")
            except Exception as e:
                print(f"[WARN] Skipping {name}: {e}")
        
        return results