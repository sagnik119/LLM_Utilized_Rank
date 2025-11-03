"""
Binary search for optimal ranks with validation constraints.

This module provides utilities for:
- Searching for optimal subspace dimensions via binary search
- Validating rank choices against perplexity/accuracy constraints
- Computing utilized ranks from subspace dimensions
"""

from dataclasses import dataclass
from typing import Dict, Callable, Optional
import torch
import torch.nn as nn

from .subspace import topk_from_xtx, projector_from_vecs
from .transform import transform_weight_ps_pt
from .utils import eval_no_grad


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

    def search_layer(self, name: str, linear: nn.Linear, max_iters: int = 12) -> RankResult:
        """
        Binary search for optimal energy thresholds for a single layer.
        
        Searches for the lowest (e_S, e_T) pair that maintains validation
        metric within epsilon. Uses same threshold for both subspaces for simplicity.
        
        Args:
            name: Qualified name of the layer
            linear: Linear module to search
            max_iters: Maximum binary search iterations (default: 12, ~1e-4 resolution)
            
        Returns:
            RankResult with optimal dimensions and utilized rank
        """
        # Baseline metric
        with eval_no_grad(self.model):
            base = self.eval_fn(self.model)
        
        if name not in self.xtx_map:
            # Layer not in statistics - return full rank
            d = linear.in_features
            m = linear.out_features
            return RankResult(k_S=d, k_T=m, r=min(d, m))
        
        xtx = self.xtx_map[name]
        
        # Binary search on energy threshold
        lo, hi = self.energy_min, self.energy_max
        best = None
        best_energy = None
        
        for _ in range(max_iters):
            # Use same energy for both subspaces
            e_S = e_T = (lo + hi) / 2
            
            # Compute input subspace
            V_S, k_S = topk_from_xtx(xtx, e_S)
            P_S = projector_from_vecs(V_S)
            
            # Infer output subspace: T^T T = W · (X^T X) · W^T
            W = linear.weight.data
            ttx = W @ xtx.to(W.device) @ W.T
            V_T, k_T = topk_from_xtx(ttx.cpu(), e_T)
            P_T = projector_from_vecs(V_T)
            
            # Save and transform weight
            W_save = W.clone()
            W_prime = transform_weight_ps_pt(linear, P_S, P_T)
            linear.weight.data.copy_(W_prime)
            
            # Evaluate
            with eval_no_grad(self.model):
                m_new = self.eval_fn(self.model)
            
            ok = self._metric_ok(base, m_new)
            
            # Restore original weight
            linear.weight.data.copy_(W_save)
            
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
            d = linear.in_features
            m = linear.out_features
            return RankResult(k_S=d, k_T=m, r=min(d, m))
        
        k_S, k_T = best
        r = min(k_S, k_T)
        e_S, e_T = best_energy if best_energy else (None, None)
        
        return RankResult(k_S=k_S, k_T=k_T, r=r, e_S=e_S, e_T=e_T)
    
    def search_all_layers(self, layer_filter: Optional[Callable[[str], bool]] = None) -> Dict[str, RankResult]:
        """
        Search optimal ranks for all eligible layers.
        
        Args:
            layer_filter: Optional function to filter layer names (default: all in xtx_map)
            
        Returns:
            Dictionary mapping layer names to RankResult objects
        """
        results = {}
        
        for name, module in self.model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            if name not in self.xtx_map:
                continue
            if layer_filter and not layer_filter(name):
                continue
            
            print(f"Searching layer: {name}")
            result = self.search_layer(name, module)
            results[name] = result
            print(f"  -> k_S={result.k_S}, k_T={result.k_T}, r={result.r}")
        
        return results