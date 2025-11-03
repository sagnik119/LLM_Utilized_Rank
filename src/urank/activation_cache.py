"""
Activation statistics collection for subspace analysis.

This module provides hooks to efficiently accumulate X^T X (input covariance)
for each Linear layer without storing full activations.
"""

from dataclasses import dataclass
from typing import Dict, List
import torch
import torch.nn as nn


@dataclass
class LayerStat:
    """
    Statistics for a single layer.
    
    Attributes:
        xtx: Accumulated X^T X matrix (input covariance)
        count: Total number of tokens/samples processed
    """
    xtx: torch.Tensor
    count: int


class ActivationStatsCollector:
    """
    Collects X^T X statistics for Linear layers via forward hooks.
    
    For a Linear layer with weight (out_features, in_features) and forward
    y = x @ W^T + b, we accumulate X^T X over token positions/batch to later
    compute SVD of input space S.
    
    We DO NOT store Y. Later, for each layer, we infer T^T T = W · (X^T X) · W^T.
    
    Args:
        model: PyTorch model to instrument
        dtype: Data type for accumulation (default: torch.float32)
        device: Device for accumulation (default: None, uses CPU)
        
    Example:
        >>> collector = ActivationStatsCollector(model).start()
        >>> # Run forward passes
        >>> for batch in dataloader:
        ...     model(batch)
        >>> stats = collector.stop()
        >>> torch.save({k: v.xtx for k, v in stats.items()}, "stats.pt")
    """
    
    def __init__(self, model: nn.Module, dtype=torch.float32, device=None):
        self.model = model
        self.dtype = dtype
        self.device = device
        self.stats: Dict[str, LayerStat] = {}
        self._hooks: List[torch.utils.hooks.RemovableHandle] = []

    def _hook(self, name: str, module: nn.Linear, inputs):
        """
        Forward pre-hook to accumulate X^T X for a layer.
        
        Args:
            name: Qualified name of the layer
            module: Linear module
            inputs: Input tuple from forward pass
        """
        (x,) = inputs  # [..., in_features]
        # Flatten to (num_tokens, in_features)
        x2d = x.reshape(-1, x.shape[-1]).to(self.dtype)
        xtx = x2d.T @ x2d  # (d, d)
        
        if name not in self.stats:
            self.stats[name] = LayerStat(
                xtx=xtx.detach().cpu(),
                count=x2d.shape[0]
            )
        else:
            s = self.stats[name]
            s.xtx += xtx.detach().cpu()
            s.count += x2d.shape[0]

    def start(self):
        """
        Start collecting statistics by registering hooks.
        
        Returns:
            self for method chaining
        """
        layer_count = 0
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                # Create a closure to properly capture the name
                def make_hook(layer_name):
                    def hook_fn(mod, inp):
                        return self._hook(layer_name, mod, inp)
                    return hook_fn
                
                hook = module.register_forward_pre_hook(make_hook(name))
                self._hooks.append(hook)
                layer_count += 1
        
        print(f"Registered hooks on {layer_count} Linear layers")
        return self

    def stop(self) -> Dict[str, LayerStat]:
        """
        Stop collecting statistics and remove all hooks.
        
        Returns:
            Dictionary mapping layer names to LayerStat objects
        """
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        return self.stats