"""
Model instrumentation utilities for layer iteration and replacement.

This module provides utilities for:
- Iterating through all Linear layers in a model
- Replacing Linear layers with factored low-rank versions
"""

from typing import Iterator, Tuple
import torch.nn as nn


def iter_linear_layers(model: nn.Module) -> Iterator[Tuple[str, nn.Linear]]:
    """
    Iterate through all nn.Linear layers with their qualified names.
    
    This function recursively traverses the model hierarchy and yields
    each Linear layer along with its fully qualified name.
    
    Args:
        model: PyTorch model to iterate through
        
    Yields:
        Tuple of (layer_name, linear_module) for each Linear layer
        
    Example:
        >>> for name, layer in iter_linear_layers(model):
        ...     print(f"{name}: {layer.in_features} -> {layer.out_features}")
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            yield name, module


def replace_linear(parent: nn.Module, attr: str, left: nn.Linear, right: nn.Linear):
    """
    Replace a Linear layer with two factored Linears in sequence.
    
    The original layer parent.attr is replaced with nn.Sequential(right, left),
    implementing the factorization W ≈ left @ right where:
    - right: (r, d) - projects input from d dimensions to rank r
    - left: (m, r) - projects from rank r to m output dimensions
    
    Args:
        parent: Parent module containing the layer to replace
        attr: Attribute name of the layer in parent
        left: Left factor (m, r) - output projection
        right: Right factor (r, d) - input projection
        
    Example:
        >>> parent = model.transformer.h[0]
        >>> left = nn.Linear(r, out_features, bias=False)
        >>> right = nn.Linear(in_features, r, bias=False)
        >>> replace_linear(parent, 'mlp.c_fc', left, right)
    """
    setattr(parent, attr, nn.Sequential(right, left))


def get_parent_and_attr(model: nn.Module, name: str) -> Tuple[nn.Module, str]:
    """
    Get parent module and attribute name for a layer specified by qualified name.
    
    Args:
        model: Root model
        name: Qualified name of the layer (e.g., "transformer.h.0.mlp.c_fc")
        
    Returns:
        Tuple of (parent_module, attribute_name)
        
    Example:
        >>> parent, attr = get_parent_and_attr(model, "transformer.h.0.attn.c_attn")
        >>> layer = getattr(parent, attr)
    """
    if '.' not in name:
        return model, name
    
    parent_name, attr = name.rsplit('.', 1)
    parent = model.get_submodule(parent_name)
    return parent, attr