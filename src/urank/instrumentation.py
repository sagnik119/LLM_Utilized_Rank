"""
Model instrumentation utilities for layer iteration and replacement.

This module provides utilities for:
- Iterating through all Linear layers in a model
- Replacing Linear layers with factored low-rank versions
"""

from typing import Iterator, Tuple, List
import torch.nn as nn


# Architecture-specific layer patterns
GPT2_PATTERNS = [
    "attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj"
]

LLAMA_PATTERNS = [
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
]


def get_architecture_patterns(model: nn.Module) -> List[str]:
    """
    Get layer name patterns for a model's architecture.
    
    Args:
        model: PyTorch model
        
    Returns:
        List of layer name patterns to match
        
    Raises:
        ValueError: If architecture is not recognized
    """
    cls_name = model.__class__.__name__.lower()
    
    if "gpt2" in cls_name or "gpt-2" in cls_name:
        return GPT2_PATTERNS
    elif "llama" in cls_name:
        return LLAMA_PATTERNS
    else:
        # Default: return all linear layers
        return []


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


def iter_candidate_linear_modules(model: nn.Module) -> Iterator[Tuple[str, nn.Linear]]:
    """
    Iterate through candidate Linear layers for compression (architecture-aware).
    
    Only yields layers matching architecture-specific patterns (attention, MLP).
    
    Args:
        model: PyTorch model
        
    Yields:
        Tuple of (layer_name, linear_module) for candidate layers
        
    Example:
        >>> for name, layer in iter_candidate_linear_modules(model):
        ...     print(f"Candidate: {name}")
    """
    patterns = get_architecture_patterns(model)
    
    if not patterns:
        # No patterns defined, yield all linear layers
        for name, module in iter_linear_layers(model):
            yield name, module
    else:
        # Filter by architecture patterns
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and any(p in name for p in patterns):
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