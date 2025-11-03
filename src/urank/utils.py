"""
Utility functions for model evaluation and device management.

This module provides helper functions for:
- Safe evaluation mode context management
- Device detection for modules
"""

from contextlib import contextmanager
import torch
import torch.nn as nn


@contextmanager
def eval_no_grad(model: nn.Module):
    """
    Context manager to temporarily set model to eval mode with no gradient computation.
    
    Automatically restores the original training state when exiting the context.
    
    Args:
        model: PyTorch model to evaluate
        
    Example:
        >>> with eval_no_grad(model):
        ...     output = model(input_data)
    """
    was_training = model.training
    try:
        model.eval()
        with torch.no_grad():
            yield
    finally:
        if was_training:
            model.train()


def device_of(module: nn.Module) -> torch.device:
    """
    Get the device of a module from its first parameter.
    
    Args:
        module: PyTorch module
        
    Returns:
        Device where the module parameters are located
        
    Raises:
        StopIteration: If module has no parameters
    """
    return next(module.parameters()).device


def count_parameters(module: nn.Module) -> int:
    """
    Count total number of parameters in a module.
    
    Args:
        module: PyTorch module
        
    Returns:
        Total parameter count
    """
    return sum(p.numel() for p in module.parameters())