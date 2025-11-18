"""
Custom GPT-2 model class that supports factorized low-rank linear layers.

This module provides a drop-in replacement for GPT2LMHeadModel that can handle
layers factorized as Sequential(A, B) while maintaining compatibility with
Hugging Face transformers for fine-tuning.
"""

import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Config
from transformers.models.gpt2.modeling_gpt2 import GPT2Block
from typing import Optional


class FactorizedLinear(nn.Module):
    """
    A factorized linear layer that computes y = (B @ A) @ x without fusing.
    
    This keeps the low-rank structure A (r, d) and B (m, r) separate,
    enabling parameter-efficient storage and fine-tuning.
    
    Args:
        in_features: Input dimension d
        out_features: Output dimension m
        rank: Bottleneck rank r
        bias: Whether to include bias term
    """
    
    def __init__(self, in_features: int, out_features: int, rank: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        
        # A: (r, d) - first projection
        self.A = nn.Linear(in_features, rank, bias=False)
        
        # B: (m, r) - second projection
        self.B = nn.Linear(rank, out_features, bias=bias)
    
    def forward(self, x):
        """Forward pass: y = B(A(x))"""
        return self.B(self.A(x))
    
    @classmethod
    def from_sequential(cls, sequential: nn.Sequential):
        """
        Create FactorizedLinear from Sequential(Linear, Linear).
        
        Args:
            sequential: nn.Sequential containing two Linear layers
        """
        if len(sequential) != 2:
            raise ValueError(f"Expected Sequential with 2 layers, got {len(sequential)}")
        
        A_module, B_module = sequential[0], sequential[1]
        
        if not (isinstance(A_module, nn.Linear) and isinstance(B_module, nn.Linear)):
            raise ValueError("Sequential must contain two nn.Linear layers")
        
        factorized = cls(
            in_features=A_module.in_features,
            out_features=B_module.out_features,
            rank=A_module.out_features,
            bias=B_module.bias is not None
        )
        
        # Copy weights
        factorized.A.weight.data = A_module.weight.data.clone()
        factorized.B.weight.data = B_module.weight.data.clone()
        
        if B_module.bias is not None:
            factorized.B.bias.data = B_module.bias.data.clone()
        
        return factorized
    
    def extra_repr(self):
        return f'in_features={self.in_features}, out_features={self.out_features}, rank={self.rank}'


class GPT2CompressedLMHeadModel(GPT2LMHeadModel):
    """
    GPT-2 Language Model with support for factorized low-rank layers.
    
    This class extends GPT2LMHeadModel to properly load and save models
    that have been compressed with factorized linear layers.
    
    Usage:
        # Load compressed model
        model = GPT2CompressedLMHeadModel.from_pretrained("path/to/compressed")
        
        # Use normally for training/inference
        outputs = model(input_ids)
        
        # Save (preserves factorized structure)
        model.save_pretrained("path/to/save")
    """
    
    def __init__(self, config: GPT2Config):
        super().__init__(config)
        self._replace_sequential_with_factorized()
    
    def _replace_sequential_with_factorized(self):
        """
        Replace all Sequential(Linear, Linear) with FactorizedLinear.
        """
        def replace_in_module(module, parent_name=""):
            for name, child in list(module.named_children()):
                full_name = f"{parent_name}.{name}" if parent_name else name
                
                # Check if this is a factorized Sequential
                if isinstance(child, nn.Sequential) and len(child) == 2:
                    if isinstance(child[0], nn.Linear) and isinstance(child[1], nn.Linear):
                        # Replace with FactorizedLinear
                        factorized = FactorizedLinear.from_sequential(child)
                        setattr(module, name, factorized)
                        print(f"Converted {full_name} to FactorizedLinear "
                              f"(rank={factorized.rank})")
                
                # Recurse
                replace_in_module(child, full_name)
        
        replace_in_module(self)
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        """
        Load a compressed GPT-2 model with factorized layers.
        
        This method first loads the model using the parent class, then
        converts any Sequential(Linear, Linear) modules to FactorizedLinear.
        """
        # Load config first
        config = GPT2Config.from_pretrained(pretrained_model_name_or_path)
        
        # Create model instance
        model = cls(config)
        
        # Load state dict
        try:
            import os
            model_path = pretrained_model_name_or_path
            
            # Try safetensors first, then pytorch_model.bin
            from safetensors.torch import load_file
            
            safetensors_path = os.path.join(model_path, "model.safetensors")
            pytorch_path = os.path.join(model_path, "pytorch_model.bin")
            
            if os.path.exists(safetensors_path):
                state_dict = load_file(safetensors_path)
                print(f"Loaded state dict from safetensors")
            elif os.path.exists(pytorch_path):
                state_dict = torch.load(pytorch_path, map_location="cpu")
                print(f"Loaded state dict from pytorch_model.bin")
            else:
                # Fall back to parent's loading
                return super().from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
            
            # Load state dict with strict=False to handle factorized keys
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            
            if missing:
                print(f"Missing keys: {len(missing)}")
            if unexpected:
                print(f"Unexpected keys: {len(unexpected)}")
            
        except Exception as e:
            print(f"Error loading custom: {e}")
            print("Falling back to parent's from_pretrained")
            return super().from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        
        return model


# Register for AutoModel
try:
    from transformers import AutoModelForCausalLM
    AutoModelForCausalLM.register(GPT2Config, GPT2CompressedLMHeadModel)
except:
    pass