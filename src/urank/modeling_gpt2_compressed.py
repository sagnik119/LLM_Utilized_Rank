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
        
        This method detects factorized state dict keys and rebuilds the architecture accordingly.
        """
        import os
        from safetensors.torch import load_file
        
        # Load state dict to inspect structure
        model_path = pretrained_model_name_or_path
        safetensors_path = os.path.join(model_path, "model.safetensors")
        pytorch_path = os.path.join(model_path, "pytorch_model.bin")
        
        if os.path.exists(safetensors_path):
            state_dict = load_file(safetensors_path)
            print(f"Loaded state dict from safetensors")
        elif os.path.exists(pytorch_path):
            state_dict = torch.load(pytorch_path, map_location="cpu")
            print(f"Loaded state dict from pytorch_model.bin")
        else:
            # Fall back to standard loading
            return super().from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        
        # Detect factorized layers by looking for .0.weight and .1.weight keys
        factorized_layers = set()
        for key in state_dict.keys():
            if ".0.weight" in key:
                # Extract base layer name (e.g., "transformer.h.0.attn.c_attn")
                base_key = key.replace(".0.weight", "")
                factorized_layers.add(base_key)
        
        print(f"Detected {len(factorized_layers)} factorized layers")
        
        # Load config
        config = GPT2Config.from_pretrained(pretrained_model_name_or_path)
        
        # Create standard model first
        model = super(GPT2CompressedLMHeadModel, cls).__init__(config)
        
        # Now replace factorized layers with FactorizedLinear modules
        for layer_name in factorized_layers:
            # Get the weights
            weight_0_key = f"{layer_name}.0.weight"
            weight_1_key = f"{layer_name}.1.weight"
            bias_1_key = f"{layer_name}.1.bias"
            
            if weight_0_key in state_dict and weight_1_key in state_dict:
                A_weight = state_dict[weight_0_key]  # (r, d)
                B_weight = state_dict[weight_1_key]  # (m, r)
                B_bias = state_dict.get(bias_1_key)  # (m,)
                
                # Create FactorizedLinear
                factorized = FactorizedLinear(
                    in_features=A_weight.shape[1],  # d
                    out_features=B_weight.shape[0],  # m
                    rank=A_weight.shape[0],          # r
                    bias=B_bias is not None
                )
                
                # Load weights
                factorized.A.weight.data = A_weight
                factorized.B.weight.data = B_weight
                if B_bias is not None:
                    factorized.B.bias.data = B_bias
                
                # Navigate to parent and replace
                parts = layer_name.split(".")
                parent = model
                for part in parts[:-1]:
                    if part.isdigit():
                        parent = parent[int(part)]
                    else:
                        parent = getattr(parent, part)
                
                setattr(parent, parts[-1], factorized)
                print(f"Replaced {layer_name} with FactorizedLinear(rank={factorized.rank})")
        
        # Load remaining (non-factorized) weights
        # Filter out factorized keys
        filtered_state_dict = {
            k: v for k, v in state_dict.items()
            if not any(f"{layer}." in k for layer in factorized_layers)
        }
        
        missing, unexpected = model.load_state_dict(filtered_state_dict, strict=False)
        print(f"Loaded non-factorized weights (missing: {len(missing)}, unexpected: {len(unexpected)})")
        
        return model


# Register for AutoModel
try:
    from transformers import AutoModelForCausalLM
    AutoModelForCausalLM.register(GPT2Config, GPT2CompressedLMHeadModel)
except:
    pass