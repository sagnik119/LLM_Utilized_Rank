"""
Custom GPT-2 model class that supports factorized low-rank linear layers.

This module provides a drop-in replacement for GPT2LMHeadModel that can handle
layers factorized as A @ B matrices, loading them from state dict.
"""

import torch
import torch.nn as nn
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel, Conv1D
from typing import Optional

# IMPORTANT: load the config from separate file
from .configuration_gpt2_compressed import GPT2CompressedConfig


class FactorizedLinear(nn.Module):
    """
    Low-rank replacement for nn.Linear or Conv1D:
        y = A(Bx)     where A ∈ R[d_out × r], B ∈ R[r × d_in]
    
    This implementation directly stores A and B as Parameters,
    making them compatible with standard PyTorch serialization.
    """
    
    def __init__(self, A: torch.Tensor, B: torch.Tensor, bias: Optional[torch.Tensor] = None):
        super().__init__()
        d_out, r = A.shape
        r2, d_in = B.shape
        assert r == r2, f"Rank mismatch: A has rank {r}, B has rank {r2}"
        
        self.d_out = d_out
        self.d_in = d_in
        self.rank = r
        
        # Store as parameters for automatic grad + serialization
        self.A = nn.Parameter(A.clone())  # (d_out, r)
        self.B = nn.Parameter(B.clone())  # (r, d_in)

        if bias is not None:
            self.bias = nn.Parameter(bias.clone())
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x):
        """
        Forward pass: y = A @ (B @ x^T)^T + bias
        
        Handles both 2D and 3D inputs:
        - x: (..., d_in) → y: (..., d_out)
        """
        # x: (..., d_in)
        x = torch.matmul(x, self.B.T)  # (..., r)
        x = torch.matmul(x, self.A.T)  # (..., d_out)
        
        if self.bias is not None:
            x = x + self.bias
        
        return x
    
    def extra_repr(self):
        return f'd_in={self.d_in}, d_out={self.d_out}, rank={self.rank}'


def convert_linear_to_factorized(module, state_dict, prefix=""):
    """
    If state_dict contains A and B for this module, load as a FactorizedLinear.
    
    Args:
        module: Original Linear or Conv1D module
        state_dict: Full model state dict
        prefix: Prefix for this module's keys in state_dict
        
    Returns:
        FactorizedLinear if A and B found, otherwise original module
    """
    A_key = prefix + ".A"
    B_key = prefix + ".B"
    bias_key = prefix + ".bias"
    
    if A_key in state_dict and B_key in state_dict:
        A = state_dict[A_key]
        B = state_dict[B_key]
        bias = state_dict.get(bias_key, None)
        return FactorizedLinear(A, B, bias)
    
    return module  # not factorized


class GPT2CompressedLMHeadModel(GPT2LMHeadModel):
    """
    GPT-2 Language Model with support for factorized low-rank layers.
    
    This class extends GPT2LMHeadModel to automatically detect and load
    factorized layers from the state dict.
    """
    config_class = GPT2CompressedConfig
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        """
        Load a compressed GPT-2 model with factorized layers.
        
        This method detects .A and .B weight keys and replaces the corresponding
        modules with FactorizedLinear before loading weights.
        """
        import os
        from safetensors.torch import load_file as safe_load
        
        kwargs.setdefault("trust_remote_code", True)
        
        # Try to load state dict first to detect structure
        model_path = str(pretrained_model_name_or_path)
        safe_path = os.path.join(model_path, "model.safetensors")
        pt_path = os.path.join(model_path, "pytorch_model.bin")
        
        state_dict = None
        if os.path.exists(safe_path):
            state_dict = safe_load(safe_path)
            print(f"[LOAD] Loaded state dict from safetensors")
        elif os.path.exists(pt_path):
            state_dict = torch.load(pt_path, map_location="cpu")
            print(f"[LOAD] Loaded state dict from pytorch_model.bin")
        
        if state_dict is None:
            # No local checkpoint, use standard loading
            return super().from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        
        # Detect factorized layers
        factorized_prefixes = set()
        for key in state_dict.keys():
            if ".A" in key:
                # Extract base name (everything before .A)
                prefix = key.split(".A")[0]
                factorized_prefixes.add(prefix)
        
        print(f"[LOAD] Detected {len(factorized_prefixes)} factorized layers")
        
        # Load config
        try:
            config = GPT2CompressedConfig.from_pretrained(model_path)
        except:
            from transformers import GPT2Config
            config = GPT2Config.from_pretrained(model_path)
        
        # Create base model
        model = cls(config)
        
        # Replace factorized modules
        for prefix in factorized_prefixes:
            # Navigate to parent module
            parts = prefix.split(".")
            parent = model
            for part in parts[:-1]:
                if part.isdigit():
                    parent = parent[int(part)]
                else:
                    parent = getattr(parent, part)
            
            attr_name = parts[-1]
            orig_module = getattr(parent, attr_name)
            
            # Replace with factorized version
            factorized = convert_linear_to_factorized(orig_module, state_dict, prefix)
            
            if isinstance(factorized, FactorizedLinear):
                setattr(parent, attr_name, factorized)
                print(f"[LOAD] Replaced {prefix} with FactorizedLinear(rank={factorized.rank})")
        
        # Load all weights (factorized modules will receive A, B, bias)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        
        if missing:
            print(f"[LOAD] Missing keys: {len(missing)}")
        if unexpected:
            print(f"[LOAD] Unexpected keys: {len(unexpected)}")
        
        return model


# Register for AutoModel and AutoConfig
try:
    from transformers import AutoConfig, AutoModelForCausalLM
    AutoConfig.register("gpt2_compressed", GPT2CompressedConfig)
    AutoModelForCausalLM.register(GPT2CompressedConfig, GPT2CompressedLMHeadModel)
except:
    pass