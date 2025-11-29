"""
Custom LLaMA model class that supports factorized low-rank linear layers.

This module provides a drop-in replacement for LlamaForCausalLM that can handle
layers factorized as A @ B matrices, loading them from state dict.
"""

import torch
import torch.nn as nn
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from typing import Optional

# IMPORTANT: load the config from separate file
from .configuration_llama_compressed import LlamaCompressedConfig


class FactorizedLinear(nn.Module):
    """
    Low-rank replacement for nn.Linear:
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
        # Save original shape and flatten to 2D
        orig_shape = x.shape
        x = x.reshape(-1, self.d_in)  # (N, d_in)
        
        # Two-stage matmul: x @ B^T @ A^T
        x = x @ self.B.T  # (N, r)
        x = x @ self.A.T  # (N, d_out)
        
        if self.bias is not None:
            x = x + self.bias
        
        # Restore original shape
        return x.reshape(*orig_shape[:-1], self.d_out)
    
    def extra_repr(self):
        return f'd_in={self.d_in}, d_out={self.d_out}, rank={self.rank}'


def convert_linear_to_factorized(module, state_dict, prefix=""):
    """
    If state_dict contains A and B for this module, load as a FactorizedLinear.
    
    Args:
        module: Original Linear module
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
        
        # Validate shapes using weight tensor (universally correct for all Linear variants)
        if hasattr(module, "weight"):
            w_out, w_in = module.weight.shape
            if A.shape[0] != w_out or B.shape[1] != w_in:
                print(f"[WARN] Shape mismatch for {prefix}: A={A.shape}, B={B.shape}, "
                      f"module=({w_out},{w_in})")
                return module
        
        return FactorizedLinear(A, B, bias)
    
    return module  # not factorized


class LlamaCompressedForCausalLM(LlamaForCausalLM):
    """
    LLaMA Language Model with support for factorized low-rank layers.
    
    This class extends LlamaForCausalLM to automatically detect and load
    factorized layers from the state dict.
    """
    config_class = LlamaCompressedConfig
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        """
        Load a compressed LLaMA model with factorized layers.
        
        This method detects .A and .B weight keys and replaces the corresponding
        modules with FactorizedLinear before loading weights.
        """
        import os
        import glob
        from safetensors.torch import load_file as safe_load
        
        kwargs.setdefault("trust_remote_code", True)
        
        # Try to load state dict first to detect structure
        model_path = str(pretrained_model_name_or_path)
        
        state_dict = None
        
        # Check for sharded safetensors (e.g., model-00001-of-00002.safetensors)
        sharded_files = sorted(glob.glob(os.path.join(model_path, "model-*-of-*.safetensors")))
        if sharded_files:
            print(f"[LOAD] Loading {len(sharded_files)} sharded safetensors files...")
            # Use load_files for proper tensor sharing and metadata
            try:
                from safetensors import safe_open
                state_dict = {}
                for shard_path in sharded_files:
                    with safe_open(shard_path, framework="pt", device="cpu") as f:
                        for key in f.keys():
                            state_dict[key] = f.get_tensor(key)
                print(f"[LOAD] Loaded {len(state_dict)} keys from sharded safetensors")
            except ImportError:
                # Fallback to manual merging if safe_open not available
                state_dict = {}
                for shard_path in sharded_files:
                    shard_dict = safe_load(shard_path)
                    state_dict.update(shard_dict)
                print(f"[LOAD] Loaded {len(state_dict)} keys from sharded safetensors (fallback)")
        else:
            # Try monolithic files
            safe_path = os.path.join(model_path, "model.safetensors")
            pt_path = os.path.join(model_path, "pytorch_model.bin")
            
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
            config = LlamaCompressedConfig.from_pretrained(model_path)
        except:
            from transformers import LlamaConfig
            config = LlamaConfig.from_pretrained(model_path)
        
        # Create base model
        model = cls(config)
        
        # Replace factorized modules
        for prefix in factorized_prefixes:
            # Navigate to parent module
            parts = prefix.split(".")
            parent = model
            
            try:
                for part in parts[:-1]:
                    if hasattr(parent, part):
                        parent = getattr(parent, part)
                    elif part.isdigit():
                        parent = parent[int(part)]
                    else:
                        raise KeyError(f"Cannot navigate to {part} in prefix {prefix}")
                
                attr_name = parts[-1]
                orig_module = getattr(parent, attr_name)
                
                # Validate module has weight attribute (supports nn.Linear and variants)
                if not hasattr(orig_module, "weight"):
                    print(f"[WARN] {prefix} has no weight attribute (got {type(orig_module).__name__}), skipping")
                    continue
                
                # Get A and B from state dict
                A = state_dict[prefix + ".A"]
                B = state_dict[prefix + ".B"]
                
                # Validate shapes using weight tensor (universally correct)
                w_out, w_in = orig_module.weight.shape
                if A.shape[0] != w_out or B.shape[1] != w_in:
                    print(f"[WARN] Shape mismatch in {prefix}: "
                          f"A={A.shape}, B={B.shape}, module=({w_out},{w_in}), skipping")
                    continue
                
                # Replace with factorized version
                factorized = convert_linear_to_factorized(orig_module, state_dict, prefix)
                
                if isinstance(factorized, FactorizedLinear):
                    setattr(parent, attr_name, factorized)
                    print(f"[LOAD] Replaced {prefix} with FactorizedLinear(rank={factorized.rank})")
            
            except Exception as e:
                print(f"[ERROR] Failed to replace {prefix}: {e}")
                continue
        
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
    AutoConfig.register("llama_compressed", LlamaCompressedConfig)
    AutoModelForCausalLM.register(LlamaCompressedConfig, LlamaCompressedForCausalLM)
except:
    pass