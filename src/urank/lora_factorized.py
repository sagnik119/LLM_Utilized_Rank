"""
LoRA-enhanced FactorizedLinear for efficient fine-tuning of compressed models.

This module provides LoRA adapters directly on the A and B factors of 
FactorizedLinear layers, allowing efficient fine-tuning without PEFT.
"""

import math
import torch
import torch.nn as nn
from typing import Optional


class LoRAFactorizedLinear(nn.Module):
    """
    FactorizedLinear with built-in LoRA adapters on both A and B factors.
    
    Base computation: y = (x @ B^T) @ A^T
    
    With LoRA:
        A_eff = A + ΔA  where ΔA = L_A @ R_A
        B_eff = B + ΔB  where ΔB = L_B @ R_B
        y = (x @ B_eff^T) @ A_eff^T
    
    Shapes:
        A: (d_out, r) - left factor
        B: (r, d_in) - right factor
        L_A: (d_out, lora_r)
        R_A: (lora_r, r)
        ΔA: (d_out, r)
        L_B: (r, lora_r)
        R_B: (lora_r, d_in)
        ΔB: (r, d_in)
    """
    
    def __init__(
        self, 
        A: torch.Tensor, 
        B: torch.Tensor, 
        bias: Optional[torch.Tensor] = None,
        lora_r: int = 0,
        lora_alpha: float = 1.0,
        lora_dropout: float = 0.0,
        freeze_base: bool = True
    ):
        super().__init__()
        d_out, r = A.shape
        r2, d_in = B.shape
        assert r == r2, f"Rank mismatch: A has rank {r}, B has rank {r2}"
        
        self.d_out = d_out
        self.d_in = d_in
        self.r = r
        self.lora_r = lora_r
        
        # Base factors (compressed model weights)
        # IMPORTANT: Must explicitly set requires_grad to freeze base weights
        A_param = A.clone()
        B_param = B.clone()
        self.A = nn.Parameter(A_param, requires_grad=not freeze_base)
        self.B = nn.Parameter(B_param, requires_grad=not freeze_base)
        
        if bias is not None:
            bias_param = bias.clone()
            self.bias = nn.Parameter(bias_param, requires_grad=not freeze_base)
        else:
            self.register_parameter('bias', None)
        
        # LoRA adapters
        if lora_r > 0:
            # ΔA = L_A @ R_A: (d_out, r)
            self.lora_A_L = nn.Parameter(torch.zeros(d_out, lora_r))
            self.lora_A_R = nn.Parameter(torch.zeros(lora_r, r))
            
            # ΔB = L_B @ R_B: (r, d_in)
            self.lora_B_L = nn.Parameter(torch.zeros(r, lora_r))
            self.lora_B_R = nn.Parameter(torch.zeros(lora_r, d_in))
            
            # Initialize LoRA weights
            nn.init.kaiming_uniform_(self.lora_A_L, a=math.sqrt(5))
            nn.init.zeros_(self.lora_A_R)
            nn.init.kaiming_uniform_(self.lora_B_L, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B_R)
            
            # Scaling factor
            self.scaling = lora_alpha / lora_r
            
            # Dropout
            if lora_dropout > 0:
                self.lora_dropout = nn.Dropout(lora_dropout)
            else:
                self.lora_dropout = nn.Identity()
        else:
            self.scaling = 1.0
            self.lora_dropout = nn.Identity()
    
    def forward(self, x):
        """
        Forward pass with LoRA adapters.
        
        Args:
            x: Input tensor (..., d_in)
            
        Returns:
            Output tensor (..., d_out)
        """
        # Save original shape and flatten to 2D
        orig_shape = x.shape
        x = x.reshape(-1, self.d_in)  # (N, d_in)
        
        # Compute effective factors
        A_eff = self.A
        B_eff = self.B
        
        # Add LoRA if enabled
        if self.lora_r > 0:
            # ΔA = L_A @ R_A: (d_out, lora_r) @ (lora_r, r) = (d_out, r)
            delta_A = (self.lora_A_L @ self.lora_A_R) * self.scaling
            delta_A = self.lora_dropout(delta_A)
            
            # ΔB = L_B @ R_B: (r, lora_r) @ (lora_r, d_in) = (r, d_in)
            delta_B = (self.lora_B_L @ self.lora_B_R) * self.scaling
            delta_B = self.lora_dropout(delta_B)
            
            # Effective factors
            A_eff = A_eff + delta_A
            B_eff = B_eff + delta_B
        
        # y = (x @ B_eff^T) @ A_eff^T
        output = x @ B_eff.T  # (N, r)
        output = output @ A_eff.T  # (N, d_out)
        
        # Add bias
        if self.bias is not None:
            output = output + self.bias
        
        # Restore original shape
        return output.reshape(*orig_shape[:-1], self.d_out)
    
    def extra_repr(self):
        s = f'd_in={self.d_in}, d_out={self.d_out}, rank={self.r}'
        if self.lora_r > 0:
            s += f', lora_r={self.lora_r}, scaling={self.scaling}'
        return s
    
    @classmethod
    def from_factorized_linear(cls, factorized_linear, lora_r=0, lora_alpha=1.0, lora_dropout=0.0, freeze_base=True):
        """
        Convert a FactorizedLinear to LoRAFactorizedLinear.
        
        Args:
            factorized_linear: Existing FactorizedLinear module
            lora_r: LoRA rank
            lora_alpha: LoRA scaling factor
            lora_dropout: LoRA dropout rate
            freeze_base: Whether to freeze A and B
            
        Returns:
            LoRAFactorizedLinear with same base weights
        """
        # Use .detach() instead of .data to properly handle requires_grad
        return cls(
            A=factorized_linear.A.detach(),
            B=factorized_linear.B.detach(),
            bias=factorized_linear.bias.detach() if factorized_linear.bias is not None else None,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            freeze_base=freeze_base
        )


def convert_to_lora_factorized(model, lora_r=8, lora_alpha=16, lora_dropout=0.05, freeze_base=True):
    """
    Convert all FactorizedLinear modules in a model to LoRAFactorizedLinear.
    
    Args:
        model: Model containing FactorizedLinear modules
        lora_r: LoRA rank
        lora_alpha: LoRA scaling factor
        lora_dropout: LoRA dropout rate
        freeze_base: Whether to freeze base A and B factors
        
    Returns:
        Number of modules converted
    """
    # Try importing both FactorizedLinear implementations
    try:
        from .modeling_gpt2_compressed import FactorizedLinear as GPT2FactorizedLinear
    except:
        GPT2FactorizedLinear = None
    
    try:
        from .modeling_llama_compressed import FactorizedLinear as LlamaFactorizedLinear
    except:
        LlamaFactorizedLinear = None
    
    count = 0
    
    # Collect all FactorizedLinear modules
    to_replace = []
    for name, module in model.named_modules():
        is_factorized = False
        
        if GPT2FactorizedLinear is not None and isinstance(module, GPT2FactorizedLinear):
            is_factorized = True
        elif LlamaFactorizedLinear is not None and isinstance(module, LlamaFactorizedLinear):
            is_factorized = True
        
        if is_factorized:
            to_replace.append((name, module))
    
    # Replace each FactorizedLinear with LoRAFactorizedLinear
    for name, module in to_replace:
        # Navigate to parent
        parts = name.split('.')
        parent = model
        for part in parts[:-1]:
            if part.isdigit():
                parent = parent[int(part)]
            else:
                parent = getattr(parent, part)
        
        # Create LoRA-enhanced version
        lora_module = LoRAFactorizedLinear.from_factorized_linear(
            module,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            freeze_base=freeze_base
        )
        
        # Replace
        setattr(parent, parts[-1], lora_module)
        count += 1
        
        print(f"[LoRA] Converted {name} to LoRAFactorizedLinear (r={lora_r})")
    
    return count


def count_lora_parameters(model):
    """
    Count LoRA parameters separately from base parameters.
    
    Returns:
        dict with 'base', 'lora', and 'total' parameter counts
    """
    base_params = 0
    lora_params = 0
    
    for name, param in model.named_parameters():
        if 'lora_' in name:
            lora_params += param.numel()
        else:
            base_params += param.numel()
    
    return {
        'base': base_params,
        'lora': lora_params,
        'total': base_params + lora_params,
        'trainable': sum(p.numel() for p in model.parameters() if p.requires_grad)
    }