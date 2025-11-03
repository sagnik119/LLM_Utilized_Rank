"""
Rank allocation policies for mixed-rank compression.

This module provides utilities for applying different rank allocation strategies
based on layer type (attention Q/K/V/O vs MLP projections).

Policies allow for:
- Uniform rank allocation across all layers
- QKV-heavier allocation (more rank to attention)
- MLP-heavier allocation (more rank to feed-forward)
"""

from dataclasses import dataclass
from typing import Dict, Optional
import re


@dataclass
class Policy:
    """
    Rank allocation policy definition.
    
    Attributes:
        name: Policy name
        type_multipliers: Dict mapping regex patterns to rank multipliers
        min_rank: Minimum rank to assign
        max_rank: Maximum rank to assign
        description: Human-readable description
    """
    name: str
    type_multipliers: Dict[str, float]
    min_rank: int = 4
    max_rank: int = 8192
    description: str = ""

    def assign(self, layer_name: str, base_rank: int) -> int:
        """
        Assign a rank to a layer based on policy.
        
        Args:
            layer_name: Qualified layer name
            base_rank: Base rank from search
            
        Returns:
            Adjusted rank according to policy
            
        Example:
            >>> policy = QKV_HEAVIER
            >>> rank = policy.assign("model.layers.0.self_attn.q_proj", 64)
            >>> print(rank)  # 80 (1.25x multiplier)
        """
        mult = 1.0
        
        # Find first matching pattern
        for pattern, multiplier in self.type_multipliers.items():
            if re.search(pattern, layer_name):
                mult = multiplier
                break
        
        # Apply multiplier and clamp
        adjusted_rank = int(round(base_rank * mult))
        adjusted_rank = max(self.min_rank, adjusted_rank)
        adjusted_rank = min(self.max_rank, adjusted_rank)
        
        return adjusted_rank


# Predefined policies

UNIFORM = Policy(
    name="uniform",
    type_multipliers={".*": 1.0},
    description="Uniform rank allocation across all layers",
)

QKV_HEAVIER = Policy(
    name="qkv_heavier",
    type_multipliers={
        r"(attn|attention)\.(q_proj|query)": 1.25,
        r"(attn|attention)\.(k_proj|key)": 1.10,
        r"(attn|attention)\.(v_proj|value)": 1.25,
        r"(attn|attention)\.(o_proj|output)": 1.00,
        r"mlp\.(up_proj|gate_proj|c_fc)": 0.85,
        r"mlp\.(down_proj|c_proj)": 0.85,
        r".*": 1.0,
    },
    description="QKV-heavier: prioritize attention projections over MLP",
)

MLP_HEAVIER = Policy(
    name="mlp_heavier",
    type_multipliers={
        r"mlp\.(up_proj|gate_proj|c_fc)": 1.25,
        r"mlp\.(down_proj|c_proj)": 1.10,
        r"(attn|attention)\.(q_proj|k_proj|v_proj|o_proj|query|key|value|output)": 0.85,
        r".*": 1.0,
    },
    description="MLP-heavier: prioritize MLP projections over attention",
)


# Registry of available policies
POLICIES = {
    "uniform": UNIFORM,
    "qkv_heavier": QKV_HEAVIER,
    "mlp_heavier": MLP_HEAVIER,
}


def get_policy(name: str) -> Policy:
    """
    Get policy by name.
    
    Args:
        name: Policy name (uniform, qkv_heavier, mlp_heavier)
        
    Returns:
        Policy object
        
    Raises:
        KeyError: If policy name not found
    """
    if name not in POLICIES:
        available = ", ".join(POLICIES.keys())
        raise KeyError(f"Policy '{name}' not found. Available: {available}")
    
    return POLICIES[name]


def list_policies() -> Dict[str, str]:
    """
    List all available policies with descriptions.
    
    Returns:
        Dictionary mapping policy names to descriptions
    """
    return {name: policy.description for name, policy in POLICIES.items()}


def load_policy_from_yaml(path: str) -> Policy:
    """
    Load a custom policy from YAML file.
    
    Expected YAML format:
        name: custom_policy
        multipliers:
          "pattern1": 1.25
          "pattern2": 0.85
        min_rank: 4
        max_rank: 8192
        description: "Custom description"
        
    Args:
        path: Path to YAML file
        
    Returns:
        Policy object
    """
    import yaml
    
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    
    return Policy(
        name=config.get("name", "custom"),
        type_multipliers=config.get("multipliers", {".*": 1.0}),
        min_rank=config.get("min_rank", 4),
        max_rank=config.get("max_rank", 8192),
        description=config.get("description", "Custom policy from YAML"),
    )