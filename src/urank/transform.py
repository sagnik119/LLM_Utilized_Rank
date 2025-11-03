"""
Weight transformation and factorization utilities.

This module provides utilities for:
- Transforming weights via projectors: W' = P_T @ W @ P_S
- Factorizing transformed weights into low-rank decompositions
"""

from typing import Tuple
import torch
import torch.nn as nn


def transform_weight_ps_pt(
    linear: nn.Linear, 
    P_S: torch.Tensor, 
    P_T: torch.Tensor
) -> torch.Tensor:
    """
    Transform a Linear layer's weight via input/output projectors.
    
    Computes W' = P_T @ W @ P_S where:
    - W: Original weight matrix (m, d)
    - P_S: Input subspace projector (d, d)
    - P_T: Output subspace projector (m, m)
    - W': Transformed weight (m, d)
    
    This reveals the "utilized rank" - the effective rank of W when
    restricted to the data-driven input and output subspaces.
    
    Args:
        linear: Linear layer with weight (out_features, in_features)
        P_S: Input projector (in_features, in_features)
        P_T: Output projector (out_features, out_features)
        
    Returns:
        Transformed weight matrix W' of same shape as original
        
    Example:
        >>> linear = nn.Linear(128, 256, bias=False)
        >>> P_S = projector_from_vecs(V_S)  # (128, 128)
        >>> P_T = projector_from_vecs(V_T)  # (256, 256)
        >>> W_prime = transform_weight_ps_pt(linear, P_S, P_T)
        >>> print(f"Utilized rank: {torch.linalg.matrix_rank(W_prime)}")
    """
    W = linear.weight.data  # (m, d)
    W_prime = P_T.to(W) @ W @ P_S.to(W)
    return W_prime


def factorize_rank(W_prime: torch.Tensor, rank: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Factorize a weight matrix at specified rank using SVD.
    
    Decomposes W' (m, d) ≈ left @ right where:
    - left: (m, r) - output projection with singular values
    - right: (r, d) - input projection
    
    The factorization minimizes ||W' - left @ right||_F in the Frobenius norm.
    
    Args:
        W_prime: Weight matrix to factorize (m, d)
        rank: Target rank r for decomposition
        
    Returns:
        Tuple of (left, right) where:
            - left: (m, r) matrix (incorporates singular values)
            - right: (r, d) matrix
            
    Example:
        >>> W_prime = transform_weight_ps_pt(linear, P_S, P_T)
        >>> r = 64  # target rank
        >>> left, right = factorize_rank(W_prime, r)
        >>> # Reconstruct: W_approx = left @ right
        >>> print(f"Shapes: {left.shape} @ {right.shape}")
        >>> print(f"Compression: {(left.numel() + right.numel()) / W_prime.numel():.2%}")
    """
    # SVD: W' = U @ S @ V^H
    U, S, Vh = torch.linalg.svd(W_prime, full_matrices=False)
    
    # Keep top-r singular vectors/values
    U_r = U[:, :rank]
    S_r = S[:rank]
    Vh_r = Vh[:rank, :]
    
    # Incorporate singular values into left factor
    left = U_r * S_r  # (m, r) - broadcasting multiplies each column by S_r[i]
    right = Vh_r      # (r, d)
    
    return left, right


def should_factorize(m: int, d: int, r: int) -> bool:
    """
    Determine if factorization reduces parameter count.
    
    Factorization is beneficial when:
    r * (m + d) < m * d
    
    Args:
        m: Output dimension
        d: Input dimension
        r: Rank
        
    Returns:
        True if factorization saves parameters
        
    Example:
        >>> should_factorize(1024, 4096, 512)  # True - saves ~25% params
        >>> should_factorize(256, 256, 200)     # False - no savings
    """
    return r * (m + d) < m * d