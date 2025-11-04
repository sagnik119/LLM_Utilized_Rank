"""
Subspace extraction and projector construction from covariance matrices.

This module provides utilities for:
- Computing top-k eigenvectors from X^T X by energy threshold
- Building orthogonal projectors from eigenvector bases
"""

from typing import Tuple
import torch


def topk_from_xtx(xtx: torch.Tensor, energy: float) -> Tuple[torch.Tensor, int]:
    """
    Compute top-k eigenvectors of X^T X by energy threshold.
    
    Returns eigenvectors V_k (d x k) ordered by descending eigenvalue,
    such that the cumulative energy (sum of top-k eigenvalues / total) >= energy.
    
    Uses device-agnostic computation (stays on CUDA if input is on CUDA) with
    numerical stabilization for robustness.
    
    Args:
        xtx: Symmetric PSD matrix X^T X of shape (d, d)
        energy: Energy threshold in [0, 1], e.g., 0.99 for 99%
        
    Returns:
        Tuple of (V_k, k) where:
            - V_k: Eigenvectors matrix (d, k) with orthonormal columns
            - k: Number of eigenvectors selected
            
    Example:
        >>> X = torch.randn(1000, 128)
        >>> xtx = X.T @ X
        >>> V_k, k = topk_from_xtx(xtx, 0.99)
        >>> print(f"Preserved 99% energy with {k} dimensions")
    """
    device = xtx.device
    
    # Symmetrize to ensure numerical PSD property
    xtx = 0.5 * (xtx + xtx.transpose(-1, -2))
    
    # Use float64 for eigendecomposition stability
    xtx = xtx.to(dtype=torch.float64)
    
    # Add tiny diagonal jitter for numerical PSD (helps with MKL errors)
    eps = xtx.diag().mean().abs() * 1e-12 if xtx.numel() > 0 else 1e-12
    xtx = xtx + eps * torch.eye(xtx.shape[-1], device=device, dtype=xtx.dtype)
    
    # Use eigh for symmetric matrices (stays on same device - CUDA preferred)
    evals, evecs = torch.linalg.eigh(xtx)
    
    # Clamp to handle numerical errors
    evals = torch.clamp(evals, min=0)
    
    # Sort in descending order
    idx = torch.argsort(evals, descending=True)
    evals = evals[idx]
    evecs = evecs[:, idx]
    
    # Compute cumulative energy ratio
    cumsum = torch.cumsum(evals, dim=0)
    total = cumsum[-1].clamp(min=1e-12)
    ratio = cumsum / total
    
    # Find k such that cumulative energy >= threshold
    k = int((ratio < energy).sum().item()) + 1
    k = min(k, len(evals))  # Ensure k doesn't exceed dimension
    
    V_k = evecs[:, :k].to(torch.float32)  # Return in float32
    return V_k, k


def projector_from_vecs(V_k: torch.Tensor) -> torch.Tensor:
    """
    Build an orthogonal projector from a basis matrix.
    
    Constructs P = V V^T, which is the orthogonal projection onto the
    subspace spanned by the columns of V. Assumes V has orthonormal columns
    (as returned by eigh).
    
    Args:
        V_k: Basis matrix (d, k) with orthonormal columns
        
    Returns:
        Projector matrix P (d, d) satisfying P^2 = P and P^T = P
        
    Example:
        >>> V_k, k = topk_from_xtx(xtx, 0.99)
        >>> P = projector_from_vecs(V_k)
        >>> # Check idempotence: P @ P ≈ P
        >>> assert torch.allclose(P @ P, P, atol=1e-5)
    """
    return V_k @ V_k.T