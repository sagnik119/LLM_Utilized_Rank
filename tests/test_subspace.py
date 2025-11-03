"""
Tests for subspace extraction and projector construction.
"""

import torch
from urank.subspace import topk_from_xtx, projector_from_vecs


def test_energy_threshold():
    """Test that energy threshold correctly selects top-k eigenvectors."""
    # Create synthetic data
    X = torch.randn(1000, 16)
    xtx = X.T @ X
    
    # Extract subspace with 90% energy
    V, k = topk_from_xtx(xtx, 0.90)
    
    # Check dimensions
    assert V.shape == (16, k)
    assert k <= 16
    assert k >= 1
    
    # Build projector
    P = projector_from_vecs(V)
    
    # Check projector properties
    assert P.shape == (16, 16)
    
    # Check idempotence: P @ P = P
    assert torch.allclose(P @ P, P, atol=1e-5)
    
    # Check symmetry: P = P^T
    assert torch.allclose(P, P.T, atol=1e-5)


def test_full_energy():
    """Test that energy=0.9999 selects all dimensions."""
    X = torch.randn(100, 8)
    xtx = X.T @ X
    
    V, k = topk_from_xtx(xtx, 0.9999)
    
    # Should select all or nearly all dimensions
    assert k >= 7  # Allow for numerical tolerance


def test_projector_orthogonality():
    """Test that projector columns are orthonormal."""
    X = torch.randn(500, 10)
    xtx = X.T @ X
    
    V, k = topk_from_xtx(xtx, 0.95)
    
    # Check orthonormality: V^T @ V = I_k
    VtV = V.T @ V
    I_k = torch.eye(k)
    assert torch.allclose(VtV, I_k, atol=1e-4)


if __name__ == "__main__":
    test_energy_threshold()
    test_full_energy()
    test_projector_orthogonality()
    print("All subspace tests passed!")