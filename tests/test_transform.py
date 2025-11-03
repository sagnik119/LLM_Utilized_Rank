"""
Tests for weight transformation and factorization.
"""

import torch
import torch.nn as nn
from urank.transform import transform_weight_ps_pt, factorize_rank, should_factorize


def test_transform_shapes():
    """Test that weight transformation preserves shapes."""
    # Create a linear layer
    lin = nn.Linear(128, 256, bias=False)
    
    # Create identity projectors
    P_S = torch.eye(128)
    P_T = torch.eye(256)
    
    # Transform weight
    W_prime = transform_weight_ps_pt(lin, P_S, P_T)
    
    # Check shape is preserved
    assert W_prime.shape == lin.weight.shape
    assert W_prime.shape == (256, 128)
    
    # With identity projectors, result should be same as original
    assert torch.allclose(W_prime, lin.weight.data, atol=1e-5)


def test_factorize_rank_shapes():
    """Test that factorization produces correct shapes."""
    # Create a weight matrix
    W = torch.randn(256, 128)
    
    # Factorize at rank 64
    r = 64
    left, right = factorize_rank(W, r)
    
    # Check shapes
    assert left.shape == (256, r)
    assert right.shape == (r, 128)
    
    # Check reconstruction
    W_approx = left @ right
    assert W_approx.shape == W.shape


def test_factorize_rank_approximation():
    """Test that factorization approximates original matrix."""
    # Create a low-rank matrix
    U = torch.randn(100, 20)
    V = torch.randn(20, 50)
    W = U @ V  # rank-20 matrix
    
    # Factorize at rank 20
    left, right = factorize_rank(W, 20)
    W_approx = left @ right
    
    # Should be very close since W is already rank 20
    assert torch.allclose(W, W_approx, atol=1e-3)


def test_should_factorize():
    """Test factorization decision logic."""
    # Case where factorization saves parameters
    assert should_factorize(1024, 4096, 512) == True  # ~25% savings
    
    # Case where factorization doesn't save parameters
    assert should_factorize(256, 256, 200) == False  # No savings
    
    # Edge case: rank equals min dimension
    assert should_factorize(100, 200, 100) == False


def test_transform_with_projectors():
    """Test transformation with non-identity projectors."""
    lin = nn.Linear(64, 128, bias=False)
    
    # Create random projectors (using first k eigenvectors of random matrices)
    M_S = torch.randn(64, 64)
    M_S = M_S @ M_S.T  # Make symmetric
    evals_S, evecs_S = torch.linalg.eigh(M_S)
    V_S = evecs_S[:, -32:]  # Top 32
    P_S = V_S @ V_S.T
    
    M_T = torch.randn(128, 128)
    M_T = M_T @ M_T.T
    evals_T, evecs_T = torch.linalg.eigh(M_T)
    V_T = evecs_T[:, -64:]  # Top 64
    P_T = V_T @ V_T.T
    
    # Transform
    W_prime = transform_weight_ps_pt(lin, P_S, P_T)
    
    # Check shape preserved
    assert W_prime.shape == lin.weight.shape
    
    # Check that transformation is different from original
    assert not torch.allclose(W_prime, lin.weight.data)


if __name__ == "__main__":
    test_transform_shapes()
    test_factorize_rank_shapes()
    test_factorize_rank_approximation()
    test_should_factorize()
    test_transform_with_projectors()
    print("All transform tests passed!")