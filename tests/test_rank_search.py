"""
Tests for rank search functionality.
"""

import torch
import torch.nn as nn
from urank.rank_search import RankSearcher, RankResult


class DummyModel(nn.Module):
    """Simple model for testing."""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(16, 8, bias=False)

    def forward(self, x):
        return self.linear(x)


def test_rank_result_creation():
    """Test RankResult dataclass."""
    result = RankResult(k_S=10, k_T=8, r=8, e_S=0.99, e_T=0.99)
    
    assert result.k_S == 10
    assert result.k_T == 8
    assert result.r == 8
    assert result.e_S == 0.99
    assert result.e_T == 0.99


def test_metric_ok():
    """Test validation metric checking."""
    model = DummyModel()
    X = torch.randn(500, 16)
    xtx = X.T @ X
    
    def mock_eval_fn(_):
        return 10.0
    
    searcher = RankSearcher(
        model,
        {"linear": xtx},
        mock_eval_fn,
        epsilon_percent=0.1
    )
    
    # Test metric within epsilon
    assert searcher._metric_ok(10.0, 10.01) == True
    
    # Test metric exceeds epsilon
    assert searcher._metric_ok(10.0, 10.5) == False


def test_search_runs():
    """Test that search completes without errors."""
    model = DummyModel()
    X = torch.randn(500, 16)
    xtx = X.T @ X
    
    def mock_eval_fn(_):
        # Mock constant metric
        return 10.0
    
    searcher = RankSearcher(
        model,
        {"linear": xtx},
        mock_eval_fn,
        epsilon_percent=0.1
    )
    
    # Run search
    result = searcher.search_layer("linear", model.linear, max_iters=3)
    
    # Check result structure
    assert isinstance(result, RankResult)
    assert result.r >= 1
    assert result.r <= min(result.k_S, result.k_T)


def test_search_missing_layer():
    """Test search behavior when layer not in statistics."""
    model = DummyModel()
    
    def mock_eval_fn(_):
        return 10.0
    
    searcher = RankSearcher(
        model,
        {},  # Empty xtx_map
        mock_eval_fn,
        epsilon_percent=0.1
    )
    
    # Should return full rank
    result = searcher.search_layer("linear", model.linear)
    
    assert result.k_S == 16  # in_features
    assert result.k_T == 8   # out_features
    assert result.r == 8     # min(in, out)


if __name__ == "__main__":
    test_rank_result_creation()
    test_metric_ok()
    test_search_runs()
    test_search_missing_layer()
    print("All rank search tests passed!")