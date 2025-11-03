"""
Evaluation utilities for model quality assessment.

This module provides utilities for:
- Perplexity evaluation on language modeling tasks
- Zero-shot and few-shot task evaluation
"""

from .perplexity import PerplexityEvaluator

__all__ = ["PerplexityEvaluator"]