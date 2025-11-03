"""
Perplexity evaluation for language models.

This module provides utilities for computing perplexity on standard
language modeling datasets like WikiText.
"""

import math
from typing import Optional
import torch
from datasets import load_dataset
from transformers import PreTrainedModel, PreTrainedTokenizer


class PerplexityEvaluator:
    """
    Evaluates language model perplexity on a dataset.
    
    Computes perplexity as exp(average_negative_log_likelihood) across
    a specified number of samples from a dataset.
    
    Args:
        model: HuggingFace model (should support labels in forward pass)
        tokenizer: HuggingFace tokenizer
        dataset_name: Name of dataset to load (default: "wikitext")
        config: Dataset config (default: "wikitext-2-raw-v1")
        split: Dataset split to use (default: "validation")
        samples: Maximum number of samples to evaluate (default: 2048)
        seq_len: Maximum sequence length for tokenization (default: 512)
        
    Example:
        >>> from transformers import AutoModelForCausalLM, AutoTokenizer
        >>> model = AutoModelForCausalLM.from_pretrained("gpt2")
        >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
        >>> evaluator = PerplexityEvaluator(model, tokenizer)
        >>> ppl = evaluator()
        >>> print(f"Perplexity: {ppl:.2f}")
    """
    
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        dataset_name: str = "wikitext",
        config: str = "wikitext-2-raw-v1",
        split: str = "validation",
        samples: int = 2048,
        seq_len: int = 512,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.dataset_name = dataset_name
        self.config = config
        self.split = split
        self.samples = samples
        self.seq_len = seq_len
        self._dataset = None
    
    @property
    def dataset(self):
        """Lazy-load dataset on first access."""
        if self._dataset is None:
            self._dataset = load_dataset(
                self.dataset_name, 
                self.config
            )[self.split]
        return self._dataset
    
    @torch.no_grad()
    def __call__(self, model: Optional[PreTrainedModel] = None) -> float:
        """
        Compute perplexity on the dataset.
        
        Args:
            model: Optional model to evaluate (default: self.model)
            
        Returns:
            Perplexity value (float)
        """
        model = model or self.model
        model.eval()
        
        total_nll = 0.0
        total_tokens = 0
        num_samples = 0
        
        for i in range(min(self.samples, len(self.dataset))):
            txt = self.dataset[i]["text"]
            
            # Skip empty samples
            if not txt or not txt.strip():
                continue
            
            # Tokenize
            ids = self.tokenizer(
                txt,
                return_tensors="pt",
                truncation=True,
                max_length=self.seq_len
            )["input_ids"].to(model.device)
            
            # Skip very short sequences
            if ids.shape[1] < 2:
                continue
            
            # Forward pass with labels (shifted internally by model)
            labels = ids.clone()
            outputs = model(input_ids=ids, labels=labels)
            
            # Accumulate negative log likelihood
            total_nll += outputs.loss.item() * ids.numel()
            total_tokens += ids.numel()
            num_samples += 1
        
        if total_tokens == 0:
            return float('inf')
        
        # Perplexity = exp(average NLL)
        avg_nll = total_nll / total_tokens
        ppl = math.exp(avg_nll)
        
        return ppl
    
    def evaluate_multiple(
        self,
        models: dict,
        verbose: bool = True
    ) -> dict:
        """
        Evaluate multiple models and return results.
        
        Args:
            models: Dictionary mapping model names to model objects
            verbose: Whether to print results (default: True)
            
        Returns:
            Dictionary mapping model names to perplexity values
        """
        results = {}
        
        for name, model in models.items():
            ppl = self(model)
            results[name] = ppl
            if verbose:
                print(f"{name}: {ppl:.2f}")
        
        return results