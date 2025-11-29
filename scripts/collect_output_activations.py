#!/usr/bin/env python3
"""
Collect output activation statistics (Y^T Y) for all Linear layers.

This script runs forward passes on a dataset and accumulates output covariance
matrices for each Linear layer without storing full activations.

For GPT-2 specifically:
- MLP layers: c_fc, c_proj
- Attention: c_attn (splits into Q/K/V), c_proj

Usage:
    python scripts/collect_output_activations.py --model gpt2 --dataset wikitext \
        --split train --samples 20000 --max-length 512 --out stats/gpt2_yty.pt
"""

import argparse
import os
import sys
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import defaultdict
from tqdm import tqdm

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.urank.instrumentation import iter_candidate_linear_modules


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="Collect output activation statistics")
    parser.add_argument("--model", type=str, required=True, help="Model name or path")
    parser.add_argument("--dataset", type=str, default="wikitext", help="Dataset name")
    parser.add_argument("--config", type=str, default="wikitext-2-raw-v1", help="Dataset config")
    parser.add_argument("--split", type=str, default="train", help="Dataset split")
    parser.add_argument("--samples", type=int, default=1000000, help="Max number of tokens to process")
    parser.add_argument("--max-length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for processing")
    parser.add_argument("--use-bf16", action="store_true", help="Use bfloat16 for faster inference")
    parser.add_argument("--out", type=str, required=True, help="Output file path")
    args = parser.parse_args()

    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load model and tokenizer with optimizations
    print(f"Loading model: {args.model}")
    dtype = torch.bfloat16 if args.use_bf16 else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Storage: name -> Y^T Y
    yty = defaultdict(lambda: None)

    def add_cov(name, y):
        """
        Accumulate Y^T Y for output activations.
        
        Args:
            name: Layer name
            y: Output tensor (B, T, D) or (B*T, D)
        """
        if y.dim() == 3:
            B, T, D = y.shape
            y_flat = y.reshape(-1, D)  # (B*T, D)
        elif y.dim() == 2:
            y_flat = y
            D = y.shape[1]
        else:
            return  # Skip unexpected shapes
        
        C = y_flat.T @ y_flat  # (D, D)
        
        if yty[name] is None:
            yty[name] = C.cpu()
        else:
            yty[name] += C.cpu()

    # Register hooks for all candidate layers (architecture-aware)
    hooks = []

    for name, module in iter_candidate_linear_modules(model):
        def make_hook(name_):
            def hook(_, __, output):
                add_cov(name_, output)
            return hook
        hooks.append(module.register_forward_hook(make_hook(name)))

    print(f"Registered hooks on {len(hooks)} layers (architecture: {model.__class__.__name__})")

    # Load dataset
    print(f"Loading dataset: {args.dataset} ({args.config}) - {args.split}")
    ds = load_dataset(args.dataset, args.config, split=args.split)
    
    # Process samples with batching
    print(f"Collecting output statistics from ~{args.samples} tokens...")
    print(f"Using batch size: {args.batch_size}")
    total_tokens = 0
    batch_texts = []

    for ex in tqdm(ds, desc="Processing batches"):
        if total_tokens >= args.samples:
            break
        
        text = ex["text"]
        if not text or not text.strip():
            continue

        batch_texts.append(text)
        
        # Process when batch is full
        if len(batch_texts) >= args.batch_size:
            # Tokenize batch
            enc = tokenizer(
                batch_texts,
                return_tensors="pt",
                max_length=args.max_length,
                truncation=True,
                padding=True,
            ).to(device)

            total_tokens += enc["input_ids"].numel()
            
            # Forward pass (outputs captured by hooks)
            _ = model(**enc, use_cache=False)
            
            # Clear batch and free memory
            batch_texts = []
            if device == "cuda":
                torch.cuda.empty_cache()
    
    # Process remaining texts in incomplete batch
    if batch_texts and total_tokens < args.samples:
        enc = tokenizer(
            batch_texts,
            return_tensors="pt",
            max_length=args.max_length,
            truncation=True,
            padding=True,
        ).to(device)
        
        total_tokens += enc["input_ids"].numel()
        _ = model(**enc, use_cache=False)

    # Remove hooks
    for h in hooks:
        h.remove()

    # Save statistics
    print(f"Saving Y^T Y stats for {len(yty)} modules to {args.out}")
    torch.save(dict(yty), args.out)
    print(f"Processed {total_tokens} total tokens")
    print("Done!")


if __name__ == "__main__":
    main()