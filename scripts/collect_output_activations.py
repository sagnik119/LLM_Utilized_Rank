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
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import defaultdict
from tqdm import tqdm


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="Collect output activation statistics")
    parser.add_argument("--model", type=str, required=True, help="Model name or path")
    parser.add_argument("--dataset", type=str, default="wikitext", help="Dataset name")
    parser.add_argument("--config", type=str, default="wikitext-2-raw-v1", help="Dataset config")
    parser.add_argument("--split", type=str, default="train", help="Dataset split")
    parser.add_argument("--samples", type=int, default=20000, help="Max token count to process")
    parser.add_argument("--max-length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--out", type=str, required=True, help="Output file path")
    args = parser.parse_args()

    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load model and tokenizer
    print(f"Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(args.model).to(device)
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

    # Register hooks for all relevant layers
    hooks = []

    for name, module in model.named_modules():
        # MLP projections
        if name.endswith(".mlp.c_fc") or name.endswith(".mlp.c_proj"):
            def make_hook(name_):
                def hook(_, __, output):
                    add_cov(name_, output)
                return hook
            hooks.append(module.register_forward_hook(make_hook(name)))

        # Attention combined QKV projection
        if name.endswith(".attn.c_attn"):
            def make_hook(name_):
                def hook(_, __, output):
                    # output: (B, T, 3*d_model) for GPT-2
                    # Split into Q, K, V
                    if output.dim() == 3:
                        B, T, threeD = output.shape
                        d = threeD // 3
                        y_q = output[..., :d]
                        y_k = output[..., d:2*d]
                        y_v = output[..., 2*d:]

                        add_cov(name_ + ".q", y_q)
                        add_cov(name_ + ".k", y_k)
                        add_cov(name_ + ".v", y_v)
                    else:
                        # Fallback: treat as single output
                        add_cov(name_, output)
                return hook
            hooks.append(module.register_forward_hook(make_hook(name)))

        # Attention output projection
        if name.endswith(".attn.c_proj"):
            def make_hook(name_):
                def hook(_, __, output):
                    add_cov(name_, output)
                return hook
            hooks.append(module.register_forward_hook(make_hook(name)))

    print(f"Registered hooks on {len(hooks)} layers")

    # Load dataset
    print(f"Loading dataset: {args.dataset} ({args.config}) - {args.split}")
    ds = load_dataset(args.dataset, args.config, split=args.split)
    
    # Process samples
    print(f"Collecting output statistics from ~{args.samples} tokens...")
    total_tokens = 0

    for ex in tqdm(ds):
        if total_tokens >= args.samples:
            break
        
        text = ex["text"]
        if not text or not text.strip():
            continue

        # Tokenize
        enc = tokenizer(
            text,
            return_tensors="pt",
            max_length=args.max_length,
            truncation=True,
        ).to(device)

        total_tokens += enc["input_ids"].numel()
        
        # Forward pass (outputs captured by hooks)
        _ = model(**enc)

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