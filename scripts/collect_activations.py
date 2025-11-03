#!/usr/bin/env python3
"""
Collect activation statistics (X^T X) for all Linear layers.

This script runs forward passes on a dataset and accumulates input covariance
matrices for each Linear layer without storing full activations.

Usage:
    python scripts/collect_activations.py --model gpt2 --dataset wikitext \\
        --split train --samples 50000 --seq-len 512 --save stats/gpt2_xtx.pt
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

from urank.activation_cache import ActivationStatsCollector


def main():
    parser = argparse.ArgumentParser(description="Collect activation statistics")
    parser.add_argument("--model", type=str, required=True, help="Model name or path")
    parser.add_argument("--dataset", type=str, default="wikitext", help="Dataset name")
    parser.add_argument("--config", type=str, default="wikitext-2-raw-v1", help="Dataset config")
    parser.add_argument("--split", type=str, default="train", help="Dataset split")
    parser.add_argument("--samples", type=int, default=50000, help="Max samples to process")
    parser.add_argument("--seq-len", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--save", type=str, required=True, help="Output file path")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    args = parser.parse_args()
    
    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load model and tokenizer
    print(f"Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(args.model).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Start collecting statistics
    collector = ActivationStatsCollector(model).start()
    
    # Load dataset
    print(f"Loading dataset: {args.dataset} ({args.config}) - {args.split}")
    ds = load_dataset(args.dataset, args.config)[args.split]
    
    # Process samples
    print(f"Collecting statistics from {args.samples} samples...")
    seen = 0
    
    with torch.no_grad():
        for ex in tqdm(ds, total=min(args.samples, len(ds))):
            if seen >= args.samples:
                break
            
            txt = ex["text"]
            if not txt or not txt.strip():
                continue
            
            # Tokenize
            ids = tokenizer(
                txt,
                return_tensors="pt",
                truncation=True,
                max_length=args.seq_len
            )["input_ids"].to(device)
            
            # Forward pass (activations captured by hooks)
            model(ids)
            
            seen += ids.numel()
    
    # Stop collection and get statistics
    stats = collector.stop()
    
    # Save as dictionary of tensors
    print(f"Saving statistics for {len(stats)} layers to {args.save}")
    torch.save({k: v.xtx for k, v in stats.items()}, args.save)
    
    print("Done!")


if __name__ == "__main__":
    main()