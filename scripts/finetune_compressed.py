#!/usr/bin/env python3
"""
Fine-tune a compressed model with factorized layers using PEFT.

This script supports both GPT-2 and LLaMA compressed models, automatically
detecting the architecture and applying appropriate fine-tuning strategies.

Usage:
    # GPT-2
    python scripts/finetune_compressed.py --preset lora \
        --model ckpts/gpt2_compressed --data alpaca_en --out outputs/lora_run
    
    # LLaMA-2
    python scripts/finetune_compressed.py --preset lora \
        --model ckpts/llama2_compressed --data alpaca_en --out outputs/lora_run
"""

import argparse
import os
import sys
import shutil
import torch
from transformers import AutoTokenizer, AutoConfig, TrainingArguments, Trainer, DataCollatorForLanguageModeling, DataCollatorForSeq2Seq
from datasets import load_dataset

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.urank.modeling_gpt2_compressed import GPT2CompressedLMHeadModel
from src.urank.modeling_llama_compressed import LlamaCompressedForCausalLM
from src.urank.lora_factorized import convert_to_lora_factorized, count_lora_parameters


def detect_architecture(model_path):
    """
    Detect model architecture from config.
    
    Returns:
        str: 'gpt2' or 'llama'
    """
    try:
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        model_type = config.model_type.lower()
        
        if model_type.startswith("gpt2"):
            return "gpt2"
        elif model_type.startswith("llama") or model_type == "llama":
            return "llama"
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")
    except Exception as e:
        raise ValueError(f"Failed to detect architecture: {e}")


def load_compressed_model(model_path, architecture):
    """Load the appropriate compressed model class based on architecture."""
    if architecture == "gpt2":
        return GPT2CompressedLMHeadModel.from_pretrained(model_path)
    elif architecture == "llama":
        return LlamaCompressedForCausalLM.from_pretrained(model_path)
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")


def load_dataset_from_name(dataset_name: str, tokenizer, split: str = "train", max_length: int = 512):
    """Load and tokenize a dataset."""
    # Common dataset mappings - use web download for Alpaca to avoid JSON errors
    DATASET_CONFIGS = {
        "alpaca_en": ("tatsu-lab/alpaca", None, "train"),
        "alpaca_gpt4": ("vicgalle/alpaca-gpt4", None, "train"),
        "wikitext2": ("wikitext", "wikitext-2-raw-v1", "train"),
        "redpajama": ("togethercomputer/RedPajama-Data-V2", "sample", "train"),
        "slimpajama": ("cadenpark/slimpajama-627m", None, "train"),
    }
    
    if dataset_name in DATASET_CONFIGS:
        config = DATASET_CONFIGS[dataset_name]
        if config[1] is not None:
            ds = load_dataset(config[0], config[1], split=config[2])
        else:
            ds = load_dataset(config[0], split=config[2])
    else:
        # Try loading directly
        if dataset_name == "c4":
            ds = load_dataset("c4", "en.noblocklist", split=split)
        elif dataset_name == "togethercomputer/RedPajama-Data-V2":
            # RedPajama V2 requires specific config
            ds = load_dataset("togethercomputer/RedPajama-Data-V2", "sample", split=split)
        elif dataset_name == "cadenpark/slimpajama-627m":
            # SlimPajama dataset
            ds = load_dataset("cadenpark/slimpajama-627m", split=split)
        else:
            ds = load_dataset(dataset_name, split=split)
    
    # Tokenize
    def tokenize_function(examples):
        # Handle instruction datasets
        if "instruction" in examples:
            texts = []
            for i in range(len(examples["instruction"])):
                text = f"### Instruction: {examples['instruction'][i]}\n"
                if "input" in examples and examples["input"][i]:
                    text += f"### Input: {examples['input'][i]}\n"
                text += f"### Response: {examples['output'][i]}"
                texts.append(text)
            return tokenizer(texts, truncation=True, max_length=max_length)
        # Handle RedPajama V2 format
        elif "raw_content" in examples:
            return tokenizer(examples["raw_content"], truncation=True, max_length=max_length)
        # Handle standard text datasets
        elif "text" in examples:
            return tokenizer(examples["text"], truncation=True, max_length=max_length)
        else:
            raise ValueError(f"Unknown dataset format: {examples.keys()}")
    
    tokenized = ds.map(tokenize_function, batched=True, remove_columns=ds.column_names)
    return tokenized


# Removed find_factorized_submodules - no longer needed with custom LoRA


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune compressed model with factorized layers"
    )
    parser.add_argument("--preset", choices=["lora", "full", "freeze"], required=True)
    parser.add_argument("--model", required=True, help="Path to compressed model")
    parser.add_argument("--data", required=True, help="Dataset name")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs (ignored if --max-train-tokens is set)")
    parser.add_argument("--max-train-tokens", type=int, default=None, help="Train for exactly this many tokens (overrides --epochs)")
    parser.add_argument("--seq-length", type=int, default=512, help="Sequence length for tokenization")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--use-bf16", action="store_true", default=True,
                        help="Use BF16 for training (default: True for A100/H100)")
    args = parser.parse_args()
    
    # Detect architecture
    print(f"Detecting architecture for: {args.model}")
    architecture = detect_architecture(args.model)
    print(f"  Detected: {architecture}")
    
    print(f"Loading compressed model from: {args.model}")
    
    # Load model with appropriate class
    model = load_compressed_model(args.model, architecture)
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")  # PATCH 1: Move to CUDA immediately
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    # Set pad token and padding side (LLaMA-specific)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id
    
    # For LLaMA, set padding side to right for causal masking
    if architecture == "llama":
        tokenizer.padding_side = "right"
    
    print(f"Model loaded with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Apply fine-tuning based on preset
    if args.preset == "lora":
        print(f"\nApplying Custom LoRA to FactorizedLinear (r={args.lora_r}, alpha={args.lora_alpha})")
        
        # Convert all FactorizedLinear modules to LoRAFactorizedLinear
        num_converted = convert_to_lora_factorized(
            model,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            freeze_base=True
        )
        
        if num_converted == 0:
            raise RuntimeError(
                "No FactorizedLinear modules found. Compression was not applied correctly. "
                "Make sure you're using a properly compressed checkpoint."
            )
        
        print(f"\n✓ Converted {num_converted} FactorizedLinear modules to LoRAFactorizedLinear")
        
        # Print parameter counts
        param_counts = count_lora_parameters(model)
        print(f"\nParameter counts:")
        print(f"  Base (frozen):     {param_counts['base']:,}")
        print(f"  LoRA (trainable):  {param_counts['lora']:,}")
        print(f"  Total:             {param_counts['total']:,}")
        print(f"  Trainable ratio:   {param_counts['lora']/param_counts['total']*100:.2f}%")
        
        model.train()
    
    elif args.preset == "full":
        print("\nFull fine-tuning (all parameters trainable)")
        # All params already trainable
    
    elif args.preset == "freeze":
        print("\nFreeze fine-tuning (only layer norms and head)")
        # Freeze all except specified
        for name, param in model.named_parameters():
            if any(x in name.lower() for x in ["ln", "layernorm", "lm_head"]):
                param.requires_grad = True
            else:
                param.requires_grad = False
        
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Trainable parameters: {trainable:,}")
    
    # Load dataset
    print(f"\nLoading dataset: {args.data}")
    train_dataset = load_dataset_from_name(args.data, tokenizer, max_length=args.seq_length)
    
    # Compute training steps based on token budget
    if args.max_train_tokens is not None:
        # Token-budget training mode
        tokens_per_batch = args.batch_size * args.grad_accum * args.seq_length
        max_steps = args.max_train_tokens // tokens_per_batch
        num_epochs = 1  # Single pass, stopping at max_steps
        
        print(f"\n{'='*60}")
        print(f"TOKEN-BUDGET TRAINING MODE")
        print(f"{'='*60}")
        print(f"Target tokens: {args.max_train_tokens:,}")
        print(f"Tokens per batch: {tokens_per_batch:,}")
        print(f"Total steps: {max_steps:,}")
        print(f"Estimated time (A100): ~{max_steps * 0.5 / 60:.1f} minutes")
        print(f"{'='*60}\n")
    else:
        # Epoch-based training mode (original behavior)
        max_steps = -1
        num_epochs = args.epochs
        print(f"\nEpoch-based training: {num_epochs} epochs")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=num_epochs,
        max_steps=max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=20,
        save_steps=1000,
        save_total_limit=2,
        bf16=args.use_bf16,  # Use BF16 for A100/H100 GPUs
        fp16=False,
        gradient_checkpointing=False,
        report_to="none",
    )
    
    # Data collator (architecture-specific)
    if architecture == "llama":
        # LLaMA requires proper attention masking
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model,
            padding=True,
            pad_to_multiple_of=8
        )
    else:
        # GPT-2 standard collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    
    # Train
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60 + "\n")
    
    trainer.train()
    
    # Save
    print(f"\nSaving model to: {args.out}")
    trainer.save_model(args.out)
    tokenizer.save_pretrained(args.out)
    
    # Copy custom architecture files to make the checkpoint self-contained
    print("\nCopying custom architecture files...")
    
    # Determine which files to copy based on architecture
    if architecture == "gpt2":
        architecture_files = [
            "configuration_gpt2_compressed.py",
            "modeling_gpt2_compressed.py",
            "__init__.py"
        ]
    elif architecture == "llama":
        architecture_files = [
            "configuration_llama_compressed.py",
            "modeling_llama_compressed.py",
            "__init__.py"
        ]
    else:
        architecture_files = []
    
    for fname in architecture_files:
        src_path = os.path.join(args.model, fname)
        if os.path.exists(src_path):
            dst_path = os.path.join(args.out, fname)
            shutil.copy2(src_path, dst_path)
            print(f"  ✓ Copied {fname}")
        else:
            print(f"  ⚠ {fname} not found in {args.model}")
    
    print("\n" + "="*60)
    print("Training completed successfully!")
    print(f"Model saved to: {args.out}")
    print("Output is now a self-contained HuggingFace repository")
    print("="*60)


if __name__ == "__main__":
    main()