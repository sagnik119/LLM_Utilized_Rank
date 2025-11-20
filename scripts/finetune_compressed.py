#!/usr/bin/env python3
"""
Fine-tune a compressed model with factorized layers using PEFT.

This script uses the custom GPT2CompressedLMHeadModel to properly load
factorized weights, then applies LoRA/Full/Freeze fine-tuning.

Usage:
    python scripts/finetune_compressed.py --preset lora \
        --model ckpts/gpt2_compressed --data alpaca_en --out outputs/lora_run
"""

import argparse
import os
import sys
import shutil
import torch
from transformers import AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, TaskType

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.urank.modeling_gpt2_compressed import GPT2CompressedLMHeadModel


def load_dataset_from_name(dataset_name: str, tokenizer, split: str = "train"):
    """Load and tokenize a dataset."""
    # Common dataset mappings - use web download for Alpaca to avoid JSON errors
    DATASET_CONFIGS = {
        "alpaca_en": ("tatsu-lab/alpaca", None, "train"),
        "alpaca_gpt4": ("vicgalle/alpaca-gpt4", None, "train"),
        "wikitext2": ("wikitext", "wikitext-2-raw-v1", "train"),
        "redpajama": ("togethercomputer/RedPajama-Data-V2", "sample", "train"),
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
            return tokenizer(texts, truncation=True, max_length=512)
        # Handle RedPajama V2 format
        elif "raw_content" in examples:
            return tokenizer(examples["raw_content"], truncation=True, max_length=512)
        # Handle standard text datasets
        elif "text" in examples:
            return tokenizer(examples["text"], truncation=True, max_length=512)
        else:
            raise ValueError(f"Unknown dataset format: {examples.keys()}")
    
    tokenized = ds.map(tokenize_function, batched=True, remove_columns=ds.column_names)
    return tokenized


def find_factorized_submodules(model):
    """
    Find all A and B submodules inside FactorizedLinear layers.
    
    Returns:
        List of module name patterns that PEFT can match
    """
    from src.urank.modeling_gpt2_compressed import FactorizedLinear
    
    factorized_targets = []
    for name, module in model.named_modules():
        if isinstance(module, FactorizedLinear):
            # Add both A and B submodules of this FactorizedLinear
            factorized_targets.append(f"{name}.A")
            factorized_targets.append(f"{name}.B")
    
    return factorized_targets


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune compressed model with factorized layers"
    )
    parser.add_argument("--preset", choices=["lora", "full", "freeze"], required=True)
    parser.add_argument("--model", required=True, help="Path to compressed model")
    parser.add_argument("--data", required=True, help="Dataset name")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    args = parser.parse_args()
    
    print(f"Loading compressed model from: {args.model}")
    
    # Load model with custom class
    model = GPT2CompressedLMHeadModel.from_pretrained(args.model)
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")  # PATCH 1: Move to CUDA immediately
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    # Set pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id
    
    print(f"Model loaded with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Apply PEFT based on preset
    if args.preset == "lora":
        print(f"\nApplying LoRA (r={args.lora_r}, alpha={args.lora_alpha})")
        
        # Find all FactorizedLinear A and B submodules
        target_modules = find_factorized_submodules(model)
        
        if not target_modules:
            print("WARNING: No FactorizedLinear modules found. Falling back to standard patterns.")
            target_modules = ["c_attn", "c_proj", "c_fc"]
        else:
            print(f"Found {len(target_modules)} FactorizedLinear submodules (A and B) for LoRA:")
            for i, name in enumerate(target_modules[:10]):  # Show first 10
                print(f"  [{i+1}] {name}")
            if len(target_modules) > 10:
                print(f"  ... and {len(target_modules) - 10} more")
        
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_modules,  # Now contains explicit paths like "transformer.h.0.attn.c_attn.A"
            bias="none"
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        model.train()  # PATCH 4: Explicitly set to training mode
    
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
    train_dataset = load_dataset_from_name(args.data, tokenizer)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=20,
        save_steps=1000,
        save_total_limit=2,
        bf16=False,                    # PATCH 3: Disable for stability
        fp16=False,                    # PATCH 3: Disable for stability
        gradient_checkpointing=False,  # PATCH 2: Disable to fix gradient flow
        report_to="none",
    )
    
    # Data collator
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
    architecture_files = [
        "configuration_gpt2_compressed.py",
        "modeling_gpt2_compressed.py",
        "__init__.py"
    ]
    
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