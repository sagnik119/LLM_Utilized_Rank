#!/usr/bin/env python3
"""
Fine-tune a compressed model to recover performance.

This script uses HuggingFace Trainer to fine-tune a compressed model
on a language modeling task.

Usage:
    python scripts/finetune.py --model ckpts/gpt2_compressed \\
        --train-dataset wikitext --epochs 2 --lr 5e-5 --out ckpts/gpt2_finetuned
"""

import argparse
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser(description="Fine-tune compressed model")
    parser.add_argument("--model", type=str, required=True, help="Model path to fine-tune")
    parser.add_argument("--train-dataset", type=str, default="wikitext", help="Training dataset")
    parser.add_argument("--config", type=str, default="wikitext-2-raw-v1", help="Dataset config")
    parser.add_argument("--epochs", type=int, default=2, help="Training epochs")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=4, help="Per-device batch size")
    parser.add_argument("--out", type=str, default="finetuned", help="Output directory")
    parser.add_argument("--max-length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--warmup-steps", type=int, default=500, help="Warmup steps")
    parser.add_argument("--logging-steps", type=int, default=100, help="Logging frequency")
    parser.add_argument("--save-steps", type=int, default=1000, help="Checkpoint save frequency")
    args = parser.parse_args()
    
    # Load model and tokenizer
    print(f"Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load dataset
    print(f"Loading dataset: {args.train_dataset}")
    ds = load_dataset(args.train_dataset, args.config)
    
    # Tokenize function
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=args.max_length,
            padding="max_length",
        )
    
    # Tokenize datasets
    print("Tokenizing dataset...")
    tokenized_ds = ds.map(
        tokenize_function,
        batched=True,
        remove_columns=ds["train"].column_names,
        desc="Tokenizing",
    )
    
    # Data collator for causal language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM, not masked LM
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.out,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        eval_strategy="steps" if "validation" in tokenized_ds else "no",
        eval_steps=args.save_steps if "validation" in tokenized_ds else None,
        fp16=True,  # Use mixed precision if available
        report_to="none",  # Disable wandb/tensorboard
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds.get("validation"),
        data_collator=data_collator,
    )
    
    # Train
    print("\nStarting training...")
    trainer.train()
    
    # Save final model
    print(f"\nSaving fine-tuned model to: {args.out}")
    model.save_pretrained(args.out, safe_serialization=True)
    tokenizer.save_pretrained(args.out)
    
    print("Done!")


if __name__ == "__main__":
    main()