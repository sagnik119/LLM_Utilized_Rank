#!/usr/bin/env python3
"""
Fine-tune a compressed model using LLaMA-Factory with PEFT options.

This script provides a wrapper around LLaMA-Factory CLI with preset configurations
for different fine-tuning strategies: LoRA, Full FT, and Freeze FT.

Prerequisites:
    - LLaMA-Factory installed with llamafactory-cli on PATH
    - Install: https://github.com/hiyouga/LLaMA-Factory

Usage:
    python scripts/finetune_llamafactory.py --preset lora \
        --model ckpts/gpt2_compressed --data wikitext2 --out outputs/lora_run
"""

import argparse
import subprocess
import os
import yaml
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune with LLaMA-Factory (LoRA/Full/Freeze)"
    )
    parser.add_argument(
        "--preset",
        choices=["lora", "full", "freeze"],
        required=True,
        help="Fine-tuning preset: lora, full, or freeze",
    )
    parser.add_argument(
        "--model", required=True, help="Base or compressed model path"
    )
    parser.add_argument(
        "--data",
        required=True,
        help="Training data name or path (per LLaMA-Factory format)",
    )
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument(
        "--cfg",
        default=None,
        help="Optional YAML to override preset configuration",
    )
    parser.add_argument(
        "--epochs", type=int, default=None, help="Override number of epochs"
    )
    parser.add_argument(
        "--lr", type=float, default=None, help="Override learning rate"
    )
    parser.add_argument(
        "--batch-size", type=int, default=None, help="Override batch size"
    )
    args = parser.parse_args()

    # Map presets to config files
    PRESETS = {
        "lora": "configs/finetune_lora.yaml",
        "full": "configs/finetune_full.yaml",
        "freeze": "configs/finetune_freeze.yaml",
    }

    cfg_path = PRESETS[args.preset]
    
    # Check if preset config exists
    if not os.path.exists(cfg_path):
        print(f"Error: Preset config not found: {cfg_path}")
        print("Please ensure configuration files are present in configs/")
        sys.exit(1)

    # Load preset configuration
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    # Override required fields
    cfg["model_name_or_path"] = args.model
    cfg["output_dir"] = args.out
    cfg["dataset"] = args.data

    # Apply command-line overrides
    if args.epochs is not None:
        cfg["num_train_epochs"] = args.epochs
    if args.lr is not None:
        cfg["learning_rate"] = args.lr
    if args.batch_size is not None:
        cfg["per_device_train_batch_size"] = args.batch_size

    # Apply user config overrides
    if args.cfg:
        with open(args.cfg) as f:
            override = yaml.safe_load(f)
        cfg.update(override)

    # Create output directory and save resolved config
    os.makedirs(args.out, exist_ok=True)
    resolved_config = os.path.join(args.out, "resolved_config.yaml")
    
    with open(resolved_config, "w") as f:
        yaml.safe_dump(cfg, f, default_flow_style=False)

    print(f"Fine-tuning with preset: {args.preset}")
    print(f"Model: {args.model}")
    print(f"Dataset: {args.data}")
    print(f"Output: {args.out}")
    print(f"Resolved config saved to: {resolved_config}")

    # Build LLaMA-Factory command
    cmd = ["llamafactory-cli", "train", resolved_config]

    print(f"\nRunning: {' '.join(cmd)}")
    print("=" * 60)

    # Execute LLaMA-Factory
    try:
        subprocess.run(cmd, check=True)
        print("\n" + "=" * 60)
        print("Fine-tuning completed successfully!")
        print(f"Model saved to: {args.out}")
    except subprocess.CalledProcessError as e:
        print(f"\nError: Fine-tuning failed with exit code {e.returncode}")
        sys.exit(1)
    except FileNotFoundError:
        print("\nError: llamafactory-cli not found.")
        print("Please install LLaMA-Factory: https://github.com/hiyouga/LLaMA-Factory")
        sys.exit(1)


if __name__ == "__main__":
    main()