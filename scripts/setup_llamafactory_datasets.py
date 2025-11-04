#!/usr/bin/env python3
"""
Setup dataset_info.json for LLaMA-Factory with common datasets.

This script creates or updates dataset_info.json in the specified directory
with entries for commonly used datasets like WikiText, OpenWebText, etc.

Usage:
    python scripts/setup_llamafactory_datasets.py --data-dir /path/to/LLaMA-Factory/data
"""

import argparse
import json
import os
import sys


DATASET_TEMPLATES = {
    "wikitext2": {
        "hf_hub_url": "wikitext",
        "subset": "wikitext-2-raw-v1",
        "split": "train",
        "formatting": "alpaca",
        "columns": {
            "prompt": "text"
        }
    },
    "wikitext103": {
        "hf_hub_url": "wikitext",
        "subset": "wikitext-103-raw-v1",
        "split": "train",
        "formatting": "alpaca",
        "columns": {
            "prompt": "text"
        }
    },
    "openwebtext": {
        "hf_hub_url": "openwebtext",
        "split": "train",
        "formatting": "alpaca",
        "columns": {
            "prompt": "text"
        }
    }
}


def main():
    parser = argparse.ArgumentParser(
        description="Setup dataset_info.json for LLaMA-Factory"
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Path to LLaMA-Factory data directory (will create dataset_info.json here)"
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Merge with existing dataset_info.json instead of overwriting"
    )
    args = parser.parse_args()

    # Ensure data directory exists
    os.makedirs(args.data_dir, exist_ok=True)
    
    dataset_info_path = os.path.join(args.data_dir, "dataset_info.json")
    
    # Load existing file if merging
    if args.merge and os.path.exists(dataset_info_path):
        print(f"Loading existing dataset_info.json from {dataset_info_path}")
        with open(dataset_info_path, "r") as f:
            dataset_info = json.load(f)
    else:
        dataset_info = {}
    
    # Add/update templates
    for name, config in DATASET_TEMPLATES.items():
        if name in dataset_info and args.merge:
            print(f"Skipping {name} (already exists)")
        else:
            dataset_info[name] = config
            print(f"Added {name}")
    
    # Save updated file
    with open(dataset_info_path, "w") as f:
        json.dump(dataset_info, f, indent=2)
    
    print(f"\nDataset info saved to: {dataset_info_path}")
    print(f"Total datasets: {len(dataset_info)}")
    print("\nAvailable datasets:")
    for name in sorted(dataset_info.keys()):
        print(f"  - {name}")


if __name__ == "__main__":
    main()