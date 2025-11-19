#!/usr/bin/env python3
"""
Export a compressed model (base + optional LoRA) as a HuggingFace-compatible
custom architecture checkpoint that lm-eval-harness can load.

This script:
1. Loads the base compressed model (with FactorizedLinear layers)
2. Optionally merges LoRA adapters if provided
3. Exports model + tokenizer + custom modeling file
4. Creates correct config.json with trust_remote_code settings
5. Produces a checkpoint that lm-eval can load directly

Usage:
    # Export base compressed model only
    python scripts/export_compressed_model.py \
        --model ckpts/gpt2_compressed \
        --out exports/gpt2_compressed_eval

    # Export compressed model + merged LoRA
    python scripts/export_compressed_model.py \
        --model ckpts/gpt2_compressed \
        --lora outputs/gpt2_lora \
        --out exports/gpt2_lora_merged

    # Then run lm-eval
    python scripts/eval_lm_eval.py \
        --model exports/gpt2_lora_merged \
        --tasks wikitext arc_easy hellaswag \
        --batch 8
"""

import argparse
import json
import shutil
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from transformers import AutoTokenizer, AutoConfig


def export_compressed_model(
    model_path: str,
    output_path: str,
    lora_path: str = None,
    tokenizer_path: str = None,
):
    """
    Export compressed model with custom architecture for lm-eval compatibility.
    
    Args:
        model_path: Path to base compressed model checkpoint
        output_path: Directory to save exported model
        lora_path: Optional path to LoRA adapter to merge
        tokenizer_path: Optional tokenizer path (defaults to original model)
    """
    from urank.modeling_gpt2_compressed import GPT2CompressedLMHeadModel
    
    print(f"Loading compressed model from: {model_path}")
    
    # Load base compressed model
    try:
        model = GPT2CompressedLMHeadModel.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        print(f"✓ Loaded compressed model ({sum(p.numel() for p in model.parameters())/1e6:.1f}M params)")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        raise
    
    # Optionally merge LoRA
    if lora_path:
        print(f"Merging LoRA adapters from: {lora_path}")
        try:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, lora_path)
            
            # Count LoRA params before merge
            lora_params = sum(p.numel() for n, p in model.named_parameters() if "lora" in n.lower())
            print(f"  LoRA params: {lora_params/1e6:.2f}M")
            
            # Merge and unload
            model = model.merge_and_unload()
            print(f"✓ Merged LoRA into base weights")
        except ImportError:
            print("✗ PEFT not installed. Install with: pip install peft")
            raise
        except Exception as e:
            print(f"✗ Failed to merge LoRA: {e}")
            raise
    
    # Create output directory
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"\nExporting to: {output_path}")
    
    # Step 1: Save model weights
    print("  [1/4] Saving model weights...")
    model.save_pretrained(output_path, safe_serialization=True)
    
    # Step 2: Save tokenizer
    print("  [2/4] Saving tokenizer...")
    tok_path = tokenizer_path or model_path
    try:
        tokenizer = AutoTokenizer.from_pretrained(tok_path)
        tokenizer.save_pretrained(output_path)
    except Exception as e:
        # Fallback to gpt2 tokenizer
        print(f"    Warning: Could not load tokenizer from {tok_path}, using gpt2")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.save_pretrained(output_path)
    
    # Step 3: Copy custom modeling file
    print("  [3/4] Copying custom modeling file...")
    src_modeling = Path(__file__).parent.parent / "src" / "urank" / "modeling_gpt2_compressed.py"
    dst_modeling = output_path / "modeling_gpt2_compressed.py"
    shutil.copy2(src_modeling, dst_modeling)
    
    # Create __init__.py
    init_file = output_path / "__init__.py"
    init_file.write_text("from .modeling_gpt2_compressed import GPT2CompressedLMHeadModel\n")
    
    # Step 4: Create/update config.json
    print("  [4/4] Creating config.json...")
    
    # Load existing config if present
    config_path = output_path / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
    else:
        # Create minimal config for GPT-2
        config = {
            "vocab_size": 50257,
            "n_embd": 768,
            "n_layer": 12,
            "n_head": 12,
            "n_positions": 1024,
            "layer_norm_epsilon": 1e-5,
            "initializer_range": 0.02,
        }
    
    # Add custom architecture fields with correct auto_map
    config.update({
        "model_type": "gpt2_compressed",
        "architectures": ["GPT2CompressedLMHeadModel"],
        "auto_map": {
            "AutoConfig": "modeling_gpt2_compressed.GPT2CompressedConfig",
            "AutoModelForCausalLM": "modeling_gpt2_compressed.GPT2CompressedLMHeadModel"
        },
    })
    
    # Ensure token IDs are set
    if "bos_token_id" not in config:
        config["bos_token_id"] = 50256
    if "eos_token_id" not in config:
        config["eos_token_id"] = 50256
    
    # Write updated config
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print("\n✓ Export complete!")
    print(f"\nTo evaluate with lm-eval-harness:")
    print(f"  python scripts/eval_lm_eval.py \\")
    print(f"    --model {output_path} \\")
    print(f"    --tasks wikitext arc_easy hellaswag \\")
    print(f"    --batch 8 \\")
    print(f"    --out results.json")
    print(f"\nNote: lm-eval will use trust_remote_code=True to load the custom architecture")


def main():
    parser = argparse.ArgumentParser(
        description="Export compressed model for lm-eval-harness compatibility"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to base compressed model checkpoint"
    )
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output directory for exported model"
    )
    parser.add_argument(
        "--lora",
        type=str,
        default=None,
        help="Optional path to LoRA adapter directory to merge"
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Optional tokenizer path (defaults to model path)"
    )
    
    args = parser.parse_args()
    
    export_compressed_model(
        model_path=args.model,
        output_path=args.out,
        lora_path=args.lora,
        tokenizer_path=args.tokenizer,
    )


if __name__ == "__main__":
    main()