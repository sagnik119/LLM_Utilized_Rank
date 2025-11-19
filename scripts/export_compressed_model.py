#!/usr/bin/env python3
"""
Export a compressed model with optional LoRA-merging into a HuggingFace-
compatible directory that lm-eval-harness can load.

This produces:
    config.json
    model.safetensors
    modeling_gpt2_compressed.py
    configuration_gpt2_compressed.py
    __init__.py
    tokenizer files
"""

import argparse
import json
import shutil
from pathlib import Path
import sys

# Add local src to import search path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

from urank.modeling_gpt2_compressed import GPT2CompressedLMHeadModel, FactorizedLinear
from urank.configuration_gpt2_compressed import GPT2CompressedConfig


# Register custom architecture globally
AutoConfig.register("gpt2_compressed", GPT2CompressedConfig)
AutoModelForCausalLM.register(GPT2CompressedConfig, GPT2CompressedLMHeadModel)


def export_compressed_model(model_path, output_path, lora_path=None, tokenizer_path=None):
    print(f"\nLoading compressed model from: {model_path}")

    # Load compressed model
    model = GPT2CompressedLMHeadModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    print(f"✓ Loaded compressed model ({sum(p.numel() for p in model.parameters())/1e6:.2f}M params)")

    # Count factorized layers
    orig_factorized = sum(1 for m in model.modules() if isinstance(m, FactorizedLinear))
    print(f"  Contains {orig_factorized} FactorizedLinear layers")

    # Optionally merge LoRA
    if lora_path:
        print(f"\nMerging LoRA adapters from: {lora_path}")
        from peft import PeftModel

        peft_model = PeftModel.from_pretrained(model, lora_path, is_trainable=False)
        print("  Merging LoRA adapters...")
        model = peft_model.merge_and_unload()

        after = sum(1 for m in model.modules() if isinstance(m, FactorizedLinear))
        if after != orig_factorized:
            print(f"  ⚠ WARNING: FactorizedLinear count changed: {orig_factorized} → {after}")
        else:
            print(f"  ✓ Factorized structure preserved ({after} layers)")

    # Create export directory
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"\nExporting to: {output_path}")

    # 1. Save weights
    print("  [1/5] Saving model weights...")
    model.save_pretrained(output_path, safe_serialization=True)

    # 2. Save tokenizer
    print("  [2/5] Saving tokenizer...")
    tok_src = tokenizer_path or model_path
    try:
        tok = AutoTokenizer.from_pretrained(tok_src)
    except:
        print("    Falling back to gpt2 tokenizer")
        tok = AutoTokenizer.from_pretrained("gpt2")
    tok.save_pretrained(output_path)

    # 3. Copy modeling file
    print("  [3/5] Copying modeling file...")
    src_model = Path(__file__).parent.parent / "src" / "urank" / "modeling_gpt2_compressed.py"
    shutil.copy2(src_model, output_path / "modeling_gpt2_compressed.py")

    # 4. Copy config file
    print("  [4/5] Copying configuration file...")
    src_cfg = Path(__file__).parent.parent / "src" / "urank" / "configuration_gpt2_compressed.py"
    shutil.copy2(src_cfg, output_path / "configuration_gpt2_compressed.py")

    # 5. Create __init__.py
    print("  [5/5] Generating __init__.py...")
    (output_path / "__init__.py").write_text(
        "from .configuration_gpt2_compressed import GPT2CompressedConfig\n"
        "from .modeling_gpt2_compressed import GPT2CompressedLMHeadModel, FactorizedLinear\n"
        "\n"
        "__all__ = ['GPT2CompressedConfig','GPT2CompressedLMHeadModel','FactorizedLinear']\n"
    )

    # Update config.json
    config_path = output_path / "config.json"
    with open(config_path) as f:
        config = json.load(f)

    config.update({
        "model_type": "gpt2_compressed",
        "architectures": ["GPT2CompressedLMHeadModel"],
        "auto_map": {
            "AutoConfig": "configuration_gpt2_compressed.GPT2CompressedConfig",
            "AutoModelForCausalLM": "modeling_gpt2_compressed.GPT2CompressedLMHeadModel",
        },
    })

    # Write updated config.json
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print("\n✓ Export complete!")
    print("\nRun evaluation via:")
    print(f"  python scripts/eval_lm_eval.py --model {output_path} --tasks wikitext arc_easy hellaswag --batch 8")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--lora", default=None)
    parser.add_argument("--tokenizer", default=None)
    args = parser.parse_args()

    export_compressed_model(args.model, args.out, args.lora, args.tokenizer)


if __name__ == "__main__":
    main()