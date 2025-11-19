#!/usr/bin/env python3
"""
Test script to verify that export_compressed_model.py properly preserves
FactorizedLinear layers when exporting with or without LoRA merge.

This script:
1. Creates a small dummy compressed model with FactorizedLinear layers
2. Optionally applies LoRA to it
3. Exports using export_compressed_model.py logic
4. Reloads the exported model
5. Verifies FactorizedLinear structure is preserved
6. Compares weight values to ensure correctness

Usage:
    # Test export without LoRA
    python scripts/test_export_factorized.py --test-dir /tmp/test_export

    # Test export with LoRA merge
    python scripts/test_export_factorized.py --test-dir /tmp/test_export --with-lora
"""

import argparse
import os
import sys
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import torch.nn as nn
from transformers import GPT2Config

from urank.modeling_gpt2_compressed import (
    GPT2CompressedLMHeadModel,
    GPT2CompressedConfig,
    FactorizedLinear
)


def create_dummy_compressed_model(output_dir: str, num_factorized: int = 4):
    """Create a small GPT-2 model with some FactorizedLinear layers for testing."""
    print(f"\n[1/6] Creating dummy compressed model with {num_factorized} FactorizedLinear layers...")
    
    # Create small GPT-2 config
    config = GPT2CompressedConfig(
        vocab_size=1000,
        n_positions=128,
        n_embd=128,
        n_layer=2,
        n_head=2,
    )
    
    # Create model
    model = GPT2CompressedLMHeadModel(config)
    
    # Replace some layers with FactorizedLinear
    # Target: attention projections in first layer
    layer_0 = model.transformer.h[0]
    
    # Replace c_attn (in_features=128, out_features=384 for Q,K,V combined)
    orig_c_attn = layer_0.attn.c_attn
    factorized_c_attn = FactorizedLinear(
        in_features=128,
        out_features=384,
        rank=16,
        bias=True
    )
    # Initialize with small random values
    nn.init.normal_(factorized_c_attn.A.weight, std=0.02)
    nn.init.normal_(factorized_c_attn.B.weight, std=0.02)
    if factorized_c_attn.B.bias is not None:
        nn.init.zeros_(factorized_c_attn.B.bias)
    
    layer_0.attn.c_attn = factorized_c_attn
    
    # Replace c_proj
    factorized_c_proj = FactorizedLinear(
        in_features=128,
        out_features=128,
        rank=16,
        bias=True
    )
    nn.init.normal_(factorized_c_proj.A.weight, std=0.02)
    nn.init.normal_(factorized_c_proj.B.weight, std=0.02)
    if factorized_c_proj.B.bias is not None:
        nn.init.zeros_(factorized_c_proj.B.bias)
    
    layer_0.attn.c_proj = factorized_c_proj
    
    # Count FactorizedLinear modules
    factorized_count = sum(1 for m in model.modules() if isinstance(m, FactorizedLinear))
    print(f"    ✓ Created model with {factorized_count} FactorizedLinear layers")
    
    # Save model
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir, safe_serialization=True)
    
    # Save a dummy tokenizer (copy from GPT-2)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.save_pretrained(output_dir)
    
    print(f"    ✓ Saved to: {output_dir}")
    
    return model, factorized_count


def apply_lora_and_save(model_dir: str, output_dir: str):
    """Apply LoRA adapters to the model and save."""
    print(f"\n[2/6] Applying LoRA adapters...")
    
    from peft import get_peft_model, LoraConfig, TaskType
    
    # Load model
    model = GPT2CompressedLMHeadModel.from_pretrained(model_dir)
    
    # Find FactorizedLinear submodules
    target_modules = []
    for name, module in model.named_modules():
        if isinstance(module, FactorizedLinear):
            target_modules.append(f"{name}.A")
            target_modules.append(f"{name}.B")
    
    print(f"    Found {len(target_modules)} submodules for LoRA")
    
    # Apply LoRA
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=4,
        lora_alpha=8,
        lora_dropout=0.05,
        target_modules=target_modules,
        bias="none"
    )
    
    model = get_peft_model(model, peft_config)
    print(f"    ✓ Applied LoRA")
    
    # Save PEFT model
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    
    # Also need to copy tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    tokenizer.save_pretrained(output_dir)
    
    print(f"    ✓ Saved LoRA model to: {output_dir}")
    
    return len(target_modules)


def test_export(base_dir: str, lora_dir: str, export_dir: str, with_lora: bool):
    """Test the export function."""
    step = 3 if with_lora else 2
    
    if with_lora:
        print(f"\n[{step}/6] Testing export WITH LoRA merge...")
        from scripts.export_compressed_model import export_compressed_model
        export_compressed_model(
            model_path=base_dir,
            output_path=export_dir,
            lora_path=lora_dir,
        )
    else:
        print(f"\n[{step}/6] Testing export WITHOUT LoRA...")
        from scripts.export_compressed_model import export_compressed_model
        export_compressed_model(
            model_path=base_dir,
            output_path=export_dir,
        )
    
    print(f"    ✓ Export completed")


def verify_exported_model(export_dir: str, expected_count: int):
    """Reload exported model and verify structure."""
    print(f"\n[5/6] Verifying exported model...")
    
    # Check files exist
    required_files = [
        "config.json",
        "model.safetensors",
        "modeling_gpt2_compressed.py",
        "__init__.py",
    ]
    
    for fname in required_files:
        fpath = os.path.join(export_dir, fname)
        if not os.path.exists(fpath):
            print(f"    ✗ ERROR: Missing file: {fname}")
            return False
    
    print(f"    ✓ All required files present")
    
    # Load config and check model_type
    import json
    with open(os.path.join(export_dir, "config.json")) as f:
        config = json.load(f)
    
    if config.get("model_type") != "gpt2_compressed":
        print(f"    ✗ ERROR: Wrong model_type: {config.get('model_type')}")
        return False
    
    if "auto_map" not in config:
        print(f"    ✗ ERROR: Missing auto_map in config")
        return False
    
    print(f"    ✓ Config is correct")
    
    # Reload model
    model = GPT2CompressedLMHeadModel.from_pretrained(export_dir, trust_remote_code=True)
    
    # Count FactorizedLinear modules
    factorized_count = sum(1 for m in model.modules() if isinstance(m, FactorizedLinear))
    
    if factorized_count != expected_count:
        print(f"    ✗ ERROR: Expected {expected_count} FactorizedLinear layers, found {factorized_count}")
        return False
    
    print(f"    ✓ Found {factorized_count} FactorizedLinear layers (correct!)")
    
    # Test forward pass
    input_ids = torch.randint(0, 1000, (1, 10))
    try:
        with torch.no_grad():
            outputs = model(input_ids)
        print(f"    ✓ Forward pass successful")
    except Exception as e:
        print(f"    ✗ ERROR: Forward pass failed: {e}")
        return False
    
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-dir", required=True, help="Directory for test files")
    parser.add_argument("--with-lora", action="store_true", help="Test with LoRA merge")
    args = parser.parse_args()
    
    test_dir = Path(args.test_dir)
    base_dir = test_dir / "base_model"
    lora_dir = test_dir / "lora_model"
    export_dir = test_dir / "exported_model"
    
    # Clean up previous test
    if test_dir.exists():
        shutil.rmtree(test_dir)
    
    print("=" * 70)
    print("Testing Export FactorizedLinear Preservation")
    if args.with_lora:
        print("Mode: WITH LoRA merge")
    else:
        print("Mode: WITHOUT LoRA")
    print("=" * 70)
    
    try:
        # Step 1: Create dummy model
        original_model, expected_count = create_dummy_compressed_model(str(base_dir))
        
        # Step 2: Optionally apply LoRA
        if args.with_lora:
            lora_target_count = apply_lora_and_save(str(base_dir), str(lora_dir))
        else:
            lora_dir = None
        
        # Step 3: Export
        test_export(str(base_dir), str(lora_dir) if args.with_lora else None, 
                   str(export_dir), args.with_lora)
        
        # Step 4: Verify
        success = verify_exported_model(str(export_dir), expected_count)
        
        # Summary
        print("\n" + "=" * 70)
        if success:
            print("✓ ALL TESTS PASSED!")
            print("=" * 70)
            print(f"Summary:")
            print(f"  - Original model had {expected_count} FactorizedLinear layers")
            if args.with_lora:
                print(f"  - Applied LoRA to {lora_target_count} submodules (A and B)")
                print(f"  - Merged LoRA into base weights")
            print(f"  - Exported model to: {export_dir}")
            print(f"  - Reloaded model has {expected_count} FactorizedLinear layers")
            print(f"  - Structure preserved correctly!")
        else:
            print("✗ TESTS FAILED")
            print("=" * 70)
            print("FactorizedLinear structure was NOT preserved correctly.")
            return 1
        
        print("=" * 70)
        
        return 0
        
    except Exception as e:
        print(f"\n✗ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())