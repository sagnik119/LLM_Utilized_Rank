#!/usr/bin/env python3
"""
Test script to verify that custom architecture registration works correctly.

This test:
1. Creates a dummy compressed model and exports it
2. Tests that AutoConfig and AutoModel can load it after registration
3. Verifies FactorizedLinear layers are preserved
4. Confirms the registration works as expected by lm-eval-harness

Usage:
    python scripts/test_architecture_registration.py
"""

import sys
import os
import shutil
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM

from urank.modeling_gpt2_compressed import GPT2CompressedLMHeadModel, FactorizedLinear
from urank.configuration_gpt2_compressed import GPT2CompressedConfig


def test_architecture_registration():
    """Test that custom architecture can be registered and loaded."""
    
    print("=" * 70)
    print("Testing Custom Architecture Registration")
    print("=" * 70)
    
    test_dir = Path("/tmp/test_arch_registration")
    if test_dir.exists():
        shutil.rmtree(test_dir)
    test_dir.mkdir(parents=True)
    
    # Step 1: Register architecture
    print("\n[1/5] Registering custom architecture...")
    try:
        AutoConfig.register("gpt2_compressed", GPT2CompressedConfig)
        AutoModelForCausalLM.register(GPT2CompressedConfig, GPT2CompressedLMHeadModel)
        print("    ✓ Successfully registered gpt2_compressed")
    except Exception as e:
        print(f"    ✗ ERROR: Registration failed: {e}")
        return False
    
    # Step 2: Create a dummy model with FactorizedLinear
    print("\n[2/5] Creating dummy compressed model...")
    config = GPT2CompressedConfig(
        vocab_size=1000,
        n_positions=128,
        n_embd=128,
        n_layer=2,
        n_head=2,
    )
    
    model = GPT2CompressedLMHeadModel(config)
    
    # Manually add a FactorizedLinear layer
    layer_0 = model.transformer.h[0]
    factorized = FactorizedLinear(
        in_features=128,
        out_features=384,
        rank=16,
        bias=True
    )
    nn.init.normal_(factorized.A.weight, std=0.02)
    nn.init.normal_(factorized.B.weight, std=0.02)
    layer_0.attn.c_attn = factorized
    
    orig_count = sum(1 for m in model.modules() if isinstance(m, FactorizedLinear))
    print(f"    ✓ Created model with {orig_count} FactorizedLinear layer(s)")
    
    # Step 3: Save the model
    print("\n[3/5] Saving model...")
    model.save_pretrained(test_dir, safe_serialization=True)
    
    # Also copy the modeling and config files (simulate export)
    src_model = Path(__file__).parent.parent / "src" / "urank" / "modeling_gpt2_compressed.py"
    src_cfg = Path(__file__).parent.parent / "src" / "urank" / "configuration_gpt2_compressed.py"
    
    shutil.copy2(src_model, test_dir / "modeling_gpt2_compressed.py")
    shutil.copy2(src_cfg, test_dir / "configuration_gpt2_compressed.py")
    
    # Create __init__.py
    (test_dir / "__init__.py").write_text(
        "from .configuration_gpt2_compressed import GPT2CompressedConfig\n"
        "from .modeling_gpt2_compressed import GPT2CompressedLMHeadModel, FactorizedLinear\n"
    )
    
    print(f"    ✓ Saved to {test_dir}")
    
    # Step 4: Test AutoConfig loading
    print("\n[4/5] Testing AutoConfig.from_pretrained()...")
    try:
        loaded_config = AutoConfig.from_pretrained(str(test_dir), trust_remote_code=True)
        
        if loaded_config.model_type != "gpt2_compressed":
            print(f"    ✗ ERROR: Wrong model_type: {loaded_config.model_type}")
            return False
        
        if not isinstance(loaded_config, GPT2CompressedConfig):
            print(f"    ✗ ERROR: Wrong config class: {type(loaded_config)}")
            return False
        
        print(f"    ✓ Loaded config: {loaded_config.model_type}")
        print(f"    ✓ Config class: {type(loaded_config).__name__}")
        
    except Exception as e:
        print(f"    ✗ ERROR: Failed to load config: {e}")
        return False
    
    # Step 5: Test AutoModelForCausalLM loading
    print("\n[5/5] Testing AutoModelForCausalLM.from_pretrained()...")
    try:
        loaded_model = AutoModelForCausalLM.from_pretrained(
            str(test_dir),
            trust_remote_code=True
        )
        
        if not isinstance(loaded_model, GPT2CompressedLMHeadModel):
            print(f"    ✗ ERROR: Wrong model class: {type(loaded_model)}")
            return False
        
        # Count FactorizedLinear layers
        loaded_count = sum(1 for m in loaded_model.modules() if isinstance(m, FactorizedLinear))
        
        if loaded_count != orig_count:
            print(f"    ✗ ERROR: Expected {orig_count} FactorizedLinear, found {loaded_count}")
            return False
        
        print(f"    ✓ Loaded model: {type(loaded_model).__name__}")
        print(f"    ✓ FactorizedLinear count: {loaded_count}")
        
        # Test forward pass
        input_ids = torch.randint(0, 1000, (1, 10))
        with torch.no_grad():
            outputs = loaded_model(input_ids)
        
        print(f"    ✓ Forward pass successful")
        
    except Exception as e:
        print(f"    ✗ ERROR: Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Clean up
    shutil.rmtree(test_dir)
    
    return True


def main():
    success = test_architecture_registration()
    
    print("\n" + "=" * 70)
    if success:
        print("✓ ALL TESTS PASSED!")
        print("=" * 70)
        print("Summary:")
        print("  - Custom architecture registered successfully")
        print("  - AutoConfig loads GPT2CompressedConfig correctly")
        print("  - AutoModelForCausalLM loads GPT2CompressedLMHeadModel correctly")
        print("  - FactorizedLinear layers preserved through save/load")
        print("  - Ready for lm-eval-harness evaluation!")
        print("=" * 70)
        return 0
    else:
        print("✗ TESTS FAILED")
        print("=" * 70)
        print("Custom architecture registration is not working correctly.")
        return 1


if __name__ == "__main__":
    sys.exit(main())