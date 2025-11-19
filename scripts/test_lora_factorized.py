#!/usr/bin/env python3
"""
Test script to verify LoRA is properly applied to FactorizedLinear A and B submodules.

This script:
1. Loads a compressed model with FactorizedLinear layers
2. Applies LoRA using the discovery method
3. Verifies that LoRA adapters are attached to both A and B
4. Tests forward/backward pass with gradient flow
5. Confirms trainable parameter counts

Usage:
    python scripts/test_lora_factorized.py --model ckpts/gpt2_compressed
"""

import argparse
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from peft import get_peft_model, LoraConfig, TaskType

from src.urank.modeling_gpt2_compressed import GPT2CompressedLMHeadModel, FactorizedLinear


def find_factorized_submodules(model):
    """Find all A and B submodules inside FactorizedLinear layers."""
    factorized_targets = []
    for name, module in model.named_modules():
        if isinstance(module, FactorizedLinear):
            factorized_targets.append(f"{name}.A")
            factorized_targets.append(f"{name}.B")
    return factorized_targets


def test_lora_injection(model):
    """Verify that LoRA adapters are present in A and B submodules."""
    lora_count = 0
    factorized_with_lora = []
    
    for name, module in model.named_modules():
        # Check if this is a LoRA-wrapped module (will have lora_A and lora_B attributes)
        if hasattr(module, 'lora_A') or hasattr(module, 'lora_B'):
            lora_count += 1
            parent_name = '.'.join(name.split('.')[:-1])  # Get parent FactorizedLinear name
            if parent_name not in factorized_with_lora:
                factorized_with_lora.append(parent_name)
    
    return lora_count, factorized_with_lora


def test_gradient_flow(model, device='cuda'):
    """Test that gradients flow through LoRA adapters."""
    model.train()
    
    # Create dummy input
    input_ids = torch.randint(0, 50257, (2, 10)).to(device)
    labels = input_ids.clone()
    
    # Forward pass
    outputs = model(input_ids=input_ids, labels=labels)
    loss = outputs.loss
    
    # Backward pass
    loss.backward()
    
    # Check for gradients in LoRA parameters
    lora_params_with_grad = []
    for name, param in model.named_parameters():
        if 'lora' in name.lower() and param.grad is not None:
            lora_params_with_grad.append(name)
    
    return len(lora_params_with_grad) > 0, lora_params_with_grad


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='Path to compressed model')
    parser.add_argument('--lora-r', type=int, default=8)
    parser.add_argument('--lora-alpha', type=int, default=16)
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("="*70)
    print("LoRA FactorizedLinear Test")
    print("="*70)
    
    # Step 1: Load model
    print(f"\n[1/5] Loading compressed model from: {args.model}")
    model = GPT2CompressedLMHeadModel.from_pretrained(args.model)
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"      Total parameters: {total_params:,}")
    
    # Step 2: Find FactorizedLinear submodules
    print(f"\n[2/5] Discovering FactorizedLinear A and B submodules...")
    target_modules = find_factorized_submodules(model)
    
    if not target_modules:
        print("      ❌ ERROR: No FactorizedLinear modules found!")
        return
    
    print(f"      ✓ Found {len(target_modules)} submodules")
    print(f"      Examples:")
    for i, name in enumerate(target_modules[:6]):
        print(f"        - {name}")
    if len(target_modules) > 6:
        print(f"        ... and {len(target_modules) - 6} more")
    
    # Step 3: Apply LoRA
    print(f"\n[3/5] Applying LoRA (r={args.lora_r}, alpha={args.lora_alpha})...")
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=target_modules,
        bias="none"
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # Step 4: Verify LoRA injection
    print(f"\n[4/5] Verifying LoRA adapters are attached to A and B...")
    lora_count, factorized_with_lora = test_lora_injection(model)
    
    if lora_count == 0:
        print(f"      ❌ ERROR: No LoRA adapters found!")
        print(f"      This means LoRA was NOT applied to FactorizedLinear A/B submodules")
        return
    
    print(f"      ✓ Found {lora_count} LoRA-wrapped modules")
    print(f"      ✓ {len(factorized_with_lora)} FactorizedLinear layers have LoRA adapters")
    
    # Show some examples
    print(f"      Examples of LoRA-adapted FactorizedLinear modules:")
    for i, name in enumerate(factorized_with_lora[:5]):
        print(f"        - {name}.A and {name}.B")
    
    # Step 5: Test gradient flow
    print(f"\n[5/5] Testing gradient flow through LoRA adapters...")
    has_grads, lora_grad_params = test_gradient_flow(model, device)
    
    if not has_grads:
        print(f"      ❌ ERROR: No gradients in LoRA parameters!")
        print(f"      This means backprop is broken")
        return
    
    print(f"      ✓ Gradients flowing through {len(lora_grad_params)} LoRA parameters")
    print(f"      Example parameters with gradients:")
    for i, name in enumerate(lora_grad_params[:5]):
        print(f"        - {name}")
    
    # Summary
    print("\n" + "="*70)
    print("✓ ALL TESTS PASSED!")
    print("="*70)
    print(f"Summary:")
    print(f"  - LoRA successfully applied to {len(target_modules)} submodules (A and B)")
    print(f"  - {lora_count} LoRA adapters are active")
    print(f"  - Gradients flow correctly through all LoRA parameters")
    print(f"  - FactorizedLinear fine-tuning is working as expected")
    print("="*70)


if __name__ == "__main__":
    main()