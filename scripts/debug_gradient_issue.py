#!/usr/bin/env python3
"""
Debug script to identify gradient flow issues in compressed model with LoRA.

This script tests:
1. Model loading with factorized layers
2. LoRA application
3. Forward pass and loss computation
4. Gradient requirements
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from transformers import AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType

from src.urank.modeling_gpt2_compressed import GPT2CompressedLMHeadModel


def main():
    print("=" * 60)
    print("Debugging Gradient Flow in Compressed Model + LoRA")
    print("=" * 60)
    
    # Load model
    print("\n1. Loading compressed model...")
    model = GPT2CompressedLMHeadModel.from_pretrained("ckpts/gpt2_compressed")
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"   Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Check what's trainable before LoRA
    print("\n2. Before LoRA - checking trainable parameters:")
    trainable_before = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Trainable: {trainable_before:,}")
    
    # Apply LoRA
    print("\n3. Applying LoRA...")
    try:
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=["A", "B"],  # Target FactorizedLinear sub-modules
            bias="none"
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    except Exception as e:
        print(f"   ERROR applying LoRA: {e}")
        print("\n4. Trying alternative target modules...")
        
        # List all module names
        print("   Available module types:")
        module_types = set()
        for name, module in model.named_modules():
            module_type = type(module).__name__
            if module_type in ["Linear", "FactorizedLinear", "Conv1D"]:
                module_types.add(module_type)
                if len(module_types) <= 5:  # Show first few examples
                    print(f"     {name}: {module_type}")
        
        print(f"\n   Found module types: {module_types}")
        sys.exit(1)
    
    # Check what's trainable after LoRA
    print("\n4. After LoRA - checking trainable parameters:")
    trainable_params = []
    for name, p in model.named_parameters():
        if p.requires_grad:
            trainable_params.append((name, p.shape))
    
    print(f"   Total trainable: {len(trainable_params)}")
    print("\n   First 10 trainable parameters:")
    for name, shape in trainable_params[:10]:
        print(f"     {name}: {shape}")
    
    # Test forward pass
    print("\n5. Testing forward pass...")
    tokenizer = AutoTokenizer.from_pretrained("ckpts/gpt2_compressed")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    text = "Hello world, this is a test."
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    try:
        outputs = model(**inputs, labels=inputs["input_ids"])
        print(f"   ✓ Forward pass successful")
        print(f"   Loss: {outputs.loss.item():.4f}")
        print(f"   Loss requires_grad: {outputs.loss.requires_grad}")
        print(f"   Loss grad_fn: {outputs.loss.grad_fn}")
    except Exception as e:
        print(f"   ✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Test backward pass
    print("\n6. Testing backward pass...")
    try:
        outputs.loss.backward()
        print(f"   ✓ Backward pass successful")
        
        # Check which parameters got gradients
        params_with_grad = []
        for name, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                params_with_grad.append(name)
        
        print(f"   Parameters with gradients: {len(params_with_grad)}")
        if params_with_grad:
            print("\n   First 5 parameters with gradients:")
            for name in params_with_grad[:5]:
                print(f"     ✓ {name}")
        else:
            print("   ✗ ERROR: No parameters received gradients!")
            
    except Exception as e:
        print(f"   ✗ Backward pass failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Additional diagnostics
        print("\n   Checking parameter dependencies:")
        for name, p in list(model.named_parameters())[:10]:
            print(f"     {name}:")
            print(f"       requires_grad: {p.requires_grad}")
            print(f"       is_leaf: {p.is_leaf}")
            print(f"       grad_fn: {p.grad_fn}")
        
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("✓ All checks passed! Model is ready for training.")
    print("=" * 60)


if __name__ == "__main__":
    main()