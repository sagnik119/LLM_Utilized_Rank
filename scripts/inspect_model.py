#!/usr/bin/env python3
"""
Inspect the structure of a compressed model to debug module names.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from transformers import AutoModelForCausalLM

def print_model_structure(model, max_depth=3, prefix="", depth=0):
    """Print model structure with module names and types."""
    if depth > max_depth:
        return
    
    for name, module in model.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        module_type = type(module).__name__
        
        # Print this module
        indent = "  " * depth
        print(f"{indent}{name}: {module_type}")
        
        # Show some details for specific types
        if hasattr(module, 'weight'):
            print(f"{indent}  └─ weight shape: {module.weight.shape}")
        
        # Recurse
        print_model_structure(module, max_depth, full_name, depth + 1)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Model path")
    parser.add_argument("--depth", type=int, default=3, help="Max depth to print")
    args = parser.parse_args()
    
    print(f"Loading model: {args.model}\n")
    
    # Try loading normally first
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model)
        print("=" * 60)
        print("Model loaded with AutoModelForCausalLM")
        print("=" * 60)
        print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}\n")
        
        print("First few layers structure:")
        print("-" * 60)
        
        # Print first transformer block in detail
        if hasattr(model, 'transformer'):
            print("\ntransformer.h[0] structure:")
            print_model_structure(model.transformer.h[0], max_depth=args.depth)
            
        print("\n" + "=" * 60)
        print("All module names (full paths):")
        print("=" * 60)
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Sequential):
                print(f"Sequential: {name} (length={len(module)})")
                for i, child in enumerate(module):
                    print(f"  [{i}] {type(child).__name__}", end="")
                    if hasattr(child, 'weight'):
                        print(f" - weight: {child.weight.shape}", end="")
                    print()
            elif isinstance(module, torch.nn.Linear):
                print(f"Linear: {name} - {module.in_features} -> {module.out_features}")
                
    except Exception as e:
        print(f"Error loading: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()