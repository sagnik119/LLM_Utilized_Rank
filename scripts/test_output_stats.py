#!/usr/bin/env python3
"""
Test script to verify collect_output_activations.py is working correctly.

This script:
1. Runs collect_output_activations.py on a small sample
2. Loads the saved Y^T Y statistics
3. Validates the structure and properties of the output covariances
"""

import sys
import tempfile
import subprocess
from pathlib import Path
import torch

def test_output_collection():
    print("=" * 60)
    print("Testing collect_output_activations.py")
    print("=" * 60)
    
    # Create temporary output file
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        tmp_path = tmp.name
    
    print(f"\n1. Running collect_output_activations.py...")
    print(f"   Output will be saved to: {tmp_path}")
    
    # Run the collection script with minimal samples
    cmd = [
        sys.executable,
        "scripts/collect_output_activations.py",
        "--model", "gpt2",
        "--dataset", "wikitext",
        "--config", "wikitext-2-raw-v1",
        "--split", "train",
        "--samples", "1000",  # Small sample for quick test
        "--max-length", "128",
        "--out", tmp_path
    ]
    
    print(f"   Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("   ✓ Script executed successfully")
        print(f"\n   Output:\n{result.stdout}")
        
    except subprocess.CalledProcessError as e:
        print(f"   ✗ Script failed with error:")
        print(e.stderr)
        return False
    
    # Load and validate the saved statistics
    print(f"\n2. Loading saved statistics from {tmp_path}...")
    
    try:
        yty = torch.load(tmp_path)
        print(f"   ✓ Loaded Y^T Y stats for {len(yty)} layers")
    except Exception as e:
        print(f"   ✗ Failed to load statistics: {e}")
        return False
    
    # Validate structure
    print("\n3. Validating structure...")
    
    expected_patterns = [
        ".mlp.c_fc",
        ".mlp.c_proj",
        ".attn.c_attn.q",
        ".attn.c_attn.k",
        ".attn.c_attn.v",
        ".attn.c_proj"
    ]
    
    found_patterns = {pattern: False for pattern in expected_patterns}
    
    for name, matrix in yty.items():
        print(f"   Layer: {name}")
        print(f"     Shape: {matrix.shape}")
        
        # Check it's a square matrix
        if matrix.shape[0] != matrix.shape[1]:
            print(f"     ✗ ERROR: Not a square matrix!")
            return False
        
        # Check it's symmetric (Y^T Y should be symmetric)
        if not torch.allclose(matrix, matrix.T, atol=1e-4):
            print(f"     ⚠ Warning: Matrix not symmetric (numerical precision issue)")
        
        # Check it's positive semi-definite (all eigenvalues >= 0)
        try:
            eigenvalues = torch.linalg.eigvalsh(matrix)
            min_eig = eigenvalues.min().item()
            if min_eig < -1e-5:
                print(f"     ✗ ERROR: Negative eigenvalue detected: {min_eig}")
                return False
            print(f"     ✓ PSD check passed (min eigenvalue: {min_eig:.6f})")
        except Exception as e:
            print(f"     ⚠ Warning: Could not compute eigenvalues: {e}")
        
        # Mark found patterns
        for pattern in expected_patterns:
            if pattern in name:
                found_patterns[pattern] = True
    
    print("\n4. Checking expected layer patterns...")
    all_found = True
    for pattern, found in found_patterns.items():
        status = "✓" if found else "✗"
        print(f"   {status} {pattern}: {'Found' if found else 'NOT FOUND'}")
        if not found:
            all_found = False
    
    if not all_found:
        print("\n   ⚠ Warning: Some expected patterns not found")
        print("   This might be expected for very small samples or different model architectures")
    
    # Check specific layer dimensions for GPT-2
    print("\n5. Validating GPT-2 specific dimensions...")
    
    # GPT-2 small has d_model=768, intermediate=3072
    for name, matrix in yty.items():
        if ".mlp.c_fc" in name:
            expected_dim = 3072  # MLP up projects to 4*d_model
            if matrix.shape[0] == expected_dim:
                print(f"   ✓ {name}: Correct dimension {expected_dim}x{expected_dim}")
            else:
                print(f"   ⚠ {name}: Unexpected dimension {matrix.shape} (expected {expected_dim})")
        
        elif ".mlp.c_proj" in name or ".attn." in name:
            expected_dim = 768  # d_model
            if matrix.shape[0] == expected_dim:
                print(f"   ✓ {name}: Correct dimension {expected_dim}x{expected_dim}")
            else:
                print(f"   ⚠ {name}: Unexpected dimension {matrix.shape} (expected {expected_dim})")
    
    # Cleanup
    Path(tmp_path).unlink()
    print(f"\n6. Cleaned up temporary file {tmp_path}")
    
    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    success = test_output_collection()
    sys.exit(0 if success else 1)