#!/usr/bin/env python3
"""
Quick test to verify that JSON sanitization works correctly for lm-eval results.

This test:
1. Creates a mock result dictionary with numpy types (like lm-eval returns)
2. Tests the sanitize_json() function
3. Verifies JSON serialization works without errors
4. Validates type conversions are correct

Usage:
    python scripts/test_json_sanitization.py
"""

import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from urank.eval.harness import sanitize_json


def test_sanitize_json():
    """Test that numpy types are properly sanitized for JSON."""
    
    print("=" * 70)
    print("Testing JSON Sanitization for lm-eval Results")
    print("=" * 70)
    
    # Create mock lm-eval results with numpy types
    mock_results = {
        "results": {
            "wikitext": {
                "word_perplexity": np.float32(22.5),
                "byte_perplexity": np.float64(19.8),
                "bits_per_byte": np.float32(4.32),
                "samples": np.int64(1000),
                "converged": np.bool_(True),
            },
            "arc_easy": {
                "acc": np.float32(0.742),
                "acc_norm": np.float32(0.718),
                "samples": np.int64(2376),
            },
            "hellaswag": {
                "acc": np.float64(0.573),
                "acc_norm": np.float64(0.751),
                "samples": np.int64(10042),
            }
        },
        "config": {
            "model": "gpt2_compressed",
            "batch_size": np.int64(8),
            "device": "cuda:0",
        },
        "versions": {
            "lm-eval": "0.4.0",
            "numpy_version": str(np.__version__),
        }
    }
    
    print("\n[1/4] Created mock results with numpy types")
    print(f"    Type of word_perplexity: {type(mock_results['results']['wikitext']['word_perplexity'])}")
    print(f"    Type of samples: {type(mock_results['results']['wikitext']['samples'])}")
    print(f"    Type of converged: {type(mock_results['results']['wikitext']['converged'])}")
    
    # Test direct JSON serialization (should fail)
    print("\n[2/4] Testing direct JSON serialization (should fail)...")
    try:
        json.dumps(mock_results)
        print("    ✗ ERROR: Direct serialization should have failed!")
        return False
    except TypeError as e:
        print(f"    ✓ Expected failure: {str(e)[:60]}...")
    
    # Sanitize the results
    print("\n[3/4] Sanitizing results with sanitize_json()...")
    sanitized = sanitize_json(mock_results)
    
    # Check types after sanitization
    print("    Type conversions:")
    print(f"      np.float32 → {type(sanitized['results']['wikitext']['word_perplexity']).__name__}")
    print(f"      np.int64 → {type(sanitized['results']['wikitext']['samples']).__name__}")
    print(f"      np.bool_ → {type(sanitized['results']['wikitext']['converged']).__name__}")
    
    # Verify correct Python types
    word_ppl = sanitized['results']['wikitext']['word_perplexity']
    samples = sanitized['results']['wikitext']['samples']
    converged = sanitized['results']['wikitext']['converged']
    
    if not isinstance(word_ppl, float):
        print(f"    ✗ ERROR: word_perplexity is {type(word_ppl)}, expected float")
        return False
    
    if not isinstance(samples, int):
        print(f"    ✗ ERROR: samples is {type(samples)}, expected int")
        return False
    
    if not isinstance(converged, bool):
        print(f"    ✗ ERROR: converged is {type(converged)}, expected bool")
        return False
    
    print("    ✓ All types converted correctly")
    
    # Test JSON serialization after sanitization
    print("\n[4/4] Testing JSON serialization after sanitization...")
    try:
        json_str = json.dumps(sanitized, indent=2)
        print(f"    ✓ Successfully serialized {len(json_str)} characters")
        
        # Try parsing it back
        parsed = json.loads(json_str)
        print(f"    ✓ Successfully parsed back")
        
        # Verify values are preserved
        if abs(parsed['results']['wikitext']['word_perplexity'] - 22.5) > 0.001:
            print("    ✗ ERROR: Value changed during serialization")
            return False
        
        print("    ✓ Values preserved correctly")
        
    except Exception as e:
        print(f"    ✗ ERROR: Serialization failed: {e}")
        return False
    
    # Test writing to file
    print("\n[5/5] Testing file write...")
    test_file = Path("/tmp/test_lm_eval_results.json")
    try:
        with open(test_file, "w") as f:
            json.dump(sanitized, f, indent=2)
        
        print(f"    ✓ Successfully wrote to {test_file}")
        
        # Read it back
        with open(test_file) as f:
            loaded = json.load(f)
        
        print(f"    ✓ Successfully loaded from file")
        
        # Verify
        if loaded['results']['arc_easy']['acc'] != sanitized['results']['arc_easy']['acc']:
            print("    ✗ ERROR: Data changed after write/read cycle")
            return False
        
        print("    ✓ Data integrity verified")
        
        # Clean up
        test_file.unlink()
        
    except Exception as e:
        print(f"    ✗ ERROR: File operations failed: {e}")
        return False
    
    return True


def main():
    success = test_sanitize_json()
    
    print("\n" + "=" * 70)
    if success:
        print("✓ ALL TESTS PASSED!")
        print("=" * 70)
        print("Summary:")
        print("  - numpy types correctly converted to Python natives")
        print("  - JSON serialization works without errors")
        print("  - File write/read cycle preserves data integrity")
        print("  - lm-eval results can now be saved to JSON")
        print("=" * 70)
        return 0
    else:
        print("✗ TESTS FAILED")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    sys.exit(main())