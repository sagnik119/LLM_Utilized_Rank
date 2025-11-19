#!/usr/bin/env python3
"""
Test script to verify search_ranks_energy.py works correctly.

This script:
1. Runs collect_output_activations.py on a small sample
2. Runs search_ranks_energy.py on the collected statistics
3. Validates the output JSON structure and energy preservation guarantees
"""

import sys
import tempfile
import subprocess
import json
from pathlib import Path
import torch


def test_energy_rank_search():
    print("=" * 60)
    print("Testing search_ranks_energy.py")
    print("=" * 60)
    
    # Create temporary files
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as stats_file, \
         tempfile.NamedTemporaryFile(suffix=".json", delete=False) as ranks_file:
        stats_path = stats_file.name
        ranks_path = ranks_file.name
    
    # Step 1: Collect output activations
    print("\n1. Collecting output activations...")
    print(f"   Stats will be saved to: {stats_path}")
    
    cmd_collect = [
        sys.executable,
        "scripts/collect_output_activations.py",
        "--model", "gpt2",
        "--dataset", "wikitext",
        "--config", "wikitext-2-raw-v1",
        "--split", "train",
        "--samples", "1000",
        "--max-length", "128",
        "--out", stats_path
    ]
    
    try:
        result = subprocess.run(cmd_collect, check=True, capture_output=True, text=True)
        print("   ✓ Activations collected successfully")
    except subprocess.CalledProcessError as e:
        print(f"   ✗ Collection failed: {e.stderr}")
        return False
    
    # Step 2: Search ranks based on energy
    print("\n2. Searching ranks based on energy...")
    print(f"   Ranks will be saved to: {ranks_path}")
    
    for energy in [0.95, 0.99]:
        print(f"\n   Testing with energy threshold: {energy}")
        
        cmd_search = [
            sys.executable,
            "scripts/search_ranks_energy.py",
            "--stats", stats_path,
            "--energy", str(energy),
            "--out", ranks_path
        ]
        
        try:
            result = subprocess.run(cmd_search, check=True, capture_output=True, text=True)
            print(f"   ✓ Rank search completed for energy={energy}")
            print(f"\n   Output:\n{result.stdout}")
        except subprocess.CalledProcessError as e:
            print(f"   ✗ Rank search failed: {e.stderr}")
            return False
        
        # Step 3: Validate output JSON
        print(f"\n3. Validating output JSON for energy={energy}...")
        
        try:
            with open(ranks_path) as f:
                ranks = json.load(f)
            print(f"   ✓ Loaded ranks for {len(ranks)} layers")
        except Exception as e:
            print(f"   ✗ Failed to load JSON: {e}")
            return False
        
        # Validate structure
        print("\n4. Validating structure...")
        
        required_fields = ["r", "d_out", "energy", "compression_ratio"]
        
        for layer_name, layer_data in ranks.items():
            # Check all required fields present
            missing = [f for f in required_fields if f not in layer_data]
            if missing:
                print(f"   ✗ Layer {layer_name} missing fields: {missing}")
                return False
            
            # Check types
            if not isinstance(layer_data["r"], int):
                print(f"   ✗ Layer {layer_name}: r should be int, got {type(layer_data['r'])}")
                return False
            
            if not isinstance(layer_data["d_out"], int):
                print(f"   ✗ Layer {layer_name}: d_out should be int, got {type(layer_data['d_out'])}")
                return False
            
            # Check ranges
            r = layer_data["r"]
            d_out = layer_data["d_out"]
            achieved_energy = layer_data["energy"]
            compression = layer_data["compression_ratio"]
            
            if not (1 <= r <= d_out):
                print(f"   ✗ Layer {layer_name}: invalid rank {r} (d_out={d_out})")
                return False
            
            if not (energy <= achieved_energy <= 1.0):
                print(f"   ✗ Layer {layer_name}: energy {achieved_energy} < target {energy}")
                return False
            
            if not (0.0 < compression <= 1.0):
                print(f"   ✗ Layer {layer_name}: invalid compression ratio {compression}")
                return False
            
            # Check compression ratio matches r/d_out
            expected_compression = r / d_out
            if abs(compression - expected_compression) > 1e-6:
                print(f"   ✗ Layer {layer_name}: compression ratio mismatch")
                return False
        
        print(f"   ✓ All {len(ranks)} layers validated")
        
        # Step 5: Verify energy preservation
        print("\n5. Verifying energy preservation with actual Y^T Y matrices...")
        
        yty_map = torch.load(stats_path, map_location="cpu")
        
        mismatches = []
        for layer_name, layer_data in ranks.items():
            if layer_name not in yty_map:
                print(f"   ⚠ Warning: {layer_name} not in Y^T Y stats")
                continue
            
            C = yty_map[layer_name]
            r = layer_data["r"]
            
            # Recompute eigenvalues
            eigvals = torch.linalg.eigvalsh(C)
            eigvals = torch.sort(eigvals, descending=True).values
            eigvals = torch.clamp(eigvals, min=0.0)
            
            total = eigvals.sum()
            cum_energy = eigvals[:r].sum() / total
            
            reported_energy = layer_data["energy"]
            
            if abs(cum_energy - reported_energy) > 1e-3:
                mismatches.append({
                    "layer": layer_name,
                    "computed": float(cum_energy),
                    "reported": reported_energy,
                    "diff": abs(cum_energy - reported_energy)
                })
        
        if mismatches:
            print(f"   ✗ Energy mismatches found in {len(mismatches)} layers:")
            for m in mismatches[:5]:  # Show first 5
                print(f"      {m['layer']}: computed={m['computed']:.4f}, reported={m['reported']:.4f}")
            return False
        
        print("   ✓ Energy values verified against Y^T Y matrices")
        
        # Step 6: Summary statistics
        print("\n6. Summary statistics:")
        
        ranks_list = [layer_data["r"] for layer_data in ranks.values()]
        d_outs = [layer_data["d_out"] for layer_data in ranks.values()]
        energies = [layer_data["energy"] for layer_data in ranks.values()]
        compressions = [layer_data["compression_ratio"] for layer_data in ranks.values()]
        
        print(f"   Layers: {len(ranks)}")
        print(f"   Rank range: [{min(ranks_list)}, {max(ranks_list)}]")
        print(f"   Average rank: {sum(ranks_list)/len(ranks_list):.1f}")
        print(f"   Average compression: {sum(compressions)/len(compressions):.3f}")
        print(f"   Energy range: [{min(energies):.4f}, {max(energies):.4f}]")
        print(f"   All energies >= {energy}: {all(e >= energy for e in energies)}")
    
    # Cleanup
    Path(stats_path).unlink()
    Path(ranks_path).unlink()
    print(f"\n7. Cleaned up temporary files")
    
    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    success = test_energy_rank_search()
    sys.exit(0 if success else 1)