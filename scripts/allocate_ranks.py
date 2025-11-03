#!/usr/bin/env python3
"""
Apply rank allocation policies to discovered ranks.

This script takes the base ranks from search_ranks.py and applies a policy
to adjust ranks based on layer type (attention vs MLP, Q/K/V/O vs up/down).

Usage:
    python scripts/allocate_ranks.py --in-ranks ranks/gpt2.json \
        --policy qkv_heavier --out ranks/gpt2_qkv.json
"""

import argparse
import json
from urank.policies import get_policy, list_policies, load_policy_from_yaml


def main():
    parser = argparse.ArgumentParser(description="Apply rank allocation policy")
    parser.add_argument(
        "--in-ranks",
        required=True,
        help="Input JSON from search_ranks.py with base ranks per layer",
    )
    parser.add_argument(
        "--policy",
        type=str,
        default=None,
        help="Policy name (uniform, qkv_heavier, mlp_heavier)",
    )
    parser.add_argument(
        "--policy-file",
        type=str,
        default=None,
        help="Path to custom policy YAML file",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output JSON file with adjusted ranks",
    )
    parser.add_argument(
        "--list-policies",
        action="store_true",
        help="List available policies and exit",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print rank adjustments",
    )
    args = parser.parse_args()

    # List policies if requested
    if args.list_policies:
        policies = list_policies()
        print("Available policies:")
        for name, description in policies.items():
            print(f"  {name}: {description}")
        return

    # Validate arguments
    if not args.policy and not args.policy_file:
        parser.error("Either --policy or --policy-file must be specified")

    # Load policy
    if args.policy_file:
        print(f"Loading custom policy from: {args.policy_file}")
        policy = load_policy_from_yaml(args.policy_file)
    else:
        print(f"Using policy: {args.policy}")
        policy = get_policy(args.policy)

    print(f"Policy description: {policy.description}")

    # Load ranks
    print(f"Loading ranks from: {args.in_ranks}")
    with open(args.in_ranks, "r") as f:
        ranks = json.load(f)

    print(f"Processing {len(ranks)} layers...")

    # Apply policy to each layer
    adjustments = []
    for name, cfg in ranks.items():
        base_r = cfg["r"]
        adjusted_r = policy.assign(name, base_r)
        
        if adjusted_r != base_r:
            adjustments.append((name, base_r, adjusted_r))
        
        cfg["r"] = adjusted_r
        cfg["policy"] = policy.name

    # Save adjusted ranks
    with open(args.out, "w") as f:
        json.dump(ranks, f, indent=2)

    print(f"Saved policy-adjusted ranks to: {args.out}")

    # Print statistics
    print(f"\nAdjustment summary:")
    print(f"  Total layers: {len(ranks)}")
    print(f"  Adjusted layers: {len(adjustments)}")
    
    if adjustments and args.verbose:
        print(f"\nRank adjustments:")
        for name, base_r, adj_r in adjustments:
            change = ((adj_r - base_r) / base_r) * 100
            print(f"  {name}: {base_r} -> {adj_r} ({change:+.1f}%)")
    
    # Compute average ranks
    avg_base = sum(cfg.get("r", 0) for cfg in ranks.values()) / len(ranks)
    print(f"\nAverage rank: {avg_base:.1f}")


if __name__ == "__main__":
    main()