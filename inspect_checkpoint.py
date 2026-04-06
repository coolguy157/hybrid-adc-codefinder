#!/usr/bin/env python
"""Quick script to inspect checkpoint file."""

import json
from pathlib import Path

checkpoint_file = Path("checkpoints/search_n4_E1.json")

if checkpoint_file.exists():
    data = json.load(open(checkpoint_file))
    print("=" * 80)
    print("CHECKPOINT FILE STRUCTURE")
    print("=" * 80)
    print(f"\nProcessed combinations: {len(data['processed_combinations'])}")
    print(f"Results saved: {len(data['results'])}")
    
    print("\n" + "=" * 80)
    print("SAMPLE CHECKPOINT DATA")
    print("=" * 80)
    
    print(f"\nFirst 3 processed combinations:")
    for i, combo in enumerate(data['processed_combinations'][:3]):
        print(f"  {i+1}. {combo}")
    
    print(f"\nFirst result:")
    result = data['results'][0]
    print(f"  - graph_key: {result['graph_key']}")
    print(f"  - orientation: {result['orientation']}")
    print(f"  - error_set_type: {result['error_set_type']}")
    print(f"  - max_clique_size: {result['max_clique_size']}")
    print(f"  - max_clique: {result['max_clique']}")
    print(f"  - admissible_vertex_count: {result['admissible_vertex_count']}")
    print(f"  - hybrid_kl_valid: {result['hybrid_kl_valid']}")
    
    print("\n" + "=" * 80)
    print("CHECKPOINT FILE INFO")
    print("=" * 80)
    print(f"File: {checkpoint_file}")
    print(f"Size: {checkpoint_file.stat().st_size / 1024:.1f} KB")
    
else:
    print(f"Checkpoint file not found: {checkpoint_file}")
