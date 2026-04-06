#!/usr/bin/env python
"""
Demonstration of checkpoint and resume functionality.
This script shows:
1. How to clear a checkpoint and restart from scratch
2. How resuming works automatically
"""

import json
from pathlib import Path
import sys

def clear_checkpoint(n, error_set_type):
    """Delete checkpoint file."""
    checkpoint_path = Path("checkpoints") / f"search_n{n}_{error_set_type}.json"
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        print(f"✓ Cleared checkpoint: {checkpoint_path}")
    else:
        print(f"✗ No checkpoint to clear: {checkpoint_path}")

def show_checkpoint_status(n, error_set_type):
    """Display current checkpoint status."""
    checkpoint_path = Path("checkpoints") / f"search_n{n}_{error_set_type}.json"
    
    print("\n" + "=" * 80)
    print(f"CHECKPOINT STATUS: n={n}, error_set={error_set_type}")
    print("=" * 80)
    
    if not checkpoint_path.exists():
        print("Status: NO CHECKPOINT (will start fresh)")
        return None
    
    data = json.load(open(checkpoint_path))
    processed = len(data['processed_combinations'])
    results = len(data['results'])
    
    print(f"Status: CHECKPOINT EXISTS")
    print(f"  - Processed combinations: {processed}")
    print(f"  - Results found: {results}")
    print(f"  - File size: {checkpoint_path.stat().st_size / 1024:.1f} KB")
    
    if results > 0:
        best_clique = max((r['max_clique_size'] for r in data['results']), default=0)
        print(f"  - Best clique size: {best_clique}")
    
    return data

def demo_resume_capability():
    """Demonstrate the checkpoint and resume feature."""
    
    print("\n" + "=" * 80)
    print("CHECKPOINT & RESUME DEMONSTRATION")
    print("=" * 80)
    
    n = 4
    error_set = "E1"
    
    print("\n1. CHECK CURRENT CHECKPOINT STATUS:")
    show_checkpoint_status(n, error_set)
    
    print("\n\n2. TO TEST RESUME FUNCTIONALITY:")
    print("   a) Edit ad_hybrid_code_search.py main() to set checkpoint_interval=0")
    print("      (This disables automatic checkpointing)")
    print("   b) Run the search, then press Ctrl+C to interrupt partway through")
    print("   c) The remaining combinations will be skipped next time")
    print("   d) Change checkpoint_interval back to 50 and run again")
    print("   e) The search will resume from where it left off!")
    
    print("\n\n3. TO START FRESH (CLEAR CHECKPOINT):")
    print("   Choose option from below:")
    print("   a) Press 'y' to clear n=4 E1 checkpoint and restart")
    print("   b) Press 'n' to leave checkpoint as-is")
    
    choice = input("\nClear checkpoint? (y/n): ").strip().lower()
    if choice == 'y':
        clear_checkpoint(n, error_set)
        print("\n✓ Next run will start from scratch")
    else:
        print("\n✓ Keeping checkpoint. Next run will resume from where it left off")
    
    print("\n" + "=" * 80)
    print("CHECKPOINT FEATURES:")
    print("=" * 80)
    print("""
    ✓ Automatic saving every 50 results (configurable via checkpoint_interval)
    ✓ Skip already-processed (graph, orientation) combinations on resume
    ✓ Merge new results with previous results
    ✓ Track progress: processed_combinations set + results list
    ✓ Zero data loss if process interrupted
    
    For larger searches (n≥6), checkpoint system prevents re-computing:
    - Each failed combination still counts toward progress
    - Clique graph construction cached by storing results
    - Can interrupt and resume without penalty
    """)

if __name__ == "__main__":
    demo_resume_capability()
