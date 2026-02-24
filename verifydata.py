import json
import os
import numpy as np
from collections import defaultdict

PUBLISHED_TOTALS = {
    (9, 2): 416, (9, 3): 69, (9, 4): 8,
    (10, 2): 2618, (10, 3): 577, (10, 4): 120,
    (11, 2): 27445, (11, 3): 11202, (11, 4): 2506
}

def main(processed_dir):
    print(f"--- Verifying data in '{processed_dir}' ---")
    counts = defaultdict(int)
    
    if not os.path.exists(processed_dir):
        print("Directory not found.")
        return

    for filename in os.listdir(processed_dir):
        if filename.endswith(".jsonl"):
            path = os.path.join(processed_dir, filename)
            with open(path, 'r') as f:
                for line in f:
                    rec = json.loads(line)
                    # Sanity check matrix dimensions
                    n = rec['n']
                    adj = np.array(rec['adjacency_matrix'])
                    if adj.shape != (n, n):
                        print(f"Error: Dimension mismatch in n={n}")
                    counts[(n, rec['d'])] += 1

    mismatches = 0
    for (n, d), expected in PUBLISHED_TOTALS.items():
        actual = counts.get((n, d), 0)
        if actual != expected:
            print(f"⚠️  Mismatch n={n} d={d}: Expected {expected}, Got {actual}")
            mismatches += 1
        else:
            print(f"✅ n={n} d={d}: {actual} records (Correct)")

    if mismatches == 0:
        print("\nAll checks passed.")

if __name__ == '__main__':
    main('./preprocessed_data')