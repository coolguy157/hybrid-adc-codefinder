# verify_data.py
import json
import os
import numpy as np
import networkx as nx
from collections import defaultdict

# --- (PUBLISHED_TOTALS dictionary is unchanged) ---
PUBLISHED_TOTALS = {
    (1, 1): 1, (2, 1): 1, (2, 2): 1, (3, 1): 2, (3, 2): 1, (3, 3): 1, (4, 1): 3,
    (4, 2): 3, (4, 3): 1, (5, 1): 6, (5, 2): 4, (5, 3): 1, (5, 4): 1, (6, 1): 11,
    (6, 2): 13, (6, 3): 4, (6, 4): 1, (7, 1): 26, (7, 2): 29, (7, 3): 4, (8, 1): 59,
    (8, 2): 107, (8, 3): 11, (8, 4): 5, (9, 1): 182, (9, 2): 416, (9, 3): 69, (9, 4): 8,
    (10, 1): 675, (10, 2): 2618, (10, 3): 577, (10, 4): 120, (11, 1): 3990,
    (11, 2): 27445, (11, 3): 11202, (11, 4): 2506, (11, 5): 1, (12, 1): 45144,
    (12, 2): 615180, (12, 3): 467519, (12, 4): 195456, (12, 5): 63, (12, 6): 1,
}

# --- (verify_sanity and verify_connectivity are unchanged) ---
def verify_sanity(record):
    n = record['n']
    adj = np.array(record['adjacency_matrix'])
    assert n == adj.shape[0] == adj.shape[1], f"Dimension mismatch for n={n}"
    assert np.all(adj.T == adj), f"Matrix not symmetric for n={n}, d={record['d']}"
    assert np.all(np.diag(adj) == 0), f"Diagonal is not zero for n={n}, d={record['d']}"

def verify_connectivity(record):
    adj = np.array(record['adjacency_matrix'])
    is_connected = nx.is_connected(nx.from_numpy_array(adj))
    assert record['is_indecomposable'] == is_connected, \
        f"Connectivity mismatch for n={record['n']}, d={record['d']}"


def test_4_cycle_graph():
    """
    Unit test for the graph6 decoding.
    This test confirms that the string 'Ch' correctly produces the adjacency
    matrix for a 4-vertex path graph (0-1-2-3).
    """
    graph6_str = "Ch"
    
    graph6_bytes = graph6_str.encode('ascii')
    graph_obj = nx.from_graph6_bytes(graph6_bytes)
    adj_matrix = nx.to_numpy_array(graph_obj, nodelist=sorted(graph_obj.nodes()), dtype=int)
    
    # FIX: This is the correct, expected matrix for the graph6 string "Ch".
    expected_adj = np.array([
        [0, 1, 0, 0],
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 0]
    ])
    
    # This assertion will now pass.
    assert np.array_equal(adj_matrix, expected_adj), "Spot check for 'Ch' graph failed"
    print("✅ Manual spot check passed.")


# --- (Main verification logic is unchanged) ---
def main(processed_dir: str):
    all_records = []
    if not os.path.exists(processed_dir):
        print(f"❌ Error: Processed data directory not found at '{processed_dir}'")
        return

    print(f"\n--- Loading data from '{processed_dir}' ---")
    for filename in sorted(os.listdir(processed_dir)):
        if filename.endswith(".jsonl"):
            filepath = os.path.join(processed_dir, filename)
            with open(filepath, 'r') as f:
                for line in f:
                    all_records.append(json.loads(line))
    
    if not all_records:
        print("❌ Error: No records found in the processed data directory.")
        return
        
    print(f"Loaded {len(all_records)} total records for verification.\n")
    
    print("--- Running per-record checks ---")
    for record in all_records:
        verify_sanity(record)
        if record['n'] > 1:
            verify_connectivity(record)
            
    print("✅ Sanity and connectivity checks passed for all records.\n")

    print("--- Running aggregate count reconciliation ---")
    counts = defaultdict(int)
    for record in all_records:
        counts[(record['n'], record['d'])] += 1
    
    mismatches = 0
    for (n, d), expected_count in PUBLISHED_TOTALS.items():
        actual_count = counts.get((n, d), 0)
        if actual_count != expected_count:
            print(f"⚠️ Count mismatch for (n={n}, d={d}): Expected {expected_count}, Got {actual_count}")
            mismatches += 1

    if mismatches == 0:
        print("✅ Aggregate count reconciliation passed for all checked values.")
    else:
        print(f"Found {mismatches} mismatches in aggregate counts.")

    print("\nVerification complete.")


if __name__ == '__main__':
    test_4_cycle_graph()
    preprocessed_directory = './preprocessed_data'
    main(preprocessed_directory)