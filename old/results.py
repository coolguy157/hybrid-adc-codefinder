import json
import os

RESULTS_FILE = './all_unique_interesting_codes.jsonl'

def load_results(filepath):
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return []

    codes = []
    with open(filepath, 'r') as f:
        for line in f:
            if not line.strip(): continue
            try:
                data = json.loads(line)
                params = data.get('parameters', data)
                codes.append(params)
            except json.JSONDecodeError:
                continue
    return codes

def print_table(codes):
    if not codes:
        print("No codes found.")
        return

    # Filter: Show only Hybrid codes where both k and m are used
    hybrid_codes = [c for c in codes if c.get('k', 0) > 0 and c.get('m', 0) > 0]
    
    # Sort by n, then efficiency
    hybrid_codes.sort(key=lambda x: (x['n'], -(x['k'] + x['m'])))

    print(f"\n--- Found {len(hybrid_codes)} Interesting Hybrid Codes (k>0, m>0) ---")
    print(f"{'n':<5} | {'k':<5} | {'m':<5} | {'Total':<8} | {'d_seed':<8}")
    print("-" * 45)
    
    for c in hybrid_codes:
        total = c['k'] + c['m']
        print(f"{c['n']:<5} | {c['k']:<5} | {c['m']:<5} | {total:<8} | {c.get('d_seed', '?'):<8}")

if __name__ == "__main__":
    data = load_results(RESULTS_FILE)
    print_table(data)