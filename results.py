import json
import os
import math

RESULTS_FILE = './cws_results.jsonl'

def load_results(filepath):
    if not os.path.exists(filepath): return []
    codes = []
    with open(filepath, 'r') as f:
        for line in f:
            try:
                codes.append(json.loads(line))
            except: continue
    return codes

def print_table(codes):
    print(f"{'n':<4} | {'Size':<6} | {'Log2(S)':<8} | {'d_seed':<6} | {'Graph6'}")
    print("-" * 50)
    
    # Sort by n, then size
    codes.sort(key=lambda x: (x['n'], -x['code_size']))
    
    for c in codes:
        # Simple heuristic: if Log2(S) is integer, it's a stabilizer code equivalent
        # if float, it's strictly non-additive CWS (Success!)
        size = c['code_size']
        log_s = c['log2_size']
        
        # Highlight non-additive codes
        is_float = not log_s.is_integer()
        mark = "*" if is_float else " "
        
        print(f"{c['n']:<4} | {size:<6} | {log_s:<8.2f}{mark} | {c['d_seed']:<6} | {c['graph6'][:10]}...")

if __name__ == "__main__":
    data = load_results(RESULTS_FILE)
    print_table(data)