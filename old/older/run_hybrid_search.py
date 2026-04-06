import json
import os
import itertools
import numpy as np
from tqdm import tqdm
from qiskit.quantum_info import Pauli
from qiskit_qec.linear import rref

# --- CONFIGURATION ---
INPUT_DIR = './preprocessed_data'
OUTPUT_FILE = './all_unique_interesting_codes.jsonl'

# Filter: Only save if k+m > this value. 
# Prevents saving thousands of trivial codes.
BEST_KNOWN_STABILIZER_K = {
    (4, 2): 2, (5, 3): 1, (6, 3): 3, (6, 4): 0, 
    (7, 3): 4, (8, 3): 5, (9, 3): 6, (10, 3): 7
}

# --- CACHE ---
EAD_ERROR_SETS_CACHE = {}

# --- HELPER FUNCTIONS ---
def generate_full_group(generators, n):
    if not generators:
        return {Pauli("I" * n)}
    identity = Pauli("I" * n)
    group = {identity}
    for gen in generators:
        group.update({elem @ gen for elem in group})
    return group

def get_code_fingerprint(n, s_gens, t_gens, w_gens):
    s_fp = tuple(sorted([str(s) for s in generate_full_group(s_gens, n)]))
    t_fp = tuple(sorted([str(t) for t in generate_full_group(t_gens, n)]))
    w_fp = tuple(sorted([str(w) for w in generate_full_group(w_gens, n)]))
    return (s_fp, t_fp, w_fp)

def _generate_ead_error_set(n):
    error_set = [Pauli("I" * n)]
    for i in range(n):
        for p_char in ['X', 'Y', 'Z']:
            p_str = ['I'] * n
            p_str[i] = p_char
            error_set.append(Pauli("".join(p_str)))
    for i in range(n):
        for j in range(i + 1, n):
            for p_pair in [('X', 'X'), ('X', 'Y'), ('Y', 'Y')]:
                p_str = ['I'] * n
                p_str[i] = p_pair[0]
                p_str[j] = p_pair[1]
                error_set.append(Pauli("".join(p_str)))
    return error_set

def get_ead_error_set(n):
    if n not in EAD_ERROR_SETS_CACHE:
        EAD_ERROR_SETS_CACHE[n] = _generate_ead_error_set(n)
    return EAD_ERROR_SETS_CACHE[n]

def check_single_hkl_condition(t_a, t_b, W_i, W_j, E_k, E_l, S_prime_gens):
    P_eff = W_i @ t_a @ E_k @ E_l @ W_j @ t_b
    commutes_with_all = all(P_eff.commutes(s) for s in S_prime_gens)
    
    if t_a != t_b:
        if commutes_with_all: return False
    else:
        if W_i != W_j:
            if commutes_with_all: return False
        else:
            if not commutes_with_all: return False
    return True

def verify_hkl_for_amplitude_damping(n, S_prime_gens, translation_ops, word_ops):
    error_set = get_ead_error_set(n)
    t_ops = list(translation_ops)
    w_ops = list(word_ops)
    for t_a in t_ops:
        for t_b in t_ops:
            for W_i in w_ops:
                for W_j in w_ops:
                    for E_k in error_set:
                        for E_l in error_set:
                            if not check_single_hkl_condition(t_a, t_b, W_i, W_j, E_k, E_l, S_prime_gens):
                                return False
    return True

# --- GENERATOR LOGIC ---
def find_null_space_gf2(matrix):
    if matrix.size == 0:
        return np.array([[]])
    
    # FIX: qiskit_qec.linear.rref returns ONLY the matrix (Boolean ndarray)
    rref_matrix = rref(matrix)
    
    rows, cols = rref_matrix.shape
    pivot_cols = []
    
    # Manually identify pivot columns (first True in each row)
    for r in range(rows):
        nonzero_indices = np.nonzero(rref_matrix[r])[0]
        if len(nonzero_indices) > 0:
            pivot_cols.append(nonzero_indices[0])
            
    free_cols = [j for j in range(cols) if j not in pivot_cols]
    
    null_space_basis = np.zeros((len(free_cols), cols), dtype=int)
    for i, free_col_idx in enumerate(free_cols):
        null_space_basis[i, free_col_idx] = 1
        for r, pivot_col_idx in enumerate(pivot_cols):
            # In GF(2), x_pivot = -sum(coeffs) is the same as sum(coeffs)
            # Assigning Boolean (True/False) to Int array converts to 1/0 automatically
            null_space_basis[i, pivot_col_idx] = rref_matrix[r, free_col_idx]
            
    return null_space_basis

def symplectic_to_pauli_dict(symplectic_vec):
    n = len(symplectic_vec) // 2
    x_vec_str = "".join(map(str, symplectic_vec[:n]))
    z_vec_str = "".join(map(str, symplectic_vec[n:]))
    return {"x": x_vec_str, "z": z_vec_str}

def stream_impure_codes(n, input_file_path):
    """Yields potential impure code configurations from a file."""
    with open(input_file_path, 'r') as f:
        for line in f:
            graph_record = json.loads(line)
            adj_matrix = np.array(graph_record['adjacency_matrix'])
            initial_stabilizers = [Pauli((adj_matrix[i], np.eye(1, n, i, dtype=bool).flatten())) for i in range(n)]

            for k_total in range(1, n):
                for demoted_indices in itertools.combinations(range(n), k_total):
                    impure_stabilizer_indices = [i for i in range(n) if i not in demoted_indices]
                    impure_stabilizers = [initial_stabilizers[i] for i in impure_stabilizer_indices]
                    if not impure_stabilizers: continue
                    
                    symplectic_rows = [np.hstack((s.x.astype(int), s.z.astype(int))) for s in impure_stabilizers]
                    check_matrix = np.array(symplectic_rows, dtype=int)
                    null_space_basis = find_null_space_gf2(check_matrix)
                    logical_generators = [symplectic_to_pauli_dict(row) for row in null_space_basis]
                    
                    yield {
                        "n": n, "d_seed": graph_record.get('d', 0), "graph6_origin": graph_record['graph6_string'],
                        "k_total": k_total,
                        "stabilizer_generators": [
                            {"x": "".join(map(str, s.x.astype(int))), "z": "".join(map(str, s.z.astype(int)))} 
                            for s in impure_stabilizers
                        ],
                        "logical_generators": logical_generators
                    }

# --- MAIN EXECUTION ---
def main():
    print(f"--- Starting Unified Streaming Search ---")
    seen_fingerprints = set()
    
    # Resumption check
    if os.path.exists(OUTPUT_FILE):
        print(f"Resuming from {OUTPUT_FILE}...")
        with open(OUTPUT_FILE, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    params = data.get('parameters', data)
                    n = params['n']
                    s_gens = [Pauli(s) for s in data['impure_stabilizers']]
                    t_gens = [Pauli(t) for t in data['translation_generators']]
                    w_gens = [Pauli(w) for w in data['word_generators']]
                    seen_fingerprints.add(get_code_fingerprint(n, s_gens, t_gens, w_gens))
                except: continue
    
    valid_found = 0

    for n in range(1, 13):
        input_path = os.path.join(INPUT_DIR, f'candidates_n{n}.jsonl')
        if not os.path.exists(input_path): continue
        
        print(f"\n--- Streaming and Searching n={n} ---")
        
        impure_iterator = stream_impure_codes(n, input_path)
        
        with open(OUTPUT_FILE, 'a') as f_out:
            for impure_code in tqdm(impure_iterator, desc=f"Processing n={n}"):
                k_total = impure_code['k_total']
                S_prime_gens = [Pauli(p['z'] + p['x']) for p in impure_code['stabilizer_generators']]
                logical_gens = [Pauli(p['z'] + p['x']) for p in impure_code['logical_generators']]

                for m in range(0, k_total + 1):
                    k = k_total - m
                    if k < 0 or 2*k > len(logical_gens): continue
                    
                    # --- FILTER: Skip if not better than known stabilizer ---
                    d_seed = impure_code['d_seed']
                    baseline_k = BEST_KNOWN_STABILIZER_K.get((n, d_seed), 0)
                    if (k + m) <= baseline_k:
                        continue

                    # Combinatorial search over logical operators
                    for word_gen_indices in itertools.combinations(range(len(logical_gens)), 2*k):
                        remaining_indices = [i for i in range(len(logical_gens)) if i not in word_gen_indices]
                        for trans_gen_indices in itertools.combinations(remaining_indices, m):
                            
                            word_gens = [logical_gens[i] for i in word_gen_indices]
                            trans_gens = [logical_gens[i] for i in trans_gen_indices]
                            t_ops_full = generate_full_group(trans_gens, n)
                            w_ops_full = generate_full_group(word_gens, n)

                            if verify_hkl_for_amplitude_damping(n, S_prime_gens, t_ops_full, w_ops_full):
                                fingerprint = get_code_fingerprint(n, S_prime_gens, trans_gens, word_gens)
                                if fingerprint not in seen_fingerprints:
                                    valid_found += 1
                                    seen_fingerprints.add(fingerprint)
                                    
                                    result = {
                                        "parameters": {"n": n, "k": k, "m": m, "d_seed": d_seed},
                                        "origin_graph": impure_code['graph6_origin'],
                                        "impure_stabilizers": [str(s) for s in S_prime_gens],
                                        "translation_generators": [str(t) for t in trans_gens],
                                        "word_generators": [str(w) for w in word_gens]
                                    }
                                    f_out.write(json.dumps(result) + '\n')
                                    f_out.flush()

    print(f"\nSearch Complete. Found {valid_found} new codes.")

if __name__ == '__main__':
    main()