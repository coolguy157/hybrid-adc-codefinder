import json
import os
import networkx as nx
import numpy as np
from tqdm import tqdm
from qiskit.quantum_info import Pauli

# --- CONFIGURATION ---
INPUT_DIR = './preprocessed_data'
OUTPUT_FILE = './cws_results.jsonl'

# Optimization: Skip graphs with N > 11 unless you have a powerful C++ clique solver
MAX_N_TO_SEARCH = 10 

# --- CACHE ---
EAD_ERROR_SETS_CACHE = {}

def get_ead_error_set(n):
    """
    Generates the Amplitude Damping Pauli Error Set.
    Includes Identity, all weight-1 Paulis, and specific weight-2 pairs (XX, XY, YY).
    """
    if n in EAD_ERROR_SETS_CACHE: return EAD_ERROR_SETS_CACHE[n]
    
    error_set = []
    # Identity
    error_set.append(Pauli("I" * n))
    
    # Weight 1
    for i in range(n):
        for p in ['X', 'Y', 'Z']:
            p_str = ['I'] * n
            p_str[i] = p
            error_set.append(Pauli("".join(p_str)))
            
    # Weight 2 (Amplitude Damping specific)
    for i in range(n):
        for j in range(i + 1, n):
            for p_pair in [('X', 'X'), ('X', 'Y'), ('Y', 'Y')]:
                p_str = ['I'] * n
                p_str[i] = p_pair[0]
                p_str[j] = p_pair[1]
                error_set.append(Pauli("".join(p_str)))
                
    EAD_ERROR_SETS_CACHE[n] = error_set
    return error_set

def get_induced_error(adj_matrix, pauli):
    """
    Maps a quantum Pauli error to a classical error pattern based on Graph G.
    Formula: Cl_G(Z^v X^u) = v + (Gamma @ u)  (mod 2)
    """
    # Qiskit Pauli stores .z and .x as boolean arrays.
    # Qiskit uses Index 0 = Qubit 0. 
    # Adjacency matrices use Row 0 = Node 0.
    # These ALIGN correctly. We do NOT need to reverse them.
    
    z = pauli.z.astype(int)
    x = pauli.x.astype(int)
    
    # v + (Adj * u)
    # The adjacency matrix acts on the X-component (u)
    induced = (z + (adj_matrix @ x)) % 2
    return tuple(induced)

def build_conflict_graph(n, adj_matrix, error_set):
    """
    Constructs the graph where Independent Sets represent quantum codes.
    Vertices: Classical bitstrings c in {0,1}^n
    Edges: Connect c1, c2 if (c1 + c2) matches an undetectable error pattern.
    """
    # 1. Calculate all induced classical errors
    # We need the set of "bad differences" D = { cl(E_i) + cl(E_j) }
    classical_errors = [get_induced_error(adj_matrix, e) for e in error_set]
    
    bad_diffs = set()
    for e1 in classical_errors:
        for e2 in classical_errors:
            # diff = e1 + e2 (mod 2)
            diff = tuple((a ^ b) for a, b in zip(e1, e2))
            bad_diffs.add(diff)
    
    # Remove 0 from bad_diffs (c + c = 0 is always allowed)
    bad_diffs.discard(tuple([0]*n))
    
    # 2. Build the Conflict Graph
    G = nx.Graph()
    G.add_nodes_from(range(2**n))
    
    # Add edges based on conflicts
    # Optimization: Iterate nodes and bad_diffs
    
    # Convert bad_diffs to integers for fast XOR operations
    bad_diff_ints = []
    for vec in bad_diffs:
        # Convert tuple (0,1,1...) to integer (Little Endian to match Qiskit-style if needed, 
        # or just consistent Big Endian for graph logic. As long as it is consistent.)
        # Here we treat index 0 as LSB to match the 'tuple' output.
        val = 0
        for i, bit in enumerate(vec):
            if bit:
                val |= (1 << i)
        bad_diff_ints.append(val)
        
    for u in range(2**n):
        for d in bad_diff_ints:
            v = u ^ d
            if v > u: # Avoid duplicates and self-loops
                G.add_edge(u, v)
                
    return G

def main():
    print(f"--- Starting CWS (Clique) Search ---")
    
    processed_graphs = set()
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r') as f:
            for line in f:
                try:
                    d = json.loads(line)
                    processed_graphs.add(d.get('graph6'))
                except: pass
        print(f"Skipping {len(processed_graphs)} previously processed graphs.")

    for n in range(4, MAX_N_TO_SEARCH + 1):
        input_path = os.path.join(INPUT_DIR, f'candidates_n{n}.jsonl')
        if not os.path.exists(input_path): continue
        
        print(f"\nProcessing n={n}...")
        error_set = get_ead_error_set(n)
        
        with open(input_path, 'r') as f_in, open(OUTPUT_FILE, 'a') as f_out:
            for line in tqdm(f_in):
                rec = json.loads(line)
                g6 = rec['graph6_string']
                
                if g6 in processed_graphs: continue
                
                adj = np.array(rec['adjacency_matrix'])
                
                # 1. Build Conflict Graph
                G_conflict = build_conflict_graph(n, adj, error_set)
                
                # 2. Find Max Independent Set
                # Using approximation for speed. For exact results on N=10, use exact solver.
                G_comp = nx.complement(G_conflict)
                
                # Heuristic clique finder (fast)
                clique = nx.algorithms.clique.max_weight_clique(G_comp, weight=None)
                
                code_size = clique[1]
                codewords = clique[0]
                
                # 3. Save if "Good" (Size > 2)
                if code_size > 2:
                    res = {
                        "n": n,
                        "d_seed": rec['d'],
                        "code_size": code_size,
                        "log2_size": float(np.log2(code_size)),
                        "graph6": g6,
                        "codewords": list(codewords)
                    }
                    f_out.write(json.dumps(res) + '\n')
                    f_out.flush()

if __name__ == '__main__':
    main()