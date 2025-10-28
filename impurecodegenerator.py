# impure_code_generator_batch.py
import json
import numpy as np
import itertools
import os
from tqdm import tqdm
from qiskit.quantum_info import Pauli
# --- CHANGE 1: Import the robust rref function from qiskit-qec ---
from qiskit_qec.linear import rref

# --- CHANGE 2: The custom rref_gf2 function is now REMOVED ---

# --- The find_null_space_gf2 function is now updated to use the library ---
def find_null_space_gf2(matrix):
    """
    Finds a basis for the null space of a binary matrix over GF(2).
    This version now uses the robust rref function from qiskit-qec.
    """
    if matrix.size == 0:
        return np.array([[]])
    
    # --- CHANGE 3: Call the library rref function ---
    rref_matrix, pivot_cols = rref(matrix)
    
    # The rest of the logic is unchanged, as it correctly computes the null space
    # from any valid RREF matrix and its pivot columns.
    rows, cols = rref_matrix.shape
    free_cols = [j for j in range(cols) if j not in pivot_cols]
    
    null_space_basis = np.zeros((len(free_cols), cols), dtype=int)
    for i, j in enumerate(free_cols):
        null_space_basis[i, j] = 1
        for k, p_col in enumerate(pivot_cols):
            null_space_basis[i, p_col] = rref_matrix[k, j]
            
    return null_space_basis

# --- Conversion and Core Logic (unchanged from previous version) ---
def symplectic_to_pauli_dict(symplectic_vec):
    n = len(symplectic_vec) // 2
    x_vec_str = "".join(map(str, symplectic_vec[:n]))
    z_vec_str = "".join(map(str, symplectic_vec[n:]))
    return {"x": x_vec_str, "z": z_vec_str}

def generate_impure_codes_from_graph(graph_record, output_file):
    n = graph_record['n']
    adj_matrix = np.array(graph_record['adjacency_matrix'])
    
    initial_stabilizers = [Pauli((adj_matrix[i], np.eye(1, n, i, dtype=bool).flatten())) for i in range(n)]

    for k_total in range(1, n):
        for demoted_indices in itertools.combinations(range(n), k_total):
            impure_stabilizer_indices = [i for i in range(n) if i not in demoted_indices]
            impure_stabilizers = [initial_stabilizers[i] for i in impure_stabilizer_indices]
            
            if not impure_stabilizers:
                continue
            
            symplectic_rows = [np.hstack((s.x.astype(int), s.z.astype(int))) for s in impure_stabilizers]
            check_matrix = np.array(symplectic_rows, dtype=int)
            
            # This call now uses the more robust underlying logic.
            null_space_basis = find_null_space_gf2(check_matrix)
            
            logical_generators = [symplectic_to_pauli_dict(row) for row in null_space_basis]
            
            output_record = {
                "n": n, "d_seed": graph_record.get('d', 0), "graph6_origin": graph_record['graph6_string'],
                "k_total": k_total,
                "stabilizer_generators": [
                    {"x": "".join(map(str, s.x.astype(int))), "z": "".join(map(str, s.z.astype(int)))} 
                    for s in impure_stabilizers
                ],
                "logical_generators": logical_generators
            }
            output_file.write(json.dumps(output_record) + '\n')

def main(input_path, output_path):
    """
    Main driver for processing a single input file of candidate graphs.
    """
    print(f"\nStarting generation from '{input_path}'...")
    try:
        # Read all candidate graphs from the input file.
        with open(input_path, 'r') as f_in:
            lines = f_in.readlines()
        
        # Open the output file for writing.
        with open(output_path, 'w') as f_out:
            # Use tqdm for a progress bar as this can be a long process.
            for line in tqdm(lines, desc=f"Processing {os.path.basename(input_path)}"):
                graph_record = json.loads(line)
                # Generate all impure codes for this one graph and write them to the file.
                generate_impure_codes_from_graph(graph_record, f_out)
                
        print(f"Successfully generated impure codes to '{output_path}'.")
    except FileNotFoundError:
        print(f"Error: Input file not found at '{input_path}'")

if __name__ == '__main__':
    # --- Automation Setup ---
    # Directory containing the pre-processed candidate graphs.
    candidates_dir = './preprocessed_data'
    
    # Directory to store the output files.
    output_dir = './impure_codes'
    os.makedirs(output_dir, exist_ok=True)

    # Loop through all supported n values (1 to 12).
    for n in range(1, 13):
        input_file = os.path.join(candidates_dir, f'candidates_n{n}.jsonl')
        output_file = os.path.join(output_dir, f'impure_codes_n{n}.jsonl')
        
        # Only run the generator if the corresponding input file exists.
        if os.path.exists(input_file):
            main(input_file, output_file)
        else:
            print(f"Skipping n={n}: Input file not found at '{input_file}'")