# search_and_filter_resumable.py
import json
import os
import itertools
from tqdm import tqdm
from qiskit.quantum_info import Pauli

# --- Global Cache for Performance ---
# This dictionary stores the generated error sets for each 'n'.
# This is a critical optimization that prevents the same error set from being
# recalculated millions of times, providing a massive speedup.
EAD_ERROR_SETS_CACHE = {}

# --- Helper Functions ---

def generate_full_group(generators, n):
    """
    Takes a list of group generators and computes the full set of operators in the group.
    
    Args:
        generators (list[Pauli]): A list of Pauli objects that generate the group.
        n (int): The number of qubits.
        
    Returns:
        set[Pauli]: A set containing all Pauli objects in the generated group.
    """
    # If there are no generators, the group is just the identity.
    if not generators:
        return {Pauli("I" * n)}
    
    identity = Pauli("I" * n)
    group = {identity}
    # Iteratively build the group by taking products of existing elements with generators.
    for gen in generators:
        group.update({elem @ gen for elem in group})
    return group

def get_code_fingerprint(n, s_gens, t_gens, w_gens):
    """
    Creates a unique, canonical signature (fingerprint) for a hybrid code.
    This is essential for the resumability feature to avoid saving duplicate results.
    A fingerprint is the same even if the generators are listed in a different order.

    Args:
        n (int): Number of qubits.
        s_gens (list[Pauli]): Generators for the impure stabilizer group.
        t_gens (list[Pauli]): Generators for the translation operators.
        w_gens (list[Pauli]): Generators for the word operators.

    Returns:
        tuple: A unique, hashable tuple representing the code.
    """
    # Generate the full groups for stabilizers, translators, and words.
    s_fp = tuple(sorted([str(s) for s in generate_full_group(s_gens, n)]))
    t_fp = tuple(sorted([str(t) for t in generate_full_group(t_gens, n)]))
    w_fp = tuple(sorted([str(w) for w in generate_full_group(w_gens, n)]))
    # The final fingerprint is a tuple of these sorted group tuples.
    return (s_fp, t_fp, w_fp)

def _generate_ead_error_set(n):
    """
    (Internal function) Generates the specific set of Pauli errors that are sufficient
    to check for correction of a single amplitude damping error.
    Source: Rigby, Olivier, and Jarvis (2019), Section IV.
    
    Args:
        n (int): The number of qubits.

    Returns:
        list[Pauli]: A list of Pauli errors to check against.
    """
    error_set = []
    # Add the identity operator.
    error_set.append(Pauli("I" * n))
    # Add all weight-1 Pauli errors (X, Y, Z on each qubit).
    for i in range(n):
        for p_char in ['X', 'Y', 'Z']:
            p_str = ['I'] * n
            p_str[i] = p_char
            error_set.append(Pauli("".join(p_str)))
    # Add the required subset of weight-2 Pauli errors.
    for i in range(n):
        for j in range(i + 1, n):
            for p_pair in [('X', 'X'), ('X', 'Y'), ('Y', 'Y')]:
                p_str = ['I'] * n
                p_str[i] = p_pair[0]
                p_str[j] = p_pair[1]
                error_set.append(Pauli("".join(p_str)))
    return error_set

def get_ead_error_set(n):
    """
    Gets the E_AD error set for a given n using a cache to avoid re-computation.
    This is the public-facing function that should be called by the verifier.
    """
    if n not in EAD_ERROR_SETS_CACHE:
        # If the set for this 'n' isn't in our cache, generate it and store it.
        EAD_ERROR_SETS_CACHE[n] = _generate_ead_error_set(n)
    # Return the cached version.
    return EAD_ERROR_SETS_CACHE[n]

def check_single_hkl_condition(t_a, t_b, W_i, W_j, E_k, E_l, S_prime_gens):
    """
    Performs a single algebraic check of the Hybrid Knill-Laflamme condition for one
    combination of operators. This is the innermost check in the verification loop.

    Returns:
        bool: True if the condition holds for this combination, False otherwise.
    """
    # Construct the effective Pauli operator for the inner product calculation.
    P_eff = W_i @ t_a @ E_k @ E_l @ W_j @ t_b
    
    # Check if this operator commutes with all of the impure stabilizer generators.
    # This algebraically determines if the corresponding inner product is zero or non-zero.
    commutes_with_all = all(P_eff.commutes(s) for s in S_prime_gens)
    
    # Case 1: Inter-code check (different classical messages).
    if t_a != t_b:
        # The inner product must be zero (orthogonality).
        # This requires P_eff to anticommute with at least one stabilizer.
        # If it commutes with all, the condition is violated.
        if commutes_with_all: return False
    # Case 2: Intra-code check (same classical message).
    else:
        # Subcase 2a: Different quantum basis states.
        if W_i != W_j:
            # The inner product must be zero (orthogonality).
            # If it commutes with all, the condition is violated.
            if commutes_with_all: return False
        # Subcase 2b: Same quantum basis state.
        else:
            # The inner product must be non-zero.
            # This requires P_eff to commute with all stabilizers.
            # If it anticommutes with any, the condition is violated.
            if not commutes_with_all: return False
            
    # If none of the violation conditions were met, this specific check passes.
    return True

def verify_hkl_for_amplitude_damping(n, S_prime_gens, translation_ops, word_ops):
    """
    The main verifier. Checks if a fully defined hybrid code structure satisfies
    the HKL conditions for the amplitude damping channel error set.

    Returns:
        bool: True if the code is valid, False otherwise.
    """
    # Get the cached error set for this 'n'.
    error_set = get_ead_error_set(n)
    
    # These nested loops iterate through every single check required by the HKL theorem.
    # A single failure at any point means the code is invalid.
    for t_a in translation_ops:
        for t_b in translation_ops:
            for W_i in word_ops:
                for W_j in word_ops:
                    for E_k in error_set:
                        for E_l in error_set:
                            if not check_single_hkl_condition(t_a, t_b, W_i, W_j, E_k, E_l, S_prime_gens):
                                return False # Early exit on failure.
    # If all checks passed, the code is valid.
    return True

def get_best_known_k(n, d):
    """
    A simple lookup table of best-known parameters for standard stabilizer codes.
    Used to determine if a found hybrid code is "interesting".
    
    Returns:
        int: The maximum known k for a standard [[n, k, d]] code.
    """
    # This table can be expanded with more known values.
    BEST_K = {(4, 2): 1, (5, 3): 1, (6, 4): 2, (7,3):3, (8,3):4, (9, 4): 3}
    return BEST_K.get((n, d), 0)

def main(input_dir, output_path):
    """
    The main driver for the resumable brute-force search.
    """
    print(f"--- Starting Resumable Batch Search & Filter ---")
    
    # --- Resumption Logic ---
    # Load all previously found unique codes to avoid duplicates.
    seen_fingerprints = set()
    total_unique_found = 0
    if os.path.exists(output_path):
        print(f"Resuming from existing file: '{output_path}'")
        with open(output_path, 'r') as f:
            for line in f:
                code_data = json.loads(line)
                n = code_data['parameters']['n']
                # Reconstruct Pauli objects to create the fingerprint.
                s_gens = [Pauli(s) for s in code_data['impure_stabilizers']]
                t_gens = [Pauli(t) for t in code_data['translation_generators']]
                w_gens = [Pauli(w) for w in code_data['word_generators']]
                fingerprint = get_code_fingerprint(n, s_gens, t_gens, w_gens)
                seen_fingerprints.add(fingerprint)
        total_unique_found = len(seen_fingerprints)
        print(f"Loaded {total_unique_found} unique fingerprints from previous runs.")

    total_checks, valid_codes_found = 0, 0

    # --- Main Search Loop ---
    # Loop over all code lengths n=1 to n=12.
    for n in range(1, 13):
        input_file = os.path.join(input_dir, f'impure_codes_n{n}.jsonl')
        if not os.path.exists(input_file):
            continue
            
        print(f"\n--- Processing n={n} ---")
        with open(input_file, 'r') as f:
            lines = f.readlines()
        
        # Open the output file in append mode ('a') so we don't overwrite previous results.
        with open(output_path, 'a') as f_out:
            # Iterate through all impure code structures for this n.
            for line in tqdm(lines, desc=f"Searching n={n}"):
                impure_code = json.loads(line)
                k_total = impure_code['k_total']
                
                # Skip if there are no stabilizers (not a valid code).
                if not impure_code['stabilizer_generators']:
                    continue
                
                # Reconstruct the Qiskit Pauli objects from the JSON data.
                S_prime_gens = [Pauli(p['z'] + p['x']) for p in impure_code['stabilizer_generators']]
                logical_gens = [Pauli(p['z'] + p['x']) for p in impure_code['logical_generators']]

                # --- Combinatorial Search: Partitioning Logicals ---
                # Loop through all possible numbers of classical bits (m).
                for m in range(0, k_total + 1):
                    k = k_total - m
                    
                    # Sanity check: ensure we have enough logical generators for this k,m split.
                    if k < 0 or 2*k > len(logical_gens): continue
                    
                    # Choose 2*k logicals to be word operator generators.
                    for word_gen_indices in itertools.combinations(range(len(logical_gens)), 2*k):
                        remaining_indices = [i for i in range(len(logical_gens)) if i not in word_gen_indices]
                        
                        # From the remaining logicals, choose m to be translation generators.
                        for trans_gen_indices in itertools.combinations(remaining_indices, m):
                            total_checks += 1
                            word_gens = [logical_gens[i] for i in word_gen_indices]
                            trans_gens = [logical_gens[i] for i in trans_gen_indices]

                            # --- Verification and Filtering ---
                            # Run the expensive HKL check on this specific configuration.
                            if verify_hkl_for_amplitude_damping(n, S_prime_gens, generate_full_group(trans_gens,n), generate_full_group(word_gens,n)):
                                valid_codes_found += 1
                                d = impure_code['d_seed']
                                best_k = get_best_known_k(n, d)
                                
                                # Check if the found code is "interesting".
                                if (k + m) > best_k:
                                    fingerprint = get_code_fingerprint(n, S_prime_gens, trans_gens, word_gens)
                                    
                                    # Check if this unique code has been seen before.
                                    if fingerprint not in seen_fingerprints:
                                        seen_fingerprints.add(fingerprint)
                                        total_unique_found += 1
                                        
                                        # Prepare the result for saving.
                                        result = {
                                            "parameters": {"n": n, "k": k, "m": m, "d_seed": d},
                                            "origin_graph": impure_code['graph6_origin'],
                                            "impure_stabilizers": [str(s) for s in S_prime_gens],
                                            "translation_generators": [str(t) for t in trans_gens],
                                            "word_generators": [str(w) for w in word_gens]
                                        }
                                        # Write immediately to the file to save progress.
                                        f_out.write(json.dumps(result) + '\n')

    print("\n--- Search Complete ---")
    print(f"Total configurations checked in this run: {total_checks}")
    print(f"Total VALID hybrid codes found in this run: {valid_codes_found}")
    print(f"Total UNIQUE interesting codes found overall: {total_unique_found}")

if __name__ == '__main__':
    input_directory = './impure_codes'
    # The output is a .jsonl file for easy appending and resuming.
    output_file = './all_unique_interesting_codes.jsonl'
    main(input_directory, output_file)