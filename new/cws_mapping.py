import numpy as np

def run_cws_mapping():
    n = 4
    # Star Graph: Qubit 0 is center, connected to 1, 2, 3
    gamma = np.array([
        [0, 1, 1, 1],
        [1, 0, 0, 0],
        [1, 0, 0, 0],
        [1, 0, 0, 0]
    ])

    paulis = ['X', 'Y', 'Z']
    
    print(f"{'Error':<8} | {'u (X)':<6} | {'v (Z)':<6} | {'u*Gamma':<8} | {'Final Cl(E)':<10}")
    print("-" * 55)

    all_cl_errors = set()

    for i in range(n):
        for p in paulis:
            # Initialize vectors
            u = np.zeros(n, dtype=int)
            v = np.zeros(n, dtype=int)
            
            # Populate u and v based on Pauli type
            if p in ['X', 'Y']: u[i] = 1
            if p in ['Z', 'Y']: v[i] = 1
            
            # Step 1: Matrix Multiplication (Propagation)
            propagation = np.dot(u, gamma) % 2
            
            # Step 2: Final XOR (Cl_G(E))
            cl_error = (v ^ propagation) % 2
            
            # Format for display
            u_str = "".join(map(str, u))
            v_str = "".join(map(str, v))
            prop_str = "".join(map(str, propagation))
            cl_str = "".join(map(str, cl_error))
            
            all_cl_errors.add(cl_str)
            
            print(f"{p}{i+1:<7} | {u_str:<6} | {v_str:<6} | {prop_str:<8} | {cl_str:<10}")

    return all_cl_errors

errors = run_cws_mapping()
print("\nUnique Classical Error Set (Cl_G):")
print(sorted(list(errors)))