"""
Debug script to verify error set computation against 4qubitexample.
"""
import numpy as np

# Star graph from 4qubitexample
gamma = np.array([
    [0, 1, 1, 1],
    [1, 0, 0, 0],
    [1, 0, 0, 0],
    [1, 0, 0, 0]
], dtype=int)

# Manual errors from 4qubitexample table
errors_manual = [
    ("1000", "0000", "X1"),      # u=1000, v=0, Cl = 0111
    ("1000", "1000", "Y1"),      # Cl = 1111
    ("0000", "1000", "Z1"),      # Cl = 1000
    ("0100", "0000", "X2"),      # Cl = u*Gamma = 1000
    ("0100", "0100", "Y2"),      # Cl = 1100
    ("0000", "0100", "Z2"),      # Cl = 0100
    ("0010", "0000", "X3"),      # Cl = 1000
    ("0010", "0010", "Y3"),      # Cl = 1010
    ("0000", "0010", "Z3"),      # Cl = 0010
    ("0001", "0000", "X4"),      # Cl = 1000
    ("0001", "0001", "Y4"),      # Cl = 1001
    ("0000", "0001", "Z4"),      # Cl = 0001
]

print("Verification of error mapping:")
print("=" * 60)

computed_classical = set()
expected_classical = {"0001", "0010", "0100", "0111", "1000", "1001", "1010", "1100", "1111"}

for u_str, v_str, name in errors_manual:
    u = np.array([int(c) for c in u_str], dtype=int)
    v = np.array([int(c) for c in v_str], dtype=int)
    
    u_gamma = np.dot(u, gamma) % 2
    cl_error = (v ^ u_gamma) % 2
    cl_str = "".join(str(bit) for bit in cl_error)
    
    computed_classical.add(cl_str)
    
    print(f"{name:4s} | u={u_str} v={v_str} | u·Gamma={(u_gamma)} | Cl={cl_str}")

print("\n" + "=" * 60)
print(f"Manual errors (from 4qubitexample): {len(errors_manual)} errors")
print(f"Expected classical set: {sorted(expected_classical)}")
print(f"Computed classical set: {sorted(computed_classical)}")
print(f"Match: {computed_classical == expected_classical}")

# Now check our error set generation
print("\n" + "=" * 60)
print("Our E{1} generation for n=4:")

# Single-qubit
single_qubit = []
for i in range(4):
    u = ["0"] * 4
    v = ["0"] * 4
    u[i] = "1"
    single_qubit.append(("".join(u), "".join(v), f"X{i+1}"))
    
    u = ["0"] * 4
    v = ["0"] * 4
    u[i] = "1"
    v[i] = "1"
    single_qubit.append(("".join(u), "".join(v), f"Y{i+1}"))
    
    u = ["0"] * 4
    v = ["0"] * 4
    v[i] = "1"
    single_qubit.append(("".join(u), "".join(v), f"Z{i+1}"))

print(f"Single-qubit errors: {len(single_qubit)}")

# Weight-2 pairs: X_i X_j, Y_i Y_j, X_i Y_j
weight2 = []
for i in range(4):
    for j in range(i+1, 4):
        u = ["0"] * 4
        v = ["0"] * 4
        u[i] = "1"
        u[j] = "1"
        weight2.append(("".join(u), "".join(v), f"X{i+1}X{j+1}"))
        
        u = ["0"] * 4
        v = ["0"] * 4
        u[i] = "1"
        u[j] = "1"
        v[i] = "1"
        v[j] = "1"
        weight2.append(("".join(u), "".join(v), f"Y{i+1}Y{j+1}"))
        
        u = ["0"] * 4
        v = ["0"] * 4
        u[i] = "1"
        u[j] = "1"
        v[j] = "1"
        weight2.append(("".join(u), "".join(v), f"X{i+1}Y{j+1}"))

print(f"Weight-2 errors: {len(weight2)}")
print(f"Total (inc. identity): {1 + len(single_qubit) + len(weight2)}")

# Compute classical errors for our E{1}
all_errors = [("0000", "0000", "I")] + single_qubit + weight2
our_classical = set()
for u_str, v_str, name in all_errors:
    u = np.array([int(c) for c in u_str], dtype=int)
    v = np.array([int(c) for c in v_str], dtype=int)
    u_gamma = np.dot(u, gamma) % 2
    cl_error = (v ^ u_gamma) % 2
    cl_str = "".join(str(bit) for bit in cl_error)
    our_classical.add(cl_str)

print(f"\nOur classical errors (first 20): {sorted(our_classical)}")
print(f"Difference from expected: {expected_classical - our_classical}")
print(f"Extra in ours: {our_classical - expected_classical}")
