import numpy as np
try:
    from qiskit_qec.linear import rref
except ImportError:
    print("CRITICAL: qiskit-qec not found.")
    exit()

def test_rref_output():
    print("--- Debugging qiskit_qec.linear.rref ---")

    # Case 1: Simple 1-row matrix [1 1]
    # In GF(2), RREF of [1 1] is [1 1] with pivot at col 0.
    matrix = np.array([[1, 1]], dtype=int)
    print(f"\n1. Input Matrix (1x2):\n{matrix}")

    output = rref(matrix)
    print(f"   Output Type: {type(output)}")
    print(f"   Output Data:\n{output}")

    # Case 2: Identity 3x3
    # RREF is Identity. Pivot cols should be 0, 1, 2.
    matrix_id = np.eye(3, dtype=int)
    print(f"\n2. Input Matrix (3x3 Identity):\n{matrix_id}")
    
    output_id = rref(matrix_id)
    print(f"   Output Type: {type(output_id)}")
    print(f"   Output Data:\n{output_id}")

    print("\n--- Diagnosis ---")
    if isinstance(output, tuple):
        print(f"It returned a TUPLE of length {len(output)}.")
        print("This matches the code logic (matrix, pivots). The error might be elsewhere.")
    elif isinstance(output, np.ndarray):
        print("It returned a SINGLE NUMPY ARRAY.")
        print("This confirms the breaking change. You MUST apply the fix.")
    else:
        print(f"It returned something unexpected: {type(output)}")

if __name__ == "__main__":
    test_rref_output()