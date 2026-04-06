"""
Graph State Error Analysis via the X–Z Rule
============================================
This script analyzes weight-1 Pauli errors on graph states using the
graph-state stabilizer formalism. It applies the X–Z rule to convert
all single-qubit Pauli errors into equivalent pure Z-type error patterns,
then represents them as classical binary strings.

Background
----------
A graph state |G⟩ is defined by a graph G = (V, E) with adjacency matrix Γ.
Its stabilizer generators are:
    K_i = X_i ⊗ ⊗_{j ∈ N(i)} Z_j
where N(i) is the set of neighbors of qubit i.

The X–Z Rule:
    - Z_i  →  Z_i           (Z errors stay as Z_i)
    - X_i  →  ⊗_{j ∈ N(i)} Z_j    (X on qubit i becomes Z on all neighbors)
    - Y_i  →  Z_i ⊗ ⊗_{j ∈ N(i)} Z_j  (Y = iXZ, so combine both rules)

This equivalence holds up to stabilizer multiplication, meaning the error
syndromes (detectable patterns) are identical.
"""

import numpy as np
import random
from typing import Optional


# ---------------------------------------------------------------------------
# 1. Graph State Database
# ---------------------------------------------------------------------------

def build_graph_database() -> dict[str, np.ndarray]:
    """
    Define a small database of LC-inequivalent graph states represented
    by their adjacency matrices (symmetric, 0-diagonal integer arrays).

    LC-inequivalence means no sequence of local complementation operations
    can map one graph into another, so they represent genuinely distinct
    entanglement classes.

    Returns
    -------
    dict mapping a descriptive name → NumPy adjacency matrix
    """
    db = {}

    # --- 3-qubit graphs ---
    # 3-qubit linear cluster: 0-1-2
    db["3q_linear_cluster"] = np.array([
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0],
    ], dtype=int)

    # 3-qubit star (same as triangle under LC, but included for illustration)
    db["3q_star"] = np.array([
        [0, 1, 1],
        [1, 0, 0],
        [1, 0, 0],
    ], dtype=int)

    # --- 4-qubit graphs ---
    # 4-qubit linear cluster: 0-1-2-3
    db["4q_linear_cluster"] = np.array([
        [0, 1, 0, 0],
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 0],
    ], dtype=int)

    # 4-qubit ring (cycle C4): 0-1-2-3-0
    db["4q_ring"] = np.array([
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
    ], dtype=int)

    # 4-qubit star: center=0, leaves=1,2,3
    db["4q_star"] = np.array([
        [0, 1, 1, 1],
        [1, 0, 0, 0],
        [1, 0, 0, 0],
        [1, 0, 0, 0],
    ], dtype=int)

    # 4-qubit complete graph K4 (all pairs connected)
    db["4q_complete_K4"] = np.array([
        [0, 1, 1, 1],
        [1, 0, 1, 1],
        [1, 1, 0, 1],
        [1, 1, 1, 0],
    ], dtype=int)

    # --- 5-qubit graphs ---
    # 5-qubit linear cluster
    db["5q_linear_cluster"] = np.array([
        [0, 1, 0, 0, 0],
        [1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0],
        [0, 0, 1, 0, 1],
        [0, 0, 0, 1, 0],
    ], dtype=int)

    # 5-qubit ring C5
    db["5q_ring_C5"] = np.array([
        [0, 1, 0, 0, 1],
        [1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0],
        [0, 0, 1, 0, 1],
        [1, 0, 0, 1, 0],
    ], dtype=int)

    # 5-qubit perfect error-correcting code graph
    # (used in the [[5,1,3]] code — ring + additional edges)
    db["5q_perfect_code"] = np.array([
        [0, 1, 0, 0, 1],
        [1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0],
        [0, 0, 1, 0, 1],
        [1, 0, 0, 1, 0],
    ], dtype=int)

    # --- 6-qubit graphs ---
    # 6-qubit ring C6
    db["6q_ring_C6"] = np.array([
        [0, 1, 0, 0, 0, 1],
        [1, 0, 1, 0, 0, 0],
        [0, 1, 0, 1, 0, 0],
        [0, 0, 1, 0, 1, 0],
        [0, 0, 0, 1, 0, 1],
        [1, 0, 0, 0, 1, 0],
    ], dtype=int)

    # 6-qubit 2D cluster (2×3 grid)
    #  0-1-2
    #  |   |   |
    #  3-4-5
    db["6q_2x3_grid"] = np.array([
        [0, 1, 0, 1, 0, 0],
        [1, 0, 1, 0, 1, 0],
        [0, 1, 0, 0, 0, 1],
        [1, 0, 0, 0, 1, 0],
        [0, 1, 0, 1, 0, 1],
        [0, 0, 1, 0, 1, 0],
    ], dtype=int)

    return db


def select_graph(
    db: dict[str, np.ndarray],
    name: Optional[str] = None,
    seed: Optional[int] = None,
) -> tuple[str, np.ndarray]:
    """
    Randomly select (or explicitly choose) a graph from the database.

    Parameters
    ----------
    db   : graph database returned by build_graph_database()
    name : if provided, select this specific graph by name
    seed : optional RNG seed for reproducibility

    Returns
    -------
    (graph_name, adjacency_matrix)
    """
    if name is not None:
        if name not in db:
            raise KeyError(f"Graph '{name}' not found. Available: {list(db)}")
        return name, db[name]

    if seed is not None:
        random.seed(seed)

    graph_name = random.choice(list(db.keys()))
    return graph_name, db[graph_name]


# ---------------------------------------------------------------------------
# 2. Adjacency Matrix Utilities
# ---------------------------------------------------------------------------

def get_neighbors(adj: np.ndarray, qubit: int) -> list[int]:
    """
    Return the list of neighbors of `qubit` in the graph.

    Parameters
    ----------
    adj   : n×n adjacency matrix
    qubit : vertex index

    Returns
    -------
    Sorted list of neighbor indices
    """
    n = adj.shape[0]
    return [j for j in range(n) if adj[qubit, j] == 1]


def print_adjacency_matrix(name: str, adj: np.ndarray) -> None:
    """Pretty-print the adjacency matrix with row/column labels."""
    n = adj.shape[0]
    header = "    " + "  ".join(f"q{j}" for j in range(n))
    print(f"\n  Graph: {name}  ({n} qubits)")
    print("  " + "─" * (len(header) - 2))
    print(f"  {header}")
    print("  " + "─" * (len(header) - 2))
    for i in range(n):
        row_str = "  ".join(str(adj[i, j]) for j in range(n))
        print(f"  q{i}  {row_str}")
    print("  " + "─" * (len(header) - 2))


# ---------------------------------------------------------------------------
# 3 & 4. Error Generation and X–Z Rule
# ---------------------------------------------------------------------------

def generate_weight1_errors(n: int) -> list[tuple[str, int]]:
    """
    Enumerate all weight-1 Pauli errors on n qubits.

    Each error is a (pauli_type, qubit_index) pair:
      - pauli_type ∈ {'X', 'Y', 'Z'}
      - qubit_index ∈ {0, 1, ..., n-1}

    Parameters
    ----------
    n : number of qubits

    Returns
    -------
    List of (pauli_label, qubit_index) tuples
    """
    errors = []
    for pauli in ("X", "Y", "Z"):
        for i in range(n):
            errors.append((pauli, i))
    return errors


def apply_xz_rule(
    pauli: str,
    qubit: int,
    adj: np.ndarray,
) -> np.ndarray:
    """
    Apply the graph-state X–Z rule to convert a single-qubit Pauli error
    into an equivalent pure Z-type error pattern (binary vector).

    Rules (mod 2 arithmetic, valid up to stabilizer multiplication):
    ----------------------------------------------------------------
    Z_i  →  e_i                         (standard basis vector for qubit i)
    X_i  →  ⊕_{j ∈ N(i)} e_j           (Z on every neighbor of i)
    Y_i  →  e_i ⊕ ⊕_{j ∈ N(i)} e_j    (Z on i AND on all neighbors of i)

    The Y rule follows from Y_i = i·X_i·Z_i and the fact that
    we combine the Z and X contributions (mod 2).

    Parameters
    ----------
    pauli : 'X', 'Y', or 'Z'
    qubit : index of the affected qubit (0-based)
    adj   : adjacency matrix of the graph

    Returns
    -------
    NumPy binary array of length n representing the Z-error pattern
    """
    n = adj.shape[0]
    z_pattern = np.zeros(n, dtype=int)

    if pauli == "Z":
        # Z_i acts as Z on qubit i — no conversion needed.
        z_pattern[qubit] = 1

    elif pauli == "X":
        # X_i is equivalent (up to stabilizers) to Z on all neighbors of i.
        # This follows from the stabilizer generator K_i = X_i ∏_{j∈N(i)} Z_j:
        # multiplying the error X_i by K_i gives ∏_{j∈N(i)} Z_j.
        neighbors = get_neighbors(adj, qubit)
        for nb in neighbors:
            z_pattern[nb] = 1  # place a Z on each neighbor

    elif pauli == "Y":
        # Y_i = i·X_i·Z_i.  Combining the X and Z rules (mod 2):
        # Y_i  →  Z_i ⊕ (Z on all neighbors of i)
        z_pattern[qubit] = 1                       # Z contribution from Z_i
        neighbors = get_neighbors(adj, qubit)
        for nb in neighbors:
            z_pattern[nb] ^= 1  # XOR: handles cases where qubit is its own neighbor

    else:
        raise ValueError(f"Unknown Pauli type: '{pauli}'. Expected X, Y, or Z.")

    return z_pattern


# ---------------------------------------------------------------------------
# 5. Binary String Conversion
# ---------------------------------------------------------------------------

def pattern_to_binary_string(z_pattern: np.ndarray) -> str:
    """
    Convert a Z-error pattern (NumPy int array) to a binary string.

    E.g. [0, 1, 0, 1] → '0101'

    Parameters
    ----------
    z_pattern : binary array of length n

    Returns
    -------
    Binary string of length n
    """
    return "".join(str(b) for b in z_pattern)


# ---------------------------------------------------------------------------
# 6. Output & Reporting
# ---------------------------------------------------------------------------

def analyze_graph_state(
    graph_name: str,
    adj: np.ndarray,
    *,
    show_duplicates: bool = True,
) -> dict:
    """
    Full analysis pipeline for a selected graph state:
      1. Print the adjacency matrix.
      2. Generate all weight-1 Pauli errors.
      3. Apply the X–Z rule to each error.
      4. Convert to binary strings.
      5. Report results (and optionally unique patterns).

    Parameters
    ----------
    graph_name      : human-readable name of the graph
    adj             : adjacency matrix (n×n NumPy array)
    show_duplicates : if False, also display the deduplicated error set

    Returns
    -------
    dict with keys:
        'errors'        : list of (label, pattern, binary_string)
        'unique_strings': set of unique binary strings
    """
    n = adj.shape[0]

    # --- Print adjacency matrix ---
    print_adjacency_matrix(graph_name, adj)

    # --- Print neighbor structure (helps understand the X–Z rule) ---
    print("\n  Neighbor structure:")
    for i in range(n):
        nbs = get_neighbors(adj, i)
        nb_str = ", ".join(f"q{j}" for j in nbs) if nbs else "none"
        print(f"    q{i}: N({i}) = {{ {nb_str} }}")

    # --- Generate errors and apply X–Z rule ---
    errors = generate_weight1_errors(n)
    results = []

    print("\n" + "═" * 62)
    print("  Weight-1 Pauli Errors → Z-Pattern Binary Strings")
    print("═" * 62)
    print(f"  {'Error':<8}  {'Z-pattern vector':<28}  {'Binary string'}")
    print("  " + "─" * 58)

    for pauli, qubit in errors:
        label = f"{pauli}_{qubit}"
        z_pattern = apply_xz_rule(pauli, qubit, adj)
        binary_str = pattern_to_binary_string(z_pattern)

        # Format the Z-pattern as a readable vector
        vec_str = "[" + " ".join(str(b) for b in z_pattern) + "]"
        print(f"  {label:<8}  {vec_str:<28}  {binary_str}")

        results.append((label, z_pattern.copy(), binary_str))

    print("  " + "─" * 58)

    # --- Unique binary strings (bonus) ---
    unique_strings = sorted(set(r[2] for r in results))
    print(f"\n  Total errors analyzed : {len(results)}")
    print(f"  Unique Z-patterns     : {len(unique_strings)}")

    if not show_duplicates or len(unique_strings) < len(results):
        print("\n" + "═" * 62)
        print("  Unique Error Patterns (deduplicated)")
        print("═" * 62)

        # Map each unique string to the errors that produce it
        pattern_map: dict[str, list[str]] = {}
        for label, _, binary_str in results:
            pattern_map.setdefault(binary_str, []).append(label)

        print(f"  {'Binary string':<20}  {'Produced by'}")
        print("  " + "─" * 50)
        for bs in unique_strings:
            producers = ", ".join(pattern_map[bs])
            print(f"  {bs:<20}  {producers}")
        print("  " + "─" * 50)

    print()
    return {"errors": results, "unique_strings": unique_strings}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(
    graph_name: Optional[str] = None,
    seed: Optional[int] = 42,
) -> None:
    """
    Entry point.

    Parameters
    ----------
    graph_name : name of a specific graph to select (None → random)
    seed       : RNG seed for reproducible random selection
    """
    print("=" * 62)
    print("  Graph State Error Analysis — X–Z Rule Mapping")
    print("=" * 62)

    # Build database and select a graph
    db = build_graph_database()
    print(f"\n  Available graphs ({len(db)} total):")
    for gname in db:
        n = db[gname].shape[0]
        print(f"    • {gname}  ({n} qubits)")

    selected_name, adj = select_graph(db, name=graph_name, seed=seed)
    print(f"\n  ► Randomly selected: '{selected_name}'\n")

    # Run the full analysis
    analyze_graph_state(selected_name, adj, show_duplicates=False)

    # --- Configurable qubit count demo ---
    print("=" * 62)
    print("  Bonus: Custom n-qubit Linear Cluster (n=4 by default)")
    print("=" * 62)
    custom_adj = make_linear_cluster(n=4)
    analyze_graph_state("custom_4q_linear_cluster", custom_adj, show_duplicates=False)


def make_linear_cluster(n: int) -> np.ndarray:
    """
    Construct the adjacency matrix for an n-qubit linear cluster state.
    Edges: i — (i+1) for i = 0, …, n-2.

    Parameters
    ----------
    n : number of qubits (must be ≥ 2)

    Returns
    -------
    n×n NumPy adjacency matrix
    """
    if n < 2:
        raise ValueError("Linear cluster requires at least 2 qubits.")
    adj = np.zeros((n, n), dtype=int)
    for i in range(n - 1):
        adj[i, i + 1] = 1
        adj[i + 1, i] = 1
    return adj


if __name__ == "__main__":
    main()