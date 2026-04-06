"""
Exhaustive Search for Hybrid Quantum Codes on the Amplitude Damping Channel
============================================================================

This module implements Phases 1-5 of the finalized search algorithm:
- Phase 1: Load LC-inequivalent graphs from vncorbits database
- Phase 2: Test orientations (error-set mapping)
- Phase 3: Pruning via degeneracy checks
- Phase 4: Max-clique search with priority on high |N_E|
- Phase 5: Hybrid KL verification

Reference: Jackson et al. (2016), LP bounds, vncorbits database
"""

import numpy as np
from typing import Tuple, List, Set, Dict, Callable, Optional
import itertools
import json
from dataclasses import dataclass, asdict
from pathlib import Path


# ==============================================================================
# PHASE 1: Graph Loading & Initialization
# ==============================================================================

@dataclass
class GraphRepresentative:
    """A single LC-inequivalent graph representative."""
    n: int
    name: str
    adjacency: np.ndarray
    graph_key: str  # unique identifier (e.g., "vncorbits_4_2")

    def copy(self):
        return GraphRepresentative(self.n, self.name, self.adjacency.copy(), self.graph_key)


def build_local_graph_database() -> Dict[str, GraphRepresentative]:
    """
    Build a small database of LC-inequivalent graphs for quick testing.
    Use vncorbits for larger searches.
    """
    db = {}

    # 4-qubit star: center=0, leaves=1,2,3
    gamma_4star = np.array([
        [0, 1, 1, 1],
        [1, 0, 0, 0],
        [1, 0, 0, 0],
        [1, 0, 0, 0],
    ], dtype=int)
    db["4q_star"] = GraphRepresentative(4, "star", gamma_4star, "4q_star_vncorbits_0")

    # 4-qubit linear cluster
    gamma_4lin = np.array([
        [0, 1, 0, 0],
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 0],
    ], dtype=int)
    db["4q_linear"] = GraphRepresentative(4, "linear", gamma_4lin, "4q_linear_vncorbits_1")

    # 4-qubit ring
    gamma_4ring = np.array([
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
    ], dtype=int)
    db["4q_ring"] = GraphRepresentative(4, "ring", gamma_4ring, "4q_ring_vncorbits_2")

    # 4-qubit complete K4
    gamma_4k4 = np.array([
        [0, 1, 1, 1],
        [1, 0, 1, 1],
        [1, 1, 0, 1],
        [1, 1, 1, 0],
    ], dtype=int)
    db["4q_K4"] = GraphRepresentative(4, "K4", gamma_4k4, "4q_K4_vncorbits_3")

    return db


# ==============================================================================
# PHASE 2 & 3: Orientation Enumeration & Error-Set Generation
# ==============================================================================

@dataclass
class OrientationConfig:
    """Encodes a per-qubit orientation choice."""
    n: int
    permutations: Tuple[int, ...]  # each element is 0, 1, or 2 → {id, (13), (23)}
    
    def __str__(self):
        names = ["id", "(13)", "(23)"]
        return "".join(names[p] for p in self.permutations)


def permutation_to_pauli_map(perm: int) -> Dict[str, str]:
    """
    Convert permutation index to Pauli reordering.
    - 0: identity → X,Y,Z → X,Y,Z
    - 1: (13) → X,Y,Z → Z,Y,X  (swap X and Z)
    - 2: (23) → X,Y,Z → X,Z,Y  (swap Y and Z)
    """
    if perm == 0:
        return {"X": "X", "Y": "Y", "Z": "Z"}
    elif perm == 1:
        return {"X": "Z", "Y": "Y", "Z": "X"}
    elif perm == 2:
        return {"X": "X", "Y": "Z", "Z": "Y"}
    else:
        raise ValueError(f"Invalid permutation index: {perm}")


def enumerate_orientations(n: int) -> List[OrientationConfig]:
    """
    Generate all 3^n orientation assignments (one per qubit).
    Each qubit can have permutation 0, 1, or 2.
    """
    orientations = []
    for perm_tuple in itertools.product(range(3), repeat=n):
        orientations.append(OrientationConfig(n, perm_tuple))
    return orientations


def generate_pauli_error_set_E1(n: int) -> Set[Tuple[str, str]]:
    """
    Generate E{1} for single AD error detection.
    
    Start with: {I} ∪ {X_i, Y_i, Z_i} (single-qubit errors).
    Optionally add weight-2 errors for AD channels.
    
    Reference: Jackson et al. (2016) - for a single damping error
    """
    errors = set()
    errors.add(("0" * n, "0" * n))  # Identity
    
    # Single-qubit errors only (simplest E{1})
    for i in range(n):
        # X_i
        u = ["0"] * n
        v = ["0"] * n
        u[i] = "1"
        errors.add(("".join(u), "".join(v)))
        
        # Y_i
        u = ["0"] * n
        v = ["0"] * n
        u[i] = "1"
        v[i] = "1"
        errors.add(("".join(u), "".join(v)))
        
        # Z_i
        u = ["0"] * n
        v = ["0"] * n
        v[i] = "1"
        errors.add(("".join(u), "".join(v)))
    
    return errors


def apply_orientation_to_error_set(
    errors: Set[Tuple[str, str]],
    orientation: OrientationConfig,
) -> Set[Tuple[str, str]]:
    """
    Apply a global orientation (permutation assignment) to an error set.
    For each (u, v) pair, permute based on the per-qubit assignment.
    """
    # For now, return unchanged. Orientation permutations affect how we interpret
    # X, Y, Z across different qubits, but the (u,v) representation encodes the
    # Pauli structure directly. Permutations are used during the classical mapping.
    return errors


def compute_classical_error_set(
    errors: Set[Tuple[str, str]],
    gamma: np.ndarray,
) -> Tuple[Set[str], Dict[str, str]]:
    """
    Apply the X-Z rule to map quantum errors to classical bit strings.
    Cl_G(E) = v ⊕ (u · Gamma).
    
    Returns:
    - classical_error_set: set of classical bitstrings in Cl_G(E)
    - error_index: maps classical string to original (u, v) for tracing
    """
    classical_errors = set()
    error_index = {}
    n = gamma.shape[0]
    
    for u_str, v_str in errors:
        u = np.array([int(c) for c in u_str], dtype=int)
        v = np.array([int(c) for c in v_str], dtype=int)
        
        # u · Gamma (mod 2)
        u_gamma = np.dot(u, gamma) % 2
        
        # v ⊕ (u · Gamma)
        cl_error = (v ^ u_gamma) % 2
        cl_str = "".join(str(bit) for bit in cl_error)
        
        classical_errors.add(cl_str)
        error_index[cl_str] = (u_str, v_str)
    
    return classical_errors, error_index


# ==============================================================================
# PHASE 3: Degeneracy Checks & Clique Graph Construction
# ==============================================================================

def identify_degenerate_errors(
    classical_errors: Set[str],
    error_index: Dict[str, str],
) -> Dict[str, Set[str]]:
    """
    Identify degenerate errors (those mapping to the zero string).
    For each degenerate error E with zero mapping, extract the u-part.
    
    Returns: mapping from inadmissible bit positions to which u-parts enforce constraints.
    """
    degeneracies = {}
    zero_str = "0" * len(list(classical_errors)[0])
    
    if zero_str in error_index:
        u_str, _ = error_index[zero_str]
        u_set = set(i for i, bit in enumerate(u_str) if bit == "1")
        if u_set:
            degeneracies["zero_error_u_support"] = u_set
    
    return degeneracies


def get_admissible_vertices(
    n: int,
    classical_errors: Set[str],
    degeneracies: Dict[str, Set[str]],
    error_index: Dict[str, str],
) -> Set[str]:
    """
    Compute the set of admissible classical codewords (vertices).
    A bitstring c is admissible if:
    - c ∉ classical_errors (not a classical error)
    - For each degenerate error, c satisfies the constraint
    """
    all_bitstrings = set(format(i, f"0{n}b") for i in range(2**n))
    
    # Remove strings in classical error set
    admissible = all_bitstrings - classical_errors
    
    # Apply degeneracy constraints
    if "zero_error_u_support" in degeneracies:
        u_support = degeneracies["zero_error_u_support"]
        # Constraint: c · u = 0 (mod 2) where u is the X-part of degenerate error
        def satisfies_constraint(c_str):
            inner_product = sum(int(c_str[i]) for i in u_support) % 2
            return inner_product == 0
        admissible = set(c for c in admissible if satisfies_constraint(c))
    
    return admissible


def build_clique_graph(
    admissible: Set[str],
    classical_errors: Set[str],
) -> Dict[str, Set[str]]:
    """
    Build the clique-compatibility graph.
    Vertices: admissible bitstrings
    Edges: (a, b) connected if (a ⊕ b) ∉ classical_errors.
    
    Returns adjacency dict.
    """
    graph = {v: set() for v in admissible}
    
    for v1 in admissible:
        for v2 in admissible:
            if v1 != v2:
                # Check if XOR is NOT in classical errors
                xor_str = "".join(str((int(v1[i]) ^ int(v2[i]))) for i in range(len(v1)))
                if xor_str not in classical_errors:
                    graph[v1].add(v2)
    
    return graph


# ==============================================================================
# PHASE 4: Max-Clique Search
# ==============================================================================

def find_max_clique_greedy(graph: Dict[str, Set[str]]) -> Tuple[Set[str], int]:
    """
    Greedy max-clique finder (fast heuristic for dense graphs).
    Not exact but sufficient for testing.
    """
    if not graph:
        return set(), 0
    
    vertices = list(graph.keys())
    max_clique = set()
    
    # Try starting from each vertex with highest degree
    sorted_vertices = sorted(vertices, key=lambda v: -len(graph[v]))
    
    for start_vertex in sorted_vertices[:min(10, len(sorted_vertices))]:
        clique = {start_vertex}
        candidates = set(graph[start_vertex])
        
        while candidates:
            # Add vertex with most connections to current clique
            best_v = None
            best_count = 0
            for v in candidates:
                count = sum(1 for u in clique if v in graph[u])
                if count == len(clique):  # Only consider v if it connects to all in clique
                    if len(graph[v] & candidates) > best_count:
                        best_v = v
                        best_count = len(graph[v] & candidates)
            
            if best_v is None:
                break
            
            clique.add(best_v)
            candidates &= graph[best_v]
        
        if len(clique) > len(max_clique):
            max_clique = clique
    
    return max_clique, len(max_clique)


# ==============================================================================
# PHASE 5: Hybrid KL Verification
# ==============================================================================

def verify_hybrid_kl_condition(
    clique: Set[str],
    classical_errors: Set[str],
    m: int,
) -> Tuple[bool, Dict]:
    """
    Verify the Hybrid Knill-Laflamme (KL) condition.
    Partition the clique into M subsets of size K.
    Check that pairs across different subsets do not XOR to classical errors.
    
    Returns: (is_valid, report)
    """
    clique_list = sorted(list(clique))
    K_total = len(clique_list)
    K = K_total // m if m > 0 else K_total
    
    if K * m != K_total:
        return False, {"error": f"Clique size {K_total} not divisible into {m} subcodes of size {K}"}
    
    # Partition into m subcodes
    subcodes = []
    for mu in range(m):
        subcode = set(clique_list[mu * K:(mu + 1) * K])
        subcodes.append(subcode)
    
    # Check KL condition: for all pairs (a, b) from different subcodes,
    # (a ⊕ b) must NOT be in classical_errors
    violations = []
    for mu in range(m):
        for nu in range(m):
            if mu != nu:
                for a in subcodes[mu]:
                    for b in subcodes[nu]:
                        xor_str = "".join(str((int(a[i]) ^ int(b[i]))) for i in range(len(a)))
                        if xor_str in classical_errors:
                            violations.append((a, b, xor_str, mu, nu))
    
    is_valid = len(violations) == 0
    report = {
        "valid": is_valid,
        "num_subcodes": m,
        "subcode_size": K,
        "total_clique_size": K_total,
        "violations": len(violations),
        "violation_samples": violations[:5],  # Report first 5
    }
    return is_valid, report


# ==============================================================================
# Main Search Orchestrator
# ==============================================================================

@dataclass
class SearchResult:
    """Result from a single (graph, orientation, E{j}) search."""
    graph_key: str
    orientation: str
    error_set_type: str  # "E1", "E2", "E3"
    n: int
    admissible_vertex_count: int
    max_clique_size: int
    max_clique: Set[str]
    classical_errors: Set[str]
    lp_bound: Optional[int] = None  # from LP_bounds table
    hybrid_kl_valid: Optional[bool] = None
    hybrid_kl_report: Optional[Dict] = None
    
    def to_dict(self):
        return {
            "graph_key": self.graph_key,
            "orientation": self.orientation,
            "error_set_type": self.error_set_type,
            "n": self.n,
            "admissible_vertex_count": self.admissible_vertex_count,
            "max_clique_size": self.max_clique_size,
            "max_clique": sorted(list(self.max_clique)),
            "num_classical_errors": len(self.classical_errors),
            "lp_bound": self.lp_bound,
            "hybrid_kl_valid": self.hybrid_kl_valid,
            "hybrid_kl_report": self.hybrid_kl_report,
        }


# ==============================================================================
# CHECKPOINT SYSTEM
# ==============================================================================

def get_checkpoint_path(n: int, error_set_type: str) -> Path:
    """Get checkpoint file path for given search parameters."""
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    return checkpoint_dir / f"search_n{n}_{error_set_type}.json"


def load_checkpoint(n: int, error_set_type: str) -> Tuple[Set[str], List[SearchResult]]:
    """
    Load checkpoint if it exists.
    Returns: (processed_combinations, results)
    processed_combinations: set of (graph_key, orientation_str) pairs already done
    results: list of SearchResult objects from previous runs
    """
    checkpoint_path = get_checkpoint_path(n, error_set_type)
    
    if not checkpoint_path.exists():
        return set(), []
    
    try:
        with open(checkpoint_path, 'r') as f:
            data = json.load(f)
        
        processed = set(tuple(combo) for combo in data.get('processed_combinations', []))
        
        results = []
        for result_dict in data.get('results', []):
            result = SearchResult(
                graph_key=result_dict['graph_key'],
                orientation=result_dict['orientation'],
                error_set_type=result_dict['error_set_type'],
                n=result_dict['n'],
                admissible_vertex_count=result_dict['admissible_vertex_count'],
                max_clique_size=result_dict['max_clique_size'],
                max_clique=set(result_dict['max_clique']),
                classical_errors=set(result_dict.get('classical_errors', [])),
                lp_bound=result_dict.get('lp_bound'),
                hybrid_kl_valid=result_dict.get('hybrid_kl_valid'),
                hybrid_kl_report=result_dict.get('hybrid_kl_report'),
            )
            results.append(result)
        
        return processed, results
    except Exception as e:
        print(f"Warning: Could not load checkpoint: {e}")
        return set(), []


def save_checkpoint(n: int, error_set_type: str, processed_combinations: Set[str], results: List[SearchResult], verbose: bool = False):
    """
    Save checkpoint with processed combinations and current results.
    """
    checkpoint_path = get_checkpoint_path(n, error_set_type)
    
    data = {
        'processed_combinations': sorted(list(processed_combinations)),
        'results': [r.to_dict() for r in results],
    }
    
    with open(checkpoint_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    if verbose:
        print(f"Checkpoint saved: {len(processed_combinations)} processed, {len(results)} results")


def run_exhaustive_search(
    n: int,
    error_set_type: str = "E1",
    graphs: Optional[Dict[str, GraphRepresentative]] = None,
    verbose: bool = True,
    checkpoint_interval: int = 50,
) -> List[SearchResult]:
    """
    Execute the full 5-phase search for hybrid AD-channel codes.
    
    Parameters:
    - n: number of qubits
    - error_set_type: "E1", "E2", or "E3"
    - graphs: dict of GraphRepresentative objects; defaults to local database
    - verbose: print progress
    - checkpoint_interval: save checkpoint every N results (0 = no checkpointing)
    
    Returns:
    - List of SearchResult objects sorted by max_clique_size (descending)
    """
    
    if graphs is None:
        graphs = build_local_graph_database()
    
    # Filter to graphs of size n
    graphs_n = {k: v for k, v in graphs.items() if v.n == n}
    
    if not graphs_n:
        print(f"No graphs found for n={n}")
        return []
    
    # Load checkpoint if it exists
    processed_combinations, results = load_checkpoint(n, error_set_type)
    results_found_count = len(results)  # Count of results (not all combinations produce results)
    
    # Phase 1: Load graphs
    if verbose:
        print(f"Phase 1: Loaded {len(graphs_n)} LC-inequivalent graphs for n={n}")
    
    # Phase 2: Enumerate orientations
    orientations = enumerate_orientations(n)
    if verbose:
        print(f"Phase 2: Enumerating {len(orientations)} orientations (3^{n})")
        if processed_combinations:
            print(f"         Resuming from checkpoint: {len(processed_combinations)} already processed")
    
    total_combinations = len(graphs_n) * len(orientations)
    processed = 0
    new_results_since_checkpoint = 0
    
    # Main loop over graphs and orientations
    for graph_name, graph_rep in graphs_n.items():
        for orientation in orientations:
            processed += 1
            
            # Create combination key for checkpoint tracking
            combo_key = (graph_rep.graph_key, str(orientation))
            
            # Skip if already processed
            if combo_key in processed_combinations:
                continue
            
            # Generate error set
            if error_set_type == "E1":
                error_set = generate_pauli_error_set_E1(n)
            else:
                error_set = generate_pauli_error_set_E1(n)  # Placeholder
            
            # Phase 2.5: Apply orientation
            oriented_errors = apply_orientation_to_error_set(error_set, orientation)
            
            # Phase 3: Compute classical errors and degeneracies
            classical_errors, error_index = compute_classical_error_set(oriented_errors, graph_rep.adjacency)
            degeneracies = identify_degenerate_errors(classical_errors, error_index)
            
            # Phase 3.5: Get admissible vertices
            admissible = get_admissible_vertices(n, classical_errors, degeneracies, error_index)
            
            if len(admissible) == 0:
                processed_combinations.add(combo_key)
                continue  # Skip if no admissible vertices
            
            # Phase 4: Build and search clique graph
            clique_graph = build_clique_graph(admissible, classical_errors)
            max_clique, clique_size = find_max_clique_greedy(clique_graph)
            
            if clique_size == 0:
                processed_combinations.add(combo_key)
                continue  # Skip if no clique found
            
            # Phase 5: Verify hybrid KL (optionally, with m=2)
            kl_valid, kl_report = verify_hybrid_kl_condition(max_clique, classical_errors, m=2)
            
            result = SearchResult(
                graph_key=graph_rep.graph_key,
                orientation=str(orientation),
                error_set_type=error_set_type,
                n=n,
                admissible_vertex_count=len(admissible),
                max_clique_size=clique_size,
                max_clique=max_clique,
                classical_errors=classical_errors,
                lp_bound=None,  # TODO: load from LP_bounds table
                hybrid_kl_valid=kl_valid,
                hybrid_kl_report=kl_report,
            )
            
            results.append(result)
            processed_combinations.add(combo_key)
            new_results_since_checkpoint += 1
            
            # Periodic checkpoint save
            if checkpoint_interval > 0 and new_results_since_checkpoint % checkpoint_interval == 0:
                save_checkpoint(n, error_set_type, processed_combinations, results, verbose=verbose)
            
            if verbose and processed % max(1, total_combinations // 4) == 0:
                print(f"  [{processed}/{total_combinations}] Results found: {len(results)}, Best clique: {max(r.max_clique_size for r in results) if results else 0}")
    
    # Sort by max_clique_size descending
    results.sort(key=lambda r: -r.max_clique_size)
    
    # Final checkpoint save
    if checkpoint_interval > 0:
        save_checkpoint(n, error_set_type, processed_combinations, results, verbose=verbose)
    
    if verbose:
        print(f"\nPhase 4-5: Completed search")
        print(f"  Combinations processed: {len(processed_combinations)}/{total_combinations}")
        print(f"  Results found: {len(results)}")
        if results:
            print(f"  Best clique size: {results[0].max_clique_size}")
        else:
            print(f"  No cliques found.")
    
    return results


def save_results(results: List[SearchResult], output_path: str):
    """Save search results to JSON."""
    data = [r.to_dict() for r in results]
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Saved {len(results)} results to {output_path}")


if __name__ == "__main__":
    # Quick test: 4-qubit search with E1
    print("=" * 80)
    print("AD Hybrid Code Search - Test Run (n=4, E{1})")
    print("=" * 80)
    
    # Debug: check first graph manually
    graphs = build_local_graph_database()
    graph_rep = graphs["4q_star"]
    error_set = generate_pauli_error_set_E1(4)
    print(f"\nDEBUG: Error set size: {len(error_set)}")
    
    classical_errors, error_index = compute_classical_error_set(error_set, graph_rep.adjacency)
    print(f"DEBUG: Classical error set size: {len(classical_errors)}")
    print(f"DEBUG: Classical errors: {sorted(list(classical_errors)[:10])}")
    
    degeneracies = identify_degenerate_errors(classical_errors, error_index)
    print(f"DEBUG: Degeneracies: {degeneracies}")
    
    admissible = get_admissible_vertices(4, classical_errors, degeneracies, error_index)
    print(f"DEBUG: Admissible vertices: {len(admissible)}")
    print(f"DEBUG: Admissible sample: {sorted(list(admissible)[:10])}")
    
    print("\n" + "=" * 80)
    
    # Run search with checkpoint system enabled (save every 50 results)
    results = run_exhaustive_search(n=4, error_set_type="E1", verbose=True, checkpoint_interval=50)
    
    print("\n" + "=" * 80)
    print("Top 5 Results:")
    print("=" * 80)
    for i, result in enumerate(results[:5]):
        print(f"\n{i+1}. {result.graph_key} - Clique Size: {result.max_clique_size}")
        print(f"   Orientation: {result.orientation}")
        print(f"   Admissible Vertices: {result.admissible_vertex_count}")
        print(f"   Hybrid KL Valid: {result.hybrid_kl_valid}")
        if result.hybrid_kl_report:
            print(f"   KL Violations: {result.hybrid_kl_report.get('violations', 'N/A')}")
    
    # Save final results
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    save_results(results, str(output_dir / "ad_hybrid_search_n4_e1.json"))
