import unittest
import numpy as np
import networkx as nx
import json
import os
import tempfile
from qiskit.quantum_info import Pauli

# --- IMPORTS ---
# We try to import from all modules. 
# We set functions to None if import fails so tests fail gracefully with a clear message.

try:
    from old.run_hybrid_search import (
        find_null_space_gf2, 
        check_single_hkl_condition, 
        _generate_ead_error_set as get_ead_hybrid, # Rename to avoid conflict
        get_code_fingerprint
    )
except ImportError:
    find_null_space_gf2 = None

try:
    from run_cws_search import (
        get_induced_error, 
        build_conflict_graph, 
        get_ead_error_set as get_ead_cws # Rename to avoid conflict
    )
except ImportError:
    get_induced_error = None

try:
    from results import load_results
except ImportError:
    load_results = None

# --- HYBRID SEARCH TESTS ---

class TestGF2LinearAlgebra(unittest.TestCase):
    def test_null_space_simple(self):
        # [1 1] -> x + z = 0 -> [1 1]
        matrix = np.array([[1, 1]], dtype=int)
        null_space = find_null_space_gf2(matrix)
        self.assertEqual(null_space.shape[1], 2)
        product = np.dot(matrix, null_space.T) % 2
        self.assertTrue(np.all(product == 0))

    def test_null_space_identity(self):
        # Identity has trivial null space
        matrix = np.eye(3, dtype=int)
        null_space = find_null_space_gf2(matrix)
        self.assertEqual(null_space.shape[0], 0)

    def test_null_space_dependent_rows(self):
        # Rows are same, so rank 1. n=3. Nullity should be 2.
        matrix = np.array([[1, 0, 1], [1, 0, 1]], dtype=int)
        null_space = find_null_space_gf2(matrix)
        product = np.dot(matrix, null_space.T) % 2
        self.assertTrue(np.all(product == 0))
        self.assertEqual(null_space.shape[0], 2)

class TestHybridErrorModels(unittest.TestCase):
    def test_ead_set_size_n2(self):
        n = 2
        error_set = get_ead_hybrid(n)
        self.assertEqual(len(error_set), 10) # 1 + 6 + 3

    def test_ead_contents(self):
        n = 2
        error_set = [str(p) for p in get_ead_hybrid(n)]
        self.assertIn("II", error_set)
        self.assertIn("XI", error_set)
        self.assertNotIn("ZZ", error_set)

class TestHKLConditions(unittest.TestCase):
    def setUp(self):
        self.n = 2
        self.S_gens = [Pauli("ZZ")]

    def test_hkl_inter_code_orthogonality(self):
        # Different translations must be distinguishable (anticommute with S)
        t_a, t_b = Pauli("XI"), Pauli("II") 
        is_valid = check_single_hkl_condition(t_a, t_b, Pauli("II"), Pauli("II"), Pauli("II"), Pauli("II"), self.S_gens)
        self.assertTrue(is_valid)

    def test_hkl_inter_code_failure(self):
        # Identical effect (commutes with S) -> Indistinguishable -> Bad
        t_a, t_b = Pauli("ZI"), Pauli("II")
        is_valid = check_single_hkl_condition(t_a, t_b, Pauli("II"), Pauli("II"), Pauli("II"), Pauli("II"), self.S_gens)
        self.assertFalse(is_valid)

# --- CWS SEARCH TESTS ---

class TestCWSLogic(unittest.TestCase):
    
    def test_induced_error_basic(self):
        # Graph: 0-1 (Path 2)
        # Adj: [[0,1], [1,0]]
        # Error: X on qubit 0 (Index 0). 
        # CWS Rule: X on node i -> Z on neighbors N(i).
        # Neighbor of 0 is 1. Result: Z on qubit 1.
        # Vector: [0, 1]
        
        adj = np.array([[0, 1], [1, 0]])
        # Qiskit Pauli("IX") means X on Qubit 0 (Little Endian Index 0)
        p = Pauli("IX") 
        
        induced = get_induced_error(adj, p)
        self.assertEqual(induced, (0, 1))

    def test_conflict_graph_size(self):
        # n=3, Error={I}. Bad Diffs = {0}.
        # Graph should have no edges. Max Independent Set = All nodes (2^3=8).
        n = 3
        adj = np.zeros((3,3), dtype=int)
        errors = [Pauli("III")]
        
        G = build_conflict_graph(n, adj, errors)
        
        self.assertEqual(G.number_of_edges(), 0)
        
        # We can calculate independent set size for empty graph easily
        # (It's just the number of nodes)
        self.assertEqual(G.number_of_nodes(), 8)

    def test_ead_generation(self):
        # Verify amplitude damping set logic is consistent
        errors = get_ead_cws(3)
        self.assertEqual(len(errors), 19) # 1 + 9 + 9

# --- REPORTING TESTS ---

class TestReporting(unittest.TestCase):
    def test_load_results(self):
        with tempfile.NamedTemporaryFile(mode='w+', delete=False) as tmp:
            tmp.write('{"parameters": {"n": 5, "k": 1, "m": 1}, "d_seed": 3}\n')
            tmp_path = tmp.name
        try:
            codes = load_results(tmp_path)
            self.assertEqual(len(codes), 1)
            self.assertEqual(codes[0]['n'], 5)
        finally:
            os.remove(tmp_path)

if __name__ == '__main__':
    unittest.main()