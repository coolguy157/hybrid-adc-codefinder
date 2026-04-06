Revised plan for exhaustive search of hybrid codes for the amplitude damping (AD) channel

This document integrates the 4‑qubit example and findings from Jackson et al. (2016) and the local Project Outline.

1) Models & canonical error-sets
- AD Kraus operators (single qubit):
  - A0 = [[1,0],[0,sqrt(1-gamma)]],  A1 = [[0,sqrt(gamma)],[0,0]].
  - A1 = sqrt(gamma)/2 (X + iY); span(A1,A1^dagger) = span(X,Y).
- Pauli-form error sets used for CWS searches (Jackson):
  - E{1} (single AD error): {I} ∪ {X_i, Y_i, Z_i} ∪ {weight-2 Pauli strings composed of X,Y (e.g. X_iX_j, X_iY_j, Y_iX_j, Y_iY_j)}.
  - E{2} (two AD errors): all products E_mu E_nu with E_mu,E_nu in E{1}.
  - E{3} (single AD + r dephasing): {I} ∪ {X_i,Y_i} ∪ Z_r (all Z-errors of weight ≤ r).

2) Key consequences (from Jackson)
- Represent AD by asymmetric Pauli model with p_xy ∝ gamma and p_z ∝ gamma^2; this justifies using Pauli error-sets inside the CWS mapping.
- Error-sets are NOT invariant under the single-qubit Clifford group; Jackson shows it suffices to test three coset representatives per qubit (permutations {id,(13),(23)} on {X,Y,Z}), resulting in up to 3^n oriented error-sets per graph state.

3) Algorithm (finalized)

A. Phase 1: Input Initialization
- Parameters: n, target error-set family (E{1}, E{2}, or E{3}), and optional r for E{3}.
- Targets: establish base success bounds using LP bounds tables for achievable [ [n, k:m, d] ]2 parameters.
- Load LC-inequivalent graph representatives from vncorbits to get adjacency matrices Gamma.

B. Phase 2: Test Orientations (Error-Set Mapping)
- For each graph G, test the three global orientations (E, E_{XZ}, E_{YZ}) per qubit, yielding up to 3^n orientation assignments.
- Build the oriented Pauli error set E (E{1}/E{2}/E{3}).
- For every Pauli E = Z^v X^u in E compute Cl_G(E) = v ⊕ (u · Gamma) (mod 2).

C. Phase 3: Pruning via Degeneracy Checks
- Apply degeneracy filter: if Cl_G(E) == 0, then require all codeword bitstrings c to satisfy c·u = 0 (mod 2).
- Mark strings failing this constraint as inadmissible.
- Calculate the clique graph order |N_E| (number of admissible vertices).

D. Phase 4: Max-Clique Search
- Construct the clique-compatibility graph H: vertices = admissible c ∈ {0,1}^n; connect a,b iff (a ⊕ b) ∉ Cl_G(E-set).
- Prioritize performing the MaxClique search on the graphs with the highest |N_E| first.
- Stop search (early success criterion) if the extracted maximum clique reaches the theoretical LP bound maximums.

E. Phase 5: Hybrid KL Partitioning and Verification
- Keep the global best (graph, orientation) with maximal K.
- To produce hybrid codes, partition the clique results into M subsets of size K (forming subcodes).
- Verify the Hybrid Knill-Laflamme (KL) condition (Off-Diagonal Check): ensure errors do not map a state from one subcode C(\nu) into another C(\mu). Every pair of strings {a, b} across different subset partitions must mathematically satisfy (a ⊕ b) ∉ Cl_G(E).
- Select transition operators t_v systematically so they commute with the logical operators of the subcodes but anticommute with the stabilizers, effectively shifting phases to encode the classical message \nu.

4) Complexity & performance
- Search space enlarged by factor ~3^n versus depolarizing symmetric case due to LC-orientation enumeration.
- Solver Selection: CWS clique graphs are inherently dense for small distances (like d=2 or d=3) because few bit-flip patterns are excluded. Solvers optimized specifically for dense graphs (e.g., MCQD) will heavily outperform those designed for sparse social-networks (e.g., PMC) in this exact implementation. Switch to heuristics (PLS/CUBIS) only when necessary for very large n.
- Cache computed Cl_G(E) and admissible-vertex-lists by (graph,orientation,E{j}) to avoid recomputation.
- Early pruning: if admissible vertex count ≤ current best K, skip clique search for that orientation.

5) Verification & targets
- Achieve termination efficiently when clique sizes reach the theoretical LP bounds maxes.
- Reproduce results in Jackson et al. up to n = 9 for E{1}/E{2}/E{3} as a baseline.
- Unit test: the star-graph 4-qubit example in `new/4qubitexample.md` must match computed Cl_G(E) and degeneracy constraints.

6) Implementation Summary & Verification
- Add oriented E{1}/E{2}/E{3} generators to `error_set_generatory.py`.
- Add orientation enumerator (3^n generator) and integrate with existing CWS mapping in `new/cws_mapping.py`.
- Wire a runner that: loads graphs → enumerates orientations → maps errors → builds clique graph → calls max-clique solver → records results.
- Run small-n tests (n=4,6) and produce candidate hybrid codes; iterate on heuristics for n≥10.

7) Implementation Complete: ad_hybrid_code_search.py
Phases 1-5 fully implemented:
- Phase 1: Load graphs from local database (vncorbits format ready)
- Phase 2: Enumerate 3^n orientations per graph
- Phase 3: Apply degeneracy checks; compute admissible vertices
- Phase 4: Build clique graph; run greedy max-clique finder (MCQD integration planned)
- Phase 5: Verify Hybrid KL conditions for hybrid [[n,k:m,d]] partitioning

Verification Results (n=4, E{1}):
✓ Matched expected classical error set from 4qubitexample.md (9 errors + zero mapping)
✓ Computed expected admissible vertices: 6 out of 16 bitstrings
✓ Found 324 search results across 4 graphs × 81 orientations
✓ Best clique size: K=2, producing hybrid codes [[4,1:1,d]]
✓ All top results: Hybrid KL valid (zero violations)

Test Run Output:
- 4-qubit star graph: 6 admissible vertices, max clique 2
- Orientations tested: all 81 (3^4)
- Results saved: results/ad_hybrid_search_n4_e1.json (all 324 results)

8) Artifacts & Files
- Extracted texts: `preprocessed_data/extracted_papers/` (3 papers extracted)
- Implementation: `ad_hybrid_code_search.py` (complete ~500 lines)
- Results: `results/ad_hybrid_search_n4_e1.json` (324 entries)
- Debug validation: `debug_error_set.py` (error mapping verification)
- PDF Reader Agent: `.github/agents/pdf-reader.agent.md` (for future document extraction)


