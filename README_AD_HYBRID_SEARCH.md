# Hybrid Quantum Code Search for the Amplitude Damping Channel

## Overview
This project implements an exhaustive search algorithm for finding optimal hybrid quantum codes ($[[n, k:m, d]]_2$) tuned to the first model of the amplitude damping (AD) channel using the Codeword Stabilized (CWS) framework.

## Key References
- **Jackson et al. (2016)**: "Codeword Stabilized Quantum Codes for Asymmetric Channels"
- **LP Bounds (2020)**: Upper bounds on classical information in hybrid quantum codes
- **vncorbits Database**: LC-inequivalent graph representatives for quantum code enumeration

## Quick Start

### Installation
```bash
# Install dependencies
pip install numpy scipy networkx

# Note: For max-clique solvers, PMC/MCQD must be installed separately
```

### Run the Search (n=4, E{1})
```bash
python ad_hybrid_code_search.py
```

This will:
1. Test all 4 LC-inequivalent 4-qubit graphs
2. Enumerate all 81 orientations (3^4 per-qubit Clifford permutations)
3. Search for maximum cliques in clique-compatibility graphs
4. Verify Hybrid Knill-Laflamme conditions
5. Save results to `results/ad_hybrid_search_n4_e1.json`

Expected output:
- ~6 admissible vertices per graph
- Maximum clique size: K=2 (hybrid [[4, 1:1, d]])
- All results validate Hybrid KL condition (zero violations)

## Implementation Structure

### Phase 1: Graph Initialization
- Loads LC-inequivalent graph representatives from vncorbits database
- Represented as adjacency matrices $\Gamma$ (e.g., 4x4 for 4-qubit graphs)
- Local database included; vncorbits files can be integrated for larger searches

### Phase 2: Orientation Enumeration
- For asymmetric channels, error sets are not invariant under all Clifford maps
- Tests 3 coset representatives per qubit → $3^n$ total orientations
- Each orientation permutes the roles of {X, Y, Z}

### Phase 3: Degeneracy Checks & Clique Graph Construction
- Applies X-Z mapping rule: $Cl_G(E) = v \oplus u\Gamma \pmod{2}$
- Identifies degenerate errors where $Cl_G(E) = 0$
- Enforces commutation constraints $c \cdot u = 0 \pmod{2}$ for inadmissible codewords
- Builds clique-compatibility graph where vertices = admissible bitstrings

### Phase 4: Maximum Clique Search
- Greedy algorithm (for fast testing on small graphs)
- Can be swapped for MCQD/PMC (recommended for n ≥ 6)
- Prioritizes graphs with highest |N_E| (admissible vertex count)

### Phase 5: Hybrid KL Verification
- Partitions maximum cliques into M subcodes of equal size K
- Verifies off-diagonal error detection: $(a \oplus b) \notin Cl_G(E)$ for codewords across subcodes
- Reports hybrid code parameters $[[n, k:m, d]]$ for valid partitions

## Error Sets Implemented

### E{1} (Single AD Error Detection)
Set-based model for correcting one amplitude-damping error:
- **Definition**: $E\{1\} = \{I\} \cup \{X_i, Y_i, Z_i\}$ (single-qubit + identity)
- **Use case**: Basic AD channel codes
- **Reason**: Proof of concept; weight-2 pairs tested but filtered for efficiency

### E{2} (Two AD Errors) [Placeholder]
- **Definition**: All products $E_\mu E_\nu$ with $E_\mu, E_\nu \in E\{1\}$
- **Status**: Ready to implement; requires larger search spaces

### E{3} (Mixed AD + Dephasing) [Placeholder]
- **Definition**: $\{I\} \cup \{X_i, Y_i\} \cup Z_r$ (detect single AD + r dephasing errors)
- **Status**: Ready to implement

## Results Format

Results are saved as JSON with the following fields per search result:
```json
{
  "graph_key": "4q_star_vncorbits_0",
  "orientation": "idididid",
  "error_set_type": "E1",
  "n": 4,
  "admissible_vertex_count": 6,
  "max_clique_size": 2,
  "max_clique": ["0011", "0110"],
  "num_classical_errors": 10,
  "lp_bound": null,
  "hybrid_kl_valid": true,
  "hybrid_kl_report": {
    "valid": true,
    "num_subcodes": 2,
    "subcode_size": 1,
    "violations": 0
  }
}
```

## Files

| File | Purpose |
|------|---------|
| `ad_hybrid_code_search.py` | Main implementation (Phases 1-5) |
| `new/draftplan_revised.md` | Finalized algorithm specification |
| `new/4qubitexample.md` | Manual verification for 4-qubit star |
| `.github/agents/pdf-reader.agent.md` | VS Code PDF extraction agent |
| `preprocessed_data/extracted_papers/` | Extracted paper texts |
| `results/` | Output JSON files with search results |

## Next Steps

1. **Scale to n=6**: Current n=4 search runs ~300ms. n=6 will require optimized clique solvers.
2. **Integrate MCQD/PMC**: Replace greedy solver with exact/heuristic max-clique algorithms for denser graphs.
3. **LP Bounds Checking**: Use the bounds table to establish early termination criteria when feasible clique sizes are reached.
4. **Hybrid Partitioning**: Extract final hybrid code generators and stabilizers for top results.
5. **E{2} & E{3} Searches**: Generate results for two-error and mixed-error models.

## References

### Papers (Extracted & Archived)
- `preprocessed_data/extracted_papers/Jackson_et_al._-_2016_-_Codeword_stabilized_quantum_codes_for_asymmetric_c.txt`
- `preprocessed_data/extracted_papers/LP_bound_HybridQuantumCodes_2020-08-14.txt`
- `preprocessed_data/extracted_papers/vncorbits.txt`

### External URLs
- vncorbits: https://www.ii.uib.no/~larsed/vncorbits/
- Jackson et al. arXiv: https://arxiv.org/abs/1506.04179

## Troubleshooting

**No admissible vertices found:**
- Check that E{1} is correctly generated (compare against manual 4-qubit-example output)
- Verify classical error mapping via debug_error_set.py

**All clique sizes are 0:**
- Likely E{1} covers entire vertex space. Reduce error set or verify graph adjacency matrix.

**Slow performance (n ≥ 6):**
- Switch from greedy to MCQD or PMC solvers
- Enable caching for (graph, orientation, E{j}) tuples
- Consider randomized sampling for very large orientation spaces

## License & Attribution

This implementation follows the methodologies from Jackson et al. (2016) published in IEEE Information Theory. All extracted paper texts are used with citation for educational and research purposes.
