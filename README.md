## The plan:
The following updated procedure for an exhaustive search program incorporates the **LC-inequivalent graph database** and integrates specific methodologies from the provided research papers.

### **Phase 1: Search Space Definition & Database Integration**
1.  **Define Target Parameters:** Establish the code length $n$, number of logical qubits $k$, and classical messages $M$ based on existing theoretical performance limits [Grassl et al., 2017].
2.  **Load Orbit Representatives ($L_n$):** Retrieve the set of non-LC-isomorphic graph representatives for length $n$ directly from the database to prune the search space from $2^{n^2}$ to a manageable size [Danielsen and Parker, 2006; Rigby et al., 2019]. 
3.  **Generate LC-Equivalent Error Sets:** For asymmetric channels (like **Amplitude Damping**), generate the three distinct LC-equivalent error sets ($E$, $E_{XZ}$, $E_{YZ}$) for each graph to account for the channel's lack of LC-invariance [Jackson et al., 2016; Rigby et al., 2019].

### **Phase 2: Heuristic Filtering for Efficiency**
4.  **Load the entire list of [adjacency matrices](https://web.archive.org/web/20240803110852/https://www.ii.uib.no/~larsed/vncorbits/) and iterate through them one-by-one.**

### **Phase 3: CWS Setup and Graph Construction**
For each graph $G$ and each LC-equivalent error set:

6.  **Map Classical Error Patterns ($Cl_G(E)$):** Transform every quantum Pauli error into its binary classical representation induced by the graph state using the $X-Z$ rule [Chuang et al., 2009; Cross et al., 2009].

7.  **Identify Inadmissible Strings ($D_G(E)$):** Identify bit strings that violate the commutation conditions necessary for degenerate codes, which are common in AD channels [Chuang et al., 2009; Jackson et al., 2016].

8.  **Build the CWS Clique Graph:** Construct a graph where vertices are admissible binary strings and edges connect strings whose XOR sum does not result in an induced classical error [Chuang et al., 2009].

### **Phase 4: Algorithmic Clique Search**
9.  **Execute Search Algorithm:** 
    *   For small $n$, use exact maximum clique finders.
    *   For $n \ge 9$ or distance $d=2$, use the **Phased Local Search (PLS)** heuristic to find the target clique size $K = 2^k \cdot M$ [Rigby et al., 2019].
10. **Store Successful Candidates:** If a clique of size $K$ is found, record the adjacency matrix and classical codewords [Chuang et al., 2009].

### **Phase 5: Hybrid Partitioning and Verification**
11. **Partition into Subcodes:** Divide the discovered classical code $C$ into $M$ orthogonal subcodes $\{C(\nu)\}$, where each subcode represents a different classical message $\nu$.
12. **Verify Hybrid Knill-Laflamme Conditions:** Ensure the subcodes satisfy the hybrid error correction condition: $\langle c(\nu)_i | E^\dagger_k E_l | c(\mu)_j \rangle = \alpha^{(\nu)}_{kl} \delta_{ij} \delta_{\mu\nu}$ [Grassl et al., 2017; Nemec, 2025]. This confirms that:
    *   Quantum information is corrected within each subcode [Grassl et al., 2017].
    *   Classical messages $\nu$ are distinguishable from $\mu$ after channel noise [Grassl et al., 2017].