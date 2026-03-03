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
(Essentially, you are converting all XYZ mized errors into just Z errors)

7.  **Identify Inadmissible Strings ($D_G(E)$):** Identify bit strings that violate the commutation conditions necessary for degenerate codes, which are common in AD channels [Chuang et al., 2009; Jackson et al., 2016].

8.  **Build the CWS Clique Graph:** Construct a graph where vertices are admissible binary strings and edges connect strings whose XOR sum does not result in an induced classical error [Chuang et al., 2009].

### **Phase 4: Algorithmic Clique Search**
9.  **Execute Search Algorithm:** 
    *   For small $n$, use exact maximum clique finders.
    *   For $n \ge 9$ or distance $d=2$, use the **Phased Local Search (PLS)** heuristic to find the target clique size $K = 2^k \cdot M$ [Rigby et al., 2019].
10. **Store Successful Candidates:** If a clique of size $K$ is found, record the adjacency matrix and classical codewords [Chuang et al., 2009].

### **Phase 5: Hybrid Partitioning and Verification**
11. **Partition into Subcodes:** Divide the discovered classical code $C$ into $M$ orthogonal subcodes $\{C(\nu)\}$, where each subcode represents a different classical message $\nu$.
12. **Verify Hybrid Knill-Laflamme Conditions:** Ensure the subcodes satisfy the hybrid error correction condition: $$\langle c(\nu)_i | E^\dagger_k E_l | c(\mu)_j \rangle = \alpha^{(\nu)}_{kl} \delta_{ij} \delta_{\mu\nu}$$ [Grassl et al., 2017; Nemec, 2025]. This confirms that:
    *   Quantum information is corrected within each subcode [Grassl et al., 2017].
    *   Classical messages $\nu$ are distinguishable from $\mu$ after channel noise [Grassl et al., 2017].

---

## Quantum Only Search
A direct stabilizer generator search is a "purely quantum" method that constructs codes by identifying an **Abelian subgroup** of the $n$-qubit Pauli group without the prerequisite of mapping to a classical bit-flip code. This approach focuses on the algebraic relations between stabilizer generators, logical operators, and the physical error set.

The following detailed steps outline how to conduct an exhaustive search using direct stabilizer generation and subspace verification:

### **Phase 1: Search Initialization and Operator Definition**
1.  **Define Target Parameters:** Set the physical length $n$, the number of qudits $k$, the number of classical messages $M$ (where $M = 2^m$), and the error set $\mathcal{E}$ (e.g., Amplitude Damping).
2.  **Initialize Operator Pool:** Generate the set of all $n$-qubit Pauli operators $P_n$. In binary form, these are represented as $2n$-bit vectors $(g_X | g_Z)$.
3.  **Define the Error Model Distance:** If the goal is a distance-$d$ code, the program must consider all products $E_i^\dagger E_j$ where $E$ are errors of weight less than $d/2$ (for correction) or weight less than $d$ (for detection).

### **Phase 2: Direct Stabilizer Group Search**
4.  **Enumerate Subgroup Generators:** Search for a set of $n-(k+m)$ linearly independent Pauli operators $\{S_1, \dots, S_{n-(k+m)}\}$ that form the **stabilizer group $S$**. 
5.  **Enforce Commutation (Symplectic Constraint):** For the group to be Abelian (commuting), the program must verify that every pair of generators $(a|b)$ and $(c|d)$ satisfies the **symplectic inner product**: $a \cdot d \oplus b \cdot c = 0$.
6.  **Exclude Negative Identity:** Ensure the group does not contain $-I$, as this would make the codespace zero-dimensional.

### **Phase 3: Logical and Transition Operator Construction**
7.  **Identify Transition Operators ($T$):** For hybrid codes, select $m$ operators $\{T_1, \dots, T_m\}$ that serve as **classical logical operators**. These must commute with all quantum logical operators but may anticommute with elements of the stabilizer group to shift between orthogonal subspaces.
8.  **Identify Quantum Logical Operators ($L$):** Construct $2k$ logical operators $X_1, \dots, X_k$ and $Z_1, \dots, Z_k$ that commute with all stabilizer generators and transition operators. Choose these from the normalizer 
9.  **Form Subspace Bases:** Define the $M$ orthogonal subspaces $C(\nu) = t_\nu C(0)$, where $C(0)$ is the seed code stabilized by $S$ and $\{t_\nu\}$ are the $M$ elements generated by the transition operators.

### **Phase 4: Direct Subspace Verification (KL Conditions)**
Instead of a clique search, the program performs matrix-based verification of the **Knill-Laflamme (KL) conditions** directly on the subspaces:

10. **Verify Internal Correction (Diagonal KL):** For each subcode $C(\nu)$ with projector $P_\nu$, ensure that for all $E_i, E_j \in \mathcal{E}$: 
    $$P_\nu E_i^\dagger E_j P_\nu = \alpha^{(\nu)}_{ij} P_\nu$$
    This confirms that quantum information is protected within that message's subspace.
11. **Verify Classical Distinguishability (Off-Diagonal Hybrid KL):** Ensure that for any two distinct classical messages $\nu \neq \mu$:
    $$P_\nu E_i^\dagger E_j P_\mu = 0$$
    This "Hybrid KL" condition confirms that errors do not map one classical message's subspace into another, allowing classical information to be retrieved independently.
12. **Check for Degeneracy:** If the code is meant to outperform standard bounds (essential for AD channels), identify errors $E \in \mathcal{E}$ that act as a multiple of the identity on the codespace ($E \in S$). These errors do not require unique detection, allowing more codewords to be packed into the space.

### **Phase 5: Handling Asymmetry and Optimality**
13. **Address Amplitude Damping Asymmetry:** Because the AD error set is not local Clifford (LC) invariant, the entire search and verification process must be repeated for the **three orientations** of the Pauli error model: $E, E_{XZ},$ and $E_{YZ}$.
14. **Compare Against LP Bounds:** Use the resulting parameters $((n, K:M, d))$ to verify if the code saturates the **Linear Programming (LP) and Shadow Enumerator bounds**.

---

## Searching in "code land":

To conduct an exhaustive search for hybrid codes completely in **"code land"** (the algebraic/classical coding perspective), a program follows the **AC06 framework** (introduced by Aggarwal and Calderbank). This method essentially works in the reverse direction of the CWS-MaxClique algorithm: rather than starting with a graph and finding codewords, it starts with a **classical code** and attempts to design a matching **quantum stabilizer state** [Chuang et al., 2009; Cross et al., 2009].

The following steps detail the execution of this search:

### **Phase 1: Search Space Initialization (Classical Foundation)**
1.  **Define Target Parameters:** Establish the code length $n$, target quantum dimension $K$ (where $K = 2^k \cdot M$ for hybrid codes), and minimum distance $d$ [Grassl et al., 2017].
2.  **Represent Code as a Boolean Function:** In code land, a quantum code is associated with a nonzero **Boolean function** $f: \{0,1\}^n \rightarrow \{0,1\}$ [Chuang et al., 2009]. The classical code $C_f$ is defined as the set of all $n$-bit vectors where the function evaluates to 1 ($f(c)=1$) [Chuang et al., 2009].
3.  **Compute the Complementary Set ($Cset_f$):** Identify the set of **classically detectable errors** for the chosen code. A vector $a$ is in $Cset_f$ if no codeword is mapped back into the code by $a$ (i.e., $f(c) \oplus f(c \oplus a) = 0$ for all $c$) [Chuang et al., 2009].
4.  **Prune via Column Reductions:** To reduce the search space from $2^{2^n}$ possible Boolean functions, only consider codes $C_f$ that are inequivalent under **column reductions** [Chuang et al., 2009]. For $K \leq n$, this is equivalent to the classification of all $(K, n')$ binary linear codes [Chuang et al., 2009].

### **Phase 2: Stabilizer Matrix Construction**
5.  **Generate Matrix Candidates ($A_f$):** Search for an $n \times 2n$ matrix $A_f$ where each column is an $n$-bit vector [Chuang et al., 2009].
6.  **Enforce Symplectic Constraints:** To be a valid quantum stabilizer state, the matrix must satisfy two algebraic conditions:
    *   **Linear Independence:** The rows of $A_f$ must be linearly independent [Chuang et al., 2009].
    *   **Symplectic Orthogonality:** The pairwise **symplectic inner product** of any two rows must be zero ($a \cdot d \oplus b \cdot c = 0$) [Grassl & Rötteler, 2008]. This ensures the corresponding Pauli operators commute [Grassl & Rötteler, 2008; Chuang et al., 2009].
7.  **Identify the Seed State:** This matrix $A_f$ corresponds to a unique quantum state $|S\rangle$ stabilized by the group $S$ [Chuang et al., 2009].
    - Specifically, through the *stabilizer formalism*, we view it as two concatenated $n\times n$ blocks $(X|Z)$. These are then the generators of our stabilizer group if the two conditions above are satisfied. 

### **Phase 3: Distance and Error Model Verification**
8.  **Define the Symplectic Error Set ($D_d$):** Generate the set of all $2n$-bit vectors $w$ with a **symplectic weight** (number of nonzero places) less than $d$ [Chuang et al., 2009].
9.  **Apply the AC06 Error Condition:** For every $w \in D_d$, calculate the induced classical error $e = A_f w^T$ [Chuang et al., 2009]. 
10. **Validate Code Distance:** The code is valid if and only if every induced error $e$ is an element of the **complementary set $Cset_f$** [Chuang et al., 2009]. This proves that no quantum error of weight $< d$ maps a state in the code to another state in the code.
11. **Account for Asymmetry (AD Channel):** Because the **Amplitude Damping** error set is not LC-invariant, the search must test candidate matrices against the three LC-equivalent Pauli error models (base, $X \leftrightarrow Z$ permuted, and $Y \leftrightarrow Z$ permuted) [Jackson et al., 2016].

### **Phase 4: Hybrid Partitioning and Degeneracy**
12. **Partition for Classical Messages:** For a hybrid code $[[n, k:m, d]]$, partition the discovered classical code $C_f$ into $M$ orthogonal subcodes $\{C(\nu)\}$, each of dimension $2^k$ [Grassl et al., 2017].
13. **Incorporate Degeneracy:** To outperform traditional codes, search for **degenerate codes** by constraining the Boolean function $f$ such that it is stabilized by specific elements of the Pauli group that map to the zero string [Chuang et al., 2009].
14. **Verify Hybrid KL Conditions:** Confirm the hybrid error-correction condition: $\langle c(\nu)_i | E^\dagger_k E_l | c(\mu)_j \rangle = \alpha^{(\nu)}_{kl} \delta_{ij} \delta_{\mu\nu}$ [Grassl et al., 2017]. This ensures classical information remains distinguishable independently of the quantum information [Grassl et al., 2017; Nemec, 2025].

### **Phase 5: Output and Comparison**
15. **Saturation and Optimality:** Compare the dimension $K$ of the found code against **Linear Programming (LP) bounds** and **shadow enumerators** to determine if the result is optimal [Grassl et al., 2017; Chuang et al., 2009].
16. **Conversion to Standard Form:** If a result is found in code land, it can be mapped back to the standard CWS form using **LC transformations**(map the generators using $g' = Lg_k L^\dagger$ and word operators similarly) to identify its corresponding graph state [Chuang et al., 2009; Cross et al., 2009].