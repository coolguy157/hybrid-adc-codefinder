The graph-based method for searching for codes over the **amplitude damping (AD)** channel relies on the **Codeword Stabilized (CWS)** framework, which translates the quantum error-correction problem into a classical search for a **maximum clique** in an induced graph [Chuang et al., 2009; Rigby et al., 2019].

The following steps detail the search process and explain what each graph represents in terms of quantum codes:

### **1. Initialization: Loading Graph State Representatives**
The program begins by loading a set of **undirected graphs ($L_n$)** that are inequivalent under local Clifford (LC) operations [Rigby et al., 2019]. 
*   **Quantum Correspondence:** Each graph $G$ in the database corresponds to a **graph state $|G\rangle$**, which serves as the **codeword stabilizer state**—the fundamental "seed" state that defines the basis of the quantum code [Cross et al., 2009; Chuang et al., 2009].

### **2. Asymmetric Error Model Mapping**
Because the AD channel is asymmetric, the program must represent its Kraus operators ($A_0, A_1$) in terms of their **linear span in the Pauli group** [Jackson et al., 2016]. For a code to correct $t$ errors, it must detect the set $E\{t\}$, which includes Pauli $X$ and $Y$ operators (representing bit flips and combined flips) and $Z$ operators (representing phase flips) [Jackson et al., 2016].
*   **Asymmetry Handling:** Unlike the depolarizing channel, AD errors are not invariant under LC operations [Jackson et al., 2016]. Therefore, for every graph $G$, the program must test **three LC-equivalent orientations** of the error set: the base set ($E$), a set with $X \leftrightarrow Z$ swapped ($E_{XZ}$), and a set with $Y \leftrightarrow Z$ swapped ($E_{YZ}$) [Jackson et al., 2016; Rigby et al., 2019].

### **3. Applying the X-Z Rule: Induced Classical Errors**
For each graph and each error set orientation, the program uses the **X-Z rule** to map quantum Pauli errors into binary strings [Cross et al., 2009]. An $X$ error at node $i$ is "pushed" along the edges of the graph state to become a $Z$ error on all neighboring nodes [Cross et al., 2009; Rigby et al., 2019].
*   **Quantum Correspondence:** This mapping, $Cl_G(E) = v \oplus u\Gamma$ (where $\Gamma$ is the adjacency matrix), transforms the complex quantum noise model into a **classical bit-flip error model** [Chuang et al., 2009; Cross et al., 2009].

### **4. Degeneracy Filtering: Identifying Inadmissible Strings**
The program identifies a set of bit strings $D_G(E)$ that are "inadmissible" because they violate the commutation requirements for **degenerate codes** [Chuang et al., 2009]. 
* If the error mapping $Cl_G$ is nonzero, then it is detectable , otherwise we need to check if it actually affects the code, so the error must commute with all chosen codeword operators
### **2. The Secondary Check: Commutation for Degeneracy**
When a quantum error maps to zero, the result is only valid if the error $E$ **commutes** with all chosen codeword operators $Z_c$. 
*   **The Algebraic Requirement:** For a codeword $c$ to be admissible when $Cl_G(E) = 0$, the condition $Z_c E = E Z_c$ must be met.
*   **The Binary String Check:** In standard form, this commutation check is simplified to a binary inner product: $c \cdot u = 0$, where $u$ is the string representing the locations of $X$ operators in the error $E$. 

*   **Populating $Cl_G(E)$:** If the result is non-zero, the string is marked in the set of classical errors that the code's distance must account for.
*   **Pruning via $D_G(E)$:** If the result is zero, the program iterates through all possible $n$-bit strings $i$. Any string where the inner product $i \cdot u \neq 0$ is added to the **inadmissible set $D_G(E)$**. These strings are subsequently removed from the pool of potential vertices for the CWS clique graph, ensuring any found code automatically satisfies the diagonal Knill-Laflamme conditions. 
*   **Quantum Correspondence:** Degeneracy is critical for AD channels as it allows some low-weight errors to act as a multiple of identity on the code space [Jackson et al., 2016]. These bit strings correspond to quantum basis vectors that would be corrupted by errors mapping to the zero classical string [Chuang et al., 2009; Jackson et al., 2016].

### **5. Construction of the CWS Clique Graph**
The program constructs a new graph, the **CWS clique graph ($GE$)**, which is used solely for the search [Chuang et al., 2009].
*   **Vertices ($V$):** Each vertex in the clique graph corresponds to an **admissible binary string** $x$ that can potentially be used as a codeword [Chuang et al., 2009].
*   **Edges ($E$):** An edge connects two vertices $x_i, x_j$ if their XOR sum is not an element of the induced classical error set $Cl_G(E)$ [Chuang et al., 2009]. 
*   **Quantum Correspondence:** This graph represents the **compatibility** of basis vectors. If two vertices share an edge, the corresponding quantum states $|x_i\rangle, |x_j\rangle$ (formed by applying $Z^{x_i}$ to the graph state) remain orthogonal even after the channel noise occurs [Chuang et al., 2009; Cross et al., 2009].

### **6. Maximum Clique Search and Hybrid Partitioning**
The program executes a search algorithm (like **Phased Local Search**) to find the largest possible **clique** of size $K$ in the clique graph [Rigby et al., 2019]. 
*   **Quantum Correspondence:** The found clique $C$ defines the **word operators** $W = \{Z^c : c \in C\}$ of the quantum code [Chuang et al., 2009].
*   **Hybrid Codes:** For **hybrid quantum-classical codes**, the clique is further partitioned into $M$ subcodes $\{C(\nu)\}$. Each subcode protects $k$ qudits while its index $\nu$ encodes a **classical message** [Grassl et al., 2017; Nemec, 2025]. The program then verifies that errors do not map one subcode's subspace into another, satisfying the **Hybrid Knill-Laflamme condition** [Grassl et al., 2017; Nemec, 2025].

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
