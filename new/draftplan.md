To implement an exhaustive search program for quantum codes based on the 4-qubit Codeword Stabilized (CWS) example, you must integrate the structural theorems of the CWS framework with a graph-theoretic search loop. 

Since you are excluding amplitude damping channel symmetry, the search focuses on **symmetric codes** where the error set $\mathcal{E}$ is defined by a fixed distance $d$ (e.g., all Pauli errors of weight less than $d$).

### **Phase 1: Input and Error Set Initialization**
1.  **Define Parameters:** Set $n$ (physical qubits) and $d$ (target distance). For your 4-qubit example, $n=4$.
2.  **Generate the Error Set ($\mathcal{E}$):** Construct the set of all Pauli errors $E = Z_v X_u$ with weight $w(E) < d$. 
    *   *Example ($d=2$):* $\mathcal{E}$ includes all single-qubit $X_i, Y_i, Z_i$.
3.  **Load Graphs ($L_n$):** To perform an **exhaustive** search, you only need to test graphs that are not Local Clifford (LC) isomorphic. For $n=4$, there are only **6** such inequivalent graphs.

### **Phase 2: The Graph Processing Loop (The Core Algorithm)**
For each graph $G$ in the set of LC-inequivalent graphs, execute the following three subroutines:

#### **Subroutine A: Setup ($Cl_G(E)$ and $D_G(E)$)**
This step transforms quantum constraints into classical bitstrings.
*   **Classical Mapping ($CL$):** For every error $E = Z_v X_u \in \mathcal{E}$, calculate the induced classical error string using the **X-Z rule**: $Cl_G(E) = v \oplus u\Gamma$, where $\Gamma$ is the adjacency matrix of $G$.
*   **Degeneracy Check ($D$):** Identify "inadmissible" strings. If $Cl_G(E) = \mathbf{0}$, iterate through all $2^n$ bitstrings $i$. If the binary inner product $i \cdot u \neq 0 \pmod 2$, then bitstring $i$ is added to the inadmissible set $D_G(E)$.

#### **Subroutine B: Clique Graph Construction**
Build a graph $G_E$ where nodes represent potential codewords:
*   **Vertices ($V$):** Include all $n$-bit strings $s$ that are not in the induced classical error set ($CL[s]=0$) and not inadmissible ($D[s]=0$).
*   **Edges ($E$):** Connect two vertices $v, w$ if their XOR sum is not an element of the classical error set: $v \oplus w \notin Cl_G(\mathcal{E})$. 

#### **Subroutine C: MaxClique Subroutine**
Call a subroutine (like **findMaxClique**) to find the largest fully connected set of nodes in $G_E$.
*   The size of this clique represents the maximum dimension **$K$** for that specific graph.
*   The elements of the clique are the bitstrings used for your **word operators** $\{Z^c : c \in \text{Clique}\}$.

### **Phase 3: Results and Code Generation**
1.  **Identify the Global Maximum:** Compare the clique size $K$ across all 6 LC-inequivalent graphs. The graph(s) yielding the largest $K$ are your optimal CWS codes for length $n$.
2.  **Basis Construction:** For the best graph found, apply the word operators to the graph state $|G\rangle$ to form the code basis.
    *   *Example:* If the clique is $\{0000, 0011\}$, your basis is $\{|G\rangle, Z_3 Z_4 |G\rangle\}$.
3.  **Hybrid Partitioning (Optional):** If a hybrid code $[[n, k:m, d]]$ is desired, partition the found clique of size $KM$ into $M$ subcodes of size $K$. 
    *   Select one subcode as the seed code.
    *   Identify a **transition operator** $t_\nu$ (a bitstring from the clique not in the seed) to map the seed to the other subcodes.

### **Complexity Considerations for Implementation**
*   **Search Space:** Because $n=4$ is small, you can use an **exact clique finder**. For $n \ge 10$, you would switch to a heuristic like Phased Local Search (PLS).
*   **Efficiency:** Instead of storing the entire clique graph in memory, you can compute edges "on the fly" during the max-clique search to save space.
*   **Redundancy:** By using only LC-inequivalent graphs, you reduce the required graph tests from $2^{n^2}$ to approximately $3^n$ (specifically 6 graphs for $n=4$).

---

# Using Vncorbits

The **vncorbits database** serves as the primary source for the **Phase 1: Input Initialization** of an exhaustive search program. It provides the set of non-LC-isomorphic graph representatives needed to cover all possible quantum codes for a given number of qubits ($n$) without redundant calculations.

According to the sources, here is specifically how it is used in the beginning steps:

### **1. Pruning the Graph Search Space**
To find all possible CWS codes for a given length $n$, one would theoretically need to test all $2^{n^2}$ possible graphs. However, the database identifies **local Clifford (LC) equivalent** graph orbits. Because any CWS code is LC-equivalent to one in standard form based on a graph state, it is sufficient to test only one representative from each of these LC-inequivalent orbits. 
*   For your **4-qubit example**, the database reveals there are only **6** total inequivalent graphs to test, rather than the hundreds of possible distinct graphs. 
*   For larger searches, like $n=11$, it reduces the search space from billions of possible graphs to a manageable **45,144** representatives.

### **2. Providing Adjacency Matrix Inputs**
The database stores its representatives in **`graph6` format**, a compact binary representation of a graph’s structure. 
*   In the **Setup subroutine** of your implementation, you would load these strings and convert them into the **adjacency matrix ($\Gamma$)**.
*   Note that the database explicitly states that a graph with adjacency matrix $\Gamma$ represents a code with a generator matrix $\Gamma + \omega I$.

### **3. Initializing Symmetric and Asymmetric Searches**
*   **Symmetric Searches:** For standard distance-$d$ codes (symmetric codes), the database provides a complete list of starting points.
*   **Asymmetric Searches (Amplitude Damping):** Even when dealing with asymmetric channels where error sets are not LC-invariant, the database is still used as the base set of graphs. The search is made exhaustive by repeating the process for each graph against **three specific LC-equivalent error set orientations** ($E, E_{XZ}, E_{YZ}$).

### **4. Pre-filtered Graph Selection**
The database allows you to pre-select graphs based on their properties before starting the heavy max-clique search:
*   **Indecomposable Graphs:** You can choose to download only connected (indecomposable) graphs, which correspond to simpler, more efficient codes.
*   **Distance Filtering:** The database lists the minimum distance $d$ of the stabilizer states (the weight of the lowest nonzero element in the stabilizer group). This can help you focus on graphs that already possess a certain baseline level of protection.

In summary, the vncorbits database acts as the **standardized "input list"** for your program, allowing you to skip the exponential cost of generating and filtering $n$-node graphs from scratch.

---

# Possible Libraries

PMC (Parallel Maximum Clique) Library: A high-performance C++ library specifically designed for large, sparse social and information networks. It uses core numbers and advanced pruning to solve for networks with millions of nodes in seconds.Source: ryanrossi/pmc on GitHubMaxCliqueDyn (MCQD): An extremely fast exact algorithm that uses an improved approximate coloring (ColorSort) and dynamic vertex sorting to prune the search space. It is widely used in bioinformatics (e.g., protein structure comparison).Source: MaxCliqueDyn at InsilabFast Max-Clique Finder: Developed at Northwestern University, this algorithm uses hierarchical pruning and is optimized for massive sparse graphs, often outperforming older standards like Östergård's Cliquer.Source: Northwestern CUCIS MaxClique2. Fastest Heuristics (Massive Graphs)When graphs have billions of edges, exact solutions may be too slow. Heuristics find "large" cliques very quickly but do not guarantee they are the "maximum."PMC Heuristic: Included in the PMC library, this greedy approach can find the largest clique in over half of studied social networks without needing the full branch-and-bound search.CUBIS (Complete-Upper-Bound-Induced Subgraph): A recent (2024) approach that decomposes massive graphs into small-scale subgraphs based on core numbers, allowing for approximately linear runtime on networks with up to 20 million nodes.