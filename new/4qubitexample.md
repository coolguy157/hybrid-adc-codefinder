To perform a search for quantum codes optimized for the amplitude damping (AD) channel using the Codeword Stabilized (CWS) framework, a program must execute specific algebraic checks to handle the channel's asymmetry and potential degeneracy.

The following example uses a 4-qubit star graph (where qubit 1 is the center connected to 2, 3, and 4) to demonstrate these checks.

  

1. Setup: Mapping the AD Error Set

Because the AD channel is defined by Kraus operators that are not Pauli operators, the search first maps them to their linear span in the Pauli group. To correct a single AD error, the code must detect the set $E^{\{1\}} = \{I, X_i, Y_i, Z_i, X_i X_j, Y_i Y_j, X_i Y_j\}$.

The program applies the X-Z rule ($Cl_G(E) = v \oplus u\Gamma$) to transform these quantum errors into classical bit-flip patterns.
Adjacency Matrix ($\Gamma$): $\begin{pmatrix} 0& 1  & 1 & 1 \\ 1 & 0 & 0 & 0 \\ 1 & 0 & 0 & 0 \\ 1 & 0 & 0 & 0\end{pmatrix}$

Mapping a Phase Flip ($Z_1$): $u=0, v=1000$. $Cl_G(Z_1) = \mathbf{1000}$.

Mapping a Bit Flip ($X_2$): $u=0100, v=0$. $Cl_G(X_2) = 0 \oplus (\text{Row 2}) = \mathbf{1000}$.

#### **1. Represent the Quantum Error ($E$)**

Any Pauli error can be split into an $X$-part ($u$) and a $Z$-part ($v$).

- **Example:** Error $X_1$ (Bit flip on the center).
    
- $u = [1, 0, 0, 0]$ (There is an $X$ on qubit 1).
    
- $v = [0, 0, 0, 0]$ (There are no $Z$s).
    

#### **2. The Matrix Multiplication ($u\Gamma$)**

This is where the "magic" happens. In a graph state, an $X$ error on a qubit is physically equivalent to $Z$ errors on all of its neighbors. The adjacency matrix tells us exactly who those neighbors are.

- We multiply the vector $u$ by the matrix $\Gamma$:
    
    $$u\Gamma = [1, 0, 0, 0] \begin{pmatrix} 0 & 1 & 1 & 1 \\ 1 & 0 & 0 & 0 \\ 1 & 0 & 0 & 0 \\ 1 & 0 & 0 & 0 \end{pmatrix} = [0, 1, 1, 1]$$
    
- **Result:** The $X$ error on the center has "propagated" to become classical bit flips on qubits 2, 3, and 4.
    

#### **3. The Final XOR ($v \oplus u\Gamma$)**

Finally, we combine the original $Z$ errors ($v$) with the propagated errors ($u\Gamma$) using XOR (modulo 2 addition).

- If we had a $Z_2$ error as well, $v$ would be $[0, 1, 0, 0]$.
    
- Final String: $[0, 1, 0, 0] \oplus [0, 1, 1, 1] = [0, 0, 1, 1]$.
    
2. The Degeneracy Check (Zero-Mapping)

Degeneracy occurs when a non-identity quantum error $E$ maps to the all-zero classical string ($Cl_G(E) = \mathbf{0000}$).

Example Case: Consider the weight-2 error $E = Z_1 X_2$.$v = 1000$ (the $Z$ part).

$u\Gamma = 1000$ (the $X$ part mapping through Row 2).

Mapping: $Cl_G(Z_1 X_2) = 1000 \oplus 1000 = \mathbf{0000}$.

---
Error    | u (X)  | v (Z)  | u*Gamma  | Final Cl(E)
-------------------------------------------------------
X1       | 1000   | 0000   | 0111     | 0111      
Y1       | 1000   | 1000   | 0111     | 1111      
Z1       | 0000   | 1000   | 0000     | 1000      
X2       | 0100   | 0000   | 1000     | 1000      
Y2       | 0100   | 0100   | 1000     | 1100      
Z2       | 0000   | 0100   | 0000     | 0100      
X3       | 0010   | 0000   | 1000     | 1000      
Y3       | 0010   | 0010   | 1000     | 1010      
Z3       | 0000   | 0010   | 0000     | 0010      
X4       | 0001   | 0000   | 1000     | 1000      
Y4       | 0001   | 0001   | 1000     | 1001      
Z4       | 0000   | 0001   | 0000     | 0001      

Unique Classical Error Set (Cl_G):
['0001', '0010', '0100', '0111', '1000', '1001', '1010', '1100', '1111']
**Your current pool (Strings not in ClG​):** Based on your list, 7 strings remain: `{0000, 0011, 0101, 0110, 1011, 1101, 1110}`

### **Step 2: Apply the Degeneracy Constraint**

As you noted in your initial example, the weight-2 error Z1​X2​ is degenerate because it maps to 0000. For the code to be valid, all codewords must satisfy the commutation check c⋅u=0(mod2), where u is the X-part of the degenerate error.

- **For Z1​X2​**: The X-part u is `0100`.
    
- **Constraint**: The second bit (c2​) of every codeword must be **0**.
    

**Refined Vertex Pool:** We prune any string from Step 1 where the second bit is `1`:

1. `0000` (OK)
    
2. `0011` (OK)
    
3. `0101` (Pruned - bit 2 is 1)
    
4. `0110` (Pruned - bit 2 is 1)
    
5. `1011` (OK)
    
6. `1101` (Pruned - bit 2 is 1)
    
7. `1110` (Pruned - bit 2 is 1)
    

**Final Vertex Pool for Clique Search:** `{0000, 0011, 1011}`

### **Step 3: Build the Clique Graph (Connectivity)**

Now, we look at the remaining three strings and determine which ones can "see" each other. Two strings ci​ and cj​ have an edge between them if their XOR sum is **not** in your error set: ci​⊕cj​∈/ClG​.

1. **Check (0000, 0011)**: XOR sum = `0011`. Is `0011` in ClG​? **No.** (Edge exists)
    
2. **Check (0000, 1011)**: XOR sum = `1011`. Is `1011` in ClG​? **No.** (Edge exists)
    
3. **Check (0011, 1011)**: XOR sum = `1000`. Is `1000` in ClG​? **Yes.** (No edge)
    

### **Step 4: Find the Maximum Clique and Construct the Basis**

The "Maximum Clique" is the largest subset of your vertices where every vertex is connected to every other vertex. In this case, the largest cliques are of size K=2:

- **Option A**: `{0000, 0011}`
    
- **Option B**: `{0000, 1011}`
    

Choosing **Option A**, your final quantum code basis is formed by applying these strings as Z operators to your star graph state ∣G⟩:

- **Logical ∣0⟩L​**: Z0000∣G⟩=∣G⟩
    
- **Logical ∣1⟩L​**: Z0011∣G⟩=Z3​Z4​∣G⟩
    

### **Summary for the AD Channel**

If you were strictly searching for **Amplitude Damping (AD)** codes, you would now repeat this entire process but swap the EXZ​ and EYZ​ orientations as mentioned in your critique. A "good" graph for the AD channel is one where the clique size K remains large across all three orientations.

In this 4-qubit example, you found a code with K=2 (1 logical qubit). If you find a graph that yields K=3 or K=4 for the AD error set, you have discovered a code that outperforms standard 4-qubit stabilizer codes.