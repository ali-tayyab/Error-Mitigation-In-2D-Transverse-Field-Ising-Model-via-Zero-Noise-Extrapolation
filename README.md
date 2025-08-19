### **Noise-Aware ZNE for 2D Transverse-Field Ising Dynamics** (PennyLane)

Simulate real-time dynamics of a 2D transverse-field Ising model on a scaled lattice, add realistic gate noise, and mitigate it using **Zero Noise Extrapolation (ZNE)**—including a **noise-aware** variant with tunable gain factors.



### **1) Problem Overview**

I study the time evolution under a 2D transverse-field Ising Hamiltonian aligned with a superconducting device topology. The full target system uses **127 qubits**, but for classical feasibility I demonstrate on **9 qubits** arranged in a **$3 \times 3$** grid with nearest-neighbor couplings. As an example local observable, I report the final-time expectation value **$\langle Z_{4} \rangle$**.



### **2) Model and Notation**

The Hamiltonian and parameters are defined as:

$$
\mathcal{H} = -K \sum_{\langle i,j \rangle} Z_i Z_j + g \sum_i X_i,
$$

- $K>0$: Ising coupling,  
- $g$: transverse field,  
- $\langle i,j \rangle$: nearest neighbors on the device graph (or $3\times3$ grid in the demo).  

Time evolution over total time $\tau$ is approximated by **first-order Trotterization** with step size $\Delta \tau$:

$$
U(\tau) \approx 
\Bigg(
\prod_{\langle i,j \rangle} e^{\, i\, \Delta \tau\, K\, Z_i Z_j}
\;\prod_i e^{-\, i\, \Delta \tau\, g\, X_i}
\Bigg)^{\tau / \Delta \tau}.
$$

Gate-parameter mapping to hardware-native rotations:

$$
R_{ZZ}(\phi_K) \ \text{with} \ \phi_K = -2K\Delta \tau, 
\qquad
R_{X}(\phi_g) \ \text{with} \ \phi_g = 2g\Delta \tau.
$$

For circuit simplification and CNOT-efficient decompositions, I set

$$
\phi_K = -\frac{\pi}{2} \quad \text{(fixed)}, 
$$

and scan over $\phi_g$.

At the special points $\phi_g=0$ and $\phi_g=\frac{\pi}{2}$, classical simulation becomes trivial; I interpolate between them and track $\langle Z_4 \rangle$.


### **3) Noisy Simulation Setup**

I incorporate **depolarizing noise** after every gate. Physically, with total probability $p$, a single-qubit Pauli error is applied uniformly at random:

$$
\{X, Y, Z\} \quad \text{each with probability } \frac{p}{3}.
$$

In simulation, I model this via the **Kraus operators** of the depolarizing channel, which produces classical mixtures and necessitates a **mixed-state simulator** (not a pure-state backend).



### **4) Zero Noise Extrapolation (ZNE)**

**Goal.** Recover a better estimate of an ideal expectation value using only noisy runs plus classical post-processing.

- Let $F$ denote the ideal (noiseless) expectation and $F'$ the measured (noisy) value.  
- Introduce a **noise gain** $G$ with $G=1$ the native device noise.  
- Evaluate $F'$ at $G \in \{1, G_2, G_3, \dots\}$ (increasing noise),  
- Fit a curve in $G$ and **extrapolate** to $G=0$ to estimate the zero-noise limit.  

A simple linear ansatz often used in practice:

$$
F'(G) \approx a + b\,G \quad \Rightarrow \quad F(0) \approx a.
$$

Higher-order or Richardson-style fits can also be employed when more points are available.



### **5) Noise-Aware ZNE (Gain Scaling)**

When I have a parametric noise description, I can **amplify** that noise in a controlled fashion instead of folding gates blindly.

- In my simulation, the noise channel is fully known, so I rescale its strength with a gain $G$:
  
  $$
  p \ \longrightarrow\  G \cdot p,
  $$
  
  where $G \in \{1.0,\ 1.2,\ 1.6\}$.

- Practically, I instantiate multiple noisy devices (or backends) with scaled noise parameters and run the same circuit to obtain $\{F'(G)\}$ for extrapolation.

This yields a **noise-aware** dataset for ZNE without the sampling explosion associated with probabilistic error cancellation.



### **6) Lattice and Connectivity**

- **Demo**: $3 \times 3$ square lattice (9 qubits), nearest-neighbor edges.  
- **Target hardware concept**: graph connectivity matching a 127-qubit architecture.  
- Each Trotter step applies $R_{ZZ}(\phi_K)$ on lattice edges and $R_X(\phi_g)$ on each site.  

### **7) Outputs**

- Final-time local observable: $\langle Z_4 \rangle$ vs. $\phi_g$.  
- Curves for different gains $G \in \{1.0,1.2,1.6\}$.  
- Extrapolated zero-noise estimate at $G \to 0$.  
### 8) ### **References**
- [Evidence for the utility of quantum computing before fault tolerance](https://www.nature.com/articles/s41586-023-06096-3)
### **9) Quick Start**

#### Environment
- Python 3.10+  
- [PennyLane](https://pennylane.ai/)  
- JAX  
- (Optional) Plotting: matplotlib  

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install pennylane matplotlib jax
