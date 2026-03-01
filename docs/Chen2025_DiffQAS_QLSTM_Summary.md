# Chen & Tiwari (2025) — "Quantum Long Short-Term Memory with Differentiable Architecture Search"

**Citation:** Chen, S. Y.-C., & Tiwari, P. (2025). Quantum Long Short-term Memory with Differentiable Architecture Search. *IEEE International Conference on Quantum Artificial Intelligence (QAI) 2025*. https://arxiv.org/abs/2508.14955

**Venue:** IEEE QAI 2025

---

## Core Contribution

The paper introduces **DiffQAS-QLSTM**, a framework that integrates Differentiable Quantum Architecture Search (DiffQAS) into Quantum LSTM models. Instead of manually selecting a fixed quantum circuit ansatz, DiffQAS maintains a **weighted superposition of candidate circuits** and jointly optimizes both circuit parameters and architectural selection weights via gradient-based training. The framework outperforms all handcrafted QLSTM baselines on five time-series prediction benchmarks.

---

## 1. Motivation: The Circuit Design Problem

### 1.1 The Problem

Designing variational quantum circuits requires choosing:
- Encoding gates (how classical data enters the circuit)
- Variational gates (trainable rotation gates)
- Entanglement patterns (CNOT connectivity)
- Circuit depth (number of layers)

These choices are typically made **manually** by domain experts, with no guarantee of optimality. As demonstrated by the paper's results, a poor choice (e.g., Config 3: Rz encoding + Rz variational) can yield **100x worse performance** than a good choice (Config 1: Ry encoding + Ry variational) on the same task.

### 1.2 The Solution

DiffQAS treats circuit architecture selection as a **continuous optimization problem**, enabling gradient-based search over a discrete design space. This eliminates manual circuit design while automatically finding task-optimal architectures.

---

## 2. DiffQAS Framework

### 2.1 Architecture Search Space

A quantum circuit C is composed of modular units S₁, S₂, ..., Sₙ. Each module Sᵢ is selected from a predefined set of candidate subcircuits Bᵢ. The overall search space comprises:

```
N = |B₁| × |B₂| × ... × |Bₙ|
```

distinct circuit configurations.

### 2.2 Weighted Ensemble

Rather than selecting a single circuit, DiffQAS computes a **weighted sum** over all candidate circuits:

```
f_C = Σ(j=1 to N) w_j · f_{C_j}(x; θ_j)
```

where:
- **f_{C_j}**: Output of candidate circuit j
- **w_j**: Learnable structural weight for candidate j
- **θ_j**: Trainable circuit parameters for candidate j (in NonShared mode)

### 2.3 Gradient-Based Optimization

Gradients with respect to structural weights:

```
∇_{w_j} L(f_C)
```

can be computed via standard automatic differentiation, enabling end-to-end training of both architecture weights and circuit parameters.

**Note:** The paper does not explicitly specify the normalization scheme for weights (softmax, Gumbel-Softmax, or unnormalized), nor whether architecture and circuit parameters use separate optimizers or learning rates.

### 2.4 Candidate Circuit Pool

**Encoding circuit options (6 choices):**
- Rotation gates: Ry, Rz, Rx (3 choices)
- Initialization: with or without Hadamard gate (2 choices)
- Total: 2 × 3 = 6 configurations

**Variational circuit options (6 choices):**
- Entanglement patterns: 2 types
- Parameterized rotation gates: Ry, Rz, Rx (3 choices)
- Total: 2 × 3 = 6 configurations

**Overall search space:** 6 × 6 = **36 unique circuit realizations**

---

## 3. QLSTM Architecture

### 3.1 Classical LSTM Recap

A standard LSTM cell processes sequential data through four gating mechanisms:

```
f_t = σ(W_f · [h_{t-1}, x_t] + b_f)    [forget gate]
i_t = σ(W_i · [h_{t-1}, x_t] + b_i)    [input gate]
C̃_t = tanh(W_c · [h_{t-1}, x_t] + b_c) [cell candidate]
o_t = σ(W_o · [h_{t-1}, x_t] + b_o)    [output gate]

c_t = f_t ⊙ c_{t-1} + i_t ⊙ C̃_t       [cell state update]
h_t = o_t ⊙ tanh(c_t)                   [hidden state output]
```

### 3.2 Quantum LSTM (QLSTM)

The QLSTM replaces the classical weight matrices with Quantum Neural Networks (QNNs):

```
f_t = σ(QNN₁(v_t))     [forget gate]
i_t = σ(QNN₂(v_t))     [input gate]
C̃_t = tanh(QNN₃(v_t))  [cell candidate]
o_t = σ(QNN₄(v_t))     [output gate]
```

where v_t = [h_{t-1}, x_t] is the concatenation of previous hidden state and current input.

Each QNN follows the standard VQC pattern:
1. Encode v_t into quantum state via encoding gates
2. Apply variational circuit with trainable rotations
3. Measure observables for output

### 3.3 DiffQAS-QLSTM

Each of the 4 QLSTM gates uses DiffQAS instead of a single fixed QNN:

```
f_t = σ(DiffQAS-QNN₁(v_t))     [forget gate — searches over 36 circuits]
i_t = σ(DiffQAS-QNN₂(v_t))     [input gate — searches over 36 circuits]
C̃_t = tanh(DiffQAS-QNN₃(v_t))  [cell candidate — searches over 36 circuits]
o_t = σ(DiffQAS-QNN₄(v_t))     [output gate — searches over 36 circuits]
```

---

## 4. Model Variants

### 4.1 Parameter Modes

| Variant | Circuit Params | Architecture Weights | Description |
|---------|---------------|---------------------|-------------|
| **NonShared** | Each candidate has own θ_j | Trainable w_j | Most expressive — independent parameters per candidate |
| **Shared** | Single shared θ for all | Trainable w_j | Reduced parameter count — all candidates share one parameter set |
| **Reservoir-NonShared** | Fixed random θ_j | Trainable w_j | Tests architecture structure alone |
| **Reservoir-Shared** | Fixed random shared θ | Trainable w_j | Most constrained |

### 4.2 Handcrafted Baselines (Table I)

Six fixed-architecture configurations with no architecture search:

| Config | Encoding Gate | Trainable Rotation Gate |
|--------|--------------|------------------------|
| Config 1 | Ry | Ry |
| Config 2 | Rz | Ry |
| Config 3 | Rz | Rz |
| Config 4 | Ry | Rz |
| Config 5 | Rx | Rz |
| Config 6 | Rx | Ry |

---

## 5. Experimental Setup

### 5.1 Datasets

Five time-series prediction benchmarks, all using the protocol: predict observation N+1 given preceding N observations (N=4).

| Dataset | Type | Complexity |
|---------|------|-----------|
| **Bessel** | Bessel function J₂ | Oscillatory, smooth |
| **Damped SHM** | Damped simple harmonic motion | Exponential decay + oscillation |
| **Delayed Quantum Control** | Quantum control signal | Non-trivial temporal dynamics |
| **NARMA-5** | Nonlinear autoregressive moving average, order 5 | Nonlinear, moderate memory |
| **NARMA-10** | Nonlinear autoregressive moving average, order 10 | Nonlinear, long memory |

### 5.2 Missing Implementation Details

The paper does not specify:
- Number of qubits per QNN module
- Circuit depth / number of variational layers
- Training epochs
- Learning rate or optimizer type
- Batch size
- Data split methodology
- Computational overhead of DiffQAS vs single-circuit evaluation

---

## 6. Results

### 6.1 Main Results (Table II — Test MSE, lower is better)

| Model | Bessel | Damped SHM | Delayed Control | NARMA-5 | NARMA-10 |
|-------|--------|-----------|-----------------|---------|----------|
| **DiffQAS-NonShared** | **0.000229** | **0.000019** | 0.001859 | 0.000030 | 0.000094 |
| DiffQAS-Shared | 0.000117 | 0.000036 | 0.002486 | 0.000472 | 0.000385 |
| DiffQAS-Reservoir-NonShared | 0.006529 | 0.007489 | 0.003567 | **0.000016** | **0.000074** |
| DiffQAS-Reservoir-Shared | 0.006546 | 0.010023 | 0.004227 | 0.000494 | 0.000412 |
| Config 1 (Ry/Ry) | 0.001324 | 0.000601 | 0.005507 | 0.000032 | 0.000123 |
| Config 2 (Rz/Ry) | 0.002768 | 0.003803 | 0.004591 | 0.000273 | 0.000188 |
| Config 3 (Rz/Rz) | 0.023947 | 0.046938 | 0.084120 | 0.000273 | 0.000188 |
| Config 4 (Ry/Rz) | 0.007316 | 0.010588 | **0.001931** | **0.000025** | 0.000101 |
| Config 5 (Rx/Rz) | 0.023947 | 0.046938 | 0.084120 | 0.000273 | 0.000188 |
| Config 6 (Rx/Ry) | 0.024024 | 0.046886 | 0.077084 | 0.000273 | 0.000188 |

### 6.2 Performance Analysis

**DiffQAS-NonShared vs Best Handcrafted Baseline:**

| Dataset | DiffQAS-NonShared | Best Baseline | Improvement |
|---------|------------------|---------------|-------------|
| Bessel | 0.000229 | 0.001324 (Config 1) | **5.8x better** |
| Damped SHM | 0.000019 | 0.000601 (Config 1) | **31.6x better** |
| Delayed Control | 0.001859 | 0.001931 (Config 4) | **1.04x better** |
| NARMA-5 | 0.000030 | 0.000025 (Config 4) | 0.83x (Config 4 wins) |
| NARMA-10 | 0.000094 | 0.000101 (Config 4) | **1.07x better** |

DiffQAS-NonShared achieves the best or near-best performance on every task without any manual tuning, while the best handcrafted config varies by task (Config 1 for Bessel/SHM, Config 4 for NARMA/Delayed Control).

**DiffQAS-NonShared vs Worst Handcrafted Baseline:**

| Dataset | DiffQAS-NonShared | Worst Baseline | Ratio |
|---------|------------------|----------------|-------|
| Bessel | 0.000229 | 0.024024 (Config 6) | **105x better** |
| Damped SHM | 0.000019 | 0.046938 (Config 3) | **2,470x better** |
| Delayed Control | 0.001859 | 0.084120 (Config 3) | **45x better** |

---

## 7. Key Findings

### Finding 1: Automated Search Eliminates Manual Design Risk

The performance gap between the best and worst handcrafted configs is **enormous** (up to 2,470x on Damped SHM). DiffQAS eliminates the risk of choosing a catastrophically bad architecture by searching over all options simultaneously.

### Finding 2: NonShared Parameters Are Critical

DiffQAS-NonShared consistently outperforms DiffQAS-Shared, indicating that allowing each candidate circuit to maintain its own trainable parameters is important. The architecture weights then learn to emphasize the best-performing candidates with their specialized parameters.

### Finding 3: Trainable Parameters Are Essential

Reservoir variants (fixed random parameters, only architecture weights trained) show significantly degraded performance compared to fully trainable variants. This demonstrates that:
- Architecture structure alone is insufficient
- Parameter optimization and architecture search are complementary
- The "right architecture" without good parameters still underperforms

### Finding 4: No Single Best Architecture Exists

The best handcrafted config varies by task:
- **Bessel, Damped SHM**: Config 1 (Ry/Ry) is best
- **NARMA-5, Delayed Control**: Config 4 (Ry/Rz) is best
- **NARMA-10**: Config 4 (Ry/Rz) is best

This task-dependency is precisely what DiffQAS addresses — it finds the task-optimal architecture automatically.

---

## 8. Relevance to Our QRL QTSTransformer

### 8.1 Current Architecture (v3)

Our v3 uses a **fixed Sim14 ansatz** with a manually chosen gate sequence:
```
RY → CRX → RY → CRX (reverse direction)
```
repeated for n_ansatz_layers=2. This is analogous to picking a single "Config" from the paper's baseline table.

### 8.2 DiffQAS Integration Opportunity

DiffQAS could be applied to our QTSTransformer at two levels:

**Level 1 — Ansatz Search:**
Replace the fixed Sim14 circuit with a weighted ensemble of candidate ansatze:
- Sim14 (RY-CRX-RY-CRX)
- Sim15 (RZ-CRX-RZ-CRX)
- Hardware-efficient (RY-CNOT-RY-CNOT)
- IsingXX/YY/ZZ variants
- Others

**Level 2 — Gate-Level Search:**
Within each layer position, search over individual gate choices:
- Encoding: {Ry, Rz, Rx} × {with/without Hadamard}
- Entanglement: {linear, circular, all-to-all} × {CRX, CRZ, CNOT}
- Trainable rotations: {Ry, Rz, Rx}

### 8.3 Combined with ANO

DiffQAS + ANO would enable searching over both:
1. **Circuit architecture** (which gates, which entanglement)
2. **Measurement strategy** (which observables, which locality k)

This is a fully differentiable end-to-end quantum model design framework.

### 8.4 Computational Considerations

**Training overhead**: DiffQAS requires evaluating all N candidate circuits per forward pass during the search phase. For N=36 candidates, this is 36x the cost of a single circuit evaluation.

**Mitigation strategies**:
1. **Two-phase training**: Search phase (expensive, short) → Fix best architecture → Production training (normal cost)
2. **Reduced search space**: Limit candidates to a smaller set (e.g., 6 instead of 36)
3. **Progressive pruning**: Start with all candidates, gradually prune low-weight ones

---

## 9. Connection to Related Papers

| Paper | Focus | Contribution to Our Framework |
|-------|-------|------------------------------|
| **Chen et al. (2025)** — Learnable Observables | Observable optimization | Learnable Hermitian measurement matrices |
| **Lin et al. (2025)** — ANO on QNNs | Scalable observables | k-local truncation, sliding/pairwise schemes |
| **Lin et al. (2025)** — QRL by ANO | RL application | ANO validated in DQN/A3C on RL benchmarks |
| **Chen & Tiwari (2025)** — DiffQAS-QLSTM (this paper) | Architecture search | Differentiable search over circuit design space |

Together, these four papers provide the complete toolkit:
- **What to measure**: Learnable Hermitian observables (Chen 2025)
- **How to scale measurement**: k-local ANO (Lin 2025a)
- **Where to apply it**: Reinforcement learning (Lin 2025b)
- **How to design the circuit**: Differentiable architecture search (Chen & Tiwari 2025)

---

## 10. Limitations

1. **Missing implementation details**: No qubit counts, learning rates, optimizers, epochs, or batch sizes specified
2. **Small-scale tasks only**: Time-series prediction with N=4 lookback; no complex sequential decision-making tasks
3. **No computational cost analysis**: No discussion of DiffQAS overhead vs single-circuit training
4. **No hardware validation**: Simulation only
5. **No comparison with other NAS methods**: No comparison against evolutionary, RL-based, or Bayesian architecture search
6. **Weight normalization unspecified**: Unclear whether softmax, Gumbel-Softmax, or unnormalized weights are used
