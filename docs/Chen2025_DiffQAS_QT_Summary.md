# Chen et al. (2025) — "Differentiable Quantum Architecture Search in Quantum-Enhanced Neural Network Parameter Generation"

**Citation:** Chen, S. Y.-C., Liu, C.-Y., Chen, K.-C., Huang, W.-J., Chang, Y.-J., & Huang, W.-H. (2025). Differentiable Quantum Architecture Search in Quantum-Enhanced Neural Network Parameter Generation. https://arxiv.org/abs/2505.09653

**Affiliations:** Wells Fargo, National Taiwan University, Imperial College London, Hon Hai Research Institute, Chung Yuan Christian University, Jij Inc.

---

## Core Contribution

The paper combines **Differentiable Quantum Architecture Search (DiffQAS)** with **Quantum-Train (QT)** — a framework where a QNN generates the weights of a classical neural network rather than processing data directly. DiffQAS automates the circuit design within QT, eliminating manual architecture selection. The combined DiffQAS-QT framework achieves massive parameter compression (up to 380x) while matching or exceeding handcrafted baselines across classification, time-series prediction, and reinforcement learning tasks. Critically, **inference requires no quantum hardware** — the QNN is only used during training to generate classical network parameters.

---

## 1. The Quantum-Train (QT) Paradigm

### 1.1 Key Insight: QNN as Parameter Generator

Unlike standard VQC approaches where quantum circuits process data, QT uses the QNN purely to **generate classical neural network weights**:

```
QNN(gamma) -> measurement probabilities -> M_beta -> classical network parameters
```

The classical network then processes data entirely on classical hardware.

### 1.2 Mathematical Formulation

For a classical neural network with **p** parameters:

1. **Qubit count**: n_qt = ceil(log2(p)) qubits
2. **Quantum state**: |psi(gamma)> parameterized by quantum circuit parameters gamma
3. **Measurement**: Yields probabilities |<phi_i|psi(gamma)>|^2 for each of 2^n_qt computational basis states
4. **Mapping function**: M_beta transforms probabilities to classical parameters:

```
M_beta(|phi_i>, |<phi_i|psi(gamma)>|^2) = kappa_i, for i = 1, 2, ..., p
```

where kappa_i is the i-th classical network parameter.

### 1.3 Mapping Function M_beta

- Implemented as a **multi-layer perceptron (MLP)**
- **Inputs**: (1) basis state label |phi_i> in binary representation (length n_qt), (2) corresponding measurement probability
- **Output**: Single classical parameter kappa_i
- Called once per basis state, generating all p classical parameters

### 1.4 Parameter Compression

Training optimizes both QNN parameters gamma and mapping parameters beta. Since n_qt = O(log p), the total trainable parameter count is O(polylog(p)) — exponentially fewer than the p classical parameters being generated.

### 1.5 Inference Without Quantum Hardware

During inference:
1. Run the trained QNN **once** to generate all classical network parameters
2. Deploy the classical network on classical hardware
3. No further quantum computation needed

This means the quantum advantage is in **training efficiency** (parameter compression), not inference speedup.

---

## 2. DiffQAS Framework

### 2.1 Architecture Search Space

Each quantum circuit layer is composed of three design choices:

| Dimension | Options | Count |
|-----------|---------|-------|
| Hadamard initialization | Yes / No | 2 |
| Entanglement pattern | Ent / Cyc | 2 |
| Rotation gate | Rx / Ry / Rz | 3 |

**Total candidates per layer: 2 x 2 x 3 = 12 configurations**

### 2.2 All 12 Candidate Configurations

| Config | Hadamard | Entanglement | Rotation |
|--------|----------|--------------|----------|
| 1 | Yes | Ent | Rx |
| 2 | Yes | Ent | Ry |
| 3 | Yes | Ent | Rz |
| 4 | Yes | Cyc | Rx |
| 5 | Yes | Cyc | Ry |
| 6 | Yes | Cyc | Rz |
| 7 | No | Ent | Rx |
| 8 | No | Ent | Ry |
| 9 | No | Ent | Rz |
| 10 | No | Cyc | Rx |
| 11 | No | Cyc | Ry |
| 12 | No | Cyc | Rz |

**Entanglement patterns:**
- **Ent**: Linear/nearest-neighbor entanglement topology
- **Cyc**: Circular/cyclic entanglement topology

### 2.3 Weighted Ensemble

The output is a weighted sum over all candidates:

```
f_C = sum(j=1 to 12) w_j * f_{C_j}(Theta)
```

where:
- **w_j**: Learnable structural weight for candidate j
- **Theta**: Shared parameter vector across all candidates (shared-parameter strategy)
- **f_{C_j}**: Output of candidate circuit j

**Note:** The paper uses **shared parameters** (single Theta for all candidates), not NonShared as in Chen & Tiwari's DiffQAS-QLSTM. Weight normalization scheme is not explicitly specified.

### 2.4 Gradient-Based Optimization

Gradients with respect to structural weights:

```
nabla_{w_j} L(f_C)
```

are computed via standard automatic differentiation, enabling end-to-end training of architecture weights, circuit parameters, and mapping function parameters simultaneously.

---

## 3. Experimental Setup and Results

### 3.1 Classification

**Datasets:**
- MNIST: digits 1 vs 5
- FashionMNIST: 1 vs 5, 5 vs 7

**Configuration:**
- Optimizer: Adam, lr = 1e-3
- Batch size: 100
- QNN depth: 15 layers

**Parameter Compression:**

| Component | Classical | DiffQAS-QT | Compression |
|-----------|----------|-----------|-------------|
| Network parameters | 108,866 | — | — |
| Trainable params | 108,866 | **286** | **380x** |

**Results:**
- DiffQAS-QT exceeds 98% test accuracy on FashionMNIST 1 vs 5
- Matches or exceeds manually-designed baselines
- Shows improved stability and generalization over individual configs
- Exact accuracy numbers provided only via learning curves (Figures 5-7), not tables

### 3.2 Time-Series Prediction

**Datasets:** Bessel J2, Damped SHM, Delayed Quantum Control, NARMA-5, NARMA-10

**Configuration:**
- Optimizer: RMSProp (lr=0.01, alpha=0.99, eps=1e-8)
- Batch size: 10
- QNN depth: 10 layers
- Input window: 4 consecutive values, predict next value
- Architecture: QT generates LSTM parameters

**Parameter Compression:**

| Component | Classical LSTM | DiffQAS-QT | Compression |
|-----------|---------------|-----------|-------------|
| Trainable params | 1,781 | **135** | **13x** |

**Results (Table II — Test MSE, lower is better):**

| Model | Bessel | Damped SHM | Delayed QC | NARMA-5 | NARMA-10 |
|-------|--------|-----------|-----------|---------|----------|
| **DiffQAS-QT** | **0.000040** | **0.000036** | 0.001710 | 0.000050 | 0.000117 |
| Config 1 (H-Ent-Rx) | 0.000515 | 0.000155 | 0.000715 | 0.000038 | 0.000095 |
| Config 2 (H-Ent-Ry) | 0.000525 | 0.000695 | 0.001190 | 0.000068 | 0.000103 |
| Config 3 (H-Ent-Rz) | 0.000515 | 0.000155 | 0.000676 | 0.000038 | 0.000095 |
| Config 4 (H-Cyc-Rx) | 0.000515 | 0.000155 | 0.002011 | 0.000038 | 0.000095 |
| Config 5 (H-Cyc-Ry) | 0.002902 | 0.000820 | 0.001111 | 0.000024 | 0.000098 |
| Config 6 (H-Cyc-Rz) | 0.000515 | 0.000155 | 0.002638 | 0.000038 | 0.000095 |
| Config 7 (Ent-Rx) | 0.000601 | 0.000040 | 0.001679 | 0.000052 | 0.000120 |
| Config 8 (Ent-Ry) | 0.000141 | 0.008873 | 0.001723 | 0.000091 | 0.000081 |
| Config 9 (Ent-Rz) | 0.001391 | 0.000154 | 0.000956 | 0.000045 | 0.000125 |
| Config 10 (Cyc-Rx) | 0.000216 | 0.000157 | 0.001839 | 0.000040 | 0.000121 |
| Config 11 (Cyc-Ry) | 0.007428 | 0.032072 | **0.000211** | 0.000054 | 0.000140 |
| Config 12 (Cyc-Rz) | 0.001391 | 0.000154 | 0.000956 | 0.000045 | 0.000125 |

**Performance Analysis:**

| Dataset | DiffQAS-QT | Best Baseline | Improvement |
|---------|-----------|---------------|-------------|
| Bessel | **0.000040** | 0.000141 (Config 8) | **3.5x better** |
| Damped SHM | **0.000036** | 0.000040 (Config 7) | **1.1x better** |
| Delayed QC | 0.001710 | **0.000211** (Config 11) | Config 11 wins (8.1x) |
| NARMA-5 | 0.000050 | **0.000024** (Config 5) | Config 5 wins (2.1x) |
| NARMA-10 | 0.000117 | **0.000081** (Config 8) | Config 8 wins (1.4x) |

DiffQAS-QT achieves the best MSE on Bessel and Damped SHM, but is outperformed by specific handcrafted configs on the other three tasks. However, it provides **consistent, stable performance** across all tasks without manual tuning.

### 3.3 Reinforcement Learning

**Environment:** MiniGrid-Empty-5x5 and MiniGrid-Empty-6x6

**Algorithm:** Asynchronous Advantage Actor-Critic (A3C)

**Configuration:**
- Workers: 16 parallel processes
- Discount factor gamma: 0.9
- Update interval: every 5 steps
- QNN depth: 10 layers
- Optimizer: Adam (lr=1e-4, betas=(0.92, 0.999))
- Total training: 50,000 episodes

**Parameter Compression:**

| Component | Classical | DiffQAS-QT | Compression |
|-----------|----------|-----------|-------------|
| Policy network | 6,023 | 157 | 38x |
| Value network | 5,825 | 157 | 37x |
| **Total** | **11,848** | **314** | **38x** |

**Results:**
- DiffQAS-QT achieves **higher average rewards** and **significantly lower variance** in the final 5,000 training episodes on both grid sizes
- Outperforms most of the 12 baseline configurations
- Results shown only as learning curves (Figures 13-14); no numerical values in tables
- Key advantage: training stability — DiffQAS-QT converges more consistently than individual configs

---

## 4. Key Findings

### Finding 1: Automated Architecture Search Matches Expert Design

DiffQAS-QT matches or exceeds the best handcrafted configuration on most tasks (Bessel, Damped SHM, classification, RL) while eliminating the need for expert circuit design. On tasks where a specific config wins (NARMA-5, Delayed QC), the gap is small and DiffQAS-QT still provides robust performance.

### Finding 2: Massive Parameter Compression via QT

The QT framework achieves compression ratios of 13x-380x across different tasks:
- Classification: 108,866 -> 286 params (380x)
- Time-series: 1,781 -> 135 params (13x)
- RL: 11,848 -> 314 params (38x)

### Finding 3: Shared Parameters Are Sufficient

Unlike Chen & Tiwari's DiffQAS-QLSTM (which found NonShared parameters critical), this paper uses **shared parameters** across all candidates and achieves strong results. This suggests the QT mapping function M_beta compensates for parameter sharing by learning to extract specialized outputs from the shared quantum state.

### Finding 4: Training Stability

DiffQAS-QT consistently shows **lower variance** and more stable convergence than individual handcrafted configurations, particularly visible in the RL experiments. The ensemble effect of maintaining weighted candidates provides implicit regularization.

### Finding 5: Cross-Domain Versatility

The framework performs well across three distinct ML paradigms (supervised classification, sequential prediction, interactive RL) without task-specific modifications to the DiffQAS mechanism.

---

## 5. Comparison: QT vs Direct VQC Approaches

| Aspect | Standard VQC (e.g., our QTSTransformer) | Quantum-Train (QT) |
|--------|----------------------------------------|--------------------|
| **Data processing** | Quantum circuit processes data | Classical network processes data |
| **QNN role** | Feature extraction / transformation | Parameter generation |
| **Inference** | Requires quantum circuit execution | Fully classical |
| **Parameter count** | QNN params = model params | QNN params << classical model params |
| **Hardware dependency** | Needed for both training and inference | Only needed during training |
| **Expressivity source** | Circuit ansatz + measurement | Classical network architecture |
| **Inductive bias** | Quantum (entanglement, superposition) | Classical (standard NN) |

---

## 6. Comparison: Two DiffQAS Approaches

| Aspect | DiffQAS-QLSTM (Chen & Tiwari 2025) | DiffQAS-QT (this paper) |
|--------|-------------------------------------|------------------------|
| **QNN role** | Direct data processing (QLSTM gates) | Parameter generation for classical NN |
| **Search space** | 36 candidates (6 encoding x 6 variational) | 12 candidates (2 init x 2 entangle x 3 rotation) |
| **Parameter mode** | NonShared best (independent params per candidate) | Shared (single Theta across all candidates) |
| **Inference** | Requires quantum execution | Fully classical |
| **Tasks tested** | Time-series prediction (5 benchmarks) | Classification, time-series, RL |
| **RL tested** | No | Yes (A3C on MiniGrid) |
| **Compression** | None (standard VQC parameter count) | 13x-380x via QT |
| **Best advantage** | 31.6x MSE improvement on Damped SHM | 3.5x MSE improvement on Bessel |
| **Key finding** | NonShared params critical | Shared params sufficient (QT compensates) |

---

## 7. Relevance to Our QRL QTSTransformer

### 7.1 DiffQAS for Circuit Design

The DiffQAS approach from both this paper and Chen & Tiwari (2025) can be applied to our QTSTransformer's Sim14 ansatz. Instead of a fixed RY-CRX-RY-CRX sequence, maintain weighted candidates:

**Candidate pool for our 8-qubit system:**
- {Hadamard init, No init} x {linear, cyclic entanglement} x {RY, RX, RZ rotations} = 12 candidates
- Or extended: add CRX/CRZ/CNOT entanglement options for larger search space

**Integration points:**
1. **QSVT polynomial circuit**: Search over encoding gate choices
2. **Sim14 ansatz**: Search over rotation + entanglement patterns
3. **Both simultaneously**: Larger search space but more comprehensive

### 7.2 QT Paradigm — Not Directly Applicable

The Quantum-Train approach (QNN generates classical NN params) is fundamentally different from our architecture, where the quantum circuit directly processes temporal features. QT would eliminate the quantum inductive bias that gives our model its advantage (Breakout +492%, SpaceInvaders +18%). Our quantum advantage stems from the circuit's direct role in temporal processing, not parameter compression.

**However**, QT could be explored for:
- Compressing the classical CNN backbone (1.95M params)
- Generating the output head parameters
- As a separate line of investigation (classical inference with quantum-trained weights)

### 7.3 Key Design Decisions from Both DiffQAS Papers

For integrating DiffQAS into our QTSTransformer:

| Decision | DiffQAS-QLSTM Recommendation | DiffQAS-QT Recommendation | Our Choice |
|----------|------------------------------|--------------------------|------------|
| Parameter sharing | NonShared (independent per candidate) | Shared (single Theta) | Start with Shared (cheaper), test NonShared |
| Search space size | 36 candidates | 12 candidates | 12 (manageable overhead) |
| Weight normalization | Unspecified | Unspecified | Use softmax (principled) |
| Training | End-to-end | End-to-end | End-to-end |
| Overhead | 36x cost per forward pass | 12x cost per forward pass | 12x is feasible |

### 7.4 Computational Feasibility

For our 8-qubit QTSTransformer with 12 DiffQAS candidates:
- **Current cost**: 1 QNode evaluation per forward pass
- **DiffQAS cost**: 12 QNode evaluations per forward pass (12x overhead)
- **Mitigation**: Two-phase training — search phase (expensive, ~1000 episodes) then fix best architecture for full training
- **v4 lesson**: Avoid creating 12 independent QNode objects. Instead, use a single parameterized circuit with configurable gate choices.

---

## 8. Limitations

1. **RL scope**: Only MiniGrid-Empty (trivial navigation); no complex game environments (Atari, etc.)
2. **Missing numerical RL results**: No tables with exact reward values; only learning curves
3. **QT vs direct VQC**: QT eliminates quantum inductive bias during inference — unclear if this sacrifices the advantages seen in direct VQC approaches
4. **Shared parameters only**: Does not compare Shared vs NonShared modes (unlike Chen & Tiwari)
5. **No hardware validation**: Simulation only
6. **Entanglement patterns underspecified**: "Ent" and "Cyc" topologies not precisely defined at the gate level
7. **Weight normalization unspecified**: Does not state whether softmax, Gumbel-Softmax, or unnormalized weights are used
8. **No computational cost analysis**: Does not discuss the 12x overhead of maintaining all candidates

---

## 9. Connection to Our Paper Series

| Paper | Focus | Key Contribution to Our Framework |
|-------|-------|----------------------------------|
| **Chen et al. (2025)** — Learnable Observables | Observable optimization | Learnable Hermitian measurement matrices |
| **Lin et al. (2025)** — ANO on QNNs | Scalable observables | k-local truncation, sliding/pairwise schemes |
| **Lin et al. (2025)** — QRL by ANO | RL application | ANO validated in DQN/A3C on RL benchmarks |
| **Chen & Tiwari (2025)** — DiffQAS-QLSTM | Architecture search (direct VQC) | Weighted ensemble over circuit designs |
| **Chen et al. (2025)** — DiffQAS-QT (this paper) | Architecture search (QT paradigm) | Automated circuit design + parameter compression |

Together, these five papers provide:
- **What to measure**: Learnable Hermitian observables (Chen 2025)
- **How to scale measurement**: k-local ANO (Lin 2025a)
- **Where to apply it**: Reinforcement learning (Lin 2025b)
- **How to design the circuit (direct)**: DiffQAS for VQC ansatze (Chen & Tiwari 2025)
- **How to design the circuit (QT)**: DiffQAS for parameter-generating circuits (Chen et al. 2025)

For our QTSTransformer, the most relevant approaches are:
1. **ANO** (replace fixed PauliX/Y/Z with learnable k-local observables)
2. **DiffQAS-direct** (search over Sim14 alternatives automatically)
3. DiffQAS-QT is less relevant since we want to preserve the quantum inductive bias in temporal processing.
