# Sun et al. (2023) — "Differentiable Quantum Architecture Search for Quantum Reinforcement Learning"

**Citation:** Sun, Y., Ma, Y., & Tresp, V. (2023). Differentiable Quantum Architecture Search for Quantum Reinforcement Learning. *IEEE International Conference on Quantum Computing and Engineering (QCE23) Workshop on Quantum Machine Learning*. https://arxiv.org/abs/2309.10392

**Affiliations:** Ludwig Maximilian University of Munich / Siemens AG

**Venue:** IEEE Quantum Week 2023 (QCE23), Workshop paper (4+1 pages)

---

## Core Contribution

This paper is the **first application of gradient-based Differentiable Quantum Architecture Search (DQAS) to Quantum Reinforcement Learning**. Unlike ensemble-based DiffQAS approaches (Chen & Tiwari 2025, Chen et al. 2025) that maintain weighted mixtures of full circuits, Sun et al. perform **gate-level search** — each position (placeholder) in the circuit independently selects from an operation pool. Applied to quantum DQN on CartPole and FrozenLake, automatically-discovered circuits achieve ~200% faster convergence than manually-designed baselines and reveal structurally novel gate arrangements (e.g., CNOT gates in the middle of blocks rather than at the end).

---

## 1. DQAS Framework

### 1.1 Super-Circuit Structure

The super-circuit consists of three blocks:
1. **Encoding block**: Encodes classical state into quantum state (with input weights)
2. **Parameterized block**: Contains p placeholders, each replaceable by an operation from the pool (can be stacked multiple times)
3. **Measurement block**: Extracts Q-values from quantum state

Each placeholder u_i covers **all qubits** (unlike the original DQAS which used single-qubit placeholders) and accepts one operation from the pool.

A parameterized block is:
```
U = prod(i=0 to p) u_i(theta_i)
```

### 1.2 Architecture Parameterization

Each placeholder i has architecture parameters alpha_i defining a probability distribution over the operation pool O:

```
P(k, alpha) = prod(i=1 to p) [exp(alpha_{ij}) / sum_k exp(alpha_{ik})]
```

This is a **softmax** distribution over candidate operations — a key distinction from the ensemble-weighted approaches in other DiffQAS papers.

### 1.3 Optimization Objective

The global loss aggregates over sampled architecture candidates:

```
L = sum_{U ~ P(U, alpha)} [P(U, alpha) / sum_{U' ~ P(U, alpha)} P(U', alpha)] * L_k(theta)
```

where L_k is the DQN loss for sampled architecture k.

Joint optimization of:
- **theta** (circuit parameters): via parameter shift rules
- **alpha** (architecture parameters): via gradient-based optimization through the sampling distribution

### 1.4 Progressive Search (Pruning)

Every n iterations, the progressive search mechanism:
1. Examines the architecture distribution at each placeholder
2. Identifies and removes operation candidates with consistently low probability
3. Effectively narrows the search space during training

This prevents the search from maintaining computational overhead on clearly inferior options.

---

## 2. Operation Pools

### 2.1 Pool Definitions (4-qubit system)

**op3 (larger pool):**
- Single-qubit gates on all 4 qubits: RY, RZ, Identity
- Two-qubit gates on all 4 qubits: CZ, CNOT
- RY/RZ on qubit subsets: {[1,2,3], [2,3,4], [1,2], [2,3], [3,4]}

**op4 (smaller pool):**
- Same as op3 but **excludes** CZ and RY/RZ on two-qubit subsets {[1,2], [2,3], [3,4]}

### 2.2 Key Design Choice

Each placeholder operates on **all qubits** simultaneously, with the operation pool defining which qubits are actually affected. This reduces the combinatorial search space compared to per-qubit placeholder assignment.

---

## 3. Quantum DQN Architecture

### 3.1 Q-Value Computation

Q-values are computed as:

```
Q(s, a) = 1/2 * (<0^n| U_theta^dag(s) O_a U_theta(s) |0^n> + 1)
```

The measurement result is shifted by +1 and scaled by 1/2 to normalize output to [0, 1].

### 3.2 DQN Loss

Standard DQN loss with target network:

```
L_i(theta_i) = E_{(s,a,r,s') ~ ER} [(r + gamma * max_{a'} Q^{target}(s', a'; theta_i^-) - Q(s, a; theta_i))^2]
```

where ER is the experience replay buffer and theta^- denotes target network parameters.

### 3.3 Data Re-uploading

The paper implements input and output weights from prior quantum DQN work, enabling data re-uploading within the circuit. Specific details are not elaborated.

### 3.4 Circuit Configuration

- **Qubits**: 4
- **Parameterized blocks**: 5
- **Placeholders per block**: 4
- **Total placeholders**: 20

---

## 4. Training Algorithm

### 4.1 Four-Phase Procedure

**Phase 1 — Initialization:**
- Initialize operation pool, super-circuit (alpha, theta)
- Create two QDQNs (online + target) and environment

**Phase 2 — Joint Search & Training:**
- Sample minibatch of architecture candidates from P(U, alpha)
- Compute global loss via weighted aggregation
- Update alpha (architecture params) and theta (circuit params) via gradients
- Periodically copy parameters to target network
- Apply progressive pruning every n iterations

**Phase 3 — Architecture Selection:**
- Rank agents by training performance
- Select top K architectures

**Phase 4 — Evaluation:**
- Evaluate best architecture or retrain from scratch

### 4.2 Hyperparameters

- **Optimizer**: Adam
- **Qubits**: 4
- **Blocks**: 5 parameterized blocks x 4 placeholders
- **Solving criterion**: Average reward reaches maximum (r_avg >= r_max)
- Specific learning rate, batch size, replay buffer size, and episode count are not reported

---

## 5. Experimental Results

### 5.1 Environments

| Environment | State Dim | Action Dim | Max Reward |
|-------------|-----------|------------|------------|
| CartPole-v0 | 4 | 2 | 200 |
| FrozenLake-v0 | 16 (discrete) | 4 | 1.0 |

### 5.2 Baseline

Manually designed circuit: 5 parameterized blocks, each with 3 operation columns covering all qubits, following the sequence **RY → RZ → CZ** per block.

### 5.3 Performance

| Metric | Baseline (manual) | DQAS (automatic) | Improvement |
|--------|-------------------|-------------------|-------------|
| CartPole solving point | ~800 episodes | ~400 episodes | **~2x faster** |
| FrozenLake solving point | ~1000 episodes | ~400 episodes | **~2.5x faster** |

"Solving point" = episode number where the agent first achieves maximum average reward.

### 5.4 Discovered Architectures

**Auto-fl-op3** (FrozenLake, pool op3):
- Contains: RY, CNOT, RZ operations
- **CNOT gates placed in the MIDDLE** of parameterized blocks

**Auto-cp-op4** (CartPole, pool op4):
- Contains: RZ, CNOT, RZ, RY sequence
- Again, CNOT in non-terminal positions

**Key structural finding**: The automatically-discovered circuits place entangling gates (CNOT) in the **middle** of parameterized blocks, contrasting with the manual baseline that places CZ at the **end** of each block. This suggests the conventional expert intuition about entanglement placement may be suboptimal.

### 5.5 Noise Resilience

Tested on IBM noisy simulator (qiskit-aer, "ibmq-quito" noise model):
- Discovered architectures maintain performance under realistic device noise
- Both op3 and op4 architectures show positive results on FrozenLake

### 5.6 Super-Circuit Quality Matters

Critical finding: "Performance of the automatically created circuit depends on whether the super-circuit learned well during the training process."

- Well-trained super-circuits → good discovered architectures
- Poorly-learning super-circuits → poor architectures despite optimization
- This implies the search quality is bounded by the training dynamics

---

## 6. Key Findings

### Finding 1: Gate-Level DQAS Works for QRL

This is the first demonstration that gradient-based architecture search at the individual gate level is effective for quantum RL tasks, despite the non-stationarity of RL datasets (training data distribution shifts as the policy improves).

### Finding 2: Automatic Designs Beat Expert Designs

Automatically-discovered circuits converge ~2-2.5x faster than manually-designed baselines. The improvement is consistent across both environments.

### Finding 3: Novel Gate Arrangements

The search discovers structurally different circuits from expert designs — particularly the placement of entangling gates in the middle rather than at the end of blocks. This challenges conventional circuit design wisdom.

### Finding 4: Noise Robustness

Discovered architectures generalize to noisy quantum hardware simulations, suggesting the search does not overfit to ideal simulation conditions.

### Finding 5: Super-Circuit Training Quality Is Critical

The quality of the discovered architecture is directly correlated with how well the super-circuit learns during the search phase. This implies:
- Sufficient search budget (episodes) is essential
- The search itself is a learning problem, not just combinatorial optimization

---

## 7. Comparison with Other DiffQAS Approaches

| Aspect | Sun et al. 2023 (this paper) | Chen & Tiwari 2025 (DiffQAS-QLSTM) | Chen et al. 2025 (DiffQAS-QT) |
|--------|------------------------------|-------------------------------------|-------------------------------|
| **Search granularity** | Gate-level (per placeholder) | Full-circuit ensemble | Full-circuit ensemble |
| **Search space** | Operation pool per position | 36 full circuits | 12 full circuits |
| **Output** | Single best circuit | Weighted ensemble | Weighted ensemble |
| **Sampling** | Softmax + sampling | Weighted sum (all evaluated) | Weighted sum (all evaluated) |
| **RL tested** | Yes (DQN: CartPole, FrozenLake) | No (time-series only) | Yes (A3C: MiniGrid) |
| **Qubits** | 4 | Not specified | Not specified |
| **Progressive pruning** | Yes | No | No |
| **Inference** | Single circuit (no ensemble) | Requires all circuits (or select best) | Classical (QT paradigm) |
| **Parameter sharing** | Shared theta across candidates | NonShared best | Shared |
| **Noise testing** | Yes (IBM simulator) | No | No |

### Key Architectural Distinction

Sun et al.'s approach produces a **single discrete circuit** at the end of search (by selecting the highest-probability operation at each placeholder), while the ensemble approaches (Chen & Tiwari, Chen et al.) maintain a **continuous weighted mixture** of circuits. The single-circuit output is more practical for deployment but potentially less expressive during training.

---

## 8. Relevance to Our QRL QTSTransformer

### 8.1 Direct Applicability

This paper is the **most directly relevant** DiffQAS work for our project because:
1. It targets **quantum DQN** — we use Double DQN
2. It addresses the **non-stationary dataset problem** inherent in RL
3. It demonstrates gate-level search — applicable to our Sim14 ansatz

### 8.2 Integration Strategy for Our 8-Qubit System

**Option A — Gate-level search (Sun et al. style):**

Replace each gate position in our Sim14 ansatz with a placeholder from an operation pool:

Current Sim14 (fixed):
```
[RY on all qubits] → [CRX pairs] → [RY on all qubits] → [CRX reverse pairs]
```

Searchable version:
```
[Placeholder 1: {RY, RX, RZ}] → [Placeholder 2: {CRX, CRZ, CNOT, CZ}] →
[Placeholder 3: {RY, RX, RZ}] → [Placeholder 4: {CRX_rev, CRZ_rev, CNOT_rev, CZ_rev}]
```

With 2 ansatz layers x 4 positions x ~4 options each = modest search space.

**Option B — Hybrid approach:**

Combine gate-level search (Sun et al.) with full-circuit ensemble (Chen & Tiwari) at different levels:
- Gate-level within each block (fine-grained)
- Block-level ensemble for overall circuit structure (coarse-grained)

### 8.3 Key Lessons for Our Implementation

1. **Progressive pruning**: Start with a full operation pool, prune low-probability options during search to reduce computational cost
2. **Super-circuit quality**: Ensure sufficient episodes for the search phase; poor search training → poor architecture
3. **Entanglement placement**: Don't assume conventional patterns (end-of-block) are optimal — let the search discover placement
4. **Single circuit output**: After search, deploy a single best circuit (no ensemble overhead at inference)
5. **Noise awareness**: Discovered architectures should be validated under noise

### 8.4 Limitations for Our Context

- Only tested on 4-qubit systems (we use 8 qubits)
- Only simple environments (CartPole/FrozenLake, not Atari)
- Workshop paper with limited experimental detail
- No comparison with non-gradient-based QAS methods (evolutionary, Bayesian, etc.)

---

## 9. Connection to Full Paper Series

| Paper | Year | Search Type | RL? | Key Innovation |
|-------|------|-------------|-----|----------------|
| **Sun et al.** (this paper) | 2023 | Gate-level DQAS | **DQN** | First DQAS for QRL, progressive pruning |
| **Chen et al.** — Learnable Observables | 2025 | N/A (measurement) | No | Learnable Hermitian observables |
| **Lin et al.** — ANO on QNNs | 2025 | N/A (measurement) | No | k-local truncation for scalability |
| **Lin et al.** — QRL by ANO | 2025 | N/A (measurement) | **DQN + A3C** | ANO validated in RL |
| **Chen & Tiwari** — DiffQAS-QLSTM | 2025 | Full-circuit ensemble | No | Weighted circuit ensemble for QLSTM |
| **Chen et al.** — DiffQAS-QT | 2025 | Full-circuit ensemble | **A3C** | DiffQAS + Quantum-Train, parameter compression |

**For our QTSTransformer, the combined approach would be:**
1. **Circuit design**: Gate-level DQAS (Sun et al.) or full-circuit ensemble (Chen & Tiwari) to replace fixed Sim14
2. **Measurement design**: ANO (Lin et al.) to replace fixed PauliX/Y/Z
3. **Both jointly**: End-to-end differentiable search over circuit architecture AND measurement observables

---

## 10. Limitations

1. **Short workshop paper**: Limited experimental detail (4+1 pages)
2. **Small scale**: 4 qubits, 2 simple environments (CartPole, FrozenLake)
3. **No Atari-scale testing**: Gap between FrozenLake and Atari complexity is enormous
4. **Hyperparameters not reported**: Learning rate, batch size, replay buffer size, episode count missing
5. **No parameter count analysis**: Does not discuss trainable parameter overhead
6. **No wall-clock timing**: Does not quantify computational cost of DQAS vs fixed circuit
7. **No comparison with other QAS**: Only compared against one manually-designed baseline
8. **Solving point metric**: Only reports convergence speed, not final reward quality or stability
