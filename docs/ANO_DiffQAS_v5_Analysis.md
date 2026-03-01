# Adaptive Non-Local Observables and Differentiable Quantum Architecture Search for Quantum Reinforcement Learning with QTSTransformer

**Version**: v5
**Authors**: Junghoon Park
**Date**: February 2026
**Script**: `scripts/QuantumTransformerAtari_v5_ano_dqas.py`
**Job Script**: `jobs/run_quantum_transformer_atari_v5.sh`

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Background: QTSTransformer v3 Baseline](#2-background-qtstransformer-v3-baseline)
3. [Adaptive Non-Local Observables (ANO)](#3-adaptive-non-local-observables-ano)
4. [Differentiable Quantum Architecture Search (DiffQAS)](#4-differentiable-quantum-architecture-search-diffqas)
5. [Two-Phase Training Procedure](#5-two-phase-training-procedure)
6. [Ablation Study Design](#6-ablation-study-design)
7. [Architecture and Parameter Counts](#7-architecture-and-parameter-counts)
8. [Implementation Details](#8-implementation-details)
9. [Known Limitations](#9-known-limitations)
10. [References](#10-references)

---

## 1. Introduction

The Quantum Time-Series Transformer (QTSTransformer) v3 demonstrated strong results on Atari RL benchmarks, outperforming classical DQN baselines on SpaceInvaders (+18%), Breakout (+492%), and MarioBros (+34%). However, v3 uses two design choices that are fixed by construction and potentially suboptimal:

1. **Fixed measurement observables**: PauliX, PauliY, and PauliZ measured independently on each qubit. These single-qubit Pauli operators cannot capture inter-qubit correlations without relying entirely on the variational circuit to encode such correlations into the quantum state.

2. **Fixed circuit ansatz**: The Sim14 circuit uses a manually designed RY-CRX gate arrangement. While effective, this architecture was chosen heuristically and may not be optimal for the specific learning task.

v5 addresses both limitations by introducing two complementary innovations:

- **ANO (Adaptive Non-Local Observables)**: Learnable k-local Hermitian measurement matrices that can capture multi-qubit correlations directly at the measurement stage (Lin et al. 2025).
- **DiffQAS (Differentiable Quantum Architecture Search)**: Gradient-based search over the gate-level circuit structure, automatically discovering task-optimal gate arrangements (Sun et al. 2023).

Both innovations are integrated into a single codebase with ablation flags (`--no-ano`, `--no-dqas`) that enable a controlled 2x2 factorial study.

---

## 2. Background: QTSTransformer v3 Baseline

### 2.1 Architecture Overview

The v3 architecture processes Atari game frames through a pipeline:

```
Atari Frame (210x160 RGB)
  -> Preprocessing (grayscale, 84x84, normalize)
  -> Frame Stacking (4 frames)
  -> CNN Feature Extractor (DQN Nature architecture)
  -> Sinusoidal Positional Encoding
  -> Quantum Transformer (QSVT + QFF)
  -> Q-values
```

The quantum transformer core consists of two stages:

1. **QSVT (Quantum Singular Value Transformation)**: Polynomial state preparation via linear combination of unitaries (LCU). Processes temporal dependencies across the 4 stacked frames by evolving quantum states through parametrized circuits.

2. **QFF (Quantum Feed-Forward)**: A final variational circuit followed by measurement. Extracts classical Q-values from the evolved quantum state.

### 2.2 The Sim14 Circuit Ansatz

Each ansatz layer in v3 consists of 4 "slots":

| Slot | Gate Type | Connectivity |
|------|-----------|-------------|
| 0 | RY(theta) | Each qubit independently |
| 1 | CRX(theta) | Forward ring: qubit i controls (i+1) mod n |
| 2 | RY(theta) | Each qubit independently |
| 3 | CRX(theta) | Backward ring: qubit i controls (i-1) mod n |

This pattern (single-qubit rotation -> entangling -> single-qubit rotation -> entangling) is a standard variational circuit design, but the specific choice of RY for single-qubit gates and CRX for entangling gates is heuristic.

### 2.3 Fixed Pauli Measurements

v3 measures 3 observables per qubit:

```
<PauliX_0>, <PauliX_1>, ..., <PauliX_{n-1}>,
<PauliY_0>, <PauliY_1>, ..., <PauliY_{n-1}>,
<PauliZ_0>, <PauliZ_1>, ..., <PauliZ_{n-1}>
```

This yields 3*n_qubits = 24 expectation values (for 8 qubits), which are fed to a classical feed-forward head to produce Q-values. Each Pauli observable acts on a single qubit, meaning any inter-qubit information must be entirely encoded in the quantum state by the circuit. This places the full burden of expressivity on the variational ansatz.

---

## 3. Adaptive Non-Local Observables (ANO)

### 3.1 Theoretical Motivation

The Heisenberg picture of quantum mechanics provides the key insight. The expectation value of an observable H measured on a state evolved by a unitary U(theta) can be equivalently expressed as:

```
<psi| U^dag(theta) H U(theta) |psi> = <psi| H(theta) |psi>
```

where H(theta) = U^dag(theta) H U(theta) is the "evolved observable." This equivalence implies that instead of making the circuit deeper to encode more complex correlations into the state, we can keep the circuit shallow and instead make the observable richer.

**ANO implements this principle**: rather than fixed single-qubit Pauli operators, we learn k-local Hermitian matrices that act on subsystems of k qubits. A 2-local Hermitian on qubits (i, i+1) is a general 4x4 Hermitian matrix with 16 real degrees of freedom, compared to 1 degree of freedom for a single-qubit Pauli.

### 3.2 Mathematical Construction

For locality k, each ANO observable is a K x K Hermitian matrix where K = 2^k. A general Hermitian matrix H is parameterized by:

- **D** (diagonal): K real parameters
- **A** (real off-diagonal): K(K-1)/2 real parameters
- **B** (imaginary off-diagonal): K(K-1)/2 real parameters

Total parameters per observable: K^2 = 4^k

For k=2 (our default):
- K = 4 (4x4 matrices)
- D: 4 real params
- A: 6 real params
- B: 6 real params
- Total: 16 params per observable

The matrix is constructed as:

```python
def create_Hermitian(N, A, B, D):
    h = zeros((N, N), dtype=complex128)
    count = 0
    for i in range(1, N):
        h[i-1, i-1] = D[i]          # diagonal
        for j in range(i):
            h[i, j] = A[count+j] + 1j * B[count+j]  # lower triangle
        count += i
    H = h + h.conj().T              # enforce Hermiticity
    return H
```

### 3.3 Sliding Window Placement

We use n_qubits = 8 circular sliding windows, each measuring k=2 adjacent qubits:

```
Window 0: qubits (0, 1)
Window 1: qubits (1, 2)
Window 2: qubits (2, 3)
...
Window 7: qubits (7, 0)  <- wraps around
```

This yields 8 expectation values (one per window), compared to v3's 24 (3 Paulis x 8 qubits). Despite fewer outputs, each ANO measurement captures 2-qubit correlations that single-qubit Paulis cannot access.

### 3.4 ANO Parameter Count

For 8 qubits with k=2:
- 8 windows x 16 params each = **128 ANO parameters**
- These are split into ParameterLists: `A[w]`, `B[w]`, `D[w]` for w in range(8)
- Initialized with `nn.init.normal_(std=2.0)` following Lin et al. (2025)

### 3.5 Separate Optimizer for ANO

ANO parameters are trained with a separate Adam optimizer at a higher learning rate (lr=0.01) compared to the VQC parameters (lr=0.00025). The 40x higher learning rate follows the recommendation from Lin et al. (2025), where the Hermitian parameters were trained at 100x the VQC learning rate.

The rationale: observable parameters define the "measurement basis" and need to adapt quickly to provide useful gradient signals to the slower-learning circuit parameters.

---

## 4. Differentiable Quantum Architecture Search (DiffQAS)

### 4.1 Motivation

The Sim14 ansatz uses fixed RY and CRX gates. But why RY and not RX or RZ? Why CRX and not CRY or CRZ? These choices affect the circuit's expressibility and entangling capability. DiffQAS answers this question empirically by letting the optimization process discover the best gates.

### 4.2 Gate-Level Parametric Approach

The core insight is that PennyLane's `qml.Rot(phi, theta, omega)` gate decomposes as:

```
Rot(phi, theta, omega) = RZ(omega) @ RY(theta) @ RZ(phi)
```

This is a universal single-qubit gate. At specific (phi, omega) values, it reduces to canonical gates:

| Gate | (phi, omega) | Equivalence |
|------|-------------|-------------|
| RY | (0, 0) | Rot(0, theta, 0) = RY(theta) |
| RX | (-pi/2, pi/2) | Rot(-pi/2, theta, pi/2) ~ RX(theta) |
| RZ | (pi/2, -pi/2) | Rot(pi/2, theta, -pi/2) ~ RZ(theta) |

Similarly, `qml.CRot(phi, theta, omega)` provides a parametric space for controlled gates.

### 4.3 Architecture Parameters

Each of the 4 slots per layer has 2 architecture parameters (phi, omega), shared across all qubits/pairs within that slot:

```
arch_params_qsvt: shape (n_ansatz_layers, 4, 2) = (2, 4, 2) = 16 params
arch_params_qff:  shape (1, 4, 2)               = 8 params
Total: 24 architecture parameters
```

**Initialization**: All architecture params initialized to zero, so:
- `Rot(0, theta, 0) = RY(theta)` (single-qubit slots)
- `CRot(0, theta, 0) = CRY(theta)` (controlled slots)

This means the search starts from a known-good starting point (close to v3's RY-CRX architecture).

### 4.4 Phase 1: Continuous Search

During Phase 1 (first `search_episodes` episodes), the architecture parameters are trained jointly with VQC and ANO parameters using a separate Adam optimizer (lr=0.001).

The DiffQAS circuit replaces each fixed gate with its parametric counterpart:

```python
# Slot 0 (was: RY per qubit)
phi_0, omega_0 = arch_params[layer, 0, 0], arch_params[layer, 0, 1]
qml.Rot(phi_0, angle, omega_0, wires=i)

# Slot 1 (was: CRX forward ring)
phi_1, omega_1 = arch_params[layer, 1, 0], arch_params[layer, 1, 1]
qml.CRot(phi_1, angle, omega_1, wires=[i, (i+1) % n])
```

### 4.5 Phase 2: Discretization

After search_episodes, architecture parameters are discretized by nearest-neighbor classification:

```python
def classify_single_qubit_gate(phi, omega):
    dists = {
        'RY': phi^2 + omega^2,
        'RX': (phi + pi/2)^2 + (omega - pi/2)^2,
        'RZ': (phi - pi/2)^2 + (omega + pi/2)^2,
    }
    return argmin(dists)
```

The discovered gate configuration is printed and frozen. QNodes are rebuilt with the discrete gates for the remaining ~9900 production episodes.

---

## 5. Two-Phase Training Procedure

### 5.1 Phase 1: Architecture Search (Episodes 1 to search_episodes)

**Three optimizer groups active:**

| Group | Parameters | Learning Rate | Purpose |
|-------|-----------|---------------|---------|
| VQC | Circuit rotation angles, QSVT coefficients, classical layers | 0.00025 | Main model parameters |
| ANO | Hermitian matrix params (A, B, D) | 0.01 | Observable optimization |
| Arch | Architecture params (phi, omega per slot) | 0.001 | Gate type discovery |

All three optimizers step together on every learning update.

### 5.2 Phase Transition

At episode `search_episodes`:
1. Read architecture parameters from online network
2. Classify each slot to nearest canonical gate
3. Print discovered architecture (e.g., "Layer 0, Slot 0: phi=0.12, omega=-0.08 -> RY")
4. Set phase=2 on both online and target networks
5. Freeze architecture parameters (`requires_grad_(False)`)
6. Rebuild QNodes with discrete gate functions (e.g., `qml.RY` instead of `qml.Rot`)
7. Sync target network
8. Recreate optimizers (arch_optimizer dropped)

### 5.3 Phase 2: Production Training (Episodes search_episodes+1 to num_episodes)

**Two optimizer groups active:**

| Group | Parameters | Learning Rate |
|-------|-----------|---------------|
| VQC | Same as Phase 1 | 0.00025 |
| ANO | Same as Phase 1 | 0.01 |

Architecture parameters are frozen; the discovered gate arrangement is baked into the QNode at trace time, eliminating any overhead from the parametric Rot/CRot gates.

---

## 6. Ablation Study Design

### 6.1 2x2 Factorial Design

To isolate the contribution of each innovation, we conduct a full factorial ablation:

| Condition | ANO | DiffQAS | CLI Flags | Description |
|-----------|-----|---------|-----------|-------------|
| **Baseline** | Off | Off | `--no-ano --no-dqas` | Reproduces v3 behavior exactly |
| **ANO Only** | On | Off | `--no-dqas` | Fixed RY-CRX circuit + learnable Hermitians |
| **DiffQAS Only** | Off | On | `--no-ano` | Architecture search + fixed PauliX/Y/Z |
| **Full v5** | On | On | (default) | Both innovations combined |

### 6.2 SLURM Submission

All four conditions run from the same script via the `ABLATION` environment variable:

```bash
# Submit all 4 ablation conditions
sbatch --export=ALL,ABLATION="baseline"  jobs/run_quantum_transformer_atari_v5.sh
sbatch --export=ALL,ABLATION="ano_only"  jobs/run_quantum_transformer_atari_v5.sh
sbatch --export=ALL,ABLATION="dqas_only" jobs/run_quantum_transformer_atari_v5.sh
sbatch --export=ALL,ABLATION="full"      jobs/run_quantum_transformer_atari_v5.sh
```

### 6.3 Run ID Convention

Each condition produces a distinct RUN_ID for checkpoint and results isolation:

```
QTransformerV5_{env}_Q{n_qubits}_L{n_layers}_D{degree}_K{k_local}_{ablation_tag}_Run{index}

Examples:
QTransformerV5_SpaceInvaders5_Q8_L2_D2_K2_baseline_Run1
QTransformerV5_SpaceInvaders5_Q8_L2_D2_K2_ANOonly_Run1
QTransformerV5_SpaceInvaders5_Q8_L2_D2_K2_DQASonly_Run1
QTransformerV5_SpaceInvaders5_Q8_L2_D2_K2_full_Run1
```

### 6.4 Hypotheses

The ablation study tests four hypotheses:

1. **ANO improves over baseline**: Learnable observables provide richer gradient signals and better capture inter-qubit correlations, leading to higher game scores.

2. **DiffQAS improves over baseline**: The automatically discovered gate arrangement is better suited to the RL task than the fixed RY-CRX heuristic.

3. **ANO + DiffQAS is synergistic**: The combination outperforms either innovation alone, because richer measurements (ANO) benefit from a better-matched circuit (DiffQAS) and vice versa.

4. **Both innovations are necessary**: Neither ANO alone nor DiffQAS alone achieves the full v5 performance, justifying the combined approach.

---

## 7. Architecture and Parameter Counts

### 7.1 Component Breakdown (8 qubits, 2 ansatz layers, k=2, SpaceInvaders)

| Component | Baseline | ANO Only | DiffQAS Only | Full v5 |
|-----------|----------|----------|-------------|---------|
| CNN (shared across conditions) | 1,946,784 | 1,946,784 | 1,946,784 | 1,946,784 |
| feature_projection | 8,256 | 8,256 | 8,256 | 8,256 |
| QSVT params | 7 | 7 | 7 | 7 |
| QFF params | 32 | 32 | 32 | 32 |
| ANO (A, B, D) | 0 | **128** | 0 | **128** |
| Arch params (Phase 1 only) | 0 | 0 | 24 | 24 |
| output_ff (6 actions) | 3,974 | **1,926** | 3,974 | **1,926** |
| **Transformer subtotal** | 12,269 | **10,349** | 12,293 | **10,373** |
| **Full model total** | 1,959,053 | **1,957,133** | 1,959,077 | **1,957,157** |

**Key observation**: v5 full is ~15% *smaller* than baseline in the transformer component. ANO's output dimension (8 windows) is smaller than fixed Paulis (24 = 3x8), saving 2,048 parameters in the output_ff layer. This more than offsets the 128 ANO parameters, resulting in a net parameter reduction.

### 7.2 Why Fewer Parameters Can Be Better

The fixed Pauli approach produces 24 expectation values, many of which may be redundant (e.g., PauliX and PauliY on the same qubit carry overlapping information for certain states). ANO's 8 k-local observables are each 4x4 Hermitian matrices with 16 free parameters, giving them the capacity to capture exactly the correlations that matter for the task. This is a more parameter-efficient representation.

---

## 8. Implementation Details

### 8.1 QNode Construction

PennyLane QNodes are statically traced, meaning the circuit structure must be fixed at trace time. This creates 6 possible QNode variants (3 circuit types x 2 measurement types):

**Circuit variants:**
- `dqas_sim14_circuit` with `arch_params` argument (Phase 1)
- `discrete_sim14_circuit` with baked `gate_config` (Phase 2)
- `sim14_circuit` with fixed RY-CRX (no DiffQAS)

**Measurement variants:**
- ANO: `qml.Hermitian(H[w], wires=window)` with `*H_flat` splatted as positional args
- Fixed: `qml.PauliX/Y/Z(i)` for each qubit

The `_build_qnodes()` method selects the correct variant based on `self.phase`, `self.use_ano`, and `self.use_dqas`, and is called at initialization and again at phase transition.

### 8.2 Hermitian Matrix Argument Passing

PennyLane QNodes require all tensor arguments to be passed positionally when mixing with `*args` unpacking. The ANO Hermitian matrices are passed as `*H_flat` (splatting the list of per-window matrices):

```python
# Correct: all positional
exps = self.qff_qnode_expval(
    normalized_mixed_timestep,   # initial_state
    self.qff_params,              # params
    self.arch_params_qff,         # arch_params (Phase 1 only)
    *H_list                       # H_flat[0], H_flat[1], ..., H_flat[7]
)
```

### 8.3 Checkpoint Save/Load with Phase Recovery

Checkpoints store the complete training state including phase information:

```python
checkpoint = {
    'episode': episode,
    'net_state': self.net.state_dict(),
    'vqc_optimizer_state': ...,
    'ano_optimizer_state': ...,     # None if --no-ano
    'arch_optimizer_state': ...,    # None if --no-dqas or Phase 2
    'phase': self.phase,
    'gate_config_qsvt': ...,       # None until Phase 2
    'gate_config_qff': ...,        # None until Phase 2
    'exploration_rate': ...,
    'torch_rng_state': ...,
    'numpy_rng_state': ...,
    'python_rng_state': ...
}
```

When loading a Phase 2 checkpoint, the gate configurations are restored first, QNodes are rebuilt with discrete gates, and then the model weights are loaded. This ensures the QNode structure matches the saved parameters.

### 8.4 QSVT Variants for Phase 1 vs Phase 2

Two parallel QSVT helper functions handle the difference:

- **Phase 1**: `evaluate_polynomial_state_pl_dqas()` passes `arch_params` to the QSVT QNode
- **Phase 2 / No DiffQAS**: `evaluate_polynomial_state_pl()` does not pass arch_params

The forward pass dispatches to the correct variant based on `self.use_dqas and self.phase == 1`.

---

## 9. Known Limitations

### 9.1 CRot Cannot Represent CRX Exactly

The PennyLane `Rot(phi, theta, omega)` gate decomposes as `RZ(omega) @ RY(theta) @ RZ(phi)`. For single-qubit gates, this covers all rotations up to global phase, which is unobservable. However, for controlled gates, the "global phase" on the target qubit becomes a *relative phase* conditioned on the control qubit, and is therefore physically significant.

Specifically:
- **CRot(-pi/2, theta, pi/2) != CRX(theta)**: Maximum unitary distance = 0.686
- **CRot(0, theta, 0) = CRY(theta)**: Exact match
- **CRot(pi/2, theta, -pi/2) = CRZ(theta)**: Exact match (up to global phase on the 2-qubit space)

This means the DiffQAS search space for controlled gates effectively covers {CRY, CRZ} but not CRX. Since v3's baseline uses CRX, the search starts near a point (phi=0, omega=0 -> CRY) that is *not* the v3 gate. This is acceptable because:

1. The search space {CRY, CRZ + continuous interpolation} is still broader than fixed CRX alone.
2. The search may discover that CRY or CRZ (or an interpolation) is actually better for the task.
3. The VQC rotation parameters can compensate for the gate type change.

### 9.2 Search Phase Duration Sensitivity

The default `search_episodes=100` was chosen as ~1% of the total 10,000 episodes. Too few search episodes may not give architecture parameters enough gradient signal to converge; too many wastes production training time with the less efficient parametric gates. This hyperparameter may need task-specific tuning.

### 9.3 Discretization Gap

The phase transition involves a discrete approximation: continuous (phi, omega) values are snapped to the nearest canonical gate. If the optimal point lies between canonical gates, the discretized circuit may underperform the continuous one. In practice, we expect the continuous values to converge close to canonical gates during search, minimizing this gap.

---

## 10. References

1. **Lin, H.-Y., Tseng, H.-H., Chen, S. Y.-C., & Yoo, S. (2025).** Adaptive Non-local Observable on Quantum Neural Networks. *2025 IEEE International Conference on Quantum Computing and Engineering (QCE)*. [arXiv:2504.13414](https://arxiv.org/abs/2504.13414)

2. **Sun, Y., Ma, Y., & Tresp, V. (2023).** Differentiable Quantum Architecture Search for Quantum Reinforcement Learning. *IEEE International Conference on Quantum Computing and Engineering (QCE23) Workshop on Quantum Machine Learning*. [arXiv:2309.10392](https://arxiv.org/abs/2309.10392)

3. **Vaswani, A., Shazeer, N., Parmar, N., et al. (2017).** Attention is All You Need. *Advances in Neural Information Processing Systems (NeurIPS)*.

4. **Mnih, V., Kavukcuoglu, K., Silver, D., et al. (2015).** Human-level control through deep reinforcement learning. *Nature*, 518(7540), 529-533.

5. **Chen, S. Y.-C. & Tiwari, P. (2025).** Differentiable Quantum Architecture Search for Quantum LSTM. *IEEE Quantum Week 2025*.

6. **Chen, S. Y.-C., Huang, C.-M., Hsing, C.-W., Goan, H.-S., & Kao, Y.-J. (2025).** Differentiable Quantum Architecture Search for Quantum Transformers. *Physical Review Research*.
