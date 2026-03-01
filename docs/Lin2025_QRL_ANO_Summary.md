# Lin et al. (2025) — "Quantum Reinforcement Learning by Adaptive Non-local Observables"

**Citation:** Lin, H.-Y., Tseng, H.-H., Chen, S. Y.-C., & Yoo, S. (2025). Quantum Reinforcement Learning by Adaptive Non-local Observables. *2025 IEEE International Conference on Quantum Computing and Engineering (QCE)*. https://arxiv.org/abs/2507.19629

**Venue:** IEEE Quantum Week 2025 (QCE 2025), Albuquerque, NM

**Compute:** NERSC (Contract No. DE-AC02-05CH11231, award HEPERCAP0033786)

---

## Core Contribution

The paper integrates **Adaptive Non-Local Observables (ANO)** into VQC-based reinforcement learning agents for both **DQN** and **A3C** frameworks. By jointly optimizing circuit rotation parameters θ and multi-qubit measurement parameters φ, ANO-VQC agents consistently outperform baseline VQCs with fixed local Pauli measurements across multiple benchmark environments. Ablation studies demonstrate that adaptive measurements expand the representational capacity of quantum models **without increasing circuit depth**.

---

## 1. Background: Why ANO for RL?

### 1.1 The Fixed Observable Limitation in QRL

Standard VQC-based RL agents use fixed Pauli observables (typically Z) for measurement, which constrains expectation value outputs to [-1, 1]. This is particularly problematic for RL because:

- **Q-values can be unbounded**: Cumulative discounted rewards often exceed [-1, 1]
- **Policy gradients need rich gradients**: Fixed observables limit the gradient landscape
- **Multi-action outputs are constrained**: Each action's Q-value is limited to the same narrow range

### 1.2 The ANO Solution

ANO replaces fixed Pauli measurements with **learnable k-local Hermitian matrices** H(φ) whose eigenvalue spectrum adapts during training. This:

1. Expands the output range to match task-specific Q-value scales
2. Captures inter-qubit correlations that single-qubit Paulis miss
3. Increases expressivity without deepening the circuit (Heisenberg picture equivalence)

---

## 2. Architecture

### 2.1 VQC with ANO

The ANO-VQC model computes:

```
f_{θ,φ}(s) = ⟨ψ₀| W†(s) U†(θ) H(φ) U(θ) W(s) |ψ₀⟩
```

where:
- **W(s)**: Input encoding — Hadamard gates on all qubits, followed by single-qubit rotations R(s_i)
- **U(θ)**: Variational layer — alternating CNOT gates between neighboring qubits and local rotations R(θ_i)
- **H(φ)**: Adaptive k-local observable — trainable K×K Hermitian matrix (K = 2^k)

### 2.2 Multi-Output Mechanism

For an action space of dimension |A|, the first |A| outputs from different k-qubit measurement groupings serve as action logits. For example, with a 4-qubit system and 3-local observables:

- Group 1: qubits (0, 1, 2) → Q(s, a₁)
- Group 2: qubits (1, 2, 3) → Q(s, a₂)

Each group has its own trainable Hermitian matrix.

### 2.3 ANO-DQN

The ANO-VQC serves as the Q-function approximator:

```
Q_{θ,φ}(s, a) = f_{θ,φ}(s)_a
```

**Bellman loss:**
```
L(θ, φ) = E[(R(s,a) + γ max_{a'} Q_{θ',φ'}(s',a') - Q_{θ,φ}(s,a))²]
```

Both rotation parameters θ and observable parameters φ are jointly optimized.

### 2.4 ANO-A3C

The Asynchronous Advantage Actor-Critic framework uses two ANO-VQC networks:

- **Actor**: Policy π_{θ,φ}(a|s) via Gibbs/softmax transformation of ANO-VQC outputs
- **Critic**: Value function V_{ϑ,φ'}(s) with separate parameters

**A3C loss components:**
- Policy loss: -log π_θ(a_t|s_t) · A_t (advantage-weighted)
- Value loss: (G_t^(n) - V_ψ(s_t))²
- Entropy regularization: -β Σ_a π_θ(a|s_t) log π_θ(a|s_t)

Asynchronous parallel workers update shared parameters.

---

## 3. Experimental Setup

### 3.1 Environments

| Environment | State Dim | Action Dim | Reward | Challenge |
|-------------|-----------|------------|--------|-----------|
| **CartPole-v1** | 4 (x, ẋ, θ, θ̇) | 2 (left/right) | +1 per timestep upright | Balance control |
| **MountainCar-v0** | 2 (position, velocity) | 3 (left/none/right) | Sparse (reach goal) | Sparse reward, momentum |
| **MiniGrid 8×8** | High-dim grid | Discrete | +1 on reaching goal | Sparse reward, navigation |
| **MiniGrid SimpleCrossing S9N1** | High-dim grid | Discrete | +1 on reaching goal | Narrow corridor navigation |

### 3.2 Ablation Configurations

Three variants tested for each environment:

| Variant | Rotation U(θ) | ANO H(φ) | Description |
|---------|--------------|-----------|-------------|
| **ANO + Rotation** | Yes | Yes | Full model — both circuit and observable trainable |
| **Rotation Only** | Yes | No (fixed Pauli Z) | Standard VQC baseline |
| **Measurement Only** | No (U = Identity) | Yes | Tests observable expressivity alone |

### 3.3 Locality Configurations

| Experiment | Qubits | ANO Locality (k) | Observable Size |
|------------|--------|-------------------|-----------------|
| CartPole (DQN & A3C) | 4 | 3-local | 8×8 Hermitian |
| MountainCar (DQN) | 4 | 3-local | 8×8 Hermitian |
| MountainCar (DQN) | 6 | 6-local | 64×64 Hermitian |
| MiniGrid (A3C) | 4 | — | Classical linear reduction to 4 features |

### 3.4 State Preprocessing

- **CartPole**: Direct 4D state → 4-qubit encoding
- **MountainCar**: 2D state duplicated to test different locality levels (4 or 6 qubits)
- **MiniGrid**: Classical linear layer reduces high-dimensional grid state to 4 features for ANO input

---

## 4. Results

### 4.1 CartPole — DQN

| Variant | Convergence Speed | Final Reward | Stability |
|---------|-------------------|--------------|-----------|
| **3-local + Rotation** | **Fastest** | **500 (max)** | **Most stable** |
| Rotation Only | Slower | Lower | Moderate |
| Measurement Only | Similar to rotation-only | Similar | Moderate |

The combined ANO+Rotation variant reaches the 500-step maximum reward cap earliest and maintains it with the greatest stability.

### 4.2 CartPole — A3C

| Variant | Moving Avg Reward (by ep ~12K) |
|---------|-------------------------------|
| **3-local + Rotation** | **~400** |
| Measurement Only | ~250 |
| Rotation Only | <100 |

A3C shows a **dramatic gap**: ANO+Rotation is 4x better than rotation-only. Notably, measurement-only surpasses rotation-only under A3C (but not under DQN), suggesting policy gradient methods better exploit rich observables.

### 4.3 MountainCar — DQN

| Variant | Performance |
|---------|-------------|
| **6-local + Rotation** | **Best** |
| **6-local, No Rotation** | **Comparable to 6-local + Rotation** |
| 3-local + Rotation | Suboptimal |
| Rotation Only (fixed Z) | Insufficient expressivity |

**Critical finding**: At 6-local, rotation gates provide **negligible additional benefit**. The observable alone captures sufficient correlations. This directly validates the Heisenberg picture argument: rich observables can substitute for deep circuits.

### 4.4 MiniGrid 8×8 — A3C

| Variant | Success Rate | Convergence | Variance |
|---------|-------------|-------------|----------|
| **ANO + Rotation** | **~95%** | **Fastest** | **Lowest** |
| Rotation Only | ~95% | Slowest | **Highest** |
| Measurement Only | ~95% | Moderate | Moderate |

All variants eventually converge, but ANO+Rotation arrives fastest with minimal variance.

### 4.5 MiniGrid SimpleCrossing S9N1 — A3C

| Variant | Success Rate |
|---------|-------------|
| **ANO + Rotation** | **>80%** |
| Rotation Only | ~40% |
| Measurement Only | ~30% |

The hardest environment shows the clearest separation. ANO+Rotation achieves **2x** the success rate of rotation-only. This demonstrates ANO is most valuable in **complex tasks** where fixed observables are insufficient.

---

## 5. Key Findings

### Finding 1: ANO + Rotation Consistently Wins

Across all environments and both RL algorithms, the combined model (trainable circuit + trainable observable) achieves the best performance. Neither component alone matches the combined approach.

### Finding 2: Locality > Circuit Depth

Increasing measurement locality (k) provides greater expressivity gains than adding variational circuit layers. This has direct hardware implications — ANO enables **shallower circuits** that are more NISQ-friendly.

### Finding 3: At Sufficient Locality, Rotations Become Optional

MountainCar 6-local demonstrates that when the observable is expressive enough, the variational circuit contributes negligibly. This validates the Heisenberg picture equivalence experimentally.

### Finding 4: A3C Benefits More from ANO than DQN

Policy gradient methods (A3C) better exploit rich observables than value-based methods (DQN). Under A3C:
- Measurement-only variant learns steadily to moderate rewards
- Under DQN, measurement-only stagnates

This may be because A3C's policy gradient directly differentiates through the softmax of observable outputs, making spectral range expansion more impactful.

### Finding 5: ANO Is Most Valuable for Hard Tasks

The performance gap between ANO+Rotation and baselines **widens** as task complexity increases:
- CartPole: Moderate gap
- MiniGrid 8×8: Speed/variance gap
- MiniGrid SimpleCrossing: **2x success rate gap**

---

## 6. Relevance to Our QRL QTSTransformer

### 6.1 Direct Comparison

| Aspect | Lin et al. QRL-ANO | Our QRL QTSTransformer v3 |
|--------|-------------------|--------------------------|
| **Environments** | CartPole, MountainCar, MiniGrid | **Atari** (SpaceInvaders, Breakout, etc.) |
| **State processing** | Direct encoding or linear reduction | **CNN + QSVT temporal transformer** |
| **Observables** | **Learnable k-local ANO** | Fixed PauliX/Y/Z per qubit |
| **Circuit** | Simple encoding + 1 variational layer | Sim14 ansatz, 2 layers |
| **RL algorithm** | DQN and A3C | Double DQN |
| **Qubits** | 4-6 | 8 |
| **Output mechanism** | k-qubit groupings → action logits | 3×n_qubits expvals → FF head |

### 6.2 Integration Opportunities

1. **Replace fixed Pauli measurements with ANO**: Our v3 QFF uses `[PauliX(i), PauliY(i), PauliZ(i)]` for 24 fixed measurements. ANO would replace these with learnable k-local Hermitian matrices.

2. **Reduce circuit depth**: Lin's ablation shows rich observables compensate for shallow circuits. We could potentially reduce from 2 ansatz layers to 1, cutting quantum simulation cost roughly in half.

3. **Expand Q-value range**: Our current output is bounded by [-1, 1] per observable, then mapped through an FF head. ANO would let the quantum circuit itself output values matched to the reward scale.

4. **Separate optimizers**: Following Chen et al.'s finding, use a higher learning rate (10-100x) for observable parameters vs. circuit parameters.

### 6.3 Proposed ANO Configuration for 8-Qubit System

| Scheme | k | Windows | Params per Window | Total ANO Params |
|--------|---|---------|-------------------|-----------------|
| Sliding 2-local | 2 | 7 | 16 | 112 |
| Sliding 3-local | 3 | 6 | 64 | 384 |
| Pairwise 2-local | 2 | 28 | 16 | 448 |

For comparison, our current quantum transformer has ~12,781 total params. Adding 112-448 ANO params is a modest increase (<4%).

### 6.4 Expected Impact

Based on Lin et al.'s results:
- **Breakout** (where fixed-observable classical baseline fails): ANO could further improve quantum advantage
- **SpaceInvaders** (18% quantum advantage): ANO could widen the gap
- **DonkeyKong/MarioBros** (capacity-limited): ANO might help without needing more qubits

---

## 7. Limitations

1. **Simple environments only**: CartPole, MountainCar, MiniGrid — no Atari-scale experiments
2. **Small qubit counts**: 4-6 qubits; scalability to 8+ not demonstrated
3. **Missing implementation details**: Paper does not specify optimizers, learning rates, batch sizes, or replay buffer configurations
4. **No hardware validation**: Simulation only; no demonstration on noisy quantum devices
5. **Limited RL algorithms**: DQN and A3C only; no PPO, SAC, or other modern algorithms tested

---

## 8. Connection to Related ANO Papers

This paper is the third in a series by the same group:

| Paper | Focus | Key Contribution |
|-------|-------|-----------------|
| **Chen et al. (2025)** — "Learning to Measure QNNs" | Foundational theory | Learnable Hermitian observables, separate optimizer finding |
| **Lin et al. (2025)** — "Adaptive Non-Local Observable on QNNs" | Scalability | k-local truncation, sliding/pairwise schemes |
| **Lin et al. (2025)** — "QRL by ANO" (this paper) | RL application | ANO in DQN/A3C, ablation on RL benchmarks |

The three papers form a complete pipeline: Chen provides the "what" (learnable measurement), the first Lin paper provides the "how to scale it" (k-local), and this paper provides the "where to apply it" (reinforcement learning).
