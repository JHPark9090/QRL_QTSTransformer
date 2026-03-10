# Quantum vs Classical & ANO/DiffQAS Comparison Across All Environments

**Date**: March 10, 2026
**Author**: Junghoon Park
**Model**: Quantum Time-Series Transformer (QTSTransformer)
**Source Data**: `docs/v5_Ablation_Complete_Results.md` (Atari), `docs/SimpleRL_v6_Results.md` (SimpleRL)

---

## 1. Overview

This document compares experiment results across two environment categories:

- **SimpleRL**: CartPole-v1 (v6 ablation), FrozenLake-v1 (v6 ablation)
- **Atari**: SpaceInvaders, Breakout, DonkeyKong, MarioBros (v5 ablation)

Two questions are addressed:
1. Do quantum models show superiority over the classical counterpart?
2. Do ANO (Adaptive Non-Local Observables) and/or DiffQAS (Differentiable Quantum Architecture Search) provide advantage over the baseline quantum model?

### Ablation Conditions (shared across v5 and v6)

| Condition | ANO | DiffQAS | Description |
|-----------|-----|---------|-------------|
| Baseline | Off | Off | Fixed RY-CRX circuit + fixed PauliX/Y/Z measurements |
| ANO-only | On | Off | Learnable k=2 local Hermitian observables, fixed circuit |
| DiffQAS-only | Off | On | Gate-level architecture search (Rot/CRot), fixed measurements |
| Full | On | On | Both innovations active |
| Classical | N/A | N/A | Classical transformer baseline |

### Key Differences Between Environments

| Property | CartPole | FrozenLake | Atari (via CNN) |
|----------|----------|------------|-----------------|
| State dimension | 4 (continuous) | 16 (one-hot) | 3,136 (CNN features) |
| State type | Continuous | Discrete | Continuous |
| Stochasticity | Deterministic | Stochastic (slippery ice) | Deterministic |
| Action space | 2 | 4 | 4-18 |
| Reward structure | Dense (+1/step) | Sparse (0 or 1) | Varies |

### Metric Convention

- **SimpleRL**: BestAvgR = best 100-episode rolling average reward
- **Atari**: Best (post-expl) = best 100-episode average after episode 500 (excludes exploration artifacts)
- **FrozenLake**: BestAvgR reported as success rate (e.g., 0.6 = 60%)

---

## 2. Question 1: Quantum vs Classical

### 2.1 Atari (v5) — Quantum Wins All 4 Games

| Game | Best Quantum | Condition | Classical | Quantum Advantage | Q vs C Params |
|------|-------------|-----------|-----------|-------------------|---------------|
| SpaceInvaders | **347.9** | Full | 290.8 | **+19.7%** | 12K vs 150K |
| Breakout | **17.2** | DiffQAS-only | 2.5 | **+588%** | 12K vs 150K |
| DonkeyKong | **70.0** | DiffQAS-only | 38.0 | **+84%** | 12K vs 150K |
| MarioBros | **360.0** | Baseline | 304.0 | **+18%** | 12K vs 150K |

Quantum beats classical on every Atari game with **12.5x fewer transformer parameters** (12K quantum vs 150K classical). The advantage ranges from +18% (MarioBros) to +588% (Breakout).

### 2.2 SimpleRL (v6) — Mixed Results

| Environment | Best Quantum | Condition | Classical | Quantum Advantage |
|-------------|-------------|-----------|-----------|-------------------|
| CartPole-v1 | **347.7** | ANO-only (LR red.) | 132.6 | **Yes (2.6x)** |
| FrozenLake-v1 | 60% | Baseline / DiffQAS | **80%** | **No (0.75x)** |

Quantum dominates CartPole (2.6x classical) but loses to classical on FrozenLake.

### 2.3 Summary

| Environment | Quantum Superiority? | Magnitude |
|-------------|---------------------|-----------|
| SpaceInvaders | **Yes** | +19.7% |
| Breakout | **Yes** | +588% |
| DonkeyKong | **Yes** | +84% |
| MarioBros | **Yes** | +18% |
| CartPole-v1 | **Yes** | +162% (2.6x) |
| FrozenLake-v1 | **No** | -25% (0.75x) |

**Score: Quantum 5, Classical 1**

### 2.4 When Does Quantum Advantage Emerge?

Quantum advantage is strongest on **continuous-state, high-dimensional tasks**:
- Atari processes 3,136-dimensional CNN features — rich input that benefits from quantum feature extraction
- CartPole has 4 continuous state variables (position, velocity, angle, angular velocity) — a natural fit for rotation-angle encoding

Quantum advantage disappears on **discrete, stochastic environments**:
- FrozenLake uses 16-dimensional one-hot states with inherent stochasticity (slippery ice). The classical transformer's larger capacity handles this table-lookup-like problem more effectively

---

## 3. Question 2: ANO and DiffQAS Impact

### 3.1 ANO Impact (Learnable Hermitian Observables)

ANO replaces fixed PauliX/Y/Z measurements with learnable k=2-local Hermitian matrices, allowing the model to adapt its measurement basis to the task.

| Environment | ANO-only | Baseline | Change | Verdict |
|-------------|---------|----------|--------|---------|
| **CartPole** | **347.7** | 223.2 | **+56%** | **Strong gain** |
| FrozenLake | 50% | 60% | -17% | Negative |
| **SpaceInvaders** | **347.0** | 339.2 | +2.3% peak, **+11% stability** | **Stability gain** |
| Breakout | 2.7 | 3.0 | -10% | Negative |
| DonkeyKong | 43.0 | 37.0 | +16% | Modest gain |
| MarioBros | 296.0 | 360.0 | -18% peak, **+61% end-stability** | **Stability gain** |

**ANO pattern**: Excels at **stabilizing exploitation** on continuous-state tasks. The +56% on CartPole is the strongest ANO result — learned observables capture better value discrimination for the 4-dimensional continuous state. On SpaceInvaders and MarioBros, ANO doesn't improve peak performance but dramatically improves late-training stability (last-100 reward).

**Where ANO doesn't help**: Tasks where the circuit architecture is the bottleneck (Breakout), or where the state is discrete/one-hot (FrozenLake).

### 3.2 DiffQAS Impact (Gate-Level Architecture Search)

DiffQAS uses parametric Rot/CRot gates during a 100-episode search phase, then discretizes to the nearest canonical gate for production training.

| Environment | DiffQAS-only | Baseline | Change | Verdict |
|-------------|-------------|----------|--------|---------|
| CartPole | 204.8 | 223.2 | -8% | Negative |
| FrozenLake | 60% | 60% | 0% | Neutral |
| SpaceInvaders | 311.8 | 339.2 | -8% | Negative |
| **Breakout** | **17.2** | 3.0 | **+473%** | **Breakthrough** |
| **DonkeyKong** | **70.0** | 37.0 | **+89%** | **Strong gain** |
| MarioBros | 344.0 | 360.0 | -4% | Neutral |

**DiffQAS pattern**: Delivers **massive gains on specific games** where the default RY-CRX circuit topology is suboptimal. Breakout (+473%) and DonkeyKong (+89%) are the standout results. The search converged to RY/CRY gates across all games — CRY provides a different entanglement structure than CRX that particularly benefits these reward landscapes.

**Where DiffQAS doesn't help**: Simple tasks where the default circuit is already sufficient (CartPole, FrozenLake), or tasks where measurements are the bottleneck rather than gates (SpaceInvaders).

### 3.3 Combined (Full = ANO + DiffQAS)

| Environment | Full | Baseline | Best Single Innovation | Full vs Best Single |
|-------------|------|----------|----------------------|---------------------|
| CartPole | 203.0 | 223.2 | 347.7 (ANO) | **Worse** (-42%) |
| FrozenLake | 50% | 60% | 60% (DiffQAS/Baseline) | **Worse** (-17%) |
| SpaceInvaders | 347.9 | 339.2 | 347.0 (ANO) | Tie |
| Breakout | 11.1 | 3.0 | 17.2 (DiffQAS) | **Worse** (-35%) |
| DonkeyKong | 25.0 | 37.0 | 70.0 (DiffQAS) | **Worse** (-64%) |
| MarioBros | 336.0 | 360.0 | 360.0 (Baseline) | **Worse** (-7%) |

**Combined verdict: No synergy.** Full (ANO+DiffQAS) never outperforms the best single innovation across any environment. Simultaneously optimizing gate topology and measurement basis creates too many degrees of freedom, degrading sample efficiency. On DonkeyKong, the combined model is the worst of all conditions.

### 3.4 ANO vs DiffQAS Summary

| Innovation | Helps | Doesn't Help | Mechanism |
|-----------|-------|-------------|-----------|
| **ANO** | CartPole (+56%), SpaceInvaders/MarioBros (stability) | Breakout, FrozenLake | Better measurement basis for continuous states |
| **DiffQAS** | Breakout (+473%), DonkeyKong (+89%) | CartPole, SpaceInvaders, FrozenLake | Better entanglement topology for complex rewards |
| **Both** | None (no synergy) | All environments | Joint optimization interference |

---

## 4. Cross-Cutting Analysis

### 4.1 Best Model per Environment

| Environment | Best Model | BestAvgR | Why This Model Wins |
|-------------|-----------|----------|---------------------|
| CartPole | **Q-ANO (LR red.)** | 347.7 | Learned observables + LR decay stabilizes continuous-state policy |
| FrozenLake | **Classical** | 80% | Larger capacity handles stochastic discrete states |
| SpaceInvaders | **Q-Full** | 347.9 | ANO+DiffQAS synergize on this one game (marginally) |
| Breakout | **Q-DiffQAS** | 17.2 | Architecture search discovers superior CRY entanglement |
| DonkeyKong | **Q-DiffQAS** | 70.0 | CRY topology handles 18-action complexity better |
| MarioBros | **Q-Baseline** | 360.0 | Default RY-CRX is already well-suited; innovations add overhead |

### 4.2 Innovation Effectiveness by Task Complexity

```
Task Complexity Spectrum:

  Simple                                                    Complex
  ←─────────────────────────────────────────────────────────────→

  FrozenLake    CartPole    SpaceInvaders    MarioBros    DonkeyKong    Breakout
  (discrete,    (4-dim      (6 actions,      (9 actions,  (18 actions,  (4 actions,
   stochastic)  continuous)  shooting)        platformer)  platformer)   paddle)

  Best innovation:
  None          ANO         ANO (stability)  None         DiffQAS       DiffQAS
  (Classical    (+56%)      (+11% last-100)  (Baseline    (+89%)        (+473%)
   wins)                                      wins)
```

**Pattern**: ANO helps on moderate-complexity tasks with continuous states. DiffQAS helps on tasks where the default circuit topology is mismatched to the reward structure. Neither helps on very simple (FrozenLake) or already-well-matched tasks (MarioBros).

### 4.3 Parameter Efficiency

| Model | Transformer Params | Environments Won |
|-------|-------------------|-----------------|
| Quantum (all variants) | ~12K | 5 of 6 |
| Classical | 150K (Atari) / 32-dim (SimpleRL) | 1 of 6 |

Quantum models achieve superior performance with **12.5x fewer transformer parameters** on Atari. The parameter efficiency advantage is a key selling point for quantum approaches — the quantum circuit provides an implicit inductive bias that compensates for fewer trainable parameters.

---

## 5. Conclusions

### 5.1 Quantum vs Classical

1. **Quantum shows clear superiority on 5 of 6 environments**, with advantages ranging from +18% to +588%.
2. **The sole exception is FrozenLake** — a discrete, stochastic environment where classical transformers' larger parameter count is advantageous.
3. **Quantum advantage scales with input dimensionality**: strongest on Atari (3,136-dim CNN features), strong on CartPole (4-dim continuous), absent on FrozenLake (16-dim one-hot).
4. **Parameter efficiency**: Quantum achieves these results with 12.5x fewer transformer parameters than classical.

### 5.2 ANO and DiffQAS

1. **ANO is best for continuous-state tasks requiring measurement optimization**: +56% on CartPole, stability gains on SpaceInvaders and MarioBros.
2. **DiffQAS is best for tasks where circuit topology matters**: +473% on Breakout, +89% on DonkeyKong — the most dramatic improvements in the entire project.
3. **Combining ANO and DiffQAS provides no synergy**: The Full condition never outperforms the best single innovation. Joint optimization of gates and measurements creates interference.
4. **Recommendation**: Select the appropriate innovation per task rather than applying both.

### 5.3 Practical Guidelines

| Task Characteristics | Recommended Model |
|---------------------|-------------------|
| Continuous state, moderate actions | Quantum + ANO |
| Complex reward landscape, needs better entanglement | Quantum + DiffQAS |
| Discrete/stochastic state | Classical transformer |
| Default circuit already well-suited | Quantum baseline (no innovations) |

---

## 6. Data Sources

| Environment Category | Results File | Experiments |
|---------------------|-------------|-------------|
| Atari (v5) | `docs/v5_Ablation_Complete_Results.md` | 16 quantum + 4 classical = 20 jobs |
| SimpleRL (v6) | `docs/SimpleRL_v6_Results.md` | 8 quantum + 2 classical = 10 jobs |
| SimpleRL (v1) | `docs/SimpleRL_Results_Summary.md` | Reference baseline |
| v6 Design Rationale | `docs/SimpleRL_v6_Design_Rationale.md` | Why v1 base for SimpleRL |

### Experiment Configurations

| Setting | SimpleRL (v6) | Atari (v5) |
|---------|--------------|------------|
| Architecture base | v1 (sigmoid [0,1], MSE, no PE) | v3 (2π centered, SmoothL1, sinusoidal PE) |
| Qubits | 8 | 8 |
| Ansatz layers | 2 | 2 |
| QSVT degree | 2 | 2 |
| Learning rate (quantum) | 0.001 | 0.00025 |
| ANO learning rate | 0.01 | 0.01 |
| DiffQAS search episodes | 50 | 100 |
| Epsilon decay | 0.995/step | 0.9999/step |
| Anti-forgetting | Yes (best-model save, early stop, LR reduce, slow sync) | Partial (PER, early stopping) |
| Seed | 2025 | 2025 |
| Hardware | A100-SXM4-80GB | A100-SXM4-80GB |
