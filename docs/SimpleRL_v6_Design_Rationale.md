# SimpleRL v6: Design Rationale and Findings

**Date**: March 6, 2026
**Authors**: Junghoon Park
**Status**: v6 experiments submitted (Jobs 49735467-49735477)

---

## 1. Background

The Quantum Time-Series Transformer (QTSTransformer) architecture was first validated on simple RL environments (CartPole-v1, FrozenLake-v1) using the **v1** design, then scaled to Atari games where architectural upgrades (**v3/v5**) significantly improved performance. When we applied the v5 architecture back to SimpleRL environments, all models failed to learn. This document explains why, and what v6 does differently.

---

## 2. The v3/v5 Upgrades That Helped Atari

Between v1 and v5, three architectural changes were introduced in v3:

| Change | v1 | v3/v5 |
|--------|-----|-------|
| **Angle scaling** | `sigmoid(x)` → [0, 1] | `(sigmoid(x) - 0.5) × 2π` → [-π, π] |
| **Positional encoding** | None | Sinusoidal PE (Vaswani et al., 2017) |
| **Loss function** | `MSELoss` | `SmoothL1Loss` (Huber loss) |
| **Gradient clipping** | None | `clip_grad_norm_(10.0)` |

These changes produced substantial improvements on Atari:

| Game | v2 Peak AvgR | v3 Peak AvgR | Improvement |
|------|-------------|-------------|-------------|
| SpaceInvaders | 283.1 | **336.3** | +19% |
| Breakout | 2.9 | **14.2** | +390% |
| MarioBros | 392.0 | **504.0** | +29% |

v5 then added ANO (Adaptive Non-Local Observables) and DiffQAS (Differentiable Quantum Architecture Search) on top of the v3 base, with further gains — most notably DiffQAS achieving +473% on Breakout.

---

## 3. Why v3/v5 Changes Hurt SimpleRL

### 3.1 The Core Problem: Input Dimensionality

Atari and SimpleRL operate at fundamentally different scales:

| Property | CartPole-v1 | FrozenLake-v1 | Atari (via CNN) |
|----------|-------------|---------------|-----------------|
| State dimension | 4 | 16 (one-hot) | 3,136 |
| State type | Continuous | Discrete | CNN features |
| Values range | [-4.8, 4.8] | {0, 1} | [0, 1] |
| Temporal complexity | Low | Low | High |

The v3 changes were designed for the high-dimensional, high-complexity Atari setting. When applied to low-dimensional SimpleRL, they introduce unnecessary complexity that destabilizes training.

### 3.2 2π Angle Scaling: Beneficial for Atari, Harmful for CartPole

**Why it helps Atari**: With 3,136-dimensional CNN features projected to rotation angles, the sigmoid [0, 1] range severely constrains the expressible quantum states. The centered 2π scaling provides the full Bloch sphere rotation range [-π, π], enabling the circuit to express richer transformations. This is critical when the input has high intrinsic complexity.

**Why it hurts CartPole**: CartPole has only 4 state dimensions (cart position, cart velocity, pole angle, pole angular velocity). The sigmoid [0, 1] range provides sufficient expressivity for this simple mapping. The 2π scaling introduces:
- **Initialization instability**: At initialization, `sigmoid(0) = 0.5`, so centered scaling gives `(0.5 - 0.5) × 2π = 0`. But small weight perturbations cause large angle changes (the sigmoid-to-2π mapping has high sensitivity near 0.5), making early training chaotic.
- **Gradient landscape roughness**: Larger rotation ranges create more oscillatory loss landscapes, making gradient descent less reliable for the simple mapping that CartPole requires.

### 3.3 Sinusoidal Positional Encoding: Unnecessary for SimpleRL

**Why it helps Atari**: Atari agents process sequences of 4 stacked video frames. The temporal ordering matters enormously — frame 1 vs frame 4 contains velocity information. PE helps the transformer distinguish frame positions.

**Why it's unnecessary for CartPole**: The state already contains explicit velocity information (cart velocity, pole angular velocity). The 4-timestep history provides redundant temporal context that the model can learn without positional encoding. Adding PE introduces additional signal that the quantum circuit must learn to ignore, consuming limited circuit capacity.

### 3.4 SmoothL1Loss vs MSELoss

**SmoothL1Loss** (Huber loss) is linear for large errors and quadratic for small errors. This is preferred in Atari where Q-value targets can have high variance due to sparse rewards and long episodes.

**MSELoss** provides stronger gradients for all error magnitudes. For CartPole's simple reward structure (+1 per step, dense), MSELoss drives faster convergence. The v1 CartPole results showed learning starting as early as episode 70 with MSELoss.

### 3.5 Empirical Evidence

**v1 (CartPole)**: Peak AvgR = **500** (perfect) at episode 77-106. Reached optimal policy in ~70 episodes.

**v5 baseline on CartPole** (with epsilon decay fix): Peak AvgR = **80.6** at episode 190. Never approached optimal. The v3 changes reduced peak performance by **84%**.

---

## 4. Catastrophic Forgetting

### 4.1 The Pattern

Both v1 and v5 exhibit the same failure mode on CartPole:

```
v1:  Ep 0-70: Learning (AvgR 14 → 54)
     Ep 77-106: Peak (AvgR 500, 30 consecutive perfect scores)
     Ep 107-340: Collapse (AvgR 500 → 78)
     Final AvgR: 175

v5 baseline (fixed):
     Ep 0-70: Learning (AvgR 16 → 18)
     Ep 80-190: Peak (AvgR 28 → 80.6)
     Ep 200-1000: Collapse (AvgR 80 → 20)
     Final AvgR: 24
```

The model discovers a good policy, then continued training overwrites it. This is **catastrophic forgetting** — the same phenomenon observed in continual learning, but here it occurs within a single task.

### 4.2 Root Causes

**1. Target Network Overwriting**: With `sync_every=100`, the target network is updated frequently. Once the policy starts degrading (due to a few bad Q-value updates), the target network quickly adopts the degraded policy, creating a feedback loop.

**2. Replay Buffer Poisoning**: As the policy degrades, new bad experiences enter the replay buffer and dilute the good experiences from the peak period. With `memory_size=10000` and ~20 steps per CartPole episode, the buffer turns over every ~500 episodes — meaning peak-era experiences are completely flushed within a few hundred episodes.

**3. Constant Learning Rate**: The same learning rate that enabled rapid initial learning also enables rapid forgetting. Once the optimal policy is found, continued updates at the same rate cause the weights to drift away from the optimum.

**4. No Stopping Criterion**: Without early stopping, training continues indefinitely past the peak, accumulating damage to the learned policy.

### 4.3 Loss Explosion as a Symptom

In v1, the loss trajectory tells the story:
- Ep 0-100: Loss ~7 (learning)
- Ep 100-130: Loss stable (peak performance)
- Ep 130-500: Loss explodes to ~170 (catastrophic forgetting)

The loss explosion indicates the Q-value estimates are becoming increasingly inconsistent, confirming that the model is "unlearning" its policy.

---

## 5. v6 Anti-Forgetting Mechanisms

v6 addresses each root cause with a targeted mechanism:

### 5.1 Best-Model Checkpointing

**Addresses**: Inability to recover peak performance after forgetting.

The agent tracks the best 100-episode average reward seen during training. Whenever a new best is achieved, the model weights are saved to `best_model.chkpt` separately from the latest checkpoint. This ensures the peak policy is always recoverable regardless of subsequent training.

```python
if avg_reward > agent.best_avg_reward:
    agent.best_avg_reward = avg_reward
    agent.save_best_model(episode, metrics)
```

### 5.2 Early Stopping

**Addresses**: Continued training past the optimum.

For CartPole, if the 100-episode average reward exceeds 490 for 20 consecutive episodes, training halts. This prevents the model from entering the forgetting phase entirely.

| Environment | Threshold | Patience |
|-------------|-----------|----------|
| CartPole-v1 | AvgR ≥ 490 | 20 episodes |
| FrozenLake-v1 | AvgR ≥ 0.70 | 50 episodes |
| MountainCar-v0 | AvgR ≥ -110 | 20 episodes |
| Acrobot-v1 | AvgR ≥ -80 | 20 episodes |

### 5.3 Learning Rate Reduction

**Addresses**: Excessive weight updates after convergence.

When the average reward does not improve for 50 consecutive episodes, the learning rate is halved (minimum 1e-5). This progressively reduces the step size, making it harder for the model to drift away from good solutions.

```python
if episodes_since_improvement >= lr_reduce_patience:
    for pg in optimizer.param_groups:
        pg['lr'] = max(pg['lr'] * 0.5, 1e-5)
```

The reduction applies to all three optimizer groups (VQC, ANO, architecture search) simultaneously.

### 5.4 Slower Target Network Sync

**Addresses**: Feedback loop between degrading online and target networks.

`sync_every` is increased from 100 to 500 learn steps. This means the target network retains a "memory" of the good policy for 5× longer, providing stable Q-value targets even if the online network temporarily degrades.

| Setting | v1/v5 | v6 |
|---------|-------|-----|
| `sync_every` | 100 steps | 500 steps |
| Effective lag | ~5 episodes | ~25 episodes |

---

## 6. v6 Architecture Summary

v6 combines the proven v1 base with v5 innovations and anti-forgetting:

```
              v1 Base (validated)          v5 Innovations (from Atari)
              ──────────────────           ──────────────────────────
              Sigmoid [0,1] scaling        ANO: Learnable Hermitian observables
              MSELoss                      DiffQAS: Gate-level architecture search
              No PE                        2x2 factorial ablation
              No gradient clipping
                        │                              │
                        └──────────┬───────────────────┘
                                   │
                          v6 (this work)
                          ──────────────
                          + Best model checkpointing
                          + Early stopping
                          + LR reduction on plateau
                          + Slow target sync (500 steps)
```

### Ablation Conditions (same 2×2 design as v5)

| Condition | ANO | DiffQAS | Description |
|-----------|-----|---------|-------------|
| `baseline` | Off | Off | v1 architecture (fixed RY-CRX + PauliX/Y/Z) |
| `ano_only` | On | Off | Learnable Hermitian measurements, fixed circuit |
| `dqas_only` | Off | On | Architecture search, fixed Pauli measurements |
| `full` | On | On | Both innovations active |

Plus a **classical transformer baseline** with matched anti-forgetting mechanisms for fair comparison.

---

## 7. Experimental Setup

| Parameter | CartPole-v1 | FrozenLake-v1 |
|-----------|-------------|---------------|
| Episodes | 500 | 3,000 |
| Qubits | 8 | 8 |
| Ansatz layers | 2 | 2 |
| QSVT degree | 2 | 2 |
| Timesteps | 4 | 4 |
| Learning rate | 0.001 | 0.001 |
| Epsilon decay | 0.995/step | 0.995/step |
| Batch size | 64 | 64 |
| Memory size | 10,000 | 10,000 |
| sync_every | 500 | 500 |
| Early stop | AvgR ≥ 490, 20 eps | AvgR ≥ 0.70, 50 eps |
| LR reduce | patience=50, factor=0.5 | patience=50, factor=0.5 |

**Jobs submitted**: 49735467-49735477 (10 total: 4 quantum + 1 classical × 2 environments)

---

## 8. Expected Outcomes

Based on v1 results and the anti-forgetting mechanisms:

1. **CartPole baseline** should reach peak AvgR ~500 (matching v1), and early stopping should preserve it
2. **ANO/DiffQAS conditions** may improve sample efficiency (faster learning) or peak stability
3. **Classical baseline** provides the reference for quantum vs classical comparison
4. **FrozenLake** should show stable ~50% success rate (matching v1) without forgetting (stochastic environment naturally prevents overconfident overfitting)
5. **Best model checkpoints** guarantee that peak performance is always recoverable even if early stopping doesn't trigger

---

## 9. Key Takeaway

**Architecture choices must match task complexity.** The v3 upgrades (2π scaling, PE, Huber loss) are well-suited for high-dimensional Atari with CNN features, but they over-complicate the simple CartPole/FrozenLake mapping. v6 respects this by using the validated v1 base for SimpleRL while preserving the v5 research innovations (ANO, DiffQAS) and adding robust anti-forgetting mechanisms. This separation allows fair evaluation of whether ANO and DiffQAS provide genuine improvements on simple environments, independent of architectural confounds.
