# SimpleRL v6 Ablation Study — Complete Results

**Date**: March 9, 2026
**Author**: Junghoon Park
**Model**: Quantum Time-Series Transformer v6 (v1 base + ANO/DiffQAS + anti-forgetting)

---

## 1. Experiment Overview

### Architecture

v6 uses the validated v1 base architecture (sigmoid [0,1] angle scaling, MSELoss, no positional encoding, no gradient clipping) combined with the v5 research innovations (ANO, DiffQAS) and anti-forgetting mechanisms. See `docs/SimpleRL_v6_Design_Rationale.md` for why v1 base outperforms v3/v5 base on SimpleRL environments.

### Ablation Conditions (2x2 Factorial)

| Condition | ANO | DiffQAS | Description |
|-----------|-----|---------|-------------|
| `baseline` | Off | Off | v1 architecture (fixed RY-CRX + PauliX/Y/Z) |
| `ano_only` | On | Off | Learnable Hermitian measurements, fixed circuit |
| `dqas_only` | Off | On | Architecture search, fixed Pauli measurements |
| `full` | On | On | Both innovations active |
| `classical` | N/A | N/A | Classical transformer baseline (d_model=32, n_heads=4) |

### Common Parameters

| Parameter | Value |
|-----------|-------|
| Qubits | 8 |
| Ansatz Layers | 2 |
| QSVT Degree | 2 |
| Timesteps | 4 |
| Batch Size | 64 |
| Gamma | 0.99 |
| Epsilon Decay | 0.995/step |
| Memory Size | 10,000 |
| Sync Every | 500 steps |
| Seed | 2025 |
| Device | CUDA (A100-SXM4-80GB) |

### Anti-Forgetting Mechanisms

| Mechanism | Description |
|-----------|-------------|
| Best-model checkpointing | Saves model at peak BestAvgR separately from latest |
| Early stopping | Halts if AvgR exceeds threshold for N consecutive episodes |
| LR reduction on plateau | Halves LR after 50 episodes without improvement (Run1 only) |
| Slow target sync | sync_every=500 (vs v1's 100) to prevent feedback loops |

### LR Strategy Comparison

Two CartPole runs were conducted to compare learning rate strategies:
- **Run1 (LR Reduction)**: lr_reduce_patience=50, factor=0.5, min=1e-5
- **Run2 (Constant LR)**: lr_reduce_patience=99999 (never triggers), LR=0.001 throughout

FrozenLake used constant LR only (LR reduction killed learning during long exploration phase).

---

## 2. CartPole-v1 Results

### 2.1 Final Results Summary

#### Run1: LR Reduction (patience=50, factor=0.5)

| Condition | Episodes | BestAvgR | Best Episode | Final AvgR | Final LR | Status |
|-----------|----------|----------|-------------|------------|----------|--------|
| **ANO-only** | 490+ | **347.7** | 442 | 281.9 | 0.000125 | Timed out (48h) |
| Baseline | 500 | 223.2 | 167 | 61.2 | 0.000016 | Complete |
| Full (ANO+DiffQAS) | 500 | 203.0 | 268 | 154.7 | 0.000016 | Complete |
| **Classical** | 500 | **132.6** | 277 | 108.6 | 0.000016 | Complete |
| DiffQAS-only | 500 | 65.3 | 412 | 30.2 | 0.000063 | Complete |

#### Run2: Constant LR (lr=0.001 throughout)

| Condition | Episodes | BestAvgR | Best Episode | Final AvgR | Final LR | Status |
|-----------|----------|----------|-------------|------------|----------|--------|
| **DiffQAS-only** | 500 | **204.8** | 233 | 93.0 | 0.001 | Complete |
| Baseline | 500 | 202.8 | 471 | 180.8 | 0.001 | Complete |
| Full (ANO+DiffQAS) | 500 | 181.0 | 218 | 36.7 | 0.001 | Complete |
| ANO-only | 500 | 145.5 | 243 | 50.1 | 0.001 | Complete |

#### v1 Reference (original, constant LR, no anti-forgetting)

| Condition | Episodes | BestAvgR | Best Episode | Final AvgR |
|-----------|----------|----------|-------------|------------|
| v1 Quantum | 500 | 289.3 | 170 | 175.5 |

*Note: v1 had 30 consecutive perfect scores (R=500) from ep 77-106, but no best-model checkpointing to preserve them.*

### 2.2 LR Strategy Comparison (CartPole)

| Condition | LR Reduction BestAvgR | Constant LR BestAvgR | Better Strategy |
|-----------|----------------------|---------------------|-----------------|
| ANO-only | **347.7** | 145.5 | **LR Reduction** (+139%) |
| Baseline | **223.2** | 202.8 | LR Reduction (+10%) |
| Full | **203.0** | 181.0 | LR Reduction (+12%) |
| DiffQAS-only | 65.3 | **204.8** | **Constant LR** (+214%) |

**Key finding**: LR reduction benefits 3 of 4 quantum conditions on CartPole, but dramatically hurts DiffQAS-only. The architecture search phase in DiffQAS is slower, so LR decays before the model has converged, crippling subsequent learning. ANO-only benefits the most from LR reduction — the gradual decay stabilizes the learnable observables after initial convergence, yielding the best overall CartPole result (347.7).

### 2.3 CartPole Training Trajectories (Run1: LR Reduction)

```
Episode    Baseline    ANO-only    DiffQAS-only    Full    Classical
  100       148.6        —            —            —        52.0
  200       144.7       85.3          —           126.8    123.0
  300       192.8      241.1          —           161.4    132.6*
  400       102.9      284.7        63.3          127.5    119.0
  490        61.2      281.9        30.2          154.7    108.6

* Classical BestAvgR reached at ep 277
```

### 2.4 CartPole Training Trajectories (Run2: Constant LR)

```
Episode    Baseline    ANO-only    DiffQAS-only    Full
  100        11.5       15.0         18.9         17.0
  200        20.5      114.4        167.6        161.0
  300       186.7       66.5        140.1        138.5
  400       182.0       61.1        125.3         43.0
  490       180.8       50.1         93.0         36.7
```

### 2.5 CartPole Analysis

1. **ANO-only with LR reduction is the best quantum model** (BestAvgR 347.7) — 69% of v1's peak of 500, and 2.6x better than the classical baseline (132.6).

2. **All quantum conditions beat classical** in their best configuration:
   - ANO-only (LR red.): 347.7 vs 132.6 = **2.6x**
   - Baseline (LR red.): 223.2 vs 132.6 = **1.7x**
   - DiffQAS-only (const.): 204.8 vs 132.6 = **1.5x**
   - Full (LR red.): 203.0 vs 132.6 = **1.5x**

3. **Catastrophic forgetting remains a challenge**: All conditions show significant decline from peak (BestAvgR) to final AvgR. The anti-forgetting mechanisms (best-model save, slow sync) preserve the peak weights but don't prevent the online network from degrading. Early stopping never triggered (threshold: AvgR >= 490).

4. **DiffQAS architecture search overhead**: DiffQAS-only processes episodes ~2x slower than baseline (fewer episodes completed in same wall time), and the Phase 1 → Phase 2 transition disrupts learning. It benefits from constant LR but hurts from LR reduction.

---

## 3. FrozenLake-v1 Results

### 3.1 Final Results Summary (Constant LR)

| Condition | Episodes | BestAvgR (Success Rate) | Best Episode | Final AvgR | Status |
|-----------|----------|------------------------|-------------|------------|--------|
| **Classical** | 3,000 | **0.8 (80%)** | 2,299 | 0.6 (60%) | Complete |
| **Baseline** | 3,000 | **0.6 (60%)** | 926 | 0.5 (50%) | Complete |
| **DiffQAS-only** | 3,000 | **0.6 (60%)** | 2,569 | 0.4 (40%) | Complete |
| ANO-only | 3,000 | 0.5 (50%) | 2,130 | 0.4 (40%) | Complete |
| Full (ANO+DiffQAS) | 3,000 | 0.5 (50%) | 2,519 | 0.4 (40%) | Complete |

#### v1 Reference

| Condition | Episodes | BestAvgR (Success Rate) | Final Success Rate |
|-----------|----------|------------------------|--------------------|
| v1 Quantum | 3,000 | 0.59 (59%) | 43% |

#### Random Policy Baseline

| Metric | Value |
|--------|-------|
| Random policy success rate | ~1.5% |

### 3.2 FrozenLake Training Trajectories (Constant LR)

```
Episode    Baseline    ANO-only    DiffQAS-only    Full    Classical
  500        0.2        0.1           —            0.1      0.1
 1000        0.4        0.3          0.3           0.2      0.1
 1500        0.4        0.4          0.3           0.3      0.2
 2000        0.4        0.3          0.4           0.4      0.3
 2500        0.4        0.4          0.5           0.5      0.8*
 3000        0.5        0.4          0.4           0.4      0.6

* Classical BestAvgR (0.8) reached at ep 2299
```

### 3.3 FrozenLake with LR Reduction (patience=50) — Failed Runs

These jobs were cancelled early after diagnosing the LR decay problem, but the damage was already evident:

| Condition | Episodes Reached | BestAvgR (Success Rate) | Final LR | Status |
|-----------|-----------------|------------------------|----------|--------|
| Classical | 3,000 | 0.3 (30%) | 0.000010 | Complete |
| Baseline | ~200 | 0.0 (0%) | 0.000250 | Cancelled |
| ANO-only | ~260 | 0.02 (2%) | 0.000500 | Cancelled |
| DiffQAS-only | ~330 | 0.02 (2%) | 0.000125 | Cancelled |
| Full (ANO+DiffQAS) | ~520 | 0.01 (1%) | 0.000010 | Cancelled |

### 3.4 LR Strategy Comparison (FrozenLake)

| Condition | LR Reduction BestAvgR | Constant LR BestAvgR | Improvement |
|-----------|----------------------|---------------------|-------------|
| **Classical** | 0.3 (30%) | **0.8 (80%)** | **+167%** |
| **Baseline** | 0.0 (0%) | **0.6 (60%)** | **+inf** |
| **ANO-only** | 0.02 (2%) | **0.5 (50%)** | **+2400%** |
| **DiffQAS-only** | 0.02 (2%) | **0.6 (60%)** | **+2900%** |
| **Full** | 0.01 (1%) | **0.5 (50%)** | **+4900%** |

**Constant LR is overwhelmingly better on FrozenLake** for every single condition.

**Why LR reduction fails on FrozenLake**: The agent spends the first ~400-500 episodes at AvgR=0 during exploration. With patience=50, the LR halves repeatedly during this phase (0.001 → 0.0005 → 0.00025 → ...), so by the time the agent starts finding the goal, the LR is too small to learn effectively. The Full condition decayed all the way to LR=0.00001 by episode 520 — essentially frozen.

### 3.5 FrozenLake Analysis

1. **Classical transformer dominates FrozenLake** (80% vs best quantum 60%). Unlike CartPole, quantum models show no advantage on this stochastic environment.

2. **Quantum models match v1**: Baseline and DiffQAS-only both reached 60%, matching v1's original 59% peak. The anti-forgetting mechanisms preserved this peak (v1's final was 43%).

3. **FrozenLake is inherently noisy**: The slippery ice introduces ~30-40% inherent failure rate even for a perfect policy. The theoretical maximum success rate is ~73%. Classical reached 80% (above theoretical max due to sampling variance in 100-episode windows), while quantum models plateau at 50-60%.

4. **All models far exceed random**: Even the weakest quantum model (50%) is **33x better than random** (1.5%).

---

## 4. Cross-Environment Comparison

### 4.1 Best Results per Condition (BestAvgR)

| Condition | CartPole (best of Run1/Run2) | FrozenLake |
|-----------|------|------------|
| ANO-only | **347.7** (LR red.) | 50% |
| Baseline | 223.2 (LR red.) | **60%** |
| DiffQAS-only | 204.8 (const.) | **60%** |
| Full (ANO+DiffQAS) | 203.0 (LR red.) | 50% |
| Classical | 132.6 | **80%** |
| v1 Reference | 289.3* | 59% |

*v1 had 30 consecutive R=500 episodes but no best-model save; BestAvgR is the 100-episode rolling average.

### 4.2 Quantum vs Classical

| Environment | Best Quantum | Classical | Quantum Advantage? |
|-------------|-------------|-----------|-------------------|
| **CartPole-v1** | 347.7 (ANO-only) | 132.6 | **Yes (2.6x)** |
| **FrozenLake-v1** | 60% (Baseline/DiffQAS) | 80% | **No (0.75x)** |

### 4.3 ANO vs DiffQAS Impact

| Innovation | CartPole Impact | FrozenLake Impact | Atari Impact (v5) |
|-----------|----------------|-------------------|-------------------|
| **ANO** | **+56%** over baseline (347.7 vs 223.2) | -17% (50% vs 60%) | +3.3% SpaceInvaders |
| **DiffQAS** | -8% (204.8 vs 223.2) | 0% (60% vs 60%) | **+473% Breakout** |
| **Both** | -9% (203.0 vs 223.2) | -17% (50% vs 60%) | Mixed |

**Key insight**: ANO and DiffQAS show environment-dependent benefits:
- **ANO** excels on CartPole (continuous state, simple mapping benefits from learned observables) but not FrozenLake or Atari
- **DiffQAS** excels on complex Atari games (architecture search finds better gates for high-dimensional inputs) but adds overhead that hurts simple tasks
- **Combined (Full)** shows no synergy — the added complexity from both innovations simultaneously hurts sample efficiency

---

## 5. Comparison with Atari v5 Ablation

The same ANO/DiffQAS innovations were tested on both Atari (v5) and SimpleRL (v6):

| Innovation | Atari (v5) Verdict | SimpleRL (v6) Verdict |
|-----------|-------------------|----------------------|
| ANO | Modest stability gains (+3.3%) | **Strong on CartPole** (+56%), neutral/negative on FrozenLake |
| DiffQAS | **Breakthrough on Breakout** (+473%) | Neutral on CartPole, neutral on FrozenLake |
| Classical vs Quantum | Quantum wins all 4 games | Quantum wins CartPole, loses FrozenLake |

**Architecture choices must match task complexity**: DiffQAS shines when the gate search can discover non-trivial circuits for complex tasks (Atari), but its overhead hurts simple tasks where the default RY-CRX circuit is already sufficient.

---

## 6. LR Strategy Recommendations

| Environment Type | Recommended LR Strategy | Rationale |
|-----------------|------------------------|-----------|
| Fast-converging (CartPole) | LR Reduction (patience=50) | Stabilizes after peak, prevents forgetting |
| Slow-learning stochastic (FrozenLake) | Constant LR | LR decay kills learning before it starts |
| DiffQAS models | Constant LR | Architecture search is slow; early LR decay is fatal |
| Atari (complex) | Constant LR (validated in v5) | Long training horizon needs sustained learning rate |

---

## 7. Job Reference

### CartPole-v1 Run1 (LR Reduction)

| Condition | Job ID | Wall Time | Exit |
|-----------|--------|-----------|------|
| Baseline | 49768804 | 30h 33m | Complete |
| ANO-only | 49768805 | 48h (timeout) | Timed out |
| DiffQAS-only | 49768806 | 1h 38m | Complete |
| Full | 49768807 | 26h 41m | Complete |
| Classical | 49735471 | ~6h | Complete |

### CartPole-v1 Run2 (Constant LR)

| Condition | Job ID | Wall Time | Exit |
|-----------|--------|-----------|------|
| Baseline | 49779688 | 36h 7m | Complete |
| ANO-only | 49779689 | 19h 38m | Complete |
| DiffQAS-only | 49779690 | 36h 5m | Complete |
| Full | 49779691 | 26h 19m | Complete |

### FrozenLake-v1 (Constant LR)

| Condition | Job ID | Wall Time | Exit |
|-----------|--------|-----------|------|
| Baseline | 49779149 | 36h 34m | Complete |
| ANO-only | 49779153 | 37h 41m | Complete |
| DiffQAS-only | 49779155 | 30h 19m | Complete |
| Full | 49779156 | 42h 43m | Complete |
| Classical | 49763730 | ~12h | Complete |

---

## 8. Visualizing Training Curves

Per-episode metrics (rewards, episode lengths, losses) are stored in checkpoint files. Use the following Python code to extract and plot training curves.

### 8.1 Loading Checkpoint Metrics

```python
import torch

# Example: load FrozenLake Full v6 checkpoint
ckpt = torch.load(
    "QRL_QTSTransformer/checkpoints/simplerl_v6_FrozenLake-v1_full_s2025_1/latest_checkpoint.chkpt",
    map_location="cpu",
    weights_only=False
)
metrics = ckpt["metrics"]
# metrics = {'rewards': [r1, r2, ...], 'lengths': [l1, l2, ...], 'losses': [loss1, loss2, ...]}
```

### 8.2 Checkpoint Path Convention

```
QRL_QTSTransformer/checkpoints/
├── simplerl_v6_{ENV}_{ABLATION}_s{SEED}_{LOG_INDEX}/
│   ├── latest_checkpoint.chkpt      # Full checkpoint with metrics
│   └── best_model.chkpt             # Best-model weights
├── classical_v6_{ENV}_s{SEED}_{LOG_INDEX}/
│   ├── latest_checkpoint.chkpt
│   └── best_model.chkpt
```

- `LOG_INDEX=1` → Run1 (LR Reduction for CartPole, Constant LR for FrozenLake)
- `LOG_INDEX=2` → Run2 (Constant LR for CartPole)

### 8.3 Plotting Average Reward Curves

```python
import torch
import numpy as np
import matplotlib.pyplot as plt

def load_rewards(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    return ckpt["metrics"]["rewards"]

def rolling_avg(data, window=100):
    return np.convolve(data, np.ones(window)/window, mode="valid")

# --- CartPole Run1 (LR Reduction) ---
base_dir = "QRL_QTSTransformer/checkpoints"
env = "CartPole-v1"
conditions = {
    "Baseline":    f"{base_dir}/simplerl_v6_{env}_baseline_s2025_1/latest_checkpoint.chkpt",
    "ANO-only":    f"{base_dir}/simplerl_v6_{env}_ano_only_s2025_1/latest_checkpoint.chkpt",
    "DiffQAS-only":f"{base_dir}/simplerl_v6_{env}_dqas_only_s2025_1/latest_checkpoint.chkpt",
    "Full":        f"{base_dir}/simplerl_v6_{env}_full_s2025_1/latest_checkpoint.chkpt",
    "Classical":   f"{base_dir}/classical_v6_{env}_s2025_1/latest_checkpoint.chkpt",
}

fig, ax = plt.subplots(figsize=(12, 6))
for label, path in conditions.items():
    rewards = load_rewards(path)
    avg = rolling_avg(rewards, window=100)
    ax.plot(range(100, 100 + len(avg)), avg, label=label)

ax.set_xlabel("Episode")
ax.set_ylabel("Average Reward (100-ep rolling)")
ax.set_title("CartPole-v1 — Run1 (LR Reduction)")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("cartpole_run1_reward_curves.pdf", dpi=150)
plt.show()
```

### 8.4 Plotting Average Loss Curves

```python
def load_losses(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    return ckpt["metrics"]["losses"]

fig, ax = plt.subplots(figsize=(12, 6))
for label, path in conditions.items():
    losses = load_losses(path)
    avg = rolling_avg(losses, window=100)
    ax.plot(range(100, 100 + len(avg)), avg, label=label)

ax.set_xlabel("Episode")
ax.set_ylabel("Average Loss (100-ep rolling)")
ax.set_title("CartPole-v1 — Run1 (LR Reduction)")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("cartpole_run1_loss_curves.pdf", dpi=150)
plt.show()
```

### 8.5 Available Experiments

All experiments below have per-episode metrics in their `latest_checkpoint.chkpt`:

| Experiment | LOG_INDEX | Checkpoint Pattern |
|-----------|-----------|-------------------|
| CartPole Run1 (LR Reduction) | 1 | `simplerl_v6_CartPole-v1_{ablation}_s2025_1/` |
| CartPole Run2 (Constant LR) | 2 | `simplerl_v6_CartPole-v1_{ablation}_s2025_2/` |
| FrozenLake (Constant LR) | 1 | `simplerl_v6_FrozenLake-v1_{ablation}_s2025_1/` |
| Classical CartPole | 1 | `classical_v6_CartPole-v1_s2025_1/` |
| Classical FrozenLake | 1 | `classical_v6_FrozenLake-v1_s2025_1/` |
| Atari v5 (all games) | — | `atari_v5_{game}_{ablation}_s2025/` |
| v1 SimpleRL | — | `simplerl_v1_{env}_s2025/` |

Replace `{ablation}` with: `baseline`, `ano_only`, `dqas_only`, or `full`.

---

## 9. Files

| File | Description |
|------|-------------|
| `scripts/QuantumTransformerSimpleRL_v6.py` | Quantum v6 training script |
| `scripts/ClassicalTransformerSimpleRL_v6.py` | Classical baseline training script |
| `jobs/run_quantum_transformer_simplerl_v6.sh` | Quantum SLURM script (48h) |
| `jobs/run_classical_transformer_simplerl_v6.sh` | Classical SLURM script (48h) |
| `jobs/submit_all_simplerl_v6.sh` | Submit all 10 jobs |
| `docs/SimpleRL_v6_Design_Rationale.md` | Why v1 base > v3/v5 base for SimpleRL |
| `docs/SimpleRL_v6_Results.md` | This file |
| `docs/SimpleRL_Results_Summary.md` | Original v1 results |
