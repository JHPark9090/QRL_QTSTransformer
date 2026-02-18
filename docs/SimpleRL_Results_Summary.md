# Quantum Time-Series Transformer - SimpleRL Results Summary

**Date**: February 6, 2026
**Model**: Quantum Time-Series Transformer with QSVT
**Environments**: CartPole-v1, FrozenLake-v1

---

## Configuration

| Parameter | Value |
|-----------|-------|
| Qubits | 8 |
| Ansatz Layers | 2 |
| QSVT Degree | 2 |
| Timesteps | 4 |
| Batch Size | 64 |
| Learning Rate | 0.001 |
| Gamma | 0.99 |
| Seed | 2025 |
| Device | CUDA (A100-SXM4-80GB) |

---

## CartPole-v1 Results

### Training Summary

| Metric | Value |
|--------|-------|
| Total Episodes | 500 |
| Total Steps | 81,975 |
| Peak Success | 30 consecutive perfect scores (Ep 77-106) |
| Best Avg Reward (last 100) | 289.3 (Ep 170) |
| Final Avg Reward | 175.5 |

### Training Phases

| Phase | Episodes | Avg Reward | Description |
|-------|----------|------------|-------------|
| Learning | 0-70 | 14 → 54 | Initial exploration and learning |
| **Peak Performance** | 77-106 | **500** | 30 consecutive perfect scores |
| Gradual Decline | 107-200 | 500 → 226 | Performance starting to degrade |
| Collapse | 210-340 | 189 → 78 | Catastrophic forgetting |
| Recovery | 350-450 | 80 → 216 | Partial recovery |
| Final | 460-500 | 216 → 175 | Stabilized but below peak |

### Key Episodes (Individual Rewards)

```
Ep  77-106: 500, 500, 500, ... (30 consecutive perfect scores)
Ep 107: 211 (first drop)
Ep 115: 159
Ep 120: 157
Ep 121: 63  (collapse begins)
```

### Performance at Different Stopping Points

| Stop At | Last 10 Avg | Last 20 Avg | Recommendation |
|---------|-------------|-------------|----------------|
| Ep 100 | 500.0 | 500.0 | Optimal |
| Ep 106 | 500.0 | 500.0 | Optimal |
| Ep 110 | 471.1 | 485.6 | Good |
| Ep 120 | 396.5 | 433.8 | Acceptable |
| Ep 130 | 112.8 | 254.7 | Too late |

### Analysis

**Catastrophic Forgetting Observed**: The model achieved perfect performance (500/500) for 30 consecutive episodes (77-106), but continued training caused the model to "unlearn" its optimal policy.

**Loss Explosion**: Loss increased from ~7 (Ep 130) to ~170 (Ep 490), indicating training instability.

**Recommendation**: Implement early stopping when avg reward > 490 for 20 consecutive episodes.

---

## FrozenLake-v1 Results

### Training Summary

| Metric | Value |
|--------|-------|
| Total Episodes | 3,000 |
| Environment | Slippery mode (default) |
| Peak Success Rate | 59% (Ep 2664) |
| Final Success Rate | 43% |
| Random Policy Baseline | ~1.5% |

### Training Phases

| Phase | Episodes | Success Rate | Description |
|-------|----------|--------------|-------------|
| Random Exploration | 0-400 | 0-9% | Learning basic navigation |
| Initial Learning | 500-700 | 27-33% | Starting to find the goal |
| Improvement | 800-1700 | 25-47% | Gradual improvement |
| Stable Performance | 1700-3000 | 40-52% | Fluctuating around 45% |

### Success Rate Over Training

```
Ep  500: 30.0%
Ep 1000: 25.0%
Ep 1500: 34.0%
Ep 2000: 42.0%
Ep 2500: 41.0%
Ep 2664: 59.0% (Peak)
Ep 3000: 43.0%
```

### Analysis

**Stable Training**: Unlike CartPole, FrozenLake showed stable training without catastrophic forgetting.

**Stochastic Environment**: The slippery ice means even a perfect policy has inherent failure rate (~30-40%), which prevented overconfident overfitting.

**Improvement over Random**:
- Random policy: ~1.5% success rate
- Our model peak: 59% success rate (**40x improvement**)
- Final model: 43% success rate (**29x improvement**)

**Loss Stability**: Loss remained stable at ~0.009-0.010 throughout training.

---

## Comparison

| Aspect | CartPole-v1 | FrozenLake-v1 |
|--------|-------------|---------------|
| Environment Type | Deterministic | Stochastic (slippery) |
| Peak Performance | 100% (500/500) | 59% success |
| Catastrophic Forgetting | Yes | No |
| Training Stability | Unstable | Stable |
| Loss Behavior | Exploded (7 → 170) | Stable (~0.009) |
| Early Stopping Needed | Yes (Ep ~100) | No |
| Final vs Peak | 35% of peak | 73% of peak |

---

## Recommendations for Future Runs

### CartPole
1. **Implement early stopping**: Stop when avg reward > 490 for 20 consecutive episodes
2. **Reduce learning rate**: Consider lr=0.0005 or lr=0.0001 for more stable training
3. **Add gradient clipping**: Prevent loss explosion
4. **Save best model**: Keep checkpoint of peak performance separately

### FrozenLake
1. **Current settings are appropriate**: No major changes needed
2. **Consider longer training**: May continue to improve slowly
3. **Try non-slippery mode**: `is_slippery=False` for deterministic baseline

---

## Checkpoint Locations

```
CartPole:   checkpoints/QTransformer_CartPole1_Q8_L2_D2_Run1/
FrozenLake: checkpoints/QTransformer_FrozenLake1_Q8_L2_D2_Run1/
```

Each checkpoint contains:
- `latest_checkpoint.chkpt`: Model weights, optimizer state, metrics
- `training_curves.png`: Visualization of training progress

---

## Files

- Training script: `QuantumTransformerSimpleRL.py`
- SLURM script: `run_quantum_transformer_simplerl.sh`
- This summary: `SimpleRL_Results_Summary.md`
