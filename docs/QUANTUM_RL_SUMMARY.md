# Quantum Time-Series Transformer for Reinforcement Learning
## Complete Implementation Summary

This document provides an overview of all quantum RL implementations created.

---

## 📦 Created Files

### 1. **Super Mario Bros Implementation**
- **`QuantumTransformerMario.py`** - Main training script for Super Mario Bros
- **`README_QuantumTransformerMario.md`** - Complete documentation
- **`run_quantum_transformer_mario.sh`** - SLURM batch script

### 2. **Simple Environments Implementation**
- **`QuantumTransformerSimpleRL.py`** - Training script for CartPole, FrozenLake, MountainCar, Acrobot
- **`README_QuantumTransformerSimpleRL.md`** - Complete documentation

### 3. **Existing Files (Reference)**
- **`ClassicalSuperMario.py`** - Classical baseline for comparison
- **`QTSTransformer.py`** - Original quantum transformer module

---

## 🚀 Quick Start Guide

### CartPole (Simplest - Start Here!)

```bash
# Activate environment
conda activate ./conda-envs/qml_eeg

# Train quantum agent on CartPole
python QuantumTransformerSimpleRL.py --env=CartPole-v1

# Expected: Solves in ~100-200 episodes
```

### FrozenLake (Discrete States)

```bash
python QuantumTransformerSimpleRL.py \
    --env=FrozenLake-v1 \
    --num-episodes=2000
```

### Super Mario Bros (Complex - Image-Based)

```bash
python QuantumTransformerMario.py \
    --n-qubits=6 \
    --n-layers=2 \
    --num-episodes=40000
```

---

## 🏗️ Architecture Comparison

### Simple Environments (CartPole, FrozenLake, etc.)

```
State Vector (4D for CartPole)
    ↓
State History Buffer (last 4 states)
    ↓
Quantum Transformer (no CNN needed)
    ↓
Q-values
```

**Characteristics**:
- ✅ No CNN required (states are already low-dimensional)
- ✅ Fast training (seconds to minutes)
- ✅ Good for testing quantum advantage
- ✅ Suitable for 3-6 qubits

### Super Mario Bros

```
4 Stacked Game Frames (4×84×84)
    ↓
CNN Feature Extractor
    ↓
Feature Vectors (4 timesteps × feature_dim)
    ↓
Quantum Transformer
    ↓
Q-values (2 actions)
```

**Characteristics**:
- ✅ CNN for spatial feature extraction
- ⚠️ Slower training (hours to days)
- ✅ Tests scalability to high-dimensional inputs
- ✅ Suitable for 6-8 qubits

---

## 📊 Recommended Configurations

### For Quick Experiments (Testing)

| Environment | Qubits | Layers | Episodes | Time (CPU) |
|------------|--------|--------|----------|------------|
| CartPole | 3 | 1 | 300 | ~3 min |
| FrozenLake | 3 | 1 | 1000 | ~2 min |
| MountainCar | 4 | 1 | 1000 | ~5 min |

### For Publication Results (Best Performance)

| Environment | Qubits | Layers | Episodes | Time (GPU) |
|------------|--------|--------|----------|------------|
| CartPole | 6 | 3 | 500 | ~5 min |
| FrozenLake | 6 | 3 | 2000 | ~8 min |
| MountainCar | 6 | 3 | 2000 | ~12 min |
| Acrobot | 8 | 3 | 3000 | ~20 min |
| Super Mario | 6 | 2 | 40000 | ~24 hours |

---

## 🎯 Progressive Learning Path

We recommend training on environments in this order:

### Level 1: CartPole ⭐ (Easiest)
**Why start here**:
- Fast to train (5-10 minutes)
- Clear learning signal (rewards increase quickly)
- Easy to debug
- Verifies quantum implementation works

**Command**:
```bash
python QuantumTransformerSimpleRL.py --env=CartPole-v1 --n-qubits=4
```

**Success criteria**: Average reward > 195 in < 300 episodes

---

### Level 2: FrozenLake ⭐⭐ (Discrete States)
**Why second**:
- Tests handling of discrete (one-hot encoded) states
- Introduces stochasticity (slippery ice)
- Still relatively fast

**Command**:
```bash
python QuantumTransformerSimpleRL.py --env=FrozenLake-v1 --num-episodes=2000
```

**Success criteria**: Success rate > 70%

---

### Level 3: MountainCar ⭐⭐⭐ (Sparse Rewards)
**Why third**:
- Tests sparse reward handling
- Requires careful exploration
- More challenging dynamics

**Command**:
```bash
python QuantumTransformerSimpleRL.py --env=MountainCar-v0 --num-episodes=2000
```

**Success criteria**: Consistently reaches goal in < 110 steps

---

### Level 4: Acrobot ⭐⭐⭐⭐ (Complex Dynamics)
**Why fourth**:
- Higher dimensional state (6D)
- Complex physics
- Benefits from larger quantum circuits

**Command**:
```bash
python QuantumTransformerSimpleRL.py \
    --env=Acrobot-v1 \
    --n-qubits=6 \
    --n-layers=3 \
    --num-episodes=3000
```

**Success criteria**: Reaches target in < 100 steps

---

### Level 5: Super Mario Bros ⭐⭐⭐⭐⭐ (Full Challenge)
**Why last**:
- Requires CNN + Quantum hybrid
- High-dimensional visual input
- Long training time
- Real-world application complexity

**Command**:
```bash
python QuantumTransformerMario.py \
    --n-qubits=6 \
    --n-layers=2 \
    --num-episodes=40000
```

**Success criteria**: Consistently reaches further levels

---

## 🔬 Research Experiments

### Experiment 1: Quantum Circuit Scaling

Test how quantum circuit size affects performance:

```bash
for qubits in 3 4 6 8; do
  python QuantumTransformerSimpleRL.py \
    --env=CartPole-v1 \
    --n-qubits=$qubits \
    --log-index=qubits_${qubits}
done
```

**Research question**: Does increasing qubits improve sample efficiency or final performance?

---

### Experiment 2: Temporal Context Window

Test importance of state history:

```bash
for timesteps in 1 2 4 8; do
  python QuantumTransformerSimpleRL.py \
    --env=CartPole-v1 \
    --n-timesteps=$timesteps \
    --log-index=timesteps_${timesteps}
done
```

**Research question**: How much temporal context is needed for sequential decision-making?

---

### Experiment 3: QSVT Degree

Test polynomial degree of quantum transformation:

```bash
for degree in 1 2 3 4; do
  python QuantumTransformerSimpleRL.py \
    --env=CartPole-v1 \
    --degree=$degree \
    --log-index=degree_${degree}
done
```

**Research question**: Does higher QSVT degree provide more expressive transformations?

---

### Experiment 4: Environment Complexity

Test same quantum architecture across multiple environments:

```bash
QUANTUM_CONFIG="--n-qubits=4 --n-layers=2 --degree=2"

python QuantumTransformerSimpleRL.py --env=CartPole-v1 $QUANTUM_CONFIG
python QuantumTransformerSimpleRL.py --env=FrozenLake-v1 $QUANTUM_CONFIG --num-episodes=2000
python QuantumTransformerSimpleRL.py --env=MountainCar-v0 $QUANTUM_CONFIG --num-episodes=2000
python QuantumTransformerSimpleRL.py --env=Acrobot-v1 $QUANTUM_CONFIG --num-episodes=3000
```

**Research question**: How does a fixed quantum architecture generalize across different RL tasks?

---

### Experiment 5: Quantum vs Classical

Compare quantum transformer with classical baseline:

```bash
# Quantum
python QuantumTransformerSimpleRL.py \
    --env=CartPole-v1 \
    --n-qubits=6 \
    --log-index=quantum

# Classical (classical baseline needed)
# python ClassicalDQNSimpleRL.py --env=CartPole-v1 --log-index=classical
```

**Research question**: Does quantum transformer offer advantages over classical DQN?

---

## 📈 Expected Results Summary

### CartPole-v1
- **Training time**: 5-10 minutes (CPU)
- **Episodes to solve**: 100-300
- **Final performance**: Average reward ~200 (max 500)
- **Learning curve**: Rapid improvement, plateaus at ~200

### FrozenLake-v1
- **Training time**: 3-8 minutes (CPU)
- **Episodes to solve**: 500-1500
- **Final performance**: ~70% success rate
- **Learning curve**: Slow initial progress, then steady improvement

### MountainCar-v0
- **Training time**: 10-20 minutes (CPU)
- **Episodes to solve**: 1000-2000
- **Final performance**: Reaches goal in ~110 steps
- **Learning curve**: Flat initially (sparse rewards), then sudden improvement

### Acrobot-v1
- **Training time**: 20-40 minutes (CPU)
- **Episodes to solve**: 2000-3000
- **Final performance**: Reaches target in ~100 steps
- **Learning curve**: Gradual improvement with occasional plateaus

### Super Mario Bros
- **Training time**: 12-48 hours (GPU)
- **Episodes to solve**: Highly variable (10,000-40,000)
- **Final performance**: Reaches further in level 1-1
- **Learning curve**: Very slow initial progress, then gradual improvement

---

## 🐛 Common Issues & Solutions

### Issue 1: "Module 'gym' has no attribute 'make'"
**Solution**: Update gym
```bash
pip install --upgrade gym
```

### Issue 2: Agent doesn't learn (flat reward curve)
**Solution**: Adjust exploration
```bash
python QuantumTransformerSimpleRL.py \
    --env=CartPole-v1 \
    --exploration-rate-decay=0.99
```

### Issue 3: Training too slow
**Solution**: Use smaller quantum circuit
```bash
python QuantumTransformerSimpleRL.py \
    --env=CartPole-v1 \
    --n-qubits=3 \
    --n-layers=1
```

### Issue 4: CUDA out of memory (Super Mario)
**Solution**: Reduce batch size or use CPU
```bash
python QuantumTransformerMario.py --batch-size=16 --device=cpu
```

### Issue 5: "AttributeError: 'NoneType' object has no attribute 'shape'"
**Solution**: Ensure environment returns correct state format
- Check that `env.reset()` returns a valid state
- For FrozenLake, ensure state is an integer (0-15)

---

## 📚 File Structure Overview

```
/pscratch/sd/j/junghoon/
├── QuantumTransformerSimpleRL.py          # Simple envs (CartPole, FrozenLake, etc.)
├── README_QuantumTransformerSimpleRL.md   # Documentation for simple envs
├── QuantumTransformerMario.py             # Super Mario Bros
├── README_QuantumTransformerMario.md      # Documentation for Mario
├── run_quantum_transformer_mario.sh       # SLURM script for Mario
├── QUANTUM_RL_SUMMARY.md                  # This file
├── QTSTransformer.py                      # Original quantum transformer
├── ClassicalSuperMario.py                 # Classical baseline
└── checkpoints/
    ├── SimpleRLCheckpoints/               # Simple env results
    │   ├── QTransformer_CartPolev1_Q4_L2_D2_Run1/
    │   ├── QTransformer_FrozenLakev1_Q4_L2_D2_Run1/
    │   └── ...
    └── SuperMarioCheckpoints/             # Mario results
        ├── QTransformerMario_Q6_L2_D2_Run1/
        └── ...
```

---

## 🎓 Educational Use

### For Teaching Quantum Machine Learning

**Lesson 1**: Introduction to Quantum RL
- **Environment**: CartPole-v1
- **Focus**: Basic quantum circuits, state encoding
- **Duration**: 1 hour

**Lesson 2**: Handling Different State Spaces
- **Environment**: FrozenLake-v1 (discrete) vs CartPole (continuous)
- **Focus**: One-hot encoding, state preprocessing
- **Duration**: 1.5 hours

**Lesson 3**: Temporal Processing with QSVT
- **Environment**: MountainCar-v0
- **Focus**: Quantum Singular Value Transformation, temporal context
- **Duration**: 2 hours

**Lesson 4**: Scaling to Complex Tasks
- **Environment**: Super Mario Bros
- **Focus**: Hybrid quantum-classical architectures, CNN integration
- **Duration**: 3 hours

---

## 📊 Benchmarking Protocol

To properly benchmark quantum RL agents:

1. **Multiple seeds**: Run 5+ times with different seeds
```bash
for seed in 2024 2025 2026 2027 2028; do
  python QuantumTransformerSimpleRL.py \
    --env=CartPole-v1 \
    --seed=$seed \
    --log-index=seed_${seed}
done
```

2. **Statistical analysis**: Compare mean and std across seeds

3. **Classical baseline**: Always compare with classical DQN

4. **Report metrics**:
   - Sample efficiency (episodes to reach threshold)
   - Final performance (average reward over last 100 episodes)
   - Training time
   - Number of parameters

---

## 🚀 Next Steps

### Immediate Next Steps
1. ✅ Test on CartPole to verify installation
2. ✅ Run experiments on FrozenLake and MountainCar
3. ✅ Compare different qubit configurations
4. ✅ Train on Super Mario for full challenge

### Future Enhancements
- [ ] Add Atari games support (Pong, Breakout, Space Invaders)
- [ ] Implement quantum actor-critic (A3C/PPO)
- [ ] Add multi-environment parallel training
- [ ] Implement quantum replay buffer
- [ ] Add visualization of quantum states
- [ ] Create comparison scripts for quantum vs classical

---

## 📖 References

### Quantum Computing
- PennyLane documentation: https://pennylane.ai
- Quantum machine learning: https://arxiv.org/abs/2101.11020

### Reinforcement Learning
- OpenAI Gym: https://gym.openai.com
- Deep Q-Network (DQN): https://arxiv.org/abs/1312.5602
- Double DQN: https://arxiv.org/abs/1509.06461

### Quantum RL
- Quantum reinforcement learning: https://arxiv.org/abs/1906.08482
- Variational quantum algorithms: https://arxiv.org/abs/2012.09265

---

## 💡 Tips for Success

1. **Start simple**: Begin with CartPole before moving to complex environments
2. **Monitor training**: Watch the console output and training curves
3. **Use checkpoints**: Always enable `--resume` for long training runs
4. **Experiment systematically**: Change one parameter at a time
5. **Compare with classical**: Establish classical baselines for fair comparison
6. **Be patient**: Quantum simulation is slow, especially with >6 qubits
7. **Document everything**: Save hyperparameters and results for reproducibility

---

**Good luck with your quantum reinforcement learning experiments! 🎮⚛️🤖**
