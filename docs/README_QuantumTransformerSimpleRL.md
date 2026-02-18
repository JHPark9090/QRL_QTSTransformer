# Quantum Time-Series Transformer for Simple RL Environments

This README provides comprehensive instructions for training quantum transformer agents on simple reinforcement learning environments: **CartPole**, **FrozenLake**, **MountainCar**, and **Acrobot**.

## Overview

**QuantumTransformerSimpleRL.py** implements a quantum time-series transformer for low-dimensional RL tasks. Unlike the Super Mario implementation which requires CNN feature extraction, these environments have simple state vectors that can be directly processed by the quantum transformer.

### Key Features

- ✅ **Unified Framework**: Single script handles multiple classic RL environments
- ✅ **No CNN Required**: States are directly fed to quantum transformer
- ✅ **State History Buffer**: Maintains temporal context for sequential decision-making
- ✅ **Double DQN**: Target network + experience replay for stable learning
- ✅ **Automatic State Preprocessing**: Handles both discrete and continuous states
- ✅ **Lightweight**: Fast training due to small state spaces

## Supported Environments

| Environment | State Space | Action Space | Difficulty | Typical Episodes to Solve |
|------------|-------------|--------------|------------|---------------------------|
| **CartPole-v1** | 4D continuous | 2 discrete | Easy | 100-300 |
| **FrozenLake-v1** | 16D discrete | 4 discrete | Easy-Medium | 500-1000 |
| **MountainCar-v0** | 2D continuous | 3 discrete | Medium | 1000-2000 |
| **Acrobot-v1** | 6D continuous | 3 discrete | Medium-Hard | 1000-3000 |

### Environment Details

#### CartPole-v1
- **Goal**: Balance a pole on a moving cart
- **State**: [cart position, cart velocity, pole angle, pole angular velocity]
- **Actions**: Push cart left (0) or right (1)
- **Reward**: +1 for each timestep pole remains upright
- **Success**: Average reward ≥ 195 over 100 episodes

#### FrozenLake-v1
- **Goal**: Navigate from start to goal on a frozen lake without falling in holes
- **State**: Current position (0-15) on 4×4 grid
- **Actions**: Move left (0), down (1), right (2), up (3)
- **Reward**: +1 for reaching goal, 0 otherwise
- **Success**: Average reward ≥ 0.70 over 100 episodes

#### MountainCar-v0
- **Goal**: Drive an underpowered car up a steep mountain
- **State**: [car position, car velocity]
- **Actions**: Push left (0), no push (1), push right (2)
- **Reward**: -1 for each timestep (penalizes time to goal)
- **Success**: Reaching the flag in < 110 steps

#### Acrobot-v1
- **Goal**: Swing a two-link robot arm to reach a target height
- **State**: [cos(θ₁), sin(θ₁), cos(θ₂), sin(θ₂), θ̇₁, θ̇₂]
- **Actions**: Apply torque to joint (+1, 0, -1)
- **Reward**: -1 for each timestep (penalizes time to goal)
- **Success**: Reaching target height in < 100 steps

## Architecture

```
State Vector (e.g., CartPole: [x, ẋ, θ, θ̇])
    ↓
State History Buffer (n_timesteps previous states)
    ↓
Shape: (n_timesteps, state_dim)
    ↓
Quantum Time-Series Transformer
    ├─ Feature Projection (state_dim → n_rots)
    ├─ Quantum Circuit (sim14 ansatz with QSVT)
    ├─ Multi-observable Measurement (X, Y, Z on all qubits)
    └─ Output Layer (3*n_qubits → action_dim)
    ↓
Q-values for all actions
    ↓
Epsilon-Greedy Action Selection
```

## Installation & Setup

### Step 1: Activate Environment

```bash
conda activate ./conda-envs/qml_eeg
```

### Step 2: Verify Dependencies

```bash
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import pennylane as qml; print('PennyLane:', qml.__version__)"
python -c "import gym; print('Gym:', gym.__version__)"
```

### Step 3: Install Missing Packages (if needed)

```bash
pip install gym torch pennylane matplotlib numpy
```

## How to Run

### Quick Start Examples

#### Train on CartPole (Easiest)

```bash
python QuantumTransformerSimpleRL.py --env=CartPole-v1
```

#### Train on FrozenLake

```bash
python QuantumTransformerSimpleRL.py --env=FrozenLake-v1 --num-episodes=2000
```

#### Train on MountainCar

```bash
python QuantumTransformerSimpleRL.py --env=MountainCar-v0 --num-episodes=2000
```

#### Train on Acrobot

```bash
python QuantumTransformerSimpleRL.py --env=Acrobot-v1 --num-episodes=3000
```

## Detailed Usage

### 1. Basic Training with Default Parameters

Run with defaults (4 qubits, 2 layers, degree 2):

```bash
python QuantumTransformerSimpleRL.py --env=CartPole-v1
```

**Default parameters**:
- Qubits: 4
- Ansatz layers: 2
- QSVT degree: 2
- Timesteps: 4
- Batch size: 64
- Learning rate: 0.001
- Episodes: 1000

### 2. Custom Quantum Parameters

Experiment with different quantum circuit configurations:

```bash
# Smaller quantum circuit (faster, less expressive)
python QuantumTransformerSimpleRL.py \
    --env=CartPole-v1 \
    --n-qubits=3 \
    --n-layers=1 \
    --degree=1

# Larger quantum circuit (slower, more expressive)
python QuantumTransformerSimpleRL.py \
    --env=CartPole-v1 \
    --n-qubits=6 \
    --n-layers=3 \
    --degree=3

# Adjusting temporal context window
python QuantumTransformerSimpleRL.py \
    --env=CartPole-v1 \
    --n-timesteps=8  # Use last 8 states instead of 4
```

### 3. Custom Training Parameters

Tune RL hyperparameters:

```bash
python QuantumTransformerSimpleRL.py \
    --env=CartPole-v1 \
    --batch-size=128 \
    --lr=0.0005 \
    --gamma=0.95 \
    --exploration-rate-start=1.0 \
    --exploration-rate-decay=0.99 \
    --exploration-rate-min=0.05 \
    --num-episodes=500
```

**Parameter descriptions**:
- `--batch-size`: Number of experiences to sample per training step
- `--lr`: Learning rate for Adam optimizer
- `--gamma`: Discount factor (how much to value future rewards)
- `--exploration-rate-start`: Initial epsilon for epsilon-greedy
- `--exploration-rate-decay`: Multiplicative decay per episode
- `--exploration-rate-min`: Minimum epsilon value
- `--num-episodes`: Total training episodes

### 4. Resume Training

If training was interrupted, resume from checkpoint:

```bash
python QuantumTransformerSimpleRL.py --env=CartPole-v1 --resume
```

### 5. Multiple Experimental Runs

Run multiple experiments with different configurations:

```bash
# Small quantum circuit
python QuantumTransformerSimpleRL.py \
    --env=CartPole-v1 \
    --n-qubits=3 \
    --log-index=1

# Medium quantum circuit
python QuantumTransformerSimpleRL.py \
    --env=CartPole-v1 \
    --n-qubits=4 \
    --log-index=2

# Large quantum circuit
python QuantumTransformerSimpleRL.py \
    --env=CartPole-v1 \
    --n-qubits=6 \
    --log-index=3
```

### 6. Rendering Environment (Visualization)

Watch the agent play in real-time (slows down training):

```bash
python QuantumTransformerSimpleRL.py \
    --env=CartPole-v1 \
    --render \
    --num-episodes=50
```

## Environment-Specific Recommendations

### CartPole-v1 (Easiest to Train)

**Recommended configuration**:
```bash
python QuantumTransformerSimpleRL.py \
    --env=CartPole-v1 \
    --n-qubits=4 \
    --n-layers=2 \
    --degree=2 \
    --batch-size=64 \
    --lr=0.001 \
    --num-episodes=300
```

**Expected results**: Solves in 100-200 episodes (avg reward > 195)

### FrozenLake-v1 (Stochastic Environment)

**Recommended configuration**:
```bash
python QuantumTransformerSimpleRL.py \
    --env=FrozenLake-v1 \
    --n-qubits=4 \
    --n-layers=2 \
    --degree=2 \
    --batch-size=64 \
    --lr=0.001 \
    --gamma=0.99 \
    --exploration-rate-decay=0.995 \
    --num-episodes=2000
```

**Expected results**: Reaches goal ~70% of the time after 1000-2000 episodes

**Note**: FrozenLake is slippery (stochastic transitions), so perfect performance is impossible.

### MountainCar-v0 (Sparse Rewards)

**Recommended configuration** (needs more exploration):
```bash
python QuantumTransformerSimpleRL.py \
    --env=MountainCar-v0 \
    --n-qubits=4 \
    --n-layers=2 \
    --degree=2 \
    --batch-size=128 \
    --lr=0.0005 \
    --gamma=0.99 \
    --exploration-rate-decay=0.9995 \
    --exploration-rate-min=0.01 \
    --num-episodes=2000
```

**Expected results**: Solves in 1000-2000 episodes (reaches flag consistently)

**Note**: MountainCar has sparse rewards (only get reward at goal), so needs patient training.

### Acrobot-v1 (Complex Dynamics)

**Recommended configuration**:
```bash
python QuantumTransformerSimpleRL.py \
    --env=Acrobot-v1 \
    --n-qubits=6 \
    --n-layers=3 \
    --degree=2 \
    --batch-size=128 \
    --lr=0.0005 \
    --gamma=0.99 \
    --num-episodes=3000 \
    --n-timesteps=8
```

**Expected results**: Solves in 2000-3000 episodes (reaches target height)

**Note**: Acrobot has complex physics, so benefits from larger quantum circuits and longer temporal context.

## Output Files

Training generates outputs in `SimpleRLCheckpoints/QTransformer_{env}_Q{n_qubits}_L{n_layers}_D{degree}_Run{log_index}/`:

### Files Created

- **`latest_checkpoint.chkpt`**: Model weights, optimizer state, metrics, RNG states
- **`training_curves.png`**: Three plots showing:
  - Episode rewards over time (with moving average)
  - Episode lengths over time (with moving average)
  - Training loss over time (with moving average)

### Example Directory Structure

```
SimpleRLCheckpoints/
├── QTransformer_CartPolev1_Q4_L2_D2_Run1/
│   ├── latest_checkpoint.chkpt
│   └── training_curves.png
├── QTransformer_FrozenLakev1_Q4_L2_D2_Run1/
│   ├── latest_checkpoint.chkpt
│   └── training_curves.png
└── QTransformer_MountainCarv0_Q4_L2_D2_Run1/
    ├── latest_checkpoint.chkpt
    └── training_curves.png
```

## Monitoring Training

### Console Output

The script prints progress every 10 episodes:

```
Ep  100 | Reward:  195.0 | Avg Reward:  123.4 | Avg Length: 195.0 | Loss: 0.0234 | ε: 0.543
Ep  110 | Reward:  200.0 | Avg Reward:  145.2 | Avg Length: 200.0 | Loss: 0.0189 | ε: 0.487
```

**Metrics explained**:
- **Ep**: Episode number
- **Reward**: Reward for this episode
- **Avg Reward**: Average reward over last 100 episodes
- **Avg Length**: Average episode length
- **Loss**: Average TD loss for this episode
- **ε (epsilon)**: Current exploration rate

### Training Curves

View the generated plot to see:
- **Rewards**: Should increase over time (higher is better)
- **Lengths**: Episode lengths (varies by environment)
- **Loss**: TD loss should decrease and stabilize

## Expected Training Behavior

### CartPole-v1

**Phase 1 (Episodes 0-50)**: Random exploration
- Agent balances pole for ~20-50 timesteps
- High exploration (ε ≈ 1.0 → 0.6)

**Phase 2 (Episodes 50-150)**: Learning
- Performance improves rapidly
- Agent learns basic balancing strategy
- Rewards increase to ~100-150

**Phase 3 (Episodes 150-300)**: Mastery
- Agent consistently balances pole for 200+ timesteps
- Exploration decreases (ε ≈ 0.1)
- **Success**: Average reward > 195

### FrozenLake-v1

**Phase 1 (Episodes 0-500)**: Exploration
- Agent explores the grid randomly
- Occasionally reaches goal by chance
- Success rate ~10-20%

**Phase 2 (Episodes 500-1500)**: Learning optimal path
- Agent learns which actions lead to goal
- Success rate increases to ~40-60%
- Loss decreases

**Phase 3 (Episodes 1500-2000)**: Convergence
- Agent finds near-optimal policy
- **Success**: ~70% success rate (can't be 100% due to slippery ice)

### MountainCar-v0

**Phase 1 (Episodes 0-500)**: Sparse rewards
- Agent struggles to reach goal
- Most episodes timeout at 200 steps
- Reward consistently -200

**Phase 2 (Episodes 500-1500)**: Discovery
- Agent occasionally reaches goal
- Learns to build momentum by rocking back and forth
- Reward improves to -150

**Phase 3 (Episodes 1500-2000)**: Optimization
- Agent consistently reaches goal
- **Success**: Reward > -110 (reaches in ~110 steps)

### Acrobot-v1

**Phase 1 (Episodes 0-1000)**: Exploration
- Agent swings randomly
- Rarely reaches target height
- Reward consistently -500

**Phase 2 (Episodes 1000-2500)**: Learning swing strategy
- Agent learns coordinated swinging
- Success rate increases
- Reward improves to -200

**Phase 3 (Episodes 2500-3000)**: Mastery
- Agent efficiently reaches target
- **Success**: Reward > -100 (reaches in ~100 steps)

## Performance Benchmarks

Training time estimates (per 1000 episodes):

| Environment | Qubits | CPU (32 cores) | GPU (A100) |
|------------|--------|----------------|------------|
| CartPole | 4 | ~5 min | ~2 min |
| CartPole | 6 | ~10 min | ~4 min |
| FrozenLake | 4 | ~3 min | ~1 min |
| MountainCar | 4 | ~8 min | ~3 min |
| Acrobot | 6 | ~15 min | ~6 min |

*Note: Times vary based on episode length and quantum circuit complexity.*

## Troubleshooting

### Issue: Agent Not Learning (Flat Reward Curve)

**Solutions**:

1. **Increase exploration**:
```bash
python QuantumTransformerSimpleRL.py \
    --env=CartPole-v1 \
    --exploration-rate-decay=0.99  # Slower decay
```

2. **Adjust learning rate**:
```bash
python QuantumTransformerSimpleRL.py \
    --env=CartPole-v1 \
    --lr=0.0005  # Try different learning rates
```

3. **Increase network capacity**:
```bash
python QuantumTransformerSimpleRL.py \
    --env=CartPole-v1 \
    --n-qubits=6 \
    --n-layers=3
```

### Issue: Training Too Slow

**Solutions**:

1. **Reduce quantum circuit size**:
```bash
python QuantumTransformerSimpleRL.py \
    --env=CartPole-v1 \
    --n-qubits=3 \
    --n-layers=1
```

2. **Increase batch size**:
```bash
python QuantumTransformerSimpleRL.py \
    --env=CartPole-v1 \
    --batch-size=128
```

3. **Use GPU**:
```bash
python QuantumTransformerSimpleRL.py \
    --env=CartPole-v1 \
    --device=cuda
```

### Issue: Unstable Learning (Reward Fluctuates)

**Solutions**:

1. **Increase target network sync frequency**:
Edit line 581 in QuantumTransformerSimpleRL.py:
```python
self.sync_every = 50  # Change from 100 to 50
```

2. **Use larger replay buffer**:
```bash
python QuantumTransformerSimpleRL.py \
    --env=CartPole-v1 \
    --memory-size=50000
```

3. **Lower learning rate**:
```bash
python QuantumTransformerSimpleRL.py \
    --env=CartPole-v1 \
    --lr=0.0001
```

### Issue: Out of Memory

**Solutions**:

1. **Reduce batch size**:
```bash
python QuantumTransformerSimpleRL.py \
    --env=CartPole-v1 \
    --batch-size=32
```

2. **Use CPU**:
```bash
python QuantumTransformerSimpleRL.py \
    --env=CartPole-v1 \
    --device=cpu
```

## Hyperparameter Tuning Guide

### Quantum Parameters

| Parameter | Small | Medium | Large | Effect |
|-----------|-------|--------|-------|--------|
| `n_qubits` | 3 | 4 | 6 | Model capacity, training time |
| `n_layers` | 1 | 2 | 3 | Circuit depth, expressiveness |
| `degree` | 1 | 2 | 3 | QSVT polynomial complexity |
| `n_timesteps` | 2 | 4 | 8 | Temporal context window |

### RL Parameters

| Parameter | Conservative | Balanced | Aggressive |
|-----------|-------------|----------|------------|
| `lr` | 0.0001 | 0.001 | 0.01 |
| `gamma` | 0.95 | 0.99 | 0.999 |
| `exploration_decay` | 0.999 | 0.995 | 0.99 |
| `batch_size` | 32 | 64 | 128 |

### Recommended Combinations

**Fast prototyping** (quick results, lower performance):
```bash
python QuantumTransformerSimpleRL.py \
    --env=CartPole-v1 \
    --n-qubits=3 \
    --n-layers=1 \
    --degree=1 \
    --batch-size=128 \
    --num-episodes=200
```

**Balanced** (good performance, reasonable time):
```bash
python QuantumTransformerSimpleRL.py \
    --env=CartPole-v1 \
    --n-qubits=4 \
    --n-layers=2 \
    --degree=2 \
    --batch-size=64 \
    --num-episodes=500
```

**High performance** (best results, slower):
```bash
python QuantumTransformerSimpleRL.py \
    --env=CartPole-v1 \
    --n-qubits=6 \
    --n-layers=3 \
    --degree=3 \
    --batch-size=64 \
    --num-episodes=1000
```

## Comparison with Classical Baselines

To compare quantum vs classical approaches, you can train a classical DQN on the same environment and compare results.

**Example comparison workflow**:

```bash
# Train quantum agent
python QuantumTransformerSimpleRL.py \
    --env=CartPole-v1 \
    --n-qubits=4 \
    --log-index=quantum

# Train classical agent (using standard DQN)
# [Use your classical baseline implementation]

# Compare learning curves from training_curves.png files
```

## Advanced Usage

### Systematic Hyperparameter Search

Run grid search over quantum parameters:

```bash
for qubits in 3 4 6; do
  for layers in 1 2 3; do
    for degree in 1 2 3; do
      python QuantumTransformerSimpleRL.py \
        --env=CartPole-v1 \
        --n-qubits=$qubits \
        --n-layers=$layers \
        --degree=$degree \
        --log-index=q${qubits}_l${layers}_d${degree} \
        --num-episodes=500
    done
  done
done
```

### Testing Generalization

Train on one environment, analyze on related tasks:

```bash
# Train on CartPole
python QuantumTransformerSimpleRL.py --env=CartPole-v1

# Train on Acrobot (similar balancing/swinging task)
python QuantumTransformerSimpleRL.py --env=Acrobot-v1
```

Compare if similar quantum circuit structures work well on related tasks.

## Citation & References

- **Sim et al. (2019)**: sim14 quantum circuit ansatz
- **QSVT**: Quantum Singular Value Transformation
- **OpenAI Gym**: Classic RL benchmark environments
- **PennyLane**: Quantum machine learning framework

---

**Happy Quantum Reinforcement Learning! 🎮⚛️🤖**
