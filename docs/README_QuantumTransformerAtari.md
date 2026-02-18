# Quantum Time-Series Transformer for Atari Games

This README provides comprehensive instructions for training quantum transformer agents on classic Atari games: **DonkeyKong**, **Pacman**, **Mario Bros**, **Space Invaders**, and **Tetris**.

## Overview

**QuantumTransformerAtari.py** implements a hybrid quantum-classical RL agent for Atari 2600 games using:
- **CNN Feature Extraction**: Processes 210×160 RGB frames → feature vectors
- **Quantum Time-Series Transformer**: QSVT-based temporal processing
- **Double DQN**: Target network + experience replay for stable learning
- **Modern Stack**: Uses `gymnasium` (not old `gym`) with `ale-py`

### Key Differences from Super Mario

| Aspect | Super Mario | Atari Games |
|--------|-------------|-------------|
| **Environment** | NES emulator | Atari 2600 emulator (ALE) |
| **Library** | gym-super-mario-bros | gymnasium + ale-py |
| **Resolution** | 240×256 | 210×160 |
| **Preprocessing** | Built-in wrappers | Custom gymnasium wrappers |
| **Action Space** | Game-specific | Variable (4-18 actions) |

---

## Supported Games

### 🦍 DonkeyKong-v5
- **Goal**: Rescue the princess by climbing ladders and avoiding barrels
- **Actions**: 18 (move, jump, climb)
- **Difficulty**: Medium-Hard
- **Typical Training**: 5,000-10,000 episodes

### 👻 Pacman-v5 (NOT Ms. Pacman!)
- **Goal**: Eat all dots while avoiding ghosts
- **Actions**: 9 (move in 4 directions + combinations)
- **Difficulty**: Medium
- **Typical Training**: 3,000-8,000 episodes
- **Note**: This is original Pac-Man, not Ms. Pac-Man

### 🎮 MarioBros-v5
- **Goal**: Defeat enemies by jumping from below platforms
- **Actions**: 18 (move, jump)
- **Difficulty**: Medium
- **Typical Training**: 5,000-10,000 episodes

### 👾 SpaceInvaders-v5
- **Goal**: Shoot descending aliens before they reach you
- **Actions**: 6 (move left/right, fire)
- **Difficulty**: Medium-Easy
- **Typical Training**: 2,000-5,000 episodes

### 🧱 Tetris-v5
- **Goal**: Clear lines by rotating and placing falling blocks
- **Actions**: 5 (move left/right, rotate, drop)
- **Difficulty**: Hard
- **Typical Training**: 10,000+ episodes

---

## Installation & Setup

### Step 1: Install Dependencies

```bash
# Activate your environment
conda activate ./conda-envs/qml_eeg

# Install gymnasium (modern gym replacement)
pip install gymnasium

# Install Atari support
pip install ale-py

# Install Atari ROM bundle (required for games)
pip install "gymnasium[atari, accept-rom-license]"

# Verify installation
python -c "import gymnasium as gym; import ale_py; print('✓ Ready for Atari!')"
```

### Step 2: Verify Atari ROMs

```bash
# Test that games are accessible
python -c "
import gymnasium as gym
import ale_py
gym.register_envs(ale_py)
env = gym.make('ALE/Pacman-v5')
print(f'✓ Pacman environment created')
print(f'  Action space: {env.action_space.n} actions')
env.close()
"
```

### Important Notes

- **Use `gymnasium`, not `gym`**: The old `gym` library is deprecated
- **ROM License**: Installing `gymnasium[accept-rom-license]` automatically accepts the Atari ROM license
- **ALE naming**: All games use `ALE/GameName-v5` format (e.g., `ALE/Pacman-v5`)

---

## Quick Start

### Train on Pacman (Recommended First)

```bash
python QuantumTransformerAtari.py --env=ALE/Pacman-v5
```

### Train on Space Invaders (Easiest)

```bash
python QuantumTransformerAtari.py --env=ALE/SpaceInvaders-v5
```

### Train on DonkeyKong

```bash
python QuantumTransformerAtari.py --env=ALE/DonkeyKong-v5 --num-episodes=10000
```

---

## Architecture

```
Atari Frame (210×160×3 RGB)
    ↓
Preprocessing
    ├─ Grayscale conversion (RGB → 1 channel)
    ├─ Resize to 84×84
    └─ Normalize (0-255 → 0-1)
    ↓
Frame Stacking (last 4 frames)
    ↓
Shape: (4, 84, 84)
    ↓
CNN Feature Extractor (Nature DQN architecture)
    ├─ Conv2D(4→32, kernel=8, stride=4)
    ├─ Conv2D(32→64, kernel=4, stride=2)
    ├─ Conv2D(64→64, kernel=3, stride=1)
    └─ FC(3136 → 512 → feature_dim×4)
    ↓
Reshape to (4 timesteps, feature_dim)
    ↓
Quantum Time-Series Transformer
    ├─ Feature Projection
    ├─ Quantum Circuit (QSVT)
    ├─ Multi-observable Measurement
    └─ Output Layer
    ↓
Q-values for all actions
```

---

## Detailed Usage

### 1. Basic Training with Defaults

```bash
python QuantumTransformerAtari.py --env=ALE/Pacman-v5
```

**Defaults**:
- Qubits: 6
- Layers: 2
- QSVT Degree: 2
- Batch size: 32
- Episodes: 10,000
- Learning rate: 0.00025

### 2. Custom Quantum Parameters

```bash
# Smaller quantum circuit (faster)
python QuantumTransformerAtari.py \
    --env=ALE/SpaceInvaders-v5 \
    --n-qubits=4 \
    --n-layers=1 \
    --degree=1

# Larger quantum circuit (more expressive)
python QuantumTransformerAtari.py \
    --env=ALE/DonkeyKong-v5 \
    --n-qubits=8 \
    --n-layers=3 \
    --degree=3 \
    --feature-dim=256
```

### 3. Custom Training Parameters

```bash
python QuantumTransformerAtari.py \
    --env=ALE/Tetris-v5 \
    --batch-size=64 \
    --lr=0.0001 \
    --gamma=0.99 \
    --exploration-rate-decay=0.9995 \
    --num-episodes=15000
```

### 4. Resume Training

```bash
python QuantumTransformerAtari.py \
    --env=ALE/Pacman-v5 \
    --resume
```

### 5. Watch Agent Play (Render)

```bash
python QuantumTransformerAtari.py \
    --env=ALE/SpaceInvaders-v5 \
    --render \
    --num-episodes=100
```

**Note**: Rendering significantly slows down training. Use only for visualization.

---

## Game-Specific Recommendations

### Pacman (Recommended First) 👻

**Why start here**: Medium difficulty, clear feedback, interesting quantum decision-making

```bash
python QuantumTransformerAtari.py \
    --env=ALE/Pacman-v5 \
    --n-qubits=6 \
    --n-layers=2 \
    --num-episodes=5000 \
    --batch-size=32
```

**Expected**: Score improvements visible after 1000-2000 episodes

---

### Space Invaders (Easiest) 👾

**Why easiest**: Simple mechanics, fast episodes, clear objective

```bash
python QuantumTransformerAtari.py \
    --env=ALE/SpaceInvaders-v5 \
    --n-qubits=4 \
    --n-layers=2 \
    --num-episodes=3000
```

**Expected**: Learns to shoot aliens after 500-1000 episodes

---

### DonkeyKong (Challenging) 🦍

**Why harder**: Complex platforming, precise timing required

```bash
python QuantumTransformerAtari.py \
    --env=ALE/DonkeyKong-v5 \
    --n-qubits=6 \
    --n-layers=3 \
    --degree=2 \
    --num-episodes=10000 \
    --lr=0.0001
```

**Expected**: Learns basic navigation after 3000-5000 episodes

---

### Mario Bros (Medium) 🎮

**Why medium**: Two-player mechanics adapted to single-player, moderate complexity

```bash
python QuantumTransformerAtari.py \
    --env=ALE/MarioBros-v5 \
    --n-qubits=6 \
    --n-layers=2 \
    --num-episodes=8000
```

**Expected**: Learns to defeat enemies after 2000-4000 episodes

---

### Tetris (Hardest) 🧱

**Why hardest**: Long-term planning, sparse rewards, complex state space

```bash
python QuantumTransformerAtari.py \
    --env=ALE/Tetris-v5 \
    --n-qubits=8 \
    --n-layers=3 \
    --degree=3 \
    --feature-dim=256 \
    --num-episodes=15000 \
    --lr=0.0001 \
    --gamma=0.99
```

**Expected**: Basic competence after 10,000+ episodes

---

## Training Metrics

### Console Output

```
Ep  100 | R:   245.0 | AvgR:   123.4 | L:  456 | Loss: 0.0234 | ε: 0.905
Ep  110 | R:   280.0 | AvgR:   145.2 | L:  512 | Loss: 0.0189 | ε: 0.895
```

**Metrics**:
- **Ep**: Episode number
- **R**: Reward this episode (score)
- **AvgR**: Average reward (last 100 episodes) - **KEY METRIC**
- **L**: Episode length (steps)
- **Loss**: TD loss
- **ε (epsilon)**: Exploration rate

### Success Criteria by Game

| Game | Beginner Score | Intermediate | Expert | Human Average |
|------|---------------|--------------|--------|---------------|
| Pacman | 500 | 2,000 | 5,000 | 7,500 |
| SpaceInvaders | 200 | 500 | 1,000 | 1,500 |
| DonkeyKong | 5,000 | 15,000 | 30,000 | 50,000 |
| MarioBros | 10,000 | 30,000 | 60,000 | 100,000 |
| Tetris | 100 | 500 | 1,500 | 3,000 |

---

## Output Files

Training creates:

```
AtariCheckpoints/
└── QTransformer_ALEPacmanv5_Q6_L2_D2_Run1/
    ├── latest_checkpoint.chkpt    # Model + training state
    └── training_curves.png        # Reward/length/loss plots
```

### Checkpoint Contents

- Model weights (online + target networks)
- Optimizer state
- Exploration rate
- Training metrics (rewards, lengths, losses)
- RNG states (for reproducibility)

---

## Performance Benchmarks

Training time estimates (per 1000 episodes):

| Game | Qubits | CPU (32 cores) | GPU (A100) | Typical Episodes |
|------|--------|----------------|------------|------------------|
| SpaceInvaders | 4 | ~2 hours | ~45 min | 3,000 |
| Pacman | 6 | ~3 hours | ~1.5 hours | 5,000 |
| MarioBros | 6 | ~3 hours | ~1.5 hours | 8,000 |
| DonkeyKong | 6 | ~4 hours | ~2 hours | 10,000 |
| Tetris | 8 | ~6 hours | ~3 hours | 15,000 |

**Note**: Times vary significantly based on episode length and quantum circuit complexity.

---

## Troubleshooting

### Issue 1: ImportError: No module named 'gymnasium'

**Solution**: Install gymnasium
```bash
pip install gymnasium ale-py "gymnasium[atari, accept-rom-license]"
```

### Issue 2: "Environment ALE/Pacman-v5 doesn't exist"

**Solution**: Register ALE environments
```bash
python -c "
import gymnasium as gym
import ale_py
gym.register_envs(ale_py)
env = gym.make('ALE/Pacman-v5')
print('✓ Success!')
"
```

### Issue 3: ROM not found

**Solution**: Accept ROM license
```bash
pip install "gymnasium[accept-rom-license]"
# Or manually download ROMs from Atari
```

### Issue 4: Training too slow

**Solution 1**: Reduce quantum circuit size
```bash
python QuantumTransformerAtari.py \
    --env=ALE/Pacman-v5 \
    --n-qubits=4 \
    --n-layers=1
```

**Solution 2**: Use GPU
```bash
python QuantumTransformerAtari.py \
    --env=ALE/Pacman-v5 \
    --device=cuda
```

**Solution 3**: Reduce batch size
```bash
python QuantumTransformerAtari.py \
    --env=ALE/Pacman-v5 \
    --batch-size=16
```

### Issue 5: Agent not learning

**Solution 1**: Increase exploration
```bash
python QuantumTransformerAtari.py \
    --env=ALE/Pacman-v5 \
    --exploration-rate-decay=0.9995
```

**Solution 2**: Adjust learning rate
```bash
python QuantumTransformerAtari.py \
    --env=ALE/Pacman-v5 \
    --lr=0.0005
```

**Solution 3**: Increase network capacity
```bash
python QuantumTransformerAtari.py \
    --env=ALE/Pacman-v5 \
    --n-qubits=8 \
    --feature-dim=256
```

### Issue 6: CUDA out of memory

**Solution**: Use CPU or reduce batch size
```bash
python QuantumTransformerAtari.py \
    --env=ALE/Pacman-v5 \
    --device=cpu \
    --batch-size=16
```

---

## Hyperparameter Tuning Guide

### Quantum Parameters

| Parameter | Small (Fast) | Medium (Balanced) | Large (Best) |
|-----------|--------------|-------------------|--------------|
| `n_qubits` | 4 | 6 | 8-12 |
| `n_layers` | 1 | 2 | 3-4 |
| `degree` | 1 | 2 | 3 |
| `feature_dim` | 64 | 128 | 256 |

### RL Parameters

| Parameter | Conservative | Balanced | Aggressive |
|-----------|-------------|----------|------------|
| `lr` | 0.0001 | 0.00025 | 0.001 |
| `gamma` | 0.95 | 0.99 | 0.999 |
| `exploration_decay` | 0.9999 | 0.9995 | 0.999 |
| `batch_size` | 16 | 32 | 64 |

### Recommended Configurations

**Fast prototyping** (quick results):
```bash
python QuantumTransformerAtari.py \
    --env=ALE/SpaceInvaders-v5 \
    --n-qubits=4 \
    --n-layers=1 \
    --degree=1 \
    --batch-size=64 \
    --num-episodes=1000
```

**Balanced** (good performance):
```bash
python QuantumTransformerAtari.py \
    --env=ALE/Pacman-v5 \
    --n-qubits=6 \
    --n-layers=2 \
    --degree=2 \
    --batch-size=32 \
    --num-episodes=5000
```

**High performance** (best results):
```bash
python QuantumTransformerAtari.py \
    --env=ALE/DonkeyKong-v5 \
    --n-qubits=8 \
    --n-layers=3 \
    --degree=3 \
    --feature-dim=256 \
    --batch-size=32 \
    --num-episodes=15000
```

---

## Research Experiments

### Experiment 1: Multi-Game Benchmark

Test same quantum architecture across all games:

```bash
QUANTUM_CONFIG="--n-qubits=6 --n-layers=2 --degree=2"

python QuantumTransformerAtari.py --env=ALE/SpaceInvaders-v5 $QUANTUM_CONFIG --num-episodes=3000
python QuantumTransformerAtari.py --env=ALE/Pacman-v5 $QUANTUM_CONFIG --num-episodes=5000
python QuantumTransformerAtari.py --env=ALE/MarioBros-v5 $QUANTUM_CONFIG --num-episodes=8000
python QuantumTransformerAtari.py --env=ALE/DonkeyKong-v5 $QUANTUM_CONFIG --num-episodes=10000
python QuantumTransformerAtari.py --env=ALE/Tetris-v5 $QUANTUM_CONFIG --num-episodes=15000
```

### Experiment 2: Quantum Circuit Scaling

```bash
for qubits in 4 6 8; do
  python QuantumTransformerAtari.py \
    --env=ALE/Pacman-v5 \
    --n-qubits=$qubits \
    --log-index=qubits_${qubits} \
    --num-episodes=5000
done
```

### Experiment 3: QSVT Degree Study

```bash
for degree in 1 2 3; do
  python QuantumTransformerAtari.py \
    --env=ALE/SpaceInvaders-v5 \
    --degree=$degree \
    --log-index=degree_${degree} \
    --num-episodes=3000
done
```

---

## Comparison with Classical Baselines

To properly evaluate quantum advantage, compare with classical DQN:

### Metrics to Compare

1. **Sample Efficiency**: Episodes to reach score threshold
2. **Final Performance**: Average score over last 100 episodes
3. **Training Time**: Wall-clock time to convergence
4. **Parameter Count**: Number of trainable parameters

### Example Comparison Protocol

```bash
# Train quantum agent
python QuantumTransformerAtari.py \
    --env=ALE/Pacman-v5 \
    --n-qubits=6 \
    --num-episodes=5000 \
    --log-index=quantum

# Train classical DQN (baseline)
# [Use your classical implementation]

# Compare training curves from both runs
```

---

## Tips for Success

1. **Start with SpaceInvaders or Pacman**: Easier games provide faster feedback
2. **Use small quantum circuits initially**: Test with 4 qubits before scaling up
3. **Monitor exploration rate**: Should decay slowly (ε > 0.1 for first 1000+ episodes)
4. **Be patient**: Atari games require 1000s of episodes for meaningful progress
5. **Save checkpoints frequently**: Use `--save-every=50` to avoid losing progress
6. **Use GPU if available**: Quantum simulation is compute-intensive
7. **Track multiple metrics**: Don't just look at rewards, monitor loss and Q-values too

---

## Known Limitations

1. **Quantum simulation is slow**: 6-8 qubits is practical limit on CPUs
2. **Atari games are sample-inefficient**: Need 1000s-10000s of episodes
3. **Some games are very hard**: Tetris and DonkeyKong require extensive training
4. **Action spaces vary**: Each game has different number of actions
5. **Scores vary widely**: Can't directly compare scores across games

---

## Citation & References

### Quantum Computing
- **PennyLane**: Quantum machine learning framework
- **QSVT**: Quantum Singular Value Transformation

### Deep Reinforcement Learning
- **DQN Nature**: Mnih et al., "Human-level control through deep reinforcement learning" (2015)
- **Double DQN**: Van Hasselt et al., "Deep Reinforcement Learning with Double Q-learning" (2016)

### Atari Learning Environment
- **ALE**: Bellemare et al., "The Arcade Learning Environment" (2013)
- **Gymnasium**: Modern replacement for OpenAI Gym

---

## Advanced Topics

### Custom Preprocessing

Modify `AtariPreprocessing` class in the script to experiment with:
- Different resize dimensions
- Color vs grayscale
- Frame skipping rates
- Normalization strategies

### Custom Quantum Circuits

Modify `sim14_circuit` function to test:
- Different gate sequences
- Entanglement patterns
- Circuit depth variations

### Transfer Learning

Train on one game, fine-tune on another:
```bash
# Train on Space Invaders
python QuantumTransformerAtari.py --env=ALE/SpaceInvaders-v5 --num-episodes=3000

# Fine-tune on Pacman (load checkpoint and continue)
# [Requires code modification to load cross-game checkpoints]
```

---

**Happy Quantum Atari Gaming! 🎮⚛️👾**
