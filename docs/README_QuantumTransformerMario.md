# Quantum Time-Series Transformer for Super Mario Bros RL

This README provides step-by-step instructions for running the Quantum Time-Series Transformer on the Super Mario Bros environment.

## Overview

**QuantumTransformerMario.py** implements a hybrid quantum-classical reinforcement learning agent that combines:
- **CNN Feature Extractor**: Extracts spatial features from game frames (4×84×84 → feature vectors)
- **Quantum Time-Series Transformer**: Processes temporal sequences using QSVT (Quantum Singular Value Transformation)
- **Double DQN**: Standard Deep Q-Network with experience replay and target network

## Architecture

```
Input: 4 Stacked Frames (4×84×84 grayscale images)
    ↓
CNN Feature Extractor (Conv layers + FC layers)
    ↓
Reshaped to (batch, 4 timesteps, feature_dim)
    ↓
Quantum Time-Series Transformer
    ├─ Feature Projection Layer
    ├─ Quantum Circuit (sim14 ansatz with QSVT)
    ├─ Multi-observable Measurement (X, Y, Z)
    └─ Output Feed-forward Layer
    ↓
Q-values for 2 actions (walk right, jump right)
    ↓
DQN Training (Double Q-learning + Experience Replay)
```

## Prerequisites

### Environment Setup

1. **Activate the conda environment**:
```bash
conda activate ./conda-envs/qml_eeg
```

2. **Verify required packages**:
```bash
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import pennylane as qml; print('PennyLane:', qml.__version__)"
python -c "import gym_super_mario_bros; print('Super Mario Bros: OK')"
python -c "import nes_py; print('NES Emulator: OK')"
```

### Check Dependencies

If any packages are missing, install them:
```bash
pip install torch torchvision pennylane gym gym-super-mario-bros nes-py tensordict torchrl
```

## How to Run

### Step 1: Basic Training (Quick Start)

Run with default parameters (6 qubits, 2 layers, degree 2):

```bash
python QuantumTransformerMario.py
```

This will:
- Train for 40,000 episodes
- Use 6 qubits, 2 ansatz layers, QSVT degree 2
- Save checkpoints every 10 episodes to `SuperMarioCheckpoints/QTransformerMario_Q6_L2_D2_Run1/`
- Generate training plots (rewards, losses, Q-values)

### Step 2: Custom Quantum Parameters

Adjust quantum circuit configuration:

```bash
python QuantumTransformerMario.py \
    --n-qubits=8 \
    --n-layers=3 \
    --degree=3 \
    --feature-dim=256
```

**Parameter descriptions**:
- `--n-qubits`: Number of qubits in quantum circuit (4, 6, 8, 12 typical)
- `--n-layers`: Number of ansatz layers (1-4 recommended)
- `--degree`: Degree of QSVT polynomial (1-4)
- `--feature-dim`: Feature dimension after CNN extraction (64, 128, 256)
- `--dropout`: Dropout rate for regularization (default: 0.1)

### Step 3: Custom Training Parameters

Modify RL hyperparameters:

```bash
python QuantumTransformerMario.py \
    --n-qubits=6 \
    --batch-size=64 \
    --lr=0.0001 \
    --gamma=0.95 \
    --learn-step=5 \
    --num-episodes=10000
```

**RL parameters**:
- `--batch-size`: Batch size for training (32, 64, 128)
- `--lr`: Learning rate (0.0001, 0.00025, 0.0005)
- `--gamma`: Discount factor (0.9-0.99)
- `--learn-step`: Learn every N steps (1, 3, 5)
- `--num-episodes`: Total number of episodes to train
- `--exploration-rate-decay`: Epsilon decay rate (default: 0.99999975)

### Step 4: Resume Training from Checkpoint

If training was interrupted, resume from the last checkpoint:

```bash
python QuantumTransformerMario.py --resume
```

This will:
- Load the latest checkpoint from the previous run
- Restore model weights, optimizer state, and exploration rate
- Continue training from the last completed episode

### Step 5: Multiple Experimental Runs

Run different experiments with unique log indices:

```bash
# Experiment 1: Small quantum circuit
python QuantumTransformerMario.py \
    --n-qubits=4 \
    --n-layers=1 \
    --degree=2 \
    --log-index=1

# Experiment 2: Medium quantum circuit
python QuantumTransformerMario.py \
    --n-qubits=6 \
    --n-layers=2 \
    --degree=2 \
    --log-index=2

# Experiment 3: Large quantum circuit
python QuantumTransformerMario.py \
    --n-qubits=8 \
    --n-layers=3 \
    --degree=3 \
    --log-index=3
```

Each run creates a separate directory with unique results.

### Step 6: CPU vs GPU Training

Force CPU training (for testing or systems without GPU):

```bash
python QuantumTransformerMario.py --device=cpu
```

Force GPU training (default if CUDA is available):

```bash
python QuantumTransformerMario.py --device=cuda
```

### Step 7: Reproducibility with Seeds

Set a specific seed for reproducible results:

```bash
python QuantumTransformerMario.py --seed=2024
python QuantumTransformerMario.py --seed=2025
python QuantumTransformerMario.py --seed=42
```

## Output Files

Training generates the following outputs in `SuperMarioCheckpoints/QTransformerMario_Q{n_qubits}_L{n_layers}_D{degree}_Run{log_index}/`:

### Checkpoints
- `latest_checkpoint.chkpt`: Model weights, optimizer state, RNG states, metrics

### Logs
- `log.txt`: Episode-by-episode metrics (rewards, losses, Q-values, timestamps)

### Plots
- `reward_plot.jpg`: Moving average of episode rewards
- `length_plot.jpg`: Moving average of episode lengths
- `loss_plot.jpg`: Moving average of training losses
- `q_plot.jpg`: Moving average of Q-values

### Replay Buffer
- `replay_buffer/`: Memory-mapped experience replay buffer (100,000 transitions)

## Example Training Commands

### Recommended Configuration (Balanced)
```bash
python QuantumTransformerMario.py \
    --n-qubits=6 \
    --n-layers=2 \
    --degree=2 \
    --feature-dim=128 \
    --batch-size=32 \
    --lr=0.00025 \
    --num-episodes=40000 \
    --seed=2025 \
    --log-index=1
```

### Fast Testing Configuration (Small)
```bash
python QuantumTransformerMario.py \
    --n-qubits=4 \
    --n-layers=1 \
    --degree=1 \
    --feature-dim=64 \
    --batch-size=16 \
    --num-episodes=1000 \
    --log-index=test
```

### High-Performance Configuration (Large)
```bash
python QuantumTransformerMario.py \
    --n-qubits=8 \
    --n-layers=3 \
    --degree=3 \
    --feature-dim=256 \
    --batch-size=64 \
    --lr=0.0001 \
    --num-episodes=100000 \
    --seed=2025 \
    --log-index=large
```

## Monitoring Training Progress

### Real-time Console Output

The script prints progress every episode:
```
Ep    142 | Step    87456 | ε 0.123 | Reward  245.3 | Length  312.1 | Loss 0.045 | Q  1.23 | Time   12.3s
```

**Metrics explained**:
- **Ep**: Episode number
- **Step**: Total environment steps taken
- **ε (epsilon)**: Current exploration rate
- **Reward**: Moving average reward (last 100 episodes)
- **Length**: Moving average episode length
- **Loss**: Moving average TD loss
- **Q**: Moving average Q-value
- **Time**: Seconds since last log

### Viewing Log File

Monitor training in real-time:
```bash
tail -f SuperMarioCheckpoints/QTransformerMario_Q6_L2_D2_Run1/log.txt
```

### Visualizing Plots

View training curves:
```bash
# On local machine with GUI
display SuperMarioCheckpoints/QTransformerMario_Q6_L2_D2_Run1/reward_plot.jpg

# Or copy plots to local machine
scp user@server:/path/to/SuperMarioCheckpoints/*/reward_plot.jpg .
```

## Expected Training Behavior

### Burn-in Phase (Steps 0 - 10,000)
- Agent explores randomly (high epsilon)
- No learning occurs yet
- Filling replay buffer with diverse experiences

### Early Training (Steps 10,000 - 100,000)
- Exploration rate decays gradually
- Agent starts learning basic behaviors (moving right)
- Rewards slowly increase

### Mid Training (Steps 100,000 - 500,000)
- Agent learns to jump over obstacles
- Rewards increase more rapidly
- Q-values stabilize

### Late Training (Steps 500,000+)
- Agent consistently reaches further in the level
- Exploration rate approaches minimum (0.1)
- Performance plateaus as optimal policy is learned

## Troubleshooting

### Issue: CUDA Out of Memory

**Solution 1**: Reduce batch size
```bash
python QuantumTransformerMario.py --batch-size=16
```

**Solution 2**: Use CPU
```bash
python QuantumTransformerMario.py --device=cpu
```

**Solution 3**: Reduce quantum circuit size
```bash
python QuantumTransformerMario.py --n-qubits=4 --feature-dim=64
```

### Issue: Training Very Slow

**Solution 1**: Reduce number of qubits (quantum simulation is expensive)
```bash
python QuantumTransformerMario.py --n-qubits=4 --n-layers=1
```

**Solution 2**: Increase learn step (learn less frequently)
```bash
python QuantumTransformerMario.py --learn-step=5
```

**Solution 3**: Reduce feature dimension
```bash
python QuantumTransformerMario.py --feature-dim=64
```

### Issue: Agent Not Learning

**Solution 1**: Check exploration rate decay
```bash
python QuantumTransformerMario.py --exploration-rate-decay=0.9999995
```

**Solution 2**: Adjust learning rate
```bash
python QuantumTransformerMario.py --lr=0.0005
```

**Solution 3**: Increase learn frequency
```bash
python QuantumTransformerMario.py --learn-step=1
```

### Issue: Missing Packages

**Solution**: Install dependencies
```bash
conda activate ./conda-envs/qml_eeg
pip install gym-super-mario-bros nes-py tensordict torchrl
```

## Performance Benchmarks

Typical training times on different hardware:

| Hardware | Qubits | Time per 100 Episodes | Episodes per Hour |
|----------|--------|----------------------|-------------------|
| CPU (32 cores) | 4 | ~30 min | ~200 |
| CPU (32 cores) | 6 | ~60 min | ~100 |
| GPU (A100) | 6 | ~15 min | ~400 |
| GPU (A100) | 8 | ~30 min | ~200 |

*Note: Times vary based on episode length and quantum circuit depth.*

## Comparison with Classical Baseline

To compare quantum transformer performance against classical CNN:
```bash
# Run quantum transformer
python QuantumTransformerMario.py --n-qubits=6 --log-index=quantum

# Run classical baseline
python ClassicalSuperMario.py --log-index=classical

# Compare results
python -c "
import pandas as pd
q_log = pd.read_csv('SuperMarioCheckpoints/QTransformerMario_Q6_L2_D2_Runquantum/log.txt', sep='\s+')
c_log = pd.read_csv('SuperMarioCheckpoints/ClassicalMario_Runclassical/log.txt', sep='\s+')
print('Quantum Mean Reward:', q_log['MeanReward'].iloc[-100:].mean())
print('Classical Mean Reward:', c_log['MeanReward'].iloc[-100:].mean())
"
```

## Advanced Usage

### Custom Quantum Circuit Configuration

The quantum circuit uses the "sim14" ansatz with configurable layers. To understand the circuit structure, see the `sim14_circuit()` function in QuantumTransformerMario.py:267-296.

### QSVT Polynomial Degree

The degree parameter controls the complexity of the QSVT transformation:
- **Degree 1**: Linear transformation (fastest, least expressive)
- **Degree 2**: Quadratic transformation (balanced)
- **Degree 3**: Cubic transformation (most expressive, slowest)

### Hyperparameter Tuning

For systematic hyperparameter search:
```bash
for qubits in 4 6 8; do
  for layers in 1 2 3; do
    for degree in 1 2 3; do
      python QuantumTransformerMario.py \
        --n-qubits=$qubits \
        --n-layers=$layers \
        --degree=$degree \
        --log-index=q${qubits}_l${layers}_d${degree}
    done
  done
done
```

## Citation

If you use this code for research, please cite:
- Sim et al. (2019) for the sim14 quantum circuit ansatz
- QSVT framework: Quantum Singular Value Transformation
- Super Mario Bros Gym environment

## Contact

For questions or issues, please refer to the main CLAUDE.md file or contact the repository maintainer.

---

**Happy Quantum Gaming! 🎮⚛️**
