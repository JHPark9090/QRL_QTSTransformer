# QRL-QTSTransformer

Quantum Reinforcement Learning with Quantum Time-Series Transformers.

## Overview

This project implements hybrid classical-quantum RL agents using variational quantum circuits for Q-value estimation. The core architecture is a **Quantum Time-Series Transformer** that uses QSVT (Quantum Singular Value Transformation) for temporal processing of state sequences, trained via Double DQN with experience replay.

The project spans three experiment domains of increasing complexity:

| Domain | Environments | State Input | Architecture |
|--------|-------------|-------------|--------------|
| **Classic Control** | CartPole, FrozenLake, MountainCar, Acrobot | Low-dim vectors | State History → QTSTransformer → Q-values |
| **Atari Games** | SpaceInvaders, Breakout, DonkeyKong, MarioBros, Pong, Tetris | 210x160 RGB frames | CNN → Sinusoidal PE → QTSTransformer → Q-values |
| **Super Mario Bros** | SuperMarioBros (gym-super-mario-bros) | 240x256 RGB frames | CNN → QTSTransformer → Q-values |

### Architecture

**Classic Control (simple state vectors):**
```
State Vector (e.g., 4-dim for CartPole)
  -> State History Buffer (n_timesteps=4)
  -> Quantum Time-Series Transformer
  -> Q-values
```

**Visual Environments (Atari, Super Mario Bros):**
```
Game Frame (RGB)
  -> Preprocessing (grayscale, resize, frame stack)
  -> 4 Stacked Frames (4x84x84)
  -> CNN Feature Extractor (128-dim)
  -> Sinusoidal Positional Encoding (v3)
  -> Quantum Time-Series Transformer (sim14 ansatz, [-pi, pi] scaling)
  -> Q-values (one per action)
```

**Quantum circuit**: sim14 ansatz from Sim et al. (2019) — RY, CRX rotation gates in ring topology, implemented in PennyLane with `default.qubit` backend and `diff_method="backprop"`.

## Key Results

### Classic Control (Gymnasium)

| Environment | Episodes | Peak Performance | Notes |
|-------------|----------|-----------------|-------|
| **CartPole-v1** | 500 | **500/500** (30 consecutive perfect episodes) | Catastrophic forgetting after ep 106 |
| **FrozenLake-v1** | 3,000 | **59% success** (40x over random baseline) | Stable training, stochastic environment |

See [docs/SimpleRL_Results_Summary.md](docs/SimpleRL_Results_Summary.md) for details.

### Atari Games (v3, latest)

| Game | Config | Peak 100-ep Avg | Human-Normalized | Status |
|------|--------|----------------|-----------------|--------|
| **SpaceInvaders** | Q8/D2 | **336.3** | 4.5% | Strong learner |
| **Breakout** | Q8/D2 | **14.2** | 43.4% | Steady improvement |
| **MarioBros** | Q8/D2 | **504.0** | 21.3% | Early peak, declined |
| **DonkeyKong** | Q10/D3 | **82.0** | -2.5% | Benefits from more qubits |
| Pong | Q8/D2 | -20.7 | 0.0% | No learning |
| Tetris | Q8/D2 | 0.1 | 0.3% | No learning |

v3 achieved **+19% to +390% improvement** over v2 across all learning games.

See [docs/Atari_Results_Summary_v2.md](docs/Atari_Results_Summary_v2.md) for full results and analysis.

### Super Mario Bros

**Status: Not yet run.** Training script and SLURM job are ready.

- Script: [scripts/QuantumTransformerMario.py](scripts/QuantumTransformerMario.py)
- Classical baseline: [scripts/ClassicalSuperMario.py](scripts/ClassicalSuperMario.py)
- SLURM job: [jobs/run_quantum_transformer_mario.sh](jobs/run_quantum_transformer_mario.sh)
- Architecture: CNN + QTSTransformer with DQN, using `gym-super-mario-bros` and `nes-py` wrappers

## Project Structure

```
QRL_QTSTransformer/
├── scripts/                  # Python training scripts
│   ├── QuantumTransformerAtari_v3.py    # Atari v3 (latest)
│   ├── QuantumTransformerAtari.py       # Atari v1/v2 (PER, early stopping)
│   ├── QuantumTransformerSimpleRL.py    # CartPole, FrozenLake, MountainCar, Acrobot
│   ├── QuantumTransformerMario.py       # Super Mario Bros
│   ├── ClassicalSuperMario.py           # Classical baseline for Mario
│   ├── QTSTransformer.py               # Quantum transformer module (v1)
│   ├── QTSTransformer_v2_5.py          # Quantum transformer module (v2.5)
│   ├── record_quantum_mario_video.py    # Video recording utility
│   ├── record_quantum_mario_video_TRANSFORMER.py
│   └── test_quantum_rl_setup.py         # Environment setup test
├── jobs/                     # SLURM batch scripts
│   ├── run_quantum_transformer_atari_v3.sh       # Atari v3 Q8/D2
│   ├── run_quantum_transformer_atari_v3_q10d3.sh # Atari v3 Q10/D3
│   ├── run_quantum_transformer_atari.sh           # Atari v1/v2
│   ├── run_quantum_transformer_simplerl.sh        # Classic control
│   ├── run_quantum_transformer_mario.sh           # Super Mario Bros
│   ├── run_all_simplerl.sh                        # Batch submit all simple envs
│   └── run_record_mario_video.sh                  # Record agent gameplay
├── docs/                     # Documentation and results
│   ├── Atari_Results_Summary_v2.md      # Comprehensive Atari results (v1/v2/v3)
│   ├── Atari_Results_Summary.md         # Atari v1 results
│   ├── SimpleRL_Results_Summary.md      # CartPole & FrozenLake results
│   ├── QUANTUM_RL_SUMMARY.md            # Overall project summary
│   ├── README_QuantumTransformerAtari.md
│   ├── README_QuantumTransformerMario.md
│   ├── README_QuantumTransformerSimpleRL.md
│   └── VIDEO_RECORDING_GUIDE.md
├── checkpoints/              # Model checkpoints (v1/v2/v3)
├── checkpoints_backup_v1/    # v1 checkpoint backups
├── logs/                     # SLURM job output/error logs
└── results/                  # Results data files
```

## Quick Start

### Environment Setup

```bash
# On Perlmutter (NERSC)
eval "$(conda shell.bash hook 2>/dev/null)"
conda activate /pscratch/sd/j/junghoon/conda-envs/qml_env

# Key dependencies: PennyLane 0.43.0, PyTorch 2.5.0+cu121, gymnasium, ale-py
# For Mario: gym-super-mario-bros, nes-py
```

### Classic Control

```bash
cd /pscratch/sd/j/junghoon

# CartPole
sbatch --export=ALL,ENV="CartPole-v1" QRL_QTSTransformer/jobs/run_quantum_transformer_simplerl.sh

# FrozenLake
sbatch --export=ALL,ENV="FrozenLake-v1" QRL_QTSTransformer/jobs/run_quantum_transformer_simplerl.sh

# Also supported: MountainCar-v0, Acrobot-v1
```

### Atari Games

```bash
cd /pscratch/sd/j/junghoon

# v3 Q8/D2 (default)
sbatch --export=ALL,ENV="ALE/SpaceInvaders-v5" QRL_QTSTransformer/jobs/run_quantum_transformer_atari_v3.sh

# v3 Q10/D3 (increased capacity)
sbatch --export=ALL,ENV="ALE/DonkeyKong-v5" QRL_QTSTransformer/jobs/run_quantum_transformer_atari_v3_q10d3.sh

# Supported: ALE/SpaceInvaders-v5, ALE/Breakout-v5, ALE/MarioBros-v5,
#            ALE/DonkeyKong-v5, ALE/Pong-v5, ALE/Tetris-v5
```

### Super Mario Bros

```bash
cd /pscratch/sd/j/junghoon
sbatch QRL_QTSTransformer/jobs/run_quantum_transformer_mario.sh
```

### Check Training Progress

```bash
python3 -c "
import torch, numpy as np
cp = torch.load('QRL_QTSTransformer/checkpoints/<CHECKPOINT_DIR>/latest_checkpoint.chkpt',
                map_location='cpu', weights_only=False)
print(f'Episode: {cp[\"episode\"]+1}')
print(f'Epsilon: {cp[\"exploration_rate\"]:.4f}')
r = cp['metrics']['rewards']
print(f'Last 100 avg reward: {np.mean(r[-100:]):.1f}')
"
```

## Experiment Configurations

### Classic Control

| Parameter | Value |
|-----------|-------|
| Qubits | 8 |
| Ansatz Layers | 2 |
| QSVT Degree | 2 |
| Timesteps | 4 |
| Learning Rate | 0.001 |
| Batch Size | 64 |
| Gamma | 0.99 |

### Atari — Q8/D2 (default)

| Parameter | Value |
|-----------|-------|
| Qubits | 8 |
| Ansatz Layers | 2 |
| QSVT Degree | 2 |
| Timesteps | 4 |
| Feature Dim | 128 |
| Learning Rate | 0.00025 |
| Batch Size | 32 |
| Gamma | 0.99 |
| Exploration Decay | 0.9999 |
| Max Episodes | 10,000 |

### Atari — Q10/D3 (increased capacity)

Same as Q8/D2 except: **Qubits = 10**, **QSVT Degree = 3**.

## Atari Version History

| Version | Key Changes | Best Result |
|---------|-------------|-------------|
| **v1** (Feb 2026) | Baseline DQN, uniform replay | SpaceInvaders: 280.9 |
| **v2** (Feb 2026) | + PER, early stopping, LR=0.0001 | SpaceInvaders: 283.1 |
| **v3** (Feb 2026) | + Sinusoidal PE, centered angle scaling, separate CNNs | SpaceInvaders: 336.3, Breakout: 14.2 |

### v3 Key Improvements

1. **Sinusoidal Positional Encoding** — temporal awareness of frame ordering within the 4-frame context window
2. **Centered angle scaling** — `(sigmoid - 0.5) * 2pi` maps to [-pi, pi] centered at 0, avoiding the barren plateau at theta=pi
3. **Separate CNN instances** — independent feature extractors for online/target networks, fixing a gradient-blocking bug in v1/v2

## SLURM Configuration

| Parameter | Value |
|-----------|-------|
| Account | `m4807_g` |
| Constraint | `gpu&hbm80g` |
| QOS | `shared` |
| Time Limit | 48 hours |
| Nodes | 1 |
| GPUs | 1 (A100-SXM4-80GB) |
| CPUs | 32 |

## References

**QTSTransformer model (this project is based on):**
```
J. J. Park et al., "Resting-State fMRI Analysis Using Quantum Time-Series Transformer,"
2025 IEEE International Conference on Quantum Computing and Engineering (QCE),
Albuquerque, NM, USA, 2025, pp. 2352-2363, doi: 10.1109/QCE65121.2025.00256.
```

**sim14 quantum circuit ansatz:**
```
Sim, S., Johnson, P. D., & Aspuru-Guzik, A. (2019).
Expressibility and entangling capability of parameterized quantum circuits
for hybrid quantum-classical algorithms.
Advanced Quantum Technologies, 2(12), 1900070.
```

## Author

Junghoon Kim — Lawrence Berkeley National Laboratory / UC Berkeley
