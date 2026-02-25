#!/bin/bash
#SBATCH --account=m4807_g
#SBATCH --constraint=gpu&hbm80g
#SBATCH --qos=shared
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=32
#SBATCH --job-name=QTransV4_MultiChip
#SBATCH --output=QRL_QTSTransformer/logs/atari_v4_multichip_%j.out
#SBATCH --error=QRL_QTSTransformer/logs/atari_v4_multichip_%j.err

# ================================================================================
# Quantum Time-Series Transformer V4 (Multi-Circuit) for Atari RL
# ================================================================================
# v4 multi-circuit architecture:
#   1. CNN features split into K chunks
#   2. Each chunk processed by independent quantum circuit
#   3. Concatenated expectation values -> minimal linear head
#
# Key Features:
# - CNN feature extraction from game frames
# - K parallel quantum circuits for distributed processing
# - Double DQN with experience replay
# - Automatic checkpointing and resumption
# ================================================================================

echo "========================================================================"
echo "QUANTUM TRANSFORMER V4 (MULTI-CIRCUIT) - ATARI GAMES"
echo "========================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "========================================================================"

# Project directory
PROJECT_DIR="/pscratch/sd/j/junghoon"

# Create logs directory if it doesn't exist
mkdir -p "${PROJECT_DIR}/QRL_QTSTransformer/logs"

# Checkpoints directory
CHECKPOINTS_DIR="${PROJECT_DIR}/QRL_QTSTransformer/checkpoints"
mkdir -p "${CHECKPOINTS_DIR}"

# Avoid ~/.local packages shadowing conda env
export PYTHONNOUSERSITE=1

# Load conda environment
echo "Activating conda environment..."
eval "$(conda shell.bash hook 2>/dev/null)" && conda activate /pscratch/sd/j/junghoon/conda-envs/qml_env

# scipy lazy loading workaround for PennyLane
python -c "import scipy.constants" 2>/dev/null

# Verify environment
echo ""
echo "Environment Information:"
echo "Python: $(which python)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
if python -c 'import torch; print(torch.cuda.is_available())' | grep -q "True"; then
    echo "GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0))')"
fi
echo "PennyLane: $(python -c 'import pennylane; print(pennylane.__version__)')"
echo ""

# Optional: Set CUDA memory management for large models
# export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256

# ================================================================================
# Experiment Configuration
# ================================================================================

# Environment (can be overridden via --export=ALL,ENV="ALE/SpaceInvaders-v5")
ENV="${ENV:-ALE/SpaceInvaders-v5}"

# Quantum parameters
N_QUBITS=8
N_LAYERS=2
DEGREE=2
N_TIMESTEPS=4
FEATURE_DIM=128
DROPOUT=0.1

# Multi-circuit parameters
N_CHIPS=16
CHUNK_SIZE=8

# RL parameters
BATCH_SIZE=32
GAMMA=0.99
LEARN_STEP=4
NUM_EPISODES=10000
LR=0.00025
EXPLORATION_DECAY=0.9999

# Training parameters
SEED=2025
LOG_INDEX=1
DEVICE="cuda"

echo "========================================================================"
echo "EXPERIMENT CONFIGURATION"
echo "========================================================================"
echo "Environment: $ENV"
echo ""
echo "Quantum Parameters:"
echo "  - N_QUBITS:      $N_QUBITS"
echo "  - N_LAYERS:      $N_LAYERS"
echo "  - DEGREE:        $DEGREE"
echo "  - N_TIMESTEPS:   $N_TIMESTEPS"
echo "  - FEATURE_DIM:   $FEATURE_DIM"
echo ""
echo "Multi-Circuit Parameters:"
echo "  - N_CHIPS:       $N_CHIPS"
echo "  - CHUNK_SIZE:    $CHUNK_SIZE"
echo ""
echo "RL Parameters:"
echo "  - BATCH_SIZE:    $BATCH_SIZE"
echo "  - GAMMA:         $GAMMA"
echo "  - LEARN_STEP:    $LEARN_STEP"
echo "  - NUM_EPISODES:  $NUM_EPISODES"
echo "  - LEARNING_RATE: $LR"
echo ""
echo "Training Parameters:"
echo "  - SEED:          $SEED"
echo "  - LOG_INDEX:     $LOG_INDEX"
echo "  - DEVICE:        $DEVICE"
echo "  - SAVE_DIR:      $CHECKPOINTS_DIR"
echo "========================================================================"
echo ""

# ================================================================================
# Run Training
# ================================================================================

echo "Starting Quantum Transformer V4 (Multi-Circuit) training..."
echo ""

CMD="python ${PROJECT_DIR}/QRL_QTSTransformer/scripts/QuantumTransformerAtari_v4_multichip.py \
    --env=$ENV \
    --n-qubits=$N_QUBITS \
    --n-layers=$N_LAYERS \
    --degree=$DEGREE \
    --n-timesteps=$N_TIMESTEPS \
    --feature-dim=$FEATURE_DIM \
    --dropout=$DROPOUT \
    --n-chips=$N_CHIPS \
    --chunk-size=$CHUNK_SIZE \
    --batch-size=$BATCH_SIZE \
    --gamma=$GAMMA \
    --learn-step=$LEARN_STEP \
    --num-episodes=$NUM_EPISODES \
    --lr=$LR \
    --exploration-rate-decay=$EXPLORATION_DECAY \
    --seed=$SEED \
    --log-index=$LOG_INDEX \
    --device=$DEVICE \
    --save-dir=$CHECKPOINTS_DIR \
    --save-every=50 \
    --resume"

echo "Command: $CMD"
echo ""
eval $CMD

EXIT_CODE=$?

echo ""
echo "========================================================================"
echo "QUANTUM TRANSFORMER V4 (MULTI-CIRCUIT) - ATARI - JOB COMPLETED"
echo "========================================================================"
echo "Exit code: $EXIT_CODE"
echo "End time: $(date)"
echo "========================================================================"

exit $EXIT_CODE
