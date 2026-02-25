#!/bin/bash
#SBATCH --account=m4807_g
#SBATCH --constraint=gpu&hbm80g
#SBATCH --qos=shared
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=32
#SBATCH --job-name=CTransformerAtari
#SBATCH --output=QRL_QTSTransformer/logs/classical_atari_%j.out
#SBATCH --error=QRL_QTSTransformer/logs/classical_atari_%j.err

# ================================================================================
# Classical Transformer Baseline for Atari RL
# ================================================================================
# Matched classical baseline for Quantum Transformer v3:
#   - Same CNN feature extractor (Nature DQN)
#   - Same replay buffer, DQN training loop, preprocessing
#   - nn.TransformerEncoder replaces quantum QSVT/QFF
# ================================================================================

echo "========================================================================"
echo "CLASSICAL TRANSFORMER BASELINE - ATARI GAMES"
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

# Verify environment
echo ""
echo "Environment Information:"
echo "Python: $(which python)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
if python -c 'import torch; print(torch.cuda.is_available())' | grep -q "True"; then
    echo "GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0))')"
fi
echo ""

# Optional: Set CUDA memory management for large models
# export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256

# ================================================================================
# Experiment Configuration
# ================================================================================

# Environment (can be overridden via --export=ALL,ENV="ALE/SpaceInvaders-v5")
ENV="${ENV:-ALE/SpaceInvaders-v5}"

# Transformer parameters
D_MODEL=128
N_HEADS=4
N_TRANSFORMER_LAYERS=1
D_FF=256
N_TIMESTEPS=4
FEATURE_DIM=128
DROPOUT=0.1

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
echo "Transformer Parameters:"
echo "  - D_MODEL:       $D_MODEL"
echo "  - N_HEADS:       $N_HEADS"
echo "  - N_LAYERS:      $N_TRANSFORMER_LAYERS"
echo "  - D_FF:          $D_FF"
echo "  - N_TIMESTEPS:   $N_TIMESTEPS"
echo "  - FEATURE_DIM:   $FEATURE_DIM"
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

echo "Starting Classical Transformer Baseline training..."
echo ""

CMD="python ${PROJECT_DIR}/QRL_QTSTransformer/scripts/ClassicalTransformerAtari.py \
    --env=$ENV \
    --d-model=$D_MODEL \
    --n-heads=$N_HEADS \
    --n-transformer-layers=$N_TRANSFORMER_LAYERS \
    --d-ff=$D_FF \
    --n-timesteps=$N_TIMESTEPS \
    --feature-dim=$FEATURE_DIM \
    --dropout=$DROPOUT \
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
echo "CLASSICAL TRANSFORMER BASELINE - ATARI - JOB COMPLETED"
echo "========================================================================"
echo "Exit code: $EXIT_CODE"
echo "End time: $(date)"
echo "========================================================================"

exit $EXIT_CODE
