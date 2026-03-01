#!/bin/bash
#SBATCH --account=m4807_g
#SBATCH --constraint=gpu&hbm80g
#SBATCH --qos=shared
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=32
#SBATCH --job-name=QTransV5_ANO_DQAS
#SBATCH --output=QRL_QTSTransformer/logs/atari_v5_%j.out
#SBATCH --error=QRL_QTSTransformer/logs/atari_v5_%j.err

# ================================================================================
# Quantum Time-Series Transformer V5 for Atari RL (ANO + DiffQAS)
# ================================================================================
# v5 innovations:
#   1. ANO (Adaptive Non-Local Observables): Learnable k-local Hermitian measurements
#   2. DiffQAS (Differentiable Quantum Architecture Search): Gate-level parametric search
#   3. Two-phase training: search (100 ep) -> production (~9900 ep)
#
# Ablation study — set ABLATION to control which features are active:
#   "full"      : ANO + DiffQAS (default)
#   "ano_only"  : ANO only (fixed RY-CRX circuit)
#   "dqas_only" : DiffQAS only (fixed PauliX/Y/Z measurements)
#   "baseline"  : Neither (reproduces v3 behavior)
#
# Example submission for specific ablation:
#   sbatch --export=ALL,ABLATION="baseline" run_quantum_transformer_atari_v5.sh
#   sbatch --export=ALL,ABLATION="ano_only" run_quantum_transformer_atari_v5.sh
#   sbatch --export=ALL,ABLATION="dqas_only" run_quantum_transformer_atari_v5.sh
#   sbatch --export=ALL,ABLATION="full" run_quantum_transformer_atari_v5.sh
# ================================================================================

echo "========================================================================"
echo "QUANTUM TRANSFORMER V5 - ATARI GAMES"
echo "========================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "========================================================================"

# Project directory
PROJECT_DIR="/pscratch/sd/j/junghoon"

# Create logs directory if it doesn't exist
mkdir -p "${PROJECT_DIR}/QRL_QTSTransformer/logs"

# Checkpoints directory (v5)
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

# Ablation condition (can be overridden via --export=ALL,ABLATION="baseline")
ABLATION="${ABLATION:-full}"

# Quantum parameters
N_QUBITS=8
N_LAYERS=2
DEGREE=2
N_TIMESTEPS=4
FEATURE_DIM=128
DROPOUT=0.1

# ANO parameters (v5)
ANO_K_LOCAL=2
ANO_LR=0.01

# DiffQAS parameters (v5)
ARCH_LR=0.001
SEARCH_EPISODES=100

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

# Build ablation flags
ABLATION_FLAGS=""
case "$ABLATION" in
    "baseline")
        ABLATION_FLAGS="--no-ano --no-dqas"
        ABLATION_LABEL="v3 Baseline (no ANO, no DiffQAS)"
        ;;
    "ano_only")
        ABLATION_FLAGS="--no-dqas"
        ABLATION_LABEL="ANO Only (no DiffQAS)"
        ;;
    "dqas_only")
        ABLATION_FLAGS="--no-ano"
        ABLATION_LABEL="DiffQAS Only (no ANO)"
        ;;
    "full"|*)
        ABLATION_FLAGS=""
        ABLATION_LABEL="Full v5 (ANO + DiffQAS)"
        ;;
esac

echo "========================================================================"
echo "EXPERIMENT CONFIGURATION"
echo "========================================================================"
echo "Environment: $ENV"
echo "Ablation:    $ABLATION_LABEL"
echo ""
echo "Quantum Parameters:"
echo "  - N_QUBITS:         $N_QUBITS"
echo "  - N_LAYERS:         $N_LAYERS"
echo "  - DEGREE:           $DEGREE"
echo "  - N_TIMESTEPS:      $N_TIMESTEPS"
echo "  - FEATURE_DIM:      $FEATURE_DIM"
echo ""
echo "ANO Parameters (v5):"
echo "  - ANO_K_LOCAL:      $ANO_K_LOCAL"
echo "  - ANO_LR:           $ANO_LR"
echo ""
echo "DiffQAS Parameters (v5):"
echo "  - ARCH_LR:          $ARCH_LR"
echo "  - SEARCH_EPISODES:  $SEARCH_EPISODES"
echo ""
echo "RL Parameters:"
echo "  - BATCH_SIZE:       $BATCH_SIZE"
echo "  - GAMMA:            $GAMMA"
echo "  - LEARN_STEP:       $LEARN_STEP"
echo "  - NUM_EPISODES:     $NUM_EPISODES"
echo "  - LEARNING_RATE:    $LR"
echo ""
echo "Training Parameters:"
echo "  - SEED:             $SEED"
echo "  - LOG_INDEX:        $LOG_INDEX"
echo "  - DEVICE:           $DEVICE"
echo "  - SAVE_DIR:         $CHECKPOINTS_DIR"
echo "========================================================================"
echo ""

# ================================================================================
# Run Training
# ================================================================================

echo "Starting Quantum Transformer V5 training ($ABLATION_LABEL)..."
echo ""

CMD="python ${PROJECT_DIR}/QRL_QTSTransformer/scripts/QuantumTransformerAtari_v5_ano_dqas.py \
    --env=$ENV \
    --n-qubits=$N_QUBITS \
    --n-layers=$N_LAYERS \
    --degree=$DEGREE \
    --n-timesteps=$N_TIMESTEPS \
    --feature-dim=$FEATURE_DIM \
    --dropout=$DROPOUT \
    --ano-k-local=$ANO_K_LOCAL \
    --ano-lr=$ANO_LR \
    --arch-lr=$ARCH_LR \
    --search-episodes=$SEARCH_EPISODES \
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
    --resume \
    $ABLATION_FLAGS"

echo "Command: $CMD"
echo ""
eval $CMD

EXIT_CODE=$?

echo ""
echo "========================================================================"
echo "QUANTUM TRANSFORMER V5 ($ABLATION_LABEL) - JOB COMPLETED"
echo "========================================================================"
echo "Exit code: $EXIT_CODE"
echo "End time: $(date)"
echo "========================================================================"

exit $EXIT_CODE
