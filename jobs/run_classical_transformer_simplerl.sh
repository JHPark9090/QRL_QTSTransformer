#!/bin/bash
#SBATCH --account=m4807_g
#SBATCH --constraint=gpu&hbm80g
#SBATCH --qos=shared
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=32
#SBATCH --job-name=CTransformer_SimpleRL
#SBATCH --output=QRL_QTSTransformer/logs/classical_simplerl_%j.out
#SBATCH --error=QRL_QTSTransformer/logs/classical_simplerl_%j.err

# ================================================================================
# Classical Transformer Baseline for Simple RL Environments
# ================================================================================
# Environments: CartPole-v1, FrozenLake-v1, MountainCar-v0, Acrobot-v1
#
# Example submissions:
#   sbatch --export=ALL,ENV="CartPole-v1" \
#     QRL_QTSTransformer/jobs/run_classical_transformer_simplerl.sh
#
#   sbatch --export=ALL,ENV="FrozenLake-v1" \
#     QRL_QTSTransformer/jobs/run_classical_transformer_simplerl.sh
# ================================================================================

echo "========================================================================"
echo "CLASSICAL TRANSFORMER - SIMPLE RL"
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

# ================================================================================
# Experiment Configuration
# ================================================================================

# Environment (can be overridden via --export=ALL,ENV="CartPole-v1")
ENV="${ENV:-CartPole-v1}"

# Classical transformer parameters
D_MODEL=32
N_HEADS=4
N_TRANSFORMER_LAYERS=1
D_FF=64
N_TIMESTEPS=4
DROPOUT=0.1

# RL parameters (matched to quantum v5 SimpleRL)
BATCH_SIZE=64
GAMMA=0.99
LEARN_STEP=1
LR=0.001
EXPLORATION_DECAY=0.995
EXPLORATION_MIN=0.01
MEMORY_SIZE=10000
MAX_STEPS=500

# Episode count per environment (matched to quantum)
case "$ENV" in
    "CartPole-v1")
        NUM_EPISODES=1000
        ;;
    "FrozenLake-v1")
        NUM_EPISODES=3000
        ;;
    "MountainCar-v0")
        NUM_EPISODES=2000
        ;;
    "Acrobot-v1")
        NUM_EPISODES=1000
        ;;
    *)
        NUM_EPISODES=1000
        ;;
esac

# Training parameters
SEED=2025
LOG_INDEX=1
DEVICE="cuda"

echo "========================================================================"
echo "EXPERIMENT CONFIGURATION"
echo "========================================================================"
echo "Environment: $ENV"
echo "Model:       Classical Transformer Baseline"
echo ""
echo "Transformer Parameters:"
echo "  - D_MODEL:          $D_MODEL"
echo "  - N_HEADS:          $N_HEADS"
echo "  - N_LAYERS:         $N_TRANSFORMER_LAYERS"
echo "  - D_FF:             $D_FF"
echo "  - N_TIMESTEPS:      $N_TIMESTEPS"
echo ""
echo "RL Parameters:"
echo "  - BATCH_SIZE:       $BATCH_SIZE"
echo "  - GAMMA:            $GAMMA"
echo "  - LEARN_STEP:       $LEARN_STEP"
echo "  - NUM_EPISODES:     $NUM_EPISODES"
echo "  - LEARNING_RATE:    $LR"
echo "  - EPS_DECAY:        $EXPLORATION_DECAY"
echo "  - EPS_MIN:          $EXPLORATION_MIN"
echo "  - MEMORY_SIZE:      $MEMORY_SIZE"
echo "  - MAX_STEPS:        $MAX_STEPS"
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

echo "Starting Classical Transformer SimpleRL training..."
echo ""

CMD="python ${PROJECT_DIR}/QRL_QTSTransformer/scripts/ClassicalTransformerSimpleRL.py \
    --env=$ENV \
    --d-model=$D_MODEL \
    --n-heads=$N_HEADS \
    --n-transformer-layers=$N_TRANSFORMER_LAYERS \
    --d-ff=$D_FF \
    --n-timesteps=$N_TIMESTEPS \
    --dropout=$DROPOUT \
    --batch-size=$BATCH_SIZE \
    --gamma=$GAMMA \
    --learn-step=$LEARN_STEP \
    --num-episodes=$NUM_EPISODES \
    --max-steps=$MAX_STEPS \
    --lr=$LR \
    --exploration-rate-decay=$EXPLORATION_DECAY \
    --exploration-rate-min=$EXPLORATION_MIN \
    --memory-size=$MEMORY_SIZE \
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
echo "CLASSICAL TRANSFORMER SIMPLERL - JOB COMPLETED"
echo "========================================================================"
echo "Exit code: $EXIT_CODE"
echo "End time: $(date)"
echo "========================================================================"

exit $EXIT_CODE
