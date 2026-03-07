#!/bin/bash
#SBATCH --account=m4807_g
#SBATCH --constraint=gpu&hbm80g
#SBATCH --qos=shared
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=32
#SBATCH --job-name=CTransV6_SimpleRL
#SBATCH --output=QRL_QTSTransformer/logs/classical_v6_%j.out
#SBATCH --error=QRL_QTSTransformer/logs/classical_v6_%j.err

# ================================================================================
# Classical Transformer V6 Baseline for Simple RL
# Matched to quantum v6 with anti-forgetting mechanisms
# ================================================================================
# Example:
#   sbatch --export=ALL,ENV="CartPole-v1" \
#     QRL_QTSTransformer/jobs/run_classical_transformer_simplerl_v6.sh
# ================================================================================

echo "========================================================================"
echo "CLASSICAL TRANSFORMER V6 - SIMPLE RL"
echo "========================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "========================================================================"

PROJECT_DIR="/pscratch/sd/j/junghoon"
mkdir -p "${PROJECT_DIR}/QRL_QTSTransformer/logs"
CHECKPOINTS_DIR="${PROJECT_DIR}/QRL_QTSTransformer/checkpoints"
mkdir -p "${CHECKPOINTS_DIR}"

export PYTHONNOUSERSITE=1

echo "Activating conda environment..."
eval "$(conda shell.bash hook 2>/dev/null)" && conda activate /pscratch/sd/j/junghoon/conda-envs/qml_env

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
ENV="${ENV:-CartPole-v1}"

# Classical transformer parameters
D_MODEL=32
N_HEADS=4
N_TRANSFORMER_LAYERS=1
D_FF=64
N_TIMESTEPS=4
DROPOUT=0.1

# RL parameters (matched to quantum v6)
BATCH_SIZE=64
GAMMA=0.99
LEARN_STEP=1
LR=0.001
EXPLORATION_DECAY=0.995
EXPLORATION_MIN=0.01
MEMORY_SIZE=10000
MAX_STEPS=500

# Anti-forgetting parameters
SYNC_EVERY=500
LR_REDUCE_PATIENCE=50
LR_REDUCE_FACTOR=0.5
LR_MIN=0.00001

# Per-environment settings
case "$ENV" in
    "CartPole-v1")
        NUM_EPISODES=500
        EARLY_STOP_REWARD=490.0
        EARLY_STOP_PATIENCE=20
        ;;
    "FrozenLake-v1")
        NUM_EPISODES=3000
        EARLY_STOP_REWARD=0.70
        EARLY_STOP_PATIENCE=50
        LR_REDUCE_PATIENCE=99999
        ;;
    "MountainCar-v0")
        NUM_EPISODES=2000
        EARLY_STOP_REWARD=-110.0
        EARLY_STOP_PATIENCE=20
        ;;
    "Acrobot-v1")
        NUM_EPISODES=1000
        EARLY_STOP_REWARD=-80.0
        EARLY_STOP_PATIENCE=20
        ;;
    *)
        NUM_EPISODES=1000
        EARLY_STOP_REWARD=490.0
        EARLY_STOP_PATIENCE=20
        ;;
esac

SEED=2025
LOG_INDEX=1
DEVICE="cuda"

echo "========================================================================"
echo "EXPERIMENT CONFIGURATION"
echo "========================================================================"
echo "Environment: $ENV"
echo "Model:       Classical Transformer V6 Baseline"
echo ""
echo "Transformer: d_model=$D_MODEL, heads=$N_HEADS, layers=$N_TRANSFORMER_LAYERS, d_ff=$D_FF"
echo "RL: batch=$BATCH_SIZE, gamma=$GAMMA, lr=$LR, eps_decay=$EXPLORATION_DECAY"
echo "    episodes=$NUM_EPISODES, max_steps=$MAX_STEPS, memory=$MEMORY_SIZE"
echo ""
echo "Anti-forgetting:"
echo "  sync_every=$SYNC_EVERY"
echo "  early_stop: AvgR>=$EARLY_STOP_REWARD for $EARLY_STOP_PATIENCE eps"
echo "  lr_reduce: patience=$LR_REDUCE_PATIENCE, factor=$LR_REDUCE_FACTOR, min=$LR_MIN"
echo "========================================================================"
echo ""

# ================================================================================
# Run Training
# ================================================================================
echo "Starting Classical Transformer V6 SimpleRL training..."
echo ""

CMD="python ${PROJECT_DIR}/QRL_QTSTransformer/scripts/ClassicalTransformerSimpleRL_v6.py \
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
    --early-stop-reward=$EARLY_STOP_REWARD \
    --early-stop-patience=$EARLY_STOP_PATIENCE \
    --lr-reduce-patience=$LR_REDUCE_PATIENCE \
    --lr-reduce-factor=$LR_REDUCE_FACTOR \
    --lr-min=$LR_MIN \
    --sync-every=$SYNC_EVERY \
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
echo "CLASSICAL TRANSFORMER V6 SIMPLERL - JOB COMPLETED"
echo "========================================================================"
echo "Exit code: $EXIT_CODE"
echo "End time: $(date)"
echo "========================================================================"

exit $EXIT_CODE
