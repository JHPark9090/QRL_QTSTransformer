#!/bin/bash
#SBATCH --account=m4807_g
#SBATCH --constraint=gpu&hbm80g
#SBATCH --qos=shared
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=32
#SBATCH --job-name=QTransV6_SimpleRL
#SBATCH --output=QRL_QTSTransformer/logs/simplerl_v6_%j.out
#SBATCH --error=QRL_QTSTransformer/logs/simplerl_v6_%j.err

# ================================================================================
# Quantum Time-Series Transformer V6 for Simple RL
# v1 base + ANO/DiffQAS ablation + anti-forgetting
# ================================================================================
# Ablation: ABLATION = "baseline" | "ano_only" | "dqas_only" | "full"
# Environment: ENV = "CartPole-v1" | "FrozenLake-v1" | ...
#
# Example:
#   sbatch --export=ALL,ABLATION="full",ENV="CartPole-v1" \
#     QRL_QTSTransformer/jobs/run_quantum_transformer_simplerl_v6.sh
# ================================================================================

echo "========================================================================"
echo "QUANTUM TRANSFORMER V6 - SIMPLE RL"
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
python -c "import scipy.constants" 2>/dev/null

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

# ================================================================================
# Experiment Configuration
# ================================================================================
ENV="${ENV:-CartPole-v1}"
ABLATION="${ABLATION:-full}"

# Quantum parameters (matched to v1)
N_QUBITS=8
N_LAYERS=2
DEGREE=2
N_TIMESTEPS=4
DROPOUT=0.1

# ANO parameters
ANO_K_LOCAL=2
ANO_LR=0.01

# DiffQAS parameters
ARCH_LR=0.001
SEARCH_EPISODES=50

# RL parameters (matched to v1)
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
LR_REDUCE_PATIENCE="${LR_REDUCE_PATIENCE:-50}"
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
LOG_INDEX="${LOG_INDEX:-1}"
DEVICE="cuda"

# Build ablation flags
ABLATION_FLAGS=""
case "$ABLATION" in
    "baseline")
        ABLATION_FLAGS="--no-ano --no-dqas"
        ABLATION_LABEL="v1 Baseline (no ANO, no DiffQAS)"
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
        ABLATION_LABEL="Full v6 (ANO + DiffQAS)"
        ;;
esac

echo "========================================================================"
echo "EXPERIMENT CONFIGURATION"
echo "========================================================================"
echo "Environment: $ENV"
echo "Ablation:    $ABLATION_LABEL"
echo "Base:        v1 (sigmoid [0,1], MSELoss, no PE, no grad clip)"
echo ""
echo "Quantum: Q=$N_QUBITS, L=$N_LAYERS, D=$DEGREE, T=$N_TIMESTEPS"
echo "ANO:    k=$ANO_K_LOCAL, lr=$ANO_LR"
echo "DiffQAS: arch_lr=$ARCH_LR, search_ep=$SEARCH_EPISODES"
echo ""
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
echo "Starting Quantum Transformer V6 SimpleRL training ($ABLATION_LABEL)..."
echo ""

CMD="python ${PROJECT_DIR}/QRL_QTSTransformer/scripts/QuantumTransformerSimpleRL_v6.py \
    --env=$ENV \
    --n-qubits=$N_QUBITS \
    --n-layers=$N_LAYERS \
    --degree=$DEGREE \
    --n-timesteps=$N_TIMESTEPS \
    --dropout=$DROPOUT \
    --ano-k-local=$ANO_K_LOCAL \
    --ano-lr=$ANO_LR \
    --arch-lr=$ARCH_LR \
    --search-episodes=$SEARCH_EPISODES \
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
    --resume \
    $ABLATION_FLAGS"

echo "Command: $CMD"
echo ""
eval $CMD

EXIT_CODE=$?

echo ""
echo "========================================================================"
echo "QUANTUM TRANSFORMER V6 SIMPLERL ($ABLATION_LABEL) - JOB COMPLETED"
echo "========================================================================"
echo "Exit code: $EXIT_CODE"
echo "End time: $(date)"
echo "========================================================================"

exit $EXIT_CODE
