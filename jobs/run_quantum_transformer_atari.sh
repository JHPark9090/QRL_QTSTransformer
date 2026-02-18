#!/bin/bash
#SBATCH --account=m4807_g
#SBATCH --constraint=gpu&hbm80g
#SBATCH --qos=shared
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=32
#SBATCH --job-name=QTransformerAtari
#SBATCH --output=/pscratch/sd/j/junghoon/QRL_QTSTransformer/logs/atari_%j.out
#SBATCH --error=/pscratch/sd/j/junghoon/QRL_QTSTransformer/logs/atari_%j.err

# ================================================================================
# Quantum Time-Series Transformer for Atari Games
# ================================================================================
# Supports: ALE/Pong-v5, ALE/Breakout-v5, ALE/SpaceInvaders-v5,
#           ALE/Pacman-v5, ALE/DonkeyKong-v5, ALE/MarioBros-v5, ALE/Tetris-v5
# ================================================================================

# ================================================================================
# PROJECT DIRECTORIES (ABSOLUTE PATHS)
# ================================================================================
PROJECT_DIR="/pscratch/sd/j/junghoon/QRL_QTSTransformer"
CONDA_ENV="/pscratch/sd/j/junghoon/conda-envs/qml_env"
LOGS_DIR="${PROJECT_DIR}/logs"
CHECKPOINTS_DIR="${PROJECT_DIR}/checkpoints"

echo "========================================================================"
echo "QUANTUM TRANSFORMER - ATARI GAMES"
echo "========================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "========================================================================"

# Create directories
mkdir -p "$LOGS_DIR"
mkdir -p "$CHECKPOINTS_DIR"

# Change to project directory
cd "$PROJECT_DIR" || exit 1
echo "Working directory: $(pwd)"

# ================================================================================
# CUDA ENVIRONMENT SETUP
# ================================================================================
echo ""
echo "Setting up CUDA environment..."

module load cudatoolkit/12.2

export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=0
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8

# Force unbuffered Python output for real-time SLURM logs
export PYTHONUNBUFFERED=1

# ================================================================================
# CONDA ENVIRONMENT - Use direct Python path for reliability
# ================================================================================
echo ""
echo "Setting up conda environment..."

# Clear any existing conda/python from module system
module unload python 2>/dev/null || true

# Use the full path to Python in the qml_env environment
PYTHON="${CONDA_ENV}/bin/python"

# Verify Python exists
if [ ! -f "$PYTHON" ]; then
    echo "ERROR: Python not found at: $PYTHON"
    exit 1
fi

echo "Using Python: $PYTHON"

# ================================================================================
# VERIFY GPU/CUDA SETUP
# ================================================================================
echo ""
echo "========================================================================"
echo "ENVIRONMENT VERIFICATION"
echo "========================================================================"

$PYTHON << 'EOF'
import torch
import sys

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    props = torch.cuda.get_device_properties(0)
    print(f"GPU Memory: {props.total_memory / 1024**3:.1f} GB")
    x = torch.randn(1000, 1000, device='cuda')
    y = torch.matmul(x, x)
    torch.cuda.synchronize()
    print("✓ CUDA computation test passed!")
else:
    print("ERROR: CUDA not available!")
    sys.exit(1)
EOF

if [ $? -ne 0 ]; then
    echo "ERROR: CUDA verification failed!"
    exit 1
fi

$PYTHON -c "import pennylane as qml; print(f'PennyLane version: {qml.__version__}')"

# Verify Atari environment
$PYTHON << 'EOF'
try:
    import gymnasium as gym
    import ale_py
    gym.register_envs(ale_py)
    print("✓ Gymnasium + ale-py loaded successfully")
except ImportError as e:
    print(f"ERROR: {e}")
    print("Install: pip install gymnasium ale-py gymnasium[atari] gymnasium[accept-rom-license]")
    exit(1)
EOF

if [ $? -ne 0 ]; then
    echo "ERROR: Atari environment verification failed!"
    exit 1
fi

# ================================================================================
# EXPERIMENT CONFIGURATION
# ================================================================================

# Environment - can be overridden via: sbatch --export=ALL,ENV="ALE/Breakout-v5" script.sh
ENV="${ENV:-ALE/Pong-v5}"  # Default: Pong
# Options: ALE/Pong-v5, ALE/Breakout-v5, ALE/SpaceInvaders-v5,
#          ALE/Pacman-v5, ALE/DonkeyKong-v5, ALE/MarioBros-v5, ALE/Tetris-v5

# Quantum parameters
N_QUBITS=8
N_LAYERS=2
DEGREE=2
N_TIMESTEPS=4
FEATURE_DIM=128
DROPOUT=0.1

# RL parameters
BATCH_SIZE=32
GAMMA=0.99
LEARN_STEP=4
NUM_EPISODES=10000
MAX_STEPS=10000
LR=0.0001  # Reduced from 0.00025 for stability
MEMORY_SIZE=100000

# Exploration parameters
EXPLORATION_START=1.0
EXPLORATION_DECAY=0.9999
EXPLORATION_MIN=0.1

# Training parameters
SEED=2025
LOG_INDEX=1
DEVICE="cuda"
SAVE_EVERY=50

# === ADVANCED DQN FEATURES ===
# Prioritized Experience Replay (set to "true" to enable)
USE_PER="${USE_PER:-true}"
PER_ALPHA=0.6
PER_BETA_START=0.4
PER_BETA_FRAMES=100000

# Early Stopping (set to "true" to enable)
EARLY_STOPPING="${EARLY_STOPPING:-true}"
PATIENCE=500
MIN_EPISODES=1000

# Target Network Updates ("hard" or "soft")
TARGET_UPDATE="hard"
TAU=0.001
SYNC_EVERY=1000

echo ""
echo "========================================================================"
echo "EXPERIMENT CONFIGURATION"
echo "========================================================================"
echo "Game:            $ENV"
echo ""
echo "Quantum Parameters:"
echo "  - N_QUBITS:      $N_QUBITS"
echo "  - N_LAYERS:      $N_LAYERS"
echo "  - DEGREE:        $DEGREE"
echo "  - N_TIMESTEPS:   $N_TIMESTEPS"
echo "  - FEATURE_DIM:   $FEATURE_DIM"
echo ""
echo "RL Parameters:"
echo "  - BATCH_SIZE:    $BATCH_SIZE"
echo "  - GAMMA:         $GAMMA"
echo "  - NUM_EPISODES:  $NUM_EPISODES"
echo "  - LEARNING_RATE: $LR"
echo "  - MEMORY_SIZE:   $MEMORY_SIZE"
echo ""
echo "Training Parameters:"
echo "  - SEED:          $SEED"
echo "  - DEVICE:        $DEVICE"
echo "  - SAVE_EVERY:    $SAVE_EVERY episodes"
echo "  - AUTO_RESUME:   Enabled"
echo ""
echo "Advanced DQN Features:"
echo "  - PER:           $USE_PER (α=$PER_ALPHA, β₀=$PER_BETA_START)"
echo "  - Early Stopping: $EARLY_STOPPING (patience=$PATIENCE)"
echo "  - Target Update: $TARGET_UPDATE"
echo ""
echo "Output: $CHECKPOINTS_DIR"
echo "========================================================================"
echo ""

# ================================================================================
# RUN TRAINING
# ================================================================================

echo "Starting Atari training..."
echo ""

# Build command with optional flags
CMD="$PYTHON scripts/QuantumTransformerAtari.py \
    --env=$ENV \
    --n-qubits=$N_QUBITS \
    --n-layers=$N_LAYERS \
    --degree=$DEGREE \
    --n-timesteps=$N_TIMESTEPS \
    --feature-dim=$FEATURE_DIM \
    --dropout=$DROPOUT \
    --batch-size=$BATCH_SIZE \
    --gamma=$GAMMA \
    --learn-step=$LEARN_STEP \
    --num-episodes=$NUM_EPISODES \
    --max-steps=$MAX_STEPS \
    --lr=$LR \
    --memory-size=$MEMORY_SIZE \
    --exploration-rate-start=$EXPLORATION_START \
    --exploration-rate-decay=$EXPLORATION_DECAY \
    --exploration-rate-min=$EXPLORATION_MIN \
    --seed=$SEED \
    --log-index=$LOG_INDEX \
    --device=$DEVICE \
    --save-dir=$CHECKPOINTS_DIR \
    --save-every=$SAVE_EVERY \
    --sync-every=$SYNC_EVERY \
    --target-update=$TARGET_UPDATE \
    --tau=$TAU \
    --patience=$PATIENCE \
    --min-episodes=$MIN_EPISODES \
    --per-alpha=$PER_ALPHA \
    --per-beta-start=$PER_BETA_START \
    --per-beta-frames=$PER_BETA_FRAMES \
    --resume"

# Add optional flags based on settings
if [ "$USE_PER" = "true" ]; then
    CMD="$CMD --use-per"
fi

if [ "$EARLY_STOPPING" = "true" ]; then
    CMD="$CMD --early-stopping"
fi

# Run the command
eval $CMD

EXIT_CODE=$?

# ================================================================================
# JOB SUMMARY
# ================================================================================
echo ""
echo "========================================================================"
echo "QUANTUM TRANSFORMER ATARI - JOB COMPLETED"
echo "========================================================================"
echo "Game: $ENV"
echo "Exit code: $EXIT_CODE"
echo "End time: $(date)"
echo ""
echo "Output files: $CHECKPOINTS_DIR"
echo "========================================================================"

exit $EXIT_CODE
