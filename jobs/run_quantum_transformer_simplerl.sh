#!/bin/bash
#SBATCH --account=m4807_g
#SBATCH --constraint=gpu&hbm80g
#SBATCH --qos=shared
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=32
#SBATCH --job-name=QTransformerSimpleRL
#SBATCH --output=/pscratch/sd/j/junghoon/QRL_QTSTransformer/logs/simplerl_%j.out
#SBATCH --error=/pscratch/sd/j/junghoon/QRL_QTSTransformer/logs/simplerl_%j.err

# ================================================================================
# Quantum Time-Series Transformer for Simple RL Environments
# ================================================================================
# Supports: CartPole-v1, FrozenLake-v1, MountainCar-v0, Acrobot-v1
# ================================================================================

# ================================================================================
# PROJECT DIRECTORIES (ABSOLUTE PATHS)
# ================================================================================
PROJECT_DIR="/pscratch/sd/j/junghoon/QRL_QTSTransformer"
CONDA_ENV="/pscratch/sd/j/junghoon/conda-envs/qml_env"
LOGS_DIR="${PROJECT_DIR}/logs"
CHECKPOINTS_DIR="${PROJECT_DIR}/checkpoints"

echo "========================================================================"
echo "QUANTUM TRANSFORMER - SIMPLE RL ENVIRONMENTS"
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

# Clear any existing conda/python from module system
module unload python 2>/dev/null || true

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

print(f"Python executable: {sys.executable}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    x = torch.randn(100, 100, device='cuda')
    y = torch.matmul(x, x)
    torch.cuda.synchronize()
    print("✓ CUDA computation test passed!")
else:
    print("WARNING: CUDA not available, using CPU")
EOF

$PYTHON -c "import pennylane as qml; print(f'PennyLane version: {qml.__version__}')"
$PYTHON -c "import gymnasium; print(f'Gymnasium version: {gymnasium.__version__}')"

# ================================================================================
# EXPERIMENT CONFIGURATION
# ================================================================================

# Environment - can be overridden via: sbatch --export=ALL,ENV="FrozenLake-v1" script.sh
ENV="${ENV:-CartPole-v1}"  # Default: CartPole-v1
# Options: CartPole-v1, FrozenLake-v1, MountainCar-v0, Acrobot-v1

# Quantum parameters
N_QUBITS=8
N_LAYERS=2
DEGREE=2
N_TIMESTEPS=4
DROPOUT=0.1

# RL parameters - can be overridden via: sbatch --export=ALL,NUM_EPISODES=3000 script.sh
BATCH_SIZE=64
GAMMA=0.99
LEARN_STEP=1
NUM_EPISODES="${NUM_EPISODES:-500}"  # Default: 500, can be overridden
MAX_STEPS=500
LR=0.001
MEMORY_SIZE=10000

# Exploration parameters
EXPLORATION_START=1.0
EXPLORATION_DECAY=0.995
EXPLORATION_MIN=0.01

# Training parameters
SEED=2025
LOG_INDEX=1
DEVICE="cuda"
SAVE_EVERY=10

echo ""
echo "========================================================================"
echo "EXPERIMENT CONFIGURATION"
echo "========================================================================"
echo "Environment:     $ENV"
echo ""
echo "Quantum Parameters:"
echo "  - N_QUBITS:      $N_QUBITS"
echo "  - N_LAYERS:      $N_LAYERS"
echo "  - DEGREE:        $DEGREE"
echo "  - N_TIMESTEPS:   $N_TIMESTEPS"
echo ""
echo "RL Parameters:"
echo "  - BATCH_SIZE:    $BATCH_SIZE"
echo "  - GAMMA:         $GAMMA"
echo "  - NUM_EPISODES:  $NUM_EPISODES"
echo "  - LEARNING_RATE: $LR"
echo ""
echo "Training Parameters:"
echo "  - SEED:          $SEED"
echo "  - DEVICE:        $DEVICE"
echo "  - SAVE_EVERY:    $SAVE_EVERY episodes"
echo "  - AUTO_RESUME:   Enabled"
echo ""
echo "Output: $CHECKPOINTS_DIR"
echo "========================================================================"
echo ""

# ================================================================================
# RUN TRAINING
# ================================================================================

echo "Starting training..."
echo ""

$PYTHON scripts/QuantumTransformerSimpleRL.py \
    --env=$ENV \
    --n-qubits=$N_QUBITS \
    --n-layers=$N_LAYERS \
    --degree=$DEGREE \
    --n-timesteps=$N_TIMESTEPS \
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
    --save-dir="$CHECKPOINTS_DIR" \
    --save-every=$SAVE_EVERY \
    --resume

EXIT_CODE=$?

# ================================================================================
# JOB SUMMARY
# ================================================================================
echo ""
echo "========================================================================"
echo "QUANTUM TRANSFORMER SIMPLE RL - JOB COMPLETED"
echo "========================================================================"
echo "Environment: $ENV"
echo "Exit code: $EXIT_CODE"
echo "End time: $(date)"
echo ""
echo "Output files: $CHECKPOINTS_DIR"
echo "========================================================================"

exit $EXIT_CODE
