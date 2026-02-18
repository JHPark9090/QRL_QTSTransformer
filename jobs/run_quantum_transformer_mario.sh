#!/bin/bash
#SBATCH --account=m4807_g
#SBATCH --constraint=gpu&hbm80g
#SBATCH --qos=shared
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=32
#SBATCH --job-name=QTransformerMario
#SBATCH --output=/pscratch/sd/j/junghoon/QRL_QTSTransformer/logs/quantum_transformer_mario_%j.out
#SBATCH --error=/pscratch/sd/j/junghoon/QRL_QTSTransformer/logs/quantum_transformer_mario_%j.err

# ================================================================================
# Quantum Time-Series Transformer for Super Mario Bros RL
# ================================================================================
# This script runs the quantum transformer-based RL agent on Super Mario Bros
# using hybrid quantum-classical neural networks with QSVT.
#
# Key Features:
# - CNN feature extraction from game frames
# - Quantum time-series transformer for temporal processing
# - Double DQN with experience replay
# - Automatic checkpointing and resumption
# ================================================================================

# ================================================================================
# PROJECT DIRECTORIES (ABSOLUTE PATHS)
# ================================================================================
PROJECT_DIR="/pscratch/sd/j/junghoon/QRL_QTSTransformer"
CONDA_ENV="/pscratch/sd/j/junghoon/conda-envs/qml_env"
LOGS_DIR="${PROJECT_DIR}/logs"
RESULTS_DIR="${PROJECT_DIR}/results"
CHECKPOINTS_DIR="${PROJECT_DIR}/checkpoints"

echo "========================================================================"
echo "QUANTUM TRANSFORMER MARIO - SLURM JOB STARTING"
echo "========================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "Project Directory: $PROJECT_DIR"
echo "========================================================================"

# Create directories
mkdir -p "$LOGS_DIR"
mkdir -p "$RESULTS_DIR"
mkdir -p "$CHECKPOINTS_DIR"

# Change to project directory
cd "$PROJECT_DIR" || exit 1
echo "Working directory: $(pwd)"

# ================================================================================
# CUDA ENVIRONMENT SETUP
# ================================================================================
echo ""
echo "Setting up CUDA environment..."

# Load CUDA module (NERSC Perlmutter)
module load cudatoolkit/12.2

# CUDA environment variables for optimal GPU usage
export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=0
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# PyTorch CUDA memory management
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Disable TensorFloat-32 for reproducibility (optional, comment out for speed)
# export NVIDIA_TF32_OVERRIDE=0

# Threading optimization
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8

# Force unbuffered Python output for real-time SLURM logs
export PYTHONUNBUFFERED=1

echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "PYTORCH_CUDA_ALLOC_CONF: $PYTORCH_CUDA_ALLOC_CONF"

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
echo "Python: $PYTHON"
echo "Python version: $($PYTHON --version)"

# Verify PyTorch and CUDA
$PYTHON << 'EOF'
import torch
import sys

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")
    print(f"GPU count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        props = torch.cuda.get_device_properties(i)
        print(f"    Memory: {props.total_memory / 1024**3:.1f} GB")
        print(f"    Compute capability: {props.major}.{props.minor}")

    # Test CUDA with a simple operation
    print("\nTesting CUDA computation...")
    x = torch.randn(1000, 1000, device='cuda')
    y = torch.matmul(x, x)
    torch.cuda.synchronize()
    print("✓ CUDA computation test passed!")
else:
    print("ERROR: CUDA is NOT available!")
    print("Training will fall back to CPU (very slow)")
    sys.exit(1)
EOF

if [ $? -ne 0 ]; then
    echo "ERROR: CUDA verification failed!"
    exit 1
fi

# Verify PennyLane and gym
echo ""
$PYTHON -c "import pennylane as qml; print(f'PennyLane version: {qml.__version__}')"
$PYTHON -c "import gym; print(f'Gym version: {gym.__version__}')"
$PYTHON -c "import gym_super_mario_bros; print('✓ gym_super_mario_bros available')"

# ================================================================================
# EXPERIMENT CONFIGURATION
# ================================================================================

# Quantum parameters
N_QUBITS=8
N_LAYERS=2
DEGREE=2
N_TIMESTEPS=4
FEATURE_DIM=128
DROPOUT=0.1

# RL parameters
BATCH_SIZE=32
GAMMA=0.9
LEARN_STEP=3
NUM_EPISODES=40000
LR=0.00025
EXPLORATION_DECAY=0.99999975

# Training parameters
SEED=2025
LOG_INDEX=1
DEVICE="cuda"
SAVE_EVERY=5  # Save checkpoint every N episodes (for fault tolerance)

echo ""
echo "========================================================================"
echo "EXPERIMENT CONFIGURATION"
echo "========================================================================"
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
echo "  - LEARN_STEP:    $LEARN_STEP"
echo "  - NUM_EPISODES:  $NUM_EPISODES"
echo "  - LEARNING_RATE: $LR"
echo ""
echo "Training Parameters:"
echo "  - SEED:          $SEED"
echo "  - LOG_INDEX:     $LOG_INDEX"
echo "  - DEVICE:        $DEVICE"
echo "  - SAVE_EVERY:    $SAVE_EVERY episodes"
echo "  - AUTO_RESUME:   Enabled"
echo ""
echo "Output Directories:"
echo "  - Checkpoints:   $CHECKPOINTS_DIR"
echo "  - Logs:          $LOGS_DIR"
echo "  - Results:       $RESULTS_DIR"
echo "========================================================================"
echo ""

# ================================================================================
# RUN TRAINING
# ================================================================================

echo "Starting training..."
echo ""

$PYTHON scripts/QuantumTransformerMario.py \
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
    --lr=$LR \
    --exploration-rate-decay=$EXPLORATION_DECAY \
    --seed=$SEED \
    --log-index=$LOG_INDEX \
    --device=$DEVICE \
    --save-dir="$CHECKPOINTS_DIR" \
    --save-every=$SAVE_EVERY \
    --resume  # Auto-resume from checkpoint if exists

EXIT_CODE=$?

# ================================================================================
# JOB SUMMARY
# ================================================================================
echo ""
echo "========================================================================"
echo "QUANTUM TRANSFORMER MARIO - JOB COMPLETED"
echo "========================================================================"
echo "Exit code: $EXIT_CODE"
echo "End time: $(date)"
echo ""
echo "Output files saved to:"
echo "  - Checkpoints: $CHECKPOINTS_DIR"
echo "  - SLURM logs:  $LOGS_DIR"
echo "========================================================================"

# List generated files
echo ""
echo "Generated files:"
ls -la "$CHECKPOINTS_DIR" 2>/dev/null || echo "  (no checkpoints yet)"

exit $EXIT_CODE
