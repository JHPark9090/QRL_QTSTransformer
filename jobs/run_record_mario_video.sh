#!/bin/bash
#SBATCH --account=m4138_g
#SBATCH --constraint=gpu&hbm80g
#SBATCH --qos=shared
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=16
#SBATCH --job-name=record_quantum_mario
#SBATCH --output=logs/record_mario_%j.out
#SBATCH --error=logs/record_mario_%j.err

# ================================================================================
# Record Video of Quantum RL Agent Playing Super Mario Bros
# ================================================================================
# This script loads a trained quantum RL agent and records gameplay videos.
#
# Usage:
#   sbatch run_record_mario_video.sh
#
# Modify the AGENT_TYPE variable below to choose which agent to record.
# ================================================================================

echo "========================================================================"
echo "QUANTUM MARIO VIDEO RECORDING - SLURM JOB"
echo "========================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "========================================================================"

# Create logs directory if it doesn't exist
mkdir -p logs

# Activate conda environment
echo "Activating conda environment..."
module load python
source activate ./conda-envs/qml_env

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
# CONFIGURATION
# ================================================================================

# Choose which agent to record:
# Option 1: QCNN-based agent (QuantumSuperMario.py)
# Option 2: Transformer-based agent (QuantumTransformerMario.py)

AGENT_TYPE="QCNN"  # Change to "TRANSFORMER" for transformer agent

# Checkpoint directory (modify as needed)
if [ "$AGENT_TYPE" == "QCNN" ]; then
    CHECKPOINT_DIR="SuperMarioCheckpoints/QuantumMario_Run16"
    SCRIPT="record_quantum_mario_video.py"
    OUTPUT_DIR="mario_videos_qcnn"
elif [ "$AGENT_TYPE" == "TRANSFORMER" ]; then
    CHECKPOINT_DIR="SuperMarioCheckpoints/QTransformerMario_Q6_L2_D2_Run1"
    SCRIPT="record_quantum_mario_video_TRANSFORMER.py"
    OUTPUT_DIR="mario_videos_transformer"

    # Quantum parameters (must match training configuration)
    N_QUBITS=6
    N_LAYERS=2
    DEGREE=2
    FEATURE_DIM=128
else
    echo "ERROR: Invalid AGENT_TYPE. Choose 'QCNN' or 'TRANSFORMER'"
    exit 1
fi

# Recording parameters
NUM_EPISODES=5       # Number of episodes to record
FPS=30              # Video frames per second
DEVICE="cuda"       # Use "cpu" if GPU not available

echo "========================================================================"
echo "CONFIGURATION"
echo "========================================================================"
echo "Agent Type:         $AGENT_TYPE"
echo "Checkpoint:         $CHECKPOINT_DIR"
echo "Script:             $SCRIPT"
echo "Output Directory:   $OUTPUT_DIR"
echo "Num Episodes:       $NUM_EPISODES"
echo "FPS:                $FPS"
echo "Device:             $DEVICE"
echo "========================================================================"
echo ""

# ================================================================================
# RUN VIDEO RECORDING
# ================================================================================

echo "Starting video recording..."
echo ""

if [ "$AGENT_TYPE" == "QCNN" ]; then
    # Record with QCNN agent
    python $SCRIPT \
        --checkpoint-dir=$CHECKPOINT_DIR \
        --num-episodes=$NUM_EPISODES \
        --output-dir=$OUTPUT_DIR \
        --fps=$FPS \
        --device=$DEVICE

elif [ "$AGENT_TYPE" == "TRANSFORMER" ]; then
    # Record with Transformer agent
    python $SCRIPT \
        --checkpoint-dir=$CHECKPOINT_DIR \
        --num-episodes=$NUM_EPISODES \
        --output-dir=$OUTPUT_DIR \
        --fps=$FPS \
        --device=$DEVICE \
        --n-qubits=$N_QUBITS \
        --n-layers=$N_LAYERS \
        --degree=$DEGREE \
        --feature-dim=$FEATURE_DIM
fi

EXIT_CODE=$?

echo ""
echo "========================================================================"
echo "VIDEO RECORDING - JOB COMPLETED"
echo "========================================================================"
echo "Exit code: $EXIT_CODE"
echo "End time: $(date)"

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✓ SUCCESS! Videos saved to: $OUTPUT_DIR/"
    echo ""
    echo "To view videos:"
    echo "  1. List videos:"
    echo "     ls -lh $OUTPUT_DIR/"
    echo ""
    echo "  2. Copy to local machine:"
    echo "     scp junghoon@perlmutter.nersc.gov:/pscratch/sd/j/junghoon/$OUTPUT_DIR/*.mp4 ~/Downloads/"
    echo ""
    echo "  3. Or use Globus transfer for large files"
else
    echo ""
    echo "✗ ERROR: Video recording failed (exit code: $EXIT_CODE)"
    echo "Check logs/record_mario_$SLURM_JOB_ID.err for details"
fi

echo "========================================================================"

exit $EXIT_CODE
