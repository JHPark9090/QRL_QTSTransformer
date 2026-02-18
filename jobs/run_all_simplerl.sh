#!/bin/bash
# ================================================================================
# Run All Simple RL Environments Sequentially
# ================================================================================
# This script submits separate SLURM jobs for each environment
# Usage: bash run_all_simplerl.sh
# ================================================================================

PROJECT_DIR="/pscratch/sd/j/junghoon/QRL_QTSTransformer"
cd "$PROJECT_DIR" || exit 1

echo "========================================================================"
echo "SUBMITTING ALL SIMPLE RL ENVIRONMENTS"
echo "========================================================================"
echo ""

# Common parameters
N_QUBITS=8
N_LAYERS=2
SEED=2025

# Submit CartPole
echo "Submitting CartPole-v1..."
sbatch --export=ALL,ENV="CartPole-v1" run_quantum_transformer_simplerl.sh
echo ""

# For other environments, you can create separate submission or modify the script
echo "To run other environments, edit run_quantum_transformer_simplerl.sh"
echo "and change the ENV variable to:"
echo "  - FrozenLake-v1"
echo "  - MountainCar-v0"
echo "  - Acrobot-v1"
echo ""
echo "Or run interactively:"
echo "  python QuantumTransformerSimpleRL.py --env=FrozenLake-v1 --n-qubits=8"
echo ""

echo "========================================================================"
echo "Check job status: squeue -u \$USER"
echo "========================================================================"
