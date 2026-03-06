#!/bin/bash
# ================================================================================
# Submit all SimpleRL v5 ablation experiments for CartPole and FrozenLake
# ================================================================================
# This submits 10 jobs total:
#   - 4 quantum conditions x 2 environments = 8 quantum jobs
#   - 1 classical baseline x 2 environments = 2 classical jobs
# ================================================================================

echo "Submitting SimpleRL v5 Ablation Study"
echo "======================================"
echo ""

# Quantum v5 ablation conditions for CartPole
echo "--- CartPole-v1 ---"
for ABLATION in "baseline" "ano_only" "dqas_only" "full"; do
    JOB_ID=$(sbatch --export=ALL,ABLATION="$ABLATION",ENV="CartPole-v1" \
        QRL_QTSTransformer/jobs/run_quantum_transformer_simplerl_v5.sh | awk '{print $4}')
    echo "  Quantum [$ABLATION]: Job $JOB_ID"
done

# Classical baseline for CartPole
JOB_ID=$(sbatch --export=ALL,ENV="CartPole-v1" \
    QRL_QTSTransformer/jobs/run_classical_transformer_simplerl.sh | awk '{print $4}')
echo "  Classical: Job $JOB_ID"

echo ""

# Quantum v5 ablation conditions for FrozenLake
echo "--- FrozenLake-v1 ---"
for ABLATION in "baseline" "ano_only" "dqas_only" "full"; do
    JOB_ID=$(sbatch --export=ALL,ABLATION="$ABLATION",ENV="FrozenLake-v1" \
        QRL_QTSTransformer/jobs/run_quantum_transformer_simplerl_v5.sh | awk '{print $4}')
    echo "  Quantum [$ABLATION]: Job $JOB_ID"
done

# Classical baseline for FrozenLake
JOB_ID=$(sbatch --export=ALL,ENV="FrozenLake-v1" \
    QRL_QTSTransformer/jobs/run_classical_transformer_simplerl.sh | awk '{print $4}')
echo "  Classical: Job $JOB_ID"

echo ""
echo "======================================"
echo "Total: 10 jobs submitted"
echo "  - 8 quantum (4 ablation conditions x 2 environments)"
echo "  - 2 classical baselines"
echo ""
echo "Monitor with: squeue -u $USER"
