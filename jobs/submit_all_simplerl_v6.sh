#!/bin/bash
# ================================================================================
# Submit all SimpleRL v6 ablation experiments for CartPole and FrozenLake
# ================================================================================
# 10 jobs total:
#   - 4 quantum conditions x 2 environments = 8 quantum jobs
#   - 1 classical baseline x 2 environments = 2 classical jobs
# ================================================================================

echo "Submitting SimpleRL v6 Ablation Study"
echo "======================================"
echo "v6 = v1 base + ANO/DiffQAS + anti-forgetting"
echo ""

# CartPole
echo "--- CartPole-v1 ---"
for ABLATION in "baseline" "ano_only" "dqas_only" "full"; do
    JOB_ID=$(sbatch --export=ALL,ABLATION="$ABLATION",ENV="CartPole-v1" \
        QRL_QTSTransformer/jobs/run_quantum_transformer_simplerl_v6.sh | awk '{print $4}')
    echo "  Quantum [$ABLATION]: Job $JOB_ID"
done

JOB_ID=$(sbatch --export=ALL,ENV="CartPole-v1" \
    QRL_QTSTransformer/jobs/run_classical_transformer_simplerl_v6.sh | awk '{print $4}')
echo "  Classical: Job $JOB_ID"

echo ""

# FrozenLake
echo "--- FrozenLake-v1 ---"
for ABLATION in "baseline" "ano_only" "dqas_only" "full"; do
    JOB_ID=$(sbatch --export=ALL,ABLATION="$ABLATION",ENV="FrozenLake-v1" \
        QRL_QTSTransformer/jobs/run_quantum_transformer_simplerl_v6.sh | awk '{print $4}')
    echo "  Quantum [$ABLATION]: Job $JOB_ID"
done

JOB_ID=$(sbatch --export=ALL,ENV="FrozenLake-v1" \
    QRL_QTSTransformer/jobs/run_classical_transformer_simplerl_v6.sh | awk '{print $4}')
echo "  Classical: Job $JOB_ID"

echo ""
echo "======================================"
echo "Total: 10 jobs submitted"
echo "  - 8 quantum (4 ablation conditions x 2 environments)"
echo "  - 2 classical baselines"
echo ""
echo "Monitor with: squeue -u $USER"
