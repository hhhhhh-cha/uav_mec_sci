## 主对比 10 train-seed 脚本

#!/usr/bin/env bash
set -euo pipefail

cd ~/projects/uav_mec_sci
source .venv/bin/activate

export PYTHONPATH=$(pwd)

TRAIN_SEEDS=(42 52 62 72 82 92 102 112 122 132)
EVAL_SEEDS="42,52,62,72,82,92,102,112,122,132"

mkdir -p logs/main_comparison_formal10_with_pure_refine
mkdir -p results/main_comparison_sci_v6/formal10_trainseed_eval10_with_pure_refine

for TRAIN_SEED in "${TRAIN_SEEDS[@]}"
do
  echo
  echo "======================================================================"
  echo "[START] Main comparison with Pure_MADDPG_Refine | train seed=${TRAIN_SEED}"
  echo "======================================================================"

  python3 ./eval/run_main_comparison_sci_v6.py \
    --stage2-prefix proposed_full_stage2_main_d25_seed${TRAIN_SEED}_ep700_v6_model_refine_formal10 \
    --stage1-prefix proposed_full_stage1_main_d25_seed${TRAIN_SEED}_ep200_formal10 \
    --pure-prefix pure_maddpg_main_d25_seed${TRAIN_SEED}_ep700_formal10 \
    --seeds ${EVAL_SEEDS} \
    --deadline-scale 2.5 \
    --task-local-cpu-min 2000 \
    --task-local-cpu-max 5000 \
    --episode-length 20 \
    --include-refine-controls \
    --out results/main_comparison_sci_v6/formal10_trainseed_eval10_with_pure_refine/trainseed${TRAIN_SEED}_eval10 \
    2>&1 | tee logs/main_comparison_formal10_with_pure_refine/trainseed${TRAIN_SEED}_eval10.log

  echo
  echo "======================================================================"
  echo "[DONE] Main comparison with Pure_MADDPG_Refine | train seed=${TRAIN_SEED}"
  echo "======================================================================"
done

echo
echo "All formal 10-train-seed comparisons with Pure_MADDPG_Refine finished."
echo "Results:"
echo "  results/main_comparison_sci_v6/formal10_trainseed_eval10_with_pure_refine/"