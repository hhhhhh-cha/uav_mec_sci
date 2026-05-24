#!/usr/bin/env bash
set -euo pipefail

cd ~/projects/uav_mec_sci
source .venv/bin/activate
export PYTHONPATH=$(pwd)

TRAIN_SEEDS=(42 52 62 72 82 92 102 112 122 132)
EVAL_SEEDS="42,52,62,72,82,92,102,112,122,132"

mkdir -p logs/phase1_refine_diagnostic_formal10
mkdir -p results/phase1_refine_diagnostic/formal10_trainseed_eval10

for TRAIN_SEED in "${TRAIN_SEEDS[@]}"; do
  echo
  echo "======================================================================"
  echo "[START] Phase-1 refine diagnostic | train seed=${TRAIN_SEED}"
  echo "======================================================================"

  python3 ./eval/phase1_refine_diagnostic_sci_v6.py \
    --stage2-prefix proposed_full_stage2_main_d25_seed${TRAIN_SEED}_ep700_v6_model_refine_formal10 \
    --pure-prefix pure_maddpg_main_d25_seed${TRAIN_SEED}_ep700_formal10 \
    --seeds ${EVAL_SEEDS} \
    --deadline-scale 2.5 \
    --task-local-cpu-min 2000 \
    --task-local-cpu-max 5000 \
    --episode-length 20 \
    --small-max-tasks 6 \
    --large-max-tasks 16 \
    --out results/phase1_refine_diagnostic/formal10_trainseed_eval10/trainseed${TRAIN_SEED}_eval10 \
    2>&1 | tee logs/phase1_refine_diagnostic_formal10/trainseed${TRAIN_SEED}_eval10.log

  echo
  echo "======================================================================"
  echo "[DONE] Phase-1 refine diagnostic | train seed=${TRAIN_SEED}"
  echo "======================================================================"
done

echo
echo "All Phase-1 formal 10-train-seed diagnostics finished."
echo "Results:"
echo "  results/phase1_refine_diagnostic/formal10_trainseed_eval10/"
