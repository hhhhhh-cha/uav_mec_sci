#!/usr/bin/env bash
set -euo pipefail

cd ~/projects/uav_mec_sci
source .venv/bin/activate
export PYTHONPATH=$(pwd)

TRAIN_SEEDS=(42 52 62 72 82 92 102 112 122 132)
EVAL_SEEDS="42,52,62,72,82,92,102,112,122,132"

OUT_ROOT="results/main_comparison_sci_v8/matched_formal10_trainseed_eval10_with_notr"
LOG_DIR="logs/main_comparison_sci_v8_matched_formal10_with_notr"
mkdir -p "${OUT_ROOT}" "${LOG_DIR}"

for TRAIN_SEED in "${TRAIN_SEEDS[@]}"
do
  PG_PREFIX="proposed_full_stage2_main_d25_seed${TRAIN_SEED}_ep700_v8_refine4_schedfix_pgboost_matched"
  NOPG_PREFIX="proposed_full_stage2_main_d25_seed${TRAIN_SEED}_ep700_v8_refine4_schedfix_no_pg_matched"
  NOTR_PREFIX="proposed_full_stage2_main_d25_seed${TRAIN_SEED}_ep700_v8_notr_schedfix_pgboost_matched"
  PURE_PREFIX="pure_maddpg_main_d25_seed${TRAIN_SEED}_ep700_formal10"

  python3 ./eval/run_main_comparison_sci_v8.py \
    --pgboost-prefix "${PG_PREFIX}" \
    --nopg-prefix "${NOPG_PREFIX}" \
    --pure-prefix "${PURE_PREFIX}" \
    --include-notr \
    --notr-prefix "${NOTR_PREFIX}" \
    --notr-num-layers 0 \
    --seeds "${EVAL_SEEDS}" \
    --deadline-scale 2.5 \
    --task-local-cpu-min 2000 \
    --task-local-cpu-max 5000 \
    --episode-length 20 \
    --model-refine-max-tasks 4 \
    --model-refine-ratio 1 \
    --model-refine-sched 1 \
    --include-greedy-refine \
    --out "${OUT_ROOT}/trainseed${TRAIN_SEED}_eval10" \
    2>&1 | tee "${LOG_DIR}/trainseed${TRAIN_SEED}_eval10.log"
done
