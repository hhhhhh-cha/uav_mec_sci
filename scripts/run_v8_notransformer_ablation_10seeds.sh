#!/usr/bin/env bash
set -euo pipefail

cd ~/projects/uav_mec_sci
source .venv/bin/activate
export PYTHONPATH=$(pwd)

SEEDS=(42 52 62 72 82 92 102 112 122 132)

# Proposed w/o Transformer:
# NUM_LAYERS=0 removes all Transformer self-attention/FFN blocks.
# Candidate tokens, masked pooling, fusion, ratio/scheduling heads,
# analytical solvers, Refine distillation, and MADDPG PG are retained.

STAGE1_RESULT_ROOT="results/convergence_training/notr_stage1"
STAGE2_RESULT_ROOT="results/convergence_training/v8_matched_stage1_stage2/notr_pgboost"
LOG_DIR="${STAGE2_RESULT_ROOT}/run_logs"
mkdir -p "${STAGE1_RESULT_ROOT}" "${STAGE2_RESULT_ROOT}" "${LOG_DIR}"

for S in "${SEEDS[@]}"
do
  STAGE1_PREFIX="proposed_full_stage1_main_d25_seed${S}_ep200_notr_formal10"
  STAGE2_RUN="proposed_full_stage2_main_d25_seed${S}_ep700_v8_notr_schedfix_pgboost_matched"

  echo "============================================================"
  echo "[NoTransformer Stage-1] seed=${S}"
  echo "============================================================"

  TORCH_NUM_THREADS=1 \
  DEVICE=auto \
  SEED=${S} \
  NUM_EPISODES=200 \
  EVAL_EVERY=5 \
  EPISODE_LENGTH=20 \
  DEADLINE_SCALE=2.5 \
  TASK_LOCAL_CPU_MIN=2000 \
  TASK_LOCAL_CPU_MAX=5000 \
  NUM_LAYERS=0 \
  RUN_NAME=${STAGE1_PREFIX} \
  CKPT_PREFIX=${STAGE1_PREFIX} \
  python3 -m train.train_proposed_full_stage1_converge

  echo "============================================================"
  echo "[NoTransformer Stage-2] seed=${S}"
  echo "============================================================"

  TORCH_NUM_THREADS=1 \
  DEVICE=auto \
  SEED=${S} \
  NUM_EPISODES=700 \
  EVAL_EVERY=5 \
  EPISODE_LENGTH=20 \
  DEADLINE_SCALE=2.5 \
  TASK_LOCAL_CPU_MIN=2000 \
  TASK_LOCAL_CPU_MAX=5000 \
  NUM_LAYERS=0 \
  STAGE1_PREFIX=${STAGE1_PREFIX} \
  RUN_NAME=${STAGE2_RUN} \
  CKPT_PREFIX=${STAGE2_RUN} \
  RESULT_ROOT=${STAGE2_RESULT_ROOT} \
  STRICT_STAGE1=1 \
  MODEL_REFINE=1 \
  MODEL_REFINE_MAX_TASKS=4 \
  MODEL_REFINE_RATIO=1 \
  MODEL_REFINE_SCHED=1 \
  BEST_SELECT_MODE=actoronly \
  REFINED_SCHED_CE_COEF_START=1.20 \
  REFINED_SCHED_CE_COEF_END=0.60 \
  NEIGHBOR_COLLAB_COEF_START=0.30 \
  NEIGHBOR_COLLAB_COEF_END=0.80 \
  NEIGHBOR_PROB_TARGET=0.20 \
  LOCAL_DOMINANCE_COEF_START=0.20 \
  LOCAL_DOMINANCE_COEF_END=0.50 \
  LOCAL_DOMINANCE_MARGIN=-0.20 \
  ACTOR_POLICY_COEF_START=0.001 \
  ACTOR_POLICY_COEF_END=0.003 \
  REWARD_SCALE=2e-4 \
  CRITIC_ONLY_EPISODES=60 \
  LEARNING_STARTS=800 \
  python3 -m train.train_proposed_full_stage2_converge_v8_transformer_sched \
    2>&1 | tee "${LOG_DIR}/${STAGE2_RUN}.log"

  echo "[DONE] seed=${S}"
done
