#!/usr/bin/env bash
set -euo pipefail

cd ~/projects/uav_mec_sci
source .venv/bin/activate
export PYTHONPATH=$(pwd)

RESULT_ROOT="results/convergence_training/v8_matched_stage1_stage2/pgboost"
LOG_DIR="${RESULT_ROOT}/run_logs"
CKPT_BACKUP_DIR="${RESULT_ROOT}/checkpoints"

mkdir -p "${RESULT_ROOT}" "${LOG_DIR}" "${CKPT_BACKUP_DIR}"

SEEDS=(42 52 62 72 82 92 102 112 122 132)

for S in "${SEEDS[@]}"
do
  STAGE1_PREFIX="proposed_full_stage1_main_d25_seed${S}_ep200_formal10"
  RUN_NAME="proposed_full_stage2_main_d25_seed${S}_ep700_v8_refine4_schedfix_pgboost_matched"
  CKPT_PREFIX="${RUN_NAME}"

  echo "============================================================"
  echo "Running STRICT MATCHED pgboost"
  echo "Seed          : ${S}"
  echo "Stage-1 prefix: ${STAGE1_PREFIX}"
  echo "Run name      : ${RUN_NAME}"
  echo "Result root   : ${RESULT_ROOT}"
  echo "============================================================"

  for PART in actor encoder fusion ratio_head
  do
    if [ ! -f "checkpoints/${STAGE1_PREFIX}_best_${PART}.pth" ]; then
      echo "[ERROR] Missing Stage-1 checkpoint: checkpoints/${STAGE1_PREFIX}_best_${PART}.pth"
      exit 1
    fi
  done

  TORCH_NUM_THREADS=1 \
  DEVICE=auto \
  SEED=${S} \
  NUM_EPISODES=700 \
  EVAL_EVERY=5 \
  EPISODE_LENGTH=20 \
  DEADLINE_SCALE=2.5 \
  TASK_LOCAL_CPU_MIN=2000 \
  TASK_LOCAL_CPU_MAX=5000 \
  STAGE1_PREFIX=${STAGE1_PREFIX} \
  RUN_NAME=${RUN_NAME} \
  CKPT_PREFIX=${CKPT_PREFIX} \
  RESULT_ROOT=${RESULT_ROOT} \
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
    2>&1 | tee "${LOG_DIR}/${RUN_NAME}.log"

  mkdir -p "${CKPT_BACKUP_DIR}/${RUN_NAME}"
  cp -v checkpoints/${CKPT_PREFIX}_*.pth "${CKPT_BACKUP_DIR}/${RUN_NAME}/" || true

  echo "[DONE] Seed ${S} pgboost finished."
  echo
done

echo "============================================================"
echo "All STRICT MATCHED pgboost seeds finished."
echo "Results: ${RESULT_ROOT}"
echo "Logs   : ${LOG_DIR}"
echo "Ckpts  : ${CKPT_BACKUP_DIR}"
echo "============================================================"
