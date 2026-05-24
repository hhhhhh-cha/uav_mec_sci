#!/usr/bin/env bash
set -euo pipefail

cd ~/projects/uav_mec_sci
source .venv/bin/activate
export PYTHONPATH=$(pwd)

SEEDS=(42 52 62 72 82 92 102 112 122 132)

echo "============================================================"
echo "[Precheck] formal10 Stage-1 checkpoints for Transformer runs"
echo "============================================================"

for S in "${SEEDS[@]}"; do
  P="proposed_full_stage1_main_d25_seed${S}_ep200_formal10"
  echo "checking ${P}"
  test -f checkpoints/${P}_best_actor.pth || { echo "missing checkpoints/${P}_best_actor.pth"; exit 1; }
  test -f checkpoints/${P}_best_encoder.pth || { echo "missing checkpoints/${P}_best_encoder.pth"; exit 1; }
  test -f checkpoints/${P}_best_fusion.pth || { echo "missing checkpoints/${P}_best_fusion.pth"; exit 1; }
  test -f checkpoints/${P}_best_ratio_head.pth || { echo "missing checkpoints/${P}_best_ratio_head.pth"; exit 1; }
done

echo "All formal10 Stage-1 checkpoints are ready."

for S in "${SEEDS[@]}"
do
  STAGE1_PREFIX="proposed_full_stage1_main_d25_seed${S}_ep200_formal10"

  echo "============================================================"
  echo "[1/3] Fixed Transformer Proposed wPG Stage-2 | seed=${S}"
  echo "Stage-1 prefix = ${STAGE1_PREFIX}"
  echo "============================================================"

  WPG_RUN="proposed_full_stage2_main_d25_seed${S}_ep700_v8fix_pgboost"

  TORCH_NUM_THREADS=1 \
  DEVICE=auto \
  SEED=${S} \
  NUM_EPISODES=700 \
  EVAL_EVERY=5 \
  NUM_LAYERS=2 \
  STAGE1_PREFIX=${STAGE1_PREFIX} \
  RUN_NAME=${WPG_RUN} \
  CKPT_PREFIX=${WPG_RUN} \
  RESULT_ROOT=results/convergence_training/v8fix/pgboost \
  STRICT_STAGE1=1 \
  ACTOR_POLICY_COEF_START=0.001 \
  ACTOR_POLICY_COEF_END=0.003 \
  EPISODE_LENGTH=20 \
  DEADLINE_SCALE=2.5 \
  TASK_LOCAL_CPU_MIN=2000 \
  TASK_LOCAL_CPU_MAX=5000 \
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
  REWARD_SCALE=2e-4 \
  CRITIC_ONLY_EPISODES=60 \
  LEARNING_STARTS=800 \
  python3 -m train.train_proposed_full_stage2_converge_v8_transformer_sched \
    2>&1 | tee logs/v8fix/${WPG_RUN}.log

  echo "============================================================"
  echo "[2/3] Fixed Transformer Proposed w/o PG Stage-2 | seed=${S}"
  echo "Stage-1 prefix = ${STAGE1_PREFIX}"
  echo "============================================================"

  NOPG_RUN="proposed_full_stage2_main_d25_seed${S}_ep700_v8fix_nopg"

  TORCH_NUM_THREADS=1 \
  DEVICE=auto \
  SEED=${S} \
  NUM_EPISODES=700 \
  EVAL_EVERY=5 \
  NUM_LAYERS=2 \
  STAGE1_PREFIX=${STAGE1_PREFIX} \
  RUN_NAME=${NOPG_RUN} \
  CKPT_PREFIX=${NOPG_RUN} \
  RESULT_ROOT=results/convergence_training/v8fix/nopg \
  STRICT_STAGE1=1 \
  ACTOR_POLICY_COEF_START=0.0 \
  ACTOR_POLICY_COEF_END=0.0 \
  EPISODE_LENGTH=20 \
  DEADLINE_SCALE=2.5 \
  TASK_LOCAL_CPU_MIN=2000 \
  TASK_LOCAL_CPU_MAX=5000 \
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
  REWARD_SCALE=2e-4 \
  CRITIC_ONLY_EPISODES=60 \
  LEARNING_STARTS=800 \
  python3 -m train.train_proposed_full_stage2_converge_v8_transformer_sched \
    2>&1 | tee logs/v8fix/${NOPG_RUN}.log

  echo "============================================================"
  echo "[3/3] Proposed w/o Transformer: Stage-1 + Stage-2 | seed=${S}"
  echo "This branch must run its own Stage-1 because NUM_LAYERS=0 changes encoder architecture."
  echo "============================================================"

  NOTR_STAGE1="proposed_full_stage1_main_d25_seed${S}_ep200_v8fix_notr"
  NOTR_STAGE2="proposed_full_stage2_main_d25_seed${S}_ep700_v8fix_notr_pgboost"

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
  RUN_NAME=${NOTR_STAGE1} \
  CKPT_PREFIX=${NOTR_STAGE1} \
  python3 -m train.train_proposed_full_stage1_converge \
    2>&1 | tee logs/v8fix/${NOTR_STAGE1}.log

  echo "checking no-Transformer Stage-1 checkpoint for seed=${S}"
  test -f checkpoints/${NOTR_STAGE1}_best_actor.pth || { echo "missing ${NOTR_STAGE1}_best_actor"; exit 1; }
  test -f checkpoints/${NOTR_STAGE1}_best_encoder.pth || { echo "missing ${NOTR_STAGE1}_best_encoder"; exit 1; }
  test -f checkpoints/${NOTR_STAGE1}_best_fusion.pth || { echo "missing ${NOTR_STAGE1}_best_fusion"; exit 1; }
  test -f checkpoints/${NOTR_STAGE1}_best_ratio_head.pth || { echo "missing ${NOTR_STAGE1}_best_ratio_head"; exit 1; }

  TORCH_NUM_THREADS=1 \
  DEVICE=auto \
  SEED=${S} \
  NUM_EPISODES=700 \
  EVAL_EVERY=5 \
  NUM_LAYERS=0 \
  STAGE1_PREFIX=${NOTR_STAGE1} \
  RUN_NAME=${NOTR_STAGE2} \
  CKPT_PREFIX=${NOTR_STAGE2} \
  RESULT_ROOT=results/convergence_training/v8fix/notr_pgboost \
  STRICT_STAGE1=1 \
  ACTOR_POLICY_COEF_START=0.001 \
  ACTOR_POLICY_COEF_END=0.003 \
  EPISODE_LENGTH=20 \
  DEADLINE_SCALE=2.5 \
  TASK_LOCAL_CPU_MIN=2000 \
  TASK_LOCAL_CPU_MAX=5000 \
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
  REWARD_SCALE=2e-4 \
  CRITIC_ONLY_EPISODES=60 \
  LEARNING_STARTS=800 \
  python3 -m train.train_proposed_full_stage2_converge_v8_transformer_sched \
    2>&1 | tee logs/v8fix/${NOTR_STAGE2}.log

  echo "checking Stage-2 checkpoints for seed=${S}"
  for P in "${WPG_RUN}" "${NOPG_RUN}" "${NOTR_STAGE2}"; do
    test -f checkpoints/${P}_best_actor.pth || { echo "missing ${P}_best_actor"; exit 1; }
    test -f checkpoints/${P}_best_encoder.pth || { echo "missing ${P}_best_encoder"; exit 1; }
    test -f checkpoints/${P}_best_fusion.pth || { echo "missing ${P}_best_fusion"; exit 1; }
    test -f checkpoints/${P}_best_ratio_head.pth || { echo "missing ${P}_best_ratio_head"; exit 1; }
    test -f checkpoints/${P}_best_schedule_head.pth || { echo "missing ${P}_best_schedule_head"; exit 1; }
  done

  echo "[DONE] seed=${S}"
done

echo "All v8fix proposed experiments finished."
