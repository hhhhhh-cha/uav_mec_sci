#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source .venv/bin/activate
export PYTHONPATH="$(pwd)"

SEEDS=(42 52 62 72 82 92 102 112 122 132)

for SEED in "${SEEDS[@]}"; do
  echo "================================================================================"
  echo "[V8 Stage-2] train seed=${SEED}"
  echo "================================================================================"

  TORCH_NUM_THREADS=1 \
  DEVICE=auto \
  SEED=${SEED} \
  NUM_EPISODES=700 \
  EPISODE_LENGTH=20 \
  OMEGA1=50 \
  OMEGA2=1 \
  DEADLINE_SCALE=2.5 \
  TASK_LOCAL_CPU_MIN=2000 \
  TASK_LOCAL_CPU_MAX=5000 \
  STAGE1_PREFIX=proposed_full_stage1_main_d25_seed${SEED}_ep200_formal10 \
  RUN_NAME=proposed_full_stage2_main_d25_seed${SEED}_ep700_v8_transformer_sched_formal10 \
  CKPT_PREFIX=proposed_full_stage2_main_d25_seed${SEED}_ep700_v8_transformer_sched_formal10 \
  CRITIC_ONLY_EPISODES=80 \
  LEARNING_STARTS=1000 \
  BATCH_SIZE=128 \
  POLICY_DELAY=2 \
  MODEL_REFINE=1 \
  MODEL_REFINE_EVAL=1 \
  MODEL_REFINE_MAX_TASKS=16 \
  MODEL_REFINE_RATIO=1 \
  MODEL_REFINE_SCHED=1 \
  REFINED_RATIO_DISTILL=1 \
  REFINED_RATIO_BC_COEF_START=0.80 \
  REFINED_RATIO_BC_COEF_END=0.25 \
  REFINED_SCHED_DISTILL=1 \
  REFINED_SCHED_CE_COEF_START=0.60 \
  REFINED_SCHED_CE_COEF_END=0.20 \
  SCHEDULE_ENTROPY_COEF_START=0.010 \
  SCHEDULE_ENTROPY_COEF_END=0.000 \
  ACTOR_POLICY_COEF_START=0.0003 \
  ACTOR_POLICY_COEF_END=0.0012 \
  ACTOR_MOVE_SCHED_BC_COEF_START=1.00 \
  ACTOR_MOVE_SCHED_BC_COEF_END=0.30 \
  RATIO_BC_COEF_START=0.020 \
  RATIO_BC_COEF_END=0.000 \
  RATIO_REG_COEF_START=0.002 \
  RATIO_REG_COEF_END=0.0005 \
  RATIO_FLOOR_TARGET=0.05 \
  RATIO_MEAN_TARGET=0.060 \
  RATIO_STD_TARGET=0.015 \
  NEIGHBOR_COLLAB_COEF_START=0.05 \
  NEIGHBOR_COLLAB_COEF_END=0.10 \
  LOCAL_DOMINANCE_COEF_START=0.02 \
  LOCAL_DOMINANCE_COEF_END=0.05 \
  SCHEDULE_BRANCH_LR=5e-5 \
  RATIO_BRANCH_LR=5e-5 \
  ACTOR_LR=5e-6 \
  EVAL_EVERY=20 \
  EVAL_SEED=999 \
  python3 -m train.train_proposed_full_stage2_converge_v8_transformer_sched

done
