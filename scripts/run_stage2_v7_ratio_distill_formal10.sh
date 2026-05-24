#!/usr/bin/env bash
set -euo pipefail

cd ~/projects/uav_mec_sci
source .venv/bin/activate
export PYTHONPATH=$(pwd)
export TORCH_NUM_THREADS=1
export TORCH_INTEROP_THREADS=1

TRAIN_SEEDS=(42 52 62 72 82 92 102 112 122 132)
mkdir -p logs/formal_10seed_v7_ratio_distill

for SEED_ID in "${TRAIN_SEEDS[@]}"; do
  echo
  echo "======================================================================"
  echo "[START] Proposed Stage-2 v7 ratio distillation | seed=${SEED_ID}"
  echo "======================================================================"

  DEVICE=auto \
  SEED=${SEED_ID} \
  NUM_EPISODES=700 \
  EPISODE_LENGTH=20 \
  EVAL_EVERY=5 \
  OMEGA1=50 \
  OMEGA2=1 \
  DEADLINE_SCALE=2.5 \
  TASK_LOCAL_CPU_MIN=2000 \
  TASK_LOCAL_CPU_MAX=5000 \
  STAGE1_PREFIX=proposed_full_stage1_main_d25_seed${SEED_ID}_ep200_formal10 \
  RUN_NAME=proposed_full_stage2_main_d25_seed${SEED_ID}_ep700_v7_ratio_distill_formal10 \
  CKPT_PREFIX=proposed_full_stage2_main_d25_seed${SEED_ID}_ep700_v7_ratio_distill_formal10 \
  MODEL_REFINE=1 \
  MODEL_REFINE_MAX_TASKS=16 \
  MODEL_REFINE_RATIO=1 \
  MODEL_REFINE_SCHED=1 \
  REFINED_RATIO_DISTILL=1 \
  REFINED_RATIO_BC_COEF_START=0.50 \
  REFINED_RATIO_BC_COEF_END=0.15 \
  RATIO_BC_COEF_START=0.020 \
  RATIO_BC_COEF_END=0.000 \
  RATIO_REG_COEF_START=0.002 \
  RATIO_REG_COEF_END=0.0005 \
  RATIO_FLOOR_TARGET=0.05 \
  RATIO_MEAN_TARGET=0.060 \
  RATIO_STD_TARGET=0.015 \
  ACTOR_POLICY_COEF_START=0.0003 \
  ACTOR_POLICY_COEF_END=0.0012 \
  ACTOR_MOVE_SCHED_BC_COEF_START=1.00 \
  ACTOR_MOVE_SCHED_BC_COEF_END=0.30 \
  NEIGHBOR_COLLAB_COEF_START=0.10 \
  NEIGHBOR_COLLAB_COEF_END=0.30 \
  NEIGHBOR_PROB_TARGET=0.15 \
  LOCAL_DOMINANCE_COEF_START=0.05 \
  LOCAL_DOMINANCE_COEF_END=0.15 \
  SELF_IMITATION_COEF_START=0.30 \
  SELF_IMITATION_COEF_END=0.80 \
  python3 -m train.train_proposed_full_stage2_converge_v7_ratio_distill \
    2>&1 | tee logs/formal_10seed_v7_ratio_distill/proposed_stage2_v7_seed${SEED_ID}.log

  echo
  echo "======================================================================"
  echo "[DONE] Proposed Stage-2 v7 ratio distillation | seed=${SEED_ID}"
  echo "======================================================================"
done
