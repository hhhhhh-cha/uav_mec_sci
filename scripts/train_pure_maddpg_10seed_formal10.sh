#!/usr/bin/env bash
set -euo pipefail

cd ~/projects/uav_mec_sci
source .venv/bin/activate

export PYTHONPATH=$(pwd)

SEEDS=(42 52 62 72 82 92 102 112 122 132)

mkdir -p logs/pure_maddpg_formal10

for SEED in "${SEEDS[@]}"
do
  echo
  echo "======================================================================"
  echo "[START] Pure MADDPG training | seed=${SEED}"
  echo "======================================================================"

  CKPT="checkpoints/pure_maddpg_main_d25_seed${SEED}_ep700_formal10_best_actor.pth"

  if [ -f "${CKPT}" ]; then
    echo "[SKIP] Checkpoint already exists:"
    echo "       ${CKPT}"
    continue
  fi

  TORCH_NUM_THREADS=1 \
  DEVICE=auto \
  SEED=${SEED} \
  EVAL_SEED=999 \
  M=3 \
  K=16 \
  NUM_EPISODES=700 \
  EPISODE_LENGTH=20 \
  CPU_MODE=kkt \
  DEADLINE_SCALE=2.5 \
  TASK_LOCAL_CPU_MIN=2000 \
  TASK_LOCAL_CPU_MAX=5000 \
  OMEGA1=50 \
  OMEGA2=1 \
  RUN_NAME=pure_maddpg_main_d25_seed${SEED}_ep700_formal10 \
  CKPT_PREFIX=pure_maddpg_main_d25_seed${SEED}_ep700_formal10 \
  python3 -m train.train_pure_maddpg_sci_v6 \
    2>&1 | tee logs/pure_maddpg_formal10/pure_maddpg_seed${SEED}_ep700_formal10.log

  echo
  echo "======================================================================"
  echo "[CHECK] Pure MADDPG checkpoint | seed=${SEED}"
  echo "======================================================================"

  test -f checkpoints/pure_maddpg_main_d25_seed${SEED}_ep700_formal10_best_actor.pth
  ls -lh checkpoints/pure_maddpg_main_d25_seed${SEED}_ep700_formal10_best_actor.pth

  echo
  echo "======================================================================"
  echo "[DONE] Pure MADDPG training finished | seed=${SEED}"
  echo "======================================================================"

done

echo
echo "======================================================================"
echo "All Pure MADDPG 10-seed training jobs finished."
echo "======================================================================"
echo "Check logs:"
echo "  logs/pure_maddpg_formal10/"
echo
echo "Check checkpoints:"
echo "  checkpoints/pure_maddpg_main_d25_seed*_ep700_formal10_best_actor.pth"