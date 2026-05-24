#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# FunHPC topology-stress pilot for UAV-MEC v8fix
# Purpose:
#   Compare Transformer vs no-Transformer under larger candidate UAV sets.
# Scenario:
#   M=5, K=25, seeds=(42,72,102)
# Output naming:
#   All checkpoints / logs / result folders contain the prefix "funhpc".
# ============================================================

# ---------- 0) Locate project directory ----------
if [[ -n "${PROJECT_DIR:-}" ]]; then
  :
elif [[ -f "train/train_proposed_full_stage1_converge.py" ]]; then
  PROJECT_DIR="$(pwd)"
else
  FOUND_FILE="$(find /data/coding -maxdepth 5 -type f -path '*/train/train_proposed_full_stage1_converge.py' 2>/dev/null | head -n 1 || true)"
  if [[ -z "${FOUND_FILE}" ]]; then
    echo "[ERROR] Cannot find train/train_proposed_full_stage1_converge.py under /data/coding."
    echo "Please put this script in your project root, or run: PROJECT_DIR=/path/to/project bash $0"
    exit 1
  fi
  PROJECT_DIR="$(dirname "$(dirname "${FOUND_FILE}")")"
fi

cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}"

echo "============================================================"
echo "FunHPC UAV-MEC topology-stress script"
echo "PROJECT_DIR=${PROJECT_DIR}"
echo "PYTHONPATH=${PYTHONPATH}"
echo "============================================================"

# ---------- 1) Optional: switch to a cloud-only git branch ----------
if command -v git >/dev/null 2>&1 && git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  TARGET_BRANCH="funhpc-topostress-m5k25"
  CUR_BRANCH="$(git branch --show-current || true)"
  echo "[Git] current branch: ${CUR_BRANCH:-unknown}"
  if [[ "${CUR_BRANCH}" != "${TARGET_BRANCH}" ]]; then
    if git show-ref --verify --quiet "refs/heads/${TARGET_BRANCH}"; then
      git switch "${TARGET_BRANCH}"
    else
      git switch -c "${TARGET_BRANCH}"
    fi
  fi
  echo "[Git] running on branch: $(git branch --show-current || true)"
else
  echo "[Git] not a git repo or git unavailable; skip branch creation."
fi

# ---------- 2) Basic code checks ----------
echo "============================================================"
echo "[Check] Python syntax and v8 schedule-head soft-update"
echo "============================================================"

python3 -m py_compile \
  train/train_proposed_full_stage1_converge.py \
  train/train_proposed_full_stage2_converge_v8_transformer_sched.py

if ! grep -q "def soft_update_policy_v8" train/train_proposed_full_stage2_converge_v8_transformer_sched.py; then
  echo "[ERROR] soft_update_policy_v8 not found. Your FunHPC code may not have the v2 checked patch."
  echo "Please apply uav_mec_experiment_patch_files_v2_checked.zip first."
  exit 1
fi

if ! grep -q "schedule_head" train/train_proposed_full_stage2_converge_v8_transformer_sched.py; then
  echo "[ERROR] schedule_head not found in Stage-2 v8 file. Please check the patch."
  exit 1
fi

echo "[OK] py_compile passed and schedule_head-related code exists."

python3 - <<'PY'
import torch
from env.mec_env import MultiUavMecEnv
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
env = MultiUavMecEnv(M=5, K=25, episode_length=20, deadline_scale=2.5)
obs = env.reset(seed=42)
print("env OK: M=", obs["raw_state"]["M"], "K=", obs["raw_state"]["K"])
PY

# ---------- 3) Experiment configuration ----------
SEEDS=(42 72 102)
MVAL=5
KVAL=25

mkdir -p scripts logs/funhpc_topostress_m5k25
mkdir -p results/convergence_training/funhpc_topostress_m5k25
mkdir -p checkpoints

# ---------- 4) Run experiments ----------
for S in "${SEEDS[@]}"; do
  echo "============================================================"
  echo "[FunHPC Topology Stress] M=${MVAL}, K=${KVAL}, seed=${S}"
  echo "============================================================"

  # ==========================================================
  # A) Transformer Stage-1 + Stage-2
  # ==========================================================
  TR_STAGE1="funhpc_proposed_stage1_topo_m${MVAL}k${KVAL}_seed${S}_ep200_v8fix"
  TR_STAGE2="funhpc_proposed_stage2_topo_m${MVAL}k${KVAL}_seed${S}_ep700_v8fix_pgboost"

  echo "------------------------------------------------------------"
  echo "[1/4] Transformer Stage-1 | ${TR_STAGE1}"
  echo "------------------------------------------------------------"

  TORCH_NUM_THREADS=1 \
  DEVICE=auto \
  SEED=${S} \
  M=${MVAL} \
  K=${KVAL} \
  NUM_EPISODES=200 \
  EVAL_EVERY=5 \
  EPISODE_LENGTH=20 \
  DEADLINE_SCALE=2.5 \
  TASK_LOCAL_CPU_MIN=2000 \
  TASK_LOCAL_CPU_MAX=5000 \
  NUM_LAYERS=2 \
  RUN_NAME=${TR_STAGE1} \
  CKPT_PREFIX=${TR_STAGE1} \
  python3 -m train.train_proposed_full_stage1_converge \
    2>&1 | tee "logs/funhpc_topostress_m5k25/${TR_STAGE1}.log"

  test -f "checkpoints/${TR_STAGE1}_best_actor.pth" || { echo "missing ${TR_STAGE1}_best_actor"; exit 1; }
  test -f "checkpoints/${TR_STAGE1}_best_encoder.pth" || { echo "missing ${TR_STAGE1}_best_encoder"; exit 1; }
  test -f "checkpoints/${TR_STAGE1}_best_fusion.pth" || { echo "missing ${TR_STAGE1}_best_fusion"; exit 1; }
  test -f "checkpoints/${TR_STAGE1}_best_ratio_head.pth" || { echo "missing ${TR_STAGE1}_best_ratio_head"; exit 1; }

  echo "------------------------------------------------------------"
  echo "[2/4] Transformer Stage-2 | ${TR_STAGE2}"
  echo "------------------------------------------------------------"

  TORCH_NUM_THREADS=1 \
  DEVICE=auto \
  SEED=${S} \
  M=${MVAL} \
  K=${KVAL} \
  NUM_EPISODES=700 \
  EVAL_EVERY=5 \
  EPISODE_LENGTH=20 \
  DEADLINE_SCALE=2.5 \
  TASK_LOCAL_CPU_MIN=2000 \
  TASK_LOCAL_CPU_MAX=5000 \
  NUM_LAYERS=2 \
  STAGE1_PREFIX=${TR_STAGE1} \
  RUN_NAME=${TR_STAGE2} \
  CKPT_PREFIX=${TR_STAGE2} \
  RESULT_ROOT=results/convergence_training/funhpc_topostress_m5k25/pgboost \
  STRICT_STAGE1=1 \
  ACTOR_POLICY_COEF_START=0.001 \
  ACTOR_POLICY_COEF_END=0.003 \
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
    2>&1 | tee "logs/funhpc_topostress_m5k25/${TR_STAGE2}.log"

  test -f "checkpoints/${TR_STAGE2}_best_actor.pth" || { echo "missing ${TR_STAGE2}_best_actor"; exit 1; }
  test -f "checkpoints/${TR_STAGE2}_best_encoder.pth" || { echo "missing ${TR_STAGE2}_best_encoder"; exit 1; }
  test -f "checkpoints/${TR_STAGE2}_best_fusion.pth" || { echo "missing ${TR_STAGE2}_best_fusion"; exit 1; }
  test -f "checkpoints/${TR_STAGE2}_best_ratio_head.pth" || { echo "missing ${TR_STAGE2}_best_ratio_head"; exit 1; }
  test -f "checkpoints/${TR_STAGE2}_best_schedule_head.pth" || { echo "missing ${TR_STAGE2}_best_schedule_head"; exit 1; }

  # ==========================================================
  # B) No-Transformer Stage-1 + Stage-2
  # ==========================================================
  NOTR_STAGE1="funhpc_proposed_stage1_topo_m${MVAL}k${KVAL}_seed${S}_ep200_v8fix_notr"
  NOTR_STAGE2="funhpc_proposed_stage2_topo_m${MVAL}k${KVAL}_seed${S}_ep700_v8fix_notr_pgboost"

  echo "------------------------------------------------------------"
  echo "[3/4] No-Transformer Stage-1 | ${NOTR_STAGE1}"
  echo "------------------------------------------------------------"

  TORCH_NUM_THREADS=1 \
  DEVICE=auto \
  SEED=${S} \
  M=${MVAL} \
  K=${KVAL} \
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
    2>&1 | tee "logs/funhpc_topostress_m5k25/${NOTR_STAGE1}.log"

  test -f "checkpoints/${NOTR_STAGE1}_best_actor.pth" || { echo "missing ${NOTR_STAGE1}_best_actor"; exit 1; }
  test -f "checkpoints/${NOTR_STAGE1}_best_encoder.pth" || { echo "missing ${NOTR_STAGE1}_best_encoder"; exit 1; }
  test -f "checkpoints/${NOTR_STAGE1}_best_fusion.pth" || { echo "missing ${NOTR_STAGE1}_best_fusion"; exit 1; }
  test -f "checkpoints/${NOTR_STAGE1}_best_ratio_head.pth" || { echo "missing ${NOTR_STAGE1}_best_ratio_head"; exit 1; }

  echo "------------------------------------------------------------"
  echo "[4/4] No-Transformer Stage-2 | ${NOTR_STAGE2}"
  echo "------------------------------------------------------------"

  TORCH_NUM_THREADS=1 \
  DEVICE=auto \
  SEED=${S} \
  M=${MVAL} \
  K=${KVAL} \
  NUM_EPISODES=700 \
  EVAL_EVERY=5 \
  EPISODE_LENGTH=20 \
  DEADLINE_SCALE=2.5 \
  TASK_LOCAL_CPU_MIN=2000 \
  TASK_LOCAL_CPU_MAX=5000 \
  NUM_LAYERS=0 \
  STAGE1_PREFIX=${NOTR_STAGE1} \
  RUN_NAME=${NOTR_STAGE2} \
  CKPT_PREFIX=${NOTR_STAGE2} \
  RESULT_ROOT=results/convergence_training/funhpc_topostress_m5k25/notr_pgboost \
  STRICT_STAGE1=1 \
  ACTOR_POLICY_COEF_START=0.001 \
  ACTOR_POLICY_COEF_END=0.003 \
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
    2>&1 | tee "logs/funhpc_topostress_m5k25/${NOTR_STAGE2}.log"

  test -f "checkpoints/${NOTR_STAGE2}_best_actor.pth" || { echo "missing ${NOTR_STAGE2}_best_actor"; exit 1; }
  test -f "checkpoints/${NOTR_STAGE2}_best_encoder.pth" || { echo "missing ${NOTR_STAGE2}_best_encoder"; exit 1; }
  test -f "checkpoints/${NOTR_STAGE2}_best_fusion.pth" || { echo "missing ${NOTR_STAGE2}_best_fusion"; exit 1; }
  test -f "checkpoints/${NOTR_STAGE2}_best_ratio_head.pth" || { echo "missing ${NOTR_STAGE2}_best_ratio_head"; exit 1; }
  test -f "checkpoints/${NOTR_STAGE2}_best_schedule_head.pth" || { echo "missing ${NOTR_STAGE2}_best_schedule_head"; exit 1; }

  echo "[DONE] FunHPC M=${MVAL}, K=${KVAL}, seed=${S}"
done

echo "============================================================"
echo "All FunHPC topology-stress pilot runs finished."
echo "Results: results/convergence_training/funhpc_topostress_m5k25"
echo "Logs:    logs/funhpc_topostress_m5k25"
echo "============================================================"

scp -P 40878 /data/coding/uav_mec_sci_funhpc_code_v8fix.tar.gz root@51dpbvcomzxkrgxnsnow.deepln.com:/data/coding/
ssh -p 40878 root@51dpbvcomzxkrgxnsnow.deepln.com