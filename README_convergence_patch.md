# UAV MEC convergence patch

本补丁只新增训练入口，不覆盖旧文件：

- `train/train_proposed_full_stage1_converge.py`
- `train/train_proposed_full_stage2_converge.py`

核心修改：

1. 自动选择 `cuda`；无 GPU 时设置 `torch.set_num_threads(1)`，避免 WSL/CPU 下 PyTorch 线程过度竞争。
2. Stage-1 默认与 Stage-2 使用一致的 `K=16, episode_length=20`，避免 Stage-1 只见到前 5 个时隙的状态分布。
3. Stage-2 默认严格加载 Stage-1 actor/encoder/fusion/ratio_head，不再在缺少 warm-start 时悄悄从随机 Stage-2 开训。
4. Stage-2 critic 仍然从零初始化，不加载 Stage-1 critic。
5. 所有关键参数可用环境变量覆盖，便于做 smoke test、单 seed 长训练和多 seed 实验。

## 运行顺序

```bash
source .venv/bin/activate

# 1) Stage-1 warm-start
TORCH_NUM_THREADS=1 \
DEVICE=auto \
SEED=72 \
NUM_EPISODES=200 \
EPISODE_LENGTH=20 \
CKPT_PREFIX=proposed_full_stage1_converge_k16_seed72 \
python3 -m train.train_proposed_full_stage1_converge

# 2) Stage-2 convergence training
TORCH_NUM_THREADS=1 \
DEVICE=auto \
SEED=72 \
NUM_EPISODES=700 \
EPISODE_LENGTH=20 \
STAGE1_PREFIX=proposed_full_stage1_converge_k16_seed72 \
RUN_NAME=proposed_full_stage2_converge_k16_seed72 \
CKPT_PREFIX=proposed_full_stage2_converge_k16_seed72 \
python3 -m train.train_proposed_full_stage2_converge
```

## 快速 smoke test

```bash
STRICT_STAGE1=0 NUM_EPISODES=1 EVAL_EVERY=1 RUN_NAME=smoke_test \
python3 -m train.train_proposed_full_stage2_converge
```

## 输出路径

- Stage-1 checkpoints: `checkpoints/<CKPT_PREFIX>_best_actor.pth` 等
- Stage-2 checkpoints: `checkpoints/<CKPT_PREFIX>_best_actor.pth` 等
- Stage-2 logs: `results/convergence_training/<RUN_NAME>/train_log.csv` and `eval_log.csv`




# Stage-1 logging patch

Files:
- `train_proposed_full_stage1_converge.py`: modified Stage-1 warm-start training script with `train_log.csv`, `eval_log.csv`, and `config.json` output.
- `plot_stagewise_convergence.py`: merges Stage-1 and Stage-2 `eval_log.csv` files and plots stage-wise convergence.

Recommended run:
```bash
cd ~/projects/uav_mec_sci
cp /path/to/train_proposed_full_stage1_converge.py train/train_proposed_full_stage1_converge.py
cp /path/to/plot_stagewise_convergence.py ./plot_stagewise_convergence.py

TORCH_NUM_THREADS=1 \
DEVICE=auto \
SEED=72 \
NUM_EPISODES=200 \
EPISODE_LENGTH=20 \
OMEGA1=50 \
OMEGA2=1 \
TASK_LOCAL_CPU_MIN=1000 \
TASK_LOCAL_CPU_MAX=3000 \
DEADLINE_SCALE=5.0 \
RUN_NAME=proposed_full_stage1_damaged_w50_seed72_logged \
CKPT_PREFIX=proposed_full_stage1_damaged_w50_seed72_logged \
EVAL_EVERY=5 \
EVAL_SEED=999 \
python3 -m train.train_proposed_full_stage1_converge




# Stage-wise training code v2

Files:
- `train_proposed_full_stage1_converge.py`: Stage-1 warm-start training with full CSV logging, safe mobility projection, hidden feasibility diagnostics, ratio/execution statistics, and config.json output.
- `train_proposed_full_stage2_converge.py`: Stage-2 CTDE fine-tuning with fixed evaluation protocol, `system_cost`, ratio statistics, execution split, safe-mobility evaluation, and import aligned to `train_proposed_full_stage1_converge`.

Recommended Stage-1 run:
```bash
cd ~/projects/uav_mec_sci
source .venv/bin/activate

TORCH_NUM_THREADS=1 \
DEVICE=auto \
SEED=72 \
NUM_EPISODES=120 \
EPISODE_LENGTH=20 \
OMEGA1=50 \
OMEGA2=1 \
TASK_LOCAL_CPU_MIN=1000 \
TASK_LOCAL_CPU_MAX=3000 \
DEADLINE_SCALE=5.0 \
RUN_NAME=proposed_full_stage1_damaged_w50_seed72_logged_v2_ep120 \
CKPT_PREFIX=proposed_full_stage1_damaged_w50_seed72_logged_v2_ep120 \
EVAL_EVERY=5 \
EVAL_SEED=999 \
python3 -m train.train_proposed_full_stage1_converge
```

Recommended Stage-2 run:
```bash
TORCH_NUM_THREADS=1 \
DEVICE=auto \
SEED=72 \
NUM_EPISODES=250 \
EPISODE_LENGTH=20 \
OMEGA1=50 \
OMEGA2=1 \
TASK_LOCAL_CPU_MIN=1000 \
TASK_LOCAL_CPU_MAX=3000 \
DEADLINE_SCALE=5.0 \
STAGE1_PREFIX=proposed_full_stage1_damaged_w50_seed72_logged_v2_ep120 \
RUN_NAME=proposed_full_stage2_damaged_w50_seed72_logged_v2_warm20_margin \
CKPT_PREFIX=proposed_full_stage2_damaged_w50_seed72_logged_v2_warm20_margin \
RATIO_REG_COEF=0.1 \
RATIO_FLOOR_TARGET=0.05 \
RATIO_MEAN_TARGET=0.095 \
RATIO_STD_TARGET=0.02 \
ACTOR_POLICY_COEF=0.00003 \
ACTOR_MOVE_SCHED_BC_COEF=5.0 \
RATIO_BC_COEF=0.05 \
CRITIC_ONLY_EPISODES=20 \
EVAL_EVERY=5 \
EVAL_SEED=999 \
python3 -m train.train_proposed_full_stage2_converge


## 绘制两阶段收敛图
python3 plot_stagewise_convergence.py \
  --stage1 results/convergence_training/proposed_full_stage1_damaged_w50_seed72_logged_v2_ep120 \
  --stage2 results/convergence_training/proposed_full_stage2_damaged_w50_seed72_logged_v2_warm20_margin \
  --out results/convergence_outputs/pic_train/stagewise_damaged_w50_seed72_v2 \
  --smooth 5
## 绘制八张子图
python3 plot_stagewise_loss_8panel.py \
  --stage1 results/convergence_training/proposed_full_stage1_damaged_w50_seed72_logged_v2_ep120 \
  --stage2 results/convergence_training/proposed_full_stage2_damaged_w50_seed72_logged_v2_warm20_margin \
  --out results/convergence_outputs/pic_train/stagewise_loss_8panel_seed72_v3 \
  --train-smooth 20 \
  --eval-smooth 5 \
  --seed-label 72


python3 plot_stagewise_loss_8panel.py \
  --stage1 results/convergence_training/proposed_full_stage1_damaged_w50_seed72_logged_v2_ep120 \
  --stage2 results/convergence_training/proposed_full_stage2_damaged_w50_seed72_lossstable_ep700 \
  --out results/convergence_outputs/pic_train/stagewise_loss_8panel_seed72_v3 \
  --stage1-smooth 20 \
  --stage2-smooth 20 \
  --eval-smooth 5 \
  --seed-label 72



cd ~/projects/uav_mec_sci
source .venv/bin/activate

python3 plot_final_stagewise_figures.py \
  --stage1 results/convergence_training/proposed_full_stage1_main_d25_seed72_ep120 \
  --stage2 results/convergence_training/proposed_full_stage2_main_d25_seed72_ep700 \
  --out results/convergence_outputs/pic_train/final_env_seed72_figures_3 \
  --seed-label 72 \
  --stage1-smooth 10 \
  --stage2-smooth 20 \
  --eval-smooth 5



cd ~/projects/uav_mec_sci
source .venv/bin/activate

DEVICE=auto \
SEED=72 \
NUM_EPISODES=250 \
EPISODE_LENGTH=20 \
DEADLINE_SCALE=2.5 \
TASK_LOCAL_CPU_MIN=2000 \
TASK_LOCAL_CPU_MAX=5000 \
STAGE1_PREFIX=proposed_full_stage1_main_d25_seed72_ep120 \
RUN_NAME=proposed_full_stage2_main_d25_seed72_ep700_v3_collab \
CKPT_PREFIX=proposed_full_stage2_main_d25_seed72_ep700_v3_collab \
python3 -m train.train_proposed_full_stage2_converge_v3_collab


cd ~/projects/uav_mec_sci
source .venv/bin/activate

DEVICE=auto \
SEED=72 \
NUM_EPISODES=250 \
EPISODE_LENGTH=20 \
DEADLINE_SCALE=2.5 \
TASK_LOCAL_CPU_MIN=2000 \
TASK_LOCAL_CPU_MAX=5000 \
STAGE1_PREFIX=proposed_full_stage1_main_d25_seed72_ep120 \
RUN_NAME=proposed_full_stage2_main_d25_seed72_ep250_v5_sil \
CKPT_PREFIX=proposed_full_stage2_main_d25_seed72_ep250_v5_sil \
python3 -m train.train_proposed_full_stage2_converge_v5_sil_trust


python3 ./eval/eval_refinement_ablation.py \
  --stage2-prefix proposed_full_stage2_main_d25_seed72_ep700_v6_model_refine \
  --stage1-prefix proposed_full_stage1_main_d25_seed72_ep120 \
  --include-stage1 \
  --include-random \
  --seeds 72 \
  --deadline-scale 2.5 \
  --task-local-cpu-min 2000 \
  --task-local-cpu-max 5000 \
  --episode-length 20 \
  --model-refine-max-tasks 6 \
  --model-refine-ratio 1 \
  --model-refine-sched 1 \
  --out results/ablation_refinement/seed72_v6

## 跑v6权重的命令（多个跑）
chmod +x scripts/formal_10seed_train_and_eval.sh
bash scripts/formal_10seed_train_and_eval.sh

## 跑v6权重的命令（单个跑）
cd ~/projects/uav_mec_sci
source .venv/bin/activate

DEVICE=auto \
SEED=72 \
NUM_EPISODES=700 \
EPISODE_LENGTH=20 \
DEADLINE_SCALE=2.5 \
TASK_LOCAL_CPU_MIN=2000 \
TASK_LOCAL_CPU_MAX=5000 \
STAGE1_PREFIX=proposed_full_stage1_main_d25_seed72_ep120 \
RUN_NAME=proposed_full_stage2_main_d25_seed72_ep700_v6_model_refine \
CKPT_PREFIX=proposed_full_stage2_main_d25_seed72_ep700_v6_model_refine \
MODEL_REFINE=1 \
MODEL_REFINE_MAX_TASKS=6 \
MODEL_REFINE_SCHED=1 \
MODEL_REFINE_RATIO=1 \
python3 -m train.train_proposed_full_stage2_converge_v6_model_refine

## 跑pure_maddpg训练权重（多个跑）
chmod +x scripts/train_pure_maddpg_10seed_formal10.sh
bash scripts/train_pure_maddpg_10seed_formal10.sh

## 跑pure_maddpg训练权重命令（单个跑）
cd ~/projects/uav_mec_sci
source .venv/bin/activate

DEVICE=auto \
SEED=72 \
NUM_EPISODES=700 \
EPISODE_LENGTH=20 \
DEADLINE_SCALE=2.5 \
TASK_LOCAL_CPU_MIN=2000 \
TASK_LOCAL_CPU_MAX=5000 \
OMEGA1=50 \
OMEGA2=1 \
RUN_NAME=pure_maddpg_main_d25_seed72_ep700 \
CKPT_PREFIX=pure_maddpg_main_d25_seed72_ep700 \
python3 -m train.train_pure_maddpg_sci_v6

## 跑著对比实验：一个训练 checkpoint + 多个 eval seeds
cd ~/projects/uav_mec_sci
source .venv/bin/activate
# 1. 先检查 10 个 seed 的 Proposed / Pure 权重是否齐全
# 2. 再运行主对比
chmod +x scripts/run_main_comparison_formal10_trainseeds.sh
bash scripts/run_main_comparison_formal10_trainseeds.sh
# 3. 最后汇总最终主表
python3 scripts/aggregate_formal10_main_comparison.py

# 跑主对比实验的命令（单个单个跑）
cd ~/projects/uav_mec_sci
source .venv/bin/activate

PYTHONPATH=$(pwd) python3 ./eval/run_main_comparison_sci_v6.py \
  --stage2-prefix proposed_full_stage2_main_d25_seed72_ep700_v6_model_refine \
  --stage1-prefix proposed_full_stage1_main_d25_seed72_ep120 \
  --pure-prefix pure_maddpg_main_d25_seed72_ep700 \
  --seeds 42,52,62,72,82,92,102,112,122,132 \
  --deadline-scale 2.5 \
  --task-local-cpu-min 2000 \
  --task-local-cpu-max 5000 \
  --episode-length 20 \
  --include-refine-controls \
  --out results/main_comparison_sci_v6/final_d25_eval10_with_pure_all

绘制十组种子收敛图
cd ~/projects/uav_mec_sci
source .venv/bin/activate

python3 plot_formal10_convergence_loss.py \
  --root results/convergence_training \
  --seeds 42,52,62,72,82,92,102,112,122,132 \
  --band ci95 \
  --train-smooth 15 \
  --eval-smooth 3 \
  --out results/convergence_outputs/formal10_stage2_loss_ci95



PYTHONPATH=$(pwd) python3 ./eval/run_main_comparison_sci_v6.py \
  --stage2-prefix proposed_full_stage2_main_d25_seed72_ep700_v6_model_refine_formal10 \
  --stage1-prefix proposed_full_stage1_main_d25_seed72_ep200_formal10 \
  --pure-prefix pure_maddpg_main_d25_seed72_ep700_formal10 \
  --seeds 42,52,62,72,82,92,102,112,122,132 \
  --deadline-scale 2.5 \
  --task-local-cpu-min 2000 \
  --task-local-cpu-max 5000 \
  --episode-length 20 \
  --model-refine-max-tasks 16 \
  --model-refine-ratio 0 \
  --model-refine-sched 1 \
  --include-refine-controls \
  --out results/main_comparison_sci_v6/diagnostic_refine16_ratio0_sched1

cd ~/projects/uav_mec_sci
source .venv/bin/activate
export PYTHONPATH=$(pwd)

TORCH_NUM_THREADS=1 \
DEVICE=auto \
SEED=72 \
NUM_EPISODES=10 \
EVAL_EVERY=1 \
EPISODE_LENGTH=20 \
DEADLINE_SCALE=2.5 \
TASK_LOCAL_CPU_MIN=2000 \
TASK_LOCAL_CPU_MAX=5000 \
STAGE1_PREFIX=proposed_full_stage1_main_d25_seed72_ep200_formal10 \
RUN_NAME=smoke_v8_save_check \
CKPT_PREFIX=smoke_v8_save_check \
MODEL_REFINE=1 \
MODEL_REFINE_MAX_TASKS=16 \
MODEL_REFINE_RATIO=1 \
MODEL_REFINE_SCHED=1 \
BEST_SELECT_MODE=actoronly \
python3 -m train.train_proposed_full_stage2_converge_v8_transformer_sched

#v8 绘制论文图片
cd ~/projects/uav_mec_sci
source .venv/bin/activate
export PYTHONPATH=$(pwd)

python3 plot_sci_v8_paper_figures.py \
  --pgboost-root results/convergence_training/v8_matched_stage1_stage2/pgboost \
  --nopg-root results/convergence_training/v8_matched_stage1_stage2/no_pg \
  --main-root results/main_comparison_sci_v8/matched_formal10_trainseed_eval10 \
  --out results/paper_figures_v8 \
  --smooth-train 9 \
  --smooth-eval 3 \
  --band std
如果你想让误差带用 95% CI，而不是标准差，改成：
--band ci95
```



