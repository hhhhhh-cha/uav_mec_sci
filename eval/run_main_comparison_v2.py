from __future__ import annotations

# 这个脚本的职责不是训练，而是统一评测五类方法：
# Random / Greedy_Simple / Greedy / Pure MADDPG / Proposed_full_stage2_fix
#
# Proposed_full_stage2_fix 加载：
#   proposed_full_stage2_fix_best_actor.pth
#   proposed_full_stage2_fix_best_encoder.pth
#   proposed_full_stage2_fix_best_fusion.pth
#   proposed_full_stage2_fix_best_ratio_head.pth
#
# 注意：
#   这里 Pure MADDPG 默认不强制要求 checkpoint。
#   如果 pure_maddpg_actor_ckpt=None，则会用随机初始化的 Pure MADDPG actor 跑评测。
#   这可以用于跑通流程，但若用于论文正式主对比，最好还是补训练后的 Pure MADDPG 权重。


from __future__ import annotations

"""
Main comparison script under the NEW stable setting:

Methods:
    1) Random
    2) Greedy
    3) Pure MADDPG
    4) Proposed_full_stage2_fix

Important:
    - This script is for unified evaluation, NOT training.
    - All methods are evaluated under the same environment setting.
    - Proposed uses the validated main setting:
        episode_length = 20
        deadline_scale = 5.0
        task_local_cpu_min = 2e3
        task_local_cpu_max = 6e3
"""
import argparse
import csv
import json
import time
import random
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import torch

from env.mec_env import MultiUavMecEnv
from env.association import build_access_association

from policy.random_policy import generate_random_high_action
from policy.greedy_policy import generate_greedy_high_action
from policy.pure_maddpg_policy import PureMADDPGPolicy

from policy.proposed_policy import build_default_proposed_policy
from model.mlp_actor import MLPActor
from model.proposed_obs_builder import build_global_observation


# ============================================================
# Paths
# ============================================================
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parent.parent
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

BASE_RESULT_DIR = PROJECT_ROOT / "results" / "main_comparison_v2"
BASE_RESULT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_run_result_dir() -> Path:
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = BASE_RESULT_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


# ============================================================
# Config
# ============================================================
@dataclass
class ComparisonConfig:
    seed: int = 72
    num_eval_episodes: int = 200

    # env params
    M: int = 3
    K: int = 16
    episode_length: int = 20
    area_size: float = 100.0
    altitude: float = 20.0
    neighbor_radius: float = 50.0
    delta_t: float = 1.0
    max_speed: float = 15.0
    min_uav_distance: float = 3.0

    cpu_mode: str = "kkt"
    prop_rho: float = 0.45
    omega1: float = 50.0
    omega2: float = 1.0
    penalty_coeff: float = 50.0
    R_min: float = 0.05
    deadline_scale: float = 5.0

    uav_energy_min: float = 2600.0
    uav_energy_max: float = 3800.0

    task_local_cpu_min: float = 2.0e3
    task_local_cpu_max: float = 6.0e3

    # checkpoints
    # update this if your pure maddpg filename is different
    pure_maddpg_actor_ckpt: Optional[str] = "pure_maddpg_k16_ep40_best_actor.pth"

    proposed_actor_ckpt: str = "proposed_full_stage2_fix_k16_ep40_best_actor.pth"
    proposed_encoder_ckpt: str = "proposed_full_stage2_fix_k16_ep40_best_encoder.pth"
    proposed_fusion_ckpt: str = "proposed_full_stage2_fix_k16_ep40_best_fusion.pth"
    proposed_ratio_head_ckpt: str = "proposed_full_stage2_fix_k16_ep40_best_ratio_head.pth"

    # save
    save_episode_csv: bool = True
    save_summary_csv: bool = True
    save_summary_json: bool = True

    # debug
    verbose: bool = True


# ============================================================
# Utils
# ============================================================
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        if isinstance(x, (list, tuple, np.ndarray)):
            arr = np.asarray(x, dtype=float)
            if arr.size == 0:
                return default
            return float(np.mean(arr))
        return float(x)
    except Exception:
        return default


def maybe_load_state_dict(
    module: torch.nn.Module,
    ckpt_path: Path,
    strict: bool = True,
    allow_partial: bool = False,
) -> bool:
    if not ckpt_path.exists():
        return False

    obj = torch.load(str(ckpt_path), map_location=DEVICE)
    if isinstance(obj, dict) and "state_dict" in obj:
        state_dict = obj["state_dict"]
    elif isinstance(obj, dict):
        state_dict = obj
    else:
        raise ValueError(f"Unsupported checkpoint format: {type(obj)}")

    if strict and not allow_partial:
        module.load_state_dict(state_dict)
        return True

    current = module.state_dict()
    matched = {}
    skipped = []

    for k, v in state_dict.items():
        if k in current and current[k].shape == v.shape:
            matched[k] = v
        else:
            old_shape = tuple(v.shape) if hasattr(v, "shape") else None
            new_shape = tuple(current[k].shape) if k in current else None
            skipped.append((k, old_shape, new_shape))

    current.update(matched)
    module.load_state_dict(current, strict=False)

    print(
        f"[INFO] Partial load for {module.__class__.__name__}: "
        f"matched={len(matched)}, skipped={len(skipped)}"
    )
    for k, old_shape, new_shape in skipped:
        print(f"  [SKIP] {k}: ckpt={old_shape}, model={new_shape}")

    return True


def build_env(cfg: ComparisonConfig, seed: Optional[int] = None) -> MultiUavMecEnv:
    env = MultiUavMecEnv(
        M=cfg.M,
        K=cfg.K,
        episode_length=cfg.episode_length,
        area_size=cfg.area_size,
        altitude=cfg.altitude,
        neighbor_radius=cfg.neighbor_radius,
        delta_t=cfg.delta_t,
        max_speed=cfg.max_speed,
        min_uav_distance=cfg.min_uav_distance,
        cpu_mode=cfg.cpu_mode,
        prop_rho=cfg.prop_rho,
        omega1=cfg.omega1,
        omega2=cfg.omega2,
        penalty_coeff=cfg.penalty_coeff,
        R_min=cfg.R_min,
        deadline_scale=cfg.deadline_scale,
        uav_energy_min=cfg.uav_energy_min,
        uav_energy_max=cfg.uav_energy_max,
        task_local_cpu_min=cfg.task_local_cpu_min,
        task_local_cpu_max=cfg.task_local_cpu_max,
        seed=seed,
    )
    return env


# ============================================================
# Policy Adapters
# ============================================================
class RandomPolicyAdapter:
    def __init__(self, seed: int = 72):
        self.rng = np.random.default_rng(seed)
        self.call_count = 0

    def act(
        self,
        state: Dict[str, Any],
        access_assoc: np.ndarray,
        deterministic: bool = True,
    ) -> Dict[str, np.ndarray]:
        # use a changing seed so random policy is truly random across calls
        local_seed = int(self.rng.integers(0, 2**31 - 1))
        self.call_count += 1
        return generate_random_high_action(
            state=state,
            access_assoc=access_assoc,
            seed=local_seed,
        )


class GreedyPolicyAdapter:
    def __init__(self, seed: int = 72):
        self.seed = seed

    def act(
        self,
        state: Dict[str, Any],
        access_assoc: np.ndarray,
        deterministic: bool = True,
    ) -> Dict[str, np.ndarray]:
        return generate_greedy_high_action(
            state=state,
            access_assoc=access_assoc,
            seed=self.seed,
        )


# ============================================================
# Proposed loader
# ============================================================
def build_proposed_full_stage2_fix_policy(
    env: MultiUavMecEnv,
    cfg: ComparisonConfig,
):
    if env.state is None:
        raise RuntimeError("env.state is None. Call env.reset() before building proposed policy.")

    obs = build_global_observation(env.state)
    obs_dim = int(np.asarray(obs, dtype=np.float32).shape[0])
    action_dim = cfg.M + cfg.M + cfg.K + cfg.K * cfg.M

    actor_net = MLPActor(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dim=256,
    ).to(DEVICE)

    actor_ok = maybe_load_state_dict(actor_net, CHECKPOINT_DIR / cfg.proposed_actor_ckpt)
    if not actor_ok:
        raise FileNotFoundError(
            f"Proposed actor checkpoint not found: {CHECKPOINT_DIR / cfg.proposed_actor_ckpt}"
        )
    actor_net.eval()

    proposed_policy = build_default_proposed_policy(
        state=env.state,
        actor_net=actor_net,
        device=str(DEVICE),
    )

    encoder_ok = maybe_load_state_dict(
        proposed_policy.encoder,
        CHECKPOINT_DIR / cfg.proposed_encoder_ckpt,
    )

    fusion_ok = maybe_load_state_dict(
        proposed_policy.fusion_net,
        CHECKPOINT_DIR / cfg.proposed_fusion_ckpt,
    )

    ratio_ok = maybe_load_state_dict(
        proposed_policy.ratio_head,
        CHECKPOINT_DIR / cfg.proposed_ratio_head_ckpt,
        strict=False,
        allow_partial=True,
    )

    if not (encoder_ok and fusion_ok):
        raise FileNotFoundError(
            "Some proposed checkpoints are missing. "
            f"encoder_ok={encoder_ok}, fusion_ok={fusion_ok}, ratio_ok={ratio_ok}"
        )

    if not ratio_ok:
        print("[WARN] Ratio head checkpoint not found. Use newly initialized ratio head.")

    proposed_policy.encoder.eval()
    proposed_policy.fusion_net.eval()
    proposed_policy.ratio_head.eval()

    return proposed_policy


# ============================================================
# Metrics
# ============================================================
def extract_metrics_from_info(info: Dict[str, Any]) -> Dict[str, float]:
    metrics = info.get("metrics", {})
    report = info.get("report", {})

    delay = safe_float(metrics.get("delay_sys", 0.0))
    energy = safe_float(metrics.get("energy_sys", 0.0))

    deadline_violation = safe_float(report.get("deadline_violation", 0.0))
    ratio_violation = safe_float(report.get("ratio_violation", 0.0))
    assoc_violation = safe_float(report.get("assoc_violation", 0.0))
    schedule_violation = safe_float(report.get("schedule_violation", 0.0))
    candidate_violation = safe_float(report.get("candidate_violation", 0.0))
    bw_violation = safe_float(report.get("bw_violation", 0.0))
    cpu_violation = safe_float(report.get("cpu_violation", 0.0))
    rate_violation = safe_float(report.get("rate_violation", 0.0))
    nan_count = safe_float(report.get("nan_count", 0.0))

    feasible_ratio = 1.0 if bool(report.get("ok", False)) else 0.0

    return {
        "delay": delay,
        "energy": energy,
        "deadline_violation": deadline_violation,
        "ratio_violation": ratio_violation,
        "assoc_violation": assoc_violation,
        "schedule_violation": schedule_violation,
        "candidate_violation": candidate_violation,
        "bw_violation": bw_violation,
        "cpu_violation": cpu_violation,
        "rate_violation": rate_violation,
        "nan_count": nan_count,
        "feasible_ratio": feasible_ratio,
    }


# ============================================================
# Records
# ============================================================
@dataclass
class EpisodeRecord:
    method: str
    episode: int
    episode_reward: float
    avg_delay: float
    avg_energy: float
    avg_deadline_violation: float
    avg_ratio_violation: float
    avg_assoc_violation: float
    avg_schedule_violation: float
    avg_candidate_violation: float
    avg_bw_violation: float
    avg_cpu_violation: float
    avg_rate_violation: float
    avg_nan_count: float
    feasible_ratio: float
    episode_steps: int


@dataclass
class SummaryRecord:
    method: str
    num_eval_episodes: int
    avg_reward: float
    avg_delay: float
    avg_energy: float
    avg_deadline_violation: float
    avg_ratio_violation: float
    avg_assoc_violation: float
    avg_schedule_violation: float
    avg_candidate_violation: float
    avg_bw_violation: float
    avg_cpu_violation: float
    avg_rate_violation: float
    avg_nan_count: float
    feasible_ratio: float
    runtime_sec: float


# ============================================================
# Evaluate one policy
# ============================================================
def evaluate_policy(
    policy,
    method_name: str,
    cfg: ComparisonConfig,
) -> Tuple[List[EpisodeRecord], SummaryRecord]:
    episode_records: List[EpisodeRecord] = []
    t0 = time.time()

    for ep in range(cfg.num_eval_episodes):
        env = build_env(cfg, seed=cfg.seed + ep)
        obs = env.reset(seed=cfg.seed + ep)

        ep_reward = 0.0
        ep_delay_sum = 0.0
        ep_energy_sum = 0.0
        ep_deadline_violation_sum = 0.0
        ep_ratio_violation_sum = 0.0
        ep_assoc_violation_sum = 0.0
        ep_schedule_violation_sum = 0.0
        ep_candidate_violation_sum = 0.0
        ep_bw_violation_sum = 0.0
        ep_cpu_violation_sum = 0.0
        ep_rate_violation_sum = 0.0
        ep_nan_count_sum = 0.0
        ep_feasible_sum = 0.0
        ep_steps = 0
        done = False

        while not done:
            state = obs["raw_state"]
            access_assoc = build_access_association(state)

            action = policy.act(
                state=state,
                access_assoc=access_assoc,
                deterministic=True,
            )

            step_action = {
                "move_dist": action["move_dist"],
                "move_angle": action["move_angle"],
                "offload_ratio": action["offload_ratio"],
                "sched_beta": action["sched_beta"],
            }

            obs, reward, done, info = env.step(step_action)
            met = extract_metrics_from_info(info)

            ep_reward += float(reward)
            ep_delay_sum += met["delay"]
            ep_energy_sum += met["energy"]
            ep_deadline_violation_sum += met["deadline_violation"]
            ep_ratio_violation_sum += met["ratio_violation"]
            ep_assoc_violation_sum += met["assoc_violation"]
            ep_schedule_violation_sum += met["schedule_violation"]
            ep_candidate_violation_sum += met["candidate_violation"]
            ep_bw_violation_sum += met["bw_violation"]
            ep_cpu_violation_sum += met["cpu_violation"]
            ep_rate_violation_sum += met["rate_violation"]
            ep_nan_count_sum += met["nan_count"]
            ep_feasible_sum += met["feasible_ratio"]
            ep_steps += 1

        denom = max(ep_steps, 1)

        rec = EpisodeRecord(
            method=method_name,
            episode=ep,
            episode_reward=float(ep_reward),
            avg_delay=float(ep_delay_sum / denom),
            avg_energy=float(ep_energy_sum / denom),
            avg_deadline_violation=float(ep_deadline_violation_sum / denom),
            avg_ratio_violation=float(ep_ratio_violation_sum / denom),
            avg_assoc_violation=float(ep_assoc_violation_sum / denom),
            avg_schedule_violation=float(ep_schedule_violation_sum / denom),
            avg_candidate_violation=float(ep_candidate_violation_sum / denom),
            avg_bw_violation=float(ep_bw_violation_sum / denom),
            avg_cpu_violation=float(ep_cpu_violation_sum / denom),
            avg_rate_violation=float(ep_rate_violation_sum / denom),
            avg_nan_count=float(ep_nan_count_sum / denom),
            feasible_ratio=float(ep_feasible_sum / denom),
            episode_steps=ep_steps,
        )
        episode_records.append(rec)

        if cfg.verbose:
            print(
                f"[{method_name}] Episode {ep:03d} | "
                f"reward={rec.episode_reward:.4f} | "
                f"delay={rec.avg_delay:.4f} | "
                f"energy={rec.avg_energy:.4f} | "
                f"deadline={rec.avg_deadline_violation:.4f} | "
                f"feasible={rec.feasible_ratio:.4f}"
            )

    runtime_sec = time.time() - t0

    summary = SummaryRecord(
        method=method_name,
        num_eval_episodes=cfg.num_eval_episodes,
        avg_reward=float(np.mean([r.episode_reward for r in episode_records])),
        avg_delay=float(np.mean([r.avg_delay for r in episode_records])),
        avg_energy=float(np.mean([r.avg_energy for r in episode_records])),
        avg_deadline_violation=float(np.mean([r.avg_deadline_violation for r in episode_records])),
        avg_ratio_violation=float(np.mean([r.avg_ratio_violation for r in episode_records])),
        avg_assoc_violation=float(np.mean([r.avg_assoc_violation for r in episode_records])),
        avg_schedule_violation=float(np.mean([r.avg_schedule_violation for r in episode_records])),
        avg_candidate_violation=float(np.mean([r.avg_candidate_violation for r in episode_records])),
        avg_bw_violation=float(np.mean([r.avg_bw_violation for r in episode_records])),
        avg_cpu_violation=float(np.mean([r.avg_cpu_violation for r in episode_records])),
        avg_rate_violation=float(np.mean([r.avg_rate_violation for r in episode_records])),
        avg_nan_count=float(np.mean([r.avg_nan_count for r in episode_records])),
        feasible_ratio=float(np.mean([r.feasible_ratio for r in episode_records])),
        runtime_sec=float(runtime_sec),
    )
    return episode_records, summary


# ============================================================
# Save
# ============================================================

def save_episode_csv(records: List[EpisodeRecord], path: Path) -> None:
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(records[0]).keys()))
        writer.writeheader()
        for r in records:
            writer.writerow(asdict(r))


def save_summary_csv(records: List[SummaryRecord], path: Path) -> None:
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(records[0]).keys()))
        writer.writeheader()
        for r in records:
            writer.writerow(asdict(r))


def save_summary_json(records: List[SummaryRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in records], f, indent=2, ensure_ascii=False)

# def save_episode_csv(records: List[EpisodeRecord], path: Path) -> None:
#     if not records:
#         return
#     with open(path, "w", newline="", encoding="utf-8") as f:
#         writer = csv.DictWriter(f, fieldnames=list(asdict(records[0]).keys()))
#         writer.writeheader()
#         for r in records:
#             writer.writerow(asdict(r))


# def save_summary_csv(records: List[SummaryRecord], path: Path) -> None:
#     if not records:
#         return
#     with open(path, "w", newline="", encoding="utf-8") as f:
#         writer = csv.DictWriter(f, fieldnames=list(asdict(records[0]).keys()))
#         writer.writeheader()
#         for r in records:
#             writer.writerow(asdict(r))


# def save_summary_json(records: List[SummaryRecord], path: Path) -> None:
#     with open(path, "w", encoding="utf-8") as f:
#         json.dump([asdict(r) for r in records], f, indent=2, ensure_ascii=False)


# ============================================================
# Main
# ============================================================
# def main():
#     cfg = ComparisonConfig()
#     set_seed(cfg.seed)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg = ComparisonConfig()
    if args.seed is not None:
        cfg.seed = args.seed

    set_seed(cfg.seed)

    result_dir = make_run_result_dir()
    print(f"[DEBUG] result_dir = {result_dir}")
    print(f"[DEBUG] result_dir exists? {result_dir.exists()}")

    with open(result_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2, ensure_ascii=False)

    print("=" * 120)
    print("UAV MEC Main Comparison v2")
    print("=" * 120)
    print("PROJECT_ROOT:", PROJECT_ROOT)
    print("CHECKPOINT_DIR:", CHECKPOINT_DIR)
    print("RESULT_DIR:", result_dir)
    print("DEVICE:", DEVICE)
    print("CFG:", asdict(cfg))
    print()

    all_summaries: List[SummaryRecord] = []

    # 1) Random
    random_policy = RandomPolicyAdapter(seed=cfg.seed)
    random_eps, random_sum = evaluate_policy(random_policy, "Random", cfg)
    all_summaries.append(random_sum)
    if cfg.save_episode_csv:
        save_episode_csv(random_eps, result_dir / "random_episode_results.csv")

    # 2) Greedy
    greedy_policy = GreedyPolicyAdapter(seed=cfg.seed)
    greedy_eps, greedy_sum = evaluate_policy(greedy_policy, "Greedy", cfg)
    all_summaries.append(greedy_sum)
    if cfg.save_episode_csv:
        save_episode_csv(greedy_eps, result_dir / "greedy_episode_results.csv")

    # 3) Pure MADDPG
    env_for_dim = build_env(cfg, seed=cfg.seed)
    obs0 = env_for_dim.reset(seed=cfg.seed)
    obs_dim = int(np.asarray(build_global_observation(obs0["raw_state"]), dtype=np.float32).shape[0])

    pure_ckpt_path = None
    if cfg.pure_maddpg_actor_ckpt is not None:
        pure_ckpt_path = str(CHECKPOINT_DIR / cfg.pure_maddpg_actor_ckpt)
        if not Path(pure_ckpt_path).exists():
            print(f"[WARN] Pure MADDPG checkpoint not found: {pure_ckpt_path}")
            print("[WARN] Falling back to randomly initialized Pure MADDPG actor.")
            pure_ckpt_path = None
    else:
        print("[WARN] pure_maddpg_actor_ckpt is None.")
        print("[WARN] Pure MADDPG will run with randomly initialized actor.")

    pure_policy = PureMADDPGPolicy(
        obs_dim=obs_dim,
        M=cfg.M,
        K=cfg.K,
        checkpoint_path=pure_ckpt_path,
        device=str(DEVICE),
    )
    pure_eps, pure_sum = evaluate_policy(pure_policy, "Pure_MADDPG", cfg)
    all_summaries.append(pure_sum)
    if cfg.save_episode_csv:
        save_episode_csv(pure_eps, result_dir / "pure_maddpg_episode_results.csv")

    # 4) Proposed
    env_for_proposed = build_env(cfg, seed=cfg.seed)
    env_for_proposed.reset(seed=cfg.seed)

    proposed_policy = build_proposed_full_stage2_fix_policy(env_for_proposed, cfg)
    proposed_eps, proposed_sum = evaluate_policy(
        proposed_policy,
        "Proposed_full_stage2_fix",
        cfg,
    )
    all_summaries.append(proposed_sum)
    if cfg.save_episode_csv:
        save_episode_csv(proposed_eps, result_dir / "proposed_full_stage2_fix_episode_results.csv")

    if cfg.save_summary_csv:
        save_summary_csv(all_summaries, result_dir / "main_comparison_summary.csv")
    if cfg.save_summary_json:
        save_summary_json(all_summaries, result_dir / "main_comparison_summary.json")

    print()
    print("=" * 120)
    print("Final Summary")
    print("=" * 120)
    for s in all_summaries:
        print(
            f"{s.method:28s} | "
            f"reward={s.avg_reward:12.4f} | "
            f"delay={s.avg_delay:10.4f} | "
            f"energy={s.avg_energy:10.4f} | "
            f"deadline={s.avg_deadline_violation:10.4f} | "
            f"ratio_vio={s.avg_ratio_violation:10.4f} | "
            f"assoc_vio={s.avg_assoc_violation:10.4f} | "
            f"sched_vio={s.avg_schedule_violation:10.4f} | "
            f"cand_vio={s.avg_candidate_violation:10.4f} | "
            f"bw_vio={s.avg_bw_violation:10.4f} | "
            f"cpu_vio={s.avg_cpu_violation:10.4f} | "
            f"rate_vio={s.avg_rate_violation:10.4f} | "
            f"nan={s.avg_nan_count:8.4f} | "
            f"feasible={s.feasible_ratio:8.4f} | "
            f"time={s.runtime_sec:8.2f}s"
        )

    print()
    print(f"[INFO] Results saved to: {result_dir}")


if __name__ == "__main__":
    main()


# import csv
# import json
# import time
# import random
# from dataclasses import dataclass, asdict
# from datetime import datetime
# from pathlib import Path
# from typing import Dict, Any, List, Optional, Tuple

# import numpy as np
# import torch

# from env.mec_env import MultiUavMecEnv
# from env.association import build_access_association

# from policy.random_policy import generate_random_high_action
# from policy.greedy_policy import generate_greedy_high_action
# from policy.greedy_simple_policy import generate_greedy_simple_high_action
# from policy.pure_maddpg_policy import PureMADDPGPolicy

# from policy.proposed_policy import build_default_proposed_policy
# from model.mlp_actor import MLPActor
# from model.proposed_obs_builder import build_global_observation


# # ============================================================
# # Path
# # ============================================================
# THIS_FILE = Path(__file__).resolve()
# PROJECT_ROOT = THIS_FILE.parent.parent
# CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

# BASE_RESULT_DIR = PROJECT_ROOT / "results" / "main_comparison_v2"
# BASE_RESULT_DIR.mkdir(parents=True, exist_ok=True)

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# def make_run_result_dir() -> Path:
#     run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
#     run_dir = BASE_RESULT_DIR / run_name
#     run_dir.mkdir(parents=True, exist_ok=False)
#     return run_dir


# # ============================================================
# # Config
# # ============================================================
# @dataclass
# class ComparisonConfig:
#     seed: int = 72
#     num_eval_episodes: int = 100

#     # env params
#     M: int = 3
#     K: int = 16
#     episode_length: int = 20
#     # episode_length: int = 20
#     area_size: float = 100.0
#     altitude: float = 20.0
#     neighbor_radius: float = 50.0
#     delta_t: float = 1.0
#     max_speed: float = 15.0
#     min_uav_distance: float = 3.0

#     cpu_mode: str = "kkt"
#     prop_rho: float = 0.45
#     omega1: float = 30.0
#     omega2: float = 1.0
#     penalty_coeff: float = 50.0
#     R_min: float = 0.05
#     deadline_scale: float = 2.5

#     uav_energy_min: float = 2600.0
#     uav_energy_max: float = 3800.0

#     # checkpoints
#     # 没有训练好的 pure MADDPG 权重时，设为 None
#     # pure_maddpg_actor_ckpt: Optional[str] = None
#     pure_maddpg_actor_ckpt: Optional[str] = "pure_maddpg_actor.pth"
#     proposed_actor_ckpt: str = "proposed_full_stage2_fix_best_actor.pth"
#     proposed_encoder_ckpt: str = "proposed_full_stage2_fix_best_encoder.pth"
#     proposed_fusion_ckpt: str = "proposed_full_stage2_fix_best_fusion.pth"
#     proposed_ratio_head_ckpt: str = "proposed_full_stage2_fix_best_ratio_head.pth"

#     # save
#     save_episode_csv: bool = True
#     save_summary_csv: bool = True
#     save_summary_json: bool = True

#     # debug
#     verbose: bool = True


# # ============================================================
# # Utils
# # ============================================================
# def set_seed(seed: int) -> None:
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(seed)


# def safe_float(x: Any, default: float = 0.0) -> float:
#     try:
#         if x is None:
#             return default
#         if isinstance(x, (list, tuple, np.ndarray)):
#             arr = np.asarray(x, dtype=float)
#             if arr.size == 0:
#                 return default
#             return float(np.mean(arr))
#         return float(x)
#     except Exception:
#         return default


# def maybe_load_state_dict(
#     module: torch.nn.Module,
#     ckpt_path: Path,
#     strict: bool = True,
#     allow_partial: bool = False,
# ) -> bool:
#     if not ckpt_path.exists():
#         return False

#     obj = torch.load(str(ckpt_path), map_location=DEVICE)
#     if isinstance(obj, dict) and "state_dict" in obj:
#         state_dict = obj["state_dict"]
#     elif isinstance(obj, dict):
#         state_dict = obj
#     else:
#         raise ValueError(f"Unsupported checkpoint format: {type(obj)}")

#     if strict and not allow_partial:
#         module.load_state_dict(state_dict)
#         return True

#     current = module.state_dict()
#     matched = {}
#     skipped = []

#     for k, v in state_dict.items():
#         if k in current and current[k].shape == v.shape:
#             matched[k] = v
#         else:
#             old_shape = tuple(v.shape) if hasattr(v, "shape") else None
#             new_shape = tuple(current[k].shape) if k in current else None
#             skipped.append((k, old_shape, new_shape))

#     current.update(matched)
#     module.load_state_dict(current, strict=False)

#     print(
#         f"[INFO] Partial load for {module.__class__.__name__}: "
#         f"matched={len(matched)}, skipped={len(skipped)}"
#     )
#     for k, old_shape, new_shape in skipped:
#         print(f"  [SKIP] {k}: ckpt={old_shape}, model={new_shape}")

#     return True


# def build_env(cfg: ComparisonConfig, seed: Optional[int] = None) -> MultiUavMecEnv:
#     env = MultiUavMecEnv(
#         M=cfg.M,
#         K=cfg.K,
#         episode_length=cfg.episode_length,
#         area_size=cfg.area_size,
#         altitude=cfg.altitude,
#         neighbor_radius=cfg.neighbor_radius,
#         delta_t=cfg.delta_t,
#         max_speed=cfg.max_speed,
#         min_uav_distance=cfg.min_uav_distance,
#         cpu_mode=cfg.cpu_mode,
#         prop_rho=cfg.prop_rho,
#         omega1=cfg.omega1,
#         omega2=cfg.omega2,
#         penalty_coeff=cfg.penalty_coeff,
#         R_min=cfg.R_min,
#         deadline_scale=cfg.deadline_scale,
#         uav_energy_min=cfg.uav_energy_min,
#         uav_energy_max=cfg.uav_energy_max,
#         seed=seed,
#     )
#     return env


# # ============================================================
# # Policy Adapters
# # ============================================================
# class RandomPolicyAdapter:
#     def __init__(self, seed: int = 42):
#         self.base_seed = seed

#     def act(
#         self,
#         state: Dict[str, Any],
#         access_assoc: np.ndarray,
#         deterministic: bool = True,
#     ) -> Dict[str, np.ndarray]:
#         slot_seed = self.base_seed + int(state.get("M", 0)) + int(state.get("K", 0))
#         return generate_random_high_action(state, access_assoc, seed=slot_seed)


# class GreedySimplePolicyAdapter:
#     def __init__(self, seed: int = 42):
#         self.seed = seed

#     def act(
#         self,
#         state: Dict[str, Any],
#         access_assoc: np.ndarray,
#         deterministic: bool = True,
#     ) -> Dict[str, np.ndarray]:
#         return generate_greedy_simple_high_action(
#             state=state,
#             access_assoc=access_assoc,
#             seed=self.seed,
#         )


# class GreedyPolicyAdapter:
#     def __init__(self, seed: int = 42):
#         self.seed = seed

#     def act(
#         self,
#         state: Dict[str, Any],
#         access_assoc: np.ndarray,
#         deterministic: bool = True,
#     ) -> Dict[str, np.ndarray]:
#         return generate_greedy_high_action(
#             state=state,
#             access_assoc=access_assoc,
#             seed=self.seed,
#         )


# # ============================================================
# # Proposed loader
# # ============================================================
# def build_proposed_full_stage2_fix_policy(
#     env: MultiUavMecEnv,
#     cfg: ComparisonConfig,
# ):
#     if env.state is None:
#         raise RuntimeError("env.state is None. Call env.reset() before building proposed policy.")

#     obs = build_global_observation(env.state)
#     obs_dim = int(np.asarray(obs, dtype=np.float32).shape[0])
#     action_dim = cfg.M + cfg.M + cfg.K + cfg.K * cfg.M

#     actor_net = MLPActor(
#         obs_dim=obs_dim,
#         action_dim=action_dim,
#         hidden_dim=256,
#     ).to(DEVICE)

#     actor_ok = maybe_load_state_dict(actor_net, CHECKPOINT_DIR / cfg.proposed_actor_ckpt)
#     if not actor_ok:
#         raise FileNotFoundError(
#             f"Proposed actor checkpoint not found: {CHECKPOINT_DIR / cfg.proposed_actor_ckpt}"
#         )
#     actor_net.eval()

#     proposed_policy = build_default_proposed_policy(
#         state=env.state,
#         actor_net=actor_net,
#         device=str(DEVICE),
#     )

#     encoder_ok = maybe_load_state_dict(
#         proposed_policy.encoder,
#         CHECKPOINT_DIR / cfg.proposed_encoder_ckpt,
#     )

#     fusion_ok = maybe_load_state_dict(
#         proposed_policy.fusion_net,
#         CHECKPOINT_DIR / cfg.proposed_fusion_ckpt,
#     )

#     ratio_ok = maybe_load_state_dict(
#         proposed_policy.ratio_head,
#         CHECKPOINT_DIR / cfg.proposed_ratio_head_ckpt,
#         strict=False,
#         allow_partial=True,
#     )

#     if not (encoder_ok and fusion_ok):
#         raise FileNotFoundError(
#             "Some proposed full_stage2_fix checkpoints are missing. "
#             f"encoder_ok={encoder_ok}, fusion_ok={fusion_ok}, ratio_ok={ratio_ok}"
#         )

#     if not ratio_ok:
#         print("[WARN] Ratio head checkpoint not found. Use new-initialized RatioHead.")

#     proposed_policy.encoder.eval()
#     proposed_policy.fusion_net.eval()
#     proposed_policy.ratio_head.eval()

#     return proposed_policy


# # ============================================================
# # Metrics
# # ============================================================
# def extract_metrics_from_info(info: Dict[str, Any]) -> Dict[str, float]:
#     metrics = info.get("metrics", {})
#     report = info.get("report", {})

#     delay = safe_float(metrics.get("delay_sys", 0.0))
#     energy = safe_float(metrics.get("energy_sys", 0.0))

#     deadline_violation = safe_float(report.get("deadline_violation", 0.0))
#     ratio_violation = safe_float(report.get("ratio_violation", 0.0))
#     assoc_violation = safe_float(report.get("assoc_violation", 0.0))
#     schedule_violation = safe_float(report.get("schedule_violation", 0.0))
#     candidate_violation = safe_float(report.get("candidate_violation", 0.0))
#     bw_violation = safe_float(report.get("bw_violation", 0.0))
#     cpu_violation = safe_float(report.get("cpu_violation", 0.0))
#     rate_violation = safe_float(report.get("rate_violation", 0.0))
#     nan_count = safe_float(report.get("nan_count", 0.0))

#     feasible_ratio = 1.0 if bool(report.get("ok", False)) else 0.0

#     return {
#         "delay": delay,
#         "energy": energy,
#         "deadline_violation": deadline_violation,
#         "ratio_violation": ratio_violation,
#         "assoc_violation": assoc_violation,
#         "schedule_violation": schedule_violation,
#         "candidate_violation": candidate_violation,
#         "bw_violation": bw_violation,
#         "cpu_violation": cpu_violation,
#         "rate_violation": rate_violation,
#         "nan_count": nan_count,
#         "feasible_ratio": feasible_ratio,
#     }


# # ============================================================
# # Records
# # ============================================================
# @dataclass
# class EpisodeRecord:
#     method: str
#     episode: int
#     episode_reward: float
#     avg_delay: float
#     avg_energy: float
#     avg_deadline_violation: float
#     avg_ratio_violation: float
#     avg_assoc_violation: float
#     avg_schedule_violation: float
#     avg_candidate_violation: float
#     avg_bw_violation: float
#     avg_cpu_violation: float
#     avg_rate_violation: float
#     avg_nan_count: float
#     feasible_ratio: float
#     episode_steps: int


# @dataclass
# class SummaryRecord:
#     method: str
#     num_eval_episodes: int
#     avg_reward: float
#     avg_delay: float
#     avg_energy: float
#     avg_deadline_violation: float
#     avg_ratio_violation: float
#     avg_assoc_violation: float
#     avg_schedule_violation: float
#     avg_candidate_violation: float
#     avg_bw_violation: float
#     avg_cpu_violation: float
#     avg_rate_violation: float
#     avg_nan_count: float
#     feasible_ratio: float
#     runtime_sec: float


# # ============================================================
# # Evaluate one policy
# # ============================================================
# def evaluate_policy(
#     policy,
#     method_name: str,
#     cfg: ComparisonConfig,
# ) -> Tuple[List[EpisodeRecord], SummaryRecord]:
#     episode_records: List[EpisodeRecord] = []
#     t0 = time.time()

#     for ep in range(cfg.num_eval_episodes):
#         env = build_env(cfg, seed=cfg.seed + ep)
#         obs = env.reset(seed=cfg.seed + ep)

#         if ep == 0 and cfg.verbose:
#             print(
#                 f"[{method_name}] DEBUG cfg energy range: "
#                 f"{cfg.uav_energy_min} ~ {cfg.uav_energy_max}"
#             )
#             print(f"[{method_name}] DEBUG sampled initial energy obs: {obs['uav_energy']}")
#             print(f"[{method_name}] DEBUG sampled initial energy raw_state: {obs['raw_state']['uav_energy']}")

#         ep_reward = 0.0
#         ep_delay_sum = 0.0
#         ep_energy_sum = 0.0
#         ep_deadline_violation_sum = 0.0
#         ep_ratio_violation_sum = 0.0
#         ep_assoc_violation_sum = 0.0
#         ep_schedule_violation_sum = 0.0
#         ep_candidate_violation_sum = 0.0
#         ep_bw_violation_sum = 0.0
#         ep_cpu_violation_sum = 0.0
#         ep_rate_violation_sum = 0.0
#         ep_nan_count_sum = 0.0
#         ep_feasible_sum = 0.0
#         ep_steps = 0
#         done = False

#         while not done:
#             state = obs["raw_state"]
#             access_assoc = build_access_association(state)

#             action = policy.act(
#                 state=state,
#                 access_assoc=access_assoc,
#                 deterministic=True,
#             )

#             if ep == 0 and ep_steps < 3 and cfg.verbose:
#                 print(f"[{method_name}] step={ep_steps}")
#                 print("move_dist:", np.round(action["move_dist"], 4))
#                 print("move_angle:", np.round(action["move_angle"], 4))
#                 print("offload_ratio:", np.round(action["offload_ratio"], 4))
#                 print("sched_beta sum:", float(np.sum(action["sched_beta"])))
#                 print(
#                     "sched_beta argmax per task:",
#                     np.argmax(action["sched_beta"], axis=2)
#                     if action["sched_beta"].ndim == 3 else "N/A",
#                 )

#             step_action = {
#                 "move_dist": action["move_dist"],
#                 "move_angle": action["move_angle"],
#                 "offload_ratio": action["offload_ratio"],
#                 "sched_beta": action["sched_beta"],
#             }

#             obs, reward, done, info = env.step(step_action)
#             met = extract_metrics_from_info(info)

#             ep_reward += float(reward)
#             ep_delay_sum += met["delay"]
#             ep_energy_sum += met["energy"]
#             ep_deadline_violation_sum += met["deadline_violation"]
#             ep_ratio_violation_sum += met["ratio_violation"]
#             ep_assoc_violation_sum += met["assoc_violation"]
#             ep_schedule_violation_sum += met["schedule_violation"]
#             ep_candidate_violation_sum += met["candidate_violation"]
#             ep_bw_violation_sum += met["bw_violation"]
#             ep_cpu_violation_sum += met["cpu_violation"]
#             ep_rate_violation_sum += met["rate_violation"]
#             ep_nan_count_sum += met["nan_count"]
#             ep_feasible_sum += met["feasible_ratio"]
#             ep_steps += 1

#         denom = max(ep_steps, 1)

#         rec = EpisodeRecord(
#             method=method_name,
#             episode=ep,
#             episode_reward=float(ep_reward),
#             avg_delay=float(ep_delay_sum / denom),
#             avg_energy=float(ep_energy_sum / denom),
#             avg_deadline_violation=float(ep_deadline_violation_sum / denom),
#             avg_ratio_violation=float(ep_ratio_violation_sum / denom),
#             avg_assoc_violation=float(ep_assoc_violation_sum / denom),
#             avg_schedule_violation=float(ep_schedule_violation_sum / denom),
#             avg_candidate_violation=float(ep_candidate_violation_sum / denom),
#             avg_bw_violation=float(ep_bw_violation_sum / denom),
#             avg_cpu_violation=float(ep_cpu_violation_sum / denom),
#             avg_rate_violation=float(ep_rate_violation_sum / denom),
#             avg_nan_count=float(ep_nan_count_sum / denom),
#             feasible_ratio=float(ep_feasible_sum / denom),
#             episode_steps=ep_steps,
#         )
#         episode_records.append(rec)

#         if cfg.verbose:
#             print(
#                 f"[{method_name}] Episode {ep:03d} | "
#                 f"reward={rec.episode_reward:.4f} | "
#                 f"delay={rec.avg_delay:.4f} | "
#                 f"energy={rec.avg_energy:.4f} | "
#                 f"deadline={rec.avg_deadline_violation:.4f} | "
#                 f"ratio_vio={rec.avg_ratio_violation:.4f} | "
#                 f"assoc_vio={rec.avg_assoc_violation:.4f} | "
#                 f"sched_vio={rec.avg_schedule_violation:.4f} | "
#                 f"cand_vio={rec.avg_candidate_violation:.4f} | "
#                 f"bw_vio={rec.avg_bw_violation:.4f} | "
#                 f"cpu_vio={rec.avg_cpu_violation:.4f} | "
#                 f"rate_vio={rec.avg_rate_violation:.4f} | "
#                 f"nan={rec.avg_nan_count:.4f} | "
#                 f"feasible={rec.feasible_ratio:.4f}"
#             )

#     runtime_sec = time.time() - t0

#     summary = SummaryRecord(
#         method=method_name,
#         num_eval_episodes=cfg.num_eval_episodes,
#         avg_reward=float(np.mean([r.episode_reward for r in episode_records])),
#         avg_delay=float(np.mean([r.avg_delay for r in episode_records])),
#         avg_energy=float(np.mean([r.avg_energy for r in episode_records])),
#         avg_deadline_violation=float(np.mean([r.avg_deadline_violation for r in episode_records])),
#         avg_ratio_violation=float(np.mean([r.avg_ratio_violation for r in episode_records])),
#         avg_assoc_violation=float(np.mean([r.avg_assoc_violation for r in episode_records])),
#         avg_schedule_violation=float(np.mean([r.avg_schedule_violation for r in episode_records])),
#         avg_candidate_violation=float(np.mean([r.avg_candidate_violation for r in episode_records])),
#         avg_bw_violation=float(np.mean([r.avg_bw_violation for r in episode_records])),
#         avg_cpu_violation=float(np.mean([r.avg_cpu_violation for r in episode_records])),
#         avg_rate_violation=float(np.mean([r.avg_rate_violation for r in episode_records])),
#         avg_nan_count=float(np.mean([r.avg_nan_count for r in episode_records])),
#         feasible_ratio=float(np.mean([r.feasible_ratio for r in episode_records])),
#         runtime_sec=float(runtime_sec),
#     )
#     return episode_records, summary


# # ============================================================
# # Save
# # ============================================================
# def save_episode_csv(records: List[EpisodeRecord], path: Path) -> None:
#     if not records:
#         return
#     with open(path, "w", newline="", encoding="utf-8") as f:
#         writer = csv.DictWriter(f, fieldnames=list(asdict(records[0]).keys()))
#         writer.writeheader()
#         for r in records:
#             writer.writerow(asdict(r))


# def save_summary_csv(records: List[SummaryRecord], path: Path) -> None:
#     if not records:
#         return
#     with open(path, "w", newline="", encoding="utf-8") as f:
#         writer = csv.DictWriter(f, fieldnames=list(asdict(records[0]).keys()))
#         writer.writeheader()
#         for r in records:
#             writer.writerow(asdict(r))


# def save_summary_json(records: List[SummaryRecord], path: Path) -> None:
#     with open(path, "w", encoding="utf-8") as f:
#         json.dump([asdict(r) for r in records], f, indent=2, ensure_ascii=False)


# # ============================================================
# # Main
# # ============================================================
# def main():
#     cfg = ComparisonConfig()
#     set_seed(cfg.seed)

#     result_dir = make_run_result_dir()

#     with open(result_dir / "config.json", "w", encoding="utf-8") as f:
#         json.dump(asdict(cfg), f, indent=2, ensure_ascii=False)

#     print("=" * 120)
#     print("UAV MEC Main Comparison v2")
#     print("=" * 120)
#     print("PROJECT_ROOT:", PROJECT_ROOT)
#     print("CHECKPOINT_DIR:", CHECKPOINT_DIR)
#     print("RESULT_DIR:", result_dir)
#     print("DEVICE:", DEVICE)
#     print("CFG:", asdict(cfg))
#     print()

#     all_summaries: List[SummaryRecord] = []

#     # 1) Random
#     random_policy = RandomPolicyAdapter(seed=cfg.seed)
#     random_eps, random_sum = evaluate_policy(random_policy, "Random", cfg)
#     all_summaries.append(random_sum)
#     if cfg.save_episode_csv:
#         save_episode_csv(random_eps, result_dir / "random_episode_results.csv")

#     # 2) Greedy Simple
#     greedy_simple_policy = GreedySimplePolicyAdapter(seed=cfg.seed)
#     greedy_simple_eps, greedy_simple_sum = evaluate_policy(
#         greedy_simple_policy, "Greedy_Simple", cfg
#     )
#     all_summaries.append(greedy_simple_sum)
#     if cfg.save_episode_csv:
#         save_episode_csv(greedy_simple_eps, result_dir / "greedy_simple_episode_results.csv")

#     # 3) Greedy
#     greedy_policy = GreedyPolicyAdapter(seed=cfg.seed)
#     greedy_eps, greedy_sum = evaluate_policy(greedy_policy, "Greedy", cfg)
#     all_summaries.append(greedy_sum)
#     if cfg.save_episode_csv:
#         save_episode_csv(greedy_eps, result_dir / "greedy_episode_results.csv")

#     # 4) Pure MADDPG
#     env_for_dim = build_env(cfg, seed=cfg.seed)
#     obs0 = env_for_dim.reset(seed=cfg.seed)
#     obs_dim = int(np.asarray(build_global_observation(obs0["raw_state"]), dtype=np.float32).shape[0])

#     pure_ckpt_path = None
#     if cfg.pure_maddpg_actor_ckpt is not None:
#         pure_ckpt_path = str(CHECKPOINT_DIR / cfg.pure_maddpg_actor_ckpt)
#         if not Path(pure_ckpt_path).exists():
#             print(f"[WARN] Pure MADDPG checkpoint not found: {pure_ckpt_path}")
#             print("[WARN] Falling back to randomly initialized Pure MADDPG actor.")
#             pure_ckpt_path = None
#     else:
#         print("[WARN] pure_maddpg_actor_ckpt is None.")
#         print("[WARN] Pure MADDPG will run with randomly initialized actor.")

#     pure_policy = PureMADDPGPolicy(
#         obs_dim=obs_dim,
#         M=cfg.M,
#         K=cfg.K,
#         checkpoint_path=pure_ckpt_path,
#         device=str(DEVICE),
#     )
#     pure_eps, pure_sum = evaluate_policy(pure_policy, "Pure_MADDPG", cfg)
#     all_summaries.append(pure_sum)
#     if cfg.save_episode_csv:
#         save_episode_csv(pure_eps, result_dir / "pure_maddpg_episode_results.csv")

#     # 5) Proposed full_stage2_fix
#     env_for_proposed = build_env(cfg, seed=cfg.seed)
#     env_for_proposed.reset(seed=cfg.seed)

#     proposed_policy = build_proposed_full_stage2_fix_policy(env_for_proposed, cfg)
#     proposed_eps, proposed_sum = evaluate_policy(
#         proposed_policy,
#         "Proposed_full_stage2_fix",
#         cfg,
#     )
#     all_summaries.append(proposed_sum)
#     if cfg.save_episode_csv:
#         save_episode_csv(proposed_eps, result_dir / "proposed_full_stage2_fix_episode_results.csv")

#     if cfg.save_summary_csv:
#         save_summary_csv(all_summaries, result_dir / "main_comparison_summary.csv")
#     if cfg.save_summary_json:
#         save_summary_json(all_summaries, result_dir / "main_comparison_summary.json")

#     print()
#     print("=" * 120)
#     print("Final Summary")
#     print("=" * 120)
#     for s in all_summaries:
#         print(
#             f"{s.method:28s} | "
#             f"reward={s.avg_reward:12.4f} | "
#             f"delay={s.avg_delay:10.4f} | "
#             f"energy={s.avg_energy:10.4f} | "
#             f"deadline={s.avg_deadline_violation:10.4f} | "
#             f"ratio_vio={s.avg_ratio_violation:10.4f} | "
#             f"assoc_vio={s.avg_assoc_violation:10.4f} | "
#             f"sched_vio={s.avg_schedule_violation:10.4f} | "
#             f"cand_vio={s.avg_candidate_violation:10.4f} | "
#             f"bw_vio={s.avg_bw_violation:10.4f} | "
#             f"cpu_vio={s.avg_cpu_violation:10.4f} | "
#             f"rate_vio={s.avg_rate_violation:10.4f} | "
#             f"nan={s.avg_nan_count:8.4f} | "
#             f"feasible={s.feasible_ratio:8.4f} | "
#             f"time={s.runtime_sec:8.2f}s"
#         )

#     print()
#     print(f"[INFO] Results saved to: {result_dir}")


# if __name__ == "__main__":
#     main()