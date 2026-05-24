#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SCI-style main comparison for the UAV-MEC hybrid learning-optimization method.

This script is intended to replace the older eval/run_main_comparison_v2.py.
It evaluates all methods under exactly the same environment configuration and
records mean/std/95% CI over evaluation seeds.

Recommended final paper modes:
  1) Proposed_Full_v6        : learned Stage-2 actor + safety projection + model refinement
  2) Proposed_ActorOnly      : learned Stage-2 actor + safety projection, no refinement
  3) Pure_MADDPG             : pure MLP actor baseline, no Transformer, no refinement
  4) Greedy                  : strong deadline-aware heuristic baseline
  5) Random                  : legal random high-level decisions

Additional refinement-control modes for rebutting the concern that refinement
alone solves the problem:
  6) Greedy_Refine           : Greedy + same model refinement
  7) Random_Refine           : Random + same model refinement

Important paper rule:
  - Do NOT use an untrained Pure MADDPG checkpoint in a formal paper table.
  - If Pure MADDPG checkpoint is missing, this script skips it by default.

Place this file in:
  ~/projects/uav_mec_sci/eval/run_main_comparison_sci_v6.py

Run example:
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
    --out results/main_comparison_sci_v6/final_d25_seed72
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

# -----------------------------------------------------------------------------
# Project imports
# -----------------------------------------------------------------------------
from env.mec_env import MultiUavMecEnv
from env.association import build_access_association
from model.mlp_actor import MLPActor
from model.proposed_obs_builder import build_global_observation, get_observation_dim
from policy.proposed_policy import build_default_proposed_policy

try:
    from policy.greedy_policy import GreedyPolicy
except Exception:
    GreedyPolicy = None

try:
    from policy.random_policy import generate_random_high_action
except Exception:
    generate_random_high_action = None

try:
    from policy.pure_maddpg_policy import PureMADDPGPolicy
except Exception:
    PureMADDPGPolicy = None

try:
    from train.train_proposed_full_stage2_converge_v6_model_refine import (
        SafeMobilityPolicyWrapper,
        ModelRefinedPolicyWrapper,
        get_actor_raw_action_dim,
    )
except Exception as exc:
    raise ImportError(
        "Cannot import v6 utilities. Copy train_proposed_full_stage2_converge_v6_model_refine.py "
        "into ~/projects/uav_mec_sci/train/ first. Original error: " + str(exc)
    )

EPS = 1e-8


# -----------------------------------------------------------------------------
# Basic helpers
# -----------------------------------------------------------------------------
def parse_seed_list(text: str) -> List[int]:
    seeds: List[int] = []
    for item in str(text).split(','):
        item = item.strip()
        if item:
            seeds.append(int(item))
    if not seeds:
        raise ValueError("At least one seed is required.")
    return seeds


def resolve_device(requested: str) -> str:
    requested = str(requested).lower()
    if requested == "cpu":
        return "cpu"
    if requested.startswith("cuda"):
        return requested if torch.cuda.is_available() else "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def set_global_seed(seed: int, device: str) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        if isinstance(x, (int, float, np.integer, np.floating)):
            val = float(x)
            return val if np.isfinite(val) else default
        arr = np.asarray(x)
        if arr.size == 1:
            val = float(arr.reshape(-1)[0])
            return val if np.isfinite(val) else default
        if arr.size > 1:
            val = float(np.nanmean(arr.astype(float)))
            return val if np.isfinite(val) else default
    except Exception:
        return default
    return default


def recursive_find_numeric(info: Any, aliases: Sequence[str]) -> float:
    if info is None:
        return 0.0
    if isinstance(info, dict):
        for key in aliases:
            if key in info:
                return safe_float(info[key], 0.0)
        for value in info.values():
            found = recursive_find_numeric(value, aliases)
            if found != 0.0:
                return found
    elif isinstance(info, (list, tuple)):
        for value in info:
            found = recursive_find_numeric(value, aliases)
            if found != 0.0:
                return found
    return 0.0


def checkpoint_paths(prefix: str, ckpt_dir: str) -> Dict[str, str]:
    base = Path(ckpt_dir)
    return {
        "actor": str(base / f"{prefix}_best_actor.pth"),
        "encoder": str(base / f"{prefix}_best_encoder.pth"),
        "fusion_net": str(base / f"{prefix}_best_fusion.pth"),
        "ratio_head": str(base / f"{prefix}_best_ratio_head.pth"),
    }


def load_state_dict_strict(module: torch.nn.Module, path: str, name: str, device: str) -> None:
    if module is None:
        raise RuntimeError(f"Module {name} is None; cannot load checkpoint {path}.")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {name} checkpoint: {path}")
    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    module.load_state_dict(state)


def set_policy_eval(policy: Any) -> None:
    for name in ["actor_net", "encoder", "fusion_net", "ratio_head", "actor"]:
        module = getattr(policy, name, None)
        if module is not None and hasattr(module, "eval"):
            module.eval()


# -----------------------------------------------------------------------------
# Environment
# -----------------------------------------------------------------------------
def make_env(args: argparse.Namespace, seed: int) -> MultiUavMecEnv:
    """Build environment using the final D2.5 setting and env.py-consistent params."""
    return MultiUavMecEnv(
        M=args.M,
        K=args.K,
        episode_length=args.episode_length,
        area_size=args.area_size,
        altitude=args.altitude,
        neighbor_radius=args.neighbor_radius,
        delta_t=args.delta_t,
        max_speed=args.max_speed,
        min_uav_distance=args.min_uav_distance,
        cpu_mode=args.cpu_mode,
        prop_rho=args.prop_rho,
        omega1=args.omega1,
        omega2=args.omega2,
        penalty_coeff=args.penalty_coeff,
        R_min=args.R_min,
        deadline_scale=args.deadline_scale,
        task_local_cpu_min=args.task_local_cpu_min,
        task_local_cpu_max=args.task_local_cpu_max,
        uav_energy_min=args.uav_energy_min,
        uav_energy_max=args.uav_energy_max,
        seed=seed,
    )


# -----------------------------------------------------------------------------
# Policies
# -----------------------------------------------------------------------------
def build_learned_proposed_policy(
    args: argparse.Namespace,
    prefix: str,
    device: str,
    build_seed: int,
) -> Any:
    env = make_env(args, build_seed)
    obs = env.reset(seed=build_seed)
    state = obs["raw_state"]

    obs_dim = get_observation_dim(state)
    actor_raw_action_dim = get_actor_raw_action_dim(state)

    actor = MLPActor(
        obs_dim=obs_dim,
        action_dim=actor_raw_action_dim,
        hidden_dim=args.hidden_dim,
    ).to(device)

    policy = build_default_proposed_policy(
        state=state,
        actor_net=actor,
        device=device,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        ff_hidden_dim=args.ff_hidden_dim,
        num_layers=args.num_layers,
    )

    paths = checkpoint_paths(prefix, args.ckpt_dir)
    load_state_dict_strict(policy.actor_net, paths["actor"], "proposed actor", device)
    load_state_dict_strict(policy.encoder, paths["encoder"], "proposed encoder", device)
    load_state_dict_strict(policy.fusion_net, paths["fusion_net"], "proposed fusion", device)
    load_state_dict_strict(policy.ratio_head, paths["ratio_head"], "proposed ratio_head", device)
    set_policy_eval(policy)
    return policy


class FunctionPolicy:
    """Wrap a function high-level policy into the policy.act interface."""

    def __init__(self, fn: Callable, seed: int = 0, name: str = "function_policy"):
        self.fn = fn
        self.seed = int(seed)
        self.name = name
        self.t = 0

    def act(
        self,
        state: Dict[str, Any],
        access_assoc: np.ndarray,
        deterministic: bool = True,
        return_aux: bool = False,
    ) -> Any:
        try:
            action = self.fn(state, access_assoc, seed=self.seed + self.t)
        except TypeError:
            action = self.fn(state, access_assoc)
        self.t += 1
        if return_aux:
            return action, {}
        return action


class FallbackRandomPolicy:
    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def act(
        self,
        state: Dict[str, Any],
        access_assoc: np.ndarray,
        deterministic: bool = True,
        return_aux: bool = False,
    ) -> Any:
        M = int(state["M"])
        K = int(state["K"])
        neighbors = state["neighbors"]
        move_dist = np.zeros(M, dtype=np.float32)
        move_angle = np.zeros(M, dtype=np.float32)
        offload_ratio = self.rng.uniform(0.0, 1.0, size=K).astype(np.float32)
        sched_beta = np.zeros((K, M, M), dtype=np.float32)
        for k in range(K):
            access_m = int(np.argmax(access_assoc[:, k]))
            legal_js = [access_m] + list(neighbors[access_m])
            j_star = int(self.rng.choice(legal_js))
            sched_beta[k, access_m, j_star] = 1.0
        action = {
            "move_dist": move_dist,
            "move_angle": move_angle,
            "offload_ratio": offload_ratio,
            "sched_beta": sched_beta,
        }
        if return_aux:
            return action, {}
        return action


class ReturnAuxCompatiblePolicy:
    """
    Adapter for policies whose act() does not support return_aux.
    This keeps the interface compatible with SafeMobilityPolicyWrapper /
    ModelRefinedPolicyWrapper.
    """

    def __init__(self, policy: Any):
        self.policy = policy

    def act(
        self,
        state: Dict[str, Any],
        access_assoc: np.ndarray,
        deterministic: bool = True,
        return_aux: bool = False,
        **kwargs: Any,
    ) -> Any:
        try:
            out = self.policy.act(
                state=state,
                access_assoc=access_assoc,
                deterministic=deterministic,
                return_aux=return_aux,
                **kwargs,
            )
            return out
        except TypeError as exc:
            if "return_aux" not in str(exc):
                raise

        try:
            action = self.policy.act(
                state=state,
                access_assoc=access_assoc,
                deterministic=deterministic,
                **kwargs,
            )
        except TypeError as exc:
            if "deterministic" not in str(exc):
                raise
            action = self.policy.act(
                state=state,
                access_assoc=access_assoc,
                **kwargs,
            )

        if return_aux:
            return action, {}
        return action


def build_random_policy(seed: int) -> Any:
    if generate_random_high_action is not None:
        return FunctionPolicy(generate_random_high_action, seed=seed, name="random")
    return FallbackRandomPolicy(seed=seed)


def build_greedy_policy(seed: int) -> Any:
    if GreedyPolicy is None:
        raise RuntimeError("policy.greedy_policy.GreedyPolicy is unavailable.")
    return ReturnAuxCompatiblePolicy(GreedyPolicy(seed=seed))


def build_pure_maddpg_policy(args: argparse.Namespace, device: str, build_seed: int) -> Optional[Any]:
    if PureMADDPGPolicy is None:
        print("[WARN] PureMADDPGPolicy is unavailable. Skip Pure_MADDPG.")
        return None
    if not args.pure_prefix:
        print("[WARN] --pure-prefix not provided. Skip Pure_MADDPG.")
        return None

    ckpt_path = Path(args.ckpt_dir) / f"{args.pure_prefix}_best_actor.pth"
    if not ckpt_path.exists():
        alt_path = Path(args.ckpt_dir) / f"{args.pure_prefix}.pth"
        if alt_path.exists():
            ckpt_path = alt_path

    if not ckpt_path.exists():
        msg = f"Pure MADDPG checkpoint not found for prefix '{args.pure_prefix}' in {args.ckpt_dir}."
        if args.allow_untrained_pure:
            print("[WARN] " + msg + " Using randomly initialized Pure MADDPG. DO NOT use this in paper.")
            ckpt_path_str = None
        else:
            print("[WARN] " + msg + " Skip Pure_MADDPG. Train it first for formal comparison.")
            return None
    else:
        ckpt_path_str = str(ckpt_path)

    env = make_env(args, build_seed)
    obs = env.reset(seed=build_seed)
    obs_dim = int(np.asarray(build_global_observation(obs["raw_state"]), dtype=np.float32).shape[0])
    policy = PureMADDPGPolicy(
        obs_dim=obs_dim,
        M=args.M,
        K=args.K,
        checkpoint_path=ckpt_path_str,
        device=device,
        hidden_dim=args.hidden_dim,
    )
    # set_policy_eval(policy)
    # return policy
    set_policy_eval(policy)
    return ReturnAuxCompatiblePolicy(policy)


def wrap_policy(env: MultiUavMecEnv, base_policy: Any, refine: bool, args: argparse.Namespace) -> Any:
    if refine:
        return ModelRefinedPolicyWrapper(
            env=env,
            policy=base_policy,
            safety_margin=args.safety_margin,
            collision_margin=args.collision_margin,
            enable_refine=True,
            max_tasks=args.model_refine_max_tasks,
            ratio_search=bool(args.model_refine_ratio),
            schedule_search=bool(args.model_refine_sched),
        )
    return SafeMobilityPolicyWrapper(
        policy=base_policy,
        safety_margin=args.safety_margin,
        collision_margin=args.collision_margin,
    )


# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------
def evaluate_policy_once(
    env: MultiUavMecEnv,
    policy: Any,
    seed: int,
) -> Dict[str, float]:
    obs = env.reset(seed=seed)
    done = False
    step_count = 0

    total_reward = 0.0
    total_delay = 0.0
    total_energy = 0.0
    feasible_count = 0.0

    sums = {
        "avg_deadline_violation": 0.0,
        "avg_ratio_violation": 0.0,
        "avg_assoc_violation": 0.0,
        "avg_schedule_violation": 0.0,
        "avg_candidate_violation": 0.0,
        "avg_bw_violation": 0.0,
        "avg_cpu_violation": 0.0,
        "avg_rate_violation": 0.0,
        "avg_nan_count": 0.0,
        "avg_move_violation": 0.0,
        "avg_boundary_violation": 0.0,
        "avg_collision_violation": 0.0,
        "avg_battery_violation": 0.0,
    }

    alias_map = {
        "avg_move_violation": ["move_violation", "motion_violation", "movement_violation", "avg_move_violation"],
        "avg_boundary_violation": ["boundary_violation", "out_of_boundary", "out_of_bounds", "avg_boundary_violation"],
        "avg_collision_violation": ["collision_violation", "collision", "safe_distance_violation", "distance_violation"],
        "avg_battery_violation": ["battery_violation", "energy_violation", "battery_negative", "avg_battery_violation"],
    }

    ratio_values: List[float] = []
    local_exec_count = 0
    neighbor_exec_count = 0
    sched_count = 0
    refine_improvement_sum = 0.0
    refine_improvement_count = 0
    refine_improved_steps = 0

    runtime_action_sec = 0.0

    while not done:
        state = obs["raw_state"]
        access_assoc = build_access_association(state)
        M = int(state["M"])
        K = int(state["K"])

        t0 = time.perf_counter()
        action = policy.act(state=state, access_assoc=access_assoc, deterministic=True)
        runtime_action_sec += time.perf_counter() - t0

        if "_model_refine_improvement" in action:
            imp = safe_float(action.get("_model_refine_improvement"), 0.0)
            refine_improvement_sum += imp
            refine_improvement_count += 1
            if imp > 1e-9:
                refine_improved_steps += 1

        if "offload_ratio" in action:
            ratio_arr = np.asarray(action["offload_ratio"], dtype=np.float32).reshape(-1)
            ratio_values.extend([float(x) for x in ratio_arr])

        if "sched_beta" in action:
            sched_beta = np.asarray(action["sched_beta"], dtype=np.float32).reshape(K, M, M)
            for k in range(K):
                access_m = int(np.argmax(access_assoc[:, k]))
                exec_j = int(np.argmax(sched_beta[k, access_m, :]))
                sched_count += 1
                if exec_j == access_m:
                    local_exec_count += 1
                else:
                    neighbor_exec_count += 1

        step_action = {
            "move_dist": action["move_dist"],
            "move_angle": action["move_angle"],
            "offload_ratio": action["offload_ratio"],
            "sched_beta": action["sched_beta"],
        }
        obs, reward, done, info = env.step(step_action)
        report = info.get("report", {})
        metrics = info.get("metrics", {})

        total_reward += float(reward)
        total_delay += safe_float(metrics.get("delay_sys"), 0.0)
        total_energy += safe_float(metrics.get("energy_sys"), 0.0)
        feasible_count += 1.0 if bool(report.get("ok", False)) else 0.0

        sums["avg_deadline_violation"] += safe_float(report.get("deadline_violation"), 0.0)
        sums["avg_ratio_violation"] += safe_float(report.get("ratio_violation"), 0.0)
        sums["avg_assoc_violation"] += safe_float(report.get("assoc_violation"), 0.0)
        sums["avg_schedule_violation"] += safe_float(report.get("schedule_violation"), 0.0)
        sums["avg_candidate_violation"] += safe_float(report.get("candidate_violation"), 0.0)
        sums["avg_bw_violation"] += safe_float(report.get("bw_violation"), 0.0)
        sums["avg_cpu_violation"] += safe_float(report.get("cpu_violation"), 0.0)
        sums["avg_rate_violation"] += safe_float(report.get("rate_violation"), 0.0)
        sums["avg_nan_count"] += safe_float(report.get("nan_count"), 0.0)
        for key, aliases in alias_map.items():
            sums[key] += recursive_find_numeric(info, aliases)

        step_count += 1

    denom = max(step_count, 1)
    ratio_np = np.asarray(ratio_values, dtype=np.float32)
    sched_denom = max(sched_count, 1)

    out = {
        "episode_reward": float(total_reward),
        "system_cost": float(-total_reward),
        "avg_delay": float(total_delay / denom),
        "avg_energy": float(total_energy / denom),
        "feasible_ratio": float(feasible_count / denom),
        "num_steps": int(step_count),
        "ratio_mean": float(np.mean(ratio_np)) if ratio_np.size else float("nan"),
        "ratio_std": float(np.std(ratio_np)) if ratio_np.size else float("nan"),
        "ratio_min": float(np.min(ratio_np)) if ratio_np.size else float("nan"),
        "ratio_max": float(np.max(ratio_np)) if ratio_np.size else float("nan"),
        "local_exec_ratio": float(local_exec_count / sched_denom),
        "neighbor_exec_ratio": float(neighbor_exec_count / sched_denom),
        "refine_improvement_per_step": float(refine_improvement_sum / max(refine_improvement_count, 1)),
        "refine_improved_step_ratio": float(refine_improved_steps / max(refine_improvement_count, 1)),
        "decision_time_per_slot_sec": float(runtime_action_sec / denom),
    }
    for key, value in sums.items():
        out[key] = float(value / denom)
    return out


def summarize_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    metrics = [
        "episode_reward", "system_cost", "avg_delay", "avg_energy", "avg_deadline_violation",
        "feasible_ratio", "avg_ratio_violation", "avg_assoc_violation", "avg_schedule_violation",
        "avg_candidate_violation", "avg_bw_violation", "avg_cpu_violation", "avg_rate_violation",
        "avg_nan_count", "avg_move_violation", "avg_boundary_violation", "avg_collision_violation",
        "avg_battery_violation", "ratio_mean", "ratio_std", "local_exec_ratio",
        "neighbor_exec_ratio", "refine_improvement_per_step", "refine_improved_step_ratio",
        "decision_time_per_slot_sec",
    ]

    methods = sorted(set(row["method"] for row in rows))
    summary: List[Dict[str, Any]] = []
    for method in methods:
        vals = [row for row in rows if row["method"] == method]
        out: Dict[str, Any] = {"method": method, "n": len(vals)}
        for metric in metrics:
            arr = np.asarray([float(v.get(metric, float("nan"))) for v in vals], dtype=np.float64)
            arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                out[f"{metric}_mean"] = float("nan")
                out[f"{metric}_std"] = float("nan")
                out[f"{metric}_ci95"] = float("nan")
                continue
            mean = float(np.mean(arr))
            std = float(np.std(arr, ddof=1)) if arr.size >= 2 else 0.0
            ci95 = float(1.96 * std / math.sqrt(arr.size)) if arr.size >= 2 else 0.0
            out[f"{metric}_mean"] = mean
            out[f"{metric}_std"] = std
            out[f"{metric}_ci95"] = ci95
        summary.append(out)

    # Sort by system cost ascending for readable ranking.
    summary.sort(key=lambda r: float(r.get("system_cost_mean", float("inf"))))
    best_cost = float(summary[0].get("system_cost_mean", float("nan"))) if summary else float("nan")
    for row in summary:
        cost = float(row.get("system_cost_mean", float("nan")))
        row["cost_gap_to_best_percent"] = float((cost - best_cost) / max(abs(best_cost), EPS) * 100.0) if np.isfinite(cost) and np.isfinite(best_cost) else float("nan")
    return summary


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def make_latex_table(summary: List[Dict[str, Any]]) -> str:
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Main comparison under the post-disaster UAV-MEC scenario.}")
    lines.append(r"\label{tab:main_comparison}")
    lines.append(r"\begin{tabular}{lcccc}")
    lines.append(r"\hline")
    lines.append(r"Method & System cost $\downarrow$ & Delay $\downarrow$ & Deadline vio. $\downarrow$ & Feasible ratio $\uparrow$ \\")
    lines.append(r"\hline")
    for row in summary:
        method = str(row["method"]).replace("_", r"\_")
        cost = f"{row['system_cost_mean']:.2f}$\\pm${row['system_cost_std']:.2f}"
        delay = f"{row['avg_delay_mean']:.2f}$\\pm${row['avg_delay_std']:.2f}"
        deadline = f"{row['avg_deadline_violation_mean']:.3f}$\\pm${row['avg_deadline_violation_std']:.3f}"
        feasible = f"{row['feasible_ratio_mean']:.3f}$\\pm${row['feasible_ratio_std']:.3f}"
        lines.append(f"{method} & {cost} & {delay} & {deadline} & {feasible} \\")
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def print_summary(summary: List[Dict[str, Any]]) -> None:
    print("\n" + "=" * 132)
    print("SCI-style main comparison summary")
    print("=" * 132)
    print(
        f"{'Method':28s} | {'Cost mean±std':22s} | {'Delay mean±std':20s} | "
        f"{'Deadline':14s} | {'Feasible':12s} | {'Decision ms/slot':16s}"
    )
    print("-" * 132)
    for row in summary:
        print(
            f"{row['method']:28s} | "
            f"{row['system_cost_mean']:9.2f}±{row['system_cost_std']:<9.2f} | "
            f"{row['avg_delay_mean']:8.3f}±{row['avg_delay_std']:<8.3f} | "
            f"{row['avg_deadline_violation_mean']:8.4f}     | "
            f"{row['feasible_ratio_mean']:8.4f}   | "
            f"{1000.0 * row['decision_time_per_slot_sec_mean']:10.3f}"
        )
    print("=" * 132 + "\n")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SCI-style main comparison for UAV-MEC v6 final method.")

    # Checkpoints.
    parser.add_argument("--stage2-prefix", type=str, required=True, help="Prefix of final Stage-2/v6 checkpoints.")
    parser.add_argument("--stage1-prefix", type=str, default="", help="Optional Stage-1 prefix for diagnostic modes.")
    parser.add_argument("--pure-prefix", type=str, default="", help="Optional Pure MADDPG actor checkpoint prefix.")
    parser.add_argument("--ckpt-dir", type=str, default="checkpoints")
    parser.add_argument("--allow-untrained-pure", action="store_true", help="Debug only. Do not use in paper.")

    # Output.
    parser.add_argument("--out", type=str, default="results/main_comparison_sci_v6")
    parser.add_argument("--seeds", type=str, default="42,52,62,72,82", help="Evaluation seeds.")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--quiet", action="store_true")

    # Env config. Default matches env.py except deadline/local CPU final D2.5 setting.
    parser.add_argument("--M", type=int, default=3)
    parser.add_argument("--K", type=int, default=16)
    parser.add_argument("--episode-length", type=int, default=20)
    parser.add_argument("--area-size", type=float, default=100.0)
    parser.add_argument("--altitude", type=float, default=50.0)
    parser.add_argument("--neighbor-radius", type=float, default=50.0)
    parser.add_argument("--delta-t", type=float, default=1.0)
    parser.add_argument("--max-speed", type=float, default=15.0)
    parser.add_argument("--min-uav-distance", type=float, default=3.0)
    parser.add_argument("--cpu-mode", type=str, default="kkt")
    parser.add_argument("--prop-rho", type=float, default=0.45)
    parser.add_argument("--omega1", type=float, default=50.0)
    parser.add_argument("--omega2", type=float, default=1.0)
    parser.add_argument("--penalty-coeff", type=float, default=50.0)
    parser.add_argument("--R-min", dest="R_min", type=float, default=0.05)
    parser.add_argument("--deadline-scale", type=float, default=2.5)
    parser.add_argument("--task-local-cpu-min", type=float, default=2000.0)
    parser.add_argument("--task-local-cpu-max", type=float, default=5000.0)
    parser.add_argument("--uav-energy-min", type=float, default=2600.0)
    parser.add_argument("--uav-energy-max", type=float, default=3800.0)

    # Proposed network architecture.
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--ff-hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=2)

    # Refinement config.
    parser.add_argument("--model-refine-max-tasks", type=int, default=6)
    parser.add_argument("--model-refine-ratio", type=int, default=1)
    parser.add_argument("--model-refine-sched", type=int, default=1)
    parser.add_argument("--safety-margin", type=float, default=1.0)
    parser.add_argument("--collision-margin", type=float, default=1e-5)

    # Method switches.
    parser.add_argument("--include-refine-controls", action="store_true", help="Add Greedy_Refine and Random_Refine control baselines.")
    parser.add_argument("--include-stage1-diagnostics", action="store_true", help="Add Stage1 actor diagnostic modes.")
    parser.add_argument("--no-random", dest="include_random", action="store_false", default=True)
    parser.add_argument("--no-greedy", dest="include_greedy", action="store_false", default=True)
    parser.add_argument("--no-proposed-actor-only", dest="include_proposed_actor_only", action="store_false", default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = parse_seed_list(args.seeds)
    device = resolve_device(args.device)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 120)
    print("SCI-style main comparison for UAV-MEC v6")
    print("=" * 120)
    print(f"device         : {device}")
    print(f"seeds          : {seeds}")
    print(f"stage2_prefix  : {args.stage2_prefix}")
    print(f"stage1_prefix  : {args.stage1_prefix or '(not used)'}")
    print(f"pure_prefix    : {args.pure_prefix or '(not used)'}")
    print(f"out            : {out_dir}")
    print("=" * 120)

    write_json(out_dir / "config.json", vars(args))

    build_seed = seeds[0]
    set_global_seed(build_seed, device)

    # Build learned policies once; dimensions are fixed under a fixed M/K/config.
    proposed_stage2 = build_learned_proposed_policy(args, args.stage2_prefix, device, build_seed)
    proposed_stage1 = None
    if args.include_stage1_diagnostics:
        if not args.stage1_prefix:
            raise ValueError("--include-stage1-diagnostics requires --stage1-prefix.")
        proposed_stage1 = build_learned_proposed_policy(args, args.stage1_prefix, device, build_seed)

    pure_policy = build_pure_maddpg_policy(args, device, build_seed)

    all_rows: List[Dict[str, Any]] = []
    global_t0 = time.time()

    for seed in seeds:
        print(f"\n[Eval seed {seed}]")
        set_global_seed(seed, device)

        modes: List[Tuple[str, Any, bool]] = []

        # Final proposed method and its actor-only diagnostic.
        modes.append(("Proposed_Full_v6", proposed_stage2, True))
        if args.include_proposed_actor_only:
            modes.append(("Proposed_ActorOnly", proposed_stage2, False))

        # Standard main baselines.
        # if pure_policy is not None:
        #     modes.append(("Pure_MADDPG", pure_policy, False))
        # if args.include_greedy:
        #     modes.append(("Greedy", build_greedy_policy(seed), False))
        # if args.include_random:
        #     modes.append(("Random", build_random_policy(seed), False))

    # Standard main baselines.
        if pure_policy is not None:
            modes.append(("Pure_MADDPG", pure_policy, False))

            # Pure MADDPG + the same one-step model refinement.
            # This control checks whether refinement alone can make the pure end-to-end DRL baseline
            # competitive with the proposed structured policy.
            if args.include_refine_controls:
                modes.append(("Pure_MADDPG_Refine", pure_policy, True))

        if args.include_greedy:
            modes.append(("Greedy", build_greedy_policy(seed), False))
        if args.include_random:
            modes.append(("Random", build_random_policy(seed), False))

        # Refinement-control baselines: important for proving refinement does not replace learning.
        if args.include_refine_controls:
            if args.include_greedy:
                modes.append(("Greedy_Refine", build_greedy_policy(seed), True))
            if args.include_random:
                modes.append(("Random_Refine", build_random_policy(seed), True))

        # Optional diagnostics for Stage-1 vs Stage-2 actor.
        if proposed_stage1 is not None:
            modes.append(("Stage1_ActorOnly", proposed_stage1, False))
            modes.append(("Stage1_Refine", proposed_stage1, True))

        for method_name, base_policy, refine in modes:
            env = make_env(args, seed)
            eval_policy = wrap_policy(env, base_policy, refine=refine, args=args)
            result = evaluate_policy_once(env, eval_policy, seed=seed)
            row: Dict[str, Any] = {
                "method": method_name,
                "seed": seed,
                "refine_enabled": int(refine),
                "stage2_prefix": args.stage2_prefix,
                "stage1_prefix": args.stage1_prefix,
                "pure_prefix": args.pure_prefix,
                "deadline_scale": args.deadline_scale,
                "task_local_cpu_min": args.task_local_cpu_min,
                "task_local_cpu_max": args.task_local_cpu_max,
                "episode_length": args.episode_length,
                "altitude": args.altitude,
                "model_refine_max_tasks": args.model_refine_max_tasks,
                "model_refine_ratio": int(args.model_refine_ratio),
                "model_refine_sched": int(args.model_refine_sched),
            }
            row.update(result)
            all_rows.append(row)

            if not args.quiet:
                print(
                    f"  {method_name:22s} | cost={row['system_cost']:10.3f} "
                    f"delay={row['avg_delay']:8.4f} feasible={row['feasible_ratio']:.3f} "
                    f"deadline={row['avg_deadline_violation']:.4f} "
                    f"local={row['local_exec_ratio']:.4f} neigh={row['neighbor_exec_ratio']:.4f} "
                    f"t={1000.0 * row['decision_time_per_slot_sec']:.2f}ms"
                )

    summary = summarize_rows(all_rows)
    write_csv(out_dir / "main_comparison_detail.csv", all_rows)
    write_csv(out_dir / "main_comparison_summary.csv", summary)
    write_json(out_dir / "main_comparison_detail.json", all_rows)
    write_json(out_dir / "main_comparison_summary.json", summary)

    latex = make_latex_table(summary)
    with open(out_dir / "main_comparison_latex_table.txt", "w", encoding="utf-8") as f:
        f.write(latex)

    print_summary(summary)
    print(f"Total runtime: {time.time() - global_t0:.2f}s")
    print("Saved:")
    print(f"  {out_dir / 'config.json'}")
    print(f"  {out_dir / 'main_comparison_detail.csv'}")
    print(f"  {out_dir / 'main_comparison_summary.csv'}")
    print(f"  {out_dir / 'main_comparison_latex_table.txt'}")

    print("\nPaper interpretation checklist:")
    print("  1) Proposed_Full_v6 should be best or statistically competitive on cost/delay.")
    print("  2) Proposed_Full_v6 should improve over Proposed_ActorOnly: refinement adds value.")
    print("  3) Proposed_Full_v6 should improve over Greedy_Refine/Random_Refine if enabled: learned actor is necessary.")
    print("  4) Pure_MADDPG must be trained under the same D2.5 environment before being used in the paper table.")
    print("  5) Report battery as diagnostic unless you explicitly add it as a hard feasibility constraint in env.step/report.ok.")


if __name__ == "__main__":
    main()
