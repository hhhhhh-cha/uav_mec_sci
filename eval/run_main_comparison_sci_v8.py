#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SCI-style main comparison for v8 Transformer-scheduling Proposed method.

Default methods:
  1. Proposed_woPG_ActorOnly
  2. Proposed_wPG_ActorOnly
  3. Proposed_wPG_Refine4
  4. Pure_MADDPG
  5. Pure_MADDPG_Refine4
  6. Greedy
  7. Greedy_Refine4
  8. Random

Optional:
  9. Random_Refine4, enabled by --include-random-refine
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Tuple

import numpy as np
import torch

from env.mec_env import MultiUavMecEnv
from env.association import build_access_association
from model.mlp_actor import MLPActor
from model.proposed_obs_builder import build_global_observation, get_observation_dim

from policy.proposed_policy_v8_transformer_sched import build_default_proposed_policy_v8

try:
    from policy.greedy_policy import GreedyPolicy
except Exception:
    GreedyPolicy = None

try:
    from policy.random_policy import generate_random_high_action
except Exception:
    generate_random_high_action = None

from train.train_proposed_full_stage1_converge import get_actor_raw_action_dim

from train.train_proposed_full_stage2_converge_v8_transformer_sched import (
    SafeMobilityPolicyWrapper,
    ModelRefinedPolicyWrapper,
)

from train.train_pure_maddpg_sci_v6 import (
    get_pure_actor_raw_action_dim,
    decode_pure_raw_action,
)

EPS = 1e-8


def parse_seed_list(text: str) -> List[int]:
    out = []
    for x in str(text).split(","):
        x = x.strip()
        if x:
            out.append(int(x))
    if not out:
        raise ValueError("Empty seed list.")
    return out


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
            y = float(x)
            return y if np.isfinite(y) else default
        arr = np.asarray(x)
        if arr.size == 1:
            y = float(arr.reshape(-1)[0])
            return y if np.isfinite(y) else default
        if arr.size > 1:
            y = float(np.nanmean(arr.astype(float)))
            return y if np.isfinite(y) else default
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


def make_env(args: argparse.Namespace, seed: int) -> MultiUavMecEnv:
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


def load_state_dict_strict(module: torch.nn.Module, path: str, name: str, device: str) -> None:
    if module is None:
        raise RuntimeError(f"{name} module is None.")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {name}: {path}")
    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    module.load_state_dict(state)


def set_policy_eval(policy: Any) -> None:
    for name in ["actor_net", "encoder", "fusion_net", "ratio_head", "schedule_head", "actor"]:
        module = getattr(policy, name, None)
        if module is not None and hasattr(module, "eval"):
            module.eval()


def v8_checkpoint_paths(prefix: str, ckpt_dir: str) -> Dict[str, str]:
    base = Path(ckpt_dir)
    return {
        "actor": str(base / f"{prefix}_best_actor.pth"),
        "encoder": str(base / f"{prefix}_best_encoder.pth"),
        "fusion": str(base / f"{prefix}_best_fusion.pth"),
        "ratio_head": str(base / f"{prefix}_best_ratio_head.pth"),
        "schedule_head": str(base / f"{prefix}_best_schedule_head.pth"),
    }


def build_proposed_v8_policy(
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

    policy = build_default_proposed_policy_v8(
        state=state,
        actor_net=actor,
        device=device,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        ff_hidden_dim=args.ff_hidden_dim,
        num_layers=args.num_layers,
    )

    paths = v8_checkpoint_paths(prefix, args.ckpt_dir)
    load_state_dict_strict(policy.actor_net, paths["actor"], "v8 actor", device)
    load_state_dict_strict(policy.encoder, paths["encoder"], "v8 encoder", device)
    load_state_dict_strict(policy.fusion_net, paths["fusion"], "v8 fusion", device)
    load_state_dict_strict(policy.ratio_head, paths["ratio_head"], "v8 ratio_head", device)
    load_state_dict_strict(policy.schedule_head, paths["schedule_head"], "v8 schedule_head", device)

    set_policy_eval(policy)
    return policy


def build_proposed_v8_policy_with_num_layers(
    args: argparse.Namespace,
    prefix: str,
    device: str,
    build_seed: int,
    num_layers: int,
) -> Any:
    """Build a proposed policy with a local num_layers override.

    This is used by the Proposed w/o Transformer ablation:
    num_layers=0 removes all Transformer self-attention/FFN blocks while
    keeping the candidate-token pipeline, fusion network, ratio/scheduling
    heads, analytical solvers, refinement, and evaluation protocol unchanged.
    """
    local_args = copy.copy(args)
    local_args.num_layers = int(num_layers)
    return build_proposed_v8_policy(local_args, prefix, device, build_seed)


class PureMADDPGActorPolicy:
    def __init__(self, args: argparse.Namespace, prefix: str, device: str, build_seed: int):
        self.args = args
        self.device = device

        env = make_env(args, build_seed)
        obs = env.reset(seed=build_seed)
        state = obs["raw_state"]

        obs_dim = get_observation_dim(state)
        action_dim = get_pure_actor_raw_action_dim(state)

        self.actor = MLPActor(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=args.hidden_dim,
        ).to(device)

        ckpt = Path(args.ckpt_dir) / f"{prefix}_best_actor.pth"
        if not ckpt.exists():
            raise FileNotFoundError(f"Missing Pure MADDPG actor checkpoint: {ckpt}")

        state_dict = torch.load(str(ckpt), map_location=device)
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        self.actor.load_state_dict(state_dict)
        self.actor.eval()

    @torch.no_grad()
    def act(
        self,
        state: Dict[str, Any],
        access_assoc: np.ndarray,
        deterministic: bool = True,
        return_aux: bool = False,
        **kwargs: Any,
    ) -> Any:
        obs_vec = build_global_observation(state)
        obs_t = torch.tensor(obs_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
        raw = self.actor(obs_t).squeeze(0).detach().cpu().numpy().astype(np.float32)
        action = decode_pure_raw_action(state, access_assoc, raw)
        if return_aux:
            return action, {}
        return action


class FunctionPolicy:
    def __init__(self, fn: Callable, seed: int = 0):
        self.fn = fn
        self.seed = int(seed)
        self.t = 0

    def act(
        self,
        state: Dict[str, Any],
        access_assoc: np.ndarray,
        deterministic: bool = True,
        return_aux: bool = False,
        **kwargs: Any,
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
        **kwargs: Any,
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
            j = int(self.rng.choice(legal_js))
            sched_beta[k, access_m, j] = 1.0

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
            return self.policy.act(
                state=state,
                access_assoc=access_assoc,
                deterministic=deterministic,
                return_aux=return_aux,
                **kwargs,
            )
        except TypeError:
            pass

        try:
            action = self.policy.act(
                state=state,
                access_assoc=access_assoc,
                deterministic=deterministic,
                **kwargs,
            )
        except TypeError:
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
        return FunctionPolicy(generate_random_high_action, seed=seed)
    return FallbackRandomPolicy(seed=seed)


def build_greedy_policy(seed: int) -> Any:
    if GreedyPolicy is None:
        raise RuntimeError("GreedyPolicy is unavailable.")
    return ReturnAuxCompatiblePolicy(GreedyPolicy(seed=seed))


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


def evaluate_policy_once(env: MultiUavMecEnv, policy: Any, seed: int) -> Dict[str, float]:
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
        "avg_move_violation": ["move_violation", "motion_violation", "movement_violation"],
        "avg_boundary_violation": ["boundary_violation", "out_of_boundary", "out_of_bounds"],
        "avg_collision_violation": ["collision_violation", "collision", "safe_distance_violation", "distance_violation"],
        "avg_battery_violation": ["battery_violation", "energy_violation", "battery_negative"],
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
        "episode_reward",
        "system_cost",
        "avg_delay",
        "avg_energy",
        "avg_deadline_violation",
        "feasible_ratio",
        "avg_ratio_violation",
        "avg_assoc_violation",
        "avg_schedule_violation",
        "avg_candidate_violation",
        "avg_bw_violation",
        "avg_cpu_violation",
        "avg_rate_violation",
        "avg_nan_count",
        "avg_move_violation",
        "avg_boundary_violation",
        "avg_collision_violation",
        "avg_battery_violation",
        "ratio_mean",
        "ratio_std",
        "local_exec_ratio",
        "neighbor_exec_ratio",
        "refine_improvement_per_step",
        "refine_improved_step_ratio",
        "decision_time_per_slot_sec",
    ]

    method_order = [
        "Proposed_wPG_Refine4",
        "Proposed_wPG_ActorOnly",
        "Proposed_woTransformer_ActorOnly",
        "Proposed_woPG_ActorOnly",
        "Pure_MADDPG_Refine4",
        "Pure_MADDPG",
        "Greedy_Refine4",
        "Greedy",
        "Random_Refine4",
        "Random",
    ]
    all_methods = [str(r["method"]) for r in rows]
    ordered_methods = method_order + sorted([m for m in set(all_methods) if m not in method_order])

    summary = []
    for method in ordered_methods:
        vals = [r for r in rows if r["method"] == method]
        if not vals:
            continue
        out: Dict[str, Any] = {"method": method, "n": len(vals)}
        for metric in metrics:
            arr = np.asarray([safe_float(v.get(metric), float("nan")) for v in vals], dtype=np.float64)
            arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                out[f"{metric}_mean"] = float("nan")
                out[f"{metric}_std"] = float("nan")
                out[f"{metric}_ci95"] = float("nan")
            else:
                mean = float(np.mean(arr))
                std = float(np.std(arr, ddof=1)) if arr.size >= 2 else 0.0
                ci95 = float(1.96 * std / math.sqrt(arr.size)) if arr.size >= 2 else 0.0
                out[f"{metric}_mean"] = mean
                out[f"{metric}_std"] = std
                out[f"{metric}_ci95"] = ci95
        summary.append(out)

    ranked = sorted(summary, key=lambda r: float(r.get("system_cost_mean", float("inf"))))
    best_cost = float(ranked[0].get("system_cost_mean", float("nan"))) if ranked else float("nan")
    for row in summary:
        cost = float(row.get("system_cost_mean", float("nan")))
        if np.isfinite(cost) and np.isfinite(best_cost):
            row["cost_gap_to_best_percent"] = float((cost - best_cost) / max(abs(best_cost), EPS) * 100.0)
        else:
            row["cost_gap_to_best_percent"] = float("nan")

    return summary


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = []
    for row in rows:
        for key in row:
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
    lines.append(r"\label{tab:main_comparison_v8}")
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
        lines.append(f"{method} & {cost} & {delay} & {deadline} & {feasible} \\\\")
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def print_summary(summary: List[Dict[str, Any]]) -> None:
    print("\n" + "=" * 140)
    print("SCI-style v8 main comparison summary")
    print("=" * 140)
    print(
        f"{'Method':28s} | {'Cost mean±std':24s} | {'Delay mean±std':22s} | "
        f"{'Deadline':12s} | {'Feasible':10s} | {'Neighbor':10s} | {'ms/slot':10s}"
    )
    print("-" * 140)
    for row in summary:
        print(
            f"{row['method']:28s} | "
            f"{row['system_cost_mean']:10.2f}±{row['system_cost_std']:<10.2f} | "
            f"{row['avg_delay_mean']:9.3f}±{row['avg_delay_std']:<9.3f} | "
            f"{row['avg_deadline_violation_mean']:9.4f} | "
            f"{row['feasible_ratio_mean']:8.4f} | "
            f"{row['neighbor_exec_ratio_mean']:8.4f} | "
            f"{1000.0 * row['decision_time_per_slot_sec_mean']:8.3f}"
        )
    print("=" * 140 + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SCI-style main comparison for v8 Transformer scheduling method.")

    parser.add_argument("--pgboost-prefix", type=str, required=True)
    parser.add_argument("--nopg-prefix", type=str, required=True)
    parser.add_argument("--pure-prefix", type=str, required=True)
    parser.add_argument("--notr-prefix", type=str, default="")
    parser.add_argument("--include-notr", action="store_true",
                        help="Include Proposed w/o Transformer self-attention ablation.")
    parser.add_argument("--include-notr-refine", action="store_true",
                        help="Also evaluate Proposed w/o Transformer with deployment refinement.")
    parser.add_argument("--notr-num-layers", type=int, default=0,
                        help="Number of Transformer blocks for the no-Transformer ablation; keep 0.")
    parser.add_argument("--ckpt-dir", type=str, default="checkpoints")

    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--seeds", type=str, default="42,52,62,72,82,92,102,112,122,132")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--quiet", action="store_true")

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

    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--ff-hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=2)

    parser.add_argument("--model-refine-max-tasks", type=int, default=4)
    parser.add_argument("--model-refine-ratio", type=int, default=1)
    parser.add_argument("--model-refine-sched", type=int, default=1)
    parser.add_argument("--safety-margin", type=float, default=1.0)
    parser.add_argument("--collision-margin", type=float, default=1e-5)

    parser.add_argument("--include-greedy-refine", action="store_true")
    parser.add_argument("--include-random-refine", action="store_true")
    parser.add_argument("--no-pure-refine", action="store_true")
    parser.add_argument("--no-random", action="store_true")
    parser.add_argument("--no-greedy", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = parse_seed_list(args.seeds)
    device = resolve_device(args.device)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 120)
    print("SCI-style main comparison for v8 Transformer scheduling method")
    print("=" * 120)
    print("device          :", device)
    print("eval seeds      :", seeds)
    print("pgboost_prefix  :", args.pgboost_prefix)
    print("nopg_prefix     :", args.nopg_prefix)
    print("pure_prefix     :", args.pure_prefix)
    if args.include_notr:
        print("notr_prefix     :", args.notr_prefix)
        print("notr_num_layers :", args.notr_num_layers)
    print("out             :", out_dir)
    print("=" * 120)

    write_json(out_dir / "config.json", vars(args))

    build_seed = seeds[0]
    set_global_seed(build_seed, device)

    proposed_pgboost = build_proposed_v8_policy(args, args.pgboost_prefix, device, build_seed)
    proposed_nopg = build_proposed_v8_policy(args, args.nopg_prefix, device, build_seed)
    pure_policy = PureMADDPGActorPolicy(args, args.pure_prefix, device, build_seed)

    proposed_notr = None
    if args.include_notr:
        if not args.notr_prefix:
            raise ValueError("--include-notr requires --notr-prefix.")
        proposed_notr = build_proposed_v8_policy_with_num_layers(
            args=args,
            prefix=args.notr_prefix,
            device=device,
            build_seed=build_seed,
            num_layers=args.notr_num_layers,
        )

    all_rows: List[Dict[str, Any]] = []
    t_start = time.time()

    for seed in seeds:
        print(f"\n[Eval seed {seed}]")
        set_global_seed(seed, device)

        refine_tag = f"Refine{int(args.model_refine_max_tasks)}"
        modes: List[Tuple[str, Any, bool]] = [
            ("Proposed_woPG_ActorOnly", proposed_nopg, False),
            ("Proposed_wPG_ActorOnly", proposed_pgboost, False),
            (f"Proposed_wPG_{refine_tag}", proposed_pgboost, True),
            ("Pure_MADDPG", pure_policy, False),
        ]

        if proposed_notr is not None:
            modes.append(("Proposed_woTransformer_ActorOnly", proposed_notr, False))
            if args.include_notr_refine:
                modes.append((f"Proposed_woTransformer_{refine_tag}", proposed_notr, True))

        if not args.no_pure_refine:
            modes.append((f"Pure_MADDPG_{refine_tag}", pure_policy, True))

        if not args.no_greedy:
            modes.append(("Greedy", build_greedy_policy(seed), False))
            if args.include_greedy_refine:
                modes.append((f"Greedy_{refine_tag}", build_greedy_policy(seed), True))

        if not args.no_random:
            modes.append(("Random", build_random_policy(seed), False))
            if args.include_random_refine:
                modes.append((f"Random_{refine_tag}", build_random_policy(seed), True))

        for method_name, base_policy, refine in modes:
            env = make_env(args, seed)
            eval_policy = wrap_policy(env, base_policy, refine=refine, args=args)
            result = evaluate_policy_once(env, eval_policy, seed=seed)

            row: Dict[str, Any] = {
                "method": method_name,
                "seed": seed,
                "refine_enabled": int(refine),
                "pgboost_prefix": args.pgboost_prefix,
                "nopg_prefix": args.nopg_prefix,
                "pure_prefix": args.pure_prefix,
                "notr_prefix": args.notr_prefix,
                "notr_num_layers": int(args.notr_num_layers),
                "deadline_scale": args.deadline_scale,
                "task_local_cpu_min": args.task_local_cpu_min,
                "task_local_cpu_max": args.task_local_cpu_max,
                "episode_length": args.episode_length,
                "model_refine_max_tasks": args.model_refine_max_tasks,
                "model_refine_ratio": int(args.model_refine_ratio),
                "model_refine_sched": int(args.model_refine_sched),
            }
            row.update(result)
            all_rows.append(row)

            if not args.quiet:
                print(
                    f"  {method_name:28s} | "
                    f"cost={result['system_cost']:.3f} | "
                    f"delay={result['avg_delay']:.3f} | "
                    f"feasible={result['feasible_ratio']:.3f} | "
                    f"neighbor={result['neighbor_exec_ratio']:.3f}"
                )

            write_csv(out_dir / "detailed_results_partial.csv", all_rows)

    summary = summarize_rows(all_rows)

    write_csv(out_dir / "detailed_results.csv", all_rows)
    write_csv(out_dir / "summary.csv", summary)
    write_json(out_dir / "summary.json", summary)

    with open(out_dir / "paper_table.tex", "w", encoding="utf-8") as f:
        f.write(make_latex_table(summary))

    print_summary(summary)
    print(f"Finished in {(time.time() - t_start) / 60.0:.2f} min")
    print("Saved:")
    print(" ", out_dir / "detailed_results.csv")
    print(" ", out_dir / "summary.csv")
    print(" ", out_dir / "summary.json")
    print(" ", out_dir / "paper_table.tex")


if __name__ == "__main__":
    main()

# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-

# """
# SCI-style main comparison for v8 Transformer-scheduling Proposed method.

# Default methods:
#   1. Proposed_woPG_ActorOnly
#   2. Proposed_wPG_ActorOnly
#   3. Proposed_wPG_Refine4
#   4. Pure_MADDPG
#   5. Pure_MADDPG_Refine4
#   6. Greedy
#   7. Greedy_Refine4
#   8. Random

# Optional:
#   9. Random_Refine4, enabled by --include-random-refine
# """

# from __future__ import annotations

# import argparse
# import csv
# import json
# import math
# import os
# import random
# import time
# from pathlib import Path
# from typing import Any, Callable, Dict, List, Sequence, Tuple

# import numpy as np
# import torch

# from env.mec_env import MultiUavMecEnv
# from env.association import build_access_association
# from model.mlp_actor import MLPActor
# from model.proposed_obs_builder import build_global_observation, get_observation_dim

# from policy.proposed_policy_v8_transformer_sched import build_default_proposed_policy_v8

# try:
#     from policy.greedy_policy import GreedyPolicy
# except Exception:
#     GreedyPolicy = None

# try:
#     from policy.random_policy import generate_random_high_action
# except Exception:
#     generate_random_high_action = None

# from train.train_proposed_full_stage1_converge import get_actor_raw_action_dim

# from train.train_proposed_full_stage2_converge_v8_transformer_sched import (
#     SafeMobilityPolicyWrapper,
#     ModelRefinedPolicyWrapper,
# )

# from train.train_pure_maddpg_sci_v6 import (
#     get_pure_actor_raw_action_dim,
#     decode_pure_raw_action,
# )

# EPS = 1e-8


# def parse_seed_list(text: str) -> List[int]:
#     out = []
#     for x in str(text).split(","):
#         x = x.strip()
#         if x:
#             out.append(int(x))
#     if not out:
#         raise ValueError("Empty seed list.")
#     return out


# def resolve_device(requested: str) -> str:
#     requested = str(requested).lower()
#     if requested == "cpu":
#         return "cpu"
#     if requested.startswith("cuda"):
#         return requested if torch.cuda.is_available() else "cpu"
#     return "cuda" if torch.cuda.is_available() else "cpu"


# def set_global_seed(seed: int, device: str) -> None:
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     if device.startswith("cuda") and torch.cuda.is_available():
#         torch.cuda.manual_seed_all(seed)


# def safe_float(x: Any, default: float = 0.0) -> float:
#     try:
#         if x is None:
#             return default
#         if isinstance(x, (int, float, np.integer, np.floating)):
#             y = float(x)
#             return y if np.isfinite(y) else default
#         arr = np.asarray(x)
#         if arr.size == 1:
#             y = float(arr.reshape(-1)[0])
#             return y if np.isfinite(y) else default
#         if arr.size > 1:
#             y = float(np.nanmean(arr.astype(float)))
#             return y if np.isfinite(y) else default
#     except Exception:
#         return default
#     return default


# def recursive_find_numeric(info: Any, aliases: Sequence[str]) -> float:
#     if info is None:
#         return 0.0
#     if isinstance(info, dict):
#         for key in aliases:
#             if key in info:
#                 return safe_float(info[key], 0.0)
#         for value in info.values():
#             found = recursive_find_numeric(value, aliases)
#             if found != 0.0:
#                 return found
#     elif isinstance(info, (list, tuple)):
#         for value in info:
#             found = recursive_find_numeric(value, aliases)
#             if found != 0.0:
#                 return found
#     return 0.0


# def make_env(args: argparse.Namespace, seed: int) -> MultiUavMecEnv:
#     return MultiUavMecEnv(
#         M=args.M,
#         K=args.K,
#         episode_length=args.episode_length,
#         area_size=args.area_size,
#         altitude=args.altitude,
#         neighbor_radius=args.neighbor_radius,
#         delta_t=args.delta_t,
#         max_speed=args.max_speed,
#         min_uav_distance=args.min_uav_distance,
#         cpu_mode=args.cpu_mode,
#         prop_rho=args.prop_rho,
#         omega1=args.omega1,
#         omega2=args.omega2,
#         penalty_coeff=args.penalty_coeff,
#         R_min=args.R_min,
#         deadline_scale=args.deadline_scale,
#         task_local_cpu_min=args.task_local_cpu_min,
#         task_local_cpu_max=args.task_local_cpu_max,
#         uav_energy_min=args.uav_energy_min,
#         uav_energy_max=args.uav_energy_max,
#         seed=seed,
#     )


# def load_state_dict_strict(module: torch.nn.Module, path: str, name: str, device: str) -> None:
#     if module is None:
#         raise RuntimeError(f"{name} module is None.")
#     if not os.path.exists(path):
#         raise FileNotFoundError(f"Missing {name}: {path}")
#     state = torch.load(path, map_location=device)
#     if isinstance(state, dict) and "state_dict" in state:
#         state = state["state_dict"]
#     module.load_state_dict(state)


# def set_policy_eval(policy: Any) -> None:
#     for name in ["actor_net", "encoder", "fusion_net", "ratio_head", "schedule_head", "actor"]:
#         module = getattr(policy, name, None)
#         if module is not None and hasattr(module, "eval"):
#             module.eval()


# def v8_checkpoint_paths(prefix: str, ckpt_dir: str) -> Dict[str, str]:
#     base = Path(ckpt_dir)
#     return {
#         "actor": str(base / f"{prefix}_best_actor.pth"),
#         "encoder": str(base / f"{prefix}_best_encoder.pth"),
#         "fusion": str(base / f"{prefix}_best_fusion.pth"),
#         "ratio_head": str(base / f"{prefix}_best_ratio_head.pth"),
#         "schedule_head": str(base / f"{prefix}_best_schedule_head.pth"),
#     }


# def build_proposed_v8_policy(
#     args: argparse.Namespace,
#     prefix: str,
#     device: str,
#     build_seed: int,
# ) -> Any:
#     env = make_env(args, build_seed)
#     obs = env.reset(seed=build_seed)
#     state = obs["raw_state"]

#     obs_dim = get_observation_dim(state)
#     actor_raw_action_dim = get_actor_raw_action_dim(state)

#     actor = MLPActor(
#         obs_dim=obs_dim,
#         action_dim=actor_raw_action_dim,
#         hidden_dim=args.hidden_dim,
#     ).to(device)

#     policy = build_default_proposed_policy_v8(
#         state=state,
#         actor_net=actor,
#         device=device,
#         embed_dim=args.embed_dim,
#         num_heads=args.num_heads,
#         ff_hidden_dim=args.ff_hidden_dim,
#         num_layers=args.num_layers,
#     )

#     paths = v8_checkpoint_paths(prefix, args.ckpt_dir)
#     load_state_dict_strict(policy.actor_net, paths["actor"], "v8 actor", device)
#     load_state_dict_strict(policy.encoder, paths["encoder"], "v8 encoder", device)
#     load_state_dict_strict(policy.fusion_net, paths["fusion"], "v8 fusion", device)
#     load_state_dict_strict(policy.ratio_head, paths["ratio_head"], "v8 ratio_head", device)
#     load_state_dict_strict(policy.schedule_head, paths["schedule_head"], "v8 schedule_head", device)

#     set_policy_eval(policy)
#     return policy


# class PureMADDPGActorPolicy:
#     def __init__(self, args: argparse.Namespace, prefix: str, device: str, build_seed: int):
#         self.args = args
#         self.device = device

#         env = make_env(args, build_seed)
#         obs = env.reset(seed=build_seed)
#         state = obs["raw_state"]

#         obs_dim = get_observation_dim(state)
#         action_dim = get_pure_actor_raw_action_dim(state)

#         self.actor = MLPActor(
#             obs_dim=obs_dim,
#             action_dim=action_dim,
#             hidden_dim=args.hidden_dim,
#         ).to(device)

#         ckpt = Path(args.ckpt_dir) / f"{prefix}_best_actor.pth"
#         if not ckpt.exists():
#             raise FileNotFoundError(f"Missing Pure MADDPG actor checkpoint: {ckpt}")

#         state_dict = torch.load(str(ckpt), map_location=device)
#         if isinstance(state_dict, dict) and "state_dict" in state_dict:
#             state_dict = state_dict["state_dict"]
#         self.actor.load_state_dict(state_dict)
#         self.actor.eval()

#     @torch.no_grad()
#     def act(
#         self,
#         state: Dict[str, Any],
#         access_assoc: np.ndarray,
#         deterministic: bool = True,
#         return_aux: bool = False,
#         **kwargs: Any,
#     ) -> Any:
#         obs_vec = build_global_observation(state)
#         obs_t = torch.tensor(obs_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
#         raw = self.actor(obs_t).squeeze(0).detach().cpu().numpy().astype(np.float32)
#         action = decode_pure_raw_action(state, access_assoc, raw)
#         if return_aux:
#             return action, {}
#         return action


# class FunctionPolicy:
#     def __init__(self, fn: Callable, seed: int = 0):
#         self.fn = fn
#         self.seed = int(seed)
#         self.t = 0

#     def act(
#         self,
#         state: Dict[str, Any],
#         access_assoc: np.ndarray,
#         deterministic: bool = True,
#         return_aux: bool = False,
#         **kwargs: Any,
#     ) -> Any:
#         try:
#             action = self.fn(state, access_assoc, seed=self.seed + self.t)
#         except TypeError:
#             action = self.fn(state, access_assoc)
#         self.t += 1
#         if return_aux:
#             return action, {}
#         return action


# class FallbackRandomPolicy:
#     def __init__(self, seed: int = 0):
#         self.rng = np.random.default_rng(seed)

#     def act(
#         self,
#         state: Dict[str, Any],
#         access_assoc: np.ndarray,
#         deterministic: bool = True,
#         return_aux: bool = False,
#         **kwargs: Any,
#     ) -> Any:
#         M = int(state["M"])
#         K = int(state["K"])
#         neighbors = state["neighbors"]

#         move_dist = np.zeros(M, dtype=np.float32)
#         move_angle = np.zeros(M, dtype=np.float32)
#         offload_ratio = self.rng.uniform(0.0, 1.0, size=K).astype(np.float32)

#         sched_beta = np.zeros((K, M, M), dtype=np.float32)
#         for k in range(K):
#             access_m = int(np.argmax(access_assoc[:, k]))
#             legal_js = [access_m] + list(neighbors[access_m])
#             j = int(self.rng.choice(legal_js))
#             sched_beta[k, access_m, j] = 1.0

#         action = {
#             "move_dist": move_dist,
#             "move_angle": move_angle,
#             "offload_ratio": offload_ratio,
#             "sched_beta": sched_beta,
#         }
#         if return_aux:
#             return action, {}
#         return action


# class ReturnAuxCompatiblePolicy:
#     def __init__(self, policy: Any):
#         self.policy = policy

#     def act(
#         self,
#         state: Dict[str, Any],
#         access_assoc: np.ndarray,
#         deterministic: bool = True,
#         return_aux: bool = False,
#         **kwargs: Any,
#     ) -> Any:
#         try:
#             return self.policy.act(
#                 state=state,
#                 access_assoc=access_assoc,
#                 deterministic=deterministic,
#                 return_aux=return_aux,
#                 **kwargs,
#             )
#         except TypeError:
#             pass

#         try:
#             action = self.policy.act(
#                 state=state,
#                 access_assoc=access_assoc,
#                 deterministic=deterministic,
#                 **kwargs,
#             )
#         except TypeError:
#             action = self.policy.act(
#                 state=state,
#                 access_assoc=access_assoc,
#                 **kwargs,
#             )

#         if return_aux:
#             return action, {}
#         return action


# def build_random_policy(seed: int) -> Any:
#     if generate_random_high_action is not None:
#         return FunctionPolicy(generate_random_high_action, seed=seed)
#     return FallbackRandomPolicy(seed=seed)


# def build_greedy_policy(seed: int) -> Any:
#     if GreedyPolicy is None:
#         raise RuntimeError("GreedyPolicy is unavailable.")
#     return ReturnAuxCompatiblePolicy(GreedyPolicy(seed=seed))


# def wrap_policy(env: MultiUavMecEnv, base_policy: Any, refine: bool, args: argparse.Namespace) -> Any:
#     if refine:
#         return ModelRefinedPolicyWrapper(
#             env=env,
#             policy=base_policy,
#             safety_margin=args.safety_margin,
#             collision_margin=args.collision_margin,
#             enable_refine=True,
#             max_tasks=args.model_refine_max_tasks,
#             ratio_search=bool(args.model_refine_ratio),
#             schedule_search=bool(args.model_refine_sched),
#         )

#     return SafeMobilityPolicyWrapper(
#         policy=base_policy,
#         safety_margin=args.safety_margin,
#         collision_margin=args.collision_margin,
#     )


# def evaluate_policy_once(env: MultiUavMecEnv, policy: Any, seed: int) -> Dict[str, float]:
#     obs = env.reset(seed=seed)
#     done = False
#     step_count = 0

#     total_reward = 0.0
#     total_delay = 0.0
#     total_energy = 0.0
#     feasible_count = 0.0

#     sums = {
#         "avg_deadline_violation": 0.0,
#         "avg_ratio_violation": 0.0,
#         "avg_assoc_violation": 0.0,
#         "avg_schedule_violation": 0.0,
#         "avg_candidate_violation": 0.0,
#         "avg_bw_violation": 0.0,
#         "avg_cpu_violation": 0.0,
#         "avg_rate_violation": 0.0,
#         "avg_nan_count": 0.0,
#         "avg_move_violation": 0.0,
#         "avg_boundary_violation": 0.0,
#         "avg_collision_violation": 0.0,
#         "avg_battery_violation": 0.0,
#     }

#     alias_map = {
#         "avg_move_violation": ["move_violation", "motion_violation", "movement_violation"],
#         "avg_boundary_violation": ["boundary_violation", "out_of_boundary", "out_of_bounds"],
#         "avg_collision_violation": ["collision_violation", "collision", "safe_distance_violation", "distance_violation"],
#         "avg_battery_violation": ["battery_violation", "energy_violation", "battery_negative"],
#     }

#     ratio_values: List[float] = []
#     local_exec_count = 0
#     neighbor_exec_count = 0
#     sched_count = 0

#     refine_improvement_sum = 0.0
#     refine_improvement_count = 0
#     refine_improved_steps = 0

#     runtime_action_sec = 0.0

#     while not done:
#         state = obs["raw_state"]
#         access_assoc = build_access_association(state)
#         M = int(state["M"])
#         K = int(state["K"])

#         t0 = time.perf_counter()
#         action = policy.act(state=state, access_assoc=access_assoc, deterministic=True)
#         runtime_action_sec += time.perf_counter() - t0

#         if "_model_refine_improvement" in action:
#             imp = safe_float(action.get("_model_refine_improvement"), 0.0)
#             refine_improvement_sum += imp
#             refine_improvement_count += 1
#             if imp > 1e-9:
#                 refine_improved_steps += 1

#         if "offload_ratio" in action:
#             ratio_arr = np.asarray(action["offload_ratio"], dtype=np.float32).reshape(-1)
#             ratio_values.extend([float(x) for x in ratio_arr])

#         if "sched_beta" in action:
#             sched_beta = np.asarray(action["sched_beta"], dtype=np.float32).reshape(K, M, M)
#             for k in range(K):
#                 access_m = int(np.argmax(access_assoc[:, k]))
#                 exec_j = int(np.argmax(sched_beta[k, access_m, :]))
#                 sched_count += 1
#                 if exec_j == access_m:
#                     local_exec_count += 1
#                 else:
#                     neighbor_exec_count += 1

#         step_action = {
#             "move_dist": action["move_dist"],
#             "move_angle": action["move_angle"],
#             "offload_ratio": action["offload_ratio"],
#             "sched_beta": action["sched_beta"],
#         }

#         obs, reward, done, info = env.step(step_action)
#         report = info.get("report", {})
#         metrics = info.get("metrics", {})

#         total_reward += float(reward)
#         total_delay += safe_float(metrics.get("delay_sys"), 0.0)
#         total_energy += safe_float(metrics.get("energy_sys"), 0.0)
#         feasible_count += 1.0 if bool(report.get("ok", False)) else 0.0

#         sums["avg_deadline_violation"] += safe_float(report.get("deadline_violation"), 0.0)
#         sums["avg_ratio_violation"] += safe_float(report.get("ratio_violation"), 0.0)
#         sums["avg_assoc_violation"] += safe_float(report.get("assoc_violation"), 0.0)
#         sums["avg_schedule_violation"] += safe_float(report.get("schedule_violation"), 0.0)
#         sums["avg_candidate_violation"] += safe_float(report.get("candidate_violation"), 0.0)
#         sums["avg_bw_violation"] += safe_float(report.get("bw_violation"), 0.0)
#         sums["avg_cpu_violation"] += safe_float(report.get("cpu_violation"), 0.0)
#         sums["avg_rate_violation"] += safe_float(report.get("rate_violation"), 0.0)
#         sums["avg_nan_count"] += safe_float(report.get("nan_count"), 0.0)

#         for key, aliases in alias_map.items():
#             sums[key] += recursive_find_numeric(info, aliases)

#         step_count += 1

#     denom = max(step_count, 1)
#     ratio_np = np.asarray(ratio_values, dtype=np.float32)
#     sched_denom = max(sched_count, 1)

#     out = {
#         "episode_reward": float(total_reward),
#         "system_cost": float(-total_reward),
#         "avg_delay": float(total_delay / denom),
#         "avg_energy": float(total_energy / denom),
#         "feasible_ratio": float(feasible_count / denom),
#         "num_steps": int(step_count),
#         "ratio_mean": float(np.mean(ratio_np)) if ratio_np.size else float("nan"),
#         "ratio_std": float(np.std(ratio_np)) if ratio_np.size else float("nan"),
#         "ratio_min": float(np.min(ratio_np)) if ratio_np.size else float("nan"),
#         "ratio_max": float(np.max(ratio_np)) if ratio_np.size else float("nan"),
#         "local_exec_ratio": float(local_exec_count / sched_denom),
#         "neighbor_exec_ratio": float(neighbor_exec_count / sched_denom),
#         "refine_improvement_per_step": float(refine_improvement_sum / max(refine_improvement_count, 1)),
#         "refine_improved_step_ratio": float(refine_improved_steps / max(refine_improvement_count, 1)),
#         "decision_time_per_slot_sec": float(runtime_action_sec / denom),
#     }

#     for key, value in sums.items():
#         out[key] = float(value / denom)

#     return out


# def summarize_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
#     metrics = [
#         "episode_reward",
#         "system_cost",
#         "avg_delay",
#         "avg_energy",
#         "avg_deadline_violation",
#         "feasible_ratio",
#         "avg_ratio_violation",
#         "avg_assoc_violation",
#         "avg_schedule_violation",
#         "avg_candidate_violation",
#         "avg_bw_violation",
#         "avg_cpu_violation",
#         "avg_rate_violation",
#         "avg_nan_count",
#         "avg_move_violation",
#         "avg_boundary_violation",
#         "avg_collision_violation",
#         "avg_battery_violation",
#         "ratio_mean",
#         "ratio_std",
#         "local_exec_ratio",
#         "neighbor_exec_ratio",
#         "refine_improvement_per_step",
#         "refine_improved_step_ratio",
#         "decision_time_per_slot_sec",
#     ]

#     method_order = [
#         "Proposed_wPG_Refine4",
#         "Proposed_wPG_ActorOnly",
#         "Proposed_woPG_ActorOnly",
#         "Pure_MADDPG_Refine4",
#         "Pure_MADDPG",
#         "Greedy_Refine4",
#         "Greedy",
#         "Random_Refine4",
#         "Random",
#     ]

#     summary = []
#     for method in method_order:
#         vals = [r for r in rows if r["method"] == method]
#         if not vals:
#             continue
#         out: Dict[str, Any] = {"method": method, "n": len(vals)}
#         for metric in metrics:
#             arr = np.asarray([safe_float(v.get(metric), float("nan")) for v in vals], dtype=np.float64)
#             arr = arr[np.isfinite(arr)]
#             if arr.size == 0:
#                 out[f"{metric}_mean"] = float("nan")
#                 out[f"{metric}_std"] = float("nan")
#                 out[f"{metric}_ci95"] = float("nan")
#             else:
#                 mean = float(np.mean(arr))
#                 std = float(np.std(arr, ddof=1)) if arr.size >= 2 else 0.0
#                 ci95 = float(1.96 * std / math.sqrt(arr.size)) if arr.size >= 2 else 0.0
#                 out[f"{metric}_mean"] = mean
#                 out[f"{metric}_std"] = std
#                 out[f"{metric}_ci95"] = ci95
#         summary.append(out)

#     ranked = sorted(summary, key=lambda r: float(r.get("system_cost_mean", float("inf"))))
#     best_cost = float(ranked[0].get("system_cost_mean", float("nan"))) if ranked else float("nan")
#     for row in summary:
#         cost = float(row.get("system_cost_mean", float("nan")))
#         if np.isfinite(cost) and np.isfinite(best_cost):
#             row["cost_gap_to_best_percent"] = float((cost - best_cost) / max(abs(best_cost), EPS) * 100.0)
#         else:
#             row["cost_gap_to_best_percent"] = float("nan")

#     return summary


# def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
#     path.parent.mkdir(parents=True, exist_ok=True)
#     if not rows:
#         return
#     fieldnames = []
#     for row in rows:
#         for key in row:
#             if key not in fieldnames:
#                 fieldnames.append(key)
#     with open(path, "w", newline="", encoding="utf-8") as f:
#         writer = csv.DictWriter(f, fieldnames=fieldnames)
#         writer.writeheader()
#         for row in rows:
#             writer.writerow(row)


# def write_json(path: Path, obj: Any) -> None:
#     path.parent.mkdir(parents=True, exist_ok=True)
#     with open(path, "w", encoding="utf-8") as f:
#         json.dump(obj, f, indent=2, ensure_ascii=False)


# def make_latex_table(summary: List[Dict[str, Any]]) -> str:
#     lines = []
#     lines.append(r"\begin{table}[t]")
#     lines.append(r"\centering")
#     lines.append(r"\caption{Main comparison under the post-disaster UAV-MEC scenario.}")
#     lines.append(r"\label{tab:main_comparison_v8}")
#     lines.append(r"\begin{tabular}{lcccc}")
#     lines.append(r"\hline")
#     lines.append(r"Method & System cost $\downarrow$ & Delay $\downarrow$ & Deadline vio. $\downarrow$ & Feasible ratio $\uparrow$ \\")
#     lines.append(r"\hline")
#     for row in summary:
#         method = str(row["method"]).replace("_", r"\_")
#         cost = f"{row['system_cost_mean']:.2f}$\\pm${row['system_cost_std']:.2f}"
#         delay = f"{row['avg_delay_mean']:.2f}$\\pm${row['avg_delay_std']:.2f}"
#         deadline = f"{row['avg_deadline_violation_mean']:.3f}$\\pm${row['avg_deadline_violation_std']:.3f}"
#         feasible = f"{row['feasible_ratio_mean']:.3f}$\\pm${row['feasible_ratio_std']:.3f}"
#         lines.append(f"{method} & {cost} & {delay} & {deadline} & {feasible} \\\\")
#     lines.append(r"\hline")
#     lines.append(r"\end{tabular}")
#     lines.append(r"\end{table}")
#     return "\n".join(lines) + "\n"


# def print_summary(summary: List[Dict[str, Any]]) -> None:
#     print("\n" + "=" * 140)
#     print("SCI-style v8 main comparison summary")
#     print("=" * 140)
#     print(
#         f"{'Method':28s} | {'Cost mean±std':24s} | {'Delay mean±std':22s} | "
#         f"{'Deadline':12s} | {'Feasible':10s} | {'Neighbor':10s} | {'ms/slot':10s}"
#     )
#     print("-" * 140)
#     for row in summary:
#         print(
#             f"{row['method']:28s} | "
#             f"{row['system_cost_mean']:10.2f}±{row['system_cost_std']:<10.2f} | "
#             f"{row['avg_delay_mean']:9.3f}±{row['avg_delay_std']:<9.3f} | "
#             f"{row['avg_deadline_violation_mean']:9.4f} | "
#             f"{row['feasible_ratio_mean']:8.4f} | "
#             f"{row['neighbor_exec_ratio_mean']:8.4f} | "
#             f"{1000.0 * row['decision_time_per_slot_sec_mean']:8.3f}"
#         )
#     print("=" * 140 + "\n")


# def parse_args() -> argparse.Namespace:
#     parser = argparse.ArgumentParser(description="SCI-style main comparison for v8 Transformer scheduling method.")

#     parser.add_argument("--pgboost-prefix", type=str, required=True)
#     parser.add_argument("--nopg-prefix", type=str, required=True)
#     parser.add_argument("--pure-prefix", type=str, required=True)
#     parser.add_argument("--ckpt-dir", type=str, default="checkpoints")

#     parser.add_argument("--out", type=str, required=True)
#     parser.add_argument("--seeds", type=str, default="42,52,62,72,82,92,102,112,122,132")
#     parser.add_argument("--device", type=str, default="auto")
#     parser.add_argument("--quiet", action="store_true")

#     parser.add_argument("--M", type=int, default=3)
#     parser.add_argument("--K", type=int, default=16)
#     parser.add_argument("--episode-length", type=int, default=20)
#     parser.add_argument("--area-size", type=float, default=100.0)
#     parser.add_argument("--altitude", type=float, default=50.0)
#     parser.add_argument("--neighbor-radius", type=float, default=50.0)
#     parser.add_argument("--delta-t", type=float, default=1.0)
#     parser.add_argument("--max-speed", type=float, default=15.0)
#     parser.add_argument("--min-uav-distance", type=float, default=3.0)
#     parser.add_argument("--cpu-mode", type=str, default="kkt")
#     parser.add_argument("--prop-rho", type=float, default=0.45)
#     parser.add_argument("--omega1", type=float, default=50.0)
#     parser.add_argument("--omega2", type=float, default=1.0)
#     parser.add_argument("--penalty-coeff", type=float, default=50.0)
#     parser.add_argument("--R-min", dest="R_min", type=float, default=0.05)
#     parser.add_argument("--deadline-scale", type=float, default=2.5)
#     parser.add_argument("--task-local-cpu-min", type=float, default=2000.0)
#     parser.add_argument("--task-local-cpu-max", type=float, default=5000.0)
#     parser.add_argument("--uav-energy-min", type=float, default=2600.0)
#     parser.add_argument("--uav-energy-max", type=float, default=3800.0)

#     parser.add_argument("--hidden-dim", type=int, default=256)
#     parser.add_argument("--embed-dim", type=int, default=128)
#     parser.add_argument("--num-heads", type=int, default=4)
#     parser.add_argument("--ff-hidden-dim", type=int, default=256)
#     parser.add_argument("--num-layers", type=int, default=2)

#     parser.add_argument("--model-refine-max-tasks", type=int, default=4)
#     parser.add_argument("--model-refine-ratio", type=int, default=1)
#     parser.add_argument("--model-refine-sched", type=int, default=1)
#     parser.add_argument("--safety-margin", type=float, default=1.0)
#     parser.add_argument("--collision-margin", type=float, default=1e-5)

#     parser.add_argument("--include-greedy-refine", action="store_true")
#     parser.add_argument("--include-random-refine", action="store_true")
#     parser.add_argument("--no-pure-refine", action="store_true")
#     parser.add_argument("--no-random", action="store_true")
#     parser.add_argument("--no-greedy", action="store_true")

#     return parser.parse_args()


# def main() -> None:
#     args = parse_args()
#     seeds = parse_seed_list(args.seeds)
#     device = resolve_device(args.device)

#     out_dir = Path(args.out)
#     out_dir.mkdir(parents=True, exist_ok=True)

#     print("=" * 120)
#     print("SCI-style main comparison for v8 Transformer scheduling method")
#     print("=" * 120)
#     print("device          :", device)
#     print("eval seeds      :", seeds)
#     print("pgboost_prefix  :", args.pgboost_prefix)
#     print("nopg_prefix     :", args.nopg_prefix)
#     print("pure_prefix     :", args.pure_prefix)
#     print("out             :", out_dir)
#     print("=" * 120)

#     write_json(out_dir / "config.json", vars(args))

#     build_seed = seeds[0]
#     set_global_seed(build_seed, device)

#     proposed_pgboost = build_proposed_v8_policy(args, args.pgboost_prefix, device, build_seed)
#     proposed_nopg = build_proposed_v8_policy(args, args.nopg_prefix, device, build_seed)
#     pure_policy = PureMADDPGActorPolicy(args, args.pure_prefix, device, build_seed)

#     all_rows: List[Dict[str, Any]] = []
#     t_start = time.time()

#     for seed in seeds:
#         print(f"\n[Eval seed {seed}]")
#         set_global_seed(seed, device)

#         modes: List[Tuple[str, Any, bool]] = [
#             ("Proposed_woPG_ActorOnly", proposed_nopg, False),
#             ("Proposed_wPG_ActorOnly", proposed_pgboost, False),
#             ("Proposed_wPG_Refine4", proposed_pgboost, True),
#             ("Pure_MADDPG", pure_policy, False),
#         ]

#         if not args.no_pure_refine:
#             modes.append(("Pure_MADDPG_Refine4", pure_policy, True))

#         if not args.no_greedy:
#             modes.append(("Greedy", build_greedy_policy(seed), False))
#             if args.include_greedy_refine:
#                 modes.append(("Greedy_Refine4", build_greedy_policy(seed), True))

#         if not args.no_random:
#             modes.append(("Random", build_random_policy(seed), False))
#             if args.include_random_refine:
#                 modes.append(("Random_Refine4", build_random_policy(seed), True))

#         for method_name, base_policy, refine in modes:
#             env = make_env(args, seed)
#             eval_policy = wrap_policy(env, base_policy, refine=refine, args=args)
#             result = evaluate_policy_once(env, eval_policy, seed=seed)

#             row: Dict[str, Any] = {
#                 "method": method_name,
#                 "seed": seed,
#                 "refine_enabled": int(refine),
#                 "pgboost_prefix": args.pgboost_prefix,
#                 "nopg_prefix": args.nopg_prefix,
#                 "pure_prefix": args.pure_prefix,
#                 "deadline_scale": args.deadline_scale,
#                 "task_local_cpu_min": args.task_local_cpu_min,
#                 "task_local_cpu_max": args.task_local_cpu_max,
#                 "episode_length": args.episode_length,
#                 "model_refine_max_tasks": args.model_refine_max_tasks,
#                 "model_refine_ratio": int(args.model_refine_ratio),
#                 "model_refine_sched": int(args.model_refine_sched),
#             }
#             row.update(result)
#             all_rows.append(row)

#             if not args.quiet:
#                 print(
#                     f"  {method_name:28s} | "
#                     f"cost={result['system_cost']:.3f} | "
#                     f"delay={result['avg_delay']:.3f} | "
#                     f"feasible={result['feasible_ratio']:.3f} | "
#                     f"neighbor={result['neighbor_exec_ratio']:.3f}"
#                 )

#             write_csv(out_dir / "detailed_results_partial.csv", all_rows)

#     summary = summarize_rows(all_rows)

#     write_csv(out_dir / "detailed_results.csv", all_rows)
#     write_csv(out_dir / "summary.csv", summary)
#     write_json(out_dir / "summary.json", summary)

#     with open(out_dir / "paper_table.tex", "w", encoding="utf-8") as f:
#         f.write(make_latex_table(summary))

#     print_summary(summary)
#     print(f"Finished in {(time.time() - t_start) / 60.0:.2f} min")
#     print("Saved:")
#     print(" ", out_dir / "detailed_results.csv")
#     print(" ", out_dir / "summary.csv")
#     print(" ", out_dir / "summary.json")
#     print(" ", out_dir / "paper_table.tex")


# if __name__ == "__main__":
#     main()
