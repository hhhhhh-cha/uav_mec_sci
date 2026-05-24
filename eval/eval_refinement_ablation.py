#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate whether the final method is still DRL-driven or dominated by the
model-based one-step refinement.

This script runs the key ablation modes needed for the paper:

1) Stage-2 Actor-only
   Transformer-MADDPG actor + safe mobility projection + analytical BW/CPU solver
   with MODEL_REFINE disabled.

2) Stage-2 Actor + model refinement
   Transformer-MADDPG actor + safe mobility projection + one-step model-based
   high-level action refinement + analytical BW/CPU solver.

3) Non-learned policy + model refinement
   Greedy/Random high-level action + the same safe projection + the same
   model refinement + the same analytical BW/CPU solver.

Optional:
4) Stage-1 Actor-only and Stage-1 Actor + refinement, to check whether Stage-2
   actor learned anything beyond the warm-start policy.

Expected conclusion format:
- If Stage-2 Actor-only improves over Stage-1 Actor-only, DRL training is useful.
- If Stage-2 Actor + refinement improves over Stage-2 Actor-only, refinement is useful.
- If Stage-2 Actor + refinement improves over Greedy/Random + refinement,
  the learned actor is still necessary and refinement is not a standalone solver.

Place this file in the project root, e.g. ~/projects/uav_mec_sci/eval_refinement_ablation.py
and run with python3 eval_refinement_ablation.py ...
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from env.mec_env import MultiUavMecEnv
from env.association import build_access_association
from model.mlp_actor import MLPActor
from model.proposed_obs_builder import get_observation_dim
from policy.proposed_policy import build_default_proposed_policy

# Reuse the exact wrappers and refinement function from the v6 training file.
# This guarantees the evaluation uses the same safety projection and model-based
# one-step refinement as the final proposed method.
try:
    from train.train_proposed_full_stage2_converge_v6_model_refine import (
        SafeMobilityPolicyWrapper,
        ModelRefinedPolicyWrapper,
        get_actor_raw_action_dim,
    )
except Exception as e:
    raise ImportError(
        "Cannot import v6 training utilities. Make sure "
        "train/train_proposed_full_stage2_converge_v6_model_refine.py exists. "
        f"Original error: {e}"
    )

# Strong heuristic and random baselines. If the user's repo uses slightly
# different exports, we fall back to a small local random policy below.
try:
    from policy.greedy_policy import GreedyPolicy
except Exception:
    GreedyPolicy = None

try:
    from policy.random_policy import generate_random_high_action
except Exception:
    generate_random_high_action = None

try:
    from policy.proposed_placeholder_policy import generate_proposed_placeholder_action
except Exception:
    generate_proposed_placeholder_action = None


EPS = 1e-8


# -----------------------------------------------------------------------------
# Runtime helpers
# -----------------------------------------------------------------------------
def parse_seed_list(text: str) -> List[int]:
    seeds: List[int] = []
    for item in str(text).split(','):
        item = item.strip()
        if not item:
            continue
        seeds.append(int(item))
    if not seeds:
        raise ValueError("At least one seed must be provided.")
    return seeds


def resolve_device(requested: str) -> str:
    requested = str(requested).lower()
    if requested == "cpu":
        return "cpu"
    if requested.startswith("cuda"):
        return requested if torch.cuda.is_available() else "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def set_global_seed(seed: int, device: str) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_env(args: argparse.Namespace, seed: int) -> MultiUavMecEnv:
    return MultiUavMecEnv(
        M=args.M,
        K=args.K,
        episode_length=args.episode_length,
        cpu_mode=args.cpu_mode,
        omega1=args.omega1,
        omega2=args.omega2,
        deadline_scale=args.deadline_scale,
        task_local_cpu_min=args.task_local_cpu_min,
        task_local_cpu_max=args.task_local_cpu_max,
        uav_energy_min=args.uav_energy_min,
        uav_energy_max=args.uav_energy_max,
        seed=seed,
    )


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
        raise RuntimeError(f"Module {name} is None; cannot load {path}.")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {name} checkpoint: {path}")
    state = torch.load(path, map_location=device)
    module.load_state_dict(state)


def set_policy_eval(policy: Any) -> None:
    for name in ["actor_net", "encoder", "fusion_net", "ratio_head"]:
        module = getattr(policy, name, None)
        if module is not None:
            module.eval()


# -----------------------------------------------------------------------------
# Policy builders
# -----------------------------------------------------------------------------
def build_learned_policy(
    env: MultiUavMecEnv,
    prefix: str,
    device: str,
    ckpt_dir: str,
    seed: int,
) -> Any:
    """Build ProposedPolicy and load actor/encoder/fusion/ratio-head weights."""
    obs = env.reset(seed=seed)
    state = obs["raw_state"]
    obs_dim = get_observation_dim(state)
    actor_raw_action_dim = get_actor_raw_action_dim(state)

    actor = MLPActor(
        obs_dim=obs_dim,
        action_dim=actor_raw_action_dim,
        hidden_dim=256,
    ).to(device)

    policy = build_default_proposed_policy(
        state=state,
        actor_net=actor,
        device=device,
        embed_dim=128,
        num_heads=4,
        ff_hidden_dim=256,
        num_layers=2,
    )

    paths = checkpoint_paths(prefix, ckpt_dir)
    load_state_dict_strict(policy.actor_net, paths["actor"], "actor", device)
    load_state_dict_strict(policy.encoder, paths["encoder"], "encoder", device)
    load_state_dict_strict(policy.fusion_net, paths["fusion_net"], "fusion", device)
    load_state_dict_strict(policy.ratio_head, paths["ratio_head"], "ratio_head", device)
    set_policy_eval(policy)
    return policy


class FunctionPolicy:
    """Wrap a high-level action generator into the policy.act(...) interface."""

    def __init__(self, fn: Callable, seed: Optional[int] = None, name: str = "function_policy"):
        self.fn = fn
        self.seed = seed
        self.name = name
        self.t = 0

    def act(self, state: Dict[str, Any], access_assoc: np.ndarray, deterministic: bool = True, return_aux: bool = False):
        # Use a changing seed for random policy while keeping deterministic reproducibility.
        if self.seed is None:
            action = self.fn(state, access_assoc)
        else:
            try:
                action = self.fn(state, access_assoc, seed=self.seed + self.t)
            except TypeError:
                action = self.fn(state, access_assoc)
        self.t += 1
        if return_aux:
            return action, {}
        return action

class ActInterfaceAdapter:
    """
    Adapt policies whose act(...) signature does not support deterministic
    or return_aux. This is mainly for GreedyPolicy compatibility with
    SafeMobilityPolicyWrapper / ModelRefinedPolicyWrapper.
    """

    def __init__(self, policy: Any, name: str = "adapted_policy"):
        self.policy = policy
        self.name = name

    def act(
        self,
        state: Dict[str, Any],
        access_assoc: np.ndarray,
        deterministic: bool = True,
        return_aux: bool = False,
    ):
        try:
            action = self.policy.act(
                state=state,
                access_assoc=access_assoc,
                deterministic=deterministic,
                return_aux=return_aux,
            )
            return action
        except TypeError:
            pass

        try:
            action = self.policy.act(
                state=state,
                access_assoc=access_assoc,
                deterministic=deterministic,
            )
        except TypeError:
            try:
                action = self.policy.act(
                    state=state,
                    access_assoc=access_assoc,
                )
            except TypeError:
                action = self.policy.act(state, access_assoc)

        if return_aux:
            return action, {}
        return action

class FallbackRandomPolicy:
    """Legal random high-level policy if policy.random_policy is unavailable."""

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def act(self, state: Dict[str, Any], access_assoc: np.ndarray, deterministic: bool = True, return_aux: bool = False):
        M, K = access_assoc.shape
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


def build_greedy_or_placeholder_policy(seed: int) -> Any:
    if GreedyPolicy is not None:
        return ActInterfaceAdapter(GreedyPolicy(seed=seed), name="greedy")
    if generate_proposed_placeholder_action is not None:
        return FunctionPolicy(generate_proposed_placeholder_action, seed=None, name="placeholder")
    raise RuntimeError("Neither GreedyPolicy nor generate_proposed_placeholder_action is available.")

def build_random_policy(seed: int) -> Any:
    if generate_random_high_action is not None:
        return FunctionPolicy(generate_random_high_action, seed=seed, name="random")
    return FallbackRandomPolicy(seed=seed)


def wrap_policy_for_mode(
    env: MultiUavMecEnv,
    base_policy: Any,
    refine: bool,
    args: argparse.Namespace,
) -> Any:
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
def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        arr = np.asarray(x)
        if arr.size == 1:
            val = float(arr.reshape(-1)[0])
            if np.isfinite(val):
                return val
        if isinstance(x, (int, float, np.integer, np.floating)):
            val = float(x)
            return val if np.isfinite(val) else default
    except Exception:
        return default
    return default


def _recursive_find_numeric(info: Any, aliases: Sequence[str]) -> float:
    if info is None:
        return 0.0
    if isinstance(info, dict):
        for key in aliases:
            if key in info:
                return _safe_float(info[key], 0.0)
        for value in info.values():
            found = _recursive_find_numeric(value, aliases)
            if found != 0.0:
                return found
    elif isinstance(info, (list, tuple)):
        for value in info:
            found = _recursive_find_numeric(value, aliases)
            if found != 0.0:
                return found
    return 0.0


def evaluate_policy_once(
    env: MultiUavMecEnv,
    policy: Any,
    seed: int,
) -> Dict[str, float]:
    obs = env.reset(seed=seed)
    done = False

    total_reward = 0.0
    total_delay = 0.0
    total_energy = 0.0
    total_deadline_violation = 0.0
    feasible_count = 0
    step_count = 0

    violation_sums = {
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

    ratio_values: List[float] = []
    local_exec_count = 0
    neighbor_exec_count = 0
    sched_count = 0

    refine_improve_sum = 0.0
    refine_improve_count = 0
    refine_improved_steps = 0

    while not done:
        state = obs["raw_state"]
        access_assoc = build_access_association(state)
        M = int(state["M"])
        K = int(state["K"])

        action = policy.act(
            state=state,
            access_assoc=access_assoc,
            deterministic=True,
            return_aux=False,
        )

        if "_model_refine_improvement" in action:
            imp = _safe_float(action.get("_model_refine_improvement"), 0.0)
            refine_improve_sum += imp
            refine_improve_count += 1
            if imp > 1e-8:
                refine_improved_steps += 1

        if "offload_ratio" in action:
            ratio_arr = np.asarray(action["offload_ratio"], dtype=np.float32).reshape(-1)
            ratio_values.extend(ratio_arr.tolist())

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

        obs, reward, done, info = env.step(action)
        report = info.get("report", {}) if isinstance(info, dict) else {}
        metrics = info.get("metrics", {}) if isinstance(info, dict) else {}

        total_reward += float(reward)
        total_delay += _safe_float(metrics.get("delay_sys"), 0.0)
        total_energy += _safe_float(metrics.get("energy_sys"), 0.0)
        total_deadline_violation += _safe_float(report.get("deadline_violation"), 0.0)
        feasible_count += int(bool(report.get("ok", False)))
        step_count += 1

        violation_sums["avg_ratio_violation"] += _safe_float(report.get("ratio_violation"), 0.0)
        violation_sums["avg_assoc_violation"] += _safe_float(report.get("assoc_violation"), 0.0)
        violation_sums["avg_schedule_violation"] += _safe_float(report.get("schedule_violation"), 0.0)
        violation_sums["avg_candidate_violation"] += _safe_float(report.get("candidate_violation"), 0.0)
        violation_sums["avg_bw_violation"] += _safe_float(report.get("bw_violation"), 0.0)
        violation_sums["avg_cpu_violation"] += _safe_float(report.get("cpu_violation"), 0.0)
        violation_sums["avg_rate_violation"] += _safe_float(report.get("rate_violation"), 0.0)
        violation_sums["avg_nan_count"] += _safe_float(report.get("nan_count"), 0.0)

        violation_sums["avg_move_violation"] += _recursive_find_numeric(
            info, ["move_violation", "motion_violation", "movement_violation", "avg_move_violation"]
        )
        violation_sums["avg_boundary_violation"] += _recursive_find_numeric(
            info, ["boundary_violation", "out_of_boundary", "out_of_bounds", "avg_boundary_violation"]
        )
        violation_sums["avg_collision_violation"] += _recursive_find_numeric(
            info, ["collision_violation", "collision", "safe_distance_violation", "distance_violation", "avg_collision_violation"]
        )
        violation_sums["avg_battery_violation"] += _recursive_find_numeric(
            info, ["battery_violation", "energy_violation", "battery_negative", "avg_battery_violation"]
        )

    denom = max(step_count, 1)
    sched_denom = max(sched_count, 1)
    ratio_np = np.asarray(ratio_values, dtype=np.float32)

    out = {
        "episode_reward": total_reward,
        "system_cost": -total_reward,
        "avg_delay": total_delay / denom,
        "avg_energy": total_energy / denom,
        "avg_deadline_violation": total_deadline_violation / denom,
        "feasible_ratio": feasible_count / denom,
        "num_steps": float(step_count),
        "ratio_mean": float(np.mean(ratio_np)) if ratio_np.size else float("nan"),
        "ratio_std": float(np.std(ratio_np)) if ratio_np.size else float("nan"),
        "ratio_min": float(np.min(ratio_np)) if ratio_np.size else float("nan"),
        "ratio_max": float(np.max(ratio_np)) if ratio_np.size else float("nan"),
        "local_exec_ratio": local_exec_count / sched_denom,
        "neighbor_exec_ratio": neighbor_exec_count / sched_denom,
        "refine_improvement_per_step": refine_improve_sum / max(refine_improve_count, 1),
        "refine_improved_step_ratio": refine_improved_steps / max(refine_improve_count, 1),
    }

    for key, val in violation_sums.items():
        out[key] = val / denom
    return out


def summarize_rows(rows: List[Dict[str, Any]], group_keys: Sequence[str]) -> List[Dict[str, Any]]:
    metrics = [
        "system_cost",
        "avg_delay",
        "avg_energy",
        "avg_deadline_violation",
        "feasible_ratio",
        "ratio_mean",
        "ratio_std",
        "local_exec_ratio",
        "neighbor_exec_ratio",
        "avg_battery_violation",
        "refine_improvement_per_step",
        "refine_improved_step_ratio",
    ]

    groups: Dict[Tuple, List[Dict[str, Any]]] = {}
    for row in rows:
        key = tuple(row[g] for g in group_keys)
        groups.setdefault(key, []).append(row)

    summary: List[Dict[str, Any]] = []
    for key, vals in sorted(groups.items()):
        out = {g: k for g, k in zip(group_keys, key)}
        out["n"] = len(vals)
        for m in metrics:
            arr = np.asarray([float(v[m]) for v in vals if m in v and np.isfinite(float(v[m]))], dtype=np.float64)
            if arr.size == 0:
                out[f"{m}_mean"] = float("nan")
                out[f"{m}_std"] = float("nan")
            else:
                out[f"{m}_mean"] = float(np.mean(arr))
                out[f"{m}_std"] = float(np.std(arr, ddof=1)) if arr.size >= 2 else 0.0
        summary.append(out)
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


def print_summary_table(summary: List[Dict[str, Any]]) -> None:
    cols = [
        "mode",
        "n",
        "system_cost_mean",
        "avg_delay_mean",
        "avg_deadline_violation_mean",
        "feasible_ratio_mean",
        "local_exec_ratio_mean",
        "neighbor_exec_ratio_mean",
        "refine_improvement_per_step_mean",
    ]
    print("\n" + "=" * 110)
    print("Ablation summary")
    print("=" * 110)
    header = " | ".join([c[:28].ljust(28) for c in cols])
    print(header)
    print("-" * len(header))
    for row in summary:
        parts = []
        for c in cols:
            v = row.get(c, "")
            if isinstance(v, float):
                parts.append(f"{v:.6g}".ljust(28))
            else:
                parts.append(str(v).ljust(28))
        print(" | ".join(parts))
    print("=" * 110 + "\n")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Actor-only / Actor+refinement / Greedy-Random+refinement ablations."
    )
    parser.add_argument("--stage2-prefix", type=str, required=True,
                        help="Checkpoint prefix for the trained Stage-2/v6 policy.")
    parser.add_argument("--stage1-prefix", type=str, default="",
                        help="Optional Stage-1 checkpoint prefix for Stage-1 actor comparisons.")
    parser.add_argument("--ckpt-dir", type=str, default="checkpoints")
    parser.add_argument("--out", type=str, default="results/ablation_refinement")
    parser.add_argument("--seeds", type=str, default="72",
                        help="Comma-separated evaluation seeds, e.g. 42,52,62,72,82.")
    parser.add_argument("--device", type=str, default="auto")

    # Environment configuration. Match training exactly.
    parser.add_argument("--M", type=int, default=3)
    parser.add_argument("--K", type=int, default=16)
    parser.add_argument("--episode-length", type=int, default=20)
    parser.add_argument("--cpu-mode", type=str, default="kkt")
    parser.add_argument("--omega1", type=float, default=50.0)
    parser.add_argument("--omega2", type=float, default=1.0)
    parser.add_argument("--deadline-scale", type=float, default=2.5)
    parser.add_argument("--task-local-cpu-min", type=float, default=2000.0)
    parser.add_argument("--task-local-cpu-max", type=float, default=5000.0)
    parser.add_argument("--uav-energy-min", type=float, default=2600.0)
    parser.add_argument("--uav-energy-max", type=float, default=3800.0)

    # Refinement settings. Match final proposed method exactly.
    parser.add_argument("--model-refine-max-tasks", type=int, default=6)
    parser.add_argument("--model-refine-ratio", type=int, default=1)
    parser.add_argument("--model-refine-sched", type=int, default=1)
    parser.add_argument("--safety-margin", type=float, default=1.0)
    parser.add_argument("--collision-margin", type=float, default=1e-5)

    # What to run.
    parser.add_argument("--include-stage1", action="store_true",
                        help="Also evaluate Stage-1 Actor-only and Stage-1 Actor+refinement.")
    parser.add_argument("--include-random", action="store_true",
                        help="Also evaluate Random+refinement. Recommended for paper appendix.")
    parser.add_argument("--include-greedy", action="store_true", default=True,
                        help="Evaluate Greedy/heuristic+refinement. Enabled by default.")
    parser.add_argument("--no-greedy", dest="include_greedy", action="store_false")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = parse_seed_list(args.seeds)
    device = resolve_device(args.device)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("Refinement ablation evaluation")
    print("=" * 100)
    print(f"device        : {device}")
    print(f"seeds         : {seeds}")
    print(f"stage2_prefix : {args.stage2_prefix}")
    print(f"stage1_prefix : {args.stage1_prefix if args.stage1_prefix else '(not evaluated)'}")
    print(f"out           : {out_dir}")
    print("=" * 100)

    all_rows: List[Dict[str, Any]] = []

    # Build learned policies once using the first seed's state dimensions.
    # Dimensions are fixed for given M/K/config, so the same policy object can be
    # evaluated under all evaluation seeds.
    build_seed = seeds[0]
    set_global_seed(build_seed, device)
    build_env = make_env(args, build_seed)
    stage2_policy = build_learned_policy(build_env, args.stage2_prefix, device, args.ckpt_dir, build_seed)
    stage1_policy = None
    if args.include_stage1:
        if not args.stage1_prefix:
            raise ValueError("--include-stage1 requires --stage1-prefix.")
        stage1_policy = build_learned_policy(build_env, args.stage1_prefix, device, args.ckpt_dir, build_seed)

    for seed in seeds:
        print(f"\n[Seed {seed}] evaluating...")
        set_global_seed(seed, device)

        modes: List[Tuple[str, Any, bool]] = []
        if stage1_policy is not None:
            modes.append(("stage1_actor_only", stage1_policy, False))
            modes.append(("stage1_actor_refine", stage1_policy, True))

        modes.append(("stage2_actor_only", stage2_policy, False))
        modes.append(("stage2_actor_refine", stage2_policy, True))

        if args.include_greedy:
            modes.append(("greedy_refine", build_greedy_or_placeholder_policy(seed), True))
        if args.include_random:
            modes.append(("random_refine", build_random_policy(seed), True))

        for mode_name, base_policy, refine in modes:
            env = make_env(args, seed)
            eval_policy = wrap_policy_for_mode(env, base_policy, refine=refine, args=args)
            result = evaluate_policy_once(env, eval_policy, seed=seed)
            row: Dict[str, Any] = {
                "mode": mode_name,
                "seed": seed,
                "refine_enabled": int(refine),
                "stage2_prefix": args.stage2_prefix,
                "stage1_prefix": args.stage1_prefix,
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
            print(
                f"  {mode_name:22s} | cost={row['system_cost']:.3f} "
                f"delay={row['avg_delay']:.4f} feasible={row['feasible_ratio']:.3f} "
                f"deadline={row['avg_deadline_violation']:.4f} "
                f"local={row['local_exec_ratio']:.4f} neigh={row['neighbor_exec_ratio']:.4f} "
                f"ref_imp={row['refine_improvement_per_step']:.4f}"
            )

    summary = summarize_rows(all_rows, group_keys=["mode"])

    write_csv(out_dir / "ablation_detail.csv", all_rows)
    write_csv(out_dir / "ablation_summary.csv", summary)
    with open(out_dir / "ablation_detail.json", "w", encoding="utf-8") as f:
        json.dump(all_rows, f, ensure_ascii=False, indent=2)
    with open(out_dir / "ablation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print_summary_table(summary)
    print("Saved:")
    print(f"  {out_dir / 'ablation_detail.csv'}")
    print(f"  {out_dir / 'ablation_summary.csv'}")
    print(f"  {out_dir / 'ablation_detail.json'}")
    print(f"  {out_dir / 'ablation_summary.json'}")

    print("\nInterpretation checklist:")
    print("  1) stage2_actor_only > stage1_actor_only  => Stage-2 actor learned useful behavior.")
    print("  2) stage2_actor_refine > stage2_actor_only => model refinement adds value.")
    print("  3) stage2_actor_refine > greedy/random_refine => the learned actor still matters.")
    print("  If (2) holds but (3) fails, refinement is too dominant and the DRL contribution is weak.")


if __name__ == "__main__":
    main()
