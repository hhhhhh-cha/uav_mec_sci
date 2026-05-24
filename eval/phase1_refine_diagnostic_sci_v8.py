#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase-1/Stage-3 diagnostic evaluation for v8 Transformer-scheduling checkpoints.

Purpose
-------
This script does NOT retrain any model and does NOT change the Transformer+MADDPG
architecture. It only evaluates the same trained checkpoints under several
one-step model-refinement configurations to answer three questions:

1) Is Proposed worse mainly because max_tasks=6 refinement is too narrow?
2) Is the improvement mainly from ratio search or schedule search?
3) Which delay components explain the cost gap: local delay, uplink delay,
   backhaul delay, or edge execution delay?

Recommended location
--------------------
Put this file at:
    ~/projects/uav_mec_sci/eval/phase1_refine_diagnostic_sci_v6.py

Recommended command
-------------------
Run from project root:

PYTHONPATH=$(pwd) python3 ./eval/phase1_refine_diagnostic_sci_v6.py \
  --stage2-prefix proposed_full_stage2_main_d25_seed72_ep700_v6_model_refine \
  --pure-prefix pure_maddpg_main_d25_seed72_ep700 \
  --seeds 42,52,62,72,82,92,102,112,122,132 \
  --deadline-scale 2.5 \
  --task-local-cpu-min 2000 \
  --task-local-cpu-max 5000 \
  --episode-length 20 \
  --out results/phase1_refine_diagnostic/seed72

Outputs
-------
  phase1_refine_detail.csv
  phase1_refine_summary.csv
  phase1_refine_correlations.csv
  config.json
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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

# -----------------------------------------------------------------------------
# Make project-root imports robust when this file is run as ./eval/xxx.py
# -----------------------------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1] if THIS_FILE.parent.name == "eval" else Path.cwd()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from env.association import build_access_association
from eval.run_main_comparison_sci_v6 import (  # reuse your existing v6 loaders
    parse_seed_list,
    resolve_device,
    set_global_seed,
    safe_float,
    recursive_find_numeric,
    make_env,
    checkpoint_paths,
    load_state_dict_strict,
    set_policy_eval,
    build_pure_maddpg_policy,
    build_greedy_policy,
    build_random_policy,
    write_csv,
    write_json,
)
from train.train_proposed_full_stage2_converge_v6_model_refine import (
    apply_safe_mobility_projection,
    refine_action_by_one_step_model,
)

EPS = 1e-8

from model.mlp_actor import MLPActor
from model.proposed_obs_builder import get_observation_dim
from train.train_proposed_full_stage1_converge import get_actor_raw_action_dim
from policy.proposed_policy_v8_transformer_sched import build_default_proposed_policy_v8


def build_learned_proposed_policy(args: argparse.Namespace, prefix: str, device: str, build_seed: int) -> Any:
    """Build v8 policy and load actor/encoder/fusion/ratio/schedule checkpoints."""
    env = make_env(args, build_seed)
    obs = env.reset(seed=build_seed)
    state = obs["raw_state"]

    obs_dim = get_observation_dim(state)
    actor_raw_action_dim = get_actor_raw_action_dim(state)
    actor = MLPActor(obs_dim=obs_dim, action_dim=actor_raw_action_dim, hidden_dim=args.hidden_dim).to(device)

    policy = build_default_proposed_policy_v8(
        state=state,
        actor_net=actor,
        device=device,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        ff_hidden_dim=args.ff_hidden_dim,
        num_layers=args.num_layers,
    )

    paths = checkpoint_paths(prefix, args.ckpt_dir)
    paths["schedule_head"] = str(Path(args.ckpt_dir) / f"{prefix}_best_schedule_head.pth")
    load_state_dict_strict(policy.actor_net, paths["actor"], "proposed actor", device)
    load_state_dict_strict(policy.encoder, paths["encoder"], "proposed encoder", device)
    load_state_dict_strict(policy.fusion_net, paths["fusion_net"], "proposed fusion", device)
    load_state_dict_strict(policy.ratio_head, paths["ratio_head"], "proposed ratio_head", device)
    load_state_dict_strict(policy.schedule_head, paths["schedule_head"], "proposed schedule_head", device)
    set_policy_eval(policy)
    return policy



# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------
def _np1d(x: Any, dtype=np.float32) -> np.ndarray:
    return np.asarray(x, dtype=dtype).reshape(-1)


def _copy_action_for_env(action: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Return only env-accepted action keys, copied as numpy arrays."""
    return {
        "move_dist": np.asarray(action["move_dist"], dtype=np.float32).copy(),
        "move_angle": np.asarray(action["move_angle"], dtype=np.float32).copy(),
        "offload_ratio": np.asarray(action["offload_ratio"], dtype=np.float32).copy(),
        "sched_beta": np.asarray(action["sched_beta"], dtype=np.float32).copy(),
    }


def _sched_exec_indices(action: Dict[str, Any], access_assoc: np.ndarray, M: int, K: int) -> np.ndarray:
    sched_beta = np.asarray(action["sched_beta"], dtype=np.float32).reshape(K, M, M)
    out = np.zeros(K, dtype=np.int64)
    for k in range(K):
        access_m = int(np.argmax(access_assoc[:, k]))
        out[k] = int(np.argmax(sched_beta[k, access_m, :]))
    return out


def _arr_sum(metrics: Dict[str, Any], key: str) -> float:
    if key not in metrics:
        return 0.0
    arr = np.asarray(metrics[key], dtype=np.float64)
    if arr.size == 0:
        return 0.0
    val = float(np.nansum(arr))
    return val if np.isfinite(val) else 0.0


def _mean_or_nan(values: List[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.mean(arr))


def _std_or_nan(values: List[float], ddof: int = 0) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    if arr.size <= ddof:
        return 0.0
    return float(np.std(arr, ddof=ddof))


def _pearson(x: List[float], y: List[float]) -> float:
    xx = np.asarray(x, dtype=np.float64)
    yy = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(xx) & np.isfinite(yy)
    xx = xx[mask]
    yy = yy[mask]
    if xx.size < 3:
        return float("nan")
    if float(np.std(xx)) < EPS or float(np.std(yy)) < EPS:
        return float("nan")
    return float(np.corrcoef(xx, yy)[0, 1])


# -----------------------------------------------------------------------------
# Diagnostic mode definition
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class ModeSpec:
    method: str
    policy_kind: str  # proposed / pure / greedy / random
    refine: bool
    max_tasks: int = 0
    ratio_search: bool = False
    schedule_search: bool = False


class DiagnosticRefineWrapper:
    """
    Wrapper that applies the same safe mobility projection and same one-step
    model refinement as v6, but additionally records before/after diagnostics.
    """

    def __init__(
        self,
        env: Any,
        policy: Any,
        refine: bool,
        max_tasks: int,
        ratio_search: bool,
        schedule_search: bool,
        safety_margin: float = 1.0,
        collision_margin: float = 1e-5,
    ):
        self.env = env
        self.policy = policy
        self.refine = bool(refine)
        self.max_tasks = int(max_tasks)
        self.ratio_search = bool(ratio_search)
        self.schedule_search = bool(schedule_search)
        self.safety_margin = float(safety_margin)
        self.collision_margin = float(collision_margin)

    def _base_act(self, state: Dict[str, Any], access_assoc: np.ndarray, deterministic: bool) -> Dict[str, Any]:
        try:
            out = self.policy.act(
                state=state,
                access_assoc=access_assoc,
                deterministic=deterministic,
                return_aux=False,
            )
        except TypeError:
            try:
                out = self.policy.act(
                    state=state,
                    access_assoc=access_assoc,
                    deterministic=deterministic,
                )
            except TypeError:
                out = self.policy.act(state=state, access_assoc=access_assoc)
        if isinstance(out, tuple):
            out = out[0]
        return out

    def act(self, state: Dict[str, Any], access_assoc: np.ndarray, deterministic: bool = True) -> Dict[str, Any]:
        M = int(state["M"])
        K = int(state["K"])

        action0 = self._base_act(state, access_assoc, deterministic=deterministic)
        action0 = apply_safe_mobility_projection(
            action0,
            state,
            safety_margin=self.safety_margin,
            collision_margin=self.collision_margin,
        )
        action0 = _copy_action_for_env(action0)

        ratio_before = np.asarray(action0["offload_ratio"], dtype=np.float32).reshape(K).copy()
        sched_before = _sched_exec_indices(action0, access_assoc, M, K)

        if self.refine:
            action1 = refine_action_by_one_step_model(
                env=self.env,
                state=state,
                access_assoc=access_assoc,
                action=action0,
                enable=True,
                max_tasks=self.max_tasks,
                ratio_search=self.ratio_search,
                schedule_search=self.schedule_search,
            )
            refine_cost_before = safe_float(
                action1.get("_model_refine_cost_before") if isinstance(action1, dict) else None,
                0.0,
            )
            refine_cost_after = safe_float(
                action1.get("_model_refine_cost_after") if isinstance(action1, dict) else None,
                0.0,
            )
            refine_improvement = safe_float(
                action1.get("_model_refine_improvement") if isinstance(action1, dict) else None,
                0.0,
            )
            action1 = _copy_action_for_env(action1)
            action1["_model_refine_cost_before"] = refine_cost_before
            action1["_model_refine_cost_after"] = refine_cost_after
            action1["_model_refine_improvement"] = refine_improvement
        else:
            action1 = action0
            action1["_model_refine_cost_before"] = 0.0
            action1["_model_refine_cost_after"] = 0.0
            action1["_model_refine_improvement"] = 0.0

        ratio_after = np.asarray(action1["offload_ratio"], dtype=np.float32).reshape(K).copy()
        sched_after = _sched_exec_indices(action1, access_assoc, M, K)

        action1["_phase1_refine_enabled"] = int(self.refine)
        action1["_phase1_refine_max_tasks"] = int(self.max_tasks)
        action1["_phase1_ratio_search"] = int(self.ratio_search)
        action1["_phase1_schedule_search"] = int(self.schedule_search)
        action1["_phase1_ratio_before_mean"] = float(np.mean(ratio_before))
        action1["_phase1_ratio_after_mean"] = float(np.mean(ratio_after))
        action1["_phase1_ratio_before_std"] = float(np.std(ratio_before))
        action1["_phase1_ratio_after_std"] = float(np.std(ratio_after))
        action1["_phase1_ratio_abs_delta_mean"] = float(np.mean(np.abs(ratio_after - ratio_before)))
        action1["_phase1_ratio_changed_frac"] = float(np.mean(np.abs(ratio_after - ratio_before) > 1e-6))
        action1["_phase1_sched_changed_frac"] = float(np.mean(sched_after != sched_before))
        return action1


# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------
def evaluate_policy_once(env: Any, policy: Any, seed: int) -> Dict[str, float]:
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
        "delay_local_sum": 0.0,
        "delay_up_sum": 0.0,
        "delay_bh_sum": 0.0,
        "delay_exec_sum": 0.0,
        "delay_edge_sum": 0.0,
        "delay_total_sum": 0.0,
        "energy_tx_sum": 0.0,
        "energy_cmp_sum": 0.0,
        "energy_fly_sum": 0.0,
    }

    alias_map = {
        "avg_move_violation": ["move_violation", "motion_violation", "movement_violation", "avg_move_violation"],
        "avg_boundary_violation": ["boundary_violation", "out_of_boundary", "out_of_bounds", "avg_boundary_violation"],
        "avg_collision_violation": ["collision_violation", "collision", "safe_distance_violation", "distance_violation"],
        "avg_battery_violation": ["battery_violation", "energy_violation", "battery_negative", "avg_battery_violation"],
    }

    ratio_values: List[float] = []
    ratio_before_means: List[float] = []
    ratio_after_means: List[float] = []
    ratio_abs_delta_means: List[float] = []
    ratio_changed_fracs: List[float] = []
    sched_changed_fracs: List[float] = []

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

        refine_imp = safe_float(action.get("_model_refine_improvement"), 0.0)
        if int(action.get("_phase1_refine_enabled", 0)) == 1:
            refine_improvement_sum += refine_imp
            refine_improvement_count += 1
            if refine_imp > 1e-9:
                refine_improved_steps += 1

        ratio_before_means.append(safe_float(action.get("_phase1_ratio_before_mean"), float("nan")))
        ratio_after_means.append(safe_float(action.get("_phase1_ratio_after_mean"), float("nan")))
        ratio_abs_delta_means.append(safe_float(action.get("_phase1_ratio_abs_delta_mean"), float("nan")))
        ratio_changed_fracs.append(safe_float(action.get("_phase1_ratio_changed_frac"), float("nan")))
        sched_changed_fracs.append(safe_float(action.get("_phase1_sched_changed_frac"), float("nan")))

        ratio_arr = np.asarray(action["offload_ratio"], dtype=np.float32).reshape(-1)
        ratio_values.extend([float(x) for x in ratio_arr])

        sched_beta = np.asarray(action["sched_beta"], dtype=np.float32).reshape(K, M, M)
        for k in range(K):
            access_m = int(np.argmax(access_assoc[:, k]))
            exec_j = int(np.argmax(sched_beta[k, access_m, :]))
            sched_count += 1
            if exec_j == access_m:
                local_exec_count += 1
            else:
                neighbor_exec_count += 1

        step_action = _copy_action_for_env(action)
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

        sums["delay_local_sum"] += _arr_sum(metrics, "delay_local")
        sums["delay_up_sum"] += _arr_sum(metrics, "delay_up")
        sums["delay_bh_sum"] += _arr_sum(metrics, "delay_bh")
        sums["delay_exec_sum"] += _arr_sum(metrics, "delay_exec")
        sums["delay_edge_sum"] += _arr_sum(metrics, "delay_edge")
        sums["delay_total_sum"] += _arr_sum(metrics, "delay_total")
        sums["energy_tx_sum"] += _arr_sum(metrics, "energy_tx")
        sums["energy_cmp_sum"] += _arr_sum(metrics, "energy_cmp")
        sums["energy_fly_sum"] += _arr_sum(metrics, "energy_fly")

        step_count += 1

    denom = max(step_count, 1)
    sched_denom = max(sched_count, 1)
    ratio_np = np.asarray(ratio_values, dtype=np.float64)

    out: Dict[str, float] = {
        "episode_reward": float(total_reward),
        "system_cost": float(-total_reward),
        "avg_delay": float(total_delay / denom),
        "avg_energy": float(total_energy / denom),
        "feasible_ratio": float(feasible_count / denom),
        "num_steps": float(step_count),
        "ratio_mean": float(np.mean(ratio_np)) if ratio_np.size else float("nan"),
        "ratio_std": float(np.std(ratio_np)) if ratio_np.size else float("nan"),
        "ratio_min": float(np.min(ratio_np)) if ratio_np.size else float("nan"),
        "ratio_p25": float(np.percentile(ratio_np, 25)) if ratio_np.size else float("nan"),
        "ratio_p50": float(np.percentile(ratio_np, 50)) if ratio_np.size else float("nan"),
        "ratio_p75": float(np.percentile(ratio_np, 75)) if ratio_np.size else float("nan"),
        "ratio_max": float(np.max(ratio_np)) if ratio_np.size else float("nan"),
        "local_exec_ratio": float(local_exec_count / sched_denom),
        "neighbor_exec_ratio": float(neighbor_exec_count / sched_denom),
        "refine_improvement_per_step": float(refine_improvement_sum / max(refine_improvement_count, 1)),
        "refine_improved_step_ratio": float(refine_improved_steps / max(refine_improvement_count, 1)),
        "ratio_before_mean_per_slot": _mean_or_nan(ratio_before_means),
        "ratio_after_mean_per_slot": _mean_or_nan(ratio_after_means),
        "ratio_abs_delta_mean_per_slot": _mean_or_nan(ratio_abs_delta_means),
        "ratio_changed_frac_per_slot": _mean_or_nan(ratio_changed_fracs),
        "sched_changed_frac_per_slot": _mean_or_nan(sched_changed_fracs),
        "decision_time_per_slot_sec": float(runtime_action_sec / denom),
    }

    for key, value in sums.items():
        out[key] = float(value / denom)

    # Useful normalized components: share relative to summed task delay_total.
    delay_total = max(abs(out.get("delay_total_sum", 0.0)), EPS)
    out["delay_local_over_total"] = float(out.get("delay_local_sum", 0.0) / delay_total)
    out["delay_up_over_total"] = float(out.get("delay_up_sum", 0.0) / delay_total)
    out["delay_bh_over_total"] = float(out.get("delay_bh_sum", 0.0) / delay_total)
    out["delay_exec_over_total"] = float(out.get("delay_exec_sum", 0.0) / delay_total)
    out["delay_edge_over_total"] = float(out.get("delay_edge_sum", 0.0) / delay_total)
    return out


# -----------------------------------------------------------------------------
# Summary / correlations
# -----------------------------------------------------------------------------
def summarize_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not rows:
        return []

    metric_names: List[str] = []
    skip_keys = {
        "method", "seed", "policy_kind", "refine_enabled", "refine_max_tasks",
        "refine_ratio_search", "refine_schedule_search", "stage2_prefix",
        "pure_prefix", "deadline_scale", "task_local_cpu_min", "task_local_cpu_max",
        "episode_length",
    }
    for row in rows:
        for key, val in row.items():
            if key in skip_keys:
                continue
            try:
                float(val)
            except Exception:
                continue
            if key not in metric_names:
                metric_names.append(key)

    methods = sorted(set(str(row["method"]) for row in rows))
    summary: List[Dict[str, Any]] = []
    for method in methods:
        vals = [row for row in rows if str(row["method"]) == method]
        out: Dict[str, Any] = {"method": method, "n": len(vals)}
        # preserve mode metadata if constant
        for meta in ["policy_kind", "refine_enabled", "refine_max_tasks", "refine_ratio_search", "refine_schedule_search"]:
            unique = sorted(set(str(v.get(meta, "")) for v in vals))
            out[meta] = unique[0] if len(unique) == 1 else ";".join(unique)

        for metric in metric_names:
            arr = np.asarray([safe_float(v.get(metric), float("nan")) for v in vals], dtype=np.float64)
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

    summary.sort(key=lambda r: safe_float(r.get("system_cost_mean"), float("inf")))
    best = safe_float(summary[0].get("system_cost_mean"), float("nan")) if summary else float("nan")
    for row in summary:
        cost = safe_float(row.get("system_cost_mean"), float("nan"))
        row["cost_gap_to_best_percent"] = float((cost - best) / max(abs(best), EPS) * 100.0) if np.isfinite(cost) and np.isfinite(best) else float("nan")
    return summary


def make_correlations(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    metrics = [
        "ratio_mean",
        "ratio_std",
        "ratio_abs_delta_mean_per_slot",
        "ratio_changed_frac_per_slot",
        "sched_changed_frac_per_slot",
        "local_exec_ratio",
        "neighbor_exec_ratio",
        "delay_local_sum",
        "delay_up_sum",
        "delay_bh_sum",
        "delay_exec_sum",
        "delay_edge_sum",
        "energy_tx_sum",
        "energy_cmp_sum",
        "energy_fly_sum",
        "refine_improvement_per_step",
    ]
    groups = ["ALL"] + sorted(set(str(r["method"]) for r in rows))
    out: List[Dict[str, Any]] = []
    for group in groups:
        vals = rows if group == "ALL" else [r for r in rows if str(r["method"]) == group]
        if len(vals) < 3:
            continue
        y_cost = [safe_float(r.get("system_cost"), float("nan")) for r in vals]
        y_delay = [safe_float(r.get("avg_delay"), float("nan")) for r in vals]
        for metric in metrics:
            x = [safe_float(r.get(metric), float("nan")) for r in vals]
            out.append(
                {
                    "group": group,
                    "metric": metric,
                    "n": len(vals),
                    "pearson_with_system_cost": _pearson(x, y_cost),
                    "pearson_with_avg_delay": _pearson(x, y_delay),
                }
            )
    return out


def print_summary(summary: List[Dict[str, Any]]) -> None:
    print("\n" + "=" * 180)
    print("Phase-1 refinement diagnostic summary")
    print("=" * 180)
    print(
        f"{'Method':34s} | {'Cost mean±std':22s} | {'Delay':17s} | {'Ratio':15s} | "
        f"{'Local/Neigh':17s} | {'RΔ':8s} | {'SΔ':8s} | {'RefImp':12s} | {'Feas':6s} | {'ms/slot':8s}"
    )
    print("-" * 180)
    for row in summary:
        print(
            f"{str(row['method']):34s} | "
            f"{safe_float(row.get('system_cost_mean')):9.2f}±{safe_float(row.get('system_cost_std')):<9.2f} | "
            f"{safe_float(row.get('avg_delay_mean')):8.3f}±{safe_float(row.get('avg_delay_std')):<6.3f} | "
            f"{safe_float(row.get('ratio_mean_mean')):6.4f}±{safe_float(row.get('ratio_mean_std')):<6.4f} | "
            f"{safe_float(row.get('local_exec_ratio_mean')):5.3f}/{safe_float(row.get('neighbor_exec_ratio_mean')):<5.3f} | "
            f"{safe_float(row.get('ratio_changed_frac_per_slot_mean')):6.3f} | "
            f"{safe_float(row.get('sched_changed_frac_per_slot_mean')):6.3f} | "
            f"{safe_float(row.get('refine_improvement_per_step_mean')):10.3f} | "
            f"{safe_float(row.get('feasible_ratio_mean')):5.3f} | "
            f"{1000.0 * safe_float(row.get('decision_time_per_slot_sec_mean')):7.2f}"
        )
    print("=" * 180 + "\n")


# -----------------------------------------------------------------------------
# CLI / main
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase-1 refinement diagnostic for UAV-MEC v6.")

    # Checkpoints.
    parser.add_argument("--stage2-prefix", type=str, required=True, help="Prefix of final Stage-2/v6 checkpoints.")
    parser.add_argument("--pure-prefix", type=str, default="", help="Optional Pure MADDPG actor checkpoint prefix.")
    parser.add_argument("--ckpt-dir", type=str, default="checkpoints")
    parser.add_argument("--allow-untrained-pure", action="store_true", help="Debug only. Do not use in paper.")

    # Output and runtime.
    parser.add_argument("--out", type=str, default="results/phase1_refine_diagnostic")
    parser.add_argument("--seeds", type=str, default="42,52,62,72,82,92,102,112,122,132")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--quiet", action="store_true")

    # Which extra baselines to evaluate.
    parser.add_argument("--no-pure", dest="include_pure", action="store_false", default=True)
    parser.add_argument("--include-greedy", action="store_true", default=False)
    parser.add_argument("--include-random", action="store_true", default=False)

    # Env config. Keep identical to run_main_comparison_sci_v6.py defaults.
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

    # Refinement safety config.
    parser.add_argument("--safety-margin", type=float, default=1.0)
    parser.add_argument("--collision-margin", type=float, default=1e-5)

    # Phase-1 diagnostic sizes.
    parser.add_argument("--small-max-tasks", type=int, default=6, help="Original v6 local refinement width.")
    parser.add_argument("--large-max-tasks", type=int, default=16, help="Full-task refinement width for K=16.")

    return parser.parse_args()


def build_modes(args: argparse.Namespace, pure_policy_available: bool) -> List[ModeSpec]:
    small = int(args.small_max_tasks)
    large = int(args.large_max_tasks)
    modes: List[ModeSpec] = [
        ModeSpec("Proposed_ActorOnly_Safe", "proposed", False, 0, False, False),
        ModeSpec(f"Proposed_Refine{small}_Both", "proposed", True, small, True, True),
        ModeSpec(f"Proposed_Refine{large}_Both", "proposed", True, large, True, True),
        ModeSpec(f"Proposed_Refine{large}_RatioOnly", "proposed", True, large, True, False),
        ModeSpec(f"Proposed_Refine{large}_SchedOnly", "proposed", True, large, False, True),
    ]

    if args.include_pure and pure_policy_available:
        modes.extend(
            [
                ModeSpec("Pure_MADDPG_ActorOnly_Safe", "pure", False, 0, False, False),
                ModeSpec(f"Pure_MADDPG_Refine{small}_Both", "pure", True, small, True, True),
                ModeSpec(f"Pure_MADDPG_Refine{large}_Both", "pure", True, large, True, True),
            ]
        )

    if args.include_greedy:
        modes.extend(
            [
                ModeSpec("Greedy_ActorOnly_Safe", "greedy", False, 0, False, False),
                ModeSpec(f"Greedy_Refine{large}_Both", "greedy", True, large, True, True),
            ]
        )

    if args.include_random:
        modes.extend(
            [
                ModeSpec("Random_ActorOnly_Safe", "random", False, 0, False, False),
                ModeSpec(f"Random_Refine{large}_Both", "random", True, large, True, True),
            ]
        )

    return modes


def main() -> None:
    args = parse_args()
    seeds = parse_seed_list(args.seeds)
    device = resolve_device(args.device)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 120)
    print("Phase-1 refinement diagnostic for UAV-MEC v6")
    print("=" * 120)
    print(f"project_root       : {PROJECT_ROOT}")
    print(f"device             : {device}")
    print(f"seeds              : {seeds}")
    print(f"stage2_prefix      : {args.stage2_prefix}")
    print(f"pure_prefix        : {args.pure_prefix or '(not used)'}")
    print(f"small/large tasks  : {args.small_max_tasks}/{args.large_max_tasks}")
    print(f"out                : {out_dir}")
    print("=" * 120)

    write_json(out_dir / "config.json", vars(args))

    build_seed = seeds[0]
    set_global_seed(build_seed, device)

    proposed_policy = build_learned_proposed_policy(args, args.stage2_prefix, device, build_seed)
    pure_policy: Optional[Any] = None
    if args.include_pure and args.pure_prefix:
        pure_policy = build_pure_maddpg_policy(args, device, build_seed)
    elif args.include_pure and not args.pure_prefix:
        print("[WARN] --pure-prefix not provided; Pure MADDPG diagnostic modes will be skipped.")

    modes = build_modes(args, pure_policy_available=(pure_policy is not None))
    print("Diagnostic modes:")
    for m in modes:
        print(
            f"  {m.method:34s} | policy={m.policy_kind:8s} refine={int(m.refine)} "
            f"max_tasks={m.max_tasks:2d} ratio={int(m.ratio_search)} sched={int(m.schedule_search)}"
        )

    all_rows: List[Dict[str, Any]] = []
    global_t0 = time.time()

    for seed in seeds:
        print(f"\n[Eval seed {seed}]")
        set_global_seed(seed, device)

        for mode in modes:
            if mode.policy_kind == "proposed":
                base_policy = proposed_policy
            elif mode.policy_kind == "pure":
                if pure_policy is None:
                    continue
                base_policy = pure_policy
            elif mode.policy_kind == "greedy":
                base_policy = build_greedy_policy(seed)
            elif mode.policy_kind == "random":
                base_policy = build_random_policy(seed)
            else:
                raise ValueError(f"Unknown policy_kind: {mode.policy_kind}")

            env = make_env(args, seed)
            eval_policy = DiagnosticRefineWrapper(
                env=env,
                policy=base_policy,
                refine=mode.refine,
                max_tasks=mode.max_tasks,
                ratio_search=mode.ratio_search,
                schedule_search=mode.schedule_search,
                safety_margin=args.safety_margin,
                collision_margin=args.collision_margin,
            )
            result = evaluate_policy_once(env, eval_policy, seed=seed)

            row: Dict[str, Any] = {
                "method": mode.method,
                "seed": seed,
                "policy_kind": mode.policy_kind,
                "refine_enabled": int(mode.refine),
                "refine_max_tasks": int(mode.max_tasks),
                "refine_ratio_search": int(mode.ratio_search),
                "refine_schedule_search": int(mode.schedule_search),
                "stage2_prefix": args.stage2_prefix,
                "pure_prefix": args.pure_prefix,
                "deadline_scale": args.deadline_scale,
                "task_local_cpu_min": args.task_local_cpu_min,
                "task_local_cpu_max": args.task_local_cpu_max,
                "episode_length": args.episode_length,
            }
            row.update(result)
            all_rows.append(row)

            if not args.quiet:
                print(
                    f"  {mode.method:34s} | cost={row['system_cost']:10.3f} "
                    f"delay={row['avg_delay']:8.4f} ratio={row['ratio_mean']:.4f} "
                    f"local/neigh={row['local_exec_ratio']:.3f}/{row['neighbor_exec_ratio']:.3f} "
                    f"RΔ={row['ratio_changed_frac_per_slot']:.3f} SΔ={row['sched_changed_frac_per_slot']:.3f} "
                    f"feas={row['feasible_ratio']:.3f} "
                    f"t={1000.0 * row['decision_time_per_slot_sec']:.2f}ms"
                )

    summary = summarize_rows(all_rows)
    correlations = make_correlations(all_rows)

    write_csv(out_dir / "phase1_refine_detail.csv", all_rows)
    write_csv(out_dir / "phase1_refine_summary.csv", summary)
    write_csv(out_dir / "phase1_refine_correlations.csv", correlations)
    write_json(out_dir / "phase1_refine_detail.json", all_rows)
    write_json(out_dir / "phase1_refine_summary.json", summary)
    write_json(out_dir / "phase1_refine_correlations.json", correlations)

    print_summary(summary)
    print(f"Total runtime: {time.time() - global_t0:.2f}s")
    print("Saved:")
    print(f"  {out_dir / 'config.json'}")
    print(f"  {out_dir / 'phase1_refine_detail.csv'}")
    print(f"  {out_dir / 'phase1_refine_summary.csv'}")
    print(f"  {out_dir / 'phase1_refine_correlations.csv'}")

    print("\nHow to read this diagnostic:")
    print("  A) If Proposed_Refine16_Both is much better than Proposed_Refine6_Both, the original local search width is insufficient.")
    print("  B) If Proposed_Refine16_RatioOnly gives most of the gain, the main defect is offloading-ratio learning.")
    print("  C) If Proposed_Refine16_SchedOnly gives most of the gain, the main defect is task-execution scheduling.")
    print("  D) Compare delay_local_sum / delay_up_sum / delay_bh_sum / delay_exec_sum to locate the real delay source.")
    print("  E) Use ratio_changed_frac_per_slot and sched_changed_frac_per_slot to judge whether refine is heavily correcting the actor.")


if __name__ == "__main__":
    main()
