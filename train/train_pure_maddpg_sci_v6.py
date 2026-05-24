#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SCI-style Pure MADDPG baseline training for the UAV-MEC project.

Purpose
-------
This file replaces the earlier train_pure_maddpg.py / train_pure_maddpg1.py
for the final main-comparison experiments.

Baseline definition
-------------------
Pure_MADDPG here means:
  - no Transformer encoder;
  - no topology-aware context module;
  - no ratio head / fusion module;
  - no model-based one-step refinement;
  - one MLP actor directly outputs high-level decisions:
        move_dist(M), move_angle(M), offload_ratio(K), sched_score(K*M);
  - the same environment-side analytical bandwidth/CPU solvers are used as in
    other methods, so the comparison isolates the high-level policy architecture.

Default final environment
-------------------------
  M=3, K=16, episode_length=20,
  deadline_scale=2.5, task_local_cpu=[2000, 5000],
  omega1=50, omega2=1, cpu_mode=kkt.

Run example
-----------
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
"""

from __future__ import annotations

import copy
import csv
import json
import os
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from env.mec_env import MultiUavMecEnv
from env.association import build_access_association
from model.mlp_actor import MLPActor
from model.mlp_critic import MLPCritic
from model.proposed_obs_builder import build_global_observation, get_observation_dim
from solver.bandwidth_solver import solve_bandwidth_allocation
from solver.cpu_solver import solve_cpu_allocation

EPS = 1e-8


# =============================================================================
# Environment variable helpers
# =============================================================================
def _env_int(name: str, default: int) -> int:
    val = os.environ.get(name)
    return int(val) if val not in (None, "") else int(default)


def _env_float(name: str, default: float) -> float:
    val = os.environ.get(name)
    return float(val) if val not in (None, "") else float(default)


def _env_str(name: str, default: str) -> str:
    val = os.environ.get(name)
    return str(val) if val not in (None, "") else str(default)


def _env_bool(name: str, default: bool = False) -> bool:
    val = os.environ.get(name)
    if val is None:
        return bool(default)
    return str(val).strip().lower() in {"1", "true", "yes", "y", "on"}


def resolve_device() -> str:
    requested = _env_str("DEVICE", "auto").lower()
    if requested == "cpu":
        return "cpu"
    if requested.startswith("cuda"):
        return requested if torch.cuda.is_available() else "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def configure_torch_runtime(device: str) -> None:
    if device == "cpu":
        torch.set_num_threads(_env_int("TORCH_NUM_THREADS", 1))
        torch.set_num_interop_threads(_env_int("TORCH_INTEROP_THREADS", 1))
    elif device.startswith("cuda"):
        torch.backends.cudnn.benchmark = True


# =============================================================================
# Replay buffer
# =============================================================================
class FullReplayBuffer:
    def __init__(self, obs_dim: int, action_dim: int, capacity: int = 30000):
        self.capacity = int(capacity)
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.buffer: deque = deque(maxlen=self.capacity)

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        next_action: np.ndarray,
        done: float,
    ) -> None:
        self.buffer.append(
            {
                "obs": np.asarray(obs, dtype=np.float32),
                "action": np.asarray(action, dtype=np.float32),
                "reward": float(reward),
                "next_obs": np.asarray(next_obs, dtype=np.float32),
                "next_action": np.asarray(next_action, dtype=np.float32),
                "done": float(done),
            }
        )

    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        if len(self.buffer) < batch_size:
            raise ValueError(f"buffer has {len(self.buffer)} samples, cannot sample {batch_size}")
        idx = np.random.choice(len(self.buffer), size=batch_size, replace=False)
        batch = [self.buffer[i] for i in idx]
        return {
            "obs": np.stack([b["obs"] for b in batch], axis=0),
            "action": np.stack([b["action"] for b in batch], axis=0),
            "reward": np.asarray([b["reward"] for b in batch], dtype=np.float32).reshape(-1, 1),
            "next_obs": np.stack([b["next_obs"] for b in batch], axis=0),
            "next_action": np.stack([b["next_action"] for b in batch], axis=0),
            "done": np.asarray([b["done"] for b in batch], dtype=np.float32).reshape(-1, 1),
        }

    def __len__(self) -> int:
        return len(self.buffer)


# =============================================================================
# Full-action layout helpers
# =============================================================================
def get_pure_actor_raw_action_dim(state: Dict[str, Any]) -> int:
    """[move_dist(M), move_angle(M), offload_ratio(K), sched_score(K*M)]"""
    M = int(state["M"])
    K = int(state["K"])
    return M + M + K + K * M


def get_full_action_dim(state: Dict[str, Any]) -> int:
    """[dist(M), angle(M), ratio(K), beta(K*M*M), bandwidth(M*K), cpu(M*K)]"""
    M = int(state["M"])
    K = int(state["K"])
    return M + M + K + K * M * M + M * K + M * K


def flatten_full_action(action: Dict[str, Any]) -> np.ndarray:
    return np.concatenate(
        [
            np.asarray(action["move_dist"], dtype=np.float32).reshape(-1),
            np.asarray(action["move_angle"], dtype=np.float32).reshape(-1),
            np.asarray(action["offload_ratio"], dtype=np.float32).reshape(-1),
            np.asarray(action["sched_beta"], dtype=np.float32).reshape(-1),
            np.asarray(action["bandwidth_alloc"], dtype=np.float32).reshape(-1),
            np.asarray(action["cpu_alloc"], dtype=np.float32).reshape(-1),
        ],
        axis=0,
    ).astype(np.float32)


def soft_update(target_net: nn.Module, source_net: nn.Module, tau: float = 0.005) -> None:
    for tp, sp in zip(target_net.parameters(), source_net.parameters()):
        tp.data.copy_(tau * sp.data + (1.0 - tau) * tp.data)


# =============================================================================
# Safe mobility projection, same semantics as the proposed method
# =============================================================================
def _get_uav_positions_from_state(state: Dict[str, Any], M: int) -> np.ndarray | None:
    for key in ["uav_pos", "uav_positions", "uav_xy", "q_uav", "q", "positions"]:
        if key in state:
            arr = np.asarray(state[key], dtype=np.float32)
            if arr.ndim == 2 and arr.shape[0] >= M and arr.shape[1] >= 2:
                return arr[:M, :2].copy()
    if "uav_x" in state and "uav_y" in state:
        x = np.asarray(state["uav_x"], dtype=np.float32).reshape(-1)
        y = np.asarray(state["uav_y"], dtype=np.float32).reshape(-1)
        if x.size >= M and y.size >= M:
            return np.stack([x[:M], y[:M]], axis=1).astype(np.float32)
    return None


def _next_positions(q_xy: np.ndarray, move_dist: np.ndarray, move_angle: np.ndarray) -> np.ndarray:
    dx = move_dist * np.cos(move_angle)
    dy = move_dist * np.sin(move_angle)
    return q_xy + np.stack([dx, dy], axis=1)


def _all_pairwise_safe(pos_xy: np.ndarray, min_dist: float) -> bool:
    M = pos_xy.shape[0]
    for i in range(M):
        for j in range(i + 1, M):
            if np.linalg.norm(pos_xy[i] - pos_xy[j]) < min_dist:
                return False
    return True


def apply_safe_mobility_projection(
    action: Dict[str, Any],
    state: Dict[str, Any],
    safety_margin: float = 1.0,
    collision_margin: float = 1e-5,
    binary_iters: int = 24,
) -> Dict[str, Any]:
    if "move_dist" not in action or "move_angle" not in action:
        return action

    M = int(state.get("M", len(np.asarray(action["move_dist"]).reshape(-1))))
    move_dist = np.asarray(action["move_dist"], dtype=np.float32).reshape(-1).copy()
    move_angle = np.asarray(action["move_angle"], dtype=np.float32).reshape(-1).copy()
    if move_dist.size < M or move_angle.size < M:
        return action

    q_xy = _get_uav_positions_from_state(state, M)
    max_speed = float(state.get("max_speed", 15.0))
    delta_t = float(state.get("delta_t", 1.0))
    max_move = max(max_speed * delta_t, EPS)

    move_dist = np.nan_to_num(move_dist[:M], nan=0.0, posinf=max_move, neginf=0.0)
    move_angle = np.nan_to_num(move_angle[:M], nan=0.0, posinf=np.pi, neginf=-np.pi)
    move_dist = np.clip(move_dist, 0.0, max_move).astype(np.float32)
    move_angle = np.clip(move_angle, -np.pi, np.pi).astype(np.float32)

    if q_xy is None:
        out = dict(action)
        out["move_dist"] = move_dist.astype(np.float32)
        out["move_angle"] = move_angle.astype(np.float32)
        return out

    area_size = float(state.get("area_size", 100.0))
    low = float(safety_margin)
    high = float(area_size - safety_margin)
    min_dist = float(state.get("min_uav_distance", 0.0)) + float(collision_margin)

    # Boundary projection: shrink each move along its proposed direction.
    for m in range(M):
        d0 = float(move_dist[m])
        theta = float(move_angle[m])
        if d0 <= EPS:
            continue
        proposed = q_xy[m] + np.array([d0 * np.cos(theta), d0 * np.sin(theta)], dtype=np.float32)
        if low <= proposed[0] <= high and low <= proposed[1] <= high:
            continue
        lo, hi = 0.0, d0
        for _ in range(binary_iters):
            mid = 0.5 * (lo + hi)
            p_mid = q_xy[m] + np.array([mid * np.cos(theta), mid * np.sin(theta)], dtype=np.float32)
            if low <= p_mid[0] <= high and low <= p_mid[1] <= high:
                lo = mid
            else:
                hi = mid
        move_dist[m] = max(0.0, lo)

    # Collision projection: shrink all movements together if needed.
    if min_dist > 0.0:
        proposed_pos = _next_positions(q_xy, move_dist, move_angle)
        if not _all_pairwise_safe(proposed_pos, min_dist):
            if _all_pairwise_safe(q_xy, min_dist):
                lo, hi = 0.0, 1.0
                for _ in range(binary_iters):
                    alpha = 0.5 * (lo + hi)
                    pos_mid = _next_positions(q_xy, move_dist * alpha, move_angle)
                    if _all_pairwise_safe(pos_mid, min_dist):
                        lo = alpha
                    else:
                        hi = alpha
                move_dist = move_dist * lo
            else:
                move_dist = np.zeros_like(move_dist, dtype=np.float32)

    out = dict(action)
    out["move_dist"] = move_dist.astype(np.float32)
    out["move_angle"] = move_angle.astype(np.float32)
    return out


# =============================================================================
# Pure MADDPG action decode and low-level allocation for critic input
# =============================================================================
def decode_offload_ratio_np(
    offload_raw: np.ndarray,
    min_ratio: float = 0.05,
    max_ratio: float = 1.0,
    temperature: float = 2.0,
) -> np.ndarray:
    x = np.asarray(offload_raw, dtype=np.float32) / float(temperature)
    ratio01 = 1.0 / (1.0 + np.exp(-x))
    ratio = min_ratio + (max_ratio - min_ratio) * ratio01
    return np.clip(ratio, min_ratio, max_ratio).astype(np.float32)


def decode_offload_ratio_torch(
    offload_raw: torch.Tensor,
    min_ratio: float = 0.05,
    max_ratio: float = 1.0,
    temperature: float = 2.0,
) -> torch.Tensor:
    ratio01 = torch.sigmoid(offload_raw / float(temperature))
    ratio = min_ratio + (max_ratio - min_ratio) * ratio01
    return torch.clamp(ratio, min=min_ratio, max=max_ratio)


def decode_pure_raw_action(
    state: Dict[str, Any],
    access_assoc: np.ndarray,
    raw_action: np.ndarray,
) -> Dict[str, np.ndarray]:
    M = int(state["M"])
    K = int(state["K"])
    neighbors = state["neighbors"]
    max_move = float(state["max_speed"]) * float(state["delta_t"])

    raw_action = np.asarray(raw_action, dtype=np.float32).reshape(-1)
    expected_dim = get_pure_actor_raw_action_dim(state)
    if raw_action.shape[0] != expected_dim:
        raise ValueError(f"Raw action dim mismatch: got {raw_action.shape[0]}, expected {expected_dim}")

    p = 0
    move_dist_raw = raw_action[p:p + M]
    p += M
    move_dist = 0.5 * (np.tanh(move_dist_raw) + 1.0) * max_move

    move_angle_raw = raw_action[p:p + M]
    p += M
    move_angle = np.pi * np.tanh(move_angle_raw)

    offload_raw = raw_action[p:p + K]
    p += K
    offload_ratio = decode_offload_ratio_np(
        offload_raw,
        min_ratio=_env_float("PURE_RATIO_MIN", 0.05),
        max_ratio=_env_float("PURE_RATIO_MAX", 1.0),
        temperature=_env_float("PURE_RATIO_TEMPERATURE", 2.0),
    )

    sched_score = raw_action[p:p + K * M].reshape(K, M)
    sched_beta = np.zeros((K, M, M), dtype=np.float32)
    for k in range(K):
        access_m = int(np.argmax(access_assoc[:, k]))
        legal_js = [access_m] + list(neighbors[access_m])
        legal_scores = [float(sched_score[k, j]) for j in legal_js]
        best_j = int(legal_js[int(np.argmax(legal_scores))])
        sched_beta[k, access_m, best_j] = 1.0

    return {
        "move_dist": move_dist.astype(np.float32),
        "move_angle": move_angle.astype(np.float32),
        "offload_ratio": offload_ratio.astype(np.float32),
        "sched_beta": sched_beta.astype(np.float32),
    }


def enrich_pure_action_to_full_action(
    state: Dict[str, Any],
    access_assoc: np.ndarray,
    action: Dict[str, Any],
    omega1: float,
    omega2: float,
) -> Dict[str, np.ndarray]:
    full_action = dict(action)
    offload_ratio = np.asarray(full_action["offload_ratio"], dtype=np.float32)
    sched_beta = np.asarray(full_action["sched_beta"], dtype=np.float32)

    bandwidth_alloc = solve_bandwidth_allocation(
        state=state,
        access_assoc=access_assoc,
        offload_ratio=offload_ratio,
    )
    cpu_alloc, _ = solve_cpu_allocation(
        state=state,
        access_assoc=access_assoc,
        sched_beta=sched_beta,
        offload_ratio=offload_ratio,
        omega1=omega1,
        omega2=omega2,
    )

    full_action["bandwidth_alloc"] = np.asarray(bandwidth_alloc, dtype=np.float32)
    full_action["cpu_alloc"] = np.asarray(cpu_alloc, dtype=np.float32)
    return full_action


# =============================================================================
# Critic action normalization
# =============================================================================
def _safe_np(x, dtype=np.float32) -> np.ndarray:
    return np.asarray(x, dtype=dtype)


def normalize_full_action_np(action_vec: np.ndarray, state: Dict[str, Any]) -> np.ndarray:
    M = int(state["M"])
    K = int(state["K"])
    out = np.asarray(action_vec, dtype=np.float32).copy().reshape(-1)
    p = 0
    max_move = max(float(state["max_speed"]) * float(state["delta_t"]), EPS)
    out[p:p + M] = out[p:p + M] / max_move
    p += M
    out[p:p + M] = out[p:p + M] / np.pi
    p += M
    p += K
    p += K * M * M
    B_max = _safe_np(state.get("B_max", np.ones(M, dtype=np.float32))).reshape(M, 1)
    B_scale = np.repeat(B_max, K, axis=1).reshape(-1)
    out[p:p + M * K] = out[p:p + M * K] / np.maximum(B_scale, EPS)
    p += M * K
    cpu_max = _safe_np(state.get("uav_cpu_max", np.ones(M, dtype=np.float32))).reshape(M, 1)
    cpu_scale = np.repeat(cpu_max, K, axis=1).reshape(-1)
    out[p:p + M * K] = out[p:p + M * K] / np.maximum(cpu_scale, EPS)
    return np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=-1.0).astype(np.float32)


def normalize_full_action_tensor(full_action: torch.Tensor, state: Dict[str, Any]) -> torch.Tensor:
    M = int(state["M"])
    K = int(state["K"])
    device = full_action.device
    dtype = full_action.dtype
    parts: List[torch.Tensor] = []
    p = 0
    max_move = max(float(state["max_speed"]) * float(state["delta_t"]), EPS)
    parts.append(full_action[:, p:p + M] / max_move)
    p += M
    parts.append(full_action[:, p:p + M] / np.pi)
    p += M
    parts.append(full_action[:, p:p + K])
    p += K
    parts.append(full_action[:, p:p + K * M * M])
    p += K * M * M
    B_max = torch.tensor(
        _safe_np(state.get("B_max", np.ones(M, dtype=np.float32))).reshape(M, 1),
        dtype=dtype,
        device=device,
    )
    B_scale = B_max.repeat(1, K).reshape(1, -1).clamp_min(EPS)
    parts.append(full_action[:, p:p + M * K] / B_scale)
    p += M * K
    cpu_max = torch.tensor(
        _safe_np(state.get("uav_cpu_max", np.ones(M, dtype=np.float32))).reshape(M, 1),
        dtype=dtype,
        device=device,
    )
    cpu_scale = cpu_max.repeat(1, K).reshape(1, -1).clamp_min(EPS)
    parts.append(full_action[:, p:p + M * K] / cpu_scale)
    return torch.cat(parts, dim=1)


def build_surrogate_full_action_tensor(
    state: Dict[str, Any],
    access_assoc: np.ndarray,
    raw_action_pred: torch.Tensor,
    bandwidth_alloc_np: np.ndarray,
    cpu_alloc_np: np.ndarray,
) -> torch.Tensor:
    device = raw_action_pred.device
    M = int(state["M"])
    K = int(state["K"])
    neighbors = state["neighbors"]
    max_move = max(float(state["max_speed"]) * float(state["delta_t"]), EPS)

    p = 0
    move_dist_raw = raw_action_pred[:, p:p + M]
    p += M
    move_angle_raw = raw_action_pred[:, p:p + M]
    p += M
    offload_raw = raw_action_pred[:, p:p + K]
    p += K
    sched_raw = raw_action_pred[:, p:p + K * M].reshape(-1, K, M)

    move_dist = 0.5 * (torch.tanh(move_dist_raw) + 1.0) * max_move
    move_angle = torch.tanh(move_angle_raw) * np.pi
    offload_ratio = decode_offload_ratio_torch(
        offload_raw,
        min_ratio=_env_float("PURE_RATIO_MIN", 0.05),
        max_ratio=_env_float("PURE_RATIO_MAX", 1.0),
        temperature=_env_float("PURE_RATIO_TEMPERATURE", 2.0),
    )

    batch_size = raw_action_pred.shape[0]
    sched_beta = torch.zeros((batch_size, K, M, M), dtype=torch.float32, device=device)
    for k in range(K):
        access_m = int(np.argmax(access_assoc[:, k]))
        legal_js = [access_m] + list(neighbors[access_m])
        legal_scores = sched_raw[:, k, legal_js]
        legal_prob = torch.softmax(legal_scores, dim=-1)
        for idx, j in enumerate(legal_js):
            sched_beta[:, k, access_m, j] = legal_prob[:, idx]

    bandwidth_alloc = torch.tensor(
        np.asarray(bandwidth_alloc_np, dtype=np.float32).reshape(1, -1),
        dtype=torch.float32,
        device=device,
    ).repeat(batch_size, 1)
    cpu_alloc = torch.tensor(
        np.asarray(cpu_alloc_np, dtype=np.float32).reshape(1, -1),
        dtype=torch.float32,
        device=device,
    ).repeat(batch_size, 1)

    return torch.cat(
        [
            move_dist,
            move_angle,
            offload_ratio,
            sched_beta.reshape(batch_size, -1),
            bandwidth_alloc,
            cpu_alloc,
        ],
        dim=1,
    )


# =============================================================================
# Evaluation
# =============================================================================
def _extract_violation(info: Dict[str, Any], key: str) -> float:
    report = info.get("report", {}) if isinstance(info, dict) else {}
    if key in report:
        return float(report.get(key, 0.0))
    if key in info:
        try:
            return float(info.get(key, 0.0))
        except Exception:
            return 0.0
    return 0.0


@torch.no_grad()
def evaluate_pure_policy_rollout(
    env: MultiUavMecEnv,
    actor: MLPActor,
    device: str,
    seed: int,
) -> Dict[str, float]:
    obs = env.reset(seed=seed)
    done = False
    actor.eval()

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

    while not done:
        state = obs["raw_state"]
        access_assoc = build_access_association(state)
        M = int(state["M"])
        K = int(state["K"])
        obs_vec = build_global_observation(state)
        obs_tensor = torch.tensor(obs_vec, dtype=torch.float32, device=device).unsqueeze(0)
        raw_action = actor(obs_tensor).squeeze(0).detach().cpu().numpy().astype(np.float32)

        action = decode_pure_raw_action(state, access_assoc, raw_action)
        action = apply_safe_mobility_projection(action, state)

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

        obs, reward, done, info = env.step(action)
        report = info.get("report", {})
        metrics = info.get("metrics", {})
        total_reward += float(reward)
        total_delay += float(metrics.get("delay_sys", 0.0))
        total_energy += float(metrics.get("energy_sys", 0.0))
        total_deadline_violation += float(report.get("deadline_violation", 0.0))
        feasible_count += int(bool(report.get("ok", False)))

        violation_sums["avg_ratio_violation"] += float(report.get("ratio_violation", 0.0))
        violation_sums["avg_assoc_violation"] += float(report.get("assoc_violation", 0.0))
        violation_sums["avg_schedule_violation"] += float(report.get("schedule_violation", 0.0))
        violation_sums["avg_candidate_violation"] += float(report.get("candidate_violation", 0.0))
        violation_sums["avg_bw_violation"] += float(report.get("bw_violation", 0.0))
        violation_sums["avg_cpu_violation"] += float(report.get("cpu_violation", 0.0))
        violation_sums["avg_rate_violation"] += float(report.get("rate_violation", 0.0))
        violation_sums["avg_nan_count"] += float(report.get("nan_count", 0.0))
        violation_sums["avg_move_violation"] += float(report.get("move_violation", 0.0))
        violation_sums["avg_boundary_violation"] += float(report.get("boundary_violation", 0.0))
        violation_sums["avg_collision_violation"] += float(report.get("collision_violation", 0.0))
        violation_sums["avg_battery_violation"] += float(report.get("battery_violation", 0.0))
        step_count += 1

    denom = max(step_count, 1)
    ratio_np = np.asarray(ratio_values, dtype=np.float32)
    sched_denom = max(sched_count, 1)
    out = {
        "episode_reward": float(total_reward),
        "system_cost": float(-total_reward),
        "avg_delay": float(total_delay / denom),
        "avg_energy": float(total_energy / denom),
        "avg_deadline_violation": float(total_deadline_violation / denom),
        "feasible_ratio": float(feasible_count / denom),
        "num_steps": int(step_count),
        "ratio_mean": float(np.mean(ratio_np)) if ratio_np.size else float("nan"),
        "ratio_std": float(np.std(ratio_np)) if ratio_np.size else float("nan"),
        "ratio_min": float(np.min(ratio_np)) if ratio_np.size else float("nan"),
        "ratio_max": float(np.max(ratio_np)) if ratio_np.size else float("nan"),
        "local_exec_ratio": float(local_exec_count / sched_denom),
        "neighbor_exec_ratio": float(neighbor_exec_count / sched_denom),
    }
    for k, v in violation_sums.items():
        out[k] = float(v / denom)
    return out


# =============================================================================
# Main
# =============================================================================
def make_env(seed: int) -> MultiUavMecEnv:
    return MultiUavMecEnv(
        M=_env_int("M", 3),
        K=_env_int("K", 16),
        episode_length=_env_int("EPISODE_LENGTH", 20),
        cpu_mode=_env_str("CPU_MODE", "kkt"),
        omega1=_env_float("OMEGA1", 50.0),
        omega2=_env_float("OMEGA2", 1.0),
        deadline_scale=_env_float("DEADLINE_SCALE", 2.5),
        task_local_cpu_min=_env_float("TASK_LOCAL_CPU_MIN", 2.0e3),
        task_local_cpu_max=_env_float("TASK_LOCAL_CPU_MAX", 5.0e3),
        uav_energy_min=_env_float("UAV_ENERGY_MIN", 2600.0),
        uav_energy_max=_env_float("UAV_ENERGY_MAX", 3800.0),
        seed=seed,
    )


def main() -> None:
    device = resolve_device()
    configure_torch_runtime(device)
    base_seed = _env_int("SEED", 72)
    torch.manual_seed(base_seed)
    np.random.seed(base_seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(base_seed)

    env = make_env(base_seed)
    obs = env.reset(seed=base_seed)
    state = obs["raw_state"]

    obs_dim = get_observation_dim(state)
    actor_raw_action_dim = get_pure_actor_raw_action_dim(state)
    critic_action_dim = get_full_action_dim(state)

    actor = MLPActor(obs_dim=obs_dim, action_dim=actor_raw_action_dim, hidden_dim=_env_int("HIDDEN_DIM", 256)).to(device)
    target_actor = copy.deepcopy(actor).to(device)
    critic = MLPCritic(obs_dim=obs_dim, action_dim=critic_action_dim, hidden_dim=_env_int("HIDDEN_DIM", 256)).to(device)
    target_critic = copy.deepcopy(critic).to(device)

    actor_opt = optim.Adam(actor.parameters(), lr=_env_float("PURE_ACTOR_LR", 1e-4))
    critic_opt = optim.Adam(critic.parameters(), lr=_env_float("PURE_CRITIC_LR", 5e-5))
    mse_loss = nn.MSELoss()

    buffer = FullReplayBuffer(obs_dim=obs_dim, action_dim=critic_action_dim, capacity=_env_int("BUFFER_CAPACITY", 30000))

    gamma = _env_float("GAMMA", 0.95)
    tau = _env_float("TAU", 0.005)
    batch_size = _env_int("BATCH_SIZE", 128)
    learning_starts = _env_int("LEARNING_STARTS", 1000)
    num_episodes = _env_int("NUM_EPISODES", 700)
    eval_every = _env_int("EVAL_EVERY", 5)
    eval_seed = _env_int("EVAL_SEED", 999)
    reward_scale = _env_float("REWARD_SCALE", 1e-4)
    policy_delay = _env_int("POLICY_DELAY", 2)
    actor_policy_coef = _env_float("ACTOR_POLICY_COEF", 0.001)
    actor_l2_coef = _env_float("ACTOR_L2_COEF", 1e-5)
    grad_clip_norm = _env_float("GRAD_CLIP_NORM", 1.0)

    # Mild anti-collapse regularization for the pure baseline. Keep it modest; this is a baseline.
    offload_floor_target = _env_float("PURE_OFFLOAD_FLOOR_TARGET", 0.08)
    offload_mean_target = _env_float("PURE_OFFLOAD_MEAN_TARGET", 0.12)
    offload_std_target = _env_float("PURE_OFFLOAD_STD_TARGET", 0.02)
    offload_reg_coef = _env_float("PURE_OFFLOAD_REG_COEF", 0.05)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    default_run_name = f"pure_maddpg_main_d25_seed{base_seed}_ep{num_episodes}_{timestamp}"
    run_name = _env_str("RUN_NAME", default_run_name)
    ckpt_prefix = _env_str("CKPT_PREFIX", run_name)
    result_dir = Path("results") / "convergence_training" / run_name
    ckpt_dir = Path(_env_str("CKPT_DIR", "checkpoints"))
    result_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_log_path = result_dir / "train_log.csv"
    eval_log_path = result_dir / "eval_log.csv"
    config_path = result_dir / "config.json"

    run_config = {
        "stage": "pure_maddpg_baseline",
        "run_name": run_name,
        "ckpt_prefix": ckpt_prefix,
        "seed": base_seed,
        "eval_seed": eval_seed,
        "device": device,
        "M": env.M,
        "K": env.K,
        "episode_length": env.episode_length,
        "cpu_mode": env.cpu_mode,
        "omega1": env.omega1,
        "omega2": env.omega2,
        "deadline_scale": env.deadline_scale,
        "task_local_cpu_min": env.task_local_cpu_min,
        "task_local_cpu_max": env.task_local_cpu_max,
        "uav_energy_min": env.uav_energy_min,
        "uav_energy_max": env.uav_energy_max,
        "num_episodes": num_episodes,
        "batch_size": batch_size,
        "learning_starts": learning_starts,
        "gamma": gamma,
        "tau": tau,
        "reward_scale": reward_scale,
        "policy_delay": policy_delay,
        "actor_policy_coef": actor_policy_coef,
        "offload_reg_coef": offload_reg_coef,
        "safe_mobility_projection": True,
        "model_refinement": False,
        "obs_dim": obs_dim,
        "actor_raw_action_dim": actor_raw_action_dim,
        "critic_action_dim": critic_action_dim,
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)

    with open(train_log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "episode",
            "episode_reward",
            "avg_actor_policy_loss",
            "avg_actor_l2_loss",
            "avg_offload_reg_loss",
            "avg_total_actor_loss",
            "avg_critic_loss",
            "avg_q_value",
            "avg_target_q_value",
            "avg_td_abs_error",
            "steps",
            "buffer_size",
            "critic_updates",
            "actor_updates",
        ])

    with open(eval_log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "eval_episode",
            "episode_reward",
            "system_cost",
            "avg_delay",
            "avg_energy",
            "avg_deadline_violation",
            "feasible_ratio",
            "num_steps",
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
            "ratio_min",
            "ratio_max",
            "local_exec_ratio",
            "neighbor_exec_ratio",
        ])

    best_actor_state = copy.deepcopy(actor.state_dict())
    best_eval_reward = -float("inf")
    global_update_step = 0

    print("Pure MADDPG SCI baseline config:")
    print(json.dumps(run_config, indent=2))
    print(f"train_log_path={train_log_path}")
    print(f"eval_log_path={eval_log_path}")

    for ep in range(num_episodes):
        obs = env.reset(seed=base_seed + ep)
        done = False

        episode_reward = 0.0
        episode_actor_policy_loss = 0.0
        episode_actor_l2_loss = 0.0
        episode_offload_reg_loss = 0.0
        episode_total_actor_loss = 0.0
        episode_critic_loss = 0.0
        episode_q_value = 0.0
        episode_target_q_value = 0.0
        episode_td_abs_error = 0.0
        critic_update_count = 0
        actor_update_count = 0
        step_count = 0

        while not done:
            state = obs["raw_state"]
            access_assoc = build_access_association(state)
            obs_vec = build_global_observation(state)
            obs_tensor = torch.tensor(obs_vec, dtype=torch.float32, device=device).unsqueeze(0)

            # Exploration noise in raw action space.
            with torch.no_grad():
                raw_action_np = actor(obs_tensor).squeeze(0).cpu().numpy().astype(np.float32)
            M_cur = int(state["M"])
            K_cur = int(state["K"])
            raw_action_np = raw_action_np.copy()
            raw_action_np[: 2 * M_cur] += np.random.normal(0.0, _env_float("MOVE_NOISE_STD", 0.12), size=(2 * M_cur,)).astype(np.float32)
            raw_action_np[2 * M_cur: 2 * M_cur + K_cur] += np.random.normal(0.0, _env_float("RATIO_NOISE_STD", 0.20), size=(K_cur,)).astype(np.float32)
            raw_action_np[2 * M_cur + K_cur:] += np.random.normal(0.0, _env_float("SCHED_NOISE_STD", 0.10), size=(K_cur * M_cur,)).astype(np.float32)

            action_high = decode_pure_raw_action(state, access_assoc, raw_action_np)
            action_high = apply_safe_mobility_projection(action_high, state)
            action_full = enrich_pure_action_to_full_action(
                state=state,
                access_assoc=access_assoc,
                action=action_high,
                omega1=env.omega1,
                omega2=env.omega2,
            )
            flat_action_norm = normalize_full_action_np(flatten_full_action(action_full), state)

            next_obs, reward, done, info = env.step(action_high)
            next_obs_vec = build_global_observation(next_obs["raw_state"])
            scaled_reward = float(reward) * reward_scale

            if not done:
                next_state = next_obs["raw_state"]
                next_access_assoc = build_access_association(next_state)
                next_obs_tensor = torch.tensor(next_obs_vec, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    next_raw_action_np = target_actor(next_obs_tensor).squeeze(0).cpu().numpy().astype(np.float32)
                next_action_high = decode_pure_raw_action(next_state, next_access_assoc, next_raw_action_np)
                next_action_high = apply_safe_mobility_projection(next_action_high, next_state)
                next_action_full = enrich_pure_action_to_full_action(
                    state=next_state,
                    access_assoc=next_access_assoc,
                    action=next_action_high,
                    omega1=env.omega1,
                    omega2=env.omega2,
                )
                next_flat_action_norm = normalize_full_action_np(flatten_full_action(next_action_full), next_state)
            else:
                next_flat_action_norm = np.zeros((critic_action_dim,), dtype=np.float32)

            buffer.add(obs_vec, flat_action_norm, scaled_reward, next_obs_vec, next_flat_action_norm, float(done))

            # Critic update
            if len(buffer) >= max(batch_size, learning_starts):
                batch = buffer.sample(batch_size)
                obs_b = torch.tensor(batch["obs"], dtype=torch.float32, device=device)
                action_b = torch.tensor(batch["action"], dtype=torch.float32, device=device)
                reward_b = torch.tensor(batch["reward"], dtype=torch.float32, device=device)
                next_obs_b = torch.tensor(batch["next_obs"], dtype=torch.float32, device=device)
                next_action_b = torch.tensor(batch["next_action"], dtype=torch.float32, device=device)
                done_b = torch.tensor(batch["done"], dtype=torch.float32, device=device)

                with torch.no_grad():
                    target_q = target_critic(next_obs_b, next_action_b)
                    y = reward_b + gamma * (1.0 - done_b) * target_q

                q_val = critic(obs_b, action_b)
                critic_loss = mse_loss(q_val, y)
                critic_opt.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=grad_clip_norm)
                critic_opt.step()

                td_abs_error = torch.abs(q_val.detach() - y.detach()).mean()
                episode_critic_loss += float(critic_loss.item())
                episode_q_value += float(q_val.detach().mean().item())
                episode_target_q_value += float(y.detach().mean().item())
                episode_td_abs_error += float(td_abs_error.item())
                critic_update_count += 1
                global_update_step += 1

                # Delayed actor update.
                if global_update_step % max(policy_delay, 1) == 0:
                    raw_pred = actor(obs_tensor)
                    surrogate_full = build_surrogate_full_action_tensor(
                        state=state,
                        access_assoc=access_assoc,
                        raw_action_pred=raw_pred,
                        bandwidth_alloc_np=action_full["bandwidth_alloc"],
                        cpu_alloc_np=action_full["cpu_alloc"],
                    )
                    surrogate_full_norm = normalize_full_action_tensor(surrogate_full, state)

                    # Freeze critic during actor update to avoid accumulating unnecessary gradients.
                    for p_critic in critic.parameters():
                        p_critic.requires_grad_(False)
                    actor_policy_loss = -critic(obs_tensor, surrogate_full_norm).mean()
                    for p_critic in critic.parameters():
                        p_critic.requires_grad_(True)

                    actor_l2_loss = (raw_pred ** 2).mean()
                    offload_raw_pred = raw_pred[:, 2 * M_cur: 2 * M_cur + K_cur]
                    offload_ratio_pred = decode_offload_ratio_torch(
                        offload_raw_pred,
                        min_ratio=_env_float("PURE_RATIO_MIN", 0.05),
                        max_ratio=_env_float("PURE_RATIO_MAX", 1.0),
                        temperature=_env_float("PURE_RATIO_TEMPERATURE", 2.0),
                    )
                    offload_floor_loss = torch.relu(offload_floor_target - offload_ratio_pred).mean()
                    offload_mean_loss = torch.relu(offload_mean_target - offload_ratio_pred.mean())
                    offload_std = torch.std(offload_ratio_pred.reshape(-1), unbiased=False)
                    offload_div_loss = torch.relu(offload_std_target - offload_std)
                    offload_reg_loss = 8.0 * offload_floor_loss + 6.0 * offload_mean_loss + 5.0 * offload_div_loss

                    total_actor_loss = (
                        actor_policy_coef * actor_policy_loss
                        + actor_l2_coef * actor_l2_loss
                        + offload_reg_coef * offload_reg_loss
                    )

                    actor_opt.zero_grad()
                    total_actor_loss.backward()
                    torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=grad_clip_norm)
                    actor_opt.step()

                    episode_actor_policy_loss += float(actor_policy_loss.item())
                    episode_actor_l2_loss += float(actor_l2_loss.item())
                    episode_offload_reg_loss += float(offload_reg_loss.item())
                    episode_total_actor_loss += float(total_actor_loss.item())
                    actor_update_count += 1

                soft_update(target_actor, actor, tau=tau)
                soft_update(target_critic, critic, tau=tau)

            obs = next_obs
            episode_reward += float(reward)
            step_count += 1

        train_row = [
            ep,
            episode_reward,
            episode_actor_policy_loss / max(actor_update_count, 1),
            episode_actor_l2_loss / max(actor_update_count, 1),
            episode_offload_reg_loss / max(actor_update_count, 1),
            episode_total_actor_loss / max(actor_update_count, 1),
            episode_critic_loss / max(critic_update_count, 1) if critic_update_count else np.nan,
            episode_q_value / max(critic_update_count, 1) if critic_update_count else np.nan,
            episode_target_q_value / max(critic_update_count, 1) if critic_update_count else np.nan,
            episode_td_abs_error / max(critic_update_count, 1) if critic_update_count else np.nan,
            step_count,
            len(buffer),
            critic_update_count,
            actor_update_count,
        ]
        with open(train_log_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(train_row)

        if (ep + 1) % eval_every == 0 or ep == 0:
            eval_env = make_env(eval_seed)
            eval_result = evaluate_pure_policy_rollout(eval_env, actor, device, seed=eval_seed)
            print("\n==============================")
            print(f"Pure MADDPG Eval @ episode {ep}")
            print("==============================")
            for key in [
                "episode_reward", "system_cost", "avg_delay", "avg_energy", "avg_deadline_violation",
                "feasible_ratio", "ratio_mean", "ratio_std", "local_exec_ratio", "neighbor_exec_ratio",
            ]:
                print(f"{key}: {eval_result.get(key)}")

            eval_row = [
                ep,
                eval_result["episode_reward"],
                eval_result["system_cost"],
                eval_result["avg_delay"],
                eval_result["avg_energy"],
                eval_result["avg_deadline_violation"],
                eval_result["feasible_ratio"],
                eval_result["num_steps"],
                eval_result["avg_ratio_violation"],
                eval_result["avg_assoc_violation"],
                eval_result["avg_schedule_violation"],
                eval_result["avg_candidate_violation"],
                eval_result["avg_bw_violation"],
                eval_result["avg_cpu_violation"],
                eval_result["avg_rate_violation"],
                eval_result["avg_nan_count"],
                eval_result["avg_move_violation"],
                eval_result["avg_boundary_violation"],
                eval_result["avg_collision_violation"],
                eval_result["avg_battery_violation"],
                eval_result["ratio_mean"],
                eval_result["ratio_std"],
                eval_result["ratio_min"],
                eval_result["ratio_max"],
                eval_result["local_exec_ratio"],
                eval_result["neighbor_exec_ratio"],
            ]
            with open(eval_log_path, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(eval_row)

            if eval_result["episode_reward"] > best_eval_reward:
                best_eval_reward = float(eval_result["episode_reward"])
                best_actor_state = copy.deepcopy(actor.state_dict())
                torch.save(best_actor_state, ckpt_dir / f"{ckpt_prefix}_best_actor.pth")
                torch.save(critic.state_dict(), ckpt_dir / f"{ckpt_prefix}_critic.pth")
                print(f"[BEST] reward={best_eval_reward:.6f}; saved {ckpt_prefix}_best_actor.pth")

        print(
            f"Episode {ep:04d} | reward={episode_reward:.3f} | "
            f"critic_loss={train_row[6]} | actor_updates={actor_update_count} | buffer={len(buffer)}"
        )

    # Save final actor/critic as well.
    torch.save(actor.state_dict(), ckpt_dir / f"{ckpt_prefix}_final_actor.pth")
    torch.save(critic.state_dict(), ckpt_dir / f"{ckpt_prefix}_final_critic.pth")
    if not (ckpt_dir / f"{ckpt_prefix}_best_actor.pth").exists():
        torch.save(best_actor_state, ckpt_dir / f"{ckpt_prefix}_best_actor.pth")
        torch.save(critic.state_dict(), ckpt_dir / f"{ckpt_prefix}_critic.pth")

    print("\nPure MADDPG SCI baseline training finished successfully.")
    print("Saved:")
    print(f"  {ckpt_dir / (ckpt_prefix + '_best_actor.pth')}")
    print(f"  {ckpt_dir / (ckpt_prefix + '_critic.pth')}")
    print(f"  {ckpt_dir / (ckpt_prefix + '_final_actor.pth')}")
    print(f"  {ckpt_dir / (ckpt_prefix + '_final_critic.pth')}")
    print("Logs:")
    print(f"  {train_log_path}")
    print(f"  {eval_log_path}")


if __name__ == "__main__":
    main()
