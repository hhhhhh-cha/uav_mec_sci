# 加载 proposed_full_stage1_best_*.pth
# 组装完整 ProposedPolicy
# 保留 encoder / fusion / ratio head / actor
# Stage-2 收敛训练版 v4（trust-region / warm-start anchored）：
#   1) critic action 输入归一化
#   2) Stage-2 critic 从零初始化，不加载 Stage-1 critic
#   3) learning_starts + batch_size=128
#   4) policy_delay=2 延迟 actor 更新，并且 actor 更新时冻结 critic 参数
#   5) 使用 Stage-1 reference policy 作为 trust-region anchor，防止 Stage-2 actor 把 warm-start 好策略带坏
#   6) 加入 local-dominance margin loss，抑制 collaborative scheduling 向本地执行塌缩
#   7) 加入 rollback-on-degradation，evaluation 退化时自动回滚到历史 best policy

import copy
import csv
import os
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from env.mec_env import MultiUavMecEnv
from env.association import build_access_association
from model.mlp_actor import MLPActor
from model.mlp_critic import MLPCritic
from model.proposed_obs_builder import build_global_observation, get_observation_dim
from policy.proposed_placeholder_policy import generate_proposed_placeholder_action
from policy.proposed_policy import build_default_proposed_policy

from contextlib import contextmanager

from train.train_proposed_full_stage1_converge import (
    EPS,
    FullReplayBuffer,
    build_teacher_raw_target,
    evaluate_full_policy_rollout,
    forward_ratio_branch,
    get_actor_raw_action_dim,
    get_full_action_dim,
    flatten_full_action,
    soft_update,
    soft_update_policy,
)



@contextmanager
def policy_eval_mode(policy):
    modules = [
        getattr(policy, "actor_net", None),
        getattr(policy, "encoder", None),
        getattr(policy, "fusion_net", None),
        getattr(policy, "ratio_head", None),
    ]
    modules = [m for m in modules if m is not None]
    old_modes = [m.training for m in modules]

    for m in modules:
        m.eval()

    try:
        yield
    finally:
        for m, old in zip(modules, old_modes):
            m.train(old)




def load_policy_weights_inplace(dst_policy, src_policy):
    """Load actor/encoder/fusion/ratio weights without replacing module objects.

    This keeps optimizer parameter references valid after rollback.
    """
    for name in ["actor_net", "encoder", "fusion_net", "ratio_head"]:
        dst_module = getattr(dst_policy, name, None)
        src_module = getattr(src_policy, name, None)
        if dst_module is not None and src_module is not None:
            dst_module.load_state_dict(src_module.state_dict())

# =========================================================
# Checkpoint loading
# =========================================================
def load_if_exists(module: torch.nn.Module, path: str, name: str):
    if module is None:
        return
    if os.path.exists(path):
        module.load_state_dict(torch.load(path, map_location="cpu"))
        print(f"Loaded {name} from: {path}")
    else:
        print(f"WARNING: {name} checkpoint not found: {path}")


# =========================================================
# Offloading-ratio regularization
# =========================================================
def compute_ratio_regularization_loss(
    ratio_pred: torch.Tensor,
    ratio_floor_target: float = 0.15,
    ratio_mean_target: float = 0.20,
    ratio_std_target: float = 0.06,
):
    """
    Regularize the Proposed ratio head to avoid overly conservative offloading.

    Terms:
        1) floor loss:     discourage per-task ratios below ratio_floor_target
        2) mean loss:      discourage batch/task mean below ratio_mean_target
        3) diversity loss: discourage collapsed ratios with too-small std
    """
    ratio_valid = ratio_pred.reshape(-1)

    floor_target = torch.tensor(
        ratio_floor_target,
        dtype=ratio_valid.dtype,
        device=ratio_valid.device,
    )
    mean_target = torch.tensor(
        ratio_mean_target,
        dtype=ratio_valid.dtype,
        device=ratio_valid.device,
    )
    std_target = torch.tensor(
        ratio_std_target,
        dtype=ratio_valid.dtype,
        device=ratio_valid.device,
    )

    ratio_floor_loss = torch.relu(floor_target - ratio_valid).mean()
    ratio_mean_loss = torch.relu(mean_target - ratio_valid.mean())

    # Use unbiased=False to avoid NaN/instability when the number of samples is small.
    ratio_std = torch.std(ratio_valid, unbiased=False)
    ratio_diversity_loss = torch.relu(std_target - ratio_std)

    ratio_reg_loss = (
        8.0 * ratio_floor_loss
        + 6.0 * ratio_mean_loss
        + 5.0 * ratio_diversity_loss
    )

    return ratio_reg_loss, ratio_floor_loss, ratio_mean_loss, ratio_diversity_loss


# =========================================================
# Action normalization for critic
# =========================================================
def _safe_np(x, dtype=np.float32):
    return np.asarray(x, dtype=dtype)


def normalize_full_action_np(action_vec: np.ndarray, state: Dict[str, Any]) -> np.ndarray:
    """
    Normalize full action before feeding it into critic.

    Full action layout:
        [ move_dist(M),
          move_angle(M),
          offload_ratio(K),
          sched_beta(K*M*M),
          bandwidth_alloc(M*K),
          cpu_alloc(M*K) ]

    Rationale:
        move_dist       : [0, max_speed * delta_t] -> [0, 1]
        move_angle      : [-pi, pi] -> [-1, 1]
        offload_ratio   : already [0, 1]
        sched_beta      : already [0, 1]
        bandwidth_alloc : divide by B_max[m]
        cpu_alloc       : divide by uav_cpu_max[m]
    """
    M = int(state["M"])
    K = int(state["K"])

    out = np.asarray(action_vec, dtype=np.float32).copy().reshape(-1)
    max_move = max(float(state["max_speed"]) * float(state["delta_t"]), EPS)

    p = 0
    out[p:p + M] = out[p:p + M] / max_move
    p += M

    out[p:p + M] = out[p:p + M] / np.pi
    p += M

    # offload_ratio: already [0, 1]
    p += K

    # sched_beta: already [0, 1]
    p += K * M * M

    # bandwidth_alloc: [M, K], scale by per-UAV B_max
    B_max = _safe_np(state.get("B_max", np.ones(M, dtype=np.float32))).reshape(M, 1)
    B_scale = np.repeat(B_max, K, axis=1).reshape(-1)
    out[p:p + M * K] = out[p:p + M * K] / np.maximum(B_scale, EPS)
    p += M * K

    # cpu_alloc: [M, K], scale by per-UAV uav_cpu_max
    cpu_max = _safe_np(state.get("uav_cpu_max", np.ones(M, dtype=np.float32))).reshape(M, 1)
    cpu_scale = np.repeat(cpu_max, K, axis=1).reshape(-1)
    out[p:p + M * K] = out[p:p + M * K] / np.maximum(cpu_scale, EPS)

    # Avoid rare numerical pollution.
    out = np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=-1.0)
    return out.astype(np.float32)


def normalize_full_action_tensor(full_action: torch.Tensor, state: Dict[str, Any]) -> torch.Tensor:
    """Torch version of normalize_full_action_np, preserving actor gradients."""
    M = int(state["M"])
    K = int(state["K"])
    device = full_action.device
    dtype = full_action.dtype

    parts = []
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


# =========================================================
# Differentiable surrogate full-action builder
# Used only for actor policy-gradient-like update
# =========================================================
def build_surrogate_full_action_tensor(
    state: Dict[str, Any],
    access_assoc: np.ndarray,
    raw_action_pred: torch.Tensor,
    ratio_pred: torch.Tensor,
    bandwidth_alloc_np: np.ndarray,
    cpu_alloc_np: np.ndarray,
) -> torch.Tensor:
    """
    Build a differentiable surrogate full action for critic-guided actor update.
    The output is still in physical scale. Normalize it with
    normalize_full_action_tensor(...) before sending it into critic.
    """
    device = raw_action_pred.device
    M = int(state["M"])
    K = int(state["K"])
    neighbors = state["neighbors"]

    max_speed = float(state["max_speed"])
    delta_t = float(state["delta_t"])
    max_move = max(max_speed * delta_t, EPS)

    p = 0
    move_dist_raw = raw_action_pred[:, p:p + M]
    p += M

    move_angle_raw = raw_action_pred[:, p:p + M]
    p += M

    _offload_raw_unused = raw_action_pred[:, p:p + K]
    p += K

    sched_raw = raw_action_pred[:, p:p + K * M].reshape(1, K, M)

    move_dist = 0.5 * (torch.tanh(move_dist_raw) + 1.0) * max_move
    move_angle = torch.tanh(move_angle_raw) * np.pi
    offload_ratio = ratio_pred.unsqueeze(0)

    sched_beta = torch.zeros((1, K, M, M), dtype=torch.float32, device=device)

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
    )
    cpu_alloc = torch.tensor(
        np.asarray(cpu_alloc_np, dtype=np.float32).reshape(1, -1),
        dtype=torch.float32,
        device=device,
    )

    full_action = torch.cat(
        [
            move_dist,
            move_angle,
            offload_ratio,
            sched_beta.reshape(1, -1),
            bandwidth_alloc,
            cpu_alloc,
        ],
        dim=1,
    )
    return full_action




def compute_neighbor_collaboration_loss(
    raw_action_pred: torch.Tensor,
    state: Dict[str, Any],
    access_assoc: np.ndarray,
    neighbor_prob_target: float = 0.22,
) -> torch.Tensor:
    """
    Differentiable anti-collapse regularization for collaborative scheduling.

    Motivation:
        In the d25 strict logs, local_exec_ratio increases from about 0.74 to
        about 0.85, while system_cost and delay worsen. This term prevents the
        scheduling branch from collapsing to pure local execution when legal
        neighboring UAV candidates exist.

    It only acts on tasks whose access UAV has at least one legal neighbor.
    It does NOT force an illegal neighbor choice.
    """
    device = raw_action_pred.device
    dtype = raw_action_pred.dtype

    M = int(state["M"])
    K = int(state["K"])
    neighbors = state.get("neighbors", [[] for _ in range(M)])

    p = M + M + K
    sched_raw = raw_action_pred[:, p:p + K * M].reshape(-1, K, M)

    losses = []
    for k in range(K):
        access_m = int(np.argmax(access_assoc[:, k]))
        legal_js = [access_m] + list(neighbors[access_m])
        legal_js = [int(j) for j in legal_js if 0 <= int(j) < M]
        neighbor_js = [j for j in legal_js if j != access_m]

        if len(neighbor_js) == 0:
            continue

        legal_scores = sched_raw[:, k, legal_js]
        legal_prob = torch.softmax(legal_scores, dim=-1)

        neighbor_indices = [legal_js.index(j) for j in neighbor_js]
        neighbor_prob = legal_prob[:, neighbor_indices].sum(dim=-1)

        target = torch.tensor(
            float(neighbor_prob_target),
            dtype=dtype,
            device=device,
        )
        losses.append(torch.relu(target - neighbor_prob).mean())

    if len(losses) == 0:
        return torch.zeros((), dtype=dtype, device=device)

    return torch.stack(losses).mean()




def compute_local_dominance_margin_loss(
    raw_action_pred: torch.Tensor,
    state: Dict[str, Any],
    access_assoc: np.ndarray,
    local_margin: float = 0.35,
) -> torch.Tensor:
    """
    Penalize overly dominant local-execution logits when neighbor candidates exist.

    This is different from directly forcing neighbor execution. It only discourages
    the local logit from being much larger than the best legal neighbor logit.
    Therefore, local execution can still be selected when it is genuinely preferred,
    but the scheduler is prevented from collapsing into a local-only policy.
    """
    device = raw_action_pred.device
    dtype = raw_action_pred.dtype

    M = int(state["M"])
    K = int(state["K"])
    neighbors = state.get("neighbors", [[] for _ in range(M)])

    p = M + M + K
    sched_raw = raw_action_pred[:, p:p + K * M].reshape(-1, K, M)

    losses = []
    margin = torch.tensor(float(local_margin), dtype=dtype, device=device)
    for k in range(K):
        access_m = int(np.argmax(access_assoc[:, k]))
        legal_js = [access_m] + list(neighbors[access_m])
        legal_js = [int(j) for j in legal_js if 0 <= int(j) < M]
        neighbor_js = [j for j in legal_js if j != access_m]
        if len(neighbor_js) == 0:
            continue

        local_score = sched_raw[:, k, access_m]
        neighbor_scores = sched_raw[:, k, neighbor_js]
        best_neighbor_score = torch.max(neighbor_scores, dim=-1).values
        losses.append(torch.relu(local_score - best_neighbor_score - margin).mean())

    if len(losses) == 0:
        return torch.zeros((), dtype=dtype, device=device)
    return torch.stack(losses).mean()

def linear_schedule_value(start: float, end: float, progress: float) -> float:
    progress = float(np.clip(progress, 0.0, 1.0))
    return float(start + progress * (end - start))


# =========================================================
# Safe mobility projection
# =========================================================
def _get_uav_positions_from_state(state: Dict[str, Any], M: int) -> np.ndarray:
    """
    Robustly extract UAV 2D positions from the raw state.
    Returns shape [M, 2]. If the key is unavailable, returns None.
    """
    candidate_keys = [
        "uav_pos",
        "uav_positions",
        "uav_xy",
        "q_uav",
        "q",
        "positions",
    ]

    for key in candidate_keys:
        if key in state:
            arr = np.asarray(state[key], dtype=np.float32)
            if arr.ndim == 2 and arr.shape[0] >= M and arr.shape[1] >= 2:
                return arr[:M, :2].copy()

    # Some codebases store x/y separately.
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
    collision_margin: float = 0.0,
    binary_iters: int = 24,
) -> Dict[str, Any]:
    """
    Project the mobility action into a physically safer flight region.

    This is a lightweight safety shield applied after actor inference and before
    env.step(...). It preserves the actor's move whenever feasible, but shrinks
    the movement distance if the proposed next position would violate boundary
    or inter-UAV safety distance.

    It does NOT alter offloading, scheduling, bandwidth, or CPU allocation.
    """
    if "move_dist" not in action or "move_angle" not in action:
        return action

    M = int(state.get("M", len(np.asarray(action["move_dist"]).reshape(-1))))
    move_dist = np.asarray(action["move_dist"], dtype=np.float32).reshape(-1).copy()
    move_angle = np.asarray(action["move_angle"], dtype=np.float32).reshape(-1).copy()

    if move_dist.size < M or move_angle.size < M:
        return action

    q_xy = _get_uav_positions_from_state(state, M)
    if q_xy is None:
        # If positions cannot be found, at least enforce the motion budget.
        max_move = max(float(state.get("max_speed", 0.0)) * float(state.get("delta_t", 1.0)), EPS)
        safe_action = dict(action)
        safe_action["move_dist"] = np.clip(move_dist[:M], 0.0, max_move).astype(np.float32)
        safe_action["move_angle"] = move_angle[:M].astype(np.float32)
        return safe_action

    area_size = float(state.get("area_size", 100.0))
    max_speed = float(state.get("max_speed", 15.0))
    delta_t = float(state.get("delta_t", 1.0))
    max_move = max(max_speed * delta_t, EPS)
    min_dist = float(state.get("min_uav_distance", 0.0)) + float(collision_margin)

    # Basic numeric cleanup and motion-budget clipping.
    move_dist = np.nan_to_num(move_dist[:M], nan=0.0, posinf=max_move, neginf=0.0)
    move_angle = np.nan_to_num(move_angle[:M], nan=0.0, posinf=np.pi, neginf=-np.pi)
    move_dist = np.clip(move_dist, 0.0, max_move).astype(np.float32)
    move_angle = np.clip(move_angle, -np.pi, np.pi).astype(np.float32)

    low = float(safety_margin)
    high = float(area_size - safety_margin)

    # Boundary projection: shrink each UAV's distance along its actor-proposed direction.
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

    # Collision projection: if the set of next positions is unsafe, shrink all moves together.
    if min_dist > 0.0:
        proposed_pos = _next_positions(q_xy, move_dist, move_angle)
        if not _all_pairwise_safe(proposed_pos, min_dist):
            # If current positions are already unsafe, shrinking cannot fully fix it.
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

    safe_action = dict(action)
    safe_action["move_dist"] = move_dist.astype(np.float32)
    safe_action["move_angle"] = move_angle.astype(np.float32)
    return safe_action


class SafeMobilityPolicyWrapper:
    """
    Wrapper used only for evaluation, so evaluate_full_policy_rollout(...)
    computes reward / delay / feasible_ratio under exactly the same safe
    mobility projection used in training execution.
    """
    def __init__(self, policy, safety_margin: float = 1.0, collision_margin: float = 0.0):
        self.policy = policy
        self.safety_margin = safety_margin
        self.collision_margin = collision_margin

    def act(self, state, access_assoc, deterministic=True, return_aux=False):
        action = self.policy.act(
            state=state,
            access_assoc=access_assoc,
            deterministic=deterministic,
            return_aux=return_aux,
        )
        if return_aux:
            # If a future policy returns (action, aux), project only the action part.
            action_part, aux = action
            action_part = apply_safe_mobility_projection(
                action_part,
                state,
                safety_margin=self.safety_margin,
                collision_margin=self.collision_margin,
            )
            return action_part, aux

        return apply_safe_mobility_projection(
            action,
            state,
            safety_margin=self.safety_margin,
            collision_margin=self.collision_margin,
        )


# =========================================================
# Evaluation with extra feasibility diagnostics
# =========================================================
def _to_float_or_none(x):
    """Convert scalar-like values to float. Return None if conversion is unsafe."""
    if x is None:
        return None
    if isinstance(x, (bool, np.bool_)):
        return float(x)
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    try:
        arr = np.asarray(x)
        if arr.size == 1:
            return float(arr.reshape(-1)[0])
    except Exception:
        return None
    return None


def _recursive_find_numeric(info: Any, aliases) -> float:
    """
    Robustly search an info dict for a violation scalar.
    This supports both top-level keys and nested reports, e.g.
    info["feasibility_report"]["collision_violation"].
    """
    if info is None:
        return 0.0

    if isinstance(info, dict):
        for key in aliases:
            if key in info:
                value = _to_float_or_none(info[key])
                if value is not None:
                    return value
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


@torch.no_grad()
def evaluate_full_policy_rollout_with_extra_constraints(env, policy, seed: int):
    """
    Keep the original Stage-1 evaluation output, and additionally log hidden
    feasibility terms that are often not included in the old eval_log.csv.

    The reward/delay/feasible_ratio and the extra diagnostics are both computed
    under the same safe mobility projection.
    """
    safe_policy = SafeMobilityPolicyWrapper(policy, safety_margin=1.0, collision_margin=1e-5)
    result = evaluate_full_policy_rollout(env=env, policy=safe_policy, seed=seed)

    obs = env.reset(seed=seed)
    done = False
    n = 0

    extra_sum = {
        "avg_move_violation": 0.0,
        "avg_boundary_violation": 0.0,
        "avg_collision_violation": 0.0,
        "avg_battery_violation": 0.0,
    }

    alias_map = {
        "avg_move_violation": [
            "move_violation", "motion_violation", "movement_violation",
            "avg_move_violation", "avg_motion_violation",
        ],
        "avg_boundary_violation": [
            "boundary_violation", "out_of_boundary", "out_of_bounds",
            "avg_boundary_violation",
        ],
        "avg_collision_violation": [
            "collision_violation", "collision", "safe_distance_violation",
            "distance_violation", "avg_collision_violation",
        ],
        "avg_battery_violation": [
            "battery_violation", "energy_violation", "battery_negative",
            "avg_battery_violation", "avg_energy_violation",
        ],
    }

    while not done:
        state = obs["raw_state"]
        access_assoc = build_access_association(state)
        action = safe_policy.act(
            state=state,
            access_assoc=access_assoc,
            deterministic=True,
            return_aux=False,
        )
        obs, _, done, info = env.step(action)

        for out_key, aliases in alias_map.items():
            extra_sum[out_key] += _recursive_find_numeric(info, aliases)

        n += 1

    denom = max(n, 1)
    for key in extra_sum:
        result[key] = extra_sum[key] / denom

    return result


# =========================================================
# Runtime configuration helpers
# =========================================================
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


def _configure_torch_runtime(device: str) -> None:
    """Avoid severe CPU oversubscription on WSL/CPU runs; keep CUDA fast when available."""
    if device == "cpu":
        torch.set_num_threads(_env_int("TORCH_NUM_THREADS", 1))
        torch.set_num_interop_threads(_env_int("TORCH_INTEROP_THREADS", 1))
    elif device.startswith("cuda"):
        torch.backends.cudnn.benchmark = True


def _resolve_stage1_paths(prefix: str) -> Dict[str, str]:
    return {
        "actor": f"checkpoints/{prefix}_best_actor.pth",
        "encoder": f"checkpoints/{prefix}_best_encoder.pth",
        "fusion_net": f"checkpoints/{prefix}_best_fusion.pth",
        "ratio_head": f"checkpoints/{prefix}_best_ratio_head.pth",
    }


# =========================================================
# Main
# =========================================================
def main():
    requested_device = _env_str("DEVICE", "auto").lower()
    if requested_device == "cpu":
        device = "cpu"
    elif requested_device.startswith("cuda"):
        device = requested_device if torch.cuda.is_available() else "cpu"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    _configure_torch_runtime(device)

    base_seed = _env_int("SEED", 72)
    torch.manual_seed(base_seed)
    np.random.seed(base_seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(base_seed)

    print(f"device: {device}")
    print(f"seed: {base_seed}")

    # -------------------------------------------------
    # Env config
    # -------------------------------------------------
    env = MultiUavMecEnv(
        M=_env_int("M", 3),
        K=_env_int("K", 16),
        episode_length=_env_int("EPISODE_LENGTH", 20),
        cpu_mode=_env_str("CPU_MODE", "kkt"),
        omega1=_env_float("OMEGA1", 50.0),
        omega2=_env_float("OMEGA2", 1.0),
        deadline_scale=_env_float("DEADLINE_SCALE", 5.0),
        task_local_cpu_min=_env_float("TASK_LOCAL_CPU_MIN", 2.0e3),
        task_local_cpu_max=_env_float("TASK_LOCAL_CPU_MAX", 6.0e3),
        uav_energy_min=_env_float("UAV_ENERGY_MIN", 2600.0),
        uav_energy_max=_env_float("UAV_ENERGY_MAX", 3800.0),
        seed=base_seed,
    )

    obs = env.reset(seed=base_seed)
    state = obs["raw_state"]

    obs_dim = get_observation_dim(state)
    actor_raw_action_dim = get_actor_raw_action_dim(state)
    critic_action_dim = get_full_action_dim(state)

    print("obs_dim:", obs_dim)
    print("actor_raw_action_dim:", actor_raw_action_dim)
    print("critic_action_dim:", critic_action_dim)

    # -------------------------------------------------
    # Build full proposed policy
    # -------------------------------------------------
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

    # Load Stage-1 actor-side checkpoints only.
    # Do NOT load Stage-1 critic here: Stage-1 critic is intentionally not used.
    stage1_prefix = _env_str("STAGE1_PREFIX", "proposed_full_stage1_fix_k16_ep40")
    stage1_paths = _resolve_stage1_paths(stage1_prefix)
    strict_stage1 = _env_bool("STRICT_STAGE1", True)
    missing_stage1 = [path for path in stage1_paths.values() if not os.path.exists(path)]
    if missing_stage1 and strict_stage1:
        raise FileNotFoundError(
            "Stage-1 warm-start checkpoints are missing. Run Stage-1 first or set "
            "STRICT_STAGE1=0 for a debugging-only random Stage-2 run. Missing: "
            + ", ".join(missing_stage1)
        )
    load_if_exists(policy.actor_net, stage1_paths["actor"], "actor")
    load_if_exists(policy.encoder, stage1_paths["encoder"], "encoder")
    load_if_exists(policy.fusion_net, stage1_paths["fusion_net"], "fusion_net")
    load_if_exists(policy.ratio_head, stage1_paths["ratio_head"], "ratio_head")

    target_policy = copy.deepcopy(policy)

    # Stage-1 reference policy: frozen trust-region anchor.
    # The d25 seed72 logs show that the best policy is the warm-start policy at eval episode 0;
    # unconstrained Stage-2 actor updates gradually shift scheduling toward local execution.
    # This frozen copy provides a stable reference for mobility/scheduling and ratio-head outputs.
    stage1_reference_policy = copy.deepcopy(policy)
    for _module in [
        getattr(stage1_reference_policy, "actor_net", None),
        getattr(stage1_reference_policy, "encoder", None),
        getattr(stage1_reference_policy, "fusion_net", None),
        getattr(stage1_reference_policy, "ratio_head", None),
    ]:
        if _module is not None:
            _module.eval()
            for _p in _module.parameters():
                _p.requires_grad_(False)

    # -------------------------------------------------
    # Critic: Stage-2 starts from scratch
    # -------------------------------------------------
    critic = MLPCritic(
        obs_dim=obs_dim,
        action_dim=critic_action_dim,
        hidden_dim=256,
    ).to(device)
    print("Stage-2 critic starts from scratch. Stage-1 critic is intentionally NOT loaded.")

    target_critic = copy.deepcopy(critic).to(device)

    # -------------------------------------------------
    # Optimizers
    # -------------------------------------------------
    # Use separate learning rates: keep mobility/scheduling actor conservative,
    # but let the Transformer/fusion/ratio branch respond strongly enough to
    # the offloading regularization.
    actor_params = []
    ratio_branch_params = []

    if policy.actor_net is not None:
        actor_params += list(policy.actor_net.parameters())
    if policy.encoder is not None:
        ratio_branch_params += list(policy.encoder.parameters())
    if policy.fusion_net is not None:
        ratio_branch_params += list(policy.fusion_net.parameters())
    if policy.ratio_head is not None:
        ratio_branch_params += list(policy.ratio_head.parameters())

    learnable_params = actor_params + ratio_branch_params
    if len(learnable_params) == 0:
        raise RuntimeError("No learnable parameters found for Stage-2 actor-side update.")

    actor_opt = optim.Adam(
        [
            {"params": actor_params, "lr": _env_float("ACTOR_LR", 1e-5)},
            {"params": ratio_branch_params, "lr": _env_float("RATIO_BRANCH_LR", 5e-5)},
        ]
    )
    critic_opt = optim.Adam(critic.parameters(), lr=5e-5)

    mse_loss = nn.MSELoss()

    # -------------------------------------------------
    # Replay
    # -------------------------------------------------
    buffer = FullReplayBuffer(
        obs_dim=obs_dim,
        action_dim=critic_action_dim,
        capacity=30000,
    )

    # -------------------------------------------------
    # Hyperparameters
    # -------------------------------------------------
    gamma = _env_float("GAMMA", 0.95)
    tau = _env_float("TAU", 0.005)
    batch_size = _env_int("BATCH_SIZE", 128)
    learning_starts = _env_int("LEARNING_STARTS", 1000)
    # Paper-grade convergence run: 700 by default; use NUM_EPISODES=20 only for smoke tests.
    num_episodes = _env_int("NUM_EPISODES", 700)
    reward_scale = _env_float("REWARD_SCALE", 1e-4)
    policy_delay = _env_int("POLICY_DELAY", 2)
    grad_clip_norm = _env_float("GRAD_CLIP_NORM", 1.0)

    # critic_only_episodes = 150

    # actor_policy_coef = 0.003
    # actor_move_sched_bc_coef = 0.5
    # ratio_bc_coef = 0.2
    # Dynamic Stage-2 actor-side coefficients.
    # The d25 seed72 log shows that the original fixed weights make Stage-2
    # drift from the Stage-1 warm-start: local_exec_ratio rises and system cost
    # worsens. Therefore, v3 keeps the warm-start scheduling structure stable
    # while allowing a modest critic-guided fine-tuning signal.
    # v4: critic-guided deterministic actor gradient is intentionally weak because
    # the surrogate full action uses old bandwidth/CPU allocations and does not
    # fully match the environment post-processing. The trust-region anchor is the
    # main stabilizer; policy-gradient fine-tuning is only a small correction.
    actor_policy_coef_start = _env_float("ACTOR_POLICY_COEF_START", 0.0)
    actor_policy_coef_end = _env_float("ACTOR_POLICY_COEF_END", 0.0015)

    # Reuse the old CSV column name avg_move_sched_bc_loss, but this loss is now
    # Stage-1-reference anchor loss instead of placeholder-teacher BC.
    actor_move_sched_bc_coef_start = _env_float("ACTOR_MOVE_SCHED_BC_COEF_START", 2.00)
    actor_move_sched_bc_coef_end = _env_float("ACTOR_MOVE_SCHED_BC_COEF_END", 1.20)

    ratio_bc_coef_start = _env_float("RATIO_BC_COEF_START", 0.050)
    ratio_bc_coef_end = _env_float("RATIO_BC_COEF_END", 0.020)

    # The original ratio regularization target (floor=0.15, mean=0.20, std=0.06)
    # is too aggressive for the d25 environment and dominates total_actor_loss.
    # v3 uses moderate targets to prevent ratio collapse without fighting the critic.
    ratio_reg_coef_start = _env_float("RATIO_REG_COEF_START", 0.030)
    ratio_reg_coef_end = _env_float("RATIO_REG_COEF_END", 0.010)
    ratio_floor_target = _env_float("RATIO_FLOOR_TARGET", 0.05)
    ratio_mean_target = _env_float("RATIO_MEAN_TARGET", 0.10)
    ratio_std_target = _env_float("RATIO_STD_TARGET", 0.020)

    # Collaboration anti-collapse term. It discourages the scheduler from
    # collapsing to local-only execution when legal neighbor candidates exist.
    neighbor_collab_coef_start = _env_float("NEIGHBOR_COLLAB_COEF_START", 0.30)
    neighbor_collab_coef_end = _env_float("NEIGHBOR_COLLAB_COEF_END", 0.80)
    neighbor_prob_target = _env_float("NEIGHBOR_PROB_TARGET", 0.28)
    local_dominance_coef_start = _env_float("LOCAL_DOMINANCE_COEF_START", 0.10)
    local_dominance_coef_end = _env_float("LOCAL_DOMINANCE_COEF_END", 0.35)
    local_dominance_margin = _env_float("LOCAL_DOMINANCE_MARGIN", 0.35)

    rollback_patience = _env_int("ROLLBACK_PATIENCE", 2)
    rollback_tolerance = _env_float("ROLLBACK_TOLERANCE", 100.0)
    rollback_lr_decay = _env_float("ROLLBACK_LR_DECAY", 0.70)

    # Shorten critic warm-up so actor/encoder/fusion/ratio-head can update within a 200-episode verification run.
    critic_only_episodes = _env_int("CRITIC_ONLY_EPISODES", 100)

    actor_l2_coef = _env_float("ACTOR_L2_COEF", 1e-5)

    eval_every = _env_int("EVAL_EVERY", 5)
    global_update_step = 0

    # -------------------------------------------------
    # Log / save dirs
    # -------------------------------------------------
    M = int(state["M"])
    K = int(state["K"])
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    default_run_name = (
        f"proposed_full_stage2_converge_k{K}_seed{base_seed}"
        f"_cw{critic_only_episodes}"
        f"_rr{ratio_reg_coef_start:g}-{ratio_reg_coef_end:g}"
        f"_nc{neighbor_collab_coef_start:g}-{neighbor_collab_coef_end:g}"
        f"_{timestamp}"
    )
    run_name = _env_str("RUN_NAME", default_run_name)
    result_dir = os.path.join("results", "convergence_training", run_name)
    os.makedirs(result_dir, exist_ok=True)

    train_log_path = os.path.join(result_dir, "train_log.csv")
    eval_log_path = os.path.join(result_dir, "eval_log.csv")

    with open(train_log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "episode",
            "episode_reward",
            "avg_actor_policy_loss",
            "avg_move_sched_bc_loss",
            "avg_ratio_bc_loss",
            "avg_ratio_reg_loss",
            "avg_ratio_floor_loss",
            "avg_ratio_mean_loss",
            "avg_ratio_diversity_loss",
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

    # movement + scheduling masks in raw actor output
    move_sched_mask = np.zeros((actor_raw_action_dim,), dtype=np.float32)
    move_sched_mask[: M + M] = 1.0
    move_sched_mask[M + M + K:] = 1.0
    move_sched_mask_t = torch.tensor(
        move_sched_mask, dtype=torch.float32, device=device
    ).unsqueeze(0)

    best_policy_state = copy.deepcopy(policy)
    best_eval_reward = -float("inf")
    # Save the truly best validation policy. If Stage-2 actor updates degrade the
    # warm-start, this prevents the final checkpoint from being worse than the
    # Stage-1 initialized policy. For ablation, set EVAL_BEST_START_EPISODE.
    eval_best_start_episode = _env_int("EVAL_BEST_START_EPISODE", 0)

    print("Stage-2 converge config:")
    print(f"  num_episodes={num_episodes}, episode_length={env.episode_length}, batch_size={batch_size}, learning_starts={learning_starts}")
    print(f"  critic_only_episodes={critic_only_episodes}, policy_delay={policy_delay}, reward_scale={reward_scale}")
    print(f"  actor_policy_coef={actor_policy_coef_start}->{actor_policy_coef_end}")
    print(f"  actor_move_sched_bc_coef={actor_move_sched_bc_coef_start}->{actor_move_sched_bc_coef_end}")
    print(f"  ratio_bc_coef={ratio_bc_coef_start}->{ratio_bc_coef_end}")
    print(f"  ratio_reg_coef={ratio_reg_coef_start}->{ratio_reg_coef_end}, floor/mean/std targets={ratio_floor_target}/{ratio_mean_target}/{ratio_std_target}")
    print(f"  neighbor_collab_coef={neighbor_collab_coef_start}->{neighbor_collab_coef_end}, neighbor_prob_target={neighbor_prob_target}")
    print(f"  local_dominance_coef={local_dominance_coef_start}->{local_dominance_coef_end}, local_margin={local_dominance_margin}")
    print(f"  rollback_patience={rollback_patience}, tolerance={rollback_tolerance}, lr_decay={rollback_lr_decay}")
    print(f"  run_name={run_name}")
    print(f"  stage1_prefix={stage1_prefix}, strict_stage1={strict_stage1}")

    # -------------------------------------------------
    # Training
    # -------------------------------------------------
    for ep in range(num_episodes):
        obs = env.reset(seed=base_seed + ep)
        done = False

        episode_reward = 0.0
        episode_actor_policy_loss = 0.0
        episode_move_sched_bc_loss = 0.0
        episode_ratio_bc_loss = 0.0
        episode_ratio_reg_loss = 0.0
        episode_ratio_floor_loss = 0.0
        episode_ratio_mean_loss = 0.0
        episode_ratio_diversity_loss = 0.0
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

            # -----------------------------------------
            # teacher action for BC regularization
            # -----------------------------------------
            teacher_action = generate_proposed_placeholder_action(
                state=state,
                access_assoc=access_assoc,
            )
            teacher_raw_target = build_teacher_raw_target(
                state=state,
                access_assoc=access_assoc,
                teacher_action=teacher_action,
            )
            teacher_offload_ratio = np.asarray(
                teacher_action["offload_ratio"],
                dtype=np.float32,
            )

            # -----------------------------------------
            # execute current full proposed action
            # -----------------------------------------
            action = policy.act(
                state=state,
                access_assoc=access_assoc,
                deterministic=True,
                return_aux=False,
            )
            action = apply_safe_mobility_projection(
                action,
                state,
                safety_margin=1.0,
                collision_margin=1e-5
            )

            flat_action = flatten_full_action(action)
            flat_action_norm = normalize_full_action_np(flat_action, state)

            next_obs, reward, done, info = env.step(action)
            next_obs_vec = build_global_observation(next_obs["raw_state"])

            if not done:
                next_state = next_obs["raw_state"]
                next_access_assoc = build_access_association(next_state)
                next_action_target = target_policy.act(
                    state=next_state,
                    access_assoc=next_access_assoc,
                    deterministic=True,
                    return_aux=False,
                )
                next_action_target = apply_safe_mobility_projection(
                    next_action_target,
                    next_state,
                    safety_margin=1.0,
                    collision_margin=1e-5,
                )
                next_flat_action = flatten_full_action(next_action_target)
                next_flat_action_norm = normalize_full_action_np(next_flat_action, next_state)
            else:
                next_flat_action_norm = np.zeros((critic_action_dim,), dtype=np.float32)

            scaled_reward = float(reward) * reward_scale

            buffer.add(
                obs=obs_vec,
                action=flat_action_norm,
                reward=scaled_reward,
                next_obs=next_obs_vec,
                next_action=next_flat_action_norm,
                done=float(done),
            )

            # -----------------------------------------
            # critic update from replay
            # -----------------------------------------
            did_critic_update = False
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
                critic_loss = F.smooth_l1_loss(q_val, y)
                td_abs_error = torch.abs(q_val - y).mean()

                critic_opt.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=grad_clip_norm)
                critic_opt.step()

                critic_update_count += 1
                global_update_step += 1
                did_critic_update = True

                episode_critic_loss += float(critic_loss.item())
                episode_q_value += float(q_val.mean().item())
                episode_target_q_value += float(y.mean().item())
                episode_td_abs_error += float(td_abs_error.item())

            # -----------------------------------------
            # delayed actor / encoder / ratio-head update
            # -----------------------------------------
            actor_updated_this_step = False
            if (
                did_critic_update
                and ep >= critic_only_episodes
                and (global_update_step % policy_delay == 0)
            ):
                obs_tensor = torch.tensor(
                    obs_vec, dtype=torch.float32, device=device
                ).unsqueeze(0)
                teacher_raw_target_t = torch.tensor(
                    teacher_raw_target, dtype=torch.float32, device=device
                ).unsqueeze(0)
                teacher_offload_ratio_t = torch.tensor(
                    teacher_offload_ratio, dtype=torch.float32, device=device
                )

                # Dynamic coefficient schedule. progress=0 right after critic-only
                # warm-up, progress=1 at the end of training.
                coef_progress = (ep - critic_only_episodes) / max(num_episodes - critic_only_episodes, 1)
                actor_policy_coef_t = linear_schedule_value(
                    actor_policy_coef_start, actor_policy_coef_end, coef_progress
                )
                actor_move_sched_bc_coef_t = linear_schedule_value(
                    actor_move_sched_bc_coef_start, actor_move_sched_bc_coef_end, coef_progress
                )
                ratio_bc_coef_t = linear_schedule_value(
                    ratio_bc_coef_start, ratio_bc_coef_end, coef_progress
                )
                ratio_reg_coef_t = linear_schedule_value(
                    ratio_reg_coef_start, ratio_reg_coef_end, coef_progress
                )
                neighbor_collab_coef_t = linear_schedule_value(
                    neighbor_collab_coef_start, neighbor_collab_coef_end, coef_progress
                )
                local_dominance_coef_t = linear_schedule_value(
                    local_dominance_coef_start, local_dominance_coef_end, coef_progress
                )

                raw_pred = policy.actor_net(obs_tensor)

                # Use unclamped ratio for ratio_bc_loss / ratio_reg_loss so the ratio head
                # receives gradients even when it wants to sit below the execution floor.
                ratio_pred = forward_ratio_branch(
                    policy=policy,
                    state=state,
                    access_assoc=access_assoc,
                    use_hard_min=False,
                )

                # The critic-side surrogate action must still remain physically legal.
                ratio_pred_for_action = torch.clamp(
                    ratio_pred,
                    min=policy.ratio_floor,
                    max=policy.ratio_ceiling,
                )

                surrogate_full_action = build_surrogate_full_action_tensor(
                    state=state,
                    access_assoc=access_assoc,
                    raw_action_pred=raw_pred,
                    ratio_pred=ratio_pred_for_action,
                    bandwidth_alloc_np=np.asarray(action["bandwidth_alloc"], dtype=np.float32),
                    cpu_alloc_np=np.asarray(action["cpu_alloc"], dtype=np.float32),
                )
                surrogate_full_action_norm = normalize_full_action_tensor(surrogate_full_action, state)

                # Freeze critic weights during actor update.
                for p_critic in critic.parameters():
                    p_critic.requires_grad_(False)

                actor_policy_loss = -critic(obs_tensor, surrogate_full_action_norm).mean()

                # Trust-region anchor to the frozen Stage-1 policy, not to the placeholder teacher.
                # This prevents Stage-2 from destroying the warm-start scheduling structure.
                with torch.no_grad():
                    ref_raw_pred = stage1_reference_policy.actor_net(obs_tensor)
                    ref_ratio_pred = forward_ratio_branch(
                        policy=stage1_reference_policy,
                        state=state,
                        access_assoc=access_assoc,
                        use_hard_min=False,
                    )

                move_sched_pred = raw_pred * move_sched_mask_t
                move_sched_target = ref_raw_pred * move_sched_mask_t
                actor_move_sched_bc_loss = mse_loss(move_sched_pred, move_sched_target)

                ratio_bc_loss = mse_loss(ratio_pred, ref_ratio_pred)
                ratio_reg_loss, ratio_floor_loss, ratio_mean_loss, ratio_diversity_loss = (
                    compute_ratio_regularization_loss(
                        ratio_pred=ratio_pred,
                        ratio_floor_target=ratio_floor_target,
                        ratio_mean_target=ratio_mean_target,
                        ratio_std_target=ratio_std_target,
                    )
                )
                neighbor_collab_loss = compute_neighbor_collaboration_loss(
                    raw_action_pred=raw_pred,
                    state=state,
                    access_assoc=access_assoc,
                    neighbor_prob_target=neighbor_prob_target,
                )
                local_dominance_loss = compute_local_dominance_margin_loss(
                    raw_action_pred=raw_pred,
                    state=state,
                    access_assoc=access_assoc,
                    local_margin=local_dominance_margin,
                )
                actor_l2_loss = (raw_pred ** 2).mean()

                total_actor_loss = (
                    actor_policy_coef_t * actor_policy_loss
                    + actor_move_sched_bc_coef_t * actor_move_sched_bc_loss
                    + ratio_bc_coef_t * ratio_bc_loss
                    + ratio_reg_coef_t * ratio_reg_loss
                    + neighbor_collab_coef_t * neighbor_collab_loss
                    + local_dominance_coef_t * local_dominance_loss
                    + actor_l2_coef * actor_l2_loss
                )

                actor_opt.zero_grad()
                total_actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(learnable_params, max_norm=grad_clip_norm)
                actor_opt.step()

                for p_critic in critic.parameters():
                    p_critic.requires_grad_(True)

                soft_update_policy(target_policy, policy, tau=tau)
                soft_update(target_critic, critic, tau=tau)

                actor_updated_this_step = True
                actor_update_count += 1
                episode_actor_policy_loss += float(actor_policy_loss.item())
                episode_move_sched_bc_loss += float(actor_move_sched_bc_loss.item())
                episode_ratio_bc_loss += float(ratio_bc_loss.item())
                episode_ratio_reg_loss += float(ratio_reg_loss.item())
                episode_ratio_floor_loss += float(ratio_floor_loss.item())
                episode_ratio_mean_loss += float(ratio_mean_loss.item())
                episode_ratio_diversity_loss += float(ratio_diversity_loss.item())
                episode_total_actor_loss += float(total_actor_loss.item())

                if step_count < 2:
                    with torch.no_grad():
                        print("[STAGE2 DEBUG]")
                        ratio_np = ratio_pred.detach().cpu().numpy()
                        print("  ratio_pred[:5] =", ratio_np[:5])
                        print("  teacher_raw[:5] =", teacher_offload_ratio[:5])
                        print("  ratio_pred mean/std/min/max =",
                              float(ratio_np.mean()),
                              float(ratio_np.std()),
                              float(ratio_np.min()),
                              float(ratio_np.max()))
                        print("  ratio_reg_loss =", float(ratio_reg_loss.item()))
                        print("  ratio_floor_loss =", float(ratio_floor_loss.item()))
                        print("  ratio_mean_loss =", float(ratio_mean_loss.item()))
                        print("  ratio_diversity_loss =", float(ratio_diversity_loss.item()))

            # During critic-only warm-up or delayed actor steps, keep target_critic tracking critic.
            # target_policy is updated only when the actor is updated.
            if did_critic_update and not actor_updated_this_step:
                soft_update(target_critic, critic, tau=tau)

            obs = next_obs
            episode_reward += float(reward)
            step_count += 1

        avg_actor_policy_loss_value = (
            episode_actor_policy_loss / actor_update_count if actor_update_count > 0 else np.nan
        )
        avg_move_sched_bc_loss_value = (
            episode_move_sched_bc_loss / actor_update_count if actor_update_count > 0 else np.nan
        )
        avg_ratio_bc_loss_value = (
            episode_ratio_bc_loss / actor_update_count if actor_update_count > 0 else np.nan
        )
        avg_ratio_reg_loss_value = (
            episode_ratio_reg_loss / actor_update_count if actor_update_count > 0 else np.nan
        )
        avg_ratio_floor_loss_value = (
            episode_ratio_floor_loss / actor_update_count if actor_update_count > 0 else np.nan
        )
        avg_ratio_mean_loss_value = (
            episode_ratio_mean_loss / actor_update_count if actor_update_count > 0 else np.nan
        )
        avg_ratio_diversity_loss_value = (
            episode_ratio_diversity_loss / actor_update_count if actor_update_count > 0 else np.nan
        )
        avg_total_actor_loss_value = (
            episode_total_actor_loss / actor_update_count if actor_update_count > 0 else np.nan
        )
        avg_critic_loss_value = (
            episode_critic_loss / critic_update_count if critic_update_count > 0 else np.nan
        )
        avg_q_value = episode_q_value / critic_update_count if critic_update_count > 0 else np.nan
        avg_target_q_value = episode_target_q_value / critic_update_count if critic_update_count > 0 else np.nan
        avg_td_abs_error = episode_td_abs_error / critic_update_count if critic_update_count > 0 else np.nan

        print(f"\nEpisode {ep}")
        print("episode_reward:", episode_reward)
        print("avg_actor_policy_loss:", avg_actor_policy_loss_value)
        print("avg_move_sched_bc_loss:", avg_move_sched_bc_loss_value)
        print("avg_ratio_bc_loss:", avg_ratio_bc_loss_value)
        print("avg_ratio_reg_loss:", avg_ratio_reg_loss_value)
        print("avg_ratio_floor_loss:", avg_ratio_floor_loss_value)
        print("avg_ratio_mean_loss:", avg_ratio_mean_loss_value)
        print("avg_ratio_diversity_loss:", avg_ratio_diversity_loss_value)
        print("avg_total_actor_loss:", avg_total_actor_loss_value)
        print("avg_critic_loss:", avg_critic_loss_value)
        print("avg_q_value:", avg_q_value)
        print("avg_target_q_value:", avg_target_q_value)
        print("avg_td_abs_error:", avg_td_abs_error)
        print("steps:", step_count)
        print("buffer_size:", len(buffer))
        print("critic_updates:", critic_update_count)
        print("actor_updates:", actor_update_count)

        with open(train_log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                ep,
                episode_reward,
                avg_actor_policy_loss_value,
                avg_move_sched_bc_loss_value,
                avg_ratio_bc_loss_value,
                avg_ratio_reg_loss_value,
                avg_ratio_floor_loss_value,
                avg_ratio_mean_loss_value,
                avg_ratio_diversity_loss_value,
                avg_total_actor_loss_value,
                avg_critic_loss_value,
                avg_q_value,
                avg_target_q_value,
                avg_td_abs_error,
                step_count,
                len(buffer),
                critic_update_count,
                actor_update_count,
            ])

        # -----------------------------------------
        # periodic evaluation
        # -----------------------------------------
        if (ep + 1) % eval_every == 0 or ep == 0:
            eval_env = MultiUavMecEnv(
                M=M,
                K=K,
                episode_length=env.episode_length,
                cpu_mode=env.cpu_mode,
                omega1=env.omega1,
                omega2=env.omega2,
                deadline_scale=env.deadline_scale,
                task_local_cpu_min=env.task_local_cpu_min,
                task_local_cpu_max=env.task_local_cpu_max,
                uav_energy_min=env.uav_energy_min,
                uav_energy_max=env.uav_energy_max,
                seed=_env_int("EVAL_SEED", 999),
            )

            with policy_eval_mode(policy):
                eval_result = evaluate_full_policy_rollout_with_extra_constraints(
                    env=eval_env,
                    policy=policy,
                    seed=_env_int("EVAL_SEED", 999),
                )

            print("\n==============================")
            print(f"Full Proposed Stage-2 Eval @ episode {ep}")
            print("==============================")
            print("episode_reward:", eval_result["episode_reward"])
            print("system_cost:", eval_result.get("system_cost", -eval_result["episode_reward"]))
            print("avg_delay:", eval_result["avg_delay"])
            print("avg_energy:", eval_result["avg_energy"])
            print("avg_deadline_violation:", eval_result["avg_deadline_violation"])
            print("feasible_ratio:", eval_result["feasible_ratio"])
            print("num_steps:", eval_result["num_steps"])
            print("avg_ratio_violation:", eval_result["avg_ratio_violation"])
            print("avg_assoc_violation:", eval_result["avg_assoc_violation"])
            print("avg_schedule_violation:", eval_result["avg_schedule_violation"])
            print("avg_candidate_violation:", eval_result["avg_candidate_violation"])
            print("avg_bw_violation:", eval_result["avg_bw_violation"])
            print("avg_cpu_violation:", eval_result["avg_cpu_violation"])
            print("avg_rate_violation:", eval_result["avg_rate_violation"])
            print("avg_nan_count:", eval_result["avg_nan_count"])
            print("avg_move_violation:", eval_result["avg_move_violation"])
            print("avg_boundary_violation:", eval_result["avg_boundary_violation"])
            print("avg_collision_violation:", eval_result["avg_collision_violation"])
            print("avg_battery_violation:", eval_result["avg_battery_violation"])
            print("ratio_mean:", eval_result.get("ratio_mean", float("nan")))
            print("ratio_std:", eval_result.get("ratio_std", float("nan")))
            print("ratio_min:", eval_result.get("ratio_min", float("nan")))
            print("ratio_max:", eval_result.get("ratio_max", float("nan")))
            print("local_exec_ratio:", eval_result.get("local_exec_ratio", float("nan")))
            print("neighbor_exec_ratio:", eval_result.get("neighbor_exec_ratio", float("nan")))

            with open(eval_log_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    ep,
                    eval_result["episode_reward"],
                    eval_result.get("system_cost", -eval_result["episode_reward"]),
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
                    eval_result.get("ratio_mean", float("nan")),
                    eval_result.get("ratio_std", float("nan")),
                    eval_result.get("ratio_min", float("nan")),
                    eval_result.get("ratio_max", float("nan")),
                    eval_result.get("local_exec_ratio", float("nan")),
                    eval_result.get("neighbor_exec_ratio", float("nan")),
                ])

            if ep >= eval_best_start_episode and eval_result["episode_reward"] > best_eval_reward:
                best_eval_reward = eval_result["episode_reward"]
                best_policy_state = copy.deepcopy(policy)
                eval_bad_count = 0
                print(f"[BEST] updated best eval reward to {best_eval_reward:.6f} at episode {ep}")
            elif ep >= eval_best_start_episode:
                # Degradation guard: if the current evaluated policy is clearly worse than
                # the best warm-start/fine-tuned policy for several evaluations, restore it.
                if eval_result["episode_reward"] < best_eval_reward - rollback_tolerance:
                    eval_bad_count += 1
                else:
                    eval_bad_count = 0

                if eval_bad_count >= rollback_patience:
                    print(
                        f"[ROLLBACK] eval reward {eval_result['episode_reward']:.6f} is worse than "
                        f"best {best_eval_reward:.6f}. Restoring best policy and decaying actor-side LR."
                    )
                    load_policy_weights_inplace(policy, best_policy_state)
                    load_policy_weights_inplace(target_policy, best_policy_state)
                    for group in actor_opt.param_groups:
                        group["lr"] = max(group["lr"] * rollback_lr_decay, 1e-7)
                    eval_bad_count = 0

    # The saved checkpoints below always use best_policy_state, so they never
    # correspond to a degraded last-episode actor.

    # -------------------------------------------------
    # Save best stage-2 checkpoints
    # -------------------------------------------------
    os.makedirs("checkpoints", exist_ok=True)

    ckpt_prefix = _env_str("CKPT_PREFIX", run_name)

    if best_policy_state.actor_net is not None:
        torch.save(
            best_policy_state.actor_net.state_dict(),
            f"checkpoints/{ckpt_prefix}_best_actor.pth",
        )
    if best_policy_state.encoder is not None:
        torch.save(
            best_policy_state.encoder.state_dict(),
            f"checkpoints/{ckpt_prefix}_best_encoder.pth",
        )
    if best_policy_state.fusion_net is not None:
        torch.save(
            best_policy_state.fusion_net.state_dict(),
            f"checkpoints/{ckpt_prefix}_best_fusion.pth",
        )
    if best_policy_state.ratio_head is not None:
        torch.save(
            best_policy_state.ratio_head.state_dict(),
            f"checkpoints/{ckpt_prefix}_best_ratio_head.pth",
        )

    torch.save(
        critic.state_dict(),
        f"checkpoints/{ckpt_prefix}_critic.pth",
    )

    print("\nFull proposed Stage-2 stable training finished successfully.")
    print("Saved:")
    print(f"  checkpoints/{ckpt_prefix}_best_actor.pth")
    print(f"  checkpoints/{ckpt_prefix}_best_encoder.pth")
    print(f"  checkpoints/{ckpt_prefix}_best_fusion.pth")
    print(f"  checkpoints/{ckpt_prefix}_best_ratio_head.pth")
    print(f"  checkpoints/{ckpt_prefix}_critic.pth")
    print("Logs:")
    print(f"  {train_log_path}")
    print(f"  {eval_log_path}")


if __name__ == "__main__":
    main()
