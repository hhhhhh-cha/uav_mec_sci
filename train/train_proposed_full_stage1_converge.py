# # 它要做的事是：
# # 用 ProposedPolicy 替代旧的简化策略接口
# # actor 训练目标先保留简单版
# # encoder / fusion / ratio head 也纳入网络参数
# # 先把“完整 Proposed 方法训练骨架”跑通

# import copy
# from collections import deque
# from typing import Any, Dict, List, Optional, Tuple

# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim

# from env.mec_env import MultiUavMecEnv
# from env.association import build_access_association
# from model.mlp_actor import MLPActor
# from model.mlp_critic import MLPCritic
# from model.proposed_obs_builder import build_global_observation, get_observation_dim
# from policy.proposed_placeholder_policy import generate_proposed_placeholder_action
# from policy.proposed_policy import build_default_proposed_policy, ProposedPolicy

import copy
import csv
import json
import os
import time
from contextlib import contextmanager
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from env.mec_env import MultiUavMecEnv
from env.association import build_access_association
from model.mlp_actor import MLPActor
from model.mlp_critic import MLPCritic
from model.proposed_obs_builder import build_global_observation, get_observation_dim
from policy.proposed_placeholder_policy import generate_proposed_placeholder_action
from policy.proposed_policy import build_default_proposed_policy, ProposedPolicy


EPS = 1e-8


# =========================================================
# Helpers: dimensions / flatten / soft update
# =========================================================
def get_actor_raw_action_dim(state: Dict[str, Any]) -> int:
    """
    Raw actor output layout:
        [ move_dist(M), move_angle(M), offload_ratio(K), sched_score(K*M) ]
    """
    M = int(state["M"])
    K = int(state["K"])
    return M + M + K + K * M


def get_full_action_dim(state: Dict[str, Any]) -> int:
    """
    Full action layout used by critic:
        [ move_dist(M),
          move_angle(M),
          offload_ratio(K),
          sched_beta(K*M*M),
          bandwidth_alloc(M*K),
          cpu_alloc(M*K) ]
    """
    M = int(state["M"])
    K = int(state["K"])
    return M + M + K + K * M * M + M * K + M * K


def flatten_full_action(action: Dict[str, Any]) -> np.ndarray:
    move_dist = np.asarray(action["move_dist"], dtype=np.float32).reshape(-1)
    move_angle = np.asarray(action["move_angle"], dtype=np.float32).reshape(-1)
    offload_ratio = np.asarray(action["offload_ratio"], dtype=np.float32).reshape(-1)
    sched_beta = np.asarray(action["sched_beta"], dtype=np.float32).reshape(-1)
    bandwidth_alloc = np.asarray(action["bandwidth_alloc"], dtype=np.float32).reshape(-1)
    cpu_alloc = np.asarray(action["cpu_alloc"], dtype=np.float32).reshape(-1)

    return np.concatenate(
        [
            move_dist,
            move_angle,
            offload_ratio,
            sched_beta,
            bandwidth_alloc,
            cpu_alloc,
        ],
        axis=0,
    ).astype(np.float32)


def soft_update(target_net: nn.Module, source_net: nn.Module, tau: float = 0.005):
    for tp, sp in zip(target_net.parameters(), source_net.parameters()):
        tp.data.copy_(tau * sp.data + (1.0 - tau) * tp.data)


def soft_update_policy(target_policy: ProposedPolicy, source_policy: ProposedPolicy, tau: float = 0.005):
    if target_policy.actor_net is not None and source_policy.actor_net is not None:
        soft_update(target_policy.actor_net, source_policy.actor_net, tau=tau)

    if target_policy.encoder is not None and source_policy.encoder is not None:
        soft_update(target_policy.encoder, source_policy.encoder, tau=tau)

    if target_policy.fusion_net is not None and source_policy.fusion_net is not None:
        soft_update(target_policy.fusion_net, source_policy.fusion_net, tau=tau)

    if target_policy.ratio_head is not None and source_policy.ratio_head is not None:
        soft_update(target_policy.ratio_head, source_policy.ratio_head, tau=tau)


# =========================================================
# Helpers: teacher target
# =========================================================
def build_teacher_raw_target(state, access_assoc, teacher_action):
    """
    Convert env-style teacher action dict into raw-action target format.

    Raw actor layout:
        [ move_dist(M), move_angle(M), offload_ratio(K), sched_score(K*M) ]
    """
    M = int(state["M"])
    K = int(state["K"])
    max_speed = float(state["max_speed"])
    delta_t = float(state["delta_t"])
    max_move = max(max_speed * delta_t, EPS)

    move_dist = np.asarray(teacher_action["move_dist"], dtype=np.float32).reshape(M)
    move_angle = np.asarray(teacher_action["move_angle"], dtype=np.float32).reshape(M)
    offload_ratio = np.asarray(teacher_action["offload_ratio"], dtype=np.float32).reshape(K)
    sched_beta = np.asarray(teacher_action["sched_beta"], dtype=np.float32).reshape(K, M, M)

    move_dist_target = 2.0 * (move_dist / max_move) - 1.0
    move_dist_target = np.clip(move_dist_target, -1.0, 1.0)

    move_angle_target = move_angle / np.pi
    move_angle_target = np.clip(move_angle_target, -1.0, 1.0)

    offload_target = 2.0 * offload_ratio - 1.0
    offload_target = np.clip(offload_target, -1.0, 1.0)

    sched_score_target = -np.ones((K, M), dtype=np.float32)
    for k in range(K):
        access_m = int(np.argmax(access_assoc[:, k]))
        chosen_js = np.where(sched_beta[k, access_m, :] > 0.5)[0]
        if len(chosen_js) == 1:
            sched_score_target[k, int(chosen_js[0])] = 1.0

    raw_target = np.concatenate(
        [
            move_dist_target.reshape(-1),
            move_angle_target.reshape(-1),
            offload_target.reshape(-1),
            sched_score_target.reshape(-1),
        ],
        axis=0,
    ).astype(np.float32)

    return raw_target


def split_actor_raw_action(raw_action: torch.Tensor, M: int, K: int):
    """
    raw layout:
        [ move_dist(M), move_angle(M), offload_ratio(K), sched_score(K*M) ]
    """
    p = 0
    move_dist_raw = raw_action[:, p:p + M]
    p += M

    move_angle_raw = raw_action[:, p:p + M]
    p += M

    offload_raw = raw_action[:, p:p + K]
    p += K

    sched_raw = raw_action[:, p:p + K * M].reshape(-1, K, M)

    return move_dist_raw, move_angle_raw, offload_raw, sched_raw


# =========================================================
# Helpers: state feature extraction for ratio branch
# =========================================================
def _safe_get(state: Dict[str, Any], key: str, default=None):
    return state[key] if key in state else default


def _to_numpy(x, dtype=np.float32):
    return np.asarray(x, dtype=dtype)


def _extract_neighbors(state: Dict[str, Any]) -> List[List[int]]:
    neighbors = _safe_get(state, "neighbors", None)
    if neighbors is None:
        M = int(state["M"])
        return [[] for _ in range(M)]
    return neighbors


def _extract_uav_positions(state: Dict[str, Any]) -> np.ndarray:
    for key in ["uav_pos", "uav_positions", "q", "uav_xy"]:
        if key in state:
            arr = np.asarray(state[key], dtype=np.float32)
            if arr.ndim == 2 and arr.shape[1] >= 2:
                return arr[:, :2]
    raise KeyError("Cannot find UAV position array in state.")


def _extract_task_size(state: Dict[str, Any]) -> np.ndarray:
    for key in ["task_size", "D", "D_k", "task_data_size"]:
        if key in state:
            return _to_numpy(state[key])
    raise KeyError("Cannot find task size in state.")


def _extract_task_cycles(state: Dict[str, Any]) -> np.ndarray:
    for key in ["task_cycles", "C", "C_k", "task_cpu_cycles"]:
        if key in state:
            return _to_numpy(state[key])
    raise KeyError("Cannot find task cycles in state.")


def _extract_task_deadline(state: Dict[str, Any]) -> np.ndarray:
    for key in ["task_deadline", "deadline", "tau_max", "task_max_delay"]:
        if key in state:
            return _to_numpy(state[key])
    raise KeyError("Cannot find task deadline in state.")


def _extract_available_cpu(state: Dict[str, Any]) -> np.ndarray:
    for key in ["uav_available_cpu", "available_cpu", "uav_cpu_avail", "uav_cpu_max"]:
        if key in state:
            return _to_numpy(state[key])
    M = int(state["M"])
    return np.ones(M, dtype=np.float32)


def _extract_workload(state: Dict[str, Any]) -> np.ndarray:
    for key in ["uav_workload", "workload", "uav_load", "queue_len"]:
        if key in state:
            return _to_numpy(state[key])
    M = int(state["M"])
    return np.zeros(M, dtype=np.float32)


def _extract_tx_power(state: Dict[str, Any], access_m: int) -> float:
    for key in ["uav_tx_power", "P_tx", "tx_power"]:
        if key in state:
            val = state[key]
            arr = np.asarray(val, dtype=np.float32)
            if arr.ndim == 0:
                return float(arr)
            return float(arr[access_m])
    return 1.0


def _extract_backhaul_bw(state: Dict[str, Any]) -> float:
    for key in ["backhaul_bandwidth", "B_bh", "bh_bandwidth"]:
        if key in state:
            return float(state[key])
    return 1.0


def _extract_noise_power(state: Dict[str, Any]) -> float:
    for key in ["noise_power", "sigma2", "noise_var"]:
        if key in state:
            return float(state[key])
    return 1e-9


def _compute_a2a_gain_and_rate(state: Dict[str, Any], access_m: int, exec_j: int) -> Tuple[float, float]:
    if exec_j == access_m:
        return 1.0, 1e6

    uav_pos = _extract_uav_positions(state)
    p_m = uav_pos[access_m]
    p_j = uav_pos[exec_j]

    dist = float(np.linalg.norm(p_m - p_j))
    dist = max(dist, 1.0)

    beta0 = float(_safe_get(state, "beta0", 1.0))
    gain = beta0 * (dist ** -2)

    B_bh = _extract_backhaul_bw(state)
    P_tx = _extract_tx_power(state, access_m)
    sigma2 = _extract_noise_power(state)

    rate = B_bh * np.log2(1.0 + P_tx * gain / max(sigma2, EPS))
    return float(gain), float(rate)


def build_ratio_branch_inputs(
    state: Dict[str, Any],
    access_assoc: np.ndarray,
    max_candidates: int,
    device: str,
):
    """
    Build:
      token_tensor: [K, Nc, 7]
      mask_tensor:  [K, Nc]
      task_feat:    [K, 3]
      uav_feat:     [K, 4]
    """
    M = int(state["M"])
    K = int(state["K"])

    task_size = _extract_task_size(state)
    task_cycles = _extract_task_cycles(state)
    task_deadline = _extract_task_deadline(state)

    available_cpu = _extract_available_cpu(state)
    workload = _extract_workload(state)
    uav_pos = _extract_uav_positions(state)
    neighbors = _extract_neighbors(state)

    token_batches = []
    mask_batches = []
    task_feat_batches = []
    uav_feat_batches = []

    for k in range(K):
        access_m = int(np.argmax(access_assoc[:, k]))
        candidates = [access_m] + list(neighbors[access_m])
        candidates = candidates[:max_candidates]

        tokens = np.zeros((max_candidates, 7), dtype=np.float32)
        mask = np.zeros((max_candidates,), dtype=np.float32)

        for idx, j in enumerate(candidates):
            h_a2a, r_a2a = _compute_a2a_gain_and_rate(state, access_m, j)
            tokens[idx] = np.asarray(
                [
                    float(h_a2a),
                    float(r_a2a),
                    float(workload[j]),
                    float(available_cpu[j]),
                    float(task_size[k]),
                    float(task_cycles[k]),
                    float(task_deadline[k]),
                ],
                dtype=np.float32,
            )
            mask[idx] = 1.0

        task_feat = np.asarray(
            [
                float(task_size[k]),
                float(task_cycles[k]),
                float(task_deadline[k]),
            ],
            dtype=np.float32,
        )

        uav_feat = np.asarray(
            [
                float(uav_pos[access_m, 0]),
                float(uav_pos[access_m, 1]),
                float(available_cpu[access_m]),
                float(workload[access_m]),
            ],
            dtype=np.float32,
        )

        token_batches.append(tokens)
        mask_batches.append(mask)
        task_feat_batches.append(task_feat)
        uav_feat_batches.append(uav_feat)

    token_tensor = torch.tensor(np.asarray(token_batches), dtype=torch.float32, device=device)
    mask_tensor = torch.tensor(np.asarray(mask_batches), dtype=torch.float32, device=device)
    task_feat_tensor = torch.tensor(np.asarray(task_feat_batches), dtype=torch.float32, device=device)
    uav_feat_tensor = torch.tensor(np.asarray(uav_feat_batches), dtype=torch.float32, device=device)

    return token_tensor, mask_tensor, task_feat_tensor, uav_feat_tensor


# def forward_ratio_branch(policy: ProposedPolicy, state: Dict[str, Any], access_assoc: np.ndarray) -> torch.Tensor:
def forward_ratio_branch(
    policy: ProposedPolicy,
    state: Dict[str, Any],
    access_assoc: np.ndarray,
    use_hard_min: bool = True,
) -> torch.Tensor:
    """
    Differentiable forward aligned with deployment path:
        encoder -> fusion -> ratio_head(prior_ratio=...)
    output:
        offload_ratio_pred: [K]
    """
    if policy.encoder is None or policy.fusion_net is None or policy.ratio_head is None:
        raise RuntimeError("policy encoder/fusion_net/ratio_head is None in full-stage1 training.")

    token_tensor, mask_tensor, task_feat_tensor, uav_feat_tensor = build_ratio_branch_inputs(
        state=state,
        access_assoc=access_assoc,
        max_candidates=policy.max_candidates,
        device=policy.device,
    )

    ratio_prior_np = policy._compute_ratio_prior(state, access_assoc)
    prior_tensor = torch.tensor(
        ratio_prior_np,
        dtype=torch.float32,
        device=policy.device,
    ).unsqueeze(-1)

    encoded_tokens, pooled_context = policy.encoder(token_tensor, mask_tensor)
    fused_feature = policy.fusion_net(
        topo_context=pooled_context,
        task_feat=task_feat_tensor,
        uav_feat=uav_feat_tensor,
    )

    # offload_ratio_pred = policy.ratio_head(
    #     fused_feature,
    #     prior_ratio=prior_tensor,
    #     temperature=1.0,
    #     hard_min=policy.ratio_floor,
    # ).squeeze(-1)

    # offload_ratio_pred = torch.clamp(
    #     offload_ratio_pred,
    #     min=policy.ratio_floor,
    #     max=policy.ratio_ceiling,
    # )
    # return offload_ratio_pred

    offload_ratio_pred = policy.ratio_head(
        fused_feature,
        prior_ratio=prior_tensor,
        temperature=1.0,
        hard_min=policy.ratio_floor if use_hard_min else None,
    ).squeeze(-1)

    if use_hard_min:
        offload_ratio_pred = torch.clamp(
            offload_ratio_pred,
            min=policy.ratio_floor,
            max=policy.ratio_ceiling,
        )
    else:
        # 训练 actor / ratio head 时不要在下界处截断梯度
        offload_ratio_pred = torch.clamp(
            offload_ratio_pred,
            min=1e-4,
            max=policy.ratio_ceiling,
        )

    return offload_ratio_pred


def build_stage1_ratio_teacher(
    state,
    access_assoc,
    min_ratio: float = 0.05,
    max_ratio: float = 0.35,
):
    """
    Stage-1 teacher for offloading ratio.
    Generates a simple feasible-biased ratio target for BC warm start.

    Args:
        state: environment raw_state dict
        access_assoc: [M, K] association matrix
        min_ratio: lower bound of offloading ratio
        max_ratio: upper bound of offloading ratio

    Returns:
        np.ndarray of shape [K], each in [min_ratio, max_ratio]
    """
    import numpy as np

    task_deadline = np.asarray(state["task_deadline"], dtype=np.float32)
    task_size = np.asarray(state["task_size"], dtype=np.float32)
    task_cycles = np.asarray(state["task_cycles"], dtype=np.float32)

    size_norm = task_size / (np.max(task_size) + 1e-8)
    cyc_norm = task_cycles / (np.max(task_cycles) + 1e-8)
    ddl_norm = task_deadline / (np.max(task_deadline) + 1e-8)

    urgency = 1.0 - ddl_norm

    # 更保守：降低对大任务/高时限紧迫度的放大量
    score = 0.35 * size_norm + 0.35 * cyc_norm + 0.30 * urgency

    # 压缩到 [0.05, 0.35]
    ratio = min_ratio + (max_ratio - min_ratio) * score

    # 再额外做一次温和压缩，避免过激 teacher
    ratio = 0.7 * ratio + 0.3 * min_ratio

    ratio = np.clip(ratio, min_ratio, max_ratio)
    return ratio.astype(np.float32)


# =========================================================
# Replay buffer
# =========================================================
class FullReplayBuffer:
    def __init__(self, obs_dim: int, action_dim: int, capacity: int = 10000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.obs_dim = obs_dim
        self.action_dim = action_dim

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        next_action: np.ndarray,
        done: float,
    ):
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

    def sample(self, batch_size: int):
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

    def __len__(self):
        return len(self.buffer)


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
# Evaluation
# =========================================================
@torch.no_grad()
def evaluate_full_policy_rollout(env: MultiUavMecEnv, policy: ProposedPolicy, seed: int = 72):
    """
    Evaluation used by Stage-1 warm-start logging.

    In addition to reward/delay/energy/feasible ratio, this function records
    offloading-ratio statistics and local/neighbor execution split. These fields
    are needed for stage-wise convergence plots and for explaining why Stage-2
    starts from a strong warm-start policy.
    """
    obs = env.reset(seed=seed)
    done = False

    total_reward = 0.0
    total_delay = 0.0
    total_energy = 0.0
    total_deadline_violation = 0.0
    feasible_count = 0
    step_count = 0

    total_ratio_violation = 0.0
    total_assoc_violation = 0.0
    total_schedule_violation = 0.0
    total_candidate_violation = 0.0
    total_bw_violation = 0.0
    total_cpu_violation = 0.0
    total_rate_violation = 0.0
    total_nan_count = 0.0

    ratio_values = []
    local_exec_count = 0
    neighbor_exec_count = 0
    sched_count = 0

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

        # Ratio and scheduling diagnostics before env.step(...)
        if "offload_ratio" in action:
            ratio_arr = np.asarray(action["offload_ratio"], dtype=np.float32).reshape(-1)
            if ratio_arr.size > 0:
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

        report = info["report"]
        metrics = info["metrics"]

        total_reward += float(reward)
        total_delay += float(metrics["delay_sys"])
        total_energy += float(metrics["energy_sys"])
        total_deadline_violation += float(report.get("deadline_violation", 0.0))
        feasible_count += int(report.get("ok", False))
        step_count += 1

        total_ratio_violation += float(report.get("ratio_violation", 0.0))
        total_assoc_violation += float(report.get("assoc_violation", 0.0))
        total_schedule_violation += float(report.get("schedule_violation", 0.0))
        total_candidate_violation += float(report.get("candidate_violation", 0.0))
        total_bw_violation += float(report.get("bw_violation", 0.0))
        total_cpu_violation += float(report.get("cpu_violation", 0.0))
        total_rate_violation += float(report.get("rate_violation", 0.0))
        total_nan_count += float(report.get("nan_count", 0.0))

    denom = max(step_count, 1)
    ratio_np = np.asarray(ratio_values, dtype=np.float32)
    if ratio_np.size == 0:
        ratio_mean = ratio_std = ratio_min = ratio_max = float("nan")
    else:
        ratio_mean = float(np.mean(ratio_np))
        ratio_std = float(np.std(ratio_np))
        ratio_min = float(np.min(ratio_np))
        ratio_max = float(np.max(ratio_np))

    sched_denom = max(sched_count, 1)

    return {
        "episode_reward": total_reward,
        "system_cost": -total_reward,
        "avg_delay": total_delay / denom,
        "avg_energy": total_energy / denom,
        "avg_deadline_violation": total_deadline_violation / denom,
        "feasible_ratio": feasible_count / denom,
        "num_steps": step_count,

        "avg_ratio_violation": total_ratio_violation / denom,
        "avg_assoc_violation": total_assoc_violation / denom,
        "avg_schedule_violation": total_schedule_violation / denom,
        "avg_candidate_violation": total_candidate_violation / denom,
        "avg_bw_violation": total_bw_violation / denom,
        "avg_cpu_violation": total_cpu_violation / denom,
        "avg_rate_violation": total_rate_violation / denom,
        "avg_nan_count": total_nan_count / denom,

        "ratio_mean": ratio_mean,
        "ratio_std": ratio_std,
        "ratio_min": ratio_min,
        "ratio_max": ratio_max,
        "local_exec_ratio": local_exec_count / sched_denom,
        "neighbor_exec_ratio": neighbor_exec_count / sched_denom,
    }

def load_if_exists(module, path, name, device="cpu"):
    if module is None:
        return
    if os.path.exists(path):
        module.load_state_dict(torch.load(path, map_location=device))
        print(f"Loaded {name} from: {path}")
    else:
        print(f"WARNING: {name} checkpoint not found: {path}")


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


@contextmanager
def policy_eval_mode(policy: ProposedPolicy):
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


def _configure_torch_runtime(device: str) -> None:
    if device == "cpu":
        torch.set_num_threads(_env_int("TORCH_NUM_THREADS", 1))
        torch.set_num_interop_threads(_env_int("TORCH_INTEROP_THREADS", 1))
    elif device.startswith("cuda"):
        torch.backends.cudnn.benchmark = True


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
        # Keep Stage-1 and Stage-2 episode distributions consistent.
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
    print("DEBUG deadline_scale =", env.deadline_scale)
    print("obs_dim:", obs_dim)
    print("actor_raw_action_dim:", actor_raw_action_dim)
    print("critic_action_dim:", critic_action_dim)

    # -------------------------------------------------
    # Full proposed policy
    # -------------------------------------------------
    actor = MLPActor(
        obs_dim=obs_dim,
        action_dim=actor_raw_action_dim,
        hidden_dim=256,
    ).to(device)

    # Encoder hyperparameters.
    # For the "Proposed w/o Transformer" ablation, set NUM_LAYERS=0.
    # This keeps candidate-token construction, fusion, ratio head, scheduling
    # structure, analytical solvers, and training protocol unchanged, while
    # removing all self-attention/FFN Transformer blocks.
    embed_dim = _env_int("EMBED_DIM", 128)
    num_heads = _env_int("NUM_HEADS", 4)
    ff_hidden_dim = _env_int("FF_HIDDEN_DIM", 256)
    num_layers = _env_int("NUM_LAYERS", 2)

    policy = build_default_proposed_policy(
        state=state,
        actor_net=actor,
        device=device,
        embed_dim=embed_dim,
        num_heads=num_heads,
        ff_hidden_dim=ff_hidden_dim,
        num_layers=num_layers,
    )

    # Warm-start old stable modules except ratio head
    # load_if_exists(policy.actor_net, "checkpoints/proposed_full_stage2_best_actor.pth", "old stage2 actor", device=device)
    # load_if_exists(policy.encoder, "checkpoints/proposed_full_stage2_best_encoder.pth", "old stage2 encoder", device=device)
    # load_if_exists(policy.fusion_net, "checkpoints/proposed_full_stage2_best_fusion.pth", "old stage2 fusion", device=device)
    # do NOT load old ratio_head because architecture changed

    target_policy = copy.deepcopy(policy)

    # -------------------------------------------------
    # Critic for full-action skeleton
    # -------------------------------------------------
    critic = MLPCritic(
        obs_dim=obs_dim,
        action_dim=critic_action_dim,
        hidden_dim=256,
    ).to(device)

    target_critic = copy.deepcopy(critic).to(device)

    # -------------------------------------------------
    # Optimizers
    # -------------------------------------------------
    learnable_modules = []
    if policy.actor_net is not None:
        learnable_modules += list(policy.actor_net.parameters())
    if policy.encoder is not None:
        learnable_modules += list(policy.encoder.parameters())
    if policy.fusion_net is not None:
        learnable_modules += list(policy.fusion_net.parameters())
    if policy.ratio_head is not None:
        learnable_modules += list(policy.ratio_head.parameters())

    actor_opt = optim.Adam(learnable_modules, lr=_env_float("STAGE1_ACTOR_LR", 1e-3))
    critic_opt = optim.Adam(critic.parameters(), lr=_env_float("STAGE1_CRITIC_LR", 1e-3))

    mse_loss = nn.MSELoss()

    # -------------------------------------------------
    # Replay buffer
    # -------------------------------------------------
    buffer = FullReplayBuffer(
        obs_dim=obs_dim,
        action_dim=critic_action_dim,
        capacity=20000,
    )

    # -------------------------------------------------
    # Hyperparameters
    # -------------------------------------------------
    gamma = _env_float("GAMMA", 0.99)
    tau = _env_float("TAU", 0.005)
    batch_size = _env_int("BATCH_SIZE", 32)
    num_episodes = _env_int("NUM_EPISODES", 200)

    # actor_move_sched_coef = 1.0
    # ratio_bc_coef = 1.0
    # actor_l2_coef = 1e-4

    actor_move_sched_coef = _env_float("ACTOR_MOVE_SCHED_COEF", 1.0)
    ratio_bc_coef = _env_float("RATIO_BC_COEF", 6.0)
    actor_l2_coef = _env_float("ACTOR_L2_COEF", 1e-5)

    # movement+scheduling masks
    M = int(state["M"])
    K = int(state["K"])
    move_sched_mask = np.zeros((actor_raw_action_dim,), dtype=np.float32)
    move_sched_mask[: M + M] = 1.0
    move_sched_mask[M + M + K:] = 1.0
    move_sched_mask_t = torch.tensor(move_sched_mask, dtype=torch.float32, device=device).unsqueeze(0)

    # -------------------------------------------------
    # Log / save dirs for Stage-1 convergence
    # -------------------------------------------------
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    default_run_name = f"proposed_full_stage1_converge_k{K}_seed{base_seed}_{timestamp}"
    run_name = _env_str("RUN_NAME", default_run_name)
    result_dir = os.path.join("results", "convergence_training", run_name)
    os.makedirs(result_dir, exist_ok=True)

    train_log_path = os.path.join(result_dir, "train_log.csv")
    eval_log_path = os.path.join(result_dir, "eval_log.csv")
    config_path = os.path.join(result_dir, "config.json")

    eval_seed = _env_int("EVAL_SEED", 999)
    eval_every = _env_int("EVAL_EVERY", 5)

    run_config = {
        "stage": "stage1_warm_start",
        "run_name": run_name,
        "seed": base_seed,
        "eval_seed": eval_seed,
        "device": device,
        "M": M,
        "K": K,
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
        "gamma": gamma,
        "tau": tau,
        "actor_move_sched_coef": actor_move_sched_coef,
        "ratio_bc_coef": ratio_bc_coef,
        "actor_l2_coef": actor_l2_coef,
        "embed_dim": embed_dim,
        "num_heads": num_heads,
        "ff_hidden_dim": ff_hidden_dim,
        "num_layers": num_layers,
        "encoder_ablation": "meanpool_no_self_attention" if num_layers == 0 else "transformer",
        "safe_mobility_projection": True,
        "collision_margin": 1e-5,
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)

    with open(train_log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "episode",
            "episode_reward",
            "system_cost",
            "avg_move_sched_loss",
            "avg_ratio_loss",
            "avg_total_actor_loss",
            "avg_critic_loss",
            "avg_q_value",
            "avg_target_q_value",
            "avg_td_abs_error",
            "avg_ratio_pred_mean",
            "avg_ratio_pred_std",
            "avg_teacher_ratio_mean",
            "avg_teacher_base_ratio_mean",
            "steps",
            "buffer_size",
            "critic_updates",
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

    best_policy_state = copy.deepcopy(policy)
    best_eval_reward = -float("inf")

    print("Stage-1 converge config:")
    print(f"  num_episodes={num_episodes}, episode_length={env.episode_length}, batch_size={batch_size}")
    print(f"  actor_move_sched_coef={actor_move_sched_coef}, ratio_bc_coef={ratio_bc_coef}, actor_l2_coef={actor_l2_coef}")
    print(f"  encoder: embed_dim={embed_dim}, num_heads={num_heads}, ff_hidden_dim={ff_hidden_dim}, num_layers={num_layers}")
    if num_layers == 0:
        print("  [ABLATION] NUM_LAYERS=0: Transformer self-attention blocks are removed; encoder becomes token-wise projection + masked mean pooling.")
    print(f"  run_name={run_name}")
    print(f"  train_log_path={train_log_path}")
    print(f"  eval_log_path={eval_log_path}")

    # -------------------------------------------------
    # Training loop
    # -------------------------------------------------
    for ep in range(num_episodes):
        obs = env.reset(seed=base_seed + ep)
        done = False

        episode_reward = 0.0
        episode_move_sched_loss = 0.0
        episode_ratio_loss = 0.0
        episode_total_actor_loss = 0.0
        episode_critic_loss = 0.0
        episode_q_value = 0.0
        episode_target_q_value = 0.0
        episode_td_abs_error = 0.0
        episode_ratio_pred_mean = 0.0
        episode_ratio_pred_std = 0.0
        episode_teacher_ratio_mean = 0.0
        episode_teacher_base_ratio_mean = 0.0
        critic_update_count = 0
        step_count = 0

        while not done:
            state = obs["raw_state"]
            access_assoc = build_access_association(state)
            obs_vec = build_global_observation(state)

            # -----------------------------------------
            # Teacher action from placeholder
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
            # teacher_offload_ratio = np.asarray(
            #     teacher_action["offload_ratio"],
            #     dtype=np.float32,
            # )
            teacher_offload_ratio_base = np.asarray(
                teacher_action["offload_ratio"],
                dtype=np.float32,
            )

            teacher_offload_ratio = build_stage1_ratio_teacher(
                state=state,
                access_assoc=access_assoc,
            )

            teacher_offload_ratio_t = torch.tensor(
                teacher_offload_ratio,
                dtype=torch.float32,
                device=device,
            )

            # -----------------------------------------
            # Current full proposed action
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
                collision_margin=1e-5,
            )
            flat_action = flatten_full_action(action)

            # -----------------------------------------
            # Environment step
            # -----------------------------------------
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
            else:
                next_flat_action = np.zeros((critic_action_dim,), dtype=np.float32)

            buffer.add(
                obs=obs_vec,
                action=flat_action,
                reward=reward,
                next_obs=next_obs_vec,
                next_action=next_flat_action,
                done=float(done),
            )

            # -----------------------------------------
            # Actor / encoder / ratio-head BC update
            # -----------------------------------------
            obs_tensor = torch.tensor(obs_vec, dtype=torch.float32, device=device).unsqueeze(0)
            teacher_raw_target_t = torch.tensor(teacher_raw_target, dtype=torch.float32, device=device).unsqueeze(0)
            teacher_offload_ratio_t = torch.tensor(teacher_offload_ratio, dtype=torch.float32, device=device)

            raw_pred = policy.actor_net(obs_tensor)
            move_sched_pred = raw_pred * move_sched_mask_t
            move_sched_target = teacher_raw_target_t * move_sched_mask_t
            actor_move_sched_loss = mse_loss(move_sched_pred, move_sched_target)

            ratio_pred = forward_ratio_branch(
                policy=policy,
                state=state,
                access_assoc=access_assoc,
            )
            ratio_bc_loss = mse_loss(ratio_pred, teacher_offload_ratio_t)

            actor_l2_loss = (raw_pred ** 2).mean()
            total_actor_loss = (
                actor_move_sched_coef * actor_move_sched_loss
                + ratio_bc_coef * ratio_bc_loss
                + actor_l2_coef * actor_l2_loss
            )

            actor_opt.zero_grad()
            total_actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(learnable_modules, max_norm=5.0)
            actor_opt.step()

            # -----------------------------------------
            # Critic skeleton update
            # -----------------------------------------
            if len(buffer) >= batch_size:
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
                torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=5.0)
                critic_opt.step()

                td_abs_error = torch.abs(q_val - y).mean()

                episode_critic_loss += float(critic_loss.item())
                episode_q_value += float(q_val.mean().item())
                episode_target_q_value += float(y.mean().item())
                episode_td_abs_error += float(td_abs_error.item())
                critic_update_count += 1

                soft_update_policy(target_policy, policy, tau=tau)
                soft_update(target_critic, critic, tau=tau)

            # if step_count < 2:
            #     with torch.no_grad():
            #         print("[STAGE1 DEBUG]")
            #         print("  ratio_pred[:5] =", ratio_pred.detach().cpu().numpy()[:5])
            #         print("  teacher_raw[:5] =", teacher_offload_ratio[:5])

            if step_count < 2:
                with torch.no_grad():
                    ratio_prior_dbg = policy._compute_ratio_prior(state, access_assoc)
                    print("[STAGE1 DEBUG]")
                    print("  ratio_pred[:5]        =", ratio_pred.detach().cpu().numpy()[:5])
                    print("  teacher_base[:5]      =", teacher_offload_ratio_base[:5])
                    print("  teacher_enhanced[:5]  =", teacher_offload_ratio[:5])
                    print("  ratio_prior[:5]       =", ratio_prior_dbg[:5])
                    print("  mean_ratio_pred       =", float(ratio_pred.mean().item()))
                    print("  mean_teacher_ratio    =", float(np.mean(teacher_offload_ratio)))

            obs = next_obs
            episode_reward += float(reward)
            episode_move_sched_loss += float(actor_move_sched_loss.item())
            episode_ratio_loss += float(ratio_bc_loss.item())
            episode_total_actor_loss += float(total_actor_loss.item())
            with torch.no_grad():
                episode_ratio_pred_mean += float(ratio_pred.mean().item())
                episode_ratio_pred_std += float(ratio_pred.std(unbiased=False).item())
            episode_teacher_ratio_mean += float(np.mean(teacher_offload_ratio))
            episode_teacher_base_ratio_mean += float(np.mean(teacher_offload_ratio_base))
            step_count += 1

        denom_steps = max(step_count, 1)
        denom_critic = max(critic_update_count, 1)
        avg_move_sched_loss = episode_move_sched_loss / denom_steps
        avg_ratio_loss = episode_ratio_loss / denom_steps
        avg_total_actor_loss = episode_total_actor_loss / denom_steps
        avg_critic_loss = episode_critic_loss / denom_critic if critic_update_count > 0 else np.nan
        avg_q_value = episode_q_value / denom_critic if critic_update_count > 0 else np.nan
        avg_target_q_value = episode_target_q_value / denom_critic if critic_update_count > 0 else np.nan
        avg_td_abs_error = episode_td_abs_error / denom_critic if critic_update_count > 0 else np.nan
        avg_ratio_pred_mean = episode_ratio_pred_mean / denom_steps
        avg_ratio_pred_std = episode_ratio_pred_std / denom_steps
        avg_teacher_ratio_mean = episode_teacher_ratio_mean / denom_steps
        avg_teacher_base_ratio_mean = episode_teacher_base_ratio_mean / denom_steps

        print(f"\nEpisode {ep}")
        print("episode_reward:", episode_reward)
        print("avg_move_sched_loss:", avg_move_sched_loss)
        print("avg_ratio_loss:", avg_ratio_loss)
        print("avg_total_actor_loss:", avg_total_actor_loss)
        print("avg_critic_loss:", avg_critic_loss)
        print("avg_td_abs_error:", avg_td_abs_error)
        print("avg_ratio_pred_mean:", avg_ratio_pred_mean)
        print("avg_teacher_ratio_mean:", avg_teacher_ratio_mean)
        print("steps:", step_count)
        print("buffer_size:", len(buffer))
        print("critic_updates:", critic_update_count)

        with open(train_log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                ep,
                episode_reward,
                -episode_reward,
                avg_move_sched_loss,
                avg_ratio_loss,
                avg_total_actor_loss,
                avg_critic_loss,
                avg_q_value,
                avg_target_q_value,
                avg_td_abs_error,
                avg_ratio_pred_mean,
                avg_ratio_pred_std,
                avg_teacher_ratio_mean,
                avg_teacher_base_ratio_mean,
                step_count,
                len(buffer),
                critic_update_count,
            ])

        # -----------------------------------------
        # Evaluation
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
                seed=eval_seed,
            )

            with policy_eval_mode(policy):
                eval_result = evaluate_full_policy_rollout_with_extra_constraints(
                    env=eval_env,
                    policy=policy,
                    seed=eval_seed,
                )

            print("DEBUG eval deadline_scale =", eval_env.deadline_scale)
            print("DEBUG eval local_cpu =", eval_env.task_local_cpu_min, eval_env.task_local_cpu_max)

            print("\n==============================")
            print(f"Full Proposed Stage-1 Eval @ episode {ep}")
            print("==============================")
            print("episode_reward:", eval_result["episode_reward"])
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
            print("ratio_mean:", eval_result["ratio_mean"])
            print("ratio_std:", eval_result["ratio_std"])
            print("ratio_min:", eval_result["ratio_min"])
            print("ratio_max:", eval_result["ratio_max"])
            print("local_exec_ratio:", eval_result["local_exec_ratio"])
            print("neighbor_exec_ratio:", eval_result["neighbor_exec_ratio"])

            with open(eval_log_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
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
                ])

            if eval_result["episode_reward"] > best_eval_reward:
                best_eval_reward = eval_result["episode_reward"]
                best_policy_state = copy.deepcopy(policy)

    # -------------------------------------------------
    # Save checkpoints
    # -------------------------------------------------

    # os.makedirs("checkpoints", exist_ok=True)

    # if best_policy_state.actor_net is not None:
    #     torch.save(
    #         best_policy_state.actor_net.state_dict(),
    #         "checkpoints/proposed_full_stage1_fix_best_actor.pth",
    #     )
    # if best_policy_state.encoder is not None:
    #     torch.save(
    #         best_policy_state.encoder.state_dict(),
    #         "checkpoints/proposed_full_stage1_fix_best_encoder.pth",
    #     )
    # if best_policy_state.fusion_net is not None:
    #     torch.save(
    #         best_policy_state.fusion_net.state_dict(),
    #         "checkpoints/proposed_full_stage1_fix_best_fusion.pth",
    #     )
    # if best_policy_state.ratio_head is not None:
    #     torch.save(
    #         best_policy_state.ratio_head.state_dict(),
    #         "checkpoints/proposed_full_stage1_fix_best_ratio_head.pth",
    #     )

    # torch.save(
    #     critic.state_dict(),
    #     "checkpoints/proposed_full_stage1_fix_critic.pth",
    # )

    # print("\nFull proposed Stage-1 training finished successfully.")
    # print("Saved:")
    # print("  checkpoints/proposed_full_stage1_fix_best_actor.pth")
    # print("  checkpoints/proposed_full_stage1_fix_best_encoder.pth")
    # print("  checkpoints/proposed_full_stage1_fix_best_fusion.pth")
    # print("  checkpoints/proposed_full_stage1_fix_best_ratio_head.pth")
    # print("  checkpoints/proposed_full_stage1_fix_critic.pth")

    os.makedirs("checkpoints", exist_ok=True)

    ckpt_prefix = _env_str("CKPT_PREFIX", f"proposed_full_stage1_converge_k{K}_seed{base_seed}")

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

    print("\nFull proposed Stage-1 converge warm-start training finished successfully.")
    print("Saved:")
    print(f"  checkpoints/{ckpt_prefix}_best_actor.pth")
    print(f"  checkpoints/{ckpt_prefix}_best_encoder.pth")
    print(f"  checkpoints/{ckpt_prefix}_best_fusion.pth")
    print(f"  checkpoints/{ckpt_prefix}_best_ratio_head.pth")
    print(f"  checkpoints/{ckpt_prefix}_critic.pth")
    print("Logs:")
    print(f"  {train_log_path}")
    print(f"  {eval_log_path}")
    print(f"  {config_path}")


if __name__ == "__main__":
    main()




# # # 它要做的事是：
# # # 用 ProposedPolicy 替代旧的简化策略接口
# # # actor 训练目标先保留简单版
# # # encoder / fusion / ratio head 也纳入网络参数
# # # 先把“完整 Proposed 方法训练骨架”跑通

# # import copy
# # from collections import deque
# # from typing import Any, Dict, List, Optional, Tuple

# # import numpy as np
# # import torch
# # import torch.nn as nn
# # import torch.optim as optim

# # from env.mec_env import MultiUavMecEnv
# # from env.association import build_access_association
# # from model.mlp_actor import MLPActor
# # from model.mlp_critic import MLPCritic
# # from model.proposed_obs_builder import build_global_observation, get_observation_dim
# # from policy.proposed_placeholder_policy import generate_proposed_placeholder_action
# # from policy.proposed_policy import build_default_proposed_policy, ProposedPolicy

# import copy
# import csv
# import json
# import os
# import time
# from contextlib import contextmanager
# from collections import deque
# from typing import Any, Dict, List, Optional, Tuple

# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim

# from env.mec_env import MultiUavMecEnv
# from env.association import build_access_association
# from model.mlp_actor import MLPActor
# from model.mlp_critic import MLPCritic
# from model.proposed_obs_builder import build_global_observation, get_observation_dim
# from policy.proposed_placeholder_policy import generate_proposed_placeholder_action
# from policy.proposed_policy import build_default_proposed_policy, ProposedPolicy


# EPS = 1e-8


# # =========================================================
# # Helpers: dimensions / flatten / soft update
# # =========================================================
# def get_actor_raw_action_dim(state: Dict[str, Any]) -> int:
#     """
#     Raw actor output layout:
#         [ move_dist(M), move_angle(M), offload_ratio(K), sched_score(K*M) ]
#     """
#     M = int(state["M"])
#     K = int(state["K"])
#     return M + M + K + K * M


# def get_full_action_dim(state: Dict[str, Any]) -> int:
#     """
#     Full action layout used by critic:
#         [ move_dist(M),
#           move_angle(M),
#           offload_ratio(K),
#           sched_beta(K*M*M),
#           bandwidth_alloc(M*K),
#           cpu_alloc(M*K) ]
#     """
#     M = int(state["M"])
#     K = int(state["K"])
#     return M + M + K + K * M * M + M * K + M * K


# def flatten_full_action(action: Dict[str, Any]) -> np.ndarray:
#     move_dist = np.asarray(action["move_dist"], dtype=np.float32).reshape(-1)
#     move_angle = np.asarray(action["move_angle"], dtype=np.float32).reshape(-1)
#     offload_ratio = np.asarray(action["offload_ratio"], dtype=np.float32).reshape(-1)
#     sched_beta = np.asarray(action["sched_beta"], dtype=np.float32).reshape(-1)
#     bandwidth_alloc = np.asarray(action["bandwidth_alloc"], dtype=np.float32).reshape(-1)
#     cpu_alloc = np.asarray(action["cpu_alloc"], dtype=np.float32).reshape(-1)

#     return np.concatenate(
#         [
#             move_dist,
#             move_angle,
#             offload_ratio,
#             sched_beta,
#             bandwidth_alloc,
#             cpu_alloc,
#         ],
#         axis=0,
#     ).astype(np.float32)


# def soft_update(target_net: nn.Module, source_net: nn.Module, tau: float = 0.005):
#     for tp, sp in zip(target_net.parameters(), source_net.parameters()):
#         tp.data.copy_(tau * sp.data + (1.0 - tau) * tp.data)


# def soft_update_policy(target_policy: ProposedPolicy, source_policy: ProposedPolicy, tau: float = 0.005):
#     if target_policy.actor_net is not None and source_policy.actor_net is not None:
#         soft_update(target_policy.actor_net, source_policy.actor_net, tau=tau)

#     if target_policy.encoder is not None and source_policy.encoder is not None:
#         soft_update(target_policy.encoder, source_policy.encoder, tau=tau)

#     if target_policy.fusion_net is not None and source_policy.fusion_net is not None:
#         soft_update(target_policy.fusion_net, source_policy.fusion_net, tau=tau)

#     if target_policy.ratio_head is not None and source_policy.ratio_head is not None:
#         soft_update(target_policy.ratio_head, source_policy.ratio_head, tau=tau)


# # =========================================================
# # Helpers: teacher target
# # =========================================================
# def build_teacher_raw_target(state, access_assoc, teacher_action):
#     """
#     Convert env-style teacher action dict into raw-action target format.

#     Raw actor layout:
#         [ move_dist(M), move_angle(M), offload_ratio(K), sched_score(K*M) ]
#     """
#     M = int(state["M"])
#     K = int(state["K"])
#     max_speed = float(state["max_speed"])
#     delta_t = float(state["delta_t"])
#     max_move = max(max_speed * delta_t, EPS)

#     move_dist = np.asarray(teacher_action["move_dist"], dtype=np.float32).reshape(M)
#     move_angle = np.asarray(teacher_action["move_angle"], dtype=np.float32).reshape(M)
#     offload_ratio = np.asarray(teacher_action["offload_ratio"], dtype=np.float32).reshape(K)
#     sched_beta = np.asarray(teacher_action["sched_beta"], dtype=np.float32).reshape(K, M, M)

#     move_dist_target = 2.0 * (move_dist / max_move) - 1.0
#     move_dist_target = np.clip(move_dist_target, -1.0, 1.0)

#     move_angle_target = move_angle / np.pi
#     move_angle_target = np.clip(move_angle_target, -1.0, 1.0)

#     offload_target = 2.0 * offload_ratio - 1.0
#     offload_target = np.clip(offload_target, -1.0, 1.0)

#     sched_score_target = -np.ones((K, M), dtype=np.float32)
#     for k in range(K):
#         access_m = int(np.argmax(access_assoc[:, k]))
#         chosen_js = np.where(sched_beta[k, access_m, :] > 0.5)[0]
#         if len(chosen_js) == 1:
#             sched_score_target[k, int(chosen_js[0])] = 1.0

#     raw_target = np.concatenate(
#         [
#             move_dist_target.reshape(-1),
#             move_angle_target.reshape(-1),
#             offload_target.reshape(-1),
#             sched_score_target.reshape(-1),
#         ],
#         axis=0,
#     ).astype(np.float32)

#     return raw_target


# def split_actor_raw_action(raw_action: torch.Tensor, M: int, K: int):
#     """
#     raw layout:
#         [ move_dist(M), move_angle(M), offload_ratio(K), sched_score(K*M) ]
#     """
#     p = 0
#     move_dist_raw = raw_action[:, p:p + M]
#     p += M

#     move_angle_raw = raw_action[:, p:p + M]
#     p += M

#     offload_raw = raw_action[:, p:p + K]
#     p += K

#     sched_raw = raw_action[:, p:p + K * M].reshape(-1, K, M)

#     return move_dist_raw, move_angle_raw, offload_raw, sched_raw


# # =========================================================
# # Helpers: state feature extraction for ratio branch
# # =========================================================
# def _safe_get(state: Dict[str, Any], key: str, default=None):
#     return state[key] if key in state else default


# def _to_numpy(x, dtype=np.float32):
#     return np.asarray(x, dtype=dtype)


# def _extract_neighbors(state: Dict[str, Any]) -> List[List[int]]:
#     neighbors = _safe_get(state, "neighbors", None)
#     if neighbors is None:
#         M = int(state["M"])
#         return [[] for _ in range(M)]
#     return neighbors


# def _extract_uav_positions(state: Dict[str, Any]) -> np.ndarray:
#     for key in ["uav_pos", "uav_positions", "q", "uav_xy"]:
#         if key in state:
#             arr = np.asarray(state[key], dtype=np.float32)
#             if arr.ndim == 2 and arr.shape[1] >= 2:
#                 return arr[:, :2]
#     raise KeyError("Cannot find UAV position array in state.")


# def _extract_task_size(state: Dict[str, Any]) -> np.ndarray:
#     for key in ["task_size", "D", "D_k", "task_data_size"]:
#         if key in state:
#             return _to_numpy(state[key])
#     raise KeyError("Cannot find task size in state.")


# def _extract_task_cycles(state: Dict[str, Any]) -> np.ndarray:
#     for key in ["task_cycles", "C", "C_k", "task_cpu_cycles"]:
#         if key in state:
#             return _to_numpy(state[key])
#     raise KeyError("Cannot find task cycles in state.")


# def _extract_task_deadline(state: Dict[str, Any]) -> np.ndarray:
#     for key in ["task_deadline", "deadline", "tau_max", "task_max_delay"]:
#         if key in state:
#             return _to_numpy(state[key])
#     raise KeyError("Cannot find task deadline in state.")


# def _extract_available_cpu(state: Dict[str, Any]) -> np.ndarray:
#     for key in ["uav_available_cpu", "available_cpu", "uav_cpu_avail", "uav_cpu_max"]:
#         if key in state:
#             return _to_numpy(state[key])
#     M = int(state["M"])
#     return np.ones(M, dtype=np.float32)


# def _extract_workload(state: Dict[str, Any]) -> np.ndarray:
#     for key in ["uav_workload", "workload", "uav_load", "queue_len"]:
#         if key in state:
#             return _to_numpy(state[key])
#     M = int(state["M"])
#     return np.zeros(M, dtype=np.float32)


# def _extract_tx_power(state: Dict[str, Any], access_m: int) -> float:
#     for key in ["uav_tx_power", "P_tx", "tx_power"]:
#         if key in state:
#             val = state[key]
#             arr = np.asarray(val, dtype=np.float32)
#             if arr.ndim == 0:
#                 return float(arr)
#             return float(arr[access_m])
#     return 1.0


# def _extract_backhaul_bw(state: Dict[str, Any]) -> float:
#     for key in ["backhaul_bandwidth", "B_bh", "bh_bandwidth"]:
#         if key in state:
#             return float(state[key])
#     return 1.0


# def _extract_noise_power(state: Dict[str, Any]) -> float:
#     for key in ["noise_power", "sigma2", "noise_var"]:
#         if key in state:
#             return float(state[key])
#     return 1e-9


# def _compute_a2a_gain_and_rate(state: Dict[str, Any], access_m: int, exec_j: int) -> Tuple[float, float]:
#     if exec_j == access_m:
#         return 1.0, 1e6

#     uav_pos = _extract_uav_positions(state)
#     p_m = uav_pos[access_m]
#     p_j = uav_pos[exec_j]

#     dist = float(np.linalg.norm(p_m - p_j))
#     dist = max(dist, 1.0)

#     beta0 = float(_safe_get(state, "beta0", 1.0))
#     gain = beta0 * (dist ** -2)

#     B_bh = _extract_backhaul_bw(state)
#     P_tx = _extract_tx_power(state, access_m)
#     sigma2 = _extract_noise_power(state)

#     rate = B_bh * np.log2(1.0 + P_tx * gain / max(sigma2, EPS))
#     return float(gain), float(rate)


# def build_ratio_branch_inputs(
#     state: Dict[str, Any],
#     access_assoc: np.ndarray,
#     max_candidates: int,
#     device: str,
# ):
#     """
#     Build:
#       token_tensor: [K, Nc, 7]
#       mask_tensor:  [K, Nc]
#       task_feat:    [K, 3]
#       uav_feat:     [K, 4]
#     """
#     M = int(state["M"])
#     K = int(state["K"])

#     task_size = _extract_task_size(state)
#     task_cycles = _extract_task_cycles(state)
#     task_deadline = _extract_task_deadline(state)

#     available_cpu = _extract_available_cpu(state)
#     workload = _extract_workload(state)
#     uav_pos = _extract_uav_positions(state)
#     neighbors = _extract_neighbors(state)

#     token_batches = []
#     mask_batches = []
#     task_feat_batches = []
#     uav_feat_batches = []

#     for k in range(K):
#         access_m = int(np.argmax(access_assoc[:, k]))
#         candidates = [access_m] + list(neighbors[access_m])
#         candidates = candidates[:max_candidates]

#         tokens = np.zeros((max_candidates, 7), dtype=np.float32)
#         mask = np.zeros((max_candidates,), dtype=np.float32)

#         for idx, j in enumerate(candidates):
#             h_a2a, r_a2a = _compute_a2a_gain_and_rate(state, access_m, j)
#             tokens[idx] = np.asarray(
#                 [
#                     float(h_a2a),
#                     float(r_a2a),
#                     float(workload[j]),
#                     float(available_cpu[j]),
#                     float(task_size[k]),
#                     float(task_cycles[k]),
#                     float(task_deadline[k]),
#                 ],
#                 dtype=np.float32,
#             )
#             mask[idx] = 1.0

#         task_feat = np.asarray(
#             [
#                 float(task_size[k]),
#                 float(task_cycles[k]),
#                 float(task_deadline[k]),
#             ],
#             dtype=np.float32,
#         )

#         uav_feat = np.asarray(
#             [
#                 float(uav_pos[access_m, 0]),
#                 float(uav_pos[access_m, 1]),
#                 float(available_cpu[access_m]),
#                 float(workload[access_m]),
#             ],
#             dtype=np.float32,
#         )

#         token_batches.append(tokens)
#         mask_batches.append(mask)
#         task_feat_batches.append(task_feat)
#         uav_feat_batches.append(uav_feat)

#     token_tensor = torch.tensor(np.asarray(token_batches), dtype=torch.float32, device=device)
#     mask_tensor = torch.tensor(np.asarray(mask_batches), dtype=torch.float32, device=device)
#     task_feat_tensor = torch.tensor(np.asarray(task_feat_batches), dtype=torch.float32, device=device)
#     uav_feat_tensor = torch.tensor(np.asarray(uav_feat_batches), dtype=torch.float32, device=device)

#     return token_tensor, mask_tensor, task_feat_tensor, uav_feat_tensor


# # def forward_ratio_branch(policy: ProposedPolicy, state: Dict[str, Any], access_assoc: np.ndarray) -> torch.Tensor:
# def forward_ratio_branch(
#     policy: ProposedPolicy,
#     state: Dict[str, Any],
#     access_assoc: np.ndarray,
#     use_hard_min: bool = True,
# ) -> torch.Tensor:
#     """
#     Differentiable forward aligned with deployment path:
#         encoder -> fusion -> ratio_head(prior_ratio=...)
#     output:
#         offload_ratio_pred: [K]
#     """
#     if policy.encoder is None or policy.fusion_net is None or policy.ratio_head is None:
#         raise RuntimeError("policy encoder/fusion_net/ratio_head is None in full-stage1 training.")

#     token_tensor, mask_tensor, task_feat_tensor, uav_feat_tensor = build_ratio_branch_inputs(
#         state=state,
#         access_assoc=access_assoc,
#         max_candidates=policy.max_candidates,
#         device=policy.device,
#     )

#     ratio_prior_np = policy._compute_ratio_prior(state, access_assoc)
#     prior_tensor = torch.tensor(
#         ratio_prior_np,
#         dtype=torch.float32,
#         device=policy.device,
#     ).unsqueeze(-1)

#     encoded_tokens, pooled_context = policy.encoder(token_tensor, mask_tensor)
#     fused_feature = policy.fusion_net(
#         topo_context=pooled_context,
#         task_feat=task_feat_tensor,
#         uav_feat=uav_feat_tensor,
#     )

#     # offload_ratio_pred = policy.ratio_head(
#     #     fused_feature,
#     #     prior_ratio=prior_tensor,
#     #     temperature=1.0,
#     #     hard_min=policy.ratio_floor,
#     # ).squeeze(-1)

#     # offload_ratio_pred = torch.clamp(
#     #     offload_ratio_pred,
#     #     min=policy.ratio_floor,
#     #     max=policy.ratio_ceiling,
#     # )
#     # return offload_ratio_pred

#     offload_ratio_pred = policy.ratio_head(
#         fused_feature,
#         prior_ratio=prior_tensor,
#         temperature=1.0,
#         hard_min=policy.ratio_floor if use_hard_min else None,
#     ).squeeze(-1)

#     if use_hard_min:
#         offload_ratio_pred = torch.clamp(
#             offload_ratio_pred,
#             min=policy.ratio_floor,
#             max=policy.ratio_ceiling,
#         )
#     else:
#         # 训练 actor / ratio head 时不要在下界处截断梯度
#         offload_ratio_pred = torch.clamp(
#             offload_ratio_pred,
#             min=1e-4,
#             max=policy.ratio_ceiling,
#         )

#     return offload_ratio_pred


# def build_stage1_ratio_teacher(
#     state,
#     access_assoc,
#     min_ratio: float = 0.05,
#     max_ratio: float = 0.35,
# ):
#     """
#     Stage-1 teacher for offloading ratio.
#     Generates a simple feasible-biased ratio target for BC warm start.

#     Args:
#         state: environment raw_state dict
#         access_assoc: [M, K] association matrix
#         min_ratio: lower bound of offloading ratio
#         max_ratio: upper bound of offloading ratio

#     Returns:
#         np.ndarray of shape [K], each in [min_ratio, max_ratio]
#     """
#     import numpy as np

#     task_deadline = np.asarray(state["task_deadline"], dtype=np.float32)
#     task_size = np.asarray(state["task_size"], dtype=np.float32)
#     task_cycles = np.asarray(state["task_cycles"], dtype=np.float32)

#     size_norm = task_size / (np.max(task_size) + 1e-8)
#     cyc_norm = task_cycles / (np.max(task_cycles) + 1e-8)
#     ddl_norm = task_deadline / (np.max(task_deadline) + 1e-8)

#     urgency = 1.0 - ddl_norm

#     # 更保守：降低对大任务/高时限紧迫度的放大量
#     score = 0.35 * size_norm + 0.35 * cyc_norm + 0.30 * urgency

#     # 压缩到 [0.05, 0.35]
#     ratio = min_ratio + (max_ratio - min_ratio) * score

#     # 再额外做一次温和压缩，避免过激 teacher
#     ratio = 0.7 * ratio + 0.3 * min_ratio

#     ratio = np.clip(ratio, min_ratio, max_ratio)
#     return ratio.astype(np.float32)


# # =========================================================
# # Replay buffer
# # =========================================================
# class FullReplayBuffer:
#     def __init__(self, obs_dim: int, action_dim: int, capacity: int = 10000):
#         self.capacity = capacity
#         self.buffer = deque(maxlen=capacity)
#         self.obs_dim = obs_dim
#         self.action_dim = action_dim

#     def add(
#         self,
#         obs: np.ndarray,
#         action: np.ndarray,
#         reward: float,
#         next_obs: np.ndarray,
#         next_action: np.ndarray,
#         done: float,
#     ):
#         self.buffer.append(
#             {
#                 "obs": np.asarray(obs, dtype=np.float32),
#                 "action": np.asarray(action, dtype=np.float32),
#                 "reward": float(reward),
#                 "next_obs": np.asarray(next_obs, dtype=np.float32),
#                 "next_action": np.asarray(next_action, dtype=np.float32),
#                 "done": float(done),
#             }
#         )

#     def sample(self, batch_size: int):
#         idx = np.random.choice(len(self.buffer), size=batch_size, replace=False)
#         batch = [self.buffer[i] for i in idx]

#         return {
#             "obs": np.stack([b["obs"] for b in batch], axis=0),
#             "action": np.stack([b["action"] for b in batch], axis=0),
#             "reward": np.asarray([b["reward"] for b in batch], dtype=np.float32).reshape(-1, 1),
#             "next_obs": np.stack([b["next_obs"] for b in batch], axis=0),
#             "next_action": np.stack([b["next_action"] for b in batch], axis=0),
#             "done": np.asarray([b["done"] for b in batch], dtype=np.float32).reshape(-1, 1),
#         }

#     def __len__(self):
#         return len(self.buffer)


# # =========================================================
# # Safe mobility projection
# # =========================================================
# def _get_uav_positions_from_state(state: Dict[str, Any], M: int) -> np.ndarray:
#     """
#     Robustly extract UAV 2D positions from the raw state.
#     Returns shape [M, 2]. If the key is unavailable, returns None.
#     """
#     candidate_keys = [
#         "uav_pos",
#         "uav_positions",
#         "uav_xy",
#         "q_uav",
#         "q",
#         "positions",
#     ]

#     for key in candidate_keys:
#         if key in state:
#             arr = np.asarray(state[key], dtype=np.float32)
#             if arr.ndim == 2 and arr.shape[0] >= M and arr.shape[1] >= 2:
#                 return arr[:M, :2].copy()

#     # Some codebases store x/y separately.
#     if "uav_x" in state and "uav_y" in state:
#         x = np.asarray(state["uav_x"], dtype=np.float32).reshape(-1)
#         y = np.asarray(state["uav_y"], dtype=np.float32).reshape(-1)
#         if x.size >= M and y.size >= M:
#             return np.stack([x[:M], y[:M]], axis=1).astype(np.float32)

#     return None


# def _next_positions(q_xy: np.ndarray, move_dist: np.ndarray, move_angle: np.ndarray) -> np.ndarray:
#     dx = move_dist * np.cos(move_angle)
#     dy = move_dist * np.sin(move_angle)
#     return q_xy + np.stack([dx, dy], axis=1)


# def _all_pairwise_safe(pos_xy: np.ndarray, min_dist: float) -> bool:
#     M = pos_xy.shape[0]
#     for i in range(M):
#         for j in range(i + 1, M):
#             if np.linalg.norm(pos_xy[i] - pos_xy[j]) < min_dist:
#                 return False
#     return True


# def apply_safe_mobility_projection(
#     action: Dict[str, Any],
#     state: Dict[str, Any],
#     safety_margin: float = 1.0,
#     collision_margin: float = 0.0,
#     binary_iters: int = 24,
# ) -> Dict[str, Any]:
#     """
#     Project the mobility action into a physically safer flight region.

#     This is a lightweight safety shield applied after actor inference and before
#     env.step(...). It preserves the actor's move whenever feasible, but shrinks
#     the movement distance if the proposed next position would violate boundary
#     or inter-UAV safety distance.

#     It does NOT alter offloading, scheduling, bandwidth, or CPU allocation.
#     """
#     if "move_dist" not in action or "move_angle" not in action:
#         return action

#     M = int(state.get("M", len(np.asarray(action["move_dist"]).reshape(-1))))
#     move_dist = np.asarray(action["move_dist"], dtype=np.float32).reshape(-1).copy()
#     move_angle = np.asarray(action["move_angle"], dtype=np.float32).reshape(-1).copy()

#     if move_dist.size < M or move_angle.size < M:
#         return action

#     q_xy = _get_uav_positions_from_state(state, M)
#     if q_xy is None:
#         # If positions cannot be found, at least enforce the motion budget.
#         max_move = max(float(state.get("max_speed", 0.0)) * float(state.get("delta_t", 1.0)), EPS)
#         safe_action = dict(action)
#         safe_action["move_dist"] = np.clip(move_dist[:M], 0.0, max_move).astype(np.float32)
#         safe_action["move_angle"] = move_angle[:M].astype(np.float32)
#         return safe_action

#     area_size = float(state.get("area_size", 100.0))
#     max_speed = float(state.get("max_speed", 15.0))
#     delta_t = float(state.get("delta_t", 1.0))
#     max_move = max(max_speed * delta_t, EPS)
#     min_dist = float(state.get("min_uav_distance", 0.0)) + float(collision_margin)

#     # Basic numeric cleanup and motion-budget clipping.
#     move_dist = np.nan_to_num(move_dist[:M], nan=0.0, posinf=max_move, neginf=0.0)
#     move_angle = np.nan_to_num(move_angle[:M], nan=0.0, posinf=np.pi, neginf=-np.pi)
#     move_dist = np.clip(move_dist, 0.0, max_move).astype(np.float32)
#     move_angle = np.clip(move_angle, -np.pi, np.pi).astype(np.float32)

#     low = float(safety_margin)
#     high = float(area_size - safety_margin)

#     # Boundary projection: shrink each UAV's distance along its actor-proposed direction.
#     for m in range(M):
#         d0 = float(move_dist[m])
#         theta = float(move_angle[m])
#         if d0 <= EPS:
#             continue

#         proposed = q_xy[m] + np.array([d0 * np.cos(theta), d0 * np.sin(theta)], dtype=np.float32)
#         if low <= proposed[0] <= high and low <= proposed[1] <= high:
#             continue

#         lo, hi = 0.0, d0
#         for _ in range(binary_iters):
#             mid = 0.5 * (lo + hi)
#             p_mid = q_xy[m] + np.array([mid * np.cos(theta), mid * np.sin(theta)], dtype=np.float32)
#             if low <= p_mid[0] <= high and low <= p_mid[1] <= high:
#                 lo = mid
#             else:
#                 hi = mid
#         move_dist[m] = max(0.0, lo)

#     # Collision projection: if the set of next positions is unsafe, shrink all moves together.
#     if min_dist > 0.0:
#         proposed_pos = _next_positions(q_xy, move_dist, move_angle)
#         if not _all_pairwise_safe(proposed_pos, min_dist):
#             # If current positions are already unsafe, shrinking cannot fully fix it.
#             if _all_pairwise_safe(q_xy, min_dist):
#                 lo, hi = 0.0, 1.0
#                 for _ in range(binary_iters):
#                     alpha = 0.5 * (lo + hi)
#                     pos_mid = _next_positions(q_xy, move_dist * alpha, move_angle)
#                     if _all_pairwise_safe(pos_mid, min_dist):
#                         lo = alpha
#                     else:
#                         hi = alpha
#                 move_dist = move_dist * lo
#             else:
#                 move_dist = np.zeros_like(move_dist, dtype=np.float32)

#     safe_action = dict(action)
#     safe_action["move_dist"] = move_dist.astype(np.float32)
#     safe_action["move_angle"] = move_angle.astype(np.float32)
#     return safe_action


# class SafeMobilityPolicyWrapper:
#     """
#     Wrapper used only for evaluation, so evaluate_full_policy_rollout(...)
#     computes reward / delay / feasible_ratio under exactly the same safe
#     mobility projection used in training execution.
#     """
#     def __init__(self, policy, safety_margin: float = 1.0, collision_margin: float = 0.0):
#         self.policy = policy
#         self.safety_margin = safety_margin
#         self.collision_margin = collision_margin

#     def act(self, state, access_assoc, deterministic=True, return_aux=False):
#         action = self.policy.act(
#             state=state,
#             access_assoc=access_assoc,
#             deterministic=deterministic,
#             return_aux=return_aux,
#         )
#         if return_aux:
#             # If a future policy returns (action, aux), project only the action part.
#             action_part, aux = action
#             action_part = apply_safe_mobility_projection(
#                 action_part,
#                 state,
#                 safety_margin=self.safety_margin,
#                 collision_margin=self.collision_margin,
#             )
#             return action_part, aux

#         return apply_safe_mobility_projection(
#             action,
#             state,
#             safety_margin=self.safety_margin,
#             collision_margin=self.collision_margin,
#         )


# # =========================================================
# # Evaluation with extra feasibility diagnostics
# # =========================================================
# def _to_float_or_none(x):
#     """Convert scalar-like values to float. Return None if conversion is unsafe."""
#     if x is None:
#         return None
#     if isinstance(x, (bool, np.bool_)):
#         return float(x)
#     if isinstance(x, (int, float, np.integer, np.floating)):
#         return float(x)
#     try:
#         arr = np.asarray(x)
#         if arr.size == 1:
#             return float(arr.reshape(-1)[0])
#     except Exception:
#         return None
#     return None


# def _recursive_find_numeric(info: Any, aliases) -> float:
#     """
#     Robustly search an info dict for a violation scalar.
#     This supports both top-level keys and nested reports, e.g.
#     info["feasibility_report"]["collision_violation"].
#     """
#     if info is None:
#         return 0.0

#     if isinstance(info, dict):
#         for key in aliases:
#             if key in info:
#                 value = _to_float_or_none(info[key])
#                 if value is not None:
#                     return value
#         for value in info.values():
#             found = _recursive_find_numeric(value, aliases)
#             if found != 0.0:
#                 return found

#     elif isinstance(info, (list, tuple)):
#         for value in info:
#             found = _recursive_find_numeric(value, aliases)
#             if found != 0.0:
#                 return found

#     return 0.0


# @torch.no_grad()
# def evaluate_full_policy_rollout_with_extra_constraints(env, policy, seed: int):
#     """
#     Keep the original Stage-1 evaluation output, and additionally log hidden
#     feasibility terms that are often not included in the old eval_log.csv.

#     The reward/delay/feasible_ratio and the extra diagnostics are both computed
#     under the same safe mobility projection.
#     """
#     safe_policy = SafeMobilityPolicyWrapper(policy, safety_margin=1.0, collision_margin=1e-5)
#     result = evaluate_full_policy_rollout(env=env, policy=safe_policy, seed=seed)

#     obs = env.reset(seed=seed)
#     done = False
#     n = 0

#     extra_sum = {
#         "avg_move_violation": 0.0,
#         "avg_boundary_violation": 0.0,
#         "avg_collision_violation": 0.0,
#         "avg_battery_violation": 0.0,
#     }

#     alias_map = {
#         "avg_move_violation": [
#             "move_violation", "motion_violation", "movement_violation",
#             "avg_move_violation", "avg_motion_violation",
#         ],
#         "avg_boundary_violation": [
#             "boundary_violation", "out_of_boundary", "out_of_bounds",
#             "avg_boundary_violation",
#         ],
#         "avg_collision_violation": [
#             "collision_violation", "collision", "safe_distance_violation",
#             "distance_violation", "avg_collision_violation",
#         ],
#         "avg_battery_violation": [
#             "battery_violation", "energy_violation", "battery_negative",
#             "avg_battery_violation", "avg_energy_violation",
#         ],
#     }

#     while not done:
#         state = obs["raw_state"]
#         access_assoc = build_access_association(state)
#         action = safe_policy.act(
#             state=state,
#             access_assoc=access_assoc,
#             deterministic=True,
#             return_aux=False,
#         )
#         obs, _, done, info = env.step(action)

#         for out_key, aliases in alias_map.items():
#             extra_sum[out_key] += _recursive_find_numeric(info, aliases)

#         n += 1

#     denom = max(n, 1)
#     for key in extra_sum:
#         result[key] = extra_sum[key] / denom

#     return result



# # =========================================================
# # Evaluation
# # =========================================================
# @torch.no_grad()
# def evaluate_full_policy_rollout(env: MultiUavMecEnv, policy: ProposedPolicy, seed: int = 72):
#     """
#     Evaluation used by Stage-1 warm-start logging.

#     In addition to reward/delay/energy/feasible ratio, this function records
#     offloading-ratio statistics and local/neighbor execution split. These fields
#     are needed for stage-wise convergence plots and for explaining why Stage-2
#     starts from a strong warm-start policy.
#     """
#     obs = env.reset(seed=seed)
#     done = False

#     total_reward = 0.0
#     total_delay = 0.0
#     total_energy = 0.0
#     total_deadline_violation = 0.0
#     feasible_count = 0
#     step_count = 0

#     total_ratio_violation = 0.0
#     total_assoc_violation = 0.0
#     total_schedule_violation = 0.0
#     total_candidate_violation = 0.0
#     total_bw_violation = 0.0
#     total_cpu_violation = 0.0
#     total_rate_violation = 0.0
#     total_nan_count = 0.0

#     ratio_values = []
#     local_exec_count = 0
#     neighbor_exec_count = 0
#     sched_count = 0

#     while not done:
#         state = obs["raw_state"]
#         access_assoc = build_access_association(state)
#         M = int(state["M"])
#         K = int(state["K"])

#         action = policy.act(
#             state=state,
#             access_assoc=access_assoc,
#             deterministic=True,
#             return_aux=False,
#         )

#         # Ratio and scheduling diagnostics before env.step(...)
#         if "offload_ratio" in action:
#             ratio_arr = np.asarray(action["offload_ratio"], dtype=np.float32).reshape(-1)
#             if ratio_arr.size > 0:
#                 ratio_values.extend(ratio_arr.tolist())

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

#         obs, reward, done, info = env.step(action)

#         report = info["report"]
#         metrics = info["metrics"]

#         total_reward += float(reward)
#         total_delay += float(metrics["delay_sys"])
#         total_energy += float(metrics["energy_sys"])
#         total_deadline_violation += float(report.get("deadline_violation", 0.0))
#         feasible_count += int(report.get("ok", False))
#         step_count += 1

#         total_ratio_violation += float(report.get("ratio_violation", 0.0))
#         total_assoc_violation += float(report.get("assoc_violation", 0.0))
#         total_schedule_violation += float(report.get("schedule_violation", 0.0))
#         total_candidate_violation += float(report.get("candidate_violation", 0.0))
#         total_bw_violation += float(report.get("bw_violation", 0.0))
#         total_cpu_violation += float(report.get("cpu_violation", 0.0))
#         total_rate_violation += float(report.get("rate_violation", 0.0))
#         total_nan_count += float(report.get("nan_count", 0.0))

#     denom = max(step_count, 1)
#     ratio_np = np.asarray(ratio_values, dtype=np.float32)
#     if ratio_np.size == 0:
#         ratio_mean = ratio_std = ratio_min = ratio_max = float("nan")
#     else:
#         ratio_mean = float(np.mean(ratio_np))
#         ratio_std = float(np.std(ratio_np))
#         ratio_min = float(np.min(ratio_np))
#         ratio_max = float(np.max(ratio_np))

#     sched_denom = max(sched_count, 1)

#     return {
#         "episode_reward": total_reward,
#         "system_cost": -total_reward,
#         "avg_delay": total_delay / denom,
#         "avg_energy": total_energy / denom,
#         "avg_deadline_violation": total_deadline_violation / denom,
#         "feasible_ratio": feasible_count / denom,
#         "num_steps": step_count,

#         "avg_ratio_violation": total_ratio_violation / denom,
#         "avg_assoc_violation": total_assoc_violation / denom,
#         "avg_schedule_violation": total_schedule_violation / denom,
#         "avg_candidate_violation": total_candidate_violation / denom,
#         "avg_bw_violation": total_bw_violation / denom,
#         "avg_cpu_violation": total_cpu_violation / denom,
#         "avg_rate_violation": total_rate_violation / denom,
#         "avg_nan_count": total_nan_count / denom,

#         "ratio_mean": ratio_mean,
#         "ratio_std": ratio_std,
#         "ratio_min": ratio_min,
#         "ratio_max": ratio_max,
#         "local_exec_ratio": local_exec_count / sched_denom,
#         "neighbor_exec_ratio": neighbor_exec_count / sched_denom,
#     }

# def load_if_exists(module, path, name, device="cpu"):
#     if module is None:
#         return
#     if os.path.exists(path):
#         module.load_state_dict(torch.load(path, map_location=device))
#         print(f"Loaded {name} from: {path}")
#     else:
#         print(f"WARNING: {name} checkpoint not found: {path}")


# # =========================================================
# # Runtime configuration helpers
# # =========================================================
# def _env_int(name: str, default: int) -> int:
#     val = os.environ.get(name)
#     return int(val) if val not in (None, "") else int(default)


# def _env_float(name: str, default: float) -> float:
#     val = os.environ.get(name)
#     return float(val) if val not in (None, "") else float(default)


# def _env_str(name: str, default: str) -> str:
#     val = os.environ.get(name)
#     return str(val) if val not in (None, "") else str(default)


# def _env_bool(name: str, default: bool = False) -> bool:
#     val = os.environ.get(name)
#     if val is None:
#         return bool(default)
#     return str(val).strip().lower() in {"1", "true", "yes", "y", "on"}


# @contextmanager
# def policy_eval_mode(policy: ProposedPolicy):
#     modules = [
#         getattr(policy, "actor_net", None),
#         getattr(policy, "encoder", None),
#         getattr(policy, "fusion_net", None),
#         getattr(policy, "ratio_head", None),
#     ]
#     modules = [m for m in modules if m is not None]
#     old_modes = [m.training for m in modules]
#     for m in modules:
#         m.eval()
#     try:
#         yield
#     finally:
#         for m, old in zip(modules, old_modes):
#             m.train(old)


# def _configure_torch_runtime(device: str) -> None:
#     if device == "cpu":
#         torch.set_num_threads(_env_int("TORCH_NUM_THREADS", 1))
#         torch.set_num_interop_threads(_env_int("TORCH_INTEROP_THREADS", 1))
#     elif device.startswith("cuda"):
#         torch.backends.cudnn.benchmark = True


# # =========================================================
# # Main
# # =========================================================
# def main():
#     requested_device = _env_str("DEVICE", "auto").lower()
#     if requested_device == "cpu":
#         device = "cpu"
#     elif requested_device.startswith("cuda"):
#         device = requested_device if torch.cuda.is_available() else "cpu"
#     else:
#         device = "cuda" if torch.cuda.is_available() else "cpu"
#     _configure_torch_runtime(device)

#     base_seed = _env_int("SEED", 72)
#     torch.manual_seed(base_seed)
#     np.random.seed(base_seed)
#     if device.startswith("cuda"):
#         torch.cuda.manual_seed_all(base_seed)

#     print(f"device: {device}")
#     print(f"seed: {base_seed}")

#     # -------------------------------------------------
#     # Env config
#     # -------------------------------------------------
#     env = MultiUavMecEnv(
#         M=_env_int("M", 3),
#         K=_env_int("K", 16),
#         # Keep Stage-1 and Stage-2 episode distributions consistent.
#         episode_length=_env_int("EPISODE_LENGTH", 20),
#         cpu_mode=_env_str("CPU_MODE", "kkt"),
#         omega1=_env_float("OMEGA1", 50.0),
#         omega2=_env_float("OMEGA2", 1.0),
#         deadline_scale=_env_float("DEADLINE_SCALE", 5.0),
#         task_local_cpu_min=_env_float("TASK_LOCAL_CPU_MIN", 2.0e3),
#         task_local_cpu_max=_env_float("TASK_LOCAL_CPU_MAX", 6.0e3),
#         uav_energy_min=_env_float("UAV_ENERGY_MIN", 2600.0),
#         uav_energy_max=_env_float("UAV_ENERGY_MAX", 3800.0),
#         seed=base_seed,
#     )

#     obs = env.reset(seed=base_seed)
#     state = obs["raw_state"]

#     obs_dim = get_observation_dim(state)
#     actor_raw_action_dim = get_actor_raw_action_dim(state)
#     critic_action_dim = get_full_action_dim(state)
#     print("DEBUG deadline_scale =", env.deadline_scale)
#     print("obs_dim:", obs_dim)
#     print("actor_raw_action_dim:", actor_raw_action_dim)
#     print("critic_action_dim:", critic_action_dim)

#     # -------------------------------------------------
#     # Full proposed policy
#     # -------------------------------------------------
#     actor = MLPActor(
#         obs_dim=obs_dim,
#         action_dim=actor_raw_action_dim,
#         hidden_dim=256,
#     ).to(device)

#     policy = build_default_proposed_policy(
#         state=state,
#         actor_net=actor,
#         device=device,
#         embed_dim=128,
#         num_heads=4,
#         ff_hidden_dim=256,
#         num_layers=2,
#     )

#     # Warm-start old stable modules except ratio head
#     # load_if_exists(policy.actor_net, "checkpoints/proposed_full_stage2_best_actor.pth", "old stage2 actor", device=device)
#     # load_if_exists(policy.encoder, "checkpoints/proposed_full_stage2_best_encoder.pth", "old stage2 encoder", device=device)
#     # load_if_exists(policy.fusion_net, "checkpoints/proposed_full_stage2_best_fusion.pth", "old stage2 fusion", device=device)
#     # do NOT load old ratio_head because architecture changed

#     target_policy = copy.deepcopy(policy)

#     # -------------------------------------------------
#     # Critic for full-action skeleton
#     # -------------------------------------------------
#     critic = MLPCritic(
#         obs_dim=obs_dim,
#         action_dim=critic_action_dim,
#         hidden_dim=256,
#     ).to(device)

#     target_critic = copy.deepcopy(critic).to(device)

#     # -------------------------------------------------
#     # Optimizers
#     # -------------------------------------------------
#     learnable_modules = []
#     if policy.actor_net is not None:
#         learnable_modules += list(policy.actor_net.parameters())
#     if policy.encoder is not None:
#         learnable_modules += list(policy.encoder.parameters())
#     if policy.fusion_net is not None:
#         learnable_modules += list(policy.fusion_net.parameters())
#     if policy.ratio_head is not None:
#         learnable_modules += list(policy.ratio_head.parameters())

#     actor_opt = optim.Adam(learnable_modules, lr=_env_float("STAGE1_ACTOR_LR", 1e-3))
#     critic_opt = optim.Adam(critic.parameters(), lr=_env_float("STAGE1_CRITIC_LR", 1e-3))

#     mse_loss = nn.MSELoss()

#     # -------------------------------------------------
#     # Replay buffer
#     # -------------------------------------------------
#     buffer = FullReplayBuffer(
#         obs_dim=obs_dim,
#         action_dim=critic_action_dim,
#         capacity=20000,
#     )

#     # -------------------------------------------------
#     # Hyperparameters
#     # -------------------------------------------------
#     gamma = _env_float("GAMMA", 0.99)
#     tau = _env_float("TAU", 0.005)
#     batch_size = _env_int("BATCH_SIZE", 32)
#     num_episodes = _env_int("NUM_EPISODES", 200)

#     # actor_move_sched_coef = 1.0
#     # ratio_bc_coef = 1.0
#     # actor_l2_coef = 1e-4

#     actor_move_sched_coef = _env_float("ACTOR_MOVE_SCHED_COEF", 1.0)
#     ratio_bc_coef = _env_float("RATIO_BC_COEF", 6.0)
#     actor_l2_coef = _env_float("ACTOR_L2_COEF", 1e-5)

#     # movement+scheduling masks
#     M = int(state["M"])
#     K = int(state["K"])
#     move_sched_mask = np.zeros((actor_raw_action_dim,), dtype=np.float32)
#     move_sched_mask[: M + M] = 1.0
#     move_sched_mask[M + M + K:] = 1.0
#     move_sched_mask_t = torch.tensor(move_sched_mask, dtype=torch.float32, device=device).unsqueeze(0)

#     # -------------------------------------------------
#     # Log / save dirs for Stage-1 convergence
#     # -------------------------------------------------
#     timestamp = time.strftime("%Y%m%d_%H%M%S")
#     default_run_name = f"proposed_full_stage1_converge_k{K}_seed{base_seed}_{timestamp}"
#     run_name = _env_str("RUN_NAME", default_run_name)
#     result_dir = os.path.join("results", "convergence_training", run_name)
#     os.makedirs(result_dir, exist_ok=True)

#     train_log_path = os.path.join(result_dir, "train_log.csv")
#     eval_log_path = os.path.join(result_dir, "eval_log.csv")
#     config_path = os.path.join(result_dir, "config.json")

#     eval_seed = _env_int("EVAL_SEED", 999)
#     eval_every = _env_int("EVAL_EVERY", 5)

#     run_config = {
#         "stage": "stage1_warm_start",
#         "run_name": run_name,
#         "seed": base_seed,
#         "eval_seed": eval_seed,
#         "device": device,
#         "M": M,
#         "K": K,
#         "episode_length": env.episode_length,
#         "cpu_mode": env.cpu_mode,
#         "omega1": env.omega1,
#         "omega2": env.omega2,
#         "deadline_scale": env.deadline_scale,
#         "task_local_cpu_min": env.task_local_cpu_min,
#         "task_local_cpu_max": env.task_local_cpu_max,
#         "uav_energy_min": env.uav_energy_min,
#         "uav_energy_max": env.uav_energy_max,
#         "num_episodes": num_episodes,
#         "batch_size": batch_size,
#         "gamma": gamma,
#         "tau": tau,
#         "actor_move_sched_coef": actor_move_sched_coef,
#         "ratio_bc_coef": ratio_bc_coef,
#         "actor_l2_coef": actor_l2_coef,
#         "safe_mobility_projection": True,
#         "collision_margin": 1e-5,
#     }
#     with open(config_path, "w", encoding="utf-8") as f:
#         json.dump(run_config, f, indent=2)

#     with open(train_log_path, "w", newline="", encoding="utf-8") as f:
#         writer = csv.writer(f)
#         writer.writerow([
#             "episode",
#             "episode_reward",
#             "system_cost",
#             "avg_move_sched_loss",
#             "avg_ratio_loss",
#             "avg_total_actor_loss",
#             "avg_critic_loss",
#             "avg_q_value",
#             "avg_target_q_value",
#             "avg_td_abs_error",
#             "avg_ratio_pred_mean",
#             "avg_ratio_pred_std",
#             "avg_teacher_ratio_mean",
#             "avg_teacher_base_ratio_mean",
#             "steps",
#             "buffer_size",
#             "critic_updates",
#         ])

#     with open(eval_log_path, "w", newline="", encoding="utf-8") as f:
#         writer = csv.writer(f)
#         writer.writerow([
#             "eval_episode",
#             "episode_reward",
#             "system_cost",
#             "avg_delay",
#             "avg_energy",
#             "avg_deadline_violation",
#             "feasible_ratio",
#             "num_steps",
#             "avg_ratio_violation",
#             "avg_assoc_violation",
#             "avg_schedule_violation",
#             "avg_candidate_violation",
#             "avg_bw_violation",
#             "avg_cpu_violation",
#             "avg_rate_violation",
#             "avg_nan_count",
#             "avg_move_violation",
#             "avg_boundary_violation",
#             "avg_collision_violation",
#             "avg_battery_violation",
#             "ratio_mean",
#             "ratio_std",
#             "ratio_min",
#             "ratio_max",
#             "local_exec_ratio",
#             "neighbor_exec_ratio",
#         ])

#     best_policy_state = copy.deepcopy(policy)
#     best_eval_reward = -float("inf")

#     print("Stage-1 converge config:")
#     print(f"  num_episodes={num_episodes}, episode_length={env.episode_length}, batch_size={batch_size}")
#     print(f"  actor_move_sched_coef={actor_move_sched_coef}, ratio_bc_coef={ratio_bc_coef}, actor_l2_coef={actor_l2_coef}")
#     print(f"  run_name={run_name}")
#     print(f"  train_log_path={train_log_path}")
#     print(f"  eval_log_path={eval_log_path}")

#     # -------------------------------------------------
#     # Training loop
#     # -------------------------------------------------
#     for ep in range(num_episodes):
#         obs = env.reset(seed=base_seed + ep)
#         done = False

#         episode_reward = 0.0
#         episode_move_sched_loss = 0.0
#         episode_ratio_loss = 0.0
#         episode_total_actor_loss = 0.0
#         episode_critic_loss = 0.0
#         episode_q_value = 0.0
#         episode_target_q_value = 0.0
#         episode_td_abs_error = 0.0
#         episode_ratio_pred_mean = 0.0
#         episode_ratio_pred_std = 0.0
#         episode_teacher_ratio_mean = 0.0
#         episode_teacher_base_ratio_mean = 0.0
#         critic_update_count = 0
#         step_count = 0

#         while not done:
#             state = obs["raw_state"]
#             access_assoc = build_access_association(state)
#             obs_vec = build_global_observation(state)

#             # -----------------------------------------
#             # Teacher action from placeholder
#             # -----------------------------------------
#             teacher_action = generate_proposed_placeholder_action(
#                 state=state,
#                 access_assoc=access_assoc,
#             )
#             teacher_raw_target = build_teacher_raw_target(
#                 state=state,
#                 access_assoc=access_assoc,
#                 teacher_action=teacher_action,
#             )
#             # teacher_offload_ratio = np.asarray(
#             #     teacher_action["offload_ratio"],
#             #     dtype=np.float32,
#             # )
#             teacher_offload_ratio_base = np.asarray(
#                 teacher_action["offload_ratio"],
#                 dtype=np.float32,
#             )

#             teacher_offload_ratio = build_stage1_ratio_teacher(
#                 state=state,
#                 access_assoc=access_assoc,
#             )

#             teacher_offload_ratio_t = torch.tensor(
#                 teacher_offload_ratio,
#                 dtype=torch.float32,
#                 device=device,
#             )

#             # -----------------------------------------
#             # Current full proposed action
#             # -----------------------------------------
#             action = policy.act(
#                 state=state,
#                 access_assoc=access_assoc,
#                 deterministic=True,
#                 return_aux=False,
#             )
#             action = apply_safe_mobility_projection(
#                 action,
#                 state,
#                 safety_margin=1.0,
#                 collision_margin=1e-5,
#             )
#             flat_action = flatten_full_action(action)

#             # -----------------------------------------
#             # Environment step
#             # -----------------------------------------
#             next_obs, reward, done, info = env.step(action)
#             next_obs_vec = build_global_observation(next_obs["raw_state"])

#             if not done:
#                 next_state = next_obs["raw_state"]
#                 next_access_assoc = build_access_association(next_state)
#                 next_action_target = target_policy.act(
#                     state=next_state,
#                     access_assoc=next_access_assoc,
#                     deterministic=True,
#                     return_aux=False,
#                 )
#                 next_action_target = apply_safe_mobility_projection(
#                     next_action_target,
#                     next_state,
#                     safety_margin=1.0,
#                     collision_margin=1e-5,
#                 )
#                 next_flat_action = flatten_full_action(next_action_target)
#             else:
#                 next_flat_action = np.zeros((critic_action_dim,), dtype=np.float32)

#             buffer.add(
#                 obs=obs_vec,
#                 action=flat_action,
#                 reward=reward,
#                 next_obs=next_obs_vec,
#                 next_action=next_flat_action,
#                 done=float(done),
#             )

#             # -----------------------------------------
#             # Actor / encoder / ratio-head BC update
#             # -----------------------------------------
#             obs_tensor = torch.tensor(obs_vec, dtype=torch.float32, device=device).unsqueeze(0)
#             teacher_raw_target_t = torch.tensor(teacher_raw_target, dtype=torch.float32, device=device).unsqueeze(0)
#             teacher_offload_ratio_t = torch.tensor(teacher_offload_ratio, dtype=torch.float32, device=device)

#             raw_pred = policy.actor_net(obs_tensor)
#             move_sched_pred = raw_pred * move_sched_mask_t
#             move_sched_target = teacher_raw_target_t * move_sched_mask_t
#             actor_move_sched_loss = mse_loss(move_sched_pred, move_sched_target)

#             ratio_pred = forward_ratio_branch(
#                 policy=policy,
#                 state=state,
#                 access_assoc=access_assoc,
#             )
#             ratio_bc_loss = mse_loss(ratio_pred, teacher_offload_ratio_t)

#             actor_l2_loss = (raw_pred ** 2).mean()
#             total_actor_loss = (
#                 actor_move_sched_coef * actor_move_sched_loss
#                 + ratio_bc_coef * ratio_bc_loss
#                 + actor_l2_coef * actor_l2_loss
#             )

#             actor_opt.zero_grad()
#             total_actor_loss.backward()
#             torch.nn.utils.clip_grad_norm_(learnable_modules, max_norm=5.0)
#             actor_opt.step()

#             # -----------------------------------------
#             # Critic skeleton update
#             # -----------------------------------------
#             if len(buffer) >= batch_size:
#                 batch = buffer.sample(batch_size)

#                 obs_b = torch.tensor(batch["obs"], dtype=torch.float32, device=device)
#                 action_b = torch.tensor(batch["action"], dtype=torch.float32, device=device)
#                 reward_b = torch.tensor(batch["reward"], dtype=torch.float32, device=device)
#                 next_obs_b = torch.tensor(batch["next_obs"], dtype=torch.float32, device=device)
#                 next_action_b = torch.tensor(batch["next_action"], dtype=torch.float32, device=device)
#                 done_b = torch.tensor(batch["done"], dtype=torch.float32, device=device)

#                 with torch.no_grad():
#                     target_q = target_critic(next_obs_b, next_action_b)
#                     y = reward_b + gamma * (1.0 - done_b) * target_q

#                 q_val = critic(obs_b, action_b)
#                 critic_loss = mse_loss(q_val, y)

#                 critic_opt.zero_grad()
#                 critic_loss.backward()
#                 torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=5.0)
#                 critic_opt.step()

#                 td_abs_error = torch.abs(q_val - y).mean()

#                 episode_critic_loss += float(critic_loss.item())
#                 episode_q_value += float(q_val.mean().item())
#                 episode_target_q_value += float(y.mean().item())
#                 episode_td_abs_error += float(td_abs_error.item())
#                 critic_update_count += 1

#                 soft_update_policy(target_policy, policy, tau=tau)
#                 soft_update(target_critic, critic, tau=tau)

#             # if step_count < 2:
#             #     with torch.no_grad():
#             #         print("[STAGE1 DEBUG]")
#             #         print("  ratio_pred[:5] =", ratio_pred.detach().cpu().numpy()[:5])
#             #         print("  teacher_raw[:5] =", teacher_offload_ratio[:5])

#             if step_count < 2:
#                 with torch.no_grad():
#                     ratio_prior_dbg = policy._compute_ratio_prior(state, access_assoc)
#                     print("[STAGE1 DEBUG]")
#                     print("  ratio_pred[:5]        =", ratio_pred.detach().cpu().numpy()[:5])
#                     print("  teacher_base[:5]      =", teacher_offload_ratio_base[:5])
#                     print("  teacher_enhanced[:5]  =", teacher_offload_ratio[:5])
#                     print("  ratio_prior[:5]       =", ratio_prior_dbg[:5])
#                     print("  mean_ratio_pred       =", float(ratio_pred.mean().item()))
#                     print("  mean_teacher_ratio    =", float(np.mean(teacher_offload_ratio)))

#             obs = next_obs
#             episode_reward += float(reward)
#             episode_move_sched_loss += float(actor_move_sched_loss.item())
#             episode_ratio_loss += float(ratio_bc_loss.item())
#             episode_total_actor_loss += float(total_actor_loss.item())
#             with torch.no_grad():
#                 episode_ratio_pred_mean += float(ratio_pred.mean().item())
#                 episode_ratio_pred_std += float(ratio_pred.std(unbiased=False).item())
#             episode_teacher_ratio_mean += float(np.mean(teacher_offload_ratio))
#             episode_teacher_base_ratio_mean += float(np.mean(teacher_offload_ratio_base))
#             step_count += 1

#         denom_steps = max(step_count, 1)
#         denom_critic = max(critic_update_count, 1)
#         avg_move_sched_loss = episode_move_sched_loss / denom_steps
#         avg_ratio_loss = episode_ratio_loss / denom_steps
#         avg_total_actor_loss = episode_total_actor_loss / denom_steps
#         avg_critic_loss = episode_critic_loss / denom_critic if critic_update_count > 0 else np.nan
#         avg_q_value = episode_q_value / denom_critic if critic_update_count > 0 else np.nan
#         avg_target_q_value = episode_target_q_value / denom_critic if critic_update_count > 0 else np.nan
#         avg_td_abs_error = episode_td_abs_error / denom_critic if critic_update_count > 0 else np.nan
#         avg_ratio_pred_mean = episode_ratio_pred_mean / denom_steps
#         avg_ratio_pred_std = episode_ratio_pred_std / denom_steps
#         avg_teacher_ratio_mean = episode_teacher_ratio_mean / denom_steps
#         avg_teacher_base_ratio_mean = episode_teacher_base_ratio_mean / denom_steps

#         print(f"\nEpisode {ep}")
#         print("episode_reward:", episode_reward)
#         print("avg_move_sched_loss:", avg_move_sched_loss)
#         print("avg_ratio_loss:", avg_ratio_loss)
#         print("avg_total_actor_loss:", avg_total_actor_loss)
#         print("avg_critic_loss:", avg_critic_loss)
#         print("avg_td_abs_error:", avg_td_abs_error)
#         print("avg_ratio_pred_mean:", avg_ratio_pred_mean)
#         print("avg_teacher_ratio_mean:", avg_teacher_ratio_mean)
#         print("steps:", step_count)
#         print("buffer_size:", len(buffer))
#         print("critic_updates:", critic_update_count)

#         with open(train_log_path, "a", newline="", encoding="utf-8") as f:
#             writer = csv.writer(f)
#             writer.writerow([
#                 ep,
#                 episode_reward,
#                 -episode_reward,
#                 avg_move_sched_loss,
#                 avg_ratio_loss,
#                 avg_total_actor_loss,
#                 avg_critic_loss,
#                 avg_q_value,
#                 avg_target_q_value,
#                 avg_td_abs_error,
#                 avg_ratio_pred_mean,
#                 avg_ratio_pred_std,
#                 avg_teacher_ratio_mean,
#                 avg_teacher_base_ratio_mean,
#                 step_count,
#                 len(buffer),
#                 critic_update_count,
#             ])

#         # -----------------------------------------
#         # Evaluation
#         # -----------------------------------------
#         if (ep + 1) % eval_every == 0 or ep == 0:
#             eval_env = MultiUavMecEnv(
#                 M=M,
#                 K=K,
#                 episode_length=env.episode_length,
#                 cpu_mode=env.cpu_mode,
#                 omega1=env.omega1,
#                 omega2=env.omega2,
#                 deadline_scale=env.deadline_scale,
#                 task_local_cpu_min=env.task_local_cpu_min,
#                 task_local_cpu_max=env.task_local_cpu_max,
#                 uav_energy_min=env.uav_energy_min,
#                 uav_energy_max=env.uav_energy_max,
#                 seed=eval_seed,
#             )

#             with policy_eval_mode(policy):
#                 eval_result = evaluate_full_policy_rollout_with_extra_constraints(
#                     env=eval_env,
#                     policy=policy,
#                     seed=eval_seed,
#                 )

#             print("DEBUG eval deadline_scale =", eval_env.deadline_scale)
#             print("DEBUG eval local_cpu =", eval_env.task_local_cpu_min, eval_env.task_local_cpu_max)

#             print("\n==============================")
#             print(f"Full Proposed Stage-1 Eval @ episode {ep}")
#             print("==============================")
#             print("episode_reward:", eval_result["episode_reward"])
#             print("avg_delay:", eval_result["avg_delay"])
#             print("avg_energy:", eval_result["avg_energy"])
#             print("avg_deadline_violation:", eval_result["avg_deadline_violation"])
#             print("feasible_ratio:", eval_result["feasible_ratio"])
#             print("num_steps:", eval_result["num_steps"])
#             print("avg_ratio_violation:", eval_result["avg_ratio_violation"])
#             print("avg_assoc_violation:", eval_result["avg_assoc_violation"])
#             print("avg_schedule_violation:", eval_result["avg_schedule_violation"])
#             print("avg_candidate_violation:", eval_result["avg_candidate_violation"])
#             print("avg_bw_violation:", eval_result["avg_bw_violation"])
#             print("avg_cpu_violation:", eval_result["avg_cpu_violation"])
#             print("avg_rate_violation:", eval_result["avg_rate_violation"])
#             print("avg_nan_count:", eval_result["avg_nan_count"])
#             print("avg_move_violation:", eval_result["avg_move_violation"])
#             print("avg_boundary_violation:", eval_result["avg_boundary_violation"])
#             print("avg_collision_violation:", eval_result["avg_collision_violation"])
#             print("avg_battery_violation:", eval_result["avg_battery_violation"])
#             print("ratio_mean:", eval_result["ratio_mean"])
#             print("ratio_std:", eval_result["ratio_std"])
#             print("ratio_min:", eval_result["ratio_min"])
#             print("ratio_max:", eval_result["ratio_max"])
#             print("local_exec_ratio:", eval_result["local_exec_ratio"])
#             print("neighbor_exec_ratio:", eval_result["neighbor_exec_ratio"])

#             with open(eval_log_path, "a", newline="", encoding="utf-8") as f:
#                 writer = csv.writer(f)
#                 writer.writerow([
#                     ep,
#                     eval_result["episode_reward"],
#                     eval_result["system_cost"],
#                     eval_result["avg_delay"],
#                     eval_result["avg_energy"],
#                     eval_result["avg_deadline_violation"],
#                     eval_result["feasible_ratio"],
#                     eval_result["num_steps"],
#                     eval_result["avg_ratio_violation"],
#                     eval_result["avg_assoc_violation"],
#                     eval_result["avg_schedule_violation"],
#                     eval_result["avg_candidate_violation"],
#                     eval_result["avg_bw_violation"],
#                     eval_result["avg_cpu_violation"],
#                     eval_result["avg_rate_violation"],
#                     eval_result["avg_nan_count"],
#                     eval_result["avg_move_violation"],
#                     eval_result["avg_boundary_violation"],
#                     eval_result["avg_collision_violation"],
#                     eval_result["avg_battery_violation"],
#                     eval_result["ratio_mean"],
#                     eval_result["ratio_std"],
#                     eval_result["ratio_min"],
#                     eval_result["ratio_max"],
#                     eval_result["local_exec_ratio"],
#                     eval_result["neighbor_exec_ratio"],
#                 ])

#             if eval_result["episode_reward"] > best_eval_reward:
#                 best_eval_reward = eval_result["episode_reward"]
#                 best_policy_state = copy.deepcopy(policy)

#     # -------------------------------------------------
#     # Save checkpoints
#     # -------------------------------------------------

#     # os.makedirs("checkpoints", exist_ok=True)

#     # if best_policy_state.actor_net is not None:
#     #     torch.save(
#     #         best_policy_state.actor_net.state_dict(),
#     #         "checkpoints/proposed_full_stage1_fix_best_actor.pth",
#     #     )
#     # if best_policy_state.encoder is not None:
#     #     torch.save(
#     #         best_policy_state.encoder.state_dict(),
#     #         "checkpoints/proposed_full_stage1_fix_best_encoder.pth",
#     #     )
#     # if best_policy_state.fusion_net is not None:
#     #     torch.save(
#     #         best_policy_state.fusion_net.state_dict(),
#     #         "checkpoints/proposed_full_stage1_fix_best_fusion.pth",
#     #     )
#     # if best_policy_state.ratio_head is not None:
#     #     torch.save(
#     #         best_policy_state.ratio_head.state_dict(),
#     #         "checkpoints/proposed_full_stage1_fix_best_ratio_head.pth",
#     #     )

#     # torch.save(
#     #     critic.state_dict(),
#     #     "checkpoints/proposed_full_stage1_fix_critic.pth",
#     # )

#     # print("\nFull proposed Stage-1 training finished successfully.")
#     # print("Saved:")
#     # print("  checkpoints/proposed_full_stage1_fix_best_actor.pth")
#     # print("  checkpoints/proposed_full_stage1_fix_best_encoder.pth")
#     # print("  checkpoints/proposed_full_stage1_fix_best_fusion.pth")
#     # print("  checkpoints/proposed_full_stage1_fix_best_ratio_head.pth")
#     # print("  checkpoints/proposed_full_stage1_fix_critic.pth")

#     os.makedirs("checkpoints", exist_ok=True)

#     ckpt_prefix = _env_str("CKPT_PREFIX", f"proposed_full_stage1_converge_k{K}_seed{base_seed}")

#     if best_policy_state.actor_net is not None:
#         torch.save(
#             best_policy_state.actor_net.state_dict(),
#             f"checkpoints/{ckpt_prefix}_best_actor.pth",
#         )
#     if best_policy_state.encoder is not None:
#         torch.save(
#             best_policy_state.encoder.state_dict(),
#             f"checkpoints/{ckpt_prefix}_best_encoder.pth",
#         )
#     if best_policy_state.fusion_net is not None:
#         torch.save(
#             best_policy_state.fusion_net.state_dict(),
#             f"checkpoints/{ckpt_prefix}_best_fusion.pth",
#         )
#     if best_policy_state.ratio_head is not None:
#         torch.save(
#             best_policy_state.ratio_head.state_dict(),
#             f"checkpoints/{ckpt_prefix}_best_ratio_head.pth",
#         )

#     torch.save(
#         critic.state_dict(),
#         f"checkpoints/{ckpt_prefix}_critic.pth",
#     )

#     print("\nFull proposed Stage-1 converge warm-start training finished successfully.")
#     print("Saved:")
#     print(f"  checkpoints/{ckpt_prefix}_best_actor.pth")
#     print(f"  checkpoints/{ckpt_prefix}_best_encoder.pth")
#     print(f"  checkpoints/{ckpt_prefix}_best_fusion.pth")
#     print(f"  checkpoints/{ckpt_prefix}_best_ratio_head.pth")
#     print(f"  checkpoints/{ckpt_prefix}_critic.pth")
#     print("Logs:")
#     print(f"  {train_log_path}")
#     print(f"  {eval_log_path}")
#     print(f"  {config_path}")


# if __name__ == "__main__":
#     main()

