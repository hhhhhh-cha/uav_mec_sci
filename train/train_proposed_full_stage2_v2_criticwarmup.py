# 加载 proposed_full_stage1_best_*.pth
# 组装完整 ProposedPolicy
# 保留 encoder / fusion / ratio head / actor
# Stage-2 稳定版：
#   1) critic action 输入归一化
#   2) Stage-2 critic 从零初始化，不加载 Stage-1 critic
#   3) learning_starts + batch_size=128
#   4) policy_delay=2 延迟 actor 更新，并且 actor 更新时冻结 critic 参数

import copy
import csv
import os
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

from train.train_proposed_full_stage1 import (
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


# =========================================================
# Runtime action shield / wrapper
# =========================================================
def apply_zero_move_shield(action: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Debug/safety shield: force UAV movement distance to zero.

    Important:
        Use this consistently in training execution, target action generation,
        actor surrogate action, and evaluation. Otherwise the replayed critic
        action and the actually executed action are inconsistent.
    """
    action = dict(action)
    action["move_dist"] = np.zeros_like(action["move_dist"], dtype=np.float32)
    return action


class ZeroMovePolicyWrapper:
    """Policy wrapper used only for evaluation with the same zero-move shield."""
    def __init__(self, policy):
        self.policy = policy

    def act(self, state, access_assoc, deterministic=True, return_aux=False):
        out = self.policy.act(
            state=state,
            access_assoc=access_assoc,
            deterministic=deterministic,
            return_aux=return_aux,
        )
        if return_aux:
            # If the underlying policy returns (action, aux), shield only action.
            action, aux = out
            return apply_zero_move_shield(action), aux
        return apply_zero_move_shield(out)


def zero_move_surrogate_tensor(full_action: torch.Tensor, M: int) -> torch.Tensor:
    """Force the move_dist slice of a differentiable full-action tensor to zero."""
    return torch.cat(
        [torch.zeros_like(full_action[:, :M]), full_action[:, M:]],
        dim=1,
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


def evaluate_full_policy_rollout_with_extra_constraints(env, policy, seed: int):
    """
    Keep the original Stage-1 evaluation output, and additionally log hidden
    feasibility terms that are often not included in the old eval_log.csv:
        - movement / motion violation
        - boundary violation
        - collision violation
        - battery / energy violation

    If a key does not exist in env.step(...).info, its average is recorded as 0.0.
    """
    wrapped_policy = ZeroMovePolicyWrapper(policy)
    result = evaluate_full_policy_rollout(env=env, policy=wrapped_policy, seed=seed)

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
        action = wrapped_policy.act(
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
# Main
# =========================================================
def main():
    device = "cpu"
    base_seed = 72
    torch.manual_seed(base_seed)
    np.random.seed(base_seed)

    # -------------------------------------------------
    # Env config
    # -------------------------------------------------
    env = MultiUavMecEnv(
        M=3,
        K=16,
        episode_length=20,
        cpu_mode="kkt",
        omega1=50.0,
        omega2=1.0,
        deadline_scale=5.0,
        task_local_cpu_min=2.0e3,
        task_local_cpu_max=6.0e3,
        uav_energy_min=2600.0,
        uav_energy_max=3800.0,
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
    # Do NOT load Stage-1 critic here: Stage-1 critic is only a skeleton / weak warm-start.
    load_if_exists(policy.actor_net, "checkpoints/proposed_full_stage1_fix_k16_ep40_best_actor.pth", "actor")
    load_if_exists(policy.encoder, "checkpoints/proposed_full_stage1_fix_k16_ep40_best_encoder.pth", "encoder")
    load_if_exists(policy.fusion_net, "checkpoints/proposed_full_stage1_fix_k16_ep40_best_fusion.pth", "fusion_net")
    load_if_exists(policy.ratio_head, "checkpoints/proposed_full_stage1_fix_k16_ep40_best_ratio_head.pth", "ratio_head")

    target_policy = copy.deepcopy(policy)

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
    learnable_params = []
    if policy.actor_net is not None:
        learnable_params += list(policy.actor_net.parameters())
    if policy.encoder is not None:
        learnable_params += list(policy.encoder.parameters())
    if policy.fusion_net is not None:
        learnable_params += list(policy.fusion_net.parameters())
    if policy.ratio_head is not None:
        learnable_params += list(policy.ratio_head.parameters())

    actor_opt = optim.Adam(learnable_params, lr=5e-5)
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
    gamma = 0.95
    tau = 0.005
    batch_size = 128
    learning_starts = 1000
    num_episodes = 500
    reward_scale = 1e-4
    policy_delay = 2
    grad_clip_norm = 1.0

    critic_only_episodes = 150

    actor_policy_coef = 0.003
    actor_move_sched_bc_coef = 0.5
    ratio_bc_coef = 0.2
    actor_l2_coef = 1e-5

    eval_every = 5
    global_update_step = 0

    # -------------------------------------------------
    # Log / save dirs
    # -------------------------------------------------
    run_name = "stable_proposed_full_stage2_seed72_criticwarmup150_bc05_policy0003_fixed_eval_zeromove"
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
        ])

    # movement + scheduling masks in raw actor output
    M = int(state["M"])
    K = int(state["K"])
    move_sched_mask = np.zeros((actor_raw_action_dim,), dtype=np.float32)
    move_sched_mask[: M + M] = 1.0
    move_sched_mask[M + M + K:] = 1.0
    move_sched_mask_t = torch.tensor(
        move_sched_mask, dtype=torch.float32, device=device
    ).unsqueeze(0)

    best_policy_state = copy.deepcopy(policy)
    best_eval_reward = -float("inf")

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
            action = apply_zero_move_shield(action)

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
                next_action_target = apply_zero_move_shield(next_action_target)
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

                raw_pred = policy.actor_net(obs_tensor)
                ratio_pred = forward_ratio_branch(
                    policy=policy,
                    state=state,
                    access_assoc=access_assoc,
                )

                surrogate_full_action = build_surrogate_full_action_tensor(
                    state=state,
                    access_assoc=access_assoc,
                    raw_action_pred=raw_pred,
                    ratio_pred=ratio_pred,
                    bandwidth_alloc_np=np.asarray(action["bandwidth_alloc"], dtype=np.float32),
                    cpu_alloc_np=np.asarray(action["cpu_alloc"], dtype=np.float32),
                )
                surrogate_full_action = zero_move_surrogate_tensor(surrogate_full_action, M)
                surrogate_full_action_norm = normalize_full_action_tensor(surrogate_full_action, state)

                # Freeze critic weights during actor update.
                for p_critic in critic.parameters():
                    p_critic.requires_grad_(False)

                actor_policy_loss = -critic(obs_tensor, surrogate_full_action_norm).mean()

                move_sched_pred = raw_pred * move_sched_mask_t
                move_sched_target = teacher_raw_target_t * move_sched_mask_t
                actor_move_sched_bc_loss = mse_loss(move_sched_pred, move_sched_target)

                ratio_bc_loss = mse_loss(ratio_pred, teacher_offload_ratio_t)
                actor_l2_loss = (raw_pred ** 2).mean()

                total_actor_loss = (
                    actor_policy_coef * actor_policy_loss
                    + actor_move_sched_bc_coef * actor_move_sched_bc_loss
                    + ratio_bc_coef * ratio_bc_loss
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
                episode_total_actor_loss += float(total_actor_loss.item())

                if step_count < 2:
                    with torch.no_grad():
                        print("[STAGE2 DEBUG]")
                        print("  ratio_pred[:5] =", ratio_pred.detach().cpu().numpy()[:5])
                        print("  teacher_raw[:5] =", teacher_offload_ratio[:5])

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
                M=3,
                K=16,
                episode_length=20,
                cpu_mode="kkt",
                omega1=50.0,
                omega2=1.0,
                deadline_scale=5.0,
                task_local_cpu_min=2.0e3,
                task_local_cpu_max=6.0e3,
                uav_energy_min=2600.0,
                uav_energy_max=3800.0,
                seed=999,
            )

            eval_result = evaluate_full_policy_rollout_with_extra_constraints(
                env=eval_env,
                policy=policy,
                seed=999,
            )

            print("\n==============================")
            print(f"Full Proposed Stage-2 Eval @ episode {ep}")
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

            with open(eval_log_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    ep,
                    eval_result["episode_reward"],
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
                ])

            if eval_result["episode_reward"] > best_eval_reward:
                best_eval_reward = eval_result["episode_reward"]
                best_policy_state = copy.deepcopy(policy)

    # -------------------------------------------------
    # Save best stage-2 checkpoints
    # -------------------------------------------------
    os.makedirs("checkpoints", exist_ok=True)

    ckpt_prefix = "proposed_full_stage2_stable_k16_seed72_criticwarmup150_bc05_fixed_eval_zeromove"

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
