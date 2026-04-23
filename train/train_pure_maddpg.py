from __future__ import annotations

import copy
import os
from typing import Any, Dict

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

from train.train_proposed_full_stage1 import (
    EPS,
    FullReplayBuffer,
    get_full_action_dim,
    flatten_full_action,
    soft_update,
)


# =========================================================
# Pure MADDPG helpers
# =========================================================
def get_pure_actor_raw_action_dim(state: Dict[str, Any]) -> int:
    """
    Raw actor layout:
        [ move_dist(M), move_angle(M), offload_ratio(K), sched_score(K*M) ]
    """
    M = int(state["M"])
    K = int(state["K"])
    return M + M + K + K * M

def decode_offload_ratio_np(
    offload_raw: np.ndarray,
    min_ratio: float = 0.05,
    max_ratio: float = 1.0,
    temperature: float = 2.0,
) -> np.ndarray:
    """
    Stable offload decoding for execution path.
    - use sigmoid instead of tanh to reduce hard saturation
    - keep a minimum ratio floor
    """
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
    """
    Stable offload decoding for differentiable surrogate path.
    """
    ratio01 = torch.sigmoid(offload_raw / float(temperature))
    ratio = min_ratio + (max_ratio - min_ratio) * ratio01
    return torch.clamp(ratio, min=min_ratio, max=max_ratio)

def decode_pure_raw_action(
    state: Dict[str, Any],
    access_assoc: np.ndarray,
    raw_action: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Decode raw actor output into environment high-level action.
    Pure MADDPG only predicts:
        - move_dist
        - move_angle
        - offload_ratio
        - sched_beta
    """
    M = int(state["M"])
    K = int(state["K"])
    neighbors = state["neighbors"]
    max_speed = float(state["max_speed"])
    delta_t = float(state["delta_t"])
    max_move = max_speed * delta_t

    raw_action = np.asarray(raw_action, dtype=np.float32).reshape(-1)
    expected_dim = M + M + K + K * M
    if raw_action.shape[0] != expected_dim:
        raise ValueError(
            f"Raw action dim mismatch: got {raw_action.shape[0]}, expected {expected_dim}"
        )

    idx = 0

    move_dist_raw = raw_action[idx: idx + M]
    idx += M
    move_dist = 0.5 * (np.tanh(move_dist_raw) + 1.0) * max_move

    move_angle_raw = raw_action[idx: idx + M]
    idx += M
    move_angle = np.pi * np.tanh(move_angle_raw)

    offload_raw = raw_action[idx: idx + K]
    idx += K
    # offload_ratio = 0.5 * (np.tanh(offload_raw) + 1.0)
    offload_ratio = decode_offload_ratio_np(
        offload_raw,
        min_ratio=0.05,
        max_ratio=1.0,
        temperature=2.0,
    )
    offload_ratio = np.clip(offload_ratio, 0.05, 1.0)

    sched_score = raw_action[idx: idx + K * M].reshape(K, M)
    sched_beta = np.zeros((K, M, M), dtype=np.float32)

    for k in range(K):
        access_m = int(np.argmax(access_assoc[:, k]))
        legal_js = [access_m] + list(neighbors[access_m])

        legal_scores = [float(sched_score[k, j]) for j in legal_js]
        best_local_idx = int(np.argmax(legal_scores))
        best_j = int(legal_js[best_local_idx])

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
    action: Dict[str, np.ndarray],
    omega1: float,
    omega2: float,
) -> Dict[str, np.ndarray]:
    """
    Pure MADDPG actor only outputs high-level action.
    This function supplements the low-level analytical allocations so that
    the action matches the full-action format required by critic/replay.

    Adds:
        - bandwidth_alloc [M, K]
        - cpu_alloc       [M, K]
    """
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


def build_surrogate_full_action_tensor(
    state: Dict[str, Any],
    access_assoc: np.ndarray,
    raw_action_pred: torch.Tensor,
    bandwidth_alloc_np: np.ndarray,
    cpu_alloc_np: np.ndarray,
) -> torch.Tensor:
    """
    Differentiable surrogate full-action builder for critic-guided actor update.

    Full action layout:
        [ move_dist(M),
          move_angle(M),
          offload_ratio(K),
          sched_beta(K*M*M),
          bandwidth_alloc(M*K),
          cpu_alloc(M*K) ]
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

    offload_raw = raw_action_pred[:, p:p + K]
    p += K

    sched_raw = raw_action_pred[:, p:p + K * M].reshape(-1, K, M)

    move_dist = 0.5 * (torch.tanh(move_dist_raw) + 1.0) * max_move
    move_angle = torch.tanh(move_angle_raw) * np.pi
    # offload_ratio = 0.5 * (torch.tanh(offload_raw) + 1.0)
    offload_ratio = decode_offload_ratio_torch(
        offload_raw,
        min_ratio=0.05,
        max_ratio=1.0,
        temperature=2.0,
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

    full_action = torch.cat(
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
    return full_action


@torch.no_grad()
def evaluate_pure_policy_rollout(
    env: MultiUavMecEnv,
    actor: MLPActor,
    device: str,
    omega1: float,
    omega2: float,
    seed: int = 123,
):
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

    actor.eval()

    while not done:
        state = obs["raw_state"]
        access_assoc = build_access_association(state)
        obs_vec = build_global_observation(state)

        obs_tensor = torch.tensor(
            obs_vec, dtype=torch.float32, device=device
        ).unsqueeze(0)

        raw_action = actor(obs_tensor).squeeze(0).detach().cpu().numpy().astype(np.float32)

        action = decode_pure_raw_action(
            state=state,
            access_assoc=access_assoc,
            raw_action=raw_action,
        )
        action_full = enrich_pure_action_to_full_action(
            state=state,
            access_assoc=access_assoc,
            action=action,
            omega1=omega1,
            omega2=omega2,
        )

        obs, reward, done, info = env.step(action_full)

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

    return {
        "episode_reward": total_reward,
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
    }


# =========================================================
# Main
# =========================================================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(72)
    np.random.seed(72)

    cpu_omega1 = 50.0
    cpu_omega2 = 1.0

    # -------------------------------------------------
    # Env config
    # -------------------------------------------------
    env = MultiUavMecEnv(
        M=3,
        K=16,
        episode_length=20,
        cpu_mode="kkt",
        omega1=cpu_omega1,
        omega2=cpu_omega2,
        deadline_scale=5,
        # task_local_cpu_min=3.0e3,
        # task_local_cpu_max=8.0e3,
        task_local_cpu_min=2.0e3,
        task_local_cpu_max=6.0e3,
        uav_energy_min=2600.0,
        uav_energy_max=3800.0,
        seed=72,
    )

    obs = env.reset(seed=72)
    state = obs["raw_state"]

    obs_dim = get_observation_dim(state)
    actor_raw_action_dim = get_pure_actor_raw_action_dim(state)
    critic_action_dim = get_full_action_dim(state)

    print("obs_dim:", obs_dim)
    print("actor_raw_action_dim:", actor_raw_action_dim)
    print("critic_action_dim:", critic_action_dim)
    print("device:", device)

    # -------------------------------------------------
    # Networks
    # -------------------------------------------------
    actor = MLPActor(
        obs_dim=obs_dim,
        action_dim=actor_raw_action_dim,
        hidden_dim=256,
    ).to(device)

    target_actor = copy.deepcopy(actor).to(device)

    critic = MLPCritic(
        obs_dim=obs_dim,
        action_dim=critic_action_dim,
        hidden_dim=256,
    ).to(device)

    target_critic = copy.deepcopy(critic).to(device)

    # -------------------------------------------------
    # Optimizers
    # -------------------------------------------------
    actor_opt = optim.Adam(actor.parameters(), lr=2e-4)
    critic_opt = optim.Adam(critic.parameters(), lr=1e-4)

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
    gamma = 0.99
    tau = 0.005
    batch_size = 32
    num_episodes = 20
    eval_every = 5
    actor_policy_coef = 0.1
    actor_l2_coef = 1e-5

    best_actor_state = copy.deepcopy(actor.state_dict())
    best_eval_reward = -float("inf")

    print("M:", env.M)
    print("K:", env.K)
    print("episode_length:", env.episode_length)
    print("obs_dim:", obs_dim)
    print("actor_raw_action_dim:", actor_raw_action_dim)
    print("critic_action_dim:", critic_action_dim)
    print("device:", device)

    # -------------------------------------------------
    # Training
    # -------------------------------------------------
    for ep in range(num_episodes):
        obs = env.reset(seed=72 + ep)
        done = False

        episode_reward = 0.0
        episode_actor_policy_loss = 0.0
        episode_critic_loss = 0.0
        update_count = 0
        step_count = 0

        while not done:
            state = obs["raw_state"]
            access_assoc = build_access_association(state)
            obs_vec = build_global_observation(state)

            obs_tensor = torch.tensor(
                obs_vec, dtype=torch.float32, device=device
            ).unsqueeze(0)

            # -----------------------------------------
            # Current actor action
            # -----------------------------------------
            # with torch.no_grad():
            #     raw_action_np = actor(obs_tensor).squeeze(0).cpu().numpy().astype(np.float32)

            with torch.no_grad():
                raw_action_np = actor(obs_tensor).squeeze(0).cpu().numpy().astype(np.float32)

            # add exploration noise only during training
            M_cur = int(state["M"])
            K_cur = int(state["K"])

            move_noise = np.random.normal(0.0, 0.15, size=(2 * M_cur,)).astype(np.float32)
            offload_noise = np.random.normal(0.0, 0.35, size=(K_cur,)).astype(np.float32)
            sched_noise = np.random.normal(0.0, 0.10, size=(K_cur * M_cur,)).astype(np.float32)

            raw_action_np = raw_action_np.copy()
            raw_action_np[: 2 * M_cur] += move_noise
            raw_action_np[2 * M_cur : 2 * M_cur + K_cur] += offload_noise
            raw_action_np[2 * M_cur + K_cur :] += sched_noise

            action = decode_pure_raw_action(
                state=state,
                access_assoc=access_assoc,
                raw_action=raw_action_np,
            )
            action_full = enrich_pure_action_to_full_action(
                state=state,
                access_assoc=access_assoc,
                action=action,
                omega1=cpu_omega1,
                omega2=cpu_omega2,
            )

            flat_action = flatten_full_action(action_full)

            next_obs, reward, done, info = env.step(action_full)
            next_obs_vec = build_global_observation(next_obs["raw_state"])

            # -----------------------------------------
            # Target next action for critic TD target
            # -----------------------------------------
            if not done:
                next_state = next_obs["raw_state"]
                next_access_assoc = build_access_association(next_state)

                next_obs_tensor = torch.tensor(
                    next_obs_vec, dtype=torch.float32, device=device
                ).unsqueeze(0)

                with torch.no_grad():
                    next_raw_action_np = (
                        target_actor(next_obs_tensor)
                        .squeeze(0)
                        .cpu()
                        .numpy()
                        .astype(np.float32)
                    )

                next_action = decode_pure_raw_action(
                    state=next_state,
                    access_assoc=next_access_assoc,
                    raw_action=next_raw_action_np,
                )
                next_action_full = enrich_pure_action_to_full_action(
                    state=next_state,
                    access_assoc=next_access_assoc,
                    action=next_action,
                    omega1=cpu_omega1,
                    omega2=cpu_omega2,
                )

                next_flat_action = flatten_full_action(next_action_full)
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
            # Critic update
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

                episode_critic_loss += float(critic_loss.item())
                update_count += 1

            # -----------------------------------------
            # Actor update
            # -----------------------------------------
            # raw_pred = actor(obs_tensor)

            # surrogate_full_action = build_surrogate_full_action_tensor(
            #     state=state,
            #     access_assoc=access_assoc,
            #     raw_action_pred=raw_pred,
            #     bandwidth_alloc_np=action_full["bandwidth_alloc"],
            #     cpu_alloc_np=action_full["cpu_alloc"],
            # )

            # actor_policy_loss = -critic(obs_tensor, surrogate_full_action).mean()
            # actor_l2_loss = (raw_pred ** 2).mean()

            # total_actor_loss = (
            #     actor_policy_coef * actor_policy_loss
            #     + actor_l2_coef * actor_l2_loss
            # )

            raw_pred = actor(obs_tensor)

            surrogate_full_action = build_surrogate_full_action_tensor(
                state=state,
                access_assoc=access_assoc,
                raw_action_pred=raw_pred,
                bandwidth_alloc_np=action_full["bandwidth_alloc"],
                cpu_alloc_np=action_full["cpu_alloc"],
            )

            actor_policy_loss = -critic(obs_tensor, surrogate_full_action).mean()
            actor_l2_loss = (raw_pred ** 2).mean()

            M_cur = int(state["M"])
            K_cur = int(state["K"])

            offload_raw_pred = raw_pred[:, 2 * M_cur : 2 * M_cur + K_cur]
            offload_ratio_pred = decode_offload_ratio_torch(
                offload_raw_pred,
                min_ratio=0.05,
                max_ratio=1.0,
                temperature=2.0,
            )

            # 1) punish sticking to the lower bound
            # offload_floor_target = 0.12
            offload_floor_target = 0.15
            offload_floor_loss = torch.relu(
                offload_floor_target - offload_ratio_pred
            ).mean()

            # 2) encourage task-wise variation, avoid all tasks same ratio
            offload_std = torch.std(offload_ratio_pred, dim=1).mean()
            # offload_diversity_loss = torch.relu(0.03 - offload_std)
            offload_diversity_loss = torch.relu(0.06 - offload_std)

            # 3) keep average offload from collapsing too low
            offload_mean = offload_ratio_pred.mean()
            # offload_mean_loss = torch.relu(0.15 - offload_mean)
            offload_mean_loss = torch.relu(0.20 - offload_mean)

            total_actor_loss = (
                actor_policy_coef * actor_policy_loss
                + actor_l2_coef * actor_l2_loss
                # + 2.0 * offload_floor_loss
                # + 1.0 * offload_diversity_loss
                # + 1.5 * offload_mean_loss
                # + 6.0 * offload_floor_loss
                # + 4.0 * offload_diversity_loss
                # + 5.0 * offload_mean_loss
                + 8.0 * offload_floor_loss
                + 5.0 * offload_diversity_loss
                + 6.0 * offload_mean_loss
            )

            actor_opt.zero_grad()
            total_actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=5.0)
            actor_opt.step()

            # -----------------------------------------
            # Target soft update
            # -----------------------------------------
            soft_update(target_actor, actor, tau=tau)
            soft_update(target_critic, critic, tau=tau)

            # if step_count < 2:
            #     with torch.no_grad():
            #         raw_dbg = raw_pred.squeeze(0).detach().cpu().numpy()
            #         print("[PURE MADDPG DEBUG]")
            #         print("  raw_action[:8] =", np.round(raw_dbg[:8], 4))
            #         print("  offload_ratio =", np.round(action_full["offload_ratio"], 4))
            #         print("  sched_beta sum =", float(np.sum(action_full["sched_beta"])))
            #         print("  bw_sum =", float(np.sum(action_full["bandwidth_alloc"])))
            #         print("  cpu_sum =", float(np.sum(action_full["cpu_alloc"])))

            if step_count < 2:
                with torch.no_grad():
                    raw_dbg = raw_pred.squeeze(0).detach().cpu().numpy()
                    M_cur = int(state["M"])
                    K_cur = int(state["K"])
                    offload_raw_dbg = raw_dbg[2 * M_cur : 2 * M_cur + K_cur]

                    print("[PURE MADDPG DEBUG]")
                    print("  offload_raw =", np.round(offload_raw_dbg, 4))
                    print("  offload_ratio =", np.round(action_full["offload_ratio"], 4))
                    print("  offload_mean =", float(np.mean(action_full["offload_ratio"])))
                    print("  offload_std =", float(np.std(action_full["offload_ratio"])))
                    print("  sched_beta sum =", float(np.sum(action_full["sched_beta"])))
                    print("  bw_sum =", float(np.sum(action_full["bandwidth_alloc"])))
                    print("  cpu_sum =", float(np.sum(action_full["cpu_alloc"])))

            obs = next_obs
            episode_reward += float(reward)
            episode_actor_policy_loss += float(actor_policy_loss.item())
            step_count += 1

        print(f"\nEpisode {ep}")
        print("episode_reward:", episode_reward)
        print("avg_actor_policy_loss:", episode_actor_policy_loss / max(step_count, 1))
        if update_count > 0:
            print("avg_critic_loss:", episode_critic_loss / update_count)
        else:
            print("avg_critic_loss: N/A")
        print("steps:", step_count)
        print("buffer_size:", len(buffer))

        # -----------------------------------------
        # Periodic evaluation
        # -----------------------------------------
        if (ep + 1) % eval_every == 0 or ep == 0:
            eval_env = MultiUavMecEnv(
                M=3,
                # K=8,
                # episode_length=5,
                K=16,
                episode_length=20,
                cpu_mode="kkt",
                omega1=cpu_omega1,
                omega2=cpu_omega2,
                deadline_scale=5,
                # task_local_cpu_min=3.0e3,
                # task_local_cpu_max=8.0e3,
                task_local_cpu_min=2.0e3,
                task_local_cpu_max=6.0e3,
                uav_energy_min=2600.0,
                uav_energy_max=3800.0,
                seed=999,
            )

            eval_result = evaluate_pure_policy_rollout(
                env=eval_env,
                actor=actor,
                device=device,
                omega1=cpu_omega1,
                omega2=cpu_omega2,
                seed=999,
            )

            print("\n==============================")
            print(f"Pure MADDPG Eval @ episode {ep}")
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



            if eval_result["episode_reward"] > best_eval_reward:
                best_eval_reward = eval_result["episode_reward"]
                best_actor_state = copy.deepcopy(actor.state_dict())

    # -------------------------------------------------
    # Save checkpoints
    # -------------------------------------------------
    os.makedirs("checkpoints", exist_ok=True)

    torch.save(
        best_actor_state,
        "checkpoints/pure_maddpg_k16_ep40_best_actor.pth",
    )
    torch.save(
        critic.state_dict(),
        "checkpoints/pure_maddpg_k16_ep40_critic.pth",
    )

    print("\nPure MADDPG training finished successfully.")
    print("Saved:")
    print("  checkpoints/pure_maddpg_k16_ep40_best_actor.pth")
    print("  checkpoints/pure_maddpg_k16_ep40_critic.pth")


if __name__ == "__main__":
    main()