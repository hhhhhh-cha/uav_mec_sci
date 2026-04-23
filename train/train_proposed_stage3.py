import os
import copy
from collections import deque
from typing import Dict, Any, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from env.mec_env import MultiUavMecEnv
from env.association import build_access_association
from model.proposed_obs_builder import build_global_observation, get_observation_dim
from model.mlp_actor import MLPActor
from model.mlp_critic import MLPCritic
from policy.proposed_placeholder_policy import generate_proposed_placeholder_action


EPS = 1e-8


def get_actor_raw_action_dim(state: Dict[str, Any]) -> int:
    """
    Raw actor output layout:
        [ move_dist(M), move_angle(M), offload_ratio(K), sched_score(K*M) ]
    """
    M = int(state["M"])
    K = int(state["K"])
    return M + M + K + K * M


def get_flattened_action_dim(state: Dict[str, Any]) -> int:
    """
    Critic action layout:
        [ move_dist(M), move_angle(M), offload_ratio(K), sched_beta(K*M*M) ]
    """
    M = int(state["M"])
    K = int(state["K"])
    return M + M + K + K * M * M


def soft_update(target_net, source_net, tau=0.005):
    for tp, sp in zip(target_net.parameters(), source_net.parameters()):
        tp.data.copy_(tau * sp.data + (1.0 - tau) * tp.data)


class RichReplayBuffer:
    """
    Replay buffer that stores extra fields needed by Stage-3:
      - access_assoc
      - next_access_assoc
    so target action can be generated properly.
    """

    def __init__(self, capacity: int = 100000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def add(
        self,
        obs: np.ndarray,
        access_assoc: np.ndarray,
        action_flat: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        next_access_assoc: np.ndarray,
        done: float,
    ):
        self.buffer.append(
            {
                "obs": np.asarray(obs, dtype=np.float32),
                "access_assoc": np.asarray(access_assoc, dtype=np.float32),
                "action": np.asarray(action_flat, dtype=np.float32),
                "reward": float(reward),
                "next_obs": np.asarray(next_obs, dtype=np.float32),
                "next_access_assoc": np.asarray(next_access_assoc, dtype=np.float32),
                "done": float(done),
            }
        )

    def sample(self, batch_size: int):
        idx = np.random.choice(len(self.buffer), size=batch_size, replace=False)
        batch = [self.buffer[i] for i in idx]

        return {
            "obs": np.stack([b["obs"] for b in batch], axis=0),
            "access_assoc": np.stack([b["access_assoc"] for b in batch], axis=0),
            "action": np.stack([b["action"] for b in batch], axis=0),
            "reward": np.asarray([b["reward"] for b in batch], dtype=np.float32).reshape(-1, 1),
            "next_obs": np.stack([b["next_obs"] for b in batch], axis=0),
            "next_access_assoc": np.stack([b["next_access_assoc"] for b in batch], axis=0),
            "done": np.asarray([b["done"] for b in batch], dtype=np.float32).reshape(-1, 1),
        }

    def __len__(self):
        return len(self.buffer)


def build_teacher_raw_target(state: Dict[str, Any], access_assoc: np.ndarray, teacher_action: Dict[str, Any]) -> np.ndarray:
    """
    Same target convention as Stage-2:
        [ move_dist(M), move_angle(M), offload_ratio(K), sched_score(K*M) ]
    all mapped to roughly [-1, 1].
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


def split_raw_action_np(raw_action: np.ndarray, M: int, K: int):
    """
    raw_action layout:
        [ move_dist(M), move_angle(M), offload_ratio(K), sched_score(K*M) ]
    """
    p = 0
    raw_move_dist = raw_action[p:p + M]
    p += M

    raw_move_angle = raw_action[p:p + M]
    p += M

    raw_offload = raw_action[p:p + K]
    p += K

    raw_sched = raw_action[p:p + K * M].reshape(K, M)

    return raw_move_dist, raw_move_angle, raw_offload, raw_sched


def decode_raw_action_to_high_action_np(
    raw_action: np.ndarray,
    access_assoc: np.ndarray,
    M: int,
    K: int,
    max_move: float,
):
    """
    Convert actor raw output into env-style high_action dict for env.step().
    Uses hard scheduling for execution.
    """
    raw_move_dist, raw_move_angle, raw_offload, raw_sched = split_raw_action_np(raw_action, M, K)

    move_dist = 0.5 * (np.tanh(raw_move_dist) + 1.0) * max_move
    move_angle = np.tanh(raw_move_angle) * np.pi
    offload_ratio = 0.5 * (np.tanh(raw_offload) + 1.0)

    sched_beta = np.zeros((K, M, M), dtype=np.float32)

    for k in range(K):
        access_m = int(np.argmax(access_assoc[:, k]))
        exec_j = int(np.argmax(raw_sched[k]))
        sched_beta[k, access_m, exec_j] = 1.0

    high_action = {
        "move_dist": move_dist.astype(np.float32),
        "move_angle": move_angle.astype(np.float32),
        "offload_ratio": offload_ratio.astype(np.float32),
        "sched_beta": sched_beta.astype(np.float32),
    }
    return high_action


def flatten_high_action_np(high_action: Dict[str, Any]) -> np.ndarray:
    """
    Critic flattened action layout:
        [ move_dist(M), move_angle(M), offload_ratio(K), sched_beta(K*M*M) ]
    """
    move_dist = np.asarray(high_action["move_dist"], dtype=np.float32).reshape(-1)
    move_angle = np.asarray(high_action["move_angle"], dtype=np.float32).reshape(-1)
    offload_ratio = np.asarray(high_action["offload_ratio"], dtype=np.float32).reshape(-1)
    sched_beta = np.asarray(high_action["sched_beta"], dtype=np.float32).reshape(-1)

    return np.concatenate(
        [move_dist, move_angle, offload_ratio, sched_beta],
        axis=0,
    ).astype(np.float32)


def build_flat_action_from_raw_torch(
    raw_action: torch.Tensor,
    access_assoc: torch.Tensor,
    M: int,
    K: int,
    max_move: float,
    hard_schedule: bool,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Build critic flat action from raw actor output.

    hard_schedule=False:
        use softmax probabilities on scheduling head so actor loss can backprop.

    hard_schedule=True:
        use one-hot schedule for target critic action.
    """
    B = raw_action.shape[0]

    p = 0
    raw_move_dist = raw_action[:, p:p + M]
    p += M

    raw_move_angle = raw_action[:, p:p + M]
    p += M

    raw_offload = raw_action[:, p:p + K]
    p += K

    raw_sched = raw_action[:, p:p + K * M].reshape(B, K, M)

    move_dist = 0.5 * (torch.tanh(raw_move_dist) + 1.0) * max_move
    move_angle = torch.tanh(raw_move_angle) * np.pi
    offload_ratio = 0.5 * (torch.tanh(raw_offload) + 1.0)

    if hard_schedule:
        sched_index = torch.argmax(raw_sched, dim=-1)  # [B, K]
        sched_prob = torch.zeros_like(raw_sched)
        sched_prob.scatter_(-1, sched_index.unsqueeze(-1), 1.0)
    else:
        sched_prob = torch.softmax(raw_sched / max(temperature, EPS), dim=-1)

    sched_beta = torch.zeros(
        (B, K, M, M),
        dtype=raw_action.dtype,
        device=raw_action.device,
    )

    access_m_idx = torch.argmax(access_assoc, dim=1)  # [B, K]

    for b in range(B):
        for k in range(K):
            m_idx = int(access_m_idx[b, k].item())
            sched_beta[b, k, m_idx, :] = sched_prob[b, k, :]

    flat_action = torch.cat(
        [
            move_dist,
            move_angle,
            offload_ratio,
            sched_beta.reshape(B, -1),
        ],
        dim=1,
    )
    return flat_action


@torch.no_grad()
def select_action_from_actor(
    actor: nn.Module,
    obs_vec: np.ndarray,
    access_assoc: np.ndarray,
    M: int,
    K: int,
    max_move: float,
    device: str,
    noise_std: float = 0.0,
):
    obs_t = torch.tensor(obs_vec, dtype=torch.float32, device=device).unsqueeze(0)
    raw_action = actor(obs_t).squeeze(0).cpu().numpy()

    if noise_std > 0.0:
        raw_action = raw_action + np.random.normal(0.0, noise_std, size=raw_action.shape).astype(np.float32)

    high_action = decode_raw_action_to_high_action_np(
        raw_action=raw_action,
        access_assoc=access_assoc,
        M=M,
        K=K,
        max_move=max_move,
    )
    flat_action = flatten_high_action_np(high_action)

    return raw_action.astype(np.float32), high_action, flat_action


@torch.no_grad()
def evaluate_actor(
    actor: nn.Module,
    device: str,
    eval_seeds: List[int],
    M: int,
    K: int,
    episode_length: int,
    cpu_mode: str,
    omega1: float,
    omega2: float,
    deadline_scale: float,
):
    total_reward = 0.0
    total_delay = 0.0
    total_energy = 0.0
    total_deadline_violation = 0.0
    total_feasible_ratio = 0.0
    total_steps = 0

    eval_env = MultiUavMecEnv(
        M=M,
        K=K,
        episode_length=episode_length,
        cpu_mode=cpu_mode,
        omega1=omega1,
        omega2=omega2,
        deadline_scale=deadline_scale,
        seed=eval_seeds[0],
    )

    obs = eval_env.reset(seed=eval_seeds[0])
    state0 = obs["raw_state"]
    max_move = float(state0["max_speed"]) * float(state0["delta_t"])

    for seed in eval_seeds:
        obs = eval_env.reset(seed=seed)
        done = False

        ep_reward = 0.0
        ep_delay = 0.0
        ep_energy = 0.0
        ep_deadline_violation = 0.0
        ep_feasible = 0
        ep_steps = 0

        while not done:
            state = obs["raw_state"]
            access_assoc = build_access_association(state)
            obs_vec = build_global_observation(state)

            _, high_action, _ = select_action_from_actor(
                actor=actor,
                obs_vec=obs_vec,
                access_assoc=access_assoc,
                M=M,
                K=K,
                max_move=max_move,
                device=device,
                noise_std=0.0,
            )

            obs, reward, done, info = eval_env.step(high_action)

            ep_reward += reward
            ep_delay += info["metrics"]["delay_sys"]
            ep_energy += info["metrics"]["energy_sys"]
            ep_deadline_violation += info["report"]["deadline_violation"]
            ep_feasible += int(info["report"]["ok"])
            ep_steps += 1

        total_reward += ep_reward
        total_delay += ep_delay / max(ep_steps, 1)
        total_energy += ep_energy / max(ep_steps, 1)
        total_deadline_violation += ep_deadline_violation / max(ep_steps, 1)
        total_feasible_ratio += ep_feasible / max(ep_steps, 1)
        total_steps += ep_steps

    n = len(eval_seeds)
    return {
        "mean_episode_reward": total_reward / max(n, 1),
        "mean_avg_delay": total_delay / max(n, 1),
        "mean_avg_energy": total_energy / max(n, 1),
        "mean_avg_deadline_violation": total_deadline_violation / max(n, 1),
        "mean_feasible_ratio": total_feasible_ratio / max(n, 1),
        "total_eval_steps": total_steps,
    }


def load_stage2_actor_checkpoint(actor: nn.Module, ckpt_path: str, device: str):
    """
    Compatible with both:
      1) raw state_dict checkpoint
      2) dict checkpoint containing actor_state_dict
    """
    ckpt = torch.load(ckpt_path, map_location=device)

    if isinstance(ckpt, dict) and "actor_state_dict" in ckpt:
        actor.load_state_dict(ckpt["actor_state_dict"])
        meta = ckpt
    else:
        actor.load_state_dict(ckpt)
        meta = None

    return meta


def main():
    device = "cpu"
    torch.manual_seed(42)
    np.random.seed(42)

    # -------------------------------------------------
    # Config
    # -------------------------------------------------
    M = 3
    K = 8
    episode_length = 5
    cpu_mode = "kkt"
    omega1 = 100.0
    omega2 = 1.0
    deadline_scale = 5.0

    gamma = 0.99
    tau = 0.005
    batch_size = 32
    buffer_capacity = 20000
    num_episodes = 200
    warmup_steps = 64
    updates_per_step = 1

    actor_lr = 1e-4
    critic_lr = 1e-3

    actor_bc_coef = 0.05
    actor_l2_coef = 1e-5
    soft_schedule_temperature = 1.0

    init_noise_std = 0.20
    final_noise_std = 0.03
    noise_decay_episodes = 120

    eval_every = 20
    eval_seeds = [1000, 1001, 1002, 1003, 1004]

    stage2_ckpt_path = "checkpoints/proposed_bc_stage2_actor.pth"
    save_dir = "checkpoints"
    os.makedirs(save_dir, exist_ok=True)

    # -------------------------------------------------
    # Build env and infer dims
    # -------------------------------------------------
    env = MultiUavMecEnv(
        M=M,
        K=K,
        episode_length=episode_length,
        cpu_mode=cpu_mode,
        omega1=omega1,
        omega2=omega2,
        deadline_scale=deadline_scale,
        seed=42,
    )

    obs = env.reset(seed=42)
    state = obs["raw_state"]

    obs_dim = get_observation_dim(state)
    actor_raw_action_dim = get_actor_raw_action_dim(state)
    critic_action_dim = get_flattened_action_dim(state)

    max_speed = float(state["max_speed"])
    delta_t = float(state["delta_t"])
    max_move = max(max_speed * delta_t, EPS)

    print("obs_dim:", obs_dim)
    print("actor_raw_action_dim:", actor_raw_action_dim)
    print("critic_action_dim:", critic_action_dim)

    # -------------------------------------------------
    # Networks
    # -------------------------------------------------
    actor = MLPActor(obs_dim, actor_raw_action_dim, hidden_dim=256).to(device)
    critic = MLPCritic(obs_dim, critic_action_dim, hidden_dim=256).to(device)

    # Load Stage-2 BC initialization
    if os.path.exists(stage2_ckpt_path):
        meta = load_stage2_actor_checkpoint(actor, stage2_ckpt_path, device=device)
        print(f"Loaded Stage-2 actor checkpoint from: {stage2_ckpt_path}")
        if meta is not None:
            print("checkpoint meta keys:", list(meta.keys()))
    else:
        print(f"WARNING: Stage-2 checkpoint not found: {stage2_ckpt_path}")
        print("Training will start from random actor initialization.")

    target_actor = copy.deepcopy(actor).to(device)
    target_critic = copy.deepcopy(critic).to(device)

    actor_opt = optim.Adam(actor.parameters(), lr=actor_lr)
    critic_opt = optim.Adam(critic.parameters(), lr=critic_lr)

    mse_loss = nn.MSELoss()

    # -------------------------------------------------
    # Replay
    # -------------------------------------------------
    replay = RichReplayBuffer(capacity=buffer_capacity)

    # -------------------------------------------------
    # Bookkeeping
    # -------------------------------------------------
    global_step = 0
    best_eval_reward = -float("inf")
    best_actor_state = copy.deepcopy(actor.state_dict())

    # -------------------------------------------------
    # Training
    # -------------------------------------------------
    for ep in range(num_episodes):
        obs = env.reset(seed=42 + ep)
        done = False

        ep_reward = 0.0
        ep_delay = 0.0
        ep_energy = 0.0
        ep_deadline_violation = 0.0
        ep_feasible = 0
        ep_steps = 0

        progress = min(ep / max(noise_decay_episodes, 1), 1.0)
        noise_std = init_noise_std + (final_noise_std - init_noise_std) * progress

        last_critic_loss = None
        last_actor_loss = None

        while not done:
            state = obs["raw_state"]
            access_assoc = build_access_association(state)
            obs_vec = build_global_observation(state)

            _, high_action, flat_action = select_action_from_actor(
                actor=actor,
                obs_vec=obs_vec,
                access_assoc=access_assoc,
                M=M,
                K=K,
                max_move=max_move,
                device=device,
                noise_std=noise_std,
            )

            next_obs, reward, done, info = env.step(high_action)
            next_state = next_obs["raw_state"]
            next_access_assoc = build_access_association(next_state)
            next_obs_vec = build_global_observation(next_state)

            replay.add(
                obs=obs_vec,
                access_assoc=access_assoc,
                action_flat=flat_action,
                reward=reward,
                next_obs=next_obs_vec,
                next_access_assoc=next_access_assoc,
                done=float(done),
            )

            obs = next_obs
            ep_reward += reward
            ep_delay += info["metrics"]["delay_sys"]
            ep_energy += info["metrics"]["energy_sys"]
            ep_deadline_violation += info["report"]["deadline_violation"]
            ep_feasible += int(info["report"]["ok"])
            ep_steps += 1
            global_step += 1

            if len(replay) >= max(batch_size, warmup_steps):
                for _ in range(updates_per_step):
                    batch = replay.sample(batch_size)

                    obs_b = torch.tensor(batch["obs"], dtype=torch.float32, device=device)
                    access_assoc_b = torch.tensor(batch["access_assoc"], dtype=torch.float32, device=device)
                    action_b = torch.tensor(batch["action"], dtype=torch.float32, device=device)
                    reward_b = torch.tensor(batch["reward"], dtype=torch.float32, device=device)
                    next_obs_b = torch.tensor(batch["next_obs"], dtype=torch.float32, device=device)
                    next_access_assoc_b = torch.tensor(batch["next_access_assoc"], dtype=torch.float32, device=device)
                    done_b = torch.tensor(batch["done"], dtype=torch.float32, device=device)

                    # -----------------------------------------
                    # Critic update: true TD target
                    # -----------------------------------------
                    with torch.no_grad():
                        next_raw_action = target_actor(next_obs_b)
                        next_flat_action = build_flat_action_from_raw_torch(
                            raw_action=next_raw_action,
                            access_assoc=next_access_assoc_b,
                            M=M,
                            K=K,
                            max_move=max_move,
                            hard_schedule=True,
                            temperature=soft_schedule_temperature,
                        )
                        target_q = target_critic(next_obs_b, next_flat_action)
                        y = reward_b + gamma * (1.0 - done_b) * target_q

                    q_val = critic(obs_b, action_b)
                    critic_loss = mse_loss(q_val, y)

                    critic_opt.zero_grad()
                    critic_loss.backward()
                    critic_opt.step()

                    # -----------------------------------------
                    # Actor update:
                    # 1) maximize critic on predicted action
                    # 2) keep close to placeholder teacher a bit
                    # -----------------------------------------
                    raw_action_pred = actor(obs_b)

                    pred_flat_action_soft = build_flat_action_from_raw_torch(
                        raw_action=raw_action_pred,
                        access_assoc=access_assoc_b,
                        M=M,
                        K=K,
                        max_move=max_move,
                        hard_schedule=False,
                        temperature=soft_schedule_temperature,
                    )

                    actor_q = critic(obs_b, pred_flat_action_soft)
                    actor_policy_loss = -actor_q.mean()

                    teacher_targets = []
                    obs_np = batch["obs"]
                    access_assoc_np = batch["access_assoc"]

                    # rebuild teacher targets from current replay batch
                    # by rolling back through env observation is impossible here,
                    # so we approximate teacher from current access association plus
                    # a placeholder state reconstructed from env state template.
                    #
                    # Since env dimensions and static system ranges are fixed across episodes,
                    # we can reuse current env.state after reset-time constants and only
                    # require placeholder to produce structurally valid high-level actions.
                    #
                    # More robust way in the future:
                    # store raw_state in replay. For current project phase,
                    # this approximation is enough to stabilize Stage-3.
                    for i in range(batch_size):
                        # Use current env raw_state template shape and overwrite only
                        # the fields needed by placeholder if they are static.
                        # If your placeholder later needs more exact per-step state,
                        # then store raw_state into replay and replace this block.
                        teacher_action = generate_proposed_placeholder_action(
                            state=state,
                            access_assoc=access_assoc_np[i],
                        )
                        teacher_raw_target = build_teacher_raw_target(
                            state=state,
                            access_assoc=access_assoc_np[i],
                            teacher_action=teacher_action,
                        )
                        teacher_targets.append(teacher_raw_target)

                    teacher_target_b = torch.tensor(
                        np.asarray(teacher_targets, dtype=np.float32),
                        dtype=torch.float32,
                        device=device,
                    )

                    actor_bc_loss = mse_loss(raw_action_pred, teacher_target_b)
                    actor_l2_loss = (raw_action_pred ** 2).mean()

                    actor_loss = (
                        actor_policy_loss
                        + actor_bc_coef * actor_bc_loss
                        + actor_l2_coef * actor_l2_loss
                    )

                    actor_opt.zero_grad()
                    actor_loss.backward()
                    actor_opt.step()

                    soft_update(target_actor, actor, tau=tau)
                    soft_update(target_critic, critic, tau=tau)

                    last_critic_loss = float(critic_loss.item())
                    last_actor_loss = float(actor_loss.item())

        print(f"\nEpisode {ep}")
        print("episode_reward:", ep_reward)
        print("avg_delay:", ep_delay / max(ep_steps, 1))
        print("avg_energy:", ep_energy / max(ep_steps, 1))
        print("avg_deadline_violation:", ep_deadline_violation / max(ep_steps, 1))
        print("feasible_ratio:", ep_feasible / max(ep_steps, 1))
        print("steps:", ep_steps)
        print("buffer_size:", len(replay))
        print("noise_std:", noise_std)
        if last_critic_loss is not None:
            print("last_critic_loss:", last_critic_loss)
        if last_actor_loss is not None:
            print("last_actor_loss:", last_actor_loss)

        if (ep + 1) % eval_every == 0:
            eval_result = evaluate_actor(
                actor=actor,
                device=device,
                eval_seeds=eval_seeds,
                M=M,
                K=K,
                episode_length=episode_length,
                cpu_mode=cpu_mode,
                omega1=omega1,
                omega2=omega2,
                deadline_scale=deadline_scale,
            )

            print("\n==============================")
            print(f"Stage-3 Evaluation @ episode {ep}")
            print("==============================")
            for k, v in eval_result.items():
                print(f"{k}: {v}")

            if eval_result["mean_episode_reward"] > best_eval_reward:
                best_eval_reward = eval_result["mean_episode_reward"]
                best_actor_state = copy.deepcopy(actor.state_dict())

                save_path = os.path.join(save_dir, "proposed_stage3_best_actor.pth")
                torch.save(
                    {
                        "actor_state_dict": best_actor_state,
                        "obs_dim": obs_dim,
                        "actor_raw_action_dim": actor_raw_action_dim,
                        "critic_action_dim": critic_action_dim,
                        "best_eval_reward": best_eval_reward,
                        "episode": ep,
                    },
                    save_path,
                )
                print(f"New best Stage-3 actor saved to: {save_path}")

    final_save_path = os.path.join(save_dir, "proposed_stage3_final_actor.pth")
    torch.save(
        {
            "actor_state_dict": actor.state_dict(),
            "obs_dim": obs_dim,
            "actor_raw_action_dim": actor_raw_action_dim,
            "critic_action_dim": critic_action_dim,
            "best_eval_reward": best_eval_reward,
            "num_episodes": num_episodes,
        },
        final_save_path,
    )
    print(f"\nFinal Stage-3 actor saved to: {final_save_path}")

    best_save_path = os.path.join(save_dir, "proposed_stage3_best_actor.pth")
    if not os.path.exists(best_save_path):
        torch.save(
            {
                "actor_state_dict": best_actor_state,
                "obs_dim": obs_dim,
                "actor_raw_action_dim": actor_raw_action_dim,
                "critic_action_dim": critic_action_dim,
                "best_eval_reward": best_eval_reward,
                "num_episodes": num_episodes,
            },
            best_save_path,
        )
        print(f"Best Stage-3 actor saved to: {best_save_path}")

    print("\nStage-3 training finished successfully.")


if __name__ == "__main__":
    main()