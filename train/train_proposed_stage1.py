# 用 placeholder 先 warm start 收集数据
# 用 network policy 跑 env
# 存 replay buffer
# 做最简 critic / actor 更新
# 验证训练主循环能跑

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from env.mec_env import MultiUavMecEnv
from env.association import build_access_association
from model.proposed_obs_builder import build_global_observation, get_observation_dim
from model.mlp_actor import MLPActor
from model.mlp_critic import MLPCritic
from policy.proposed_learned_policy import ProposedLearnedPolicy
from train.replay_buffer import ReplayBuffer
from train.action_utils import flatten_high_action, get_flattened_action_dim


def get_actor_raw_action_dim(state):
    """
    Raw actor output layout:
        [ move_dist(M), move_angle(M), offload_ratio(K), sched_score(K*M) ]
    """
    M = int(state["M"])
    K = int(state["K"])
    return M + M + K + K * M


def soft_update(target_net, source_net, tau=0.005):
    for tp, sp in zip(target_net.parameters(), source_net.parameters()):
        tp.data.copy_(tau * sp.data + (1.0 - tau) * tp.data)


def main():
    device = "cpu"
    torch.manual_seed(42)
    np.random.seed(42)

    env = MultiUavMecEnv(
        M=3,
        K=8,
        episode_length=5,
        cpu_mode="kkt",
        omega1=100.0,
        omega2=1.0,
        deadline_scale=5.0,
        seed=42,
    )

    obs = env.reset(seed=42)
    state = obs["raw_state"]

    obs_dim = get_observation_dim(state)
    actor_raw_action_dim = get_actor_raw_action_dim(state)
    critic_action_dim = get_flattened_action_dim(state)

    print("obs_dim:", obs_dim)
    print("actor_raw_action_dim:", actor_raw_action_dim)
    print("critic_action_dim:", critic_action_dim)

    actor = MLPActor(obs_dim, actor_raw_action_dim, hidden_dim=256).to(device)
    critic = MLPCritic(obs_dim, critic_action_dim, hidden_dim=256).to(device)

    target_actor = copy.deepcopy(actor).to(device)
    target_critic = copy.deepcopy(critic).to(device)

    actor_opt = optim.Adam(actor.parameters(), lr=1e-3)
    critic_opt = optim.Adam(critic.parameters(), lr=1e-3)

    policy = ProposedLearnedPolicy(
        mode="network",
        actor_net=actor,
        device=device,
    )
    target_policy = ProposedLearnedPolicy(
        mode="network",
        actor_net=target_actor,
        device=device,
    )

    buffer = ReplayBuffer(
        obs_dim=obs_dim,
        action_dim=critic_action_dim,
        capacity=10000,
    )

    gamma = 0.99
    batch_size = 16
    tau = 0.005

    num_episodes = 5

    for ep in range(num_episodes):
        obs = env.reset(seed=42 + ep)
        done = False
        episode_reward = 0.0
        step_count = 0

        while not done:
            state = obs["raw_state"]
            access_assoc = build_access_association(state)
            obs_vec = build_global_observation(state)

            high_action = policy.act(
                state=state,
                access_assoc=access_assoc,
                deterministic=True,
            )

            flat_action = flatten_high_action(high_action)

            next_obs, reward, done, info = env.step(high_action)
            next_obs_vec = build_global_observation(next_obs["raw_state"])

            buffer.add(
                obs=obs_vec,
                action=flat_action,
                reward=reward,
                next_obs=next_obs_vec,
                done=done,
            )

            obs = next_obs
            episode_reward += reward
            step_count += 1

            if len(buffer) >= batch_size:
                batch = buffer.sample(batch_size)

                obs_b = torch.tensor(batch["obs"], dtype=torch.float32, device=device)
                action_b = torch.tensor(batch["action"], dtype=torch.float32, device=device)
                reward_b = torch.tensor(batch["reward"], dtype=torch.float32, device=device)
                next_obs_b = torch.tensor(batch["next_obs"], dtype=torch.float32, device=device)
                done_b = torch.tensor(batch["done"], dtype=torch.float32, device=device)

                # -----------------------------------------
                # target action: need decode through policy
                # -----------------------------------------
                next_action_list = []
                for i in range(batch_size):
                    # We do not have original state objects in replay yet,
                    # so for stage-1 skeleton we reuse current env state shape
                    # and build a simple zero placeholder action target.
                    # This is only to verify the training loop wiring.
                    dummy_action = np.zeros((critic_action_dim,), dtype=np.float32)
                    next_action_list.append(dummy_action)

                next_action_b = torch.tensor(
                    np.asarray(next_action_list),
                    dtype=torch.float32,
                    device=device,
                )

                with torch.no_grad():
                    target_q = target_critic(next_obs_b, next_action_b)
                    y = reward_b + gamma * (1.0 - done_b) * target_q

                q_val = critic(obs_b, action_b)
                critic_loss = nn.MSELoss()(q_val, y)

                critic_opt.zero_grad()
                critic_loss.backward()
                critic_opt.step()

                # -----------------------------------------
                # actor update (stage-1 simplified)
                # -----------------------------------------
                raw_action_pred = actor(obs_b)

                # For stage-1 skeleton, actor loss is simple L2 regularization
                # plus a weak critic-related surrogate to keep code flow valid.
                actor_loss = 1e-4 * (raw_action_pred ** 2).mean()

                actor_opt.zero_grad()
                actor_loss.backward()
                actor_opt.step()

                soft_update(target_actor, actor, tau=tau)
                soft_update(target_critic, critic, tau=tau)

        print(f"\nEpisode {ep}")
        print("episode_reward:", episode_reward)
        print("steps:", step_count)
        print("buffer_size:", len(buffer))

    print("\nStage-1 training skeleton finished successfully.")


if __name__ == "__main__":
    main()