# 这版就是基于你当前 full_stage2 的结果做的保守稳健调参版，目标不是大幅探索，而是：
# 继续沿用完整 Proposed 框架
# 保持 feasible_ratio = 1.0、deadline_violation = 0.0
# 减弱 critic 把 actor 拉偏的力度
# 增强 placeholder 附近的小幅精修能力

# 你刚才的 full_stage2 日志非常清楚地表明：评估几乎冻结在 -8691.48x 平台，avg_delay 完全不变，而 actor_policy_loss 和 move_sched_bc_loss 却持续变大，这说明当前策略在“偏离老师”，但没有换来更好的 eval 表现。

import copy
import os
from typing import Any, Dict, Tuple

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
from train.train_proposed_full_stage2 import (
    build_surrogate_full_action_tensor,
    load_if_exists,
)


def main():
    device = "cpu"
    torch.manual_seed(42)
    np.random.seed(42)

    # -------------------------------------------------
    # Env config
    # -------------------------------------------------
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

    # Load stage-1 best by default.
    # You can later switch these to stage2 best if you want another run.
    load_if_exists(policy.actor_net, "checkpoints/proposed_full_stage1_best_actor.pth", "actor")
    load_if_exists(policy.encoder, "checkpoints/proposed_full_stage1_best_encoder.pth", "encoder")
    load_if_exists(policy.fusion_net, "checkpoints/proposed_full_stage1_best_fusion.pth", "fusion_net")
    load_if_exists(policy.ratio_head, "checkpoints/proposed_full_stage1_best_ratio_head.pth", "ratio_head")

    target_policy = copy.deepcopy(policy)

    # -------------------------------------------------
    # Critic
    # -------------------------------------------------
    critic = MLPCritic(
        obs_dim=obs_dim,
        action_dim=critic_action_dim,
        hidden_dim=256,
    ).to(device)

    if os.path.exists("checkpoints/proposed_full_stage1_critic.pth"):
        critic.load_state_dict(
            torch.load(
                "checkpoints/proposed_full_stage1_critic.pth",
                map_location=device,
            )
        )
        print("Loaded critic from: checkpoints/proposed_full_stage1_critic.pth")
    else:
        print("WARNING: critic checkpoint not found, critic starts random.")

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

    # Stage-2b: smaller actor LR, still allow critic learning
    actor_opt = optim.Adam(learnable_params, lr=1e-4)
    critic_opt = optim.Adam(critic.parameters(), lr=5e-4)

    mse_loss = nn.MSELoss()

    # -------------------------------------------------
    # Replay
    # -------------------------------------------------
    buffer = FullReplayBuffer(
        obs_dim=obs_dim,
        action_dim=critic_action_dim,
        capacity=20000,
    )

    # -------------------------------------------------
    # Hyperparameters
    # -------------------------------------------------
    gamma = 0.99
    tau = 0.005
    batch_size = 16
    num_episodes = 40
    eval_every = 5

    # Conservative tuning:
    # weaken critic-guided pull, strengthen BC anchoring
    actor_policy_coef = 0.05
    actor_move_sched_bc_coef = 1.0
    ratio_bc_coef = 0.5
    actor_l2_coef = 1e-5

    # extra stabilization: clip policy loss magnitude
    max_policy_loss_abs = 50.0

    # movement+scheduling mask in raw actor output
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
        obs = env.reset(seed=42 + ep)
        done = False

        episode_reward = 0.0
        episode_actor_policy_loss = 0.0
        episode_move_sched_bc_loss = 0.0
        episode_ratio_bc_loss = 0.0
        episode_critic_loss = 0.0
        update_count = 0
        step_count = 0

        while not done:
            state = obs["raw_state"]
            access_assoc = build_access_association(state)
            obs_vec = build_global_observation(state)

            # -----------------------------------------
            # teacher action for anchoring
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
            flat_action = flatten_full_action(action)

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
            # critic update
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
            # actor / encoder / ratio-head update
            # -----------------------------------------
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

            actor_policy_loss = -critic(obs_tensor, surrogate_full_action).mean()
            actor_policy_loss = torch.clamp(
                actor_policy_loss,
                min=-max_policy_loss_abs,
                max=max_policy_loss_abs,
            )

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
            torch.nn.utils.clip_grad_norm_(learnable_params, max_norm=5.0)
            actor_opt.step()

            # target soft update
            soft_update_policy(target_policy, policy, tau=tau)
            soft_update(target_critic, critic, tau=tau)

            # bookkeeping
            obs = next_obs
            episode_reward += reward
            episode_actor_policy_loss += float(actor_policy_loss.item())
            episode_move_sched_bc_loss += float(actor_move_sched_bc_loss.item())
            episode_ratio_bc_loss += float(ratio_bc_loss.item())
            step_count += 1

        print(f"\nEpisode {ep}")
        print("episode_reward:", episode_reward)
        print("avg_actor_policy_loss:", episode_actor_policy_loss / max(step_count, 1))
        print("avg_move_sched_bc_loss:", episode_move_sched_bc_loss / max(step_count, 1))
        print("avg_ratio_bc_loss:", episode_ratio_bc_loss / max(step_count, 1))
        if update_count > 0:
            print("avg_critic_loss:", episode_critic_loss / update_count)
        else:
            print("avg_critic_loss: N/A")
        print("steps:", step_count)
        print("buffer_size:", len(buffer))

        # -----------------------------------------
        # evaluation
        # -----------------------------------------
        if (ep + 1) % eval_every == 0 or ep == 0:
            eval_env = MultiUavMecEnv(
                M=3,
                K=8,
                episode_length=5,
                cpu_mode="kkt",
                omega1=100.0,
                omega2=1.0,
                deadline_scale=5.0,
                seed=999,
            )

            eval_result = evaluate_full_policy_rollout(
                env=eval_env,
                policy=policy,
                seed=999,
            )

            print("\n==============================")
            print(f"Full Proposed Stage-2b Eval @ episode {ep}")
            print("==============================")
            print("episode_reward:", eval_result["episode_reward"])
            print("avg_delay:", eval_result["avg_delay"])
            print("avg_energy:", eval_result["avg_energy"])
            print("avg_deadline_violation:", eval_result["avg_deadline_violation"])
            print("feasible_ratio:", eval_result["feasible_ratio"])
            print("num_steps:", eval_result["num_steps"])

            if eval_result["episode_reward"] > best_eval_reward:
                best_eval_reward = eval_result["episode_reward"]
                best_policy_state = copy.deepcopy(policy)

    # -------------------------------------------------
    # Save best stage-2b checkpoints
    # -------------------------------------------------
    os.makedirs("checkpoints", exist_ok=True)

    if best_policy_state.actor_net is not None:
        torch.save(
            best_policy_state.actor_net.state_dict(),
            "checkpoints/proposed_full_stage2b_best_actor.pth",
        )
    if best_policy_state.encoder is not None:
        torch.save(
            best_policy_state.encoder.state_dict(),
            "checkpoints/proposed_full_stage2b_best_encoder.pth",
        )
    if best_policy_state.fusion_net is not None:
        torch.save(
            best_policy_state.fusion_net.state_dict(),
            "checkpoints/proposed_full_stage2b_best_fusion.pth",
        )
    if best_policy_state.ratio_head is not None:
        torch.save(
            best_policy_state.ratio_head.state_dict(),
            "checkpoints/proposed_full_stage2b_best_ratio_head.pth",
        )

    torch.save(
        critic.state_dict(),
        "checkpoints/proposed_full_stage2b_critic.pth",
    )

    print("\nFull proposed Stage-2b training finished successfully.")
    print("Saved:")
    print("  checkpoints/proposed_full_stage2b_best_actor.pth")
    print("  checkpoints/proposed_full_stage2b_best_encoder.pth")
    print("  checkpoints/proposed_full_stage2b_best_fusion.pth")
    print("  checkpoints/proposed_full_stage2b_best_ratio_head.pth")
    print("  checkpoints/proposed_full_stage2b_critic.pth")


if __name__ == "__main__":
    main()