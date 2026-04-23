# 加载 proposed_full_stage1_best_*.pth
# 组装完整 ProposedPolicy
# 保留 encoder / fusion / ratio head / actor
# 加入更正式的 actor-critic 更新
# 同时保留一小部分 imitation regularization
# 定期 evaluation
# 保存 best full proposed checkpoint

# Stage-1
# 主要是 imitation warm-start：
# actor 学 movement + scheduling
# ratio branch 学 offloading ratio
# critic 只是 skeleton

# Stage-2
# 开始加入真正的critic-guided policy term：
# actor_policy_loss = -critic(obs, surrogate_action)
# movement 分支会通过这个项被继续优化
# scheduling 分支也会通过这个项被继续优化
# ratio branch 也会通过这个项被带着走
# 同时保留少量 BC，防止策略一下子崩掉


import copy
import os
from typing import Any, Dict, List

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
        bandwidth_alloc_np.reshape(1, -1),
        dtype=torch.float32,
        device=device,
    )
    cpu_alloc = torch.tensor(
        cpu_alloc_np.reshape(1, -1),
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
# Main
# =========================================================
def main():
    device = "cpu"
    torch.manual_seed(72)
    np.random.seed(72)

    # -------------------------------------------------
    # Env config
    # -------------------------------------------------
    env = MultiUavMecEnv(
        M=3,
        # K=8,
        # episode_length=5,
        K=16,
        episode_length=20,
        cpu_mode="kkt",
        omega1=50.0,
        omega2=1.0,
        deadline_scale=5.0,
        # task_local_cpu_min=3.0e3,
        # task_local_cpu_max=8.0e3,
        # task_local_cpu_min=1.0e3,
        # task_local_cpu_max=4.0e3,
        task_local_cpu_min=2.0e3,
        task_local_cpu_max=6.0e3,
        uav_energy_min=2600.0,
        uav_energy_max=3800.0,
        seed=72,
    )

    obs = env.reset(seed=72)
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

    # load stage-1 checkpoints
    # load_if_exists(policy.actor_net, "checkpoints/proposed_full_stage1_fix_best_actor.pth", "actor")
    # load_if_exists(policy.encoder, "checkpoints/proposed_full_stage1_fix_best_encoder.pth", "encoder")
    # load_if_exists(policy.fusion_net, "checkpoints/proposed_full_stage1_fix_best_fusion.pth", "fusion_net")
    # load_if_exists(policy.ratio_head, "checkpoints/proposed_full_stage1_fix_best_ratio_head.pth", "ratio_head")

    load_if_exists(policy.actor_net, "checkpoints/proposed_full_stage1_fix_k16_ep40_best_actor.pth", "actor")
    load_if_exists(policy.encoder, "checkpoints/proposed_full_stage1_fix_k16_ep40_best_encoder.pth", "encoder")
    load_if_exists(policy.fusion_net, "checkpoints/proposed_full_stage1_fix_k16_ep40_best_fusion.pth", "fusion_net")
    load_if_exists(policy.ratio_head, "checkpoints/proposed_full_stage1_fix_k16_ep40_best_ratio_head.pth", "ratio_head")

    target_policy = copy.deepcopy(policy)

    # -------------------------------------------------
    # Critic
    # -------------------------------------------------
    critic = MLPCritic(
        obs_dim=obs_dim,
        action_dim=critic_action_dim,
        hidden_dim=256,
    ).to(device)

    # if os.path.exists("checkpoints/proposed_full_stage1_fix_critic.pth"):
    #     critic.load_state_dict(torch.load("checkpoints/proposed_full_stage1_fix_critic.pth", map_location=device))
    #     print("Loaded critic from: checkpoints/proposed_full_stage1_fix_critic.pth")
    # else:
    #     print("WARNING: critic checkpoint not found, critic starts random.")

    
    if os.path.exists("checkpoints/proposed_full_stage1_fix_k16_ep40_critic.pth"):
        critic.load_state_dict(torch.load("checkpoints/proposed_full_stage1_fix_k16_ep40_critic.pth", map_location=device))
        print("Loaded critic from: checkpoints/proposed_full_stage1_fix_k16_ep40_critic.pth")
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

    actor_opt = optim.Adam(learnable_params, lr=5e-4)
    critic_opt = optim.Adam(critic.parameters(), lr=1e-4)

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
    batch_size = 32
    num_episodes = 40

    # actor_policy_coef = 0.1
    # actor_move_sched_bc_coef = 0.2
    # ratio_bc_coef = 0.2
    actor_policy_coef = 0.05
    actor_move_sched_bc_coef = 0.2
    ratio_bc_coef = 0.2
    actor_l2_coef = 1e-5

    eval_every = 5

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
        obs = env.reset(seed=72 + ep)
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
            # teacher action
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
            # critic update from replay
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

            soft_update_policy(target_policy, policy, tau=tau)
            soft_update(target_critic, critic, tau=tau)

            if step_count < 2:
                with torch.no_grad():
                    print("[STAGE2 DEBUG]")
                    print("  ratio_pred[:5] =", ratio_pred.detach().cpu().numpy()[:5])
                    print("  teacher_raw[:5] =", teacher_offload_ratio[:5])

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
        # periodic evaluation
        # -----------------------------------------
        if (ep + 1) % eval_every == 0 or ep == 0:
            eval_env = MultiUavMecEnv(
                M=3,
                # K=8,
                # episode_length=5,
                K=16,
                episode_length=20,
                cpu_mode="kkt",
                omega1=50.0,
                omega2=1.0,
                deadline_scale=5.0,
                # task_local_cpu_min=3.0e3,
                # task_local_cpu_max=8.0e3,
                # task_local_cpu_min=1.0e3,
                # task_local_cpu_max=4.0e3,
                task_local_cpu_min=2.0e3,
                task_local_cpu_max=6.0e3,
                uav_energy_min=2600.0,
                uav_energy_max=3800.0,
                seed=999,
            )

            eval_result = evaluate_full_policy_rollout(
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

            if eval_result["episode_reward"] > best_eval_reward:
                best_eval_reward = eval_result["episode_reward"]
                best_policy_state = copy.deepcopy(policy)

    # -------------------------------------------------
    # Save best stage-2 checkpoints
    # -------------------------------------------------
    # os.makedirs("checkpoints", exist_ok=True)

    # if best_policy_state.actor_net is not None:
    #     torch.save(
    #         best_policy_state.actor_net.state_dict(),
    #         "checkpoints/proposed_full_stage2_fix_best_actor.pth",
    #     )
    # if best_policy_state.encoder is not None:
    #     torch.save(
    #         best_policy_state.encoder.state_dict(),
    #         "checkpoints/proposed_full_stage2_fix_best_encoder.pth",
    #     )
    # if best_policy_state.fusion_net is not None:
    #     torch.save(
    #         best_policy_state.fusion_net.state_dict(),
    #         "checkpoints/proposed_full_stage2_fix_best_fusion.pth",
    #     )
    # if best_policy_state.ratio_head is not None:
    #     torch.save(
    #         best_policy_state.ratio_head.state_dict(),
    #         "checkpoints/proposed_full_stage2_fix_best_ratio_head.pth",
    #     )

    # torch.save(
    #     critic.state_dict(),
    #     "checkpoints/proposed_full_stage2_fix_critic.pth",
    # )

    # print("\nFull proposed Stage-2 training finished successfully.")
    # print("Saved:")
    # print("  checkpoints/proposed_full_stage2_fix_best_actor.pth")
    # print("  checkpoints/proposed_full_stage2_fix_best_encoder.pth")
    # print("  checkpoints/proposed_full_stage2_fix_best_fusion.pth")
    # print("  checkpoints/proposed_full_stage2_fix_best_ratio_head.pth")
    # print("  checkpoints/proposed_full_stage2_fix_critic.pth")

    os.makedirs("checkpoints", exist_ok=True)

    if best_policy_state.actor_net is not None:
        torch.save(
            best_policy_state.actor_net.state_dict(),
            "checkpoints/proposed_full_stage2_fix_k16_ep40_best_actor.pth",
        )
    if best_policy_state.encoder is not None:
        torch.save(
            best_policy_state.encoder.state_dict(),
            "checkpoints/proposed_full_stage2_fix_k16_ep40_best_encoder.pth",
        )
    if best_policy_state.fusion_net is not None:
        torch.save(
            best_policy_state.fusion_net.state_dict(),
            "checkpoints/proposed_full_stage2_fix_k16_ep40_best_fusion.pth",
        )
    if best_policy_state.ratio_head is not None:
        torch.save(
            best_policy_state.ratio_head.state_dict(),
            "checkpoints/proposed_full_stage2_fix_k16_ep40_best_ratio_head.pth",
        )

    torch.save(
        critic.state_dict(),
        "checkpoints/proposed_full_stage2_fix_k16_ep40_critic.pth",
    )

    print("\nFull proposed Stage-2 training finished successfully.")
    print("Saved:")
    print("  checkpoints/proposed_full_stage2_fix_k16_ep40_best_actor.pth")
    print("  checkpoints/proposed_full_stage2_fix_k16_ep40_best_encoder.pth")
    print("  checkpoints/proposed_full_stage2_fix_k16_ep40_best_fusion.pth")
    print("  checkpoints/proposed_full_stage2_fix_k16_ep40_best_ratio_head.pth")
    print("  checkpoints/proposed_full_stage2_fix_k16_ep40_critic.pth")


if __name__ == "__main__":
    main()


# import copy
# import os
# from typing import Any, Dict, List

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

# from train.train_proposed_full_stage1 import (
#     EPS,
#     FullReplayBuffer,
#     build_teacher_raw_target,
#     evaluate_full_policy_rollout,
#     forward_ratio_branch,
#     get_actor_raw_action_dim,
#     get_full_action_dim,
#     flatten_full_action,
#     soft_update,
#     soft_update_policy,
# )


# # =========================================================
# # Checkpoint loading
# # =========================================================
# def load_if_exists(module: torch.nn.Module, path: str, name: str):
#     if os.path.exists(path):
#         module.load_state_dict(torch.load(path, map_location="cpu"))
#         print(f"Loaded {name} from: {path}")
#     else:
#         print(f"WARNING: {name} checkpoint not found: {path}")


# # =========================================================
# # Differentiable surrogate full-action builder
# # Used only for actor policy-gradient-like update
# # =========================================================
# def build_surrogate_full_action_tensor(
#     state: Dict[str, Any],
#     access_assoc: np.ndarray,
#     raw_action_pred: torch.Tensor,      # [1, raw_dim]
#     ratio_pred: torch.Tensor,           # [K]
#     bandwidth_alloc_np: np.ndarray,     # [M, K], detached constants
#     cpu_alloc_np: np.ndarray,           # [M, K], detached constants
# ) -> torch.Tensor:
#     """
#     Build a differentiable surrogate full action for critic-guided actor update.

#     Full action layout:
#         [ move_dist(M),
#           move_angle(M),
#           offload_ratio(K),
#           sched_beta(K*M*M),
#           bandwidth_alloc(M*K),
#           cpu_alloc(M*K) ]
#     """
#     device = raw_action_pred.device
#     M = int(state["M"])
#     K = int(state["K"])
#     neighbors = state["neighbors"]

#     max_speed = float(state["max_speed"])
#     delta_t = float(state["delta_t"])
#     max_move = max(max_speed * delta_t, EPS)

#     p = 0
#     move_dist_raw = raw_action_pred[:, p:p + M]
#     p += M

#     move_angle_raw = raw_action_pred[:, p:p + M]
#     p += M

#     _offload_raw_unused = raw_action_pred[:, p:p + K]
#     p += K

#     sched_raw = raw_action_pred[:, p:p + K * M].reshape(1, K, M)

#     # differentiable movement decode
#     move_dist = 0.5 * (torch.tanh(move_dist_raw) + 1.0) * max_move     # [1, M]
#     move_angle = torch.tanh(move_angle_raw) * np.pi                    # [1, M]

#     # differentiable ratio branch output
#     offload_ratio = ratio_pred.unsqueeze(0)                            # [1, K]

#     # differentiable soft scheduling
#     sched_beta = torch.zeros((1, K, M, M), dtype=torch.float32, device=device)

#     for k in range(K):
#         access_m = int(np.argmax(access_assoc[:, k]))
#         legal_js = [access_m] + list(neighbors[access_m])

#         legal_scores = sched_raw[:, k, legal_js]                       # [1, |legal_js|]
#         legal_prob = torch.softmax(legal_scores, dim=-1)               # differentiable

#         for idx, j in enumerate(legal_js):
#             sched_beta[:, k, access_m, j] = legal_prob[:, idx]

#     bandwidth_alloc = torch.tensor(
#         bandwidth_alloc_np.reshape(1, -1),
#         dtype=torch.float32,
#         device=device,
#     )
#     cpu_alloc = torch.tensor(
#         cpu_alloc_np.reshape(1, -1),
#         dtype=torch.float32,
#         device=device,
#     )

#     full_action = torch.cat(
#         [
#             move_dist,
#             move_angle,
#             offload_ratio,
#             sched_beta.reshape(1, -1),
#             bandwidth_alloc,
#             cpu_alloc,
#         ],
#         dim=1,
#     )
#     return full_action


# # =========================================================
# # Main
# # =========================================================
# def main():
#     device = "cpu"
#     torch.manual_seed(42)
#     np.random.seed(42)

#     # -------------------------------------------------
#     # Env config
#     # -------------------------------------------------
#     env = MultiUavMecEnv(
#         M=3,
#         K=8,
#         episode_length=5,
#         cpu_mode="kkt",
#         omega1=100.0,
#         omega2=1.0,
#         deadline_scale=5.0,
#         seed=42,
#     )

#     obs = env.reset(seed=42)
#     state = obs["raw_state"]

#     obs_dim = get_observation_dim(state)
#     actor_raw_action_dim = get_actor_raw_action_dim(state)
#     critic_action_dim = get_full_action_dim(state)

#     print("obs_dim:", obs_dim)
#     print("actor_raw_action_dim:", actor_raw_action_dim)
#     print("critic_action_dim:", critic_action_dim)

#     # -------------------------------------------------
#     # Build full proposed policy
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

#     # load stage-1 checkpoints
#     load_if_exists(policy.actor_net, "checkpoints/proposed_full_stage1_best_actor.pth", "actor")
#     load_if_exists(policy.encoder, "checkpoints/proposed_full_stage1_best_encoder.pth", "encoder")
#     load_if_exists(policy.fusion_net, "checkpoints/proposed_full_stage1_best_fusion.pth", "fusion_net")
#     load_if_exists(policy.ratio_head, "checkpoints/proposed_full_stage1_best_ratio_head.pth", "ratio_head")

#     target_policy = copy.deepcopy(policy)

#     # -------------------------------------------------
#     # Critic
#     # -------------------------------------------------
#     critic = MLPCritic(
#         obs_dim=obs_dim,
#         action_dim=critic_action_dim,
#         hidden_dim=256,
#     ).to(device)

#     if os.path.exists("checkpoints/proposed_full_stage1_critic.pth"):
#         critic.load_state_dict(torch.load("checkpoints/proposed_full_stage1_critic.pth", map_location=device))
#         print("Loaded critic from: checkpoints/proposed_full_stage1_critic.pth")
#     else:
#         print("WARNING: critic checkpoint not found, critic starts random.")

#     target_critic = copy.deepcopy(critic).to(device)

#     # -------------------------------------------------
#     # Optimizers
#     # -------------------------------------------------
#     learnable_params = []
#     if policy.actor_net is not None:
#         learnable_params += list(policy.actor_net.parameters())
#     if policy.encoder is not None:
#         learnable_params += list(policy.encoder.parameters())
#     if policy.fusion_net is not None:
#         learnable_params += list(policy.fusion_net.parameters())
#     if policy.ratio_head is not None:
#         learnable_params += list(policy.ratio_head.parameters())

#     actor_opt = optim.Adam(learnable_params, lr=5e-4)
#     critic_opt = optim.Adam(critic.parameters(), lr=1e-3)

#     mse_loss = nn.MSELoss()

#     # -------------------------------------------------
#     # Replay
#     # -------------------------------------------------
#     buffer = FullReplayBuffer(
#         obs_dim=obs_dim,
#         action_dim=critic_action_dim,
#         capacity=20000,
#     )

#     # -------------------------------------------------
#     # Hyperparameters
#     # -------------------------------------------------
#     gamma = 0.99
#     tau = 0.005
#     batch_size = 16
#     num_episodes = 40

#     # Stage-2:
#     # keep some BC, but now add critic-guided policy term
#     actor_policy_coef = 1.0
#     actor_move_sched_bc_coef = 0.2
#     ratio_bc_coef = 0.2
#     actor_l2_coef = 1e-5

#     eval_every = 5

#     # movement + scheduling masks in raw actor output
#     M = int(state["M"])
#     K = int(state["K"])
#     move_sched_mask = np.zeros((actor_raw_action_dim,), dtype=np.float32)
#     move_sched_mask[: M + M] = 1.0
#     move_sched_mask[M + M + K:] = 1.0
#     move_sched_mask_t = torch.tensor(
#         move_sched_mask, dtype=torch.float32, device=device
#     ).unsqueeze(0)

#     best_policy_state = copy.deepcopy(policy)
#     best_eval_reward = -float("inf")

#     # -------------------------------------------------
#     # Training
#     # -------------------------------------------------
#     for ep in range(num_episodes):
#         obs = env.reset(seed=42 + ep)
#         done = False

#         episode_reward = 0.0
#         episode_actor_policy_loss = 0.0
#         episode_move_sched_bc_loss = 0.0
#         episode_ratio_bc_loss = 0.0
#         episode_critic_loss = 0.0
#         update_count = 0
#         step_count = 0

#         while not done:
#             state = obs["raw_state"]
#             access_assoc = build_access_association(state)
#             obs_vec = build_global_observation(state)

#             # -----------------------------------------
#             # teacher action (for light BC regularization)
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

#             teacher_offload_ratio = np.asarray(
#                 teacher_action["offload_ratio"],
#                 dtype=np.float32,
#             )

#             teacher_offload_ratio_refined, _ = policy._refine_high_level_action(
#                 state=state,
#                 access_assoc=access_assoc,
#                 offload_ratio=teacher_offload_ratio,
#                 sched_beta=np.asarray(teacher_action["sched_beta"], dtype=np.float32),
#             )

#             teacher_offload_ratio_refined_t = torch.tensor(
#                 teacher_offload_ratio_refined,
#                 dtype=torch.float32,
#                 device=device,
#             )

#             # -----------------------------------------
#             # execute current full proposed action
#             # -----------------------------------------
#             action = policy.act(
#                 state=state,
#                 access_assoc=access_assoc,
#                 deterministic=True,
#                 return_aux=False,
#             )
#             flat_action = flatten_full_action(action)

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
#             # critic update from replay
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
#                 critic_opt.step()

#                 episode_critic_loss += float(critic_loss.item())
#                 update_count += 1

#             # -----------------------------------------
#             # actor / encoder / ratio-head update
#             # now includes critic-guided policy loss
#             # -----------------------------------------
#             obs_tensor = torch.tensor(
#                 obs_vec, dtype=torch.float32, device=device
#             ).unsqueeze(0)
#             teacher_raw_target_t = torch.tensor(
#                 teacher_raw_target, dtype=torch.float32, device=device
#             ).unsqueeze(0)
#             teacher_offload_ratio_t = torch.tensor(
#                 teacher_offload_ratio, dtype=torch.float32, device=device
#             )

#             raw_pred = policy.actor_net(obs_tensor)  # [1, raw_dim]
#             ratio_pred = forward_ratio_branch(
#                 policy=policy,
#                 state=state,
#                 access_assoc=access_assoc,
#             )  # [K]

#             # differentiable surrogate action for critic-guided policy update
#             surrogate_full_action = build_surrogate_full_action_tensor(
#                 state=state,
#                 access_assoc=access_assoc,
#                 raw_action_pred=raw_pred,
#                 ratio_pred=ratio_pred,
#                 bandwidth_alloc_np=np.asarray(action["bandwidth_alloc"], dtype=np.float32),
#                 cpu_alloc_np=np.asarray(action["cpu_alloc"], dtype=np.float32),
#             )

#             actor_policy_loss = -critic(obs_tensor, surrogate_full_action).mean()

#             move_sched_pred = raw_pred * move_sched_mask_t
#             move_sched_target = teacher_raw_target_t * move_sched_mask_t
#             actor_move_sched_bc_loss = mse_loss(move_sched_pred, move_sched_target)

#             # ratio_bc_loss = mse_loss(ratio_pred, teacher_offload_ratio_t)
#             ratio_bc_loss = mse_loss(ratio_pred, teacher_offload_ratio_refined_t)
            
#             actor_l2_loss = (raw_pred ** 2).mean()

#             total_actor_loss = (
#                 actor_policy_coef * actor_policy_loss
#                 + actor_move_sched_bc_coef * actor_move_sched_bc_loss
#                 + ratio_bc_coef * ratio_bc_loss
#                 + actor_l2_coef * actor_l2_loss
#             )

#             actor_opt.zero_grad()
#             total_actor_loss.backward()
#             actor_opt.step()

#             # target soft update
#             soft_update_policy(target_policy, policy, tau=tau)
#             soft_update(target_critic, critic, tau=tau)

#             # bookkeeping
#             obs = next_obs
#             episode_reward += reward
#             episode_actor_policy_loss += float(actor_policy_loss.item())
#             episode_move_sched_bc_loss += float(actor_move_sched_bc_loss.item())
#             episode_ratio_bc_loss += float(ratio_bc_loss.item())
#             step_count += 1

#         print(f"\nEpisode {ep}")
#         print("episode_reward:", episode_reward)
#         print("avg_actor_policy_loss:", episode_actor_policy_loss / max(step_count, 1))
#         print("avg_move_sched_bc_loss:", episode_move_sched_bc_loss / max(step_count, 1))
#         print("avg_ratio_bc_loss:", episode_ratio_bc_loss / max(step_count, 1))
#         if update_count > 0:
#             print("avg_critic_loss:", episode_critic_loss / update_count)
#         else:
#             print("avg_critic_loss: N/A")
#         print("steps:", step_count)
#         print("buffer_size:", len(buffer))

#         # -----------------------------------------
#         # periodic evaluation
#         # -----------------------------------------
#         if (ep + 1) % eval_every == 0 or ep == 0:
#             eval_env = MultiUavMecEnv(
#                 M=3,
#                 K=8,
#                 episode_length=5,
#                 cpu_mode="kkt",
#                 omega1=100.0,
#                 omega2=1.0,
#                 deadline_scale=5.0,
#                 seed=999,
#             )

#             eval_result = evaluate_full_policy_rollout(
#                 env=eval_env,
#                 policy=policy,
#                 seed=999,
#             )

#             print("\n==============================")
#             print(f"Full Proposed Stage-2 Eval @ episode {ep}")
#             print("==============================")
#             print("episode_reward:", eval_result["episode_reward"])
#             print("avg_delay:", eval_result["avg_delay"])
#             print("avg_energy:", eval_result["avg_energy"])
#             print("avg_deadline_violation:", eval_result["avg_deadline_violation"])
#             print("feasible_ratio:", eval_result["feasible_ratio"])
#             print("num_steps:", eval_result["num_steps"])

#             if eval_result["episode_reward"] > best_eval_reward:
#                 best_eval_reward = eval_result["episode_reward"]
#                 best_policy_state = copy.deepcopy(policy)

#     # -------------------------------------------------
#     # Save best stage-2 checkpoints
#     # -------------------------------------------------
#     os.makedirs("checkpoints", exist_ok=True)

#     if best_policy_state.actor_net is not None:
#         torch.save(
#             best_policy_state.actor_net.state_dict(),
#             "checkpoints/proposed_full_stage2_best_actor.pth",
#         )
#     if best_policy_state.encoder is not None:
#         torch.save(
#             best_policy_state.encoder.state_dict(),
#             "checkpoints/proposed_full_stage2_best_encoder.pth",
#         )
#     if best_policy_state.fusion_net is not None:
#         torch.save(
#             best_policy_state.fusion_net.state_dict(),
#             "checkpoints/proposed_full_stage2_best_fusion.pth",
#         )
#     if best_policy_state.ratio_head is not None:
#         torch.save(
#             best_policy_state.ratio_head.state_dict(),
#             "checkpoints/proposed_full_stage2_best_ratio_head.pth",
#         )

#     torch.save(
#         critic.state_dict(),
#         "checkpoints/proposed_full_stage2_critic.pth",
#     )

#     print("\nFull proposed Stage-2 training finished successfully.")
#     print("Saved:")
#     print("  checkpoints/proposed_full_stage2_best_actor.pth")
#     print("  checkpoints/proposed_full_stage2_best_encoder.pth")
#     print("  checkpoints/proposed_full_stage2_best_fusion.pth")
#     print("  checkpoints/proposed_full_stage2_best_ratio_head.pth")
#     print("  checkpoints/proposed_full_stage2_critic.pth")


# if __name__ == "__main__":
#     main()

