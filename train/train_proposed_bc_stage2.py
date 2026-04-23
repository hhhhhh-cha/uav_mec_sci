# 它做的事是：
# 用 placeholder 生成 teacher action
# network actor 输出 raw action
# decode 后得到 env-style action
# 把 actor 输出动作和 teacher 动作做 supervised loss
# 训练 actor 学 placeholder

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


from env.mec_env import MultiUavMecEnv
from env.association import build_access_association
from model.proposed_obs_builder import build_global_observation, get_observation_dim
from model.mlp_actor import MLPActor
from policy.proposed_placeholder_policy import generate_proposed_placeholder_action
from policy.proposed_learned_policy import ProposedLearnedPolicy


EPS = 1e-8


def get_actor_raw_action_dim(state):
    """
    Raw actor output layout:
        [ move_dist(M), move_angle(M), offload_ratio(K), sched_score(K*M) ]
    """
    M = int(state["M"])
    K = int(state["K"])
    return M + M + K + K * M


def build_teacher_raw_target(state, access_assoc, teacher_action):
    """
    Convert env-style teacher action dict into the raw-action target format
    expected by the actor output.

    Raw actor layout:
        [ move_dist(M), move_angle(M), offload_ratio(K), sched_score(K*M) ]

    Design choice:
    - move_dist target is normalized into tanh-preimage-compatible range
      approximately by simple scaling
    - move_angle target is normalized by pi
    - offload ratio target is mapped from [0,1] to [-1,1] style target range
    - sched_score target uses +1 for chosen execution UAV, -1 otherwise
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

    # -------------------------------------------------
    # move_dist target: map [0, max_move] -> [-1, 1]
    # -------------------------------------------------
    move_dist_target = 2.0 * (move_dist / max_move) - 1.0
    move_dist_target = np.clip(move_dist_target, -1.0, 1.0)

    # -------------------------------------------------
    # move_angle target: map [-pi, pi] -> [-1, 1]
    # -------------------------------------------------
    move_angle_target = move_angle / np.pi
    move_angle_target = np.clip(move_angle_target, -1.0, 1.0)

    # -------------------------------------------------
    # offload target: map [0, 1] -> [-1, 1]
    # -------------------------------------------------
    offload_target = 2.0 * offload_ratio - 1.0
    offload_target = np.clip(offload_target, -1.0, 1.0)

    # -------------------------------------------------
    # scheduling score target: [K, M]
    # only chosen j gets +1, others -1
    # -------------------------------------------------
    sched_score_target = -np.ones((K, M), dtype=np.float32)

    for k in range(K):
        access_m = int(np.argmax(access_assoc[:, k]))
        chosen_js = np.where(sched_beta[k, access_m, :] > 0.5)[0]
        if len(chosen_js) == 1:
            j = int(chosen_js[0])
            sched_score_target[k, j] = 1.0

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


def evaluate_policy_rollout(env, policy, seed=123):
    """
    Run one episode using the learned policy and return summary metrics.
    """
    obs = env.reset(seed=seed)
    done = False

    total_reward = 0.0
    total_delay = 0.0
    total_energy = 0.0
    total_deadline_violation = 0.0
    feasible_count = 0
    step_count = 0

    while not done:
        state = obs["raw_state"]
        access_assoc = build_access_association(state)

        high_action = policy.act(
            state=state,
            access_assoc=access_assoc,
            deterministic=True,
        )

        obs, reward, done, info = env.step(high_action)

        total_reward += reward
        total_delay += info["metrics"]["delay_sys"]
        total_energy += info["metrics"]["energy_sys"]
        total_deadline_violation += info["report"]["deadline_violation"]
        feasible_count += int(info["report"]["ok"])
        step_count += 1

    return {
        "episode_reward": total_reward,
        "avg_delay": total_delay / step_count,
        "avg_energy": total_energy / step_count,
        "avg_deadline_violation": total_deadline_violation / step_count,
        "feasible_ratio": feasible_count / step_count,
        "num_steps": step_count,
    }


def collect_bc_dataset(num_episodes=20, base_seed=100):
    """
    Collect supervised dataset:
        obs -> teacher_raw_target
    using the proposed placeholder policy as teacher.
    """
    obs_list = []
    target_list = []

    for ep in range(num_episodes):
        seed = base_seed + ep

        env = MultiUavMecEnv(
            M=3,
            K=8,
            episode_length=5,
            cpu_mode="kkt",
            omega1=100.0,
            omega2=1.0,
            deadline_scale=5.0,
            seed=seed,
        )

        obs = env.reset(seed=seed)
        done = False

        while not done:
            state = obs["raw_state"]
            access_assoc = build_access_association(state)

            obs_vec = build_global_observation(state)

            teacher_action = generate_proposed_placeholder_action(
                state=state,
                access_assoc=access_assoc,
            )
            teacher_raw_target = build_teacher_raw_target(
                state=state,
                access_assoc=access_assoc,
                teacher_action=teacher_action,
            )

            obs_list.append(obs_vec)
            target_list.append(teacher_raw_target)

            obs, _, done, _ = env.step(teacher_action)

    obs_arr = np.asarray(obs_list, dtype=np.float32)
    target_arr = np.asarray(target_list, dtype=np.float32)

    return obs_arr, target_arr


def main():
    device = "cpu"
    torch.manual_seed(42)
    np.random.seed(42)

    # -------------------------------------------------
    # Build one env just to infer dimensions
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
    action_dim = get_actor_raw_action_dim(state)

    print("obs_dim:", obs_dim)
    print("actor_raw_action_dim:", action_dim)

    # -------------------------------------------------
    # Collect BC dataset from placeholder teacher
    # -------------------------------------------------
    obs_data, target_data = collect_bc_dataset(
        num_episodes=20,
        base_seed=100,
    )

    print("bc dataset obs shape:", obs_data.shape)
    print("bc dataset target shape:", target_data.shape)

    obs_tensor = torch.tensor(obs_data, dtype=torch.float32, device=device)
    target_tensor = torch.tensor(target_data, dtype=torch.float32, device=device)

    # -------------------------------------------------
    # Actor to be trained
    # -------------------------------------------------
    actor = MLPActor(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dim=256,
    ).to(device)

    optimizer = optim.Adam(actor.parameters(), lr=1e-3)
    mse_loss = nn.MSELoss()

    # -------------------------------------------------
    # Behavior cloning training
    # -------------------------------------------------
    num_epochs = 300
    batch_size = 32
    num_samples = obs_tensor.shape[0]

    best_actor = None
    best_loss = float("inf")

    for epoch in range(num_epochs):
        perm = torch.randperm(num_samples, device=device)

        epoch_loss = 0.0
        batch_count = 0

        for start in range(0, num_samples, batch_size):
            idx = perm[start:start + batch_size]

            obs_b = obs_tensor[idx]
            target_b = target_tensor[idx]

            pred_b = actor(obs_b)
            loss = mse_loss(pred_b, target_b)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.item())
            batch_count += 1

        epoch_loss /= max(batch_count, 1)

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_actor = copy.deepcopy(actor)

        if epoch % 50 == 0 or epoch == num_epochs - 1:
            print(f"epoch={epoch}, bc_loss={epoch_loss:.6f}")

    if best_actor is None:
        best_actor = actor

    # -------------------------------------------------
    # Evaluate learned actor in env through unified policy
    # -------------------------------------------------
    learned_policy = ProposedLearnedPolicy(
        mode="network",
        actor_net=best_actor,
        device=device,
    )

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

    result = evaluate_policy_rollout(
        env=eval_env,
        policy=learned_policy,
        seed=999,
    )

    print("\n==============================")
    print("BC Stage-2 Evaluation")
    print("==============================")
    print("episode_reward:", result["episode_reward"])
    print("avg_delay:", result["avg_delay"])
    print("avg_energy:", result["avg_energy"])
    print("avg_deadline_violation:", result["avg_deadline_violation"])
    print("feasible_ratio:", result["feasible_ratio"])
    print("num_steps:", result["num_steps"])
    print("best_bc_loss:", best_loss)
    print("avg_ratio_violation:", eval_result["avg_ratio_violation"])
    print("avg_assoc_violation:", eval_result["avg_assoc_violation"])
    print("avg_schedule_violation:", eval_result["avg_schedule_violation"])
    print("avg_candidate_violation:", eval_result["avg_candidate_violation"])
    print("avg_bw_violation:", eval_result["avg_bw_violation"])
    print("avg_cpu_violation:", eval_result["avg_cpu_violation"])
    print("avg_rate_violation:", eval_result["avg_rate_violation"])
    print("avg_nan_count:", eval_result["avg_nan_count"])

    import os

    os.makedirs("checkpoints", exist_ok=True)
    save_path = "checkpoints/proposed_bc_stage2_actor.pth"
    torch.save(
        {
            "actor_state_dict": best_actor.state_dict(),
            "obs_dim": obs_dim,
            "action_dim": action_dim,
            "best_bc_loss": best_loss,
            "num_epochs": num_epochs,
            "dataset_size": num_samples,
        },
        save_path,
    )
    print(f"Stage-2 actor saved to: {save_path}")
if __name__ == "__main__":
    main()