from env.mec_env import MultiUavMecEnv
from env.association import build_access_association
from policy.conservative_policy import generate_conservative_high_action
from policy.random_policy import generate_random_high_action
from policy.proposed_placeholder_policy import generate_proposed_placeholder_action


def rollout_one_episode(env, policy_name, seed=0):
    obs = env.reset(seed=seed)
    done = False

    total_reward = 0.0
    feasible_count = 0
    total_deadline_violation = 0.0
    total_delay = 0.0
    total_energy = 0.0
    step_count = 0

    while not done:
        state = obs["raw_state"]
        access_assoc = build_access_association(state)

        if policy_name == "conservative":
            action = generate_conservative_high_action(state, access_assoc)
        elif policy_name == "random":
            action = generate_random_high_action(
                state,
                access_assoc,
                seed=seed + step_count,
            )
        elif policy_name == "proposed_placeholder":
            action = generate_proposed_placeholder_action(state, access_assoc)
        else:
            raise ValueError(f"Unknown policy_name: {policy_name}")

        obs, reward, done, info = env.step(action)

        total_reward += reward
        total_delay += info["metrics"]["delay_sys"]
        total_energy += info["metrics"]["energy_sys"]
        total_deadline_violation += info["report"]["deadline_violation"]
        feasible_count += int(info["report"]["ok"])
        step_count += 1

    return {
        "policy": policy_name,
        "episode_reward": total_reward,
        "avg_delay": total_delay / step_count,
        "avg_energy": total_energy / step_count,
        "avg_deadline_violation": total_deadline_violation / step_count,
        "feasible_ratio": feasible_count / step_count,
        "num_steps": step_count,
    }


def evaluate_policy(policy_name, seeds):
    """
    Run one policy over multiple episodes with different seeds,
    then report the average metrics.
    """
    reward_list = []
    delay_list = []
    energy_list = []
    deadline_violation_list = []
    feasible_ratio_list = []

    for seed in seeds:
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

        result = rollout_one_episode(env, policy_name=policy_name, seed=seed)

        reward_list.append(result["episode_reward"])
        delay_list.append(result["avg_delay"])
        energy_list.append(result["avg_energy"])
        deadline_violation_list.append(result["avg_deadline_violation"])
        feasible_ratio_list.append(result["feasible_ratio"])

    num_episodes = len(seeds)
    # print(f"policy={policy_name}, seed={seed}, result={result}")

    return {
        "policy": policy_name,
        "num_episodes": num_episodes,
        "avg_episode_reward": sum(reward_list) / num_episodes,
        "avg_delay": sum(delay_list) / num_episodes,
        "avg_energy": sum(energy_list) / num_episodes,
        "avg_deadline_violation": sum(deadline_violation_list) / num_episodes,
        "avg_feasible_ratio": sum(feasible_ratio_list) / num_episodes,
    }


def main():
    seeds = list(range(42, 52))  # 10 episodes: 42, 43, ..., 51

    for policy_name in ["conservative", "proposed_placeholder", "random"]:
        result = evaluate_policy(policy_name=policy_name, seeds=seeds)

        print("\n==============================")
        print("policy:", result["policy"])
        print("num_episodes:", result["num_episodes"])
        print("avg_episode_reward:", result["avg_episode_reward"])
        print("avg_delay:", result["avg_delay"])
        print("avg_energy:", result["avg_energy"])
        print("avg_deadline_violation:", result["avg_deadline_violation"])
        print("avg_feasible_ratio:", result["avg_feasible_ratio"])


if __name__ == "__main__":
    main()





# from env.mec_env import MultiUavMecEnv
# from env.association import build_access_association
# from policy.conservative_policy import generate_conservative_high_action
# from policy.random_policy import generate_random_high_action
# from policy.proposed_placeholder_policy import generate_proposed_placeholder_action


# def rollout_one_episode(env, policy_name, seed=0):
#     obs = env.reset(seed=seed)
#     done = False

#     total_reward = 0.0
#     feasible_count = 0
#     total_deadline_violation = 0.0
#     total_delay = 0.0
#     total_energy = 0.0
#     step_count = 0

#     while not done:
#         state = obs["raw_state"]
#         access_assoc = build_access_association(state)

#         if policy_name == "conservative":
#             action = generate_conservative_high_action(state, access_assoc)
#         elif policy_name == "random":
#             action = generate_random_high_action(state, access_assoc, seed=seed + step_count)
#         elif policy_name == "proposed_placeholder":
#             action = generate_proposed_placeholder_action(state, access_assoc)
#         else:
#             raise ValueError(f"Unknown policy_name: {policy_name}")

#         obs, reward, done, info = env.step(action)

#         total_reward += reward
#         total_delay += info["metrics"]["delay_sys"]
#         total_energy += info["metrics"]["energy_sys"]
#         total_deadline_violation += info["report"]["deadline_violation"]
#         feasible_count += int(info["report"]["ok"])
#         step_count += 1

#     return {
#         "policy": policy_name,
#         "episode_reward": total_reward,
#         "avg_delay": total_delay / step_count,
#         "avg_energy": total_energy / step_count,
#         "avg_deadline_violation": total_deadline_violation / step_count,
#         "feasible_ratio": feasible_count / step_count,
#         "num_steps": step_count,
#     }


# def main():
#     for policy_name in ["conservative", "proposed_placeholder", "random"]:
#         env = MultiUavMecEnv(
#             M=3,
#             K=8,
#             episode_length=5,
#             cpu_mode="kkt",
#             omega1=100.0,
#             omega2=1.0,
#             deadline_scale=5.0,
#             seed=42,
#         )

#         result = rollout_one_episode(env, policy_name=policy_name, seed=42)
#         print("\n==============================")
#         print("policy:", result["policy"])
#         print("episode_reward:", result["episode_reward"])
#         print("avg_delay:", result["avg_delay"])
#         print("avg_energy:", result["avg_energy"])
#         print("avg_deadline_violation:", result["avg_deadline_violation"])
#         print("feasible_ratio:", result["feasible_ratio"])
#         print("num_steps:", result["num_steps"])


# if __name__ == "__main__":
#     main()