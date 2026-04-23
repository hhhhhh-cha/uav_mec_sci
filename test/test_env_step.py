import numpy as np

from env.mec_env import MultiUavMecEnv
from env.association import build_access_association, heuristic_generate_high_action


def generate_conservative_high_action(state, access_assoc):
    """
    Conservative hand-crafted action for environment sanity check.

    Design:
    1) no movement
    2) moderate offloading ratio
    3) execute on access UAV itself (no A2A backhaul)
    """
    M, K = access_assoc.shape

    move_dist = np.zeros(M, dtype=float)
    move_angle = np.zeros(M, dtype=float)

    # Conservative partial offloading
    offload_ratio = 0.2 * np.ones(K, dtype=float)

    # sched_beta[k, access_m, access_m] = 1
    sched_beta = np.zeros((K, M, M), dtype=float)
    for k in range(K):
        access_m = int(np.argmax(access_assoc[:, k]))
        sched_beta[k, access_m, access_m] = 1.0

    return {
        "move_dist": move_dist,
        "move_angle": move_angle,
        "offload_ratio": offload_ratio,
        "sched_beta": sched_beta,
    }


def main():
    env = MultiUavMecEnv(
        M=3,
        K=8,
        episode_length=5,
        cpu_mode="kkt",   # 或 "prop"
        omega1=100.0,
        omega2=1.0,
        deadline_scale=5.0,   # 先用宽松版本做联调
        seed=42,
    )

    obs = env.reset(seed=42)
    print("reset done, slot =", obs["slot"])

    done = False
    step_idx = 0

    # 切换开关：
    # True  -> 用保守动作测试
    # False -> 用你当前 heuristic_generate_high_action
    use_conservative_action = True

    while not done:
        state = obs["raw_state"]
        access_assoc = build_access_association(state)

        if use_conservative_action:
            high_action = generate_conservative_high_action(
                state=state,
                access_assoc=access_assoc,
            )
        else:
            high_action = heuristic_generate_high_action(
                state=state,
                access_assoc=access_assoc,
                seed=100 + step_idx,
            )

        obs, reward, done, info = env.step(high_action)

        print(f"\n===== step {step_idx} =====")
        print("reward:", reward)
        print("done:", done)
        print("delay_sys:", info["metrics"]["delay_sys"])
        print("energy_sys:", info["metrics"]["energy_sys"])
        print("feasible:", info["report"]["ok"])
        print("deadline_violation:", info["report"]["deadline_violation"])
        print("battery:", obs["raw_state"]["uav_energy"])
        print("energy_fly:", info["metrics"].get("energy_fly"))
        print("energy_tx:", info["metrics"].get("energy_tx"))
        print("energy_cmp:", info["metrics"].get("energy_cmp"))
        print("uav_energy_before_step:", info.get("uav_energy_before"))
        print("uav_energy_after_step:", info.get("uav_energy_after"))
        step_idx += 1


if __name__ == "__main__":
    main()




# import numpy as np

# from env.mec_env import MultiUavMecEnv
# from env.association import build_access_association, heuristic_generate_high_action


# def main():
#     env = MultiUavMecEnv(
#         M=3,
#         K=8,
#         episode_length=5,
#         cpu_mode="kkt",   # 或 "prop"
#         omega1=100.0,
#         omega2=1.0,
#         seed=42,
#     )

#     obs = env.reset(seed=42)
#     print("reset done, slot =", obs["slot"])

#     done = False
#     step_idx = 0

#     while not done:
#         state = obs["raw_state"]
#         access_assoc = build_access_association(state)

#         # 先临时继续用你现在的 heuristic 作为高层动作占位
#         high_action = heuristic_generate_high_action(
#             state=state,
#             access_assoc=access_assoc,
#             seed=100 + step_idx,
#         )

#         obs, reward, done, info = env.step(high_action)

#         print(f"\n===== step {step_idx} =====")
#         print("reward:", reward)
#         print("done:", done)
#         print("delay_sys:", info["metrics"]["delay_sys"])
#         print("energy_sys:", info["metrics"]["energy_sys"])
#         print("feasible:", info["report"]["ok"])
#         print("deadline_violation:", info["report"]["deadline_violation"])
#         print("battery:", obs["raw_state"]["uav_energy"])
#         print("energy_fly:", info["metrics"].get("energy_fly"))
#         print("energy_tx:", info["metrics"].get("energy_tx"))
#         print("energy_cmp:", info["metrics"].get("energy_cmp"))
#         print("uav_energy_before_step:", info.get("uav_energy_before"))
#         print("uav_energy_after_step:", info.get("uav_energy_after"))
#         step_idx += 1


# if __name__ == "__main__":
#     main()