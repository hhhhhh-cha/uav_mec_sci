from env.mec_env import MultiUavMecEnv
from env.association import build_access_association
from policy.proposed_placeholder_policy import generate_proposed_placeholder_action


def main():
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
    print("reset done, slot =", obs["slot"])

    done = False
    step_idx = 0

    total_reward = 0.0
    total_delay = 0.0
    total_energy = 0.0
    total_deadline_violation = 0.0
    feasible_count = 0

    while not done:
        state = obs["raw_state"]
        access_assoc = build_access_association(state)

        high_action = generate_proposed_placeholder_action(
            state=state,
            access_assoc=access_assoc,
        )

        obs, reward, done, info = env.step(high_action)

        total_reward += reward
        total_delay += info["metrics"]["delay_sys"]
        total_energy += info["metrics"]["energy_sys"]
        total_deadline_violation += info["report"]["deadline_violation"]
        feasible_count += int(info["report"]["ok"])

        print(f"\n===== step {step_idx} =====")
        print("reward:", reward)
        print("done:", done)
        print("delay_sys:", info["metrics"]["delay_sys"])
        print("energy_sys:", info["metrics"]["energy_sys"])
        print("feasible:", info["report"]["ok"])
        print("deadline_violation:", info["report"]["deadline_violation"])
        print("battery:", obs["raw_state"]["uav_energy"])
        print("offload_ratio:", info["high_action"]["offload_ratio"])
        print("energy_fly:", info["metrics"].get("energy_fly"))
        print("energy_tx:", info["metrics"].get("energy_tx"))
        print("energy_cmp:", info["metrics"].get("energy_cmp"))

        step_idx += 1

    print("\n==============================")
    print("Proposed Placeholder Summary")
    print("==============================")
    print("episode_reward:", total_reward)
    print("avg_delay:", total_delay / step_idx)
    print("avg_energy:", total_energy / step_idx)
    print("avg_deadline_violation:", total_deadline_violation / step_idx)
    print("feasible_ratio:", feasible_count / step_idx)
    print("num_steps:", step_idx)


if __name__ == "__main__":
    main()