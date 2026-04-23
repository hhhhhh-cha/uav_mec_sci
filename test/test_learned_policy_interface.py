import numpy as np
import torch

from env.mec_env import MultiUavMecEnv
from env.association import build_access_association
from model.proposed_obs_builder import build_global_observation, get_observation_dim
from model.mlp_actor import MLPActor
from policy.proposed_learned_policy import ProposedLearnedPolicy


def get_action_dim(state):
    """
    Raw action layout:
        [ move_dist(M), move_angle(M), offload_ratio(K), sched_score(K*M) ]
    """
    M = int(state["M"])
    K = int(state["K"])
    return M + M + K + K * M


def main():
    device = "cpu"

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
    action_dim = get_action_dim(state)

    print("obs_dim:", obs_dim)
    print("action_dim:", action_dim)

    actor = MLPActor(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dim=256,
    ).to(device)

    policy = ProposedLearnedPolicy(
        mode="network",
        actor_net=actor,
        device=device,
    )

    done = False
    step_idx = 0

    while not done:
        state = obs["raw_state"]
        access_assoc = build_access_association(state)

        obs_vec = build_global_observation(state)
        print(f"\n===== step {step_idx} =====")
        print("obs_vec.shape:", obs_vec.shape)

        high_action = policy.act(
            state=state,
            access_assoc=access_assoc,
            deterministic=True,
        )

        print("move_dist shape:", np.asarray(high_action["move_dist"]).shape)
        print("move_angle shape:", np.asarray(high_action["move_angle"]).shape)
        print("offload_ratio shape:", np.asarray(high_action["offload_ratio"]).shape)
        print("sched_beta shape:", np.asarray(high_action["sched_beta"]).shape)

        obs, reward, done, info = env.step(high_action)

        print("reward:", reward)
        print("delay_sys:", info["metrics"]["delay_sys"])
        print("energy_sys:", info["metrics"]["energy_sys"])
        print("feasible:", info["report"]["ok"])
        print("deadline_violation:", info["report"]["deadline_violation"])

        step_idx += 1

    print("\nInterface test finished successfully.")


if __name__ == "__main__":
    main()