# 这个测试的目的很明确，不是训练，而是先验证这条主方法链是否都能接起来：
# proposed_policy.py
# transformer_encoder.py
# proposed_learned_policy.py
# env.step()
# bandwidth_solver.py
# cpu_solver.py
# 也就是先检查你论文第四节这条完整接口，能不能“从 state 进，到 full action 出，再送进环境”。这一步对应你论文里从高层决策到低层解析分配的主流程

import numpy as np
import torch

from env.mec_env import MultiUavMecEnv
from env.association import build_access_association
from model.mlp_actor import MLPActor
from model.proposed_obs_builder import get_observation_dim
from policy.proposed_policy import build_default_proposed_policy


def get_actor_raw_action_dim(state):
    """
    Raw actor output layout:
        [ move_dist(M), move_angle(M), offload_ratio(K), sched_score(K*M) ]
    """
    M = int(state["M"])
    K = int(state["K"])
    return M + M + K + K * M


def print_action_summary(action):
    print("\n[Action Summary]")
    print("move_dist shape:", np.asarray(action["move_dist"]).shape)
    print("move_angle shape:", np.asarray(action["move_angle"]).shape)
    print("offload_ratio shape:", np.asarray(action["offload_ratio"]).shape)
    print("sched_beta shape:", np.asarray(action["sched_beta"]).shape)

    if "bandwidth_alloc" in action:
        print("bandwidth_alloc shape:", np.asarray(action["bandwidth_alloc"]).shape)
    if "cpu_alloc" in action:
        print("cpu_alloc shape:", np.asarray(action["cpu_alloc"]).shape)

    print("move_dist:", np.asarray(action["move_dist"]))
    print("move_angle:", np.asarray(action["move_angle"]))
    print("offload_ratio:", np.asarray(action["offload_ratio"]))

    sched_beta = np.asarray(action["sched_beta"])
    K, M, _ = sched_beta.shape
    exec_list = []
    for k in range(K):
        access_m = int(np.argmax(np.sum(sched_beta[k], axis=1)))
        exec_j = int(np.argmax(sched_beta[k, access_m]))
        exec_list.append((k, access_m, exec_j))
    print("task/access/exec:", exec_list)


def validate_action_shapes(state, action):
    M = int(state["M"])
    K = int(state["K"])

    assert np.asarray(action["move_dist"]).shape == (M,)
    assert np.asarray(action["move_angle"]).shape == (M,)
    assert np.asarray(action["offload_ratio"]).shape == (K,)
    assert np.asarray(action["sched_beta"]).shape == (K, M, M)

    if "bandwidth_alloc" in action:
        assert np.asarray(action["bandwidth_alloc"]).shape == (M, K)
    if "cpu_alloc" in action:
        assert np.asarray(action["cpu_alloc"]).shape == (M, K)


def run_placeholder_mode_test():
    print("\n" + "=" * 80)
    print("Test 1: ProposedPolicy in placeholder mode")
    print("=" * 80)

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
    access_assoc = build_access_association(state)

    policy = build_default_proposed_policy(
        state=state,
        actor_net=None,
        device="cpu",
    )

    action, aux = policy.act(
        state=state,
        access_assoc=access_assoc,
        deterministic=True,
        return_aux=True,
    )

    validate_action_shapes(state, action)
    print_action_summary(action)

    next_obs, reward, done, info = env.step(action)

    print("\n[Env Step Result - Placeholder]")
    print("reward:", reward)
    print("done:", done)
    print("delay_sys:", info["metrics"]["delay_sys"])
    print("energy_sys:", info["metrics"]["energy_sys"])
    print("deadline_violation:", info["report"]["deadline_violation"])
    print("feasible:", info["report"]["ok"])

    print("\nPlaceholder mode full pipeline test passed.")
    return next_obs


def run_network_mode_test():
    print("\n" + "=" * 80)
    print("Test 2: ProposedPolicy in network mode")
    print("=" * 80)

    env = MultiUavMecEnv(
        M=3,
        K=8,
        episode_length=5,
        cpu_mode="kkt",
        omega1=100.0,
        omega2=1.0,
        deadline_scale=5.0,
        seed=123,
    )

    obs = env.reset(seed=123)
    state = obs["raw_state"]
    access_assoc = build_access_association(state)

    obs_dim = get_observation_dim(state)
    action_dim = get_actor_raw_action_dim(state)

    actor = MLPActor(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dim=256,
    ).to("cpu")

    policy = build_default_proposed_policy(
        state=state,
        actor_net=actor,
        device="cpu",
    )

    action, aux = policy.act(
        state=state,
        access_assoc=access_assoc,
        deterministic=True,
        return_aux=True,
    )

    validate_action_shapes(state, action)
    print_action_summary(action)

    print("\n[Aux Info]")
    if len(aux) == 0:
        print("aux is empty")
    else:
        for k, v in aux.items():
            if hasattr(v, "shape"):
                print(f"{k} shape: {v.shape}")
            elif isinstance(v, list):
                print(f"{k} length: {len(v)}")
            else:
                print(f"{k}: {type(v)}")

    next_obs, reward, done, info = env.step(action)

    print("\n[Env Step Result - Network]")
    print("reward:", reward)
    print("done:", done)
    print("delay_sys:", info["metrics"]["delay_sys"])
    print("energy_sys:", info["metrics"]["energy_sys"])
    print("deadline_violation:", info["report"]["deadline_violation"])
    print("feasible:", info["report"]["ok"])

    print("\nNetwork mode full pipeline test passed.")
    return next_obs


def run_two_step_smoke_test():
    print("\n" + "=" * 80)
    print("Test 3: Two-step smoke rollout")
    print("=" * 80)

    env = MultiUavMecEnv(
        M=3,
        K=8,
        episode_length=5,
        cpu_mode="kkt",
        omega1=100.0,
        omega2=1.0,
        deadline_scale=5.0,
        seed=7,
    )

    obs = env.reset(seed=7)
    state = obs["raw_state"]

    obs_dim = get_observation_dim(state)
    action_dim = get_actor_raw_action_dim(state)

    actor = MLPActor(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dim=256,
    ).to("cpu")

    policy = build_default_proposed_policy(
        state=state,
        actor_net=actor,
        device="cpu",
    )

    total_reward = 0.0

    for step in range(2):
        state = obs["raw_state"]
        access_assoc = build_access_association(state)

        action = policy.act(
            state=state,
            access_assoc=access_assoc,
            deterministic=True,
            return_aux=False,
        )

        validate_action_shapes(state, action)

        obs, reward, done, info = env.step(action)
        total_reward += reward

        print(f"\n[Step {step}]")
        print("reward:", reward)
        print("delay_sys:", info["metrics"]["delay_sys"])
        print("energy_sys:", info["metrics"]["energy_sys"])
        print("deadline_violation:", info["report"]["deadline_violation"])
        print("feasible:", info["report"]["ok"])

        if done:
            break

    print("\nTwo-step smoke rollout passed.")
    print("total_reward:", total_reward)


def main():
    np.random.seed(42)
    torch.manual_seed(42)

    run_placeholder_mode_test()
    run_network_mode_test()
    run_two_step_smoke_test()

    print("\n" + "=" * 80)
    print("All ProposedPolicy full-pipeline tests passed successfully.")
    print("=" * 80)


if __name__ == "__main__":
    main()