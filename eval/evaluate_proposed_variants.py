# 这个脚本的目标很明确：
# 用同一套 seeds
# 在同一套环境参数
# 统一比较下面几种 Proposed 变体：
# placeholder
# full_stage1
# full_stage2
# full_stage2b

# 输出指标包括：
# mean_episode_reward
# mean_avg_delay
# mean_avg_energy
# mean_avg_deadline_violation
# mean_feasible_ratio
# 这样你就能非常清楚地判断，当前哪一个 checkpoint 才是你论文里 Proposed 方法的最佳代表版本。


import os
from typing import Dict, List, Optional

import numpy as np
import torch

from env.mec_env import MultiUavMecEnv
from env.association import build_access_association
from model.mlp_actor import MLPActor
from model.proposed_obs_builder import get_observation_dim
from policy.proposed_policy import build_default_proposed_policy
from train.train_proposed_full_stage1 import get_actor_raw_action_dim


# =========================================================
# Config
# =========================================================
DEVICE = "cpu"

ENV_CONFIG = {
    "M": 3,
    "K": 8,
    "episode_length": 5,
    "cpu_mode": "kkt",
    "omega1": 100.0,
    "omega2": 1.0,
    "deadline_scale": 5.0,
}

EVAL_SEEDS = [1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009]


# =========================================================
# Checkpoint utils
# =========================================================
def load_if_exists(module: torch.nn.Module, path: str, name: str):
    if os.path.exists(path):
        module.load_state_dict(torch.load(path, map_location="cpu"))
        print(f"Loaded {name} from: {path}")
        return True
    print(f"WARNING: {name} checkpoint not found: {path}")
    return False


# =========================================================
# Policy builders
# =========================================================
def build_placeholder_policy():
    env = MultiUavMecEnv(seed=42, **ENV_CONFIG)
    obs = env.reset(seed=42)
    state = obs["raw_state"]

    policy = build_default_proposed_policy(
        state=state,
        actor_net=None,
        device=DEVICE,
        embed_dim=128,
        num_heads=4,
        ff_hidden_dim=256,
        num_layers=2,
    )
    return policy


def build_learnable_policy_with_checkpoints(
    actor_path: str,
    encoder_path: str,
    fusion_path: str,
    ratio_head_path: str,
):
    env = MultiUavMecEnv(seed=42, **ENV_CONFIG)
    obs = env.reset(seed=42)
    state = obs["raw_state"]

    obs_dim = get_observation_dim(state)
    actor_raw_action_dim = get_actor_raw_action_dim(state)

    actor = MLPActor(
        obs_dim=obs_dim,
        action_dim=actor_raw_action_dim,
        hidden_dim=256,
    ).to(DEVICE)

    policy = build_default_proposed_policy(
        state=state,
        actor_net=actor,
        device=DEVICE,
        embed_dim=128,
        num_heads=4,
        ff_hidden_dim=256,
        num_layers=2,
    )

    ok_actor = load_if_exists(policy.actor_net, actor_path, "actor")
    ok_encoder = load_if_exists(policy.encoder, encoder_path, "encoder")
    ok_fusion = load_if_exists(policy.fusion_net, fusion_path, "fusion_net")
    ok_ratio = load_if_exists(policy.ratio_head, ratio_head_path, "ratio_head")

    all_ok = ok_actor and ok_encoder and ok_fusion and ok_ratio
    return policy, all_ok


# =========================================================
# Evaluation
# =========================================================
@torch.no_grad()
def evaluate_policy_over_seeds(policy, seeds: List[int]) -> Dict[str, float]:
    rewards = []
    delays = []
    energies = []
    deadline_violations = []
    feasible_ratios = []
    num_steps_all = []

    for seed in seeds:
        env = MultiUavMecEnv(seed=seed, **ENV_CONFIG)

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

            action = policy.act(
                state=state,
                access_assoc=access_assoc,
                deterministic=True,
                return_aux=False,
            )

            obs, reward, done, info = env.step(action)

            total_reward += reward
            total_delay += info["metrics"]["delay_sys"]
            total_energy += info["metrics"]["energy_sys"]
            total_deadline_violation += info["report"]["deadline_violation"]
            feasible_count += int(info["report"]["ok"])
            step_count += 1

        rewards.append(total_reward)
        delays.append(total_delay / max(step_count, 1))
        energies.append(total_energy / max(step_count, 1))
        deadline_violations.append(total_deadline_violation / max(step_count, 1))
        feasible_ratios.append(feasible_count / max(step_count, 1))
        num_steps_all.append(step_count)

    return {
        "mean_episode_reward": float(np.mean(rewards)),
        "std_episode_reward": float(np.std(rewards)),
        "mean_avg_delay": float(np.mean(delays)),
        "std_avg_delay": float(np.std(delays)),
        "mean_avg_energy": float(np.mean(energies)),
        "std_avg_energy": float(np.std(energies)),
        "mean_avg_deadline_violation": float(np.mean(deadline_violations)),
        "std_avg_deadline_violation": float(np.std(deadline_violations)),
        "mean_feasible_ratio": float(np.mean(feasible_ratios)),
        "std_feasible_ratio": float(np.std(feasible_ratios)),
        "mean_num_steps": float(np.mean(num_steps_all)),
    }


def print_result_block(name: str, result: Dict[str, float]):
    print("\n" + "=" * 80)
    print(name)
    print("=" * 80)
    for k, v in result.items():
        print(f"{k}: {v}")


# =========================================================
# Main
# =========================================================
def main():
    np.random.seed(42)
    torch.manual_seed(42)

    print("Evaluation seeds:", EVAL_SEEDS)
    print("Environment config:", ENV_CONFIG)

    all_results = {}

    # -------------------------------------------------
    # 1) Placeholder
    # -------------------------------------------------
    print("\nBuilding placeholder policy...")
    placeholder_policy = build_placeholder_policy()
    placeholder_result = evaluate_policy_over_seeds(
        policy=placeholder_policy,
        seeds=EVAL_SEEDS,
    )
    all_results["placeholder"] = placeholder_result
    print_result_block("Placeholder", placeholder_result)

    # -------------------------------------------------
    # 2) Full Stage-1
    # -------------------------------------------------
    print("\nBuilding full_stage1 policy...")
    full_stage1_policy, ok_stage1 = build_learnable_policy_with_checkpoints(
        actor_path="checkpoints/proposed_full_stage1_best_actor.pth",
        encoder_path="checkpoints/proposed_full_stage1_best_encoder.pth",
        fusion_path="checkpoints/proposed_full_stage1_best_fusion.pth",
        ratio_head_path="checkpoints/proposed_full_stage1_best_ratio_head.pth",
    )
    if ok_stage1:
        stage1_result = evaluate_policy_over_seeds(
            policy=full_stage1_policy,
            seeds=EVAL_SEEDS,
        )
        all_results["full_stage1"] = stage1_result
        print_result_block("Full Stage-1", stage1_result)
    else:
        print("Skipping full_stage1 because some checkpoints are missing.")

    # -------------------------------------------------
    # 3) Full Stage-2
    # -------------------------------------------------
    print("\nBuilding full_stage2 policy...")
    full_stage2_policy, ok_stage2 = build_learnable_policy_with_checkpoints(
        actor_path="checkpoints/proposed_full_stage2_best_actor.pth",
        encoder_path="checkpoints/proposed_full_stage2_best_encoder.pth",
        fusion_path="checkpoints/proposed_full_stage2_best_fusion.pth",
        ratio_head_path="checkpoints/proposed_full_stage2_best_ratio_head.pth",
    )
    if ok_stage2:
        stage2_result = evaluate_policy_over_seeds(
            policy=full_stage2_policy,
            seeds=EVAL_SEEDS,
        )
        all_results["full_stage2"] = stage2_result
        print_result_block("Full Stage-2", stage2_result)
    else:
        print("Skipping full_stage2 because some checkpoints are missing.")

    # -------------------------------------------------
    # 4) Full Stage-2b
    # -------------------------------------------------
    print("\nBuilding full_stage2b policy...")
    full_stage2b_policy, ok_stage2b = build_learnable_policy_with_checkpoints(
        actor_path="checkpoints/proposed_full_stage2b_best_actor.pth",
        encoder_path="checkpoints/proposed_full_stage2b_best_encoder.pth",
        fusion_path="checkpoints/proposed_full_stage2b_best_fusion.pth",
        ratio_head_path="checkpoints/proposed_full_stage2b_best_ratio_head.pth",
    )
    if ok_stage2b:
        stage2b_result = evaluate_policy_over_seeds(
            policy=full_stage2b_policy,
            seeds=EVAL_SEEDS,
        )
        all_results["full_stage2b"] = stage2b_result
        print_result_block("Full Stage-2b", stage2b_result)
    else:
        print("Skipping full_stage2b because some checkpoints are missing.")

    # -------------------------------------------------
    # Summary ranking by reward
    # -------------------------------------------------
    print("\n" + "#" * 80)
    print("Summary Ranking by mean_episode_reward")
    print("#" * 80)

    sortable = []
    for name, result in all_results.items():
        sortable.append((name, result["mean_episode_reward"]))

    sortable.sort(key=lambda x: x[1], reverse=True)

    for rank, (name, score) in enumerate(sortable, start=1):
        print(f"{rank}. {name}: {score}")

    # -------------------------------------------------
    # Compact comparison table
    # -------------------------------------------------
    print("\n" + "#" * 80)
    print("Compact Comparison")
    print("#" * 80)

    for name, result in all_results.items():
        print(
            f"{name} | "
            f"reward={result['mean_episode_reward']:.6f} | "
            f"delay={result['mean_avg_delay']:.6f} | "
            f"energy={result['mean_avg_energy']:.6f} | "
            f"deadline_violation={result['mean_avg_deadline_violation']:.6f} | "
            f"feasible_ratio={result['mean_feasible_ratio']:.6f}"
        )

    print("\nEvaluation finished successfully.")


if __name__ == "__main__":
    main()