from __future__ import annotations

import copy

import random
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch

from env.mec_env import MultiUavMecEnv
from env.association import build_access_association

from policy.greedy_policy import generate_greedy_high_action
from policy.proposed_policy import build_default_proposed_policy

from model.mlp_actor import MLPActor
from model.proposed_obs_builder import build_global_observation


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPS = 1e-8


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _extract_uav_pos_from_state(state):
    for key in ["uav_pos", "uav_positions", "q", "uav_xy"]:
        if key in state:
            arr = np.asarray(state[key], dtype=np.float32)
            if arr.ndim == 2 and arr.shape[1] >= 2:
                return arr[:, :2]
    raise KeyError("Cannot find UAV position array in state.")


def _print_mobility_debug(tag, state_before, state_after, action):
    q_before = _extract_uav_pos_from_state(state_before)
    q_after = _extract_uav_pos_from_state(state_after)

    move_dist_cmd = np.asarray(action["move_dist"], dtype=np.float32)
    move_angle_cmd = np.asarray(action["move_angle"], dtype=np.float32)

    delta_xy = q_after - q_before
    real_move_dist = np.linalg.norm(delta_xy, axis=1)
    real_move_angle = np.arctan2(delta_xy[:, 1], delta_xy[:, 0] + 1e-12)

    print(f"\n[{tag} MOBILITY DEBUG]")
    print("q_before:")
    print(np.round(q_before, 4))
    print("q_after :")
    print(np.round(q_after, 4))
    print("delta_xy:")
    print(np.round(delta_xy, 6))

    print("cmd move_dist :", np.round(move_dist_cmd, 6))
    print("real move_dist:", np.round(real_move_dist, 6))
    print("cmd move_angle :", np.round(move_angle_cmd, 6))
    print("real move_angle:", np.round(real_move_angle, 6))


def _extract_task_pos_from_state(state):
    for key in ["task_pos", "task_positions", "td_pos", "user_pos", "w"]:
        if key in state:
            arr = np.asarray(state[key], dtype=np.float32)
            if arr.ndim == 2 and arr.shape[1] >= 2:
                return arr[:, :2]
    return None


def _print_channel_distance_debug(tag, state_before, state_after, access_assoc):
    q_before = _extract_uav_pos_from_state(state_before)
    q_after = _extract_uav_pos_from_state(state_after)
    M, K = access_assoc.shape

    print(f"\n[{tag} DISTANCE DEBUG]")

    # A2A
    print("A2A pair distance change:")
    for m in range(M):
        for j in range(m + 1, M):
            d0 = float(np.linalg.norm(q_before[m] - q_before[j]))
            d1 = float(np.linalg.norm(q_after[m] - q_after[j]))
            print(f"  ({m},{j}): before={d0:.4f}, after={d1:.4f}, delta={d1-d0:.6f}")

    # A2G for associated task
    task_pos = _extract_task_pos_from_state(state_before)
    if task_pos is not None:
        print("A2G access distance change:")
        for k in range(K):
            m = int(np.argmax(access_assoc[:, k]))
            d0 = float(np.linalg.norm(q_before[m] - task_pos[k]))
            d1 = float(np.linalg.norm(q_after[m] - task_pos[k]))
            print(f"  task {k} via UAV {m}: before={d0:.4f}, after={d1:.4f}, delta={d1-d0:.6f}")


def maybe_load_state_dict(
    module: torch.nn.Module,
    ckpt_path: Path,
    strict: bool = True,
    allow_partial: bool = False,
) -> bool:
    if not ckpt_path.exists():
        return False

    obj = torch.load(str(ckpt_path), map_location=DEVICE)
    if isinstance(obj, dict) and "state_dict" in obj:
        state_dict = obj["state_dict"]
    elif isinstance(obj, dict):
        state_dict = obj
    else:
        raise ValueError(f"Unsupported checkpoint format: {type(obj)}")

    if strict and not allow_partial:
        module.load_state_dict(state_dict)
        return True

    current = module.state_dict()
    matched = {}
    skipped = []

    for k, v in state_dict.items():
        if k in current and current[k].shape == v.shape:
            matched[k] = v
        else:
            ckpt_shape = tuple(v.shape) if hasattr(v, "shape") else None
            model_shape = tuple(current[k].shape) if k in current else None
            skipped.append((k, ckpt_shape, model_shape))

    current.update(matched)
    module.load_state_dict(current, strict=False)

    print(
        f"[INFO] Partial load for {module.__class__.__name__}: "
        f"matched={len(matched)}, skipped={len(skipped)}"
    )
    for k, ckpt_shape, model_shape in skipped:
        print(f"  [SKIP] {k}: ckpt={ckpt_shape}, model={model_shape}")

    return True


def build_env(seed: int = 42, profile: str = "baseline") -> MultiUavMecEnv:
    common_kwargs = dict(
        M=3,
        K=8,
        episode_length=20,
        area_size=100.0,
        altitude=20.0,
        neighbor_radius=50.0,
        delta_t=1.0,
        max_speed=15.0,
        min_uav_distance=3.0,
        cpu_mode="kkt",
        prop_rho=0.45,
        omega1=100.0,
        omega2=1.0,
        penalty_coeff=50.0,
        R_min=0.05,
        uav_energy_min=2600.0,
        uav_energy_max=3800.0,
        seed=seed,
    )

    if profile == "baseline":
        return MultiUavMecEnv(
            **common_kwargs,
            deadline_scale=5.0,
            task_local_cpu_min=3.0e3,
            task_local_cpu_max=8.0e3,
            uav_cpu_min=2.0e4,
            uav_cpu_max_init=5.0e4,
        )

    elif profile == "tight_deadline":
        return MultiUavMecEnv(
            **common_kwargs,
            deadline_scale=2.5,
            task_local_cpu_min=3.0e3,
            task_local_cpu_max=8.0e3,
            uav_cpu_min=2.0e4,
            uav_cpu_max_init=5.0e4,
        )

    elif profile == "tight_deadline_low_localcpu":
        return MultiUavMecEnv(
            **common_kwargs,
            deadline_scale=2.5,
            task_local_cpu_min=1.0e3,
            task_local_cpu_max=3.0e3,
            uav_cpu_min=2.0e4,
            uav_cpu_max_init=5.0e4,
        )

    else:
        raise ValueError(f"Unknown profile: {profile}")
# def build_env(seed: int = 42) -> MultiUavMecEnv:
#     return MultiUavMecEnv(
#         M=3,
#         K=8,
#         episode_length=20,
#         area_size=100.0,
#         altitude=20.0,
#         neighbor_radius=50.0,
#         delta_t=1.0,
#         max_speed=15.0,
#         min_uav_distance=3.0,
#         cpu_mode="kkt",
#         prop_rho=0.45,
#         omega1=100.0,
#         omega2=1.0,
#         penalty_coeff=50.0,
#         R_min=0.05,
#         deadline_scale=5.0,
#         uav_energy_min=2600.0,
#         uav_energy_max=3800.0,
#         seed=seed,
#     )


def build_proposed_full_stage2_policy(env: MultiUavMecEnv):
    if env.state is None:
        raise RuntimeError("env.state is None. Call env.reset() first.")

    obs = build_global_observation(env.state)
    obs_dim = int(np.asarray(obs, dtype=np.float32).shape[0])
    M = env.M
    K = env.K
    action_dim = M + M + K + K * M

    actor_net = MLPActor(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dim=256,
    ).to(DEVICE)

    actor_ok = maybe_load_state_dict(
        actor_net,
        CHECKPOINT_DIR / "proposed_full_stage2_best_actor.pth",
    )
    if not actor_ok:
        raise FileNotFoundError(
            f"Missing actor checkpoint: {CHECKPOINT_DIR / 'proposed_full_stage2_best_actor.pth'}"
        )
    actor_net.eval()

    proposed_policy = build_default_proposed_policy(
        state=env.state,
        actor_net=actor_net,
        device=str(DEVICE),
    )

    encoder_ok = maybe_load_state_dict(
        proposed_policy.encoder,
        CHECKPOINT_DIR / "proposed_full_stage2_best_encoder.pth",
    )
    fusion_ok = maybe_load_state_dict(
        proposed_policy.fusion_net,
        CHECKPOINT_DIR / "proposed_full_stage2_best_fusion.pth",
    )
    ratio_ok = maybe_load_state_dict(
        proposed_policy.ratio_head,
        CHECKPOINT_DIR / "proposed_full_stage2_best_ratio_head.pth",
        strict=False,
        allow_partial=True,
    )

    if not (encoder_ok and fusion_ok):
        raise FileNotFoundError(
            "Some proposed checkpoints are missing: "
            f"encoder_ok={encoder_ok}, fusion_ok={fusion_ok}, ratio_ok={ratio_ok}"
        )

    if not ratio_ok:
        print("[WARN] Ratio head checkpoint not found. Use new-initialized RatioHead.")

    proposed_policy.encoder.eval()
    proposed_policy.fusion_net.eval()
    proposed_policy.ratio_head.eval()

    return proposed_policy


def summarize_sched_beta(sched_beta: np.ndarray) -> Dict[int, tuple]:
    """
    Return {task_k: (access_m, exec_j)} for nonzero scheduling entries.
    """
    K, M, _ = sched_beta.shape
    out: Dict[int, tuple] = {}
    for k in range(K):
        idx = np.argwhere(sched_beta[k] > 0.5)
        if idx.shape[0] > 0:
            m = int(idx[0, 0])
            j = int(idx[0, 1])
            out[k] = (m, j)
        else:
            out[k] = (-1, -1)
    return out


def action_diff_stats(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    da = np.asarray(a, dtype=float)
    db = np.asarray(b, dtype=float)
    diff = np.abs(da - db)
    return {
        "mean_abs": float(np.mean(diff)),
        "max_abs": float(np.max(diff)),
    }


def print_action_block(name: str, action: Dict[str, np.ndarray]) -> None:
    print(f"\n[{name}]")
    print("move_dist      :", np.round(action["move_dist"], 4))
    print("move_angle     :", np.round(action["move_angle"], 4))
    print("offload_ratio  :", np.round(action["offload_ratio"], 4))
    print("sched(task->m,j):", summarize_sched_beta(np.asarray(action["sched_beta"], dtype=float)))


def rebuild_action_with_overrides(
    policy,
    state: Dict[str, Any],
    access_assoc: np.ndarray,
    base_action: Dict[str, np.ndarray],
    offload_ratio_override: np.ndarray = None,
    sched_beta_override: np.ndarray = None,
    name: str = "",
):
    move_dist = np.asarray(base_action["move_dist"], dtype=np.float32).copy()
    move_angle = np.asarray(base_action["move_angle"], dtype=np.float32).copy()

    if offload_ratio_override is None:
        offload_ratio = np.asarray(base_action["offload_ratio"], dtype=np.float32).copy()
    else:
        offload_ratio = np.asarray(offload_ratio_override, dtype=np.float32).copy()

    if sched_beta_override is None:
        sched_beta = np.asarray(base_action["sched_beta"], dtype=np.float32).copy()
    else:
        sched_beta = np.asarray(sched_beta_override, dtype=np.float32).copy()

    low_action = policy._solve_low_level(
        state=state,
        access_assoc=access_assoc,
        offload_ratio=offload_ratio,
        sched_beta=sched_beta,
    )

    action = {
        "move_dist": move_dist,
        "move_angle": move_angle,
        "offload_ratio": offload_ratio,
        "sched_beta": sched_beta,
        "bandwidth_alloc": np.asarray(low_action["bandwidth_alloc"], dtype=np.float32),
        "cpu_alloc": np.asarray(low_action["cpu_alloc"], dtype=np.float32),
    }

    if name:
        print(f"\n[ABLATION BUILD] {name}")
        print("offload_ratio :", np.round(action["offload_ratio"], 4))
        print("sched(task->m,j):", summarize_sched_beta(action["sched_beta"]))

    return action

def main():
    seed = 42
    num_slots_to_debug = 3

    set_seed(seed)

    print("=" * 120)
    print("DEBUG COMPARE ACTIONS: Greedy vs Proposed")
    print("=" * 120)
    print("PROJECT_ROOT :", PROJECT_ROOT)
    print("CHECKPOINT_DIR:", CHECKPOINT_DIR)
    print("DEVICE       :", DEVICE)

    # env_greedy = build_env(seed=seed)
    # obs_greedy = env_greedy.reset(seed=seed)

    # env_prop = build_env(seed=seed)
    # obs_prop = env_prop.reset(seed=seed)
    # env_greedy = build_env(seed=seed)
    # # 组1
    # env_greedy = build_env(seed=seed, profile="baseline")
    # obs_greedy = env_greedy.reset(seed=seed)

    # env_prop = build_env(seed=seed, profile="baseline")
    # obs_prop = env_prop.reset(seed=seed)

    # env_prop_fixed_ratio = build_env(seed=seed, profile="baseline")
    # obs_prop_fixed_ratio = env_prop_fixed_ratio.reset(seed=seed)

    # env_prop_greedy_sched = build_env(seed=seed, profile="baseline")
    # obs_prop_greedy_sched = env_prop_greedy_sched.reset(seed=seed)

    # env_prop_fixed_ratio_greedy_sched = build_env(seed=seed, profile="baseline")
    # obs_prop_fixed_ratio_greedy_sched = env_prop_fixed_ratio_greedy_sched.reset(seed=seed)
 
    # # 组2
    # env_greedy = build_env(seed=seed, profile="tight_deadline")
    # obs_greedy = env_greedy.reset(seed=seed)

    # env_prop = build_env(seed=seed, profile="tight_deadline")
    # obs_prop = env_prop.reset(seed=seed)

    # env_prop_fixed_ratio = build_env(seed=seed, profile="tight_deadline")
    # obs_prop_fixed_ratio = env_prop_fixed_ratio.reset(seed=seed)

    # env_prop_greedy_sched = build_env(seed=seed, profile="tight_deadline")
    # obs_prop_greedy_sched = env_prop_greedy_sched.reset(seed=seed)

    # env_prop_fixed_ratio_greedy_sched = build_env(seed=seed, profile="tight_deadline")
    # obs_prop_fixed_ratio_greedy_sched = env_prop_fixed_ratio_greedy_sched.reset(seed=seed)

    # 组3
    env_greedy = build_env(seed=seed, profile="tight_deadline_low_localcpu")
    obs_greedy = env_greedy.reset(seed=seed)

    env_prop = build_env(seed=seed, profile="tight_deadline_low_localcpu")
    obs_prop = env_prop.reset(seed=seed)

    env_prop_fixed_ratio = build_env(seed=seed, profile="tight_deadline_low_localcpu")
    obs_prop_fixed_ratio = env_prop_fixed_ratio.reset(seed=seed)

    env_prop_greedy_sched = build_env(seed=seed, profile="tight_deadline_low_localcpu")
    obs_prop_greedy_sched = env_prop_greedy_sched.reset(seed=seed)

    env_prop_fixed_ratio_greedy_sched = build_env(seed=seed, profile="tight_deadline_low_localcpu")
    obs_prop_fixed_ratio_greedy_sched = env_prop_fixed_ratio_greedy_sched.reset(seed=seed)



    proposed_policy = build_proposed_full_stage2_policy(env_prop)

    for slot in range(num_slots_to_debug):
        print("\n" + "=" * 120)
        print(f"SLOT {slot}")
        print("=" * 120)

        # state_g = obs_greedy["raw_state"]
        # state_p = obs_prop["raw_state"]

        # access_assoc_g = build_access_association(state_g)
        # access_assoc_p = build_access_association(state_p)

        # state_g_before = obs_greedy["raw_state"]
        # state_p_before = obs_prop["raw_state"]
        state_g_before = copy.deepcopy(obs_greedy["raw_state"])
        state_p_before = copy.deepcopy(obs_prop["raw_state"])
        state_p_fr_before = copy.deepcopy(obs_prop_fixed_ratio["raw_state"])
        state_p_gs_before = copy.deepcopy(obs_prop_greedy_sched["raw_state"])
        state_p_both_before = copy.deepcopy(obs_prop_fixed_ratio_greedy_sched["raw_state"])

        access_assoc_g = build_access_association(state_g_before)
        access_assoc_p = build_access_association(state_p_before)
        access_assoc_p_fr = build_access_association(state_p_fr_before)
        access_assoc_p_gs = build_access_association(state_p_gs_before)
        access_assoc_p_both = build_access_association(state_p_both_before)

        print("access_assoc_g:\n", access_assoc_g.astype(int))
        print("access_assoc_p:\n", access_assoc_p.astype(int))

        # greedy_action_raw = generate_greedy_high_action(
        #     state=state_g,
        #     access_assoc=access_assoc_g,
        #     seed=seed,
        # )

        # proposed_action_raw, proposed_aux = proposed_policy.act(
        #     state=state_p,
        #     access_assoc=access_assoc_p,
        #     deterministic=True,
        #     return_aux=True,
        # )

        greedy_action_raw = generate_greedy_high_action(
            state=state_g_before,
            access_assoc=access_assoc_g,
            seed=seed,
        )

        proposed_action_raw, proposed_aux = proposed_policy.act(
            state=state_p_before,
            access_assoc=access_assoc_p,
            deterministic=True,
            return_aux=True,
        )

        K = int(state_p_before["K"])

        # A1: fixed ratio only, keep proposed mobility + proposed scheduling
        proposed_fixed_ratio05 = rebuild_action_with_overrides(
            policy=proposed_policy,
            state=state_p_before,
            access_assoc=access_assoc_p,
            base_action=proposed_action_raw,
            offload_ratio_override=np.full((K,), 0.05, dtype=np.float32),
            sched_beta_override=None,
            name="Proposed + FixedRatio0.05",
        )

        # A2: greedy scheduling only, keep proposed mobility + proposed ratio
        proposed_greedy_sched = rebuild_action_with_overrides(
            policy=proposed_policy,
            state=state_p_before,
            access_assoc=access_assoc_p,
            base_action=proposed_action_raw,
            offload_ratio_override=None,
            sched_beta_override=np.asarray(greedy_action_raw["sched_beta"], dtype=np.float32),
            name="Proposed + GreedySched",
        )

        # A3: fixed ratio + greedy scheduling, only keep proposed mobility
        proposed_fixed_ratio05_greedy_sched = rebuild_action_with_overrides(
            policy=proposed_policy,
            state=state_p_before,
            access_assoc=access_assoc_p,
            base_action=proposed_action_raw,
            offload_ratio_override=np.full((K,), 0.05, dtype=np.float32),
            sched_beta_override=np.asarray(greedy_action_raw["sched_beta"], dtype=np.float32),
            name="Proposed + FixedRatio0.05 + GreedySched",
        )

        print_action_block("Greedy RAW", greedy_action_raw)
        print_action_block("Proposed RAW", proposed_action_raw)

        print("\n[RAW DIFF STATS]")
        print(
            "move_dist     :", action_diff_stats(
                greedy_action_raw["move_dist"], proposed_action_raw["move_dist"]
            )
        )
        print(
            "move_angle    :", action_diff_stats(
                greedy_action_raw["move_angle"], proposed_action_raw["move_angle"]
            )
        )
        print(
            "offload_ratio :", action_diff_stats(
                greedy_action_raw["offload_ratio"], proposed_action_raw["offload_ratio"]
            )
        )

        greedy_action_san = env_greedy._sanitize_high_action(greedy_action_raw)
        proposed_action_san = env_prop._sanitize_high_action(proposed_action_raw)

        print_action_block("Greedy SANITIZED", greedy_action_san)
        print_action_block("Proposed SANITIZED", proposed_action_san)
        print_action_block("Proposed + FixedRatio0.05", proposed_fixed_ratio05)
        print_action_block("Proposed + GreedySched", proposed_greedy_sched)
        print_action_block(
            "Proposed + FixedRatio0.05 + GreedySched",
            proposed_fixed_ratio05_greedy_sched,
        )

        print("\n[SANITIZED DIFF STATS]")
        print(
            "move_dist     :", action_diff_stats(
                greedy_action_san["move_dist"], proposed_action_san["move_dist"]
            )
        )
        print(
            "move_angle    :", action_diff_stats(
                greedy_action_san["move_angle"], proposed_action_san["move_angle"]
            )
        )
        print(
            "offload_ratio :", action_diff_stats(
                greedy_action_san["offload_ratio"], proposed_action_san["offload_ratio"]
            )
        )

        greedy_sched = summarize_sched_beta(np.asarray(greedy_action_san["sched_beta"], dtype=float))
        prop_sched = summarize_sched_beta(np.asarray(proposed_action_san["sched_beta"], dtype=float))

        same_sched_count = sum(
            1 for k in greedy_sched.keys() if greedy_sched[k] == prop_sched[k]
        )
        print(
            f"\n[SCHED COMPARISON] same task assignments = {same_sched_count}/{len(greedy_sched)}"
        )
        print("Greedy sched   :", greedy_sched)
        print("Proposed sched :", prop_sched)

        if "pooled_context" in proposed_aux:
            pooled_context = np.asarray(proposed_aux["pooled_context"], dtype=float)
            fused_feature = np.asarray(proposed_aux["fused_feature"], dtype=float)
            ratio_prior = np.asarray(proposed_aux.get("ratio_prior", []), dtype=float)
            ratio_final = np.asarray(proposed_action_raw["offload_ratio"], dtype=float)

            print("\n[PROPOSED AUX]")
            print("pooled_context shape:", pooled_context.shape)
            print("fused_feature shape :", fused_feature.shape)
            print("ratio_prior         :", np.round(ratio_prior, 4))
            print("ratio_final         :", np.round(ratio_final, 4))
            print("ratio_delta         :", np.round(ratio_final - ratio_prior, 4))

        # obs_greedy, reward_g, done_g, info_g = env_greedy.step(greedy_action_raw)
        # obs_prop, reward_p, done_p, info_p = env_prop.step(proposed_action_raw)

        # print("\n[STEP RESULT]")
        obs_greedy, reward_g, done_g, info_g = env_greedy.step(greedy_action_raw)
        obs_prop, reward_p, done_p, info_p = env_prop.step(proposed_action_raw)

        obs_prop_fixed_ratio, reward_p_fr, done_p_fr, info_p_fr = env_prop_fixed_ratio.step(
            proposed_fixed_ratio05
        )
        obs_prop_greedy_sched, reward_p_gs, done_p_gs, info_p_gs = env_prop_greedy_sched.step(
            proposed_greedy_sched
        )
        obs_prop_fixed_ratio_greedy_sched, reward_p_both, done_p_both, info_p_both = (
            env_prop_fixed_ratio_greedy_sched.step(proposed_fixed_ratio05_greedy_sched)
        )

        # state_g_after = obs_greedy["raw_state"]
        # state_p_after = obs_prop["raw_state"]
        state_g_after = copy.deepcopy(obs_greedy["raw_state"])
        state_p_after = copy.deepcopy(obs_prop["raw_state"])

        _print_mobility_debug(
            "GREEDY",
            state_g_before,
            state_g_after,
            greedy_action_raw,
        )
        _print_channel_distance_debug(
            "GREEDY",
            state_g_before,
            state_g_after,
            access_assoc_g,
        )

        _print_mobility_debug(
            "PROPOSED",
            state_p_before,
            state_p_after,
            proposed_action_raw,
        )
        _print_channel_distance_debug(
            "PROPOSED",
            state_p_before,
            state_p_after,
            access_assoc_p,
        )

        print("\n[STEP RESULT]")
        print(
            f"Greedy                         reward={reward_g:.4f}, "
            f"feasible={bool(info_g['report'].get('ok', False))}, "
            f"delay={float(info_g['metrics'].get('delay_sys', 0.0)):.4f}, "
            f"energy={float(info_g['metrics'].get('energy_sys', 0.0)):.4f}"
        )
        print(
            f"Proposed                       reward={reward_p:.4f}, "
            f"feasible={bool(info_p['report'].get('ok', False))}, "
            f"delay={float(info_p['metrics'].get('delay_sys', 0.0)):.4f}, "
            f"energy={float(info_p['metrics'].get('energy_sys', 0.0)):.4f}"
        )
        print(
            f"Proposed + FixedRatio0.05      reward={reward_p_fr:.4f}, "
            f"feasible={bool(info_p_fr['report'].get('ok', False))}, "
            f"delay={float(info_p_fr['metrics'].get('delay_sys', 0.0)):.4f}, "
            f"energy={float(info_p_fr['metrics'].get('energy_sys', 0.0)):.4f}"
        )
        print(
            f"Proposed + GreedySched         reward={reward_p_gs:.4f}, "
            f"feasible={bool(info_p_gs['report'].get('ok', False))}, "
            f"delay={float(info_p_gs['metrics'].get('delay_sys', 0.0)):.4f}, "
            f"energy={float(info_p_gs['metrics'].get('energy_sys', 0.0)):.4f}"
        )
        print(
            f"Proposed + FixedRatio+GreedySched reward={reward_p_both:.4f}, "
            f"feasible={bool(info_p_both['report'].get('ok', False))}, "
            f"delay={float(info_p_both['metrics'].get('delay_sys', 0.0)):.4f}, "
            f"energy={float(info_p_both['metrics'].get('energy_sys', 0.0)):.4f}"
        )

        # print("\n[STEP RESULT]")

        # print(
        #     f"Greedy   reward={reward_g:.4f}, feasible={bool(info_g['report'].get('ok', False))}, "
        #     f"delay={float(info_g['metrics'].get('delay_sys', 0.0)):.4f}, "
        #     f"energy={float(info_g['metrics'].get('energy_sys', 0.0)):.4f}"
        # )
        # print(
        #     f"Proposed reward={reward_p:.4f}, feasible={bool(info_p['report'].get('ok', False))}, "
        #     f"delay={float(info_p['metrics'].get('delay_sys', 0.0)):.4f}, "
        #     f"energy={float(info_p['metrics'].get('energy_sys', 0.0)):.4f}"
        # )

        # print("\n[ENERGY]")
        # print("Greedy before:", np.round(info_g["uav_energy_before"], 4))
        # print("Greedy after :", np.round(info_g["uav_energy_after"], 4))
        # print("Prop before  :", np.round(info_p["uav_energy_before"], 4))
        # print("Prop after   :", np.round(info_p["uav_energy_after"], 4))

        if done_g or done_p:
            break

    print("\n" + "=" * 120)
    print("DEBUG FINISHED")
    print("=" * 120)


if __name__ == "__main__":
    main()


# from __future__ import annotations

# import random
# from pathlib import Path
# from typing import Dict, Any

# import numpy as np
# import torch

# from env.mec_env import MultiUavMecEnv
# from env.association import build_access_association

# from policy.greedy_policy import generate_greedy_high_action
# from policy.proposed_policy import build_default_proposed_policy

# from model.mlp_actor import MLPActor
# from model.proposed_obs_builder import build_global_observation


# PROJECT_ROOT = Path(__file__).resolve().parent.parent
# CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# EPS = 1e-8


# def set_seed(seed: int) -> None:
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(seed)


# def maybe_load_state_dict(module: torch.nn.Module, ckpt_path: Path) -> bool:
#     if not ckpt_path.exists():
#         return False

#     obj = torch.load(str(ckpt_path), map_location=DEVICE)
#     if isinstance(obj, dict) and "state_dict" in obj:
#         module.load_state_dict(obj["state_dict"])
#     elif isinstance(obj, dict):
#         module.load_state_dict(obj)
#     else:
#         raise ValueError(f"Unsupported checkpoint format: {type(obj)}")
#     return True


# def build_env(seed: int = 42) -> MultiUavMecEnv:
#     return MultiUavMecEnv(
#         M=3,
#         K=8,
#         episode_length=20,
#         area_size=100.0,
#         altitude=20.0,
#         neighbor_radius=50.0,
#         delta_t=1.0,
#         max_speed=15.0,
#         min_uav_distance=3.0,
#         cpu_mode="kkt",
#         prop_rho=0.45,
#         omega1=100.0,
#         omega2=1.0,
#         penalty_coeff=50.0,
#         R_min=0.05,
#         deadline_scale=5.0,
#         uav_energy_min=2600.0,
#         uav_energy_max=3800.0,
#         seed=seed,
#     )


# def build_proposed_full_stage2_policy(env: MultiUavMecEnv):
#     if env.state is None:
#         raise RuntimeError("env.state is None. Call env.reset() first.")

#     obs = build_global_observation(env.state)
#     obs_dim = int(np.asarray(obs, dtype=np.float32).shape[0])
#     M = env.M
#     K = env.K
#     action_dim = M + M + K + K * M

#     actor_net = MLPActor(
#         obs_dim=obs_dim,
#         action_dim=action_dim,
#         hidden_dim=256,
#     ).to(DEVICE)

#     actor_ok = maybe_load_state_dict(
#         actor_net, CHECKPOINT_DIR / "proposed_full_stage2_best_actor.pth"
#     )
#     if not actor_ok:
#         raise FileNotFoundError(
#             f"Missing actor checkpoint: {CHECKPOINT_DIR / 'proposed_full_stage2_best_actor.pth'}"
#         )
#     actor_net.eval()

#     proposed_policy = build_default_proposed_policy(
#         state=env.state,
#         actor_net=actor_net,
#         device=str(DEVICE),
#     )

#     encoder_ok = maybe_load_state_dict(
#         proposed_policy.encoder,
#         CHECKPOINT_DIR / "proposed_full_stage2_best_encoder.pth",
#     )
#     fusion_ok = maybe_load_state_dict(
#         proposed_policy.fusion_net,
#         CHECKPOINT_DIR / "proposed_full_stage2_best_fusion.pth",
#     )
#     ratio_ok = maybe_load_state_dict(
#         proposed_policy.ratio_head,
#         CHECKPOINT_DIR / "proposed_full_stage2_best_ratio_head.pth",
#     )

#     if not (encoder_ok and fusion_ok and ratio_ok):
#         raise FileNotFoundError(
#             "Some proposed checkpoints are missing: "
#             f"encoder_ok={encoder_ok}, fusion_ok={fusion_ok}, ratio_ok={ratio_ok}"
#         )

#     proposed_policy.encoder.eval()
#     proposed_policy.fusion_net.eval()
#     proposed_policy.ratio_head.eval()

#     return proposed_policy


# def summarize_sched_beta(sched_beta: np.ndarray) -> Dict[int, tuple]:
#     """
#     Return {task_k: (access_m, exec_j)} for nonzero scheduling entries.
#     """
#     K, M, _ = sched_beta.shape
#     out: Dict[int, tuple] = {}
#     for k in range(K):
#         idx = np.argwhere(sched_beta[k] > 0.5)
#         if idx.shape[0] > 0:
#             m = int(idx[0, 0])
#             j = int(idx[0, 1])
#             out[k] = (m, j)
#         else:
#             out[k] = (-1, -1)
#     return out


# def action_diff_stats(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
#     da = np.asarray(a, dtype=float)
#     db = np.asarray(b, dtype=float)
#     diff = np.abs(da - db)
#     return {
#         "mean_abs": float(np.mean(diff)),
#         "max_abs": float(np.max(diff)),
#     }


# def print_action_block(name: str, action: Dict[str, np.ndarray]) -> None:
#     print(f"\n[{name}]")
#     print("move_dist      :", np.round(action["move_dist"], 4))
#     print("move_angle     :", np.round(action["move_angle"], 4))
#     print("offload_ratio  :", np.round(action["offload_ratio"], 4))
#     print("sched(task->m,j):", summarize_sched_beta(np.asarray(action["sched_beta"], dtype=float)))


# def main():
#     seed = 42
#     num_slots_to_debug = 3

#     set_seed(seed)

#     print("=" * 120)
#     print("DEBUG COMPARE ACTIONS: Greedy vs Proposed")
#     print("=" * 120)
#     print("PROJECT_ROOT :", PROJECT_ROOT)
#     print("CHECKPOINT_DIR:", CHECKPOINT_DIR)
#     print("DEVICE       :", DEVICE)

#     env_greedy = build_env(seed=seed)
#     obs_greedy = env_greedy.reset(seed=seed)

#     env_prop = build_env(seed=seed)
#     obs_prop = env_prop.reset(seed=seed)

#     proposed_policy = build_proposed_full_stage2_policy(env_prop)

#     for slot in range(num_slots_to_debug):
#         print("\n" + "=" * 120)
#         print(f"SLOT {slot}")
#         print("=" * 120)

#         state_g = obs_greedy["raw_state"]
#         state_p = obs_prop["raw_state"]

#         access_assoc_g = build_access_association(state_g)
#         access_assoc_p = build_access_association(state_p)

#         print("access_assoc_g:\n", access_assoc_g.astype(int))
#         print("access_assoc_p:\n", access_assoc_p.astype(int))

#         greedy_action_raw = generate_greedy_high_action(
#             state=state_g,
#             access_assoc=access_assoc_g,
#             seed=seed,
#         )

#         proposed_action_raw, proposed_aux = proposed_policy.act(
#             state=state_p,
#             access_assoc=access_assoc_p,
#             deterministic=True,
#             return_aux=True,
#         )

#         print_action_block("Greedy RAW", greedy_action_raw)
#         print_action_block("Proposed RAW", proposed_action_raw)

#         print("\n[RAW DIFF STATS]")
#         print(
#             "move_dist     :", action_diff_stats(
#                 greedy_action_raw["move_dist"], proposed_action_raw["move_dist"]
#             )
#         )
#         print(
#             "move_angle    :", action_diff_stats(
#                 greedy_action_raw["move_angle"], proposed_action_raw["move_angle"]
#             )
#         )
#         print(
#             "offload_ratio :", action_diff_stats(
#                 greedy_action_raw["offload_ratio"], proposed_action_raw["offload_ratio"]
#             )
#         )

#         greedy_action_san = env_greedy._sanitize_high_action(greedy_action_raw)
#         proposed_action_san = env_prop._sanitize_high_action(proposed_action_raw)

#         print_action_block("Greedy SANITIZED", greedy_action_san)
#         print_action_block("Proposed SANITIZED", proposed_action_san)

#         print("\n[SANITIZED DIFF STATS]")
#         print(
#             "move_dist     :", action_diff_stats(
#                 greedy_action_san["move_dist"], proposed_action_san["move_dist"]
#             )
#         )
#         print(
#             "move_angle    :", action_diff_stats(
#                 greedy_action_san["move_angle"], proposed_action_san["move_angle"]
#             )
#         )
#         print(
#             "offload_ratio :", action_diff_stats(
#                 greedy_action_san["offload_ratio"], proposed_action_san["offload_ratio"]
#             )
#         )

#         greedy_sched = summarize_sched_beta(np.asarray(greedy_action_san["sched_beta"], dtype=float))
#         prop_sched = summarize_sched_beta(np.asarray(proposed_action_san["sched_beta"], dtype=float))

#         same_sched_count = sum(
#             1 for k in greedy_sched.keys() if greedy_sched[k] == prop_sched[k]
#         )
#         print(
#             f"\n[SCHED COMPARISON] same task assignments = {same_sched_count}/{len(greedy_sched)}"
#         )
#         print("Greedy sched   :", greedy_sched)
#         print("Proposed sched :", prop_sched)

#         if "pooled_context" in proposed_aux:
#             pooled_context = np.asarray(proposed_aux["pooled_context"], dtype=float)
#             fused_feature = np.asarray(proposed_aux["fused_feature"], dtype=float)
#             print("\n[PROPOSED AUX]")
#             print("pooled_context shape:", pooled_context.shape)
#             print("fused_feature shape :", fused_feature.shape)

#         obs_greedy, reward_g, done_g, info_g = env_greedy.step(greedy_action_raw)
#         obs_prop, reward_p, done_p, info_p = env_prop.step(proposed_action_raw)

#         print("\n[STEP RESULT]")
#         print(
#             f"Greedy   reward={reward_g:.4f}, feasible={bool(info_g['report'].get('ok', False))}, "
#             f"delay={float(info_g['metrics'].get('delay_sys', 0.0)):.4f}, "
#             f"energy={float(info_g['metrics'].get('energy_sys', 0.0)):.4f}"
#         )
#         print(
#             f"Proposed reward={reward_p:.4f}, feasible={bool(info_p['report'].get('ok', False))}, "
#             f"delay={float(info_p['metrics'].get('delay_sys', 0.0)):.4f}, "
#             f"energy={float(info_p['metrics'].get('energy_sys', 0.0)):.4f}"
#         )

#         print("\n[ENERGY]")
#         print("Greedy before:", np.round(info_g["uav_energy_before"], 4))
#         print("Greedy after :", np.round(info_g["uav_energy_after"], 4))
#         print("Prop before  :", np.round(info_p["uav_energy_before"], 4))
#         print("Prop after   :", np.round(info_p["uav_energy_after"], 4))

#         if done_g or done_p:
#             break

#     print("\n" + "=" * 120)
#     print("DEBUG FINISHED")
#     print("=" * 120)


# if __name__ == "__main__":
#     main()