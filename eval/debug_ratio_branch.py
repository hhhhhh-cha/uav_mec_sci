from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch

from env.mec_env import MultiUavMecEnv
from env.association import build_access_association
from model.mlp_actor import MLPActor
from model.proposed_obs_builder import build_global_observation
from policy.proposed_policy import build_default_proposed_policy

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def build_env(seed: int = 42) -> MultiUavMecEnv:
    return MultiUavMecEnv(
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
        deadline_scale=5.0,
        uav_energy_min=2600.0,
        uav_energy_max=3800.0,
        seed=seed,
    )


def build_policy(env: MultiUavMecEnv):
    if env.state is None:
        raise RuntimeError("env.state is None. Call env.reset() first.")

    obs = build_global_observation(env.state)
    obs_dim = int(np.asarray(obs, dtype=np.float32).shape[0])
    M = env.M
    K = env.K
    action_dim = M + M + K + K * M

    actor_net = MLPActor(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=256).to(DEVICE)
    maybe_load_state_dict(
        actor_net,
        CHECKPOINT_DIR / "proposed_full_stage2_best_actor.pth",
    )
    actor_net.eval()

    policy = build_default_proposed_policy(
        state=env.state,
        actor_net=actor_net,
        device=str(DEVICE),
    )

    maybe_load_state_dict(
        policy.encoder,
        CHECKPOINT_DIR / "proposed_full_stage2_best_encoder.pth",
    )
    maybe_load_state_dict(
        policy.fusion_net,
        CHECKPOINT_DIR / "proposed_full_stage2_best_fusion.pth",
    )

    ratio_ok = maybe_load_state_dict(
        policy.ratio_head,
        CHECKPOINT_DIR / "proposed_full_stage2_best_ratio_head.pth",
        strict=False,
        allow_partial=True,
    )
    if not ratio_ok:
        print("[WARN] Ratio head checkpoint not found. Use new-initialized RatioHead.")

    policy.encoder.eval()
    policy.fusion_net.eval()
    policy.ratio_head.eval()
    return policy


def main():
    seed = 42
    set_seed(seed)

    env = build_env(seed=seed)
    obs = env.reset(seed=seed)
    state = obs["raw_state"]
    access_assoc = build_access_association(state)
    policy = build_policy(env)

    action, aux = policy.act(
        state=state,
        access_assoc=access_assoc,
        deterministic=True,
        return_aux=True,
    )

    ratio_prior = np.asarray(aux.get("ratio_prior", []), dtype=float)
    ratio_final = np.asarray(action["offload_ratio"], dtype=float)

    print("=" * 80)
    print("RATIO BRANCH DEBUG")
    print("=" * 80)
    print("ratio_prior:", np.round(ratio_prior, 4))
    print("ratio_final:", np.round(ratio_final, 4))
    print("delta      :", np.round(ratio_final - ratio_prior, 4))
    print(
        "min/final mean/max:",
        float(np.min(ratio_final)),
        float(np.mean(ratio_final)),
        float(np.max(ratio_final)),
    )


if __name__ == "__main__":
    main()