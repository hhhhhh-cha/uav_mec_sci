import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

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
    get_actor_raw_action_dim,
    get_full_action_dim,
    flatten_full_action,
    soft_update,
    soft_update_policy,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class RetrainV2Config:
    seed: int = 42
    device: str = DEVICE

    # hard environment for fine-tune
    M: int = 3
    K: int = 8
    episode_length: int = 5
    cpu_mode: str = "kkt"
    omega1: float = 100.0
    omega2: float = 1.0
    penalty_coeff: float = 50.0

    # harsher scenario
    deadline_scale: float = 2.5
    task_local_cpu_min: float = 2.0e3
    task_local_cpu_max: float = 5.0e3
    uav_cpu_min: float = 2.0e4
    uav_cpu_max_init: float = 5.0e4
    uav_energy_min: float = 2600.0
    uav_energy_max: float = 3800.0

    # model
    actor_hidden_dim: int = 256
    critic_hidden_dim: int = 256
    embed_dim: int = 128
    num_heads: int = 4
    ff_hidden_dim: int = 256
    num_layers: int = 2

    # hard fine-tune only
    num_episodes: int = 40
    batch_size: int = 32
    buffer_capacity: int = 30000
    actor_lr: float = 2e-4
    critic_lr: float = 5e-4
    gamma: float = 0.99
    tau: float = 0.005

    actor_policy_coef: float = 1.0
    actor_move_sched_bc_coef: float = 0.1
    ratio_bc_coef: float = 0.1
    actor_l2_coef: float = 1e-5
    eval_every: int = 5

    # load from old stable stage2
    init_actor_ckpt: str = "proposed_full_stage2_best_actor.pth"
    init_encoder_ckpt: str = "proposed_full_stage2_best_encoder.pth"
    init_fusion_ckpt: str = "proposed_full_stage2_best_fusion.pth"
    init_ratio_ckpt: str = "proposed_full_stage2_best_ratio_head.pth"
    init_critic_ckpt: str = "proposed_full_stage2_critic.pth"

    # save as hard fine-tuned model
    save_actor_ckpt: str = "proposed_full_stage2_hard_best_actor.pth"
    save_encoder_ckpt: str = "proposed_full_stage2_hard_best_encoder.pth"
    save_fusion_ckpt: str = "proposed_full_stage2_hard_best_fusion.pth"
    save_ratio_ckpt: str = "proposed_full_stage2_hard_best_ratio_head.pth"
    save_critic_ckpt: str = "proposed_full_stage2_hard_critic.pth"


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_env_v2(cfg: RetrainV2Config, seed: int) -> MultiUavMecEnv:
    return MultiUavMecEnv(
        M=cfg.M,
        K=cfg.K,
        episode_length=cfg.episode_length,
        cpu_mode=cfg.cpu_mode,
        omega1=cfg.omega1,
        omega2=cfg.omega2,
        penalty_coeff=cfg.penalty_coeff,
        deadline_scale=cfg.deadline_scale,
        task_local_cpu_min=cfg.task_local_cpu_min,
        task_local_cpu_max=cfg.task_local_cpu_max,
        uav_cpu_min=cfg.uav_cpu_min,
        uav_cpu_max_init=cfg.uav_cpu_max_init,
        uav_energy_min=cfg.uav_energy_min,
        uav_energy_max=cfg.uav_energy_max,
        seed=seed,
    )


def save_policy_modules(
    policy: ProposedPolicy,
    actor_path: Path,
    encoder_path: Path,
    fusion_path: Path,
    ratio_path: Path,
) -> None:
    actor_path.parent.mkdir(parents=True, exist_ok=True)
    if policy.actor_net is not None:
        torch.save(policy.actor_net.state_dict(), actor_path)
    if policy.encoder is not None:
        torch.save(policy.encoder.state_dict(), encoder_path)
    if policy.fusion_net is not None:
        torch.save(policy.fusion_net.state_dict(), fusion_path)
    if policy.ratio_head is not None:
        torch.save(policy.ratio_head.state_dict(), ratio_path)


def load_if_exists(module: torch.nn.Module, path: Path, name: str) -> bool:
    if path.exists():
        module.load_state_dict(torch.load(path, map_location="cpu"))
        print(f"Loaded {name} from: {path}")
        return True
    print(f"WARNING: {name} checkpoint not found: {path}")
    return False


def build_policy_and_critic(cfg: RetrainV2Config, seed: int):
    env = build_env_v2(cfg, seed=seed)
    obs = env.reset(seed=seed)
    state = obs["raw_state"]

    obs_dim = get_observation_dim(state)
    actor_raw_action_dim = get_actor_raw_action_dim(state)
    critic_action_dim = get_full_action_dim(state)

    actor = MLPActor(
        obs_dim=obs_dim,
        action_dim=actor_raw_action_dim,
        hidden_dim=cfg.actor_hidden_dim,
    ).to(cfg.device)

    policy = build_default_proposed_policy(
        state=state,
        actor_net=actor,
        device=cfg.device,
        embed_dim=cfg.embed_dim,
        num_heads=cfg.num_heads,
        ff_hidden_dim=cfg.ff_hidden_dim,
        num_layers=cfg.num_layers,
    )

    critic = MLPCritic(
        obs_dim=obs_dim,
        action_dim=critic_action_dim,
        hidden_dim=cfg.critic_hidden_dim,
    ).to(cfg.device)

    return env, state, obs_dim, actor_raw_action_dim, critic_action_dim, policy, critic


def build_move_sched_mask(actor_raw_action_dim: int, state: Dict[str, Any], device: str) -> torch.Tensor:
    M = int(state["M"])
    K = int(state["K"])
    move_sched_mask = np.zeros((actor_raw_action_dim,), dtype=np.float32)
    move_sched_mask[: M + M] = 1.0
    move_sched_mask[M + M + K:] = 1.0
    return torch.tensor(move_sched_mask, dtype=torch.float32, device=device).unsqueeze(0)


def build_surrogate_full_action_tensor(
    state: Dict[str, Any],
    access_assoc: np.ndarray,
    raw_action_pred: torch.Tensor,
    ratio_pred: torch.Tensor,
    bandwidth_alloc_np: np.ndarray,
    cpu_alloc_np: np.ndarray,
) -> torch.Tensor:
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
    _offload_unused = raw_action_pred[:, p:p + K]
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

    return torch.cat(
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


def forward_ratio_branch_aligned(
    policy: ProposedPolicy,
    state: Dict[str, Any],
    access_assoc: np.ndarray,
) -> torch.Tensor:
    """
    Aligned with deployed policy semantics:
      encoder -> fusion -> ratio_head(prior_ratio=..., hard_min=...)
    """
    if policy.encoder is None or policy.fusion_net is None or policy.ratio_head is None:
        raise RuntimeError("policy encoder/fusion_net/ratio_head is None.")

    K = int(state["K"])
    Nc = int(policy.max_candidates)

    token_batches = []
    mask_batches = []
    task_feat_batches = []
    uav_feat_batches = []

    for k in range(K):
        access_m = int(np.argmax(access_assoc[:, k]))
        tokens, mask = policy._build_candidate_tokens_for_task(
            state=state,
            task_idx=k,
            access_m=access_m,
            max_candidates=Nc,
        )
        task_feat = policy._normalize_task_feature(state, k)
        uav_feat = policy._normalize_uav_feature(state, access_m)

        token_batches.append(tokens)
        mask_batches.append(mask)
        task_feat_batches.append(task_feat)
        uav_feat_batches.append(uav_feat)

    token_tensor = torch.tensor(np.asarray(token_batches), dtype=torch.float32, device=policy.device)
    mask_tensor = torch.tensor(np.asarray(mask_batches), dtype=torch.float32, device=policy.device)
    task_feat_tensor = torch.tensor(np.asarray(task_feat_batches), dtype=torch.float32, device=policy.device)
    uav_feat_tensor = torch.tensor(np.asarray(uav_feat_batches), dtype=torch.float32, device=policy.device)

    ratio_prior_np = policy._compute_ratio_prior(state, access_assoc)
    prior_tensor = torch.tensor(
        ratio_prior_np,
        dtype=torch.float32,
        device=policy.device,
    ).unsqueeze(-1)

    encoded_tokens, pooled_context = policy.encoder(token_tensor, mask_tensor)
    fused_feature = policy.fusion_net(
        topo_context=pooled_context,
        task_feat=task_feat_tensor,
        uav_feat=uav_feat_tensor,
    )

    ratio_pred = policy.ratio_head(
        fused_feature,
        prior_ratio=prior_tensor,
        temperature=1.0,
        hard_min=policy.ratio_floor,
    ).squeeze(-1)

    ratio_pred = torch.clamp(
        ratio_pred,
        min=policy.ratio_floor,
        max=policy.ratio_ceiling,
    )
    return ratio_pred


def build_stage1_ratio_teacher(
    policy: ProposedPolicy,
    state: Dict[str, Any],
    access_assoc: np.ndarray,
    base_ratio: np.ndarray,
) -> np.ndarray:
    """
    Build a stronger ratio teacher than the placeholder's raw offload ratio.

    Idea:
    1) start from placeholder ratio
    2) compare with policy prior
    3) if local execution is too tight w.r.t. deadline, force a higher teacher ratio
    """
    task_size = np.asarray(state["task_size"], dtype=np.float32)
    task_cycles = np.asarray(state["task_cycles"], dtype=np.float32)
    task_deadline = np.asarray(state["task_deadline"], dtype=np.float32)
    task_local_cpu = np.asarray(state["task_local_cpu"], dtype=np.float32)

    ratio_prior = policy._compute_ratio_prior(state, access_assoc)
    teacher = np.maximum(np.asarray(base_ratio, dtype=np.float32), ratio_prior).copy()

    K = int(state["K"])
    for k in range(K):
        Dk = float(task_size[k])
        Ck = float(task_cycles[k])
        tau_k = max(float(task_deadline[k]), 1e-8)
        f_loc = max(float(task_local_cpu[k]), 1e-8)

        local_only_delay = Dk * Ck / f_loc

        # if local execution is much tighter than the deadline, push ratio teacher upward
        if local_only_delay > 1.05 * tau_k:
            lam_min = 1.0 - tau_k * f_loc / max(Dk * Ck, 1e-8)
            teacher[k] = max(teacher[k], lam_min + 0.10)

        # mild floor for training diversity
        teacher[k] = max(teacher[k], 0.08)

    teacher = np.clip(teacher, policy.ratio_floor, policy.ratio_ceiling).astype(np.float32)
    return teacher


def fine_tune_hard_from_stage2(cfg: RetrainV2Config) -> None:
    print("=" * 80)
    print("Hard-scenario fine-tuning from old stage2 checkpoints")
    print("=" * 80)
    set_seed(cfg.seed)

    env, state, obs_dim, actor_raw_action_dim, critic_action_dim, policy, critic = build_policy_and_critic(
        cfg,
        seed=cfg.seed,
    )

    print("device:", cfg.device)
    print("obs_dim:", obs_dim)
    print("actor_raw_action_dim:", actor_raw_action_dim)
    print("critic_action_dim:", critic_action_dim)

    load_if_exists(policy.actor_net, CHECKPOINT_DIR / cfg.init_actor_ckpt, "init actor")
    load_if_exists(policy.encoder, CHECKPOINT_DIR / cfg.init_encoder_ckpt, "init encoder")
    load_if_exists(policy.fusion_net, CHECKPOINT_DIR / cfg.init_fusion_ckpt, "init fusion")
    # load_if_exists(policy.ratio_head, CHECKPOINT_DIR / cfg.init_ratio_ckpt, "init ratio_head")
    load_if_exists(critic, CHECKPOINT_DIR / cfg.init_critic_ckpt, "init critic")

    target_policy = copy.deepcopy(policy)
    target_critic = copy.deepcopy(critic).to(cfg.device)

    params = []
    if policy.actor_net is not None:
        params += list(policy.actor_net.parameters())
    if policy.encoder is not None:
        params += list(policy.encoder.parameters())
    if policy.fusion_net is not None:
        params += list(policy.fusion_net.parameters())
    if policy.ratio_head is not None:
        params += list(policy.ratio_head.parameters())

    actor_opt = optim.Adam(params, lr=cfg.actor_lr)
    critic_opt = optim.Adam(critic.parameters(), lr=cfg.critic_lr)
    mse_loss = nn.MSELoss()

    buffer = FullReplayBuffer(
        obs_dim=obs_dim,
        action_dim=critic_action_dim,
        capacity=cfg.buffer_capacity,
    )
    move_sched_mask_t = build_move_sched_mask(actor_raw_action_dim, state, cfg.device)

    best_policy_state = copy.deepcopy(policy)
    best_eval_reward = -float("inf")

    for ep in range(cfg.num_episodes):
        obs = env.reset(seed=cfg.seed + ep)
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
            teacher_sched_beta = np.asarray(
                teacher_action["sched_beta"],
                dtype=np.float32,
            )

            teacher_offload_ratio_refined, _ = policy._refine_high_level_action(
                state=state,
                access_assoc=access_assoc,
                offload_ratio=teacher_offload_ratio,
                sched_beta=teacher_sched_beta,
            )

            action = policy.act(
                state=state,
                access_assoc=access_assoc,
                deterministic=True,
                return_aux=False,
            )
            flat_action = flatten_full_action(action)

            next_obs, reward, done, _info = env.step(action)
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

            if len(buffer) >= cfg.batch_size:
                batch = buffer.sample(cfg.batch_size)
                obs_b = torch.tensor(batch["obs"], dtype=torch.float32, device=cfg.device)
                action_b = torch.tensor(batch["action"], dtype=torch.float32, device=cfg.device)
                reward_b = torch.tensor(batch["reward"], dtype=torch.float32, device=cfg.device)
                next_obs_b = torch.tensor(batch["next_obs"], dtype=torch.float32, device=cfg.device)
                next_action_b = torch.tensor(batch["next_action"], dtype=torch.float32, device=cfg.device)
                done_b = torch.tensor(batch["done"], dtype=torch.float32, device=cfg.device)

                with torch.no_grad():
                    target_q = target_critic(next_obs_b, next_action_b)
                    y = reward_b + cfg.gamma * (1.0 - done_b) * target_q

                q_val = critic(obs_b, action_b)
                critic_loss = mse_loss(q_val, y)

                critic_opt.zero_grad()
                critic_loss.backward()
                critic_opt.step()

                episode_critic_loss += float(critic_loss.item())
                update_count += 1

            obs_tensor = torch.tensor(obs_vec, dtype=torch.float32, device=cfg.device).unsqueeze(0)
            teacher_raw_target_t = torch.tensor(teacher_raw_target, dtype=torch.float32, device=cfg.device).unsqueeze(0)
            teacher_offload_ratio_refined_t = torch.tensor(
                teacher_offload_ratio_refined,
                dtype=torch.float32,
                device=cfg.device,
            )

            raw_pred = policy.actor_net(obs_tensor)
            ratio_pred = forward_ratio_branch_aligned(
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

            ratio_bc_loss = mse_loss(ratio_pred, teacher_offload_ratio_refined_t)
            actor_l2_loss = (raw_pred ** 2).mean()

            total_actor_loss = (
                cfg.actor_policy_coef * actor_policy_loss
                + cfg.actor_move_sched_bc_coef * actor_move_sched_bc_loss
                + cfg.ratio_bc_coef * ratio_bc_loss
                + cfg.actor_l2_coef * actor_l2_loss
            )

            actor_opt.zero_grad()
            total_actor_loss.backward()
            actor_opt.step()

            soft_update_policy(target_policy, policy, tau=cfg.tau)
            soft_update(target_critic, critic, tau=cfg.tau)

            if step_count < 2:
                with torch.no_grad():
                    print("[HARD FT DEBUG]")
                    print("  ratio_pred[:5] =", ratio_pred.detach().cpu().numpy()[:5])
                    print("  teacher_refined[:5] =", teacher_offload_ratio_refined[:5])

            obs = next_obs
            episode_reward += reward
            episode_actor_policy_loss += float(actor_policy_loss.item())
            episode_move_sched_bc_loss += float(actor_move_sched_bc_loss.item())
            episode_ratio_bc_loss += float(ratio_bc_loss.item())
            step_count += 1

        print(f"\n[Hard_FT] Episode {ep}")
        print("episode_reward:", episode_reward)
        print("avg_actor_policy_loss:", episode_actor_policy_loss / max(step_count, 1))
        print("avg_move_sched_bc_loss:", episode_move_sched_bc_loss / max(step_count, 1))
        print("avg_ratio_bc_loss:", episode_ratio_bc_loss / max(step_count, 1))
        print("avg_critic_loss:", episode_critic_loss / max(update_count, 1) if update_count else "N/A")
        print("steps:", step_count)
        print("buffer_size:", len(buffer))

        if (ep + 1) % cfg.eval_every == 0 or ep == 0:
            eval_env = build_env_v2(cfg, seed=999)
            eval_result = evaluate_full_policy_rollout(
                env=eval_env,
                policy=policy,
                seed=999,
            )

            print("\n==============================")
            print(f"Hard fine-tune Eval @ episode {ep}")
            print("==============================")
            print("episode_reward:", eval_result["episode_reward"])
            print("avg_delay:", eval_result["avg_delay"])
            print("avg_energy:", eval_result["avg_energy"])
            print("avg_deadline_violation:", eval_result["avg_deadline_violation"])
            print("feasible_ratio:", eval_result["feasible_ratio"])
            print("num_steps:", eval_result["num_steps"])

            if eval_result["episode_reward"] > best_eval_reward:
                best_eval_reward = eval_result["episode_reward"]
                best_policy_state = copy.deepcopy(policy)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    save_policy_modules(
        best_policy_state,
        CHECKPOINT_DIR / cfg.save_actor_ckpt,
        CHECKPOINT_DIR / cfg.save_encoder_ckpt,
        CHECKPOINT_DIR / cfg.save_fusion_ckpt,
        CHECKPOINT_DIR / cfg.save_ratio_ckpt,
    )
    torch.save(critic.state_dict(), CHECKPOINT_DIR / cfg.save_critic_ckpt)

    print("\nHard fine-tuning finished.")
    print("Saved to:")
    print(CHECKPOINT_DIR / cfg.save_actor_ckpt)
    print(CHECKPOINT_DIR / cfg.save_encoder_ckpt)
    print(CHECKPOINT_DIR / cfg.save_fusion_ckpt)
    print(CHECKPOINT_DIR / cfg.save_ratio_ckpt)
    print(CHECKPOINT_DIR / cfg.save_critic_ckpt)


def main() -> None:
    cfg = RetrainV2Config()
    print("Using config:")
    for k, v in cfg.__dict__.items():
        print(f"  {k}: {v}")

    fine_tune_hard_from_stage2(cfg)
    print("\nHard fine-tuning completed successfully.")


if __name__ == "__main__":
    main()