# # 它要做的事是：
# # 用 ProposedPolicy 替代旧的简化策略接口
# # actor 训练目标先保留简单版
# # encoder / fusion / ratio head 也纳入网络参数
# # 先把“完整 Proposed 方法训练骨架”跑通

# import copy
# from collections import deque
# from typing import Any, Dict, List, Optional, Tuple

# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim

# from env.mec_env import MultiUavMecEnv
# from env.association import build_access_association
# from model.mlp_actor import MLPActor
# from model.mlp_critic import MLPCritic
# from model.proposed_obs_builder import build_global_observation, get_observation_dim
# from policy.proposed_placeholder_policy import generate_proposed_placeholder_action
# from policy.proposed_policy import build_default_proposed_policy, ProposedPolicy

import copy
import os
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

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


EPS = 1e-8


# =========================================================
# Helpers: dimensions / flatten / soft update
# =========================================================
def get_actor_raw_action_dim(state: Dict[str, Any]) -> int:
    """
    Raw actor output layout:
        [ move_dist(M), move_angle(M), offload_ratio(K), sched_score(K*M) ]
    """
    M = int(state["M"])
    K = int(state["K"])
    return M + M + K + K * M


def get_full_action_dim(state: Dict[str, Any]) -> int:
    """
    Full action layout used by critic:
        [ move_dist(M),
          move_angle(M),
          offload_ratio(K),
          sched_beta(K*M*M),
          bandwidth_alloc(M*K),
          cpu_alloc(M*K) ]
    """
    M = int(state["M"])
    K = int(state["K"])
    return M + M + K + K * M * M + M * K + M * K


def flatten_full_action(action: Dict[str, Any]) -> np.ndarray:
    move_dist = np.asarray(action["move_dist"], dtype=np.float32).reshape(-1)
    move_angle = np.asarray(action["move_angle"], dtype=np.float32).reshape(-1)
    offload_ratio = np.asarray(action["offload_ratio"], dtype=np.float32).reshape(-1)
    sched_beta = np.asarray(action["sched_beta"], dtype=np.float32).reshape(-1)
    bandwidth_alloc = np.asarray(action["bandwidth_alloc"], dtype=np.float32).reshape(-1)
    cpu_alloc = np.asarray(action["cpu_alloc"], dtype=np.float32).reshape(-1)

    return np.concatenate(
        [
            move_dist,
            move_angle,
            offload_ratio,
            sched_beta,
            bandwidth_alloc,
            cpu_alloc,
        ],
        axis=0,
    ).astype(np.float32)


def soft_update(target_net: nn.Module, source_net: nn.Module, tau: float = 0.005):
    for tp, sp in zip(target_net.parameters(), source_net.parameters()):
        tp.data.copy_(tau * sp.data + (1.0 - tau) * tp.data)


def soft_update_policy(target_policy: ProposedPolicy, source_policy: ProposedPolicy, tau: float = 0.005):
    if target_policy.actor_net is not None and source_policy.actor_net is not None:
        soft_update(target_policy.actor_net, source_policy.actor_net, tau=tau)

    if target_policy.encoder is not None and source_policy.encoder is not None:
        soft_update(target_policy.encoder, source_policy.encoder, tau=tau)

    if target_policy.fusion_net is not None and source_policy.fusion_net is not None:
        soft_update(target_policy.fusion_net, source_policy.fusion_net, tau=tau)

    if target_policy.ratio_head is not None and source_policy.ratio_head is not None:
        soft_update(target_policy.ratio_head, source_policy.ratio_head, tau=tau)


# =========================================================
# Helpers: teacher target
# =========================================================
def build_teacher_raw_target(state, access_assoc, teacher_action):
    """
    Convert env-style teacher action dict into raw-action target format.

    Raw actor layout:
        [ move_dist(M), move_angle(M), offload_ratio(K), sched_score(K*M) ]
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

    move_dist_target = 2.0 * (move_dist / max_move) - 1.0
    move_dist_target = np.clip(move_dist_target, -1.0, 1.0)

    move_angle_target = move_angle / np.pi
    move_angle_target = np.clip(move_angle_target, -1.0, 1.0)

    offload_target = 2.0 * offload_ratio - 1.0
    offload_target = np.clip(offload_target, -1.0, 1.0)

    sched_score_target = -np.ones((K, M), dtype=np.float32)
    for k in range(K):
        access_m = int(np.argmax(access_assoc[:, k]))
        chosen_js = np.where(sched_beta[k, access_m, :] > 0.5)[0]
        if len(chosen_js) == 1:
            sched_score_target[k, int(chosen_js[0])] = 1.0

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


def split_actor_raw_action(raw_action: torch.Tensor, M: int, K: int):
    """
    raw layout:
        [ move_dist(M), move_angle(M), offload_ratio(K), sched_score(K*M) ]
    """
    p = 0
    move_dist_raw = raw_action[:, p:p + M]
    p += M

    move_angle_raw = raw_action[:, p:p + M]
    p += M

    offload_raw = raw_action[:, p:p + K]
    p += K

    sched_raw = raw_action[:, p:p + K * M].reshape(-1, K, M)

    return move_dist_raw, move_angle_raw, offload_raw, sched_raw


# =========================================================
# Helpers: state feature extraction for ratio branch
# =========================================================
def _safe_get(state: Dict[str, Any], key: str, default=None):
    return state[key] if key in state else default


def _to_numpy(x, dtype=np.float32):
    return np.asarray(x, dtype=dtype)


def _extract_neighbors(state: Dict[str, Any]) -> List[List[int]]:
    neighbors = _safe_get(state, "neighbors", None)
    if neighbors is None:
        M = int(state["M"])
        return [[] for _ in range(M)]
    return neighbors


def _extract_uav_positions(state: Dict[str, Any]) -> np.ndarray:
    for key in ["uav_pos", "uav_positions", "q", "uav_xy"]:
        if key in state:
            arr = np.asarray(state[key], dtype=np.float32)
            if arr.ndim == 2 and arr.shape[1] >= 2:
                return arr[:, :2]
    raise KeyError("Cannot find UAV position array in state.")


def _extract_task_size(state: Dict[str, Any]) -> np.ndarray:
    for key in ["task_size", "D", "D_k", "task_data_size"]:
        if key in state:
            return _to_numpy(state[key])
    raise KeyError("Cannot find task size in state.")


def _extract_task_cycles(state: Dict[str, Any]) -> np.ndarray:
    for key in ["task_cycles", "C", "C_k", "task_cpu_cycles"]:
        if key in state:
            return _to_numpy(state[key])
    raise KeyError("Cannot find task cycles in state.")


def _extract_task_deadline(state: Dict[str, Any]) -> np.ndarray:
    for key in ["task_deadline", "deadline", "tau_max", "task_max_delay"]:
        if key in state:
            return _to_numpy(state[key])
    raise KeyError("Cannot find task deadline in state.")


def _extract_available_cpu(state: Dict[str, Any]) -> np.ndarray:
    for key in ["uav_available_cpu", "available_cpu", "uav_cpu_avail", "uav_cpu_max"]:
        if key in state:
            return _to_numpy(state[key])
    M = int(state["M"])
    return np.ones(M, dtype=np.float32)


def _extract_workload(state: Dict[str, Any]) -> np.ndarray:
    for key in ["uav_workload", "workload", "uav_load", "queue_len"]:
        if key in state:
            return _to_numpy(state[key])
    M = int(state["M"])
    return np.zeros(M, dtype=np.float32)


def _extract_tx_power(state: Dict[str, Any], access_m: int) -> float:
    for key in ["uav_tx_power", "P_tx", "tx_power"]:
        if key in state:
            val = state[key]
            arr = np.asarray(val, dtype=np.float32)
            if arr.ndim == 0:
                return float(arr)
            return float(arr[access_m])
    return 1.0


def _extract_backhaul_bw(state: Dict[str, Any]) -> float:
    for key in ["backhaul_bandwidth", "B_bh", "bh_bandwidth"]:
        if key in state:
            return float(state[key])
    return 1.0


def _extract_noise_power(state: Dict[str, Any]) -> float:
    for key in ["noise_power", "sigma2", "noise_var"]:
        if key in state:
            return float(state[key])
    return 1e-9


def _compute_a2a_gain_and_rate(state: Dict[str, Any], access_m: int, exec_j: int) -> Tuple[float, float]:
    if exec_j == access_m:
        return 1.0, 1e6

    uav_pos = _extract_uav_positions(state)
    p_m = uav_pos[access_m]
    p_j = uav_pos[exec_j]

    dist = float(np.linalg.norm(p_m - p_j))
    dist = max(dist, 1.0)

    beta0 = float(_safe_get(state, "beta0", 1.0))
    gain = beta0 * (dist ** -2)

    B_bh = _extract_backhaul_bw(state)
    P_tx = _extract_tx_power(state, access_m)
    sigma2 = _extract_noise_power(state)

    rate = B_bh * np.log2(1.0 + P_tx * gain / max(sigma2, EPS))
    return float(gain), float(rate)


def build_ratio_branch_inputs(
    state: Dict[str, Any],
    access_assoc: np.ndarray,
    max_candidates: int,
    device: str,
):
    """
    Build:
      token_tensor: [K, Nc, 7]
      mask_tensor:  [K, Nc]
      task_feat:    [K, 3]
      uav_feat:     [K, 4]
    """
    M = int(state["M"])
    K = int(state["K"])

    task_size = _extract_task_size(state)
    task_cycles = _extract_task_cycles(state)
    task_deadline = _extract_task_deadline(state)

    available_cpu = _extract_available_cpu(state)
    workload = _extract_workload(state)
    uav_pos = _extract_uav_positions(state)
    neighbors = _extract_neighbors(state)

    token_batches = []
    mask_batches = []
    task_feat_batches = []
    uav_feat_batches = []

    for k in range(K):
        access_m = int(np.argmax(access_assoc[:, k]))
        candidates = [access_m] + list(neighbors[access_m])
        candidates = candidates[:max_candidates]

        tokens = np.zeros((max_candidates, 7), dtype=np.float32)
        mask = np.zeros((max_candidates,), dtype=np.float32)

        for idx, j in enumerate(candidates):
            h_a2a, r_a2a = _compute_a2a_gain_and_rate(state, access_m, j)
            tokens[idx] = np.asarray(
                [
                    float(h_a2a),
                    float(r_a2a),
                    float(workload[j]),
                    float(available_cpu[j]),
                    float(task_size[k]),
                    float(task_cycles[k]),
                    float(task_deadline[k]),
                ],
                dtype=np.float32,
            )
            mask[idx] = 1.0

        task_feat = np.asarray(
            [
                float(task_size[k]),
                float(task_cycles[k]),
                float(task_deadline[k]),
            ],
            dtype=np.float32,
        )

        uav_feat = np.asarray(
            [
                float(uav_pos[access_m, 0]),
                float(uav_pos[access_m, 1]),
                float(available_cpu[access_m]),
                float(workload[access_m]),
            ],
            dtype=np.float32,
        )

        token_batches.append(tokens)
        mask_batches.append(mask)
        task_feat_batches.append(task_feat)
        uav_feat_batches.append(uav_feat)

    token_tensor = torch.tensor(np.asarray(token_batches), dtype=torch.float32, device=device)
    mask_tensor = torch.tensor(np.asarray(mask_batches), dtype=torch.float32, device=device)
    task_feat_tensor = torch.tensor(np.asarray(task_feat_batches), dtype=torch.float32, device=device)
    uav_feat_tensor = torch.tensor(np.asarray(uav_feat_batches), dtype=torch.float32, device=device)

    return token_tensor, mask_tensor, task_feat_tensor, uav_feat_tensor


# def forward_ratio_branch(policy: ProposedPolicy, state: Dict[str, Any], access_assoc: np.ndarray) -> torch.Tensor:
def forward_ratio_branch(
    policy: ProposedPolicy,
    state: Dict[str, Any],
    access_assoc: np.ndarray,
    use_hard_min: bool = True,
) -> torch.Tensor:
    """
    Differentiable forward aligned with deployment path:
        encoder -> fusion -> ratio_head(prior_ratio=...)
    output:
        offload_ratio_pred: [K]
    """
    if policy.encoder is None or policy.fusion_net is None or policy.ratio_head is None:
        raise RuntimeError("policy encoder/fusion_net/ratio_head is None in full-stage1 training.")

    token_tensor, mask_tensor, task_feat_tensor, uav_feat_tensor = build_ratio_branch_inputs(
        state=state,
        access_assoc=access_assoc,
        max_candidates=policy.max_candidates,
        device=policy.device,
    )

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

    # offload_ratio_pred = policy.ratio_head(
    #     fused_feature,
    #     prior_ratio=prior_tensor,
    #     temperature=1.0,
    #     hard_min=policy.ratio_floor,
    # ).squeeze(-1)

    # offload_ratio_pred = torch.clamp(
    #     offload_ratio_pred,
    #     min=policy.ratio_floor,
    #     max=policy.ratio_ceiling,
    # )
    # return offload_ratio_pred

    offload_ratio_pred = policy.ratio_head(
        fused_feature,
        prior_ratio=prior_tensor,
        temperature=1.0,
        hard_min=policy.ratio_floor if use_hard_min else None,
    ).squeeze(-1)

    if use_hard_min:
        offload_ratio_pred = torch.clamp(
            offload_ratio_pred,
            min=policy.ratio_floor,
            max=policy.ratio_ceiling,
        )
    else:
        # 训练 actor / ratio head 时不要在下界处截断梯度
        offload_ratio_pred = torch.clamp(
            offload_ratio_pred,
            min=1e-4,
            max=policy.ratio_ceiling,
        )

    return offload_ratio_pred


def build_stage1_ratio_teacher(
    state,
    access_assoc,
    min_ratio: float = 0.05,
    max_ratio: float = 0.35,
):
    """
    Stage-1 teacher for offloading ratio.
    Generates a simple feasible-biased ratio target for BC warm start.

    Args:
        state: environment raw_state dict
        access_assoc: [M, K] association matrix
        min_ratio: lower bound of offloading ratio
        max_ratio: upper bound of offloading ratio

    Returns:
        np.ndarray of shape [K], each in [min_ratio, max_ratio]
    """
    import numpy as np

    task_deadline = np.asarray(state["task_deadline"], dtype=np.float32)
    task_size = np.asarray(state["task_size"], dtype=np.float32)
    task_cycles = np.asarray(state["task_cycles"], dtype=np.float32)

    size_norm = task_size / (np.max(task_size) + 1e-8)
    cyc_norm = task_cycles / (np.max(task_cycles) + 1e-8)
    ddl_norm = task_deadline / (np.max(task_deadline) + 1e-8)

    urgency = 1.0 - ddl_norm

    # 更保守：降低对大任务/高时限紧迫度的放大量
    score = 0.35 * size_norm + 0.35 * cyc_norm + 0.30 * urgency

    # 压缩到 [0.05, 0.35]
    ratio = min_ratio + (max_ratio - min_ratio) * score

    # 再额外做一次温和压缩，避免过激 teacher
    ratio = 0.7 * ratio + 0.3 * min_ratio

    ratio = np.clip(ratio, min_ratio, max_ratio)
    return ratio.astype(np.float32)


# =========================================================
# Replay buffer
# =========================================================
class FullReplayBuffer:
    def __init__(self, obs_dim: int, action_dim: int, capacity: int = 10000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.obs_dim = obs_dim
        self.action_dim = action_dim

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        next_action: np.ndarray,
        done: float,
    ):
        self.buffer.append(
            {
                "obs": np.asarray(obs, dtype=np.float32),
                "action": np.asarray(action, dtype=np.float32),
                "reward": float(reward),
                "next_obs": np.asarray(next_obs, dtype=np.float32),
                "next_action": np.asarray(next_action, dtype=np.float32),
                "done": float(done),
            }
        )

    def sample(self, batch_size: int):
        idx = np.random.choice(len(self.buffer), size=batch_size, replace=False)
        batch = [self.buffer[i] for i in idx]

        return {
            "obs": np.stack([b["obs"] for b in batch], axis=0),
            "action": np.stack([b["action"] for b in batch], axis=0),
            "reward": np.asarray([b["reward"] for b in batch], dtype=np.float32).reshape(-1, 1),
            "next_obs": np.stack([b["next_obs"] for b in batch], axis=0),
            "next_action": np.stack([b["next_action"] for b in batch], axis=0),
            "done": np.asarray([b["done"] for b in batch], dtype=np.float32).reshape(-1, 1),
        }

    def __len__(self):
        return len(self.buffer)


# =========================================================
# Evaluation
# =========================================================
@torch.no_grad()
def evaluate_full_policy_rollout(env: MultiUavMecEnv, policy: ProposedPolicy, seed: int = 72):
    obs = env.reset(seed=seed)
    done = False

    total_reward = 0.0
    total_delay = 0.0
    total_energy = 0.0
    total_deadline_violation = 0.0
    feasible_count = 0
    step_count = 0

    # 新增：各类失败约束累计
    total_ratio_violation = 0.0
    total_assoc_violation = 0.0
    total_schedule_violation = 0.0
    total_candidate_violation = 0.0
    total_bw_violation = 0.0
    total_cpu_violation = 0.0
    total_rate_violation = 0.0
    total_nan_count = 0.0

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

        report = info["report"]
        metrics = info["metrics"]

        total_reward += reward
        total_delay += metrics["delay_sys"]
        total_energy += metrics["energy_sys"]
        total_deadline_violation += report.get("deadline_violation", 0.0)
        feasible_count += int(report.get("ok", False))
        step_count += 1

        # 新增：累计细分 violation
        total_ratio_violation += report.get("ratio_violation", 0.0)
        total_assoc_violation += report.get("assoc_violation", 0.0)
        total_schedule_violation += report.get("schedule_violation", 0.0)
        total_candidate_violation += report.get("candidate_violation", 0.0)
        total_bw_violation += report.get("bw_violation", 0.0)
        total_cpu_violation += report.get("cpu_violation", 0.0)
        total_rate_violation += report.get("rate_violation", 0.0)
        total_nan_count += report.get("nan_count", 0.0)

    denom = max(step_count, 1)

    return {
        "episode_reward": total_reward,
        "avg_delay": total_delay / denom,
        "avg_energy": total_energy / denom,
        "avg_deadline_violation": total_deadline_violation / denom,
        "feasible_ratio": feasible_count / denom,
        "num_steps": step_count,

        # 新增：返回细分诊断
        "avg_ratio_violation": total_ratio_violation / denom,
        "avg_assoc_violation": total_assoc_violation / denom,
        "avg_schedule_violation": total_schedule_violation / denom,
        "avg_candidate_violation": total_candidate_violation / denom,
        "avg_bw_violation": total_bw_violation / denom,
        "avg_cpu_violation": total_cpu_violation / denom,
        "avg_rate_violation": total_rate_violation / denom,
        "avg_nan_count": total_nan_count / denom,
    }

def load_if_exists(module, path, name, device="cpu"):
    if module is None:
        return
    if os.path.exists(path):
        module.load_state_dict(torch.load(path, map_location=device))
        print(f"Loaded {name} from: {path}")
    else:
        print(f"WARNING: {name} checkpoint not found: {path}")


# =========================================================
# Main
# =========================================================
def main():
    device = "cpu"
    torch.manual_seed(72)
    np.random.seed(72)

    # -------------------------------------------------
    # Env config
    # -------------------------------------------------
    env = MultiUavMecEnv(
        M=3,
        # K=8,
        # episode_length=5,
        K=16,
        episode_length=5,
        cpu_mode="kkt",
        omega1=10.0,
        omega2=1.0,
        # deadline_scale=2.0,
        deadline_scale=5.0,
        # deadline_scale=8.0,
        # task_local_cpu_min=3.0e3,
        # task_local_cpu_max=8.0e3,
        # task_local_cpu_min=1.0e3,
        # task_local_cpu_max=4.0e3,
        task_local_cpu_min=2.0e3,
        task_local_cpu_max=6.0e3,
        uav_energy_min=2600.0,
        uav_energy_max=3800.0,
        # seed=42,
        seed=72,
    )

    obs = env.reset(seed=72)
    state = obs["raw_state"]

    obs_dim = get_observation_dim(state)
    actor_raw_action_dim = get_actor_raw_action_dim(state)
    critic_action_dim = get_full_action_dim(state)
    print("DEBUG deadline_scale =", env.deadline_scale)
    print("obs_dim:", obs_dim)
    print("actor_raw_action_dim:", actor_raw_action_dim)
    print("critic_action_dim:", critic_action_dim)

    # -------------------------------------------------
    # Full proposed policy
    # -------------------------------------------------
    actor = MLPActor(
        obs_dim=obs_dim,
        action_dim=actor_raw_action_dim,
        hidden_dim=256,
    ).to(device)

    policy = build_default_proposed_policy(
        state=state,
        actor_net=actor,
        device=device,
        embed_dim=128,
        num_heads=4,
        ff_hidden_dim=256,
        num_layers=2,
    )

    # Warm-start old stable modules except ratio head
    # load_if_exists(policy.actor_net, "checkpoints/proposed_full_stage2_best_actor.pth", "old stage2 actor", device=device)
    # load_if_exists(policy.encoder, "checkpoints/proposed_full_stage2_best_encoder.pth", "old stage2 encoder", device=device)
    # load_if_exists(policy.fusion_net, "checkpoints/proposed_full_stage2_best_fusion.pth", "old stage2 fusion", device=device)
    # do NOT load old ratio_head because architecture changed

    target_policy = copy.deepcopy(policy)

    # -------------------------------------------------
    # Critic for full-action skeleton
    # -------------------------------------------------
    critic = MLPCritic(
        obs_dim=obs_dim,
        action_dim=critic_action_dim,
        hidden_dim=256,
    ).to(device)

    target_critic = copy.deepcopy(critic).to(device)

    # -------------------------------------------------
    # Optimizers
    # -------------------------------------------------
    learnable_modules = []
    if policy.actor_net is not None:
        learnable_modules += list(policy.actor_net.parameters())
    if policy.encoder is not None:
        learnable_modules += list(policy.encoder.parameters())
    if policy.fusion_net is not None:
        learnable_modules += list(policy.fusion_net.parameters())
    if policy.ratio_head is not None:
        learnable_modules += list(policy.ratio_head.parameters())

    actor_opt = optim.Adam(learnable_modules, lr=1e-3)
    critic_opt = optim.Adam(critic.parameters(), lr=1e-3)

    mse_loss = nn.MSELoss()

    # -------------------------------------------------
    # Replay buffer
    # -------------------------------------------------
    buffer = FullReplayBuffer(
        obs_dim=obs_dim,
        action_dim=critic_action_dim,
        capacity=20000,
    )

    # -------------------------------------------------
    # Hyperparameters
    # -------------------------------------------------
    gamma = 0.99
    tau = 0.005
    batch_size = 32
    num_episodes = 200

    # actor_move_sched_coef = 1.0
    # ratio_bc_coef = 1.0
    # actor_l2_coef = 1e-4

    actor_move_sched_coef = 1.0
    ratio_bc_coef = 6.0
    actor_l2_coef = 1e-5

    # movement+scheduling masks
    M = int(state["M"])
    K = int(state["K"])
    move_sched_mask = np.zeros((actor_raw_action_dim,), dtype=np.float32)
    move_sched_mask[: M + M] = 1.0
    move_sched_mask[M + M + K:] = 1.0
    move_sched_mask_t = torch.tensor(move_sched_mask, dtype=torch.float32, device=device).unsqueeze(0)

    best_policy_state = copy.deepcopy(policy)
    best_eval_reward = -float("inf")

    # -------------------------------------------------
    # Training loop
    # -------------------------------------------------
    for ep in range(num_episodes):
        obs = env.reset(seed=72 + ep)
        done = False

        episode_reward = 0.0
        episode_move_sched_loss = 0.0
        episode_ratio_loss = 0.0
        step_count = 0

        while not done:
            state = obs["raw_state"]
            access_assoc = build_access_association(state)
            obs_vec = build_global_observation(state)

            # -----------------------------------------
            # Teacher action from placeholder
            # -----------------------------------------
            teacher_action = generate_proposed_placeholder_action(
                state=state,
                access_assoc=access_assoc,
            )
            teacher_raw_target = build_teacher_raw_target(
                state=state,
                access_assoc=access_assoc,
                teacher_action=teacher_action,
            )
            # teacher_offload_ratio = np.asarray(
            #     teacher_action["offload_ratio"],
            #     dtype=np.float32,
            # )
            teacher_offload_ratio_base = np.asarray(
                teacher_action["offload_ratio"],
                dtype=np.float32,
            )

            teacher_offload_ratio = build_stage1_ratio_teacher(
                state=state,
                access_assoc=access_assoc,
            )

            teacher_offload_ratio_t = torch.tensor(
                teacher_offload_ratio,
                dtype=torch.float32,
                device=device,
            )

            # -----------------------------------------
            # Current full proposed action
            # -----------------------------------------
            action = policy.act(
                state=state,
                access_assoc=access_assoc,
                deterministic=True,
                return_aux=False,
            )
            flat_action = flatten_full_action(action)

            # -----------------------------------------
            # Environment step
            # -----------------------------------------
            next_obs, reward, done, info = env.step(action)
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

            # -----------------------------------------
            # Actor / encoder / ratio-head BC update
            # -----------------------------------------
            obs_tensor = torch.tensor(obs_vec, dtype=torch.float32, device=device).unsqueeze(0)
            teacher_raw_target_t = torch.tensor(teacher_raw_target, dtype=torch.float32, device=device).unsqueeze(0)
            teacher_offload_ratio_t = torch.tensor(teacher_offload_ratio, dtype=torch.float32, device=device)

            raw_pred = policy.actor_net(obs_tensor)
            move_sched_pred = raw_pred * move_sched_mask_t
            move_sched_target = teacher_raw_target_t * move_sched_mask_t
            actor_move_sched_loss = mse_loss(move_sched_pred, move_sched_target)

            ratio_pred = forward_ratio_branch(
                policy=policy,
                state=state,
                access_assoc=access_assoc,
            )
            ratio_bc_loss = mse_loss(ratio_pred, teacher_offload_ratio_t)

            actor_l2_loss = (raw_pred ** 2).mean()
            total_actor_loss = (
                actor_move_sched_coef * actor_move_sched_loss
                + ratio_bc_coef * ratio_bc_loss
                + actor_l2_coef * actor_l2_loss
            )

            actor_opt.zero_grad()
            total_actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(learnable_modules, max_norm=5.0)
            actor_opt.step()

            # -----------------------------------------
            # Critic skeleton update
            # -----------------------------------------
            if len(buffer) >= batch_size:
                batch = buffer.sample(batch_size)

                obs_b = torch.tensor(batch["obs"], dtype=torch.float32, device=device)
                action_b = torch.tensor(batch["action"], dtype=torch.float32, device=device)
                reward_b = torch.tensor(batch["reward"], dtype=torch.float32, device=device)
                next_obs_b = torch.tensor(batch["next_obs"], dtype=torch.float32, device=device)
                next_action_b = torch.tensor(batch["next_action"], dtype=torch.float32, device=device)
                done_b = torch.tensor(batch["done"], dtype=torch.float32, device=device)

                with torch.no_grad():
                    target_q = target_critic(next_obs_b, next_action_b)
                    y = reward_b + gamma * (1.0 - done_b) * target_q

                q_val = critic(obs_b, action_b)
                critic_loss = mse_loss(q_val, y)

                critic_opt.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=5.0)
                critic_opt.step()

                soft_update_policy(target_policy, policy, tau=tau)
                soft_update(target_critic, critic, tau=tau)

            # if step_count < 2:
            #     with torch.no_grad():
            #         print("[STAGE1 DEBUG]")
            #         print("  ratio_pred[:5] =", ratio_pred.detach().cpu().numpy()[:5])
            #         print("  teacher_raw[:5] =", teacher_offload_ratio[:5])

            if step_count < 2:
                with torch.no_grad():
                    ratio_prior_dbg = policy._compute_ratio_prior(state, access_assoc)
                    print("[STAGE1 DEBUG]")
                    print("  ratio_pred[:5]        =", ratio_pred.detach().cpu().numpy()[:5])
                    print("  teacher_base[:5]      =", teacher_offload_ratio_base[:5])
                    print("  teacher_enhanced[:5]  =", teacher_offload_ratio[:5])
                    print("  ratio_prior[:5]       =", ratio_prior_dbg[:5])
                    print("  mean_ratio_pred       =", float(ratio_pred.mean().item()))
                    print("  mean_teacher_ratio    =", float(np.mean(teacher_offload_ratio)))

            obs = next_obs
            episode_reward += reward
            episode_move_sched_loss += float(actor_move_sched_loss.item())
            episode_ratio_loss += float(ratio_bc_loss.item())
            step_count += 1

        print(f"\nEpisode {ep}")
        print("episode_reward:", episode_reward)
        print("avg_move_sched_loss:", episode_move_sched_loss / max(step_count, 1))
        print("avg_ratio_loss:", episode_ratio_loss / max(step_count, 1))
        print("steps:", step_count)
        print("buffer_size:", len(buffer))

        # -----------------------------------------
        # Evaluation
        # -----------------------------------------
        if (ep + 1) % 5 == 0 or ep == 0:
            eval_env = MultiUavMecEnv(
                M=3,
                K=16,
                episode_length=5,
                cpu_mode="kkt",
                omega1=10.0,
                omega2=1.0,
                # deadline_scale=2.0,
                deadline_scale=5.0,
                # deadline_scale=8.0,
                # task_local_cpu_min=3.0e3,
                # task_local_cpu_max=8.0e3,
                # task_local_cpu_min=1.0e3,
                # task_local_cpu_max=4.0e3,
                task_local_cpu_min=2.0e3,
                task_local_cpu_max=6.0e3,
                uav_energy_min=2600.0,
                uav_energy_max=3800.0,
                seed=999,
            )

            eval_result = evaluate_full_policy_rollout(
                env=eval_env,
                policy=policy,
                seed=999,
            )

            print("DEBUG eval deadline_scale =", eval_env.deadline_scale)

            print("\n==============================")
            print(f"Full Proposed Stage-1 Eval @ episode {ep}")
            print("==============================")
            print("episode_reward:", eval_result["episode_reward"])
            print("avg_delay:", eval_result["avg_delay"])
            print("avg_energy:", eval_result["avg_energy"])
            print("avg_deadline_violation:", eval_result["avg_deadline_violation"])
            print("feasible_ratio:", eval_result["feasible_ratio"])
            print("num_steps:", eval_result["num_steps"])
            print("avg_ratio_violation:", eval_result["avg_ratio_violation"])
            print("avg_assoc_violation:", eval_result["avg_assoc_violation"])
            print("avg_schedule_violation:", eval_result["avg_schedule_violation"])
            print("avg_candidate_violation:", eval_result["avg_candidate_violation"])
            print("avg_bw_violation:", eval_result["avg_bw_violation"])
            print("avg_cpu_violation:", eval_result["avg_cpu_violation"])
            print("avg_rate_violation:", eval_result["avg_rate_violation"])
            print("avg_nan_count:", eval_result["avg_nan_count"])

            if eval_result["episode_reward"] > best_eval_reward:
                best_eval_reward = eval_result["episode_reward"]
                best_policy_state = copy.deepcopy(policy)

    # -------------------------------------------------
    # Save checkpoints
    # -------------------------------------------------

    # os.makedirs("checkpoints", exist_ok=True)

    # if best_policy_state.actor_net is not None:
    #     torch.save(
    #         best_policy_state.actor_net.state_dict(),
    #         "checkpoints/proposed_full_stage1_fix_best_actor.pth",
    #     )
    # if best_policy_state.encoder is not None:
    #     torch.save(
    #         best_policy_state.encoder.state_dict(),
    #         "checkpoints/proposed_full_stage1_fix_best_encoder.pth",
    #     )
    # if best_policy_state.fusion_net is not None:
    #     torch.save(
    #         best_policy_state.fusion_net.state_dict(),
    #         "checkpoints/proposed_full_stage1_fix_best_fusion.pth",
    #     )
    # if best_policy_state.ratio_head is not None:
    #     torch.save(
    #         best_policy_state.ratio_head.state_dict(),
    #         "checkpoints/proposed_full_stage1_fix_best_ratio_head.pth",
    #     )

    # torch.save(
    #     critic.state_dict(),
    #     "checkpoints/proposed_full_stage1_fix_critic.pth",
    # )

    # print("\nFull proposed Stage-1 training finished successfully.")
    # print("Saved:")
    # print("  checkpoints/proposed_full_stage1_fix_best_actor.pth")
    # print("  checkpoints/proposed_full_stage1_fix_best_encoder.pth")
    # print("  checkpoints/proposed_full_stage1_fix_best_fusion.pth")
    # print("  checkpoints/proposed_full_stage1_fix_best_ratio_head.pth")
    # print("  checkpoints/proposed_full_stage1_fix_critic.pth")

    os.makedirs("checkpoints", exist_ok=True)

    if best_policy_state.actor_net is not None:
        torch.save(
            best_policy_state.actor_net.state_dict(),
            "checkpoints/proposed_full_stage1_fix_k16_ep40_best_actor.pth",
        )
    if best_policy_state.encoder is not None:
        torch.save(
            best_policy_state.encoder.state_dict(),
            "checkpoints/proposed_full_stage1_fix_k16_ep40_best_encoder.pth",
        )
    if best_policy_state.fusion_net is not None:
        torch.save(
            best_policy_state.fusion_net.state_dict(),
            "checkpoints/proposed_full_stage1_fix_k16_ep40_best_fusion.pth",
        )
    if best_policy_state.ratio_head is not None:
        torch.save(
            best_policy_state.ratio_head.state_dict(),
            "checkpoints/proposed_full_stage1_fix_k16_ep40_best_ratio_head.pth",
        )

    torch.save(
        critic.state_dict(),
        "checkpoints/proposed_full_stage1_fix_k16_ep40_critic.pth",
    )

    print("\nFull proposed Stage-1 training finished successfully.")
    print("Saved:")
    print("  checkpoints/proposed_full_stage1_fix_k16_ep40_best_actor.pth")
    print("  checkpoints/proposed_full_stage1_fix_k16_ep40_best_encoder.pth")
    print("  checkpoints/proposed_full_stage1_fix_k16_ep40_best_fusion.pth")
    print("  checkpoints/proposed_full_stage1_fix_k16_ep40_best_ratio_head.pth")
    print("  checkpoints/proposed_full_stage1_fix_k16_ep40_critic.pth")


if __name__ == "__main__":
    main()


# EPS = 1e-8


# # =========================================================
# # Helpers: dimensions / flatten / soft update
# # =========================================================
# def get_actor_raw_action_dim(state: Dict[str, Any]) -> int:
#     """
#     Raw actor output layout:
#         [ move_dist(M), move_angle(M), offload_ratio(K), sched_score(K*M) ]
#     """
#     M = int(state["M"])
#     K = int(state["K"])
#     return M + M + K + K * M


# def get_full_action_dim(state: Dict[str, Any]) -> int:
#     """
#     Full action layout used by critic:
#         [ move_dist(M),
#           move_angle(M),
#           offload_ratio(K),
#           sched_beta(K*M*M),
#           bandwidth_alloc(M*K),
#           cpu_alloc(M*K) ]
#     """
#     M = int(state["M"])
#     K = int(state["K"])
#     return M + M + K + K * M * M + M * K + M * K


# def flatten_full_action(action: Dict[str, Any]) -> np.ndarray:
#     move_dist = np.asarray(action["move_dist"], dtype=np.float32).reshape(-1)
#     move_angle = np.asarray(action["move_angle"], dtype=np.float32).reshape(-1)
#     offload_ratio = np.asarray(action["offload_ratio"], dtype=np.float32).reshape(-1)
#     sched_beta = np.asarray(action["sched_beta"], dtype=np.float32).reshape(-1)
#     bandwidth_alloc = np.asarray(action["bandwidth_alloc"], dtype=np.float32).reshape(-1)
#     cpu_alloc = np.asarray(action["cpu_alloc"], dtype=np.float32).reshape(-1)

#     return np.concatenate(
#         [
#             move_dist,
#             move_angle,
#             offload_ratio,
#             sched_beta,
#             bandwidth_alloc,
#             cpu_alloc,
#         ],
#         axis=0,
#     ).astype(np.float32)


# def soft_update(target_net: nn.Module, source_net: nn.Module, tau: float = 0.005):
#     for tp, sp in zip(target_net.parameters(), source_net.parameters()):
#         tp.data.copy_(tau * sp.data + (1.0 - tau) * tp.data)


# def soft_update_policy(target_policy: ProposedPolicy, source_policy: ProposedPolicy, tau: float = 0.005):
#     if target_policy.actor_net is not None and source_policy.actor_net is not None:
#         soft_update(target_policy.actor_net, source_policy.actor_net, tau=tau)

#     if target_policy.encoder is not None and source_policy.encoder is not None:
#         soft_update(target_policy.encoder, source_policy.encoder, tau=tau)

#     if target_policy.fusion_net is not None and source_policy.fusion_net is not None:
#         soft_update(target_policy.fusion_net, source_policy.fusion_net, tau=tau)

#     if target_policy.ratio_head is not None and source_policy.ratio_head is not None:
#         soft_update(target_policy.ratio_head, source_policy.ratio_head, tau=tau)


# # =========================================================
# # Helpers: teacher target
# # =========================================================
# def build_teacher_raw_target(state, access_assoc, teacher_action):
#     """
#     Convert env-style teacher action dict into raw-action target format.

#     Raw actor layout:
#         [ move_dist(M), move_angle(M), offload_ratio(K), sched_score(K*M) ]
#     """
#     M = int(state["M"])
#     K = int(state["K"])
#     max_speed = float(state["max_speed"])
#     delta_t = float(state["delta_t"])
#     max_move = max(max_speed * delta_t, EPS)

#     move_dist = np.asarray(teacher_action["move_dist"], dtype=np.float32).reshape(M)
#     move_angle = np.asarray(teacher_action["move_angle"], dtype=np.float32).reshape(M)
#     offload_ratio = np.asarray(teacher_action["offload_ratio"], dtype=np.float32).reshape(K)
#     sched_beta = np.asarray(teacher_action["sched_beta"], dtype=np.float32).reshape(K, M, M)

#     move_dist_target = 2.0 * (move_dist / max_move) - 1.0
#     move_dist_target = np.clip(move_dist_target, -1.0, 1.0)

#     move_angle_target = move_angle / np.pi
#     move_angle_target = np.clip(move_angle_target, -1.0, 1.0)

#     offload_target = 2.0 * offload_ratio - 1.0
#     offload_target = np.clip(offload_target, -1.0, 1.0)

#     sched_score_target = -np.ones((K, M), dtype=np.float32)
#     for k in range(K):
#         access_m = int(np.argmax(access_assoc[:, k]))
#         chosen_js = np.where(sched_beta[k, access_m, :] > 0.5)[0]
#         if len(chosen_js) == 1:
#             sched_score_target[k, int(chosen_js[0])] = 1.0

#     raw_target = np.concatenate(
#         [
#             move_dist_target.reshape(-1),
#             move_angle_target.reshape(-1),
#             offload_target.reshape(-1),
#             sched_score_target.reshape(-1),
#         ],
#         axis=0,
#     ).astype(np.float32)

#     return raw_target


# def split_actor_raw_action(raw_action: torch.Tensor, M: int, K: int):
#     """
#     raw layout:
#         [ move_dist(M), move_angle(M), offload_ratio(K), sched_score(K*M) ]
#     """
#     p = 0
#     move_dist_raw = raw_action[:, p:p + M]
#     p += M

#     move_angle_raw = raw_action[:, p:p + M]
#     p += M

#     offload_raw = raw_action[:, p:p + K]
#     p += K

#     sched_raw = raw_action[:, p:p + K * M].reshape(-1, K, M)

#     return move_dist_raw, move_angle_raw, offload_raw, sched_raw


# # =========================================================
# # Helpers: state feature extraction for ratio branch
# # =========================================================
# def _safe_get(state: Dict[str, Any], key: str, default=None):
#     return state[key] if key in state else default


# def _to_numpy(x, dtype=np.float32):
#     return np.asarray(x, dtype=dtype)


# def _extract_neighbors(state: Dict[str, Any]) -> List[List[int]]:
#     neighbors = _safe_get(state, "neighbors", None)
#     if neighbors is None:
#         M = int(state["M"])
#         return [[] for _ in range(M)]
#     return neighbors


# def _extract_uav_positions(state: Dict[str, Any]) -> np.ndarray:
#     for key in ["uav_pos", "uav_positions", "q", "uav_xy"]:
#         if key in state:
#             arr = np.asarray(state[key], dtype=np.float32)
#             if arr.ndim == 2 and arr.shape[1] >= 2:
#                 return arr[:, :2]
#     raise KeyError("Cannot find UAV position array in state.")


# def _extract_task_size(state: Dict[str, Any]) -> np.ndarray:
#     for key in ["task_size", "D", "D_k", "task_data_size"]:
#         if key in state:
#             return _to_numpy(state[key])
#     raise KeyError("Cannot find task size in state.")


# def _extract_task_cycles(state: Dict[str, Any]) -> np.ndarray:
#     for key in ["task_cycles", "C", "C_k", "task_cpu_cycles"]:
#         if key in state:
#             return _to_numpy(state[key])
#     raise KeyError("Cannot find task cycles in state.")


# def _extract_task_deadline(state: Dict[str, Any]) -> np.ndarray:
#     for key in ["task_deadline", "deadline", "tau_max", "task_max_delay"]:
#         if key in state:
#             return _to_numpy(state[key])
#     raise KeyError("Cannot find task deadline in state.")


# def _extract_available_cpu(state: Dict[str, Any]) -> np.ndarray:
#     for key in ["uav_available_cpu", "available_cpu", "uav_cpu_avail", "uav_cpu_max"]:
#         if key in state:
#             return _to_numpy(state[key])
#     M = int(state["M"])
#     return np.ones(M, dtype=np.float32)


# def _extract_workload(state: Dict[str, Any]) -> np.ndarray:
#     for key in ["uav_workload", "workload", "uav_load", "queue_len"]:
#         if key in state:
#             return _to_numpy(state[key])
#     M = int(state["M"])
#     return np.zeros(M, dtype=np.float32)


# def _extract_tx_power(state: Dict[str, Any], access_m: int) -> float:
#     for key in ["uav_tx_power", "P_tx", "tx_power"]:
#         if key in state:
#             val = state[key]
#             arr = np.asarray(val, dtype=np.float32)
#             if arr.ndim == 0:
#                 return float(arr)
#             return float(arr[access_m])
#     return 1.0


# def _extract_backhaul_bw(state: Dict[str, Any]) -> float:
#     for key in ["backhaul_bandwidth", "B_bh", "bh_bandwidth"]:
#         if key in state:
#             return float(state[key])
#     return 1.0


# def _extract_noise_power(state: Dict[str, Any]) -> float:
#     for key in ["noise_power", "sigma2", "noise_var"]:
#         if key in state:
#             return float(state[key])
#     return 1e-9


# def _compute_a2a_gain_and_rate(state: Dict[str, Any], access_m: int, exec_j: int) -> Tuple[float, float]:
#     if exec_j == access_m:
#         return 1.0, 1e6

#     uav_pos = _extract_uav_positions(state)
#     p_m = uav_pos[access_m]
#     p_j = uav_pos[exec_j]

#     dist = float(np.linalg.norm(p_m - p_j))
#     dist = max(dist, 1.0)

#     beta0 = float(_safe_get(state, "beta0", 1.0))
#     gain = beta0 * (dist ** -2)

#     B_bh = _extract_backhaul_bw(state)
#     P_tx = _extract_tx_power(state, access_m)
#     sigma2 = _extract_noise_power(state)

#     rate = B_bh * np.log2(1.0 + P_tx * gain / max(sigma2, EPS))
#     return float(gain), float(rate)


# def build_ratio_branch_inputs(
#     state: Dict[str, Any],
#     access_assoc: np.ndarray,
#     max_candidates: int,
#     device: str,
# ):
#     """
#     Build:
#       token_tensor: [K, Nc, 7]
#       mask_tensor:  [K, Nc]
#       task_feat:    [K, 3]
#       uav_feat:     [K, 4]
#     """
#     M = int(state["M"])
#     K = int(state["K"])

#     task_size = _extract_task_size(state)
#     task_cycles = _extract_task_cycles(state)
#     task_deadline = _extract_task_deadline(state)

#     available_cpu = _extract_available_cpu(state)
#     workload = _extract_workload(state)
#     uav_pos = _extract_uav_positions(state)
#     neighbors = _extract_neighbors(state)

#     token_batches = []
#     mask_batches = []
#     task_feat_batches = []
#     uav_feat_batches = []

#     for k in range(K):
#         access_m = int(np.argmax(access_assoc[:, k]))
#         candidates = [access_m] + list(neighbors[access_m])
#         candidates = candidates[:max_candidates]

#         tokens = np.zeros((max_candidates, 7), dtype=np.float32)
#         mask = np.zeros((max_candidates,), dtype=np.float32)

#         for idx, j in enumerate(candidates):
#             h_a2a, r_a2a = _compute_a2a_gain_and_rate(state, access_m, j)
#             tokens[idx] = np.asarray(
#                 [
#                     float(h_a2a),
#                     float(r_a2a),
#                     float(workload[j]),
#                     float(available_cpu[j]),
#                     float(task_size[k]),
#                     float(task_cycles[k]),
#                     float(task_deadline[k]),
#                 ],
#                 dtype=np.float32,
#             )
#             mask[idx] = 1.0

#         task_feat = np.asarray(
#             [
#                 float(task_size[k]),
#                 float(task_cycles[k]),
#                 float(task_deadline[k]),
#             ],
#             dtype=np.float32,
#         )

#         uav_feat = np.asarray(
#             [
#                 float(uav_pos[access_m, 0]),
#                 float(uav_pos[access_m, 1]),
#                 float(available_cpu[access_m]),
#                 float(workload[access_m]),
#             ],
#             dtype=np.float32,
#         )

#         token_batches.append(tokens)
#         mask_batches.append(mask)
#         task_feat_batches.append(task_feat)
#         uav_feat_batches.append(uav_feat)

#     token_tensor = torch.tensor(np.asarray(token_batches), dtype=torch.float32, device=device)
#     mask_tensor = torch.tensor(np.asarray(mask_batches), dtype=torch.float32, device=device)
#     task_feat_tensor = torch.tensor(np.asarray(task_feat_batches), dtype=torch.float32, device=device)
#     uav_feat_tensor = torch.tensor(np.asarray(uav_feat_batches), dtype=torch.float32, device=device)

#     return token_tensor, mask_tensor, task_feat_tensor, uav_feat_tensor


# # def forward_ratio_branch(policy: ProposedPolicy, state: Dict[str, Any], access_assoc: np.ndarray) -> torch.Tensor:
# #     """
# #     Differentiable forward for:
# #         encoder -> fusion -> ratio_head
# #     output:
# #         offload_ratio_pred: [K]
# #     """
# #     if policy.encoder is None or policy.fusion_net is None or policy.ratio_head is None:
# #         raise RuntimeError("policy encoder/fusion_net/ratio_head is None in full-stage1 training.")

# #     token_tensor, mask_tensor, task_feat_tensor, uav_feat_tensor = build_ratio_branch_inputs(
# #         state=state,
# #         access_assoc=access_assoc,
# #         max_candidates=policy.max_candidates,
# #         device=policy.device,
# #     )

# #     encoded_tokens, pooled_context = policy.encoder(token_tensor, mask_tensor)
# #     fused_feature = policy.fusion_net(
# #         topo_context=pooled_context,
# #         task_feat=task_feat_tensor,
# #         uav_feat=uav_feat_tensor,
# #     )
# #     offload_ratio_pred = policy.ratio_head(fused_feature).squeeze(-1)
# #     return offload_ratio_pred
# def forward_ratio_branch(policy: ProposedPolicy, state: Dict[str, Any], access_assoc: np.ndarray) -> torch.Tensor:
#     """
#     Differentiable forward aligned with deployment path:
#         encoder -> fusion -> ratio_head(prior_ratio=...)
#     output:
#         offload_ratio_pred: [K]
#     """
#     if policy.encoder is None or policy.fusion_net is None or policy.ratio_head is None:
#         raise RuntimeError("policy encoder/fusion_net/ratio_head is None in full-stage1 training.")

#     token_tensor, mask_tensor, task_feat_tensor, uav_feat_tensor = build_ratio_branch_inputs(
#         state=state,
#         access_assoc=access_assoc,
#         max_candidates=policy.max_candidates,
#         device=policy.device,
#     )

#     # build prior ratio using the SAME logic family as deployment
#     ratio_prior_np = policy._compute_ratio_prior(state, access_assoc)
#     prior_tensor = torch.tensor(
#         ratio_prior_np,
#         dtype=torch.float32,
#         device=policy.device,
#     ).unsqueeze(-1)   # [K, 1]

#     encoded_tokens, pooled_context = policy.encoder(token_tensor, mask_tensor)
#     fused_feature = policy.fusion_net(
#         topo_context=pooled_context,
#         task_feat=task_feat_tensor,
#         uav_feat=uav_feat_tensor,
#     )

#     offload_ratio_pred = policy.ratio_head(
#         fused_feature,
#         prior_ratio=prior_tensor,
#         temperature=1.0,
#         hard_min=policy.ratio_floor,
#     ).squeeze(-1)

#     offload_ratio_pred = torch.clamp(
#         offload_ratio_pred,
#         min=policy.ratio_floor,
#         max=policy.ratio_ceiling,
#     )
#     return offload_ratio_pred

# # =========================================================
# # Replay buffer
# # =========================================================
# class FullReplayBuffer:
#     def __init__(self, obs_dim: int, action_dim: int, capacity: int = 10000):
#         self.capacity = capacity
#         self.buffer = deque(maxlen=capacity)
#         self.obs_dim = obs_dim
#         self.action_dim = action_dim

#     def add(
#         self,
#         obs: np.ndarray,
#         action: np.ndarray,
#         reward: float,
#         next_obs: np.ndarray,
#         next_action: np.ndarray,
#         done: float,
#     ):
#         self.buffer.append(
#             {
#                 "obs": np.asarray(obs, dtype=np.float32),
#                 "action": np.asarray(action, dtype=np.float32),
#                 "reward": float(reward),
#                 "next_obs": np.asarray(next_obs, dtype=np.float32),
#                 "next_action": np.asarray(next_action, dtype=np.float32),
#                 "done": float(done),
#             }
#         )

#     def sample(self, batch_size: int):
#         idx = np.random.choice(len(self.buffer), size=batch_size, replace=False)
#         batch = [self.buffer[i] for i in idx]

#         return {
#             "obs": np.stack([b["obs"] for b in batch], axis=0),
#             "action": np.stack([b["action"] for b in batch], axis=0),
#             "reward": np.asarray([b["reward"] for b in batch], dtype=np.float32).reshape(-1, 1),
#             "next_obs": np.stack([b["next_obs"] for b in batch], axis=0),
#             "next_action": np.stack([b["next_action"] for b in batch], axis=0),
#             "done": np.asarray([b["done"] for b in batch], dtype=np.float32).reshape(-1, 1),
#         }

#     def __len__(self):
#         return len(self.buffer)


# # =========================================================
# # Evaluation
# # =========================================================
# @torch.no_grad()
# def evaluate_full_policy_rollout(env: MultiUavMecEnv, policy: ProposedPolicy, seed: int = 123):
#     obs = env.reset(seed=seed)
#     done = False

#     total_reward = 0.0
#     total_delay = 0.0
#     total_energy = 0.0
#     total_deadline_violation = 0.0
#     feasible_count = 0
#     step_count = 0

#     while not done:
#         state = obs["raw_state"]
#         access_assoc = build_access_association(state)

#         action = policy.act(
#             state=state,
#             access_assoc=access_assoc,
#             deterministic=True,
#             return_aux=False,
#         )

#         obs, reward, done, info = env.step(action)

#         total_reward += reward
#         total_delay += info["metrics"]["delay_sys"]
#         total_energy += info["metrics"]["energy_sys"]
#         total_deadline_violation += info["report"]["deadline_violation"]
#         feasible_count += int(info["report"]["ok"])
#         step_count += 1

#     return {
#         "episode_reward": total_reward,
#         "avg_delay": total_delay / max(step_count, 1),
#         "avg_energy": total_energy / max(step_count, 1),
#         "avg_deadline_violation": total_deadline_violation / max(step_count, 1),
#         "feasible_ratio": feasible_count / max(step_count, 1),
#         "num_steps": step_count,
#     }


# # =========================================================
# # Main
# # =========================================================
# def main():
#     device = "cpu"
#     torch.manual_seed(42)
#     np.random.seed(42)

#     # -------------------------------------------------
#     # Env config
#     # -------------------------------------------------
#     env = MultiUavMecEnv(
#         M=3,
#         K=8,
#         episode_length=5,
#         cpu_mode="kkt",
#         omega1=100.0,
#         omega2=1.0,
#         deadline_scale=5.0,
#         seed=42,
#     )

#     obs = env.reset(seed=42)
#     state = obs["raw_state"]

#     obs_dim = get_observation_dim(state)
#     actor_raw_action_dim = get_actor_raw_action_dim(state)
#     critic_action_dim = get_full_action_dim(state)

#     print("obs_dim:", obs_dim)
#     print("actor_raw_action_dim:", actor_raw_action_dim)
#     print("critic_action_dim:", critic_action_dim)

#     # -------------------------------------------------
#     # Full proposed policy
#     # -------------------------------------------------
#     actor = MLPActor(
#         obs_dim=obs_dim,
#         action_dim=actor_raw_action_dim,
#         hidden_dim=256,
#     ).to(device)

#     policy = build_default_proposed_policy(
#         state=state,
#         actor_net=actor,
#         device=device,
#         embed_dim=128,
#         num_heads=4,
#         ff_hidden_dim=256,
#         num_layers=2,
#     )

#     target_policy = copy.deepcopy(policy)

#     # -------------------------------------------------
#     # Critic for full-action skeleton
#     # -------------------------------------------------
#     critic = MLPCritic(
#         obs_dim=obs_dim,
#         action_dim=critic_action_dim,
#         hidden_dim=256,
#     ).to(device)

#     target_critic = copy.deepcopy(critic).to(device)

#     # -------------------------------------------------
#     # Optimizers
#     # -------------------------------------------------
#     learnable_modules = []
#     if policy.actor_net is not None:
#         learnable_modules += list(policy.actor_net.parameters())
#     if policy.encoder is not None:
#         learnable_modules += list(policy.encoder.parameters())
#     if policy.fusion_net is not None:
#         learnable_modules += list(policy.fusion_net.parameters())
#     if policy.ratio_head is not None:
#         learnable_modules += list(policy.ratio_head.parameters())

#     actor_opt = optim.Adam(learnable_modules, lr=1e-3)
#     critic_opt = optim.Adam(critic.parameters(), lr=1e-3)

#     mse_loss = nn.MSELoss()

#     # -------------------------------------------------
#     # Replay buffer
#     # -------------------------------------------------
#     buffer = FullReplayBuffer(
#         obs_dim=obs_dim,
#         action_dim=critic_action_dim,
#         capacity=10000,
#     )

#     # -------------------------------------------------
#     # Hyperparameters
#     # -------------------------------------------------
#     gamma = 0.99
#     tau = 0.005
#     batch_size = 16
#     num_episodes = 10

#     # full-stage1 = warm-start skeleton:
#     # 1) high-level actor (movement+scheduling) imitates placeholder
#     # 2) ratio branch imitates placeholder offloading ratio
#     # 3) critic learns on executed full action for pipeline verification
#     actor_move_sched_coef = 1.0
#     ratio_bc_coef = 1.0
#     actor_l2_coef = 1e-4

#     # movement+scheduling masks
#     M = int(state["M"])
#     K = int(state["K"])
#     move_sched_mask = np.zeros((actor_raw_action_dim,), dtype=np.float32)
#     move_sched_mask[: M + M] = 1.0
#     move_sched_mask[M + M + K:] = 1.0
#     move_sched_mask_t = torch.tensor(move_sched_mask, dtype=torch.float32, device=device).unsqueeze(0)

#     best_policy_state = copy.deepcopy(policy)
#     best_eval_reward = -float("inf")

#     # -------------------------------------------------
#     # Training loop
#     # -------------------------------------------------
#     for ep in range(num_episodes):
#         obs = env.reset(seed=42 + ep)
#         done = False

#         episode_reward = 0.0
#         episode_move_sched_loss = 0.0
#         episode_ratio_loss = 0.0
#         step_count = 0

#         while not done:
#             state = obs["raw_state"]
#             access_assoc = build_access_association(state)
#             obs_vec = build_global_observation(state)

#             # -----------------------------------------
#             # Teacher action from placeholder
#             # -----------------------------------------
#             teacher_action = generate_proposed_placeholder_action(
#                 state=state,
#                 access_assoc=access_assoc,
#             )
#             teacher_raw_target = build_teacher_raw_target(
#                 state=state,
#                 access_assoc=access_assoc,
#                 teacher_action=teacher_action,
#             )
#             # teacher_offload_ratio = np.asarray(
#             #     teacher_action["offload_ratio"],
#             #     dtype=np.float32,
#             # )
#             teacher_offload_ratio_refined, _ = policy._refine_high_level_action(
#                 state=state,
#                 access_assoc=access_assoc,
#                 offload_ratio=teacher_offload_ratio,
#                 sched_beta=np.asarray(teacher_action["sched_beta"], dtype=np.float32),
#             )
#             teacher_offload_ratio_refined_t = torch.tensor(
#                 teacher_offload_ratio_refined,
#                 dtype=torch.float32,
#                 device=device,
# )

#             # -----------------------------------------
#             # Current full proposed action
#             # -----------------------------------------
#             action = policy.act(
#                 state=state,
#                 access_assoc=access_assoc,
#                 deterministic=True,
#                 return_aux=False,
#             )
#             flat_action = flatten_full_action(action)

#             # -----------------------------------------
#             # Environment step
#             # -----------------------------------------
#             next_obs, reward, done, info = env.step(action)
#             next_obs_vec = build_global_observation(next_obs["raw_state"])

#             # next action from target policy for TD target
#             if not done:
#                 next_state = next_obs["raw_state"]
#                 next_access_assoc = build_access_association(next_state)
#                 next_action_target = target_policy.act(
#                     state=next_state,
#                     access_assoc=next_access_assoc,
#                     deterministic=True,
#                     return_aux=False,
#                 )
#                 next_flat_action = flatten_full_action(next_action_target)
#             else:
#                 next_flat_action = np.zeros((critic_action_dim,), dtype=np.float32)

#             buffer.add(
#                 obs=obs_vec,
#                 action=flat_action,
#                 reward=reward,
#                 next_obs=next_obs_vec,
#                 next_action=next_flat_action,
#                 done=float(done),
#             )

#             # -----------------------------------------
#             # Actor / encoder / ratio-head BC update
#             # -----------------------------------------
#             obs_tensor = torch.tensor(obs_vec, dtype=torch.float32, device=device).unsqueeze(0)
#             teacher_raw_target_t = torch.tensor(teacher_raw_target, dtype=torch.float32, device=device).unsqueeze(0)
#             teacher_offload_ratio_t = torch.tensor(teacher_offload_ratio, dtype=torch.float32, device=device)

#             raw_pred = policy.actor_net(obs_tensor)  # [1, raw_dim]
#             move_sched_pred = raw_pred * move_sched_mask_t
#             move_sched_target = teacher_raw_target_t * move_sched_mask_t
#             actor_move_sched_loss = mse_loss(move_sched_pred, move_sched_target)

#             ratio_pred = forward_ratio_branch(
#                 policy=policy,
#                 state=state,
#                 access_assoc=access_assoc,
#             )  # [K]
#             # ratio_bc_loss = mse_loss(ratio_pred, teacher_offload_ratio_t)
#             ratio_bc_loss = mse_loss(ratio_pred, teacher_offload_ratio_refined_t)
            
#             actor_l2_loss = (raw_pred ** 2).mean()
#             total_actor_loss = (
#                 actor_move_sched_coef * actor_move_sched_loss
#                 + ratio_bc_coef * ratio_bc_loss
#                 + actor_l2_coef * actor_l2_loss
#             )

#             actor_opt.zero_grad()
#             total_actor_loss.backward()
#             actor_opt.step()

#             # -----------------------------------------
#             # Critic skeleton update
#             # -----------------------------------------
#             if len(buffer) >= batch_size:
#                 batch = buffer.sample(batch_size)

#                 obs_b = torch.tensor(batch["obs"], dtype=torch.float32, device=device)
#                 action_b = torch.tensor(batch["action"], dtype=torch.float32, device=device)
#                 reward_b = torch.tensor(batch["reward"], dtype=torch.float32, device=device)
#                 next_obs_b = torch.tensor(batch["next_obs"], dtype=torch.float32, device=device)
#                 next_action_b = torch.tensor(batch["next_action"], dtype=torch.float32, device=device)
#                 done_b = torch.tensor(batch["done"], dtype=torch.float32, device=device)

#                 with torch.no_grad():
#                     target_q = target_critic(next_obs_b, next_action_b)
#                     y = reward_b + gamma * (1.0 - done_b) * target_q

#                 q_val = critic(obs_b, action_b)
#                 critic_loss = mse_loss(q_val, y)

#                 critic_opt.zero_grad()
#                 critic_loss.backward()
#                 critic_opt.step()

#                 # soft-update targets
#                 soft_update_policy(target_policy, policy, tau=tau)
#                 soft_update(target_critic, critic, tau=tau)

#             obs = next_obs
#             episode_reward += reward
#             episode_move_sched_loss += float(actor_move_sched_loss.item())
#             episode_ratio_loss += float(ratio_bc_loss.item())
#             step_count += 1

#         print(f"\nEpisode {ep}")
#         print("episode_reward:", episode_reward)
#         print("avg_move_sched_loss:", episode_move_sched_loss / max(step_count, 1))
#         print("avg_ratio_loss:", episode_ratio_loss / max(step_count, 1))
#         print("steps:", step_count)
#         print("buffer_size:", len(buffer))

#         # -----------------------------------------
#         # Evaluation
#         # -----------------------------------------
#         eval_env = MultiUavMecEnv(
#             M=3,
#             K=8,
#             episode_length=5,
#             cpu_mode="kkt",
#             omega1=100.0,
#             omega2=1.0,
#             deadline_scale=5.0,
#             seed=999,
#         )

#         eval_result = evaluate_full_policy_rollout(
#             env=eval_env,
#             policy=policy,
#             seed=999,
#         )

#         print("\n==============================")
#         print(f"Full Proposed Stage-1 Eval @ episode {ep}")
#         print("==============================")
#         print("episode_reward:", eval_result["episode_reward"])
#         print("avg_delay:", eval_result["avg_delay"])
#         print("avg_energy:", eval_result["avg_energy"])
#         print("avg_deadline_violation:", eval_result["avg_deadline_violation"])
#         print("feasible_ratio:", eval_result["feasible_ratio"])
#         print("num_steps:", eval_result["num_steps"])

#         if eval_result["episode_reward"] > best_eval_reward:
#             best_eval_reward = eval_result["episode_reward"]
#             best_policy_state = copy.deepcopy(policy)

#     # -------------------------------------------------
#     # Save checkpoints
#     # -------------------------------------------------
#     import os

#     os.makedirs("checkpoints", exist_ok=True)

#     # save full best policy modules separately
#     if best_policy_state.actor_net is not None:
#         torch.save(
#             best_policy_state.actor_net.state_dict(),
#             "checkpoints/proposed_full_stage1_best_actor.pth",
#         )
#     if best_policy_state.encoder is not None:
#         torch.save(
#             best_policy_state.encoder.state_dict(),
#             "checkpoints/proposed_full_stage1_best_encoder.pth",
#         )
#     if best_policy_state.fusion_net is not None:
#         torch.save(
#             best_policy_state.fusion_net.state_dict(),
#             "checkpoints/proposed_full_stage1_best_fusion.pth",
#         )
#     if best_policy_state.ratio_head is not None:
#         torch.save(
#             best_policy_state.ratio_head.state_dict(),
#             "checkpoints/proposed_full_stage1_best_ratio_head.pth",
#         )

#     torch.save(
#         critic.state_dict(),
#         "checkpoints/proposed_full_stage1_critic.pth",
#     )

#     print("\nFull proposed Stage-1 training skeleton finished successfully.")
#     print("Saved:")
#     print("  checkpoints/proposed_full_stage1_best_actor.pth")
#     print("  checkpoints/proposed_full_stage1_best_encoder.pth")
#     print("  checkpoints/proposed_full_stage1_best_fusion.pth")
#     print("  checkpoints/proposed_full_stage1_best_ratio_head.pth")
#     print("  checkpoints/proposed_full_stage1_critic.pth")


# if __name__ == "__main__":
#     main()