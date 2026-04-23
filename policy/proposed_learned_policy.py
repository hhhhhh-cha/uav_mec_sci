# 这个文件的作用是：
# 先把“可训练版策略”的接口搭起来
# 当前可以先支持两种模式：
# mode="placeholder"：继续调用你现在稳定的 placeholder
# mode="network"：以后接神经网络输出
# 这样你的主方法接口就彻底统一了。

import numpy as np

from model.proposed_obs_builder import build_global_observation
from policy.proposed_placeholder_policy import generate_proposed_placeholder_action

EPS = 1e-8


class ProposedLearnedPolicy:
    """
    Unified interface for the proposed method.

    Current supported modes:
    - placeholder: use the stable rule-based proposed placeholder
    - network:     reserved for the trainable neural version

    This lets you keep one policy interface while incrementally upgrading
    from placeholder -> learned policy.
    """

    def __init__(self, mode="placeholder", actor_net=None, device="cpu"):
        self.mode = str(mode)
        self.actor_net = actor_net
        self.device = device

    def act(self, state, access_assoc, deterministic=True):
        if self.mode == "placeholder":
            return generate_proposed_placeholder_action(state, access_assoc)

        if self.mode == "network":
            if self.actor_net is None:
                raise RuntimeError("actor_net is None while mode='network'.")

            obs = build_global_observation(state)
            raw_action = self._forward_actor(obs, deterministic=deterministic)
            action = self._decode_raw_action(
                state=state,
                access_assoc=access_assoc,
                raw_action=raw_action,
            )
            return action

        raise ValueError(f"Unknown policy mode: {self.mode}")

    def _forward_actor(self, obs, deterministic=True):
        """
        Forward actor network and return raw action vector.

        Expected actor output layout:
        [ move_dist(M), move_angle(M), offload_ratio(K), sched_score(K*M) ]

        For now we assume actor_net(obs_batch) -> raw_action_batch
        and use batch size = 1.
        """
        import torch

        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            raw = self.actor_net(obs_tensor)

        raw = raw.squeeze(0).detach().cpu().numpy().astype(np.float32)
        return raw

    def _decode_raw_action(self, state, access_assoc, raw_action):
        """
        Decode raw actor output into environment-compatible high-level action.

        Current simplified design:
        - movement is decoded but can later be frozen if desired
        - sched_beta chooses among candidate execution UAVs using per-task scores
        """
        M = int(state["M"])
        K = int(state["K"])
        neighbors = state["neighbors"]
        max_speed = float(state["max_speed"])
        delta_t = float(state["delta_t"])
        max_move = max_speed * delta_t

        expected_dim = M + M + K + K * M
        if raw_action.shape[0] != expected_dim:
            raise ValueError(
                f"Raw action dim mismatch: got {raw_action.shape[0]}, expected {expected_dim}"
            )

        idx = 0

        # -------------------------------------------------
        # movement distance in [0, max_move]
        # -------------------------------------------------
        move_dist_raw = raw_action[idx: idx + M]
        idx += M
        move_dist = 0.5 * (np.tanh(move_dist_raw) + 1.0) * max_move

        # debug: temporarily amplify mobility magnitude
        move_dist_scale = 20.0
        move_dist = np.clip(move_dist * move_dist_scale, 0.0, max_move)


        # -------------------------------------------------
        # movement angle in [-pi, pi]
        # -------------------------------------------------
        move_angle_raw = raw_action[idx: idx + M]
        idx += M
        move_angle = np.pi * np.tanh(move_angle_raw)

        # -------------------------------------------------
        # offloading ratio in [0, 1]
        # -------------------------------------------------
        offload_raw = raw_action[idx: idx + K]
        idx += K
        offload_ratio = 0.5 * (np.tanh(offload_raw) + 1.0)

        # -------------------------------------------------
        # scheduling scores -> one-hot over legal candidate set
        # raw layout: [K, M], but only access_m row is used for each task
        # -------------------------------------------------
        sched_score = raw_action[idx: idx + K * M].reshape(K, M)
        sched_beta = np.zeros((K, M, M), dtype=float)

        for k in range(K):
            access_m = int(np.argmax(access_assoc[:, k]))
            legal_js = [access_m] + list(neighbors[access_m])

            legal_scores = [float(sched_score[k, j]) for j in legal_js]
            best_local_idx = int(np.argmax(legal_scores))
            best_j = int(legal_js[best_local_idx])

            sched_beta[k, access_m, best_j] = 1.0

        return {
            "move_dist": move_dist.astype(float),
            "move_angle": move_angle.astype(float),
            "offload_ratio": offload_ratio.astype(float),
            "sched_beta": sched_beta.astype(float),
        }