# # 可直接用于主对比实验的纯 MADDPG 基线接口：
# # 不走 Transformer
# # 不走 ratio head / fusion
# # 只用一个 MLP actor 直接输出：
# # move_dist(M)
# # move_angle(M)
# # offload_ratio(K)
# # sched_score(K*M)
# # 接口和你现有环境完全一致


from __future__ import annotations

from typing import Dict, Any, Optional
from pathlib import Path

import numpy as np
import torch

from model.mlp_actor import MLPActor
from model.proposed_obs_builder import build_global_observation

EPS = 1e-8


def decode_offload_ratio_np(
    offload_raw: np.ndarray,
    min_ratio: float = 0.05,
    max_ratio: float = 1.0,
    temperature: float = 2.0,
) -> np.ndarray:
    """
    Map raw actor output to [min_ratio, max_ratio].
    """
    x = np.asarray(offload_raw, dtype=np.float32) / float(temperature)
    ratio01 = 1.0 / (1.0 + np.exp(-x))
    ratio = min_ratio + (max_ratio - min_ratio) * ratio01
    return np.clip(ratio, min_ratio, max_ratio).astype(float)


class PureMADDPGPolicy:
    """
    Pure MADDPG baseline:
    - no Transformer encoder
    - no ratio head / fusion module
    - direct MLP actor -> high-level action

    Interface:
        act(state, access_assoc, deterministic=True) -> high_action dict
    """

    def __init__(
        self,
        obs_dim: int,
        M: int,
        K: int,
        checkpoint_path: Optional[str] = None,
        device: str = "cpu",
        hidden_dim: int = 256,
    ):
        self.obs_dim = int(obs_dim)
        self.M = int(M)
        self.K = int(K)
        self.device = torch.device(device)

        # [move_dist(M), move_angle(M), offload_ratio(K), sched_score(K*M)]
        self.action_dim = self.M + self.M + self.K + self.K * self.M

        self.actor = MLPActor(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            hidden_dim=hidden_dim,
        ).to(self.device)

        self.loaded = False
        if checkpoint_path is not None:
            ckpt = Path(checkpoint_path)
            if ckpt.exists():
                obj = torch.load(str(ckpt), map_location=self.device)
                if isinstance(obj, dict) and "state_dict" in obj:
                    self.actor.load_state_dict(obj["state_dict"])
                elif isinstance(obj, dict):
                    self.actor.load_state_dict(obj)
                else:
                    raise ValueError(f"Unsupported checkpoint format: {type(obj)}")
                self.loaded = True

        self.actor.eval()

    def act(
        self,
        state: Dict[str, Any],
        access_assoc: np.ndarray,
        deterministic: bool = True,
    ) -> Dict[str, np.ndarray]:
        obs = build_global_observation(state)
        raw_action = self._forward_actor(obs)
        return self._decode_raw_action(state, access_assoc, raw_action)

    def _forward_actor(self, obs: np.ndarray) -> np.ndarray:
        obs_tensor = torch.tensor(
            obs,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        with torch.no_grad():
            raw = self.actor(obs_tensor)

        raw = raw.squeeze(0).detach().cpu().numpy().astype(np.float32)
        return raw

    def _decode_raw_action(
        self,
        state: Dict[str, Any],
        access_assoc: np.ndarray,
        raw_action: np.ndarray,
    ) -> Dict[str, np.ndarray]:
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

        # 1) movement distance
        move_dist_raw = raw_action[idx: idx + M]
        idx += M
        move_dist = 0.5 * (np.tanh(move_dist_raw) + 1.0) * max_move

        # 2) movement angle
        move_angle_raw = raw_action[idx: idx + M]
        idx += M
        move_angle = np.pi * np.tanh(move_angle_raw)

        # 3) offloading ratio
        offload_raw = raw_action[idx: idx + K]
        idx += K
        offload_ratio = decode_offload_ratio_np(
            offload_raw,
            min_ratio=0.05,
            max_ratio=1.0,
            temperature=2.0,
        )
        offload_ratio = np.clip(offload_ratio, 0.05, 1.0)

        # 4) scheduling score -> legal candidate selection
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

# from __future__ import annotations

# from typing import Dict, Any, Optional
# from pathlib import Path

# import numpy as np
# import torch

# from model.mlp_actor import MLPActor
# from model.proposed_obs_builder import build_global_observation

# EPS = 1e-8


# class PureMADDPGPolicy:
#     """
#     Pure MADDPG baseline:
#     - no Transformer encoder
#     - no ratio head / fusion module
#     - direct MLP actor -> high-level action

#     Interface:
#         act(state, access_assoc, deterministic=True) -> high_action dict
#     """

#     def __init__(
#         self,
#         obs_dim: int,
#         M: int,
#         K: int,
#         checkpoint_path: Optional[str] = None,
#         device: str = "cpu",
#         hidden_dim: int = 256,
#     ):
#         self.obs_dim = int(obs_dim)
#         self.M = int(M)
#         self.K = int(K)
#         self.device = torch.device(device)

#         self.action_dim = self.M + self.M + self.K + self.K * self.M

#         self.actor = MLPActor(
#             obs_dim=self.obs_dim,
#             action_dim=self.action_dim,
#             hidden_dim=hidden_dim,
#         ).to(self.device)

#         self.loaded = False
#         if checkpoint_path is not None:
#             ckpt = Path(checkpoint_path)
#             if ckpt.exists():
#                 obj = torch.load(str(ckpt), map_location=self.device)
#                 if isinstance(obj, dict) and "state_dict" in obj:
#                     self.actor.load_state_dict(obj["state_dict"])
#                 elif isinstance(obj, dict):
#                     self.actor.load_state_dict(obj)
#                 else:
#                     raise ValueError(f"Unsupported checkpoint format: {type(obj)}")
#                 self.loaded = True

#         self.actor.eval()

#     def act(
#         self,
#         state: Dict[str, Any],
#         access_assoc: np.ndarray,
#         deterministic: bool = True,
#     ) -> Dict[str, np.ndarray]:
#         obs = build_global_observation(state)
#         raw_action = self._forward_actor(obs)
#         return self._decode_raw_action(state, access_assoc, raw_action)

#     def _forward_actor(self, obs: np.ndarray) -> np.ndarray:
#         obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
#         with torch.no_grad():
#             raw = self.actor(obs_tensor)
#         raw = raw.squeeze(0).detach().cpu().numpy().astype(np.float32)
#         return raw

#     def _decode_raw_action(
#         self,
#         state: Dict[str, Any],
#         access_assoc: np.ndarray,
#         raw_action: np.ndarray,
#     ) -> Dict[str, np.ndarray]:
#         M = int(state["M"])
#         K = int(state["K"])
#         neighbors = state["neighbors"]
#         max_speed = float(state["max_speed"])
#         delta_t = float(state["delta_t"])
#         max_move = max_speed * delta_t

#         expected_dim = M + M + K + K * M
#         if raw_action.shape[0] != expected_dim:
#             raise ValueError(
#                 f"Raw action dim mismatch: got {raw_action.shape[0]}, expected {expected_dim}"
#             )

#         idx = 0

#         move_dist_raw = raw_action[idx: idx + M]
#         idx += M
#         move_dist = 0.5 * (np.tanh(move_dist_raw) + 1.0) * max_move

#         move_angle_raw = raw_action[idx: idx + M]
#         idx += M
#         move_angle = np.pi * np.tanh(move_angle_raw)

#         offload_raw = raw_action[idx: idx + K]
#         idx += K
#         # offload_ratio = 0.5 * (np.tanh(offload_raw) + 1.0)
#         offload_ratio = decode_offload_ratio_np(
#             offload_raw,
#             min_ratio=0.05,
#             max_ratio=1.0,
#             temperature=2.0,
#         )
    
#     def decode_offload_ratio_np(
#         offload_raw: np.ndarray,
#         min_ratio: float = 0.05,
#         max_ratio: float = 1.0,
#         temperature: float = 2.0,
#     ) -> np.ndarray:
#         x = np.asarray(offload_raw, dtype=np.float32) / float(temperature)
#         ratio01 = 1.0 / (1.0 + np.exp(-x))
#         ratio = min_ratio + (max_ratio - min_ratio) * ratio01
#         return np.clip(ratio, min_ratio, max_ratio).astype(float)

#         offload_ratio = np.clip(offload_ratio, 0.05, 1.0)

#         sched_score = raw_action[idx: idx + K * M].reshape(K, M)
#         sched_beta = np.zeros((K, M, M), dtype=float)

#         for k in range(K):
#             access_m = int(np.argmax(access_assoc[:, k]))
#             legal_js = [access_m] + list(neighbors[access_m])

#             legal_scores = [float(sched_score[k, j]) for j in legal_js]
#             best_local_idx = int(np.argmax(legal_scores))
#             best_j = int(legal_js[best_local_idx])

#             sched_beta[k, access_m, best_j] = 1.0

#         return {
#             "move_dist": move_dist.astype(float),
#             "move_angle": move_angle.astype(float),
#             "offload_ratio": offload_ratio.astype(float),
#             "sched_beta": sched_beta.astype(float),
#         }