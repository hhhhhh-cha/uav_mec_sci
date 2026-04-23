# 移动：每架 UAV 朝自己当前关联任务的几何中心移动
# 卸载比：本地 CPU 压力越大，越倾向于多卸载
# 协同调度：优先选 “可用 CPU 大 + A2A 速率高” 的执行 UAV

from typing import Dict, Any, Optional
import numpy as np

from env.association import heuristic_generate_high_action


class GreedyPolicy:
    """
    Strong heuristic baseline aligned with the paper-oriented greedy design.

    This is intentionally mapped to the deadline-aware heuristic in
    env/association.py, instead of using a weaker geometric-only rule.
    """

    def __init__(self, seed: Optional[int] = None):
        self.seed = seed

    def act(
        self,
        state: Dict[str, Any],
        access_assoc: np.ndarray,
        deterministic: bool = True,
    ) -> Dict[str, np.ndarray]:
        return heuristic_generate_high_action(
            state=state,
            access_assoc=access_assoc,
            seed=self.seed,
        )


def generate_greedy_high_action(
    state: Dict[str, Any],
    access_assoc: np.ndarray,
    seed: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """
    Module-level export for evaluation scripts.
    """
    policy = GreedyPolicy(seed=seed)
    return policy.act(
        state=state,
        access_assoc=access_assoc,
        deterministic=True,
    )

# from typing import Dict, Any
# import numpy as np

# EPS = 1e-8


# def generate_greedy_high_action(state: Dict[str, Any], access_assoc: np.ndarray):
#     """
#     Greedy baseline under the same high-level action interface.

#     Heuristic design:
#     1) UAV mobility:
#        move each UAV toward the centroid of its currently associated tasks.
#     2) Offloading ratio:
#        larger local-delay pressure -> larger offloading ratio.
#     3) Scheduling:
#        choose execution UAV by a greedy score combining
#        available CPU and A2A forwarding rate.
#     """
#     M = int(state["M"])
#     K = int(state["K"])

#     uav_pos = np.asarray(state["uav_pos"], dtype=float)
#     td_pos = np.asarray(state["td_pos"], dtype=float)
#     task_size = np.asarray(state["task_size"], dtype=float)
#     task_cycles = np.asarray(state["task_cycles"], dtype=float)
#     task_deadline = np.asarray(state["task_deadline"], dtype=float)
#     task_local_cpu = np.asarray(state["task_local_cpu"], dtype=float)

#     uav_cpu_avail = np.asarray(state["uav_cpu_avail"], dtype=float)
#     rate_a2a = np.asarray(state["rate_a2a"], dtype=float)
#     neighbors = state["neighbors"]

#     max_speed = float(state["max_speed"])
#     delta_t = float(state["delta_t"])
#     max_move = max_speed * delta_t

#     move_dist = np.zeros((M,), dtype=float)
#     move_angle = np.zeros((M,), dtype=float)
#     offload_ratio = np.zeros((K,), dtype=float)
#     sched_beta = np.zeros((K, M, M), dtype=float)

#     # -------------------------------------------------
#     # 1) Mobility: move toward centroid of associated TDs
#     # -------------------------------------------------
#     for m in range(M):
#         task_idx = np.where(access_assoc[m] > 0.5)[0]
#         if len(task_idx) == 0:
#             move_dist[m] = 0.0
#             move_angle[m] = 0.0
#             continue

#         centroid = np.mean(td_pos[task_idx], axis=0)
#         vec = centroid - uav_pos[m]
#         dist = float(np.linalg.norm(vec))

#         if dist <= EPS:
#             move_dist[m] = 0.0
#             move_angle[m] = 0.0
#         else:
#             move_dist[m] = min(0.6 * dist, max_move)
#             move_angle[m] = float(np.arctan2(vec[1], vec[0]))

#     # -------------------------------------------------
#     # 2) Offloading ratio: local execution harder -> more offload
#     # -------------------------------------------------
#     local_delay_proxy = (task_size * task_cycles) / np.maximum(task_local_cpu, EPS)
#     urgency_proxy = local_delay_proxy / np.maximum(task_deadline, EPS)

#     offload_ratio = np.clip(0.25 + 0.55 * urgency_proxy, 0.05, 0.95)

#     # -------------------------------------------------
#     # 3) Scheduling: choose execution UAV greedily
#     # -------------------------------------------------
#     cpu_max = np.max(uav_cpu_avail) if np.max(uav_cpu_avail) > EPS else 1.0

#     for k in range(K):
#         access_m = int(np.argmax(access_assoc[:, k]))
#         legal_js = [access_m] + list(neighbors[access_m])

#         best_score = -1e18
#         best_j = access_m

#         for j in legal_js:
#             cpu_score = float(uav_cpu_avail[j] / cpu_max)

#             if j == access_m:
#                 rate_score = 1.0
#             else:
#                 denom = np.max(rate_a2a[access_m]) if np.max(rate_a2a[access_m]) > EPS else 1.0
#                 rate_score = float(rate_a2a[access_m, j] / denom)

#             score = 0.7 * cpu_score + 0.3 * rate_score

#             if score > best_score:
#                 best_score = score
#                 best_j = j

#         sched_beta[k, access_m, best_j] = 1.0

#     return {
#         "move_dist": move_dist.astype(float),
#         "move_angle": move_angle.astype(float),
#         "offload_ratio": offload_ratio.astype(float),
#         "sched_beta": sched_beta.astype(float),
#     }