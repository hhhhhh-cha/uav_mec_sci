from typing import Dict, Any

import numpy as np
from env.task_model import compute_delay_and_energy_from_action

EPS = 1e-8


def check_feasibility(
    state: Dict[str, Any],
    access_assoc: np.ndarray,
    offload_ratio: np.ndarray,
    sched_beta: np.ndarray,
    bw_alloc: np.ndarray,
    cpu_alloc: np.ndarray,
    metrics: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Global feasibility check for the minimal low-level validation stage.

    Checks:
    1) offloading ratio in [0, 1]
    2) each task is associated with exactly one access UAV
    3) schedule is one-hot under the associated access UAV
    4) chosen execution node belongs to candidate execution set
    5) bandwidth budget constraints
    6) cpu budget constraints
    7) uplink minimum-rate constraint
    8) deadline constraint
    9) numeric stability
    """
    M, K = access_assoc.shape
    B_max = state["B_max"]
    uav_cpu_max = state["uav_cpu_max"]
    neighbors = state["neighbors"]
    R_min = float(state["R_min"])
    rate_up_raw = state["rate_up"]
    task_deadline = state["task_deadline"]
    delay_total = metrics["delay_total"]

    report = {
        "ok": True,
        "ratio_violation": 0.0,
        "assoc_violation": 0.0,
        "schedule_violation": 0.0,
        "candidate_violation": 0.0,
        "bw_violation": 0.0,
        "cpu_violation": 0.0,
        "rate_violation": 0.0,
        "deadline_violation": 0.0,
        "nan_count": 0,
    }

    # -------------------------------------------------
    # 1) offloading ratio in [0, 1]
    # -------------------------------------------------
    ratio_low = float(np.min(offload_ratio))
    ratio_high = float(np.max(offload_ratio))
    if ratio_low < -1e-8 or ratio_high > 1.0 + 1e-8:
        report["ratio_violation"] = max(-ratio_low, ratio_high - 1.0)
        report["ok"] = False

    # -------------------------------------------------
    # 2) each task is associated with exactly one UAV
    # -------------------------------------------------
    col_sum = np.sum(access_assoc, axis=0)
    assoc_violation = float(np.max(np.abs(col_sum - 1.0)))
    report["assoc_violation"] = assoc_violation
    if assoc_violation > 1e-8:
        report["ok"] = False

    # -------------------------------------------------
    # 3) one-hot schedule under associated access UAV
    # 4) candidate legality
    # -------------------------------------------------
    for k in range(K):
        access_m = int(np.argmax(access_assoc[:, k]))
        s = float(np.sum(sched_beta[k, access_m, :]))

        if abs(s - 1.0) > 1e-8:
            report["schedule_violation"] = max(report["schedule_violation"], abs(s - 1.0))
            report["ok"] = False

        chosen_js = np.where(sched_beta[k, access_m, :] > 0.5)[0]
        if len(chosen_js) != 1:
            report["candidate_violation"] = max(report["candidate_violation"], 1.0)
            report["ok"] = False
            continue

        j = int(chosen_js[0])
        cand_set = [access_m] + neighbors[access_m]
        if j not in cand_set:
            report["candidate_violation"] = max(report["candidate_violation"], 1.0)
            report["ok"] = False

    # -------------------------------------------------
    # 5) bandwidth budget
    # -------------------------------------------------
    for m in range(M):
        violation = max(0.0, float(np.sum(bw_alloc[m]) - B_max[m]))
        report["bw_violation"] = max(report["bw_violation"], violation)
        if violation > 1e-6:
            report["ok"] = False

    # -------------------------------------------------
    # 6) cpu budget
    # -------------------------------------------------
    for j in range(M):
        violation = max(0.0, float(np.sum(cpu_alloc[j]) - uav_cpu_max[j]))
        report["cpu_violation"] = max(report["cpu_violation"], violation)
        if violation > 1e-6:
            report["ok"] = False

    # -------------------------------------------------
    # 7) uplink minimum-rate constraint
    # Eq. (42h): R_{m,k}^{up,t} >= R_min * x_{m,k}^t
    #
    # In current code:
    # R_{m,k}^{up,t} = B_{m,k}^t * log2(1 + gamma_{m,k}^{up,t})
    # and rate_up_raw stores log2(1 + gamma)
    # -------------------------------------------------
    for k in range(K):
        access_m = int(np.argmax(access_assoc[:, k]))
        actual_rate = float(bw_alloc[access_m, k]) * float(rate_up_raw[access_m, k])

        # only active access links with positive offloading really need positive uplink service
        if offload_ratio[k] > EPS:
            violation = max(0.0, R_min - actual_rate)
            report["rate_violation"] = max(report["rate_violation"], violation)
            if violation > 1e-6:
                report["ok"] = False

    # -------------------------------------------------
    # 8) deadline constraint
    # Eq. (42i): T_k^t <= tau_k^max
    # -------------------------------------------------
    for k in range(K):
        violation = max(0.0, float(delay_total[k]) - float(task_deadline[k]))
        report["deadline_violation"] = max(report["deadline_violation"], violation)
        if violation > 1e-6:
            report["ok"] = False

    # -------------------------------------------------
    # 9) numeric stability
    # -------------------------------------------------
    arrays_to_check = [
        access_assoc,
        offload_ratio,
        sched_beta,
        bw_alloc,
        cpu_alloc,
        delay_total,
    ]
    nan_count = 0
    for arr in arrays_to_check:
        nan_count += int(np.isnan(arr).sum())
        nan_count += int(np.isinf(arr).sum())
    report["nan_count"] = nan_count
    if nan_count > 0:
        report["ok"] = False

    return report


def compute_delay_and_energy(
    state: Dict[str, Any],
    access_assoc: np.ndarray,
    offload_ratio: np.ndarray,
    sched_beta: np.ndarray,
    bw_alloc: np.ndarray,
    cpu_alloc: np.ndarray,
) -> Dict[str, Any]:
    """
    Wrapper that delegates delay and energy computation to env.task_model,
    keeping backward compatibility with the current codebase.
    """
    return compute_delay_and_energy_from_action(
        state=state,
        access_assoc=access_assoc,
        offload_ratio=offload_ratio,
        sched_beta=sched_beta,
        bw_alloc=bw_alloc,
        cpu_alloc=cpu_alloc,
    )

# 正式版本
# def compute_delay_and_energy(
#     state: Dict[str, Any],
#     access_assoc: np.ndarray,
#     offload_ratio: np.ndarray,
#     sched_beta: np.ndarray,
#     bw_alloc: np.ndarray,
#     cpu_alloc: np.ndarray,
# ) -> Dict[str, Any]:
#     M, K = access_assoc.shape

#     task_size = state["task_size"]
#     task_cycles = state["task_cycles"]
#     task_local_cpu = state["task_local_cpu"]
#     rate_up_raw = state["rate_up"]
#     rate_a2a = state["rate_a2a"]
#     kappa_vec = state["kappa_vec"]
#     uav_tx_power = state["uav_tx_power"]

#     delay_local = np.zeros((K,), dtype=float)
#     delay_up = np.zeros((K,), dtype=float)
#     delay_bh = np.zeros((K,), dtype=float)
#     delay_exec = np.zeros((K,), dtype=float)
#     delay_edge = np.zeros((K,), dtype=float)
#     delay_total = np.zeros((K,), dtype=float)

#     energy_tx = np.zeros((M,), dtype=float)
#     energy_cmp = np.zeros((M,), dtype=float)

#     delay_bh_tensor = np.zeros((K, M, M), dtype=float)
#     delay_exec_tensor = np.zeros((K, M, M), dtype=float)

#     cmp_energy_scale = 1e-4

#     for k in range(K):
#         Dk = float(task_size[k])
#         Ck = float(task_cycles[k])
#         lam = float(offload_ratio[k])

#         access_m = int(np.argmax(access_assoc[:, k]))

#         local_cpu = max(float(task_local_cpu[k]), EPS)
#         delay_local[k] = (1.0 - lam) * Dk * Ck / local_cpu

#         if lam > EPS:
#             r_up = max(float(bw_alloc[access_m, k]) * float(rate_up_raw[access_m, k]), EPS)
#             delay_up[k] = lam * Dk / r_up
#         else:
#             delay_up[k] = 0.0

#         chosen_js = np.where(sched_beta[k, access_m, :] > 0.5)[0]

#         if len(chosen_js) != 1:
#             delay_edge[k] = delay_up[k]
#             delay_total[k] = max(delay_local[k], delay_edge[k])
#             continue

#         j_star = int(chosen_js[0])

#         if lam > EPS:
#             if j_star != access_m:
#                 r_bh = max(float(rate_a2a[access_m, j_star]), EPS)
#                 delay_bh[k] = lam * Dk / r_bh
#                 delay_bh_tensor[k, access_m, j_star] = delay_bh[k]
#                 energy_tx[access_m] += float(uav_tx_power[access_m]) * delay_bh[k]
#             else:
#                 delay_bh[k] = 0.0

#             f_exec = max(float(cpu_alloc[j_star, k]), EPS)
#             delay_exec[k] = lam * Dk * Ck / f_exec
#             delay_exec_tensor[k, access_m, j_star] = delay_exec[k]

#             energy_cmp[j_star] += (
#                 cmp_energy_scale
#                 * float(kappa_vec[j_star])
#                 * lam * Dk * Ck * (f_exec ** 2)
#             )
#         else:
#             delay_bh[k] = 0.0
#             delay_exec[k] = 0.0

#         delay_edge[k] = delay_up[k] + delay_bh[k] + delay_exec[k]
#         delay_total[k] = max(delay_local[k], delay_edge[k])

#     metrics = {
#         "delay_local": delay_local,
#         "delay_up": delay_up,
#         "delay_bh": delay_bh,
#         "delay_exec": delay_exec,
#         "delay_edge": delay_edge,
#         "delay_total": delay_total,
#         "delay_bh_tensor": delay_bh_tensor,
#         "delay_exec_tensor": delay_exec_tensor,
#         "delay_sys": float(np.sum(delay_total)),
#         "energy_tx": energy_tx,
#         "energy_cmp": energy_cmp,
#         "energy_sys": float(np.sum(energy_tx) + np.sum(energy_cmp)),
#     }

#     return metrics

# 简化测试版本
# def compute_delay_and_energy(
#     state: Dict[str, Any],
#     access_assoc: np.ndarray,
#     offload_ratio: np.ndarray,
#     sched_beta: np.ndarray,
#     bw_alloc: np.ndarray,
#     cpu_alloc: np.ndarray,
# ) -> Dict[str, Any]:
#     """
#     Minimal delay and energy computation for feasibility validation.

#     This is not yet the final full paper-faithful implementation.
#     It is only used to verify that the whole single-slot decision chain runs.

#     Implemented quantities:
#     - local delay
#     - uplink delay
#     - A2A backhaul delay
#     - execution delay
#     - total delay
#     - relay transmission energy
#     - execution computation energy
#     """
#     M, K = access_assoc.shape
#     task_size = state["task_size"]
#     task_cycles = state["task_cycles"]
#     task_local_cpu = state["task_local_cpu"]
#     rate_up_raw = state["rate_up"]
#     rate_a2a = state["rate_a2a"]
#     kappa_vec = state["kappa_vec"]
#     uav_tx_power = state["uav_tx_power"]

#     delay_local = np.zeros((K,), dtype=float)
#     delay_up = np.zeros((K,), dtype=float)
#     delay_bh = np.zeros((K,), dtype=float)
#     delay_exec = np.zeros((K,), dtype=float)
#     delay_edge = np.zeros((K,), dtype=float)
#     delay_total = np.zeros((K,), dtype=float)

#     energy_tx = np.zeros((M,), dtype=float)
#     energy_cmp = np.zeros((M,), dtype=float)

#     for k in range(K):
#         access_m = int(np.argmax(access_assoc[:, k]))
#         chosen_js = np.where(sched_beta[k, access_m, :] > 0.5)[0]
#         if len(chosen_js) != 1:
#             continue
#         j = int(chosen_js[0])

#         lam = float(offload_ratio[k])
#         Dk = float(task_size[k])
#         Ck = float(task_cycles[k])

#         # Local delay: T_loc = (1-lambda) D C / f_loc
#         local_work = (1.0 - lam) * Dk * Ck
#         delay_local[k] = local_work / max(float(task_local_cpu[k]), EPS)

#         if lam > EPS:
#             # Uplink delay: lambda D / (B * log2(1+sinr))
#             up_rate_eff = max(float(bw_alloc[access_m, k]) * float(rate_up_raw[access_m, k]), EPS)
#             delay_up[k] = lam * Dk / up_rate_eff

#             # Backhaul delay if j != access_m
#             if j != access_m:
#                 bh_rate_eff = max(float(rate_a2a[access_m, j]), EPS)
#                 delay_bh[k] = lam * Dk / bh_rate_eff
#                 energy_tx[access_m] += float(uav_tx_power[access_m]) * delay_bh[k]

#             # Execution delay
#             exec_cpu = max(float(cpu_alloc[j, k]), EPS)
#             delay_exec[k] = lam * Dk * Ck / exec_cpu

#             # Computation energy
#             energy_cmp[j] += float(kappa_vec[j]) * lam * Dk * Ck * (exec_cpu ** 2)

#         delay_edge[k] = delay_up[k] + delay_bh[k] + delay_exec[k]
#         delay_total[k] = max(delay_local[k], delay_edge[k])

#     metrics = {
#         "delay_local": delay_local,
#         "delay_up": delay_up,
#         "delay_bh": delay_bh,
#         "delay_exec": delay_exec,
#         "delay_edge": delay_edge,
#         "delay_total": delay_total,
#         "delay_sys": float(np.sum(delay_total)),
#         "energy_tx": energy_tx,
#         "energy_cmp": energy_cmp,
#         "energy_sys": float(np.sum(energy_tx) + np.sum(energy_cmp)),
#     }

#     return metrics