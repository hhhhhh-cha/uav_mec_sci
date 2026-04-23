from typing import Dict, Any

import numpy as np
# 对每个接入UAV m，再关联任务集合Kmt上，求解一个严格凸问题，最终得到闭式最优解B

EPS = 1e-8


def solve_bandwidth_allocation(
    state: Dict[str, Any],
    access_assoc: np.ndarray,
    offload_ratio: np.ndarray,
) -> np.ndarray:
    """
    Solve the uplink bandwidth allocation subproblem for all access UAVs.

    Paper correspondence:
        a_{m,k}^t = lambda_k^t D_k^t / log2(1 + gamma_{m,k}^{up,t})
        B_{m,k}^{t,*} = sqrt(a_{m,k}^t) / sum_{k'} sqrt(a_{m,k'}^t) * B_m^max

    Args:
        state:
            must contain
                task_size: [K]
                sinr_up:   [M, K]
                B_max:     [M]
        access_assoc: [M, K], binary
        offload_ratio:[K], in [0,1]

    Returns:
        bw_alloc: [M, K]
    """
    task_size = state["task_size"]   # D_k
    sinr_up = state["sinr_up"]       # gamma_{m,k}^{up}
    B_max = state["B_max"]           # B_m^max

    M, K = access_assoc.shape
    bw_alloc = np.zeros((M, K), dtype=float)

    for m in range(M):
        # tasks associated with UAV m
        task_idx = np.where(access_assoc[m] > 0.5)[0].tolist()

        # only tasks with positive offloading need uplink bandwidth
        active_tasks = [k for k in task_idx if offload_ratio[k] > EPS]

        if len(active_tasks) == 0:
            continue

        a_vals = []
        for k in active_tasks:
            denom = np.log2(1.0 + max(float(sinr_up[m, k]), EPS))
            denom = max(denom, EPS)

            a_mk = offload_ratio[k] * task_size[k] / denom
            a_vals.append(max(float(a_mk), EPS))

        a_vals = np.array(a_vals, dtype=float)
        sqrt_a = np.sqrt(a_vals)
        denom_sum = float(np.sum(sqrt_a))

        # closed-form allocation
        if denom_sum < EPS:
            share = float(B_max[m]) / max(len(active_tasks), 1)
            for k in active_tasks:
                bw_alloc[m, k] = share
        else:
            for idx, k in enumerate(active_tasks):
                bw_alloc[m, k] = float(B_max[m]) * sqrt_a[idx] / denom_sum

    return bw_alloc


def compute_bandwidth_problem_coefficients(
    state: Dict[str, Any],
    access_assoc: np.ndarray,
    offload_ratio: np.ndarray,
) -> np.ndarray:
    """
    Compute a_{m,k}^t in the paper:
        a_{m,k}^t = lambda_k^t D_k^t / log2(1 + gamma_{m,k}^{up,t})

    Returns:
        a_mk: [M, K]
    """
    task_size = state["task_size"]
    sinr_up = state["sinr_up"]

    M, K = access_assoc.shape
    a_mk = np.zeros((M, K), dtype=float)

    for m in range(M):
        for k in range(K):
            if access_assoc[m, k] <= 0.5:
                continue
            if offload_ratio[k] <= EPS:
                continue

            denom = np.log2(1.0 + max(float(sinr_up[m, k]), EPS))
            denom = max(denom, EPS)
            a_mk[m, k] = offload_ratio[k] * task_size[k] / denom

    return a_mk


def validate_bandwidth_allocation(
    state: Dict[str, Any],
    access_assoc: np.ndarray,
    offload_ratio: np.ndarray,
    bw_alloc: np.ndarray,
) -> Dict[str, Any]:
    """
    Validate:
    1) nonnegativity
    2) sum_k B_{m,k} <= B_m^max
    3) no positive bandwidth is assigned to non-associated tasks
    """
    B_max = state["B_max"]
    M, K = access_assoc.shape

    neg_violation = 0.0
    budget_violation = 0.0
    support_violation = 0.0
    ok = True

    # 1) nonnegativity
    min_bw = float(np.min(bw_alloc))
    if min_bw < -1e-8:
        neg_violation = -min_bw
        ok = False

    # 2) per-UAV bandwidth budget
    for m in range(M):
        violation = max(0.0, float(np.sum(bw_alloc[m]) - B_max[m]))
        budget_violation = max(budget_violation, violation)
        if violation > 1e-6:
            ok = False

    # 3) support check
    for m in range(M):
        for k in range(K):
            # if task k is not associated with m, no positive bandwidth should be allocated
            if access_assoc[m, k] <= 0.5 and bw_alloc[m, k] > 1e-8:
                support_violation = max(support_violation, float(bw_alloc[m, k]))
                ok = False

            # if lambda is zero, bandwidth should ideally be zero
            if offload_ratio[k] <= EPS and bw_alloc[m, k] > 1e-8:
                support_violation = max(support_violation, float(bw_alloc[m, k]))
                ok = False

    return {
        "ok": ok,
        "neg_violation": float(neg_violation),
        "budget_violation": float(budget_violation),
        "support_violation": float(support_violation),
    }