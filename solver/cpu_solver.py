from typing import Dict, Any, Tuple

import numpy as np
# 对每个执行 UAV j，在其最终执行任务集合 𝑈𝑗𝑡上，求解严格凸 CPU 分配问题。
# 做法是：
# 固定对偶变量 𝜇𝑗𝑡
# 对每个任务求唯一正根
# 外层二分搜索𝜇𝑗𝑡
# 直到满足总 CPU 约束

EPS = 1e-8

def solve_cpu_allocation_proportional(
    state,
    access_assoc,
    sched_beta,
    offload_ratio,
    rho: float = 0.45,
):
    """
    Proportional CPU allocation with configurable budget ratio rho.

    rho = 1.0  -> use full CPU budget
    rho = 0.5  -> use half CPU budget
    """

    import numpy as np

    EPS = 1e-8
    M, K = access_assoc.shape
    cpu_alloc = np.zeros((M, K), dtype=float)
    uav_cpu_max = state["uav_cpu_max"]
    task_size = state["task_size"]
    task_cycles = state["task_cycles"]

    rho = float(np.clip(rho, 0.0, 1.0))

    for j in range(M):
        exec_tasks = []
        workloads = []

        for k in range(K):
            if offload_ratio[k] <= EPS:
                continue

            access_m = int(np.argmax(access_assoc[:, k]))
            if sched_beta[k, access_m, j] > 0.5:
                W = offload_ratio[k] * task_size[k] * task_cycles[k]
                exec_tasks.append(k)
                workloads.append(float(W))

        if len(exec_tasks) == 0:
            continue

        workloads = np.array(workloads, dtype=float)
        total_W = float(np.sum(workloads))

        # only use rho * f_max_j instead of full budget
        budget_j = rho * float(uav_cpu_max[j])

        if total_W <= EPS:
            share = budget_j / len(exec_tasks)
            for k in exec_tasks:
                cpu_alloc[j, k] = share
        else:
            for idx, k in enumerate(exec_tasks):
                cpu_alloc[j, k] = budget_j * workloads[idx] / total_W

    return cpu_alloc
# def solve_cpu_allocation_proportional(
#     state,
#     access_assoc,
#     sched_beta,
#     offload_ratio,
# ):
#     import numpy as np

#     EPS = 1e-8
#     M, K = access_assoc.shape
#     cpu_alloc = np.zeros((M, K), dtype=float)
#     uav_cpu_max = state["uav_cpu_max"]
#     task_size = state["task_size"]
#     task_cycles = state["task_cycles"]

#     for j in range(M):
#         exec_tasks = []
#         workloads = []

#         for k in range(K):
#             if offload_ratio[k] <= EPS:
#                 continue

#             access_m = int(np.argmax(access_assoc[:, k]))
#             if sched_beta[k, access_m, j] > 0.5:
#                 W = offload_ratio[k] * task_size[k] * task_cycles[k]
#                 exec_tasks.append(k)
#                 workloads.append(float(W))

#         if len(exec_tasks) == 0:
#             continue

#         workloads = np.array(workloads, dtype=float)
#         total_W = float(np.sum(workloads))

#         if total_W <= EPS:
#             share = float(uav_cpu_max[j]) / len(exec_tasks)
#             for k in exec_tasks:
#                 cpu_alloc[j, k] = share
#         else:
#             for idx, k in enumerate(exec_tasks):
#                 cpu_alloc[j, k] = float(uav_cpu_max[j]) * workloads[idx] / total_W

#     return cpu_alloc


def solve_positive_root_for_one_task(
    W: float,
    mu: float,
    omega1: float,
    omega2: float,
    kappa_j: float,
    f_low: float = 1e-8,
    f_high: float = 1e3,
    tol: float = 1e-8,
    max_iter: int = 200,
) -> float:
    """
    Solve the positive root of:
        -omega1 * W / f^2 + 2 * omega2 * kappa_j * W * f + mu = 0,  f > 0

    This corresponds to the stationarity condition in the paper.

    Args:
        W: workload term, W_{k,j}^t = lambda_k^t D_k^t C_k^t
        mu: dual variable
        omega1, omega2: objective weights
        kappa_j: switched-capacitance coefficient
    """
    if W <= 1e-12:
        return 0.0

    def g(f: float) -> float:
        return -omega1 * W / (f * f) + 2.0 * omega2 * kappa_j * W * f + mu

    lo, hi = f_low, f_high

    # expand upper bound until g(hi) >= 0
    while g(hi) < 0.0:
        hi *= 2.0
        if hi > 1e12:
            break

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        val = g(mid)

        if abs(val) < tol:
            return mid

        if val > 0.0:
            hi = mid
        else:
            lo = mid

    return 0.5 * (lo + hi)


def solve_f_given_mu(
    task_workloads: Dict[Tuple[int, int], float],
    mu: float,
    omega1: float,
    omega2: float,
    kappa_j: float,
) -> Tuple[Dict[Tuple[int, int], float], float]:
    """
    Given a fixed dual variable mu, solve all task CPU allocations on one UAV j.

    Args:
        task_workloads:
            dict[(k, access_m)] = W_{k,j}^t
    Returns:
        alloc:
            dict[(k, access_m)] = f_{j,k}^{exe,t}
        total:
            sum of allocated CPU on UAV j
    """
    alloc = {}
    total = 0.0

    for key, W in task_workloads.items():
        f_val = solve_positive_root_for_one_task(
            W=W,
            mu=mu,
            omega1=omega1,
            omega2=omega2,
            kappa_j=kappa_j,
        )
        alloc[key] = f_val
        total += f_val

    return alloc, total


def solve_cpu_allocation_for_one_uav(
    task_workloads: Dict[Tuple[int, int], float],
    f_max_j: float,
    omega1: float,
    omega2: float,
    kappa_j: float,
    mu_low: float = 0.0,
    mu_high: float = 1.0,
    tol: float = 1e-6,
    max_iter: int = 100,
) -> Tuple[Dict[Tuple[int, int], float], float, int]:
    """
    Solve the CPU allocation subproblem for one execution UAV j.

    Paper correspondence:
        min sum [ omega1 * W/f + omega2 * kappa_j * W * f^2 ]
        s.t. sum f <= f_j^max, f > 0

    Returns:
        alloc_j:
            dict[(k, access_m)] = f_{j,k}^{exe,t}
        mu_star:
            dual variable
        iter_count:
            number of outer dual-search iterations
    """
    if len(task_workloads) == 0:
        return {}, 0.0, 0

    iter_count = 0

    # Expand mu_high until total allocation <= capacity
    while True:
        alloc_hi, total_hi = solve_f_given_mu(
            task_workloads=task_workloads,
            mu=mu_high,
            omega1=omega1,
            omega2=omega2,
            kappa_j=kappa_j,
        )
        iter_count += 1

        if total_hi <= f_max_j:
            break

        mu_high *= 2.0
        if mu_high > 1e12:
            break

    # If mu = 0 already feasible, constraint inactive
    alloc_lo, total_lo = solve_f_given_mu(
        task_workloads=task_workloads,
        mu=mu_low,
        omega1=omega1,
        omega2=omega2,
        kappa_j=kappa_j,
    )
    iter_count += 1

    if total_lo <= f_max_j:
        return alloc_lo, 0.0, iter_count

    lo, hi = mu_low, mu_high
    best_alloc = None

    for _ in range(max_iter):
        mu_mid = 0.5 * (lo + hi)

        alloc_mid, total_mid = solve_f_given_mu(
            task_workloads=task_workloads,
            mu=mu_mid,
            omega1=omega1,
            omega2=omega2,
            kappa_j=kappa_j,
        )
        iter_count += 1
        best_alloc = alloc_mid

        if abs(total_mid - f_max_j) < tol:
            return alloc_mid, mu_mid, iter_count

        if total_mid > f_max_j:
            lo = mu_mid
        else:
            hi = mu_mid

    return best_alloc if best_alloc is not None else {}, 0.5 * (lo + hi), iter_count


def solve_cpu_allocation(
    state: Dict[str, Any],
    access_assoc: np.ndarray,
    sched_beta: np.ndarray,
    offload_ratio: np.ndarray,
    omega1: float = 1.0,
    omega2: float = 1.0,
) -> Tuple[np.ndarray, int]:
    """
    Solve CPU allocation for all execution UAVs.

    Args:
        state:
            task_size:   [K]
            task_cycles: [K]
            uav_cpu_max: [M]
            kappa_vec:   [M]
        access_assoc: [M, K]
        sched_beta:   [K, M, M]
        offload_ratio:[K]

    Returns:
        cpu_alloc: [M, K], where cpu_alloc[j, k] = f_{j,k}^{exe,t}
        total_bisect_iters: total number of dual-search iterations
    """
    task_size = state["task_size"]
    task_cycles = state["task_cycles"]
    uav_cpu_max = state["uav_cpu_max"]
    kappa_vec = state["kappa_vec"]

    M, K = access_assoc.shape
    cpu_alloc = np.zeros((M, K), dtype=float)
    total_bisect_iters = 0

    for j in range(M):
        # U_j^t = {(k,m) | x_{m,k} beta_{k,m,j} = 1}
        task_workloads = {}

        for k in range(K):
            if offload_ratio[k] <= EPS:
                continue

            access_m = int(np.argmax(access_assoc[:, k]))

            if sched_beta[k, access_m, j] > 0.5:
                W_kj = offload_ratio[k] * task_size[k] * task_cycles[k]
                task_workloads[(k, access_m)] = float(W_kj)

        alloc_j, mu_j, iter_j = solve_cpu_allocation_for_one_uav(
            task_workloads=task_workloads,
            f_max_j=float(uav_cpu_max[j]),
            omega1=omega1,
            omega2=omega2,
            kappa_j=float(kappa_vec[j]),
        )
        total_bisect_iters += iter_j

        for (k, access_m), f_val in alloc_j.items():
            cpu_alloc[j, k] = float(f_val)

    return cpu_alloc, total_bisect_iters


def compute_cpu_problem_workloads(
    state: Dict[str, Any],
    access_assoc: np.ndarray,
    sched_beta: np.ndarray,
    offload_ratio: np.ndarray,
) -> Dict[int, Dict[Tuple[int, int], float]]:
    """
    Compute W_{k,j}^t = lambda_k^t D_k^t C_k^t for each execution UAV j.

    Returns:
        workloads_by_uav[j][(k, access_m)] = W_{k,j}^t
    """
    task_size = state["task_size"]
    task_cycles = state["task_cycles"]

    M, K = access_assoc.shape
    workloads_by_uav = {}

    for j in range(M):
        workloads_by_uav[j] = {}

        for k in range(K):
            if offload_ratio[k] <= EPS:
                continue

            access_m = int(np.argmax(access_assoc[:, k]))
            if sched_beta[k, access_m, j] > 0.5:
                W_kj = offload_ratio[k] * task_size[k] * task_cycles[k]
                workloads_by_uav[j][(k, access_m)] = float(W_kj)

    return workloads_by_uav


def validate_cpu_allocation(
    state: Dict[str, Any],
    access_assoc: np.ndarray,
    sched_beta: np.ndarray,
    offload_ratio: np.ndarray,
    cpu_alloc: np.ndarray,
) -> Dict[str, Any]:
    """
    Validate:
    1) nonnegativity
    2) sum_k f_{j,k} <= f_j^max
    3) positive CPU should only appear on actually executed offloaded tasks
    """
    uav_cpu_max = state["uav_cpu_max"]
    M, K = access_assoc.shape

    neg_violation = 0.0
    budget_violation = 0.0
    support_violation = 0.0
    ok = True

    min_cpu = float(np.min(cpu_alloc))
    if min_cpu < -1e-8:
        neg_violation = -min_cpu
        ok = False

    for j in range(M):
        violation = max(0.0, float(np.sum(cpu_alloc[j]) - uav_cpu_max[j]))
        budget_violation = max(budget_violation, violation)
        if violation > 1e-6:
            ok = False

    for j in range(M):
        for k in range(K):
            if cpu_alloc[j, k] <= 1e-8:
                continue

            access_m = int(np.argmax(access_assoc[:, k]))

            # task k must be offloaded
            if offload_ratio[k] <= EPS:
                support_violation = max(support_violation, float(cpu_alloc[j, k]))
                ok = False
                continue

            # and actually scheduled to execution UAV j
            if sched_beta[k, access_m, j] <= 0.5:
                support_violation = max(support_violation, float(cpu_alloc[j, k]))
                ok = False

    return {
        "ok": ok,
        "neg_violation": float(neg_violation),
        "budget_violation": float(budget_violation),
        "support_violation": float(support_violation),
    }