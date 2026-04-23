import numpy as np

EPS = 1e-8


def _estimate_local_delay(state, k: int, lam: float) -> float:
    """
    Estimate local delay:
        T_loc = (1-lambda) * D * C / f_loc
    """
    Dk = float(state["task_size"][k])
    Ck = float(state["task_cycles"][k])
    f_loc = max(float(state["task_local_cpu"][k]), EPS)
    return (1.0 - lam) * Dk * Ck / f_loc


def _estimate_edge_delay_for_candidate(
    state,
    access_m: int,
    exec_j: int,
    k: int,
    lam: float,
    access_task_count: np.ndarray,
):
    """
    Rough edge-delay estimate used for high-level placeholder policy:
        T_edge_est = T_up_est + T_bh_est + T_exe_est

    Notes:
    - uplink uses equal-share bandwidth estimate on access UAV
    - backhaul delay is 0 when exec_j == access_m
    - execution delay uses a rough CPU-share estimate on execution UAV
    """
    Dk = float(state["task_size"][k])
    Ck = float(state["task_cycles"][k])

    B_max = state["B_max"]
    rate_up_raw = state["rate_up"]
    rate_a2a = state["rate_a2a"]
    uav_cpu_max = state["uav_cpu_max"]
    neighbors = state["neighbors"]

    # 1) uplink delay estimate
    num_tasks_m = max(int(access_task_count[access_m]), 1)
    bw_est = float(B_max[access_m]) / num_tasks_m
    r_up_est = max(bw_est * float(rate_up_raw[access_m, k]), EPS)
    t_up_est = lam * Dk / r_up_est if lam > EPS else 0.0

    # 2) backhaul delay estimate
    if exec_j == access_m:
        t_bh_est = 0.0
    else:
        r_bh_est = max(float(rate_a2a[access_m, exec_j]), EPS)
        t_bh_est = lam * Dk / r_bh_est

    # 3) execution delay estimate
    # rough equal-share CPU estimate
    # candidate size = self + neighbors of access UAV
    num_exec_est = max(1, 1 + len(neighbors[access_m]))
    f_exec_est = max(float(uav_cpu_max[exec_j]) / num_exec_est, EPS)
    t_exec_est = lam * Dk * Ck / f_exec_est

    return t_up_est + t_bh_est + t_exec_est


def _choose_adaptive_offload_ratio(state, access_m: int, k: int) -> float:
    """
    Adaptive offloading ratio placeholder v2.

    Design goal:
    - make lambda more discriminative across tasks
    - prefer low offloading when local execution is already easy
    - increase offloading only when local execution is relatively tight
    - use uplink quality as a secondary adjustment

    Output range is intentionally wider than v1: [0.05, 0.80].
    """
    Dk = float(state["task_size"][k])
    Ck = float(state["task_cycles"][k])
    deadline = max(float(state["task_deadline"][k]), EPS)
    f_loc = max(float(state["task_local_cpu"][k]), EPS)
    up_quality = float(state["rate_up"][access_m, k])

    # Full-local processing delay estimate
    t_local_full = Dk * Ck / f_loc

    # Tightness ratio:
    # < 1 means local execution may finish within deadline
    # > 1 means local execution is relatively difficult
    tightness = t_local_full / deadline

    # Base lambda by local tightness
    if tightness <= 0.6:
        lam = 0.08
    elif tightness <= 1.0:
        lam = 0.22
    elif tightness <= 1.5:
        lam = 0.45
    else:
        lam = 0.65

    # Secondary adjustment by uplink quality
    # better uplink -> slightly more willing to offload
    if up_quality >= 4.0:
        lam += 0.08
    elif up_quality >= 2.5:
        lam += 0.04
    elif up_quality < 1.5:
        lam -= 0.08

    # Small extra protection:
    # if local is already much easier than deadline, keep lambda low
    if tightness <= 0.4:
        lam = min(lam, 0.12)

    return float(np.clip(lam, 0.05, 0.80))


def generate_proposed_placeholder_action(state, access_assoc):
    """
    Proposed-method placeholder policy (stage-1 version).

    Current design:
    1) freeze mobility: no movement
    2) adaptively choose offload ratio per task
    3) choose execution UAV from candidate set by minimizing:
           max(T_loc_est, T_edge_est)
       which is closer to your paper's completion-delay objective

    This is NOT the final trained policy.
    It is a structured placeholder aligned with the proposed pipeline.
    """
    M, K = access_assoc.shape
    neighbors = state["neighbors"]

    # Stage-1: freeze mobility
    move_dist = np.zeros(M, dtype=float)
    move_angle = np.zeros(M, dtype=float)

    offload_ratio = np.zeros(K, dtype=float)
    sched_beta = np.zeros((K, M, M), dtype=float)

    # rough task counts per access UAV for uplink-share estimation
    access_task_count = np.sum(access_assoc, axis=1)

    for k in range(K):
        access_m = int(np.argmax(access_assoc[:, k]))
        cand_js = [access_m] + list(neighbors[access_m])

        # Step 1: adaptive lambda
        lam = _choose_adaptive_offload_ratio(state, access_m, k)
        offload_ratio[k] = lam

        # Step 2: estimate total completion delay for each candidate j
        t_loc_est = _estimate_local_delay(state, k, lam)

        best_j = access_m
        best_total_cost = float("inf")

        for j in cand_js:
            t_edge_est = _estimate_edge_delay_for_candidate(
                state=state,
                access_m=access_m,
                exec_j=j,
                k=k,
                lam=lam,
                access_task_count=access_task_count,
            )

            # paper-consistent completion style:
            # T_total ~= max(T_loc, T_edge)
            t_total_est = max(t_loc_est, t_edge_est)

            if t_total_est < best_total_cost:
                best_total_cost = t_total_est
                best_j = j

        sched_beta[k, access_m, best_j] = 1.0

    return {
        "move_dist": move_dist,
        "move_angle": move_angle,
        "offload_ratio": offload_ratio,
        "sched_beta": sched_beta,
    }