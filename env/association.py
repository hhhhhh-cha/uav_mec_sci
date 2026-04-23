from typing import Dict, Any, Optional

import numpy as np
# 负责两件事：
# （1）按 strongest uplink rule 生成 access_assoc
# （2）按 生成候选执行集

# 同时加了两个合法性检查函数，方便后面测试时先查：
# （1）接入关联是不是每个任务唯一
# （2）调度决策是不是 one-hot 且候选节点合法


EPS = 1e-8

def build_access_association(state: Dict[str, Any]) -> np.ndarray:
    """
    Build access association according to strongest uplink channel rule.

    Corresponds to the paper:
        x_{m,k}^t = 1, if m = argmax_{m'} h_{m',k}^t
                   0, otherwise

    Returns:
        access_assoc: shape [M, K], binary matrix
    """
    gain_a2g = state["gain_a2g"]
    M, K = gain_a2g.shape

    access_assoc = np.zeros((M, K), dtype=float)

    for k in range(K):
        m_star = int(np.argmax(gain_a2g[:, k]))
        access_assoc[m_star, k] = 1.0

    return access_assoc


def build_tasks_of_uav(access_assoc: np.ndarray) -> Dict[int, list]:
    """
    Build K_m^t: tasks associated with each UAV m.

    Returns:
        tasks_of_uav[m] = [k1, k2, ...]
    """
    M, K = access_assoc.shape
    tasks_of_uav = {}

    for m in range(M):
        task_idx = np.where(access_assoc[m] > 0.5)[0].tolist()
        tasks_of_uav[m] = task_idx

    return tasks_of_uav


def build_candidate_exec_set(state: Dict[str, Any]) -> Dict[int, list]:
    """
    Build candidate execution set:
        J_m^t = {m} U N_m^t

    Returns:
        cand_exec_set[m] = [m] + neighbors[m]
    """
    M = state["M"]
    neighbors = state["neighbors"]

    cand_exec_set = {}
    for m in range(M):
        cand_exec_set[m] = [m] + neighbors[m]

    return cand_exec_set


def _estimate_local_delay_for_task(state: Dict[str, Any], k: int, lam: float) -> float:
    """
    Estimate local delay:
        T_k^{loc,t} = (1-lambda_k^t) D_k^t C_k^t / f_k^{loc,t}
    """
    Dk = float(state["task_size"][k])
    Ck = float(state["task_cycles"][k])
    f_loc = max(float(state["task_local_cpu"][k]), EPS)
    return (1.0 - lam) * Dk * Ck / f_loc


def _estimate_edge_delay_for_candidate(
    state: Dict[str, Any],
    k: int,
    access_m: int,
    exec_j: int,
    lam: float,
    cpu_share: float,
    num_tasks_m: int,
) -> float:
    if lam <= EPS:
        return 0.0

    Dk = float(state["task_size"][k])
    Ck = float(state["task_cycles"][k])

    B_max_m = float(state["B_max"][access_m])
    rate_up_raw = max(float(state["rate_up"][access_m, k]), EPS)

    B_guess = B_max_m / max(num_tasks_m, 1)
    up_rate_eff = max(B_guess * rate_up_raw, EPS)
    T_up = lam * Dk / up_rate_eff

    if exec_j != access_m:
        r_bh = max(float(state["rate_a2a"][access_m, exec_j]), EPS)
        T_bh = lam * Dk / r_bh
    else:
        T_bh = 0.0

    cpu_share = max(cpu_share, EPS)
    T_exec = lam * Dk * Ck / cpu_share

    return T_up + T_bh + T_exec

# def _estimate_edge_delay_for_candidate(
#     state: Dict[str, Any],
#     k: int,
#     access_m: int,
#     exec_j: int,
#     lam: float,
#     cpu_share: float,
# ) -> float:
#     """
#     Rough heuristic estimate of edge-side delay for candidate execution UAV j.

#     We use a lightweight estimate:
#         T_edge ≈ T_up + T_bh + T_exec

#     where
#         T_up   ≈ lambda D / (B_guess * log2(1+sinr))
#         T_bh   ≈ lambda D / R_a2a, if j != m
#         T_exec ≈ lambda D C / cpu_share
#     """
#     if lam <= EPS:
#         return 0.0

#     Dk = float(state["task_size"][k])
#     Ck = float(state["task_cycles"][k])

#     # rough uplink estimate:
#     # use a guessed uplink bandwidth share instead of exact bw solver output
#     B_max_m = float(state["B_max"][access_m])
#     rate_up_raw = max(float(state["rate_up"][access_m, k]), EPS)

#     # heuristic guess: each associated task gets about an equal share
#     up_rate_eff = max(B_max_m * 0.5 * rate_up_raw, EPS)
#     T_up = lam * Dk / up_rate_eff

#     # backhaul delay
#     if exec_j != access_m:
#         r_bh = max(float(state["rate_a2a"][access_m, exec_j]), EPS)
#         T_bh = lam * Dk / r_bh
#     else:
#         T_bh = 0.0

#     # execution delay
#     cpu_share = max(cpu_share, EPS)
#     T_exec = lam * Dk * Ck / cpu_share

#     return T_up + T_bh + T_exec


def random_generate_high_action(
    state: Dict[str, Any],
    access_assoc: np.ndarray,
    seed: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """
    Generate random but legal high-level actions for first-stage feasibility tests.

    Outputs:
        move_dist:    [M]
        move_angle:   [M]
        offload_ratio:[K]
        sched_beta:   [K, M, M]

    Notes:
        - This is only for solver validation.
        - It is NOT the final actor output.
        - For each task k, only the associated access UAV m can schedule it.
        - The chosen execution UAV j must belong to J_m^t.
    """
    rng = np.random.default_rng(seed)

    M, K = access_assoc.shape
    cand_exec_set = build_candidate_exec_set(state)

    # movement is unused in the current low-level test, so keep zeros
    move_dist = np.zeros((M,), dtype=float)
    move_angle = np.zeros((M,), dtype=float)

    # lambda_k in [0, 1]
    offload_ratio = rng.uniform(0.0, 1.0, size=(K,))

    # sched_beta[k, m, j]
    sched_beta = np.zeros((K, M, M), dtype=float)

    for k in range(K):
        access_m = int(np.argmax(access_assoc[:, k]))
        candidates = cand_exec_set[access_m]

        # even if lambda_k is very small, keep schedule one-hot for consistency
        chosen_j = int(rng.choice(candidates))
        sched_beta[k, access_m, chosen_j] = 1.0

    return {
        "move_dist": move_dist,
        "move_angle": move_angle,
        "offload_ratio": offload_ratio,
        "sched_beta": sched_beta,
    }


def heuristic_generate_high_action(
    state: Dict[str, Any],
    access_assoc: np.ndarray,
    seed: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """
    Deadline-aware heuristic high-level action generator.

    Main ideas:
    1) If local execution can already meet the deadline, use conservative offloading.
    2) If local execution cannot meet the deadline, compute a minimum required
       offloading ratio to reduce the local-delay bottleneck.
    3) Prefer execution at the access UAV itself.
    4) Forward to a neighbor only when that neighbor has clearly better CPU and
       the A2A link is sufficiently good.
    5) Keep the action legal under the candidate execution set.

    This is still a heuristic baseline, not the final RL policy.
    """
    rng = np.random.default_rng(seed)

    M, K = access_assoc.shape
    cand_exec_set = build_candidate_exec_set(state)

    tasks_of_uav = build_tasks_of_uav(access_assoc)

    move_dist = np.zeros((M,), dtype=float)
    move_angle = np.zeros((M,), dtype=float)

    offload_ratio = np.zeros((K,), dtype=float)
    sched_beta = np.zeros((K, M, M), dtype=float)

    R_min = float(state["R_min"])
    uav_cpu_avail = state["uav_cpu_avail"]
    task_deadline = state["task_deadline"]
    task_size = state["task_size"]
    task_cycles = state["task_cycles"]
    task_local_cpu = state["task_local_cpu"]

    for k in range(K):
        access_m = int(np.argmax(access_assoc[:, k]))
        candidates = cand_exec_set[access_m]
        num_tasks_m = len(tasks_of_uav[access_m])

        Dk = float(task_size[k])
        Ck = float(task_cycles[k])
        f_loc = max(float(task_local_cpu[k]), EPS)
        deadline_k = float(task_deadline[k])

        uplink_quality = float(state["rate_up"][access_m, k])
        local_cpu_access = float(uav_cpu_avail[access_m])

        # -------------------------------------------------
        # Step 1: local-only delay baseline
        # -------------------------------------------------
        # T_loc when lambda = 0
        local_only_delay = Dk * Ck / f_loc

        # -------------------------------------------------
        # Step 2: choose initial lambda
        # -------------------------------------------------
        # Case A: local already meets deadline -> conservative offloading
        if local_only_delay <= deadline_k:
            if uplink_quality < 1.2 * R_min:
                lam = 0.0
            elif uplink_quality < 3.0 * R_min:
                lam = 0.1
            elif uplink_quality < 6.0 * R_min:
                lam = 0.3
            else:
                lam = 0.5

        # Case B: local cannot meet deadline -> must offload at least part of task
        else:
            # Need:
            #   (1 - lambda) D C / f_loc <= deadline
            # => lambda >= 1 - deadline * f_loc / (D C)
            lam_min = 1.0 - deadline_k * f_loc / max(Dk * Ck, EPS)
            lam_min = float(np.clip(lam_min, 0.0, 1.0))

            # add a small safety margin
            lam = min(1.0, lam_min + 0.15)

            # but if uplink is extremely weak, avoid full aggressive offloading
            if uplink_quality < 1.2 * R_min:
                lam = max(lam_min, min(lam, 0.3))
            elif uplink_quality < 3.0 * R_min:
                lam = max(lam_min, min(lam, 0.5))
            else:
                lam = max(lam_min, lam)

        # final clip
        # lam = float(np.clip(lam, 0.0, 1.0))
        # offload_ratio[k] = lam
        # final clip
        # lam = float(np.clip(lam, 0.0, 1.0))
        # final clip with temporary minimum offloading floor for debug
        lam = float(np.clip(lam, 0.05, 1.0))

        if state.get("debug_heuristic", False) and k < 3:
            print(
                f"[HEURISTIC DEBUG] "
                f"k={k}, access_m={access_m}, "
                f"local_only_delay={local_only_delay:.6f}, "
                f"deadline={deadline_k:.6f}, "
                f"uplink_quality={uplink_quality:.6f}, "
                f"lam={lam:.6f}"
            )

        offload_ratio[k] = lam

        # -------------------------------------------------
        # Step 3: choose execution UAV
        # -------------------------------------------------
        # Default: access UAV itself
        best_j = access_m
        best_score = float("inf")

        # Evaluate all legal candidates
        for j in candidates:
            cpu_j = float(uav_cpu_avail[j])

            # Rough execution CPU share guess
            # Prefer not to overestimate
            cpu_share_j = max(0.4 * cpu_j, EPS)

            # Neighbor forwarding conditions
            if j != access_m:
                a2a_rate = float(state["rate_a2a"][access_m, j])

                # weak A2A link -> skip
                if a2a_rate < 2.0 * R_min:
                    continue

                # only forward if neighbor CPU is clearly better
                if cpu_j <= 1.15 * local_cpu_access:
                    continue

            edge_delay_j = _estimate_edge_delay_for_candidate(
                state=state,
                k=k,
                access_m=access_m,
                exec_j=j,
                lam=lam,
                cpu_share=cpu_share_j,
                num_tasks_m=num_tasks_m,
            )

            # add a slight preference for local execution at access UAV
            # to avoid unnecessary forwarding when scores are similar
            if j != access_m:
                edge_delay_j *= 1.03

            score_j = edge_delay_j + 1e-6 * rng.uniform()

            if score_j < best_score:
                best_score = score_j
                best_j = j

        # -------------------------------------------------
        # Step 4: deadline-aware correction
        # -------------------------------------------------
        local_delay_now = _estimate_local_delay_for_task(state, k, lam)
        chosen_cpu_share = max(0.4 * float(uav_cpu_avail[best_j]), EPS)
        chosen_edge_delay = _estimate_edge_delay_for_candidate(
            state=state,
            k=k,
            access_m=access_m,
            exec_j=best_j,
            lam=lam,
            cpu_share=chosen_cpu_share,
            num_tasks_m=num_tasks_m,
        )

        # If both local and edge are still too slow, push lambda upward when possible
        if max(local_delay_now, chosen_edge_delay) > deadline_k:
            if local_delay_now > deadline_k and chosen_edge_delay < local_delay_now:
                lam = min(1.0, max(lam, min(0.55, lam + 0.1)))
                offload_ratio[k] = lam

                local_delay_now = _estimate_local_delay_for_task(state, k, lam)
                chosen_edge_delay = _estimate_edge_delay_for_candidate(
                    state=state,
                    k=k,
                    access_m=access_m,
                    exec_j=best_j,
                    lam=lam,
                    cpu_share=chosen_cpu_share,
                    num_tasks_m=num_tasks_m,
                )

        # If access UAV itself is still too weak, re-check strong neighbors
        if chosen_edge_delay > deadline_k and best_j == access_m:
            for j in candidates:
                if j == access_m:
                    continue

                cpu_j = float(uav_cpu_avail[j])
                a2a_rate = float(state["rate_a2a"][access_m, j])

                if a2a_rate < 2.5 * R_min:
                    continue
                if cpu_j <= 1.25 * local_cpu_access:
                    continue

                cpu_share_j = max(0.5 * cpu_j, EPS)
                edge_delay_j = _estimate_edge_delay_for_candidate(
                    state=state,
                    k=k,
                    access_m=access_m,
                    exec_j=j,
                    lam=lam,
                    cpu_share=cpu_share_j,
                    num_tasks_m=num_tasks_m,
                )

                if edge_delay_j < chosen_edge_delay:
                    chosen_edge_delay = edge_delay_j
                    best_j = j

        # -------------------------------------------------
        # Step 5: final schedule one-hot
        # -------------------------------------------------
        sched_beta[k, access_m, best_j] = 1.0

    return {
        "move_dist": move_dist,
        "move_angle": move_angle,
        "offload_ratio": offload_ratio,
        "sched_beta": sched_beta,
    }

def validate_access_association(access_assoc: np.ndarray) -> Dict[str, Any]:
    """
    Check whether each task is associated with exactly one access UAV.
    """
    _, K = access_assoc.shape
    col_sum = np.sum(access_assoc, axis=0)

    max_violation = float(np.max(np.abs(col_sum - 1.0)))
    ok = bool(np.all(np.abs(col_sum - 1.0) <= 1e-8))

    return {
        "ok": ok,
        "max_violation": max_violation,
        "col_sum": col_sum,
    }


def validate_sched_beta_legal(
    state: Dict[str, Any],
    access_assoc: np.ndarray,
    sched_beta: np.ndarray,
) -> Dict[str, Any]:
    """
    Validate that:
    1) each task has exactly one execution node under its associated access UAV
    2) the chosen execution UAV belongs to the candidate execution set
    """
    M, K = access_assoc.shape
    cand_exec_set = build_candidate_exec_set(state)

    one_hot_violation = 0.0
    candidate_violation = 0.0
    ok = True

    for k in range(K):
        access_m = int(np.argmax(access_assoc[:, k]))
        s = float(np.sum(sched_beta[k, access_m, :]))
        one_hot_violation = max(one_hot_violation, abs(s - 1.0))
        if abs(s - 1.0) > 1e-8:
            ok = False

        chosen_js = np.where(sched_beta[k, access_m, :] > 0.5)[0]
        if len(chosen_js) != 1:
            candidate_violation = max(candidate_violation, 1.0)
            ok = False
            continue

        j = int(chosen_js[0])
        if j not in cand_exec_set[access_m]:
            candidate_violation = max(candidate_violation, 1.0)
            ok = False

    return {
        "ok": ok,
        "one_hot_violation": float(one_hot_violation),
        "candidate_violation": float(candidate_violation),
    }




# from typing import Dict, Any, Optional

# import numpy as np
# # 负责两件事：
# # （1）按 strongest uplink rule 生成 access_assoc
# # （2）按 生成候选执行集

# # 同时加了两个合法性检查函数，方便后面测试时先查：
# # （1）接入关联是不是每个任务唯一
# # （2）调度决策是不是 one-hot 且候选节点合法


# EPS = 1e-8

# def build_access_association(state: Dict[str, Any]) -> np.ndarray:
#     """
#     Build access association according to strongest uplink channel rule.

#     Corresponds to the paper:
#         x_{m,k}^t = 1, if m = argmax_{m'} h_{m',k}^t
#                    0, otherwise

#     Returns:
#         access_assoc: shape [M, K], binary matrix
#     """
#     gain_a2g = state["gain_a2g"]
#     M, K = gain_a2g.shape

#     access_assoc = np.zeros((M, K), dtype=float)

#     for k in range(K):
#         m_star = int(np.argmax(gain_a2g[:, k]))
#         access_assoc[m_star, k] = 1.0

#     return access_assoc


# def build_tasks_of_uav(access_assoc: np.ndarray) -> Dict[int, list]:
#     """
#     Build K_m^t: tasks associated with each UAV m.

#     Returns:
#         tasks_of_uav[m] = [k1, k2, ...]
#     """
#     M, K = access_assoc.shape
#     tasks_of_uav = {}

#     for m in range(M):
#         task_idx = np.where(access_assoc[m] > 0.5)[0].tolist()
#         tasks_of_uav[m] = task_idx

#     return tasks_of_uav


# def build_candidate_exec_set(state: Dict[str, Any]) -> Dict[int, list]:
#     """
#     Build candidate execution set:
#         J_m^t = {m} U N_m^t

#     Returns:
#         cand_exec_set[m] = [m] + neighbors[m]
#     """
#     M = state["M"]
#     neighbors = state["neighbors"]

#     cand_exec_set = {}
#     for m in range(M):
#         cand_exec_set[m] = [m] + neighbors[m]

#     return cand_exec_set


# def _estimate_local_delay_for_task(state: Dict[str, Any], k: int, lam: float) -> float:
#     """
#     Estimate local delay:
#         T_k^{loc,t} = (1-lambda_k^t) D_k^t C_k^t / f_k^{loc,t}
#     """
#     Dk = float(state["task_size"][k])
#     Ck = float(state["task_cycles"][k])
#     f_loc = max(float(state["task_local_cpu"][k]), EPS)
#     return (1.0 - lam) * Dk * Ck / f_loc


# def _estimate_edge_delay_for_candidate(
#     state: Dict[str, Any],
#     k: int,
#     access_m: int,
#     exec_j: int,
#     lam: float,
#     cpu_share: float,
#     num_tasks_m: int,
# ) -> float:
#     if lam <= EPS:
#         return 0.0

#     Dk = float(state["task_size"][k])
#     Ck = float(state["task_cycles"][k])

#     B_max_m = float(state["B_max"][access_m])
#     rate_up_raw = max(float(state["rate_up"][access_m, k]), EPS)

#     B_guess = B_max_m / max(num_tasks_m, 1)
#     up_rate_eff = max(B_guess * rate_up_raw, EPS)
#     T_up = lam * Dk / up_rate_eff

#     if exec_j != access_m:
#         r_bh = max(float(state["rate_a2a"][access_m, exec_j]), EPS)
#         T_bh = lam * Dk / r_bh
#     else:
#         T_bh = 0.0

#     cpu_share = max(cpu_share, EPS)
#     T_exec = lam * Dk * Ck / cpu_share

#     return T_up + T_bh + T_exec

# # def _estimate_edge_delay_for_candidate(
# #     state: Dict[str, Any],
# #     k: int,
# #     access_m: int,
# #     exec_j: int,
# #     lam: float,
# #     cpu_share: float,
# # ) -> float:
# #     """
# #     Rough heuristic estimate of edge-side delay for candidate execution UAV j.

# #     We use a lightweight estimate:
# #         T_edge ≈ T_up + T_bh + T_exec

# #     where
# #         T_up   ≈ lambda D / (B_guess * log2(1+sinr))
# #         T_bh   ≈ lambda D / R_a2a, if j != m
# #         T_exec ≈ lambda D C / cpu_share
# #     """
# #     if lam <= EPS:
# #         return 0.0

# #     Dk = float(state["task_size"][k])
# #     Ck = float(state["task_cycles"][k])

# #     # rough uplink estimate:
# #     # use a guessed uplink bandwidth share instead of exact bw solver output
# #     B_max_m = float(state["B_max"][access_m])
# #     rate_up_raw = max(float(state["rate_up"][access_m, k]), EPS)

# #     # heuristic guess: each associated task gets about an equal share
# #     up_rate_eff = max(B_max_m * 0.5 * rate_up_raw, EPS)
# #     T_up = lam * Dk / up_rate_eff

# #     # backhaul delay
# #     if exec_j != access_m:
# #         r_bh = max(float(state["rate_a2a"][access_m, exec_j]), EPS)
# #         T_bh = lam * Dk / r_bh
# #     else:
# #         T_bh = 0.0

# #     # execution delay
# #     cpu_share = max(cpu_share, EPS)
# #     T_exec = lam * Dk * Ck / cpu_share

# #     return T_up + T_bh + T_exec


# def random_generate_high_action(
#     state: Dict[str, Any],
#     access_assoc: np.ndarray,
#     seed: Optional[int] = None,
# ) -> Dict[str, np.ndarray]:
#     """
#     Generate random but legal high-level actions for first-stage feasibility tests.

#     Outputs:
#         move_dist:    [M]
#         move_angle:   [M]
#         offload_ratio:[K]
#         sched_beta:   [K, M, M]

#     Notes:
#         - This is only for solver validation.
#         - It is NOT the final actor output.
#         - For each task k, only the associated access UAV m can schedule it.
#         - The chosen execution UAV j must belong to J_m^t.
#     """
#     rng = np.random.default_rng(seed)

#     M, K = access_assoc.shape
#     cand_exec_set = build_candidate_exec_set(state)

#     # movement is unused in the current low-level test, so keep zeros
#     move_dist = np.zeros((M,), dtype=float)
#     move_angle = np.zeros((M,), dtype=float)

#     # lambda_k in [0, 1]
#     offload_ratio = rng.uniform(0.0, 1.0, size=(K,))

#     # sched_beta[k, m, j]
#     sched_beta = np.zeros((K, M, M), dtype=float)

#     for k in range(K):
#         access_m = int(np.argmax(access_assoc[:, k]))
#         candidates = cand_exec_set[access_m]

#         # even if lambda_k is very small, keep schedule one-hot for consistency
#         chosen_j = int(rng.choice(candidates))
#         sched_beta[k, access_m, chosen_j] = 1.0

#     return {
#         "move_dist": move_dist,
#         "move_angle": move_angle,
#         "offload_ratio": offload_ratio,
#         "sched_beta": sched_beta,
#     }


# def heuristic_generate_high_action(
#     state: Dict[str, Any],
#     access_assoc: np.ndarray,
#     seed: Optional[int] = None,
# ) -> Dict[str, np.ndarray]:
#     """
#     Deadline-aware heuristic high-level action generator.

#     Main ideas:
#     1) If local execution can already meet the deadline, use conservative offloading.
#     2) If local execution cannot meet the deadline, compute a minimum required
#        offloading ratio to reduce the local-delay bottleneck.
#     3) Prefer execution at the access UAV itself.
#     4) Forward to a neighbor only when that neighbor has clearly better CPU and
#        the A2A link is sufficiently good.
#     5) Keep the action legal under the candidate execution set.

#     This is still a heuristic baseline, not the final RL policy.
#     """
#     rng = np.random.default_rng(seed)

#     M, K = access_assoc.shape
#     cand_exec_set = build_candidate_exec_set(state)

#     tasks_of_uav = build_tasks_of_uav(access_assoc)

#     move_dist = np.zeros((M,), dtype=float)
#     move_angle = np.zeros((M,), dtype=float)

#     offload_ratio = np.zeros((K,), dtype=float)
#     sched_beta = np.zeros((K, M, M), dtype=float)

#     R_min = float(state["R_min"])
#     uav_cpu_avail = state["uav_cpu_avail"]
#     task_deadline = state["task_deadline"]
#     task_size = state["task_size"]
#     task_cycles = state["task_cycles"]
#     task_local_cpu = state["task_local_cpu"]

#     for k in range(K):
#         access_m = int(np.argmax(access_assoc[:, k]))
#         candidates = cand_exec_set[access_m]
#         num_tasks_m = len(tasks_of_uav[access_m])

#         Dk = float(task_size[k])
#         Ck = float(task_cycles[k])
#         f_loc = max(float(task_local_cpu[k]), EPS)
#         deadline_k = float(task_deadline[k])

#         uplink_quality = float(state["rate_up"][access_m, k])
#         local_cpu_access = float(uav_cpu_avail[access_m])

#         # -------------------------------------------------
#         # Step 1: local-only delay baseline
#         # -------------------------------------------------
#         # T_loc when lambda = 0
#         local_only_delay = Dk * Ck / f_loc

#         # -------------------------------------------------
#         # Step 2: choose initial lambda
#         # -------------------------------------------------
#         # Case A: local already meets deadline -> conservative offloading
#         if local_only_delay <= deadline_k:
#             if uplink_quality < 1.2 * R_min:
#                 lam = 0.0
#             elif uplink_quality < 3.0 * R_min:
#                 lam = 0.1
#             elif uplink_quality < 6.0 * R_min:
#                 lam = 0.3
#             else:
#                 lam = 0.5

#         # Case B: local cannot meet deadline -> must offload at least part of task
#         else:
#             # Need:
#             #   (1 - lambda) D C / f_loc <= deadline
#             # => lambda >= 1 - deadline * f_loc / (D C)
#             lam_min = 1.0 - deadline_k * f_loc / max(Dk * Ck, EPS)
#             lam_min = float(np.clip(lam_min, 0.0, 1.0))

#             # add a small safety margin
#             lam = min(1.0, lam_min + 0.15)

#             # but if uplink is extremely weak, avoid full aggressive offloading
#             if uplink_quality < 1.2 * R_min:
#                 lam = max(lam_min, min(lam, 0.3))
#             elif uplink_quality < 3.0 * R_min:
#                 lam = max(lam_min, min(lam, 0.5))
#             else:
#                 lam = max(lam_min, lam)

#         # final clip
#         # lam = float(np.clip(lam, 0.0, 1.0))
#         # offload_ratio[k] = lam
#         # final clip
#         # lam = float(np.clip(lam, 0.0, 1.0))
#         # final clip with temporary minimum offloading floor for debug
#         lam = float(np.clip(lam, 0.05, 1.0))

#         if k < 3:
#             print(
#                 f"[HEURISTIC DEBUG] "
#                 f"k={k}, access_m={access_m}, "
#                 f"local_only_delay={local_only_delay:.6f}, "
#                 f"deadline={deadline_k:.6f}, "
#                 f"uplink_quality={uplink_quality:.6f}, "
#                 f"lam={lam:.6f}"
#             )

#         offload_ratio[k] = lam

#         # -------------------------------------------------
#         # Step 3: choose execution UAV
#         # -------------------------------------------------
#         # Default: access UAV itself
#         best_j = access_m
#         best_score = float("inf")

#         # Evaluate all legal candidates
#         for j in candidates:
#             cpu_j = float(uav_cpu_avail[j])

#             # Rough execution CPU share guess
#             # Prefer not to overestimate
#             cpu_share_j = max(0.4 * cpu_j, EPS)

#             # Neighbor forwarding conditions
#             if j != access_m:
#                 a2a_rate = float(state["rate_a2a"][access_m, j])

#                 # weak A2A link -> skip
#                 if a2a_rate < 2.0 * R_min:
#                     continue

#                 # only forward if neighbor CPU is clearly better
#                 if cpu_j <= 1.15 * local_cpu_access:
#                     continue

#             edge_delay_j = _estimate_edge_delay_for_candidate(
#                 state=state,
#                 k=k,
#                 access_m=access_m,
#                 exec_j=j,
#                 lam=lam,
#                 cpu_share=cpu_share_j,
#                 num_tasks_m=num_tasks_m,
#             )

#             # add a slight preference for local execution at access UAV
#             # to avoid unnecessary forwarding when scores are similar
#             if j != access_m:
#                 edge_delay_j *= 1.03

#             score_j = edge_delay_j + 1e-6 * rng.uniform()

#             if score_j < best_score:
#                 best_score = score_j
#                 best_j = j

#         # -------------------------------------------------
#         # Step 4: deadline-aware correction
#         # -------------------------------------------------
#         local_delay_now = _estimate_local_delay_for_task(state, k, lam)
#         chosen_cpu_share = max(0.4 * float(uav_cpu_avail[best_j]), EPS)
#         chosen_edge_delay = _estimate_edge_delay_for_candidate(
#             state=state,
#             k=k,
#             access_m=access_m,
#             exec_j=best_j,
#             lam=lam,
#             cpu_share=chosen_cpu_share,
#             num_tasks_m=num_tasks_m,
#         )

#         # If both local and edge are still too slow, push lambda upward when possible
#         if max(local_delay_now, chosen_edge_delay) > deadline_k:
#             if local_delay_now > deadline_k and chosen_edge_delay < local_delay_now:
#                 lam = min(1.0, max(lam, min(0.55, lam + 0.1)))
#                 offload_ratio[k] = lam

#                 local_delay_now = _estimate_local_delay_for_task(state, k, lam)
#                 chosen_edge_delay = _estimate_edge_delay_for_candidate(
#                     state=state,
#                     k=k,
#                     access_m=access_m,
#                     exec_j=best_j,
#                     lam=lam,
#                     cpu_share=chosen_cpu_share,
#                     num_tasks_m=num_tasks_m,
#                 )

#         # If access UAV itself is still too weak, re-check strong neighbors
#         if chosen_edge_delay > deadline_k and best_j == access_m:
#             for j in candidates:
#                 if j == access_m:
#                     continue

#                 cpu_j = float(uav_cpu_avail[j])
#                 a2a_rate = float(state["rate_a2a"][access_m, j])

#                 if a2a_rate < 2.5 * R_min:
#                     continue
#                 if cpu_j <= 1.25 * local_cpu_access:
#                     continue

#                 cpu_share_j = max(0.5 * cpu_j, EPS)
#                 edge_delay_j = _estimate_edge_delay_for_candidate(
#                     state=state,
#                     k=k,
#                     access_m=access_m,
#                     exec_j=j,
#                     lam=lam,
#                     cpu_share=cpu_share_j,
#                     num_tasks_m=num_tasks_m,
#                 )

#                 if edge_delay_j < chosen_edge_delay:
#                     chosen_edge_delay = edge_delay_j
#                     best_j = j

#         # -------------------------------------------------
#         # Step 5: final schedule one-hot
#         # -------------------------------------------------
#         sched_beta[k, access_m, best_j] = 1.0

#     return {
#         "move_dist": move_dist,
#         "move_angle": move_angle,
#         "offload_ratio": offload_ratio,
#         "sched_beta": sched_beta,
#     }

# def validate_access_association(access_assoc: np.ndarray) -> Dict[str, Any]:
#     """
#     Check whether each task is associated with exactly one access UAV.
#     """
#     _, K = access_assoc.shape
#     col_sum = np.sum(access_assoc, axis=0)

#     max_violation = float(np.max(np.abs(col_sum - 1.0)))
#     ok = bool(np.all(np.abs(col_sum - 1.0) <= 1e-8))

#     return {
#         "ok": ok,
#         "max_violation": max_violation,
#         "col_sum": col_sum,
#     }


# def validate_sched_beta_legal(
#     state: Dict[str, Any],
#     access_assoc: np.ndarray,
#     sched_beta: np.ndarray,
# ) -> Dict[str, Any]:
#     """
#     Validate that:
#     1) each task has exactly one execution node under its associated access UAV
#     2) the chosen execution UAV belongs to the candidate execution set
#     """
#     M, K = access_assoc.shape
#     cand_exec_set = build_candidate_exec_set(state)

#     one_hot_violation = 0.0
#     candidate_violation = 0.0
#     ok = True

#     for k in range(K):
#         access_m = int(np.argmax(access_assoc[:, k]))
#         s = float(np.sum(sched_beta[k, access_m, :]))
#         one_hot_violation = max(one_hot_violation, abs(s - 1.0))
#         if abs(s - 1.0) > 1e-8:
#             ok = False

#         chosen_js = np.where(sched_beta[k, access_m, :] > 0.5)[0]
#         if len(chosen_js) != 1:
#             candidate_violation = max(candidate_violation, 1.0)
#             ok = False
#             continue

#         j = int(chosen_js[0])
#         if j not in cand_exec_set[access_m]:
#             candidate_violation = max(candidate_violation, 1.0)
#             ok = False

#     return {
#         "ok": ok,
#         "one_hot_violation": float(one_hot_violation),
#         "candidate_violation": float(candidate_violation),
#     }