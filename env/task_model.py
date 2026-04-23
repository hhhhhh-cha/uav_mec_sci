from typing import Dict, Any

import numpy as np

EPS = 1e-8


def compute_local_delay(
    task_size: float,
    task_cycles: float,
    offload_ratio: float,
    task_local_cpu: float,
) -> float:
    """
    T_loc = (1 - lambda) * D * C / f_loc
    """
    f_loc = max(float(task_local_cpu), EPS)
    lam = float(offload_ratio)
    Dk = float(task_size)
    Ck = float(task_cycles)
    return (1.0 - lam) * Dk * Ck / f_loc


def compute_uplink_delay(
    bw_alloc_mk: float,
    rate_up_raw_mk: float,
    task_size: float,
    offload_ratio: float,
) -> float:
    """
    T_up = lambda * D / R_up
    where current code stores:
        R_up = B_mk * log2(1 + SINR)
    """
    lam = float(offload_ratio)
    if lam <= EPS:
        return 0.0

    Dk = float(task_size)
    r_up = max(float(bw_alloc_mk) * float(rate_up_raw_mk), EPS)
    return lam * Dk / r_up


def compute_backhaul_delay(
    rate_a2a_mj: float,
    task_size: float,
    offload_ratio: float,
    is_forwarded: bool,
) -> float:
    """
    T_bh = lambda * D / R_a2a, if j != m
    """
    lam = float(offload_ratio)
    if lam <= EPS or not is_forwarded:
        return 0.0

    Dk = float(task_size)
    r_bh = max(float(rate_a2a_mj), EPS)
    return lam * Dk / r_bh


def compute_execution_delay(
    cpu_alloc_jk: float,
    task_size: float,
    task_cycles: float,
    offload_ratio: float,
) -> float:
    """
    T_exec = lambda * D * C / f_exec
    """
    lam = float(offload_ratio)
    if lam <= EPS:
        return 0.0

    Dk = float(task_size)
    Ck = float(task_cycles)
    f_exec = max(float(cpu_alloc_jk), EPS)
    return lam * Dk * Ck / f_exec


def compute_tx_energy(
    uav_tx_power_m: float,
    backhaul_delay: float,
    is_forwarded: bool,
) -> float:
    """
    Current engineering-consistent form:
        E_tx = P_tx * T_bh
    This is equivalent to:
        P_tx * lambda * D / R_a2a
    """
    if not is_forwarded:
        return 0.0
    return float(uav_tx_power_m) * float(backhaul_delay)


def compute_cmp_energy(
    kappa_j: float,
    task_size: float,
    task_cycles: float,
    offload_ratio: float,
    cpu_alloc_jk: float,
    energy_scale: float = 1e-4,
) -> float:
    """
    Paper-inspired computation energy:
        E_cmp ~ kappa * lambda * D * C * f^2

    We keep the same engineering scale factor as the current codebase
    to maintain numerical stability.
    """
    lam = float(offload_ratio)
    if lam <= EPS:
        return 0.0

    Dk = float(task_size)
    Ck = float(task_cycles)
    f_exec = max(float(cpu_alloc_jk), EPS)

    return (
        float(energy_scale)
        * float(kappa_j)
        * lam * Dk * Ck * (f_exec ** 2)
    )


def compute_delay_and_energy_from_action(
    state: Dict[str, Any],
    access_assoc: np.ndarray,
    offload_ratio: np.ndarray,
    sched_beta: np.ndarray,
    bw_alloc: np.ndarray,
    cpu_alloc: np.ndarray,
) -> Dict[str, Any]:
    """
    Unified task-delay and UAV-side energy computation.

    Returns:
        same metric dictionary style as the current feasibility_check.py
    """
    M, K = access_assoc.shape

    task_size = np.asarray(state["task_size"], dtype=float)
    task_cycles = np.asarray(state["task_cycles"], dtype=float)
    task_local_cpu = np.asarray(state["task_local_cpu"], dtype=float)
    rate_up_raw = np.asarray(state["rate_up"], dtype=float)
    rate_a2a = np.asarray(state["rate_a2a"], dtype=float)
    kappa_vec = np.asarray(state["kappa_vec"], dtype=float)
    uav_tx_power = np.asarray(state["uav_tx_power"], dtype=float)

    delay_local = np.zeros((K,), dtype=float)
    delay_up = np.zeros((K,), dtype=float)
    delay_bh = np.zeros((K,), dtype=float)
    delay_exec = np.zeros((K,), dtype=float)
    delay_edge = np.zeros((K,), dtype=float)
    delay_total = np.zeros((K,), dtype=float)

    energy_tx = np.zeros((M,), dtype=float)
    energy_cmp = np.zeros((M,), dtype=float)

    delay_bh_tensor = np.zeros((K, M, M), dtype=float)
    delay_exec_tensor = np.zeros((K, M, M), dtype=float)

    for k in range(K):
        Dk = float(task_size[k])
        Ck = float(task_cycles[k])
        lam = float(offload_ratio[k])

        access_m = int(np.argmax(access_assoc[:, k]))

        # 1) local delay
        delay_local[k] = compute_local_delay(
            task_size=Dk,
            task_cycles=Ck,
            offload_ratio=lam,
            task_local_cpu=float(task_local_cpu[k]),
        )

        # 2) uplink delay
        delay_up[k] = compute_uplink_delay(
            bw_alloc_mk=float(bw_alloc[access_m, k]),
            rate_up_raw_mk=float(rate_up_raw[access_m, k]),
            task_size=Dk,
            offload_ratio=lam,
        )

        chosen_js = np.where(sched_beta[k, access_m, :] > 0.5)[0]

        if len(chosen_js) != 1:
            delay_edge[k] = delay_up[k]
            delay_total[k] = max(delay_local[k], delay_edge[k])
            continue

        j_star = int(chosen_js[0])
        is_forwarded = bool(j_star != access_m)

        # 3) backhaul delay
        delay_bh[k] = compute_backhaul_delay(
            rate_a2a_mj=float(rate_a2a[access_m, j_star]),
            task_size=Dk,
            offload_ratio=lam,
            is_forwarded=is_forwarded,
        )
        if is_forwarded:
            delay_bh_tensor[k, access_m, j_star] = delay_bh[k]

        # 4) execution delay
        delay_exec[k] = compute_execution_delay(
            cpu_alloc_jk=float(cpu_alloc[j_star, k]),
            task_size=Dk,
            task_cycles=Ck,
            offload_ratio=lam,
        )
        delay_exec_tensor[k, access_m, j_star] = delay_exec[k]

        # 5) edge delay and total delay
        delay_edge[k] = delay_up[k] + delay_bh[k] + delay_exec[k]
        delay_total[k] = max(delay_local[k], delay_edge[k])

        # 6) transmission energy
        tx_e = compute_tx_energy(
            uav_tx_power_m=float(uav_tx_power[access_m]),
            backhaul_delay=delay_bh[k],
            is_forwarded=is_forwarded,
        )
        energy_tx[access_m] += tx_e

        # 7) computation energy
        cmp_e = compute_cmp_energy(
            kappa_j=float(kappa_vec[j_star]),
            task_size=Dk,
            task_cycles=Ck,
            offload_ratio=lam,
            cpu_alloc_jk=float(cpu_alloc[j_star, k]),
            energy_scale=1e-4,
        )
        energy_cmp[j_star] += cmp_e

    metrics = {
        "delay_local": delay_local,
        "delay_up": delay_up,
        "delay_bh": delay_bh,
        "delay_exec": delay_exec,
        "delay_edge": delay_edge,
        "delay_total": delay_total,
        "delay_bh_tensor": delay_bh_tensor,
        "delay_exec_tensor": delay_exec_tensor,
        "delay_sys": float(np.sum(delay_total)),
        "energy_tx": energy_tx,
        "energy_cmp": energy_cmp,
        "energy_sys": float(np.sum(energy_tx) + np.sum(energy_cmp)),
    }
    return metrics