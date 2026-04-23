import numpy as np

EPS = 1e-8


def _safe_norm(x, denom):
    return float(x) / max(float(denom), EPS)


def build_global_observation(state):
    """
    Build a flat global observation vector for the current simplified
    trainable proposed-policy version.

    Current design:
    - This is a single shared observation for the whole system.
    - Later, it can be split into per-agent local observations for MADDPG.

    Included features:
    1) UAV positions / energy / CPU
    2) task features
    3) access-link quality summary
    4) topology summary
    """
    M = int(state["M"])
    K = int(state["K"])
    area_size = float(state["area_size"])

    uav_pos = state["uav_pos"]
    uav_energy = state["uav_energy"]
    uav_cpu_avail = state["uav_cpu_avail"]
    uav_cpu_max = state["uav_cpu_max"]

    td_pos = state["td_pos"]
    task_size = state["task_size"]
    task_cycles = state["task_cycles"]
    task_deadline = state["task_deadline"]
    task_local_cpu = state["task_local_cpu"]

    gain_a2g = state["gain_a2g"]
    rate_up = state["rate_up"]
    dist_a2a = state["dist_a2a"]
    rate_a2a = state["rate_a2a"]
    neighbors = state["neighbors"]

    feat = []

    # -------------------------------------------------
    # UAV features
    # -------------------------------------------------
    energy_scale = max(float(np.max(uav_energy)), 1.0)
    cpu_scale = max(float(np.max(uav_cpu_max)), 1.0)

    for m in range(M):
        feat.extend([
            _safe_norm(uav_pos[m, 0], area_size),
            _safe_norm(uav_pos[m, 1], area_size),
            _safe_norm(uav_energy[m], energy_scale),
            _safe_norm(uav_cpu_avail[m], cpu_scale),
            _safe_norm(uav_cpu_max[m], cpu_scale),
            len(neighbors[m]) / max(M - 1, 1),
        ])

    # -------------------------------------------------
    # Task features
    # -------------------------------------------------
    size_scale = max(float(np.max(task_size)), 1.0)
    cycle_scale = max(float(np.max(task_cycles)), 1.0)
    deadline_scale = max(float(np.max(task_deadline)), 1.0)
    local_cpu_scale = max(float(np.max(task_local_cpu)), 1.0)
    rate_up_scale = max(float(np.max(rate_up)), 1.0)

    for k in range(K):
        best_access_m = int(np.argmax(rate_up[:, k]))
        best_up_rate = float(rate_up[best_access_m, k])

        feat.extend([
            _safe_norm(td_pos[k, 0], area_size),
            _safe_norm(td_pos[k, 1], area_size),
            _safe_norm(task_size[k], size_scale),
            _safe_norm(task_cycles[k], cycle_scale),
            _safe_norm(task_deadline[k], deadline_scale),
            _safe_norm(task_local_cpu[k], local_cpu_scale),
            _safe_norm(best_access_m, max(M - 1, 1)),
            _safe_norm(best_up_rate, rate_up_scale),
        ])

    # -------------------------------------------------
    # Topology summary
    # -------------------------------------------------
    if M > 1:
        a2a_dist_vals = dist_a2a[dist_a2a > 0]
        a2a_rate_vals = rate_a2a[rate_a2a > 0]
        mean_a2a_dist = float(np.mean(a2a_dist_vals)) if a2a_dist_vals.size > 0 else 0.0
        mean_a2a_rate = float(np.mean(a2a_rate_vals)) if a2a_rate_vals.size > 0 else 0.0
    else:
        mean_a2a_dist = 0.0
        mean_a2a_rate = 0.0

    max_dist_scale = max(area_size, 1.0)
    max_rate_scale = max(float(np.max(rate_a2a)), float(np.max(rate_up)), 1.0)

    feat.extend([
        _safe_norm(mean_a2a_dist, max_dist_scale),
        _safe_norm(mean_a2a_rate, max_rate_scale),
    ])

    return np.asarray(feat, dtype=np.float32)


def get_observation_dim(state):
    obs = build_global_observation(state)
    return int(obs.shape[0])