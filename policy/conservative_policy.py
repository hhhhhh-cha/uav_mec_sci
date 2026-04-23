import numpy as np


def generate_conservative_high_action(state, access_assoc):
    """
    Conservative policy for sanity-check and stable feasible rollout.

    Strategy:
    1) no movement
    2) moderate offloading
    3) execute on access UAV itself
    """
    M, K = access_assoc.shape

    move_dist = np.zeros(M, dtype=float)
    move_angle = np.zeros(M, dtype=float)
    offload_ratio = 0.2 * np.ones(K, dtype=float)

    sched_beta = np.zeros((K, M, M), dtype=float)
    for k in range(K):
        access_m = int(np.argmax(access_assoc[:, k]))
        sched_beta[k, access_m, access_m] = 1.0

    return {
        "move_dist": move_dist,
        "move_angle": move_angle,
        "offload_ratio": offload_ratio,
        "sched_beta": sched_beta,
    }