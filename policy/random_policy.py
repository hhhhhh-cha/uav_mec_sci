import numpy as np


def generate_random_high_action(state, access_assoc, seed=None):
    rng = np.random.default_rng(seed)

    M, K = access_assoc.shape
    neighbors = state["neighbors"]

    move_dist = np.zeros(M, dtype=float)
    move_angle = np.zeros(M, dtype=float)
    offload_ratio = rng.uniform(0.0, 1.0, size=K)

    sched_beta = np.zeros((K, M, M), dtype=float)

    for k in range(K):
        access_m = int(np.argmax(access_assoc[:, k]))
        legal_js = [access_m] + list(neighbors[access_m])
        j_star = int(rng.choice(legal_js))
        sched_beta[k, access_m, j_star] = 1.0

    return {
        "move_dist": move_dist,
        "move_angle": move_angle,
        "offload_ratio": offload_ratio,
        "sched_beta": sched_beta,
    }