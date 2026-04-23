from typing import Dict, Any, List
import math

import numpy as np

EPS = 1e-8
C_LIGHT = 3.0e8


def compute_a2g_channels(state: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """
    Paper-aligned A2G model:
      - distance
      - elevation angle
      - LoS probability
      - LoS/NLoS path loss
      - average path loss
      - average channel gain
      - uplink SINR
      - uplink spectral efficiency log2(1+SINR)

    This follows the modeling structure in the paper draft, while keeping
    interference implementation lightweight and numerically stable.
    """
    M = int(state["M"])
    K = int(state["K"])

    uav_pos = np.asarray(state["uav_pos"], dtype=float)
    td_pos = np.asarray(state["td_pos"], dtype=float)
    altitude = float(state["altitude"])

    noise_power = float(state["noise_power"])
    td_tx_power = np.asarray(state["td_tx_power"], dtype=float)

    # -----------------------------
    # paper-like environment params
    # -----------------------------
    fc = float(state.get("carrier_freq", 2.0e9))      # Hz
    env_a = float(state.get("los_a", 9.61))
    env_b = float(state.get("los_b", 0.16))
    eta_los = float(state.get("eta_los_db", 1.0))
    eta_nlos = float(state.get("eta_nlos_db", 20.0))

    # lightweight interference control
    reuse_interference_factor = float(state.get("reuse_interference_factor", 0.15))

    dist_a2g = np.zeros((M, K), dtype=float)
    elev_angle = np.zeros((M, K), dtype=float)
    p_los = np.zeros((M, K), dtype=float)
    p_nlos = np.zeros((M, K), dtype=float)

    pathloss_los_db = np.zeros((M, K), dtype=float)
    pathloss_nlos_db = np.zeros((M, K), dtype=float)
    pathloss_avg_db = np.zeros((M, K), dtype=float)

    gain_a2g = np.zeros((M, K), dtype=float)
    sinr_up = np.zeros((M, K), dtype=float)
    rate_up = np.zeros((M, K), dtype=float)  # store spectral efficiency log2(1+sinr)

    for m in range(M):
        for k in range(K):
            horizontal_d = float(np.linalg.norm(uav_pos[m] - td_pos[k]))
            d = math.sqrt(horizontal_d ** 2 + altitude ** 2)
            d = max(d, 1.0)

            dist_a2g[m, k] = d
            elev_angle[m, k] = 180.0 / math.pi * math.asin(altitude / max(d, EPS))

            # LoS probability
            p_l = 1.0 / (1.0 + env_a * math.exp(-env_b * (elev_angle[m, k] - env_a)))
            p_n = 1.0 - p_l
            p_los[m, k] = p_l
            p_nlos[m, k] = p_n

            # free-space path loss in dB
            fspl_db = 20.0 * math.log10(4.0 * math.pi * fc * d / C_LIGHT)

            pl_los = fspl_db + eta_los
            pl_nlos = fspl_db + eta_nlos
            pl_avg = p_l * pl_los + p_n * pl_nlos

            pathloss_los_db[m, k] = pl_los
            pathloss_nlos_db[m, k] = pl_nlos
            pathloss_avg_db[m, k] = pl_avg

            gain = 10.0 ** (-pl_avg / 10.0)
            gain_a2g[m, k] = gain

    # -------------------------------------------------
    # lightweight inter-UAV reuse interference
    # -------------------------------------------------
    for m in range(M):
        for k in range(K):
            signal = float(td_tx_power[k]) * float(gain_a2g[m, k])

            interference = 0.0
            for mp in range(M):
                if mp == m:
                    continue
                interference += reuse_interference_factor * float(td_tx_power[k]) * float(gain_a2g[mp, k])

            denom = noise_power + interference
            gamma = signal / max(denom, EPS)

            sinr_up[m, k] = gamma
            rate_up[m, k] = math.log2(1.0 + max(gamma, EPS))

    return {
        "dist_a2g": dist_a2g,
        "elev_angle": elev_angle,
        "p_los": p_los,
        "p_nlos": p_nlos,
        "pathloss_los_db": pathloss_los_db,
        "pathloss_nlos_db": pathloss_nlos_db,
        "pathloss_avg_db": pathloss_avg_db,
        "gain_a2g": gain_a2g,
        "sinr_up": sinr_up,
        "rate_up": rate_up,
    }


def compute_a2a_channels(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Paper-aligned A2A model:
      h_{m,j}^{A2A} = beta0 * d^{-2}
      R_{m,j}^{A2A} = B_bh * log2(1 + P_tx * h / sigma^2)

    Also returns dynamic neighbors under communication radius.
    """
    M = int(state["M"])

    uav_pos = np.asarray(state["uav_pos"], dtype=float)
    noise_power = float(state["noise_power"])
    uav_tx_power = np.asarray(state["uav_tx_power"], dtype=float)
    backhaul_bandwidth = float(state["backhaul_bandwidth"])
    neighbor_radius = float(state["neighbor_radius"])
    beta0_a2a = float(state["beta0_a2a"])

    neighbors: List[List[int]] = []
    dist_a2a = np.zeros((M, M), dtype=float)
    gain_a2a = np.zeros((M, M), dtype=float)
    rate_a2a = np.zeros((M, M), dtype=float)

    for m in range(M):
        nb = []
        for j in range(M):
            if m == j:
                continue

            d = float(np.linalg.norm(uav_pos[m] - uav_pos[j]))
            d = max(d, 1.0)
            dist_a2a[m, j] = d

            if d <= neighbor_radius:
                nb.append(j)

            gain = beta0_a2a * (d ** -2)
            gain_a2a[m, j] = gain

            snr = float(uav_tx_power[m]) * gain / max(noise_power, EPS)
            rate_a2a[m, j] = backhaul_bandwidth * math.log2(1.0 + max(snr, EPS))

        neighbors.append(nb)

    return {
        "neighbors": neighbors,
        "dist_a2a": dist_a2a,
        "gain_a2a": gain_a2a,
        "rate_a2a": rate_a2a,
    }


def fill_channel_state(state: Dict[str, Any]) -> None:
    """
    Update state dict in-place with both A2G and A2A fields.
    """
    a2g_dict = compute_a2g_channels(state)
    a2a_dict = compute_a2a_channels(state)

    state.update(a2g_dict)
    state.update(a2a_dict)