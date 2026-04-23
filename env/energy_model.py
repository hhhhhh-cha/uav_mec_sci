# 实现论文里的 rotor-wing propulsion power
# 实现 E_fly = P_fly(v) * Δt
# 给你一个统一接口，后面环境里直接调用

from typing import Dict, Any

import numpy as np

EPS = 1e-8


def ensure_energy_params(state: Dict[str, Any]) -> None:
    """
    Fill default rotor-wing energy-model parameters if missing.

    These defaults are engineering-safe placeholders and can be adjusted later
    according to the final paper setting.
    """
    defaults = {
        # rotor-wing propulsion model parameters
        "P0": 79.86,         # profile power in hovering
        "Pi": 88.63,         # induced power in hovering
        "U_tip": 120.0,      # rotor blade tip speed
        "v0": 4.03,          # mean rotor induced velocity in hover
        "d0": 0.6,           # fuselage drag ratio
        "rho_air": 1.225,    # air density
        "rotor_solidity": 0.05,
        "rotor_disc_area": 0.503,
    }

    for k, v in defaults.items():
        if k not in state:
            state[k] = float(v)


def compute_rotor_wing_power(state: Dict[str, Any], move_dist: np.ndarray) -> np.ndarray:
    """
    Paper-aligned rotor-wing propulsion power model.

    For UAV m:
        P_fly(v_m^t) = P0 * (1 + 3 v^2 / U_tip^2)
                     + 0.5 * d0 * rho * s * A * v^3
                     + Pi * sqrt( sqrt(1 + v^4 / (4 v0^4)) - v^2 / (2 v0^2) )

    Notes:
    - We use numerically stable clipping to avoid invalid sqrt due to tiny
      floating-point errors.
    """
    ensure_energy_params(state)

    delta_t = float(state["delta_t"])
    v = np.asarray(move_dist, dtype=float) / max(delta_t, EPS)

    P0 = float(state["P0"])
    Pi = float(state["Pi"])
    U_tip = float(state["U_tip"])
    v0 = float(state["v0"])
    d0 = float(state["d0"])
    rho_air = float(state["rho_air"])
    rotor_solidity = float(state["rotor_solidity"])
    rotor_disc_area = float(state["rotor_disc_area"])

    term1 = P0 * (1.0 + 3.0 * (v ** 2) / max(U_tip ** 2, EPS))
    term2 = 0.5 * d0 * rho_air * rotor_solidity * rotor_disc_area * (v ** 3)

    inside_sqrt1 = 1.0 + (v ** 4) / max(4.0 * (v0 ** 4), EPS)
    sqrt1 = np.sqrt(np.maximum(inside_sqrt1, 0.0))

    inside_sqrt2 = sqrt1 - (v ** 2) / max(2.0 * (v0 ** 2), EPS)
    inside_sqrt2 = np.maximum(inside_sqrt2, 0.0)

    term3 = Pi * np.sqrt(inside_sqrt2)

    power = term1 + term2 + term3
    power = np.maximum(power, 0.0)
    return power.astype(float)


def compute_flight_energy(state: Dict[str, Any], move_dist: np.ndarray) -> np.ndarray:
    """
    E_fly = P_fly(v) * delta_t
    """
    power = compute_rotor_wing_power(state, move_dist)
    delta_t = float(state["delta_t"])
    energy = power * delta_t
    return np.asarray(energy, dtype=float)


def compute_total_uav_energy_cost(
    state: Dict[str, Any],
    energy_fly: np.ndarray,
    energy_tx: np.ndarray,
    energy_cmp: np.ndarray,
) -> np.ndarray:
    """
    Total one-slot UAV-side energy cost per UAV:
        E_fly + E_tx + E_cmp
    """
    energy_fly = np.asarray(energy_fly, dtype=float)
    energy_tx = np.asarray(energy_tx, dtype=float)
    energy_cmp = np.asarray(energy_cmp, dtype=float)

    total_cost = energy_fly + energy_tx + energy_cmp
    return total_cost.astype(float)