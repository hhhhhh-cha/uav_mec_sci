# from typing import Dict, Any, Optional, Tuple
import math

# import numpy as np

# from env.association import build_access_association
# from solver.bandwidth_solver import solve_bandwidth_allocation
# from solver.cpu_solver import (
#     solve_cpu_allocation,
#     solve_cpu_allocation_proportional,
# )
# from solver.feasibility_check import compute_delay_and_energy, check_feasibility

from typing import Dict, Any, Optional, Tuple
import numpy as np

from env.energy_model import compute_flight_energy, compute_total_uav_energy_cost
from env.association import build_access_association
from env.channel_model import fill_channel_state
from solver.bandwidth_solver import solve_bandwidth_allocation
from solver.cpu_solver import (
    solve_cpu_allocation,
    solve_cpu_allocation_proportional,
)
from solver.feasibility_check import compute_delay_and_energy, check_feasibility

EPS = 1e-8


class MultiUavMecEnv:
    """
    Multi-UAV MEC environment for the main proposed-method pipeline.

    Current design goal:
    1) Support reset() / step()
    2) Support mobility + channel/topology update
    3) Support high-level action -> low-level analytical allocation -> reward
    4) Keep compatibility with current solver / association codebase

    This is the main-line environment skeleton before full MADDPG training.
    """

    def __init__(
        self,
        M: int = 3,
        # K: int = 8,
        K: int = 16,
        episode_length: int = 20,
        area_size: float = 100.0,
        altitude: float = 50.0,
        neighbor_radius: float = 50.0,
        delta_t: float = 1.0,
        max_speed: float = 15.0,
        min_uav_distance: float = 3.0,
        cpu_mode: str = "kkt",   # "kkt" or "prop"
        prop_rho: float = 0.45,
        # omega1: float = 100.0,
        omega1: float = 50.0,
        omega2: float = 1.0,
        penalty_coeff: float = 50.0,
        R_min: float = 0.05,
        # R_min: float = 0.02,
        # R_min: float = 0.08,
        deadline_scale: float = 5.0,
        # deadline_scale: float = 2.0,
        # deadline_scale: float = 8.0,
        seed: Optional[int] = None,
        # uav_energy_min: float = 800.0,
        # uav_energy_max: float = 1200.0,
        uav_energy_min: float = 2600.0,
        uav_energy_max: float = 3800.0,
        # task_local_cpu_min: float = 1.0e3,
        # task_local_cpu_max: float = 5.0e3,
        # task_local_cpu_min: float = 3.0e3,
        # task_local_cpu_max: float = 8.0e3,

        task_local_cpu_min: float = 2.0e3,
        task_local_cpu_max: float = 6.0e3,
        # uav_cpu_min: float = 2.0e4,
        # uav_cpu_max_init: float = 5.0e4,

        uav_cpu_min: float = 3.0e4,
        uav_cpu_max_init: float = 6.0e4,

        # uav_cpu_min: float = 1.0e4,
        # uav_cpu_max_init: float = 3.0e4,

        # uav_cpu_min: float = 4.0e4,
        # uav_cpu_max_init: float = 8.0e4,



    ):
        self.M = int(M)
        self.K = int(K)
        self.episode_length = int(episode_length)

        self.area_size = float(area_size)
        self.altitude = float(altitude)
        self.neighbor_radius = float(neighbor_radius)

        self.delta_t = float(delta_t)
        self.max_speed = float(max_speed)
        self.min_uav_distance = float(min_uav_distance)

        self.cpu_mode = str(cpu_mode)
        self.prop_rho = float(prop_rho)

        self.omega1 = float(omega1)
        self.omega2 = float(omega2)
        self.penalty_coeff = float(penalty_coeff)

        self.R_min = float(R_min)
        self.deadline_scale = float(deadline_scale)

        self.base_seed = seed
        self.rng = np.random.default_rng(seed)

        self.t = 0
        self.state: Optional[Dict[str, Any]] = None

        self.uav_energy_min = float(uav_energy_min)
        self.uav_energy_max = float(uav_energy_max)

        self.task_local_cpu_min = float(task_local_cpu_min)
        self.task_local_cpu_max = float(task_local_cpu_max)
        self.uav_cpu_min = float(uav_cpu_min)
        self.uav_cpu_max_init = float(uav_cpu_max_init)

    # =========================================================
    # Public API
    # =========================================================
    def reset(self, seed: Optional[int] = None) -> Dict[str, Any]:
        """
        Reset episode and generate initial single-slot state.
        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.t = 0
        self.state = self._generate_initial_state()
        return self._build_observation()

    def step(self, high_action: Dict[str, np.ndarray]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        Execute one slot:
        1) clip / sanitize action
        2) build access association
        3) low-level analytical allocation
        4) compute delay / energy / feasibility / reward
        5) update battery
        6) move UAVs
        7) regenerate task set for next slot
        8) recompute channels/topology
        """
        if self.state is None:
            raise RuntimeError("Environment is not reset. Call reset() first.")

        action = self._sanitize_high_action(high_action)

        # -----------------------------------------------------
        # 1) Access association under current state
        # -----------------------------------------------------
        access_assoc = build_access_association(self.state)

        offload_ratio = action["offload_ratio"]
        sched_beta = action["sched_beta"]
        move_dist = action["move_dist"]
        move_angle = action["move_angle"]

        # -----------------------------------------------------
        # 2) Low-level analytical allocation
        # -----------------------------------------------------
        bw_alloc = solve_bandwidth_allocation(
            state=self.state,
            access_assoc=access_assoc,
            offload_ratio=offload_ratio,
        )

        if self.cpu_mode == "kkt":
            cpu_alloc, total_bisect_iters = solve_cpu_allocation(
                state=self.state,
                access_assoc=access_assoc,
                sched_beta=sched_beta,
                offload_ratio=offload_ratio,
                omega1=self.omega1,
                omega2=self.omega2,
            )
        elif self.cpu_mode == "prop":
            cpu_alloc = solve_cpu_allocation_proportional(
                state=self.state,
                access_assoc=access_assoc,
                sched_beta=sched_beta,
                offload_ratio=offload_ratio,
                rho=self.prop_rho,
            )
            total_bisect_iters = 0
        else:
            raise ValueError(f"Unknown cpu_mode: {self.cpu_mode}")

        # -----------------------------------------------------
        # 3) Delay / tx-energy / cmp-energy
        # -----------------------------------------------------
        metrics = compute_delay_and_energy(
            state=self.state,
            access_assoc=access_assoc,
            offload_ratio=offload_ratio,
            sched_beta=sched_beta,
            bw_alloc=bw_alloc,
            cpu_alloc=cpu_alloc,
        )

        # -----------------------------------------------------
        # 4) Flight energy
        # -----------------------------------------------------
        energy_fly = self._compute_flight_energy(move_dist)
        metrics["energy_fly"] = energy_fly
        metrics["energy_sys"] = float(metrics["energy_sys"] + np.sum(energy_fly))

        # -----------------------------------------------------
        # 5) Feasibility under current slot
        # -----------------------------------------------------
        report = check_feasibility(
            state=self.state,
            access_assoc=access_assoc,
            offload_ratio=offload_ratio,
            sched_beta=sched_beta,
            bw_alloc=bw_alloc,
            cpu_alloc=cpu_alloc,
            metrics=metrics,
        )

        # add extra hard mobility checks + debug-only energy-budget diagnostic
        extra_report = self._check_extra_constraints(
            move_dist=move_dist,
            move_angle=move_angle,
            energy_fly=energy_fly,
            energy_tx=metrics.get("energy_tx", np.zeros(self.M)),
            energy_cmp=metrics.get("energy_cmp", np.zeros(self.M)),
        )
        report = self._merge_reports(report, extra_report)

        # -----------------------------------------------------
        # 6) Reward
        # r_t = - omega1 * T_sys^t - omega2 * E_sys^t - zeta * penalty
        # -----------------------------------------------------
        # penalty_value = self._compute_penalty(report)
        # reward = -self.omega1 * float(metrics["delay_sys"]) \
        #          - self.omega2 * float(metrics["energy_sys"]) \
        #          - self.penalty_coeff * penalty_value

        penalty_value = self._compute_penalty(report)

        offload_mean = float(np.mean(offload_ratio))
        deadline_pressure = float(report.get("deadline_violation", 0.0))
        offload_bonus = 5.0 * max(0.0, offload_mean - 0.05) * (1.0 + deadline_pressure)

        reward = (
            -self.omega1 * float(metrics["delay_sys"])
            - self.omega2 * float(metrics["energy_sys"])
            - self.penalty_coeff * penalty_value
            + offload_bonus
        )

        # -----------------------------------------------------
        # 7) Update UAV residual energy state after current slot
        # -----------------------------------------------------
        uav_energy_before = self.state["uav_energy"].copy()
        self._update_uav_energy(metrics)
        uav_energy_after = self.state["uav_energy"].copy()


        # -----------------------------------------------------
        # 8) Move UAVs to next slot
        # -----------------------------------------------------
        self._update_uav_positions(move_dist, move_angle)

        # -----------------------------------------------------
        # 9) Advance slot index
        # -----------------------------------------------------
        self.t += 1
        done = bool(self.t >= self.episode_length)

        # -----------------------------------------------------
        # 10) Generate next-slot tasks and recompute channels/topology
        # -----------------------------------------------------
        if not done:
            self._refresh_task_state()
            self._recompute_channels_and_topology()

        next_obs = self._build_observation()

        info = {
            "slot": self.t,
            "access_assoc": access_assoc,
            "bw_alloc": bw_alloc,
            "cpu_alloc": cpu_alloc,
            "metrics": metrics,
            "report": report,
            "penalty_value": penalty_value,
            "total_bisect_iters": total_bisect_iters,
            "high_action": action,
            "uav_energy_before": uav_energy_before,
            "uav_energy_after": uav_energy_after,
        }

        return next_obs, float(reward), done, info

    # =========================================================
    # Initial State Generation
    # =========================================================
    def _generate_initial_state(self) -> Dict[str, Any]:
        """
        Generate initial state for one episode.
        UAV geometry and resource budgets are initialized here.
        """
        M = self.M
        K = self.K

        # -----------------------------
        # 1) UAV / TD positions
        # -----------------------------
        uav_pos = self.rng.uniform(0.0, self.area_size, size=(M, 2))
        td_pos = self.rng.uniform(0.0, self.area_size, size=(K, 2))

        # -----------------------------
        # 2) Task parameters
        # -----------------------------
        task_size = self.rng.uniform(5.0, 20.0, size=(K,))
        task_cycles = self.rng.uniform(500.0, 1500.0, size=(K,))
        task_deadline = self.deadline_scale * self.rng.uniform(5.0, 20.0, size=(K,))
        # task_deadline = self.deadline_scale * self.rng.uniform(3.0, 20.0, size=(K,))
        # task_deadline = self.deadline_scale * self.rng.uniform(8.0, 20.0, size=(K,))
       
        # task_local_cpu = self.rng.uniform(3.0e3, 8.0e3, size=(K,))

        task_local_cpu = self.rng.uniform(
            self.task_local_cpu_min,
            self.task_local_cpu_max,
            size=(K,),
        )
        # -----------------------------
        # 3) UAV resources
        # -----------------------------
        # uav_energy = self.rng.uniform(80.0, 120.0, size=(M,))
        print("DEBUG DEFAULT uav_cpu_min =", self.uav_cpu_min)
        print("DEBUG DEFAULT uav_cpu_max_init =", self.uav_cpu_max_init)               
 
        uav_energy = self.rng.uniform(self.uav_energy_min, self.uav_energy_max, size=(M,))
        # uav_cpu_max = self.rng.uniform(2.0e4, 5.0e4, size=(M,))
        uav_cpu_max = self.rng.uniform(
            self.uav_cpu_min,
            self.uav_cpu_max_init,
            size=(M,),
        )
        print("DEBUG sampled uav_cpu_max[:3] =", uav_cpu_max[:3])
        print("DEBUG sampled uav_cpu_max min/max =", uav_cpu_max.min(), uav_cpu_max.max())

        uav_cpu_avail = uav_cpu_max.copy()
        B_max = self.rng.uniform(10.0, 30.0, size=(M,))
        kappa_vec = self.rng.uniform(1.0e-6, 5.0e-6, size=(M,))

        # -----------------------------
        # 4) Communication params
        # -----------------------------
        # noise_power = 1e-3
        noise_power = 1e-8
        td_tx_power = self.rng.uniform(0.5, 1.0, size=(K,))
        uav_tx_power = self.rng.uniform(1.0, 2.0, size=(M,))
        backhaul_bandwidth = 10.0
        beta0_a2g = 50.0
        beta0_a2a = 50.0

        state = {
            "M": M,
            "K": K,

            # episode / physical params
            "delta_t": self.delta_t,
            "area_size": self.area_size,
            "altitude": self.altitude,
            "neighbor_radius": self.neighbor_radius,
            "max_speed": self.max_speed,
            "min_uav_distance": self.min_uav_distance,

            # geometry
            "uav_pos": uav_pos,
            "td_pos": td_pos,

            # tasks
            "task_size": task_size,
            "task_cycles": task_cycles,
            "task_deadline": task_deadline,
            "task_local_cpu": task_local_cpu,

            # uav resources
            "uav_energy": uav_energy,
            "uav_cpu_max": uav_cpu_max,
            "uav_cpu_avail": uav_cpu_avail,
            "B_max": B_max,
            "kappa_vec": kappa_vec,

            # communication params
            "noise_power": noise_power,
            "td_tx_power": td_tx_power,
            "uav_tx_power": uav_tx_power,
            "backhaul_bandwidth": backhaul_bandwidth,
            "R_min": self.R_min,
            "beta0_a2g": beta0_a2g,
            "beta0_a2a": beta0_a2a,

            # rotor-wing propulsion model parameters
            "P0": 79.86,
            "Pi": 88.63,
            "U_tip": 120.0,
            "v0": 4.03,
            "d0": 0.6,
            "rho_air": 1.225,
            "rotor_solidity": 0.05,
            "rotor_disc_area": 0.503,

            # paper-like A2G params
            "carrier_freq": 2.0e9,
            "los_a": 9.61,
            "los_b": 0.16,
            "eta_los_db": 1.0,
            "eta_nlos_db": 20.0,

            # lightweight reuse interference factor
            "reuse_interference_factor": 0.15,
        }


        # derive topology / channels
        self._fill_topology_and_channel_fields(state)
        return state

    # =========================================================
    # Observation
    # =========================================================
    def _build_observation(self) -> Dict[str, Any]:
        """
        Current version returns a dict observation.
        Later, this can be converted into per-agent observation tensors.
        """
        if self.state is None:
            raise RuntimeError("State is None.")

        obs = {
            "slot": self.t,
            "uav_pos": self.state["uav_pos"].copy(),
            "uav_energy": self.state["uav_energy"].copy(),
            "uav_cpu_avail": self.state["uav_cpu_avail"].copy(),
            "td_pos": self.state["td_pos"].copy(),
            "task_size": self.state["task_size"].copy(),
            "task_cycles": self.state["task_cycles"].copy(),
            "task_deadline": self.state["task_deadline"].copy(),
            "task_local_cpu": self.state["task_local_cpu"].copy(),
            "gain_a2g": self.state["gain_a2g"].copy(),
            "sinr_up": self.state["sinr_up"].copy(),
            "rate_up": self.state["rate_up"].copy(),
            "dist_a2a": self.state["dist_a2a"].copy(),
            "gain_a2a": self.state["gain_a2a"].copy(),
            "rate_a2a": self.state["rate_a2a"].copy(),
            "neighbors": [list(x) for x in self.state["neighbors"]],
            "raw_state": self.state,
        }
        return obs

    # =========================================================
    # Action Processing
    # =========================================================
    def _sanitize_high_action(self, high_action: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Ensure action values are legal and have correct shapes.
        """
        if self.state is None:
            raise RuntimeError("State is None.")

        M = self.M
        K = self.K
        neighbors = self.state["neighbors"]

        move_dist = np.asarray(high_action["move_dist"], dtype=float).reshape(M)
        move_angle = np.asarray(high_action["move_angle"], dtype=float).reshape(M)
        offload_ratio = np.asarray(high_action["offload_ratio"], dtype=float).reshape(K)
        sched_beta = np.asarray(high_action["sched_beta"], dtype=float).reshape(K, M, M)

        # movement clipping
        max_move = self.max_speed * self.delta_t
        move_dist = np.clip(move_dist, 0.0, max_move)

        # angle wrap to [-pi, pi]
        move_angle = (move_angle + np.pi) % (2.0 * np.pi) - np.pi

        # lambda clip
        offload_ratio = np.clip(offload_ratio, 0.0, 1.0)

        # sched_beta legalize:
        # for each task k, only associated UAV m can choose j in J_m
        access_assoc = build_access_association(self.state)
        sched_beta_clean = np.zeros((K, M, M), dtype=float)

        for k in range(K):
            access_m = int(np.argmax(access_assoc[:, k]))
            legal_js = [access_m] + list(neighbors[access_m])

            raw_scores = sched_beta[k, access_m, legal_js]
            best_local_idx = int(np.argmax(raw_scores))
            j_star = int(legal_js[best_local_idx])

            sched_beta_clean[k, access_m, j_star] = 1.0

        return {
            "move_dist": move_dist,
            "move_angle": move_angle,
            "offload_ratio": offload_ratio,
            "sched_beta": sched_beta_clean,
        }

    # =========================================================
    # Dynamics / State Evolution
    # =========================================================
    def _update_uav_positions(self, move_dist: np.ndarray, move_angle: np.ndarray) -> None:
        """
        Update q_{m}^{t+1} = q_m^t + d_m^t [cos theta, sin theta]
        and clip into the area.
        """
        if self.state is None:
            raise RuntimeError("State is None.")

        pos = self.state["uav_pos"]

        dx = move_dist * np.cos(move_angle)
        dy = move_dist * np.sin(move_angle)

        pos[:, 0] += dx
        pos[:, 1] += dy

        pos[:, 0] = np.clip(pos[:, 0], 0.0, self.area_size)
        pos[:, 1] = np.clip(pos[:, 1], 0.0, self.area_size)

        self.state["uav_pos"] = pos

    def _refresh_task_state(self) -> None:
        """
        Generate next-slot task set.
        For now, tasks are re-sampled every slot, which matches the paper's
        per-slot task generation assumption.
        """
        if self.state is None:
            raise RuntimeError("State is None.")

        K = self.K

        self.state["td_pos"] = self.rng.uniform(0.0, self.area_size, size=(K, 2))
        self.state["task_size"] = self.rng.uniform(5.0, 20.0, size=(K,))
        self.state["task_cycles"] = self.rng.uniform(500.0, 1500.0, size=(K,))
        self.state["task_deadline"] = self.deadline_scale * self.rng.uniform(5.0, 20.0, size=(K,))
        # self.state["task_local_cpu"] = self.rng.uniform(3.0e3, 8.0e3, size=(K,))
        self.state["task_local_cpu"] = self.rng.uniform(
            self.task_local_cpu_min,
            self.task_local_cpu_max,
            size=(K,),
        )
        self.state["td_tx_power"] = self.rng.uniform(0.5, 1.0, size=(K,))

        # CPU availability resets per slot in current simplified version
        self.state["uav_cpu_avail"] = self.state["uav_cpu_max"].copy()

    def _recompute_channels_and_topology(self) -> None:
        """
        Recompute A2A/A2G quantities after movement / task refresh.
        """
        if self.state is None:
            raise RuntimeError("State is None.")

        self._fill_topology_and_channel_fields(self.state)

    # =========================================================
    # Topology / Channel Computation
    # =========================================================
    # def _fill_topology_and_channel_fields(self, state: Dict[str, Any]) -> None:
    #     """
    #     Compute neighbors, A2A channels, A2G channels, rates, etc.
    #     """
    def _fill_topology_and_channel_fields(self, state: Dict[str, Any]) -> None:
        """
        Compute topology and channel-related fields in-place.

        All A2G/A2A modeling is delegated to env.channel_model.fill_channel_state
        to avoid duplicated logic and accidental overwriting.
        """
        fill_channel_state(state)

    # def _fill_topology_and_channel_fields(self, state: Dict[str, Any]) -> None:
    #     """
    #     Compute neighbors, A2A channels, A2G channels, rates, etc.
    #     Delegated to env.channel_model for better modularity and
    #     stronger alignment with the paper system model.
    #     """
    #     fill_channel_state(state)
    #     M = state["M"]
    #     K = state["K"]

    #     uav_pos = state["uav_pos"]
    #     td_pos = state["td_pos"]
    #     altitude = state["altitude"]
    #     neighbor_radius = state["neighbor_radius"]

    #     noise_power = state["noise_power"]
    #     uav_tx_power = state["uav_tx_power"]
    #     td_tx_power = state["td_tx_power"]
    #     backhaul_bandwidth = state["backhaul_bandwidth"]
    #     beta0_a2g = state["beta0_a2g"]
    #     beta0_a2a = state["beta0_a2a"]

    #     # -----------------------------
    #     # A2A
    #     # -----------------------------
    #     neighbors = []
    #     dist_a2a = np.zeros((M, M), dtype=float)
    #     gain_a2a = np.zeros((M, M), dtype=float)
    #     rate_a2a = np.zeros((M, M), dtype=float)

    #     for m in range(M):
    #         nb = []
    #         for j in range(M):
    #             if m == j:
    #                 continue

    #             d = float(np.linalg.norm(uav_pos[m] - uav_pos[j]))
    #             dist_a2a[m, j] = d

    #             if d <= neighbor_radius:
    #                 nb.append(j)

    #             gain = beta0_a2a / max(d * d, 1.0)
    #             gain_a2a[m, j] = gain

    #             snr = float(uav_tx_power[m]) * gain / max(noise_power, EPS)
    #             rate_a2a[m, j] = float(backhaul_bandwidth) * math.log2(1.0 + max(snr, EPS))

    #         neighbors.append(nb)

    #     # -----------------------------
    #     # A2G
    #     # -----------------------------
    #     dist_a2g = np.zeros((M, K), dtype=float)
    #     elev_angle = np.zeros((M, K), dtype=float)
    #     gain_a2g = np.zeros((M, K), dtype=float)
    #     sinr_up = np.zeros((M, K), dtype=float)
    #     rate_up = np.zeros((M, K), dtype=float)

    #     for m in range(M):
    #         for k in range(K):
    #             horizontal_d = float(np.linalg.norm(uav_pos[m] - td_pos[k]))
    #             d = math.sqrt(horizontal_d ** 2 + altitude ** 2)
    #             dist_a2g[m, k] = d

    #             elev_angle[m, k] = 180.0 / math.pi * math.asin(altitude / max(d, EPS))

    #             gain = beta0_a2g / max(d * d, 1.0)
    #             gain_a2g[m, k] = gain

    #             sinr = float(td_tx_power[k]) * gain / max(noise_power, EPS)
    #             sinr_up[m, k] = sinr
    #             rate_up[m, k] = math.log2(1.0 + max(sinr, EPS))

    #     state["neighbors"] = neighbors
    #     state["dist_a2a"] = dist_a2a
    #     state["gain_a2a"] = gain_a2a
    #     state["rate_a2a"] = rate_a2a

    #     state["dist_a2g"] = dist_a2g
    #     state["elev_angle"] = elev_angle
    #     state["gain_a2g"] = gain_a2g
    #     state["sinr_up"] = sinr_up
    #     state["rate_up"] = rate_up

    # =========================================================
    # Energy / Reward / Penalty
    # =========================================================
    def _compute_flight_energy(self, move_dist: np.ndarray) -> np.ndarray:
        """
        Paper-aligned rotor-wing propulsion energy model:
            E_fly = P_fly(v) * delta_t
        """
        if self.state is None:
            raise RuntimeError("State is None.")

        return compute_flight_energy(self.state, move_dist)
    # def _compute_flight_energy(self, move_dist: np.ndarray) -> np.ndarray:
    #     """
    #     Simplified flight energy:
    #         E_fly = P(v) * delta_t

    #     Here we use a stable simplified polynomial surrogate rather than the
    #     full rotor-wing expression, to first make the main pipeline run.
    #     Later, this can be replaced by the exact paper model.
    #     """
    #     v = move_dist / max(self.delta_t, EPS)

    #     # simplified propulsion power surrogate
    #     # strictly positive and increasing with speed
    #     p_hover = 8.0
    #     p_linear = 0.25 * v
    #     p_quad = 0.03 * (v ** 2)

    #     power = p_hover + p_linear + p_quad
    #     energy_fly = power * self.delta_t
    #     return energy_fly.astype(float)

    def _update_uav_energy(self, metrics: Dict[str, Any]) -> None:
        if self.state is None:
            raise RuntimeError("State is None.")

        energy_fly = metrics.get("energy_fly", np.zeros((self.M,), dtype=float))
        energy_tx = metrics.get("energy_tx", np.zeros((self.M,), dtype=float))
        energy_cmp = metrics.get("energy_cmp", np.zeros((self.M,), dtype=float))

        total_cost = compute_total_uav_energy_cost(
            state=self.state,
            energy_fly=energy_fly,
            energy_tx=energy_tx,
            energy_cmp=energy_cmp,
        )

        self.state["uav_energy"] = self.state["uav_energy"] - total_cost
        self.state["uav_energy"] = np.maximum(self.state["uav_energy"], 0.0)

    # def _update_uav_energy(self, metrics: Dict[str, Any]) -> None:
    #     if self.state is None:
    #         raise RuntimeError("State is None.")

    #     energy_fly = metrics.get("energy_fly", np.zeros((self.M,), dtype=float))
    #     energy_tx = metrics.get("energy_tx", np.zeros((self.M,), dtype=float))
    #     energy_cmp = metrics.get("energy_cmp", np.zeros((self.M,), dtype=float))

    #     total_cost = energy_fly + energy_tx + energy_cmp
    #     self.state["uav_energy"] = self.state["uav_energy"] - total_cost

    #     # debug-friendly clipping
    #     self.state["uav_energy"] = np.maximum(self.state["uav_energy"], 0.0)

    def _compute_penalty(self, report: Dict[str, Any]) -> float:
        """
        Aggregate only HARD operational constraint violations into a scalar
        penalty.

        Important:
            UAV energy consumption is already penalized by the objective term
            -omega2 * energy_sys in the reward. Therefore, battery/energy-budget
            overrun is kept only as a debug diagnostic and is NOT included here.
        """
        penalty = 0.0
        hard_keys = [
            "ratio_violation",
            "assoc_violation",
            "schedule_violation",
            "candidate_violation",
            "bw_violation",
            "cpu_violation",
            "rate_violation",
            "deadline_violation",
            "move_violation",
            "boundary_violation",
            "collision_violation",
            "nan_count",
        ]

        for k in hard_keys:
            if k in report:
                penalty += float(report[k])

        return penalty

    # =========================================================
    # Extra Constraints
    # =========================================================
    def _check_extra_constraints(
        self,
        move_dist: np.ndarray,
        move_angle: np.ndarray,
        energy_fly: np.ndarray,
        energy_tx: np.ndarray,
        energy_cmp: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Extra diagnostics not covered by current feasibility_check.py:
        1) motion budget
        2) boundary feasibility after movement
        3) minimum inter-UAV safety distance after movement
        4) energy-budget overrun diagnostic

        Note:
            The energy-budget/battery item is NOT treated as a hard feasibility
            constraint in the energy-aware formulation. It is recorded only for
            debugging and analysis. UAV energy is optimized through energy_sys
            in the reward/objective.
        """
        if self.state is None:
            raise RuntimeError("State is None.")

        report = {
            "move_violation": 0.0,
            "boundary_violation": 0.0,
            "collision_violation": 0.0,
            # Debug-only energy budget diagnostic. It is not a hard constraint.
            "energy_budget_overrun_debug": 0.0,
            # Backward-compatible alias. Also debug-only.
            "battery_violation": 0.0,
        }

        max_move = self.max_speed * self.delta_t

        # 1) motion budget
        move_violation = max(0.0, float(np.max(move_dist) - max_move))
        report["move_violation"] = move_violation

        # predict next positions
        cur_pos = self.state["uav_pos"]
        next_pos = cur_pos.copy()
        next_pos[:, 0] += move_dist * np.cos(move_angle)
        next_pos[:, 1] += move_dist * np.sin(move_angle)

        # 2) boundary
        boundary_violation = 0.0
        for m in range(self.M):
            x, y = float(next_pos[m, 0]), float(next_pos[m, 1])
            boundary_violation = max(boundary_violation, max(0.0, -x))
            boundary_violation = max(boundary_violation, max(0.0, x - self.area_size))
            boundary_violation = max(boundary_violation, max(0.0, -y))
            boundary_violation = max(boundary_violation, max(0.0, y - self.area_size))
        report["boundary_violation"] = boundary_violation

        # 3) safety distance
        collision_violation = 0.0
        for m in range(self.M):
            for j in range(m + 1, self.M):
                d = float(np.linalg.norm(next_pos[m] - next_pos[j]))
                violation = max(0.0, self.min_uav_distance - d)
                collision_violation = max(collision_violation, violation)
        report["collision_violation"] = collision_violation

        # 4) energy-budget overrun diagnostic only.
        # This is intentionally NOT a hard feasibility constraint.
        remain_energy = self.state["uav_energy"] - energy_fly - energy_tx - energy_cmp
        energy_overrun = max(0.0, float(np.max(-remain_energy)))
        report["energy_budget_overrun_debug"] = energy_overrun
        report["battery_violation"] = energy_overrun  # backward-compatible debug alias

        return report

    def _merge_reports(self, report_a: Dict[str, Any], report_b: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge feasibility reports and update 'ok' using only HARD operational
        constraints.

        Energy/battery overrun is deliberately excluded from 'ok' because the
        current formulation treats UAV energy as an optimization objective
        rather than as a hard battery-capacity constraint.
        """
        merged = dict(report_a)
        merged.update(report_b)

        hard_keys = [
            "ratio_violation",
            "assoc_violation",
            "schedule_violation",
            "candidate_violation",
            "bw_violation",
            "cpu_violation",
            "rate_violation",
            "deadline_violation",
            "move_violation",
            "boundary_violation",
            "collision_violation",
            "nan_count",
        ]

        merged["ok"] = True
        for k in hard_keys:
            if float(merged.get(k, 0.0)) > 1e-8:
                merged["ok"] = False
                break

        return merged


# =============================================================
# Backward-Compatible Utility Functions
# =============================================================
def random_generate_state(
    M: int = 3,
    K: int = 16,
    area_size: float = 100.0,
    altitude: float = 20.0,
    neighbor_radius: float = 50.0,
    seed: Optional[int] = None,
    R_min: float = 0.05,
    deadline_scale: float = 5.0,
) -> Dict[str, Any]:
    """
    Backward-compatible helper.
    Generates one state using the new environment internals.
    """
    env = MultiUavMecEnv(
        M=M,
        K=K,
        episode_length=20,
        area_size=area_size,
        altitude=altitude,
        neighbor_radius=neighbor_radius,
        R_min=R_min,
        deadline_scale=deadline_scale,
        seed=seed,
    )
    env.reset(seed=seed)
    if env.state is None:
        raise RuntimeError("Failed to generate state.")
    return env.state


def pretty_print_state_summary(state: Dict[str, Any]) -> None:
    """
    Print a concise summary for debugging.
    """
    print("=" * 60)
    print("Environment State Summary")
    print("=" * 60)
    print(f"M (UAVs): {state['M']}")
    print(f"K (Tasks): {state['K']}")
    print(f"Area size: {state['area_size']}")
    print(f"Altitude: {state['altitude']}")
    print(f"Neighbor radius: {state['neighbor_radius']}")
    print(f"UAV positions shape: {state['uav_pos'].shape}")
    print(f"TD positions shape: {state['td_pos'].shape}")
    print(f"A2G gain shape: {state['gain_a2g'].shape}")
    print(f"A2A gain shape: {state['gain_a2a'].shape}")
    print(f"Current UAV energy: {state['uav_energy']}")
    print("=" * 60)

































# # from typing import Dict, Any, Optional, Tuple
# import math

# # import numpy as np

# # from env.association import build_access_association
# # from solver.bandwidth_solver import solve_bandwidth_allocation
# # from solver.cpu_solver import (
# #     solve_cpu_allocation,
# #     solve_cpu_allocation_proportional,
# # )
# # from solver.feasibility_check import compute_delay_and_energy, check_feasibility

# from typing import Dict, Any, Optional, Tuple
# import numpy as np

# from env.energy_model import compute_flight_energy, compute_total_uav_energy_cost
# from env.association import build_access_association
# from env.channel_model import fill_channel_state
# from solver.bandwidth_solver import solve_bandwidth_allocation
# from solver.cpu_solver import (
#     solve_cpu_allocation,
#     solve_cpu_allocation_proportional,
# )
# from solver.feasibility_check import compute_delay_and_energy, check_feasibility

# EPS = 1e-8


# class MultiUavMecEnv:
#     """
#     Multi-UAV MEC environment for the main proposed-method pipeline.

#     Current design goal:
#     1) Support reset() / step()
#     2) Support mobility + channel/topology update
#     3) Support high-level action -> low-level analytical allocation -> reward
#     4) Keep compatibility with current solver / association codebase

#     This is the main-line environment skeleton before full MADDPG training.
#     """

#     def __init__(
#         self,
#         M: int = 3,
#         # K: int = 8,
#         K: int = 16,
#         episode_length: int = 20,
#         area_size: float = 100.0,
#         altitude: float = 50.0,
#         neighbor_radius: float = 50.0,
#         delta_t: float = 1.0,
#         max_speed: float = 15.0,
#         min_uav_distance: float = 3.0,
#         cpu_mode: str = "kkt",   # "kkt" or "prop"
#         prop_rho: float = 0.45,
#         # omega1: float = 100.0,
#         omega1: float = 50.0,
#         omega2: float = 1.0,
#         penalty_coeff: float = 50.0,
#         R_min: float = 0.05,
#         # R_min: float = 0.02,
#         # R_min: float = 0.08,
#         deadline_scale: float = 5.0,
#         # deadline_scale: float = 2.0,
#         # deadline_scale: float = 8.0,
#         seed: Optional[int] = None,
#         # uav_energy_min: float = 800.0,
#         # uav_energy_max: float = 1200.0,
#         uav_energy_min: float = 2600.0,
#         uav_energy_max: float = 3800.0,
#         # task_local_cpu_min: float = 1.0e3,
#         # task_local_cpu_max: float = 5.0e3,
#         # task_local_cpu_min: float = 3.0e3,
#         # task_local_cpu_max: float = 8.0e3,

#         task_local_cpu_min: float = 2.0e3,
#         task_local_cpu_max: float = 6.0e3,
#         # uav_cpu_min: float = 2.0e4,
#         # uav_cpu_max_init: float = 5.0e4,

#         uav_cpu_min: float = 3.0e4,
#         uav_cpu_max_init: float = 6.0e4,

#         # uav_cpu_min: float = 1.0e4,
#         # uav_cpu_max_init: float = 3.0e4,

#         # uav_cpu_min: float = 4.0e4,
#         # uav_cpu_max_init: float = 8.0e4,



#     ):
#         self.M = int(M)
#         self.K = int(K)
#         self.episode_length = int(episode_length)

#         self.area_size = float(area_size)
#         self.altitude = float(altitude)
#         self.neighbor_radius = float(neighbor_radius)

#         self.delta_t = float(delta_t)
#         self.max_speed = float(max_speed)
#         self.min_uav_distance = float(min_uav_distance)

#         self.cpu_mode = str(cpu_mode)
#         self.prop_rho = float(prop_rho)

#         self.omega1 = float(omega1)
#         self.omega2 = float(omega2)
#         self.penalty_coeff = float(penalty_coeff)

#         self.R_min = float(R_min)
#         self.deadline_scale = float(deadline_scale)

#         self.base_seed = seed
#         self.rng = np.random.default_rng(seed)

#         self.t = 0
#         self.state: Optional[Dict[str, Any]] = None

#         self.uav_energy_min = float(uav_energy_min)
#         self.uav_energy_max = float(uav_energy_max)

#         self.task_local_cpu_min = float(task_local_cpu_min)
#         self.task_local_cpu_max = float(task_local_cpu_max)
#         self.uav_cpu_min = float(uav_cpu_min)
#         self.uav_cpu_max_init = float(uav_cpu_max_init)

#     # =========================================================
#     # Public API
#     # =========================================================
#     def reset(self, seed: Optional[int] = None) -> Dict[str, Any]:
#         """
#         Reset episode and generate initial single-slot state.
#         """
#         if seed is not None:
#             self.rng = np.random.default_rng(seed)

#         self.t = 0
#         self.state = self._generate_initial_state()
#         return self._build_observation()

#     def step(self, high_action: Dict[str, np.ndarray]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
#         """
#         Execute one slot:
#         1) clip / sanitize action
#         2) build access association
#         3) low-level analytical allocation
#         4) compute delay / energy / feasibility / reward
#         5) update battery
#         6) move UAVs
#         7) regenerate task set for next slot
#         8) recompute channels/topology
#         """
#         if self.state is None:
#             raise RuntimeError("Environment is not reset. Call reset() first.")

#         action = self._sanitize_high_action(high_action)

#         # -----------------------------------------------------
#         # 1) Access association under current state
#         # -----------------------------------------------------
#         access_assoc = build_access_association(self.state)

#         offload_ratio = action["offload_ratio"]
#         sched_beta = action["sched_beta"]
#         move_dist = action["move_dist"]
#         move_angle = action["move_angle"]

#         # -----------------------------------------------------
#         # 2) Low-level analytical allocation
#         # -----------------------------------------------------
#         bw_alloc = solve_bandwidth_allocation(
#             state=self.state,
#             access_assoc=access_assoc,
#             offload_ratio=offload_ratio,
#         )

#         if self.cpu_mode == "kkt":
#             cpu_alloc, total_bisect_iters = solve_cpu_allocation(
#                 state=self.state,
#                 access_assoc=access_assoc,
#                 sched_beta=sched_beta,
#                 offload_ratio=offload_ratio,
#                 omega1=self.omega1,
#                 omega2=self.omega2,
#             )
#         elif self.cpu_mode == "prop":
#             cpu_alloc = solve_cpu_allocation_proportional(
#                 state=self.state,
#                 access_assoc=access_assoc,
#                 sched_beta=sched_beta,
#                 offload_ratio=offload_ratio,
#                 rho=self.prop_rho,
#             )
#             total_bisect_iters = 0
#         else:
#             raise ValueError(f"Unknown cpu_mode: {self.cpu_mode}")

#         # -----------------------------------------------------
#         # 3) Delay / tx-energy / cmp-energy
#         # -----------------------------------------------------
#         metrics = compute_delay_and_energy(
#             state=self.state,
#             access_assoc=access_assoc,
#             offload_ratio=offload_ratio,
#             sched_beta=sched_beta,
#             bw_alloc=bw_alloc,
#             cpu_alloc=cpu_alloc,
#         )

#         # -----------------------------------------------------
#         # 4) Flight energy
#         # -----------------------------------------------------
#         energy_fly = self._compute_flight_energy(move_dist)
#         metrics["energy_fly"] = energy_fly
#         metrics["energy_sys"] = float(metrics["energy_sys"] + np.sum(energy_fly))

#         # -----------------------------------------------------
#         # 5) Feasibility under current slot
#         # -----------------------------------------------------
#         report = check_feasibility(
#             state=self.state,
#             access_assoc=access_assoc,
#             offload_ratio=offload_ratio,
#             sched_beta=sched_beta,
#             bw_alloc=bw_alloc,
#             cpu_alloc=cpu_alloc,
#             metrics=metrics,
#         )

#         # add extra mobility / battery feasibility checks
#         extra_report = self._check_extra_constraints(
#             move_dist=move_dist,
#             move_angle=move_angle,
#             energy_fly=energy_fly,
#             energy_tx=metrics.get("energy_tx", np.zeros(self.M)),
#             energy_cmp=metrics.get("energy_cmp", np.zeros(self.M)),
#         )
#         report = self._merge_reports(report, extra_report)

#         # -----------------------------------------------------
#         # 6) Reward
#         # r_t = - omega1 * T_sys^t - omega2 * E_sys^t - zeta * penalty
#         # -----------------------------------------------------
#         # penalty_value = self._compute_penalty(report)
#         # reward = -self.omega1 * float(metrics["delay_sys"]) \
#         #          - self.omega2 * float(metrics["energy_sys"]) \
#         #          - self.penalty_coeff * penalty_value

#         penalty_value = self._compute_penalty(report)

#         offload_mean = float(np.mean(offload_ratio))
#         deadline_pressure = float(report.get("deadline_violation", 0.0))
#         offload_bonus = 5.0 * max(0.0, offload_mean - 0.05) * (1.0 + deadline_pressure)

#         reward = (
#             -self.omega1 * float(metrics["delay_sys"])
#             - self.omega2 * float(metrics["energy_sys"])
#             - self.penalty_coeff * penalty_value
#             + offload_bonus
#         )

#         # -----------------------------------------------------
#         # 7) Update UAV battery after current slot
#         # -----------------------------------------------------
#         uav_energy_before = self.state["uav_energy"].copy()
#         self._update_uav_energy(metrics)
#         uav_energy_after = self.state["uav_energy"].copy()


#         # -----------------------------------------------------
#         # 8) Move UAVs to next slot
#         # -----------------------------------------------------
#         self._update_uav_positions(move_dist, move_angle)

#         # -----------------------------------------------------
#         # 9) Advance slot index
#         # -----------------------------------------------------
#         self.t += 1
#         done = bool(self.t >= self.episode_length)

#         # -----------------------------------------------------
#         # 10) Generate next-slot tasks and recompute channels/topology
#         # -----------------------------------------------------
#         if not done:
#             self._refresh_task_state()
#             self._recompute_channels_and_topology()

#         next_obs = self._build_observation()

#         info = {
#             "slot": self.t,
#             "access_assoc": access_assoc,
#             "bw_alloc": bw_alloc,
#             "cpu_alloc": cpu_alloc,
#             "metrics": metrics,
#             "report": report,
#             "penalty_value": penalty_value,
#             "total_bisect_iters": total_bisect_iters,
#             "high_action": action,
#             "uav_energy_before": uav_energy_before,
#             "uav_energy_after": uav_energy_after,
#         }

#         return next_obs, float(reward), done, info

#     # =========================================================
#     # Initial State Generation
#     # =========================================================
#     def _generate_initial_state(self) -> Dict[str, Any]:
#         """
#         Generate initial state for one episode.
#         UAV geometry and resource budgets are initialized here.
#         """
#         M = self.M
#         K = self.K

#         # -----------------------------
#         # 1) UAV / TD positions
#         # -----------------------------
#         uav_pos = self.rng.uniform(0.0, self.area_size, size=(M, 2))
#         td_pos = self.rng.uniform(0.0, self.area_size, size=(K, 2))

#         # -----------------------------
#         # 2) Task parameters
#         # -----------------------------
#         task_size = self.rng.uniform(5.0, 20.0, size=(K,))
#         task_cycles = self.rng.uniform(500.0, 1500.0, size=(K,))
#         task_deadline = self.deadline_scale * self.rng.uniform(5.0, 20.0, size=(K,))
#         # task_deadline = self.deadline_scale * self.rng.uniform(3.0, 20.0, size=(K,))
#         # task_deadline = self.deadline_scale * self.rng.uniform(8.0, 20.0, size=(K,))
       
#         # task_local_cpu = self.rng.uniform(3.0e3, 8.0e3, size=(K,))

#         task_local_cpu = self.rng.uniform(
#             self.task_local_cpu_min,
#             self.task_local_cpu_max,
#             size=(K,),
#         )
#         # -----------------------------
#         # 3) UAV resources
#         # -----------------------------
#         # uav_energy = self.rng.uniform(80.0, 120.0, size=(M,))
#         print("DEBUG DEFAULT uav_cpu_min =", self.uav_cpu_min)
#         print("DEBUG DEFAULT uav_cpu_max_init =", self.uav_cpu_max_init)               
 
#         uav_energy = self.rng.uniform(self.uav_energy_min, self.uav_energy_max, size=(M,))
#         # uav_cpu_max = self.rng.uniform(2.0e4, 5.0e4, size=(M,))
#         uav_cpu_max = self.rng.uniform(
#             self.uav_cpu_min,
#             self.uav_cpu_max_init,
#             size=(M,),
#         )
#         print("DEBUG sampled uav_cpu_max[:3] =", uav_cpu_max[:3])
#         print("DEBUG sampled uav_cpu_max min/max =", uav_cpu_max.min(), uav_cpu_max.max())

#         uav_cpu_avail = uav_cpu_max.copy()
#         B_max = self.rng.uniform(10.0, 30.0, size=(M,))
#         kappa_vec = self.rng.uniform(1.0e-6, 5.0e-6, size=(M,))

#         # -----------------------------
#         # 4) Communication params
#         # -----------------------------
#         # noise_power = 1e-3
#         noise_power = 1e-8
#         td_tx_power = self.rng.uniform(0.5, 1.0, size=(K,))
#         uav_tx_power = self.rng.uniform(1.0, 2.0, size=(M,))
#         backhaul_bandwidth = 10.0
#         beta0_a2g = 50.0
#         beta0_a2a = 50.0

#         state = {
#             "M": M,
#             "K": K,

#             # episode / physical params
#             "delta_t": self.delta_t,
#             "area_size": self.area_size,
#             "altitude": self.altitude,
#             "neighbor_radius": self.neighbor_radius,
#             "max_speed": self.max_speed,
#             "min_uav_distance": self.min_uav_distance,

#             # geometry
#             "uav_pos": uav_pos,
#             "td_pos": td_pos,

#             # tasks
#             "task_size": task_size,
#             "task_cycles": task_cycles,
#             "task_deadline": task_deadline,
#             "task_local_cpu": task_local_cpu,

#             # uav resources
#             "uav_energy": uav_energy,
#             "uav_cpu_max": uav_cpu_max,
#             "uav_cpu_avail": uav_cpu_avail,
#             "B_max": B_max,
#             "kappa_vec": kappa_vec,

#             # communication params
#             "noise_power": noise_power,
#             "td_tx_power": td_tx_power,
#             "uav_tx_power": uav_tx_power,
#             "backhaul_bandwidth": backhaul_bandwidth,
#             "R_min": self.R_min,
#             "beta0_a2g": beta0_a2g,
#             "beta0_a2a": beta0_a2a,

#             # rotor-wing propulsion model parameters
#             "P0": 79.86,
#             "Pi": 88.63,
#             "U_tip": 120.0,
#             "v0": 4.03,
#             "d0": 0.6,
#             "rho_air": 1.225,
#             "rotor_solidity": 0.05,
#             "rotor_disc_area": 0.503,

#             # paper-like A2G params
#             "carrier_freq": 2.0e9,
#             "los_a": 9.61,
#             "los_b": 0.16,
#             "eta_los_db": 1.0,
#             "eta_nlos_db": 20.0,

#             # lightweight reuse interference factor
#             "reuse_interference_factor": 0.15,
#         }


#         # derive topology / channels
#         self._fill_topology_and_channel_fields(state)
#         return state

#     # =========================================================
#     # Observation
#     # =========================================================
#     def _build_observation(self) -> Dict[str, Any]:
#         """
#         Current version returns a dict observation.
#         Later, this can be converted into per-agent observation tensors.
#         """
#         if self.state is None:
#             raise RuntimeError("State is None.")

#         obs = {
#             "slot": self.t,
#             "uav_pos": self.state["uav_pos"].copy(),
#             "uav_energy": self.state["uav_energy"].copy(),
#             "uav_cpu_avail": self.state["uav_cpu_avail"].copy(),
#             "td_pos": self.state["td_pos"].copy(),
#             "task_size": self.state["task_size"].copy(),
#             "task_cycles": self.state["task_cycles"].copy(),
#             "task_deadline": self.state["task_deadline"].copy(),
#             "task_local_cpu": self.state["task_local_cpu"].copy(),
#             "gain_a2g": self.state["gain_a2g"].copy(),
#             "sinr_up": self.state["sinr_up"].copy(),
#             "rate_up": self.state["rate_up"].copy(),
#             "dist_a2a": self.state["dist_a2a"].copy(),
#             "gain_a2a": self.state["gain_a2a"].copy(),
#             "rate_a2a": self.state["rate_a2a"].copy(),
#             "neighbors": [list(x) for x in self.state["neighbors"]],
#             "raw_state": self.state,
#         }
#         return obs

#     # =========================================================
#     # Action Processing
#     # =========================================================
#     def _sanitize_high_action(self, high_action: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
#         """
#         Ensure action values are legal and have correct shapes.
#         """
#         if self.state is None:
#             raise RuntimeError("State is None.")

#         M = self.M
#         K = self.K
#         neighbors = self.state["neighbors"]

#         move_dist = np.asarray(high_action["move_dist"], dtype=float).reshape(M)
#         move_angle = np.asarray(high_action["move_angle"], dtype=float).reshape(M)
#         offload_ratio = np.asarray(high_action["offload_ratio"], dtype=float).reshape(K)
#         sched_beta = np.asarray(high_action["sched_beta"], dtype=float).reshape(K, M, M)

#         # movement clipping
#         max_move = self.max_speed * self.delta_t
#         move_dist = np.clip(move_dist, 0.0, max_move)

#         # angle wrap to [-pi, pi]
#         move_angle = (move_angle + np.pi) % (2.0 * np.pi) - np.pi

#         # lambda clip
#         offload_ratio = np.clip(offload_ratio, 0.0, 1.0)

#         # sched_beta legalize:
#         # for each task k, only associated UAV m can choose j in J_m
#         access_assoc = build_access_association(self.state)
#         sched_beta_clean = np.zeros((K, M, M), dtype=float)

#         for k in range(K):
#             access_m = int(np.argmax(access_assoc[:, k]))
#             legal_js = [access_m] + list(neighbors[access_m])

#             raw_scores = sched_beta[k, access_m, legal_js]
#             best_local_idx = int(np.argmax(raw_scores))
#             j_star = int(legal_js[best_local_idx])

#             sched_beta_clean[k, access_m, j_star] = 1.0

#         return {
#             "move_dist": move_dist,
#             "move_angle": move_angle,
#             "offload_ratio": offload_ratio,
#             "sched_beta": sched_beta_clean,
#         }

#     # =========================================================
#     # Dynamics / State Evolution
#     # =========================================================
#     def _update_uav_positions(self, move_dist: np.ndarray, move_angle: np.ndarray) -> None:
#         """
#         Update q_{m}^{t+1} = q_m^t + d_m^t [cos theta, sin theta]
#         and clip into the area.
#         """
#         if self.state is None:
#             raise RuntimeError("State is None.")

#         pos = self.state["uav_pos"]

#         dx = move_dist * np.cos(move_angle)
#         dy = move_dist * np.sin(move_angle)

#         pos[:, 0] += dx
#         pos[:, 1] += dy

#         pos[:, 0] = np.clip(pos[:, 0], 0.0, self.area_size)
#         pos[:, 1] = np.clip(pos[:, 1], 0.0, self.area_size)

#         self.state["uav_pos"] = pos

#     def _refresh_task_state(self) -> None:
#         """
#         Generate next-slot task set.
#         For now, tasks are re-sampled every slot, which matches the paper's
#         per-slot task generation assumption.
#         """
#         if self.state is None:
#             raise RuntimeError("State is None.")

#         K = self.K

#         self.state["td_pos"] = self.rng.uniform(0.0, self.area_size, size=(K, 2))
#         self.state["task_size"] = self.rng.uniform(5.0, 20.0, size=(K,))
#         self.state["task_cycles"] = self.rng.uniform(500.0, 1500.0, size=(K,))
#         self.state["task_deadline"] = self.deadline_scale * self.rng.uniform(5.0, 20.0, size=(K,))
#         # self.state["task_local_cpu"] = self.rng.uniform(3.0e3, 8.0e3, size=(K,))
#         self.state["task_local_cpu"] = self.rng.uniform(
#             self.task_local_cpu_min,
#             self.task_local_cpu_max,
#             size=(K,),
#         )
#         self.state["td_tx_power"] = self.rng.uniform(0.5, 1.0, size=(K,))

#         # CPU availability resets per slot in current simplified version
#         self.state["uav_cpu_avail"] = self.state["uav_cpu_max"].copy()

#     def _recompute_channels_and_topology(self) -> None:
#         """
#         Recompute A2A/A2G quantities after movement / task refresh.
#         """
#         if self.state is None:
#             raise RuntimeError("State is None.")

#         self._fill_topology_and_channel_fields(self.state)

#     # =========================================================
#     # Topology / Channel Computation
#     # =========================================================
#     # def _fill_topology_and_channel_fields(self, state: Dict[str, Any]) -> None:
#     #     """
#     #     Compute neighbors, A2A channels, A2G channels, rates, etc.
#     #     """
#     def _fill_topology_and_channel_fields(self, state: Dict[str, Any]) -> None:
#         """
#         Compute topology and channel-related fields in-place.

#         All A2G/A2A modeling is delegated to env.channel_model.fill_channel_state
#         to avoid duplicated logic and accidental overwriting.
#         """
#         fill_channel_state(state)

#     # def _fill_topology_and_channel_fields(self, state: Dict[str, Any]) -> None:
#     #     """
#     #     Compute neighbors, A2A channels, A2G channels, rates, etc.
#     #     Delegated to env.channel_model for better modularity and
#     #     stronger alignment with the paper system model.
#     #     """
#     #     fill_channel_state(state)
#     #     M = state["M"]
#     #     K = state["K"]

#     #     uav_pos = state["uav_pos"]
#     #     td_pos = state["td_pos"]
#     #     altitude = state["altitude"]
#     #     neighbor_radius = state["neighbor_radius"]

#     #     noise_power = state["noise_power"]
#     #     uav_tx_power = state["uav_tx_power"]
#     #     td_tx_power = state["td_tx_power"]
#     #     backhaul_bandwidth = state["backhaul_bandwidth"]
#     #     beta0_a2g = state["beta0_a2g"]
#     #     beta0_a2a = state["beta0_a2a"]

#     #     # -----------------------------
#     #     # A2A
#     #     # -----------------------------
#     #     neighbors = []
#     #     dist_a2a = np.zeros((M, M), dtype=float)
#     #     gain_a2a = np.zeros((M, M), dtype=float)
#     #     rate_a2a = np.zeros((M, M), dtype=float)

#     #     for m in range(M):
#     #         nb = []
#     #         for j in range(M):
#     #             if m == j:
#     #                 continue

#     #             d = float(np.linalg.norm(uav_pos[m] - uav_pos[j]))
#     #             dist_a2a[m, j] = d

#     #             if d <= neighbor_radius:
#     #                 nb.append(j)

#     #             gain = beta0_a2a / max(d * d, 1.0)
#     #             gain_a2a[m, j] = gain

#     #             snr = float(uav_tx_power[m]) * gain / max(noise_power, EPS)
#     #             rate_a2a[m, j] = float(backhaul_bandwidth) * math.log2(1.0 + max(snr, EPS))

#     #         neighbors.append(nb)

#     #     # -----------------------------
#     #     # A2G
#     #     # -----------------------------
#     #     dist_a2g = np.zeros((M, K), dtype=float)
#     #     elev_angle = np.zeros((M, K), dtype=float)
#     #     gain_a2g = np.zeros((M, K), dtype=float)
#     #     sinr_up = np.zeros((M, K), dtype=float)
#     #     rate_up = np.zeros((M, K), dtype=float)

#     #     for m in range(M):
#     #         for k in range(K):
#     #             horizontal_d = float(np.linalg.norm(uav_pos[m] - td_pos[k]))
#     #             d = math.sqrt(horizontal_d ** 2 + altitude ** 2)
#     #             dist_a2g[m, k] = d

#     #             elev_angle[m, k] = 180.0 / math.pi * math.asin(altitude / max(d, EPS))

#     #             gain = beta0_a2g / max(d * d, 1.0)
#     #             gain_a2g[m, k] = gain

#     #             sinr = float(td_tx_power[k]) * gain / max(noise_power, EPS)
#     #             sinr_up[m, k] = sinr
#     #             rate_up[m, k] = math.log2(1.0 + max(sinr, EPS))

#     #     state["neighbors"] = neighbors
#     #     state["dist_a2a"] = dist_a2a
#     #     state["gain_a2a"] = gain_a2a
#     #     state["rate_a2a"] = rate_a2a

#     #     state["dist_a2g"] = dist_a2g
#     #     state["elev_angle"] = elev_angle
#     #     state["gain_a2g"] = gain_a2g
#     #     state["sinr_up"] = sinr_up
#     #     state["rate_up"] = rate_up

#     # =========================================================
#     # Energy / Reward / Penalty
#     # =========================================================
#     def _compute_flight_energy(self, move_dist: np.ndarray) -> np.ndarray:
#         """
#         Paper-aligned rotor-wing propulsion energy model:
#             E_fly = P_fly(v) * delta_t
#         """
#         if self.state is None:
#             raise RuntimeError("State is None.")

#         return compute_flight_energy(self.state, move_dist)
#     # def _compute_flight_energy(self, move_dist: np.ndarray) -> np.ndarray:
#     #     """
#     #     Simplified flight energy:
#     #         E_fly = P(v) * delta_t

#     #     Here we use a stable simplified polynomial surrogate rather than the
#     #     full rotor-wing expression, to first make the main pipeline run.
#     #     Later, this can be replaced by the exact paper model.
#     #     """
#     #     v = move_dist / max(self.delta_t, EPS)

#     #     # simplified propulsion power surrogate
#     #     # strictly positive and increasing with speed
#     #     p_hover = 8.0
#     #     p_linear = 0.25 * v
#     #     p_quad = 0.03 * (v ** 2)

#     #     power = p_hover + p_linear + p_quad
#     #     energy_fly = power * self.delta_t
#     #     return energy_fly.astype(float)

#     def _update_uav_energy(self, metrics: Dict[str, Any]) -> None:
#         if self.state is None:
#             raise RuntimeError("State is None.")

#         energy_fly = metrics.get("energy_fly", np.zeros((self.M,), dtype=float))
#         energy_tx = metrics.get("energy_tx", np.zeros((self.M,), dtype=float))
#         energy_cmp = metrics.get("energy_cmp", np.zeros((self.M,), dtype=float))

#         total_cost = compute_total_uav_energy_cost(
#             state=self.state,
#             energy_fly=energy_fly,
#             energy_tx=energy_tx,
#             energy_cmp=energy_cmp,
#         )

#         self.state["uav_energy"] = self.state["uav_energy"] - total_cost
#         self.state["uav_energy"] = np.maximum(self.state["uav_energy"], 0.0)

#     # def _update_uav_energy(self, metrics: Dict[str, Any]) -> None:
#     #     if self.state is None:
#     #         raise RuntimeError("State is None.")

#     #     energy_fly = metrics.get("energy_fly", np.zeros((self.M,), dtype=float))
#     #     energy_tx = metrics.get("energy_tx", np.zeros((self.M,), dtype=float))
#     #     energy_cmp = metrics.get("energy_cmp", np.zeros((self.M,), dtype=float))

#     #     total_cost = energy_fly + energy_tx + energy_cmp
#     #     self.state["uav_energy"] = self.state["uav_energy"] - total_cost

#     #     # debug-friendly clipping
#     #     self.state["uav_energy"] = np.maximum(self.state["uav_energy"], 0.0)

#     def _compute_penalty(self, report: Dict[str, Any]) -> float:
#         """
#         Aggregate constraint violations into a scalar penalty.
#         """
#         penalty = 0.0
#         keys = [
#             "ratio_violation",
#             "assoc_violation",
#             "schedule_violation",
#             "candidate_violation",
#             "bw_violation",
#             "cpu_violation",
#             "rate_violation",
#             "deadline_violation",
#             "move_violation",
#             "boundary_violation",
#             "collision_violation",
#             "battery_violation",
#             "nan_count",
#         ]

#         for k in keys:
#             if k in report:
#                 penalty += float(report[k])

#         return penalty

#     # =========================================================
#     # Extra Constraints
#     # =========================================================
#     def _check_extra_constraints(
#         self,
#         move_dist: np.ndarray,
#         move_angle: np.ndarray,
#         energy_fly: np.ndarray,
#         energy_tx: np.ndarray,
#         energy_cmp: np.ndarray,
#     ) -> Dict[str, Any]:
#         """
#         Extra constraints not covered by current feasibility_check.py:
#         1) motion budget
#         2) boundary feasibility after movement
#         3) minimum inter-UAV safety distance after movement
#         4) battery non-negativity after one-slot energy spending
#         """
#         if self.state is None:
#             raise RuntimeError("State is None.")

#         report = {
#             "move_violation": 0.0,
#             "boundary_violation": 0.0,
#             "collision_violation": 0.0,
#             "battery_violation": 0.0,
#         }

#         max_move = self.max_speed * self.delta_t

#         # 1) motion budget
#         move_violation = max(0.0, float(np.max(move_dist) - max_move))
#         report["move_violation"] = move_violation

#         # predict next positions
#         cur_pos = self.state["uav_pos"]
#         next_pos = cur_pos.copy()
#         next_pos[:, 0] += move_dist * np.cos(move_angle)
#         next_pos[:, 1] += move_dist * np.sin(move_angle)

#         # 2) boundary
#         boundary_violation = 0.0
#         for m in range(self.M):
#             x, y = float(next_pos[m, 0]), float(next_pos[m, 1])
#             boundary_violation = max(boundary_violation, max(0.0, -x))
#             boundary_violation = max(boundary_violation, max(0.0, x - self.area_size))
#             boundary_violation = max(boundary_violation, max(0.0, -y))
#             boundary_violation = max(boundary_violation, max(0.0, y - self.area_size))
#         report["boundary_violation"] = boundary_violation

#         # 3) safety distance
#         collision_violation = 0.0
#         for m in range(self.M):
#             for j in range(m + 1, self.M):
#                 d = float(np.linalg.norm(next_pos[m] - next_pos[j]))
#                 violation = max(0.0, self.min_uav_distance - d)
#                 collision_violation = max(collision_violation, violation)
#         report["collision_violation"] = collision_violation

#         # 4) battery
#         remain_energy = self.state["uav_energy"] - energy_fly - energy_tx - energy_cmp
#         battery_violation = max(0.0, float(np.max(-remain_energy)))
#         report["battery_violation"] = battery_violation

#         return report

#     def _merge_reports(self, report_a: Dict[str, Any], report_b: Dict[str, Any]) -> Dict[str, Any]:
#         """
#         Merge feasibility reports and update 'ok'.
#         """
#         merged = dict(report_a)
#         merged.update(report_b)

#         merged["ok"] = True
#         for k, v in merged.items():
#             if k == "ok":
#                 continue
#             if float(v) > 1e-8:
#                 merged["ok"] = False
#                 break

#         return merged


# # =============================================================
# # Backward-Compatible Utility Functions
# # =============================================================
# def random_generate_state(
#     M: int = 3,
#     K: int = 16,
#     area_size: float = 100.0,
#     altitude: float = 20.0,
#     neighbor_radius: float = 50.0,
#     seed: Optional[int] = None,
#     R_min: float = 0.05,
#     deadline_scale: float = 5.0,
# ) -> Dict[str, Any]:
#     """
#     Backward-compatible helper.
#     Generates one state using the new environment internals.
#     """
#     env = MultiUavMecEnv(
#         M=M,
#         K=K,
#         episode_length=20,
#         area_size=area_size,
#         altitude=altitude,
#         neighbor_radius=neighbor_radius,
#         R_min=R_min,
#         deadline_scale=deadline_scale,
#         seed=seed,
#     )
#     env.reset(seed=seed)
#     if env.state is None:
#         raise RuntimeError("Failed to generate state.")
#     return env.state


# def pretty_print_state_summary(state: Dict[str, Any]) -> None:
#     """
#     Print a concise summary for debugging.
#     """
#     print("=" * 60)
#     print("Environment State Summary")
#     print("=" * 60)
#     print(f"M (UAVs): {state['M']}")
#     print(f"K (Tasks): {state['K']}")
#     print(f"Area size: {state['area_size']}")
#     print(f"Altitude: {state['altitude']}")
#     print(f"Neighbor radius: {state['neighbor_radius']}")
#     print(f"UAV positions shape: {state['uav_pos'].shape}")
#     print(f"TD positions shape: {state['td_pos'].shape}")
#     print(f"A2G gain shape: {state['gain_a2g'].shape}")
#     print(f"A2A gain shape: {state['gain_a2a'].shape}")
#     print(f"Current UAV energy: {state['uav_energy']}")
#     print("=" * 60)



# # import math
# # from typing import Dict, Any, Optional

# # import numpy as np

# # # 负责生成最小的单时隙状态：
# # # UAV的位置、TD位置、任务参数（D、C、Tmax）
# # # UAV 资源、A2G/A2A简化信道、邻居集
# # EPS = 1e-8


# # def random_generate_state(
# #     M: int = 3,
# #     K: int = 8,
# #     area_size: float = 100.0,
# #     altitude: float = 20.0,
# #     neighbor_radius: float = 50.0,
# #     seed: Optional[int] = None,
# #     R_min = 0.05,
# #     deadline_scale = 1.0,
# # ) -> Dict[str, Any]:
# #     """
# #     Generate a random single-slot environment state for the minimal feasibility test.

# #     This version is intentionally lightweight:
# #     - It is used to verify the solvability of the low-level analytical layer first.
# #     - It does not yet implement the full training environment.
# #     - It uses simplified A2G/A2A channel calculations to keep the first-stage test stable.
# #     Args:
# #         M: number of UAVs
# #         K: number of tasks / TDs
# #         area_size: 2D square area side length
# #         altitude: fixed UAV altitude H
# #         neighbor_radius: communication range for feasible A2A cooperation
# #         seed: random seed

# #     Returns:
# #         state dict
# #     """
# #     rng = np.random.default_rng(seed)

# #     # -----------------------------
# #     # 1. UAV / TD positions
# #     # -----------------------------
# #     uav_pos = rng.uniform(0.0, area_size, size=(M, 2))   # q_m^t
# #     td_pos = rng.uniform(0.0, area_size, size=(K, 2))    # w_k

# #     # -----------------------------
# #     # 2. Task parameters
# #     # -----------------------------
# #     # D_k^t: input data size
# #     task_size = rng.uniform(5.0, 20.0, size=(K,))

# #     # C_k^t: required CPU cycles per bit
# #     task_cycles = rng.uniform(500.0, 1500.0, size=(K,))

# #     # tau_k^max
# #     # task_deadline = rng.uniform(0.5, 3.0, size=(K,))
# #     task_deadline = deadline_scale * rng.uniform(5.0, 20.0, size=(K,))


# #     # local CPU frequency at TD side (for local delay calculation later)
# #     # task_local_cpu = rng.uniform(1.0e3, 3.0e3, size=(K,))
# #     task_local_cpu = rng.uniform(3.0e3, 8.0e3, size=(K,))
    

# #     # -----------------------------
# #     # 3. UAV-side resources
# #     # -----------------------------
# #     uav_energy = rng.uniform(80.0, 120.0, size=(M,))
# #     # uav_cpu_max = rng.uniform(5.0e3, 1.5e4, size=(M,))
# #     uav_cpu_max = rng.uniform(2.0e4, 5.0e4, size=(M,))
# #     uav_cpu_avail = uav_cpu_max.copy()

# #     # uplink bandwidth budget B_m^max
# #     B_max = rng.uniform(10.0, 30.0, size=(M,))

# #     # switched-capacitance coefficient kappa_j
# #     kappa_vec = rng.uniform(1.0e-6, 5.0e-6, size=(M,))

# #     # -----------------------------
# #     # 4. Simple communication parameters
# #     # -----------------------------
# #     # These are placeholders for the minimal version.
# #     # noise_power = 1.0
# #     # td_tx_power = rng.uniform(2.0, 5.0, size=(K,))
# #     # uav_tx_power = rng.uniform(5.0, 10.0, size=(M,))
# #     # backhaul_bandwidth = 5.0
# #     noise_power = 1e-3
# #     td_tx_power = rng.uniform(0.5, 1.0, size=(K,))
# #     uav_tx_power = rng.uniform(1.0, 2.0, size=(M,))
# #     backhaul_bandwidth = 10.0
# #     beta0_a2g = 50.0
# #     beta0_a2a = 50.0

# #     # minimum uplink rate threshold for active access links
# #     R_min = float(R_min)

# #     # -----------------------------
# #     # 5. A2A distances / gains / rates
# #     # -----------------------------
# #     neighbors = []
# #     dist_a2a = np.zeros((M, M), dtype=float)
# #     gain_a2a = np.zeros((M, M), dtype=float)
# #     rate_a2a = np.zeros((M, M), dtype=float)

# #     for m in range(M):
# #         nb = []
# #         for j in range(M):
# #             if m == j:
# #                 continue

# #             d = float(np.linalg.norm(uav_pos[m] - uav_pos[j]))
# #             dist_a2a[m, j] = d

# #             if d <= neighbor_radius:
# #                 nb.append(j)

# #             # simplified gain: beta0 * d^-2, beta0=1
# #             gain = beta0_a2a / max(d * d, 1.0)
# #             gain_a2a[m, j] = gain

# #             snr = uav_tx_power[m] * gain / max(noise_power, EPS)
# #             rate_a2a[m, j] = backhaul_bandwidth * math.log2(1.0 + max(snr, EPS))

# #         neighbors.append(nb)

# #     # -----------------------------
# #     # 6. A2G distances / gains / rates
# #     # -----------------------------
# #     dist_a2g = np.zeros((M, K), dtype=float)
# #     elev_angle = np.zeros((M, K), dtype=float)
# #     gain_a2g = np.zeros((M, K), dtype=float)
# #     rate_up = np.zeros((M, K), dtype=float)
# #     sinr_up = np.zeros((M, K), dtype=float)

# #     for m in range(M):
# #         for k in range(K):
# #             horizontal_d = float(np.linalg.norm(uav_pos[m] - td_pos[k]))
# #             d = math.sqrt(horizontal_d**2 + altitude**2)
# #             dist_a2g[m, k] = d

# #             # elevation angle in degrees
# #             elev_angle[m, k] = 180.0 / math.pi * math.asin(altitude / max(d, EPS))

# #             # simplified average gain: 1 / d^2
# #             gain = beta0_a2g / max(d * d, 1.0)
# #             gain_a2g[m, k] = gain

# #             # simplified SINR / rate
# #             sinr = td_tx_power[k] * gain / max(noise_power, EPS)
# #             sinr_up[m, k] = sinr
# #             rate_up[m, k] = math.log2(1.0 + max(sinr, EPS))

# #     state = {
# #         # sizes
# #         "M": M,
# #         "K": K,

# #         # geometry
# #         "area_size": area_size,
# #         "altitude": altitude,
# #         "neighbor_radius": neighbor_radius,
# #         "uav_pos": uav_pos,
# #         "td_pos": td_pos,

# #         # tasks
# #         "task_size": task_size,
# #         "task_cycles": task_cycles,
# #         "task_deadline": task_deadline,
# #         "task_local_cpu": task_local_cpu,

# #         # uav-side resources
# #         "uav_energy": uav_energy,
# #         "uav_cpu_max": uav_cpu_max,
# #         "uav_cpu_avail": uav_cpu_avail,
# #         "B_max": B_max,
# #         "kappa_vec": kappa_vec,

# #         # communication params
# #         "noise_power": noise_power,
# #         "td_tx_power": td_tx_power,
# #         "uav_tx_power": uav_tx_power,
# #         "backhaul_bandwidth": backhaul_bandwidth,
# #         "R_min": R_min,

# #         # topology and channels
# #         "neighbors": neighbors,
# #         "dist_a2g": dist_a2g,
# #         "elev_angle": elev_angle,
# #         "gain_a2g": gain_a2g,
# #         "sinr_up": sinr_up,
# #         "rate_up": rate_up,
# #         "dist_a2a": dist_a2a,
# #         "gain_a2a": gain_a2a,
# #         "rate_a2a": rate_a2a,
# #     }

# #     return state


# # def pretty_print_state_summary(state: Dict[str, Any]) -> None:
# #     """
# #     Print a concise summary for debugging.
# #     """
# #     print("=" * 60)
# #     print("Environment State Summary")
# #     print("=" * 60)
# #     print(f"M (UAVs): {state['M']}")
# #     print(f"K (Tasks): {state['K']}")
# #     print(f"Area size: {state['area_size']}")
# #     print(f"Altitude: {state['altitude']}")
# #     print(f"Neighbor radius: {state['neighbor_radius']}")
# #     print(f"UAV positions shape: {state['uav_pos'].shape}")
# #     print(f"TD positions shape: {state['td_pos'].shape}")
# #     print(f"A2G gain shape: {state['gain_a2g'].shape}")
# #     print(f"A2A gain shape: {state['gain_a2a'].shape}")
# #     print("=" * 60)