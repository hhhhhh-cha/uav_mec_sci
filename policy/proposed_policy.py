# # # 这个文件的职责是把下面几件事真正串起来：
# # # 调用 transformer_encoder.py
# # # 生成每个 task 的 context
# # # 生成 offloading ratio
# # # 调用 actor 输出 mobility / scheduling
# # # 调用 bandwidth_solver.py 和 cpu_solver.py
# # # 最终输出完整的 proposed action / metrics 接口

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from model.transformer_encoder import (
    MaskedTransformerEncoder,
    RatioHead,
    TaskContextFusion,
)
from policy.proposed_learned_policy import ProposedLearnedPolicy
from policy.proposed_placeholder_policy import generate_proposed_placeholder_action

EPS = 1e-8


def _safe_get(state: Dict[str, Any], key: str, default=None):
    return state[key] if key in state else default


def _to_numpy(x, dtype=np.float32):
    return np.asarray(x, dtype=dtype)


def _extract_uav_positions(state: Dict[str, Any]) -> np.ndarray:
    for key in ["uav_pos", "uav_positions", "q", "uav_xy"]:
        if key in state:
            arr = np.asarray(state[key], dtype=np.float32)
            if arr.ndim == 2 and arr.shape[1] >= 2:
                return arr[:, :2]
    raise KeyError("Cannot find UAV position array in state.")


def _extract_task_positions(state: Dict[str, Any]) -> Optional[np.ndarray]:
    for key in ["task_pos", "task_positions", "td_pos", "user_pos", "w"]:
        if key in state:
            arr = np.asarray(state[key], dtype=np.float32)
            if arr.ndim == 2 and arr.shape[1] >= 2:
                return arr[:, :2]
    return None


def _extract_neighbors(state: Dict[str, Any]) -> List[List[int]]:
    neighbors = _safe_get(state, "neighbors", None)
    if neighbors is None:
        M = int(state["M"])
        return [[] for _ in range(M)]
    return neighbors


def _extract_task_size(state: Dict[str, Any]) -> np.ndarray:
    for key in ["task_size", "D", "D_k", "task_data_size"]:
        if key in state:
            return _to_numpy(state[key])
    raise KeyError("Cannot find task size in state.")


def _extract_task_cycles(state: Dict[str, Any]) -> np.ndarray:
    for key in ["task_cycles", "C", "C_k", "task_cpu_cycles"]:
        if key in state:
            return _to_numpy(state[key])
    raise KeyError("Cannot find task cycles in state.")


def _extract_task_deadline(state: Dict[str, Any]) -> np.ndarray:
    for key in ["task_deadline", "deadline", "tau_max", "task_max_delay"]:
        if key in state:
            return _to_numpy(state[key])
    raise KeyError("Cannot find task deadline in state.")


def _extract_available_cpu(state: Dict[str, Any]) -> np.ndarray:
    for key in ["uav_available_cpu", "available_cpu", "uav_cpu_avail", "uav_cpu_max"]:
        if key in state:
            return _to_numpy(state[key])
    M = int(state["M"])
    return np.ones(M, dtype=np.float32)


def _extract_workload(state: Dict[str, Any]) -> np.ndarray:
    for key in ["uav_workload", "workload", "uav_load", "queue_len"]:
        if key in state:
            return _to_numpy(state[key])
    M = int(state["M"])
    return np.zeros(M, dtype=np.float32)


def _extract_tx_power(state: Dict[str, Any], access_m: int) -> float:
    for key in ["uav_tx_power", "P_tx", "tx_power"]:
        if key in state:
            val = state[key]
            arr = np.asarray(val, dtype=np.float32)
            if arr.ndim == 0:
                return float(arr)
            return float(arr[access_m])
    return 1.0


def _extract_backhaul_bw(state: Dict[str, Any]) -> float:
    for key in ["backhaul_bandwidth", "B_bh", "bh_bandwidth"]:
        if key in state:
            return float(state[key])
    return 1.0


def _extract_noise_power(state: Dict[str, Any]) -> float:
    for key in ["noise_power", "sigma2", "noise_var"]:
        if key in state:
            return float(state[key])
    return 1e-9


def _compute_a2a_gain_and_rate(
    state: Dict[str, Any],
    access_m: int,
    exec_j: int,
) -> Tuple[float, float]:
    """
    Compute approximate A2A channel gain and rate for token construction.
    """
    if exec_j == access_m:
        return 1.0, 1e6

    uav_pos = _extract_uav_positions(state)
    p_m = uav_pos[access_m]
    p_j = uav_pos[exec_j]

    dist = float(np.linalg.norm(p_m - p_j))
    dist = max(dist, 1.0)

    beta0 = float(_safe_get(state, "beta0", 1.0))
    gain = beta0 * (dist ** -2)

    B_bh = _extract_backhaul_bw(state)
    P_tx = _extract_tx_power(state, access_m)
    sigma2 = _extract_noise_power(state)

    rate = B_bh * np.log2(1.0 + P_tx * gain / max(sigma2, EPS))
    return float(gain), float(rate)


def _import_bandwidth_solver():
    try:
        from solver.bandwidth_solver import solve_bandwidth_allocation
        return solve_bandwidth_allocation
    except Exception:
        return None


def _import_cpu_solver():
    candidates = [
        "solve_cpu_allocation_kkt",
        "solve_cpu_allocation",
        "solve_cpu_allocation_proportional",
    ]
    try:
        import solver.cpu_solver as cpu_solver_module
        for name in candidates:
            if hasattr(cpu_solver_module, name):
                return getattr(cpu_solver_module, name)
    except Exception:
        pass
    return None


def _fallback_bandwidth_allocation(
    state: Dict[str, Any],
    access_assoc: np.ndarray,
    offload_ratio: np.ndarray,
) -> np.ndarray:
    M, K = access_assoc.shape
    task_size = _extract_task_size(state)
    Bmax = _safe_get(state, "uav_bandwidth_max", _safe_get(state, "bandwidth_max", 1.0))
    Bmax = np.asarray(Bmax, dtype=np.float32)
    if Bmax.ndim == 0:
        Bmax = np.full(M, float(Bmax), dtype=np.float32)

    bw = np.zeros((M, K), dtype=np.float32)

    for m in range(M):
        task_idx = np.where(access_assoc[m] > 0.5)[0]
        if len(task_idx) == 0:
            continue

        weights = np.asarray(
            [max(float(offload_ratio[k] * task_size[k]), EPS) for k in task_idx],
            dtype=np.float32,
        )
        weights = weights / max(weights.sum(), EPS)

        for i, k in enumerate(task_idx):
            bw[m, k] = Bmax[m] * weights[i]

    return bw


def _fallback_cpu_allocation(
    state: Dict[str, Any],
    access_assoc: np.ndarray,
    sched_beta: np.ndarray,
    offload_ratio: np.ndarray,
) -> np.ndarray:
    M, K = access_assoc.shape
    task_size = _extract_task_size(state)
    task_cycles = _extract_task_cycles(state)
    cpu_max = _extract_available_cpu(state)

    cpu_alloc = np.zeros((M, K), dtype=np.float32)

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

        workloads = np.asarray(workloads, dtype=np.float32)
        weights = workloads / max(workloads.sum(), EPS)

        for i, k in enumerate(exec_tasks):
            cpu_alloc[j, k] = cpu_max[j] * weights[i]

    return cpu_alloc


class ProposedPolicy:
    """
    Full proposed-method interface aligned with the paper structure:

    1) build candidate execution set
    2) topology-aware context encoding (Transformer)
    3) high-level decisions:
         - mobility
         - offloading ratio
         - collaborative scheduling
    4) low-level analytical resource allocation:
         - bandwidth
         - CPU
    """

    def __init__(
        self,
        mode: str = "placeholder",
        actor_net: Optional[torch.nn.Module] = None,
        encoder: Optional[MaskedTransformerEncoder] = None,
        ratio_head: Optional[RatioHead] = None,
        fusion_net: Optional[TaskContextFusion] = None,
        device: str = "cpu",
        max_candidates: Optional[int] = None,
        ratio_floor: float = 0.05,
        ratio_ceiling: float = 0.50,
        forward_bias: float = 1.0,
        local_keep_margin: float = 0.15,
        conservative_ratio_cap_base: float = 0.12,
        conservative_ratio_cap_min: float = 0.06,
        conservative_ratio_cap_max: float = 0.18,
    ):
        self.mode = str(mode)
        self.actor_net = actor_net
        self.encoder = encoder
        self.ratio_head = ratio_head
        self.fusion_net = fusion_net
        self.device = device
        self.max_candidates = max_candidates
        self.ratio_floor = float(ratio_floor)
        self.ratio_ceiling = float(ratio_ceiling)
        self.forward_bias = float(forward_bias)
        self.local_keep_margin = float(local_keep_margin)
        self.conservative_ratio_cap_base = float(conservative_ratio_cap_base)
        self.conservative_ratio_cap_min = float(conservative_ratio_cap_min)
        self.conservative_ratio_cap_max = float(conservative_ratio_cap_max)

        self.learned_policy = ProposedLearnedPolicy(
            mode="network" if actor_net is not None else "placeholder",
            actor_net=actor_net,
            device=device,
        )

        self.solve_bandwidth_allocation = _import_bandwidth_solver()
        self.solve_cpu_allocation = _import_cpu_solver()

    def act(
        self,
        state: Dict[str, Any],
        access_assoc: np.ndarray,
        deterministic: bool = True,
        return_aux: bool = False,
    ):
        """
        Main entry.

        Returns:
            action dict with keys:
              - move_dist
              - move_angle
              - offload_ratio
              - sched_beta
              - bandwidth_alloc
              - cpu_alloc
        """
        if self.mode == "placeholder":
            high_action = generate_proposed_placeholder_action(state, access_assoc)
            low_action = self._solve_low_level(
                state=state,
                access_assoc=access_assoc,
                offload_ratio=np.asarray(high_action["offload_ratio"], dtype=np.float32),
                sched_beta=np.asarray(high_action["sched_beta"], dtype=np.float32),
            )
            full_action = {**high_action, **low_action}
            if return_aux:
                return full_action, {}
            return full_action

        if self.mode != "network":
            raise ValueError(f"Unknown mode: {self.mode}")

        # Step 1: get base mobility + scheduling from learned policy
        base_high_action = self.learned_policy.act(
            state=state,
            access_assoc=access_assoc,
            deterministic=deterministic,
        )

        move_dist = np.asarray(base_high_action["move_dist"], dtype=np.float32)
        move_angle = np.asarray(base_high_action["move_angle"], dtype=np.float32)
        sched_beta = np.asarray(base_high_action["sched_beta"], dtype=np.float32)

        # Step 2: use Transformer context to generate offloading ratio
        if self.encoder is not None and self.ratio_head is not None and self.fusion_net is not None:
            offload_ratio, aux_info = self._build_offloading_ratio_from_context(
                state=state,
                access_assoc=access_assoc,
            )
        else:
            offload_ratio = np.asarray(base_high_action["offload_ratio"], dtype=np.float32)
            aux_info = {}

        # NOTE:
        # Deliberately disable hard post-processing/refinement here.
        # We want to observe the network's true offloading and scheduling outputs
        # without conservative caps or local-first overrides.

        high_action = {
            "move_dist": move_dist.astype(np.float32),
            "move_angle": move_angle.astype(np.float32),
            "offload_ratio": offload_ratio.astype(np.float32),
            "sched_beta": sched_beta.astype(np.float32),
        }

        low_action = self._solve_low_level(
            state=state,
            access_assoc=access_assoc,
            offload_ratio=offload_ratio,
            sched_beta=sched_beta,
        )

        full_action = {**high_action, **low_action}

        if return_aux:
            aux_info = dict(aux_info)
            aux_info["refined_offload_ratio"] = offload_ratio.copy()
            aux_info["refined_sched_beta"] = sched_beta.copy()
            return full_action, aux_info

        return full_action

    def _normalize_task_feature(self, state: Dict[str, Any], task_idx: int) -> np.ndarray:
        task_size = _extract_task_size(state)
        task_cycles = _extract_task_cycles(state)
        task_deadline = _extract_task_deadline(state)
        task_local_cpu = _to_numpy(state.get("task_local_cpu", np.ones_like(task_size)))

        size_scale = max(float(np.max(task_size)), 1.0)
        cycle_scale = max(float(np.max(task_cycles)), 1.0)
        deadline_scale = max(float(np.max(task_deadline)), 1.0)
        local_cpu_scale = max(float(np.max(task_local_cpu)), 1.0)

        return np.asarray(
            [
                float(task_size[task_idx]) / size_scale,
                float(task_cycles[task_idx]) / cycle_scale,
                float(task_deadline[task_idx]) / deadline_scale,
            ],
            dtype=np.float32,
        )

    def _normalize_uav_feature(self, state: Dict[str, Any], uav_idx: int) -> np.ndarray:
        area_size = max(float(state.get("area_size", 1.0)), 1.0)
        available_cpu = _extract_available_cpu(state)
        workload = _extract_workload(state)
        uav_pos = _extract_uav_positions(state)

        cpu_scale = max(float(np.max(available_cpu)), 1.0)
        workload_scale = max(float(np.max(workload)), 1.0)

        return np.asarray(
            [
                float(uav_pos[uav_idx, 0]) / area_size,
                float(uav_pos[uav_idx, 1]) / area_size,
                float(available_cpu[uav_idx]) / cpu_scale,
                float(workload[uav_idx]) / workload_scale,
            ],
            dtype=np.float32,
        )

    def _compute_ratio_prior(self, state: Dict[str, Any], access_assoc: np.ndarray) -> np.ndarray:
        K = int(state["K"])
        task_size = _extract_task_size(state)
        task_cycles = _extract_task_cycles(state)
        task_deadline = _extract_task_deadline(state)
        task_local_cpu = _to_numpy(state.get("task_local_cpu", np.ones(K, dtype=np.float32)))
        rate_up = np.asarray(state.get("rate_up", np.ones((int(state["M"]), K))), dtype=np.float32)

        global_rate_scale = max(float(np.max(rate_up)), EPS)
        prior = np.zeros((K,), dtype=np.float32)

        for k in range(K):
            access_m = int(np.argmax(access_assoc[:, k]))
            Dk = float(task_size[k])
            Ck = float(task_cycles[k])
            deadline = max(float(task_deadline[k]), EPS)
            f_loc = max(float(task_local_cpu[k]), EPS)

            local_only_delay = Dk * Ck / f_loc
            tightness = local_only_delay / deadline
            lam_prior = 0.10 + 0.30 * np.tanh(0.8 * (tightness - 1.0)) + 0.20

            if local_only_delay > deadline:
                lam_min = 1.0 - deadline * f_loc / max(Dk * Ck, EPS)
                lam_prior = max(lam_prior, lam_min + 0.10)

            uplink_norm = float(rate_up[access_m, k]) / global_rate_scale
            lam_prior += 0.10 * (uplink_norm - 0.5)
            prior[k] = float(np.clip(lam_prior, self.ratio_floor, self.ratio_ceiling))

        return prior

    def _build_offloading_ratio_from_context(
        self,
        state: Dict[str, Any],
        access_assoc: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        K = int(state["K"])

        token_batches = []
        mask_batches = []
        task_feat_batches = []
        uav_feat_batches = []
        task_indices = []

        Nc = self._get_max_candidates(state)
        ratio_prior = self._compute_ratio_prior(state, access_assoc)

        for k in range(K):
            access_m = int(np.argmax(access_assoc[:, k]))
            tokens, mask = self._build_candidate_tokens_for_task(
                state=state,
                task_idx=k,
                access_m=access_m,
                max_candidates=Nc,
            )

            task_feat = self._normalize_task_feature(state, k)
            uav_feat = self._normalize_uav_feature(state, access_m)

            token_batches.append(tokens)
            mask_batches.append(mask)
            task_feat_batches.append(task_feat)
            uav_feat_batches.append(uav_feat)
            task_indices.append((k, access_m))

        token_tensor = torch.tensor(np.asarray(token_batches), dtype=torch.float32, device=self.device)
        mask_tensor = torch.tensor(np.asarray(mask_batches), dtype=torch.float32, device=self.device)
        task_feat_tensor = torch.tensor(np.asarray(task_feat_batches), dtype=torch.float32, device=self.device)
        uav_feat_tensor = torch.tensor(np.asarray(uav_feat_batches), dtype=torch.float32, device=self.device)
        prior_tensor = torch.tensor(ratio_prior, dtype=torch.float32, device=self.device).unsqueeze(-1)

        with torch.no_grad():
            encoded_tokens, pooled_context = self.encoder(token_tensor, mask_tensor)
            fused_feature = self.fusion_net(
                topo_context=pooled_context,
                task_feat=task_feat_tensor,
                uav_feat=uav_feat_tensor,
            )
            offload_ratio_t = self.ratio_head(
                fused_feature,
                prior_ratio=prior_tensor,
                temperature=1.0,
                hard_min=self.ratio_floor,
            ).squeeze(-1)

        offload_ratio = offload_ratio_t.detach().cpu().numpy().astype(np.float32)
        offload_ratio = np.clip(offload_ratio, self.ratio_floor, self.ratio_ceiling)

        aux_info = {
            "encoded_tokens": encoded_tokens.detach().cpu().numpy(),
            "pooled_context": pooled_context.detach().cpu().numpy(),
            "fused_feature": fused_feature.detach().cpu().numpy(),
            "ratio_prior": ratio_prior.copy(),
            "final_offload_ratio": offload_ratio.copy(),
            "task_indices": task_indices,
        }
        return offload_ratio, aux_info

    def _build_candidate_tokens_for_task(
        self,
        state: Dict[str, Any],
        task_idx: int,
        access_m: int,
        max_candidates: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        neighbors = _extract_neighbors(state)
        task_size = _extract_task_size(state)
        task_cycles = _extract_task_cycles(state)
        task_deadline = _extract_task_deadline(state)
        available_cpu = _extract_available_cpu(state)
        workload = _extract_workload(state)

        size_scale = max(float(np.max(task_size)), 1.0)
        cycle_scale = max(float(np.max(task_cycles)), 1.0)
        deadline_scale = max(float(np.max(task_deadline)), 1.0)
        cpu_scale = max(float(np.max(available_cpu)), 1.0)
        workload_scale = max(float(np.max(workload)), 1.0)

        a2a_rate_vals = []
        for j in [access_m] + list(neighbors[access_m]):
            _, r_a2a = _compute_a2a_gain_and_rate(state=state, access_m=access_m, exec_j=j)
            a2a_rate_vals.append(float(r_a2a))
        a2a_rate_scale = max(max(a2a_rate_vals) if len(a2a_rate_vals) > 0 else 1.0, 1.0)

        candidates = [access_m] + list(neighbors[access_m])
        candidates = candidates[:max_candidates]

        tokens = np.zeros((max_candidates, 7), dtype=np.float32)
        mask = np.zeros((max_candidates,), dtype=np.float32)

        for idx, j in enumerate(candidates):
            h_a2a, r_a2a = _compute_a2a_gain_and_rate(
                state=state,
                access_m=access_m,
                exec_j=j,
            )
            token = np.asarray(
                [
                    float(h_a2a),
                    float(r_a2a) / a2a_rate_scale,
                    float(workload[j]) / workload_scale,
                    float(available_cpu[j]) / cpu_scale,
                    float(task_size[task_idx]) / size_scale,
                    float(task_cycles[task_idx]) / cycle_scale,
                    float(task_deadline[task_idx]) / deadline_scale,
                ],
                dtype=np.float32,
            )
            tokens[idx] = token
            mask[idx] = 1.0

        return tokens, mask

    def _estimate_local_exec_delay(
        self,
        state: Dict[str, Any],
        task_idx: int,
        access_m: int,
        lam: float,
    ) -> float:
        task_size = _extract_task_size(state)
        task_cycles = _extract_task_cycles(state)
        available_cpu = _extract_available_cpu(state)

        Dk = float(task_size[task_idx])
        Ck = float(task_cycles[task_idx])
        f_m = max(float(available_cpu[access_m]), EPS)
        return lam * Dk * Ck / f_m

    def _estimate_forward_exec_delay(
        self,
        state: Dict[str, Any],
        task_idx: int,
        access_m: int,
        exec_j: int,
        lam: float,
    ) -> float:
        task_size = _extract_task_size(state)
        task_cycles = _extract_task_cycles(state)
        available_cpu = _extract_available_cpu(state)

        Dk = float(task_size[task_idx])
        Ck = float(task_cycles[task_idx])
        f_j = max(float(available_cpu[exec_j]), EPS)

        _, r_a2a = _compute_a2a_gain_and_rate(
            state=state,
            access_m=access_m,
            exec_j=exec_j,
        )

        backhaul_delay = lam * Dk / max(float(r_a2a), EPS)
        exec_delay = lam * Dk * Ck / f_j
        return backhaul_delay + exec_delay

    # Keep these helpers for optional future ablation / reuse,
    # but they are not used in act() right now.
    def _apply_conservative_ratio_cap(
        self,
        state: Dict[str, Any],
        access_assoc: np.ndarray,
        offload_ratio: np.ndarray,
    ) -> np.ndarray:
        K = int(state["K"])
        task_size = _extract_task_size(state)
        task_cycles = _extract_task_cycles(state)
        task_deadline = _extract_task_deadline(state)
        task_local_cpu = _to_numpy(
            state.get("task_local_cpu", np.ones(K, dtype=np.float32))
        )

        new_ratio = np.asarray(offload_ratio, dtype=np.float32).copy()

        for k in range(K):
            Dk = float(task_size[k])
            Ck = float(task_cycles[k])
            tau_k = max(float(task_deadline[k]), EPS)
            f_loc = max(float(task_local_cpu[k]), EPS)

            local_only_delay = Dk * Ck / f_loc
            deadline_scale = np.clip(tau_k / max(local_only_delay, EPS), 0.6, 1.4)

            ratio_cap = self.conservative_ratio_cap_base * deadline_scale
            ratio_cap = float(np.clip(
                ratio_cap,
                self.conservative_ratio_cap_min,
                self.conservative_ratio_cap_max,
            ))

            if local_only_delay <= 0.8 * tau_k:
                ratio_cap = min(ratio_cap, 0.10)

            new_ratio[k] = min(float(new_ratio[k]), ratio_cap)

        new_ratio = np.clip(new_ratio, self.ratio_floor, self.ratio_ceiling)
        return new_ratio

    def _refine_sched_beta_local_first(
        self,
        state: Dict[str, Any],
        access_assoc: np.ndarray,
        offload_ratio: np.ndarray,
        sched_beta: np.ndarray,
    ) -> np.ndarray:
        M, K = access_assoc.shape
        refined = np.asarray(sched_beta, dtype=np.float32).copy()
        neighbors = _extract_neighbors(state)
        available_cpu = _extract_available_cpu(state)
        workload = _extract_workload(state)

        for k in range(K):
            lam = float(offload_ratio[k])
            if lam <= EPS:
                continue

            access_m = int(np.argmax(access_assoc[:, k]))
            candidates = [access_m] + list(neighbors[access_m])

            cur_j = access_m
            for j in candidates:
                if refined[k, access_m, j] > 0.5:
                    cur_j = j
                    break

            if cur_j == access_m:
                continue

            local_cpu = max(float(available_cpu[access_m]), EPS)
            neigh_cpu = max(float(available_cpu[cur_j]), EPS)
            local_work = float(workload[access_m])
            neigh_work = float(workload[cur_j])

            local_delay = self._estimate_local_exec_delay(
                state=state,
                task_idx=k,
                access_m=access_m,
                lam=lam,
            )
            forward_delay = self._estimate_forward_exec_delay(
                state=state,
                task_idx=k,
                access_m=access_m,
                exec_j=cur_j,
                lam=lam,
            )

            forward_score = forward_delay + self.forward_bias
            local_better = local_delay <= (1.0 + self.local_keep_margin) * forward_score
            weak_neighbor_advantage = (neigh_cpu <= 1.50 * local_cpu) and (neigh_work >= 0.75 * local_work)

            # softened threshold if you ever turn this back on
            if lam <= 0.08:
                refined[k, access_m, :] = 0.0
                refined[k, access_m, access_m] = 1.0
                continue

            if local_better or weak_neighbor_advantage:
                refined[k, access_m, :] = 0.0
                refined[k, access_m, access_m] = 1.0

        return refined

    def _refine_high_level_action(
        self,
        state: Dict[str, Any],
        access_assoc: np.ndarray,
        offload_ratio: np.ndarray,
        sched_beta: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        offload_ratio_refined = self._apply_conservative_ratio_cap(
            state=state,
            access_assoc=access_assoc,
            offload_ratio=offload_ratio,
        )

        sched_beta_refined = self._refine_sched_beta_local_first(
            state=state,
            access_assoc=access_assoc,
            offload_ratio=offload_ratio_refined,
            sched_beta=sched_beta,
        )

        return offload_ratio_refined, sched_beta_refined

    def _get_max_candidates(self, state: Dict[str, Any]) -> int:
        if self.max_candidates is not None:
            return int(self.max_candidates)

        neighbors = _extract_neighbors(state)
        max_deg = 1
        for nei in neighbors:
            max_deg = max(max_deg, 1 + len(nei))
        return int(max_deg)

    def _solve_low_level(
        self,
        state: Dict[str, Any],
        access_assoc: np.ndarray,
        offload_ratio: np.ndarray,
        sched_beta: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        if self.solve_bandwidth_allocation is not None:
            try:
                bandwidth_alloc = self.solve_bandwidth_allocation(
                    state=state,
                    access_assoc=access_assoc,
                    offload_ratio=offload_ratio,
                )
            except TypeError:
                bandwidth_alloc = self.solve_bandwidth_allocation(
                    state,
                    access_assoc,
                    offload_ratio,
                )
        else:
            bandwidth_alloc = _fallback_bandwidth_allocation(
                state=state,
                access_assoc=access_assoc,
                offload_ratio=offload_ratio,
            )

        if self.solve_cpu_allocation is not None:
            try:
                cpu_result = self.solve_cpu_allocation(
                    state=state,
                    access_assoc=access_assoc,
                    sched_beta=sched_beta,
                    offload_ratio=offload_ratio,
                )
            except TypeError:
                cpu_result = self.solve_cpu_allocation(
                    state,
                    access_assoc,
                    sched_beta,
                    offload_ratio,
                )
        else:
            cpu_result = _fallback_cpu_allocation(
                state=state,
                access_assoc=access_assoc,
                sched_beta=sched_beta,
                offload_ratio=offload_ratio,
            )

        if isinstance(cpu_result, tuple) or isinstance(cpu_result, list):
            cpu_alloc = cpu_result[0]
        else:
            cpu_alloc = cpu_result

        bandwidth_alloc = np.asarray(bandwidth_alloc, dtype=np.float32)
        cpu_alloc = np.asarray(cpu_alloc, dtype=np.float32)
        return {
            "bandwidth_alloc": bandwidth_alloc,
            "cpu_alloc": cpu_alloc,
        }


def build_default_proposed_policy(
    state: Dict[str, Any],
    actor_net: Optional[torch.nn.Module] = None,
    device: str = "cpu",
    embed_dim: int = 128,
    num_heads: int = 4,
    ff_hidden_dim: int = 256,
    num_layers: int = 2,
):
    token_dim = 7
    task_dim = 3
    uav_dim = 4
    fused_dim = 128

    encoder = MaskedTransformerEncoder(
        input_dim=token_dim,
        embed_dim=embed_dim,
        num_heads=num_heads,
        ff_hidden_dim=ff_hidden_dim,
        num_layers=num_layers,
        dropout=0.1,
    ).to(device)

    fusion_net = TaskContextFusion(
        topo_dim=embed_dim,
        task_dim=task_dim,
        uav_dim=uav_dim,
        hidden_dim=128,
        out_dim=fused_dim,
    ).to(device)

    ratio_head = RatioHead(
        input_dim=fused_dim,
        hidden_dim=128,
        min_ratio=0.02,
        max_ratio=0.98,
        # Increase residual capacity so the ratio head can move upward from
        # a conservative prior when Stage-2 ratio regularization is active.
        residual_scale=2.0,
    ).to(device)

    mode = "network" if actor_net is not None else "placeholder"

    max_candidates = 1
    neighbors = _extract_neighbors(state)
    for nei in neighbors:
        max_candidates = max(max_candidates, 1 + len(nei))

    return ProposedPolicy(
        mode=mode,
        actor_net=actor_net,
        encoder=encoder,
        ratio_head=ratio_head,
        fusion_net=fusion_net,
        device=device,
        max_candidates=max_candidates,
        ratio_floor=0.05,
        ratio_ceiling=0.50,
        forward_bias=1.0,
        local_keep_margin=0.15,
        conservative_ratio_cap_base=0.12,
        conservative_ratio_cap_min=0.06,
        conservative_ratio_cap_max=0.18,
    )

# # from typing import Any, Dict, List, Optional, Tuple

# # import numpy as np
# # import torch

# # from model.proposed_obs_builder import build_global_observation
# # from model.transformer_encoder import (
# #     MaskedTransformerEncoder,
# #     RatioHead,
# #     TaskContextFusion,
# # )
# # from policy.proposed_learned_policy import ProposedLearnedPolicy
# # from policy.proposed_placeholder_policy import generate_proposed_placeholder_action

# # EPS = 1e-8


# # def _safe_get(state: Dict[str, Any], key: str, default=None):
# #     return state[key] if key in state else default


# # def _to_numpy(x, dtype=np.float32):
# #     return np.asarray(x, dtype=dtype)


# # def _extract_uav_positions(state: Dict[str, Any]) -> np.ndarray:
# #     """
# #     Expected shape: [M, 2]
# #     """
# #     for key in ["uav_pos", "uav_positions", "q", "uav_xy"]:
# #         if key in state:
# #             arr = np.asarray(state[key], dtype=np.float32)
# #             if arr.ndim == 2 and arr.shape[1] >= 2:
# #                 return arr[:, :2]
# #     raise KeyError("Cannot find UAV position array in state.")


# # def _extract_task_positions(state: Dict[str, Any]) -> Optional[np.ndarray]:
# #     for key in ["task_pos", "task_positions", "td_pos", "user_pos", "w"]:
# #         if key in state:
# #             arr = np.asarray(state[key], dtype=np.float32)
# #             if arr.ndim == 2 and arr.shape[1] >= 2:
# #                 return arr[:, :2]
# #     return None


# # def _extract_neighbors(state: Dict[str, Any]) -> List[List[int]]:
# #     neighbors = _safe_get(state, "neighbors", None)
# #     if neighbors is None:
# #         M = int(state["M"])
# #         return [[] for _ in range(M)]
# #     return neighbors


# # def _extract_task_size(state: Dict[str, Any]) -> np.ndarray:
# #     for key in ["task_size", "D", "D_k", "task_data_size"]:
# #         if key in state:
# #             return _to_numpy(state[key])
# #     raise KeyError("Cannot find task size in state.")


# # def _extract_task_cycles(state: Dict[str, Any]) -> np.ndarray:
# #     for key in ["task_cycles", "C", "C_k", "task_cpu_cycles"]:
# #         if key in state:
# #             return _to_numpy(state[key])
# #     raise KeyError("Cannot find task cycles in state.")


# # def _extract_task_deadline(state: Dict[str, Any]) -> np.ndarray:
# #     for key in ["task_deadline", "deadline", "tau_max", "task_max_delay"]:
# #         if key in state:
# #             return _to_numpy(state[key])
# #     raise KeyError("Cannot find task deadline in state.")


# # def _extract_available_cpu(state: Dict[str, Any]) -> np.ndarray:
# #     for key in ["uav_available_cpu", "available_cpu", "uav_cpu_avail", "uav_cpu_max"]:
# #         if key in state:
# #             return _to_numpy(state[key])
# #     M = int(state["M"])
# #     return np.ones(M, dtype=np.float32)


# # def _extract_workload(state: Dict[str, Any]) -> np.ndarray:
# #     for key in ["uav_workload", "workload", "uav_load", "queue_len"]:
# #         if key in state:
# #             return _to_numpy(state[key])
# #     M = int(state["M"])
# #     return np.zeros(M, dtype=np.float32)


# # def _extract_tx_power(state: Dict[str, Any], access_m: int) -> float:
# #     for key in ["uav_tx_power", "P_tx", "tx_power"]:
# #         if key in state:
# #             val = state[key]
# #             arr = np.asarray(val, dtype=np.float32)
# #             if arr.ndim == 0:
# #                 return float(arr)
# #             return float(arr[access_m])
# #     return 1.0


# # def _extract_backhaul_bw(state: Dict[str, Any]) -> float:
# #     for key in ["backhaul_bandwidth", "B_bh", "bh_bandwidth"]:
# #         if key in state:
# #             return float(state[key])
# #     return 1.0


# # def _extract_noise_power(state: Dict[str, Any]) -> float:
# #     for key in ["noise_power", "sigma2", "noise_var"]:
# #         if key in state:
# #             return float(state[key])
# #     return 1e-9


# # def _compute_a2a_gain_and_rate(
# #     state: Dict[str, Any],
# #     access_m: int,
# #     exec_j: int,
# # ) -> Tuple[float, float]:
# #     """
# #     Compute approximate A2A channel gain and rate for token construction.
# #     Follows the modeling intention in the paper, with fallbacks for engineering use.
# #     """
# #     if exec_j == access_m:
# #         return 1.0, 1e6

# #     uav_pos = _extract_uav_positions(state)
# #     p_m = uav_pos[access_m]
# #     p_j = uav_pos[exec_j]

# #     dist = float(np.linalg.norm(p_m - p_j))
# #     dist = max(dist, 1.0)

# #     beta0 = float(_safe_get(state, "beta0", 1.0))
# #     gain = beta0 * (dist ** -2)

# #     B_bh = _extract_backhaul_bw(state)
# #     P_tx = _extract_tx_power(state, access_m)
# #     sigma2 = _extract_noise_power(state)

# #     rate = B_bh * np.log2(1.0 + P_tx * gain / max(sigma2, EPS))
# #     return float(gain), float(rate)


# # def _import_bandwidth_solver():
# #     try:
# #         from solver.bandwidth_solver import solve_bandwidth_allocation
# #         return solve_bandwidth_allocation
# #     except Exception:
# #         return None


# # def _import_cpu_solver():
# #     candidates = [
# #         "solve_cpu_allocation_kkt",
# #         "solve_cpu_allocation",
# #         "solve_cpu_allocation_proportional",
# #     ]
# #     try:
# #         import solver.cpu_solver as cpu_solver_module
# #         for name in candidates:
# #             if hasattr(cpu_solver_module, name):
# #                 return getattr(cpu_solver_module, name)
# #     except Exception:
# #         pass
# #     return None


# # def _fallback_bandwidth_allocation(
# #     state: Dict[str, Any],
# #     access_assoc: np.ndarray,
# #     offload_ratio: np.ndarray,
# # ) -> np.ndarray:
# #     """
# #     Simple proportional fallback if analytical solver import fails.
# #     """
# #     M, K = access_assoc.shape
# #     task_size = _extract_task_size(state)
# #     Bmax = _safe_get(state, "uav_bandwidth_max", _safe_get(state, "bandwidth_max", 1.0))
# #     Bmax = np.asarray(Bmax, dtype=np.float32)
# #     if Bmax.ndim == 0:
# #         Bmax = np.full(M, float(Bmax), dtype=np.float32)

# #     bw = np.zeros((M, K), dtype=np.float32)

# #     for m in range(M):
# #         task_idx = np.where(access_assoc[m] > 0.5)[0]
# #         if len(task_idx) == 0:
# #             continue

# #         weights = np.asarray(
# #             [max(float(offload_ratio[k] * task_size[k]), EPS) for k in task_idx],
# #             dtype=np.float32,
# #         )
# #         weights = weights / max(weights.sum(), EPS)

# #         for i, k in enumerate(task_idx):
# #             bw[m, k] = Bmax[m] * weights[i]

# #     return bw


# # def _fallback_cpu_allocation(
# #     state: Dict[str, Any],
# #     access_assoc: np.ndarray,
# #     sched_beta: np.ndarray,
# #     offload_ratio: np.ndarray,
# # ) -> np.ndarray:
# #     """
# #     Simple proportional CPU fallback if analytical solver import fails.
# #     """
# #     M, K = access_assoc.shape
# #     task_size = _extract_task_size(state)
# #     task_cycles = _extract_task_cycles(state)
# #     cpu_max = _extract_available_cpu(state)

# #     cpu_alloc = np.zeros((M, K), dtype=np.float32)

# #     for j in range(M):
# #         exec_tasks = []
# #         workloads = []

# #         for k in range(K):
# #             if offload_ratio[k] <= EPS:
# #                 continue

# #             access_m = int(np.argmax(access_assoc[:, k]))
# #             if sched_beta[k, access_m, j] > 0.5:
# #                 W = offload_ratio[k] * task_size[k] * task_cycles[k]
# #                 exec_tasks.append(k)
# #                 workloads.append(float(W))

# #         if len(exec_tasks) == 0:
# #             continue

# #         workloads = np.asarray(workloads, dtype=np.float32)
# #         weights = workloads / max(workloads.sum(), EPS)

# #         for i, k in enumerate(exec_tasks):
# #             cpu_alloc[j, k] = cpu_max[j] * weights[i]

# #     return cpu_alloc


# # class ProposedPolicy:
# #     """
# #     Full proposed-method interface aligned with the paper structure:

# #     1) build candidate execution set
# #     2) topology-aware context encoding (Transformer)
# #     3) high-level decisions:
# #          - mobility
# #          - offloading ratio
# #          - collaborative scheduling
# #     4) low-level analytical resource allocation:
# #          - bandwidth
# #          - CPU
# #     """

# #     def __init__(
# #         self,
# #         mode: str = "placeholder",
# #         actor_net: Optional[torch.nn.Module] = None,
# #         encoder: Optional[MaskedTransformerEncoder] = None,
# #         ratio_head: Optional[RatioHead] = None,
# #         fusion_net: Optional[TaskContextFusion] = None,
# #         device: str = "cpu",
# #         max_candidates: Optional[int] = None,
# #     ):
# #         self.mode = str(mode)
# #         self.actor_net = actor_net
# #         self.encoder = encoder
# #         self.ratio_head = ratio_head
# #         self.fusion_net = fusion_net
# #         self.device = device
# #         self.max_candidates = max_candidates

# #         self.learned_policy = ProposedLearnedPolicy(
# #             mode="network" if actor_net is not None else "placeholder",
# #             actor_net=actor_net,
# #             device=device,
# #         )

# #         self.solve_bandwidth_allocation = _import_bandwidth_solver()
# #         self.solve_cpu_allocation = _import_cpu_solver()

# #     def act(
# #         self,
# #         state: Dict[str, Any],
# #         access_assoc: np.ndarray,
# #         deterministic: bool = True,
# #         return_aux: bool = False,
# #     ):
# #         """
# #         Main entry.

# #         Returns:
# #             action dict with keys:
# #               - move_dist
# #               - move_angle
# #               - offload_ratio
# #               - sched_beta
# #               - bandwidth_alloc
# #               - cpu_alloc
# #         """
# #         if self.mode == "placeholder":
# #             high_action = generate_proposed_placeholder_action(state, access_assoc)
# #             low_action = self._solve_low_level(
# #                 state=state,
# #                 access_assoc=access_assoc,
# #                 offload_ratio=np.asarray(high_action["offload_ratio"], dtype=np.float32),
# #                 sched_beta=np.asarray(high_action["sched_beta"], dtype=np.float32),
# #             )
# #             full_action = {**high_action, **low_action}
# #             if return_aux:
# #                 return full_action, {}
# #             return full_action

# #         if self.mode != "network":
# #             raise ValueError(f"Unknown mode: {self.mode}")

# #         # ---------------------------------------------
# #         # Step 1: get base mobility + scheduling from learned policy
# #         # ---------------------------------------------
# #         base_high_action = self.learned_policy.act(
# #             state=state,
# #             access_assoc=access_assoc,
# #             deterministic=deterministic,
# #         )

# #         move_dist = np.asarray(base_high_action["move_dist"], dtype=np.float32)
# #         move_angle = np.asarray(base_high_action["move_angle"], dtype=np.float32)
# #         sched_beta = np.asarray(base_high_action["sched_beta"], dtype=np.float32)

# #         # ---------------------------------------------
# #         # Step 2: use Transformer context to generate offloading ratio
# #         # ---------------------------------------------
# #         if self.encoder is not None and self.ratio_head is not None and self.fusion_net is not None:
# #             offload_ratio, aux_info = self._build_offloading_ratio_from_context(
# #                 state=state,
# #                 access_assoc=access_assoc,
# #             )
# #         else:
# #             # fallback to actor-decoded offloading ratio from learned policy
# #             offload_ratio = np.asarray(base_high_action["offload_ratio"], dtype=np.float32)
# #             aux_info = {}

# #         high_action = {
# #             "move_dist": move_dist.astype(np.float32),
# #             "move_angle": move_angle.astype(np.float32),
# #             "offload_ratio": offload_ratio.astype(np.float32),
# #             "sched_beta": sched_beta.astype(np.float32),
# #         }

# #         # ---------------------------------------------
# #         # Step 3: low-level analytical resource allocation
# #         # ---------------------------------------------
# #         low_action = self._solve_low_level(
# #             state=state,
# #             access_assoc=access_assoc,
# #             offload_ratio=offload_ratio,
# #             sched_beta=sched_beta,
# #         )

# #         full_action = {**high_action, **low_action}

# #         if return_aux:
# #             return full_action, aux_info
# #         return full_action

# #     def _build_offloading_ratio_from_context(
# #         self,
# #         state: Dict[str, Any],
# #         access_assoc: np.ndarray,
# #     ) -> Tuple[np.ndarray, Dict[str, Any]]:
# #         """
# #         Build topology-aware task context and output lambda_k via ratio head.
# #         """
# #         M = int(state["M"])
# #         K = int(state["K"])

# #         task_size = _extract_task_size(state)
# #         task_cycles = _extract_task_cycles(state)
# #         task_deadline = _extract_task_deadline(state)

# #         available_cpu = _extract_available_cpu(state)
# #         workload = _extract_workload(state)
# #         uav_pos = _extract_uav_positions(state)

# #         token_batches = []
# #         mask_batches = []
# #         task_feat_batches = []
# #         uav_feat_batches = []
# #         task_indices = []

# #         Nc = self._get_max_candidates(state)

# #         for k in range(K):
# #             access_m = int(np.argmax(access_assoc[:, k]))
# #             tokens, mask = self._build_candidate_tokens_for_task(
# #                 state=state,
# #                 task_idx=k,
# #                 access_m=access_m,
# #                 max_candidates=Nc,
# #             )

# #             # task-local feature u_k^t
# #             task_feat = np.asarray(
# #                 [
# #                     float(task_size[k]),
# #                     float(task_cycles[k]),
# #                     float(task_deadline[k]),
# #                 ],
# #                 dtype=np.float32,
# #             )

# #             # local UAV state feature s_m^t
# #             uav_feat = np.asarray(
# #                 [
# #                     float(uav_pos[access_m, 0]),
# #                     float(uav_pos[access_m, 1]),
# #                     float(available_cpu[access_m]),
# #                     float(workload[access_m]),
# #                 ],
# #                 dtype=np.float32,
# #             )

# #             token_batches.append(tokens)
# #             mask_batches.append(mask)
# #             task_feat_batches.append(task_feat)
# #             uav_feat_batches.append(uav_feat)
# #             task_indices.append((k, access_m))

# #         token_tensor = torch.tensor(np.asarray(token_batches), dtype=torch.float32, device=self.device)
# #         mask_tensor = torch.tensor(np.asarray(mask_batches), dtype=torch.float32, device=self.device)
# #         task_feat_tensor = torch.tensor(np.asarray(task_feat_batches), dtype=torch.float32, device=self.device)
# #         uav_feat_tensor = torch.tensor(np.asarray(uav_feat_batches), dtype=torch.float32, device=self.device)

# #         # with torch.no_grad():
# #         #     encoded_tokens, pooled_context = self.encoder(token_tensor, mask_tensor)
# #         #     fused_feature = self.fusion_net(
# #         #         topo_context=pooled_context,
# #         #         task_feat=task_feat_tensor,
# #         #         uav_feat=uav_feat_tensor,
# #         #     )
# #         #     # offload_ratio = self.ratio_head(fused_feature).squeeze(-1)

# #         #     ratio_raw = self.ratio_head.net(fused_feature).squeeze(-1)
# #         #     offload_ratio = torch.sigmoid(ratio_raw)

# #         #     print("[RATIO DEBUG] ratio_raw[:5] =", ratio_raw[:5].detach().cpu().numpy())
# #         #     print("[RATIO DEBUG] ratio_sigmoid[:5] =", offload_ratio[:5].detach().cpu().numpy())

# #         # offload_ratio = offload_ratio.detach().cpu().numpy().astype(np.float32)
# #         # offload_ratio = np.clip(offload_ratio, 0.0, 1.0)

# #         with torch.no_grad():
# #             encoded_tokens, pooled_context = self.encoder(token_tensor, mask_tensor)
# #             fused_feature = self.fusion_net(
# #                 topo_context=pooled_context,
# #                 task_feat=task_feat_tensor,
# #                 uav_feat=uav_feat_tensor,
# #             )

# #             # DEBUG: inspect raw ratio logits before sigmoid
# #             ratio_raw = self.ratio_head.net(fused_feature).squeeze(-1)
# #             offload_ratio_t = torch.sigmoid(ratio_raw)

# #         print("[RATIO DEBUG] ratio_raw[:5] =", ratio_raw[:5].detach().cpu().numpy())
# #         print("[RATIO DEBUG] ratio_sigmoid[:5] =", offload_ratio_t[:5].detach().cpu().numpy())

# #         offload_ratio = offload_ratio_t.detach().cpu().numpy().astype(np.float32)
# #         # offload_ratio = np.clip(offload_ratio, 0.0, 1.0)
# #         offload_ratio = np.clip(offload_ratio, 0.05, 1.0)

# #         aux_info = {
# #             "encoded_tokens": encoded_tokens.detach().cpu().numpy(),
# #             "pooled_context": pooled_context.detach().cpu().numpy(),
# #             "fused_feature": fused_feature.detach().cpu().numpy(),
# #             "task_indices": task_indices,
# #         }
# #         return offload_ratio, aux_info

# #     def _build_candidate_tokens_for_task(
# #         self,
# #         state: Dict[str, Any],
# #         task_idx: int,
# #         access_m: int,
# #         max_candidates: int,
# #     ) -> Tuple[np.ndarray, np.ndarray]:
# #         """
# #         Build candidate token sequence C_{k,m}^t with zero-padding and mask.

# #         Token format:
# #             [ h_a2a, R_a2a, workload_j, avail_cpu_j, D_k, C_k, deadline_k ]
# #         """
# #         neighbors = _extract_neighbors(state)
# #         task_size = _extract_task_size(state)
# #         task_cycles = _extract_task_cycles(state)
# #         task_deadline = _extract_task_deadline(state)
# #         available_cpu = _extract_available_cpu(state)
# #         workload = _extract_workload(state)

# #         candidates = [access_m] + list(neighbors[access_m])
# #         candidates = candidates[:max_candidates]

# #         tokens = np.zeros((max_candidates, 7), dtype=np.float32)
# #         mask = np.zeros((max_candidates,), dtype=np.float32)

# #         for idx, j in enumerate(candidates):
# #             h_a2a, r_a2a = _compute_a2a_gain_and_rate(
# #                 state=state,
# #                 access_m=access_m,
# #                 exec_j=j,
# #             )
# #             token = np.asarray(
# #                 [
# #                     float(h_a2a),
# #                     float(r_a2a),
# #                     float(workload[j]),
# #                     float(available_cpu[j]),
# #                     float(task_size[task_idx]),
# #                     float(task_cycles[task_idx]),
# #                     float(task_deadline[task_idx]),
# #                 ],
# #                 dtype=np.float32,
# #             )
# #             tokens[idx] = token
# #             mask[idx] = 1.0

# #         return tokens, mask

# #     def _get_max_candidates(self, state: Dict[str, Any]) -> int:
# #         if self.max_candidates is not None:
# #             return int(self.max_candidates)

# #         neighbors = _extract_neighbors(state)
# #         max_deg = 1
# #         for nei in neighbors:
# #             max_deg = max(max_deg, 1 + len(nei))
# #         return int(max_deg)

# #     def _solve_low_level(
# #         self,
# #         state: Dict[str, Any],
# #         access_assoc: np.ndarray,
# #         offload_ratio: np.ndarray,
# #         sched_beta: np.ndarray,
# #     ) -> Dict[str, np.ndarray]:
# #         """
# #         Solve:
# #           - bandwidth allocation
# #           - CPU allocation

# #         Prefer analytical solvers from solver/ if available.
# #         Fall back to proportional versions if needed.
# #         """
# #         if self.solve_bandwidth_allocation is not None:
# #             try:
# #                 bandwidth_alloc = self.solve_bandwidth_allocation(
# #                     state=state,
# #                     access_assoc=access_assoc,
# #                     offload_ratio=offload_ratio,
# #                 )
# #             except TypeError:
# #                 bandwidth_alloc = self.solve_bandwidth_allocation(
# #                     state,
# #                     access_assoc,
# #                     offload_ratio,
# #                 )
# #         else:
# #             bandwidth_alloc = _fallback_bandwidth_allocation(
# #                 state=state,
# #                 access_assoc=access_assoc,
# #                 offload_ratio=offload_ratio,
# #             )

# #         if self.solve_cpu_allocation is not None:
# #             try:
# #                 cpu_result = self.solve_cpu_allocation(
# #                     state=state,
# #                     access_assoc=access_assoc,
# #                     sched_beta=sched_beta,
# #                     offload_ratio=offload_ratio,
# #                 )
# #             except TypeError:
# #                 cpu_result = self.solve_cpu_allocation(
# #                     state,
# #                     access_assoc,
# #                     sched_beta,
# #                     offload_ratio,
# #                 )
# #         else:
# #             cpu_result = _fallback_cpu_allocation(
# #                 state=state,
# #                 access_assoc=access_assoc,
# #                 sched_beta=sched_beta,
# #                 offload_ratio=offload_ratio,
# #             )

# #         # -------------------------------------------------
# #         # Compatible with multiple return styles:
# #         # 1) cpu_alloc
# #         # 2) (cpu_alloc, extra_info)
# #         # 3) [cpu_alloc, extra_info, ...]
# #         # -------------------------------------------------
# #         if isinstance(cpu_result, tuple) or isinstance(cpu_result, list):
# #             cpu_alloc = cpu_result[0]
# #         else:
# #             cpu_alloc = cpu_result

# #         bandwidth_alloc = np.asarray(bandwidth_alloc, dtype=np.float32)
# #         cpu_alloc = np.asarray(cpu_alloc, dtype=np.float32)
# #         return {
# #             "bandwidth_alloc": bandwidth_alloc,
# #             "cpu_alloc": cpu_alloc,
# #         }


# # def build_default_proposed_policy(
# #     state: Dict[str, Any],
# #     actor_net: Optional[torch.nn.Module] = None,
# #     device: str = "cpu",
# #     embed_dim: int = 128,
# #     num_heads: int = 4,
# #     ff_hidden_dim: int = 256,
# #     num_layers: int = 2,
# # ):
# #     """
# #     Convenience builder for the full proposed method.
# #     """
# #     token_dim = 7
# #     task_dim = 3
# #     uav_dim = 4
# #     fused_dim = 128

# #     encoder = MaskedTransformerEncoder(
# #         input_dim=token_dim,
# #         embed_dim=embed_dim,
# #         num_heads=num_heads,
# #         ff_hidden_dim=ff_hidden_dim,
# #         num_layers=num_layers,
# #         dropout=0.1,
# #     ).to(device)

# #     fusion_net = TaskContextFusion(
# #         topo_dim=embed_dim,
# #         task_dim=task_dim,
# #         uav_dim=uav_dim,
# #         hidden_dim=128,
# #         out_dim=fused_dim,
# #     ).to(device)

# #     ratio_head = RatioHead(
# #         input_dim=fused_dim,
# #         hidden_dim=128,
# #     ).to(device)

# #     mode = "network" if actor_net is not None else "placeholder"

# #     max_candidates = 1
# #     neighbors = _extract_neighbors(state)
# #     for nei in neighbors:
# #         max_candidates = max(max_candidates, 1 + len(nei))

# #     return ProposedPolicy(
# #         mode=mode,
# #         actor_net=actor_net,
# #         encoder=encoder,
# #         ratio_head=ratio_head,
# #         fusion_net=fusion_net,
# #         device=device,
# #         max_candidates=max_candidates,
# #     )