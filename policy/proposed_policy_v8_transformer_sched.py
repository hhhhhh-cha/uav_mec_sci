"""
Stage-3 / v8 proposed policy.

Key change from policy.proposed_policy.ProposedPolicy:
    - Transformer encoder still provides the ratio context for lambda.
    - A new TransformerScheduleHead directly generates collaborative scheduling
      logits over candidate execution UAV tokens.
    - MLP actor is kept for UAV mobility only in deployment; its raw scheduling
      scores are no longer the final beta decision.

This preserves the paper's hierarchical variable partition:
    High-level learned: mobility Q via (d, theta), offloading Lambda, scheduling Psi.
    Low-level analytical: bandwidth B and CPU F.
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from model.transformer_encoder import MaskedTransformerEncoder, RatioHead, TaskContextFusion
from model.transformer_schedule_head import TransformerScheduleHead
from policy.proposed_policy import ProposedPolicy, _extract_neighbors

EPS = 1e-8


class ProposedPolicyV8TransformerSched(ProposedPolicy):
    def __init__(self, *args, schedule_head: Optional[TransformerScheduleHead] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.schedule_head = schedule_head

    def act(
        self,
        state: Dict[str, Any],
        access_assoc: np.ndarray,
        deterministic: bool = True,
        return_aux: bool = False,
    ):
        if self.mode == "placeholder":
            return super().act(state, access_assoc, deterministic=deterministic, return_aux=return_aux)
        if self.mode != "network":
            raise ValueError(f"Unknown mode: {self.mode}")

        # MLP actor is used for mobility. Its raw scheduling output is intentionally ignored in v8.
        base_high_action = self.learned_policy.act(
            state=state,
            access_assoc=access_assoc,
            deterministic=deterministic,
        )
        move_dist = np.asarray(base_high_action["move_dist"], dtype=np.float32)
        move_angle = np.asarray(base_high_action["move_angle"], dtype=np.float32)

        if (
            self.encoder is not None
            and self.ratio_head is not None
            and self.fusion_net is not None
            and self.schedule_head is not None
        ):
            offload_ratio, sched_beta, aux_info = self._build_ratio_and_schedule_from_context(
                state=state,
                access_assoc=access_assoc,
            )
        else:
            offload_ratio = np.asarray(base_high_action["offload_ratio"], dtype=np.float32)
            sched_beta = np.asarray(base_high_action["sched_beta"], dtype=np.float32)
            aux_info = {}

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
            aux_info["v8_transformer_sched"] = True
            aux_info["final_offload_ratio"] = offload_ratio.copy()
            aux_info["final_sched_beta"] = sched_beta.copy()
            return full_action, aux_info
        return full_action

    def _candidate_j_matrix(self, state: Dict[str, Any], access_assoc: np.ndarray, Nc: int) -> np.ndarray:
        K = int(state["K"])
        neighbors = _extract_neighbors(state)
        cand = -np.ones((K, Nc), dtype=np.int64)
        for k in range(K):
            access_m = int(np.argmax(access_assoc[:, k]))
            legal_js = [access_m] + list(neighbors[access_m])
            legal_js = [int(j) for j in legal_js if 0 <= int(j) < int(state["M"])]
            legal_js = legal_js[:Nc]
            for n, j in enumerate(legal_js):
                cand[k, n] = int(j)
        return cand

    def _build_ratio_and_schedule_from_context(
        self,
        state: Dict[str, Any],
        access_assoc: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        M = int(state["M"])
        K = int(state["K"])
        Nc = self._get_max_candidates(state)

        token_batches = []
        mask_batches = []
        task_feat_batches = []
        uav_feat_batches = []
        task_indices = []
        ratio_prior = self._compute_ratio_prior(state, access_assoc)
        candidate_j = self._candidate_j_matrix(state, access_assoc, Nc)

        for k in range(K):
            access_m = int(np.argmax(access_assoc[:, k]))
            tokens, mask = self._build_candidate_tokens_for_task(
                state=state,
                task_idx=k,
                access_m=access_m,
                max_candidates=Nc,
            )
            token_batches.append(tokens)
            mask_batches.append(mask)
            task_feat_batches.append(self._normalize_task_feature(state, k))
            uav_feat_batches.append(self._normalize_uav_feature(state, access_m))
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
            sched_logits_t = self.schedule_head(
                encoded_tokens=encoded_tokens,
                mask=mask_tensor,
                task_feat=task_feat_tensor,
                uav_feat=uav_feat_tensor,
            )

        offload_ratio = offload_ratio_t.detach().cpu().numpy().astype(np.float32)
        offload_ratio = np.clip(offload_ratio, self.ratio_floor, self.ratio_ceiling)

        sched_logits = sched_logits_t.detach().cpu().numpy().astype(np.float32)
        sched_beta = np.zeros((K, M, M), dtype=np.float32)
        mask_np = np.asarray(mask_batches, dtype=np.float32)
        for k in range(K):
            access_m = int(np.argmax(access_assoc[:, k]))
            valid_idx = np.where(mask_np[k] > 0.5)[0]
            if valid_idx.size == 0:
                sched_beta[k, access_m, access_m] = 1.0
                continue
            best_n = int(valid_idx[np.argmax(sched_logits[k, valid_idx])])
            best_j = int(candidate_j[k, best_n])
            if best_j < 0:
                best_j = access_m
            sched_beta[k, access_m, best_j] = 1.0

        aux_info = {
            "encoded_tokens": encoded_tokens.detach().cpu().numpy(),
            "pooled_context": pooled_context.detach().cpu().numpy(),
            "ratio_prior": ratio_prior.copy(),
            "final_offload_ratio": offload_ratio.copy(),
            "schedule_logits": sched_logits.copy(),
            "candidate_j": candidate_j.copy(),
            "task_indices": task_indices,
        }
        return offload_ratio, sched_beta, aux_info


def build_default_proposed_policy_v8(
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
        residual_scale=2.0,
    ).to(device)

    schedule_head = TransformerScheduleHead(
        embed_dim=embed_dim,
        task_dim=task_dim,
        uav_dim=uav_dim,
        hidden_dim=128,
        dropout=0.05,
        use_context_features=True,
    ).to(device)

    max_candidates = 1
    neighbors = _extract_neighbors(state)
    for nei in neighbors:
        max_candidates = max(max_candidates, 1 + len(nei))

    mode = "network" if actor_net is not None else "placeholder"
    return ProposedPolicyV8TransformerSched(
        mode=mode,
        actor_net=actor_net,
        encoder=encoder,
        ratio_head=ratio_head,
        fusion_net=fusion_net,
        schedule_head=schedule_head,
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
