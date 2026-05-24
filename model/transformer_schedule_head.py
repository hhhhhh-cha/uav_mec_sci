"""
Transformer-guided collaborative scheduling head for Stage-3 / v8.

Purpose
-------
The original proposed policy used Transformer context mainly for the offloading
ratio branch, while collaborative scheduling beta was decoded from the MLP actor.
This module makes beta explicitly Transformer-guided: every candidate execution
UAV token receives a scheduling logit, and infeasible / padded candidates are
masked before softmax or argmax.

Input shapes
------------
encoded_tokens : [K, Nc, embed_dim]
mask           : [K, Nc], 1=valid candidate, 0=padding/infeasible
optional task_feat : [K, task_dim]
optional uav_feat  : [K, uav_dim]

Output
------
masked_logits  : [K, Nc]
"""

from typing import Optional

import torch
import torch.nn as nn


class TransformerScheduleHead(nn.Module):
    """Token-level scheduling head over candidate execution UAVs."""

    def __init__(
        self,
        embed_dim: int,
        task_dim: int = 3,
        uav_dim: int = 4,
        hidden_dim: int = 128,
        dropout: float = 0.05,
        use_context_features: bool = True,
    ):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.task_dim = int(task_dim)
        self.uav_dim = int(uav_dim)
        self.hidden_dim = int(hidden_dim)
        self.use_context_features = bool(use_context_features)

        in_dim = self.embed_dim
        if self.use_context_features:
            in_dim += self.task_dim + self.uav_dim

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        encoded_tokens: torch.Tensor,
        mask: torch.Tensor,
        task_feat: Optional[torch.Tensor] = None,
        uav_feat: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if encoded_tokens.dim() != 3:
            raise ValueError(f"encoded_tokens must be [K,Nc,E], got {tuple(encoded_tokens.shape)}")
        K, Nc, E = encoded_tokens.shape
        if E != self.embed_dim:
            raise ValueError(f"embed_dim mismatch: got {E}, expected {self.embed_dim}")

        x = encoded_tokens
        if self.use_context_features:
            if task_feat is None or uav_feat is None:
                raise ValueError("task_feat and uav_feat are required when use_context_features=True")
            if task_feat.shape[0] != K or uav_feat.shape[0] != K:
                raise ValueError("task_feat/uav_feat first dimension must match K")
            task_rep = task_feat.unsqueeze(1).expand(K, Nc, task_feat.shape[-1])
            uav_rep = uav_feat.unsqueeze(1).expand(K, Nc, uav_feat.shape[-1])
            x = torch.cat([x, task_rep, uav_rep], dim=-1)

        logits = self.net(x).squeeze(-1)
        mask = mask.float()
        logits = logits.masked_fill(mask < 0.5, -1e9)
        return logits
