# # 这一文件在你论文里的职责
# # 它对应你第四节 IV-B：
# # 输入：候选执行节点特征 token
# # 做 masked self-attention
# # 输出 pooled context
# # 后面给 ratio head / actor 融合用
# # 你现在先把这个模块写出来，后面再在 proposed_policy.py 里调用它

import math
from typing import Optional, Tuple


EPS = 1e-6

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedTransformerEncoder(nn.Module):
    """
    Topology-aware masked Transformer encoder for candidate execution nodes.

    Input:
        x:    [B, N, input_dim]
        mask: [B, N], where
              mask[b, n] = 1 means valid token
              mask[b, n] = 0 means padded / invalid token

    Output:
        h:      [B, N, embed_dim]   encoded token representations
        pooled: [B, embed_dim]      masked mean pooled context
    """

    def __init__(
        self,
        input_dim: int,
        embed_dim: int = 128,
        num_heads: int = 4,
        ff_hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads.")

        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_hidden_dim = ff_hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

        self.input_proj = nn.Linear(input_dim, embed_dim)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    ff_hidden_dim=ff_hidden_dim,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.out_norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x:    [B, N, input_dim]
            mask: [B, N], 1=valid, 0=invalid

        Returns:
            h:      [B, N, embed_dim]
            pooled: [B, embed_dim]
        """
        if x.dim() != 3:
            raise ValueError(f"x must have shape [B, N, input_dim], got {tuple(x.shape)}")

        B, N, D = x.shape
        if D != self.input_dim:
            raise ValueError(f"input_dim mismatch: got {D}, expected {self.input_dim}")

        if mask is None:
            mask = torch.ones((B, N), dtype=torch.float32, device=x.device)
        else:
            mask = mask.float()

        h = self.input_proj(x)

        for layer in self.layers:
            h = layer(h, mask)

        h = self.out_norm(h)
        pooled = masked_mean_pool(h, mask)

        return h, pooled


class TransformerBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ff_hidden_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.attn = MultiHeadSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.drop1 = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_hidden_dim, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x, mask)
        x = self.norm1(x + self.drop1(attn_out))

        ffn_out = self.ffn(x)
        x = self.norm2(x + self.drop2(ffn_out))
        return x


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float):
        super().__init__()

        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads.")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        x:    [B, N, E]
        mask: [B, N], 1=valid, 0=invalid
        """
        B, N, E = x.shape

        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B,H,N,D]
        k = self.k_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [B,H,N,N]

        # key padding mask: invalid keys should not be attended to
        key_mask = mask.unsqueeze(1).unsqueeze(2)  # [B,1,1,N]
        scores = scores.masked_fill(key_mask < 0.5, -1e9)

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)  # [B,H,N,D]
        out = out.transpose(1, 2).contiguous().view(B, N, E)
        out = self.o_proj(out)

        # zero-out invalid query positions as well
        out = out * mask.unsqueeze(-1)
        return out


def masked_mean_pool(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    x:    [B, N, E]
    mask: [B, N], 1=valid, 0=invalid
    """
    mask = mask.float()
    denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
    pooled = (x * mask.unsqueeze(-1)).sum(dim=1) / denom
    return pooled


class RatioHead(nn.Module):
    """
    Engineering-stable ratio head.

    Instead of predicting lambda from scratch, the head predicts a bounded
    residual on top of an optional prior ratio. This avoids the common failure
    mode where the branch collapses to near-zero because the raw logit drifts to
    a large negative value early in training or when checkpoint quality is weak.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        min_ratio: float = 0.02,
        max_ratio: float = 0.98,
        residual_scale: float = 1.5,
    ):
        super().__init__()
        self.min_ratio = float(min_ratio)
        self.max_ratio = float(max_ratio)
        self.residual_scale = float(residual_scale)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def _safe_logit(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.clamp(x, EPS, 1.0 - EPS)
        return torch.log(x) - torch.log(1.0 - x)

    def forward(
        self,
        fused_feature: torch.Tensor,
        prior_ratio: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        hard_min: Optional[float] = None,
    ) -> torch.Tensor:
        raw_delta = self.net(fused_feature)
        bounded_delta = self.residual_scale * torch.tanh(raw_delta)

        if prior_ratio is None:
            prior_logit = 0.0
        else:
            prior_logit = self._safe_logit(prior_ratio)

        logit = prior_logit + bounded_delta / max(float(temperature), EPS)
        ratio = torch.sigmoid(logit)
        ratio = self.min_ratio + (self.max_ratio - self.min_ratio) * ratio

        if hard_min is not None:
            ratio = torch.clamp(ratio, min=float(hard_min))

        return ratio


class TaskContextFusion(nn.Module):
    """
    Fuse:
      - pooled topology-aware context z_{k,m}^t
      - task-local feature u_k^t
      - local UAV state feature s_m^t

    into one fused feature.
    """

    def __init__(
        self,
        topo_dim: int,
        task_dim: int,
        uav_dim: int,
        hidden_dim: int = 128,
        out_dim: int = 128,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(topo_dim + task_dim + uav_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU(),
        )

    def forward(
        self,
        topo_context: torch.Tensor,
        task_feat: torch.Tensor,
        uav_feat: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([topo_context, task_feat, uav_feat], dim=-1)
        return self.net(x)

# import math
# from typing import Optional, Tuple

# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# class MaskedTransformerEncoder(nn.Module):
#     """
#     Topology-aware masked Transformer encoder for candidate execution nodes.

#     Input:
#         x:    [B, N, input_dim]
#         mask: [B, N], where
#               mask[b, n] = 1 means valid token
#               mask[b, n] = 0 means padded / invalid token

#     Output:
#         h:      [B, N, embed_dim]   encoded token representations
#         pooled: [B, embed_dim]      masked mean pooled context
#     """

#     def __init__(
#         self,
#         input_dim: int,
#         embed_dim: int = 128,
#         num_heads: int = 4,
#         ff_hidden_dim: int = 256,
#         num_layers: int = 2,
#         dropout: float = 0.1,
#     ):
#         super().__init__()

#         if embed_dim % num_heads != 0:
#             raise ValueError("embed_dim must be divisible by num_heads.")

#         self.input_dim = input_dim
#         self.embed_dim = embed_dim
#         self.num_heads = num_heads
#         self.ff_hidden_dim = ff_hidden_dim
#         self.num_layers = num_layers
#         self.dropout = dropout

#         self.input_proj = nn.Linear(input_dim, embed_dim)
#         self.layers = nn.ModuleList(
#             [
#                 TransformerBlock(
#                     embed_dim=embed_dim,
#                     num_heads=num_heads,
#                     ff_hidden_dim=ff_hidden_dim,
#                     dropout=dropout,
#                 )
#                 for _ in range(num_layers)
#             ]
#         )
#         self.out_norm = nn.LayerNorm(embed_dim)

#     def forward(
#         self,
#         x: torch.Tensor,
#         mask: Optional[torch.Tensor] = None,
#     ) -> Tuple[torch.Tensor, torch.Tensor]:
#         """
#         Args:
#             x:    [B, N, input_dim]
#             mask: [B, N], 1=valid, 0=invalid

#         Returns:
#             h:      [B, N, embed_dim]
#             pooled: [B, embed_dim]
#         """
#         if x.dim() != 3:
#             raise ValueError(f"x must have shape [B, N, input_dim], got {tuple(x.shape)}")

#         B, N, D = x.shape
#         if D != self.input_dim:
#             raise ValueError(f"input_dim mismatch: got {D}, expected {self.input_dim}")

#         if mask is None:
#             mask = torch.ones((B, N), dtype=torch.float32, device=x.device)
#         else:
#             mask = mask.float()

#         h = self.input_proj(x)

#         for layer in self.layers:
#             h = layer(h, mask)

#         h = self.out_norm(h)
#         pooled = masked_mean_pool(h, mask)

#         return h, pooled


# class TransformerBlock(nn.Module):
#     def __init__(
#         self,
#         embed_dim: int,
#         num_heads: int,
#         ff_hidden_dim: int,
#         dropout: float,
#     ):
#         super().__init__()
#         self.attn = MultiHeadSelfAttention(
#             embed_dim=embed_dim,
#             num_heads=num_heads,
#             dropout=dropout,
#         )
#         self.norm1 = nn.LayerNorm(embed_dim)
#         self.drop1 = nn.Dropout(dropout)

#         self.ffn = nn.Sequential(
#             nn.Linear(embed_dim, ff_hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(ff_hidden_dim, embed_dim),
#         )
#         self.norm2 = nn.LayerNorm(embed_dim)
#         self.drop2 = nn.Dropout(dropout)

#     def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
#         attn_out = self.attn(x, mask)
#         x = self.norm1(x + self.drop1(attn_out))

#         ffn_out = self.ffn(x)
#         x = self.norm2(x + self.drop2(ffn_out))
#         return x


# class MultiHeadSelfAttention(nn.Module):
#     def __init__(self, embed_dim: int, num_heads: int, dropout: float):
#         super().__init__()

#         if embed_dim % num_heads != 0:
#             raise ValueError("embed_dim must be divisible by num_heads.")

#         self.embed_dim = embed_dim
#         self.num_heads = num_heads
#         self.head_dim = embed_dim // num_heads

#         self.q_proj = nn.Linear(embed_dim, embed_dim)
#         self.k_proj = nn.Linear(embed_dim, embed_dim)
#         self.v_proj = nn.Linear(embed_dim, embed_dim)
#         self.o_proj = nn.Linear(embed_dim, embed_dim)

#         self.dropout = nn.Dropout(dropout)

#     def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
#         """
#         x:    [B, N, E]
#         mask: [B, N], 1=valid, 0=invalid
#         """
#         B, N, E = x.shape

#         q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B,H,N,D]
#         k = self.k_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
#         v = self.v_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

#         scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [B,H,N,N]

#         # key padding mask: invalid keys should not be attended to
#         key_mask = mask.unsqueeze(1).unsqueeze(2)  # [B,1,1,N]
#         scores = scores.masked_fill(key_mask < 0.5, -1e9)

#         attn = torch.softmax(scores, dim=-1)
#         attn = self.dropout(attn)

#         out = torch.matmul(attn, v)  # [B,H,N,D]
#         out = out.transpose(1, 2).contiguous().view(B, N, E)
#         out = self.o_proj(out)

#         # zero-out invalid query positions as well
#         out = out * mask.unsqueeze(-1)
#         return out


# def masked_mean_pool(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
#     """
#     x:    [B, N, E]
#     mask: [B, N], 1=valid, 0=invalid
#     """
#     mask = mask.float()
#     denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
#     pooled = (x * mask.unsqueeze(-1)).sum(dim=1) / denom
#     return pooled


# class RatioHead(nn.Module):
#     """
#     Ratio head for generating offloading ratio lambda in [0, 1].

#     Input:
#         fused_feature: [B, D]

#     Output:
#         lambda_ratio: [B, 1]
#     """

#     def __init__(self, input_dim: int, hidden_dim: int = 128):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, 1),
#         )

#     def forward(self, fused_feature: torch.Tensor) -> torch.Tensor:
#         return torch.sigmoid(self.net(fused_feature))


# class TaskContextFusion(nn.Module):
#     """
#     Fuse:
#       - pooled topology-aware context z_{k,m}^t
#       - task-local feature u_k^t
#       - local UAV state feature s_m^t

#     into one fused feature.
#     """

#     def __init__(
#         self,
#         topo_dim: int,
#         task_dim: int,
#         uav_dim: int,
#         hidden_dim: int = 128,
#         out_dim: int = 128,
#     ):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(topo_dim + task_dim + uav_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, out_dim),
#             nn.ReLU(),
#         )

#     def forward(
#         self,
#         topo_context: torch.Tensor,
#         task_feat: torch.Tensor,
#         uav_feat: torch.Tensor,
#     ) -> torch.Tensor:
#         x = torch.cat([topo_context, task_feat, uav_feat], dim=-1)
#         return self.net(x)