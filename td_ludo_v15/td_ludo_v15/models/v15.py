"""V15 GraphTransformer model.

Architecture (see ../../V15_DESIGN_PLAN.md):
    Input:  (B, T, 15, 15, 3) — per-cell triplet × T frames (T=history_len)
    Flatten per node: T × 3 features per board cell
    Add learned positional embedding (225, d_model)
    Prepend learnable CLS token
    n_layers × Graph Transformer layers with edge-biased attention
    Policy head: per-board-node MLP → 225 logits → masked softmax
    Value head:  CLS slice → MLP → sigmoid scalar

Params target: ~3.0M for V15 (matches V13.5 teacher capacity).
Original default was d_model=192, n_layers=4 ≈ 1.3M which under-fit the
teacher in cross-arch distillation — bumped to d=256, n_layers=8 on
2026-05-14.

V15.1 variant (2026-05-17): tiny arch (d=128, n_layers=4, ffn=256) with
history_len=2 (current + 1 prev frame). Aim ~0.7M params. Hypothesis:
the 8-frame history was redundant capacity that diluted gradient signal
during distillation; a shorter window and smaller model should fit the
teacher's per-state argmax just as well at a fraction of the cost.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..game.cells import CLS_INDEX, NUM_BOARD_CELLS, NUM_NODES
from ..game.graph import EDGE_TYPE_MATRIX, NUM_EDGE_TYPES
from ._blocks import GTLayer


class V15GraphTransformer(nn.Module):
    """V15 model: Graph Transformer over 226 nodes (225 board cells + 1 CLS).

    Args:
        d_model:    hidden dim (V15=256, V15.1=128)
        n_heads:    attention heads (V15=8, V15.1=4)
        n_layers:   number of GT layers (V15=8, V15.1=4)
        ffn_dim:    FFN inner dim (V15=512, V15.1=256)
        history_len: number of stacked frames T (V15=8, V15.1=2).
                     in_features is derived as history_len * 3 if not given.
        in_features: per-node input dim. If None, computed from history_len.
        attn_dropout, ffn_dropout: regularization (default 0)
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 8,
        ffn_dim: int = 512,
        history_len: int = 8,
        in_features: int | None = None,
        attn_dropout: float = 0.0,
        ffn_dropout: float = 0.0,
    ):
        super().__init__()
        if in_features is None:
            in_features = history_len * 3
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.history_len = history_len
        self.in_features = in_features

        # Input MLP: per-cell triplet × 8 frames → d_model
        # Two-layer with GELU. Each board cell is processed by this shared MLP.
        self.input_mlp = nn.Sequential(
            nn.Linear(in_features, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # Learned positional embedding for board cells (225 entries).
        # CLS gets its own learnable embedding separately.
        self.pos_emb = nn.Embedding(NUM_BOARD_CELLS, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls_token, std=0.02)

        # Static edge-type matrix as a buffer (227 × 227).
        # `register_buffer` makes it move with `.to(device)`.
        self.register_buffer(
            "edge_type_matrix",
            torch.from_numpy(EDGE_TYPE_MATRIX.astype(np.int64)),
            persistent=False,
        )

        # Stack of GT layers
        self.layers = nn.ModuleList([
            GTLayer(
                d_model=d_model,
                n_heads=n_heads,
                ffn_dim=ffn_dim,
                num_edge_types=NUM_EDGE_TYPES,
                attn_dropout=attn_dropout,
                ffn_dropout=ffn_dropout,
            )
            for _ in range(n_layers)
        ])
        self.final_ln = nn.LayerNorm(d_model)

        # Policy head — shared per-cell MLP applied to each of the 225
        # board nodes. CLS is excluded from policy.
        self.policy_mlp = nn.Sequential(
            nn.Linear(d_model, d_model // 3),
            nn.GELU(),
            nn.Linear(d_model // 3, 1),
        )

        # Value head — CLS token → sigmoid scalar (win prob).
        self.value_mlp = nn.Sequential(
            nn.Linear(d_model, d_model // 3),
            nn.GELU(),
            nn.Linear(d_model // 3, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        legal_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, T, 15, 15, 3) float — per-cell triplet × T frames in
               current-player POV. T must equal self.history_len.
            legal_mask: (B, 225) bool/float — 1 where this cell is a legal
                source cell for the move, 0 otherwise. Cells with 0 get
                their policy logit masked to -inf before softmax.

        Returns:
            policy: (B, 225) — masked softmax over board cells
            value: (B,) — sigmoid scalar (win probability)
        """
        B = x.shape[0]
        # (B, T, 15, 15, 3) → (B, 225, T*3)
        # Reorder: permute axes so cells are the leading non-batch dim and
        # frames+features collapse into the per-node feature vector.
        # Currently axes are (B, T, R, C, F). We want (B, R*C, T*F).
        x = x.permute(0, 2, 3, 1, 4).contiguous().view(B, NUM_BOARD_CELLS, -1)

        # Per-cell input MLP
        node_emb = self.input_mlp(x.float())  # (B, 225, d_model)

        # Add positional embedding
        pos_ids = torch.arange(NUM_BOARD_CELLS, device=x.device)
        node_emb = node_emb + self.pos_emb(pos_ids).unsqueeze(0)

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)
        nodes = torch.cat([node_emb, cls], dim=1)  # (B, 226, d_model)

        # GT layers
        for layer in self.layers:
            nodes = layer(nodes, self.edge_type_matrix)
        nodes = self.final_ln(nodes)

        # Split: board nodes + CLS
        board_nodes = nodes[:, :NUM_BOARD_CELLS, :]   # (B, 225, d_model)
        cls_node = nodes[:, CLS_INDEX, :]             # (B, d_model)

        # Policy: per-board-node MLP → (B, 225)
        policy_logits = self.policy_mlp(board_nodes).squeeze(-1)  # (B, 225)

        # Apply legal mask if provided
        if legal_mask is not None:
            # Replace illegal cells with -inf before softmax
            mask_f = legal_mask.float()
            # Avoid -inf when all-zero (e.g., terminal states); guard with
            # very-large-negative bias instead of literal -inf for fp16/bf16
            # compatibility.
            policy_logits = policy_logits.masked_fill(mask_f < 0.5, -1e9)
        policy = F.softmax(policy_logits, dim=-1)

        # Value: CLS → scalar sigmoid
        value = torch.sigmoid(self.value_mlp(cls_node)).squeeze(-1)  # (B,)

        return policy, value

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
