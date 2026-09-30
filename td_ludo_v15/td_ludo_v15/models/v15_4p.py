"""V15 4-Player GraphTransformer model.

Architecture:
    Input:  (B, T, 15, 15, 5) — per-cell quintuplet × T frames (default T=1)
    Flatten per node: T × 5 features per board cell
    Add learned positional embedding (225, d_model)
    Prepend learnable CLS token (226th node)
    n_layers × Graph Transformer layers with edge-biased attention (4P graph)
    Policy head: per-board-node MLP → 225 logits → masked softmax
    Value head:  CLS slice → MLP → 4 logits → softmax ([v0, v1, v2, v3] win probs)

Params: ~700K parameters for default 4×128 (history_len=1).
"""
from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..game.cells import CLS_INDEX, NUM_BOARD_CELLS, NUM_NODES
from ..game.graph_4p import EDGE_TYPE_MATRIX_4P, NUM_EDGE_TYPES
from ._blocks import GTLayer


class V15_4P_GraphTransformer(nn.Module):
    """V15 4-Player Graph Transformer over 226 nodes (225 board cells + 1 CLS).

    Args:
        d_model:     hidden dim (default 128)
        n_heads:     attention heads (default 4)
        n_layers:    number of GT layers (default 4)
        ffn_dim:     FFN inner dim (default 256)
        history_len: number of stacked frames T (default 1)
        in_features: per-node input dim. If None, derived as history_len * 5.
        attn_dropout, ffn_dropout: regularization (default 0.0)
    """

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        ffn_dim: int = 256,
        history_len: int = 1,
        in_features: Optional[int] = None,
        attn_dropout: float = 0.0,
        ffn_dropout: float = 0.0,
    ):
        super().__init__()
        if in_features is None:
            in_features = history_len * 5
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.history_len = history_len
        self.in_features = in_features

        # Input MLP: per-cell quintuplet × T frames → d_model
        self.input_mlp = nn.Sequential(
            nn.Linear(in_features, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # Learned positional embedding for board cells (225 entries)
        self.pos_emb = nn.Embedding(NUM_BOARD_CELLS, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls_token, std=0.02)

        # Static 4-player edge-type matrix buffer (226 × 226)
        self.register_buffer(
            "edge_type_matrix",
            torch.from_numpy(EDGE_TYPE_MATRIX_4P.astype(np.int64)),
            persistent=False,
        )

        # GT layers
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

        # Policy head: shared per-node MLP → 225 source-cell logits
        self.policy_mlp = nn.Sequential(
            nn.Linear(d_model, d_model // 3),
            nn.GELU(),
            nn.Linear(d_model // 3, 1),
        )

        # Value head: CLS token → 4 logits ([P0, P1, P2, P3] win probabilities)
        self.value_mlp = nn.Sequential(
            nn.Linear(d_model, d_model // 3),
            nn.GELU(),
            nn.Linear(d_model // 3, 4),
        )

    def forward(
        self,
        x: torch.Tensor,
        legal_mask: Optional[torch.Tensor] = None,
        return_logits: bool = False,
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        """
        Args:
            x: (B, T, 15, 15, 5) or (B, 15, 15, 5) float32 in current mover POV.
            legal_mask: (B, 225) float/bool — 1 where cell is a legal source cell.
            return_logits: if True, returns (policy, value, policy_logits, value_logits).

        Returns:
            policy: (B, 225) — masked softmax over board cells
            value: (B, 4) — softmax distribution over all 4 players winning
        """
        if x.dim() == 4:
            # (B, 15, 15, 5) → (B, 1, 15, 15, 5)
            x = x.unsqueeze(1)
        B = x.shape[0]

        # (B, T, 15, 15, 5) → (B, 225, T*5)
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

        # Split: 225 board nodes + CLS
        board_nodes = nodes[:, :NUM_BOARD_CELLS, :]  # (B, 225, d_model)
        cls_node = nodes[:, CLS_INDEX, :]            # (B, d_model)

        # Policy: per-board-node MLP → (B, 225)
        raw_policy_logits = self.policy_mlp(board_nodes).squeeze(-1)  # (B, 225)

        # Mask illegal moves
        if legal_mask is not None:
            mask_f = legal_mask.float()
            policy_logits = raw_policy_logits.masked_fill(mask_f < 0.5, -1e9)
        else:
            policy_logits = raw_policy_logits
        policy = F.softmax(policy_logits, dim=-1)

        # Value: CLS → 4 logits → softmax
        raw_value_logits = self.value_mlp(cls_node)  # (B, 4)
        value = F.softmax(raw_value_logits, dim=-1)

        if return_logits:
            return policy, value, raw_policy_logits, raw_value_logits
        return policy, value

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
