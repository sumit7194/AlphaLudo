"""AlphaLudo V15-4PW (Winner-Distilled) GraphTransformer model.

Extends V15-4P with an auxiliary Standings/Progress Head for 4-player game-theory awareness.

Heads:
1. Policy Head: 225 source-cell logits -> masked softmax
2. Value Head: 4 logits -> softmax distribution over relative winner ([Me, Next, Opp, Prev])
3. Standings Aux Head: 4 logits -> Sigmoid -> normalized progress fractions in [0.0, 1.0]
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure td_ludo_v15 is accessible
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_V15_ROOT = _REPO_ROOT / "td_ludo_v15"
for p in (str(_V15_ROOT), str(_REPO_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from td_ludo_v15.game.cells import CLS_INDEX, NUM_BOARD_CELLS, NUM_NODES
from td_ludo_v15.game.graph_4p import EDGE_TYPE_MATRIX_4P, NUM_EDGE_TYPES
from td_ludo_v15.models._blocks import GTLayer


class V15_4PW_GraphTransformer(nn.Module):
    """V15 4-Player Winner-Distilled Graph Transformer over 226 nodes (225 board cells + 1 CLS)."""

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

        # Input MLP: per-cell quintuplet -> d_model
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

        # 1. Policy head: shared per-node MLP -> 225 source-cell logits
        self.policy_mlp = nn.Sequential(
            nn.Linear(d_model, d_model // 3),
            nn.GELU(),
            nn.Linear(d_model // 3, 1),
        )

        # 2. Value head: CLS token -> 4 logits ([P0, P1, P2, P3] win probabilities)
        self.value_mlp = nn.Sequential(
            nn.Linear(d_model, d_model // 3),
            nn.GELU(),
            nn.Linear(d_model // 3, 4),
        )

        # 3. Standings Aux head: CLS token -> 4 logits -> Sigmoid ([Me, Next, Opp, Prev] progress in [0, 1])
        self.standings_mlp = nn.Sequential(
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
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        """Forward pass.
        Args:
            x: (B, 15, 15, 5) float32 in mover POV.
            legal_mask: (B, 225) float/bool — 1 where cell is legal source cell.
            return_logits: if True, returns (policy, value, standings, pol_logits, val_logits, standings_logits).

        Returns:
            policy: (B, 225) masked softmax
            value: (B, 4) softmax win probabilities
            standings: (B, 4) sigmoid progress fractions in [0.0, 1.0]
        """
        if x.dim() == 4:
            x = x.unsqueeze(1)
        B = x.shape[0]

        # (B, 1, 15, 15, 5) -> (B, 225, 5)
        x = x.permute(0, 2, 3, 1, 4).contiguous().view(B, NUM_BOARD_CELLS, -1)

        # Per-cell input MLP
        node_emb = self.input_mlp(x.float())  # (B, 225, d_model)

        # Add positional embedding
        pos_ids = torch.arange(NUM_BOARD_CELLS, device=x.device)
        node_emb = node_emb + self.pos_emb(pos_ids).unsqueeze(0)

        # Prepend CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        h = torch.cat([node_emb, cls_tokens], dim=1)  # (B, 226, d_model)

        # Pass through GT layers
        for layer in self.layers:
            h = layer(h, self.edge_type_matrix)
        h = self.final_ln(h)

        # Extract board nodes and CLS node
        board_nodes = h[:, :NUM_BOARD_CELLS, :]  # (B, 225, d_model)
        cls_node = h[:, CLS_INDEX, :]            # (B, d_model)

        # 1. Policy head
        policy_logits = self.policy_mlp(board_nodes).squeeze(-1)  # (B, 225)
        if legal_mask is not None:
            mask_bool = legal_mask.bool()
            policy_logits = policy_logits.masked_fill(~mask_bool, -1e9)
        policy = F.softmax(policy_logits, dim=-1)

        # 2. Value head
        value_logits = self.value_mlp(cls_node)  # (B, 4)
        value = F.softmax(value_logits, dim=-1)

        # 3. Standings aux head
        standings_logits = self.standings_mlp(cls_node)  # (B, 4)
        standings = torch.sigmoid(standings_logits)       # (B, 4) in [0.0, 1.0]

        if return_logits:
            return policy, value, standings, policy_logits, value_logits, standings_logits
        return policy, value, standings

    def load_sl_pretrained(self, checkpoint_path: str):
        """Loads pretrained weights from earlier SL distillation run."""
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        print(f"📥 Loaded pretrained weights from {checkpoint_path}")
        print(f"   Newly initialized layers: {missing}")
        if unexpected:
            print(f"   Unexpected keys ignored: {unexpected}")

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
