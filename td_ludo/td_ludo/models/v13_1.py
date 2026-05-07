"""V13.1 — MinimalCNN with auxiliary feature-prediction heads.

Architecture: 12 ResBlocks × 160 channels, 14ch raw input. ~5.6M params.

Key idea: V13.1 receives the SAME 14-channel raw input as V13 (token positions
+ dice one-hot — no engineered strategic features). During SL it ALSO has to
predict STATIC board-layout features that V11 encoder gives V12.2 for free
but V13's 14ch input does not:
  - safe_square_map  (15×15 binary, 8 cells)  → which cells are safe squares
  - home_path_map    (15×15 binary, 5 cells)  → which cells lead to home center

State-dependent features (danger, capture, own positions) are NOT aux'd
because they're either already in the input (own/opp positions) or derivable
from input + dice (so adding aux is redundant). Only static layout info that
the input doesn't directly contain gets an aux head — those are the things
the model would otherwise have to discover indirectly via policy observation.

Inputs/outputs match MinimalCNN14 for drop-in compatibility with existing
RL pipeline. Aux heads are extra outputs the SL trainer reads but the RL
pipeline ignores (forward without aux=True returns the same shape as
MinimalCNN14).

Forward returns:
    (policy, win_prob, moves_remaining)        — same as MinimalCNN14
or with aux=True:
    (policy, win_prob, moves_remaining,
     safe_map_logits, home_path_logits)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)


class MinimalCNN14Aux(nn.Module):
    """V13.1: MinimalCNN14 + 3 auxiliary feature-prediction heads.

    Default config: 12 ResBlocks × 160 channels (was 10×128 in V13).
    """
    OWN_TOKEN_CHANNELS = (0, 1, 2, 3)

    def __init__(
        self,
        num_res_blocks: int = 12,
        num_channels: int = 160,
        in_channels: int = 14,
        aux_hidden: int = 64,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.num_res_blocks = num_res_blocks
        self.in_channels = in_channels

        # Stem
        self.conv_input = nn.Conv2d(
            in_channels, num_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn_input = nn.BatchNorm2d(num_channels)

        # CNN backbone
        self.res_blocks = nn.ModuleList(
            [ResidualBlock(num_channels) for _ in range(num_res_blocks)]
        )

        # ── Main heads (match MinimalCNN14 interface) ─────────────────
        self.policy_fc1 = nn.Linear(num_channels, aux_hidden)
        self.policy_fc2 = nn.Linear(aux_hidden, 1)

        self.win_fc1 = nn.Linear(num_channels, aux_hidden)
        self.win_fc2 = nn.Linear(aux_hidden, 1)

        self.moves_fc1 = nn.Linear(num_channels, aux_hidden)
        self.moves_fc2 = nn.Linear(aux_hidden, 1)

        # ── Auxiliary heads (2 STATIC spatial maps, each 15×15 binary) ──
        # Both targets are deterministic given current_player and constant
        # in the canonical (post-rotation) view, so the aux loss forces the
        # backbone to encode the static board layout — info the 14ch input
        # does not contain explicitly.
        self.aux_safe_conv = nn.Conv2d(num_channels, 1, kernel_size=1)
        self.aux_home_path_conv = nn.Conv2d(num_channels, 1, kernel_size=1)

    # ── Backbone + feature extraction (same as MinimalCNN14) ─────────
    def _cnn_backbone(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn_input(self.conv_input(x)))
        for block in self.res_blocks:
            out = block(out)
        return out

    def _extract_own_token_features(
        self, x: torch.Tensor, cnn_features: torch.Tensor
    ) -> torch.Tensor:
        own_mask = x[:, list(self.OWN_TOKEN_CHANNELS)]
        own_features = torch.einsum("btij,bcij->btc", own_mask, cnn_features)
        return own_features

    def _apply_legal_mask(
        self, policy_logits: torch.Tensor, legal_mask: torch.Tensor | None
    ) -> torch.Tensor:
        if legal_mask is None:
            return policy_logits
        all_illegal = (legal_mask.sum(dim=1, keepdim=True) == 0)
        policy_logits = policy_logits.masked_fill(~legal_mask.bool(), float("-inf"))
        if all_illegal.any():
            policy_logits = torch.where(
                all_illegal.expand_as(policy_logits),
                torch.zeros_like(policy_logits),
                policy_logits,
            )
        return policy_logits

    # ── Aux head computation ─────────────────────────────────────────
    def _aux_outputs(self, cnn_features: torch.Tensor):
        """Compute the 2 spatial aux head logits, each shape (B, 15, 15)."""
        safe_map_logits = self.aux_safe_conv(cnn_features).squeeze(1)
        home_path_logits = self.aux_home_path_conv(cnn_features).squeeze(1)
        return safe_map_logits, home_path_logits

    # ── Public forward ───────────────────────────────────────────────
    def forward(self, x: torch.Tensor, legal_mask: torch.Tensor | None = None,
                aux: bool = False):
        cnn_features = self._cnn_backbone(x)
        own_features = self._extract_own_token_features(x, cnn_features)

        # Policy
        p = F.relu(self.policy_fc1(own_features))
        policy_logits = self.policy_fc2(p).squeeze(-1)
        policy_logits = self._apply_legal_mask(policy_logits, legal_mask)
        policy = F.softmax(policy_logits, dim=1)

        # Value & moves (global average pool)
        pooled = F.adaptive_avg_pool2d(cnn_features, 1).flatten(1)
        w = F.relu(self.win_fc1(pooled))
        win_prob = torch.sigmoid(self.win_fc2(w)).squeeze(-1)
        m = F.relu(self.moves_fc1(pooled))
        moves_remaining = F.softplus(self.moves_fc2(m)).squeeze(-1)

        if not aux:
            return policy, win_prob, moves_remaining

        safe_map_logits, home_path_logits = self._aux_outputs(cnn_features)
        return (policy, win_prob, moves_remaining,
                safe_map_logits, home_path_logits)

    def forward_policy_only(
        self, x: torch.Tensor, legal_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """RL-time interface: returns pre-softmax legal-masked logits.
        Matches MinimalCNN14 / AlphaLudoV12 so the V11 player drives both."""
        cnn_features = self._cnn_backbone(x)
        own_features = self._extract_own_token_features(x, cnn_features)
        p = F.relu(self.policy_fc1(own_features))
        policy_logits = self.policy_fc2(p).squeeze(-1)
        return self._apply_legal_mask(policy_logits, legal_mask)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    print("=== V13.1 MinimalCNN14Aux ===")
    model = MinimalCNN14Aux()
    print(f"params: {model.count_parameters():,}")
    print(f"  ResBlocks={model.num_res_blocks} × Channels={model.num_channels}")

    x = torch.randn(2, 14, 15, 15)
    for batch in range(2):
        for ch in range(4):
            x[batch, ch] = 0
            r, c = torch.randint(0, 15, (2,)).tolist()
            x[batch, ch, r, c] = 1.0
    mask = torch.tensor([[1.0, 1.0, 0.0, 1.0], [1.0, 0.0, 1.0, 0.0]])

    # Default forward (RL-compatible)
    p, w, m = model(x, mask)
    print(f"\nRL forward:")
    print(f"  policy: {tuple(p.shape)}, win: {tuple(w.shape)}, moves: {tuple(m.shape)}")

    # Aux forward (SL-compatible)
    p, w, m, smap, hpmap = model(x, mask, aux=True)
    print(f"\nSL forward (with aux):")
    print(f"  policy: {tuple(p.shape)}, win: {tuple(w.shape)}, moves: {tuple(m.shape)}")
    print(f"  safe_map: {tuple(smap.shape)}, home_path_map: {tuple(hpmap.shape)}")

    # Forward-policy-only interface
    logits = model.forward_policy_only(x, mask)
    print(f"\nforward_policy_only: {tuple(logits.shape)}")
