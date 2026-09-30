"""AlphaLudo V13.7 — Tabula Rasa AlphaZero Dual-Head ResNet for 2-Player Ludo.

Architecture:
  - Input: (B, 17, 15, 15) float32
      Ch 0-3: Own 4 tokens (one-hot per token in canonical POV)
      Ch 4-7: Opponent 4 tokens (one-hot per token in canonical POV)
      Ch 8-13: Dice roll 1..6 (full plane one-hot)
      Ch 14: Safe squares (8 cells, 0.5)
      Ch 15: My home path (5 cells, 1.0)
      Ch 16: Opponent home path (5 cells, 1.0)
  - Backbone:
      Conv stem: 17 -> 128 channels, 3x3, BatchNorm, ReLU
      ResBlocks: 6 Residual Blocks (each: 2x 3x3 Conv-BN-ReLU + residual skip)
  - Heads:
      Policy Head: Per-token spatial gather via einsum over own token channels 0..3
                   Linear(128, 64) -> ReLU -> Linear(64, 1) -> (B, 4) logits with legal mask
      Value Head:  AdaptiveAvgPool2d(1) -> Linear(128, 64) -> ReLU -> Linear(64, 1) -> Tanh -> (B,) in [-1, 1]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)


class AlphaLudoV137(nn.Module):
    OWN_TOKEN_CHANNELS = (0, 1, 2, 3)

    def __init__(
        self,
        in_channels: int = 17,
        num_res_blocks: int = 6,
        num_channels: int = 128,
        head_hidden: int = 64,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_res_blocks = num_res_blocks
        self.num_channels = num_channels

        # Stem
        self.stem_conv = nn.Conv2d(
            in_channels, num_channels, kernel_size=3, padding=1, bias=False
        )
        self.stem_bn = nn.BatchNorm2d(num_channels)

        # ResNet Backbone
        self.res_blocks = nn.ModuleList(
            [ResidualBlock(num_channels) for _ in range(num_res_blocks)]
        )

        # Policy Head (Per-token spatial feature pooling)
        self.policy_fc1 = nn.Linear(num_channels, head_hidden)
        self.policy_fc2 = nn.Linear(head_hidden, 1)

        # Value Head (Zero-sum scalar payoff in [-1.0, +1.0])
        self.value_fc1 = nn.Linear(num_channels, head_hidden)
        self.value_fc2 = nn.Linear(head_hidden, 1)

    def _extract_own_token_features(
        self, x: torch.Tensor, cnn_features: torch.Tensor
    ) -> torch.Tensor:
        """Gathers CNN spatial features at each own token's board coordinates.
        x: (B, 17, 15, 15), cnn_features: (B, C, 15, 15)
        Returns: (B, 4, C)
        """
        own_mask = x[:, list(self.OWN_TOKEN_CHANNELS)]  # (B, 4, 15, 15)
        return torch.einsum("btij,bcij->btc", own_mask, cnn_features)

    def _apply_legal_mask(
        self, logits: torch.Tensor, legal_mask: torch.Tensor | None
    ) -> torch.Tensor:
        if legal_mask is None:
            return logits
        # Mask illegal actions with -1e4 instead of -inf to prevent 0.0 * -inf = NaN in loss
        mask_val = -1e4
        return torch.where(legal_mask.bool(), logits, torch.full_like(logits, mask_val))


    def forward(
        self, x: torch.Tensor, legal_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.
        Args:
            x: (B, 17, 15, 15) float32 tensor
            legal_mask: (B, 4) binary float tensor or None
        Returns:
            policy_logits: (B, 4) unnormalized log-probabilities with illegal moves masked out
            value: (B,) scalar in [-1.0, +1.0]
        """
        out = F.relu(self.stem_bn(self.stem_conv(x)))
        for block in self.res_blocks:
            out = block(out)

        # 1. Policy head
        own_features = self._extract_own_token_features(x, out)  # (B, 4, C)
        p = F.relu(self.policy_fc1(own_features))
        raw_logits = self.policy_fc2(p).squeeze(-1)  # (B, 4)
        policy_logits = self._apply_legal_mask(raw_logits, legal_mask)

        # 2. Value head
        pooled = F.adaptive_avg_pool2d(out, 1).flatten(1)  # (B, C)
        v = F.relu(self.value_fc1(pooled))
        value = torch.tanh(self.value_fc2(v)).squeeze(-1)  # (B,)

        return policy_logits, value

    @torch.no_grad()
    def predict_p_v(
        self, x: torch.Tensor, legal_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Inference helper returning normalized probabilities and scalar value."""
        self.eval()
        logits, value = self.forward(x, legal_mask)
        probs = F.softmax(logits, dim=-1)
        return probs, value
