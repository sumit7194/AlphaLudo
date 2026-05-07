import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """Standard CNN ResBlock."""
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

class MinimalCNN14(nn.Module):
    """
    14-Channel Minimal CNN for Distillation.
    Input: (B, 14, 15, 15)
    0-3: Own tokens
    4-7: Opp tokens
    8-13: Dice one-hot
    
    10 ResBlocks x 128 channels. Pure CNN. No transformer.
    """
    OWN_TOKEN_CHANNELS = (0, 1, 2, 3)

    def __init__(
        self,
        num_res_blocks: int = 10,
        num_channels: int = 128,
        in_channels: int = 14,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.num_res_blocks = num_res_blocks
        self.in_channels = in_channels

        # ---- Stem ----
        self.conv_input = nn.Conv2d(
            in_channels, num_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn_input = nn.BatchNorm2d(num_channels)

        # ---- CNN backbone ----
        self.res_blocks = nn.ModuleList(
            [ResidualBlock(num_channels) for _ in range(num_res_blocks)]
        )

        # ---- Heads ----
        # Policy head (permutation equivariant, applied to the 4 token features extracted from CNN)
        self.policy_fc1 = nn.Linear(num_channels, 64)
        self.policy_fc2 = nn.Linear(64, 1)

        # Value and Moves heads (Global Average Pooling)
        feat = num_channels
        self.win_fc1 = nn.Linear(feat, 64)
        self.win_fc2 = nn.Linear(64, 1)

        self.moves_fc1 = nn.Linear(feat, 64)
        self.moves_fc2 = nn.Linear(64, 1)

    def _cnn_backbone(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn_input(self.conv_input(x)))
        for block in self.res_blocks:
            out = block(out)
        return out

    def _extract_own_token_features(
        self, x: torch.Tensor, cnn_features: torch.Tensor
    ) -> torch.Tensor:
        """Gather per-token features from CNN at the cells where own tokens sit."""
        own_mask = x[:, list(self.OWN_TOKEN_CHANNELS)]
        # einsum: sum over spatial dims (i, j) of mask * features
        # (B, 4, H, W) * (B, C, H, W) -> (B, 4, C)
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

    def forward(self, x: torch.Tensor, legal_mask: torch.Tensor | None = None):
        # Backbone
        cnn_features = self._cnn_backbone(x)  # (B, C, H, W)

        # Policy Head (Per-Token Extraction)
        own_features = self._extract_own_token_features(x, cnn_features) # (B, 4, C)
        p = F.relu(self.policy_fc1(own_features))
        policy_logits = self.policy_fc2(p).squeeze(-1) # (B, 4)
        policy_logits = self._apply_legal_mask(policy_logits, legal_mask)
        policy = F.softmax(policy_logits, dim=1)
        # Value & Moves Head (Global Average Pooling)
        pooled = F.adaptive_avg_pool2d(cnn_features, 1).flatten(1) # (B, C)

        w = F.relu(self.win_fc1(pooled))
        win_prob = torch.sigmoid(self.win_fc2(w)).squeeze(-1)

        m = F.relu(self.moves_fc1(pooled))
        moves_remaining = F.softplus(self.moves_fc2(m)).squeeze(-1)

        return policy, win_prob, moves_remaining

    def forward_policy_only(
        self, x: torch.Tensor, legal_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """V13 RL path: PPO sampler reads logits (pre-softmax), applies its
        own temperature, then samples. Matches AlphaLudoV12's interface so
        the V11 player can drive both architectures with no per-model
        branching."""
        cnn_features = self._cnn_backbone(x)
        own_features = self._extract_own_token_features(x, cnn_features)
        p = F.relu(self.policy_fc1(own_features))
        policy_logits = self.policy_fc2(p).squeeze(-1)
        return self._apply_legal_mask(policy_logits, legal_mask)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

if __name__ == "__main__":
    print("=== MinimalCNN14 Distillation Model ===")
    model = MinimalCNN14()
    print(f"params: {model.count_parameters():,}")

    x = torch.randn(2, 14, 15, 15)
    # Ensure one-hot for token channels 0-3
    for batch in range(2):
        for ch in range(4):
            x[batch, ch] = 0
            r, c = torch.randint(0, 15, (2,)).tolist()
            x[batch, ch, r, c] = 1.0
            
    mask = torch.tensor([[1.0, 1.0, 0.0, 1.0], [1.0, 0.0, 1.0, 0.0]])
    policy, win_prob, moves = model(x, mask)
    print(f"\\npolicy:           {tuple(policy.shape)} sums={policy.sum(dim=1).tolist()}")
    print(f"win_prob:         {tuple(win_prob.shape)} values={win_prob.tolist()}")
    print(f"moves_remaining:  {tuple(moves.shape)} values={moves.tolist()}")
