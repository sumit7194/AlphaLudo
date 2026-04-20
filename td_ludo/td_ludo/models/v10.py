"""
AlphaLudo V10 — Slim multi-task architecture.

Designed from V6.3 mech-interp findings:
  - V6.3 had 10 ResBlocks but only block 0 mattered (others redundant per CKA)
  - All 128 channels active but packed with redundancy
  - Value head was uncalibrated (PPO normalized returns, not probabilities)
  - Ch 25 (consecutive_sixes) was completely unused (KL=0)

V10 fixes:
  - 6 ResBlocks (down from 10) — still has safety margin
  - 96 channels (down from 128) — slight compression
  - 28 input channels: V6.3 minus dead ch25 + 2 new strategic channels
    * ch26: forced_single_token_flag (option-value exhaustion trigger)
    * ch27: my_leader_progress (endgame proximity)
  - 3 heads (replaces policy + value): policy, win_prob, moves_remaining
  - Trained jointly from scratch so backbone learns outcome-prediction features

Parameters: ~1.0M (vs V6.3's 3.0M — 3x smaller, much faster on CPU).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)


class AlphaLudoV10(nn.Module):
    """V10: Slim multi-task CNN. Policy + win_prob + moves_remaining heads."""

    def __init__(self, num_res_blocks=6, num_channels=96, in_channels=28):
        super().__init__()
        self.num_channels = num_channels
        self.num_res_blocks = num_res_blocks
        self.in_channels = in_channels

        # Stem
        self.conv_input = nn.Conv2d(
            in_channels, num_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn_input = nn.BatchNorm2d(num_channels)

        # Backbone
        self.res_blocks = nn.ModuleList(
            [ResidualBlock(num_channels) for _ in range(num_res_blocks)]
        )

        feat = num_channels

        # Policy head — which of 4 tokens to move
        self.policy_fc1 = nn.Linear(feat, 48)
        self.policy_fc2 = nn.Linear(48, 4)

        # Win-prob head — sigmoid output, trained with BCE on {0,1} outcomes
        self.win_fc1 = nn.Linear(feat, 48)
        self.win_fc2 = nn.Linear(48, 1)

        # Moves-remaining head — softplus output (>=0), trained with MSE on own-turn count
        self.moves_fc1 = nn.Linear(feat, 48)
        self.moves_fc2 = nn.Linear(48, 1)

    def _backbone(self, x):
        out = F.relu(self.bn_input(self.conv_input(x)))
        for block in self.res_blocks:
            out = block(out)
        out = F.adaptive_avg_pool2d(out, 1)
        return out.flatten(start_dim=1)

    def _apply_legal_mask(self, policy_logits, legal_mask):
        if legal_mask is None:
            return policy_logits
        all_illegal = (legal_mask.sum(dim=1, keepdim=True) == 0)
        policy_logits = policy_logits.masked_fill(~legal_mask.bool(), float('-inf'))
        if all_illegal.any():
            policy_logits = torch.where(
                all_illegal.expand_as(policy_logits),
                torch.zeros_like(policy_logits),
                policy_logits,
            )
        return policy_logits

    def forward(self, x, legal_mask=None):
        """Full forward: returns (policy, win_prob, moves_remaining)."""
        features = self._backbone(x)

        # Policy
        p = F.relu(self.policy_fc1(features))
        policy_logits = self._apply_legal_mask(self.policy_fc2(p), legal_mask)
        policy = F.softmax(policy_logits, dim=1)

        # Win probability (sigmoid)
        w = F.relu(self.win_fc1(features))
        win_prob = torch.sigmoid(self.win_fc2(w)).squeeze(-1)

        # Moves remaining (softplus ensures >= 0)
        m = F.relu(self.moves_fc1(features))
        moves_remaining = F.softplus(self.moves_fc2(m)).squeeze(-1)

        return policy, win_prob, moves_remaining

    def forward_policy_only(self, x, legal_mask=None):
        """Fast inference path — returns policy logits only."""
        features = self._backbone(x)
        p = F.relu(self.policy_fc1(features))
        return self._apply_legal_mask(self.policy_fc2(p), legal_mask)

    def forward_with_features(self, x, legal_mask=None):
        """Return all heads plus GAP features (useful for probing/analysis)."""
        features = self._backbone(x)
        p = F.relu(self.policy_fc1(features))
        policy_logits = self._apply_legal_mask(self.policy_fc2(p), legal_mask)
        policy = F.softmax(policy_logits, dim=1)

        w = F.relu(self.win_fc1(features))
        win_prob = torch.sigmoid(self.win_fc2(w)).squeeze(-1)

        m = F.relu(self.moves_fc1(features))
        moves_remaining = F.softplus(self.moves_fc2(m)).squeeze(-1)

        return policy, win_prob, moves_remaining, features

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = AlphaLudoV10()
    print(f"V10 params: {model.count_parameters():,}")
    x = torch.randn(2, 28, 15, 15)
    mask = torch.tensor([[1.0, 1.0, 0.0, 1.0], [1.0, 0.0, 1.0, 0.0]])
    policy, win_prob, moves = model(x, mask)
    print(f"policy: {policy.shape}, win_prob: {win_prob.shape}, moves: {moves.shape}")
    print(f"policy row sums: {policy.sum(dim=1)}")
    print(f"win_prob: {win_prob}")
    print(f"moves_remaining: {moves}")
