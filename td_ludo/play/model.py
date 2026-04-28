"""
AlphaLudo inference-only models for gameplay.

V6.1 (AlphaLudoV5):  ResNet-10, 128ch, 24-channel strategic input.
V6.3 (AlphaLudoV63): Same backbone with 27 input channels (+ bonus_turn,
                     consecutive_sixes, two_roll_capture_map) and an aux head.
                     Aux head is loaded but unused at inference.
V11.1 (AlphaLudoV11): ResNet-4 + 1 Transformer attention layer, 96ch backbone,
                     attn_dim=64, 28-channel input. 3-head: policy + win_prob + moves_remaining.
                     Imported from the main package to keep one source of truth.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

# Make td_ludo.models importable (the main package containing V11)
_PLAY_DIR = os.path.dirname(os.path.abspath(__file__))
_TD_LUDO_DIR = os.path.dirname(_PLAY_DIR)
if _TD_LUDO_DIR not in sys.path:
    sys.path.insert(0, _TD_LUDO_DIR)

# Re-export V11 + V12 from canonical implementations in the main package.
from td_ludo.models.v11 import AlphaLudoV11  # noqa: F401
from td_ludo.models.v12 import AlphaLudoV12  # noqa: F401


class ResidualBlock(nn.Module):
    def __init__(self, channels=128):
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


def _apply_legal_mask(logits, legal_mask):
    if legal_mask is None:
        return logits
    all_illegal = (legal_mask.sum(dim=1, keepdim=True) == 0)
    logits = logits.masked_fill(~legal_mask.bool(), float('-inf'))
    if all_illegal.any():
        logits = torch.where(
            all_illegal.expand_as(logits),
            torch.zeros_like(logits),
            logits,
        )
    return logits


class AlphaLudoV5(nn.Module):
    """V6.1 Strategic: 128 channels, 10 residual blocks, 24-channel input, ~3M params."""

    def __init__(self, num_res_blocks=10, num_channels=128, in_channels=17):
        super().__init__()
        self.num_channels = num_channels

        self.conv_input = nn.Conv2d(in_channels, num_channels, 3, padding=1, bias=False)
        self.bn_input = nn.BatchNorm2d(num_channels)
        self.res_blocks = nn.ModuleList([
            ResidualBlock(num_channels) for _ in range(num_res_blocks)
        ])

        self.policy_fc1 = nn.Linear(num_channels, 64)
        self.policy_fc2 = nn.Linear(64, 4)
        self.value_fc1 = nn.Linear(num_channels, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def _backbone(self, x):
        out = F.relu(self.bn_input(self.conv_input(x)))
        for block in self.res_blocks:
            out = block(out)
        out = F.adaptive_avg_pool2d(out, 1)
        return out.flatten(start_dim=1)

    def forward_policy_only(self, x, legal_mask=None):
        features = self._backbone(x)
        p = F.relu(self.policy_fc1(features))
        return _apply_legal_mask(self.policy_fc2(p), legal_mask)

    def forward(self, x, legal_mask=None):
        features = self._backbone(x)
        p = F.relu(self.policy_fc1(features))
        logits = _apply_legal_mask(self.policy_fc2(p), legal_mask)
        policy = F.softmax(logits, dim=1)
        v = F.relu(self.value_fc1(features))
        value = self.value_fc2(v)
        return policy, value


class AlphaLudoV63(nn.Module):
    """V6.3: same backbone as V5 but with 27 input channels and an auxiliary
    capture-prediction head. The aux head is loaded for state_dict compatibility
    but ignored at inference (forward_policy_only skips it entirely).
    """

    def __init__(self, num_res_blocks=10, num_channels=128, in_channels=27):
        super().__init__()
        self.num_channels = num_channels

        self.conv_input = nn.Conv2d(in_channels, num_channels, 3, padding=1, bias=False)
        self.bn_input = nn.BatchNorm2d(num_channels)
        self.res_blocks = nn.ModuleList([
            ResidualBlock(num_channels) for _ in range(num_res_blocks)
        ])

        self.policy_fc1 = nn.Linear(num_channels, 64)
        self.policy_fc2 = nn.Linear(64, 4)
        self.value_fc1 = nn.Linear(num_channels, 64)
        self.value_fc2 = nn.Linear(64, 1)

        # Aux head — present so the V6.3 state_dict loads cleanly. Unused here.
        self.aux_capture_fc1 = nn.Linear(num_channels, 64)
        self.aux_capture_fc2 = nn.Linear(64, 1)

    def _backbone(self, x):
        out = F.relu(self.bn_input(self.conv_input(x)))
        for block in self.res_blocks:
            out = block(out)
        out = F.adaptive_avg_pool2d(out, 1)
        return out.flatten(start_dim=1)

    def forward_policy_only(self, x, legal_mask=None):
        features = self._backbone(x)
        p = F.relu(self.policy_fc1(features))
        return _apply_legal_mask(self.policy_fc2(p), legal_mask)

    def forward(self, x, legal_mask=None):
        features = self._backbone(x)
        p = F.relu(self.policy_fc1(features))
        logits = _apply_legal_mask(self.policy_fc2(p), legal_mask)
        policy = F.softmax(logits, dim=1)
        v = F.relu(self.value_fc1(features))
        value = self.value_fc2(v)
        return policy, value
