"""
AlphaLudo V6.1 — Inference-only model for gameplay.
Architecture: ResNet-10 (128ch, 24-channel strategic input) with 4-token policy head.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


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
        logits = self.policy_fc2(p)
        if legal_mask is not None:
            all_illegal = (legal_mask.sum(dim=1, keepdim=True) == 0)
            logits = logits.masked_fill(~legal_mask.bool(), float('-inf'))
            if all_illegal.any():
                logits = torch.where(
                    all_illegal.expand_as(logits),
                    torch.zeros_like(logits),
                    logits
                )
        return logits

    def forward(self, x, legal_mask=None):
        features = self._backbone(x)
        p = F.relu(self.policy_fc1(features))
        logits = self.policy_fc2(p)
        if legal_mask is not None:
            all_illegal = (legal_mask.sum(dim=1, keepdim=True) == 0)
            logits = logits.masked_fill(~legal_mask.bool(), float('-inf'))
            if all_illegal.any():
                logits = torch.where(
                    all_illegal.expand_as(logits),
                    torch.zeros_like(logits),
                    logits
                )
        policy = F.softmax(logits, dim=1)
        v = F.relu(self.value_fc1(features))
        value = self.value_fc2(v)
        return policy, value
