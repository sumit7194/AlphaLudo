"""
AlphaLudo V6.3 — Bonus-Turn Awareness + Capture Prediction

Same CNN backbone as V6.1 (AlphaLudoV5: 128ch, 10 ResBlocks) but with:
  - 27 input channels (V6.1's 24 + 3 bonus-turn channels)
  - Auxiliary capture-prediction head (Linear 128→64→1, sigmoid)

Architecture:
  Input: (B, 27, 15, 15)
    Ch 0-23: V6.1 strategic encoding (tokens, danger, capture, safe landing, etc.)
    Ch 24:   bonus_turn_flag (broadcast 1.0 if dice == 6)
    Ch 25:   consecutive_sixes (broadcast 0.0/0.5/1.0)
    Ch 26:   two_roll_capture_map (opponent positions capturable in 6+X combo)

  CNN Backbone (identical to V5/V6.1):
    Stem: Conv2d(27→128, 3×3) + BN + ReLU
    10× ResidualBlock(128)
    GAP → (B, 128)

  Heads:
    Policy: Linear(128→64) → ReLU → Linear(64→4) → softmax
    Value:  Linear(128→64) → ReLU → Linear(64→1)
    Aux:    Linear(128→64) → ReLU → Linear(64→1) → sigmoid
            (predicts "will capture within next 5 own turns?")

  Total: ~3.01M params (~8K more than V6.1 from aux head)
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


class AlphaLudoV63(nn.Module):
    """V6.3: V6.1 CNN + bonus-turn channels + auxiliary capture prediction."""

    def __init__(self, num_res_blocks=10, num_channels=128, in_channels=27):
        super().__init__()
        self.num_channels = num_channels

        # Stem (same structure as V5, but in_channels=27)
        self.conv_input = nn.Conv2d(in_channels, num_channels, kernel_size=3,
                                     padding=1, bias=False)
        self.bn_input = nn.BatchNorm2d(num_channels)

        # Backbone
        self.res_blocks = nn.ModuleList([
            ResidualBlock(num_channels) for _ in range(num_res_blocks)
        ])

        feature_size = num_channels

        # Policy head (4 token outputs — the Actor)
        self.policy_fc1 = nn.Linear(feature_size, 64)
        self.policy_fc2 = nn.Linear(64, 4)

        # Value head (win probability — the Critic)
        self.value_fc1 = nn.Linear(feature_size, 64)
        self.value_fc2 = nn.Linear(64, 1)

        # Auxiliary capture prediction head
        # Predicts: "will this player capture an opponent within the next
        # 5 of their own turns?" — sigmoid output in [0, 1].
        self.aux_capture_fc1 = nn.Linear(feature_size, 64)
        self.aux_capture_fc2 = nn.Linear(64, 1)

    def _backbone(self, x):
        """Shared backbone: stem + residual blocks + GAP → features."""
        out = F.relu(self.bn_input(self.conv_input(x)))
        for block in self.res_blocks:
            out = block(out)
        out = F.adaptive_avg_pool2d(out, 1)
        return out.flatten(start_dim=1)

    def _apply_legal_mask(self, policy_logits, legal_mask):
        """Apply legal move mask to policy logits."""
        if legal_mask is not None:
            all_illegal = (legal_mask.sum(dim=1, keepdim=True) == 0)
            policy_logits = policy_logits.masked_fill(~legal_mask.bool(), float('-inf'))
            if all_illegal.any():
                policy_logits = torch.where(
                    all_illegal.expand_as(policy_logits),
                    torch.zeros_like(policy_logits),
                    policy_logits,
                )
        return policy_logits

    def forward(self, x, legal_mask=None, detach_aux=False):
        """
        Full forward pass returning policy, value, and auxiliary prediction.

        Args:
            x: (B, 27, 15, 15) state tensor
            legal_mask: (B, 4) float tensor, 1.0 for legal, 0.0 for illegal
            detach_aux: if True, detach features before aux head so aux loss
                        does not backprop into the backbone.

        Returns:
            policy: (B, 4) probability distribution over tokens
            value: (B, 1) win value (unbounded)
            aux_capture: (B, 1) capture logits (raw, no sigmoid) if detach_aux,
                         else sigmoid probabilities for backward compat.
        """
        features = self._backbone(x)

        # Policy (Actor)
        p = F.relu(self.policy_fc1(features))
        policy_logits = self._apply_legal_mask(self.policy_fc2(p), legal_mask)
        policy = F.softmax(policy_logits, dim=1)

        # Value (Critic)
        v = F.relu(self.value_fc1(features))
        value = self.value_fc2(v)

        # Auxiliary capture prediction
        aux_features = features.detach() if detach_aux else features
        a = F.relu(self.aux_capture_fc1(aux_features))
        aux_out = self.aux_capture_fc2(a)

        if detach_aux:
            # Return raw logits — caller uses BCE_with_logits + pos_weight
            return policy, value, aux_out
        else:
            return policy, value, torch.sigmoid(aux_out)

    def forward_policy_only(self, x, legal_mask=None):
        """
        Fast forward pass returning only policy logits (for eval/play).
        Skips value and auxiliary heads.
        """
        features = self._backbone(x)
        p = F.relu(self.policy_fc1(features))
        return self._apply_legal_mask(self.policy_fc2(p), legal_mask)

    def load_v61_weights(self, v61_state_dict):
        """
        Load CNN backbone + policy/value heads from a V6.1 (AlphaLudoV5)
        checkpoint. The stem conv is expanded from 24→27 input channels
        with zero-padding on the new channels. The auxiliary head keeps
        its random initialization.

        Returns the number of parameters transferred.
        """
        own_state = self.state_dict()
        loaded = 0
        skipped = 0

        for key, v61_val in v61_state_dict.items():
            if key not in own_state:
                skipped += 1
                continue

            if key == 'conv_input.weight':
                # Stem conv: V6.1 is (128, 24, 3, 3), V6.3 is (128, 27, 3, 3)
                # Copy the 24 existing channels, leave channels 24-26 at zero.
                assert v61_val.shape == (self.num_channels, 24, 3, 3), \
                    f"Unexpected V6.1 stem shape: {v61_val.shape}"
                new_weight = torch.zeros_like(own_state[key])
                new_weight[:, :24, :, :] = v61_val
                own_state[key] = new_weight
                loaded += 1
            elif own_state[key].shape == v61_val.shape:
                own_state[key] = v61_val
                loaded += 1
            else:
                skipped += 1

        self.load_state_dict(own_state)

        # Count aux head params that stayed at random init
        aux_params = sum(1 for k in own_state if 'aux_capture' in k)

        print(f"[V6.3] Loaded {loaded} params from V6.1, skipped {skipped}. "
              f"Aux head ({aux_params} params) at random init. "
              f"Stem conv expanded 24→27ch (new channels zero-init).")
        return loaded

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = AlphaLudoV63(num_res_blocks=10, num_channels=128, in_channels=27)
    total = model.count_parameters()
    print(f"AlphaLudo V6.3 — Total: {total:,}")

    # Test forward
    B = 2
    x = torch.randn(B, 27, 15, 15)
    legal = torch.ones(B, 4)
    policy, value, aux = model(x, legal)
    print(f"Policy: {policy.shape}, Value: {value.shape}, Aux: {aux.shape}")
    assert policy.shape == (B, 4)
    assert value.shape == (B, 1)
    assert aux.shape == (B, 1)
    assert (aux >= 0).all() and (aux <= 1).all(), "aux must be in [0, 1]"
    print("Forward pass OK")

    # Test policy-only
    logits = model.forward_policy_only(x, legal)
    print(f"Policy-only logits: {logits.shape}")
    assert logits.shape == (B, 4)
    print("Policy-only OK")
