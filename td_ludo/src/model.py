"""
AlphaLudo v3 Neural Network - Direct Token Selection

Key Improvements over v2:
1. Policy head outputs 4 logits (one per token) instead of 225 spatial
2. Legal move masking applied before softmax
3. Auxiliary safety head helps value learning
4. Designed for TD(λ) training with MCTS Q-value targets

Architecture:
- Input: (B, 17, 15, 15) spatial tensor
- Backbone: ResNet-10 (128 channels)
- Policy Head: GAP → FC(128→64→4)
- Value Head: GAP → FC(128→64→1)
- Aux Safety Head: GAP → FC(128→64→4) [optional training signal]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Standard Residual block with 128 filters."""
    def __init__(self, channels=128):
        super(ResidualBlock, self).__init__()
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


class AlphaLudoV3(nn.Module):
    """
    v3 Architecture - Direct Token Selection
    
    Changes from v2 (AlphaLudoTopNet):
    - Policy: 225 spatial → 4 token logits
    - Added auxiliary safety prediction head
    - Streamlined for faster training
    """
    def __init__(self, num_res_blocks=10, num_channels=128, in_channels=17):
        super(AlphaLudoV3, self).__init__()
        
        self.num_channels = num_channels
        
        # Stem: 17 Input Channels → 128 Filters
        self.conv_input = nn.Conv2d(in_channels, num_channels, kernel_size=3, padding=1, bias=False)
        self.bn_input = nn.BatchNorm2d(num_channels)
        
        # Backbone: 10 Residual Blocks
        self.res_blocks = nn.ModuleList([
            ResidualBlock(num_channels) for _ in range(num_res_blocks)
        ])
        
        # Global Average Pooling output size: 128
        feature_size = num_channels
        
        # --- Policy Head (4 token outputs) ---
        self.policy_fc1 = nn.Linear(feature_size, 64)
        self.policy_fc2 = nn.Linear(64, 4)
        
        # --- Value Head ---
        self.value_fc1 = nn.Linear(feature_size, 64)
        self.value_fc2 = nn.Linear(64, 1)
        
        # --- Auxiliary Safety Head (helps value learning) ---
        # Predicts "safety score" for each token (0=danger, 1=safe)
        self.aux_fc1 = nn.Linear(feature_size, 64)
        self.aux_fc2 = nn.Linear(64, 4)
        
    def forward(self, x, legal_mask=None):
        """
        Forward pass with direct token selection.
        
        Args:
            x: (B, 17, 15, 15) spatial tensor
            legal_mask: (B, 4) binary mask of legal moves (optional)
            
        Returns:
            policy: (B, 4) probabilities per token (masked if provided)
            value: (B, 1) game outcome prediction
            aux_safety: (B, 4) token safety scores (for training)
        """
        # Spatial backbone
        out = F.relu(self.bn_input(self.conv_input(x)))
        
        for block in self.res_blocks:
            out = block(out)
        
        # Global Average Pooling: (B, 128, 15, 15) → (B, 128)
        out = F.adaptive_avg_pool2d(out, 1)
        features = out.flatten(start_dim=1)  # (B, 128)
        
        # Policy Head → (B, 4) logits
        p = F.relu(self.policy_fc1(features))
        policy_logits = self.policy_fc2(p)  # (B, 4)
        
        # Apply legal move mask before softmax
        if legal_mask is not None:
            # Check if all moves are illegal (edge case - shouldn't happen in practice)
            all_illegal = (legal_mask.sum(dim=1, keepdim=True) == 0)
            
            # Mask out illegal moves with large negative value
            policy_logits = policy_logits.masked_fill(~legal_mask.bool(), float('-inf'))
            
            # For rows where all moves are illegal, set to uniform (avoid NaN)
            if all_illegal.any():
                policy_logits = torch.where(
                    all_illegal.expand_as(policy_logits),
                    torch.zeros_like(policy_logits),  # Uniform logits
                    policy_logits
                )
        
        policy = F.softmax(policy_logits, dim=1)
        
        # Value Head
        v = F.relu(self.value_fc1(features))
        value = torch.tanh(self.value_fc2(v))  # (B, 1)
        
        # Auxiliary Safety Head
        a = F.relu(self.aux_fc1(features))
        aux_safety = torch.sigmoid(self.aux_fc2(a))  # (B, 4)
        
        return policy, value, aux_safety
    
    def forward_policy_value(self, x, legal_mask=None):
        """Inference-only forward (skip auxiliary head for speed)."""
        out = F.relu(self.bn_input(self.conv_input(x)))
        
        for block in self.res_blocks:
            out = block(out)
        
        out = F.adaptive_avg_pool2d(out, 1)
        features = out.flatten(start_dim=1)
        
        # Policy
        p = F.relu(self.policy_fc1(features))
        policy_logits = self.policy_fc2(p)
        
        if legal_mask is not None:
            policy_logits = policy_logits.masked_fill(~legal_mask.bool(), float('-inf'))
        
        policy = F.softmax(policy_logits, dim=1)
        
        # Value
        v = F.relu(self.value_fc1(features))
        value = torch.tanh(self.value_fc2(v))
        
        return policy, value
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class AlphaLudoV4(nn.Module):
    """
    v4 Architecture - Slim & Fast
    
    Same inputs/outputs as v3, but massively reduced capacity for speed and RL stability.
    - Channels: 32 (down from 128)
    - Depth: 3 ResNet blocks (down from 10)
    """
    def __init__(self, num_res_blocks=3, num_channels=32, in_channels=17):
        super(AlphaLudoV4, self).__init__()
        
        self.num_channels = num_channels
        
        # Stem
        self.conv_input = nn.Conv2d(in_channels, num_channels, kernel_size=3, padding=1, bias=False)
        self.bn_input = nn.BatchNorm2d(num_channels)
        
        # Backbone
        self.res_blocks = nn.ModuleList([
            ResidualBlock(num_channels) for _ in range(num_res_blocks)
        ])
        
        feature_size = num_channels
        
        # Policy Head (4 token outputs)
        self.policy_fc1 = nn.Linear(feature_size, 32)
        self.policy_fc2 = nn.Linear(32, 4)
        
        # Value Head
        self.value_fc1 = nn.Linear(feature_size, 32)
        self.value_fc2 = nn.Linear(32, 1)
        
        # Auxiliary Safety Head
        self.aux_fc1 = nn.Linear(feature_size, 32)
        self.aux_fc2 = nn.Linear(32, 4)
        
    def forward(self, x, legal_mask=None):
        out = F.relu(self.bn_input(self.conv_input(x)))
        for block in self.res_blocks:
            out = block(out)
        
        out = F.adaptive_avg_pool2d(out, 1)
        features = out.flatten(start_dim=1)
        
        # Policy
        p = F.relu(self.policy_fc1(features))
        policy_logits = self.policy_fc2(p)
        
        if legal_mask is not None:
            all_illegal = (legal_mask.sum(dim=1, keepdim=True) == 0)
            policy_logits = policy_logits.masked_fill(~legal_mask.bool(), float('-inf'))
            if all_illegal.any():
                policy_logits = torch.where(
                    all_illegal.expand_as(policy_logits),
                    torch.zeros_like(policy_logits),
                    policy_logits
                )
        
        policy = F.softmax(policy_logits, dim=1)
        
        # Value
        v = F.relu(self.value_fc1(features))
        value = torch.tanh(self.value_fc2(v))
        
        # Aux
        a = F.relu(self.aux_fc1(features))
        aux_safety = torch.sigmoid(self.aux_fc2(a))
        
        return policy, value, aux_safety
    
    def forward_policy_value(self, x, legal_mask=None):
        out = F.relu(self.bn_input(self.conv_input(x)))
        for block in self.res_blocks:
            out = block(out)
        
        out = F.adaptive_avg_pool2d(out, 1)
        features = out.flatten(start_dim=1)
        
        p = F.relu(self.policy_fc1(features))
        policy_logits = self.policy_fc2(p)
        if legal_mask is not None:
            all_illegal = (legal_mask.sum(dim=1, keepdim=True) == 0)
            policy_logits = policy_logits.masked_fill(~legal_mask.bool(), float('-inf'))
            if all_illegal.any():
                policy_logits = torch.where(
                    all_illegal.expand_as(policy_logits),
                    torch.zeros_like(policy_logits),
                    policy_logits
                )
        policy = F.softmax(policy_logits, dim=1)
        
        v = F.relu(self.value_fc1(features))
        value = torch.tanh(self.value_fc2(v))
        
        return policy, value
        
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class AlphaLudoV5(nn.Module):
    """
    v5 Architecture - Actor-Critic Medium
    
    Designed for policy-gradient (REINFORCE + baseline) training.
    Bigger than V4 for better feature extraction, no aux head.
    
    - Channels: 64 (up from 32)
    - Depth: 5 ResNet blocks (up from 3)
    - Policy Head: GAP → FC(64→64→4) → softmax (with legal move masking)
    - Value Head: GAP → FC(64→64→1) → tanh (baseline for variance reduction)
    - No Aux Safety Head (removed — was never trained)
    - ~250K parameters
    """
    def __init__(self, num_res_blocks=5, num_channels=64, in_channels=17):
        super(AlphaLudoV5, self).__init__()
        
        self.num_channels = num_channels
        
        # Stem
        self.conv_input = nn.Conv2d(in_channels, num_channels, kernel_size=3, padding=1, bias=False)
        self.bn_input = nn.BatchNorm2d(num_channels)
        
        # Backbone
        self.res_blocks = nn.ModuleList([
            ResidualBlock(num_channels) for _ in range(num_res_blocks)
        ])
        
        feature_size = num_channels
        
        # Policy Head (4 token outputs — the Actor)
        self.policy_fc1 = nn.Linear(feature_size, 64)
        self.policy_fc2 = nn.Linear(64, 4)
        
        # Value Head (win probability — the Critic / baseline)
        self.value_fc1 = nn.Linear(feature_size, 64)
        self.value_fc2 = nn.Linear(64, 1)
        
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
                    policy_logits
                )
        return policy_logits
        
    def forward(self, x, legal_mask=None):
        """
        Full forward pass returning policy probabilities and value.
        
        Args:
            x: (B, 17, 15, 15) state tensor
            legal_mask: (B, 4) float tensor, 1.0 for legal tokens, 0.0 for illegal
            
        Returns:
            policy: (B, 4) probability distribution over tokens
            value: (B, 1) win probability in [-1, +1]
        """
        features = self._backbone(x)
        
        # Policy (Actor)
        p = F.relu(self.policy_fc1(features))
        policy_logits = self._apply_legal_mask(self.policy_fc2(p), legal_mask)
        policy = F.softmax(policy_logits, dim=1)
        
        # Value (Critic)
        v = F.relu(self.value_fc1(features))
        value = self.value_fc2(v)  # Removed tanh to allow unbound dense returns
        
        return policy, value
    
    def forward_policy_only(self, x, legal_mask=None):
        """
        Fast forward pass returning only policy logits (for inference).
        Skips value head computation.
        
        Returns:
            policy_logits: (B, 4) raw logits (apply softmax/temperature externally)
        """
        features = self._backbone(x)
        p = F.relu(self.policy_fc1(features))
        return self._apply_legal_mask(self.policy_fc2(p), legal_mask)
        
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = AlphaLudoV3()
    print(f"AlphaLudo v3 Parameters: {model.count_parameters():,}")
    
    model4 = AlphaLudoV4()
    print(f"AlphaLudo v4 Parameters: {model4.count_parameters():,}")
    
    model5 = AlphaLudoV5()
    print(f"AlphaLudo v5 Parameters: {model5.count_parameters():,}")
    
    # Test V5 forward pass
    x = torch.randn(2, 11, 15, 15)
    legal_mask = torch.tensor([[1, 1, 0, 0], [1, 0, 1, 1]], dtype=torch.float32)
    
    policy, value = model5(x, legal_mask)
    print(f"V5 Policy: {policy.shape}, Value: {value.shape}")
    print(f"Policy sum: {policy.sum(dim=1)}")  # Should be 1.0
    print(f"Sample policy: {policy[0]}")  # Token 2,3 should be 0 (masked)
    
    # Test policy-only forward
    logits = model5.forward_policy_only(x, legal_mask)
    print(f"Policy logits: {logits.shape}")

