"""
AlphaLudo Neural Network (Optimized for Speed)

Optimization:
1. Removed SE blocks (too slow)
2. Use Global Average Pooling (GAP) to reduce Linear layer size drastically
3. Reduced head channels
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """Standard Residual block."""
    def __init__(self, channels):
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


class AlphaLudoNet(nn.Module):
    """
    Fast AlphaLudo Network.
    Uses Global Average Pooling to minimize parameters and compute.
    """
    def __init__(self, num_res_blocks=8, num_channels=64, dropout=0.1):
        super(AlphaLudoNet, self).__init__()
        
        self.num_channels = num_channels
        
        # Stem
        self.conv_input = nn.Conv2d(8, num_channels, kernel_size=3, padding=1, bias=False)
        self.bn_input = nn.BatchNorm2d(num_channels)
        
        # Backbone
        self.res_blocks = nn.ModuleList([
            ResidualBlock(num_channels) for _ in range(num_res_blocks)
        ])
        
        # Policy Head (Actor)
        # Conv 1x1 -> BN -> ReLU -> GAP -> Linear -> Softmax
        self.policy_conv = nn.Conv2d(num_channels, 32, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(32)
        # Global Average Pool will make it 32 values
        self.policy_fc1 = nn.Linear(32, 128) 
        self.policy_dropout = nn.Dropout(dropout)
        self.policy_fc2 = nn.Linear(128, 4)
        
        # Value Head (Critic)
        # Conv 1x1 -> BN -> ReLU -> GAP -> Linear -> Tanh
        self.value_conv = nn.Conv2d(num_channels, 16, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(16)
        # Global Average Pool will make it 16 values
        self.value_fc1 = nn.Linear(16, 64)
        self.value_dropout = nn.Dropout(dropout)
        self.value_fc2 = nn.Linear(64, 1)
        
    def forward(self, x):
        # Input: (B, 8, 15, 15)
        out = F.relu(self.bn_input(self.conv_input(x)))
        
        # Backbone
        for block in self.res_blocks:
            out = block(out)
        
        # Policy Head
        p = F.relu(self.policy_bn(self.policy_conv(out)))
        p = F.adaptive_avg_pool2d(p, 1).view(p.size(0), -1) # Global Average Pool
        p = F.relu(self.policy_fc1(p))
        p = self.policy_dropout(p)
        policy_logits = self.policy_fc2(p)
        policy = F.softmax(policy_logits, dim=1)
        
        # Value Head
        v = F.relu(self.value_bn(self.value_conv(out)))
        v = F.adaptive_avg_pool2d(v, 1).view(v.size(0), -1) # Global Average Pool
        v = F.relu(self.value_fc1(v))
        v = self.value_dropout(v)
        value = torch.tanh(self.value_fc2(v))
        
        return policy, value
    
    def count_parameters(self):
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

if __name__ == "__main__":
    model = AlphaLudoNet()
    print(f"Model parameters: {model.count_parameters():,}")
    x = torch.randn(2, 8, 15, 15)
    policy, value = model(x)
    print(f"Policy shape: {policy.shape}, Value shape: {value.shape}")
