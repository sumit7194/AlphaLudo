
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


class AlphaLudoTopNet(nn.Module):
    """
    Mastery Level AlphaLudo Network - 18 Channel Async Plan.
    
    Architecture:
    - Input: (B, 18, 15, 15) - Full spatial stack (Pieces, Home, Safe, Dice, Context)
    - Stem: 18 -> 128 Filters
    - Backbone: ResNet-10 (128 filters)
    - Global Average Pooling -> (B, 128)
    - Policy Head: FC(128->256->225)
    - Value Head: FC(128->256->1)
    """
    def __init__(self, num_res_blocks=10, num_channels=128, in_channels=18):
        super(AlphaLudoTopNet, self).__init__()
        
        self.num_channels = num_channels
        
        # Stem: 18 Input Channels -> 128 Filters
        self.conv_input = nn.Conv2d(in_channels, num_channels, kernel_size=3, padding=1, bias=False)
        self.bn_input = nn.BatchNorm2d(num_channels)
        
        # Backbone: 10 Residual Blocks
        self.res_blocks = nn.ModuleList([
            ResidualBlock(num_channels) for _ in range(num_res_blocks)
        ])
        
        # Global Average Pooling output size: 128
        feature_size = num_channels
        
        # --- Policy Head ---
        self.policy_fc1 = nn.Linear(feature_size, 256)
        self.policy_fc2 = nn.Linear(256, 225)
        
        # --- Value Head ---
        self.value_fc1 = nn.Linear(feature_size, 256)
        self.value_fc2 = nn.Linear(256, 1)
        
    def forward(self, x):
        """
        Forward pass with single spatial tensor.
        
        Args:
            x: (B, 18, 15, 15) spatial tensor
            
        Returns:
            policy_logits: (B, 225)
            value: (B, 1)
        """
        # Spatial backbone
        out = F.relu(self.bn_input(self.conv_input(x)))
        
        for block in self.res_blocks:
            out = block(out)
        
        # Global Average Pooling: (B, 128, 15, 15) -> (B, 128)
        out = F.adaptive_avg_pool2d(out, 1)
        features = out.flatten(start_dim=1)  # (B, 128)
        
        # Policy Head
        p = F.relu(self.policy_fc1(features))
        policy_logits = self.policy_fc2(p)  # (B, 225)
        
        # Value Head
        v = F.relu(self.value_fc1(features))
        value = torch.tanh(self.value_fc2(v))  # (B, 1)
        
        return policy_logits, value
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = AlphaLudoTopNet()
    print(f"Mastery Model Parameters: {model.count_parameters():,}")
    x = torch.randn(2, 18, 15, 15)
    logits, val = model(x)
    print(f"Logits: {logits.shape}, Value: {val.shape}")


