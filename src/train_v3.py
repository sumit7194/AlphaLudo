"""
AlphaLudo v3 Trainer - Enhanced Training with TD(λ) and Auxiliary Losses

Key Improvements:
1. Supports 4-action policy output (token selection)
2. TD(λ) value targets for better credit assignment
3. Auxiliary safety loss for faster value learning
4. Learning rate warmup
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from src.model_v3 import AlphaLudoV3
from src.config import LEARNING_RATE, LR_WARMUP_STEPS, TD_LAMBDA, TD_GAMMA, AUX_LOSS_WEIGHT


def compute_td_lambda_returns(values, final_outcome, gamma=0.99, lambda_=0.95):
    """
    Compute TD(λ) returns for a sequence of states.
    
    Instead of assigning final outcome to all states equally,
    use bootstrapped returns with decay for better credit assignment.
    
    Args:
        values: List of predicted values at each state (from eval, not training)
        final_outcome: +1 for win, -1 for loss
        gamma: Discount factor
        lambda_: TD(λ) mixing parameter
        
    Returns:
        List of TD(λ) targets for each state
    """
    n = len(values)
    if n == 0:
        return []
    
    returns = [0.0] * n
    G = final_outcome  # Bootstrap from final outcome
    
    for t in reversed(range(n)):
        # TD(λ) target: blend between MC (final outcome) and TD (next value)
        if t == n - 1:
            returns[t] = final_outcome  # Last state uses actual outcome
        else:
            # G_t = r + γ * ((1-λ) * V(s_{t+1}) + λ * G_{t+1})
            # For Ludo, r=0 for all intermediate steps
            next_val = values[t + 1] if t + 1 < n else final_outcome
            returns[t] = gamma * ((1 - lambda_) * next_val + lambda_ * G)
            G = returns[t]
    
    return returns


class TrainerV3:
    """
    v3 Trainer with TD(λ), auxiliary loss, and LR warmup.
    """
    def __init__(self, model, device, learning_rate=LEARNING_RATE, warmup_steps=LR_WARMUP_STEPS):
        self.model = model
        self.device = device
        self.base_lr = learning_rate
        self.warmup_steps = warmup_steps
        self.current_step = 0
        self.total_epochs = 0
        
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        self.model.to(device)
        
    def get_lr(self):
        """Get current learning rate with linear warmup."""
        if self.current_step < self.warmup_steps:
            return self.base_lr * (self.current_step + 1) / self.warmup_steps
        return self.base_lr
    
    def update_lr(self):
        """Update learning rate in optimizer."""
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr
        
    def train_step(self, states, target_policies, target_values, 
                   legal_masks=None, safety_targets=None):
        """
        v3 Training step with 4-action policy.
        
        Args:
            states: (B, 18, 15, 15) - Spatial tensor
            target_policies: (B, 4) - Token selection probabilities
            target_values: (B,) or (B, 1) - TD(λ) value targets
            legal_masks: (B, 4) - Optional legal move masks
            safety_targets: (B, 4) - Optional token safety targets for aux loss
            
        Returns:
            total_loss, policy_loss, value_loss, aux_loss (all floats)
        """
        self.model.train()
        self.current_step += 1
        self.update_lr()
        
        # Move to device
        states = states.to(self.device)
        target_policies = target_policies.to(self.device)
        target_values = target_values.to(self.device)
        
        if legal_masks is not None:
            legal_masks = legal_masks.to(self.device)
        
        if target_values.dim() == 1:
            target_values = target_values.unsqueeze(1)
            
        # --- NaN Checks ---
        if torch.isnan(states).any():
            print("[TrainerV3] CRITICAL: Input 'states' contains NaN!")
            return float('nan'), 0.0, 0.0, 0.0
        if torch.isnan(target_policies).any():
            print("[TrainerV3] CRITICAL: Input 'target_policies' contains NaN!")
            return float('nan'), 0.0, 0.0, 0.0
        if torch.isnan(target_values).any():
            print("[TrainerV3] CRITICAL: Input 'target_values' contains NaN!")
            return float('nan'), 0.0, 0.0, 0.0

        # Forward pass
        policy, value, aux_safety = self.model(states, legal_masks)
        
        # --- Check Outputs ---
        if torch.isnan(policy).any():
            print("[TrainerV3] CRITICAL: Model produced NaN policy!")
            return float('nan'), 0.0, 0.0, 0.0

        # --- Value Loss (MSE) ---
        value_loss = F.mse_loss(value, target_values)
        
        # --- Policy Loss (Cross-Entropy with soft targets) ---
        # policy is already softmax'd, need log for KL div
        # Use KL divergence: KL(target || pred) = sum(target * log(target/pred))
        # Or just cross-entropy: -sum(target * log(pred))
        policy_log = torch.log(policy + 1e-8)
        policy_loss = -torch.sum(target_policies * policy_log, dim=1).mean()
        
        # --- Auxiliary Safety Loss (optional) ---
        aux_loss = torch.tensor(0.0, device=self.device)
        if safety_targets is not None:
            safety_targets = safety_targets.to(self.device)
            aux_loss = F.binary_cross_entropy(aux_safety, safety_targets)
        
        # --- Total Loss ---
        total_loss = policy_loss + value_loss + AUX_LOSS_WEIGHT * aux_loss
        
        if torch.isnan(total_loss):
            print(f"[TrainerV3] Warning: Loss is NaN. P={policy_loss.item():.4f}, V={value_loss.item():.4f}")
            return float('nan'), policy_loss.item(), value_loss.item(), aux_loss.item()
            
        # Backward
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        return total_loss.item(), policy_loss.item(), value_loss.item(), aux_loss.item()
    
    def save_checkpoint(self, path):
        """Save model checkpoint."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'total_epochs': self.total_epochs,
            'current_step': self.current_step
        }, path)

    def load_checkpoint(self, path):
        """Load model checkpoint. Returns True if successful."""
        if not os.path.exists(path):
            return False
        try:
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.total_epochs = checkpoint.get('total_epochs', 0)
            self.current_step = checkpoint.get('current_step', 0)
            return True
        except Exception as e:
            print(f"[TrainerV3] Failed to load checkpoint: {e}")
            return False


if __name__ == "__main__":
    # Quick test
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = AlphaLudoV3()
    trainer = TrainerV3(model, device)
    
    print(f"Model parameters: {model.count_parameters():,}")
    
    # Test training step
    states = torch.randn(16, 18, 15, 15)
    policies = torch.zeros(16, 4)
    policies[:, 0] = 0.5
    policies[:, 1] = 0.5
    values = torch.randn(16)
    legal_masks = torch.ones(16, 4)
    
    loss, p_loss, v_loss, aux_loss = trainer.train_step(states, policies, values, legal_masks)
    print(f"Loss: {loss:.4f}, Policy: {p_loss:.4f}, Value: {v_loss:.4f}, Aux: {aux_loss:.4f}")
