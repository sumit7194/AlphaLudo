"""
Trainer for AlphaLudo.

Updates neural network weights using self-play data.

Features:
- Apple MPS (Metal) GPU support for M1/M2/M3 Macs
- Learning rate scheduling
- Gradient clipping
- Full checkpoint save/load (model + optimizer)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import os


def get_device():
    """Get the best available device (MPS > CUDA > CPU)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


class Trainer:
    """
    Trains the neural network on self-play data.
    """
    
    def __init__(self, model, learning_rate=0.001, weight_decay=1e-4, device=None):
        """
        Args:
            model: Neural network model.
            learning_rate: Learning rate for optimizer.
            weight_decay: L2 regularization weight.
            device: Device to train on (auto-detected if None).
        """
        self.device = device or get_device()
        self.model = model.to(self.device)
        self.learning_rate = learning_rate
        
        self.optimizer = optim.Adam(
            model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler (reduce on plateau)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10
        )
        
        self.policy_loss_fn = nn.CrossEntropyLoss()
        self.value_loss_fn = nn.MSELoss()
        
        # Training stats
        self.total_steps = 0
        self.total_epochs = 0
        
        print(f"Trainer initialized on device: {self.device}")
    
    def train_step(self, states_spatial, states_scalar, policy_targets, value_targets):
        """
        Perform one training step with spatial + scalar inputs.
        
        Args:
            states_spatial: Batch of spatial tensors (B, 9, 15, 15).
            states_scalar: Batch of scalar tensors (B, 4).
            policy_targets: Batch of policy targets (B, 225).
            value_targets: Batch of value targets (B, 1).
            
        Returns:
            Tuple of (total_loss, policy_loss, value_loss).
        """
        self.model.train()
        
        # Move to device
        states_spatial = states_spatial.to(self.device)
        states_scalar = states_scalar.to(self.device)
        policy_targets = policy_targets.to(self.device)
        value_targets = value_targets.to(self.device)
        
        self.optimizer.zero_grad()
        
        # Forward pass (2 outputs: policy, value)
        policy_pred_log, value_pred = self.model(states_spatial, states_scalar)
        
        # Convert logits to probs (apply softmax)
        policy_pred = torch.softmax(policy_pred_log, dim=1)
        
        # Policy Loss: Cross-Entropy (soft targets)
        policy_loss = -(policy_targets * torch.log(policy_pred + 1e-8)).sum(dim=1).mean()
        
        # Value Loss: MSE
        value_loss = self.value_loss_fn(value_pred, value_targets)
        
        # Total Loss
        total_loss = policy_loss + value_loss
        
        # Backward pass
        total_loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        self.total_steps += 1
        
        return total_loss.item(), policy_loss.item(), value_loss.item()
    
    def train_epoch(self, replay_buffer, batch_size=32, num_batches=100):
        """
        Train for multiple batches.
        
        Args:
            replay_buffer: ReplayBuffer with training examples.
            batch_size: Mini-batch size.
            num_batches: Number of batches to train.
            
        Returns:
            Average losses (total, policy, value).
        """
        if len(replay_buffer) < batch_size:
            print(f"Not enough data in buffer ({len(replay_buffer)} < {batch_size})")
            return 0, 0, 0
        
        total_losses = []
        policy_losses = []
        value_losses = []
        
        for _ in range(num_batches):
            # Note: Buffer must now return (spatial, scalar, policy, value)
            states_spatial, states_scalar, policies, values = replay_buffer.sample(batch_size)
            total, policy, value = self.train_step(states_spatial, states_scalar, policies, values)
            total_losses.append(total)
            policy_losses.append(policy)
            value_losses.append(value)
        
        avg_total = sum(total_losses) / len(total_losses)
        avg_policy = sum(policy_losses) / len(policy_losses)
        avg_value = sum(value_losses) / len(value_losses)
        
        # Update learning rate scheduler
        self.scheduler.step(avg_total)
        self.total_epochs += 1
        
        return avg_total, avg_policy, avg_value
    
    def save_checkpoint(self, path):
        """
        Save full checkpoint atomically (write to tmp -> rename).
        Prevents corruption if process is killed during write.
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'total_steps': self.total_steps,
            'total_epochs': self.total_epochs,
            'learning_rate': self.learning_rate,
        }
        
        # Atomic Write Pattern
        tmp_path = path + ".tmp"
        try:
            torch.save(checkpoint, tmp_path)
            # Ensure flush to disk
            pass # torch.save usually handles file, but os.fsync needed if we had fd.
            # Torch save closes file, OS might buffer.
            # Rename is atomic on POSIX
            os.replace(tmp_path, path)
            print(f"Checkpoint saved (Atomic): {path}")
        except Exception as e:
            print(f"Failed to save checkpoint to {path}: {e}")
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    
    def load_checkpoint(self, path):
        """
        Load full checkpoint.
        """
        if not os.path.exists(path):
            print(f"No checkpoint found at {path}")
            return False
        
        checkpoint = torch.load(path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['state_dict'])
        else:
            # Maybe it's just the state dict
            try:
                self.model.load_state_dict(checkpoint)
            except:
                print("Could not load model state dict")
                return False

        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint and self.scheduler is not None:
             self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.total_steps = checkpoint.get('total_steps', 0)
        self.total_epochs = checkpoint.get('total_epochs', 0)
        
        print(f"Checkpoint loaded: {path} (epoch {self.total_epochs}, step {self.total_steps})")
        return True
    
    def save_model(self, path):
        """Save model weights only (for inference)."""
        torch.save(self.model.state_dict(), path)
    
    def load_model(self, path):
        """Load model weights only."""
        self.model.load_state_dict(torch.load(path, map_location=self.device))
    
    def get_current_lr(self):
        """Get current learning rate."""
        return self.optimizer.param_groups[0]['lr']
