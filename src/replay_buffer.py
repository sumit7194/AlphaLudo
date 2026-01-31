"""
Replay Buffer for AlphaLudo.

Stores training examples and provides sampling for mini-batch training.
"""

import random
from collections import deque
import torch


class ReplayBuffer:
    """
    Fixed-size buffer to store training examples.
    
    Each example is a tuple: (state_tensor, policy_target, value_target)
    """
    
    def __init__(self, max_size=50000):
        """
        Args:
            max_size: Maximum number of examples to store. Oldest are removed first.
        """
        self.buffer = deque(maxlen=max_size)
    
    def add(self, examples):
        """
        Add examples to the buffer.
        
        Args:
            examples: List of (state, policy, value) tuples.
        """
        for example in examples:
            self.buffer.append(example)
    
    def sample(self, batch_size):
        """
        Sample a random mini-batch of examples.
        
        Args:
            batch_size: Number of examples to sample.
            
        Returns:
            Tuple of (states, policies, values) tensors.
        """
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
        
        batch = random.sample(list(self.buffer), batch_size)
        
        states = torch.stack([ex[0] for ex in batch])
        policies = torch.stack([ex[1] for ex in batch])
        values = torch.stack([ex[2] for ex in batch])
        
        return states, policies, values
    
    def __len__(self):
        return len(self.buffer)
    
    def clear(self):
        """Clear the buffer."""
        self.buffer.clear()
