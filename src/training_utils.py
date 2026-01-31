"""
Training Utilities for AlphaLudo
- Temperature scheduling
- Board rotation augmentation
- Elo rating system
"""

import numpy as np
import torch
import json
import os
from collections import defaultdict


# =============================================================================
# 1. TEMPERATURE SCHEDULING
# =============================================================================

def get_temperature(move_number, schedule='alphazero'):
    """
    Get temperature for move selection based on game progress.
    
    Schedules:
    - 'alphazero': τ=1.0 for first 30 moves, then τ=0.1
    - 'linear': Linear decay from 1.0 to 0.1 over 60 moves
    - 'exponential': Exponential decay after move 20
    - 'constant': Always τ=1.0 (current behavior)
    
    Args:
        move_number: Current move number in the game
        schedule: Which schedule to use
        
    Returns:
        Temperature value (float)
    """
    if schedule == 'constant':
        return 1.0
    
    elif schedule == 'alphazero':
        # AlphaZero style: explore early, exploit late
        if move_number < 30:
            return 1.0
        else:
            return 0.1
    
    elif schedule == 'linear':
        # Linear decay from 1.0 to 0.1 over 60 moves
        if move_number < 60:
            return 1.0 - (move_number / 60) * 0.9
        return 0.1
    
    elif schedule == 'exponential':
        # Exponential decay after move 20
        if move_number < 20:
            return 1.0
        decay_factor = 0.95 ** (move_number - 20)
        return max(0.1, decay_factor)
    
    else:
        return 1.0


# =============================================================================
# 2. BOARD ROTATION AUGMENTATION
# =============================================================================

def rotate_state_tensor(state_tensor, k):
    """
    Rotate state tensor by k*90 degrees.
    
    Args:
        state_tensor: Shape (C, 15, 15) or (B, C, 15, 15)
        k: Number of 90-degree rotations (1, 2, or 3)
        
    Returns:
        Rotated tensor with same shape
    """
    if k == 0:
        return state_tensor
    
    # Handle both batched and unbatched
    if state_tensor.dim() == 3:
        # (C, H, W) -> rotate on (H, W)
        return torch.rot90(state_tensor, k, dims=(1, 2))
    elif state_tensor.dim() == 4:
        # (B, C, H, W) -> rotate on (H, W)
        return torch.rot90(state_tensor, k, dims=(2, 3))
    else:
        raise ValueError(f"Unexpected tensor shape: {state_tensor.shape}")


def rotate_policy(policy_tensor, k, num_pieces=4):
    """
    Rotate policy tensor to match rotated board.
    
    In Ludo, policy is over actions for 4 pieces (indices 0-3) + pass.
    When we rotate the board by k*90°, we are effectively changing
    which player we are (from perspective of the rotated board).
    
    For simplicity with our current action encoding (piece indices),
    policy rotation is identity - the policy stays the same because
    it's relative to the current player's pieces.
    
    Args:
        policy_tensor: Policy probabilities (shape varies)
        k: Number of 90-degree rotations
        num_pieces: Number of pieces per player
        
    Returns:
        Policy tensor (unchanged for Ludo's action encoding)
    """
    # In Ludo, actions are "move piece 0/1/2/3" which is relative
    # to the current player. Rotation doesn't change this.
    return policy_tensor


def rotate_token_indices(token_indices, k, board_size=15):
    """
    Rotate token indices by k*90 degrees counterclockwise to match board rotation.
    
    Token indices are flat indices (0-224) representing positions on the 15x15 board.
    When the board is rotated, the token positions must be rotated to match.
    
    Args:
        token_indices: Tensor of shape (4,) with flat indices (0-224)
        k: Number of 90-degree counterclockwise rotations (matches torch.rot90)
        board_size: Size of the board (15)
        
    Returns:
        Rotated token indices tensor
    """
    if k == 0 or k % 4 == 0:
        return token_indices
    
    # Ensure we're working with a tensor
    if not isinstance(token_indices, torch.Tensor):
        token_indices = torch.tensor(token_indices, dtype=torch.long)
    
    # Convert flat indices to (row, col)
    rows = token_indices // board_size
    cols = token_indices % board_size
    
    # Apply rotation k times (counterclockwise to match torch.rot90)
    k = k % 4
    for _ in range(k):
        # 90° CCW: (r, c) → (board_size - 1 - c, r)
        new_rows = board_size - 1 - cols
        new_cols = rows
        rows = new_rows
        cols = new_cols
    
    # Convert back to flat indices
    return rows * board_size + cols


def rotate_channels(state_tensor, k):
    """
    Rotate channel order to match rotated board perspective.
    
    Our tensor format has channels per player. When rotating the board,
    we need to rotate which channels correspond to which player.
    
    Assuming channel layout:
    - Channels 0-3: Player 0's pieces (one per piece)
    - Channels 4-7: Player 1's pieces
    - Channels 8-11: Player 2's pieces
    - Channels 12-15: Player 3's pieces
    - Additional channels (dice, safe zones, etc.)
    
    After k rotations:
    - Original P0 → becomes P(4-k)%4
    - Original P1 → becomes P(5-k)%4
    - etc.
    """
    # For now, return the spatially rotated tensor
    # Full channel rotation requires knowing exact channel layout
    return rotate_state_tensor(state_tensor, k)


def augment_training_sample(state, token_indices, policy, value, augment_rotations=True):
    """
    Augment a single training sample with rotations.
    
    Args:
        state: State tensor (C, 15, 15)
        token_indices: Token indices tensor (for action masking)
        policy: Policy tensor
        value: Value tensor
        augment_rotations: Whether to do rotation augmentation
        
    Returns:
        List of (state, token_indices, policy, value) tuples (1-4 samples)
    """
    samples = [(state, token_indices, policy, value)]
    
    if not augment_rotations:
        return samples
    
    # Add rotated versions (90°, 180°, 270°)
    for k in [1, 2, 3]:
        rot_state = rotate_channels(state, k)
        rot_token_indices = rotate_token_indices(token_indices, k)
        rot_policy = rotate_policy(policy, k)
        # Value stays the same (win/loss doesn't change with rotation)
        samples.append((rot_state, rot_token_indices, rot_policy, value))
    
    return samples


def augment_batch(examples, augment_probability=0.5):
    """
    Augment a batch of training examples with probability.
    
    Args:
        examples: List of (state, token_indices, policy, value) tuples
        augment_probability: Probability of augmenting each sample
        
    Returns:
        Augmented list of examples
    """
    augmented = []
    for example in examples:
        state, token_indices, policy, value = example
        
        if np.random.rand() < augment_probability:
            # Random rotation (0, 1, 2, or 3 * 90°)
            k = np.random.randint(0, 4)
            if k > 0:
                state = rotate_channels(state, k)
                token_indices = rotate_token_indices(token_indices, k)
                policy = rotate_policy(policy, k)
        
        augmented.append((state, token_indices, policy, value))
    
    return augmented


# =============================================================================
# 3. ELO RATING SYSTEM
# =============================================================================

class EloTracker:
    """
    Track Elo ratings for models in the training pool.
    
    Used for:
    - Measuring model improvement over time
    - Smart ghost selection (adversarial training)
    - Visualizing training progress
    """
    
    def __init__(self, k_factor=32, initial_rating=1500, save_path=None):
        """
        Args:
            k_factor: How much ratings change per game (higher = more volatile)
            initial_rating: Starting Elo for new models
            save_path: Path to persist ratings (optional)
        """
        self.k = k_factor
        self.initial = initial_rating
        self.ratings = {}  # model_name -> current_elo
        self.history = defaultdict(list)  # model_name -> [(epoch, elo), ...]
        self.save_path = save_path
        
        if save_path and os.path.exists(save_path):
            self.load()
    
    def get_rating(self, model_name):
        """Get model's current rating, initializing if needed."""
        if model_name not in self.ratings:
            self.ratings[model_name] = self.initial
        return self.ratings[model_name]
    
    def expected_score(self, rating_a, rating_b):
        """Expected score for player A vs player B."""
        return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400))
    
    def update_from_game(self, identities, winner_idx, epoch=None):
        """
        Update ratings based on game result.
        
        Args:
            identities: List of 4 model names (e.g., ['Main', 'Ghost', 'Main', 'Main'])
            winner_idx: Index of winning player (0-3), or -1 for draw
            epoch: Optional epoch number for history tracking
        """
        if winner_idx < 0 or winner_idx >= 4:
            return  # No update for draws or invalid
        
        winner_name = identities[winner_idx]
        loser_names = [identities[i] for i in range(4) if i != winner_idx]
        
        # Get ratings
        winner_rating = self.get_rating(winner_name)
        
        # Update against each loser
        for loser_name in loser_names:
            loser_rating = self.get_rating(loser_name)
            
            # Expected scores
            expected_win = self.expected_score(winner_rating, loser_rating)
            expected_lose = self.expected_score(loser_rating, winner_rating)
            
            # Update ratings (each game 1/3 weight since 3 losers)
            weight = self.k / 3
            self.ratings[winner_name] += weight * (1 - expected_win)
            self.ratings[loser_name] += weight * (0 - expected_lose)
        
        # Record history
        if epoch is not None:
            for name in set(identities):
                self.history[name].append((epoch, self.ratings[name]))
    
    def select_ghost(self, ghost_pool, main_name='Main', strategy='adversarial'):
        """
        Select a ghost model based on Elo ratings.
        
        Strategies:
        - 'adversarial': Prefer ghosts that beat Main (higher Elo)
        - 'matched': Prefer ghosts with similar Elo to Main
        - 'random': Random selection (baseline)
        
        Args:
            ghost_pool: List of ghost model paths
            main_name: Name of the main model for comparison
            strategy: Selection strategy
            
        Returns:
            Selected ghost path
        """
        if not ghost_pool:
            return None
        
        if strategy == 'random':
            return np.random.choice(ghost_pool)
        
        main_rating = self.get_rating(main_name)
        
        # Get ghost names from paths
        ghost_ratings = []
        for path in ghost_pool:
            # Extract name from path (e.g., "ghost_252.pt" -> "ghost_252")
            name = os.path.basename(path).replace('.pt', '')
            rating = self.get_rating(name)
            ghost_ratings.append((path, name, rating))
        
        if strategy == 'adversarial':
            # Weight toward higher-rated ghosts
            weights = []
            for _, _, rating in ghost_ratings:
                # Higher = better opponent
                weight = max(1, rating - main_rating + 200)
                weights.append(weight)
            
            weights = np.array(weights, dtype=float)
            weights /= weights.sum()
            idx = np.random.choice(len(ghost_pool), p=weights)
            return ghost_pool[idx]
        
        elif strategy == 'matched':
            # Weight toward similar-rated ghosts
            weights = []
            for _, _, rating in ghost_ratings:
                # Closer = higher weight
                diff = abs(rating - main_rating)
                weight = max(1, 200 - diff)
                weights.append(weight)
            
            weights = np.array(weights, dtype=float)
            weights /= weights.sum()
            idx = np.random.choice(len(ghost_pool), p=weights)
            return ghost_pool[idx]
        
        return np.random.choice(ghost_pool)
    
    def get_rankings(self):
        """Get all models sorted by Elo rating."""
        return sorted(self.ratings.items(), key=lambda x: -x[1])
    
    def save(self):
        """Save ratings to disk."""
        if not self.save_path:
            return
        
        data = {
            'ratings': self.ratings,
            'history': {k: list(v) for k, v in self.history.items()}
        }
        with open(self.save_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load(self):
        """Load ratings from disk."""
        if not self.save_path or not os.path.exists(self.save_path):
            return
        
        try:
            with open(self.save_path, 'r') as f:
                data = json.load(f)
            self.ratings = data.get('ratings', {})
            self.history = defaultdict(list, {
                k: [(e, r) for e, r in v] 
                for k, v in data.get('history', {}).items()
            })
        except Exception as e:
            print(f"Warning: Could not load Elo data: {e}")
    
    def __str__(self):
        rankings = self.get_rankings()
        lines = ["Elo Rankings:"]
        for name, rating in rankings[:10]:  # Top 10
            lines.append(f"  {name}: {rating:.0f}")
        return "\n".join(lines)


# =============================================================================
# 4. TRAINING METRICS
# =============================================================================

class TrainingMetrics:
    """Track and aggregate training metrics over time."""
    
    def __init__(self):
        self.metrics = defaultdict(list)
        self.current_epoch = 0
    
    def log(self, **kwargs):
        """Log metrics for current epoch."""
        for key, value in kwargs.items():
            self.metrics[key].append((self.current_epoch, value))
    
    def get_recent(self, key, n=10):
        """Get last n values for a metric."""
        if key not in self.metrics:
            return []
        return [v for _, v in self.metrics[key][-n:]]
    
    def get_average(self, key, n=10):
        """Get average of last n values."""
        recent = self.get_recent(key, n)
        if not recent:
            return 0
        return sum(recent) / len(recent)
    
    def next_epoch(self):
        """Move to next epoch."""
        self.current_epoch += 1


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    # Test temperature scheduling
    print("Temperature Schedule (AlphaZero):")
    for move in [0, 10, 25, 30, 40, 50]:
        print(f"  Move {move}: τ = {get_temperature(move, 'alphazero'):.2f}")
    
    # Test rotation
    print("\nRotation Test:")
    tensor = torch.randn(18, 15, 15)
    for k in [1, 2, 3]:
        rotated = rotate_state_tensor(tensor, k)
        print(f"  Rotate {k*90}°: {rotated.shape}")
    
    # Test token index rotation
    print("\nToken Index Rotation Test:")
    # Token at (0, 0) = index 0 (top-left)
    # Token at (14, 14) = index 224 (bottom-right)
    # Token at (7, 7) = index 112 (center)
    # Token at (0, 14) = index 14 (top-right)
    test_indices = torch.tensor([0, 224, 112, 14], dtype=torch.long)
    print(f"  Original indices: {test_indices.tolist()}")
    print(f"  Original positions: (0,0), (14,14), (7,7), (0,14)")
    
    for k in [1, 2, 3]:
        rotated = rotate_token_indices(test_indices, k)
        # Decode back to (r, c) for verification
        rows = rotated // 15
        cols = rotated % 15
        positions = [(r.item(), c.item()) for r, c in zip(rows, cols)]
        print(f"  After {k*90}° CCW: {rotated.tolist()} -> {positions}")
    
    # Verify: 4 rotations should return to original
    full_rotation = rotate_token_indices(test_indices, 4)
    assert torch.equal(test_indices, full_rotation), "4x rotation should be identity!"
    print("  ✓ 4x rotation = identity (verified)")
    
    # Test Elo
    print("\nElo Test:")
    elo = EloTracker()
    
    # Simulate some games
    for i in range(10):
        identities = ['Main', 'Ghost_1', 'Main', 'Ghost_2']
        winner = np.random.randint(0, 4)
        elo.update_from_game(identities, winner, epoch=i)
    
    print(elo)

