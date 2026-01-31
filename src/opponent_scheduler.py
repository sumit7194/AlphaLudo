"""
Opponent Scheduler for Advanced Training Strategies.

Implements:
1. Dynamic probability scheduling based on skill gap
2. Win rate tracking per opponent type
3. Prioritized opponent sampling
"""

import json
import os
from collections import defaultdict


class OpponentScheduler:
    """
    Manages opponent distribution dynamically based on training progress.
    """
    
    # Available opponent types
    BOT_TYPES = ['Heuristic', 'Aggressive', 'Defensive', 'Racing']
    
    def __init__(self, save_path=None):
        self.save_path = save_path
        
        # Track win rates against each opponent type
        self.win_counts = defaultdict(int)
        self.game_counts = defaultdict(int)
        
        # Current probabilities
        self.probabilities = {
            'Main': 0.50,
            'Ghost': 0.20,
            'Heuristic': 0.10,
            'Aggressive': 0.07,
            'Defensive': 0.07,
            'Racing': 0.06,
        }
        
        # Load saved state if exists
        if save_path and os.path.exists(save_path):
            self.load()
    
    def get_probabilities(self, main_elo=None, heuristic_elo=None):
        """
        Get opponent probabilities, optionally adjusted by skill gap.
        
        Args:
            main_elo: Current Main model ELO
            heuristic_elo: Heuristic bot ELO
            
        Returns:
            Dict of opponent type -> probability
        """
        if main_elo is None or heuristic_elo is None:
            return self.probabilities.copy()
        
        gap = heuristic_elo - main_elo
        
        if gap > 200:
            # Far behind heuristic: heavy focus on bots
            return {
                'Main': 0.35,
                'Ghost': 0.15,
                'Heuristic': 0.15,
                'Aggressive': 0.12,
                'Defensive': 0.12,
                'Racing': 0.11,
            }
        elif gap > 100:
            # Catching up
            return {
                'Main': 0.45,
                'Ghost': 0.20,
                'Heuristic': 0.12,
                'Aggressive': 0.08,
                'Defensive': 0.08,
                'Racing': 0.07,
            }
        elif gap > 0:
            # Close
            return {
                'Main': 0.55,
                'Ghost': 0.22,
                'Heuristic': 0.08,
                'Aggressive': 0.05,
                'Defensive': 0.05,
                'Racing': 0.05,
            }
        else:
            # Ahead of heuristic: focus on self-play
            return {
                'Main': 0.65,
                'Ghost': 0.25,
                'Heuristic': 0.04,
                'Aggressive': 0.02,
                'Defensive': 0.02,
                'Racing': 0.02,
            }
    
    def assign_opponents(self, batch_size, main_elo=None, heuristic_elo=None):
        """
        Assign opponent types for each game in a batch.
        
        Returns:
            List of opponent types for each game index.
            'Main' means pure self-play.
        """
        import random
        
        probs = self.get_probabilities(main_elo, heuristic_elo)
        
        # Calculate counts
        assignments = []
        opponent_types = list(probs.keys())
        weights = [probs[t] for t in opponent_types]
        
        for _ in range(batch_size):
            selected = random.choices(opponent_types, weights=weights, k=1)[0]
            assignments.append(selected)
        
        return assignments
    
    def record_result(self, opponent_type, main_won):
        """Record game result for win rate tracking."""
        self.game_counts[opponent_type] += 1
        if main_won:
            self.win_counts[opponent_type] += 1
    
    def get_win_rate(self, opponent_type):
        """Get win rate against specific opponent."""
        if self.game_counts[opponent_type] == 0:
            return 0.5  # Unknown
        return self.win_counts[opponent_type] / self.game_counts[opponent_type]
    
    def get_all_win_rates(self):
        """Get all win rates."""
        return {
            t: self.get_win_rate(t) 
            for t in self.BOT_TYPES + ['Ghost', 'Main']
        }
    
    def save(self):
        """Save scheduler state."""
        if not self.save_path:
            return
        
        data = {
            'win_counts': dict(self.win_counts),
            'game_counts': dict(self.game_counts),
            'probabilities': self.probabilities,
        }
        
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        with open(self.save_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load(self):
        """Load scheduler state."""
        if not self.save_path or not os.path.exists(self.save_path):
            return
        
        try:
            with open(self.save_path, 'r') as f:
                data = json.load(f)
            
            self.win_counts = defaultdict(int, data.get('win_counts', {}))
            self.game_counts = defaultdict(int, data.get('game_counts', {}))
            self.probabilities = data.get('probabilities', self.probabilities)
        except Exception as e:
            print(f"Failed to load scheduler state: {e}")
    
    def __repr__(self):
        rates = self.get_all_win_rates()
        return f"OpponentScheduler(win_rates={rates})"
