"""
Elo Rating System for TD-Ludo

Tracks Elo ratings for:
- Main model (current training model)
- Ghost snapshots (past versions of the model)
- Heuristic bots (Aggressive, Defensive, Racing, Random)

Used for:
- Measuring model improvement over time
- Smart ghost selection (adversarial/matched training)
- Elo-based ghost pruning (keep strongest opponents)
- Dashboard visualization
"""

import os
import json
import numpy as np
from collections import defaultdict


class EloTracker:
    """
    N-player Elo rating system that ignores inactive seats.

    In an N-player game, the winner beats all active losers.
    Each win/loss pair contributes K / (N - 1) to the rating change.
    """
    
    def __init__(self, k_factor=32, initial_rating=1500, save_path=None):
        self.k = k_factor
        self.initial = initial_rating
        self.ratings = {}          # name -> current Elo
        self.history = defaultdict(list)  # name -> [(game_num, elo), ...]
        self.save_path = save_path
        
        if save_path and os.path.exists(save_path):
            self.load()
    
    def get_rating(self, name):
        """Get rating for a model/bot, initializing if first time."""
        if name not in self.ratings:
            self.ratings[name] = self.initial
        return self.ratings[name]
    
    def expected_score(self, rating_a, rating_b):
        """Expected score of A vs B (standard Elo formula)."""
        return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400))
    
    def update_from_game(self, identities, winner_idx, game_num=None):
        """
        Update Elo ratings from a game result.
        
        Args:
            identities: List of 4 names (e.g. ['Main', 'Heuristic', 'Main', 'Main'])
            winner_idx: Index 0-3 of winner, or -1 for no winner
            game_num: Optional game number for history tracking
        """
        if winner_idx < 0 or winner_idx >= 4:
            return

        active_entries = [
            (idx, name) for idx, name in enumerate(identities)
            if name != 'Inactive'
        ]
        if len(active_entries) < 2:
            return

        active_indices = {idx for idx, _ in active_entries}
        if winner_idx not in active_indices:
            return

        winner_name = identities[winner_idx]
        loser_names = [
            name for idx, name in active_entries
            if idx != winner_idx
        ]
        if not loser_names:
            return

        winner_rating = self.get_rating(winner_name)
        weight = self.k / max(1, len(loser_names))

        for loser_name in loser_names:
            if loser_name == winner_name:
                continue
            loser_rating = self.get_rating(loser_name)
            
            expected_win = self.expected_score(winner_rating, loser_rating)
            expected_lose = self.expected_score(loser_rating, winner_rating)
            
            self.ratings[winner_name] += weight * (1 - expected_win)
            self.ratings[loser_name] += weight * (0 - expected_lose)
        
        # Record history (capped at 5000 entries per player to prevent memory leak)
        if game_num is not None:
            max_history = 5000
            for _, name in active_entries:
                self.history[name].append((game_num, round(self.ratings[name], 1)))
                if len(self.history[name]) > max_history:
                    self.history[name] = self.history[name][-max_history:]
    
    def select_ghost(self, ghost_pool, main_name='Main', strategy='adversarial'):
        """
        Select a ghost for the next game based on Elo ratings.
        
        Strategies:
        - 'adversarial': Prefer stronger ghosts (higher Elo)
        - 'matched': Prefer ghosts with similar Elo to Main
        - 'random': Random selection
        
        Args:
            ghost_pool: List of ghost file paths
            main_name: Name of main model for comparison
            strategy: Selection strategy
            
        Returns:
            Selected ghost path, or None
        """
        if not ghost_pool:
            return None
        
        if strategy == 'random':
            return ghost_pool[np.random.randint(len(ghost_pool))]
        
        main_rating = self.get_rating(main_name)
        
        ghost_ratings = []
        for path in ghost_pool:
            name = os.path.basename(path).replace('.pt', '')
            rating = self.get_rating(name)
            ghost_ratings.append((path, name, rating))
        
        if strategy == 'adversarial':
            # Weight toward stronger ghosts
            weights = []
            for _, _, rating in ghost_ratings:
                weight = max(1, rating - main_rating + 200)
                weights.append(weight)
        elif strategy == 'matched':
            # Weight toward similar-rated ghosts
            weights = []
            for _, _, rating in ghost_ratings:
                diff = abs(rating - main_rating)
                weight = max(1, 200 - diff)
                weights.append(weight)
        else:
            weights = [1.0] * len(ghost_ratings)
        
        weights = np.array(weights, dtype=float)
        weights /= weights.sum()
        idx = np.random.choice(len(ghost_pool), p=weights)
        return ghost_pool[idx]
    
    def get_weakest_ghost(self, ghost_dir):
        """
        Find the ghost with the lowest Elo rating (candidate for pruning).
        
        Returns:
            (path, name, elo) of weakest ghost, or None
        """
        if not os.path.exists(ghost_dir):
            return None
        
        ghosts = [f for f in os.listdir(ghost_dir) 
                   if f.startswith('ghost_') and f.endswith('.pt')]
        
        if not ghosts:
            return None
        
        weakest = None
        weakest_elo = float('inf')
        
        for fname in ghosts:
            name = fname.replace('.pt', '')
            elo = self.get_rating(name)
            if elo < weakest_elo:
                weakest_elo = elo
                weakest = (os.path.join(ghost_dir, fname), name, elo)
        
        return weakest
    
    def get_rankings(self, top_n=None):
        """Get all models sorted by Elo (descending)."""
        ranked = sorted(self.ratings.items(), key=lambda x: -x[1])
        if top_n:
            return ranked[:top_n]
        return ranked
    
    def get_history_for_dashboard(self, names=None, max_points=200):
        """
        Get Elo history formatted for dashboard charts.
        
        Returns:
            dict: {name: [(game, elo), ...], ...}
        """
        result = {}
        for name, history in self.history.items():
            if names and name not in names:
                continue
            # Downsample if too many points
            if len(history) > max_points:
                step = len(history) // max_points
                result[name] = history[::step]
            else:
                result[name] = list(history)
        return result
    
    def to_dict(self):
        """Serialize for JSON API response."""
        rankings = self.get_rankings(top_n=20)
        return {
            'rankings': [{'name': n, 'elo': round(e, 1)} for n, e in rankings],
            'main_elo': round(self.get_rating('Main'), 1),
            'history': {
                name: [{'game': g, 'elo': e} for g, e in pts[-200:]]
                for name, pts in self.history.items()
            },
        }
    
    def save(self):
        """Persist to disk."""
        if not self.save_path:
            return
        try:
            data = {
                'ratings': self.ratings,
                'history': {k: list(v) for k, v in self.history.items()},
            }
            tmp = self.save_path + '.tmp'
            with open(tmp, 'w') as f:
                json.dump(data, f, indent=2)
            os.replace(tmp, self.save_path)
        except Exception as e:
            print(f"[Elo] Save failed: {e}")
    
    def load(self):
        """Load from disk."""
        if not self.save_path or not os.path.exists(self.save_path):
            return
        try:
            with open(self.save_path, 'r') as f:
                data = json.load(f)
            self.ratings = data.get('ratings', {})
            self.history = defaultdict(list, {
                k: [(g, e) for g, e in v]
                for k, v in data.get('history', {}).items()
            })
            print(f"[Elo] Loaded {len(self.ratings)} ratings")
        except Exception as e:
            print(f"[Elo] Load failed: {e}")
    
    def __str__(self):
        rankings = self.get_rankings(top_n=10)
        lines = ["Elo Rankings:"]
        for name, rating in rankings:
            lines.append(f"  {name}: {rating:.0f}")
        return "\n".join(lines)
