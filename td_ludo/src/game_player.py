"""
TD-Ludo Game Player — Actor-Critic Policy-Based Move Selection

Core gameplay loop for the Actor-Critic training paradigm:
1. At each decision point, forward pass π(a|s) with legal mask
2. Sample action from the policy distribution (with temperature)
3. Collect trajectory for ALL players (model + bots)
4. On game end, send trajectories to trainer for REINFORCE update

Key changes from TD-Gammon approach:
- NO 1-ply lookahead (we don't simulate each possible move)
- ONE forward pass per decision instead of N (much faster)
- Trajectories collected for all players (opponent learning)
- Training happens at end-of-game, not per-step
"""

import random
import time
import numpy as np
import torch
import torch.nn.functional as F
import td_ludo_cpp as ludo_cpp

from src.heuristic_bot import HeuristicLudoBot, AggressiveBot, DefensiveBot, RacingBot, RandomBot
from src.reward_shaping import compute_shaped_reward
from src.config import (
    GAME_COMPOSITION, MAX_MOVES_PER_GAME,
    TEMPERATURE_START, TEMPERATURE_END, TEMPERATURE_DECAY_GAMES,
    NUM_ACTIVE_PLAYERS
)


# Bot registry
BOT_CLASSES = {
    'Heuristic': HeuristicLudoBot,
    'Aggressive': AggressiveBot,
    'Defensive': DefensiveBot,
    'Racing': RacingBot,
    'Random': RandomBot,
}


class VectorACGamePlayer:
    """
    Plays N parallel games using C++ VectorGameState and batched policy inference.
    
    Key differences from VectorTDGamePlayer:
    - Move selection via π(a|s) sampling (not V(s') argmax)
    - Collects trajectories for ALL players (model + bots)
    - Training happens at end of each game (not per-step)
    """
    def __init__(self, trainer, batch_size, device):
        self.trainer = trainer
        self.batch_size = batch_size
        self.device = device
        
        # Initialize Vector Env
        two_player = (NUM_ACTIVE_PLAYERS == 2)
        self.env = ludo_cpp.VectorGameState(batch_size, two_player)
        
        # Per-game state tracking
        self.game_compositions = [self._random_composition() for _ in range(batch_size)]
        
        # Trajectory storage: dict of player_id → list of (state_tensor, action, legal_mask)
        # One per game slot
        self.trajectories = [{} for _ in range(batch_size)]
        
        # Stats
        self.total_games = 0
        self.total_model_wins = 0
        self.recent_wins = []
        
        # Track consecutive sixes for all games (4 players per game)
        self.consecutive_sixes = np.zeros((batch_size, 4), dtype=int)
        
        # Track move counts for max_moves enforcement
        self.move_counts = np.zeros(batch_size, dtype=int)
        
        # Initialize bots (shared instances are fine for stateless bots)
        self.bots = {name: cls() for name, cls in BOT_CLASSES.items()}

    def get_temperature(self, total_games):
        """Calculate current temperature for policy sampling."""
        if total_games >= TEMPERATURE_DECAY_GAMES:
            return TEMPERATURE_END
        progress = total_games / TEMPERATURE_DECAY_GAMES
        return TEMPERATURE_START - progress * (TEMPERATURE_START - TEMPERATURE_END)

    def play_step(self, train=True):
        """
        Advance all games by one step.
        Returns list of finished game results (dicts).
        """
        # 1. Get current states for decision making
        current_states_np = self.env.get_state_tensor()
        
        actions = []
        current_players = []
        model_decision_indices = []  # Games where model needs to make a decision
        
        # 2. Determine actions for all games
        for i in range(self.batch_size):
            game = self.env.get_game(i)
            
            if game.is_terminal:
                actions.append(-1)
                current_players.append(-1)
                continue
                
            cp = game.current_player
            current_players.append(cp)
            
            # Check max moves
            if self.move_counts[i] >= MAX_MOVES_PER_GAME:
                game.is_terminal = True
                actions.append(-1)
                continue
            
            # Dice Roll Logic
            if game.current_dice_roll == 0:
                roll = random.randint(1, 6)
                game.current_dice_roll = roll
                if roll == 6:
                    self.consecutive_sixes[i, cp] += 1
                else:
                    self.consecutive_sixes[i, cp] = 0
                    
                if self.consecutive_sixes[i, cp] >= 3:
                    next_p = (cp + 1) % 4
                    while not game.active_players[next_p]:
                        next_p = (next_p + 1) % 4
                    game.current_player = next_p
                    game.current_dice_roll = 0
                    self.consecutive_sixes[i, cp] = 0
                    actions.append(-1)
                    continue

            legal_moves = ludo_cpp.get_legal_moves(game)
            if not legal_moves:
                next_p = (cp + 1) % 4
                while not game.active_players[next_p]:
                    next_p = (next_p + 1) % 4
                game.current_player = next_p
                game.current_dice_roll = 0
                actions.append(-1)
                continue
            
            ptype = self.game_compositions[i]['player_types'][cp]
            
            if ptype in ('Model', 'SelfPlay'):
                # Model makes the decision — will be batched below
                model_decision_indices.append(i)
                actions.append(-2)  # Placeholder marker
            else:
                # Bot makes the decision — no trajectory needed
                bot = self.bots.get(ptype, self.bots['Random'])
                action = bot.select_move(game, legal_moves)
                actions.append(action)

        # 3. Batched Model Action Selection via Policy Head
        if model_decision_indices:
            temperature = self.get_temperature(self.trainer.total_games)
            
            # Gather states and legal masks for all model decisions
            batch_states = []
            batch_legal_masks = []
            batch_legal_moves = []
            
            for idx in model_decision_indices:
                game = self.env.get_game(idx)
                lmoves = ludo_cpp.get_legal_moves(game)
                batch_legal_moves.append(lmoves)
                
                # State tensor from C++ encoder
                state_tensor = ludo_cpp.encode_state(game)
                batch_states.append(state_tensor)
                
                # Legal mask (4 tokens)
                legal_mask = np.zeros(4, dtype=np.float32)
                for m in lmoves:
                    legal_mask[m] = 1.0
                batch_legal_masks.append(legal_mask)
            
            # Batch forward pass — ONE pass for all model decisions
            states_t = torch.from_numpy(np.stack(batch_states)).to(self.device, dtype=torch.float32)
            masks_t = torch.from_numpy(np.stack(batch_legal_masks)).to(self.device, dtype=torch.float32)
            
            with torch.no_grad():
                policy_logits = self.trainer.model.forward_policy_only(states_t, masks_t)
                
                # Base probabilities at T=1.0 (for PPO old_log_prob — must match training)
                probs_base = F.softmax(policy_logits, dim=1)
                
                # Apply temperature for ACTION SAMPLING (exploration)
                if temperature != 1.0:
                    sample_logits = policy_logits / temperature
                    probs_sample = F.softmax(sample_logits, dim=1)
                else:
                    probs_sample = probs_base
                
                sampled_actions = torch.multinomial(probs_sample, num_samples=1).squeeze(1)
                
                # Compute old_log_prob from base (T=1.0) distribution for PPO ratio
                old_log_probs = torch.log(
                    probs_base.gather(1, sampled_actions.unsqueeze(1)).squeeze(1) + 1e-8
                )
            
            sampled_np = sampled_actions.cpu().numpy()
            old_lp_np = old_log_probs.cpu().numpy()
            
            for j, idx in enumerate(model_decision_indices):
                action = int(sampled_np[j])
                lmoves = batch_legal_moves[j]
                
                # Safety: ensure sampled action is actually legal
                if action not in lmoves:
                    action = random.choice(lmoves)
                
                actions[idx] = action
                
                # Store model's trajectory (with old_log_prob for PPO)
                if train:
                    cp = current_players[idx]
                    if cp not in self.trajectories[idx]:
                        self.trajectories[idx][cp] = []
                    self.trajectories[idx][cp].append((
                        batch_states[j],
                        action,
                        batch_legal_masks[j],
                        float(old_lp_np[j]),   # old_log_prob at T=1.0 for PPO
                    ))
        
        # Save pre-step states for reward computation
        pre_step_states = []
        for i in range(self.batch_size):
            # Must copy the state or extract needed info if GameState mutates.
            # Fortunately get_game returns a reference the C++ object, so we need to copy?
            # Actually, `compute_shaped_reward` only needs `player_positions`.
            # Let's extract original positions for all 4 players for all batch games.
            game = self.env.get_game(i)
            # Create a simple struct/dict to hold the old positions
            old_pos = {p: list(game.player_positions[p]) for p in range(4)}
            pre_step_states.append(old_pos)
            
        # 4. Step Environment
        final_actions = [a if a >= 0 else -1 for a in actions]
        for i, a in enumerate(final_actions):
            if a >= 0:
                self.move_counts[i] += 1
                
        next_states_np, rewards_np, dones_np, info_list = self.env.step(final_actions)
        
        # 5. Handle game completions — compute dense rewards and train
        results = []
        for i in range(self.batch_size):
            # Compute shaped reward for the move that just happened in this game.
            # Only if a valid move was made by a player we are tracking (train==True).
            cp = current_players[i]
            if train and cp >= 0 and self.trajectories[i] and cp in self.trajectories[i]:
                # We need to compute the dense reward for cp.
                # Since C++ GameState is mutated, self.env.get_game(i) is now the NEXT state.
                next_game = self.env.get_game(i)
                
                # compute_shaped_reward expects objects with .player_positions
                class DummyState:
                    def __init__(self, pos):
                        self.player_positions = pos
                
                dummy_old = DummyState(pre_step_states[i])
                dummy_new = DummyState(next_game.player_positions)
                
                step_reward = compute_shaped_reward(dummy_old, dummy_new, cp)
                
                # Append to the LAST trajectory entry for this player
                last_idx = len(self.trajectories[i][cp]) - 1
                if last_idx >= 0:
                    tup = self.trajectories[i][cp][last_idx]
                    # Currently tup is (state, action, mask, old_lp)
                    if len(tup) == 4:
                        self.trajectories[i][cp][last_idx] = tup + (step_reward,)
            if dones_np[i]:
                winner = info_list[i]['winner']
                mpid = self.game_compositions[i]['model_player']
                
                if winner == -1:
                    outcome = "Timeout"
                    model_won = False
                else:
                    model_won = (winner == mpid)
                    outcome = "Win" if model_won else "Loss"
                
                # Train on this game's trajectories
                if train and winner >= 0 and self.trajectories[i]:
                    train_stats = self.trainer.train_on_game(
                        self.trajectories[i],
                        winner,
                        mpid
                    )
                    self.trainer.total_games += 1
                
                # Stats
                self.total_games += 1
                if model_won:
                    self.total_model_wins += 1
                self.recent_wins.append(1 if model_won else 0)
                if len(self.recent_wins) > 100:
                    self.recent_wins = self.recent_wins[-100:]

                # Capture identities before reset
                identities = self.game_compositions[i]['player_types']

                # Reset game
                self.env.reset_game(i)
                self.game_compositions[i] = self._random_composition()
                self.consecutive_sixes[i] = 0
                self.trajectories[i] = {}
                
                results.append({
                    'winner': winner,
                    'model_won': model_won,
                    'model_player': mpid,
                    'identities': identities,
                    'total_moves': int(self.move_counts[i]),
                    'game_duration': 0.0
                })
                self.move_counts[i] = 0
        
        return results
    
    def get_recent_win_rate(self):
        """Win rate over last 100 games."""
        if not self.recent_wins:
            return 0.0
        return sum(self.recent_wins) / len(self.recent_wins)

    def get_spectator_state(self, game_idx=0):
        """
        Get the full state of a specific game for visualization.
        Returns dict with board, positions, scores, etc.
        """
        if game_idx < 0 or game_idx >= self.batch_size:
            return None
        
        game = self.env.get_game(game_idx)
        
        return {
            'positions': game.player_positions.tolist(),
            'scores': game.scores.tolist(),
            'current_player': game.current_player,
            'dice_roll': game.current_dice_roll,
            'is_terminal': game.is_terminal,
            'identities': self.game_compositions[game_idx]['player_types'],
            'active_players': game.active_players.tolist(),
            'move_count': int(self.move_counts[game_idx])
        }

    def _random_composition(self):
        """Generate a random game composition based on config probabilities."""
        probs = GAME_COMPOSITION
        r = random.random()
        cumulative = 0.0
        game_type = 'SelfPlay'
        for gtype, prob in probs.items():
            cumulative += prob
            if r < cumulative:
                game_type = gtype
                break
        
        if NUM_ACTIVE_PLAYERS == 2:
            seats = [0, 2]
            model_player = random.choice(seats)
            opponent_seat = 2 if model_player == 0 else 0
            player_types = ['Inactive'] * 4
            player_types[model_player] = 'Model'
            if game_type == 'SelfPlay':
                player_types[opponent_seat] = 'SelfPlay'
            elif game_type == 'Random':
                player_types[opponent_seat] = 'Random'
            else:
                player_types[opponent_seat] = game_type
            return {'model_player': model_player, 'player_types': player_types}
            
        model_player = random.randint(0, 3)
        player_types = ['Model'] * 4
        if game_type != 'SelfPlay':
            bot_seats = [i for i in range(4) if i != model_player]
            if game_type == 'Random':
                for seat in bot_seats:
                    player_types[seat] = 'Random'
            else:
                primary_seat = random.choice(bot_seats)
                player_types[primary_seat] = game_type
                remaining_seats = [s for s in bot_seats if s != primary_seat]
                bot_options = list(BOT_CLASSES.keys()) + ['Random']
                for seat in remaining_seats:
                    player_types[seat] = random.choice(bot_options)
        return {'model_player': model_player, 'player_types': player_types}
