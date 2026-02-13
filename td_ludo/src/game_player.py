"""
TD-Ludo Game Player — Value-Based Move Selection

Replaces the MCTS-based VectorLeagueWorker with a simple value-based approach:
1. For each legal move, simulate it → get V(s')
2. Pick the move with the highest V(s') (with ε-greedy exploration)
3. After each move, compute TD update with shaped reward

This is the core "TD-Gammon" approach adapted for 4-player Ludo.
"""

import random
import time
import numpy as np
import torch
import td_ludo_cpp as ludo_cpp

from src.tensor_utils import state_to_tensor_mastery
from src.reward_shaping import compute_shaped_reward, get_terminal_reward
from src.heuristic_bot import HeuristicLudoBot, AggressiveBot, DefensiveBot, RacingBot, RandomBot
from src.config import (
    GAME_COMPOSITION, MAX_MOVES_PER_GAME, EPSILON_START,
    EPSILON_END, EPSILON_DECAY_GAMES, TD_GAMMA, REWARD_SHAPING,
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


class TDGamePlayer:
    """
    Plays Ludo games using value-based move selection.
    
    Architecture:
    - Model predicts V(s) for any board state
    - Move selection: for each legal move, evaluate V(next_state), pick highest
    - Exploration: ε-greedy (dice provides additional implicit exploration)
    - Training: TD(0) update after each model move
    """
    
    def __init__(self, trainer, device):
        """
        Args:
            trainer: TDTrainer instance (has model + td_update)
            device: torch device
        """
        self.trainer = trainer
        self.device = device
        
        # Instantiate bots
        self.bots = {name: cls() for name, cls in BOT_CLASSES.items()}
        
        # Stats
        self.total_games = 0
        self.total_model_wins = 0
        self.recent_wins = []  # Last 100 games
        self.recent_td_errors = []  # Last 1000 TD errors
    
    def get_epsilon(self, games_played):
        """Get current epsilon with linear decay."""
        progress = min(1.0, games_played / EPSILON_DECAY_GAMES)
        return EPSILON_START + (EPSILON_END - EPSILON_START) * progress
    
    def select_model_move(self, state, legal_moves, epsilon):
        """
        Select a move using value-based 1-ply lookahead + ε-greedy.
        
        Args:
            state: Current game state (with dice roll)
            legal_moves: List of legal token indices
            epsilon: Exploration rate
            
        Returns:
            (chosen_move, was_exploration)
        """
        if len(legal_moves) == 1:
            return legal_moves[0], False
        
        # ε-greedy exploration
        if random.random() < epsilon:
            return random.choice(legal_moves), True
        
        # Evaluate each possible move
        best_value = -float('inf')
        best_move = legal_moves[0]
        
        # Batch evaluation for efficiency
        next_states = []
        for move in legal_moves:
            next_state = ludo_cpp.apply_move(state, move)
            next_states.append(state_to_tensor_mastery(next_state))
        
        values = self.trainer.predict_value_batch(next_states)
        
        current_player = state.current_player
        for i, move in enumerate(legal_moves):
            # Value is from the perspective of current_player
            v = values[i]
            if v > best_value:
                best_value = v
                best_move = move
        
        return best_move, False
    
    def play_game(self, game_composition=None, epsilon=0.1, train=True):
        """
        Play a single game and optionally train online.
        
        Args:
            game_composition: Dict mapping player slots to 'Model' or bot type
            epsilon: Exploration rate for model moves
            train: Whether to apply TD updates
            
        Returns:
            dict with game results:
                winner: int (0-3)
                model_won: bool
                model_player: int
                total_moves: int
                avg_td_error: float
                game_duration: float
        """
        start_time = time.time()
        if NUM_ACTIVE_PLAYERS == 2:
            state = ludo_cpp.create_initial_state_2p()
        else:
            state = ludo_cpp.create_initial_state()
        
        # Assign players
        
        # Assign players
        if game_composition is None:
            game_composition = self._random_composition()
        
        model_player = game_composition['model_player']
        player_types = game_composition['player_types']  # [type_p0, type_p1, type_p2, type_p3]
        
        # Track for TD updates
        consecutive_sixes = [0, 0, 0, 0]
        move_count = 0
        td_errors = []
        
        prev_model_state_tensor = None
        prev_model_state = None  # For reward shaping
        
        while not state.is_terminal and move_count < MAX_MOVES_PER_GAME:
            # Roll dice
            if state.current_dice_roll == 0:
                state.current_dice_roll = random.randint(1, 6)
                
                # Track consecutive sixes
                cp = state.current_player
                if state.current_dice_roll == 6:
                    consecutive_sixes[cp] += 1
                else:
                    consecutive_sixes[cp] = 0
                
                # Penalty for 3 consecutive sixes
                if consecutive_sixes[cp] >= 3:
                    state.current_player = (state.current_player + 1) % 4
                    state.current_dice_roll = 0
                    consecutive_sixes[cp] = 0
                    continue
            
            # Get legal moves
            legal_moves = ludo_cpp.get_legal_moves(state)
            
            if len(legal_moves) == 0:
                # No legal moves — pass turn
                state.current_player = (state.current_player + 1) % 4
                state.current_dice_roll = 0
                continue
            
            current_player = state.current_player
            player_type = player_types[current_player]
            
            if player_type == 'Model':
                # Model's turn — value-based selection
                current_state_tensor = state_to_tensor_mastery(state)
                
                # TD Update from previous model move to this one
                if train and prev_model_state_tensor is not None:
                    if REWARD_SHAPING:
                        reward = compute_shaped_reward(
                            prev_model_state, state, model_player, 0.0, TD_GAMMA
                        )
                    else:
                        reward = 0.0
                    
                    td_err = self.trainer.td_update(
                        prev_model_state_tensor, current_state_tensor,
                        reward, done=False
                    )
                    td_errors.append(td_err)
                
                # Select move
                if len(legal_moves) == 1:
                    action = legal_moves[0]
                else:
                    action, _ = self.select_model_move(state, legal_moves, epsilon)
                
                # Store for next TD update
                prev_model_state_tensor = current_state_tensor
                prev_model_state = ludo_cpp.GameState()
                prev_model_state.current_player = state.current_player
                prev_model_state.current_dice_roll = state.current_dice_roll
                prev_model_state.is_terminal = state.is_terminal
                prev_model_state.player_positions[:] = state.player_positions[:]
                prev_model_state.scores[:] = state.scores[:]
                prev_model_state.board[:] = state.board[:]
                
            elif player_type == 'Random':
                bot = self.bots.get('Random')
                if bot:
                    action = bot.select_move(state, legal_moves)
                else:
                    action = random.choice(legal_moves)
            else:
                # Heuristic bot
                bot = self.bots.get(player_type)
                if bot:
                    action = bot.select_move(state, legal_moves)
                else:
                    action = random.choice(legal_moves)
            
            # Apply move
            state = ludo_cpp.apply_move(state, action)
            move_count += 1
        
        # Game over — final TD update
        winner = ludo_cpp.get_winner(state) if state.is_terminal else -1
        
        if train and prev_model_state_tensor is not None:
            terminal_state_tensor = state_to_tensor_mastery(state)
            
            if REWARD_SHAPING:
                raw_reward = get_terminal_reward(state, model_player)
                final_reward = compute_shaped_reward(
                    prev_model_state, state, model_player, raw_reward, TD_GAMMA
                )
            else:
                final_reward = get_terminal_reward(state, model_player)
            
            td_err = self.trainer.td_update(
                prev_model_state_tensor, terminal_state_tensor,
                final_reward, done=True
            )
            td_errors.append(td_err)
            
            # Flush any remaining gradients
            self.trainer.flush_gradients()
        
        # Track results
        model_won = (winner == model_player)
        self.total_games += 1
        if model_won:
            self.total_model_wins += 1
        
        self.recent_wins.append(1 if model_won else 0)
        if len(self.recent_wins) > 100:
            self.recent_wins = self.recent_wins[-100:]
        
        avg_td_error = np.mean(td_errors) if td_errors else 0.0
        self.recent_td_errors.extend(td_errors)
        if len(self.recent_td_errors) > 1000:
            self.recent_td_errors = self.recent_td_errors[-1000:]
        
        duration = time.time() - start_time
        
        # Build identities for Elo tracking ('Model' → 'Main')
        identities = ['Main' if pt == 'Model' else pt for pt in player_types]
        
        return {
            'winner': winner,
            'model_won': model_won,
            'model_player': model_player,
            'total_moves': move_count,
            'avg_td_error': avg_td_error,
            'game_duration': duration,
            'player_types': player_types,
            'identities': identities,
        }
    
    def _random_composition(self):
        """
        Generate a random game composition based on config probabilities.
        
        Returns dict with:
            model_player: int (0-3)
            player_types: list of 4 strings
        """
        probs = GAME_COMPOSITION
        
        # Determine game type
        r = random.random()
        cumulative = 0.0
        game_type = 'SelfPlay'
        
        for gtype, prob in probs.items():
            cumulative += prob
            if r < cumulative:
                game_type = gtype
                break
        
        # 2-Player Mode Logic (P0 vs P2)
        if NUM_ACTIVE_PLAYERS == 2:
            # P0 is Model, P2 is Opponent (or vice-versa)
            # Actually, let's keep it simple: Model is always P0 or P2
            # But wait, create_initial_state_2p activates P0 and P2.
            # So valid seats are 0 and 2.
            
            seats = [0, 2]
            model_player = random.choice(seats)
            opponent_seat = 2 if model_player == 0 else 0
            
            player_types = ['Inactive'] * 4
            player_types[model_player] = 'Model'
            
            # Opponent type
            if game_type == 'SelfPlay':
                player_types[opponent_seat] = 'SelfPlay'
            elif game_type == 'Random':
                player_types[opponent_seat] = 'Random'
            else:
                # Specific bot type
                player_types[opponent_seat] = game_type
                
            return {
                'model_player': model_player,
                'player_types': player_types,
            }

        # 4-Player Mode Logic (Standard)
        # Model always gets a random seat
        model_player = random.randint(0, 3)
        player_types = ['Model'] * 4
        
        if game_type != 'SelfPlay':
            # Place one bot of chosen type in a random non-model seat
            bot_seats = [i for i in range(4) if i != model_player]
            
            if game_type == 'Random':
                for seat in bot_seats:
                    player_types[seat] = 'Random'
            else:
                # Mix: chosen bot + random bots for remaining seats
                # Primary bot in one seat
                primary_seat = random.choice(bot_seats)
                player_types[primary_seat] = game_type
                
                # Other seats get random bot types (variety)
                remaining_seats = [s for s in bot_seats if s != primary_seat]
                bot_options = list(BOT_CLASSES.keys()) + ['Random']
                for seat in remaining_seats:
                    player_types[seat] = random.choice(bot_options)
        
        return {
            'model_player': model_player,
            'player_types': player_types,
        }
    
    def get_recent_win_rate(self):
        """Win rate over last 100 games."""
        if not self.recent_wins:
            return 0.0
        return sum(self.recent_wins) / len(self.recent_wins)
    
    def get_recent_td_error(self):
        """Average TD error over last 1000 updates."""
        if not self.recent_td_errors:
            return 0.0
        return np.mean(self.recent_td_errors)

class VectorTDGamePlayer:
    """
    Plays N parallel games using C++ VectorGameState and batched model inference.
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
        
        # Pending TD updates: (prev_tensor, prev_shaping_state, model_player_id)
        self.pending_updates = [None] * batch_size
        
        # Stats
        self.total_games = 0
        self.total_model_wins = 0
        self.recent_wins = []
        self.recent_td_errors = []
        
        # Track consecutive sixes for all games (4 players per game)
        # Shape: (batch_size, 4)
        self.consecutive_sixes = np.zeros((batch_size, 4), dtype=int)
        
        # Initialize bots (shared instances are fine for stateless bots)
        self.bots = {name: cls() for name, cls in BOT_CLASSES.items()}

    def play_step(self, epsilon=0.1, train=True):
        """
        Advance all games by one step.
        Returns list of finished game results (dicts).
        """
        # 1. Get current states (s_t) for decision making
        # shape (B, 21, 15, 15)
        current_states_np = self.env.get_state_tensor()
        
        actions = []
        current_players = []
        model_indices = []
        
        # 2. Determine actions for all games
        for i in range(self.batch_size):
            game = self.env.get_game(i)
            
            if game.is_terminal:
                actions.append(-1)
                current_players.append(-1)
                continue
                
            cp = game.current_player
            current_players.append(cp)
            
            # Dice Roll Logic
            if game.current_dice_roll == 0:
                roll = random.randint(1, 6)
                game.current_dice_roll = roll
                if roll == 6:
                    self.consecutive_sixes[i, cp] += 1
                else:
                    self.consecutive_sixes[i, cp] = 0
                    
                if self.consecutive_sixes[i, cp] >= 3:
                    game.current_player = (cp + 1) % 4
                    game.current_dice_roll = 0
                    self.consecutive_sixes[i, cp] = 0
                    actions.append(-1)
                    continue
            
            legal_moves = ludo_cpp.get_legal_moves(game)
            if not legal_moves:
                game.current_player = (cp + 1) % 4
                game.current_dice_roll = 0
                actions.append(-1)
                continue
            
            ptype = self.game_compositions[i]['player_types'][cp]
            if ptype == 'Model':
                model_indices.append(i)
                actions.append(-2) # Marker
            else:
                bot = self.bots.get(ptype, self.bots['Random'])
                action = bot.select_move(game, legal_moves)
                actions.append(action)

        # 3. Model Action Selection (Batch)
        all_hooks = [] # (game_idx, move, next_state_tensor)
        
        if model_indices:
            # Pre-gather all candidate next states
            for idx in model_indices:
                game = self.env.get_game(idx)
                lmoves = ludo_cpp.get_legal_moves(game)
                
                # ε-greedy
                if random.random() < epsilon:
                    actions[idx] = random.choice(lmoves)
                    continue
                
                if len(lmoves) == 1:
                    actions[idx] = lmoves[0]
                    continue
                
                # Prepare candidates for value estimation
                for m in lmoves:
                    # Apply move on copy (fast enough in C++)
                    next_g = ludo_cpp.apply_move(game, m)
                    tsr = state_to_tensor_mastery(next_g)
                    all_hooks.append((idx, m, tsr))
            
            if all_hooks:
                # Batch predict V(s') for candidates
                batch_tsrs = np.stack([h[2] for h in all_hooks]) 
                vals = self.trainer.predict_value_batch(torch.from_numpy(batch_tsrs).to(self.device))
                
                best_scores = {idx: -float('inf') for idx in model_indices}
                chosen_moves = {}
                
                ptr = 0
                for idx, m, _ in all_hooks:
                    v = vals[ptr]
                    ptr += 1
                    if v > best_scores.get(idx, -float('inf')):
                        best_scores[idx] = v
                        chosen_moves[idx] = m
                
                for idx in chosen_moves:
                    actions[idx] = chosen_moves[idx]

        # 4. Snapshot state for reward shaping (before step)
        prev_states_data = {}
        if train and REWARD_SHAPING and model_indices:
            for idx in model_indices:
                if actions[idx] >= 0: # If we actually moved
                    g = self.env.get_game(idx)
                    prev_states_data[idx] = (
                        np.array(g.scores, copy=True),
                        np.array(g.player_positions, copy=True),
                        np.array(g.board, copy=True),
                        g.current_player,
                        g.is_terminal
                    )

        # 5. Step Environment
        final_actions = [a if a >= 0 else -1 for a in actions]
        next_states_np, rewards_np, dones_np, info_list = self.env.step(final_actions)
        
        # 6. TD Updates
        model_updates = []
        if train:
            for idx in model_indices:
                if actions[idx] >= 0:
                    # Calculate Reward
                    raw_reward = rewards_np[idx]
                    reward = raw_reward
                    
                    if REWARD_SHAPING and idx in prev_states_data:
                        # Reconstruct prev state proxy
                        prev_s = ludo_cpp.GameState()
                        ps_scores, ps_pos, ps_board, ps_cp, ps_term = prev_states_data[idx]
                        prev_s.scores = ps_scores
                        prev_s.player_positions = ps_pos
                        prev_s.board = ps_board
                        prev_s.current_player = ps_cp
                        prev_s.is_terminal = ps_term
                        
                        curr_s = self.env.get_game(idx)
                        mpid = self.game_compositions[idx]['model_player']
                        reward = compute_shaped_reward(prev_s, curr_s, mpid, raw_reward, TD_GAMMA)
                    
                    # Store (s_t, s_{t+1}, r, d)
                    model_updates.append((
                        current_states_np[idx],
                        next_states_np[idx],
                        reward,
                        dones_np[idx]
                    ))
            
            # Perform Batch Update
            if model_updates:
                s_batch = np.stack([u[0] for u in model_updates])
                ns_batch = np.stack([u[1] for u in model_updates])
                r_batch = np.array([u[2] for u in model_updates], dtype=np.float32)
                d_batch = np.array([u[3] for u in model_updates], dtype=np.float32)
                
                td_err = self.trainer.td_update_batch(
                    torch.from_numpy(s_batch),
                    torch.from_numpy(ns_batch),
                    torch.from_numpy(r_batch),
                    torch.from_numpy(d_batch)
                )
                self.recent_td_errors.append(td_err)
                if len(self.recent_td_errors) > 1000:
                    self.recent_td_errors = self.recent_td_errors[-1000:]

        # 7. Handle resets and logging
        results = []
        for i in range(self.batch_size):
            if dones_np[i]:
                winner = info_list[i]['winner']
                mpid = self.game_compositions[i]['model_player']
                model_won = (winner == mpid)
                
                # Stats
                self.total_games += 1
                if model_won: self.total_model_wins += 1
                self.recent_wins.append(1 if model_won else 0)
                if len(self.recent_wins) > 100: self.recent_wins = self.recent_wins[-100:]

                # Capture identities before reset
                identities = self.game_compositions[i]['player_types']

                # Reset
                self.env.reset_game(i)
                self.game_compositions[i] = self._random_composition()
                self.consecutive_sixes[i] = 0
                
                results.append({
                    'winner': winner,
                    'model_won': model_won,
                    'model_player': mpid,
                    'identities': identities,
                    'total_moves': 0, # Not tracked per game easily
                    'avg_td_error': self.get_recent_td_error(),
                    'game_duration': 0.0
                })
        
        return results

    def get_epsilon(self, total_games):
        """Calculate current epsilon for exploration."""
        if total_games >= EPSILON_DECAY_GAMES:
            return EPSILON_END
        
        # Linear decay
        progress = total_games / EPSILON_DECAY_GAMES
        return EPSILON_START - progress * (EPSILON_START - EPSILON_END)

    def get_recent_td_error(self):
        if not self.recent_td_errors:
            return 0.0
        return np.mean(self.recent_td_errors)
        
    def get_spectator_state(self, game_idx=0):
        """
        Get the full state of a specific game for visualization.
        Returns dict with board, positions, scores, etc.
        """
        if game_idx < 0 or game_idx >= self.batch_size:
            return None
        
        game = self.env.get_game(game_idx)
        
        # Convert C++ binding objects to Python list/dict
        # Board is 15x15 (but visually fixed, we actually just need token positions for Ludo)
        # Actually, board array contains stack heights if we modeled it that way?
        # The C++ board logic is: 0=empty, or player_id+1? 
        # Actually bindings.cpp shows board is exposed.
        
        return {
            'positions': game.player_positions.tolist(), # 4x4
            'scores': game.scores.tolist(), # 4
            'current_player': game.current_player,
            'dice_roll': game.current_dice_roll,
            'is_terminal': game.is_terminal,
            'identities': self.game_compositions[game_idx]['player_types']
        }

    def _random_composition(self):
        # Same as TDGamePlayer._random_composition
        # Can copy-paste logic or reuse static method if refactored.
        # Copy-pasting for safety now.
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
            if game_type == 'SelfPlay': player_types[opponent_seat] = 'SelfPlay'
            elif game_type == 'Random': player_types[opponent_seat] = 'Random'
            else: player_types[opponent_seat] = game_type
            return {'model_player': model_player, 'player_types': player_types}
            
        model_player = random.randint(0, 3)
        player_types = ['Model'] * 4
        if game_type != 'SelfPlay':
            bot_seats = [i for i in range(4) if i != model_player]
            if game_type == 'Random':
                for seat in bot_seats: player_types[seat] = 'Random'
            else:
                primary_seat = random.choice(bot_seats)
                player_types[primary_seat] = game_type
                remaining_seats = [s for s in bot_seats if s != primary_seat]
                bot_options = list(BOT_CLASSES.keys()) + ['Random']
                for seat in remaining_seats: player_types[seat] = random.choice(bot_options)
        return {'model_player': model_player, 'player_types': player_types}
