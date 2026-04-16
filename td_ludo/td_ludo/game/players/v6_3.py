"""
TD-Ludo V6.3 Game Player — Bonus-Turn Awareness + Capture Tracking

Based on V6.1 game player with these additions:
- Uses encode_state_v6_3 (27 channels) instead of encode_state_v6 (24 channels)
- Tracks captures during gameplay for auxiliary prediction target
- Computes per-step aux target: "did this player capture within next 5 own turns?"
- Passes consecutive_sixes to the encoder (bonus-turn context)
"""

import os
import random
import time
import numpy as np
import torch
import torch.nn.functional as F
import td_ludo_cpp as ludo_cpp

from src.heuristic_bot import (
    HeuristicLudoBot, AggressiveBot, DefensiveBot, RacingBot, RandomBot,
    ExpertBot,
)
from src.reward_shaping import compute_shaped_reward
from src.config import (
    GAME_COMPOSITION, MAX_MOVES_PER_GAME,
    TEMPERATURE_START, TEMPERATURE_END, TEMPERATURE_DECAY_GAMES,
    NUM_ACTIVE_PLAYERS, GHOSTS_DIR,
    SELFPLAY_GHOST_FRACTION, SELFPLAY_GHOST_STRATEGY,
)


# Bot registry
BOT_CLASSES = {
    'Heuristic': HeuristicLudoBot,
    'Aggressive': AggressiveBot,
    'Defensive': DefensiveBot,
    'Racing': RacingBot,
    'Random': RandomBot,
    'Expert': ExpertBot,
}


class VectorACGamePlayer:
    """
    Plays N parallel games using C++ VectorGameState and batched policy inference.
    
    Key differences from VectorTDGamePlayer:
    - Move selection via π(a|s) sampling (not V(s') argmax)
    - Collects trajectories for ALL players (model + bots)
    - Training happens at end of each game (not per-step)
    """
    def __init__(self, trainer, batch_size, device, model_factory=None, elo_tracker=None):
        self.trainer = trainer
        self.batch_size = batch_size
        self.device = device
        self.model_factory = model_factory
        self.elo_tracker = elo_tracker
        
        # Initialize Vector Env
        two_player = (NUM_ACTIVE_PLAYERS == 2)
        self.env = ludo_cpp.VectorGameState(batch_size, two_player)

        self.ghost_cache = {}
        self.max_cached_ghosts = 4
        self.active_ghost = None
        self.active_ghost_selected_at = -1
        self.ghost_refresh_games = 1000
        
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

        # V6.3: Capture tracking for auxiliary prediction target
        # capture_log[i] = list of (player_id, own_turn_number) for each capture in game i
        self.capture_log = [[] for _ in range(batch_size)]
        # own_turn_counts[i][p] = how many turns player p has taken in game i
        self.own_turn_counts = np.zeros((batch_size, 4), dtype=int)

        # Initialize bots (shared instances are fine for stateless bots)
        self.bots = {name: cls() for name, cls in BOT_CLASSES.items()}

    def get_temperature(self, total_games):
        """Calculate current temperature for policy sampling."""
        if total_games >= TEMPERATURE_DECAY_GAMES:
            return TEMPERATURE_END
        progress = total_games / TEMPERATURE_DECAY_GAMES
        return TEMPERATURE_START - progress * (TEMPERATURE_START - TEMPERATURE_END)

    def _get_ghost_paths(self):
        """Return available ghost snapshot paths sorted newest-first."""
        if not os.path.exists(GHOSTS_DIR):
            return []
        ghosts = [
            os.path.join(GHOSTS_DIR, fname)
            for fname in os.listdir(GHOSTS_DIR)
            if fname.startswith("ghost_") and fname.endswith(".pt")
        ]
        return sorted(ghosts, reverse=True)

    def _load_ghost_model(self, ghost_path):
        """Load a ghost snapshot into an eval-only model, keeping a small cache."""
        if ghost_path in self.ghost_cache:
            model = self.ghost_cache.pop(ghost_path)
            self.ghost_cache[ghost_path] = model
            return model

        if self.model_factory is None:
            return None

        model = self.model_factory().to(self.device)
        checkpoint = torch.load(ghost_path, map_location=self.device, weights_only=False)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict)
        model.eval()

        for param in model.parameters():
            param.requires_grad_(False)

        self.ghost_cache[ghost_path] = model
        while len(self.ghost_cache) > self.max_cached_ghosts:
            old_path, old_model = next(iter(self.ghost_cache.items()))
            del self.ghost_cache[old_path]
            del old_model

        return model

    def _get_active_ghost(self):
        """Keep one active ghost opponent for a while to avoid model-loading churn."""
        ghost_pool = self._get_ghost_paths()
        if not ghost_pool:
            self.active_ghost = None
            return None

        needs_refresh = (
            self.active_ghost is None
            or self.active_ghost['path'] not in ghost_pool
            or (self.trainer.total_games - self.active_ghost_selected_at) >= self.ghost_refresh_games
        )
        if not needs_refresh:
            return self.active_ghost

        if self.elo_tracker is not None:
            ghost_path = self.elo_tracker.select_ghost(
                ghost_pool,
                main_name="Model",
                strategy=SELFPLAY_GHOST_STRATEGY,
            )
        else:
            ghost_path = random.choice(ghost_pool)

        if not ghost_path:
            self.active_ghost = None
            return None

        ghost_name = os.path.basename(ghost_path).replace(".pt", "")
        self.active_ghost = {"path": ghost_path, "name": ghost_name}
        self.active_ghost_selected_at = self.trainer.total_games
        return self.active_ghost

    def _pick_selfplay_opponent(self):
        """Choose whether self-play should face the live model or the active ghost."""
        if random.random() >= SELFPLAY_GHOST_FRACTION:
            return None
        return self._get_active_ghost()

    def play_step(self, train=True):
        """
        Advance all games by one step.
        Returns list of finished game results (dicts).
        """
        # 1. Get current states for decision making
        current_states_np = self.env.get_state_tensor()
        
        actions = []
        current_players = []
        decision_groups = {}  # (controller_type, controller_id) -> [game_indices]
        
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
            
            composition = self.game_compositions[i]
            controller = composition['controllers'][cp]
            ptype = composition['player_types'][cp]
            
            if controller in ('Model', 'SelfPlay', 'Ghost'):
                if controller == 'Ghost':
                    group_key = ('Ghost', composition['ghost_paths'][cp])
                else:
                    group_key = ('Main', None)
                decision_groups.setdefault(group_key, []).append(i)
                actions.append(-2)  # Placeholder marker
            else:
                # Bot makes the decision — no trajectory needed
                bot = self.bots.get(ptype, self.bots['Random'])
                action = bot.select_move(game, legal_moves)
                actions.append(action)

        # 3. Batched Model Action Selection via Policy Head
        if decision_groups:
            main_temperature = self.get_temperature(self.trainer.total_games)

            for (controller, controller_id), indices in decision_groups.items():
                batch_states = []
                batch_legal_masks = []
                batch_legal_moves = []

                for idx in indices:
                    game = self.env.get_game(idx)
                    lmoves = ludo_cpp.get_legal_moves(game)
                    batch_legal_moves.append(lmoves)

                    state_tensor = ludo_cpp.encode_state_v6_3(
                        game, int(self.consecutive_sixes[idx, cp])
                    )
                    batch_states.append(state_tensor)

                    legal_mask = np.zeros(4, dtype=np.float32)
                    for move in lmoves:
                        legal_mask[move] = 1.0
                    batch_legal_masks.append(legal_mask)

                states_t = torch.from_numpy(np.stack(batch_states)).to(self.device, dtype=torch.float32)
                masks_t = torch.from_numpy(np.stack(batch_legal_masks)).to(self.device, dtype=torch.float32)

                if controller == 'Ghost':
                    model = self._load_ghost_model(controller_id)
                    sample_temperature = 1.0
                else:
                    model = self.trainer.model
                    sample_temperature = main_temperature

                if model is None:
                    for j, idx in enumerate(indices):
                        lmoves = batch_legal_moves[j]
                        action = random.choice(lmoves)
                        actions[idx] = action
                    continue

                with torch.no_grad():
                    policy_logits = model.forward_policy_only(states_t, masks_t)
                    probs_base = F.softmax(policy_logits, dim=1)

                    if sample_temperature != 1.0:
                        sample_logits = policy_logits / sample_temperature
                        probs_sample = F.softmax(sample_logits, dim=1)
                    else:
                        probs_sample = probs_base

                    sampled_actions = torch.multinomial(probs_sample, num_samples=1).squeeze(1)
                    old_log_probs = torch.log(
                        probs_sample.gather(1, sampled_actions.unsqueeze(1)).squeeze(1) + 1e-8
                    )

                sampled_np = sampled_actions.cpu().numpy()
                old_lp_np = old_log_probs.cpu().numpy()

                for j, idx in enumerate(indices):
                    action = int(sampled_np[j])
                    lmoves = batch_legal_moves[j]

                    if action not in lmoves:
                        action = random.choice(lmoves)

                    actions[idx] = action

                    cp = current_players[idx]
                    if not train:
                        continue
                    if cp not in self.game_compositions[idx]['train_players']:
                        continue

                    if cp not in self.trajectories[idx]:
                        self.trajectories[idx][cp] = []
                    self.trajectories[idx][cp].append({
                        'state': batch_states[j],
                        'action': action,
                        'legal_mask': batch_legal_masks[j],
                        'old_log_prob': float(old_lp_np[j]),
                        'temperature': float(sample_temperature),
                        'own_turn_number': int(self.own_turn_counts[idx, cp]),
                    })
        
        # Save pre-step states for reward computation
        pre_step_states = []
        pre_step_scores = []
        pre_step_active = []
        for i in range(self.batch_size):
            # Must copy the state or extract needed info if GameState mutates.
            # Fortunately get_game returns a reference the C++ object, so we need to copy?
            # Actually, `compute_shaped_reward` only needs `player_positions`.
            # Let's extract original positions for all 4 players for all batch games.
            game = self.env.get_game(i)
            # Create a simple struct/dict to hold the old positions
            old_pos = {p: list(game.player_positions[p]) for p in range(4)}
            pre_step_states.append(old_pos)
            pre_step_scores.append(list(game.scores))
            pre_step_active.append(list(game.active_players))
            
        # 4. Step Environment
        final_actions = [a if a >= 0 else -1 for a in actions]
        for i, a in enumerate(final_actions):
            if a >= 0:
                self.move_counts[i] += 1
                # V6.3: track per-player turn counts
                cp = current_players[i]
                if cp >= 0:
                    self.own_turn_counts[i, cp] += 1

        next_states_np, rewards_np, dones_np, info_list = self.env.step(final_actions)

        # V6.3: Detect captures by comparing pre/post opponent positions.
        # A capture occurred if any opponent token moved from non-base to base (-1).
        for i in range(self.batch_size):
            cp = current_players[i]
            if cp < 0 or final_actions[i] < 0:
                continue
            next_game = self.env.get_game(i)
            for op in range(4):
                if op == cp:
                    continue
                for t in range(4):
                    old_pos = pre_step_states[i][op][t]
                    new_pos = int(next_game.player_positions[op][t])
                    if old_pos >= 0 and old_pos != 99 and new_pos == -1:
                        # Capture! Record who did it and on which of their own turns
                        self.capture_log[i].append(
                            (cp, int(self.own_turn_counts[i, cp]))
                        )
        
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
                    def __init__(self, pos, scores=None, active=None):
                        self.player_positions = pos
                        self.scores = scores or [0,0,0,0]
                        self.active_players = active or [True,False,True,False]
                
                dummy_old = DummyState(pre_step_states[i], pre_step_scores[i], pre_step_active[i])
                dummy_new = DummyState(next_game.player_positions, list(next_game.scores), list(next_game.active_players))
                
                step_reward = compute_shaped_reward(dummy_old, dummy_new, cp)
                
                # Append to the LAST trajectory entry for this player
                last_idx = len(self.trajectories[i][cp]) - 1
                if last_idx >= 0:
                    step = self.trajectories[i][cp][last_idx]
                    step['step_reward'] = step_reward
            if dones_np[i]:
                winner = info_list[i]['winner']
                composition = self.game_compositions[i]
                mpid = composition['model_player']
                
                if winner == -1:
                    outcome = "Timeout"
                    model_won = False
                else:
                    model_won = (winner == mpid)
                    outcome = "Win" if model_won else "Loss"
                
                # V6.3: Compute auxiliary capture targets retroactively.
                # For each trajectory step, label "did this player capture
                # within the next 5 of their OWN turns?"
                if train and winner >= 0:
                    for train_player in composition['train_players']:
                        if train_player not in self.trajectories[i]:
                            continue
                        # Get all capture turn numbers for this player
                        capture_turns = [
                            turn_num
                            for (pid, turn_num) in self.capture_log[i]
                            if pid == train_player
                        ]
                        for step in self.trajectories[i][train_player]:
                            step_turn = step.get('own_turn_number', 0)
                            captured_within_5 = any(
                                ct > step_turn and ct <= step_turn + 5
                                for ct in capture_turns
                            )
                            step['aux_capture_target'] = 1.0 if captured_within_5 else 0.0

                # Train on this game's trajectories
                if train and winner >= 0:
                    for train_player in composition['train_players']:
                        if train_player in self.trajectories[i]:
                            self.trainer.train_on_game(
                                self.trajectories[i],
                                winner,
                                train_player
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
                identities = composition['player_types']

                # Reset game
                self.env.reset_game(i)
                self.game_compositions[i] = self._random_composition()
                self.consecutive_sixes[i] = 0
                self.trajectories[i] = {}
                # V6.3: clear capture tracking for this game slot
                self.capture_log[i] = []
                self.own_turn_counts[i] = 0
                
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
            controllers = ['Inactive'] * 4
            ghost_paths = [None] * 4
            train_players = {model_player}
            player_types[model_player] = 'Model'
            controllers[model_player] = 'Model'
            if game_type == 'SelfPlay':
                ghost_info = self._pick_selfplay_opponent()
                if ghost_info is not None:
                    player_types[opponent_seat] = ghost_info['name']
                    controllers[opponent_seat] = 'Ghost'
                    ghost_paths[opponent_seat] = ghost_info['path']
                else:
                    player_types[opponent_seat] = 'SelfPlay'
                    controllers[opponent_seat] = 'SelfPlay'
                    train_players.add(opponent_seat)
            elif game_type == 'Random':
                player_types[opponent_seat] = 'Random'
                controllers[opponent_seat] = 'Random'
            else:
                player_types[opponent_seat] = game_type
                controllers[opponent_seat] = game_type
            return {
                'model_player': model_player,
                'player_types': player_types,
                'controllers': controllers,
                'ghost_paths': ghost_paths,
                'train_players': sorted(train_players),
            }
            
        model_player = random.randint(0, 3)
        player_types = ['Model'] * 4
        controllers = ['Model'] * 4
        ghost_paths = [None] * 4
        if game_type != 'SelfPlay':
            bot_seats = [i for i in range(4) if i != model_player]
            if game_type == 'Random':
                for seat in bot_seats:
                    player_types[seat] = 'Random'
                    controllers[seat] = 'Random'
            else:
                primary_seat = random.choice(bot_seats)
                player_types[primary_seat] = game_type
                controllers[primary_seat] = game_type
                remaining_seats = [s for s in bot_seats if s != primary_seat]
                bot_options = list(BOT_CLASSES.keys()) + ['Random']
                for seat in remaining_seats:
                    player_types[seat] = random.choice(bot_options)
                    controllers[seat] = player_types[seat]
        return {
            'model_player': model_player,
            'player_types': player_types,
            'controllers': controllers,
            'ghost_paths': ghost_paths,
            'train_players': [model_player],
        }
