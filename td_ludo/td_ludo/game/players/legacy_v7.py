"""
TD-Ludo V7 Game Player — Transformer with Context Window

Key differences from V6 VectorACGamePlayer:
1. Uses 1D state encoding (encode_state_1d) instead of C++ encode_state (17ch tensor)
2. Maintains a rolling context window of K=16 past turns per game
3. Passes (token_positions, continuous, actions, seq_mask) to model instead of single tensor
4. Trajectory stores 1D state data for the V7 trainer

The game environment, bots, reward shaping, ghost system, and
game composition logic are reused directly from V6.
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
from src.state_encoder_1d import encode_state_1d, make_empty_state_1d, NUM_ACTION_CLASSES
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


class TurnHistory:
    """
    Rolling context window for a single game slot.
    Stores the last K turns of (token_positions, continuous, action).
    """

    def __init__(self, context_length=16):
        self.K = context_length
        # Ring buffer of turns
        self._token_positions = []  # list of np.array(8,) int64
        self._continuous = []       # list of np.array(9,) float32
        self._actions = []          # list of int (0-4)

    def reset(self):
        self._token_positions.clear()
        self._continuous.clear()
        self._actions.clear()

    def add_turn(self, token_positions, continuous, action=4):
        """
        Record a turn. Call this BEFORE the model makes its decision.

        Args:
            token_positions: np.array(8,) int64
            continuous: np.array(9,) float32
            action: int — the action taken on the PREVIOUS turn (4=none/pass for first turn)
        """
        self._token_positions.append(token_positions)
        self._continuous.append(continuous)
        self._actions.append(action)

        # Trim to keep only last K
        if len(self._token_positions) > self.K:
            self._token_positions.pop(0)
            self._continuous.pop(0)
            self._actions.pop(0)

    def update_last_action(self, action):
        """Update the action for the most recently added turn."""
        if self._actions:
            self._actions[-1] = action

    def get_sequence(self):
        """
        Get the padded K-length sequence for model input.

        Returns:
            token_positions: np.array(K, 8) int64
            continuous: np.array(K, 9) float32
            actions: np.array(K,) int64
            seq_mask: np.array(K,) bool — True for padded positions
        """
        n_valid = len(self._token_positions)
        n_pad = self.K - n_valid

        # Pre-allocate
        tok = np.zeros((self.K, 8), dtype=np.int64)
        cont = np.zeros((self.K, 9), dtype=np.float32)
        acts = np.full(self.K, 4, dtype=np.int64)  # 4 = pass/none
        mask = np.ones(self.K, dtype=bool)  # True = padding

        # Fill valid turns (padding is at the start, valid turns at the end)
        if n_valid > 0:
            tok[n_pad:] = np.stack(self._token_positions)
            cont[n_pad:] = np.stack(self._continuous)
            acts[n_pad:] = np.array(self._actions, dtype=np.int64)
            mask[n_pad:] = False

        return tok, cont, acts, mask

    @property
    def num_turns(self):
        return len(self._token_positions)


class VectorV7GamePlayer:
    """
    V7 game player with transformer context window support.

    Same structure as VectorACGamePlayer but:
    - Encodes states as 1D vectors
    - Maintains TurnHistory per game slot
    - Passes sequences to model.forward() / model.forward_policy_only()
    """

    def __init__(self, trainer, batch_size, device, context_length=16,
                 model_factory=None, elo_tracker=None):
        self.trainer = trainer
        self.batch_size = batch_size
        self.device = device
        self.context_length = context_length
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

        # Trajectory storage: dict of player_id → list of step dicts
        self.trajectories = [{} for _ in range(batch_size)]

        # Turn history per game — tracks context window for each player
        # Key: (game_idx, player_id) → TurnHistory
        self.turn_histories = {}
        for i in range(batch_size):
            for p in range(4):
                self.turn_histories[(i, p)] = TurnHistory(context_length)

        # Track the last action each player took per game
        self.last_actions = np.full((batch_size, 4), 4, dtype=np.int64)  # 4 = none

        # Stats
        self.total_games = 0
        self.total_model_wins = 0
        self.recent_wins = []

        # Consecutive sixes & move counts
        self.consecutive_sixes = np.zeros((batch_size, 4), dtype=int)
        self.move_counts = np.zeros(batch_size, dtype=int)

        # Bots
        self.bots = {name: cls() for name, cls in BOT_CLASSES.items()}

    def get_temperature(self, total_games):
        if total_games >= TEMPERATURE_DECAY_GAMES:
            return TEMPERATURE_END
        progress = total_games / TEMPERATURE_DECAY_GAMES
        return TEMPERATURE_START - progress * (TEMPERATURE_START - TEMPERATURE_END)

    # --- Ghost management (same as V6) ---
    def _get_ghost_paths(self):
        if not os.path.exists(GHOSTS_DIR):
            return []
        ghosts = [
            os.path.join(GHOSTS_DIR, fname)
            for fname in os.listdir(GHOSTS_DIR)
            if fname.startswith("ghost_") and fname.endswith(".pt")
        ]
        return sorted(ghosts, reverse=True)

    def _load_ghost_model(self, ghost_path):
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
                ghost_pool, main_name="Model", strategy=SELFPLAY_GHOST_STRATEGY,
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
        if random.random() >= SELFPLAY_GHOST_FRACTION:
            return None
        return self._get_active_ghost()

    def play_step(self, train=True):
        """
        Advance all games by one step.
        Returns list of finished game results.
        """
        actions = []
        current_players = []
        decision_groups = {}  # (controller_type, controller_id) -> [game_indices]

        for i in range(self.batch_size):
            game = self.env.get_game(i)

            if game.is_terminal:
                actions.append(-1)
                current_players.append(-1)
                continue

            cp = game.current_player
            current_players.append(cp)

            if self.move_counts[i] >= MAX_MOVES_PER_GAME:
                game.is_terminal = True
                actions.append(-1)
                continue

            # Dice roll
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
                actions.append(-2)  # Placeholder
            else:
                bot = self.bots.get(ptype, self.bots['Random'])
                action = bot.select_move(game, legal_moves)
                actions.append(action)

        # Batched model action selection
        if decision_groups:
            main_temperature = self.get_temperature(self.trainer.total_games)

            for (controller, controller_id), indices in decision_groups.items():
                batch_sequences = []
                batch_legal_masks = []
                batch_legal_moves = []

                for idx in indices:
                    game = self.env.get_game(idx)
                    cp = current_players[idx]
                    lmoves = ludo_cpp.get_legal_moves(game)
                    batch_legal_moves.append(lmoves)

                    # Encode current state as 1D
                    tok, cont = encode_state_1d(game)

                    # Add current turn to history (action is from previous turn)
                    history = self.turn_histories[(idx, cp)]
                    last_act = int(self.last_actions[idx, cp])
                    history.add_turn(tok, cont, action=last_act)

                    # Get sequence for model
                    seq_tok, seq_cont, seq_acts, seq_mask = history.get_sequence()
                    batch_sequences.append((seq_tok, seq_cont, seq_acts, seq_mask))

                    # Legal mask
                    legal_mask = np.zeros(4, dtype=np.float32)
                    for move in lmoves:
                        legal_mask[move] = 1.0
                    batch_legal_masks.append(legal_mask)

                # Stack into batched tensors
                B = len(indices)
                all_tok = torch.from_numpy(np.stack([s[0] for s in batch_sequences])).to(self.device)
                all_cont = torch.from_numpy(np.stack([s[1] for s in batch_sequences])).to(self.device, dtype=torch.float32)
                all_acts = torch.from_numpy(np.stack([s[2] for s in batch_sequences])).to(self.device)
                all_mask = torch.from_numpy(np.stack([s[3] for s in batch_sequences])).to(self.device)
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
                        actions[idx] = random.choice(lmoves)
                    continue

                with torch.no_grad():
                    policy_logits = model.forward_policy_only(
                        all_tok, all_cont, all_acts, all_mask, masks_t
                    )
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

                    # Update last action for this player
                    self.last_actions[idx, cp] = action

                    if not train:
                        continue
                    if cp not in self.game_compositions[idx]['train_players']:
                        continue

                    if cp not in self.trajectories[idx]:
                        self.trajectories[idx][cp] = []

                    # Store the sequence snapshot for PPO training
                    self.trajectories[idx][cp].append({
                        'token_positions': batch_sequences[j][0].copy(),  # (K, 8)
                        'continuous': batch_sequences[j][1].copy(),       # (K, 9)
                        'actions_seq': batch_sequences[j][2].copy(),      # (K,)
                        'seq_mask': batch_sequences[j][3].copy(),         # (K,)
                        'action': action,
                        'legal_mask': batch_legal_masks[j].copy(),
                        'old_log_prob': float(old_lp_np[j]),
                        'temperature': float(sample_temperature),
                    })

        # Save pre-step states for reward computation
        pre_step_states = []
        for i in range(self.batch_size):
            game = self.env.get_game(i)
            old_pos = {p: list(game.player_positions[p]) for p in range(4)}
            pre_step_states.append(old_pos)

        # Step environment
        final_actions = [a if a >= 0 else -1 for a in actions]
        for i, a in enumerate(final_actions):
            if a >= 0:
                self.move_counts[i] += 1

        next_states_np, rewards_np, dones_np, info_list = self.env.step(final_actions)

        # Handle game completions
        results = []
        for i in range(self.batch_size):
            # Compute shaped reward
            cp = current_players[i]
            if train and cp >= 0 and self.trajectories[i] and cp in self.trajectories[i]:
                next_game = self.env.get_game(i)

                class DummyState:
                    def __init__(self, pos):
                        self.player_positions = pos

                dummy_old = DummyState(pre_step_states[i])
                dummy_new = DummyState(next_game.player_positions)

                step_reward = compute_shaped_reward(dummy_old, dummy_new, cp)

                last_idx = len(self.trajectories[i][cp]) - 1
                if last_idx >= 0:
                    self.trajectories[i][cp][last_idx]['step_reward'] = step_reward

            if dones_np[i]:
                winner = info_list[i]['winner']
                composition = self.game_compositions[i]
                mpid = composition['model_player']

                if winner == -1:
                    model_won = False
                else:
                    model_won = (winner == mpid)

                # Train on trajectories
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

                identities = composition['player_types']

                # Reset game
                self.env.reset_game(i)
                self.game_compositions[i] = self._random_composition()
                self.consecutive_sixes[i] = 0
                self.trajectories[i] = {}
                self.last_actions[i] = 4  # Reset to "no action"

                # Reset turn histories for this game slot
                for p in range(4):
                    self.turn_histories[(i, p)].reset()

                results.append({
                    'winner': winner,
                    'model_won': model_won,
                    'model_player': mpid,
                    'identities': identities,
                    'total_moves': int(self.move_counts[i]),
                    'game_duration': 0.0,
                })
                self.move_counts[i] = 0

        return results

    def get_recent_win_rate(self):
        if not self.recent_wins:
            return 0.0
        return sum(self.recent_wins) / len(self.recent_wins)

    def get_spectator_state(self, game_idx=0):
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
            'move_count': int(self.move_counts[game_idx]),
        }

    def _random_composition(self):
        """Generate random game composition (same logic as V6)."""
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
