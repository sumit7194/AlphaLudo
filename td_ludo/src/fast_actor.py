"""
Fast Actor for V9 Multi-Process Training

Runs game simulations on CPU with batched model inference.
Sends completed game trajectories to learner via Queue in compact format.

Compact format: instead of sending (K, 14, 15, 15) per step, we send
all unique turn grids (T, 14, 15, 15) and let the learner reconstruct
the K-length context windows. This reduces IPC bandwidth ~16x.
"""

import os
import random
import time
import numpy as np
import torch
import torch.nn.functional as F


# These are imported inside the worker function to avoid issues with
# multiprocessing 'spawn' start method
V9_IN_CHANNELS = 14
V9_EMBED_DIM = 80

NUM_ACTION_CLASSES = 5  # 0-3 pieces, 4=pass/none


class TurnHistory:
    """Lightweight turn history for actor (CPU). Same logic as TurnHistoryV9."""

    def __init__(self, K=16, embed_dim=V9_EMBED_DIM):
        self.K = K
        self.embed_dim = embed_dim
        self._grids = []
        self._actions = []
        self._cnn_features = []

    def reset(self):
        self._grids.clear()
        self._actions.clear()
        self._cnn_features.clear()

    def add_turn(self, grid, action=4, cnn_feature=None):
        self._grids.append(grid)
        self._actions.append(action)
        self._cnn_features.append(cnn_feature)
        if len(self._grids) > self.K:
            self._grids.pop(0)
            self._actions.pop(0)
            self._cnn_features.pop(0)

    def get_cached_sequence(self):
        n_valid = len(self._grids)
        n_pad = self.K - n_valid
        cached = np.zeros((self.K, self.embed_dim), dtype=np.float32)
        acts = np.full(self.K, 4, dtype=np.int64)
        mask = np.ones(self.K, dtype=bool)
        if n_valid > 0:
            acts[n_pad:] = np.array(self._actions, dtype=np.int64)
            mask[n_pad:] = False
            for j, feat in enumerate(self._cnn_features):
                if feat is not None:
                    cached[n_pad + j] = feat
        return cached, acts, mask

    @property
    def num_turns(self):
        return len(self._grids)


class FastActor:
    """
    Plays batched games on CPU, collects trajectories in compact format.
    """

    def __init__(self, model, batch_size, context_length=16, config=None):
        import td_ludo_cpp as ludo_cpp
        from src.heuristic_bot import (
            HeuristicLudoBot, AggressiveBot, DefensiveBot,
            RacingBot, RandomBot, ExpertBot,
        )

        self.ludo_cpp = ludo_cpp
        self.model = model
        self.batch_size = batch_size
        self.context_length = context_length
        self.config = config or {}

        two_player = self.config.get('num_active_players', 2) == 2
        self.env = ludo_cpp.VectorGameState(batch_size, two_player)

        self.bot_classes = {
            'Heuristic': HeuristicLudoBot,
            'Aggressive': AggressiveBot,
            'Defensive': DefensiveBot,
            'Racing': RacingBot,
            'Random': RandomBot,
            'Expert': ExpertBot,
        }
        self.bots = {name: cls() for name, cls in self.bot_classes.items()}

        # Config values (must be set before _random_composition is called)
        self.game_composition = self.config.get('game_composition', {
            'SelfPlay': 0.40, 'Expert': 0.25, 'Heuristic': 0.15,
            'Aggressive': 0.10, 'Defensive': 0.10,
        })
        self.ghosts_dir = self.config.get('ghosts_dir', '')
        self.selfplay_ghost_fraction = self.config.get('selfplay_ghost_fraction', 0.5)
        self.selfplay_ghost_strategy = self.config.get('selfplay_ghost_strategy', 'matched')
        self.max_moves = self.config.get('max_moves', 10000)
        self.num_active_players = self.config.get('num_active_players', 2)

        # Ghost cache
        self.ghost_cache = {}
        self.max_cached_ghosts = 4
        self.active_ghost = None
        self.active_ghost_selected_at = 0
        self.ghost_refresh_interval = 1000

        # Per-game state
        self.game_compositions = [self._random_composition() for _ in range(batch_size)]
        self.turn_histories = {}
        for i in range(batch_size):
            for p in range(4):
                self.turn_histories[(i, p)] = TurnHistory(context_length, V9_EMBED_DIM)

        self.last_actions = np.full((batch_size, 4), 4, dtype=np.int64)
        self.consecutive_sixes = np.zeros((batch_size, 4), dtype=int)
        self.move_counts = np.zeros(batch_size, dtype=int)

        # Trajectory collection (compact: only store per-turn data)
        self.game_trajectories = [{} for _ in range(batch_size)]

        # Stats
        self.total_games = 0
        self.model_wins = 0

    def get_temperature(self, total_games):
        t_start = self.config.get('temp_start', 1.1)
        t_end = self.config.get('temp_end', 0.95)
        t_decay = self.config.get('temp_decay_games', 20000)
        if total_games >= t_decay:
            return t_end
        progress = total_games / t_decay
        return t_start - progress * (t_start - t_end)

    def play_step(self, total_games_global=0):
        """Advance all games by one step. Returns list of completed game data dicts."""
        from src.reward_shaping import compute_shaped_reward
        ludo_cpp = self.ludo_cpp

        actions = []
        current_players = []
        decision_groups = {}

        for i in range(self.batch_size):
            game = self.env.get_game(i)

            if game.is_terminal:
                actions.append(-1)
                current_players.append(-1)
                continue

            cp = game.current_player
            current_players.append(cp)

            if self.move_counts[i] >= self.max_moves:
                game.is_terminal = True
                actions.append(-1)
                continue

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
                actions.append(-2)  # placeholder
            else:
                bot = self.bots.get(ptype, self.bots['Random'])
                action = bot.select_move(game, legal_moves)
                actions.append(action)

        # Batched model inference for decision groups
        if decision_groups:
            temperature = self.get_temperature(total_games_global)

            for (controller, controller_id), indices in decision_groups.items():
                batch_new_grids = []
                batch_legal_masks = []
                batch_legal_moves = []

                for idx in indices:
                    game = self.env.get_game(idx)
                    cp = current_players[idx]
                    lmoves = ludo_cpp.get_legal_moves(game)
                    batch_legal_moves.append(lmoves)

                    grid = ludo_cpp.encode_state_v9(game)  # (14, 15, 15)
                    history = self.turn_histories[(idx, cp)]
                    last_act = int(self.last_actions[idx, cp])
                    history.add_turn(grid, action=last_act, cnn_feature=None)
                    batch_new_grids.append(grid)

                    legal_mask = np.zeros(4, dtype=np.float32)
                    for m in lmoves:
                        legal_mask[m] = 1.0
                    batch_legal_masks.append(legal_mask)

                # Select model
                if controller == 'Ghost':
                    inference_model = self._load_ghost(controller_id)
                    sample_temp = 1.0
                else:
                    inference_model = self.model
                    sample_temp = temperature

                if inference_model is None:
                    for j, idx in enumerate(indices):
                        actions[idx] = random.choice(batch_legal_moves[j])
                    continue

                # CNN features for new grids
                with torch.no_grad():
                    new_grids_t = torch.from_numpy(np.stack(batch_new_grids)).float()
                    new_cnn = inference_model.compute_single_cnn_features(new_grids_t)
                    new_cnn_np = new_cnn.numpy()

                # Update cached CNN features and build sequences
                batch_cached = []
                for j, idx in enumerate(indices):
                    cp = current_players[idx]
                    history = self.turn_histories[(idx, cp)]
                    history._cnn_features[-1] = new_cnn_np[j]
                    cached_cnn, seq_acts, seq_mask = history.get_cached_sequence()
                    batch_cached.append((cached_cnn, seq_acts, seq_mask))

                # Batched transformer forward
                masks_t = torch.from_numpy(np.stack(batch_legal_masks)).float()
                all_cached = torch.from_numpy(
                    np.stack([s[0] for s in batch_cached])
                ).float()
                all_acts = torch.from_numpy(
                    np.stack([s[1] for s in batch_cached])
                )
                all_mask = torch.from_numpy(
                    np.stack([s[2] for s in batch_cached])
                )

                with torch.no_grad():
                    policy_logits = inference_model.forward_policy_only_cached(
                        all_cached, all_acts, all_mask, masks_t
                    )
                    if sample_temp != 1.0:
                        probs = F.softmax(policy_logits / sample_temp, dim=1)
                    else:
                        probs = F.softmax(policy_logits, dim=1)
                    # Fix any NaN/inf from masking edge cases
                    bad = torch.isnan(probs).any(dim=1) | (probs.sum(dim=1) < 0.99)
                    if bad.any():
                        probs[bad] = masks_t[bad] / masks_t[bad].sum(dim=1, keepdim=True).clamp_min(1e-8)
                    probs = probs.clamp_min(1e-8)
                    probs = probs / probs.sum(dim=1, keepdim=True)
                    sampled = torch.multinomial(probs, num_samples=1).squeeze(1)
                    old_lps = torch.log(
                        probs.gather(1, sampled.unsqueeze(1)).squeeze(1) + 1e-8
                    )

                sampled_np = sampled.numpy()
                old_lps_np = old_lps.numpy()

                for j, idx in enumerate(indices):
                    action = int(sampled_np[j])
                    lmoves = batch_legal_moves[j]
                    if action not in lmoves:
                        action = random.choice(lmoves)

                    actions[idx] = action
                    cp = current_players[idx]
                    self.last_actions[idx, cp] = action

                    # Store trajectory step (compact: just the new grid)
                    if cp in self.game_compositions[idx]['train_players']:
                        if cp not in self.game_trajectories[idx]:
                            self.game_trajectories[idx][cp] = {
                                'grids': [],
                                'actions': [],
                                'legal_masks': [],
                                'old_log_probs': [],
                                'temperatures': [],
                                'step_rewards': [],
                            }
                        traj = self.game_trajectories[idx][cp]
                        traj['grids'].append(batch_new_grids[j].copy())
                        traj['actions'].append(action)
                        traj['legal_masks'].append(batch_legal_masks[j].copy())
                        traj['old_log_probs'].append(float(old_lps_np[j]))
                        traj['temperatures'].append(float(sample_temp))
                        traj['step_rewards'].append(0.0)  # filled after env step

        # Pre-step states for reward computation
        pre_states = []
        for i in range(self.batch_size):
            game = self.env.get_game(i)
            pre_states.append(
                {p: list(game.player_positions[p]) for p in range(4)}
            )

        # Step environment
        final_actions = [a if a >= 0 else -1 for a in actions]
        for i, a in enumerate(final_actions):
            if a >= 0:
                self.move_counts[i] += 1

        _, _, dones_np, info_list = self.env.step(final_actions)

        # Compute shaped rewards and handle completions
        completed = []
        for i in range(self.batch_size):
            cp = current_players[i]

            # Shaped reward for trainable players
            if cp >= 0 and cp in self.game_trajectories[i]:
                traj = self.game_trajectories[i][cp]
                if traj['step_rewards']:
                    next_game = self.env.get_game(i)

                    class _S:
                        def __init__(self, pos):
                            self.player_positions = pos

                    reward = compute_shaped_reward(
                        _S(pre_states[i]),
                        _S(next_game.player_positions),
                        cp,
                    )
                    traj['step_rewards'][-1] = reward

            if dones_np[i]:
                winner = info_list[i]['winner']
                comp = self.game_compositions[i]
                mpid = comp['model_player']
                model_won = (winner == mpid) if winner >= 0 else False

                # Package completed game trajectories
                if winner >= 0:
                    for train_player in comp['train_players']:
                        if train_player in self.game_trajectories[i]:
                            traj = self.game_trajectories[i][train_player]
                            if traj['grids']:
                                game_data = self._package_game(
                                    traj, winner, train_player, comp
                                )
                                completed.append(game_data)

                self.total_games += 1
                if model_won:
                    self.model_wins += 1

                # Reset game slot
                self.env.reset_game(i)
                self.game_compositions[i] = self._random_composition()
                self.consecutive_sixes[i] = 0
                self.game_trajectories[i] = {}
                self.last_actions[i] = 4
                self.move_counts[i] = 0
                for p in range(4):
                    self.turn_histories[(i, p)].reset()

        return completed

    def _package_game(self, traj, winner, model_player, composition):
        """Package a completed game's trajectory in compact format."""
        n_active = self.num_active_players
        T = len(traj['grids'])

        # Compute discounted returns
        loss_penalty = -1.0 / max(1, n_active - 1)
        z = 1.0 if model_player == winner else loss_penalty
        gamma = 0.999
        returns = np.zeros(T, dtype=np.float32)
        R = 0.0
        for i in reversed(range(T)):
            r_t = traj['step_rewards'][i]
            if i == T - 1:
                r_t += z
            R = r_t + gamma * R
            returns[i] = R

        step_actions = np.array(traj['actions'], dtype=np.int64)

        # Build prev_actions: at turn j, the prev_action is what was chosen at j-1
        prev_actions = np.full(T, 4, dtype=np.int64)
        if T > 1:
            prev_actions[1:] = step_actions[:-1]

        return {
            'player_grids': np.stack(traj['grids']),       # (T, 14, 15, 15)
            'prev_actions': prev_actions,                    # (T,)
            'step_actions': step_actions,                    # (T,)
            'legal_masks': np.stack(traj['legal_masks']),   # (T, 4)
            'old_log_probs': np.array(traj['old_log_probs'], dtype=np.float32),
            'temperatures': np.array(traj['temperatures'], dtype=np.float32),
            'returns': returns,                              # (T,)
            'game_info': {
                'winner': winner,
                'model_won': model_player == winner,
                'model_player': model_player,
                'identities': composition['player_types'],
                'total_moves': T,
            },
        }

    def load_weights(self, weight_path):
        """Load model weights from file (for periodic sync from learner)."""
        try:
            state_dict = torch.load(weight_path, map_location='cpu', weights_only=True)
            self.model.load_state_dict(state_dict)
            return True
        except Exception:
            return False

    # =========================================================================
    # Ghost Management (same logic as VectorV9GamePlayer)
    # =========================================================================
    def _get_ghost_paths(self):
        if not self.ghosts_dir or not os.path.exists(self.ghosts_dir):
            return []
        ghosts = [
            os.path.join(self.ghosts_dir, f)
            for f in os.listdir(self.ghosts_dir)
            if f.startswith('ghost_') and f.endswith('.pt')
        ]
        return sorted(ghosts, reverse=True)

    def _load_ghost(self, ghost_path):
        if ghost_path in self.ghost_cache:
            return self.ghost_cache[ghost_path]
        if not os.path.exists(ghost_path):
            return None

        from src.model_v9 import AlphaLudoV9
        model = AlphaLudoV9(context_length=self.context_length)
        checkpoint = torch.load(ghost_path, map_location='cpu', weights_only=False)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)

        self.ghost_cache[ghost_path] = model
        while len(self.ghost_cache) > self.max_cached_ghosts:
            old_key = next(iter(self.ghost_cache))
            del self.ghost_cache[old_key]

        return model

    def _get_active_ghost(self):
        ghost_pool = self._get_ghost_paths()
        if not ghost_pool:
            self.active_ghost = None
            return None
        needs_refresh = (
            self.active_ghost is None
            or self.active_ghost['path'] not in ghost_pool
            or (self.total_games - self.active_ghost_selected_at) >= self.ghost_refresh_interval
        )
        if not needs_refresh:
            return self.active_ghost

        # Simple random selection (no elo tracker in actor)
        ghost_path = random.choice(ghost_pool)
        ghost_name = os.path.basename(ghost_path).replace('.pt', '')
        self.active_ghost = {'path': ghost_path, 'name': ghost_name}
        self.active_ghost_selected_at = self.total_games
        return self.active_ghost

    def _pick_selfplay_opponent(self):
        if random.random() >= self.selfplay_ghost_fraction:
            return None
        return self._get_active_ghost()

    # =========================================================================
    # Game Composition (same logic as VectorV9GamePlayer)
    # =========================================================================
    def _random_composition(self):
        probs = self.game_composition
        r = random.random()
        cumulative = 0.0
        game_type = 'SelfPlay'
        for gtype, prob in probs.items():
            cumulative += prob
            if r < cumulative:
                game_type = gtype
                break

        if self.num_active_players == 2:
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

        # 4-player mode
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
                remaining = [s for s in bot_seats if s != primary_seat]
                bot_options = list(self.bot_classes.keys()) + ['Random']
                for seat in remaining:
                    player_types[seat] = random.choice(bot_options)
                    controllers[seat] = player_types[seat]
        return {
            'model_player': model_player,
            'player_types': player_types,
            'controllers': controllers,
            'ghost_paths': ghost_paths,
            'train_players': [model_player],
        }


def actor_worker(actor_id, batch_size, context_length,
                 trajectory_queue, stats_queue,
                 weight_path, weight_version, total_games_counter,
                 stop_event,
                 config):
    """
    Actor process entry point.
    Runs on CPU, plays games, sends trajectories to learner.
    """
    import torch
    torch.set_num_threads(2)  # limit per-actor thread usage

    from src.model_v9 import AlphaLudoV9

    # Create model on CPU
    model = AlphaLudoV9(context_length=context_length)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    # Load initial weights
    if os.path.exists(weight_path):
        try:
            state_dict = torch.load(weight_path, map_location='cpu', weights_only=True)
            model.load_state_dict(state_dict)
        except Exception as e:
            print(f"[Actor {actor_id}] Warning: could not load initial weights: {e}")

    actor = FastActor(model, batch_size, context_length, config)

    local_weight_version = weight_version.value
    last_weight_check = time.time()
    weight_check_interval = 5.0  # seconds

    games_played = 0
    wins = 0
    last_stats_time = time.time()

    print(f"[Actor {actor_id}] Started (batch={batch_size}, CPU)")

    while not stop_event.is_set():
        total_games_global = total_games_counter.value
        completed = actor.play_step(total_games_global=total_games_global)

        for game_data in completed:
            try:
                trajectory_queue.put(game_data, timeout=5.0)
            except Exception:
                if stop_event.is_set():
                    break
                continue

            games_played += 1
            if game_data['game_info']['model_won']:
                wins += 1

        # Report stats periodically
        now = time.time()
        if now - last_stats_time > 10.0:
            try:
                stats_queue.put({
                    'type': 'actor_stats',
                    'actor_id': actor_id,
                    'games': games_played,
                    'wins': wins,
                    'timestamp': now,
                }, block=False)
            except Exception:
                pass
            last_stats_time = now

        # Check for weight updates
        if now - last_weight_check > weight_check_interval:
            current_version = weight_version.value
            if current_version > local_weight_version:
                if actor.load_weights(weight_path):
                    local_weight_version = current_version
            last_weight_check = now

    print(f"[Actor {actor_id}] Shutting down ({games_played} games, "
          f"{wins}/{games_played} wins)")
