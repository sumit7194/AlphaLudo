import torch
import numpy as np
import time
import os
from collections import deque
import ludo_cpp
from src.tensor_utils_mastery import state_to_tensor_mastery, get_board_coords, BOARD_SIZE
from src.vector_mcts import VectorMCTSMastery, get_action_probs_vector
from src.model_v3 import AlphaLudoV3
from src.training_utils import get_temperature, EloTracker, rotate_token_indices
from src.heuristic_bot import HeuristicLudoBot, AggressiveBot, DefensiveBot, RacingBot, get_bot
try:
    from src.visualizer import visualizer
except ImportError:
    visualizer = None

# Bot type registry for multi-heuristic training
BOT_TYPES = ['Heuristic', 'Aggressive', 'Defensive', 'Racing']


class VectorLeagueWorker:
    def __init__(self, main_model, probabilities, mcts_simulations=50, visualize=False, 
                 ghost_pool=None, elo_tracker=None, temp_schedule='alphazero', 
                 c_puct=3.0, dirichlet_alpha=0.3, dirichlet_eps=0.25, actor_id=0):
        self.actor_id = actor_id
        self.main_model = main_model
        self.mcts_simulations = mcts_simulations
        self.visualize = visualize
        self.probabilities = probabilities
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_eps = dirichlet_eps
        
        # Ghost Pool Management
        self.ghost_pool = ghost_pool if ghost_pool else []
        self.ghost_model = None
        self.ghost_fraction = 0.25  # 25% of games have a ghost opponent
        self.elo_tracker = elo_tracker  # For smart ghost selection and tracking
        self.temp_schedule = temp_schedule  # Temperature schedule: 'constant', 'alphazero', 'linear'
        self.current_ghost_name = None  # Track which ghost is loaded
        
        # Load a ghost model if pool exists
        if self.ghost_pool:
            self._load_ghost()
        
        print(f"VectorLeagueWorker initialized. Batch MCTS Sims: {mcts_simulations}. Ghosts: {len(self.ghost_pool)}. Temp: {temp_schedule}")
        
        # New Features: Time Tracking & Pause Logic
        self.game_start_times = {} # game_id -> start_timestamp
        self.paused_games = set() # Set of game_ids currently paused
        self.visualizer_instance = None # To bridge commands if needed

    def update_params(self, c_puct, dirichlet_eps, probabilities=None):
        """Dynamic Parameter Update from Auto-Tuner"""
        if c_puct != self.c_puct or dirichlet_eps != self.dirichlet_eps:
            print(f"[Worker {self.actor_id}] Updating Params: CPUCT {self.c_puct}->{c_puct}, EPS {self.dirichlet_eps}->{dirichlet_eps}")
            self.c_puct = c_puct
            self.dirichlet_eps = dirichlet_eps
        
        if probabilities and probabilities != self.probabilities:
            print(f"[Worker {self.actor_id}] Updating Game Composition: {probabilities}")
            self.probabilities = probabilities

    def pause_game(self, game_id):
        """Pause a specific game. Enforces only one game paused at a time."""
        self.paused_games.clear() # Only allow one paused game
        self.paused_games.add(game_id)
        print(f"Paused Game {game_id}")

    def resume_game(self, game_id):
        """Resume a specific game."""
        if game_id in self.paused_games:
            self.paused_games.remove(game_id)
            print(f"Resumed Game {game_id}")

    def resume_all(self):
        """Resume all paused games."""
        if self.paused_games:
            print(f"Resuming {len(self.paused_games)} paused games.")
            self.paused_games.clear()

    def _load_ghost(self, strategy='adversarial'):
        """Load a ghost model from the pool using Elo-based selection."""
        if not self.ghost_pool:
            return
        
        # Use Elo tracker for smart selection, or random if no tracker
        if self.elo_tracker:
            ghost_path = self.elo_tracker.select_ghost(self.ghost_pool, 'Main', strategy)
        else:
            ghost_path = np.random.choice(self.ghost_pool)
        
        device = next(self.main_model.parameters()).device
        try:
            checkpoint = torch.load(ghost_path, map_location=device)
            self.ghost_model = AlphaLudoTopNet()
            self.ghost_model.load_state_dict(checkpoint['model_state_dict'])
            self.ghost_model.to(device)
            self.ghost_model.eval()
            self.current_ghost_name = ghost_path.split('/')[-1].replace('.pt', '')
            print(f"  Loaded Ghost: {self.current_ghost_name}")
        except Exception as e:
            print(f"  Failed to load ghost: {e}")
            self.ghost_model = None
            self.current_ghost_name = None

    def _load_random_ghost(self):
        """Legacy method - calls _load_ghost with random selection."""
        self._load_ghost(strategy='random')

    def get_reward_shaping(self, prev_state, action, next_state, player):
        """Return 0 - pure win/loss signal only (AlphaZero style)."""
        return 0.0

    def play_batch(self, batch_size=32, temperature=1.0, epoch=None):
        batch_start_time = time.time()
        """
        Play a batch of games using C++ MCTS with ghost support.
        Optimized with 'Fast Forward' logic to cascade trivial turns without waiting for MCTS.
        """
        # Load 2 ghosts: A (adversarial), B (random)
        ghost_a_model, ghost_a_name = None, None
        ghost_b_model, ghost_b_name = None, None
        
        if self.ghost_pool and len(self.ghost_pool) >= 2:
            device = next(self.main_model.parameters()).device
            
            # Ghost A: Adversarial selection
            if self.elo_tracker:
                ghost_a_path = self.elo_tracker.select_ghost(self.ghost_pool, 'Main', 'adversarial')
            else:
                ghost_a_path = np.random.choice(self.ghost_pool)
            
            # Ghost B: Random selection (different from A)
            remaining = [p for p in self.ghost_pool if p != ghost_a_path]
            ghost_b_path = np.random.choice(remaining) if remaining else ghost_a_path
            
            # Load Ghost A
            try:
                ckpt_a = torch.load(ghost_a_path, map_location=device)
                ghost_a_model = AlphaLudoTopNet()
                ghost_a_model.load_state_dict(ckpt_a['model_state_dict'])
                ghost_a_model.to(device).eval()
                ghost_a_name = ghost_a_path.split('/')[-1].replace('.pt', '')
                print(f"  Loaded Ghost A: {ghost_a_name} (adversarial)")
            except Exception as e:
                print(f"  Failed to load ghost A: {e}")
            
            # Load Ghost B
            try:
                ckpt_b = torch.load(ghost_b_path, map_location=device)
                ghost_b_model = AlphaLudoTopNet()
                ghost_b_model.load_state_dict(ckpt_b['model_state_dict'])
                ghost_b_model.to(device).eval()
                ghost_b_name = ghost_b_path.split('/')[-1].replace('.pt', '')
                print(f"  Loaded Ghost B: {ghost_b_name} (random)")
            except Exception as e:
                print(f"  Failed to load ghost B: {e}")
        
        elif self.ghost_pool:
            # Only 1 ghost available, use it for both
            self._load_ghost(strategy='adversarial')
            ghost_a_model = self.ghost_model
            ghost_a_name = self.current_ghost_name
            ghost_b_model = self.ghost_model
            ghost_b_name = self.current_ghost_name
        
        # Store for use later
        self.ghost_model = ghost_a_model
        
        # Instantiate all Heuristic Bot variants
        self.bots = {
            'Heuristic': HeuristicLudoBot(),
            'Aggressive': AggressiveBot(),
            'Defensive': DefensiveBot(),
            'Racing': RacingBot(),
        }
        
        # Initialize C++ MCTS Engine with v3 parameters
        if not hasattr(self, 'cpp_mcts') or self.cpp_mcts_batch_size != batch_size:
            self.cpp_mcts = ludo_cpp.MCTSEngine(batch_size, self.c_puct, 
                                                 self.dirichlet_alpha, self.dirichlet_eps)
            self.cpp_mcts_batch_size = batch_size
            print(f"[Worker {self.actor_id}] Created MCTSEngine: CPUCT={self.c_puct}, Alpha={self.dirichlet_alpha}, EPS={self.dirichlet_eps}")

        # Game layout based on probabilities
        probs = self.probabilities or {'Main': 1.0}
        
        heuristic_prob = probs.get('Heuristic', 0.10)
        aggressive_prob = probs.get('Aggressive', 0.07)
        defensive_prob = probs.get('Defensive', 0.07)
        racing_prob = probs.get('Racing', 0.06)
        ghost_prob = probs.get('Ghost', 0.20)
        
        num_ghost = int(batch_size * ghost_prob)
        num_heuristic = int(batch_size * heuristic_prob)
        num_aggressive = int(batch_size * aggressive_prob)
        num_defensive = int(batch_size * defensive_prob)
        num_racing = int(batch_size * racing_prob)
        
        total_bot = num_heuristic + num_aggressive + num_defensive + num_racing
        
        # Fallback: If total_bot is 0 due to rounding but probs > 0, force 1 bot.
        if total_bot == 0 and (heuristic_prob + aggressive_prob + defensive_prob + racing_prob) > 0:
            # Pick one type proportional to their probs (or just uniform among available)
            candidates = []
            if heuristic_prob > 0: candidates.append('Heuristic')
            if aggressive_prob > 0: candidates.append('Aggressive')
            if defensive_prob > 0: candidates.append('Defensive')
            if racing_prob > 0: candidates.append('Racing')
            
            if candidates:
                chosen = np.random.choice(candidates)
                if chosen == 'Heuristic': num_heuristic = 1
                elif chosen == 'Aggressive': num_aggressive = 1
                elif chosen == 'Defensive': num_defensive = 1
                elif chosen == 'Racing': num_racing = 1
                total_bot = 1
        
        num_main = batch_size - num_ghost - total_bot
        num_main = max(0, num_main)
        
        # DEBUG: Print distribution
        # print(f"DEBUG_COUNTS: Main={num_main}, Ghost={num_ghost}, Bots={total_bot}")
        
        # Assign indices
        indices = list(range(batch_size))
        idx = 0
        
        main_indices = set(indices[idx : idx + num_main])
        idx += num_main
        ghost_indices = set(indices[idx : idx + num_ghost])
        idx += num_ghost
        heuristic_indices = set(indices[idx : idx + num_heuristic])
        idx += num_heuristic
        aggressive_indices = set(indices[idx : idx + num_aggressive])
        idx += num_aggressive
        defensive_indices = set(indices[idx : idx + num_defensive])
        idx += num_defensive
        racing_indices = set(indices[idx : idx + num_racing])
        
        # Map game index -> bot type
        game_bot_types = {}
        for i in heuristic_indices: game_bot_types[i] = 'Heuristic'
        for i in aggressive_indices: game_bot_types[i] = 'Aggressive'
        for i in defensive_indices: game_bot_types[i] = 'Defensive'
        for i in racing_indices: game_bot_types[i] = 'Racing'
        
        # Distribute Ghosts (A/B)
        ghost_a_indices = set()
        ghost_b_indices = set()
        if ghost_indices:
            ghost_list = list(ghost_indices)
            mid = len(ghost_list) // 2
            ghost_a_indices = set(ghost_list[:mid + 1])
            ghost_b_indices = set(ghost_list[mid + 1:])
            
        ghost_game_indices = ghost_a_indices | ghost_b_indices
        all_bot_indices = heuristic_indices | aggressive_indices | defensive_indices | racing_indices

        # Stats
        consecutive_sixes = np.zeros(batch_size, dtype=int)
        states = [ludo_cpp.create_initial_state() for _ in range(batch_size)]
        
        # Init Start Times
        self.game_start_times = {} 
        for i in range(batch_size):
            self.game_start_times[i] = time.time()
        histories = [[] for _ in range(batch_size)]
        move_counts = [0] * batch_size
        
        # Game Identities
        game_identities = []
        for idx in range(batch_size):
            if idx in ghost_a_indices and ghost_a_model:
                ghost_pos = np.random.randint(0, 4)
                ids = ['Main'] * 4
                ids[ghost_pos] = ghost_a_name or 'Ghost_A'
                game_identities.append(ids)
            elif idx in ghost_b_indices and ghost_b_model:
                ghost_pos = np.random.randint(0, 4)
                ids = ['Main'] * 4
                ids[ghost_pos] = ghost_b_name or 'Ghost_B'
                game_identities.append(ids)
            elif idx in game_bot_types:
                bot_type = game_bot_types[idx]
                bot_pos = np.random.randint(0, 4)
                ids = ['Main'] * 4
                ids[bot_pos] = bot_type
                game_identities.append(ids)
            else:
                game_identities.append(['Main'] * 4)
        
        # Broadcast Init
        # Broadcast Init
        if self.visualize and visualizer:
            visualizer.broadcast_batch_init(batch_size)
            if ghost_game_indices: visualizer.broadcast_ghost_games(list(ghost_game_indices))
            if all_bot_indices: visualizer.broadcast_heuristic_games(list(all_bot_indices))

        active_indices = list(range(batch_size))
        completed_episodes = 0
        target_episodes = batch_size
        all_examples = []
        results = []
        
        # --- FAST FORWARD LOOP ---
        loop_cnt = 0
        while len(active_indices) > 0:
            loop_cnt += 1
            # Note: Stop signal is now handled at data_worker level between batches
            # This allows the current batch to complete fully

            if completed_episodes >= target_episodes:
                break
            
            # Queue for processing Active Games
            # We process them until they hit a 'MCTS Block' or 'Pause'
            processing_queue = list(active_indices)
            
            # The Waiting Room for MCTS
            multi_move_states = []
            multi_move_idx_map = []
            
            # Clear paused games from queue initially
            processing_queue = [i for i in processing_queue if i not in self.paused_games]
            
            if not processing_queue and not self.paused_games:
                break # Should not happen if active_indices > 0

            # Drain the queue of trivial moves
            while processing_queue:
                idx = processing_queue.pop(0)
                
                # Double Check Pause (in case changed mid-loop if threaded, unlikely here but safe)
                if idx in self.paused_games: 
                    continue

                state = states[idx]
                
                # A. Check Terminal
                if state.is_terminal or move_counts[idx] >= 1000:
                    winner = ludo_cpp.get_winner(state) if state.is_terminal else -1
                    if self.visualize and visualizer:
                         visualizer.broadcast_game_result(idx, winner)

                    identities = game_identities[idx]
                    
                    if completed_episodes < target_episodes:
                         duration = time.time() - self.game_start_times[idx]
                         results.append({'winner': winner, 'identities': identities, 'duration': duration})
                         
                         for record in histories[idx]:
                             p = record['player']
                             if winner == -1: val = 0.0
                             elif winner == p: val = 1.0
                             else: val = -1.0
                             
                             if identities[p] == 'Main':
                                 # v3: Policy is already 4-dimensional (one per token)
                                 policy_4 = torch.tensor(record['policy'], dtype=torch.float32)
                                 all_examples.append((record['state'], policy_4, torch.tensor(val, dtype=torch.float32)))
                         
                         completed_episodes += 1
                         if self.elo_tracker and winner >= 0:
                             self.elo_tracker.update_from_game(identities, winner, epoch=epoch)
                         
                         print(f"Episode {completed_episodes}/{target_episodes} finished (Winner: {winner})")
                    
                    active_indices.remove(idx) # Done
                    continue

                # B. Roll Dice Logic
                if state.current_dice_roll == 0:
                    state.current_dice_roll = np.random.randint(1, 7)
                    if state.current_dice_roll == 6:
                        consecutive_sixes[idx] += 1
                    else:
                        consecutive_sixes[idx] = 0
                
                    # Check Penalty
                    if consecutive_sixes[idx] >= 3:
                         if self.visualize and visualizer:
                              visualizer.broadcast_move(state.current_player, -1, 6, game_id=idx)
                         
                         state.current_player = (state.current_player + 1) % 4
                         state.current_dice_roll = 0
                         consecutive_sixes[idx] = 0
                         
                         if self.visualize and visualizer:
                              visualizer.broadcast_state(state, game_id=idx)
                         
                         # Re-process this game immediately (it's next turn now)
                         processing_queue.append(idx)
                         continue
                
                # C. Check Legal Moves
                current_dice = state.current_dice_roll
                legal_moves = ludo_cpp.get_legal_moves(state)
                
                if len(legal_moves) == 0:
                     if self.visualize and visualizer:
                         visualizer.broadcast_move(state.current_player, -1, current_dice, game_id=idx)
                     state.current_player = (state.current_player + 1) % 4
                     state.current_dice_roll = 0
                     if self.visualize and visualizer:
                         visualizer.broadcast_state(state, game_id=idx)
                     
                     # Re-process immediately
                     processing_queue.append(idx)
                     continue

                # D. Forced / Heuristic / Instant Win Checks
                current_p = state.current_player
                identity = game_identities[idx][current_p]
                
                forced_action = None
                
                # D1. Heuristic Bot
                if identity in self.bots:
                    forced_action = self.bots[identity].select_move(state, legal_moves)
                
                # D2. Single Move
                elif len(legal_moves) == 1:
                    forced_action = legal_moves[0]
                
                # D3. Instant Win & Equiv State
                else:
                    # Instant Win
                    if state.scores[current_p] == 3:
                        for t in legal_moves:
                            pos = state.player_positions[current_p][t]
                            if pos >= 0 and (pos + current_dice) == 56:
                                forced_action = t
                                break
                    
                    # Equiv State
                    if forced_action is None:
                        # Logic: if all moves start from same position, they are identical
                        first_pos = state.player_positions[current_p][legal_moves[0]]
                        if all(state.player_positions[current_p][t] == first_pos for t in legal_moves):
                             forced_action = legal_moves[0]

                # E. Execute Forced Action
                if forced_action is not None:
                    # Log Training Data (100% policy)
                    state_tensor = state_to_tensor_mastery(state)
                    positions = state.player_positions[current_p]
                    token_indices = []
                    for t in range(4):
                        r, c = get_board_coords(current_p, positions[t], t)
                        token_indices.append(r * BOARD_SIZE + c)
                    token_indices_t = torch.tensor(token_indices, dtype=torch.long)
                    token_indices_rotated = rotate_token_indices(token_indices_t, current_p).tolist()
                    
                    probs = np.zeros(4)
                    probs[forced_action] = 1.0
                    
                    histories[idx].append({
                        'state': state_tensor,
                        'token_indices': token_indices_rotated,
                        'policy': probs,
                        'player': current_p
                    })
                    
                    # Apply
                    prev_state_copy = ludo_cpp.GameState()
                    prev_state_copy.current_dice_roll = state.current_dice_roll # Store for viz
                    states[idx] = ludo_cpp.apply_move(state, forced_action)
                    move_counts[idx] += 1
                    
                    if self.visualize and visualizer:
                        visualizer.broadcast_move(current_p, forced_action, prev_state_copy.current_dice_roll, game_id=idx)
                        visualizer.broadcast_state(states[idx], game_id=idx)
                    
                    # Re-process immediately
                    processing_queue.append(idx)
                    continue

                # F. MCTS Required
                # If we reach here, we have >1 move and it's not trivial.
                # Add to Batch List.
                multi_move_states.append(state)
                multi_move_idx_map.append(idx)
            
            
            # --- END OF FAST FORWARD QUEUE ---
            # Now we have a batch of states waiting for MCTS.
            
            if not multi_move_states:
                # No MCTS needed this cycle. 
                # If active games exist but none in MCTS list, and none in queue -> All Paused OR Error.
                if not active_indices: break
                if all(i in self.paused_games for i in active_indices):
                     time.sleep(0.1) # Prevent busy loop when paused
                continue 
            
            # Run Batch MCTS with v3 parameters
            # Use subset size
            self.cpp_mcts = ludo_cpp.MCTSEngine(len(multi_move_states), self.c_puct,
                                                 self.dirichlet_alpha, self.dirichlet_eps)
            self.cpp_mcts.set_roots(multi_move_states)
            
             # Determine dynamic simulations
            # FIXED: Always use full simulations. Early game throttling (50/100) was hurting exploration.
            current_sims = self.mcts_simulations

            # Run Batch MCTS
            # Optimization: Use Leaf Parallelism (Virtual Loss)
            # Instead of 1 leaf per game per loop, we pick 8 leaves per game using Virtual Loss.
            # This reduces sequential GPU roundtrips by 8x.
            
            PARALLEL_SIMS = 8
            mcts_loops = max(1, current_sims // PARALLEL_SIMS) 
            
            for i_loop in range(mcts_loops):
                # 1. Selection (Selects PARALLEL_SIMS * BATCH leaves)
                self.cpp_mcts.select_leaves(parallel_sims=PARALLEL_SIMS) 
                
                leaf_tensors_np = self.cpp_mcts.get_leaf_tensors()
                
                if leaf_tensors_np.size == 0:
                    break
                    
                # 2. Prediction (Batched) - v3: Use forward_policy_value (2 outputs, skip aux)
                device = next(self.main_model.parameters()).device
                model_dtype = next(self.main_model.parameters()).dtype
                input_tensor = torch.from_numpy(leaf_tensors_np.copy()).to(device=device, dtype=model_dtype)
                with torch.no_grad():
                    # v3 model returns (policy, value) from forward_policy_value
                    # or (policy, value, aux) from forward - use the faster one
                    if hasattr(self.main_model, 'forward_policy_value'):
                        pi_probs, v = self.main_model.forward_policy_value(input_tensor)
                    else:
                        pi_probs, v = self.main_model(input_tensor)[:2]

                # 3. Expansion & Backprop - v3: policy is already probabilities, not logits
                self.cpp_mcts.expand_and_backprop(pi_probs.float().cpu().numpy(), v.float().cpu().numpy().flatten())
            
            # Get Probabilities - v3: Returns 4-dim array per game (one per token)
            mcts_probs = self.cpp_mcts.get_action_probs(temperature)
            
            # Apply Steps
            for mcts_idx, batch_idx in enumerate(multi_move_idx_map):
                state = states[batch_idx]
                probs = np.array(mcts_probs[mcts_idx])  # Convert list to numpy (4,) array
                
                # Safety: Check for NaN policy (happens if MCTS had 0 visits)
                if np.isnan(probs).any() or probs.sum() < 1e-6:
                    legal_m = ludo_cpp.get_legal_moves(state)
                    probs = np.zeros(4)
                    if len(legal_m) > 0:
                        for m in legal_m:
                            probs[m] = 1.0 / len(legal_m)
                    
                current_p = state.current_player
                
                # Log History - v3: Simplified (no token_indices needed)
                state_tensor = state_to_tensor_mastery(state)
                
                histories[batch_idx].append({
                    'state': state_tensor,
                    'policy': probs,  # 4-dim policy
                    'player': current_p
                })
                
                # Select Action
                legal_moves = ludo_cpp.get_legal_moves(state)
                temp = get_temperature(move_counts[batch_idx], self.temp_schedule)
                valid_probs = np.array([probs[m] for m in legal_moves])
                valid_probs = np.clip(valid_probs, 1e-8, None)
                
                if temp < 0.01:
                    action_idx = np.argmax(valid_probs)
                else:
                    valid_probs = valid_probs ** (1.0 / temp)
                    valid_probs /= valid_probs.sum()
                    try: action_idx = np.random.choice(len(legal_moves), p=valid_probs)
                    except: action_idx = np.random.choice(len(legal_moves))
                
                action = legal_moves[action_idx]
                
                # Apply
                prev_dice = state.current_dice_roll
                states[batch_idx] = ludo_cpp.apply_move(state, action)
                move_counts[batch_idx] += 1
                
                if self.visualize and visualizer:
                    visualizer.broadcast_identities(game_identities[batch_idx], game_id=batch_idx)
                    ghost_game_ids = [idx for idx, g_id in enumerate(game_identities) if 'Ghost' in g_id]
                    visualizer.broadcast_ghost_games(ghost_game_ids)
                    heuristic_game_ids = [idx for idx, g_id in enumerate(game_identities) if 'Heuristic' in g_id]
                    visualizer.broadcast_heuristic_games(heuristic_game_ids)
                    visualizer.broadcast_move(current_p, action, prev_dice, game_id=batch_idx)
                    visualizer.broadcast_state(states[batch_idx], game_id=batch_idx)

        # End While
        batch_duration = time.time() - batch_start_time
        return all_examples, results, batch_duration
