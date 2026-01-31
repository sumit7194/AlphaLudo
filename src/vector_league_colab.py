
import numpy as np
import torch
import ludo_cpp
from tensor_utils_mastery import state_to_tensor_mastery, get_board_coords, BOARD_SIZE
from vector_mcts import VectorMCTSMastery, get_action_probs_vector
from model_mastery import AlphaLudoTopNet
from training_utils import get_temperature, EloTracker
try:
    from src.visualizer import visualizer
except ImportError:
    visualizer = None




class VectorLeagueColabWorker:
    def __init__(self, main_model, probabilities, mcts_simulations=50, visualize=False, 
                 reward_config=None, ghost_pool=None, elo_tracker=None, temp_schedule='alphazero'):
        self.main_model = main_model
        # ... (rest is same) ...
        
# Actually, the file content is already VectorLeagueWorker. I will rename it and add the logic.

        self.main_model = main_model
        self.mcts_simulations = mcts_simulations
        self.visualize = visualize
        self.reward_config = reward_config or {}
        self.ghost_pool = ghost_pool or []
        self.ghost_model = None
        self.ghost_fraction = 0.25  # 25% of games have a ghost opponent
        self.elo_tracker = elo_tracker  # For smart ghost selection and tracking
        self.temp_schedule = temp_schedule  # Temperature schedule: 'constant', 'alphazero', 'linear'
        self.current_ghost_name = None  # Track which ghost is loaded
        
        # Load a ghost model if pool exists
        if self.ghost_pool:
            self._load_ghost()
        
        print(f"VectorLeagueColabWorker initialized (optimized for Colab).")
        print(f"  Batch MCTS Sims: {mcts_simulations}. Ghosts: {len(self.ghost_pool)}. Temp: {temp_schedule}")
        
        # Check device
        try:
            device = next(self.main_model.parameters()).device
            print(f"  Model Device: {device}")
            if device.type == 'cuda':
                print(f"  CUDA Device: {torch.cuda.get_device_name(0)}")
            elif device.type == 'cpu':
                print("  WARNING: Running on CPU! Training will be slow.")
        except:
             pass

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
        reward = 0.0
        if self.reward_config.get('cut', 0.0) > 0.0:
            prev_pos = prev_state.player_positions
            curr_pos = next_state.player_positions
            cut_occurred = False
            for opp in range(4):
                if opp == player: continue
                for t in range(4):
                    was_on_board = (prev_pos[opp][t] != -1 and prev_pos[opp][t] != 99)
                    is_at_base = (curr_pos[opp][t] == -1)
                    if was_on_board and is_at_base:
                        cut_occurred = True
            if cut_occurred:
                reward += self.reward_config['cut']

        if self.reward_config.get('home', 0.0) > 0.0:
             prev_my = prev_state.player_positions[player]
             curr_my = next_state.player_positions[player]
             for t in range(4):
                 if prev_my[t] != 99 and curr_my[t] == 99:
                     reward += self.reward_config['home']

        if self.reward_config.get('safe', 0.0) > 0.0:
            SAFE_REL_POS = {0, 8, 13, 21, 26, 34, 39, 47}
            curr_my = next_state.player_positions[player]
            count = sum(1 for p in curr_my if p in SAFE_REL_POS)
            reward += count * self.reward_config['safe']
            
        return reward

    def play_batch(self, batch_size=32, temperature=1.0):
        """
        Play a batch of games using C++ MCTS with ghost support.
        Multi-ghost: Games 0-3 self-play, Games 4-5 ghost A, Games 6-7 ghost B.
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
        self.ghost_model = ghost_a_model  # Primary ghost
        self.current_ghost_name = ghost_a_name
        
        # Initialize C++ MCTS Engine
        if not hasattr(self, 'cpp_mcts') or self.cpp_mcts_batch_size != batch_size:
            self.cpp_mcts = ludo_cpp.MCTSEngine(batch_size)
            self.cpp_mcts_batch_size = batch_size

        # Game layout: 0-3 self-play, 4-5 ghost A, 6-7 ghost B (for 8-game batch)
        # General: 50% self-play, 25% ghost A, 25% ghost B
        half = batch_size // 2
        quarter = batch_size // 4
        
        ghost_a_games = set(range(half, half + quarter)) if ghost_a_model else set()
        ghost_b_games = set(range(half + quarter, batch_size)) if ghost_b_model else set()
        ghost_game_indices = ghost_a_games | ghost_b_games

        # Game State
        states = [ludo_cpp.create_initial_state() for _ in range(batch_size)]
        histories = [[] for _ in range(batch_size)]
        shaped_returns = [{0:0.0, 1:0.0, 2:0.0, 3:0.0} for _ in range(batch_size)]
        move_counts = [0] * batch_size
        
        # Per-game: store ghost model and name per game
        game_ghost_models = {}
        game_identities = []
        
        for idx in range(batch_size):
            if idx in ghost_a_games:
                # Ghost A in random position
                ghost_pos = np.random.randint(0, 4)
                ids = ['Main', 'Main', 'Main', 'Main']
                ids[ghost_pos] = ghost_a_name or 'Ghost_A'
                game_identities.append(ids)
                game_ghost_models[idx] = (ghost_a_model, ghost_pos, ghost_a_name)
            elif idx in ghost_b_games:
                # Ghost B in random position  
                ghost_pos = np.random.randint(0, 4)
                ids = ['Main', 'Main', 'Main', 'Main']
                ids[ghost_pos] = ghost_b_name or 'Ghost_B'
                game_identities.append(ids)
                game_ghost_models[idx] = (ghost_b_model, ghost_pos, ghost_b_name)
            else:
                game_identities.append(['Main', 'Main', 'Main', 'Main'])
                game_ghost_models[idx] = None
        
        # Collection Buffer
        completed_episodes = 0
        target_episodes = batch_size
        all_examples = []
        results = []
        
        # Broadcast to UI (order matters: batch_init first, then ghost_games)
        if self.visualize and visualizer:
            visualizer.broadcast_batch_init(batch_size)
            if ghost_game_indices:
                print(f"  Broadcasting ghost games: {list(ghost_game_indices)}")
                visualizer.broadcast_ghost_games(list(ghost_game_indices))

        active_indices = list(range(batch_size))
        shaped_returns = np.zeros((batch_size, 4))
        
        move_step = 0
        while len(active_indices) > 0:
            move_step += 1

            
            # Re-evaluate active games
            current_active_indices = []
            for idx in active_indices:
                state = states[idx]
                is_terminal = state.is_terminal or move_counts[idx] >= 1000
                
                if is_terminal:
                    winner = ludo_cpp.get_winner(state) if state.is_terminal else -1
                    if self.visualize and visualizer:
                         visualizer.broadcast_game_result(idx, winner)

                    identities = game_identities[idx]
                    
                    if completed_episodes < target_episodes:
                         results.append({'winner': winner, 'identities': identities})
                         
                         for record in histories[idx]:
                             p = record['player']
                             if winner == -1: val = 0.0
                             elif winner == p: val = 1.0
                             else: val = -1.0
                             val += shaped_returns[idx][p]
                             
                             # Only record training data for Main model decisions
                             if identities[p] == 'Main':
                                 all_examples.append((
                                     record['state'],
                                     torch.tensor(record['token_indices'], dtype=torch.long),
                                     torch.tensor(record['policy'], dtype=torch.float32),
                                     torch.tensor(val, dtype=torch.float32)
                                 ))
                         completed_episodes += 1
                         
                         # Update Elo ratings
                         if self.elo_tracker and winner >= 0:
                             self.elo_tracker.update_from_game(identities, winner)
                         
                         print(f"Episode {completed_episodes}/{target_episodes} finished (Winner: {winner}, Ghost: {idx in ghost_game_indices})")
                    
                    if completed_episodes >= target_episodes:
                        continue
                    
                    states[idx] = ludo_cpp.create_initial_state()
                    histories[idx] = []
                    shaped_returns[idx] = {0:0.0, 1:0.0, 2:0.0, 3:0.0}
                    move_counts[idx] = 0
                    
                if state.current_dice_roll == 0:
                    state.current_dice_roll = np.random.randint(1, 7)
                
                current_dice = state.current_dice_roll
                legal_moves = ludo_cpp.get_legal_moves(state)
                if len(legal_moves) == 0:
                     # Broadcast skipped turn with dice value (token=-1 means no move)
                     if self.visualize and visualizer:
                         visualizer.broadcast_move(state.current_player, -1, current_dice, game_id=idx)
                     state.current_player = (state.current_player + 1) % 4
                     state.current_dice_roll = 0
                     if self.visualize and visualizer:
                         visualizer.broadcast_state(state, game_id=idx)
                else:
                    active_indices.append(idx)
            
            if not active_indices:
                if completed_episodes >= target_episodes:
                   break
                continue
            
            # OPTIMIZATION 1: Pre-check for single-move situations
            # For games with only 1 legal move, skip MCTS entirely and process immediately
            single_move_games = {}  # idx -> (action, legal_moves)
            multi_move_states = []  # States that need MCTS
            multi_move_idx_map = []  # Maps MCTS index -> original batch index
            
            for idx in active_indices:
                state = states[idx]
                legal_moves = ludo_cpp.get_legal_moves(state)
                if len(legal_moves) == 1:
                    # Fast path: no search needed for forced moves
                    single_move_games[idx] = (legal_moves[0], legal_moves)
                elif len(legal_moves) > 1:
                    multi_move_states.append(state)
                    multi_move_idx_map.append(idx)
            
            # Process single-move games IMMEDIATELY (before MCTS)
            for idx, (action, legal_moves) in single_move_games.items():
                state = states[idx]
                current_p = state.current_player
                identities = game_identities[idx]
                
                state_tensor = state_to_tensor_mastery(state)
                positions = state.player_positions[current_p]
                token_indices = []
                for t in range(4):
                    r, c = get_board_coords(current_p, positions[t], t)
                    token_indices.append(r * BOARD_SIZE + c)
                
                # Policy for training: 100% on the only legal move
                probs = np.zeros(4)
                probs[action] = 1.0
                
                histories[idx].append({
                    'state': state_tensor,
                    'token_indices': token_indices,
                    'policy': probs,
                    'player': current_p
                })
                
                # Apply the forced move
                prev_state_copy = ludo_cpp.GameState()
                prev_state_copy.player_positions = [list(state.player_positions[i]) for i in range(4)]
                prev_state_copy.scores = list(state.scores)
                prev_state_copy.current_player = state.current_player
                prev_state_copy.current_dice_roll = state.current_dice_roll
                prev_state_copy.is_terminal = state.is_terminal
                
                states[idx] = ludo_cpp.apply_move(state, action)
                
                reward = self.get_reward_shaping(prev_state_copy, action, states[idx], current_p)
                shaped_returns[idx][current_p] += reward
                move_counts[idx] += 1
                
                # Broadcast to visualizer
                if self.visualize and visualizer:
                    visualizer.broadcast_move(current_p, action, prev_state_copy.current_dice_roll, game_id=idx)
                    visualizer.broadcast_state(states[idx], game_id=idx)
            
            # Skip MCTS entirely if no multi-move games
            if not multi_move_states:
                continue
            
            # Print Progress (Colab Optimization)
            if move_step % 10 == 0:
                print(f"  Step {move_step}: {len(active_indices)} games active", end='\r')

            # MCTS for multi-move games only
            self.cpp_mcts = ludo_cpp.MCTSEngine(len(multi_move_states))  # Resize for subset
            self.cpp_mcts.set_roots(multi_move_states)
            
            # Get early termination threshold from visualizer if available
            early_term_threshold = 0.80
            if self.visualize and visualizer and hasattr(visualizer, 'mcts_settings'):
                early_term_threshold = visualizer.mcts_settings.get('early_termination_threshold', 0.80)
            
            min_sims = 20  # Minimum simulations before checking early termination
            
            for sim_num in range(self.mcts_simulations):
                self.cpp_mcts.select_leaves()
                
                # Get leaf tensors
                leaf_tensors_np = self.cpp_mcts.get_leaf_tensors()
                if leaf_tensors_np.size == 0:
                     break
                
                device = next(self.main_model.parameters()).device
                input_tensor = torch.from_numpy(leaf_tensors_np).to(device)
                
                with torch.no_grad():
                    pi_logits, v = self.main_model(input_tensor)
                
                pi_np = pi_logits.cpu().numpy()
                v_np = v.cpu().numpy().flatten()
                
                self.cpp_mcts.expand_and_backprop(pi_np, v_np)
                
                # OPTIMIZATION 2: Early termination check
                if sim_num >= min_sims and sim_num % 10 == 0:
                    try:
                        visit_probs = self.cpp_mcts.get_action_probs(0.01)
                        max_prob = np.max(visit_probs)
                        if max_prob >= early_term_threshold:
                            break
                    except:
                        pass
            
            # Get MCTS decisions for multi-move games
            # Get MCTS decisions for multi-move games
            mcts_probs = self.cpp_mcts.get_action_probs(temperature)
            
            # Process multi-move games using MCTS results
            for mcts_idx, batch_idx in enumerate(multi_move_idx_map):
                state = states[batch_idx]
                probs = mcts_probs[mcts_idx]
                
                current_p = state.current_player
                identities = game_identities[batch_idx]
                
                state_tensor = state_to_tensor_mastery(state)
                
                positions = state.player_positions[current_p]
                token_indices = []
                for t in range(4):
                    r, c = get_board_coords(current_p, positions[t], t)
                    token_indices.append(r * BOARD_SIZE + c)
                
                histories[batch_idx].append({
                    'state': state_tensor,
                    'token_indices': token_indices,
                    'policy': probs,
                    'player': current_p
                })
                
                legal_moves = ludo_cpp.get_legal_moves(state)
                if len(legal_moves) == 0:
                    continue
                
                # Get temperature based on move count
                temp = get_temperature(move_counts[batch_idx], self.temp_schedule)
                    
                # Apply temperature to probabilities
                valid_probs = np.array([probs[m] for m in legal_moves])
                valid_probs = np.clip(valid_probs, 1e-8, None)
                
                if temp < 0.01:
                    # Greedy selection (very low temperature)
                    action_idx = np.argmax(valid_probs)
                else:
                    # Temperature-scaled sampling
                    valid_probs = valid_probs ** (1.0 / temp)
                    valid_probs /= valid_probs.sum()
                    try:
                        action_idx = np.random.choice(len(legal_moves), p=valid_probs)
                    except ValueError:
                        action_idx = np.random.choice(len(legal_moves))
                action = legal_moves[action_idx]
                
                prev_state_copy = ludo_cpp.GameState()
                prev_state_copy.player_positions = [list(state.player_positions[i]) for i in range(4)]
                prev_state_copy.scores = list(state.scores)
                prev_state_copy.current_player = state.current_player
                prev_state_copy.current_dice_roll = state.current_dice_roll
                prev_state_copy.is_terminal = state.is_terminal
                
                states[batch_idx] = ludo_cpp.apply_move(state, action)
                
                reward = self.get_reward_shaping(prev_state_copy, action, states[batch_idx], current_p)
                shaped_returns[batch_idx][current_p] += reward
                
                move_counts[batch_idx] += 1
                
                if self.visualize and visualizer:
                    visualizer.broadcast_identities(game_identities[batch_idx], game_id=batch_idx)
                    visualizer.broadcast_move(current_p, action, state.current_dice_roll, game_id=batch_idx)
                    visualizer.broadcast_state(states[batch_idx], game_id=batch_idx)

        return all_examples, results
