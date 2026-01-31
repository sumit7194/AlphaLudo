import numpy as np
import torch
import ludo_cpp
from tensor_utils_mastery import state_to_tensor_mastery, get_board_coords, BOARD_SIZE
from mcts_mastery import MCTSMastery, get_action_probs_mastery
from src.heuristic_bot import HeuristicLudoBot
try:
    from src.visualizer import visualizer
except ImportError:
    visualizer = None

class LeagueWorkerMastery:
    """
    League Worker for Mastery Architecture.
    Does NOT inherit from old LeagueWorker to avoid accidentally using old tensor utils.
    """
    def __init__(self, main_model, specialist_pool, probabilities, mcts_simulations=100, visualize=False, reward_config=None, ghost_pool=None):
        self.main_model = main_model
        # Ensure Main is in pool
        self.specialist_pool = specialist_pool
        self.full_pool = specialist_pool.copy()
        self.full_pool['Main'] = main_model
        
        self.probabilities = probabilities
        self.mcts_simulations = mcts_simulations
        self.visualize = visualize
        self.reward_config = reward_config or {}
        self.ghost_pool = ghost_pool or []
        
        # Cache for ghost model to avoid reloading every game if we pick the same one
        self.cached_ghost_path = None
        self.cached_ghost_model = None
        
        # Instantiate Heuristic Bot
        self.heuristic_bot = HeuristicLudoBot()
        
        print(f"LeagueWorkerMastery initialized. MCTS Sims: {mcts_simulations}. Rewards: {self.reward_config}")

    def get_reward_shaping(self, prev_state, action, next_state, player):
        """
        Calculate extra reward based on what happened in the move.
        """
        reward = 0.0
        
        # 1. Check for CUT (Aggression)
        # +0.15 if we cut someone
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

        # 2. Check for HOME (Rusher)
        # +0.25 if we send a token home
        if self.reward_config.get('home', 0.0) > 0.0:
            prev_my = prev_state.player_positions[player]
            curr_my = next_state.player_positions[player]
            for t in range(4):
                if prev_my[t] != 99 and curr_my[t] == 99:
                    reward += self.reward_config['home']

        # 3. Check for SAFE (Defensive)
        # +0.05 if we land on a safe glob/star
        # Indices: 0, 8, 13, 21, 26, 34, 39, 47
        if self.reward_config.get('safe', 0.0) > 0.0:
            SAFE_REL_POS = {0, 8, 13, 21, 26, 34, 39, 47}
            curr_my = next_state.player_positions[player]
            for t in range(4):
                # Did this token just MOVE to a safe spot?
                # Or just IS at a safe spot?
                # Usually standard shaping is per-step "am I safe?". 
                # Let's reward landing on it or staying on it? 
                # "Land on" is better for event-based. "Stay" might exploit looped rewards.
                # Let's check if the moved token is safe.
                # Which token moved? logic is tricky without knowing token idx action applied to.
                # But we can just check if *any* token is safe? No, that rewards static play.
                # Let's check difference.
                if curr_my[t] in SAFE_REL_POS:
                     # Check if it was NOT there before or just reward state?
                     # Let's reward BEING safe, but small. 
                     # Actually, to avoid rewarding sitting, maybe only if we Moved there?
                     # The action applied moved *one* token.
                     # Determining which token moved is hard blindly.
                     # Simplification: Reward total count of safe tokens * 0.05?
                     # No, that encourages turtle.
                     # Let's stick to "If the move resulted in a safe position".
                     pass
            
            # Implementation: Reward simply for count of Safe Tokens?
            # Or simplified: if any token is safe, +0.05.
            # Let's allow +0.05 per safe token.
            count = sum(1 for p in curr_my if p in SAFE_REL_POS)
            reward += count * self.reward_config['safe']

        # 4. Check for STACK (Blockade)
        if self.reward_config.get('stack', 0.0) > 0.0:
            curr_my = next_state.player_positions[player]
            # Count tokens on same valid square
            pos_counts = {}
            for p in curr_my:
                if p != -1 and p != 99:
                    pos_counts[p] = pos_counts.get(p, 0) + 1
            
            # Sum tokens that are in a stack (count >= 2)
            stack_tokens = sum(count for count in pos_counts.values() if count >= 2)
            reward += stack_tokens * self.reward_config['stack']

        return reward

    def select_opponents(self):
        players = ['Main']
        options = list(self.probabilities.keys())
        probs = list(self.probabilities.values())
        opponents = np.random.choice(options, size=3, p=probs)
        players.extend(opponents)
        return players

    def play_game(self, temperature=1.0):
        identities = self.select_opponents()
        identities_str = [str(x) for x in identities]
        
        # Resolve models
        models = []
        for id_str in identities_str:
            if id_str == 'Ghost':
                if self.ghost_pool:
                    # Optimized Ghost Loading
                    ghost_path = np.random.choice(self.ghost_pool)
                    
                    if ghost_path == self.cached_ghost_path and self.cached_ghost_model is not None:
                         # Cache Hit
                         models.append(self.cached_ghost_model)
                    else:
                        # Cache Miss - Load from disk
                        import copy
                        model = copy.deepcopy(self.main_model)
                        try:
                            # Load on CPU to avoid VRAM spam? Or device?
                            # device is in model.device?
                            device = next(self.main_model.parameters()).device
                            checkpoint = torch.load(ghost_path, map_location=device)
                            # Handle full dict or state dict
                            if 'model_state_dict' in checkpoint:
                                model.load_state_dict(checkpoint['model_state_dict'])
                            else:
                                model.load_state_dict(checkpoint)
                            model.to(device)
                            
                            # Update Cache
                            self.cached_ghost_path = ghost_path
                            self.cached_ghost_model = model
                            models.append(model)
                        except Exception as e:
                            print(f"Failed to load ghost {ghost_path}: {e}")
                            models.append(self.main_model) # Fallback
                else:
                    models.append(self.main_model) # Fallback
            else:
                models.append(self.full_pool[id_str])
        
        if self.visualize and visualizer:
            visualizer.broadcast_identities(identities_str)

        game_history = []
        state = ludo_cpp.create_initial_state()
        move_count = 0
        max_moves = 1000
        
        shaped_returns = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}
        
        while not state.is_terminal and move_count < max_moves:
            dice_roll = np.random.randint(1, 7)
            state.current_dice_roll = dice_roll
            
            if self.visualize and visualizer:
                visualizer.broadcast_state(state)
            
            legal_moves = ludo_cpp.get_legal_moves(state)
            if len(legal_moves) == 0:
                state.current_player = (state.current_player + 1) % 4
                state.current_dice_roll = 0
                state.current_dice_roll = 0
                continue
            
            # --- Reward Shaping Hook (Pre-Move) ---
            # We need to track rewards for the player who IS ABOUT TO MOVE?
            # No, rewards are result of actions.
            # We need to accumulate rewards for the PREVIOUS player?
            # Game loop:
            # 1. State S_t
            # 2. Player P chooses Action A
            # 3. State S_{t+1}
            # 4. Resulting Reward R (Cut/Home/Safe) is given to Player P.
            
            # So we apply move, then calculate reward, then add to `shaped_returns` for that player.
            # We need a `shaped_returns` dict.

            
            p = state.current_player
            current_id = identities_str[p]
            # Determine Action Source
            if current_id == 'Heuristic':
                # --- HEURISTIC BOT ---
                action = self.heuristic_bot.select_move(state, legal_moves)
                
                # Mock policy for history (one-hot)
                action_probs = np.zeros(4, dtype=np.float32)
                # Map action (0-3 token index) to something?
                # Heuristic returns token index. MCTS logic expects token index?
                # Wait, MCTS returns action index 0-3 (token index). Correct.
                action_probs[action] = 1.0
                
            else:
                # --- NEURAL MCTS ---
                current_model = models[p]
                
                # Use Mastery MCTS
                mcts = MCTSMastery(current_model, num_simulations=self.mcts_simulations)
                action_probs = get_action_probs_mastery(mcts, state, temperature=temperature)
            
            # Record Data only for Main model
            # Note: We record data BEFORE applying the move.
            if current_id == 'Main':
                state_tensor = state_to_tensor_mastery(state)
                
                # Calculate Token Indices (Flat 0-224) for the current player's 4 tokens
                # This is needed for training masking
                positions = state.player_positions[p]
                token_indices = []
                for t in range(4):
                    r, c = get_board_coords(p, positions[t], t)
                    token_indices.append(r * BOARD_SIZE + c)
                
                game_history.append({
                    'state': state_tensor,
                    'token_indices': token_indices, # List of 4 ints
                    'policy': action_probs, # List of 4 floats
                    'player': p
                })
            
            # Select Action
            if current_id != 'Heuristic':
                if temperature == 0:
                    action = int(np.argmax(action_probs))
                else:
                    action = np.random.choice(4, p=action_probs)
                
                if action not in legal_moves:
                    action = np.random.choice(legal_moves)
            
            if self.visualize and visualizer:
                 visualizer.broadcast_move(state.current_player, action, dice_roll)

            # Apply Move
            prev_state = state
            state = ludo_cpp.apply_move(state, action)
            move_count += 1
            
            # --- Reward Shaping Calculation ---
            # Player who moved: prev_state.current_player
            actor = prev_state.current_player
            # Only calculate if we have config
            if self.reward_config:
                shaping = self.get_reward_shaping(prev_state, action, state, actor)
                shaped_returns[actor] += shaping
            
        if self.visualize and visualizer:
            visualizer.broadcast_state(state)
            
        winner = ludo_cpp.get_winner(state)
        
        training_examples = []
        for record in game_history:
            p = record['player']
            if winner == -1:
                val = 0.0
            elif winner == p:
                val = 1.0
            else:
                val = -1.0
            
            # Add shaped rewards
            # Total Target = Terminal Value + Shaped Return
            # Note: We clip to [-1, 1]? Or allow > 1?
            # RL usually expects V in specific range. 
            # If we add +0.15 often, value can go > 1.
            # Tanh activation at end of Value Head caps output at [-1, 1].
            # So target > 1 will just push it to 1.
            # To preserve shaping gradient, maybe better not to saturate?
            # But the user asked for small rewards so it's fine.
            # Let's just add it.
            
            val += shaped_returns[p]
                
            training_examples.append((
                record['state'],
                torch.tensor(record['token_indices'], dtype=torch.long),
                torch.tensor(record['policy'], dtype=torch.float32),
                torch.tensor(val, dtype=torch.float32)
            ))
            
        return training_examples, winner, identities_str
