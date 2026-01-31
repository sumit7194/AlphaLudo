import random
import ludo_cpp
from src.tensor_utils import get_board_coords

# --- WEIGHTS CONFIGURATION (The "Personality") ---
W_WIN_GAME = 50000.0       # Immediate win
W_FINISH_TOKEN = 10000.0   # Getting a token home
W_CUT_OPPONENT = 6000.0    # Sending enemy to base
W_EXIT_BASE = 1500.0       # Rolling 6 to get out
W_ENTER_SAFE = 800.0       # Moving to a Globe/Star
W_ENTER_HOME_RUN = 500.0   # Entering the colored safe corridor
W_PROGRESS = 10.0          # Generic step forward
W_STACK = 50.0             # Small reward for stacking (new feature alignment)

# Penalties
W_DANGER_PENALTY = -2000.0 # Landing 1-6 steps ahead of an enemy
W_LEAVE_SAFE_PENALTY = -500.0 # Leaving a safe zone unnecessarily

class HeuristicLudoBot:
    def __init__(self, player_id=None):
        """
        player_id: Optional. If None, it infers from state.current_player.
        """
        self.player_id = player_id
        # Absolute indices of safe globes (0-51 scale)
        self.safe_indices = {0, 8, 13, 21, 26, 34, 39, 47} 

    def select_move(self, state, legal_moves):
        """
        Returns the index of the best token to move.
        """
        if not legal_moves:
            return -1
        
        if len(legal_moves) == 1:
            return legal_moves[0] # Forced

        # --- ADAPTIVE PERSONALITY ---
        # Adjust weights based on game state
        p = self.player_id if self.player_id is not None else state.current_player
        
        # Calculate score difference relative to leader
        my_score = state.scores[p]
        max_opp_score = max([state.scores[i] for i in range(4) if i != p])
        
        # Base Configuration copy
        w_cut = W_CUT_OPPONENT
        w_danger = W_DANGER_PENALTY
        
        if my_score > max_opp_score:
            # We are winning: Play Defensive
            w_danger *= 1.5 # Fear getting cut more
            w_cut *= 0.8    # Less need to stick neck out for kills
        elif my_score < max_opp_score:
            # We are losing: Play Aggressive
            w_cut *= 1.5    # Need to catch up
            w_danger *= 0.8 # Risk it
            
        best_score = -float('inf')
        best_move = random.choice(legal_moves)

        for token_idx in legal_moves:
            # Create a virtual copy of the state
            try:
                # Note: We rely on ludo_cpp bindings returning a copy for apply_move
                next_state = ludo_cpp.apply_move(state, token_idx)
            except Exception as e:
                print(f"Heuristic Error: {e}")
                continue
            
            # Pass dynamic weights
            score = self.evaluate_state(state, next_state, token_idx, p, w_cut, w_danger)
            
            score += random.uniform(0, 5) # Noise

            if score > best_score:
                best_score = score
                best_move = token_idx
        
        return best_move

    def evaluate_state(self, prev_state, current_state, moved_token_idx, player, w_cut, w_danger):
        """
        Compares previous state vs new state.
        """
        score = 0.0
        p = player
        
        # --- 1. DETECT IMMEDIATE EVENTS ---
        
        # Win
        if current_state.scores[p] == 4:
            return W_WIN_GAME
        
        # Score Point
        if current_state.scores[p] > prev_state.scores[p]:
            score += W_FINISH_TOKEN
            
        # Cuts (Using dynamic weight)
        cuts = 0
        for op in range(4):
            if op == p: continue
            prev_base = sum(1 for t in range(4) if prev_state.player_positions[op][t] == -1)
            curr_base = sum(1 for t in range(4) if current_state.player_positions[op][t] == -1)
            if curr_base > prev_base:
                cuts += (curr_base - prev_base)
        score += (cuts * w_cut)

        # --- 2. EVALUATE NEW POSITION ---
        
        pos = current_state.player_positions[p][moved_token_idx]
        prev_pos = prev_state.player_positions[p][moved_token_idx]
        
        # Exit Base
        if prev_pos == -1 and pos != -1:
            score += W_EXIT_BASE
            
        # Check Stack Status (User Rule: Stack = Safe)
        is_stack_now = self.is_stack(current_state, p, pos)
        was_stack_prev = self.is_stack(prev_state, p, prev_pos)
        
        # Safety Analysis
        is_safe_now = self.is_safe(p, pos) or is_stack_now
        is_safe_prev = self.is_safe(p, prev_pos) or was_stack_prev
        
        # Enter Safe Zone
        if is_safe_now:
            score += W_ENTER_SAFE
            if not is_safe_prev and prev_pos != -1:
                 score += 200 # Extra bonus for finding safety
        
        # Leave Safe Penalty
        # Only penalty if we leave safety for an UNSAFE spot
        # And we didn't score or cut
        if is_safe_prev and not is_safe_now and pos != 99:
            if cuts == 0 and current_state.scores[p] == prev_state.scores[p]:
                score += W_LEAVE_SAFE_PENALTY

        # Enter Home Run
        if prev_pos <= 50 and pos > 50 and pos != 99:
            score += W_ENTER_HOME_RUN

        # Progress
        if pos != -1 and prev_pos != -1 and pos != 99:
             score += (pos - prev_pos) * W_PROGRESS
             
        # Stack Reward
        if is_stack_now:
             score += W_STACK * 2 # Good reward

        # --- 3. DANGER ASSESSMENT ---
        if pos != 99 and pos <= 50 and not is_safe_now: # Ignore danger if safe/home
            my_abs_pos = self.get_abs_pos(p, pos)
            
            risk_factor = 0.0
            for op in range(4):
                if op == p: continue
                for t in range(4):
                    op_pos = current_state.player_positions[op][t]
                    if op_pos != -1 and op_pos != 99 and op_pos <= 50:
                        op_abs = self.get_abs_pos(op, op_pos)
                        
                        # Check if Opponent is 1-6 steps BEHIND me
                        # Distance = (My - Op) % 52
                        distance = (my_abs_pos - op_abs) % 52
                        
                        if 1 <= distance <= 6:
                            risk_factor += 1.0
                            
            score += (risk_factor * w_danger)

        return score
    
    def is_stack(self, state, player, pos):
        """Returns True if there is more than 1 token of player at pos."""
        if pos == -1 or pos == 99: return False
        count = 0
        for t in range(4):
            if state.player_positions[player][t] == pos:
                count += 1
        return count >= 2

    def get_abs_pos(self, player, rel_pos):
        """Converts relative position (0-50) to absolute board index (0-51)."""
        if rel_pos == -1 or rel_pos > 50: return -1
        # P0=0, P1=13, P2=26, P3=39
        return (rel_pos + (player * 13)) % 52

    def is_safe(self, player, rel_pos):
        """Checks if a position is a Globe or Star."""
        if rel_pos == -1: return True 
        if rel_pos > 50: return True 
        if rel_pos == 99: return True 
        
        abs_pos = self.get_abs_pos(player, rel_pos)
        return abs_pos in self.safe_indices


# --- Bot Variants for Multi-Heuristic Training ---

class AggressiveBot(HeuristicLudoBot):
    """
    Aggressive variant that prioritizes cutting opponents.
    Takes more risks, less concerned about safety.
    """
    def __init__(self, player_id=None):
        super().__init__(player_id)
        # Override weights for aggressive play
        self.w_cut_bonus = 2.0      # 2x cut reward
        self.w_danger_mult = 0.3    # Less fear of danger
        self.w_safe_mult = 0.5      # Less value on safety
    
    def evaluate_state(self, prev_state, current_state, moved_token_idx, player, w_cut, w_danger):
        # Apply aggressive multipliers
        w_cut = w_cut * self.w_cut_bonus
        w_danger = w_danger * self.w_danger_mult
        return super().evaluate_state(prev_state, current_state, moved_token_idx, player, w_cut, w_danger)


class DefensiveBot(HeuristicLudoBot):
    """
    Defensive variant that prioritizes safety and stacking.
    Avoids danger, prefers safe zones.
    """
    def __init__(self, player_id=None):
        super().__init__(player_id)
        # Override weights for defensive play
        self.w_danger_mult = 2.5    # Very fearful
        self.w_safe_bonus = 2.0     # More value on safety
        self.w_stack_bonus = 3.0    # Loves stacking
    
    def evaluate_state(self, prev_state, current_state, moved_token_idx, player, w_cut, w_danger):
        w_danger = w_danger * self.w_danger_mult
        score = super().evaluate_state(prev_state, current_state, moved_token_idx, player, w_cut, w_danger)
        
        # Extra bonus for safe positions
        pos = current_state.player_positions[player][moved_token_idx]
        if self.is_safe(player, pos):
            score += W_ENTER_SAFE * (self.w_safe_bonus - 1)  # Additional bonus
        
        # Extra bonus for stacking
        if self.is_stack(current_state, player, pos):
            score += W_STACK * (self.w_stack_bonus - 1)
        
        return score


class RacingBot(HeuristicLudoBot):
    """
    Racing variant that rushes tokens home.
    Ignores opponents, focuses on progress.
    """
    def __init__(self, player_id=None):
        super().__init__(player_id)
        # Override weights for racing play
        self.w_progress_bonus = 3.0    # 3x progress reward
        self.w_home_bonus = 2.0        # 2x finish token reward
        self.w_cut_mult = 0.2          # Mostly ignores cuts
    
    def evaluate_state(self, prev_state, current_state, moved_token_idx, player, w_cut, w_danger):
        # Reduce cut importance
        w_cut = w_cut * self.w_cut_mult
        score = super().evaluate_state(prev_state, current_state, moved_token_idx, player, w_cut, w_danger)
        
        # Extra progress bonus
        pos = current_state.player_positions[player][moved_token_idx]
        prev_pos = prev_state.player_positions[player][moved_token_idx]
        
        if pos != -1 and prev_pos != -1 and pos != 99:
            score += (pos - prev_pos) * W_PROGRESS * (self.w_progress_bonus - 1)
        
        # Extra finish bonus
        if current_state.scores[player] > prev_state.scores[player]:
            score += W_FINISH_TOKEN * (self.w_home_bonus - 1)
        
        return score


# Bot factory for easy instantiation
def get_bot(bot_type, player_id=None):
    """Factory function to create bot by type name."""
    bots = {
        'Heuristic': HeuristicLudoBot,
        'Aggressive': AggressiveBot,
        'Defensive': DefensiveBot,
        'Racing': RacingBot,
    }
    bot_class = bots.get(bot_type, HeuristicLudoBot)
    return bot_class(player_id=player_id)

