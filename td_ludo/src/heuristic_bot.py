import random
import td_ludo_cpp as ludo_cpp
try:
    from src.tensor_utils import get_board_coords
except ImportError:
    from tensor_utils import get_board_coords


# ============================================================================
# WEIGHTS CONFIGURATION (The "Personality")
# ============================================================================

# Positive rewards
W_WIN_GAME      = 50000.0   # Immediate win
W_FINISH_TOKEN  = 10000.0   # Getting a token home
W_CUT_OPPONENT  = 6000.0    # Sending enemy to base
W_EXIT_BASE     = 1500.0    # Rolling 6 to get out
W_ENTER_SAFE    = 800.0     # Moving to a Globe/Star
W_ENTER_HOME_RUN = 500.0    # Entering the colored safe corridor
W_PROGRESS      = 10.0      # Generic step forward
W_STACK         = 150.0     # Stacking tokens (safe + strategic)
W_BREAK_STACK   = 400.0     # Cutting an opponent stack (sends 2+ back)
W_STAR_TELEPORT = 350.0     # Landing on a star (teleportation value)

# Penalties
W_DANGER_PENALTY     = -2000.0  # Landing 1-6 steps ahead of an enemy
W_LEAVE_SAFE_PENALTY = -500.0   # Leaving a safe zone unnecessarily
W_LEAVE_STACK_PENALTY = -200.0  # Breaking your own stack


# ============================================================================
# Base Heuristic Bot (Balanced play)
# ============================================================================

class HeuristicLudoBot:
    """
    Strong heuristic Ludo bot with adaptive personality.
    
    Evaluates moves by simulating each one and scoring the resulting state
    based on: cuts, safety, progress, danger avoidance, and stacking.
    Adapts strategy based on score differential (defensive when ahead,
    aggressive when behind).
    """
    
    def __init__(self, player_id=None):
        """
        player_id: Optional. If None, uses state.current_player.
        """
        self.player_id = player_id
        # Safe globe positions (absolute 0-51 board indices)
        # Globes: 0, 8, 13, 21, 26, 34, 39, 47
        self.safe_globes = {0, 8, 13, 21, 26, 34, 39, 47}
        # Star positions (absolute) — these are teleport points on the board
        self.star_positions = {5, 18, 31, 44, 11, 24, 37, 50}
        # Combined safe positions
        self.safe_indices = self.safe_globes | self.star_positions

    def select_move(self, state, legal_moves):
        """
        Returns the token index of the best move to make.
        """
        if not legal_moves:
            return -1
        
        if len(legal_moves) == 1:
            return legal_moves[0]

        p = self.player_id if self.player_id is not None else state.current_player
        
        # Adaptive personality based on game state
        my_score = state.scores[p]
        max_opp_score = max(state.scores[i] for i in range(4) if i != p)
        
        w_cut = W_CUT_OPPONENT
        w_danger = W_DANGER_PENALTY
        
        if my_score > max_opp_score:
            # Winning → play defensive
            w_danger *= 1.5
            w_cut *= 0.8
        elif my_score < max_opp_score:
            # Losing → play aggressive
            w_cut *= 1.5
            w_danger *= 0.8
            
        best_score = -float('inf')
        best_move = random.choice(legal_moves)

        for token_idx in legal_moves:
            try:
                next_state = ludo_cpp.apply_move(state, token_idx)
            except Exception:
                continue
            
            score = self._evaluate(state, next_state, token_idx, p, w_cut, w_danger)
            score += random.uniform(0, 5)  # Small noise for variety

            if score > best_score:
                best_score = score
                best_move = token_idx
        
        return best_move

    def _evaluate(self, prev, curr, token_idx, player, w_cut, w_danger):
        """
        Score a (prev_state → curr_state) transition for the given player.
        Higher = better.
        """
        score = 0.0
        p = player
        
        # ---- 1. IMMEDIATE EVENTS ----
        
        # Win
        if curr.scores[p] == 4:
            return W_WIN_GAME
        
        # Score a point (token reached home)
        if curr.scores[p] > prev.scores[p]:
            score += W_FINISH_TOKEN
            
        # Cuts — count how many opponent tokens were sent to base
        cuts = 0
        for op in range(4):
            if op == p:
                continue
            prev_in_base = sum(1 for t in range(4) if prev.player_positions[op][t] == -1)
            curr_in_base = sum(1 for t in range(4) if curr.player_positions[op][t] == -1)
            new_cuts = curr_in_base - prev_in_base
            if new_cuts > 0:
                cuts += new_cuts
                if new_cuts >= 2:
                    score += W_BREAK_STACK  # Bonus for breaking a stack
        score += cuts * w_cut

        # ---- 2. POSITION QUALITY ----
        
        pos = curr.player_positions[p][token_idx]
        prev_pos = prev.player_positions[p][token_idx]
        
        # Exit base
        if prev_pos == -1 and pos != -1:
            score += W_EXIT_BASE
            
        # Stack analysis
        is_stack_now = self._is_stack(curr, p, pos)
        was_stack_prev = self._is_stack(prev, p, prev_pos)
        
        # Safety analysis (safe = globe/star OR stacked)
        is_safe_now = self._is_safe(p, pos) or is_stack_now
        is_safe_prev = self._is_safe(p, prev_pos) or was_stack_prev
        
        # Entered safe zone
        if is_safe_now:
            score += W_ENTER_SAFE
            if not is_safe_prev and prev_pos != -1:
                score += 200  # Extra bonus for transitioning to safety
        
        # Left safe zone (penalty, unless we scored or cut)
        if is_safe_prev and not is_safe_now and pos != 99:
            if cuts == 0 and curr.scores[p] == prev.scores[p]:
                score += W_LEAVE_SAFE_PENALTY
        
        # Breaking own stack (moving a token off a stack)
        if was_stack_prev and not is_stack_now and prev_pos != -1 and pos != 99:
            if cuts == 0 and curr.scores[p] == prev.scores[p]:
                score += W_LEAVE_STACK_PENALTY

        # Entered home run corridor
        if prev_pos != -1 and prev_pos <= 50 and pos > 50 and pos != 99:
            score += W_ENTER_HOME_RUN

        # Star teleport bonus (landing on a star position)
        if pos != -1 and pos <= 50 and pos != 99:
            abs_pos = self._get_abs_pos(p, pos)
            if abs_pos in self.star_positions:
                score += W_STAR_TELEPORT

        # Progress reward  
        if pos != -1 and prev_pos != -1 and pos != 99:
            if pos > prev_pos:
                score += (pos - prev_pos) * W_PROGRESS
            elif prev_pos <= 50 and pos <= 50:
                # Wrapped around the board
                score += ((52 - prev_pos) + pos) * W_PROGRESS
             
        # Stack reward
        if is_stack_now:
            score += W_STACK

        # ---- 3. DANGER ASSESSMENT ----
        if pos != 99 and pos <= 50 and not is_safe_now:
            my_abs = self._get_abs_pos(p, pos)
            
            risk_factor = 0.0
            for op in range(4):
                if op == p:
                    continue
                for t in range(4):
                    op_pos = curr.player_positions[op][t]
                    if op_pos == -1 or op_pos == 99 or op_pos > 50:
                        continue
                    op_abs = self._get_abs_pos(op, op_pos)
                    
                    # Is opponent 1-6 steps behind us? (they can reach us)
                    distance = (my_abs - op_abs) % 52
                    if 1 <= distance <= 6:
                        risk_factor += 1.0
                        # Extra danger if opponent is stacked (lose 2 tokens)
                        if self._is_stack(curr, op, op_pos):
                            risk_factor += 0.5
                            
            score += risk_factor * w_danger

        # ---- 4. LEADER TARGETING (prefer cutting the leader) ----
        if cuts > 0:
            for op in range(4):
                if op == p:
                    continue
                prev_in_base = sum(1 for t in range(4) if prev.player_positions[op][t] == -1)
                curr_in_base = sum(1 for t in range(4) if curr.player_positions[op][t] == -1)
                if curr_in_base > prev_in_base and curr.scores[op] == max(curr.scores[i] for i in range(4) if i != p):
                    score += 1500  # Bonus for cutting the leader

        return score
    
    def _is_stack(self, state, player, pos):
        """Returns True if 2+ tokens of player are at pos."""
        if pos == -1 or pos == 99:
            return False
        count = sum(1 for t in range(4) if state.player_positions[player][t] == pos)
        return count >= 2

    def _get_abs_pos(self, player, rel_pos):
        """Converts relative position (0-50) to absolute board index (0-51)."""
        if rel_pos == -1 or rel_pos > 50:
            return -1
        return (rel_pos + player * 13) % 52

    def _is_safe(self, player, rel_pos):
        """Checks if a position is a Globe, Star, base, or home."""
        if rel_pos == -1 or rel_pos > 50 or rel_pos == 99:
            return True
        abs_pos = self._get_abs_pos(player, rel_pos)
        return abs_pos in self.safe_indices


# ============================================================================
# Bot Variants for Multi-Heuristic Training
# ============================================================================

class AggressiveBot(HeuristicLudoBot):
    """
    Aggressive variant — prioritizes cutting opponents, downplays safety.
    Takes more risks, chases cuts aggressively, less concerned about danger.
    """
    def __init__(self, player_id=None):
        super().__init__(player_id)
        self.w_cut_bonus = 2.0
        self.w_danger_mult = 0.3
        self.w_safe_mult = 0.5
    
    def _evaluate(self, prev, curr, token_idx, player, w_cut, w_danger):
        w_cut *= self.w_cut_bonus
        w_danger *= self.w_danger_mult
        score = super()._evaluate(prev, curr, token_idx, player, w_cut, w_danger)
        
        # Reduce safety bonuses
        pos = curr.player_positions[player][token_idx]
        if self._is_safe(player, pos):
            score -= W_ENTER_SAFE * (1 - self.w_safe_mult)
        
        return score


class DefensiveBot(HeuristicLudoBot):
    """
    Defensive variant — maximizes safety, avoids danger, loves stacking.
    Extremely cautious, prefers safe zones and stacking defensively.
    """
    def __init__(self, player_id=None):
        super().__init__(player_id)
        self.w_danger_mult = 2.5
        self.w_safe_bonus = 2.0
        self.w_stack_bonus = 3.0
    
    def _evaluate(self, prev, curr, token_idx, player, w_cut, w_danger):
        w_danger *= self.w_danger_mult
        score = super()._evaluate(prev, curr, token_idx, player, w_cut, w_danger)
        
        pos = curr.player_positions[player][token_idx]
        if self._is_safe(player, pos):
            score += W_ENTER_SAFE * (self.w_safe_bonus - 1)
        if self._is_stack(curr, player, pos):
            score += W_STACK * (self.w_stack_bonus - 1)
        
        return score


class RacingBot(HeuristicLudoBot):
    """
    Racing variant — rushes tokens home as fast as possible.
    Ignores most combat, focuses exclusively on forward progress.
    """
    def __init__(self, player_id=None):
        super().__init__(player_id)
        self.w_progress_bonus = 3.0
        self.w_home_bonus = 2.0
        self.w_cut_mult = 0.2
    
    def _evaluate(self, prev, curr, token_idx, player, w_cut, w_danger):
        w_cut *= self.w_cut_mult
        score = super()._evaluate(prev, curr, token_idx, player, w_cut, w_danger)
        
        pos = curr.player_positions[player][token_idx]
        prev_pos = prev.player_positions[player][token_idx]
        
        if pos != -1 and prev_pos != -1 and pos != 99:
            if pos > prev_pos:
                score += (pos - prev_pos) * W_PROGRESS * (self.w_progress_bonus - 1)
        
        if curr.scores[player] > prev.scores[player]:
            score += W_FINISH_TOKEN * (self.w_home_bonus - 1)
        
        return score


class RandomBot:
    """
    Pure random move selection. Useful as a baseline opponent.
    """
    def __init__(self, player_id=None):
        self.player_id = player_id
    
    def select_move(self, state, legal_moves):
        if not legal_moves:
            return -1
        return random.choice(legal_moves)


class ExpertBot(HeuristicLudoBot):
    """
    Expert variant — Uses mathematically optimized weights discovered via Genetic Tournament.
    Highly values safety and forward progress. Avoids reckless cutting if it risks a token.
    Wins 58%+ against the baseline balanced HeuristicBot.
    """
    def __init__(self, player_id=None):
        super().__init__(player_id)
        # Optimized weights from tournament
        self.w_cut_opt = 4243.3
        self.w_danger_opt = -698.1
        self.w_progress_opt = 40.1
        self.w_safe_opt = 2917.6
        self.w_stack_opt = 279.9
        
    def _evaluate(self, prev, curr, token_idx, player, w_cut, w_danger):
        # We must re-implement the core loop to inject the exact optimized constants
        # instead of relying on super() which hardcodes global constants.
        
        score = 0.0
        p = player
        
        # 1. IMMEDIATE EVENTS
        if curr.scores[p] == 4: return W_WIN_GAME
        if curr.scores[p] > prev.scores[p]: score += W_FINISH_TOKEN
            
        cuts = 0
        for op in range(4):
            if op == p: continue
            prev_in_base = sum(1 for t in range(4) if prev.player_positions[op][t] == -1)
            curr_in_base = sum(1 for t in range(4) if curr.player_positions[op][t] == -1)
            new_cuts = curr_in_base - prev_in_base
            if new_cuts > 0:
                cuts += new_cuts
                if new_cuts >= 2: score += W_BREAK_STACK
        score += cuts * self.w_cut_opt

        # 2. POSITION QUALITY
        pos = curr.player_positions[p][token_idx]
        prev_pos = prev.player_positions[p][token_idx]
        
        if prev_pos == -1 and pos != -1: score += W_EXIT_BASE
            
        is_stack_now = self._is_stack(curr, p, pos)
        was_stack_prev = self._is_stack(prev, p, prev_pos)
        is_safe_now = self._is_safe(p, pos) or is_stack_now
        is_safe_prev = self._is_safe(p, prev_pos) or was_stack_prev
        
        if is_safe_now:
            score += self.w_safe_opt
            if not is_safe_prev and prev_pos != -1:
                score += 200
        
        if is_safe_prev and not is_safe_now and pos != 99:
            if cuts == 0 and curr.scores[p] == prev.scores[p]:
                score += W_LEAVE_SAFE_PENALTY
        
        if was_stack_prev and not is_stack_now and prev_pos != -1 and pos != 99:
            if cuts == 0 and curr.scores[p] == prev.scores[p]:
                score += W_LEAVE_STACK_PENALTY

        if prev_pos != -1 and prev_pos <= 50 and pos > 50 and pos != 99:
            score += W_ENTER_HOME_RUN

        if pos != -1 and pos <= 50 and pos != 99:
            abs_pos = self._get_abs_pos(p, pos)
            if abs_pos in self.star_positions:
                score += W_STAR_TELEPORT

        if pos != -1 and prev_pos != -1 and pos != 99:
            if pos > prev_pos:
                score += (pos - prev_pos) * self.w_progress_opt
            elif prev_pos <= 50 and pos <= 50:
                score += ((52 - prev_pos) + pos) * self.w_progress_opt
             
        if is_stack_now:
            score += self.w_stack_opt

        # 3. DANGER ASSESSMENT
        if pos != 99 and pos <= 50 and not is_safe_now:
            my_abs = self._get_abs_pos(p, pos)
            risk_factor = 0.0
            for op in range(4):
                if op == p: continue
                for t in range(4):
                    op_pos = curr.player_positions[op][t]
                    if op_pos == -1 or op_pos == 99 or op_pos > 50: continue
                    op_abs = self._get_abs_pos(op, op_pos)
                    distance = (my_abs - op_abs) % 52
                    if 1 <= distance <= 6:
                        risk_factor += 1.0
                        if self._is_stack(curr, op, op_pos):
                            risk_factor += 0.5
            score += risk_factor * self.w_danger_opt

        # 4. LEADER TARGETING
        if cuts > 0:
            for op in range(4):
                if op == p: continue
                prev_in_base = sum(1 for t in range(4) if prev.player_positions[op][t] == -1)
                curr_in_base = sum(1 for t in range(4) if curr.player_positions[op][t] == -1)
                if curr_in_base > prev_in_base and curr.scores[op] == max(curr.scores[i] for i in range(4) if i != p):
                    score += 1500

        return score


# ============================================================================
# Bot Factory
# ============================================================================

BOT_REGISTRY = {
    'Heuristic': HeuristicLudoBot,
    'Aggressive': AggressiveBot,
    'Defensive': DefensiveBot,
    'Racing': RacingBot,
    'Random': RandomBot,
    'Expert': ExpertBot,
}

def get_bot(bot_type, player_id=None):
    """Factory function to create bot by type name."""
    bot_class = BOT_REGISTRY.get(bot_type, HeuristicLudoBot)
    return bot_class(player_id=player_id)
