"""
Dense Direct Reward Shaping for TD-Ludo (v1.1 — Surgical Unbias)

Provides immediate, tangible rewards for:
- Moving a token out of the base (+0.05)
- Moving tokens forward (+0.005 per step)
- Entering the home stretch (+0.10)
- Scoring a token (+0.40)

Removed (Experiment 8 — non-potential-based / biased):
- Capturing opponent tokens: was +0.20, now 0 (let model discover capture value from outcomes)
- Getting killed: was -0.20, now 0 (let model discover death cost from outcomes)

All remaining rewards are potential-based (derived from own state changes),
which are mathematically guaranteed to preserve the optimal policy (Ng et al. 1999).
"""

import td_ludo_cpp as ludo_cpp
from src.config import NUM_ACTIVE_PLAYERS

# =============================================================================
# Constants
# =============================================================================
HOME_STRETCH_START = 51
SCORE_POSITION = 56  # Token is home

def compute_shaped_reward(state, next_state, player):
    """
    Compute dense direct rewards based on state delta.
    
    Args:
        state: Current game state (before move)
        next_state: Next game state (after move)
        player: Player index (0-3)
        
    Returns:
        float: Shaped reward
    """
        
    reward = 0.0
    
    our_pos_old = state.player_positions[player]
    our_pos_new = next_state.player_positions[player]
    
    # 1. Evaluate our own tokens (Progress, Base, Scoring, Dying)
    for i in range(4):
        p1 = our_pos_old[i]
        p2 = our_pos_new[i]
        
        if p1 == p2:
            continue
            
        # Left base
        if p1 == -1 and p2 >= 0:
            reward += 0.05
            
        # Entered home stretch
        if p1 < HOME_STRETCH_START and p2 >= HOME_STRETCH_START and p2 < SCORE_POSITION:
            reward += 0.10
            
        # Scored a token
        if p1 < SCORE_POSITION and p2 >= SCORE_POSITION:
            reward += 0.40
            
        # Standard forward progress
        if p1 >= 0 and p2 > p1:
            reward += 0.005 * (p2 - p1)
            
        # Got killed / sent to base — REMOVED (Experiment 8: non-potential-based bias)
        # if p1 >= 0 and p2 == -1:
        #     reward -= 0.20
            
    # 2. Evaluate opponent captures — REMOVED (Experiment 8: non-potential-based bias)
    # Capture reward (+0.20) removed to let the model discover the true value
    # of captures from game outcomes alone.
    # for opp in range(4):
    #     if opp == player:
    #         continue
    #     opp_pos_old = state.player_positions[opp]
    #     opp_pos_new = next_state.player_positions[opp]
    #     for i in range(4):
    #         if opp_pos_old[i] >= 0 and opp_pos_new[i] == -1:
    #             reward += 0.20
                
    return reward

def get_terminal_reward(state, player):
    """
    Get the terminal reward for a player.
    """
    if not state.is_terminal:
        return 0.0
    
    winner = ludo_cpp.get_winner(state)
    if winner == player:
        return 1.0
    elif winner >= 0:
        return -1.0 / max(1, (NUM_ACTIVE_PLAYERS - 1))
    return 0.0  # Draw
