"""
Potential-Based Reward Shaping (PBRS) for TD-Ludo

The shaped reward is:
    R_shaped(s, a, s') = R_original + γ · Φ(s') - Φ(s)

where Φ(s) is a potential function measuring board quality for a player.

Theorem (Ng et al., 1999): PBRS preserves the optimal policy.
This means the agent still learns to WIN, but gets denser signal to learn faster.
"""

import td_ludo_cpp as ludo_cpp


# =============================================================================
# Constants
# =============================================================================
# Safe positions per player (globes + home stretch)
# Globes are at absolute positions 0, 8, 13, 21, 26, 34, 39, 47
GLOBE_POSITIONS = {0, 8, 13, 21, 26, 34, 39, 47}

# Each player's start globe (their own safe globe)
PLAYER_START_GLOBE = {0: 0, 1: 13, 2: 26, 3: 39}

# Home stretch starts at relative position 51
HOME_STRETCH_START = 51
SCORE_POSITION = 56  # Token is home


# =============================================================================
# Potential Function
# =============================================================================
def potential(state, player):
    """
    Compute the potential Φ(s) for a given player.
    
    Higher potential = better board position for this player.
    
    Components:
    - Token progress (further along = better)
    - Token safety (safe zones, home stretch)
    - Tokens scored (home)
    - Opponent threat (tokens that can capture ours)
    
    Returns:
        float: Potential value for this player's position
    """
    phi = 0.0
    positions = state.player_positions[player]
    
    for token_idx in range(4):
        pos = positions[token_idx]
        
        if pos == -1:
            # Token in base — no value
            phi += 0.0
        elif pos >= SCORE_POSITION:
            # Token scored (home) — maximum value per token
            phi += 1.0
        elif pos >= HOME_STRETCH_START:
            # In home stretch (positions 51-55) — very safe, close to scoring
            progress = (pos - HOME_STRETCH_START) / (SCORE_POSITION - HOME_STRETCH_START)
            phi += 0.70 + 0.25 * progress  # 0.70 → 0.95
        else:
            # On the main board (positions 0-50)
            # Base progress: linear from 0.05 to 0.60
            progress = pos / HOME_STRETCH_START
            phi += 0.05 + 0.55 * progress
            
            # Safety bonus: on a globe or star
            abs_pos = _get_absolute_position(player, pos)
            if abs_pos in GLOBE_POSITIONS:
                phi += 0.05
    
    # Opponent threat penalty
    # Count how many opponent tokens are within striking distance (1-6 behind us)
    threats = _count_threats(state, player)
    phi -= 0.03 * threats
    
    return phi


def _get_absolute_position(player, relative_pos):
    """Convert player-relative position to absolute board position (0-51)."""
    if relative_pos < 0 or relative_pos >= HOME_STRETCH_START:
        return -1  # Not on main board
    offset = player * 13  # Each player starts 13 positions apart
    return (relative_pos + offset) % 52


def _count_threats(state, player):
    """
    Count the number of opponent tokens that threaten our tokens.
    A token is "threatened" if an opponent is within 1-6 squares behind it
    on the main board (not in safe zones).
    """
    threats = 0
    our_positions = state.player_positions[player]
    
    for token_idx in range(4):
        our_pos = our_positions[token_idx]
        if our_pos < 0 or our_pos >= HOME_STRETCH_START:
            continue  # In base or home stretch, can't be captured
            
        our_abs = _get_absolute_position(player, our_pos)
        if our_abs in GLOBE_POSITIONS:
            continue  # On a globe, safe
        
        # Check all opponents
        for opp in range(4):
            if opp == player:
                continue
            opp_positions = state.player_positions[opp]
            for opp_token in range(4):
                opp_pos = opp_positions[opp_token]
                if opp_pos < 0 or opp_pos >= HOME_STRETCH_START:
                    continue
                opp_abs = _get_absolute_position(opp, opp_pos)
                
                # Check if opponent is 1-6 squares behind our token
                for dist in range(1, 7):
                    if (opp_abs + dist) % 52 == our_abs:
                        threats += 1
                        break  # Only count once per opponent token
    
    return threats


# =============================================================================
# Shaped Reward Computation
# =============================================================================
def compute_shaped_reward(state, next_state, player, raw_reward, gamma=0.995):
    """
    Compute PBRS-shaped reward.
    
    R_shaped = R_raw + γ · Φ(s') - Φ(s)
    
    Args:
        state: Current game state (before move)
        next_state: Next game state (after move)
        player: Player index (0-3)
        raw_reward: Original reward (0 for intermediate, +1/-1 for terminal)
        gamma: Discount factor
        
    Returns:
        float: Shaped reward
    """
    phi_s = potential(state, player)
    phi_s_next = potential(next_state, player)
    
    shaped = raw_reward + gamma * phi_s_next - phi_s
    return shaped


# =============================================================================
# Terminal Reward
# =============================================================================
def get_terminal_reward(state, player):
    """
    Get the terminal reward for a player.
    
    Returns:
        +1.0 if player won
        -0.33 if player lost (distributed among 3 losers)
        0.0 if game is not terminal
    """
    if not state.is_terminal:
        return 0.0
    
    winner = ludo_cpp.get_winner(state)
    if winner == player:
        return 1.0
    elif winner >= 0:
        return -0.33  # Loss shared among 3 losers
    return 0.0  # Draw (shouldn't happen in Ludo)
