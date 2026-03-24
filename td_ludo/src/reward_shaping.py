"""
Strategic Reward Shaping for TD-Ludo (v2.0)

Builds on proven v1.1 dense rewards with 6 new strategic rewards that
incentivize setup behaviors (chasing, safety-seeking, blocking, etc.)
rather than just outcomes (captures, scoring).

EXISTING (proven, do not modify):
- Moving a token out of the base (+0.05)
- Moving tokens forward (+0.005 per step)
- Entering the home stretch (+0.10)
- Scoring a token (+0.40)
- Capturing opponent tokens: +0.20
- Getting killed: -0.20

NEW STRATEGIC (v2.0):
- Chase target acquired: +0.06 per new opponent in dice range
- Safety transition: +0.08 for moving unsafe → safe/stacked
- Danger reduction: +0.06 per own token removed from danger
- Stack formed: +0.07 per new protective stack created
- Leader capture bonus: +0.08 when capturing the score leader
- Endgame score urgency: +0.15 when scoring while opponent has 3

Design principles:
- Delta-based (reward transitions, not states) to avoid persistent bias
- max(delta, 0) pattern — only reward improvement, never penalize
- Magnitudes ≥ 0.05 (dice noise drowns weaker signals)
- All existing rewards preserved exactly
"""

import td_ludo_cpp as ludo_cpp
from src.config import NUM_ACTIVE_PLAYERS

# =============================================================================
# Constants
# =============================================================================
HOME_STRETCH_START = 51
SCORE_POSITION = 56  # Token is home

# Safe squares (absolute board positions) — invariant under +13 mod 52 rotation,
# so these work as both relative and absolute positions for any player.
SAFE_SQUARES = frozenset({0, 8, 13, 21, 26, 34, 39, 47})


# =============================================================================
# Helper Functions
# =============================================================================

def _get_abs_pos(player, rel_pos):
    """Convert player-relative position (0-50) to absolute board index (0-51)."""
    if rel_pos < 0 or rel_pos > 50:
        return -1
    return (rel_pos + player * 13) % 52


def _is_on_main_track(pos):
    """Check if position is on the main shared track (capturable zone)."""
    return 0 <= pos <= 50


def _is_safe_square(player, rel_pos):
    """Check if a main-track position is a safe/globe square."""
    if not _is_on_main_track(rel_pos):
        return False
    abs_pos = _get_abs_pos(player, rel_pos)
    return abs_pos in SAFE_SQUARES


def _is_stacked(state, player, pos):
    """Check if 2+ of player's tokens are at the same position."""
    if not _is_on_main_track(pos):
        return False
    count = 0
    for t in range(4):
        if state.player_positions[player][t] == pos:
            count += 1
    return count >= 2


def _is_safe_position(state, player, pos):
    """Check if a token at pos is safe (safe square, stacked, home stretch, base, or scored)."""
    if pos < 0 or pos > 50:
        return True  # base, home stretch, scored — all safe from capture
    return _is_safe_square(player, pos) or _is_stacked(state, player, pos)


def _count_capturable_targets_in_range(state, player, token_idx):
    """
    Count opponent tokens within dice range (1-6 steps ahead) that are capturable.
    Only counts opponents on main track, not on safe squares, not stacked.
    """
    my_pos = state.player_positions[player][token_idx]
    if not _is_on_main_track(my_pos):
        return 0

    my_abs = _get_abs_pos(player, my_pos)
    count = 0

    for opp in range(4):
        if opp == player or not state.active_players[opp]:
            continue
        for t in range(4):
            opp_pos = state.player_positions[opp][t]
            if not _is_on_main_track(opp_pos):
                continue
            opp_abs = _get_abs_pos(opp, opp_pos)
            # Distance I need to travel to reach opponent (circular)
            dist = (opp_abs - my_abs) % 52
            if 1 <= dist <= 6:
                # Check opponent is actually capturable (not safe/stacked)
                if opp_abs in SAFE_SQUARES:
                    continue
                if _is_stacked(state, opp, opp_pos):
                    continue
                count += 1

    return count


def _count_endangered_tokens(state, player):
    """
    Count how many of player's tokens are in danger:
    on main track, not safe/stacked, with an opponent 1-6 steps behind.
    """
    count = 0
    for t in range(4):
        my_pos = state.player_positions[player][t]
        if not _is_on_main_track(my_pos):
            continue
        if _is_safe_position(state, player, my_pos):
            continue

        my_abs = _get_abs_pos(player, my_pos)
        endangered = False

        for opp in range(4):
            if opp == player or not state.active_players[opp]:
                continue
            for ot in range(4):
                opp_pos = state.player_positions[opp][ot]
                if not _is_on_main_track(opp_pos):
                    continue
                opp_abs = _get_abs_pos(opp, opp_pos)
                # How far ahead am I of this opponent (they need this many to reach me)
                dist = (my_abs - opp_abs) % 52
                if 1 <= dist <= 6:
                    endangered = True
                    break
            if endangered:
                break

        if endangered:
            count += 1

    return count


def _count_stacks(state, player):
    """Count number of stack formations (2+ tokens at same main-track position)."""
    positions = {}
    for t in range(4):
        pos = state.player_positions[player][t]
        if _is_on_main_track(pos):
            positions[pos] = positions.get(pos, 0) + 1
    return sum(1 for c in positions.values() if c >= 2)


def _find_moved_token(state, next_state, player):
    """Find which token moved (returns token index 0-3, or -1 if none)."""
    for t in range(4):
        if state.player_positions[player][t] != next_state.player_positions[player][t]:
            return t
    return -1


def _identify_captured_opponent(state, next_state, player):
    """Find which opponent was captured (sent to base). Returns player index or -1."""
    for opp in range(4):
        if opp == player:
            continue
        for t in range(4):
            old_pos = state.player_positions[opp][t]
            new_pos = next_state.player_positions[opp][t]
            if old_pos >= 0 and old_pos != 99 and new_pos == -1:
                return opp
    return -1


# =============================================================================
# Main Reward Function
# =============================================================================

def compute_shaped_reward(state, next_state, player):
    """
    Compute dense strategic rewards based on state delta.

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

    capture_occurred = False
    scored_this_move = False

    # =========================================================================
    # SECTION 1: EXISTING REWARDS (v1.1, proven — do not modify)
    # =========================================================================

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
            scored_this_move = True

        # Standard forward progress
        if p1 >= 0 and p2 > p1:
            reward += 0.005 * (p2 - p1)

        # Got killed / sent to base
        if p1 >= 0 and p2 == -1:
            reward -= 0.20

    # Evaluate opponent captures
    for opp in range(4):
        if opp == player:
            continue
        opp_pos_old = state.player_positions[opp]
        opp_pos_new = next_state.player_positions[opp]
        for i in range(4):
            if opp_pos_old[i] >= 0 and opp_pos_new[i] == -1:
                reward += 0.20
                capture_occurred = True

    # =========================================================================
    # SECTION 2: NEW STRATEGIC REWARDS (v2.0)
    # =========================================================================

    moved_token = _find_moved_token(state, next_state, player)

    # --- 2a. Chase Target Acquired (+0.06) ---
    # Reward gaining new capturable opponents within dice range
    if moved_token >= 0:
        targets_before = _count_capturable_targets_in_range(state, player, moved_token)
        targets_after = _count_capturable_targets_in_range(next_state, player, moved_token)
        delta_targets = targets_after - targets_before
        if delta_targets > 0:
            reward += 0.06 * delta_targets

    # --- 2b. Safety Transition (+0.08) ---
    # Reward moving from an endangered position to a safe one.
    # Only fires when the token was actually in danger (opponent 1-6 behind),
    # not just any unsafe→safe transition (which is too frequent).
    if moved_token >= 0:
        old_pos = our_pos_old[moved_token]
        new_pos = our_pos_new[moved_token]
        if old_pos >= 0 and _is_on_main_track(old_pos):
            was_safe = _is_safe_position(state, player, old_pos)
            now_safe = _is_safe_position(next_state, player, new_pos)
            if not was_safe and now_safe:
                # Check if the token was actually endangered (opponent 1-6 behind)
                my_abs = _get_abs_pos(player, old_pos)
                was_endangered = False
                for opp in range(4):
                    if opp == player or not state.active_players[opp]:
                        continue
                    for ot in range(4):
                        opp_pos = state.player_positions[opp][ot]
                        if not _is_on_main_track(opp_pos):
                            continue
                        opp_abs = _get_abs_pos(opp, opp_pos)
                        dist = (my_abs - opp_abs) % 52
                        if 1 <= dist <= 6:
                            was_endangered = True
                            break
                    if was_endangered:
                        break
                if was_endangered:
                    reward += 0.08

    # --- 2c. Danger Reduction (+0.06 per token) ---
    # Reward reducing the count of own endangered tokens
    danger_before = _count_endangered_tokens(state, player)
    danger_after = _count_endangered_tokens(next_state, player)
    delta_danger = danger_before - danger_after  # positive = reduced danger
    if delta_danger > 0:
        reward += 0.06 * delta_danger

    # --- 2d. Stack Formed (+0.07) ---
    # Reward creating new protective stacks
    stacks_before = _count_stacks(state, player)
    stacks_after = _count_stacks(next_state, player)
    delta_stacks = stacks_after - stacks_before
    if delta_stacks > 0:
        reward += 0.07 * delta_stacks

    # --- 2e. Leader Capture Bonus (+0.08) ---
    # Extra reward for capturing the player who is currently winning
    if capture_occurred:
        victim = _identify_captured_opponent(state, next_state, player)
        if victim >= 0:
            max_opp_score = max(
                state.scores[i] for i in range(4) if i != player
            )
            if max_opp_score > 0 and state.scores[victim] == max_opp_score:
                reward += 0.08

    # --- 2f. Endgame Score Urgency (+0.15) ---
    # Bonus for scoring when opponent is close to winning (3+ tokens home)
    if scored_this_move:
        max_opp_score = max(
            state.scores[i] for i in range(4) if i != player
        )
        if max_opp_score >= 3:
            reward += 0.15

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
