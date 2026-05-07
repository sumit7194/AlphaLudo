"""Bias-correction penalties for V13 RL training.

Five targeted negative rewards that fire when the model exhibits known
policy biases identified empirically from 100K v12.2 self-play and 9
hand-flagged disagreements:

  1. Unlock-on-6 when a better non-base alternative existed
     (the 92% bias from the dice=6 conditional analysis)
  2. Missed capture (capture was available, didn't take)
  3. Missed finish (could have scored a token with the dice, didn't)
  4. Left a safe square unnecessarily when other tokens could move
  5. Moved an advanced token (pos > 35) into capture range needlessly

All penalties are *delta-shaped* — they add to compute_shaped_reward's
output. Each fires only when the cheap-to-detect "this was clearly
wrong" condition is met (so most decisions get 0 penalty).

Phase scaling: penalties strengthen as we move past opening. Empirical
phase boundaries (from 100K self-play, median game = 160 moves):
  early     (0-16):    no penalties (opening play is constrained)
  early_mid (17-48):   half-strength
  mid+      (49+):     full strength

Magnitudes are in win-prob units, comparable to the existing strategic
rewards in reward_shaping.py. Total per-step penalty is capped at
ABS_MAX_PENALTY to prevent overwhelming the policy gradient.

Usage:
    from td_ludo.game.bias_penalties import compute_bias_penalties
    total, breakdown = compute_bias_penalties(
        state, next_state, player,
        context={'dice': 6, 'legal_moves': [0, 2, 3], 'action': 0, 'move_count': 50}
    )
"""

# =============================================================================
# Constants — keep aligned with reward_shaping.py
# =============================================================================
SAFE_SQUARES = frozenset({0, 8, 13, 21, 26, 34, 39, 47})
HOME_STRETCH_START = 51
SCORE_POSITION = 99
BASE_POSITION = -1

# Empirical phase boundaries (median game length = 160 in v122 self-play)
EARLY_END = 16
EARLY_MID_END = 48

# Penalty magnitudes (win-prob units)
P_UNLOCK_BETTER_AVAIL = 0.05
P_MISS_CAPTURE_BASE = 0.10
P_MISS_FINISH = 0.15
P_LEFT_SAFE = 0.03
# Bumped 3× from 0.04 on 2026-05-05 — danger-blindness was a visible failure
# mode in V13.2 play (model pushed leader into opp capture range despite
# alternatives). dice=6 discount intentionally retained.
P_DANGER_ADVANCED_BASE = 0.12
# Penalty 6: per-cell coefficient for laggard distance when scoring 3rd token.
# Range: 0.0025 × 1 = -0.0025 (laggard 1 cell from home) to
#        0.0025 × 99 = -0.2475 (laggard at base or spawn) — capped by
#        ABS_MAX_PENALTY=0.15 in the worst cases. Fires once per game.
# Bumped 5× from 0.0005 on 2026-05-05 — earlier value was barely shifting
# behavior; this makes a far-laggard 3rd-score cost ~7-15% of value swing.
P_LAGGARD_PER_CELL = 0.0025

# Cap on total per-step penalty (asymmetric — can't go more negative than this)
ABS_MAX_PENALTY = 0.15


# =============================================================================
# Helpers
# =============================================================================

def _is_main_track(pos):
    return 0 <= pos <= 50


def _abs_pos(player, rel):
    """Player-relative pos → absolute board index, only meaningful on track."""
    if not _is_main_track(rel):
        return -1
    return (rel + player * 13) % 52


def _is_safe_square(player, rel):
    return _is_main_track(rel) and _abs_pos(player, rel) in SAFE_SQUARES


def _simulate_dest(state, player, token, dice):
    """Where would `token` land if moved with `dice`? Conservative — handles
    common cases (spawn on 6, advance on track, enter home stretch, finish
    exactly). Returns None if move would be illegal (e.g. blockers, overshoot).

    Doesn't simulate captures or stack-block rules; caller checks those.
    """
    cur = state.player_positions[player][token]
    if cur == BASE_POSITION:
        return 0 if dice == 6 else None     # spawn on 6 only
    if cur == SCORE_POSITION:
        return None                          # already scored
    new_p = cur + dice
    if new_p == SCORE_POSITION:
        return SCORE_POSITION
    if new_p > SCORE_POSITION:
        return None                          # overshoot
    return new_p


def _captures_opp_at(state, player, dest):
    """Would landing at `dest` (player-relative) capture an opp? Engine rule
    says safe squares don't allow captures."""
    if not _is_main_track(dest):
        return False
    if _is_safe_square(player, dest):
        return False
    abs_dest = _abs_pos(player, dest)
    for opp in range(4):
        if opp == player or not state.active_players[opp]:
            continue
        for ot in range(4):
            opp_pos = state.player_positions[opp][ot]
            if not _is_main_track(opp_pos):
                continue
            if _abs_pos(opp, opp_pos) == abs_dest:
                return True
    return False


def _in_capture_range(state, player, pos):
    """Is the token at `pos` (player-relative) within 1-6 cells of any opp?
    (Same definition reward_shaping.py uses for danger.)"""
    if not _is_main_track(pos):
        return False
    if _is_safe_square(player, pos):
        return False
    my_abs = _abs_pos(player, pos)
    for opp in range(4):
        if opp == player or not state.active_players[opp]:
            continue
        for ot in range(4):
            opp_pos = state.player_positions[opp][ot]
            if not _is_main_track(opp_pos):
                continue
            dist = (my_abs - _abs_pos(opp, opp_pos)) % 52
            if 1 <= dist <= 6:
                return True
    return False


def phase_scale(move_count):
    """Penalties off in opening (move 0-16), half through 17-48, full from 49+."""
    if move_count <= EARLY_END:
        return 0.0
    if move_count <= EARLY_MID_END:
        return 0.5
    return 1.0


# =============================================================================
# Main entrypoint
# =============================================================================

def compute_bias_penalties(state, next_state, player, context):
    """Compute negative rewards for the 5 known biases.

    Args:
        state: pre-move state (positions, scores, active_players)
        next_state: post-move state
        player: int, the deciding player
        context: dict with required keys:
                 'dice'        : int (1-6)
                 'legal_moves' : list[int]  (legal token indices at decision)
                 'action'      : int        (chosen token, 0-3)
                 'move_count'  : int        (current ply count for the game)

    Returns:
        (total_penalty, breakdown) where:
            total_penalty: float ≤ 0
            breakdown: dict with one entry per penalty
    """
    breakdown = {
        'unlock_with_better': 0.0,
        'missed_capture': 0.0,
        'missed_finish': 0.0,
        'left_safe': 0.0,
        'advanced_into_danger': 0.0,
        'laggard_on_3score': 0.0,
    }

    if context is None:
        return 0.0, breakdown

    dice = int(context.get('dice', 0))
    legal = list(context.get('legal_moves', []))
    action = int(context.get('action', -1))
    move_count = int(context.get('move_count', 0))
    scale = phase_scale(move_count)

    if action < 0 or action > 3 or not legal:
        return 0.0, breakdown

    own_pos = state.player_positions[player]
    chosen_pre = own_pos[action]
    chosen_post = next_state.player_positions[player][action]

    # ── Penalty 1: unlock-on-6 with better alternative ────────────────
    if dice == 6 and chosen_pre == BASE_POSITION and scale > 0:
        nonbase_alts = [t for t in legal if own_pos[t] != BASE_POSITION]
        if nonbase_alts:
            better_found = False
            for alt in nonbase_alts:
                dest = _simulate_dest(state, player, alt, dice)
                if dest is None:
                    continue
                # Three "better" cases: finish, capture, escape-danger
                if dest == SCORE_POSITION:
                    better_found = True
                    break
                if _captures_opp_at(state, player, dest):
                    better_found = True
                    break
                if (_in_capture_range(state, player, own_pos[alt])
                        and not _in_capture_range(state, player, dest)):
                    better_found = True
                    break
            if better_found:
                breakdown['unlock_with_better'] = -P_UNLOCK_BETTER_AVAIL * scale

    # ── Penalty 2: missed capture ─────────────────────────────────────
    captured_now = _did_capture(state, next_state, player)
    if not captured_now:
        for alt in legal:
            if alt == action:
                continue
            dest = _simulate_dest(state, player, alt, dice)
            if dest is None:
                continue
            if _captures_opp_at(state, player, dest):
                # Penalty scales with opp progress (captured token's pos)
                penalty = P_MISS_CAPTURE_BASE
                abs_dest = _abs_pos(player, dest)
                for opp in range(4):
                    if opp == player:
                        continue
                    for ot in range(4):
                        op = state.player_positions[opp][ot]
                        if _is_main_track(op) and _abs_pos(opp, op) == abs_dest:
                            penalty += 0.005 * op
                            break
                breakdown['missed_capture'] = -penalty
                break

    # ── Penalty 3: missed finish ─────────────────────────────────────
    chosen_finished = (chosen_pre != SCORE_POSITION and chosen_post == SCORE_POSITION)
    if not chosen_finished:
        for alt in legal:
            if alt == action:
                continue
            dest = _simulate_dest(state, player, alt, dice)
            if dest == SCORE_POSITION:
                breakdown['missed_finish'] = -P_MISS_FINISH
                break

    # ── Penalty 4: left a safe square unnecessarily ──────────────────
    # Halved when dice == 6 because a 6 grants a bonus turn — the player
    # can still re-cover or move another token, so the exposure cost is
    # genuinely lower than for non-6 rolls.
    if scale > 0 and _is_safe_square(player, chosen_pre):
        non_safe_alts = [
            t for t in legal
            if t != action
            and own_pos[t] not in (BASE_POSITION, SCORE_POSITION)
            and _is_main_track(own_pos[t])
            and not _is_safe_square(player, own_pos[t])
        ]
        if non_safe_alts:
            bonus_factor = 0.5 if dice == 6 else 1.0
            breakdown['left_safe'] = -P_LEFT_SAFE * scale * bonus_factor

    # ── Penalty 5: moved advanced token into danger ──────────────────
    if (_is_main_track(chosen_pre) and chosen_pre > 35
            and _is_main_track(chosen_post)
            and _in_capture_range(next_state, player, chosen_post)):
        # Was there a safer alternative? "Safer" = doesn't create danger
        # for an equally-or-more-advanced token of ours.
        safer_alt_exists = False
        for alt in legal:
            if alt == action:
                continue
            alt_pre = own_pos[alt]
            dest = _simulate_dest(state, player, alt, dice)
            if dest is None:
                continue
            # Skip alts that ALSO move an advanced token into danger
            if (_is_main_track(alt_pre) and alt_pre > 35
                    and _is_main_track(dest)
                    and _in_capture_range(state, player, dest)):
                continue
            safer_alt_exists = True
            break
        if safer_alt_exists:
            # Halved when dice == 6 because the bonus turn lets the player
            # immediately re-cover or counter-move the now-exposed token.
            bonus_factor = 0.5 if dice == 6 else 1.0
            mag = P_DANGER_ADVANCED_BASE * (1 + (chosen_pre - 35) / 20) * bonus_factor
            breakdown['advanced_into_danger'] = -mag

    # ── Penalty 6: laggard distance when scoring 3rd token ───────────
    # Triggers exactly once per game per player — on the score 2→3 transition.
    # Penalises situations where the player completed 3 tokens but their
    # remaining laggard is far from home (avoidable by advancing all tokens
    # together rather than racing 3 to the finish line).
    own_score_pre = int(state.scores[player])
    own_score_post = int(next_state.scores[player])
    if own_score_pre == 2 and own_score_post == 3:
        # Find the one remaining non-home token in next_state
        own_pos_post = next_state.player_positions[player]
        laggard_pos = None
        for t in range(4):
            if int(own_pos_post[t]) != SCORE_POSITION:
                laggard_pos = int(own_pos_post[t])
                break
        if laggard_pos is not None:
            # Distance to home: at-base treated as max distance.
            if laggard_pos == BASE_POSITION:
                distance = 99
            else:
                distance = max(0, 99 - laggard_pos)
            breakdown['laggard_on_3score'] = -P_LAGGARD_PER_CELL * distance

    # ── Cap total ────────────────────────────────────────────────────
    total = sum(breakdown.values())
    if total < -ABS_MAX_PENALTY:
        rescale = -ABS_MAX_PENALTY / total  # positive < 1
        for k in breakdown:
            breakdown[k] *= rescale
        total = -ABS_MAX_PENALTY

    return total, breakdown


def _did_capture(state, next_state, player):
    """True iff `player`'s move sent any opp token to base."""
    for opp in range(4):
        if opp == player or not state.active_players[opp]:
            continue
        for ot in range(4):
            old_op = state.player_positions[opp][ot]
            new_op = next_state.player_positions[opp][ot]
            if old_op >= 0 and old_op != SCORE_POSITION and new_op == BASE_POSITION:
                return True
    return False
