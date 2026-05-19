"""v1 dense reward menu — the era from training_journal.md Experiment 2.

The journal's "Dense Direct Rewards v1" was the strongest pre-V13 training
era. This module exposes that menu as a callable so we can run a
**pure-shaping** experiment (terminal reward removed entirely).

Menu (per single own decision, mostly):
    | Event                      | Magnitude |
    |----------------------------|----------:|
    | Score token (reaches home) |    +0.40 |
    | Capture enemy              |    +0.20 |
    | Got killed                 |    −0.20 |
    | Home stretch entry         |    +0.10 |
    | Spawn (exit base)          |    +0.05 |
    | Forward step on track      |    +0.005 |

Per-game total typically lands in +1.0 .. +3.0 (asymmetric — score/capture
exceed kill events on the long run, so the model has a meaningful
gradient to climb).

**Got-killed is NOT detectable inside a single (pre→post) own-move window**
because opp captures happen during opp's intervening turn. The caller
must pass `prev_own_at_base_count` (the own-at-base snapshot from cp's
last own decision) for that signal to fire. See
`compute_dense_reward_v1_with_kill_tracking` below.
"""
from __future__ import annotations

from typing import Optional


# Position sentinels — kept in sync with `td_ludo.game.progress_score`
BASE_POS = -1
HOME_STRETCH_START = 51   # tokens at pos in [51, 55] are on the home column
HOME_POS = 99             # tokens at pos == 99 are scored (off the board)

# v1 magnitudes — DO NOT scale below 0.05 (journal Exp 4/6 verified this
# collapses learning in stochastic Ludo)
REWARD_SCORE_TOKEN     = 0.40
REWARD_CAPTURE_ENEMY   = 0.20
PENALTY_GOT_KILLED     = -0.20
REWARD_HOME_STRETCH    = 0.10
REWARD_SPAWN           = 0.05
REWARD_FORWARD_STEP    = 0.005


def _count_at_base(positions, player: int) -> int:
    """How many of `player`'s tokens are at BASE_POS in this positions struct."""
    return sum(1 for p in positions[player] if int(p) == BASE_POS)


def _count_in_stretch(positions, player: int) -> int:
    return sum(
        1 for p in positions[player]
        if HOME_STRETCH_START <= int(p) < HOME_POS
    )


def compute_dense_reward_v1(old_state, new_state, player: int) -> float:
    """v1 dense reward for events that happen during `player`'s own move.

    Captures: score, capture, spawn, home-stretch-entry, forward-step.
    Does NOT capture got-killed — see
    `compute_dense_reward_v1_with_kill_tracking` for that.

    `old_state` and `new_state` should be DummyState-like (or any object
    with `.player_positions[p]` indexable per player and `.scores[p]`).
    """
    reward = 0.0
    old_pos = old_state.player_positions
    new_pos = new_state.player_positions

    # 1. Score event — own scored token count increased
    score_delta = int(new_state.scores[player]) - int(old_state.scores[player])
    if score_delta > 0:
        reward += REWARD_SCORE_TOKEN * score_delta

    # 2. Capture event — any opp's at-base count rose during my move
    for opp in range(4):
        if opp == player:
            continue
        opp_old_base = _count_at_base(old_pos, opp)
        opp_new_base = _count_at_base(new_pos, opp)
        capture_delta = opp_new_base - opp_old_base
        if capture_delta > 0:
            reward += REWARD_CAPTURE_ENEMY * capture_delta

    # 3. Spawn event — my at-base count dropped (token came out)
    own_old_base = _count_at_base(old_pos, player)
    own_new_base = _count_at_base(new_pos, player)
    spawn_delta = own_old_base - own_new_base
    if spawn_delta > 0:
        reward += REWARD_SPAWN * spawn_delta

    # 4. Home-stretch entry — count of my tokens in [51, 99) rose
    own_old_stretch = _count_in_stretch(old_pos, player)
    own_new_stretch = _count_in_stretch(new_pos, player)
    # Subtract "score event" contributions: a token going stretch → score
    # would naively count as -1 stretch + 1 score. We only want +1 score.
    stretch_delta = own_new_stretch - own_old_stretch + max(0, score_delta)
    if stretch_delta > 0:
        reward += REWARD_HOME_STRETCH * stretch_delta

    # 5. Forward step on track — own positions moving from lower to higher
    # excluding base→track transitions (counted as spawn) and
    # track→home transitions (counted as score). We sum non-negative
    # absolute deltas on the main track and home column.
    for tok in range(4):
        o = int(old_pos[player][tok])
        n = int(new_pos[player][tok])
        if o == BASE_POS or o == HOME_POS or n == BASE_POS or n == HOME_POS:
            continue  # spawn / score handled above; base→base unchanged
        if n > o:
            reward += REWARD_FORWARD_STEP * (n - o)

    return reward


def compute_kill_penalty(
    prev_own_at_base: Optional[int],
    current_own_at_base: int,
) -> float:
    """Negative reward for tokens captured by opp during their intervening turn.

    `prev_own_at_base`: own at-base count from cp's previous own decision.
        None on first decision (no prior baseline → no kill penalty).
    `current_own_at_base`: own at-base count as of cp's current decision.

    Returns 0 or a negative value.
    """
    if prev_own_at_base is None:
        return 0.0
    delta = current_own_at_base - prev_own_at_base
    if delta > 0:
        return PENALTY_GOT_KILLED * delta
    return 0.0


def total_per_game_reward_estimate() -> dict:
    """Rough analytical estimate of per-game reward distribution under v1
    menu. Useful for sanity-checking that magnitudes still total to a
    learnable signal. NOT used by training."""
    # Typical 2-player game: ~80 plies = 40 own moves
    # Own events per game (approx):
    #   spawn ~3 (1 per ~13 plies, dice=6 only)
    #   forward ~3.5 per move × 40 moves = 140 cells advanced (4 tokens × 56 ≈ 224 max)
    #   home_stretch entries ~3-4 (3-4 tokens reach stretch)
    #   score ~3-4 (3-4 tokens score)
    #   capture ~1-3 (varies by play style)
    #   got_killed ~1-3 (mirror of capture)
    return {
        "spawn":        3 * REWARD_SPAWN,           # +0.15
        "forward":      140 * REWARD_FORWARD_STEP,  # +0.70
        "home_stretch": 3.5 * REWARD_HOME_STRETCH,  # +0.35
        "score":        3.5 * REWARD_SCORE_TOKEN,   # +1.40
        "capture":      2.0 * REWARD_CAPTURE_ENEMY, # +0.40
        "got_killed":   2.0 * PENALTY_GOT_KILLED,   # -0.40
        # Total positive: +3.00
        # Total negative: -0.40
        # Net per game: ~+2.60 (matches journal Exp 2 estimate of +1 to +3)
    }
