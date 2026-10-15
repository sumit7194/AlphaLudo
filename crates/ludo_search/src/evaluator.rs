//! Position evaluation heuristics for scoring game states from any player's POV.

use ludo_core::{
    get_absolute_pos, is_safe_pos, GameState, BASE_POS, HOME_POS, NUM_PLAYERS, NUM_TOKENS,
};

/// Sum of token progress (0..56 per token). Home = 56, Base = -10 (penalty).
#[inline]
pub fn token_progress_score(player: u8, state: &GameState) -> f32 {
    let p = player as usize;
    let mut sum = 0.0;
    for t in 0..NUM_TOKENS {
        let pos = state.positions[p][t];
        if pos == HOME_POS {
            sum += 56.0;
        } else if pos == BASE_POS {
            sum -= 10.0;
        } else {
            sum += pos as f32;
        }
    }
    sum
}

/// Checks whether a token is currently safe (base, home, home stretch, or globe/star).
#[inline]
pub fn is_token_safe(player: u8, pos: i8) -> bool {
    if pos == BASE_POS || pos == HOME_POS || pos >= 51 {
        return true;
    }
    if let Some(abs) = get_absolute_pos(player, pos) {
        is_safe_pos(abs)
    } else {
        true
    }
}

/// Sum penalty for each of this player's unsafe tokens that an active opponent
/// could capture within 1..6 squares on the next turn.
#[inline]
pub fn exposure_penalty(player: u8, state: &GameState) -> f32 {
    let p = player as usize;
    let mut penalty = 0.0;

    for t in 0..NUM_TOKENS {
        let pos = state.positions[p][t];
        if is_token_safe(player, pos) {
            continue;
        }
        let my_abs = match get_absolute_pos(player, pos) {
            Some(a) => a as i16,
            None => continue,
        };

        // Check if any active opponent is 1..6 squares behind
        'opp_check: for opp in 0..NUM_PLAYERS {
            if opp == p || !state.active_players[opp] {
                continue;
            }
            for opp_t in 0..NUM_TOKENS {
                let opp_pos = state.positions[opp][opp_t];
                if opp_pos == BASE_POS || opp_pos == HOME_POS || opp_pos >= 51 {
                    continue;
                }
                if let Some(opp_abs) = get_absolute_pos(opp as u8, opp_pos) {
                    let dist = (my_abs - (opp_abs as i16)).rem_euclid(52);
                    if (1..=6).contains(&dist) {
                        penalty += 15.0;
                        break 'opp_check;
                    }
                }
            }
        }
    }

    penalty
}

/// Standard Expectimax scoring: own progress + 60*scored - exposure, minus opponent progress.
#[inline]
pub fn score_position(player: u8, state: &GameState) -> f32 {
    let p = player as usize;
    let own_prog = token_progress_score(player, state);
    let own_scored = (state.scores[p] as f32) * 60.0;
    let exposure = exposure_penalty(player, state);

    let mut opp_prog = 0.0;
    for opp in 0..NUM_PLAYERS {
        if opp == p || !state.active_players[opp] {
            continue;
        }
        opp_prog += token_progress_score(opp as u8, state);
        opp_prog += (state.scores[opp] as f32) * 60.0;
    }

    (own_prog + own_scored - exposure) * 2.0 - opp_prog
}

/// Aggressive scoring (Elo Champion): 3x bonus for putting opponents in danger,
/// reduced fear of own exposure (0.5x).
#[inline]
pub fn score_aggressive(player: u8, state: &GameState) -> f32 {
    let p = player as usize;
    let own_prog = token_progress_score(player, state);
    let own_scored = (state.scores[p] as f32) * 60.0;
    let own_exposure = exposure_penalty(player, state);

    let mut opp_prog = 0.0;
    let mut opp_exposure_total = 0.0;
    for opp in 0..NUM_PLAYERS {
        if opp == p || !state.active_players[opp] {
            continue;
        }
        opp_prog += token_progress_score(opp as u8, state);
        opp_prog += (state.scores[opp] as f32) * 60.0;
        opp_exposure_total += exposure_penalty(opp as u8, state);
    }

    (own_prog + own_scored - 0.5 * own_exposure) - opp_prog + 3.0 * opp_exposure_total
}

/// Defensive scoring: 3x penalty for own exposure, avoids risk at all costs.
#[inline]
pub fn score_defensive(player: u8, state: &GameState) -> f32 {
    let p = player as usize;
    let own_prog = token_progress_score(player, state);
    let own_scored = (state.scores[p] as f32) * 60.0;
    let own_exposure = exposure_penalty(player, state);

    let mut opp_prog = 0.0;
    for opp in 0..NUM_PLAYERS {
        if opp == p || !state.active_players[opp] {
            continue;
        }
        opp_prog += token_progress_score(opp as u8, state);
        opp_prog += (state.scores[opp] as f32) * 60.0;
    }

    (own_prog + own_scored - 3.0 * own_exposure) - 2.0 * opp_prog
}

// ─────────────────────────────────────────────────────────────────────────────
// Calibrated win-probability leaf evaluator (2026-10-02 audit fix).
//
// WHY: TwoPlayerMCTSBot's two leaf options were both blind in the early game.
//   * random rollout capped at 60 moves returning 0.0 on cutoff: 98.5% of
//     early-game simulations returned exactly 0.0;
//   * the "heuristic leaf" computed max(score_position, 0)+1 per side, and with
//     every token in base score_position is negative for BOTH players, so both
//     clip to 0 and the value is again exactly 0.
//
// This evaluator is a logistic win model fitted on Depth2-vs-Depth2 games, with
// features computed by THIS function — the fit and the search share one
// implementation, so they cannot drift apart. Features are taken at the
// pre-roll point, from the perspective of the player about to roll, which is
// exactly the state at an MCTS chance node.
// ─────────────────────────────────────────────────────────────────────────────

/// Number of features returned by [`leaf_features`].
pub const N_LEAF_FEATURES: usize = 9;

/// Player-relative features for `player` (opponent = every other active seat).
///
/// Index:
///   0 progress difference / 56           (base=0, track=pos, home lane=pos, scored=56)
///   1 base-token difference              (my tokens in base − theirs)
///   2 scored-token difference
///   3 exposed-token difference           (unsafe tokens an opponent can hit with 1..6)
///   4 home-lane difference               (tokens in 51..55: safe, not yet scored)
///   5 progress difference × stage        (stage = total progress / 448; leads matter more late)
///   6 my tokens in base == 4             (cannot move without a 6)
///   7 their tokens in base == 4
///   8 constant 1.0                       (tempo: being the player about to roll)
#[inline]
pub fn leaf_features(player: u8, state: &GameState) -> [f32; N_LEAF_FEATURES] {
    #[inline(always)]
    fn side(p: usize, s: &GameState) -> (f32, f32, f32, f32) {
        let (mut prog, mut base, mut scored, mut lane) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
        for t in 0..NUM_TOKENS {
            let pos = s.positions[p][t];
            if pos == BASE_POS {
                base += 1.0;
            } else if pos == HOME_POS {
                prog += 56.0;
                scored += 1.0;
            } else {
                prog += pos as f32;
                if pos >= 51 {
                    lane += 1.0;
                }
            }
        }
        (prog, base, scored, lane)
    }

    let me = player as usize;
    let (mp, mb, ms, ml) = side(me, state);
    let (mut op, mut ob, mut os, mut ol, mut oexp) = (0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32);
    for o in 0..NUM_PLAYERS {
        if o == me || !state.active_players[o] {
            continue;
        }
        let (p, b, s, l) = side(o, state);
        op += p;
        ob += b;
        os += s;
        ol += l;
        oexp += exposure_penalty(o as u8, state) / 15.0;
    }
    let mexp = exposure_penalty(player, state) / 15.0;
    let diff = (mp - op) / 56.0;
    let stage = ((mp + op) / 448.0).clamp(0.0, 1.0);
    [
        diff,
        mb - ob,
        ms - os,
        mexp - oexp,
        ml - ol,
        diff * stage,
        if mb >= 4.0 { 1.0 } else { 0.0 },
        if ob >= 4.0 { 1.0 } else { 0.0 },
        1.0,
    ]
}

/// Fitted logistic weights for [`leaf_features`] (2026-10-02 audit).
///
/// Fit: 5,079,506 pre-roll states from 30,000 Depth2-vs-Depth2 games (equal
/// players, so position value is not confounded by who is stronger); split by
/// GAME (27,000 train / 3,000 validation) since states within a game correlate.
///   validation log-loss 0.5748 (constant baseline 0.6929), AUC 0.7615;
///   calibrated within ~0.01 in every probability decile.
/// prog_diff and prog_diff*stage are collinear, so individual signs are not
/// interpretable — only the combination is. Regenerate with the `dumpfeat`
/// procedure documented in the training journal if the features change.
pub const LEAF_WEIGHTS: [f32; N_LEAF_FEATURES] = [
    -0.336534, -0.304019, -0.0954038, -0.157682, 0.0211639, 2.65214, -0.154032, 0.112321, 0.0549371,
];

/// Win logit for `player` (positive = favoured).
#[inline]
pub fn leaf_logit(player: u8, state: &GameState) -> f32 {
    let f = leaf_features(player, state);
    let mut z = 0.0f32;
    for i in 0..N_LEAF_FEATURES {
        z += LEAF_WEIGHTS[i] * f[i];
    }
    z
}

/// Leaf value in [-1, 1] from `player`'s perspective: 2·P(win) − 1 = tanh(logit/2).
/// Terminal states return exactly ±1.
#[inline]
pub fn leaf_value(player: u8, state: &GameState) -> f32 {
    if state.is_terminal {
        return if state.winner == player as i8 { 1.0 } else { -1.0 };
    }
    (0.5 * leaf_logit(player, state)).tanh()
}
