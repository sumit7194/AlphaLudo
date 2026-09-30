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
