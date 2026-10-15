//! Calibrated 4-player win-share evaluator (conditional logit over seats).
//!
//! WHY (2026-10-07): `evaluator::leaf_logit` was fitted on 2-player games and,
//! in 4P, compares "me" against the SUM of all three opponents — "their tokens
//! all in base" fires whenever the three together have 4 in base, the stage term
//! saturates at half the game, and the softmax over seats is ~2x overconfident.
//! Max^n MCTS on top of it stopped scaling at ~1000 simulations (34% vs 3x
//! Expectimax at N=1000, 3000 and 10000 alike).
//!
//! Model: every ACTIVE seat k gets a feature vector phi_k from its own point of
//! view; P(k wins) = softmax_k(w · phi_k) over active seats. Anything identical
//! for all seats cancels in the softmax, so the model can only express real
//! differences between players (no "me vs the sum of the others").
//!
//! The fit and the search share THIS feature code (`seat_features`), so they
//! cannot drift apart. Features are taken at pre-roll points (dice = 0), which is
//! exactly the state at an MCTS chance node / after a rollout move.

use ludo_core::{get_absolute_pos, is_safe_pos, GameState, BASE_POS, HOME_POS, NUM_PLAYERS, NUM_TOKENS};

/// Number of per-seat features.
pub const N_E4: usize = 16;

/// Feature names, index-aligned with [`seat_features`] (used by the fitting script).
pub const E4_NAMES: [&str; N_E4] = [
    "progress",         // sum of token progress / 224 (base 0, home 56)
    "base",             // tokens in base / 4
    "scored",           // tokens home / 4
    "lane",             // tokens in the home lane (51..55) / 4
    "exposed",          // own unsafe single tokens an active opponent can hit with 1..6 / 4
    "threats",          // opponent unsafe single tokens this seat can hit with 1..6 / 4
    "progress_x_stage", // progress × game stage (mean progress of active seats)
    "all_in_base",      // 1 if no token is out (needs a six to move at all)
    "laggard",          // progress of the least advanced token / 56 (base = 0)
    "order0",           // this seat rolls next (turn-order distance 0)
    "order1",           // distance 1 among active seats
    "order2",           // distance 2 among active seats (4P only)
    "risk",             // sum over own hittable tokens of P(hit before my next turn) x progress/56, / 4
    "risk_x_order0",    // risk while it is this seat's turn (it may still move away)
    "threat_stake",     // sum over hittable opponent tokens of P(this seat rolls a hit) x their progress/56, / 4
    "progress_sq",      // progress^2 (lead matters non-linearly)
];

/// Fitted weights (see `fit_eval4.py`; numbers recorded in the training journal).
/// Zero-initialised until fitted: with all-zero weights the evaluator returns a
/// uniform share, which is a safe (if useless) default.
pub const E4_WEIGHTS: [f32; N_E4] = [0.0; N_E4];

#[inline(always)]
fn track_abs(player: usize, pos: i8) -> Option<u8> {
    get_absolute_pos(player as u8, pos)
}

/// Per-seat features for every seat (inactive seats are left at zero).
/// Zero allocation; one pass over the 16 tokens plus the pairwise threat scan.
#[inline]
pub fn seat_features(state: &GameState) -> [[f32; N_E4]; NUM_PLAYERS] {
    let mut out = [[0.0f32; N_E4]; NUM_PLAYERS];

    // Absolute main-track cell (0..51) of each token, or 255 if not on the loop.
    let mut abs = [[255u8; NUM_TOKENS]; NUM_PLAYERS];
    // Tokens of each seat per absolute cell are needed to detect blockades (2+ of
    // the same colour on one cell are immune). At most 4 tokens, so count by scan.
    let mut progress = [0.0f32; NUM_PLAYERS];
    let mut n_active = 0usize;
    for p in 0..NUM_PLAYERS {
        if !state.active_players[p] {
            continue;
        }
        n_active += 1;
        let f = &mut out[p];
        let mut min_prog = 56.0f32;
        for t in 0..NUM_TOKENS {
            let pos = state.positions[p][t];
            let prog = if pos == BASE_POS {
                f[1] += 0.25;
                0.0
            } else if pos == HOME_POS {
                f[2] += 0.25;
                56.0
            } else {
                if pos >= 51 {
                    f[3] += 0.25;
                }
                if let Some(a) = track_abs(p, pos) {
                    abs[p][t] = a;
                }
                pos as f32
            };
            progress[p] += prog;
            min_prog = min_prog.min(prog);
        }
        f[0] = progress[p] / 224.0;
        f[7] = if f[1] >= 0.999 { 1.0 } else { 0.0 };
        f[8] = min_prog / 56.0;
    }

    // Hittable = alone (no own blockade) on an unsafe main-track cell, with an
    // active opponent token on the loop 1..6 cells behind it that stays on the
    // loop for that many steps. Distances are collected as a bitmask per
    // attacker so P(attacker rolls one of them) = |set| / 6.
    for p in 0..NUM_PLAYERS {
        if !state.active_players[p] {
            continue;
        }
        for t in 0..NUM_TOKENS {
            let a = abs[p][t];
            if a == 255 || is_safe_pos(a) {
                continue;
            }
            let mut stack = 0;
            for u in 0..NUM_TOKENS {
                if abs[p][u] == a {
                    stack += 1;
                }
            }
            if stack >= 2 {
                continue;
            }
            let stake = state.positions[p][t] as f32 / 56.0;
            let mut p_safe = 1.0f32;
            let mut hit = false;
            for q in 0..NUM_PLAYERS {
                if q == p || !state.active_players[q] {
                    continue;
                }
                let mut dmask = 0u8;
                for u in 0..NUM_TOKENS {
                    let b = abs[q][u];
                    if b == 255 {
                        continue;
                    }
                    let d = (a as i16 - b as i16).rem_euclid(52);
                    if (1..=6).contains(&d) && state.positions[q][u] as i16 + d <= 50 {
                        dmask |= 1 << d;
                    }
                }
                if dmask != 0 {
                    let pq = dmask.count_ones() as f32 / 6.0;
                    out[q][5] += 0.25;
                    out[q][14] += 0.25 * pq * stake;
                    p_safe *= 1.0 - pq;
                    hit = true;
                }
            }
            if hit {
                out[p][4] += 0.25;
                out[p][12] += 0.25 * (1.0 - p_safe) * stake;
            }
        }
    }

    // Game stage and turn order.
    let mut total = 0.0f32;
    for p in 0..NUM_PLAYERS {
        if state.active_players[p] {
            total += progress[p];
        }
    }
    let stage = (total / (224.0 * n_active.max(1) as f32)).clamp(0.0, 1.0);
    let mover = state.current_player as usize;
    let mut dist = 0usize;
    let mut s = mover;
    for _ in 0..NUM_PLAYERS {
        if state.active_players[s] {
            let f = &mut out[s];
            f[6] = f[0] * stage;
            f[15] = f[0] * f[0];
            match dist {
                0 => {
                    f[9] = 1.0;
                    f[13] = f[12];
                }
                1 => f[10] = 1.0,
                2 => f[11] = 1.0,
                _ => {}
            }
            dist += 1;
        }
        s = (s + 1) % NUM_PLAYERS;
    }
    out
}

/// Win logits per seat with the given weights (inactive seats: -inf).
#[inline]
pub fn e4_logits_with(state: &GameState, w: &[f32; N_E4]) -> [f32; NUM_PLAYERS] {
    let f = seat_features(state);
    let mut z = [f32::NEG_INFINITY; NUM_PLAYERS];
    for p in 0..NUM_PLAYERS {
        if state.active_players[p] {
            let mut s = 0.0f32;
            for i in 0..N_E4 {
                s += w[i] * f[p][i];
            }
            z[p] = s;
        }
    }
    z
}

/// Win shares (sum to 1 over active seats). Terminal states are exact one-hots.
#[inline]
pub fn e4_probs_with(state: &GameState, w: &[f32; N_E4]) -> [f32; NUM_PLAYERS] {
    let mut out = [0.0f32; NUM_PLAYERS];
    if state.is_terminal && state.winner >= 0 {
        out[state.winner as usize] = 1.0;
        return out;
    }
    let z = e4_logits_with(state, w);
    let mx = z.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for p in 0..NUM_PLAYERS {
        if state.active_players[p] {
            out[p] = (z[p] - mx).exp();
            sum += out[p];
        }
    }
    for v in out.iter_mut() {
        *v /= sum.max(1e-12);
    }
    out
}

/// Win shares with the fitted weights.
#[inline]
pub fn e4_probs(state: &GameState) -> [f32; NUM_PLAYERS] {
    e4_probs_with(state, &E4_WEIGHTS)
}
