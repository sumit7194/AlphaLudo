//! Unified 2-player / 4-player encoding for the V17 model family (15 × 15 × 5, mover's point of view).
//!
//! One network plays any seat count: seats are indexed in RELATIVE TURN ORDER from
//! the mover — k = 0 me, 1 next, 2 opposite, 3 previous (k = (seat − mover) mod 4).
//! A 2-player game (seats 0 and 2) therefore uses k = 0 and k = 2.
//!
//! Per cell, feature f[k] for k = 0..3 is seat k's token count on that cell
//! (0..4), or −1 if seat k can never stand there **or seat k is not in the game**.
//! f[4] is safety from the mover's view: 1 safe square on my route, 0 unsafe on my
//! route, −1 not on my route.
//!
//! Bases use the spread-fill rule: with n tokens in base, the first n of that
//! seat's four base cells (canonical order) hold 1, the rest 0.
//!
//! Special cells (never on any route):
//!   dice corner of seat k  (0,0) (0,14) (14,14) (14,0):
//!       f[k] = 1 + consecutive sixes rolled this turn by seat k   (seat active)
//!       f[4] = dice if seat k is to move, else 0                   (seat active)
//!       everything −1 if seat k is not in the game
//!   scored cell of seat k  (7,6) (6,7) (7,8) (8,7):
//!       f[k] = tokens seat k has brought home (−1 if seat inactive)
//!
//! Actions are SOURCE CELLS (225-way): a token's current cell in this frame; every
//! base token maps to the mover's first base cell (2,2). Stacked tokens share a cell
//! (they are the same move).
//!
//! Differences from the September 4P encoder (`encoder::encode_frame_4p_into`):
//! inactive seats are absent instead of showing four tokens in base; base actions
//! no longer depend on token slot; the six streak is visible.

use crate::constants::{get_absolute_pos, is_safe_pos, BASE_POS, BOARD_SIZE, HOME_POS, NUM_BOARD_CELLS};
use crate::encoder::HOME_BASE_CELLS_4P;
use crate::geometry::position_to_cell_in_pov;
use crate::state::GameState;

pub const UNI_FEATURES: usize = 5;
pub const UNI_FRAME_SIZE: usize = NUM_BOARD_CELLS * UNI_FEATURES;
pub const UNI_DICE_CELLS: [[u8; 2]; 4] = [[0, 0], [0, 14], [14, 14], [14, 0]];
pub const UNI_SCORED_CELLS: [[u8; 2]; 4] = [[7, 6], [6, 7], [7, 8], [8, 7]];

#[inline(always)]
fn cell_index(r: u8, c: u8) -> usize {
    (r as usize) * BOARD_SIZE + c as usize
}

/// Board cell index (mover's frame) of seat `owner`'s token at track position
/// `pos` (0..=55) when `pov` is the viewer.
#[inline]
pub fn uni_track_cell(pos: i8, owner: u8, pov: u8) -> usize {
    debug_assert!((0..=55).contains(&pos));
    let (r, c) = position_to_cell_in_pov(pos, owner, pov, 0);
    cell_index(r, c)
}

/// Source cell (0..225) of the mover's token `token` — the policy action for it.
#[inline]
pub fn uni_action_cell(state: &GameState, token: u8) -> usize {
    let cp = state.current_player;
    let pos = state.positions[cp as usize][token as usize];
    if pos == BASE_POS {
        let [r, c] = HOME_BASE_CELLS_4P[0][0];
        cell_index(r, c)
    } else {
        uni_track_cell(pos, cp, cp)
    }
}

/// Encodes `state` from `pov`'s point of view into `out` (len UNI_FRAME_SIZE).
pub fn encode_unified_into(state: &GameState, pov: u8, out: &mut [i8]) {
    debug_assert_eq!(out.len(), UNI_FRAME_SIZE);
    out.fill(-1);
    let f = |cell: usize, k: usize| cell * UNI_FEATURES + k;

    for k in 0..4usize {
        let seat = ((pov as usize) + k) % 4;
        if !state.active_players[seat] {
            continue;
        }
        // Route: track 0..=55 and the four base cells are reachable (0 = empty).
        for pos in 0..=55i8 {
            out[f(uni_track_cell(pos, seat as u8, pov), k)] = 0;
        }
        for &[r, c] in HOME_BASE_CELLS_4P[k].iter() {
            out[f(cell_index(r, c), k)] = 0;
        }
        // Tokens.
        let mut in_base = 0usize;
        for &pos in state.positions[seat].iter() {
            if pos == BASE_POS {
                in_base += 1;
            } else if pos != HOME_POS {
                out[f(uni_track_cell(pos, seat as u8, pov), k)] += 1;
            }
        }
        for (i, &[r, c]) in HOME_BASE_CELLS_4P[k].iter().enumerate() {
            out[f(cell_index(r, c), k)] = (i < in_base) as i8;
        }
    }

    // Safety, mover's route only (the mover is always active).
    for pos in 0..=55i8 {
        let cell = uni_track_cell(pos, pov, pov);
        let safe = if pos > 50 {
            true
        } else {
            is_safe_pos(get_absolute_pos(pov, pos).expect("track position"))
        };
        out[f(cell, 4)] = safe as i8;
    }
    for &[r, c] in HOME_BASE_CELLS_4P[0].iter() {
        out[f(cell_index(r, c), 4)] = 1;
    }

    // Special cells.
    for k in 0..4usize {
        let seat = ((pov as usize) + k) % 4;
        if !state.active_players[seat] {
            continue;
        }
        let [dr, dc] = UNI_DICE_CELLS[k];
        let dcell = cell_index(dr, dc);
        out[f(dcell, k)] = 1 + state.consecutive_sixes[seat] as i8;
        out[f(dcell, 4)] = if state.current_player as usize == seat { state.current_dice_roll as i8 } else { 0 };
        let [sr, sc] = UNI_SCORED_CELLS[k];
        out[f(cell_index(sr, sc), k)] = state.scores[seat] as i8;
    }
}

/// Legal-action mask over the 225 source cells for the current player.
/// Returns the number of distinct legal cells.
pub fn uni_legal_mask_into(state: &GameState, mask: &mut [f32]) -> usize {
    debug_assert_eq!(mask.len(), NUM_BOARD_CELLS);
    mask.fill(0.0);
    let (legal, count) = state.legal_moves();
    let mut n = 0;
    for &t in &legal[..count] {
        let c = uni_action_cell(state, t);
        if mask[c] == 0.0 {
            mask[c] = 1.0;
            n += 1;
        }
    }
    n
}
