//! Zero-copy (15, 15, 5) state encoder for AlphaLudo V15-4P neural network input.

use crate::constants::{
    is_safe_pos, BASE_POS, BOARD_SIZE, DICE_CELLS_4P, HOME_POS, PATH_COORDS_P0, SCORED_CELLS_4P,
};
use crate::geometry::position_to_cell_in_pov;
use crate::state::GameState;

pub const HOME_BASE_CELLS_4P: [[[u8; 2]; 4]; 4] = [
    [[2, 2],  [2, 3],  [3, 2],  [3, 3]],      // P0 (Top-Left)
    [[2, 12], [3, 12], [2, 11], [3, 11]],     // P1 (Top-Right)
    [[12, 12],[12, 11],[11, 12],[11, 11]],    // P2 (Bottom-Right)
    [[12, 2], [11, 2], [12, 3], [11, 3]],     // P3 (Bottom-Left)
];

pub const HOME_STRETCH_CELLS_4P: [[[u8; 2]; 5]; 4] = [
    [[7, 1], [7, 2], [7, 3], [7, 4], [7, 5]],      // P0 (runs East to 7,6)
    [[1, 7], [2, 7], [3, 7], [4, 7], [5, 7]],      // P1 (runs South to 6,7)
    [[7, 13], [7, 12], [7, 11], [7, 10], [7, 9]],  // P2 (runs West to 7,8)
    [[13, 7], [12, 7], [11, 7], [10, 7], [9, 7]],  // P3 (runs North to 8,7)
];

/// Encodes a GameState into an existing `[i8; 15 * 15 * 5]` buffer in mover POV.
/// Zero heap allocations, pure SIMD-friendly sequential writes.
#[inline]
pub fn encode_frame_4p_into(state: &GameState, pov_player: Option<u8>, out: &mut [i8]) {
    debug_assert_eq!(out.len(), BOARD_SIZE * BOARD_SIZE * 5);

    let pov = pov_player.unwrap_or(state.current_player);
    let rel_players = [
        pov,
        (pov + 1) % 4,
        (pov + 2) % 4,
        (pov + 3) % 4,
    ];

    // Initialize all cells to -1 sentinel
    out.fill(-1);

    #[inline(always)]
    fn set_cell(out: &mut [i8], r: u8, c: u8, slot: usize, val: i8) {
        let idx = ((r as usize) * BOARD_SIZE + (c as usize)) * 5 + slot;
        out[idx] = val;
    }

    #[inline(always)]
    fn set_cell_all(out: &mut [i8], r: u8, c: u8, vals: [i8; 5]) {
        let idx = ((r as usize) * BOARD_SIZE + (c as usize)) * 5;
        out[idx..idx + 5].copy_from_slice(&vals);
    }

    // 1. Shared main path cells (52 cells in P0 POV)
    for path_pos in 0..51 {
        let [r, c] = PATH_COORDS_P0[path_pos];
        let safe = if is_safe_pos(path_pos as u8) { 1 } else { 0 };
        set_cell_all(out, r, c, [0, 0, 0, 0, safe]);
    }

    // Outer loop closure at (6, 0)
    set_cell_all(out, 6, 0, [-1, 0, 0, 0, -1]);

    // 2. Mark home bases for all 4 players
    for rel_idx in 0..4 {
        for &[r, c] in &HOME_BASE_CELLS_4P[rel_idx] {
            let mut base_tuple = [-1, -1, -1, -1, if rel_idx == 0 { 1 } else { -1 }];
            base_tuple[rel_idx] = 0;
            set_cell_all(out, r, c, base_tuple);
        }
    }

    // 3. Mark home stretches for all 4 players
    for rel_idx in 0..4 {
        for &[r, c] in &HOME_STRETCH_CELLS_4P[rel_idx] {
            let mut stretch_tuple = [-1, -1, -1, -1, if rel_idx == 0 { 1 } else { -1 }];
            stretch_tuple[rel_idx] = 0;
            set_cell_all(out, r, c, stretch_tuple);
        }
    }

    // 4. Count tokens and apply spread-fill
    for rel_idx in 0..4 {
        let p_actual = rel_players[rel_idx] as usize;
        let positions = state.positions[p_actual];

        // Home base spread-fill: N tokens in base fill first N cells of base with 1
        let mut at_base = 0;
        for &pos in &positions {
            if pos == BASE_POS {
                at_base += 1;
            }
        }
        for (cell_idx, &[r, c]) in HOME_BASE_CELLS_4P[rel_idx].iter().enumerate() {
            let val = if cell_idx < at_base { 1 } else { 0 };
            set_cell(out, r, c, rel_idx, val);
        }

        // Non-base, non-home tokens: accumulate into board cell
        for &pos in &positions {
            if pos == HOME_POS || pos == BASE_POS {
                continue;
            }
            let (r, c) = position_to_cell_in_pov(pos, p_actual as u8, pov, 0);
            let idx = ((r as usize) * BOARD_SIZE + (c as usize)) * 5 + rel_idx;
            let cur = out[idx];
            out[idx] = if cur < 0 { 1 } else { cur + 1 };
        }
    }

    // 5. Special-cell overrides: Dice corners & Scored centers
    let cp = state.current_player;
    let dice = state.current_dice_roll as i8;

    // 4 Dice Corners
    for rel_idx in 0..4 {
        let p_actual = rel_players[rel_idx];
        let [r, c] = DICE_CELLS_4P[rel_idx];
        if cp == p_actual && dice > 0 {
            set_cell_all(out, r, c, [-1, -1, -1, -1, dice]);
        } else {
            set_cell_all(out, r, c, [-1, -1, -1, -1, -1]);
        }
    }

    // 4 Scored Centers
    for rel_idx in 0..4 {
        let p_actual = rel_players[rel_idx] as usize;
        let score_val = state.scores[p_actual] as i8;
        let [r, c] = SCORED_CELLS_4P[rel_idx];
        let mut scored_tuple = [-1, -1, -1, -1, -1];
        scored_tuple[rel_idx] = score_val;
        set_cell_all(out, r, c, scored_tuple);
    }
}

/// Allocates and returns a fresh `(15, 15, 5)` encoded buffer.
#[inline]
pub fn encode_frame_4p(state: &GameState, pov_player: Option<u8>) -> [i8; 15 * 15 * 5] {
    let mut buf = [-1i8; BOARD_SIZE * BOARD_SIZE * 5];
    encode_frame_4p_into(state, pov_player, &mut buf);
    buf
}

pub const V17_CHANNELS: usize = 17;
pub const V17_FRAME_SIZE: usize = V17_CHANNELS * BOARD_SIZE * BOARD_SIZE;

pub const SAFE_CELLS_V17: [[usize; 2]; 8] = [
    [1, 8], [2, 6], [6, 1], [6, 12], [8, 2], [8, 13], [12, 8], [13, 6],
];
pub const MY_HOME_PATH_V17: [[usize; 2]; 5] = [
    [7, 1], [7, 2], [7, 3], [7, 4], [7, 5],
];
pub const OPP_HOME_PATH_V17: [[usize; 2]; 5] = [
    [7, 9], [7, 10], [7, 11], [7, 12], [7, 13],
];

/// Encodes a 2-player GameState into a (17, 15, 15) float32 buffer matching V13.2 / V17 encoder.
/// Zero heap allocations, pure contiguous writes.
pub fn encode_state_v17_into(state: &GameState, out: &mut [f32]) {
    debug_assert_eq!(out.len(), V17_FRAME_SIZE);
    out.fill(0.0);

    let cp = state.current_player;
    let mut opp = (cp + 2) % 4;
    for offset in 1..4 {
        let cand = (cp + offset) % 4;
        if state.active_players[cand as usize] {
            opp = cand;
            break;
        }
    }

    let spatial_size = BOARD_SIZE * BOARD_SIZE; // 225

    // Channels 0-3: My Tokens (1.0 at token cell in mover's POV)
    for t in 0..4 {
        let pos = state.positions[cp as usize][t];
        let (r, c) = position_to_cell_in_pov(pos, cp, cp, t);
        let idx = t * spatial_size + (r as usize) * BOARD_SIZE + (c as usize);
        out[idx] += 1.0;
    }

    // Channels 4-7: Opponent Tokens (1.0 at opp token cell in mover's POV)
    for t in 0..4 {
        let pos = state.positions[opp as usize][t];
        let (r, c) = position_to_cell_in_pov(pos, opp, cp, t);
        let idx = (4 + t) * spatial_size + (r as usize) * BOARD_SIZE + (c as usize);
        out[idx] += 1.0;
    }

    // Channels 8-13: Dice Roll (one-hot plane across all 225 cells)
    let roll = state.current_dice_roll;
    if roll >= 1 && roll <= 6 {
        let ch = 8 + (roll - 1) as usize;
        out[ch * spatial_size..(ch + 1) * spatial_size].fill(1.0);
    }

    // Channel 14: Safe Squares (8 cells with 0.5)
    for &[r, c] in &SAFE_CELLS_V17 {
        out[14 * spatial_size + r * BOARD_SIZE + c] = 0.5;
    }

    // Channel 15: My Home Path (5 cells with 1.0)
    for &[r, c] in &MY_HOME_PATH_V17 {
        out[15 * spatial_size + r * BOARD_SIZE + c] = 1.0;
    }

    // Channel 16: Opp Home Path (5 cells with 1.0)
    for &[r, c] in &OPP_HOME_PATH_V17 {
        out[16 * spatial_size + r * BOARD_SIZE + c] = 1.0;
    }
}


// ─────────────────────────────────────────────────────────────────────────────
// V15 2-player per-cell triplet encoder (15, 15, 3) — bit-exact port of
// td_ludo_v15/game/encoder.py::encode_frame (verified against it on real
// shard states). Lets search engines evaluate the V15 GraphTransformer
// without a Python round-trip per position.
// ─────────────────────────────────────────────────────────────────────────────
pub const V15_FRAME_SIZE: usize = BOARD_SIZE * BOARD_SIZE * 3;
const V15_MD: [u8; 2] = [0, 0];
const V15_OD: [u8; 2] = [14, 14];
const V15_MS: [u8; 2] = [7, 6];
const V15_OS: [u8; 2] = [7, 8];
/// Mover-POV cell of every own base token (the Python cell helper maps BASE to
/// the canonical home counter (2, 2), regardless of token slot).
pub const V15_BASE_CELL: usize = 2 * BOARD_SIZE + 2;

#[inline(always)]
fn v15_set(out: &mut [i8], r: u8, c: u8, v: [i8; 3]) {
    let i = ((r as usize) * BOARD_SIZE + c as usize) * 3;
    out[i] = v[0];
    out[i + 1] = v[1];
    out[i + 2] = v[2];
}

/// The other active seat (2-player).
#[inline]
pub fn v15_opp_2p(state: &GameState, pov: u8) -> u8 {
    for p in 0..4u8 {
        if p != pov && state.active_players[p as usize] {
            return p;
        }
    }
    pov
}

/// Mover-POV cell index (0..225) of `pov`'s own token at `pos` (not HOME).
#[inline]
pub fn v15_own_cell(pos: i8, pov: u8) -> usize {
    if pos == BASE_POS {
        return V15_BASE_CELL;
    }
    let (r, c) = position_to_cell_in_pov(pos, pov, pov, 0);
    (r as usize) * BOARD_SIZE + c as usize
}

/// Encodes `state` from `pov`'s perspective into a (15, 15, 3) int8 buffer.
pub fn encode_v15_2p_into(state: &GameState, pov: u8, out: &mut [i8]) {
    debug_assert_eq!(out.len(), V15_FRAME_SIZE);
    out.fill(-1);
    let opp = v15_opp_2p(state, pov);

    // 1. main path (own route): a=0, b=0, c=safe
    for (pos, rc) in PATH_COORDS_P0.iter().enumerate() {
        let safe = matches!(pos, 0 | 8 | 13 | 21 | 26 | 34 | 39 | 47);
        v15_set(out, rc[0], rc[1], [0, 0, safe as i8]);
    }
    // 2-3. my home base + home stretch
    for rc in HOME_BASE_CELLS_4P[0].iter() {
        v15_set(out, rc[0], rc[1], [0, -1, 1]);
    }
    for rc in HOME_STRETCH_CELLS_4P[0].iter() {
        v15_set(out, rc[0], rc[1], [0, -1, 1]);
    }
    // 4-5. opp home base + home stretch
    for rc in HOME_BASE_CELLS_4P[2].iter() {
        v15_set(out, rc[0], rc[1], [-1, 0, -1]);
    }
    for rc in HOME_STRETCH_CELLS_4P[2].iter() {
        v15_set(out, rc[0], rc[1], [-1, 0, -1]);
    }
    // 6. tokens
    let own = &state.positions[pov as usize];
    let oppp = &state.positions[opp as usize];
    let own_base = own.iter().filter(|&&p| p == BASE_POS).count();
    let opp_base = oppp.iter().filter(|&&p| p == BASE_POS).count();
    for (i, rc) in HOME_BASE_CELLS_4P[0].iter().enumerate() {
        out[((rc[0] as usize) * BOARD_SIZE + rc[1] as usize) * 3] = (i < own_base) as i8;
    }
    for (i, rc) in HOME_BASE_CELLS_4P[2].iter().enumerate() {
        out[((rc[0] as usize) * BOARD_SIZE + rc[1] as usize) * 3 + 1] = (i < opp_base) as i8;
    }
    for &pos in own.iter() {
        if pos == HOME_POS || pos == BASE_POS {
            continue;
        }
        let (r, c) = position_to_cell_in_pov(pos, pov, pov, 0);
        let i = ((r as usize) * BOARD_SIZE + c as usize) * 3;
        out[i] = out[i].max(0) + 1;
    }
    for &pos in oppp.iter() {
        if pos == HOME_POS || pos == BASE_POS {
            continue;
        }
        let (r, c) = position_to_cell_in_pov(pos, opp, pov, 0);
        let i = ((r as usize) * BOARD_SIZE + c as usize) * 3 + 1;
        out[i] = out[i].max(0) + 1;
    }
    // 7. special cells
    let cp = state.current_player;
    let dice = state.current_dice_roll as i8;
    v15_set(out, V15_MD[0], V15_MD[1], [-1, -1, if cp == pov && dice > 0 { dice } else { -1 }]);
    v15_set(out, V15_OD[0], V15_OD[1], [-1, -1, if cp == opp && dice > 0 { dice } else { -1 }]);
    let scored = |p: u8| state.positions[p as usize].iter().filter(|&&x| x == HOME_POS).count() as i8;
    v15_set(out, V15_MS[0], V15_MS[1], [scored(pov), -1, -1]);
    v15_set(out, V15_OS[0], V15_OS[1], [scored(opp), -1, -1]);
}
