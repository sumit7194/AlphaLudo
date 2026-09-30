//! Board coordinate transformations, 90-degree rotations, and cell indexing.

use crate::constants::{
    BASE_COORDS, BASE_POS, BOARD_SIZE, HOME_COORD_P0, HOME_POS, HOME_RUN_LENGTH, HOME_RUN_P0,
    NUM_BOARD_CELLS, PATH_COORDS_P0,
};

#[inline(always)]
pub fn cell_to_index(row: u8, col: u8) -> usize {
    debug_assert!((row as usize) < BOARD_SIZE && (col as usize) < BOARD_SIZE);
    (row as usize) * BOARD_SIZE + (col as usize)
}

#[inline(always)]
pub fn index_to_cell(idx: usize) -> (u8, u8) {
    debug_assert!(idx < NUM_BOARD_CELLS);
    ((idx / BOARD_SIZE) as u8, (idx % BOARD_SIZE) as u8)
}

/// Rotates (row, col) k times 90° CCW around the board center (7, 7).
/// One CCW rotation: `(r, c) -> (14 - c, r)`.
#[inline]
pub fn rotate_cell_ccw(mut row: u8, mut col: u8, k: u8) -> (u8, u8) {
    for _ in 0..(k % 4) {
        let new_r = 14 - col;
        let new_c = row;
        row = new_r;
        col = new_c;
    }
    (row, col)
}

/// Rotates (row, col) k times 90° CW around the board center (7, 7).
/// One CW rotation: `(r, c) -> (c, 14 - r)`.
#[inline]
pub fn rotate_cell_cw(mut row: u8, mut col: u8, k: u8) -> (u8, u8) {
    for _ in 0..(k % 4) {
        let new_r = col;
        let new_c = 14 - row;
        row = new_r;
        col = new_c;
    }
    (row, col)
}

/// Computes the actual board cell (row, col) where `player`'s token at `pos` sits.
pub fn position_to_actual_cell(player: u8, pos: i8, slot_hint: usize) -> (u8, u8) {
    let (local_r, local_c) = if pos == BASE_POS {
        let slot = if slot_hint < 4 { slot_hint } else { 0 };
        return (BASE_COORDS[player as usize][slot][0], BASE_COORDS[player as usize][slot][1]);
    } else if pos == HOME_POS {
        (HOME_COORD_P0[0], HOME_COORD_P0[1])
    } else if pos > 50 {
        let idx = (pos - 51) as usize;
        if idx < (HOME_RUN_LENGTH as usize) {
            (HOME_RUN_P0[idx][0], HOME_RUN_P0[idx][1])
        } else {
            (HOME_COORD_P0[0], HOME_COORD_P0[1])
        }
    } else {
        (PATH_COORDS_P0[pos as usize][0], PATH_COORDS_P0[pos as usize][1])
    };

    // Rotate player times CW around (7, 7)
    rotate_cell_cw(local_r, local_c, player)
}

/// Returns the cell of `token_owner`'s token at `pos`, as seen from `pov_player`'s perspective
/// (i.e. board rotated so pov_player is Red / Player 0).
#[inline]
pub fn position_to_cell_in_pov(pos: i8, token_owner: u8, pov_player: u8, slot_hint: usize) -> (u8, u8) {
    let (act_r, act_c) = position_to_actual_cell(token_owner, pos, slot_hint);
    rotate_cell_ccw(act_r, act_c, pov_player)
}
