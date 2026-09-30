//! Board constants and coordinate lookup tables for 4-Player Ludo.

pub const BOARD_SIZE: usize = 15;
pub const NUM_BOARD_CELLS: usize = BOARD_SIZE * BOARD_SIZE; // 225
pub const NUM_PLAYERS: usize = 4;
pub const NUM_TOKENS: usize = 4;
pub const PATH_LENGTH: u8 = 52;
pub const HOME_RUN_LENGTH: u8 = 5;

pub const BASE_POS: i8 = -1;
pub const HOME_POS: i8 = 99;

/// The 8 safe squares on the 52-cell main loop (Player 0 canonical perspective).
pub const SAFE_INDICES: [u8; 8] = [0, 8, 13, 21, 26, 34, 39, 47];

#[inline(always)]
pub fn is_safe_pos(abs_pos: u8) -> bool {
    matches!(abs_pos, 0 | 8 | 13 | 21 | 26 | 34 | 39 | 47)
}

/// Converts relative position on main track (0..50) to absolute 52-loop index (0..51).
/// Returns None if relative_pos is in base (-1) or in home stretch / home (> 50).
#[inline(always)]
pub fn get_absolute_pos(player: u8, relative_pos: i8) -> Option<u8> {
    if relative_pos < 0 || relative_pos > 50 {
        None
    } else {
        Some(((relative_pos as u8) + 13 * player) % PATH_LENGTH)
    }
}

/// Player 0's 51-cell main track. Position i (0..50) -> [row, col].
pub const PATH_COORDS_P0: [[u8; 2]; 51] = [
    [6, 1],  [6, 2],  [6, 3],  [6, 4],  [6, 5],           // 0-4
    [5, 6],  [4, 6],  [3, 6],  [2, 6],  [1, 6],  [0, 6],  // 5-10
    [0, 7],  [0, 8],                                      // 11-12
    [1, 8],  [2, 8],  [3, 8],  [4, 8],  [5, 8],           // 13-17
    [6, 9],  [6, 10], [6, 11], [6, 12], [6, 13], [6, 14], // 18-23
    [7, 14], [8, 14],                                     // 24-25
    [8, 13], [8, 12], [8, 11], [8, 10], [8, 9],           // 26-30
    [9, 8],  [10, 8], [11, 8], [12, 8], [13, 8], [14, 8], // 31-36
    [14, 7], [14, 6],                                     // 37-38
    [13, 6], [12, 6], [11, 6], [10, 6], [9, 6],           // 39-43
    [8, 5],  [8, 4],  [8, 3],  [8, 2],  [8, 1],  [8, 0],  // 44-49
    [7, 0],                                               // 50 (end of main track)
];

/// Player 0's 5 home stretch cells (positions 51..55) -> [row, col].
pub const HOME_RUN_P0: [[u8; 2]; 5] = [
    [7, 1], [7, 2], [7, 3], [7, 4], [7, 5]
];

/// Player 0's HOME goal center.
pub const HOME_COORD_P0: [u8; 2] = [7, 6];

/// Base corner coordinates per [player][token_slot][row, col].
/// Symmetric coordinates post-rotation:
/// P0 sees its base tokens at (2,2), (2,3), (3,2), (3,3).
pub const BASE_COORDS: [[[u8; 2]; 4]; 4] = [
    [[2, 2],  [2, 3],  [3, 2],  [3, 3]],      // P0 (Red)
    [[2, 12], [3, 12], [2, 11], [3, 11]],     // P1 (Green)
    [[12, 12],[12, 11],[11, 12],[11, 11]],    // P2 (Yellow)
    [[12, 2], [11, 2], [12, 3], [11, 3]],     // P3 (Blue)
];

/// 4-Player Dice display cells in current player (P0) POV.
pub const DICE_CELLS_4P: [[u8; 2]; 4] = [
    [0, 0],   // P0 (Me)
    [0, 14],  // P1 (Next)
    [14, 14], // P2 (Opp)
    [14, 0],  // P3 (Prev)
];

/// 4-Player Scored tokens display cells in current player (P0) POV.
pub const SCORED_CELLS_4P: [[u8; 2]; 4] = [
    [7, 6], // P0
    [6, 7], // P1
    [7, 8], // P2
    [8, 7], // P3
];
