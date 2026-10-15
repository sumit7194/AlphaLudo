//! Unified (V17) encoder invariants.

use ludo_core::{
    encode_unified_into, uni_action_cell, uni_legal_mask_into, GameState, BASE_COORDS, HOME_BASE_CELLS_4P,
    NUM_BOARD_CELLS, UNI_FEATURES, UNI_FRAME_SIZE,
};
use rand::{Rng, SeedableRng};

fn enc(s: &GameState, pov: u8) -> Vec<i8> {
    let mut v = vec![0i8; UNI_FRAME_SIZE];
    encode_unified_into(s, pov, &mut v);
    v
}

/// Plays random legal moves to reach varied positions.
fn random_state(rng: &mut impl Rng, four: bool, plies: usize) -> GameState {
    let mut s = if four { GameState::new_4p() } else { GameState::new_2p() };
    let mut n = 0;
    while n < plies && !s.is_terminal {
        if !s.roll(rng.gen_range(1..=6)) {
            continue;
        }
        let (legal, c) = s.legal_moves();
        let t = legal[rng.gen_range(0..c)];
        s = s.apply_move(t);
        n += 1;
    }
    // Leave it at a decision point with dice rolled when possible.
    for _ in 0..50 {
        if s.is_terminal || s.roll(rng.gen_range(1..=6)) {
            break;
        }
    }
    s
}

#[test]
fn inactive_seats_are_absent_in_2p() {
    let mut rng = rand::rngs::StdRng::seed_from_u64(1);
    for _ in 0..200 {
        let s = { let n = rng.gen_range(0..120); random_state(&mut rng, false, n) };
        let pov = s.current_player;
        let e = enc(&s, pov);
        // Relative seats 1 and 3 are the inactive seats in a 2-player game.
        for cell in 0..NUM_BOARD_CELLS {
            assert_eq!(e[cell * UNI_FEATURES + 1], -1);
            assert_eq!(e[cell * UNI_FEATURES + 3], -1);
        }
    }
}

#[test]
fn rotation_invariance_4p() {
    // Relabelling every seat by +1 and viewing from mover+1 must give the same frame.
    let mut rng = rand::rngs::StdRng::seed_from_u64(2);
    for _ in 0..300 {
        let s = { let n = rng.gen_range(0..250); random_state(&mut rng, true, n) };
        let mut r = s;
        for p in 0..4 {
            let q = (p + 1) % 4;
            r.positions[q] = s.positions[p];
            r.scores[q] = s.scores[p];
            r.consecutive_sixes[q] = s.consecutive_sixes[p];
            r.active_players[q] = s.active_players[p];
        }
        r.current_player = (s.current_player + 1) % 4;
        assert_eq!(enc(&s, s.current_player), enc(&r, r.current_player));
    }
}

#[test]
fn token_counts_and_masks_are_consistent() {
    let mut rng = rand::rngs::StdRng::seed_from_u64(3);
    let mut mask = vec![0f32; NUM_BOARD_CELLS];
    for four in [false, true] {
        for _ in 0..300 {
            let s = { let n = rng.gen_range(0..250); random_state(&mut rng, four, n) };
            let pov = s.current_player;
            let e = enc(&s, pov);
            // Total tokens on the board + base per active seat = 4 - scored.
            for k in 0..4usize {
                let seat = (pov as usize + k) % 4;
                if !s.active_players[seat] {
                    continue;
                }
                let on_board: i32 = (0..NUM_BOARD_CELLS)
                    .map(|c| e[c * UNI_FEATURES + k].max(0) as i32)
                    .sum::<i32>()
                    // the dice corner and scored cell of seat k are not tokens
                    - (1 + s.consecutive_sixes[seat] as i32)
                    - s.scores[seat] as i32;
                assert_eq!(on_board, 4 - s.scores[seat] as i32, "seat {k} token count");
            }
            // Every legal token's action cell is in the mask and holds one of the mover's tokens.
            let n = uni_legal_mask_into(&s, &mut mask);
            let (legal, c) = s.legal_moves();
            assert!(n >= 1 || c == 0);
            for &t in &legal[..c] {
                let cell = uni_action_cell(&s, t);
                assert_eq!(mask[cell], 1.0);
                assert!(e[cell * UNI_FEATURES] >= 1, "action cell must show a mover token");
            }
        }
    }
}

#[test]
fn base_cells_match_geometry() {
    // HOME_BASE_CELLS_4P[k] (relative seat k in the mover's frame) must be the same
    // quadrant as seat (pov + k)'s real base, for every viewer.
    for pov in 0..4u8 {
        for k in 0..4u8 {
            let seat = (pov + k) % 4;
            let mut real: Vec<(u8, u8)> = BASE_COORDS[seat as usize]
                .iter()
                .map(|&[r, c]| ludo_core::rotate_cell_ccw(r, c, pov))
                .collect();
            let mut table: Vec<(u8, u8)> = HOME_BASE_CELLS_4P[k as usize].iter().map(|&[r, c]| (r, c)).collect();
            real.sort();
            table.sort();
            assert_eq!(real, table, "pov {pov} k {k}");
        }
    }
}
