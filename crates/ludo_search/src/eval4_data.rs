//! Training data for the 4P evaluator (`eval4`): bot-vs-bot games played in
//! parallel (rayon), sampling pre-roll states with their per-seat features and
//! the eventual winner. Also records the legacy 2P-fitted payoff vector so the
//! two evaluators can be compared on identical positions.

use ludo_core::{GameState, NUM_PLAYERS};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

use crate::eval4::{seat_features, N_E4};
use crate::evaluator::leaf_logit;
use crate::farm::{make_bot, BotKind};

pub struct E4Sample {
    pub game: u32,
    pub feats: [[f32; N_E4]; NUM_PLAYERS],
    pub active: [bool; NUM_PLAYERS],
    pub legacy: [f32; NUM_PLAYERS],
    /// Winner seat (0..3); games that hit the move cap are dropped.
    pub winner: u8,
    /// Fraction of the game elapsed (move index / total moves), for diagnostics.
    pub t: f32,
}

fn legacy_shares(s: &GameState) -> [f32; NUM_PLAYERS] {
    let mut z = [f32::NEG_INFINITY; NUM_PLAYERS];
    let mut mx = f32::NEG_INFINITY;
    for p in 0..NUM_PLAYERS {
        if s.active_players[p] {
            z[p] = leaf_logit(p as u8, s);
            mx = mx.max(z[p]);
        }
    }
    let mut out = [0.0; NUM_PLAYERS];
    let mut sum = 0.0;
    for p in 0..NUM_PLAYERS {
        if s.active_players[p] {
            out[p] = (z[p] - mx).exp();
            sum += out[p];
        }
    }
    for v in out.iter_mut() {
        *v /= sum;
    }
    out
}

/// Plays `games` games with `seats[k]` at seat k (2 or 4 entries; seat order is
/// rotated per game so no bot keeps the first-move advantage) and keeps each
/// pre-roll state with probability `keep_prob`.
pub fn generate(seats: &[BotKind], games: usize, keep_prob: f64, max_moves: usize, seed: u64) -> Vec<E4Sample> {
    let n = seats.len();
    assert!(n == 2 || n == 4);
    (0..games)
        .into_par_iter()
        .map(|g| {
            let mut rng = StdRng::seed_from_u64(seed ^ (g as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
            let mut s = if n == 4 { GameState::new_4p() } else { GameState::new_2p() };
            let seat_ids: Vec<u8> = if n == 4 { vec![0, 1, 2, 3] } else { vec![0, 2] };
            let rot = rng.gen_range(0..n);
            let mut bots: Vec<Option<Box<dyn crate::expectimax::LudoBot>>> = (0..4).map(|_| None).collect();
            for (i, &seat) in seat_ids.iter().enumerate() {
                let kind = seats[(i + rot) % n];
                bots[seat as usize] = Some(make_bot(kind, seat, n, rng.gen()));
            }
            let mut out: Vec<E4Sample> = Vec::new();
            let mut moves = 0usize;
            while !s.is_terminal && moves < max_moves {
                if s.current_dice_roll == 0 && rng.gen_bool(keep_prob) {
                    out.push(E4Sample {
                        game: g as u32,
                        feats: seat_features(&s),
                        active: s.active_players,
                        legacy: legacy_shares(&s),
                        winner: 255,
                        t: moves as f32,
                    });
                }
                if !s.roll(rng.gen_range(1..=6)) {
                    continue;
                }
                let (legal, count) = s.legal_moves();
                let a = if count == 1 {
                    legal[0]
                } else {
                    bots[s.current_player as usize].as_ref().unwrap().select_move(&s)
                };
                s = s.apply_move(a);
                moves += 1;
            }
            if !s.is_terminal {
                return Vec::new();
            }
            let total = moves.max(1) as f32;
            for r in out.iter_mut() {
                r.winner = s.winner as u8;
                r.t /= total;
            }
            out
        })
        .flatten()
        .collect()
}
