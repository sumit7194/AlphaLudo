//! Full-round expectimax for 2P/4P Ludo ("round" bot) — exact, no sampling.
//!
//! WHY (2026-10-07): Max^n MCTS in 4P did not improve with search (≈34% vs
//! 3x Expectimax at 1k, 3k and 10k simulations). Between two of my decisions
//! three opponents each roll; MCTS samples that 6×6×6 round sparsely and its
//! estimate stays biased by the rollout policy. Here the round is enumerated
//! EXACTLY: for each candidate move, every dice face of every opponent (and the
//! bonus rolls after sixes, captures and arrivals, up to `max_rolls` chance
//! layers) is expanded; each opponent answers greedily for ITSELF (Max^n
//! semantics, 1-ply on its own leaf value); the value of the move is the exact
//! expectation of MY leaf value at the start of my next turn (or the terminal
//! payoff). Zero allocation, depth-first.
//!
//! Leaf: `eval4` win shares (default) or the legacy `score_position` scale.

use ludo_core::{GameState, NUM_PLAYERS};

use crate::eval4::{e4_probs_with, seat_features, N_E4};
use crate::evaluator::score_position;
use crate::expectimax::LudoBot;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RoundLeaf {
    /// eval4 win share of the seat.
    E4([f32; N_E4]),
    /// score_position (legacy expectimax scale; comparable only within one seat).
    Score,
    /// Linear per-seat utility on eval4 features: w·phi_k − beta · sum_{j≠k} w·phi_j.
    /// beta sets how much the OTHER players' standing matters (0 = selfish race,
    /// 1/(n−1) = me vs their mean, larger = paranoid). Tuned for playing strength.
    Lin { w: [f32; N_E4], beta: f32 },
    /// General linear "me vs the others": w·phi_k − sum_{j≠k} u·phi_j
    /// (score_position is one point of this family).
    Lin2 { w: [f32; N_E4], u: [f32; N_E4] },
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RoundConfig {
    pub leaf: RoundLeaf,
    /// Chance layers expanded below the root move before falling back to the leaf.
    pub max_rolls: u8,
}

pub struct RoundBot {
    pub player_id: Option<u8>,
    pub cfg: RoundConfig,
}

impl RoundBot {
    pub fn new(player_id: Option<u8>, cfg: RoundConfig) -> Self {
        Self { player_id, cfg }
    }

    #[inline]
    fn leaf(&self, s: &GameState, seat: u8) -> f32 {
        match &self.cfg.leaf {
            RoundLeaf::E4(w) => e4_probs_with(s, w)[seat as usize],
            RoundLeaf::Score => {
                if s.is_terminal {
                    return if s.winner == seat as i8 { 1e6 } else { -1e6 };
                }
                score_position(seat, s)
            }
            RoundLeaf::Lin { w, beta } => lin_utility(s, seat, w, *beta),
            RoundLeaf::Lin2 { w, u } => lin2_utility(s, seat, w, u),
        }
    }

    /// Greedy move of the seat to move (state has dice and >= 1 legal move).
    #[inline]
    fn greedy(&self, s: &GameState) -> u8 {
        let (lm, c) = s.legal_moves();
        if c == 1 {
            return lm[0];
        }
        let p = s.current_player;
        let mut best = lm[0];
        let mut best_v = f32::NEG_INFINITY;
        for &a in &lm[..c] {
            let v = self.leaf(&s.apply_move(a), p);
            if v > best_v {
                best_v = v;
                best = a;
            }
        }
        best
    }

    /// Expected leaf value for `me` from a pre-roll state (dice = 0).
    fn expect(&self, s: &GameState, me: u8, rolls_left: u8) -> f32 {
        if s.is_terminal {
            return self.leaf(s, me);
        }
        if s.current_player == me || rolls_left == 0 {
            return self.leaf(s, me);
        }
        let mut total = 0.0f32;
        for face in 1..=6u8 {
            let mut t = *s;
            let v = if t.roll(face) {
                let a = self.greedy(&t);
                self.expect(&t.apply_move(a), me, rolls_left - 1)
            } else {
                self.expect(&t, me, rolls_left - 1)
            };
            total += v;
        }
        total / 6.0
    }

    /// Value of each legal move slot (state has dice). Returns (values, legal, count).
    pub fn move_values(&self, state: &GameState) -> ([f32; 4], [u8; 4], usize) {
        let (legal, count) = state.legal_moves();
        let me = state.current_player;
        let mut vals = [f32::NEG_INFINITY; 4];
        for i in 0..count {
            vals[i] = self.expect(&state.apply_move(legal[i]), me, self.cfg.max_rolls);
        }
        (vals, legal, count)
    }
}

impl LudoBot for RoundBot {
    fn name(&self) -> &'static str {
        "Round"
    }

    fn select_move(&self, state: &GameState) -> u8 {
        let (vals, legal, count) = self.move_values(state);
        if count == 0 {
            return 0;
        }
        let mut b = 0;
        for i in 1..count {
            if vals[i] > vals[b] {
                b = i;
            }
        }
        legal[b]
    }
}

/// Linear per-seat utility (see [`RoundLeaf::Lin`]); terminal = ±1e6.
#[inline]
pub fn lin_utility(s: &GameState, seat: u8, w: &[f32; N_E4], beta: f32) -> f32 {
    if s.is_terminal {
        return if s.winner == seat as i8 { 1e6 } else { -1e6 };
    }
    let f = seat_features(s);
    let mut own = 0.0f32;
    let mut others = 0.0f32;
    for p in 0..NUM_PLAYERS {
        if !s.active_players[p] {
            continue;
        }
        let mut z = 0.0f32;
        for i in 0..N_E4 {
            z += w[i] * f[p][i];
        }
        if p == seat as usize { own = z } else { others += z }
    }
    own - beta * others
}

/// General linear utility (see [`RoundLeaf::Lin2`]); terminal = ±1e6.
#[inline]
pub fn lin2_utility(s: &GameState, seat: u8, w: &[f32; N_E4], u: &[f32; N_E4]) -> f32 {
    if s.is_terminal {
        return if s.winner == seat as i8 { 1e6 } else { -1e6 };
    }
    let f = seat_features(s);
    let mut v = 0.0f32;
    for p in 0..NUM_PLAYERS {
        if !s.active_players[p] {
            continue;
        }
        let (c, sign) = if p == seat as usize { (w, 1.0f32) } else { (u, -1.0f32) };
        let mut z = 0.0f32;
        for i in 0..N_E4 {
            z += c[i] * f[p][i];
        }
        v += sign * z;
    }
    v
}

/// Number of active seats (helper for callers choosing `max_rolls`).
pub fn n_active(s: &GameState) -> usize {
    (0..NUM_PLAYERS).filter(|&p| s.active_players[p]).count()
}
