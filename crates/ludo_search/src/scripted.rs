//! Rust ports of the project's scripted bots (td_ludo/td_ludo/game/heuristic_bot.py):
//! `HeuristicLudoBot` ("Heuristic", balanced, adaptive) and `ExpertBot` ("Expert",
//! tournament-optimised weights). Faithful ports — same scoring terms, same
//! constants, same tie-breaking (first legal token wins ties; the Python bots add
//! uniform(0, 5) noise, reproduced here unless `noise` is off). Verified move-for-
//! move against the Python versions with noise disabled (v17/tests).
//!
//! Note: like the originals, these treat "stars" as safe squares in their own
//! scoring even though the rules only protect the 8 globes.

use std::sync::Mutex;

use ludo_core::{GameState, BASE_POS, HOME_POS};
use rand::rngs::StdRng as SmallRng;
use rand::{Rng, SeedableRng};

use crate::expectimax::LudoBot;

const W_WIN_GAME: f64 = 50000.0;
const W_FINISH_TOKEN: f64 = 10000.0;
const W_CUT_OPPONENT: f64 = 6000.0;
const W_EXIT_BASE: f64 = 1500.0;
const W_ENTER_SAFE: f64 = 800.0;
const W_ENTER_HOME_RUN: f64 = 500.0;
const W_PROGRESS: f64 = 10.0;
const W_STACK: f64 = 150.0;
const W_BREAK_STACK: f64 = 400.0;
const W_STAR_TELEPORT: f64 = 350.0;
const W_DANGER_PENALTY: f64 = -2000.0;
const W_LEAVE_SAFE_PENALTY: f64 = -500.0;
const W_LEAVE_STACK_PENALTY: f64 = -200.0;

const GLOBES: [i32; 8] = [0, 8, 13, 21, 26, 34, 39, 47];
const STARS: [i32; 8] = [5, 18, 31, 44, 11, 24, 37, 50];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScriptedKind {
    Heuristic,
    Expert,
}

/// Weights that differ between the two bots.
struct Weights {
    cut: f64,       // multiplied by the adaptive factor (Heuristic) / fixed (Expert)
    danger: f64,
    progress: f64,
    safe: f64,
    stack: f64,
}

pub struct ScriptedBot {
    pub kind: ScriptedKind,
    pub player_id: Option<u8>,
    /// Reproduce the Python bots' uniform(0, 5) score noise.
    pub noise: bool,
    rng: Mutex<SmallRng>,
}

impl ScriptedBot {
    pub fn new(kind: ScriptedKind, player_id: Option<u8>) -> Self {
        ScriptedBot { kind, player_id, noise: true, rng: Mutex::new(SmallRng::from_entropy()) }
    }

    pub fn with_seed(kind: ScriptedKind, player_id: Option<u8>, seed: u64) -> Self {
        ScriptedBot { kind, player_id, noise: true, rng: Mutex::new(SmallRng::seed_from_u64(seed)) }
    }

    #[inline]
    fn abs_pos(player: usize, rel: i8) -> i32 {
        if rel < 0 || rel > 50 {
            -1
        } else {
            (rel as i32 + 13 * player as i32) % 52
        }
    }

    #[inline]
    fn is_safe(player: usize, rel: i8) -> bool {
        if rel == BASE_POS || rel > 50 || rel == HOME_POS {
            return true;
        }
        let a = Self::abs_pos(player, rel);
        GLOBES.contains(&a) || STARS.contains(&a)
    }

    #[inline]
    fn is_stack(s: &GameState, player: usize, pos: i8) -> bool {
        if pos == BASE_POS || pos == HOME_POS {
            return false;
        }
        s.positions[player].iter().filter(|&&p| p == pos).count() >= 2
    }

    fn in_base(s: &GameState, p: usize) -> i32 {
        s.positions[p].iter().filter(|&&x| x == BASE_POS).count() as i32
    }

    fn evaluate(&self, prev: &GameState, curr: &GameState, token: usize, p: usize, w: &Weights) -> f64 {
        let mut score = 0.0;
        if curr.scores[p] == 4 {
            return W_WIN_GAME;
        }
        if curr.scores[p] > prev.scores[p] {
            score += W_FINISH_TOKEN;
        }
        let mut cuts = 0i32;
        for op in 0..4 {
            if op == p {
                continue;
            }
            let new_cuts = Self::in_base(curr, op) - Self::in_base(prev, op);
            if new_cuts > 0 {
                cuts += new_cuts;
                if new_cuts >= 2 {
                    score += W_BREAK_STACK;
                }
            }
        }
        score += cuts as f64 * w.cut;

        let pos = curr.positions[p][token];
        let prev_pos = prev.positions[p][token];
        if prev_pos == BASE_POS && pos != BASE_POS {
            score += W_EXIT_BASE;
        }
        let is_stack_now = Self::is_stack(curr, p, pos);
        let was_stack_prev = Self::is_stack(prev, p, prev_pos);
        let is_safe_now = Self::is_safe(p, pos) || is_stack_now;
        let is_safe_prev = Self::is_safe(p, prev_pos) || was_stack_prev;
        let scored_or_cut = cuts > 0 || curr.scores[p] != prev.scores[p];

        if is_safe_now {
            score += w.safe;
            if !is_safe_prev && prev_pos != BASE_POS {
                score += 200.0;
            }
        }
        if is_safe_prev && !is_safe_now && pos != HOME_POS && !scored_or_cut {
            score += W_LEAVE_SAFE_PENALTY;
        }
        if was_stack_prev && !is_stack_now && prev_pos != BASE_POS && pos != HOME_POS && !scored_or_cut {
            score += W_LEAVE_STACK_PENALTY;
        }
        if prev_pos != BASE_POS && prev_pos <= 50 && pos > 50 && pos != HOME_POS {
            score += W_ENTER_HOME_RUN;
        }
        if pos != BASE_POS && pos <= 50 && pos != HOME_POS && STARS.contains(&Self::abs_pos(p, pos)) {
            score += W_STAR_TELEPORT;
        }
        if pos != BASE_POS && prev_pos != BASE_POS && pos != HOME_POS {
            if pos > prev_pos {
                score += (pos - prev_pos) as f64 * w.progress;
            } else if prev_pos <= 50 && pos <= 50 {
                score += ((52 - prev_pos as i32) + pos as i32) as f64 * w.progress;
            }
        }
        if is_stack_now {
            score += w.stack;
        }
        if pos != HOME_POS && pos <= 50 && !is_safe_now {
            let my_abs = Self::abs_pos(p, pos);
            let mut risk = 0.0;
            for op in 0..4 {
                if op == p {
                    continue;
                }
                for t in 0..4 {
                    let op_pos = curr.positions[op][t];
                    if op_pos == BASE_POS || op_pos == HOME_POS || op_pos > 50 {
                        continue;
                    }
                    let distance = (my_abs - Self::abs_pos(op, op_pos)).rem_euclid(52);
                    if (1..=6).contains(&distance) {
                        risk += 1.0;
                        if Self::is_stack(curr, op, op_pos) {
                            risk += 0.5;
                        }
                    }
                }
            }
            score += risk * w.danger;
        }
        if cuts > 0 {
            let max_other = (0..4).filter(|&i| i != p).map(|i| curr.scores[i]).max().unwrap_or(0);
            for op in 0..4 {
                if op == p {
                    continue;
                }
                if Self::in_base(curr, op) > Self::in_base(prev, op) && curr.scores[op] == max_other {
                    score += 1500.0;
                }
            }
        }
        score
    }
}

impl LudoBot for ScriptedBot {
    fn name(&self) -> &'static str {
        match self.kind {
            ScriptedKind::Heuristic => "Heuristic",
            ScriptedKind::Expert => "Expert",
        }
    }

    fn select_move(&self, state: &GameState) -> u8 {
        let (legal, count) = state.legal_moves();
        if count == 0 {
            return 0;
        }
        if count == 1 {
            return legal[0];
        }
        let p = self.player_id.unwrap_or(state.current_player) as usize;
        let w = match self.kind {
            ScriptedKind::Heuristic => {
                let my = state.scores[p];
                let max_opp = (0..4).filter(|&i| i != p).map(|i| state.scores[i]).max().unwrap_or(0);
                let (mut cut, mut danger) = (W_CUT_OPPONENT, W_DANGER_PENALTY);
                if my > max_opp {
                    danger *= 1.5;
                    cut *= 0.8;
                } else if my < max_opp {
                    cut *= 1.5;
                    danger *= 0.8;
                }
                Weights { cut, danger, progress: W_PROGRESS, safe: W_ENTER_SAFE, stack: W_STACK }
            }
            ScriptedKind::Expert => Weights { cut: 4243.3, danger: -698.1, progress: 40.1, safe: 2917.6, stack: 279.9 },
        };
        let mut rng = self.rng.lock().expect("rng lock");
        let mut best = legal[0];
        let mut best_score = f64::NEG_INFINITY;
        for &t in &legal[..count] {
            let next = state.apply_move(t);
            let mut sc = self.evaluate(state, &next, t as usize, p, &w);
            if self.noise {
                sc += rng.gen_range(0.0..5.0);
            }
            if sc > best_score {
                best_score = sc;
                best = t;
            }
        }
        best
    }
}
