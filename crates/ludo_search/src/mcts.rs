//! 4-Player Multi-Agent Max^n MCTS with real chance nodes and an Expectimax prior.
//!
//!   - Every edge accumulates a 4-dimensional payoff vector V = [v0, v1, v2, v3].
//!   - At a decision node, player `p` selects by PUCT on ITS OWN coordinate.
//!   - Root priors come from ExpectimaxBot (smoothed), as in the original design.
//!
//! REWRITTEN in the 2026-10-02 audit, sharing the fixes made to the 2-player
//! engine (see mcts_2p.rs for the measurements):
//!   1. Each action edge used to sample ONE dice roll at first expansion and
//!      reuse it forever (no chance nodes). -> Every move leads to a chance node
//!      with a child per dice face; each simulation samples a fresh roll.
//!   2. No-move outcomes re-expanded the parent edge. -> They resolve (via
//!      GameState::roll) to a chance node for whoever rolls next.
//!   3. Every game loop double-passed after a three-six forfeit, which in 4P
//!      SKIPPED THE NEXT PLAYER'S TURN. -> GameState::roll.
//!   4. The cutoff heuristic normalised max(score_position, 0) + 1 per seat;
//!      score_position is negative for every seat while tokens sit in base, so
//!      early-game payoffs were exactly uniform. -> Softmax over each seat's
//!      leaf_logit (informative everywhere; NOTE: that model was fitted on
//!      2-player games, so these 4P payoffs are not calibrated probabilities).
//!   5. Random 60-move rollouts re-rolled already-rolled nodes. -> Short greedy
//!      rollout from the node's own dice, then the evaluator.
//!   6. A path Vec was allocated per simulation. -> Thread-local buffers.
//!
//!   RESULT, one MaxN seat vs 3x ExpectimaxBot at N=1000: 21.1% -> 32.6%
//!   (fair share 25%), 1.6x faster; N=3000 reaches 38.5%.
//!
//! 2026-10-07: search stopped scaling (34% vs 3x Expectimax at N = 1000, 3000 and
//! 10000 alike). The leaf evaluator (2P-fitted, "me vs the SUM of opponents") and
//! the score_position rollouts are now configurable ([`MaxNConfig`]): a 4P
//! conditional-logit evaluator (`eval4`) and rollouts that maximise the mover's
//! own win share. `Legacy` reproduces the previous engine exactly.

use std::cell::RefCell;

use ludo_core::{GameState, NUM_PLAYERS};
use rand::Rng;

use crate::eval4::{e4_probs_with, N_E4};
use crate::evaluator::{leaf_logit, score_position};
use crate::expectimax::{ExpectimaxBot, LudoBot};

const MAX_TREE_NODES: usize = 1 << 16;
const NONE: u32 = u32::MAX;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Decision,
    Chance,
}

#[derive(Clone, Copy)]
struct MctsNode {
    state: GameState,
    kind: Kind,
    player: u8,
    legal_actions: [u8; 4],
    legal_count: u8,
    priors: [f32; 4],
    visit_counts: [u32; 4],
    /// total_payoffs[action][player]
    total_payoffs: [[f32; 4]; 4],
    /// Decision: per action slot. Chance: per dice face (index face-1).
    children: [u32; 6],
}

impl MctsNode {
    fn chance(state: GameState) -> Self {
        Self {
            state,
            kind: Kind::Chance,
            player: state.current_player,
            legal_actions: [0; 4],
            legal_count: 0,
            priors: [0.0; 4],
            visit_counts: [0; 4],
            total_payoffs: [[0.0; 4]; 4],
            children: [NONE; 6],
        }
    }

    fn decision(state: GameState) -> Self {
        let (legal, count) = state.legal_moves();
        let mut priors = [0.0; 4];
        for p in priors.iter_mut().take(count) {
            *p = 1.0 / count.max(1) as f32;
        }
        Self {
            state,
            kind: Kind::Decision,
            player: state.current_player,
            legal_actions: legal,
            legal_count: count as u8,
            priors,
            visit_counts: [0; 4],
            total_payoffs: [[0.0; 4]; 4],
            children: [NONE; 6],
        }
    }

    fn after_roll(mut state: GameState, face: u8) -> Self {
        if state.roll(face) { MctsNode::decision(state) } else { MctsNode::chance(state) }
    }

    /// PUCT over player p's own coordinate in the payoff vector.
    fn select_puct(&self, c_puct: f32) -> usize {
        let p = self.player as usize;
        let n = self.legal_count as usize;
        let total: u32 = self.visit_counts[..n].iter().sum();
        let sqrt_total = (total.max(1) as f32).sqrt();
        let mut best = 0;
        let mut best_score = f32::NEG_INFINITY;
        for i in 0..n {
            let v = self.visit_counts[i] as f32;
            // Payoffs are probability vectors, so an unvisited edge starts at a
            // fair share rather than 0 (0 was the floor of every coordinate).
            let q = if v > 0.0 { self.total_payoffs[i][p] / v } else { 0.25 };
            let u = c_puct * self.priors[i] * sqrt_total / (1.0 + v);
            if q + u > best_score {
                best_score = q + u;
                best = i;
            }
        }
        best
    }
}

/// Leaf evaluator for the payoff vector.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LeafEval4 {
    /// Softmax over the 2P-fitted `leaf_logit` per seat (pre-2026-10-07 engine).
    Legacy,
    /// 4P conditional-logit win shares (`eval4`) with these weights.
    E4([f32; N_E4]),
}

/// Rollout move choice (each mover greedy on its own criterion).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Rollout4 {
    /// `score_position` (own progress x2 minus the SUM of opponents) — legacy.
    ScorePosition,
    /// The mover's own win share under the leaf evaluator.
    OwnShare,
}

/// Which bot supplies the root prior (its move gets 1 − prior_smooth).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Prior4 {
    /// 1-ply ExpectimaxBot (legacy).
    Expectimax,
    /// The exact full-round search (round4.rs) with this configuration.
    Round(crate::round4::RoundConfig),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MaxNConfig {
    pub sims: u32,
    pub prior: Prior4,
    pub c_puct: f32,
    pub prior_smooth: f32,
    pub rollout_depth: u16,
    pub rollout: Rollout4,
    pub leaf: LeafEval4,
}

impl MaxNConfig {
    /// The pre-2026-10-07 engine.
    pub fn legacy(sims: u32) -> Self {
        Self {
            sims,
            prior: Prior4::Expectimax,
            c_puct: 0.5,
            prior_smooth: 0.15,
            rollout_depth: 10,
            rollout: Rollout4::ScorePosition,
            leaf: LeafEval4::Legacy,
        }
    }
}

thread_local! {
    static ARENA4: RefCell<Vec<MctsNode>> = const { RefCell::new(Vec::new()) };
    static PATH4: RefCell<Vec<(u32, u8)>> = const { RefCell::new(Vec::new()) };
}

/// 4-Player Max^n MCTS Bot.
pub struct MaxNMCTSBot {
    pub player_id: Option<u8>,
    pub num_simulations: usize,
    pub c_puct: f32,
    pub prior_smooth: f32,
    /// Greedy rollout length before the leaf evaluator.
    pub rollout_depth: usize,
    pub rollout: Rollout4,
    pub leaf: LeafEval4,
    prior_bot: Box<dyn LudoBot>,
}

impl MaxNMCTSBot {
    pub fn new(player_id: Option<u8>, num_simulations: usize) -> Self {
        // Measured 2026-10-02, one MaxN seat vs 3x ExpectimaxBot (fair share 25%):
        //   pre-audit engine  N=40 24.9%   N=1000 21.1% (more search was WORSE)
        //   this engine       N=40 26.7%   N=1000 c=1.414 28.3%, c=0.7 32.2%,
        //                     c=0.5 32.6%, c=0.35 31.3%, c=3 25.8%
        //                     N=3000 c=0.5 38.5%;  vs 3x Depth2: N=1000 31.4%
        //   rollout_depth 0 23.8% / 10 32.6% / 25 32.4%;  prior_smooth 0.75 21.2%
        // Payoffs are win shares (~0.25 each), so Q differences are small and a
        // lower c_puct than the 2-player engine's is right.
        Self::with_config(player_id, MaxNConfig::legacy(num_simulations as u32))
    }

    pub fn with_config(player_id: Option<u8>, c: MaxNConfig) -> Self {
        Self {
            player_id,
            num_simulations: c.sims as usize,
            c_puct: c.c_puct,
            prior_smooth: c.prior_smooth,
            rollout_depth: c.rollout_depth as usize,
            rollout: c.rollout,
            leaf: c.leaf,
            prior_bot: match c.prior {
                Prior4::Expectimax => Box::new(ExpectimaxBot::new(player_id)),
                Prior4::Round(rc) => Box::new(crate::round4::RoundBot::new(player_id, rc)),
            },
        }
    }

    fn leaf_payoffs(&self, state: &GameState) -> [f32; 4] {
        match &self.leaf {
            LeafEval4::Legacy => Self::payoffs(state),
            LeafEval4::E4(w) => e4_probs_with(state, w),
        }
    }

    /// Payoff vector (win shares, summing to 1 over active seats).
    fn payoffs(state: &GameState) -> [f32; 4] {
        let mut out = [0.0f32; 4];
        if state.is_terminal && state.winner >= 0 {
            out[state.winner as usize] = 1.0;
            return out;
        }
        let mut logits = [f32::NEG_INFINITY; 4];
        let mut mx = f32::NEG_INFINITY;
        for p in 0..NUM_PLAYERS {
            if state.active_players[p] {
                logits[p] = leaf_logit(p as u8, state);
                mx = mx.max(logits[p]);
            }
        }
        let mut sum = 0.0;
        for p in 0..NUM_PLAYERS {
            if state.active_players[p] {
                out[p] = (logits[p] - mx).exp();
                sum += out[p];
            }
        }
        for v in out.iter_mut() {
            *v /= sum.max(1e-12);
        }
        out
    }

    /// Short greedy rollout from the node's own state (a decision node already
    /// has its dice, so it is not re-rolled), then the payoff estimate.
    fn rollout(&self, mut s: GameState, rng: &mut impl Rng) -> [f32; 4] {
        let mut depth = 0;
        let mut guard = 0;
        while !s.is_terminal && depth < self.rollout_depth && guard < 8 * self.rollout_depth + 64 {
            guard += 1;
            if s.current_dice_roll == 0 && !s.roll(rng.gen_range(1..=6)) {
                continue;
            }
            let (lm, c) = s.legal_moves();
            let p = s.current_player;
            let mut best = lm[0];
            if c > 1 {
                let mut best_sc = f32::NEG_INFINITY;
                for &a in &lm[..c] {
                    let nx = s.apply_move(a);
                    let sc = match self.rollout {
                        Rollout4::ScorePosition => score_position(p, &nx),
                        Rollout4::OwnShare => self.leaf_payoffs(&nx)[p as usize],
                    };
                    if sc > best_sc { best_sc = sc; best = a; }
                }
            }
            s = s.apply_move(best);
            depth += 1;
        }
        self.leaf_payoffs(&s)
    }

    /// Runs the search; returns (chosen token, visit distribution over tokens).
    /// `temp` 0 = most visited; > 0 samples visits^(1/temp).
    pub fn search(&self, state: &GameState, temp: f32, rng: &mut impl Rng) -> (u8, [f32; 4]) {
        let (legal, count) = state.legal_moves();
        let mut pi = [0f32; 4];
        if count == 0 {
            return (0, pi);
        }
        if count == 1 {
            pi[legal[0] as usize] = 1.0;
            return (legal[0], pi);
        }
        let visits = self.run(state, rng);
        let total: u32 = visits[..count].iter().sum();
        for i in 0..count {
            pi[legal[i] as usize] = visits[i] as f32 / total.max(1) as f32;
        }
        let pick = if temp <= 1e-6 {
            let mut b = 0;
            for i in 0..count {
                if visits[i] > visits[b] { b = i; }
            }
            b
        } else {
            let w: Vec<f64> = (0..count).map(|i| (visits[i] as f64).powf(1.0 / temp as f64)).collect();
            let sw: f64 = w.iter().sum();
            let mut x = rng.gen_range(0.0..sw.max(1e-12));
            let mut b = count - 1;
            for i in 0..count {
                if x < w[i] { b = i; break; }
                x -= w[i];
            }
            b
        };
        (legal[pick], pi)
    }

    /// Root visit counts per legal-move slot (state must have >= 2 legal moves).
    fn run(&self, state: &GameState, rng: &mut impl Rng) -> [u32; 4] {
        let (legal, count) = state.legal_moves();
        let mut root = MctsNode::decision(*state);
        let best_exp_move = self.prior_bot.select_move(state);
        let k = count as f32;
        for i in 0..count {
            root.priors[i] = if legal[i] == best_exp_move {
                1.0 - self.prior_smooth
            } else {
                self.prior_smooth / (k - 1.0)
            };
        }
        let cap = (self.num_simulations + 2).min(MAX_TREE_NODES);

        ARENA4.with(|ac| {
            PATH4.with(|pc| {
                let mut arena = ac.borrow_mut();
                let mut path = pc.borrow_mut();
                arena.clear();
                arena.reserve(cap);
                arena.push(root);

                for _ in 0..self.num_simulations {
                    path.clear();
                    let mut idx: u32 = 0;
                    let leaf: [f32; 4] = loop {
                        let node = arena[idx as usize];
                        match node.kind {
                            Kind::Decision => {
                                let slot = node.select_puct(self.c_puct);
                                path.push((idx, slot as u8));
                                let child = node.children[slot];
                                if child != NONE { idx = child; continue; }
                                let after = node.state.apply_move(node.legal_actions[slot]);
                                if after.is_terminal { break self.leaf_payoffs(&after); }
                                let v = self.rollout(after, rng);
                                if arena.len() < cap {
                                    arena.push(MctsNode::chance(after));
                                    let ni = (arena.len() - 1) as u32;
                                    arena[idx as usize].children[slot] = ni;
                                }
                                break v;
                            }
                            Kind::Chance => {
                                let face = rng.gen_range(1..=6u8);
                                let child = node.children[(face - 1) as usize];
                                if child != NONE { idx = child; continue; }
                                let new = MctsNode::after_roll(node.state, face);
                                let v = self.rollout(new.state, rng);
                                if arena.len() < cap {
                                    arena.push(new);
                                    let ni = (arena.len() - 1) as u32;
                                    arena[idx as usize].children[(face - 1) as usize] = ni;
                                }
                                break v;
                            }
                        }
                    };
                    for &(ni, slot) in path.iter() {
                        let n = &mut arena[ni as usize];
                        n.visit_counts[slot as usize] += 1;
                        for p in 0..NUM_PLAYERS {
                            n.total_payoffs[slot as usize][p] += leaf[p];
                        }
                    }
                }
                arena[0].visit_counts
            })
        })
    }
}

impl LudoBot for MaxNMCTSBot {
    fn name(&self) -> &'static str {
        "MaxNMCTS"
    }

    fn select_move(&self, state: &GameState) -> u8 {
        let mut rng = rand::thread_rng();
        self.search(state, 0.0, &mut rng).0
    }
}
