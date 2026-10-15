//! 2-Player Zero-Sum Expecti-MCTS (P0 vs P2) with real chance nodes.
//!
//! REWRITTEN in the 2026-10-02 audit. The previous engine lost 15.2% of
//! 1,000,000 games to Depth2ExpectimaxBot. Measured causes, all fixed here:
//!
//!   1. NO CHANCE NODES. Each action edge sampled ONE dice roll at first
//!      expansion and reused that child for every later simulation, so each
//!      subtree was conditioned on a single frozen roll. Same position, 8
//!      independent searches agreed with their own modal action only
//!      49% / 55% / 61% of the time (early / mid / late game).
//!      -> Every move now leads to a CHANCE node with one child per dice face;
//!         each simulation samples a fresh roll, so values average over dice.
//!
//!   2. PASS NODES RE-EXPANDED THEIR PARENT. A child with no legal move ended
//!      selection on the parent edge, which was re-expanded with a new roll and
//!      the old subtree orphaned — so the tree kept re-rolling until the next
//!      player HAD a move ("opponent rolled a 6" bias in the early game).
//!      -> No-move and three-six-forfeit outcomes resolve (via GameState::roll)
//!         to a chance node for whoever rolls next.
//!
//!   3. BLIND LEAVES. Uniform-random rollouts capped at 60 moves returned 0.0
//!      on cutoff: 98.5% of early-game simulations returned exactly 0.0. The
//!      "heuristic" alternative clipped max(score, 0)+1 per side, which is
//!      also exactly 0 whenever both players still have tokens in base.
//!      -> Leaves now end in evaluator::leaf_value, a calibrated logistic win
//!         model (val log-loss 0.5748 vs 0.6929 constant, AUC 0.762). Default
//!         is a 10-move greedy rollout then that evaluator.
//!
//!   RESULT vs Depth2ExpectimaxBot: 15.2% (1M games) -> 63.1% ±1.1 (n=2000);
//!   self-agreement across 8 searches 49/55/61% -> 84/91/86%; 7.7x faster per
//!   search (60-move random rollouts were the cost).
//!
//!   4. Rollouts called set_dice on nodes that already had dice set,
//!      double-counting consecutive sixes; and every game loop double-passed
//!      after a three-six forfeit. -> GameState::roll everywhere.
//!
//!   5. A path Vec was heap-allocated every simulation. -> Thread-local arena
//!      and path buffers: no allocation in the search loop once warm.
//!
//! Public API (`new`, the pub fields, `search`, `search_with_temp`, LudoBot) is
//! unchanged, so callers keep compiling. Leaf behaviour is chosen by
//! `leaf_mode`; `use_pure_terminal` keeps its old meaning for RandomRollout.

use std::cell::RefCell;

use ludo_core::GameState;
use rand::Rng;

use crate::evaluator::{leaf_value, score_position};
use crate::expectimax::{ExpectimaxBot, LudoBot};

/// Hard cap on tree size. Each simulation adds at most one node, so a search
/// of N simulations needs at most N + 1.
const MAX_TREE_NODES_2P: usize = 1 << 17;
const NONE: u32 = u32::MAX;

/// How a newly expanded leaf is valued.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LeafMode {
    /// Calibrated logistic win model (`evaluator::leaf_value`). Fastest.
    Evaluator,
    /// Uniform-random rollout for `rollout_depth` moves. On cutoff returns 0.0
    /// if `use_pure_terminal`, otherwise the evaluator. (Pre-audit behaviour
    /// when `use_pure_terminal = true`.)
    RandomRollout,
    /// Greedy rollout (each mover maximises `score_position`) for
    /// `rollout_depth` moves, then the evaluator. Default (strongest per cost).
    GreedyRollout,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    /// Dice already rolled, current player has >= 1 legal move.
    Decision,
    /// Dice not yet rolled (current_dice_roll == 0); children indexed by face.
    Chance,
}

#[derive(Clone, Copy)]
struct Node {
    state: GameState,
    kind: Kind,
    /// Player to act: the mover at a decision node, the roller at a chance node.
    player: u8,
    legal_actions: [u8; 4],
    legal_count: u8,
    priors: [f32; 4],
    visits: [u32; 4],
    /// Accumulated value per action, from `player`'s perspective.
    values: [f32; 4],
    /// Decision: child chance node per action slot (NONE = unexpanded or a
    /// terminal edge, which is re-scored exactly on every visit).
    /// Chance: child per dice face 1..6 at index face-1.
    children: [u32; 6],
}

impl Node {
    fn chance(state: GameState) -> Self {
        Self {
            state,
            kind: Kind::Chance,
            player: state.current_player,
            legal_actions: [0; 4],
            legal_count: 0,
            priors: [0.0; 4],
            visits: [0; 4],
            values: [0.0; 4],
            children: [NONE; 6],
        }
    }

    fn decision(state: GameState) -> Self {
        let (legal, count) = state.legal_moves();
        debug_assert!(count > 0, "decision node needs a legal move");
        let mut priors = [0.0; 4];
        for p in priors.iter_mut().take(count) {
            *p = 1.0 / count as f32;
        }
        Self {
            state,
            kind: Kind::Decision,
            player: state.current_player,
            legal_actions: legal,
            legal_count: count as u8,
            priors,
            visits: [0; 4],
            values: [0.0; 4],
            children: [NONE; 6],
        }
    }

    /// Node reached by rolling `face` from a chance node's state.
    fn after_roll(mut state: GameState, face: u8) -> Self {
        if state.roll(face) {
            Node::decision(state)
        } else {
            // three-six forfeit or no legal move: turn already passed
            Node::chance(state)
        }
    }

    #[inline]
    fn select_puct(&self, c_puct: f32) -> usize {
        let n = self.legal_count as usize;
        let total: u32 = self.visits[..n].iter().sum();
        let sqrt_total = (total.max(1) as f32).sqrt();
        let mut best = 0;
        let mut best_score = f32::NEG_INFINITY;
        for i in 0..n {
            let v = self.visits[i] as f32;
            let q = if v > 0.0 { self.values[i] / v } else { 0.0 };
            let u = c_puct * self.priors[i] * sqrt_total / (1.0 + v);
            if q + u > best_score {
                best_score = q + u;
                best = i;
            }
        }
        best
    }
}

thread_local! {
    static ARENA: RefCell<Vec<Node>> = const { RefCell::new(Vec::new()) };
    static PATH: RefCell<Vec<(u32, u8)>> = const { RefCell::new(Vec::new()) };
}

/// Samples Dirichlet(alpha=0.3) noise for k actions using Marsaglia-Tsang Gamma method.
pub(crate) fn sample_dirichlet_03(k: usize, rng: &mut impl rand::Rng) -> [f32; 4] {
    let mut out = [0.0f32; 4];
    let mut sum = 0.0f32;
    let alpha = 0.3f32;
    let d = (alpha + 1.0) - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();

    for slot in out.iter_mut().take(k) {
        let mut attempts = 0;
        let g = loop {
            attempts += 1;
            let z: f32 = rng.gen_range(-2.5..2.5);
            let v = 1.0 + c * z;
            if v <= 0.0 {
                if attempts > 50 { break 0.1; }
                continue;
            }
            let v3 = v * v * v;
            let u: f32 = rng.gen_range(0.0001..0.9999);
            if u < 1.0 - 0.0331 * z * z * z * z || u.ln() < 0.5 * z * z + d * (1.0 - v3 + v3.ln()) {
                let u2: f32 = rng.gen_range(0.0001..0.9999);
                break (d * v3) * u2.powf(1.0 / alpha);
            }
            if attempts > 50 { break 0.1; }
        };
        *slot = g.max(1e-4);
        sum += *slot;
    }
    for slot in out.iter_mut().take(k) {
        *slot = if sum > 0.0 { *slot / sum } else { 1.0 / k as f32 };
    }
    out
}

/// 2-Player Zero-Sum MCTS Bot for P0 vs P2.
pub struct TwoPlayerMCTSBot {
    pub player_id: Option<u8>,
    pub num_simulations: usize,
    pub c_puct: f32,
    pub prior_smooth: f32,
    /// Root prior from ExpectimaxBot's choice instead of uniform.
    pub use_heuristic_prior: bool,
    /// RandomRollout only: on cutoff return 0.0 (true) or the evaluator (false).
    pub use_pure_terminal: bool,
    /// Root Dirichlet noise. For self-play EXPLORATION only — never for
    /// generating teacher labels (it was on in the V15.3 teacher and, with
    /// blind leaves, decided the label).
    pub add_dirichlet: bool,
    pub dirichlet_epsilon: f32,
    pub leaf_mode: LeafMode,
    pub rollout_depth: usize,
    prior_bot: ExpectimaxBot,
}

impl TwoPlayerMCTSBot {
    pub fn new(player_id: Option<u8>, num_simulations: usize) -> Self {
        Self {
            player_id,
            num_simulations,
            c_puct: 1.414,
            prior_smooth: 0.15,
            use_heuristic_prior: false,
            use_pure_terminal: true,
            add_dirichlet: false,
            dirichlet_epsilon: 0.25,
            // Measured vs Depth2ExpectimaxBot, seat-alternated (2026-10-02):
            //   pre-audit engine (random rollout, frozen dice)  15.2%  (1M games)
            //   Evaluator,            N=3000                    58.1% ±1.1 (n=2000)
            //   GreedyRollout d=10,   N=3000                    63.1% ±1.1 (n=2000)
            //   GreedyRollout d=20,   N=3000                    63.3% ±1.5 (2x cost)
            //   GreedyRollout d=10,   N=10000                   61.2% ±1.5 (no gain)
            //   use_heuristic_prior = true                      50.8% (hurts: keep uniform)
            leaf_mode: LeafMode::GreedyRollout,
            rollout_depth: 10,
            prior_bot: ExpectimaxBot::new(player_id),
        }
    }

    /// Value of `node_state` from `root`'s perspective, per `leaf_mode`.
    fn evaluate(&self, node_state: &GameState, kind: Kind, root: u8, rng: &mut impl Rng) -> f32 {
        if node_state.is_terminal {
            return if node_state.winner == root as i8 { 1.0 } else { -1.0 };
        }
        match self.leaf_mode {
            LeafMode::Evaluator => match kind {
                // Dice unknown: the evaluator was fitted at exactly this point
                // (pre-roll, roller's perspective).
                Kind::Chance => {
                    let v = leaf_value(node_state.current_player, node_state);
                    if node_state.current_player == root { v } else { -v }
                }
                // Dice known: use it — best move for the mover, valued at the
                // resulting chance node from the mover's perspective.
                Kind::Decision => {
                    let mover = node_state.current_player;
                    let (lm, c) = node_state.legal_moves();
                    let mut best = f32::NEG_INFINITY;
                    for &a in &lm[..c] {
                        let s = node_state.apply_move(a);
                        let v = if s.is_terminal {
                            if s.winner == mover as i8 { 1.0 } else { -1.0 }
                        } else {
                            let w = leaf_value(s.current_player, &s);
                            if s.current_player == mover { w } else { -w }
                        };
                        if v > best { best = v; }
                    }
                    if mover == root { best } else { -best }
                }
            },
            LeafMode::RandomRollout | LeafMode::GreedyRollout => {
                self.rollout(*node_state, root, rng)
            }
        }
    }

    /// Rollout from a node state. A decision-node state already has its dice
    /// rolled, so its first move uses that dice (the old rollout re-rolled it,
    /// double-counting consecutive sixes).
    fn rollout(&self, mut s: GameState, root: u8, rng: &mut impl Rng) -> f32 {
        let greedy = self.leaf_mode == LeafMode::GreedyRollout;
        let mut depth = 0;
        let mut guard = 0;
        while !s.is_terminal && depth < self.rollout_depth && guard < 4 * self.rollout_depth + 64 {
            guard += 1;
            if s.current_dice_roll == 0 && !s.roll(rng.gen_range(1..=6)) {
                continue; // forfeit / no move: turn passed
            }
            let (lm, c) = s.legal_moves();
            let mv = if greedy && c > 1 {
                let p = s.current_player;
                let mut best = lm[0];
                let mut best_sc = f32::NEG_INFINITY;
                for &a in &lm[..c] {
                    let sc = score_position(p, &s.apply_move(a));
                    if sc > best_sc { best_sc = sc; best = a; }
                }
                best
            } else {
                lm[rng.gen_range(0..c)]
            };
            s = s.apply_move(mv);
            depth += 1;
        }
        if s.is_terminal {
            return if s.winner == root as i8 { 1.0 } else { -1.0 };
        }
        if self.leaf_mode == LeafMode::RandomRollout && self.use_pure_terminal {
            return 0.0;
        }
        let v = leaf_value(s.current_player, &s);
        if s.current_player == root { v } else { -v }
    }

    /// Runs the search and returns (chosen_action, root visit distribution π
    /// indexed by token id). `temperature <= 0.01` = argmax visits.
    pub fn search_with_temp(&self, state: &GameState, temperature: f32) -> (u8, [f32; 4]) {
        let (legal, count) = state.legal_moves();
        if count == 0 {
            return (0, [0.0; 4]);
        }
        if count == 1 {
            let mut pi = [0.0f32; 4];
            pi[legal[0] as usize] = 1.0;
            return (legal[0], pi);
        }

        let root_player = state.current_player;
        let mut rng = rand::thread_rng();

        let mut root = Node::decision(*state);
        if self.use_heuristic_prior {
            let best = self.prior_bot.select_move(state);
            let k = count as f32;
            for i in 0..count {
                root.priors[i] = if legal[i] == best {
                    1.0 - self.prior_smooth
                } else {
                    self.prior_smooth / (k - 1.0)
                };
            }
        }
        if self.add_dirichlet {
            let noise = sample_dirichlet_03(count, &mut rng);
            for i in 0..count {
                root.priors[i] = (1.0 - self.dirichlet_epsilon) * root.priors[i]
                    + self.dirichlet_epsilon * noise[i];
            }
        }

        let cap = (self.num_simulations + 2).min(MAX_TREE_NODES_2P);

        ARENA.with(|arena_cell| {
            PATH.with(|path_cell| {
                let mut arena = arena_cell.borrow_mut();
                let mut path = path_cell.borrow_mut();
                arena.clear();
                arena.reserve(cap); // no-op once warm: capacity persists per thread
                arena.push(root);

                for _ in 0..self.num_simulations {
                    path.clear();
                    let mut idx: u32 = 0;
                    let leaf_val: f32 = loop {
                        let node = arena[idx as usize];
                        match node.kind {
                            Kind::Decision => {
                                let slot = node.select_puct(self.c_puct);
                                path.push((idx, slot as u8));
                                let child = node.children[slot];
                                if child != NONE {
                                    idx = child;
                                    continue;
                                }
                                let after = node.state.apply_move(node.legal_actions[slot]);
                                if after.is_terminal {
                                    // Exact payoff; no node needed, re-scored on every visit.
                                    break if after.winner == root_player as i8 { 1.0 } else { -1.0 };
                                }
                                let v = self.evaluate(&after, Kind::Chance, root_player, &mut rng);
                                if arena.len() < cap {
                                    arena.push(Node::chance(after));
                                    let ni = (arena.len() - 1) as u32;
                                    arena[idx as usize].children[slot] = ni;
                                }
                                break v;
                            }
                            Kind::Chance => {
                                let face = rng.gen_range(1..=6u8);
                                let child = node.children[(face - 1) as usize];
                                if child != NONE {
                                    idx = child;
                                    continue;
                                }
                                let new = Node::after_roll(node.state, face);
                                let v = self.evaluate(&new.state, new.kind, root_player, &mut rng);
                                if arena.len() < cap {
                                    arena.push(new);
                                    let ni = (arena.len() - 1) as u32;
                                    arena[idx as usize].children[(face - 1) as usize] = ni;
                                }
                                break v;
                            }
                        }
                    };

                    // Zero-sum backup on decision edges, each from its own mover's view.
                    for &(ni, slot) in path.iter() {
                        let n = &mut arena[ni as usize];
                        n.visits[slot as usize] += 1;
                        n.values[slot as usize] += if n.player == root_player { leaf_val } else { -leaf_val };
                    }
                }

                let root = &arena[0];
                let n = root.legal_count as usize;
                let total: u32 = root.visits[..n].iter().sum();
                let mut pi = [0.0f32; 4];
                let mut best_action = root.legal_actions[0];
                let mut best_visits = 0;
                for i in 0..n {
                    let v = root.visits[i];
                    if total > 0 {
                        pi[root.legal_actions[i] as usize] = v as f32 / total as f32;
                    }
                    if v > best_visits {
                        best_visits = v;
                        best_action = root.legal_actions[i];
                    }
                }

                let chosen = if temperature <= 0.01 || total == 0 {
                    best_action
                } else {
                    // p ∝ visits^(1/T). Unvisited actions get weight 0 (the old
                    // code floored visits at 1, giving them non-zero mass).
                    let inv_t = 1.0 / temperature;
                    let mut w = [0.0f32; 4];
                    let mut sum = 0.0f32;
                    for i in 0..n {
                        w[i] = (root.visits[i] as f32).powf(inv_t);
                        sum += w[i];
                    }
                    let mut r = rng.gen_range(0.0..sum);
                    let mut pick = best_action;
                    for i in 0..n {
                        if r < w[i] {
                            pick = root.legal_actions[i];
                            break;
                        }
                        r -= w[i];
                    }
                    pick
                };
                (chosen, pi)
            })
        })
    }

    /// Argmax-visits search.
    pub fn search(&self, state: &GameState) -> (u8, [f32; 4]) {
        self.search_with_temp(state, 0.0)
    }
}

impl LudoBot for TwoPlayerMCTSBot {
    fn name(&self) -> &'static str {
        "TwoPlayerMCTS"
    }

    fn select_move(&self, state: &GameState) -> u8 {
        self.search(state).0
    }
}
