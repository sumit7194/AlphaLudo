//! Network-guided Expectimax-PUCT for 2-player Ludo (exact dice expectations).
//!
//! Why (2026-10-02): `net_mcts` samples ONE dice roll per simulation at every
//! chance node. Ludo positions swing hard with the dice, so with tens of
//! simulations per move the per-move averages are dominated by dice noise:
//! the v15.2 champion + 50-sim net-MCTS beat the bare champion only 51.7%
//! ±2.9, although its value head is strong (AUC 0.867, calibrated).
//! TD-Gammon's lesson for dice games: look ahead with the EXACT expectation
//! over rolls.
//!
//! Here, when a move is first tried, its chance node is fully expanded: all
//! six faces are rolled, identical results merged (from base, 1-5 usually
//! coincide), and every resulting decision state is evaluated by the network
//! in one batch. No-move results and forced moves are resolved recursively
//! (bounded depth, calibrated evaluator beyond). A chance node's value is the
//! probability-weighted average of its outcomes — exact, no sampling. So the
//! first visit of each root move is a precise 1-ply lookahead; further budget
//! deepens the most promising lines by PUCT.
//!
//! Values are stored from SEAT 0's perspective in [-1, 1] (2-player zero-sum);
//! the chooser at a decision node flips the sign when it is seat 2. Decision
//! node value = (network value + sum_a N(a) * Q(a)) / (1 + sum_a N(a)).
//! Budget is counted in network evaluations.

use ludo_core::GameState;
use rand::Rng;

use crate::evaluator::leaf_value;
use crate::mcts_2p::sample_dirichlet_03;
use crate::net_mcts::cell_policy_to_priors;

const NONE: u32 = u32::MAX;
/// Max chain of consecutive no-move rolls resolved inside one expansion
/// (identical results merge, so a chain is a single path).
const MAX_CHAIN: u32 = 3;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Kind {
    /// Dice known, >= 2 legal moves: priors + value from the network.
    Decision,
    /// Dice known, exactly one legal move (value = its child's).
    Forced,
    /// Dice unknown; children = distinct roll outcomes with probabilities.
    Chance,
    /// Game over (value exact).
    Terminal,
    /// Chain cap reached: calibrated evaluator, not expanded further.
    Leaf,
}

#[derive(Clone, Copy)]
struct Node {
    state: GameState,
    kind: Kind,
    legal: [u8; 4],
    n_legal: u8,
    evaluated: bool,
    priors: [f32; 4],
    /// Network value (seat-0 perspective), decision nodes.
    net_v0: f32,
    /// Current value estimate (seat-0 perspective).
    v0: f32,
    /// Decision: completed visits / virtual loss per slot.
    n: [u32; 4],
    vl: [u16; 4],
    /// Decision/Forced: child per slot. Chance: distinct outcomes.
    children: [u32; 6],
    probs: [f32; 6],
    n_out: u8,
    /// Chance: descents per outcome.
    cn: [u32; 6],
}

impl Node {
    fn blank(state: GameState, kind: Kind) -> Node {
        Node {
            state,
            kind,
            legal: [0; 4],
            n_legal: 0,
            evaluated: true,
            priors: [0.0; 4],
            net_v0: 0.0,
            v0: 0.0,
            n: [0; 4],
            vl: [0; 4],
            children: [NONE; 6],
            probs: [0.0; 6],
            n_out: 0,
            cn: [0; 6],
        }
    }
}

#[inline]
fn to_v0(seat: u8, v: f32) -> f32 {
    if seat == 0 { v } else { -v }
}

#[derive(Clone, Copy, Debug)]
pub struct NetExpectiConfig {
    /// Network evaluations per search (the root's own evaluation included).
    pub max_evals: u32,
    pub c_puct: f32,
    pub fpu_reduction: f32,
    /// Expansions launched per search per round (virtual-loss batch).
    pub expansions_per_round: u32,
    pub dirichlet_eps: f32,
    /// Temperature of the recorded policy target softmax(Q / pi_temp) over
    /// root moves (Q in [-1, 1]).
    pub pi_temp: f32,
    /// Final move = most visited (prior-guided PUCT; AlphaZero-style)
    /// instead of highest value.
    pub pick_by_visits: bool,
}

impl Default for NetExpectiConfig {
    fn default() -> Self {
        NetExpectiConfig {
            max_evals: 100,
            c_puct: 1.5,
            fpu_reduction: 0.2,
            expansions_per_round: 4,
            dirichlet_eps: 0.0,
            pi_temp: 0.05,
            pick_by_visits: false,
        }
    }
}

struct Expansion {
    /// Node indices from the root down to the expanded chance node
    /// (decision / forced / chance nodes), with the slot taken at decisions.
    start: u32,
    len: u32,
    remaining: u32,
}

pub struct NetExpectiSearch {
    arena: Vec<Node>,
    paths: Vec<(u32, u8)>,
    path: Vec<(u32, u8)>,
    expansions: Vec<Option<Expansion>>,
    /// (node awaiting the network, expansion id or NONE for the root)
    waiting: Vec<(u32, u32)>,
    root_requested: bool,
    evals_requested: u32,
    in_flight: u32,
    new_nodes: Vec<u32>,
    /// A round launched nothing while nothing was in flight: the tree is
    /// exhausted (every line ends in terminals / chain-cap leaves).
    stalled: bool,
}

impl Default for NetExpectiSearch {
    fn default() -> Self {
        Self::new()
    }
}

impl NetExpectiSearch {
    pub fn new() -> Self {
        NetExpectiSearch {
            arena: Vec::new(),
            paths: Vec::new(),
            path: Vec::new(),
            expansions: Vec::new(),
            waiting: Vec::new(),
            root_requested: false,
            evals_requested: 0,
            in_flight: 0,
            new_nodes: Vec::new(),
            stalled: false,
        }
    }

    pub fn reset(&mut self, root: GameState) {
        self.arena.clear();
        let (legal, count) = root.legal_moves();
        let mut r = Node::blank(root, Kind::Decision);
        r.legal = legal;
        r.n_legal = count as u8;
        r.evaluated = false;
        self.arena.push(r);
        self.paths.clear();
        self.expansions.clear();
        self.waiting.clear();
        self.root_requested = false;
        self.evals_requested = 0;
        self.in_flight = 0;
        self.stalled = false;
    }

    pub fn finished(&self, cfg: &NetExpectiConfig) -> bool {
        self.arena[0].evaluated && self.in_flight == 0 && (self.evals_requested >= cfg.max_evals || self.stalled)
    }

    pub fn node_state(&self, node: u32) -> &GameState {
        &self.arena[node as usize].state
    }

    fn push(&mut self, n: Node) -> u32 {
        self.arena.push(n);
        (self.arena.len() - 1) as u32
    }

    /// Creates the node for a post-move state (terminal or chance) and, for a
    /// chance node, expands it. Returns the node index.
    fn make_after_move(&mut self, after: GameState, exp: u32, depth: u32) -> u32 {
        if after.is_terminal {
            let mut t = Node::blank(after, Kind::Terminal);
            t.v0 = if after.winner == 0 { 1.0 } else { -1.0 };
            return self.push(t);
        }
        let c = self.push(Node::blank(after, Kind::Chance));
        self.expand_chance(c, exp, depth);
        c
    }

    /// Rolls all six faces at chance node `c`, merging identical results.
    fn expand_chance(&mut self, c: u32, exp: u32, depth: u32) {
        let base = self.arena[c as usize].state;
        let mut outs: [(GameState, f32, bool); 6] = [(base, 0.0, false); 6];
        let mut k = 0usize;
        for face in 1..=6u8 {
            let mut s = base;
            let decides = s.roll(face);
            if let Some(j) = outs[..k].iter().position(|o| o.0 == s) {
                outs[j].1 += 1.0 / 6.0;
            } else {
                outs[k] = (s, 1.0 / 6.0, decides);
                k += 1;
            }
        }
        for (j, &(s, p, decides)) in outs[..k].iter().enumerate() {
            let child = if decides {
                // Any position with a move to make — including a forced one —
                // costs exactly one network evaluation. (Recursing through
                // forced moves fanned out 6-way per level: one expansion
                // could spend thousands of evaluations.)
                let (legal, count) = s.legal_moves();
                let mut d = Node::blank(s, Kind::Decision);
                d.legal = legal;
                d.n_legal = count as u8;
                d.evaluated = false;
                let di = self.push(d);
                self.waiting.push((di, exp));
                self.new_nodes.push(di);
                if exp != NONE {
                    if let Some(e) = self.expansions[exp as usize].as_mut() {
                        e.remaining += 1;
                    }
                }
                di
            } else if depth + 1 >= MAX_CHAIN {
                self.leaf(s)
            } else {
                let ci = self.push(Node::blank(s, Kind::Chance));
                self.expand_chance(ci, exp, depth + 1);
                ci
            };
            let cn = &mut self.arena[c as usize];
            cn.children[j] = child;
            cn.probs[j] = p;
        }
        self.arena[c as usize].n_out = k as u8;
    }

    fn leaf(&mut self, s: GameState) -> u32 {
        let mut l = Node::blank(s, Kind::Leaf);
        // leaf_value is from the roller's perspective, pre-roll.
        l.v0 = to_v0(s.current_player, leaf_value(s.current_player, &s));
        self.push(l)
    }

    /// Recomputes values bottom-up over a freshly expanded subtree.
    fn settle(&mut self, idx: u32) -> f32 {
        let n = self.arena[idx as usize];
        let v = match n.kind {
            Kind::Terminal | Kind::Leaf => n.v0,
            Kind::Decision => self.decision_value(idx),
            Kind::Forced => self.settle(n.children[0]),
            Kind::Chance => {
                let mut v = 0.0;
                for j in 0..n.n_out as usize {
                    v += n.probs[j] * self.settle(n.children[j]);
                }
                v
            }
        };
        self.arena[idx as usize].v0 = v;
        v
    }

    fn decision_value(&self, idx: u32) -> f32 {
        let n = &self.arena[idx as usize];
        let mut num = n.net_v0;
        let mut den = 1.0;
        for a in 0..n.n_legal as usize {
            if n.n[a] > 0 && n.children[a] != NONE {
                num += n.n[a] as f32 * self.arena[n.children[a] as usize].v0;
                den += n.n[a] as f32;
            }
        }
        num / den
    }

    /// Recomputes values of the path nodes, deepest first.
    fn backup_path(&mut self, start: u32, len: u32, completed: bool) {
        for i in (start..start + len).rev() {
            let (ni, slot) = self.paths[i as usize];
            let kind = self.arena[ni as usize].kind;
            if kind == Kind::Decision && slot != u8::MAX {
                let s = slot as usize;
                let nd = &mut self.arena[ni as usize];
                nd.vl[s] -= 1;
                if completed {
                    nd.n[s] += 1;
                }
            }
            let v = match kind {
                Kind::Decision => self.decision_value(ni),
                Kind::Forced => self.arena[self.arena[ni as usize].children[0] as usize].v0,
                Kind::Chance => {
                    let n = self.arena[ni as usize];
                    (0..n.n_out as usize).map(|j| n.probs[j] * self.arena[n.children[j] as usize].v0).sum()
                }
                _ => self.arena[ni as usize].v0,
            };
            self.arena[ni as usize].v0 = v;
        }
    }

    fn select(&self, idx: u32, cfg: &NetExpectiConfig) -> usize {
        let n = &self.arena[idx as usize];
        let seat = n.state.current_player;
        let k = n.n_legal as usize;
        if idx == 0 {
            // Root: every legal move gets its exact 1-ply expansion first
            // (highest prior first) before PUCT deepens anything.
            let mut pick: Option<usize> = None;
            for a in 0..k {
                if n.children[a] == NONE && pick.map_or(true, |b| n.priors[a] > n.priors[b]) {
                    pick = Some(a);
                }
            }
            if let Some(a) = pick {
                return a;
            }
        }
        let total: u32 = (0..k).map(|a| n.n[a] + n.vl[a] as u32).sum();
        let sq = (total.max(1) as f32).sqrt();
        let fpu = to_v0(seat, n.v0) - cfg.fpu_reduction;
        let mut best = 0;
        let mut best_s = f32::NEG_INFINITY;
        for a in 0..k {
            let visits = n.n[a] + n.vl[a] as u32;
            let q = if n.children[a] != NONE && visits > 0 {
                let q = to_v0(seat, self.arena[n.children[a] as usize].v0);
                (q * n.n[a] as f32 - n.vl[a] as f32) / visits as f32
            } else {
                fpu
            };
            let s = q + cfg.c_puct * n.priors[a] * sq / (1.0 + visits as f32);
            if s > best_s {
                best_s = s;
                best = a;
            }
        }
        best
    }

    /// Launches up to `expansions_per_round` expansions; pushes the nodes that
    /// need a network evaluation into `out`.
    pub fn run_round(&mut self, cfg: &NetExpectiConfig, out: &mut Vec<u32>) {
        if !self.arena[0].evaluated {
            if !self.root_requested {
                self.root_requested = true;
                self.waiting.push((0, NONE));
                self.in_flight += 1;
                self.evals_requested += 1;
                out.push(0);
            }
            return;
        }
        let before = out.len();
        let mut launched = 0;
        let mut attempts = 0;
        while launched < cfg.expansions_per_round
            && self.evals_requested < cfg.max_evals
            && attempts < cfg.expansions_per_round * 4
        {
            attempts += 1;
            if self.descend_and_expand(cfg, out) {
                launched += 1;
            }
        }
        if out.len() == before && self.in_flight == 0 && self.evals_requested < cfg.max_evals {
            self.stalled = true;
        }
    }

    /// One descent. Returns true if it did useful work (expansion launched or
    /// a completed no-eval expansion backed up).
    fn descend_and_expand(&mut self, cfg: &NetExpectiConfig, out: &mut Vec<u32>) -> bool {
        self.path.clear();
        let mut idx = 0u32;
        loop {
            let n = self.arena[idx as usize];
            match n.kind {
                Kind::Decision => {
                    if !n.evaluated {
                        self.undo_vl();
                        return false;
                    }
                    let a = self.select(idx, cfg);
                    self.arena[idx as usize].vl[a] += 1;
                    self.path.push((idx, a as u8));
                    if n.children[a] != NONE {
                        idx = n.children[a];
                        continue;
                    }
                    // Expand this move: its post-move node (+ full dice expansion).
                    let exp_id = self.expansions.len() as u32;
                    self.expansions.push(Some(Expansion { start: 0, len: 0, remaining: 0 }));
                    self.new_nodes.clear();
                    let after = n.state.apply_move(n.legal[a]);
                    let child = self.make_after_move(after, exp_id, 0);
                    self.arena[idx as usize].children[a] = child;
                    let start = self.paths.len() as u32;
                    self.paths.extend_from_slice(&self.path);
                    let len = self.path.len() as u32;
                    let remaining = self.expansions[exp_id as usize].as_ref().unwrap().remaining;
                    if remaining == 0 {
                        self.settle(child);
                        self.backup_path(start, len, true);
                        self.paths.truncate(start as usize);
                        self.expansions[exp_id as usize] = None;
                    } else {
                        let e = self.expansions[exp_id as usize].as_mut().unwrap();
                        e.start = start;
                        e.len = len;
                        self.in_flight += remaining;
                        self.evals_requested += remaining;
                        out.extend_from_slice(&self.new_nodes);
                    }
                    return true;
                }
                Kind::Forced => {
                    self.path.push((idx, u8::MAX));
                    idx = n.children[0];
                }
                Kind::Chance => {
                    // Descend to the outcome most under-visited for its probability.
                    let mut best = 0;
                    let mut best_s = f32::NEG_INFINITY;
                    for j in 0..n.n_out as usize {
                        let s = n.probs[j] / (1.0 + n.cn[j] as f32);
                        if s > best_s {
                            best_s = s;
                            best = j;
                        }
                    }
                    self.arena[idx as usize].cn[best] += 1;
                    self.path.push((idx, u8::MAX));
                    idx = n.children[best];
                }
                Kind::Terminal | Kind::Leaf => {
                    // Nothing to expand on this line: count the visit, move on.
                    let start = self.paths.len() as u32;
                    self.paths.extend_from_slice(&self.path);
                    self.backup_path(start, self.path.len() as u32, true);
                    self.paths.truncate(start as usize);
                    return true;
                }
            }
        }
    }

    fn undo_vl(&mut self) {
        for &(ni, s) in &self.path {
            if s != u8::MAX && self.arena[ni as usize].kind == Kind::Decision {
                self.arena[ni as usize].vl[s as usize] -= 1;
            }
        }
    }

    /// Network answer for waiting node `node`: policy over 225 cells and
    /// P(mover wins).
    pub fn answer(&mut self, node: u32, policy: &[f32], p_win: f32, noise: Option<&mut impl Rng>, cfg: &NetExpectiConfig) {
        let pos = self.waiting.iter().position(|&(n, _)| n == node).expect("answer for a node that is not waiting");
        let (_, exp) = self.waiting.swap_remove(pos);
        {
            let nd = &mut self.arena[node as usize];
            let k = nd.n_legal as usize;
            let legal = nd.legal;
            let mut pri = cell_policy_to_priors(&nd.state, &legal[..k], policy);
            if node == 0 && cfg.dirichlet_eps > 0.0 {
                if let Some(r) = noise {
                    let z = sample_dirichlet_03(k, r);
                    for i in 0..k {
                        pri[i] = (1.0 - cfg.dirichlet_eps) * pri[i] + cfg.dirichlet_eps * z[i];
                    }
                }
            }
            nd.priors = pri;
            nd.net_v0 = to_v0(nd.state.current_player, 2.0 * p_win - 1.0);
            nd.v0 = nd.net_v0;
            nd.evaluated = true;
        }
        self.in_flight -= 1;
        if exp == NONE {
            return;
        }
        let done = {
            let e = self.expansions[exp as usize].as_mut().unwrap();
            e.remaining -= 1;
            e.remaining == 0
        };
        if done {
            let e = self.expansions[exp as usize].take().unwrap();
            // The expanded child hangs off the last decision edge of the path.
            let (pi, slot) = self.paths[(e.start + e.len - 1) as usize];
            let child = self.arena[pi as usize].children[slot as usize];
            self.settle(child);
            self.backup_path(e.start, e.len, true);
        }
        if self.in_flight == 0 {
            self.paths.clear();
            self.expansions.clear();
        }
    }

    /// Root move values for the mover (NEG_INFINITY if not expanded).
    fn root_q(&self) -> ([f32; 4], bool) {
        let r = &self.arena[0];
        let seat = r.state.current_player;
        let mut q = [f32::NEG_INFINITY; 4];
        let mut all = true;
        for a in 0..r.n_legal as usize {
            if r.children[a] != NONE && r.n[a] > 0 {
                q[a] = to_v0(seat, self.arena[r.children[a] as usize].v0);
            } else {
                all = false;
            }
        }
        (q, all)
    }

    /// Policy target by TOKEN index: softmax(Q / pi_temp) over root moves when
    /// every move has been expanded, else the priors.
    pub fn root_pi_temp(&self, pi_temp: f32) -> [f32; 4] {
        let r = &self.arena[0];
        let k = r.n_legal as usize;
        let (q, all) = self.root_q();
        let mut pi = [0f32; 4];
        if all {
            let mx = q[..k].iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut w = [0f32; 4];
            let mut sum = 0.0;
            for a in 0..k {
                w[a] = ((q[a] - mx) / pi_temp.max(1e-4)).exp();
                sum += w[a];
            }
            for a in 0..k {
                pi[r.legal[a] as usize] += w[a] / sum;
            }
        } else {
            for a in 0..k {
                pi[r.legal[a] as usize] += r.priors[a];
            }
        }
        pi
    }

    pub fn root_pi(&self) -> [f32; 4] {
        self.root_pi_temp(0.05)
    }

    /// Highest-value root move once every move is expanded (values are exact
    /// dice expectations, deepened where PUCT spent budget); with a budget too
    /// small to expand them all, the highest prior.
    pub fn best_move_cfg(&self, cfg: &NetExpectiConfig) -> u8 {
        if !cfg.pick_by_visits {
            return self.best_move();
        }
        let r = &self.arena[0];
        let (q, _) = self.root_q();
        let mut best = 0;
        for a in 1..r.n_legal as usize {
            if (r.n[a], q[a], r.priors[a]) > (r.n[best], q[best], r.priors[best]) {
                best = a;
            }
        }
        r.legal[best]
    }

    pub fn best_move(&self) -> u8 {
        let r = &self.arena[0];
        let (q, all) = self.root_q();
        let mut best = 0;
        for a in 1..r.n_legal as usize {
            let better = if all { (q[a], r.priors[a]) > (q[best], r.priors[best]) } else { r.priors[a] > r.priors[best] };
            if better {
                best = a;
            }
        }
        r.legal[best]
    }

    /// Root move sampled from root_pi_temp (exploration in self-play).
    pub fn sample_move(&self, rng: &mut impl Rng, pi_temp: f32) -> u8 {
        let pi = self.root_pi_temp(pi_temp);
        let x: f32 = rng.gen();
        let mut acc = 0.0;
        let r = &self.arena[0];
        for a in 0..r.n_legal as usize {
            acc += pi[r.legal[a] as usize];
            if x < acc {
                return r.legal[a];
            }
        }
        self.best_move()
    }

    /// (token, visits, value for the mover) per root move — diagnostics.
    pub fn root_stats(&self) -> Vec<(u8, u32, f32)> {
        let r = &self.arena[0];
        let seat = r.state.current_player;
        (0..r.n_legal as usize)
            .map(|a| {
                let q = if r.children[a] != NONE { to_v0(seat, self.arena[r.children[a] as usize].v0) } else { f32::NAN };
                (r.legal[a], r.n[a], q)
            })
            .collect()
    }

    pub fn in_flight(&self) -> u32 {
        self.in_flight
    }
}
