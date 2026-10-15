//! Network-guided Max^n MCTS for 2P and 4P (V17 unified encoding) + a batched
//! evaluation farm. Built 2026-10-15 for the "search headroom" test (does search
//! on top of the V17 network beat the raw network?) and as the engine for a
//! later expert-iteration stage.
//!
//! The search never runs a model: a simulation that reaches a new decision node
//! with >= 2 legal moves parks its path and emits an evaluation REQUEST (mover's
//! POV, unified encoding). The caller answers a whole batch with (policy over
//! 225 cells, 4-way win probabilities over RELATIVE seats). Values are payoff
//! vectors over ABSOLUTE seats (win probabilities, summing to 1 over active
//! seats); each decision node selects by PUCT on its mover's own coordinate
//! (Max^n). In 2P (two active seats) this is exactly zero-sum MCTS.
//!
//! Tree: decision nodes (dice known) and chance nodes (dice unknown); every
//! move leads to a chance node; each simulation samples a fresh roll. Rolls
//! with no legal move / three-six forfeits resolve via GameState::roll. Forced
//! decisions are passed through without a network call. Virtual loss = a visit
//! with payoff 0 for the selecting player.

use ludo_core::{
    encode_unified_into, uni_action_cell, uni_legal_mask_into, GameState, NUM_BOARD_CELLS, NUM_PLAYERS, UNI_FRAME_SIZE,
};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

use crate::expectimax::LudoBot;
use crate::farm::{make_bot, BotKind};

const NONE: u32 = u32::MAX;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Decision,
    Chance,
}

#[derive(Clone, Copy)]
struct Node {
    state: GameState,
    kind: Kind,
    player: u8,
    legal: [u8; 4],
    n_legal: u8,
    evaluated: bool,
    priors: [f32; 4],
    /// Network payoff vector of this node (absolute seats).
    value: [f32; 4],
    n: [u32; 4],
    /// Sum of payoff vectors per slot: w[slot][seat].
    w: [[f32; 4]; 4],
    vl: [u16; 4],
    children: [u32; 6],
}

impl Node {
    fn chance(state: GameState) -> Node {
        Node {
            state,
            kind: Kind::Chance,
            player: state.current_player,
            legal: [0; 4],
            n_legal: 0,
            evaluated: true,
            priors: [0.0; 4],
            value: [0.0; 4],
            n: [0; 4],
            w: [[0.0; 4]; 4],
            vl: [0; 4],
            children: [NONE; 6],
        }
    }

    fn decision(state: GameState) -> Node {
        let (legal, count) = state.legal_moves();
        let mut node = Node::chance(state);
        node.kind = Kind::Decision;
        node.legal = legal;
        node.n_legal = count as u8;
        node.evaluated = count == 1;
        if count == 1 {
            node.priors[0] = 1.0;
        }
        node
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NetMaxNConfig {
    pub sims: u32,
    /// PUCT constant on win-probability values in [0, 1].
    pub c_puct: f32,
    /// Unvisited edge = node's own network value for the mover − fpu_reduction.
    pub fpu_reduction: f32,
    pub leaves_per_round: u32,
}

impl Default for NetMaxNConfig {
    fn default() -> Self {
        NetMaxNConfig { sims: 200, c_puct: 1.0, fpu_reduction: 0.1, leaves_per_round: 8 }
    }
}

/// Network policy (225 cells, mover POV) -> priors over the legal tokens.
pub fn uni_policy_to_priors(state: &GameState, legal: &[u8], policy: &[f32]) -> [f32; 4] {
    let mut cells = [0usize; 4];
    for (i, &t) in legal.iter().enumerate() {
        cells[i] = uni_action_cell(state, t);
    }
    let mut pri = [0f32; 4];
    let mut sum = 0.0;
    for i in 0..legal.len() {
        let share = cells[..legal.len()].iter().filter(|&&c| c == cells[i]).count() as f32;
        pri[i] = policy[cells[i]].max(0.0) / share;
        sum += pri[i];
    }
    for i in 0..legal.len() {
        pri[i] = if sum > 0.0 { pri[i] / sum } else { 1.0 / legal.len() as f32 };
    }
    pri
}

/// Relative-seat value (index k = k seats after the mover) -> absolute seats.
#[inline]
pub fn rel_to_abs(state: &GameState, rel: &[f32]) -> [f32; 4] {
    let mover = state.current_player as usize;
    let mut out = [0f32; 4];
    let mut sum = 0.0;
    for k in 0..NUM_PLAYERS {
        let s = (mover + k) % NUM_PLAYERS;
        if state.active_players[s] {
            out[s] = rel[k].max(0.0);
            sum += out[s];
        }
    }
    if sum > 0.0 {
        for v in out.iter_mut() {
            *v /= sum;
        }
    }
    out
}

pub struct NetMaxNSearch {
    arena: Vec<Node>,
    pending: Vec<(u32, u32, u32)>,
    paths: Vec<(u32, u8)>,
    path: Vec<(u32, u8)>,
    root_requested: bool,
    sims_done: u32,
    in_flight: u32,
}

impl Default for NetMaxNSearch {
    fn default() -> Self {
        Self::new()
    }
}

impl NetMaxNSearch {
    pub fn new() -> Self {
        NetMaxNSearch {
            arena: Vec::new(),
            pending: Vec::new(),
            paths: Vec::new(),
            path: Vec::new(),
            root_requested: false,
            sims_done: 0,
            in_flight: 0,
        }
    }

    pub fn reset(&mut self, root: GameState) {
        self.arena.clear();
        self.arena.push(Node::decision(root));
        self.pending.clear();
        self.paths.clear();
        self.root_requested = false;
        self.sims_done = 0;
        self.in_flight = 0;
    }

    pub fn finished(&self, cfg: &NetMaxNConfig) -> bool {
        self.arena[0].evaluated && self.sims_done >= cfg.sims && self.in_flight == 0
    }

    #[inline]
    fn select(node: &Node, cfg: &NetMaxNConfig) -> usize {
        let k = node.n_legal as usize;
        if k == 1 {
            return 0;
        }
        let me = node.player as usize;
        let total: u32 = (0..k).map(|i| node.n[i] + node.vl[i] as u32).sum();
        let sqrt_total = (total.max(1) as f32).sqrt();
        let fpu = node.value[me] - cfg.fpu_reduction;
        let mut best = 0;
        let mut best_score = f32::NEG_INFINITY;
        for i in 0..k {
            let n = node.n[i] + node.vl[i] as u32;
            let q = if n > 0 { node.w[i][me] / n as f32 } else { fpu };
            let s = q + cfg.c_puct * node.priors[i] * sqrt_total / (1.0 + n as f32);
            if s > best_score {
                best_score = s;
                best = i;
            }
        }
        best
    }

    fn backup(arena: &mut [Node], path: &[(u32, u8)], v: &[f32; 4], had_vl: bool) {
        for &(ni, slot) in path {
            let node = &mut arena[ni as usize];
            let s = slot as usize;
            node.n[s] += 1;
            for p in 0..NUM_PLAYERS {
                node.w[s][p] += v[p];
            }
            if had_vl {
                node.vl[s] -= 1;
            }
        }
    }

    /// Runs up to `leaves_per_round` simulations; pushes requested node ids.
    pub fn run_round(&mut self, cfg: &NetMaxNConfig, rng: &mut impl Rng, out: &mut Vec<u32>) -> usize {
        let before = out.len();
        if !self.arena[0].evaluated {
            if !self.root_requested {
                self.root_requested = true;
                self.pending.push((0, self.paths.len() as u32, 0));
                self.in_flight += 1;
                out.push(0);
            }
            return out.len() - before;
        }
        let mut launched = 0;
        let mut attempts = 0;
        while launched < cfg.leaves_per_round
            && self.sims_done + self.in_flight < cfg.sims
            && attempts < cfg.leaves_per_round * 4
        {
            attempts += 1;
            match self.simulate(cfg, rng) {
                Some(Some(leaf)) => {
                    out.push(leaf);
                    launched += 1;
                }
                Some(None) => launched += 1,
                None => {}
            }
        }
        out.len() - before
    }

    fn simulate(&mut self, cfg: &NetMaxNConfig, rng: &mut impl Rng) -> Option<Option<u32>> {
        self.path.clear();
        let mut idx: u32 = 0;
        loop {
            let node = self.arena[idx as usize];
            match node.kind {
                Kind::Decision => {
                    if !node.evaluated {
                        for &(ni, s) in &self.path {
                            self.arena[ni as usize].vl[s as usize] -= 1;
                        }
                        return None;
                    }
                    let slot = Self::select(&node, cfg);
                    self.arena[idx as usize].vl[slot] += 1;
                    self.path.push((idx, slot as u8));
                    let child = node.children[slot];
                    if child != NONE {
                        idx = child;
                        continue;
                    }
                    let after = node.state.apply_move(node.legal[slot]);
                    if after.is_terminal {
                        let mut v = [0f32; 4];
                        v[after.winner as usize] = 1.0;
                        Self::backup(&mut self.arena, &self.path, &v, true);
                        self.sims_done += 1;
                        return Some(None);
                    }
                    self.arena.push(Node::chance(after));
                    let ni = (self.arena.len() - 1) as u32;
                    self.arena[idx as usize].children[slot] = ni;
                    idx = ni;
                }
                Kind::Chance => {
                    let face = rng.gen_range(1..=6u8);
                    let child = node.children[(face - 1) as usize];
                    if child != NONE {
                        idx = child;
                        continue;
                    }
                    let mut s = node.state;
                    let new = if s.roll(face) { Node::decision(s) } else { Node::chance(s) };
                    self.arena.push(new);
                    let ni = (self.arena.len() - 1) as u32;
                    self.arena[idx as usize].children[(face - 1) as usize] = ni;
                    if new.kind == Kind::Decision && !new.evaluated {
                        let start = self.paths.len() as u32;
                        self.paths.extend_from_slice(&self.path);
                        self.pending.push((ni, start, self.path.len() as u32));
                        self.in_flight += 1;
                        return Some(Some(ni));
                    }
                    idx = ni;
                }
            }
        }
    }

    pub fn node_state(&self, node: u32) -> &GameState {
        &self.arena[node as usize].state
    }

    /// Network answer for pending `node`: policy (225, mover POV) and value
    /// over RELATIVE seats (4).
    pub fn answer(&mut self, node: u32, policy: &[f32], value_rel: &[f32]) {
        let pos = self.pending.iter().position(|&(n, _, _)| n == node).expect("answer for a node that is not pending");
        let (_, start, len) = self.pending.swap_remove(pos);
        let v = {
            let nd = &mut self.arena[node as usize];
            let k = nd.n_legal as usize;
            let legal = nd.legal;
            nd.priors = uni_policy_to_priors(&nd.state, &legal[..k], policy);
            nd.value = rel_to_abs(&nd.state, value_rel);
            nd.evaluated = true;
            nd.value
        };
        if len > 0 {
            let path: Vec<(u32, u8)> = self.paths[start as usize..(start + len) as usize].to_vec();
            Self::backup(&mut self.arena, &path, &v, true);
            self.sims_done += 1;
        }
        self.in_flight -= 1;
        if self.pending.is_empty() {
            self.paths.clear();
        }
    }

    /// Root visit distribution by TOKEN index.
    pub fn root_pi(&self) -> [f32; 4] {
        let r = &self.arena[0];
        let mut pi = [0f32; 4];
        let total: u32 = r.n[..r.n_legal as usize].iter().sum();
        for i in 0..r.n_legal as usize {
            pi[r.legal[i] as usize] = if total > 0 { r.n[i] as f32 / total as f32 } else { r.priors[i] };
        }
        pi
    }

    pub fn best_move(&self) -> u8 {
        let r = &self.arena[0];
        let mut best = 0;
        for i in 1..r.n_legal as usize {
            if (r.n[i], r.priors[i]) > (r.n[best], r.priors[best]) {
                best = i;
            }
        }
        r.legal[best]
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Evaluation farm
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SeatSpec {
    /// Network-guided Max^n MCTS with this model.
    Search { model: u8, cfg: NetMaxNConfig },
    /// The bare network: argmax of its policy.
    Greedy { model: u8 },
    Bot(BotKind),
}

enum Seat {
    Empty,
    Search { model: u8, cfg: NetMaxNConfig, tree: Box<NetMaxNSearch>, active: bool },
    Greedy { model: u8 },
    Bot(Box<dyn LudoBot>),
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Wait {
    None,
    Net,
    Bot,
}

struct Game {
    state: GameState,
    seats: [Seat; 4],
    role_of_seat: [u8; 4],
    rng: StdRng,
    moves: u32,
    wait: Wait,
    done: bool,
    id: u64,
}

#[derive(Clone, Copy)]
enum Req {
    Greedy { game: u32 },
    Leaf { game: u32, node: u32 },
}

#[derive(Clone, Copy, Debug)]
pub struct EvalResult {
    pub game_id: u64,
    pub role_of_seat: [u8; 4],
    pub winner: i8,
    pub moves: u32,
}

pub struct SearchFarm {
    roles: Vec<SeatSpec>,
    games: Vec<Game>,
    n_models: usize,
    concurrency: usize,
    game_limit: u64,
    next_id: u64,
    max_moves: u32,
    rng: StdRng,
    req: Vec<Vec<Req>>,
    req_x: Vec<Vec<i8>>,
    req_mask: Vec<Vec<f32>>,
    pub results: Vec<EvalResult>,
    /// Network evaluations requested by search seats / greedy seats.
    pub search_evals: u64,
    pub greedy_evals: u64,
}

impl SearchFarm {
    /// `roles`: 2 or 4 seat specs (role i is placed on a shuffled seat each game).
    pub fn new(roles: Vec<SeatSpec>, n_models: usize, games: u64, concurrency: usize, seed: u64) -> Self {
        assert!(roles.len() == 2 || roles.len() == 4);
        SearchFarm {
            roles,
            games: Vec::new(),
            n_models,
            concurrency,
            game_limit: games,
            next_id: 0,
            max_moves: 1500,
            rng: StdRng::seed_from_u64(seed),
            req: (0..n_models).map(|_| Vec::new()).collect(),
            req_x: (0..n_models).map(|_| Vec::new()).collect(),
            req_mask: (0..n_models).map(|_| Vec::new()).collect(),
            results: Vec::new(),
            search_evals: 0,
            greedy_evals: 0,
        }
    }

    fn new_game(&mut self) -> Game {
        use rand::seq::SliceRandom;
        let n = self.roles.len();
        let state = if n == 2 { GameState::new_2p() } else { GameState::new_4p() };
        let mut seat_ids: Vec<u8> = if n == 2 { vec![0, 2] } else { vec![0, 1, 2, 3] };
        seat_ids.shuffle(&mut self.rng);
        let mut seats: [Seat; 4] = [Seat::Empty, Seat::Empty, Seat::Empty, Seat::Empty];
        let mut role_of_seat = [255u8; 4];
        for (role, &seat) in seat_ids.iter().enumerate() {
            role_of_seat[seat as usize] = role as u8;
            seats[seat as usize] = match self.roles[role] {
                SeatSpec::Search { model, cfg } => {
                    Seat::Search { model, cfg, tree: Box::new(NetMaxNSearch::new()), active: false }
                }
                SeatSpec::Greedy { model } => Seat::Greedy { model },
                SeatSpec::Bot(k) => Seat::Bot(make_bot(k, seat, n, self.rng.gen())),
            };
        }
        let id = self.next_id;
        self.next_id += 1;
        Game {
            state,
            seats,
            role_of_seat,
            rng: StdRng::seed_from_u64(self.rng.gen()),
            moves: 0,
            wait: Wait::None,
            done: false,
            id,
        }
    }

    fn push_request(&mut self, model: u8, st: &GameState, r: Req) {
        let m = model as usize;
        let base = self.req_x[m].len();
        self.req_x[m].resize(base + UNI_FRAME_SIZE, 0);
        encode_unified_into(st, st.current_player, &mut self.req_x[m][base..]);
        let mb = self.req_mask[m].len();
        self.req_mask[m].resize(mb + NUM_BOARD_CELLS, 0.0);
        uni_legal_mask_into(st, &mut self.req_mask[m][mb..]);
        self.req[m].push(r);
    }

    /// Advances game gi until it waits for the network or a bot, or ends.
    fn advance(&mut self, gi: usize) {
        loop {
            let max_moves = self.max_moves;
            let g = &mut self.games[gi];
            if g.done || g.wait != Wait::None {
                return;
            }
            if g.state.is_terminal || g.moves >= max_moves {
                g.done = true;
                let winner = if g.state.is_terminal { g.state.winner } else { -1 };
                self.results.push(EvalResult { game_id: g.id, role_of_seat: g.role_of_seat, winner, moves: g.moves });
                return;
            }
            if g.state.current_dice_roll == 0 {
                let d = g.rng.gen_range(1..=6u8);
                g.state.roll(d);
                continue;
            }
            let (legal, count) = g.state.legal_moves();
            if count == 1 {
                g.state = g.state.apply_move(legal[0]);
                g.moves += 1;
                continue;
            }
            let cp = g.state.current_player as usize;
            match &mut g.seats[cp] {
                Seat::Empty => unreachable!("empty seat to move"),
                Seat::Bot(_) => {
                    g.wait = Wait::Bot;
                    return;
                }
                Seat::Greedy { model } => {
                    let model = *model;
                    g.wait = Wait::Net;
                    let st = g.state;
                    self.greedy_evals += 1;
                    self.push_request(model, &st, Req::Greedy { game: gi as u32 });
                    return;
                }
                Seat::Search { model, cfg, tree, active } => {
                    let (model, cfg) = (*model, *cfg);
                    if !*active {
                        tree.reset(g.state);
                        *active = true;
                    }
                    if tree.finished(&cfg) {
                        let mv = tree.best_move();
                        *active = false;
                        g.state = g.state.apply_move(mv);
                        g.moves += 1;
                        continue;
                    }
                    let mut out = Vec::new();
                    tree.run_round(&cfg, &mut g.rng, &mut out);
                    if out.is_empty() {
                        // Only terminal/collided simulations this round: try again.
                        continue;
                    }
                    g.wait = Wait::Net;
                    let states: Vec<GameState> = out.iter().map(|&nd| *tree.node_state(nd)).collect();
                    self.search_evals += out.len() as u64;
                    for (k, &nd) in out.iter().enumerate() {
                        self.push_request(model, &states[k], Req::Leaf { game: gi as u32, node: nd });
                    }
                    return;
                }
            }
        }
    }

    /// Advances all games to their next network wait (bots resolved in
    /// parallel). Returns the number of pending network requests (0 = all done).
    pub fn collect(&mut self) -> usize {
        for m in 0..self.n_models {
            self.req[m].clear();
            self.req_x[m].clear();
            self.req_mask[m].clear();
        }
        // Finished games are removed only here: request indices point into `games`.
        self.games.retain(|g| !g.done);
        loop {
            let active = self.games.iter().filter(|g| !g.done).count();
            for _ in active..self.concurrency {
                if self.next_id >= self.game_limit {
                    break;
                }
                let g = self.new_game();
                self.games.push(g); // appended: existing indices stay valid
            }
            for gi in 0..self.games.len() {
                self.advance(gi);
            }
            if !self.games.iter().any(|g| g.wait == Wait::Bot) {
                let refill = self.games.iter().filter(|g| !g.done).count() < self.concurrency
                    && self.next_id < self.game_limit;
                if refill && self.req.iter().all(|r| r.is_empty()) {
                    continue;
                }
                break;
            }
            self.games.par_iter_mut().filter(|g| g.wait == Wait::Bot).for_each(|g| {
                let cp = g.state.current_player as usize;
                if let Seat::Bot(b) = &g.seats[cp] {
                    let a = b.select_move(&g.state);
                    g.state = g.state.apply_move(a);
                    g.moves += 1;
                }
                g.wait = Wait::None;
            });
        }
        self.req.iter().map(|r| r.len()).sum()
    }

    pub fn models_pending(&self) -> Vec<u8> {
        (0..self.n_models).filter(|&m| !self.req[m].is_empty()).map(|m| m as u8).collect()
    }

    pub fn request_x(&self, model: u8) -> &[i8] {
        &self.req_x[model as usize]
    }

    pub fn request_mask(&self, model: u8) -> &[f32] {
        &self.req_mask[model as usize]
    }

    pub fn request_count(&self, model: u8) -> usize {
        self.req[model as usize].len()
    }

    /// policy (n, 225) probabilities; value (n, 4) over RELATIVE seats.
    pub fn submit(&mut self, model: u8, policy: &[f32], value: &[f32]) -> Result<(), String> {
        let m = model as usize;
        let n = self.req[m].len();
        if policy.len() != n * NUM_BOARD_CELLS || value.len() != n * 4 {
            return Err(format!("model {model}: expected {n} answers"));
        }
        let reqs = std::mem::take(&mut self.req[m]);
        for (i, r) in reqs.iter().enumerate() {
            let pol = &policy[i * NUM_BOARD_CELLS..(i + 1) * NUM_BOARD_CELLS];
            let val = &value[i * 4..(i + 1) * 4];
            match *r {
                Req::Greedy { game } => {
                    let g = &mut self.games[game as usize];
                    let (legal, count) = g.state.legal_moves();
                    let pri = uni_policy_to_priors(&g.state, &legal[..count], pol);
                    let mut b = 0;
                    for k in 1..count {
                        if pri[k] > pri[b] {
                            b = k;
                        }
                    }
                    g.state = g.state.apply_move(legal[b]);
                    g.moves += 1;
                    g.wait = Wait::None;
                }
                Req::Leaf { game, node } => {
                    let g = &mut self.games[game as usize];
                    if let Seat::Search { tree, .. } = &mut g.seats[g.state.current_player as usize] {
                        tree.answer(node, pol, val);
                    }
                    g.wait = Wait::None;
                }
            }
        }
        self.req[m] = reqs;
        self.req[m].clear();
        Ok(())
    }

    pub fn in_flight(&self) -> usize {
        self.games.iter().filter(|g| !g.done).count()
    }
}
