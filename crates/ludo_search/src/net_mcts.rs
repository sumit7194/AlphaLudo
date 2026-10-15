//! Network-guided 2-player Expecti-MCTS and a batched game farm.
//!
//! The search is network-AGNOSTIC: it never runs a model. When a simulation
//! reaches a new decision node with >= 2 legal moves it parks the path and
//! emits an evaluation REQUEST (the state); the caller (Python, with the V15
//! GraphTransformer on MPS/CUDA) answers a whole batch of requests at once
//! with (policy over 225 cells, P(mover wins)), and the farm backs them up.
//!
//! Tree: decision nodes (dice known, a player chooses) and chance nodes (dice
//! unknown). Every move leads to a chance node; each simulation samples a fresh
//! roll there. Rolls with no legal move / three-six forfeits resolve (via
//! GameState::roll) to the next roller's chance node. Forced decisions (one
//! legal move) are passed through without a network call. So every network
//! call is on a post-roll decision state — exactly the distribution the V15
//! network was trained on.
//!
//! Values are zero-sum in [-1, 1]; each decision edge stores W from the
//! chooser's perspective. Many simulations per search are in flight at once
//! using virtual loss; with K concurrent games a round produces up to
//! K * leaves_per_round requests.

use ludo_core::{encode_v15_2p_into, v15_own_cell, GameState, NUM_BOARD_CELLS, V15_FRAME_SIZE};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::mcts_2p::sample_dirichlet_03;
use crate::net_expecti::{NetExpectiConfig, NetExpectiSearch};
use crate::records::{
    quantise_pi, SLRecordV2, BOT_NET_GREEDY, BOT_NET_MCTS, FLAG_EXPLORATORY, FLAG_FORCED, FLAG_HAS_PI,
    FLAG_TEACHER_SEAT,
};

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
    /// Priors valid (network answered, or forced pass-through).
    evaluated: bool,
    priors: [f32; 4],
    /// Network value of this node for its mover, in [-1, 1] (FPU baseline).
    value: f32,
    n: [u32; 4],
    w: [f32; 4],
    vl: [u16; 4],
    /// Decision: per legal slot. Chance: per dice face (index face - 1).
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
            value: 0.0,
            n: [0; 4],
            w: [0.0; 4],
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

/// Search hyper-parameters.
#[derive(Clone, Copy, Debug)]
pub struct NetMctsConfig {
    pub sims: u32,
    pub c_puct: f32,
    /// Unvisited edge value = node's own network value - fpu_reduction.
    pub fpu_reduction: f32,
    /// Simulations launched per search per round (virtual-loss batch).
    pub leaves_per_round: u32,
    pub dirichlet_eps: f32,
    /// Keep the matching subtree of the previous search (see reset_reuse).
    pub reuse_tree: bool,
}

impl Default for NetMctsConfig {
    fn default() -> Self {
        NetMctsConfig { sims: 200, c_puct: 1.5, fpu_reduction: 0.2, leaves_per_round: 8, dirichlet_eps: 0.0, reuse_tree: true }
    }
}

/// Converts a 225-cell policy (probabilities, mover POV) to token priors.
/// Tokens stacked on one cell share its mass (they are the same move).
pub fn cell_policy_to_priors(state: &GameState, legal: &[u8], policy: &[f32]) -> [f32; 4] {
    let cp = state.current_player;
    let mut cells = [0usize; 4];
    for (i, &t) in legal.iter().enumerate() {
        cells[i] = v15_own_cell(state.positions[cp as usize][t as usize], cp);
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

/// One search tree (owned by a game; the arena is reused between moves).
pub struct NetSearch {
    arena: Vec<Node>,
    spare: Vec<Node>,
    stack: Vec<(u32, u32)>,
    /// Pending leaf evaluations: (leaf node, start, len) into `paths`.
    pending: Vec<(u32, u32, u32)>,
    paths: Vec<(u32, u8)>,
    path: Vec<(u32, u8)>,
    root_requested: bool,
    sims_done: u32,
    in_flight: u32,
}

impl NetSearch {
    pub fn new() -> Self {
        NetSearch {
            arena: Vec::new(),
            spare: Vec::new(),
            stack: Vec::new(),
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

    /// Like `reset`, but if the previous tree contains an evaluated decision
    /// node for exactly this state (same seat to move, same dice — reached
    /// after our move, the rolls and the opponent's reply), that subtree is
    /// kept: its priors, value and visits carry over and count toward the
    /// simulation budget. Returns the number of reused root visits.
    pub fn reset_reuse(&mut self, root: GameState, cfg: &NetMctsConfig, rng: &mut impl Rng) -> u32 {
        let mut best: Option<(usize, u32)> = None;
        if self.pending.is_empty() {
            for (i, n) in self.arena.iter().enumerate() {
                if n.kind == Kind::Decision && n.evaluated && n.n_legal >= 2 && n.state == root {
                    let v: u32 = n.n[..n.n_legal as usize].iter().sum();
                    if best.map_or(true, |(_, bv)| v > bv) {
                        best = Some((i, v));
                    }
                }
            }
        }
        let Some((bi, visits)) = best else {
            self.reset(root);
            return 0;
        };
        // Copy the subtree into `spare` (DFS, children remapped), then swap.
        self.spare.clear();
        self.stack.clear();
        self.spare.push(self.arena[bi]);
        self.stack.push((bi as u32, 0));
        while let Some((old, new)) = self.stack.pop() {
            let width = if self.arena[old as usize].kind == Kind::Decision { 4 } else { 6 };
            for c in 0..width {
                let oc = self.arena[old as usize].children[c];
                if oc == NONE {
                    continue;
                }
                self.spare.push(self.arena[oc as usize]);
                let nc = (self.spare.len() - 1) as u32;
                self.spare[new as usize].children[c] = nc;
                self.stack.push((oc, nc));
            }
        }
        std::mem::swap(&mut self.arena, &mut self.spare);
        for n in self.arena.iter_mut() {
            n.vl = [0; 4];
        }
        if cfg.dirichlet_eps > 0.0 {
            let r = &mut self.arena[0];
            let k = r.n_legal as usize;
            let noise = sample_dirichlet_03(k, rng);
            for i in 0..k {
                r.priors[i] = (1.0 - cfg.dirichlet_eps) * r.priors[i] + cfg.dirichlet_eps * noise[i];
            }
        }
        self.pending.clear();
        self.paths.clear();
        self.root_requested = true;
        self.sims_done = visits.min(cfg.sims);
        self.in_flight = 0;
        visits
    }

    pub fn finished(&self, cfg: &NetMctsConfig) -> bool {
        self.arena[0].evaluated && self.sims_done >= cfg.sims && self.in_flight == 0
    }

    #[inline]
    fn select(node: &Node, cfg: &NetMctsConfig) -> usize {
        let k = node.n_legal as usize;
        if k == 1 {
            return 0;
        }
        let total: u32 = (0..k).map(|i| node.n[i] + node.vl[i] as u32).sum();
        let sqrt_total = (total.max(1) as f32).sqrt();
        let fpu = node.value - cfg.fpu_reduction;
        let mut best = 0;
        let mut best_score = f32::NEG_INFINITY;
        for i in 0..k {
            let n = node.n[i] + node.vl[i] as u32;
            let q = if n > 0 { (node.w[i] - node.vl[i] as f32) / n as f32 } else { fpu };
            let s = q + cfg.c_puct * node.priors[i] * sqrt_total / (1.0 + n as f32);
            if s > best_score {
                best_score = s;
                best = i;
            }
        }
        best
    }

    /// Backs up a value given as P(`mover` wins) along `path` (removing the
    /// virtual loss if `had_vl`).
    fn backup(arena: &mut [Node], path: &[(u32, u8)], mover: u8, p_win: f32, had_vl: bool) {
        let v = 2.0 * p_win - 1.0;
        for &(ni, slot) in path {
            let node = &mut arena[ni as usize];
            let val = if node.player == mover { v } else { -v };
            let s = slot as usize;
            node.n[s] += 1;
            node.w[s] += val;
            if had_vl {
                node.vl[s] -= 1;
            }
        }
    }

    /// Runs up to `leaves_per_round` simulations, pushing evaluation requests
    /// (node indices) into `out`. Returns how many requests were pushed.
    pub fn run_round(&mut self, cfg: &NetMctsConfig, rng: &mut impl Rng, out: &mut Vec<u32>) -> usize {
        let before = out.len();
        if !self.arena[0].evaluated {
            if !self.root_requested {
                self.root_requested = true;
                self.pending.push((0, self.paths.len() as u32, 0));
                self.in_flight += 1;
                out.push(0);
            }
            return 0 + (out.len() - before);
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
                Some(None) => launched += 1, // terminal: backed up immediately
                None => {}                   // collision with a pending leaf
            }
        }
        out.len() - before
    }

    /// One simulation. Some(Some(leaf)) = parked for evaluation,
    /// Some(None) = reached a terminal (done), None = collision (undone).
    fn simulate(&mut self, cfg: &NetMctsConfig, rng: &mut impl Rng) -> Option<Option<u32>> {
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
                        // Not stored: terminal edges are re-scored on every visit.
                        let w = after.winner as u8;
                        Self::backup(&mut self.arena, &self.path, w, 1.0, true);
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

    /// Applies the network answer for pending leaf `node`.
    pub fn answer(&mut self, node: u32, policy: &[f32], p_win: f32, root_noise: Option<&mut StdRng>, cfg: &NetMctsConfig) {
        let pos = self
            .pending
            .iter()
            .position(|&(n, _, _)| n == node)
            .expect("answer for a node that is not pending");
        let (_, start, len) = self.pending.swap_remove(pos);
        {
            let nd = &mut self.arena[node as usize];
            let k = nd.n_legal as usize;
            let legal = nd.legal;
            let mut pri = cell_policy_to_priors(&nd.state, &legal[..k], policy);
            if node == 0 && cfg.dirichlet_eps > 0.0 {
                if let Some(r) = root_noise {
                    let noise = sample_dirichlet_03(k, r);
                    for i in 0..k {
                        pri[i] = (1.0 - cfg.dirichlet_eps) * pri[i] + cfg.dirichlet_eps * noise[i];
                    }
                }
            }
            nd.priors = pri;
            nd.value = 2.0 * p_win - 1.0;
            nd.evaluated = true;
        }
        if len > 0 {
            let mover = self.arena[node as usize].player;
            let path: Vec<(u32, u8)> = self.paths[start as usize..(start + len) as usize].to_vec();
            Self::backup(&mut self.arena, &path, mover, p_win, true);
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

    /// Most-visited root move (ties broken by prior).
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

    /// Move sampled in proportion to root visits.
    pub fn sample_move(&self, rng: &mut impl Rng) -> u8 {
        let r = &self.arena[0];
        let k = r.n_legal as usize;
        let total: u32 = r.n[..k].iter().sum();
        if total == 0 {
            return self.best_move();
        }
        let mut x = rng.gen_range(0..total);
        for i in 0..k {
            if x < r.n[i] {
                return r.legal[i];
            }
            x -= r.n[i];
        }
        r.legal[k - 1]
    }

    pub fn root_value(&self) -> f32 {
        self.arena[0].value
    }
}

impl Default for NetSearch {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Game farm
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug)]
pub enum SeatKind {
    /// Network-guided MCTS.
    Mcts(NetMctsConfig),
    /// Network-guided Expectimax-PUCT (exact dice expectations).
    Expecti(NetExpectiConfig),
    /// Bare network, argmax of the policy.
    Greedy,
}

#[derive(Clone, Copy, Debug)]
pub struct SeatConfig {
    pub kind: SeatKind,
    /// Which network answers this seat's requests (lets two models play).
    pub model: u8,
    /// Record this seat's decisions as training data.
    pub record: bool,
    /// For the first N of this seat's searched decisions per game, PLAY a move
    /// sampled from the visit counts (the label stays the argmax).
    pub explore_moves: u32,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ReqKind {
    Greedy,
    Leaf(u32),
}

struct Game {
    id: u32,
    state: GameState,
    rng: StdRng,
    /// Seat 0 runs config A when id is even; seat 2 runs A when id is odd.
    a_seat: u8,
    moves: u32,
    search: NetSearch,
    esearch: NetExpectiSearch,
    searching: bool,
    awaiting_greedy: bool,
    decisions: [u32; 4],
    records: Vec<SLRecordV2>,
}

/// One finished game.
#[derive(Clone, Copy, Debug)]
pub struct GameResult {
    pub id: u32,
    pub a_seat: u8,
    /// Winning seat, or -1 if the move cap was hit.
    pub winner: i8,
    pub moves: u32,
}

pub struct NetFarm {
    pub seat_a: SeatConfig,
    pub seat_b: SeatConfig,
    pub max_moves: u32,
    n_games: u32,
    concurrency: usize,
    seed: u64,
    next_id: u32,
    games: Vec<Game>,
    search_rng: StdRng,
    /// Current request batch.
    req: Vec<(usize, ReqKind)>,
    pub req_x: Vec<i8>,
    pub req_mask: Vec<f32>,
    pub req_model: Vec<u8>,
    /// The state behind each request (for native evaluators / tests).
    pub req_states: Vec<GameState>,
    pub results: Vec<GameResult>,
    pub records: Vec<SLRecordV2>,
    /// Root visits carried over by tree reuse (diagnostic).
    pub reused_visits: u64,
    leaf_buf: Vec<u32>,
}

impl NetFarm {
    pub fn new(seat_a: SeatConfig, seat_b: SeatConfig, n_games: u32, concurrency: usize, seed: u64) -> Self {
        NetFarm {
            seat_a,
            seat_b,
            max_moves: 600,
            n_games,
            concurrency,
            seed,
            next_id: 0,
            games: Vec::new(),
            search_rng: StdRng::seed_from_u64(seed ^ 0x9E37_79B9_7F4A_7C15),
            req: Vec::new(),
            req_x: Vec::new(),
            req_mask: Vec::new(),
            req_model: Vec::new(),
            req_states: Vec::new(),
            results: Vec::new(),
            records: Vec::new(),
            reused_visits: 0,
            leaf_buf: Vec::new(),
        }
    }

    pub fn games_finished(&self) -> usize {
        self.results.len()
    }

    fn new_game(&mut self) -> Game {
        let id = self.next_id;
        self.next_id += 1;
        Game {
            id,
            state: GameState::new_2p(),
            rng: StdRng::seed_from_u64(self.seed.wrapping_mul(1_000_003).wrapping_add(id as u64)),
            a_seat: if id % 2 == 0 { 0 } else { 2 },
            moves: 0,
            search: NetSearch::new(),
            esearch: NetExpectiSearch::new(),
            searching: false,
            awaiting_greedy: false,
            decisions: [0; 4],
            records: Vec::new(),
        }
    }

    fn push_request(&mut self, gi: usize, kind: ReqKind, state: GameState, model: u8) {
        let base_x = self.req_x.len();
        self.req_x.resize(base_x + V15_FRAME_SIZE, 0);
        encode_v15_2p_into(&state, state.current_player, &mut self.req_x[base_x..]);
        let base_m = self.req_mask.len();
        self.req_mask.resize(base_m + NUM_BOARD_CELLS, 0.0);
        let (legal, count) = state.legal_moves();
        let cp = state.current_player;
        for &t in &legal[..count] {
            self.req_mask[base_m + v15_own_cell(state.positions[cp as usize][t as usize], cp)] = 1.0;
        }
        self.req_model.push(model);
        self.req_states.push(state);
        self.req.push((gi, kind));
    }

    fn record(g: &mut Game, cfg: &SeatConfig, label: u8, pi: [f32; 4], flags: u8, legal_count: u8) {
        if !cfg.record {
            return;
        }
        let s = &g.state;
        let mut pos = [0i8; 16];
        for p in 0..4 {
            for t in 0..4 {
                pos[p * 4 + t] = s.positions[p][t];
            }
        }
        let bot = match cfg.kind {
            SeatKind::Mcts(_) | SeatKind::Expecti(_) => BOT_NET_MCTS,
            SeatKind::Greedy => BOT_NET_GREEDY,
        };
        let teacher = if s.current_player == g.a_seat { FLAG_TEACHER_SEAT } else { 0 };
        g.records.push(SLRecordV2 {
            positions: pos,
            current_player: s.current_player as i8,
            dice_roll: s.current_dice_roll as i8,
            action: label as i8,
            target_value: 0,
            mover_bot: bot,
            legal_count,
            consecutive_sixes: s.consecutive_sixes[s.current_player as usize],
            flags: flags | teacher,
            pi: quantise_pi(&pi),
            game_id: g.id,
        });
    }

    /// Advances every game until it needs the network; builds the request
    /// batch. Returns the number of requests (0 = all games finished).
    pub fn collect(&mut self) -> usize {
        self.req.clear();
        self.req_x.clear();
        self.req_mask.clear();
        self.req_model.clear();
        self.req_states.clear();

        while self.games.len() < self.concurrency && self.next_id < self.n_games {
            let g = self.new_game();
            self.games.push(g);
        }

        let mut gi = 0;
        while gi < self.games.len() {
            let mut removed = false;
            // Advance game gi until it blocks on the network or ends.
            loop {
                let g = &mut self.games[gi];
                if g.state.is_terminal || g.moves >= self.max_moves {
                    let winner = if g.state.is_terminal { g.state.winner } else { -1 };
                    if winner >= 0 {
                        for r in g.records.iter_mut() {
                            r.target_value = if r.current_player == winner { 1 } else { -1 };
                        }
                    }
                    self.records.append(&mut g.records);
                    self.results.push(GameResult { id: g.id, a_seat: g.a_seat, winner, moves: g.moves });
                    if self.next_id < self.n_games {
                        let ng = self.new_game();
                        self.games[gi] = ng;
                        continue;
                    } else {
                        // The last game moves into slot gi and must be
                        // processed in this same pass (skipping it could
                        // leave the batch empty while games remain).
                        self.games.swap_remove(gi);
                        removed = true;
                        break;
                    }
                }
                if g.awaiting_greedy {
                    break; // answered in submit()
                }
                if g.state.current_dice_roll == 0 {
                    let d = g.rng.gen_range(1..=6u8);
                    g.state.roll(d);
                    continue;
                }
                let (legal, count) = g.state.legal_moves();
                let cp = g.state.current_player;
                let cfg = if cp == g.a_seat { self.seat_a } else { self.seat_b };
                if count == 1 {
                    let mut pi = [0f32; 4];
                    pi[legal[0] as usize] = 1.0;
                    Self::record(g, &cfg, legal[0], pi, FLAG_FORCED, 1);
                    g.state = g.state.apply_move(legal[0]);
                    g.moves += 1;
                    continue;
                }
                match cfg.kind {
                    SeatKind::Greedy => {
                        g.awaiting_greedy = true;
                        let st = g.state;
                        self.push_request(gi, ReqKind::Greedy, st, cfg.model);
                        break;
                    }
                    SeatKind::Expecti(ec) => {
                        if !g.searching {
                            g.esearch.reset(g.state);
                            g.searching = true;
                        }
                        if g.esearch.finished(&ec) {
                            let label = g.esearch.best_move_cfg(&ec);
                            let pi = g.esearch.root_pi_temp(ec.pi_temp);
                            let mut flags = FLAG_HAS_PI;
                            let played = if g.decisions[cp as usize] < cfg.explore_moves {
                                let m = g.esearch.sample_move(&mut g.rng, ec.pi_temp.max(0.1));
                                if m != label {
                                    flags |= FLAG_EXPLORATORY;
                                }
                                m
                            } else {
                                label
                            };
                            g.decisions[cp as usize] += 1;
                            Self::record(g, &cfg, label, pi, flags, count as u8);
                            g.searching = false;
                            g.state = g.state.apply_move(played);
                            g.moves += 1;
                            continue;
                        }
                        self.leaf_buf.clear();
                        let mut leaves = std::mem::take(&mut self.leaf_buf);
                        self.games[gi].esearch.run_round(&ec, &mut leaves);
                        for &leaf in &leaves {
                            let st = *self.games[gi].esearch.node_state(leaf);
                            self.push_request(gi, ReqKind::Leaf(leaf), st, cfg.model);
                        }
                        let n_new = leaves.len();
                        self.leaf_buf = leaves;
                        let es = &self.games[gi].esearch;
                        if n_new == 0 && (es.finished(&ec) || es.in_flight() == 0) {
                            continue;
                        }
                        break;
                    }
                    SeatKind::Mcts(mc) => {
                        if !g.searching {
                            if mc.reuse_tree {
                                let reused = g.search.reset_reuse(g.state, &mc, &mut g.rng);
                                self.reused_visits += reused as u64;
                            } else {
                                g.search.reset(g.state);
                            }
                            g.searching = true;
                        }
                        if g.search.finished(&mc) {
                            let label = g.search.best_move();
                            let pi = g.search.root_pi();
                            let mut flags = FLAG_HAS_PI;
                            let played = if g.decisions[cp as usize] < cfg.explore_moves {
                                let m = g.search.sample_move(&mut g.rng);
                                if m != label {
                                    flags |= FLAG_EXPLORATORY;
                                }
                                m
                            } else {
                                label
                            };
                            g.decisions[cp as usize] += 1;
                            Self::record(g, &cfg, label, pi, flags, count as u8);
                            g.searching = false;
                            g.state = g.state.apply_move(played);
                            g.moves += 1;
                            continue;
                        }
                        self.leaf_buf.clear();
                        let mut leaves = std::mem::take(&mut self.leaf_buf);
                        let g = &mut self.games[gi];
                        g.search.run_round(&mc, &mut self.search_rng, &mut leaves);
                        let n_new = leaves.len();
                        for &leaf in &leaves {
                            let st = *self.games[gi].search.node_state(leaf);
                            self.push_request(gi, ReqKind::Leaf(leaf), st, cfg.model);
                        }
                        self.leaf_buf = leaves;
                        if n_new == 0
                            && (self.games[gi].search.finished(&mc) || self.games[gi].search.in_flight == 0)
                        {
                            // Only terminals this round (or the budget is now
                            // complete): move on in this same pass, otherwise an
                            // empty batch would read as "all games finished".
                            continue;
                        }
                        break;
                    }
                }
            }
            if !removed {
                gi += 1;
            }
        }
        self.req.len()
    }

    /// Applies network answers for the current batch: `policy` is
    /// (n, 225) probabilities, `p_win` is (n,) P(mover wins).
    pub fn submit(&mut self, policy: &[f32], p_win: &[f32]) {
        assert_eq!(policy.len(), self.req.len() * NUM_BOARD_CELLS);
        assert_eq!(p_win.len(), self.req.len());
        let req = std::mem::take(&mut self.req);
        for (i, &(gi, kind)) in req.iter().enumerate() {
            let pol = &policy[i * NUM_BOARD_CELLS..(i + 1) * NUM_BOARD_CELLS];
            let g = &mut self.games[gi];
            match kind {
                ReqKind::Greedy => {
                    let (legal, count) = g.state.legal_moves();
                    let pri = cell_policy_to_priors(&g.state, &legal[..count], pol);
                    let mut best = 0;
                    for k in 1..count {
                        if pri[k] > pri[best] {
                            best = k;
                        }
                    }
                    let cp = g.state.current_player;
                    let cfg = if cp == g.a_seat { self.seat_a } else { self.seat_b };
                    let mut pi = [0f32; 4];
                    for k in 0..count {
                        pi[legal[k] as usize] = pri[k];
                    }
                    Self::record(g, &cfg, legal[best], pi, FLAG_HAS_PI, count as u8);
                    g.state = g.state.apply_move(legal[best]);
                    g.moves += 1;
                    g.awaiting_greedy = false;
                }
                ReqKind::Leaf(node) => {
                    let cp = g.state.current_player; // the searching seat is the mover
                    let cfg = if cp == g.a_seat { self.seat_a } else { self.seat_b };
                    let mut noise_rng = g.rng.clone();
                    let eps = match cfg.kind {
                        SeatKind::Mcts(mc) => {
                            g.search.answer(node, pol, p_win[i], Some(&mut noise_rng), &mc);
                            mc.dirichlet_eps
                        }
                        SeatKind::Expecti(ec) => {
                            g.esearch.answer(node, pol, p_win[i], Some(&mut noise_rng), &ec);
                            ec.dirichlet_eps
                        }
                        SeatKind::Greedy => unreachable!(),
                    };
                    if node == 0 && eps > 0.0 {
                        g.rng = noise_rng;
                    }
                }
            }
        }
        self.req = req;
        self.req.clear();
    }
}
