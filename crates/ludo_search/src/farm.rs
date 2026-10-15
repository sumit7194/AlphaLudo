//! General batched game farm for V17 RL and evaluation (2 or 4 seats).
//!
//! Each game is drawn from a weighted MATCHUP table; a matchup lists one player
//! spec per seat: a network (by model id; greedy or sampling at a temperature;
//! optionally recording its decisions for training) or a native bot. Seats are
//! shuffled per game. Rust owns games, dice, bots and records; Python answers
//! batched network requests per model (policy over 225 source cells + the
//! mover's win probability). Bot decisions are computed in parallel (rayon).
//!
//! Forced moves (one legal option) never reach the network and are not recorded.
//!
//! Contract: after every `collect()`, Python must `submit` answers for EVERY model
//! listed by `models_pending()` before calling `collect()` again.
//!
//! Recorded decisions carry what PPO needs: frame, legal cells, action cell, the
//! log-probability of the action under the distribution actually sampled from,
//! and the network's value estimate. At game end each recorded seat gets reward
//! 1.0 (won) or 0.0 (lost); games stopped by the move cap are marked unfinished.

use ludo_core::{
    encode_unified_into, encode_v15_2p_into, uni_action_cell, v15_own_cell, GameState, NUM_BOARD_CELLS,
    UNI_FRAME_SIZE, V15_FRAME_SIZE,
};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

use crate::expectimax::{Depth2ExpectimaxBot, ExpectimaxBot, LudoBot};
use crate::mcts::MaxNMCTSBot;
use crate::mcts_2p::TwoPlayerMCTSBot;
use crate::scripted::{ScriptedBot, ScriptedKind};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Encoder {
    /// ludo_core::unified (V17), 2P and 4P.
    Unified,
    /// The v15.2 champion's 15x15x3 encoding (2P only).
    V15,
}

impl Encoder {
    pub fn frame_size(self) -> usize {
        match self {
            Encoder::Unified => UNI_FRAME_SIZE,
            Encoder::V15 => V15_FRAME_SIZE,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BotKind {
    Random,
    Heuristic,
    Expert,
    Expectimax,
    Depth2,
    /// Expecti-MCTS (2P) / Max^n MCTS (4P) with this many simulations.
    Mcts(u32),
    /// Max^n MCTS with an explicit configuration (4P; also legal in 2P).
    MaxN(crate::mcts::MaxNConfig),
    /// Full-round exact expectimax (round4.rs).
    Round(crate::round4::RoundConfig),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PlayerSpec {
    Net { model: u8, temperature: f32, record: bool },
    Bot(BotKind),
}

#[derive(Clone, Debug)]
pub struct Matchup {
    pub players: Vec<PlayerSpec>, // 2 or 4 entries
    pub weight: f64,
}

/// One recorded decision.
#[derive(Clone)]
pub struct StepRecord {
    pub game_id: u64,
    pub matchup: u16,
    pub seat: u8,
    pub step: u16,
    pub x: Vec<i8>,
    pub cells: [u8; 4],
    pub n_cells: u8,
    pub action: u8,
    pub logp: f32,
    pub value: f32,
    pub legal_tokens: u8,
    pub n_players: u8,
    /// 1.0 won, 0.0 lost, NaN unfinished (filled at game end).
    pub reward: f32,
}

#[derive(Clone, Copy, Debug)]
pub struct GameResult {
    pub game_id: u64,
    pub matchup: u16,
    /// Index into the matchup's player list for each seat (255 = empty seat).
    pub role_of_seat: [u8; 4],
    /// Winning seat, or -1 if the move cap was hit.
    pub winner: i8,
    pub moves: u32,
}

enum Seat {
    Empty,
    Net { model: u8, temperature: f32, record: bool },
    Bot(Box<dyn LudoBot>),
}

struct Game {
    id: u64,
    matchup: u16,
    state: GameState,
    seats: [Seat; 4],
    role_of_seat: [u8; 4],
    rng: StdRng,
    moves: u32,
    steps: u16,
    awaiting: bool,
    pending_bot: bool,
    /// Finished with no replacement; removed at the start of the next collect()
    /// (removing it mid-pass would shift indices that pending requests use).
    done: bool,
    records: Vec<StepRecord>,
}

struct Request {
    game: usize,
}

pub struct GameFarm {
    encoders: Vec<Encoder>,
    matchups: Vec<Matchup>,
    concurrency: usize,
    pub max_moves: u32,
    game_limit: u64,
    next_id: u64,
    rng: StdRng,
    games: Vec<Game>,
    /// Pending requests per model id.
    req: Vec<Vec<Request>>,
    req_x: Vec<Vec<i8>>,
    req_mask: Vec<Vec<f32>>,
    pub records: Vec<StepRecord>,
    pub results: Vec<GameResult>,
    pub finished: u64,
}

pub fn make_bot(kind: BotKind, seat: u8, n_players: usize, seed: u64) -> Box<dyn LudoBot> {
    let s = Some(seat);
    match kind {
        BotKind::Random => Box::new(RandomBot::new(seed)),
        BotKind::Heuristic => Box::new(ScriptedBot::with_seed(ScriptedKind::Heuristic, s, seed)),
        BotKind::Expert => Box::new(ScriptedBot::with_seed(ScriptedKind::Expert, s, seed)),
        BotKind::Expectimax => Box::new(ExpectimaxBot::new(s)),
        BotKind::Depth2 => Box::new(Depth2ExpectimaxBot::new(s)),
        BotKind::Mcts(sims) => {
            if n_players == 2 {
                Box::new(TwoPlayerMCTSBot::new(s, sims as usize))
            } else {
                Box::new(MaxNMCTSBot::new(s, sims as usize))
            }
        }
        BotKind::MaxN(cfg) => Box::new(MaxNMCTSBot::with_config(s, cfg)),
        BotKind::Round(cfg) => Box::new(crate::round4::RoundBot::new(s, cfg)),
    }
}

/// Uniformly random legal move.
pub struct RandomBot {
    rng: std::sync::Mutex<StdRng>,
}

impl RandomBot {
    pub fn new(seed: u64) -> Self {
        RandomBot { rng: std::sync::Mutex::new(StdRng::seed_from_u64(seed)) }
    }
}

impl LudoBot for RandomBot {
    fn name(&self) -> &'static str {
        "Random"
    }
    fn select_move(&self, state: &GameState) -> u8 {
        let (legal, count) = state.legal_moves();
        if count == 0 {
            return 0;
        }
        legal[self.rng.lock().unwrap().gen_range(0..count)]
    }
}

impl GameFarm {
    pub fn new(encoders: Vec<Encoder>, concurrency: usize, seed: u64) -> Self {
        let n = encoders.len();
        GameFarm {
            encoders,
            matchups: Vec::new(),
            concurrency,
            max_moves: 1200,
            game_limit: u64::MAX,
            next_id: 0,
            rng: StdRng::seed_from_u64(seed),
            games: Vec::new(),
            req: (0..n).map(|_| Vec::new()).collect(),
            req_x: (0..n).map(|_| Vec::new()).collect(),
            req_mask: (0..n).map(|_| Vec::new()).collect(),
            records: Vec::new(),
            results: Vec::new(),
            finished: 0,
        }
    }

    pub fn set_matchups(&mut self, m: Vec<Matchup>) -> Result<(), String> {
        for (i, mu) in m.iter().enumerate() {
            if mu.players.len() != 2 && mu.players.len() != 4 {
                return Err(format!("matchup {i}: need 2 or 4 players"));
            }
            for p in &mu.players {
                if let PlayerSpec::Net { model, .. } = p {
                    let e = *self.encoders.get(*model as usize).ok_or(format!("matchup {i}: unknown model {model}"))?;
                    if e == Encoder::V15 && mu.players.len() == 4 {
                        return Err(format!("matchup {i}: the V15 encoding is 2-player only"));
                    }
                }
            }
        }
        if m.iter().map(|x| x.weight).sum::<f64>() <= 0.0 {
            return Err("matchup weights must sum to > 0".into());
        }
        self.matchups = m;
        Ok(())
    }

    /// Stop starting new games after `n` more have been started (eval mode).
    pub fn set_game_limit(&mut self, n: u64) {
        self.game_limit = self.next_id + n;
    }

    pub fn in_flight(&self) -> usize {
        self.games.iter().filter(|g| !g.done).count()
    }

    fn new_game(&mut self) -> Game {
        let total: f64 = self.matchups.iter().map(|m| m.weight).sum();
        let mut x = self.rng.gen_range(0.0..total);
        let mut mi = 0;
        for (i, m) in self.matchups.iter().enumerate() {
            if x < m.weight {
                mi = i;
                break;
            }
            x -= m.weight;
            mi = i;
        }
        let mu = self.matchups[mi].clone();
        let n = mu.players.len();
        let mut state = if n == 2 { GameState::new_2p() } else { GameState::new_4p() };
        let mut seat_ids: Vec<u8> = if n == 2 { vec![0, 2] } else { vec![0, 1, 2, 3] };
        seat_ids.shuffle(&mut self.rng);
        let id = self.next_id;
        self.next_id += 1;
        let mut seats: [Seat; 4] = [Seat::Empty, Seat::Empty, Seat::Empty, Seat::Empty];
        let mut role_of_seat = [255u8; 4];
        for (role, &seat) in seat_ids.iter().enumerate() {
            role_of_seat[seat as usize] = role as u8;
            seats[seat as usize] = match mu.players[role] {
                PlayerSpec::Net { model, temperature, record } => Seat::Net { model, temperature, record },
                PlayerSpec::Bot(k) => Seat::Bot(make_bot(k, seat, n, self.rng.gen())),
            };
        }
        state.current_player = if n == 2 { 0 } else { 0 };
        Game {
            id,
            matchup: mi as u16,
            state,
            seats,
            role_of_seat,
            rng: StdRng::seed_from_u64(self.rng.gen()),
            moves: 0,
            steps: 0,
            awaiting: false,
            pending_bot: false,
            done: false,
            records: Vec::new(),
        }
    }

    fn finish(&mut self, gi: usize) {
        let g = &mut self.games[gi];
        let winner = if g.state.is_terminal { g.state.winner } else { -1 };
        for r in g.records.iter_mut() {
            r.reward = if winner < 0 { f32::NAN } else if r.seat as i8 == winner { 1.0 } else { 0.0 };
        }
        self.records.append(&mut g.records);
        self.results.push(GameResult { game_id: g.id, matchup: g.matchup, role_of_seat: g.role_of_seat, winner, moves: g.moves });
        self.finished += 1;
    }

    /// Advance game gi until it needs a network answer, a bot move, or ends.
    /// Returns false if the game finished.
    fn advance(&mut self, gi: usize) -> bool {
        loop {
            let g = &mut self.games[gi];
            if g.state.is_terminal || g.moves >= self.max_moves {
                return false;
            }
            if g.awaiting || g.pending_bot {
                return true;
            }
            if g.state.current_dice_roll == 0 {
                let d = g.rng.gen_range(1..=6u8);
                g.state.roll(d);
                continue;
            }
            let (legal, count) = g.state.legal_moves();
            let cp = g.state.current_player as usize;
            match &g.seats[cp] {
                Seat::Empty => unreachable!("empty seat to move"),
                Seat::Bot(_) => {
                    if count == 1 {
                        g.state = g.state.apply_move(legal[0]);
                        g.moves += 1;
                        continue;
                    }
                    g.pending_bot = true;
                    return true;
                }
                Seat::Net { model, .. } => {
                    let model = *model;
                    // Forced moves are not decisions: no network call, not recorded
                    // (GAE runs over decision points only; reward is terminal).
                    if count == 1 {
                        g.state = g.state.apply_move(legal[0]);
                        g.moves += 1;
                        continue;
                    }
                    g.awaiting = true;
                    let enc = self.encoders[model as usize];
                    let st = self.games[gi].state;
                    self.push_request(gi, model, enc, &st);
                    return true;
                }
            }
        }
    }

    fn push_request(&mut self, gi: usize, model: u8, enc: Encoder, st: &GameState) {
        let m = model as usize;
        let fx = enc.frame_size();
        let base = self.req_x[m].len();
        self.req_x[m].resize(base + fx, 0);
        let cp = st.current_player;
        match enc {
            Encoder::Unified => encode_unified_into(st, cp, &mut self.req_x[m][base..]),
            Encoder::V15 => encode_v15_2p_into(st, cp, &mut self.req_x[m][base..]),
        }
        let bm = self.req_mask[m].len();
        self.req_mask[m].resize(bm + NUM_BOARD_CELLS, 0.0);
        let (legal, count) = st.legal_moves();
        for &t in &legal[..count] {
            let c = cell_of(enc, st, t);
            self.req_mask[m][bm + c] = 1.0;
        }
        self.req[m].push(Request { game: gi });
    }

    /// Advances all games to their next network call (bot moves resolved in
    /// parallel). Returns the total number of pending network requests
    /// (0 = no games left).
    pub fn collect(&mut self) -> usize {
        for m in 0..self.req.len() {
            self.req[m].clear();
            self.req_x[m].clear();
            self.req_mask[m].clear();
        }
        // Safe point: every previous request was answered, so indices can change.
        self.games.retain(|g| !g.done);
        loop {
            while self.games.len() < self.concurrency && self.next_id < self.game_limit && !self.matchups.is_empty() {
                let g = self.new_game();
                self.games.push(g);
            }
            for gi in 0..self.games.len() {
                loop {
                    let g = &self.games[gi];
                    if g.done || g.awaiting || g.pending_bot {
                        break;
                    }
                    if self.advance(gi) {
                        break;
                    }
                    self.finish(gi);
                    if self.next_id < self.game_limit && !self.matchups.is_empty() {
                        let ng = self.new_game();
                        self.games[gi] = ng; // replace in place, keep going with it
                    } else {
                        self.games[gi].done = true;
                    }
                }
            }
            // Resolve bot moves in parallel, then loop until only network waits remain.
            if !self.games.iter().any(|g| g.pending_bot) {
                break;
            }
            self.games.par_iter_mut().filter(|g| g.pending_bot).for_each(|g| {
                let cp = g.state.current_player as usize;
                if let Seat::Bot(b) = &g.seats[cp] {
                    let a = b.select_move(&g.state);
                    g.state = g.state.apply_move(a);
                    g.moves += 1;
                }
                g.pending_bot = false;
            });
        }
        self.req.iter().map(|r| r.len()).sum()
    }

    pub fn models_pending(&self) -> Vec<u8> {
        (0..self.req.len()).filter(|&m| !self.req[m].is_empty()).map(|m| m as u8).collect()
    }

    pub fn request_count(&self, model: u8) -> usize {
        self.req[model as usize].len()
    }

    pub fn request_x(&self, model: u8) -> &[i8] {
        &self.req_x[model as usize]
    }

    pub fn request_mask(&self, model: u8) -> &[f32] {
        &self.req_mask[model as usize]
    }

    /// Answers model `model`'s requests: `policy` (n, 225) probabilities over legal
    /// cells, `value` (n,) the mover's win probability.
    pub fn submit(&mut self, model: u8, policy: &[f32], value: &[f32]) -> Result<(), String> {
        let m = model as usize;
        let n = self.req[m].len();
        if policy.len() != n * NUM_BOARD_CELLS || value.len() != n {
            return Err(format!("model {model}: expected {n} answers"));
        }
        let enc = self.encoders[m];
        let fx = enc.frame_size();
        let reqs = std::mem::take(&mut self.req[m]);
        for (i, r) in reqs.iter().enumerate() {
            let pol = &policy[i * NUM_BOARD_CELLS..(i + 1) * NUM_BOARD_CELLS];
            let mask = &self.req_mask[m][i * NUM_BOARD_CELLS..(i + 1) * NUM_BOARD_CELLS];
            let g = &mut self.games[r.game];
            let cp = g.state.current_player;
            let (temperature, record) = match g.seats[cp as usize] {
                Seat::Net { temperature, record, .. } => (temperature, record),
                _ => unreachable!(),
            };
            let (legal, count) = g.state.legal_moves();
            // Distinct legal cells and their probabilities.
            let mut cells = [0u8; 4];
            let mut probs = [0f32; 4];
            let mut k = 0usize;
            for &t in &legal[..count] {
                let c = cell_of(enc, &g.state, t);
                if !cells[..k].contains(&(c as u8)) {
                    cells[k] = c as u8;
                    probs[k] = pol[c].max(0.0) * (mask[c] > 0.5) as i32 as f32;
                    k += 1;
                }
            }
            let (pick, logp) = choose(&probs[..k], temperature, &mut g.rng);
            let cell = cells[pick];
            let token = legal[..count].iter().copied().find(|&t| cell_of(enc, &g.state, t) as u8 == cell).unwrap();
            if record {
                g.records.push(StepRecord {
                    game_id: g.id,
                    matchup: g.matchup,
                    seat: cp,
                    step: g.steps,
                    x: self.req_x[m][i * fx..(i + 1) * fx].to_vec(),
                    cells,
                    n_cells: k as u8,
                    action: cell,
                    logp,
                    value: value[i],
                    legal_tokens: count as u8,
                    n_players: g.state.active_players.iter().filter(|&&a| a).count() as u8,
                    reward: f32::NAN,
                });
                g.steps += 1;
            }
            g.state = g.state.apply_move(token);
            g.moves += 1;
            g.awaiting = false;
        }
        self.req[m] = reqs;
        self.req[m].clear();
        Ok(())
    }
}

#[inline]
fn cell_of(enc: Encoder, st: &GameState, token: u8) -> usize {
    match enc {
        Encoder::Unified => uni_action_cell(st, token),
        Encoder::V15 => {
            let cp = st.current_player;
            v15_own_cell(st.positions[cp as usize][token as usize], cp)
        }
    }
}

/// Picks an index: greedy if temperature <= 0, else samples from p^(1/T).
/// Returns (index, log-probability of that index under the distribution used).
fn choose(p: &[f32], temperature: f32, rng: &mut StdRng) -> (usize, f32) {
    let k = p.len();
    if k == 1 {
        return (0, 0.0);
    }
    if temperature <= 0.0 {
        let mut b = 0;
        for i in 1..k {
            if p[i] > p[b] {
                b = i;
            }
        }
        return (b, 0.0);
    }
    let mut w = [0f64; 4];
    let mut sum = 0.0;
    for i in 0..k {
        w[i] = (p[i].max(1e-30) as f64).powf(1.0 / temperature as f64);
        sum += w[i];
    }
    let mut x = rng.gen_range(0.0..sum);
    let mut pick = k - 1;
    for i in 0..k {
        if x < w[i] {
            pick = i;
            break;
        }
        x -= w[i];
    }
    (pick, (w[pick] / sum).ln() as f32)
}
