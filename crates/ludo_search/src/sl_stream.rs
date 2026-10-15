//! Streaming supervised-learning data: MCTS-3000 vs MCTS-3000 self-play, no files.
//!
//! N worker threads play complete 2-player games where BOTH seats are the fixed
//! Rust Expecti-MCTS teacher. Every decision is labelled with the teacher's visit
//! distribution over the mover's legal SOURCE CELLS (unified encoding, stacked
//! tokens merged) and, once the game ends, with the winner's relative seat. Each
//! decision is kept with probability `keep_prob` (positions inside one game are
//! highly correlated). Finished games' records go into a bounded queue; workers
//! block when it is full (back-pressure), so generation never outruns the learner.
//!
//! 2026-10-07 (V17 2P+4P extension): workers can also play 4-PLAYER games with a
//! configurable 4P teacher in every seat (`teacher4`: the exact full-round search
//! or Max^n MCTS). `frac4` is the target share of 4P POSITIONS (not games: a 4P
//! game yields ~2.5x more decisions); each worker picks its next game's mode by
//! comparing the running position counts to the target, and the target can be
//! changed while the stream runs (`set_frac4`). frac4 = 0 is exactly the
//! original 2P stream.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;

use ludo_core::{encode_unified_into, uni_action_cell, GameState, UNI_FRAME_SIZE};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::farm::BotKind;
use crate::mcts::MaxNMCTSBot;
use crate::mcts_2p::TwoPlayerMCTSBot;
use crate::round4::RoundBot;

/// Winner-relative seat for unfinished games.
pub const NO_WINNER: u8 = 255;

#[derive(Clone)]
pub struct SlRecord {
    pub x: Box<[i8; UNI_FRAME_SIZE]>,
    /// Up to 4 distinct legal source cells and the teacher's probability on each.
    pub cells: [u8; 4],
    pub pi: [f32; 4],
    pub n_cells: u8,
    /// Cell of the move actually played (teacher argmax).
    pub action: u8,
    /// (winner seat − mover seat) mod 4, or NO_WINNER.
    pub winner_rel: u8,
    /// Number of legal TOKENS (1 = forced move).
    pub legal_tokens: u8,
    /// 2 or 4.
    pub n_players: u8,
}

struct Shared {
    queue: Mutex<VecDeque<SlRecord>>,
    not_full: Condvar,
    not_empty: Condvar,
    capacity: usize,
    stop: AtomicBool,
    games: AtomicU64,
    positions: AtomicU64,
    games4: AtomicU64,
    positions4: AtomicU64,
    /// Target share of 4P positions, f64 bits.
    frac4: AtomicU64,
}

pub struct SlStream {
    shared: Arc<Shared>,
    workers: Vec<JoinHandle<()>>,
}

#[derive(Clone, Copy, Debug)]
pub struct SlStreamConfig {
    pub threads: usize,
    pub sims: usize,
    pub keep_prob: f64,
    pub capacity: usize,
    pub max_moves: usize,
    pub seed: u64,
    /// Initial target share of 4P positions (0 = 2P only).
    pub frac4: f64,
    /// 4P teacher (every seat): BotKind::Round or BotKind::MaxN.
    pub teacher4: Option<BotKind>,
    pub max_moves4: usize,
}

/// 4P teacher: a move plus a distribution over tokens.
enum Teacher4 {
    Round(RoundBot),
    MaxN(MaxNMCTSBot),
}

impl Teacher4 {
    fn new(kind: BotKind, seat: u8) -> Self {
        match kind {
            BotKind::Round(c) => Teacher4::Round(RoundBot::new(Some(seat), c)),
            BotKind::MaxN(c) => Teacher4::MaxN(MaxNMCTSBot::with_config(Some(seat), c)),
            BotKind::Mcts(sims) => Teacher4::MaxN(MaxNMCTSBot::new(Some(seat), sims as usize)),
            other => panic!("unsupported 4P teacher {other:?}"),
        }
    }

    /// (move, target distribution over tokens). Round: one-hot on the best move
    /// (its values are utilities, not probabilities); MaxN: visit distribution.
    fn label(&self, s: &GameState, rng: &mut StdRng) -> (u8, [f32; 4]) {
        match self {
            Teacher4::Round(b) => {
                let (vals, legal, count) = b.move_values(s);
                let mut bi = 0;
                for i in 1..count {
                    if vals[i] > vals[bi] {
                        bi = i;
                    }
                }
                let mut p = [0f32; 4];
                p[legal[bi] as usize] = 1.0;
                (legal[bi], p)
            }
            Teacher4::MaxN(b) => b.search(s, 0.0, rng),
        }
    }
}

/// Encodes and labels one decision (shared by the 2P and 4P games).
fn make_record(s: &GameState, legal: &[u8; 4], count: usize, pi_tok: &[f32; 4], action: u8, n_players: u8) -> SlRecord {
    let cp = s.current_player;
    let mut x = Box::new([0i8; UNI_FRAME_SIZE]);
    encode_unified_into(s, cp, &mut x[..]);
    let mut cells = [0u8; 4];
    let mut pi = [0f32; 4];
    let mut n = 0usize;
    for &t in &legal[..count] {
        let c = uni_action_cell(s, t) as u8;
        match cells[..n].iter().position(|&e| e == c) {
            Some(j) => pi[j] += pi_tok[t as usize],
            None => {
                cells[n] = c;
                pi[n] = pi_tok[t as usize];
                n += 1;
            }
        }
    }
    SlRecord {
        x,
        cells,
        pi,
        n_cells: n as u8,
        action: uni_action_cell(s, action) as u8,
        winner_rel: NO_WINNER,
        legal_tokens: count as u8,
        n_players,
    }
}

fn play_game4(cfg: &SlStreamConfig, rng: &mut StdRng, out: &mut Vec<(SlRecord, u8)>) {
    let kind = cfg.teacher4.expect("teacher4 not configured");
    let mut s = GameState::new_4p();
    let bots: Vec<Teacher4> = (0..4u8).map(|p| Teacher4::new(kind, p)).collect();
    let start = out.len();
    let mut moves = 0;
    while !s.is_terminal && moves < cfg.max_moves4 {
        if !s.roll(rng.gen_range(1..=6)) {
            continue;
        }
        let (legal, count) = s.legal_moves();
        let cp = s.current_player;
        let (action, pi_tok) = if count == 1 {
            let mut p = [0f32; 4];
            p[legal[0] as usize] = 1.0;
            (legal[0], p)
        } else {
            bots[cp as usize].label(&s, rng)
        };
        if rng.gen_bool(cfg.keep_prob) {
            out.push((make_record(&s, &legal, count, &pi_tok, action, 4), cp));
        }
        s = s.apply_move(action);
        moves += 1;
    }
    if s.is_terminal {
        let w = s.winner as u8;
        for (rec, mover) in out[start..].iter_mut() {
            rec.winner_rel = (w + 4 - *mover) % 4;
        }
    }
}

fn play_game(cfg: &SlStreamConfig, rng: &mut StdRng, out: &mut Vec<(SlRecord, u8)>) {
    let mut s = GameState::new_2p();
    let bots = [TwoPlayerMCTSBot::new(Some(0), cfg.sims), TwoPlayerMCTSBot::new(Some(2), cfg.sims)];
    let start = out.len();
    let mut moves = 0;
    while !s.is_terminal && moves < cfg.max_moves {
        if !s.roll(rng.gen_range(1..=6)) {
            continue;
        }
        let (legal, count) = s.legal_moves();
        let cp = s.current_player;
        let (action, pi_tok) = if count == 1 {
            let mut p = [0f32; 4];
            p[legal[0] as usize] = 1.0;
            (legal[0], p)
        } else {
            bots[(cp / 2) as usize].search_with_temp(&s, 0.0)
        };
        if rng.gen_bool(cfg.keep_prob) {
            out.push((make_record(&s, &legal, count, &pi_tok, action, 2), cp));
        }
        s = s.apply_move(action);
        moves += 1;
    }
    if s.is_terminal {
        let w = s.winner as u8;
        for (rec, mover) in out[start..].iter_mut() {
            rec.winner_rel = (w + 4 - *mover) % 4;
        }
    }
}

impl SlStream {
    pub fn start(cfg: SlStreamConfig) -> Self {
        let shared = Arc::new(Shared {
            queue: Mutex::new(VecDeque::with_capacity(cfg.capacity)),
            not_full: Condvar::new(),
            not_empty: Condvar::new(),
            capacity: cfg.capacity,
            stop: AtomicBool::new(false),
            games: AtomicU64::new(0),
            positions: AtomicU64::new(0),
            games4: AtomicU64::new(0),
            positions4: AtomicU64::new(0),
            frac4: AtomicU64::new(cfg.frac4.clamp(0.0, 1.0).to_bits()),
        });
        let mut workers = Vec::new();
        for i in 0..cfg.threads.max(1) {
            let sh = Arc::clone(&shared);
            let c = cfg;
            workers.push(std::thread::spawn(move || {
                let mut rng = StdRng::seed_from_u64(c.seed ^ (0x9E37_79B9_7F4A_7C15u64.wrapping_mul(i as u64 + 1)));
                let mut buf: Vec<(SlRecord, u8)> = Vec::with_capacity(64);
                while !sh.stop.load(Ordering::Relaxed) {
                    buf.clear();
                    let f4 = f64::from_bits(sh.frac4.load(Ordering::Relaxed));
                    let four = if f4 <= 0.0 || c.teacher4.is_none() {
                        false
                    } else if f4 >= 1.0 {
                        true
                    } else {
                        // Steer the running position mix towards the target share.
                        let p4 = sh.positions4.load(Ordering::Relaxed) as f64;
                        let p = sh.positions.load(Ordering::Relaxed) as f64;
                        if p < 1.0 { rng.gen_bool(f4) } else { p4 / p < f4 }
                    };
                    if four {
                        play_game4(&c, &mut rng, &mut buf);
                        sh.games4.fetch_add(1, Ordering::Relaxed);
                    } else {
                        play_game(&c, &mut rng, &mut buf);
                    }
                    sh.games.fetch_add(1, Ordering::Relaxed);
                    let mut q = sh.queue.lock().unwrap();
                    for (rec, _) in buf.drain(..) {
                        while q.len() >= sh.capacity && !sh.stop.load(Ordering::Relaxed) {
                            q = sh.not_full.wait(q).unwrap();
                        }
                        if sh.stop.load(Ordering::Relaxed) {
                            return;
                        }
                        if rec.n_players == 4 {
                            sh.positions4.fetch_add(1, Ordering::Relaxed);
                        }
                        q.push_back(rec);
                        sh.positions.fetch_add(1, Ordering::Relaxed);
                    }
                    drop(q);
                    sh.not_empty.notify_all();
                }
            }));
        }
        SlStream { shared, workers }
    }

    /// Takes up to `n` records, waiting at most `timeout_ms` for the first one.
    pub fn take(&self, n: usize, timeout_ms: u64) -> Vec<SlRecord> {
        let mut q = self.shared.queue.lock().unwrap();
        if q.is_empty() {
            let (g, _) = self
                .shared
                .not_empty
                .wait_timeout_while(q, std::time::Duration::from_millis(timeout_ms), |q| q.is_empty())
                .unwrap();
            q = g;
        }
        let k = n.min(q.len());
        let out: Vec<SlRecord> = q.drain(..k).collect();
        drop(q);
        self.shared.not_full.notify_all();
        out
    }

    pub fn queued(&self) -> usize {
        self.shared.queue.lock().unwrap().len()
    }

    pub fn games(&self) -> u64 {
        self.shared.games.load(Ordering::Relaxed)
    }

    pub fn positions(&self) -> u64 {
        self.shared.positions.load(Ordering::Relaxed)
    }

    pub fn games4(&self) -> u64 {
        self.shared.games4.load(Ordering::Relaxed)
    }

    pub fn positions4(&self) -> u64 {
        self.shared.positions4.load(Ordering::Relaxed)
    }

    /// Changes the target share of 4P positions (takes effect from the next game).
    pub fn set_frac4(&self, f: f64) {
        self.shared.frac4.store(f.clamp(0.0, 1.0).to_bits(), Ordering::Relaxed);
    }

    pub fn stop(&mut self) {
        self.shared.stop.store(true, Ordering::Relaxed);
        self.shared.not_full.notify_all();
        self.shared.not_empty.notify_all();
        for w in self.workers.drain(..) {
            let _ = w.join();
        }
    }
}

impl Drop for SlStream {
    fn drop(&mut self) {
        self.stop();
    }
}
