//! High-Throughput Rust Multi-Threaded Generator for search-distillation SL data.
//!
//! Plays 2-player games (P0 vs P2, seats alternated by game number) between a
//! TEACHER bot and an OPPONENT bot and writes compact binary shards.
//!
//! FORMAT v2 (2026-10-02 audit). v1 (20-byte records, no header) had no record
//! of which bot made each move, so the V15.3 dataset is an unlabeled ~49/51
//! mix of MCTS and Depth2 moves; it discarded the search's visit distribution;
//! and forced (single-legal-move) records were indistinguishable from real
//! decisions. v2 shards start with a 16-byte header:
//!
//!   magic b"ALSL" | u16 version = 2 | u16 record_size = 32 | u32 n_records | u32 0
//!
//! followed by n_records 32-byte `SLRecordV2` records (little-endian). v1 files
//! are still readable by the Python loader (it detects the magic).
//!
//! Other fixes vs v1: dice resolution goes through GameState::roll (v1
//! double-passed after a three-six forfeit, handing the turn straight back);
//! root Dirichlet noise is OFF by default (it was on, and with the old blind
//! leaves it often decided the label); a manifest.json pins the config so a
//! resumed run cannot silently mix datasets.

use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use ludo_core::GameState;
use ludo_search::records::{
    quantise_pi, records_as_bytes, shard_header, SLRecordV2, BOT_DEPTH2, BOT_EXPECTIMAX, BOT_MCTS, FLAG_EXPLORATORY,
    FLAG_FORCED, FLAG_HAS_PI, FLAG_TEACHER_SEAT, FORMAT_VERSION,
};
use ludo_search::{Depth2ExpectimaxBot, ExpectimaxBot, LudoBot, TwoPlayerMCTSBot};
use rand::Rng;
use rayon::prelude::*;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum BotKind {
    Mcts,
    Depth2,
    Expectimax,
}

impl BotKind {
    fn parse(s: &str) -> BotKind {
        match s {
            "mcts" | "MCTS" => BotKind::Mcts,
            "depth2" | "Depth2" => BotKind::Depth2,
            "expectimax" | "Expectimax" => BotKind::Expectimax,
            other => panic!("unknown bot kind {other:?} (mcts | depth2 | expectimax)"),
        }
    }
    fn id(self) -> u8 {
        match self {
            BotKind::Mcts => BOT_MCTS,
            BotKind::Depth2 => BOT_DEPTH2,
            BotKind::Expectimax => BOT_EXPECTIMAX,
        }
    }
    fn name(self) -> &'static str {
        match self {
            BotKind::Mcts => "mcts",
            BotKind::Depth2 => "depth2",
            BotKind::Expectimax => "expectimax",
        }
    }
}

#[derive(Clone, Copy)]
struct Config {
    teacher: BotKind,
    opponent: BotKind,
    mcts_sims: usize,
    dirichlet: bool,
    /// For the first `explore_moves` MCTS decisions of each game, the move
    /// PLAYED is sampled from the visit distribution (temperature 1) for
    /// opening diversity; the recorded label is still the argmax.
    explore_moves: usize,
}

enum Bot {
    Mcts(TwoPlayerMCTSBot),
    Other(Box<dyn LudoBot>),
}

impl Bot {
    fn new(kind: BotKind, seat: u8, cfg: &Config) -> Bot {
        match kind {
            BotKind::Mcts => {
                let mut b = TwoPlayerMCTSBot::new(Some(seat), cfg.mcts_sims);
                b.add_dirichlet = cfg.dirichlet;
                Bot::Mcts(b)
            }
            BotKind::Depth2 => Bot::Other(Box::new(Depth2ExpectimaxBot::new(Some(seat)))),
            BotKind::Expectimax => Bot::Other(Box::new(ExpectimaxBot::new(Some(seat)))),
        }
    }
}

fn play_teacher_game(cfg: &Config, game_id: u32) -> Vec<SLRecordV2> {
    let mut rng = rand::thread_rng();
    let mut state = GameState::new_2p();
    let teacher_seat: u8 = if game_id % 2 == 0 { 0 } else { 2 };
    let opp_seat: u8 = 2 - teacher_seat;
    let teacher = Bot::new(cfg.teacher, teacher_seat, cfg);
    let opponent = Bot::new(cfg.opponent, opp_seat, cfg);

    let mut recs: Vec<SLRecordV2> = Vec::with_capacity(200);
    let mut mcts_decisions = 0usize;
    let mut move_count = 0;

    while !state.is_terminal && move_count < 450 {
        let d = rng.gen_range(1..=6);
        if !state.roll(d) {
            continue; // no legal move or three-six forfeit: turn already passed
        }
        let cp = state.current_player;
        let (legal, count) = state.legal_moves();

        let is_teacher = cp == teacher_seat;
        let (bot, kind) = if is_teacher { (&teacher, cfg.teacher) } else { (&opponent, cfg.opponent) };

        let mut flags = if is_teacher { FLAG_TEACHER_SEAT } else { 0 };
        let mut pi = [0u8; 4];
        let (label, played) = if count == 1 {
            flags |= FLAG_FORCED;
            pi[legal[0] as usize] = 255;
            (legal[0], legal[0])
        } else {
            match bot {
                Bot::Mcts(m) => {
                    let (argmax, pif) = m.search_with_temp(&state, 0.0);
                    pi = quantise_pi(&pif);
                    flags |= FLAG_HAS_PI;
                    let mut played = argmax;
                    if mcts_decisions < cfg.explore_moves {
                        // Sample from the same visit distribution (temperature 1).
                        let r: f32 = rng.gen();
                        let mut acc = 0.0;
                        for &a in &legal[..count] {
                            acc += pif[a as usize];
                            if r < acc {
                                played = a;
                                break;
                            }
                        }
                        if played != argmax {
                            flags |= FLAG_EXPLORATORY;
                        }
                    }
                    mcts_decisions += 1;
                    (argmax, played)
                }
                Bot::Other(b) => {
                    let a = b.select_move(&state);
                    pi[a as usize] = 255;
                    (a, a)
                }
            }
        };

        let mut pos_flat = [0i8; 16];
        for p in 0..4 {
            for t in 0..4 {
                pos_flat[p * 4 + t] = state.positions[p][t];
            }
        }
        recs.push(SLRecordV2 {
            positions: pos_flat,
            current_player: cp as i8,
            dice_roll: state.current_dice_roll as i8,
            action: label as i8,
            target_value: 0,
            mover_bot: kind.id(),
            legal_count: count as u8,
            consecutive_sixes: state.consecutive_sixes[cp as usize],
            flags,
            pi,
            game_id,
        });

        state = state.apply_move(played);
        move_count += 1;
    }

    if state.is_terminal {
        for r in recs.iter_mut() {
            r.target_value = if r.current_player == state.winner { 1 } else { -1 };
        }
    }
    recs
}

fn arg<T: std::str::FromStr>(args: &[String], name: &str, default: T) -> T {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let target_games: usize = arg(&args, "--target-games", 1_000_000);
    let shard_size: usize = arg(&args, "--shard-size", 5_000);
    let mcts_sims: usize = arg(&args, "--mcts-sims", 3_000);
    let out_dir: PathBuf = PathBuf::from(arg(&args, "--out-dir", "data/sl_teacher_v2".to_string()));
    let teacher = BotKind::parse(&arg(&args, "--teacher", "mcts".to_string()));
    let opponent = BotKind::parse(&arg(&args, "--opponent", "depth2".to_string()));
    let dirichlet = args.iter().any(|a| a == "--dirichlet");
    let explore_moves: usize = arg(&args, "--explore-moves", 0);
    let stop_file: PathBuf = PathBuf::from(arg(&args, "--stop-file", "stop".to_string()));

    let cfg = Config { teacher, opponent, mcts_sims, dirichlet, explore_moves };

    fs::create_dir_all(&out_dir).expect("Failed to create output directory");

    // Pin the dataset config. A resumed run with different settings would
    // silently produce a mixed dataset, so refuse instead.
    let manifest = format!(
        "{{\n  \"format_version\": {},\n  \"record_size\": 32,\n  \"teacher\": \"{}\",\n  \"opponent\": \"{}\",\n  \"mcts_sims\": {},\n  \"dirichlet\": {},\n  \"explore_moves\": {},\n  \"shard_size\": {},\n  \"seat_rule\": \"teacher seat = 0 if game_id even else 2\"\n}}\n",
        FORMAT_VERSION, teacher.name(), opponent.name(), mcts_sims, dirichlet, explore_moves, shard_size
    );
    let manifest_path = out_dir.join("manifest.json");
    match fs::read_to_string(&manifest_path) {
        Ok(existing) if existing != manifest => {
            eprintln!(
                "Refusing to resume: {:?} was generated with a different config.\nexisting:\n{existing}\nrequested:\n{manifest}",
                manifest_path
            );
            std::process::exit(2);
        }
        Ok(_) => {}
        Err(_) => {
            let has_shards = fs::read_dir(&out_dir)
                .map(|it| it.flatten().any(|e| e.file_name().to_string_lossy().starts_with("shard_")))
                .unwrap_or(false);
            if has_shards {
                eprintln!(
                    "Refusing to write into {:?}: it already holds shards but no manifest.json \
                     (format v1 / unknown teacher). Use a new --out-dir.",
                    out_dir
                );
                std::process::exit(2);
            }
            fs::write(&manifest_path, &manifest).expect("Failed to write manifest");
        }
    }

    println!("================================================================================");
    println!("🚀 RUST SL TEACHER GENERATOR (format v{FORMAT_VERSION})");
    println!("   Teacher / Opponent:  {} / {}", teacher.name(), opponent.name());
    println!("   Target Games:        {:>10}", target_games);
    println!("   Shard Size:          {:>10} games/shard", shard_size);
    println!("   MCTS Simulations:    {:>10} sims/move", mcts_sims);
    println!("   Dirichlet / explore: {} / {} moves", dirichlet, explore_moves);
    println!("   Output Directory:    {:?}", out_dir);
    println!("   Parallel Threads:    {:>10}", rayon::current_num_threads());
    println!("================================================================================\n");

    let num_shards = (target_games + shard_size - 1) / shard_size;
    let mut total_states = 0usize;
    let mut total_games = 0usize;
    let t_start = Instant::now();

    for shard_idx in 0..num_shards {
        let shard_file = out_dir.join(format!("shard_{:05}.bin", shard_idx));
        if shard_file.exists() {
            continue;
        }
        if stop_file.exists() || Path::new("stop").exists() || Path::new("checkpoints/stop").exists() {
            println!("\n[Stop] stop file detected! Exiting cleanly...");
            break;
        }

        let shard_start_game = shard_idx * shard_size;
        let n_games = (target_games - shard_start_game).min(shard_size);
        let t_shard = Instant::now();

        let records: Vec<SLRecordV2> = (0..n_games)
            .into_par_iter()
            .flat_map_iter(|i| play_teacher_game(&cfg, (shard_start_game + i) as u32))
            .collect();

        let tmp_file = out_dir.join(format!("shard_{:05}.bin.tmp", shard_idx));
        {
            let mut w = BufWriter::new(File::create(&tmp_file).expect("Failed to create temp shard"));
            w.write_all(&shard_header(records.len())).expect("Failed to write header");
            let bytes = records_as_bytes(&records);
            w.write_all(bytes).expect("Failed to write shard data");
            w.flush().expect("Failed to flush shard");
        }
        fs::rename(&tmp_file, &shard_file).expect("Failed to rename shard");

        total_states += records.len();
        total_games += n_games;
        let dt = t_shard.elapsed().as_secs_f64();
        println!(
            "📦 Shard {:05}/{} written: {} states ({:.1}s) | {:.1} GPM ({:.0} SPS) | Session: {} games ({} states) in {:.1}m",
            shard_idx,
            num_shards,
            records.len(),
            dt,
            n_games as f64 / dt * 60.0,
            records.len() as f64 / dt,
            total_games,
            total_states,
            t_start.elapsed().as_secs_f64() / 60.0,
        );
    }

    println!("\n✅ Generation loop finished. Total time: {:.1} minutes", t_start.elapsed().as_secs_f64() / 60.0);
}
