//! High-Throughput Rust Multi-Threaded Generator for V15.3 SL Teacher Dataset.
//!
//! Generates master-class training games between:
//!   - TwoPlayerMCTSBot (Expecti-MCTS with N=3,000 simulations)
//!   - Depth2ExpectimaxBot (True Depth-2 Expectimax search)
//! with alternating seat rotation (P0 vs P2).
//!
//! Outputs compact binary shards (20 bytes per state record) ready for zero-copy
//! streaming into PyTorch DataLoader.

use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

use ludo_core::GameState;
use ludo_search::{Depth2ExpectimaxBot, LudoBot, TwoPlayerMCTSBot};
use rand::Rng;
use rayon::prelude::*;

#[repr(C, packed)]
#[derive(Clone, Copy)]
pub struct SLRecord {
    pub positions: [i8; 16],
    pub current_player: i8,
    pub dice_roll: i8,
    pub action: i8,
    pub target_value: i8,
}

struct RawStep {
    positions: [i8; 16],
    current_player: i8,
    dice_roll: i8,
    action: i8,
}

fn play_teacher_game(mcts_sims: usize, mcts_seat: u8) -> (Vec<RawStep>, i8) {
    let mut rng = rand::thread_rng();
    let mut state = GameState::new_2p();
    let mut mcts_bot = TwoPlayerMCTSBot::new(Some(mcts_seat), mcts_sims);
    mcts_bot.add_dirichlet = true; // AlphaZero root exploration
    mcts_bot.use_pure_terminal = true;

    let depth2_seat = if mcts_seat == 0 { 2 } else { 0 };
    let depth2_bot = Depth2ExpectimaxBot::new(Some(depth2_seat));

    let mut steps: Vec<RawStep> = Vec::with_capacity(160);
    let mut move_count = 0;

    while !state.is_terminal && move_count < 450 {
        let cp = state.current_player;
        let d = rng.gen_range(1..=6);
        state.set_dice(d);

        let (legal, count) = state.legal_moves();
        if count == 0 {
            state.pass_turn();
            continue;
        }

        let mut pos_flat = [0i8; 16];
        for p in 0..4 {
            for t in 0..4 {
                pos_flat[p * 4 + t] = state.positions[p][t];
            }
        }

        let action = if count == 1 {
            legal[0]
        } else if cp == mcts_seat {
            let (act, _) = mcts_bot.search_with_temp(&state, 0.0);
            act
        } else {
            depth2_bot.select_move(&state)
        };

        steps.push(RawStep {
            positions: pos_flat,
            current_player: cp as i8,
            dice_roll: d as i8,
            action: action as i8,
        });

        state = state.apply_move(action);
        move_count += 1;
    }

    let winner = if state.is_terminal { state.winner } else { -1 };
    (steps, winner)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let target_games: usize = args
        .iter()
        .position(|a| a == "--target-games")
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(1_000_000);

    let shard_size: usize = args
        .iter()
        .position(|a| a == "--shard-size")
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(5_000);

    let mcts_sims: usize = args
        .iter()
        .position(|a| a == "--mcts-sims")
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(3_000);

    let out_dir: PathBuf = args
        .iter()
        .position(|a| a == "--out-dir")
        .and_then(|i| args.get(i + 1))
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("data/sl_teacher_v153"));

    fs::create_dir_all(&out_dir).expect("Failed to create output directory");

    println!("================================================================================");
    println!("🚀 RUST SL TEACHER GENERATOR (V15.3 AlphaZero Distillation)");
    println!("   Target Games:      {:>10}", target_games);
    println!("   Shard Size:        {:>10} games/shard", shard_size);
    println!("   MCTS Simulations:  {:>10} sims/move", mcts_sims);
    println!("   Output Directory:  {:?}", out_dir);
    println!("   Parallel Threads:  {:>10}", rayon::current_num_threads());
    println!("================================================================================\n");

    let num_shards = (target_games + shard_size - 1) / shard_size;
    let total_states_gen = Arc::new(AtomicUsize::new(0));
    let total_games_gen = Arc::new(AtomicUsize::new(0));
    let t_start = Instant::now();

    for shard_idx in 0..num_shards {
        let shard_file = out_dir.join(format!("shard_{:05}.bin", shard_idx));
        if shard_file.exists() {
            println!("  [Skip] Shard {:05} already exists, skipping...", shard_idx);
            continue;
        }

        // Check for stop file
        if Path::new("stop").exists() || Path::new("checkpoints/stop").exists() {
            println!("\n[Stop] 'stop' file detected! Exiting cleanly...");
            break;
        }

        let shard_start_game = shard_idx * shard_size;
        let current_shard_games = (target_games - shard_start_game).min(shard_size);

        let t_shard = Instant::now();

        // Parallel game generation using Rayon
        let shard_records: Vec<SLRecord> = (0..current_shard_games)
            .into_par_iter()
            .flat_map_iter(|i| {
                let game_num = shard_start_game + i;
                let mcts_seat = if game_num % 2 == 0 { 0 } else { 2 };
                let (steps, winner) = play_teacher_game(mcts_sims, mcts_seat);

                steps.into_iter().map(move |step| {
                    let target_val = if winner >= 0 {
                        if step.current_player == winner { 1i8 } else { -1i8 }
                    } else {
                        0i8
                    };
                    SLRecord {
                        positions: step.positions,
                        current_player: step.current_player,
                        dice_roll: step.dice_roll,
                        action: step.action,
                        target_value: target_val,
                    }
                })
            })
            .collect();

        // Write binary shard atomically
        let tmp_file = out_dir.join(format!("shard_{:05}.bin.tmp", shard_idx));
        {
            let file = File::create(&tmp_file).expect("Failed to create temp shard file");
            let mut writer = BufWriter::new(file);

            let byte_slice: &[u8] = unsafe {
                std::slice::from_raw_parts(
                    shard_records.as_ptr() as *const u8,
                    shard_records.len() * std::mem::size_of::<SLRecord>(),
                )
            };
            writer.write_all(byte_slice).expect("Failed to write shard data");
            writer.flush().expect("Failed to flush shard data");
        }
        fs::rename(&tmp_file, &shard_file).expect("Failed to atomically rename shard file");

        let n_states = shard_records.len();
        let total_st = total_states_gen.fetch_add(n_states, Ordering::Relaxed) + n_states;
        let total_gm = total_games_gen.fetch_add(current_shard_games, Ordering::Relaxed) + current_shard_games;

        let dt_shard = t_shard.elapsed().as_secs_f64();
        let dt_total = t_start.elapsed().as_secs_f64();
        let gps = current_shard_games as f64 / dt_shard;
        let sps = n_states as f64 / dt_shard;

        println!(
            "📦 Shard {:05}/{} written: {} states ({:.1}s) | Shard Rate: {:.1} GPM ({:.0} SPS) | Total: {} games ({} states) in {:.1}m",
            shard_idx,
            num_shards,
            n_states,
            dt_shard,
            gps * 60.0,
            sps,
            total_gm,
            total_st,
            dt_total / 60.0,
        );
    }

    println!("\n✅ Generation loop finished. Total time: {:.1} minutes", t_start.elapsed().as_secs_f64() / 60.0);
}
