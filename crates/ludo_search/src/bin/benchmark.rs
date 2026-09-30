use ludo_core::GameState;
use ludo_search::{AggressiveExpectimaxBot, Depth2ExpectimaxBot, ExpectimaxBot, MaxNMCTSBot, LudoBot};
use rand::Rng;
use std::time::Instant;

fn create_bot_by_id(id: usize, seat: u8) -> Box<dyn LudoBot> {
    match id {
        0 => Box::new(MaxNMCTSBot::new(Some(seat), 40)),
        1 => Box::new(Depth2ExpectimaxBot::new(Some(seat))),
        2 => Box::new(AggressiveExpectimaxBot::new(Some(seat))),
        _ => Box::new(ExpectimaxBot::new(Some(seat))),
    }
}

const BOT_NAMES: [&str; 4] = [
    "MaxNMCTS (40 rollouts)",
    "Depth2Expectimax (2-ply)",
    "AggressiveExpectimax (1-ply)",
    "BaselineExpectimax (1-ply)",
];

fn main() {
    println!("============================================================");
    println!("🏆 Rust Ludo Bot Zoo Tournament (Seat-Rotated 4-Way Match)");
    println!("============================================================");

    let num_rounds = 50; // 50 rounds * 4 seat permutations = 200 games
    let mut wins = [0usize; 4];
    let mut move_times_ns = [0u128; 4];
    let mut move_counts = [0usize; 4];
    let mut total_games = 0;

    let start_all = Instant::now();

    for round in 0..num_rounds {
        for rot in 0..4 {
            let mut state = GameState::new_4p();
            let mut rng = rand::thread_rng();

            // Seat assignment: seat p gets bot (p + rot) % 4
            let seat_to_bot = [
                (0 + rot) % 4,
                (1 + rot) % 4,
                (2 + rot) % 4,
                (3 + rot) % 4,
            ];

            let bots: [Box<dyn LudoBot>; 4] = [
                create_bot_by_id(seat_to_bot[0], 0),
                create_bot_by_id(seat_to_bot[1], 1),
                create_bot_by_id(seat_to_bot[2], 2),
                create_bot_by_id(seat_to_bot[3], 3),
            ];

            let mut moves = 0;
            while !state.is_terminal && moves < 800 {
                let cp = state.current_player as usize;
                let d = rng.gen_range(1..=6);
                state.set_dice(d);

                let (_, count) = state.legal_moves();
                if count == 0 {
                    state.pass_turn();
                    continue;
                }

                let bot_id = seat_to_bot[cp];
                let t0 = Instant::now();
                let a = bots[cp].select_move(&state);
                let dt = t0.elapsed().as_nanos();

                move_times_ns[bot_id] += dt;
                move_counts[bot_id] += 1;

                state = state.apply_move(a);
                moves += 1;
            }

            if state.is_terminal && state.winner >= 0 {
                let winning_seat = state.winner as usize;
                let winning_bot = seat_to_bot[winning_seat];
                wins[winning_bot] += 1;
                total_games += 1;
            }
        }
        if (round + 1) % 10 == 0 {
            print!(".");
            std::io::Write::flush(&mut std::io::stdout()).unwrap();
        }
    }
    println!("\nTournament completed in {:.2?}!\n", start_all.elapsed());

    println!("{:<30} | {:>6} | {:>8} | {:>15}", "Bot Name", "Wins", "Win Rate", "Avg Time/Move");
    println!("{:-<30}-+-{:-<6}-+-{:-<8}-+-{:-<15}", "", "", "", "");

    let total_moves_all: usize = move_counts.iter().sum();
    println!("Total games: {}", total_games);
    println!("Total moves simulated: {}", total_moves_all);
    println!("Avg total moves per game: {:.1}", total_moves_all as f64 / total_games as f64);
    println!("Avg moves per player: {:.1}", (total_moves_all as f64 / total_games as f64) / 4.0);

    for i in 0..4 {
        let wr = (wins[i] as f64 / total_games as f64) * 100.0;
        let avg_time_us = if move_counts[i] > 0 {
            (move_times_ns[i] as f64 / move_counts[i] as f64) / 1000.0
        } else {
            0.0
        };
        println!(
            "{:<30} | {:>6} | {:>7.1}% | {:>12.2} µs | {:>8} moves",
            BOT_NAMES[i], wins[i], wr, avg_time_us, move_counts[i]
        );
    }
    println!("============================================================");
}
