use ludo_core::GameState;
use ludo_search::{Depth2ExpectimaxBot, LudoBot, TwoPlayerMCTSBot};
use rand::Rng;
use std::time::Instant;

fn main() {
    println!("================================================================================");
    println!("🔬 2-Player AlphaZero MCTS High-Confidence Calibration (Expanded Budget)");
    println!("================================================================================");

    // ─────────────────────────────────────────────────────────────
    // Part 1: Decision Stability Analysis (400 Mid-Game Positions)
    // ─────────────────────────────────────────────────────────────
    let num_positions = 400;
    println!("\n📊 PART 1: Decision Stability Across {} Mid-Game Positions", num_positions);
    println!("Evaluating Action Agreement % against N=3,000 Gold Standard...\n");

    let mut rng = rand::thread_rng();
    let mut test_states: Vec<GameState> = Vec::with_capacity(num_positions);

    // Collect 400 genuine non-trivial mid-game states
    while test_states.len() < num_positions {
        let mut state = GameState::new_2p();
        let mut move_count = 0;
        let target_depth = rng.gen_range(20..85);

        while !state.is_terminal && move_count < target_depth {
            let d = rng.gen_range(1..=6);
            state.set_dice(d);

            let (legal, count) = state.legal_moves();
            if count == 0 {
                state.pass_turn();
                continue;
            }

            let pick = rng.gen_range(0..count);
            state = state.apply_move(legal[pick]);
            move_count += 1;
        }

        let d = rng.gen_range(1..=6);
        state.set_dice(d);
        let (_, count) = state.legal_moves();
        if !state.is_terminal && count >= 2 {
            test_states.push(state);
        }
    }

    let budget_levels = [50, 150, 300, 500, 800, 1200, 2000, 3000];
    let mut decisions: Vec<Vec<u8>> = Vec::with_capacity(budget_levels.len());
    let mut avg_latencies_us: Vec<f64> = Vec::with_capacity(budget_levels.len());

    for &n_sims in &budget_levels {
        let bot = TwoPlayerMCTSBot::new(None, n_sims);
        let mut bot_decisions = Vec::with_capacity(test_states.len());
        let t0 = Instant::now();

        for state in &test_states {
            let (action, _) = bot.search(state);
            bot_decisions.push(action);
        }

        let elapsed = t0.elapsed();
        let avg_us = (elapsed.as_nanos() as f64) / (test_states.len() as f64 * 1000.0);
        avg_latencies_us.push(avg_us);
        decisions.push(bot_decisions);
        print!("[N={:<4} done] ", n_sims);
        std::io::Write::flush(&mut std::io::stdout()).unwrap();
    }
    println!("\n");

    let gold_idx = budget_levels.len() - 1; // N=3000 is reference
    let gold_decisions = &decisions[gold_idx];

    println!("{:<10} | {:>18} | {:>14} | {:>14} | {:>14}", "Sims (N)", "Agreement vs 3000", "95% Conf. Int.", "Action Flips", "Avg Move Time");
    println!("{:-<10}-+-{:-<18}-+-{:-<14}-+-{:-<14}-+-{:-<14}", "", "", "", "", "");

    for i in 0..budget_levels.len() {
        let mut matches = 0;
        for j in 0..test_states.len() {
            if decisions[i][j] == gold_decisions[j] {
                matches += 1;
            }
        }
        let p = (matches as f64) / (test_states.len() as f64);
        let agreement_pct = p * 100.0;
        let flips = test_states.len() - matches;

        // 95% Wilson score / Wald interval for binomial proportion
        let se = (p * (1.0 - p) / (test_states.len() as f64)).sqrt();
        let ci_margin = 1.96 * se * 100.0;

        let lat_str = if avg_latencies_us[i] >= 1000.0 {
            format!("{:.2} ms", avg_latencies_us[i] / 1000.0)
        } else {
            format!("{:.0} µs", avg_latencies_us[i])
        };

        println!(
            "{:<10} | {:>17.1}% | {:>11.1}% ±{:.1}% | {:>14} | {:>14}",
            budget_levels[i], agreement_pct, agreement_pct, ci_margin, flips, lat_str
        );
    }

    // ─────────────────────────────────────────────────────────────
    // Part 2: Head-to-Head 2-Player Scaling Tournaments (100 Games each)
    // ─────────────────────────────────────────────────────────────
    println!("\n================================================================================");
    println!("⚔️ PART 2: High-Confidence Head-to-Head Tournaments (100 Games per Matchup)");
    println!("Seat-balanced: exactly 50 games as Player 0, 50 games as Player 2");
    println!("================================================================================");

    let matchups: [(&str, Box<dyn Fn() -> Box<dyn LudoBot>>, &str, Box<dyn Fn() -> Box<dyn LudoBot>>); 5] = [
        ("MCTS(300)", Box::new(|| Box::new(TwoPlayerMCTSBot::new(None, 300))), "MCTS(100)", Box::new(|| Box::new(TwoPlayerMCTSBot::new(None, 100)))),
        ("MCTS(600)", Box::new(|| Box::new(TwoPlayerMCTSBot::new(None, 600))), "MCTS(300)", Box::new(|| Box::new(TwoPlayerMCTSBot::new(None, 300)))),
        ("MCTS(1200)", Box::new(|| Box::new(TwoPlayerMCTSBot::new(None, 1200))), "MCTS(300)", Box::new(|| Box::new(TwoPlayerMCTSBot::new(None, 300)))),
        ("MCTS(300)", Box::new(|| Box::new(TwoPlayerMCTSBot::new(None, 300))), "Depth2Expectimax", Box::new(|| Box::new(Depth2ExpectimaxBot::new(None)))),
        ("MCTS(800)", Box::new(|| Box::new(TwoPlayerMCTSBot::new(None, 800))), "Depth2Expectimax", Box::new(|| Box::new(Depth2ExpectimaxBot::new(None)))),
    ];

    println!("{:<14} vs {:<18} | {:>10} | {:>10} | {:>16}", "Bot A", "Bot B", "Score (A-B)", "Win Rate A", "95% CI");
    println!("{:-<14}-+--{:-<18}-+-{:-<10}-+-{:-<10}-+-{:-<16}", "", "", "", "", "");

    for (name_a, make_a, name_b, make_b) in matchups {
        let bot_a = make_a();
        let bot_b = make_b();
        let total_games = 100;
        let mut wins_a = 0;
        let mut wins_b = 0;

        for g in 0..total_games {
            let mut state = GameState::new_2p();
            let seat_a = if g % 2 == 0 { 0 } else { 2 };
            let seat_b = if seat_a == 0 { 2 } else { 0 };

            let mut mc = 0;
            while !state.is_terminal && mc < 600 {
                let cp = state.current_player;
                let d = rng.gen_range(1..=6);
                state.set_dice(d);

                let (_, count) = state.legal_moves();
                if count == 0 {
                    state.pass_turn();
                    continue;
                }

                let action = if cp == seat_a {
                    bot_a.select_move(&state)
                } else {
                    bot_b.select_move(&state)
                };

                state = state.apply_move(action);
                mc += 1;
            }

            if state.is_terminal {
                if state.winner == seat_a as i8 {
                    wins_a += 1;
                } else if state.winner == seat_b as i8 {
                    wins_b += 1;
                }
            }
        }

        let p = (wins_a as f64) / (total_games as f64);
        let wr = p * 100.0;
        let se = (p * (1.0 - p) / (total_games as f64)).sqrt();
        let ci = 1.96 * se * 100.0;

        println!(
            "{:<14} vs {:<18} | {:>4} - {:<4} | {:>9.1}% | [{:4.1}% - {:4.1}%]",
            name_a, name_b, wins_a, wins_b, wr, (wr - ci).max(0.0), (wr + ci).min(100.0)
        );
    }

    println!("================================================================================");
}
