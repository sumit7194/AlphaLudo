//! NetFarm driven by a native stand-in "network": uniform policy over legal
//! cells, value = calibrated leaf evaluator. Checks that every game finishes
//! (no stalls / lost games, with and without tree reuse) and that search
//! with an informative value beats the greedy-uniform policy.

use ludo_core::NUM_BOARD_CELLS;
use ludo_search::evaluator::leaf_value;
use ludo_search::{NetExpectiConfig, NetFarm, NetMctsConfig, SeatConfig, SeatKind};

fn run(seat_a: SeatConfig, seat_b: SeatConfig, games: u32, seed: u64) -> (f64, NetFarm) {
    let mut farm = NetFarm::new(seat_a, seat_b, games, 16, seed);
    loop {
        let n = farm.collect();
        if n == 0 {
            break;
        }
        let mut pol = vec![0f32; n * NUM_BOARD_CELLS];
        let mut val = vec![0f32; n];
        for i in 0..n {
            let m = &farm.req_mask[i * NUM_BOARD_CELLS..(i + 1) * NUM_BOARD_CELLS];
            let k: f32 = m.iter().sum();
            for c in 0..NUM_BOARD_CELLS {
                pol[i * NUM_BOARD_CELLS + c] = m[c] / k;
            }
            let s = &farm.req_states[i];
            val[i] = 0.5 * (leaf_value(s.current_player, s) + 1.0);
        }
        farm.submit(&pol, &val);
    }
    assert_eq!(farm.results.len(), games as usize, "every game must finish exactly once");
    let wins = farm.results.iter().filter(|r| r.winner == r.a_seat as i8).count();
    (wins as f64 / games as f64, farm)
}

fn mcts(sims: u32, reuse: bool) -> SeatConfig {
    SeatConfig {
        kind: SeatKind::Mcts(NetMctsConfig { sims, reuse_tree: reuse, ..NetMctsConfig::default() }),
        model: 0,
        record: true,
        explore_moves: 0,
    }
}

const GREEDY: SeatConfig = SeatConfig { kind: SeatKind::Greedy, model: 0, record: false, explore_moves: 0 };

#[test]
fn farm_finishes_all_games_and_search_helps() {
    for reuse in [false, true] {
        let (wr, farm) = run(mcts(100, reuse), GREEDY, 60, 7);
        println!("reuse={reuse}: MCTS(100, leaf-eval value) vs greedy-uniform = {wr:.3}, reused visits {}", farm.reused_visits);
        assert!(wr > 0.6, "search with an informative value should beat greedy-uniform (got {wr})");
        assert!(!farm.records.is_empty());
        for r in &farm.records {
            let lc = r.legal_count;
            let pi_sum: u32 = r.pi.iter().map(|&x| x as u32).sum();
            assert!(lc >= 1 && pi_sum > 200 && pi_sum < 260);
        }
        if reuse {
            assert!(farm.reused_visits > 0);
        }
    }
}

#[test]
fn single_sim_matches_greedy_policy() {
    // With one simulation the root's most-visited move is its highest-prior
    // move, i.e. the greedy move; uniform priors break ties identically.
    let (wr, _) = run(mcts(1, false), GREEDY, 40, 3);
    assert!(wr > 0.2 && wr < 0.8);
}

#[test]
fn expecti_finishes_and_helps() {
    for evals in [1u32, 30, 100] {
        let seat = SeatConfig {
            kind: SeatKind::Expecti(NetExpectiConfig { max_evals: evals, ..NetExpectiConfig::default() }),
            model: 0,
            record: true,
            explore_moves: 0,
        };
        let (wr, farm) = run(seat, GREEDY, 300, 11);
        println!("Expecti(max_evals={evals}, leaf-eval value) vs greedy-uniform = {wr:.3}, records {}", farm.records.len());
        for r in farm.records.iter().filter(|r| r.legal_count >= 2).take(6) {
            let (a, pi, lc) = (r.action, r.pi, r.legal_count);
            println!("   label {a} pi {pi:?} legal {lc}");
        }
        if evals == 1 {
            // Root evaluation only = the greedy policy itself.
            assert!(wr > 0.3 && wr < 0.7, "1 eval should match greedy (got {wr})");
        } else {
            // 300 games: stderr ~2.8 pp. Low budgets give a smaller edge.
            let floor = if evals < 60 { 0.55 } else { 0.6 };
            assert!(wr > floor, "exact-expectation lookahead should beat greedy-uniform (got {wr})");
        }
    }
}
