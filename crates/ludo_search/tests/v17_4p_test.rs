//! Tests for the 2026-10-07 4P teacher pieces: eval4 features/shares, the exact
//! full-round search, Max^n `search`, and the mixed 2P/4P SL stream.
use ludo_core::{GameState, BASE_POS, HOME_POS};
use ludo_search::farm::BotKind;
use ludo_search::sl_stream::{SlStream, SlStreamConfig, NO_WINNER};
use ludo_search::{
    e4_probs_with, seat_features, LudoBot, MaxNConfig, MaxNMCTSBot, RoundBot, RoundConfig, RoundLeaf, E4_WEIGHTS, N_E4,
};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Plays `n` random legal moves from a fresh 4P game (pre-roll states returned).
fn random_states(n: usize, seed: u64) -> Vec<GameState> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut out = Vec::new();
    let mut s = GameState::new_4p();
    while out.len() < n {
        if s.is_terminal {
            s = GameState::new_4p();
        }
        if s.current_dice_roll == 0 {
            out.push(s);
        }
        if !s.roll(rng.gen_range(1..=6)) {
            continue;
        }
        let (lm, c) = s.legal_moves();
        s = s.apply_move(lm[rng.gen_range(0..c)]);
    }
    out
}

/// Seat k -> seat k+1 (board rotated by a quarter turn). Relative positions are
/// per-seat, so the rotated game is the same game seen one seat later.
fn rotate(s: &GameState) -> GameState {
    let mut r = *s;
    for p in 0..4 {
        let q = (p + 1) % 4;
        r.positions[q] = s.positions[p];
        r.scores[q] = s.scores[p];
        r.consecutive_sixes[q] = s.consecutive_sixes[p];
        r.active_players[q] = s.active_players[p];
    }
    r.current_player = (s.current_player + 1) % 4;
    if s.winner >= 0 {
        r.winner = ((s.winner as u8 + 1) % 4) as i8;
    }
    r
}

#[test]
fn eval4_features_rotate_with_the_seats() {
    for s in random_states(3000, 1) {
        let f = seat_features(&s);
        let g = seat_features(&rotate(&s));
        for p in 0..4 {
            for i in 0..N_E4 {
                assert!((f[p][i] - g[(p + 1) % 4][i]).abs() < 1e-5, "seat {p} feature {i}: {} vs {}", f[p][i], g[(p + 1) % 4][i]);
            }
        }
    }
}

#[test]
fn eval4_shares_are_a_distribution_over_active_seats() {
    let w = [0.6f32, -0.7, 0.4, 0.8, -0.6, 0.1, 3.0, -0.1, 0.7, 0.08, 0.05, 0.03, -0.3, 0.2, 0.1, 2.1];
    for s in random_states(500, 2) {
        let p = e4_probs_with(&s, &w);
        assert!((p.iter().sum::<f32>() - 1.0).abs() < 1e-5);
    }
    let mut two = GameState::new_2p();
    two.positions[0][0] = 10;
    let p = e4_probs_with(&two, &w);
    assert_eq!(p[1], 0.0);
    assert_eq!(p[3], 0.0);
    assert!((p[0] + p[2] - 1.0).abs() < 1e-6 && p[0] > p[2]);
    let mut done = GameState::new_4p();
    done.is_terminal = true;
    done.winner = 3;
    assert_eq!(e4_probs_with(&done, &E4_WEIGHTS), [0.0, 0.0, 0.0, 1.0]);
}

#[test]
fn eval4_exposure_respects_blockades_and_safe_squares() {
    // Seat 1's token sits 3 squares behind seat 0's token at relative 20 (absolute 20).
    let mut s = GameState::new_4p();
    s.positions[0][0] = 20; // abs 20, unsafe
    s.positions[1][0] = 4; // abs 17 -> distance 3
    let f = seat_features(&s);
    assert!(f[0][4] > 0.0, "single unsafe token within reach is exposed");
    assert!(f[1][5] > 0.0, "attacker records a threat");
    s.positions[0][1] = 20; // blockade on abs 20
    let f = seat_features(&s);
    assert_eq!(f[0][4], 0.0, "a blockade cannot be captured");
    let mut t = GameState::new_4p();
    t.positions[0][0] = 21; // abs 21 is a safe square
    t.positions[1][0] = 5; // abs 18
    assert_eq!(seat_features(&t)[0][4], 0.0, "safe squares are never exposed");
    let _ = (BASE_POS, HOME_POS);
}

#[test]
fn round_bot_values_every_legal_move_and_is_deterministic() {
    let cfgs = [
        RoundConfig { leaf: RoundLeaf::Score, max_rolls: 3 },
        RoundConfig { leaf: RoundLeaf::E4(E4_WEIGHTS), max_rolls: 2 },
        RoundConfig { leaf: RoundLeaf::Lin { w: [1.0; N_E4], beta: 0.5 }, max_rolls: 3 },
    ];
    let mut rng = StdRng::seed_from_u64(3);
    for cfg in cfgs {
        let b = RoundBot::new(None, cfg);
        let mut checked = 0;
        for mut s in random_states(400, 4) {
            if !s.roll(rng.gen_range(1..=6)) {
                continue;
            }
            let (vals, legal, count) = b.move_values(&s);
            let (lm, c) = s.legal_moves();
            assert_eq!((legal, count), (lm, c));
            assert!(vals[..count].iter().all(|v| v.is_finite()));
            let m = b.select_move(&s);
            assert!(lm[..c].contains(&m));
            assert_eq!(m, b.select_move(&s));
            checked += 1;
        }
        assert!(checked > 100);
    }
}

#[test]
fn maxn_search_returns_a_visit_distribution_over_legal_tokens() {
    let mut rng = StdRng::seed_from_u64(5);
    let bot = MaxNMCTSBot::with_config(None, MaxNConfig { sims: 200, ..MaxNConfig::legacy(200) });
    let mut n = 0;
    for mut s in random_states(300, 6) {
        if !s.roll(rng.gen_range(1..=6)) {
            continue;
        }
        let (lm, c) = s.legal_moves();
        let (a, pi) = bot.search(&s, 0.0, &mut rng);
        assert!(lm[..c].contains(&a));
        assert!((pi.iter().sum::<f32>() - 1.0).abs() < 1e-5);
        for t in 0..4u8 {
            if !lm[..c].contains(&t) {
                assert_eq!(pi[t as usize], 0.0);
            }
        }
        n += 1;
        if n == 60 {
            break;
        }
    }
}

#[test]
fn mixed_stream_produces_both_modes_with_valid_labels() {
    let mut st = SlStream::start(SlStreamConfig {
        threads: 2,
        sims: 30,
        keep_prob: 0.3,
        capacity: 20_000,
        max_moves: 600,
        seed: 9,
        frac4: 0.6,
        teacher4: Some(BotKind::Round(RoundConfig { leaf: RoundLeaf::Score, max_rolls: 2 })),
        max_moves4: 1500,
    });
    let mut recs = Vec::new();
    while recs.len() < 4000 {
        recs.extend(st.take(4000 - recs.len(), 5000));
    }
    st.stop();
    let n4 = recs.iter().filter(|r| r.n_players == 4).count();
    assert!(n4 > 1500 && n4 < 3500, "4P share {n4}/4000");
    for r in &recs {
        let s: f32 = r.pi[..r.n_cells as usize].iter().sum();
        assert!((s - 1.0).abs() < 1e-4);
        if r.n_players == 2 {
            assert!(r.winner_rel == 0 || r.winner_rel == 2 || r.winner_rel == NO_WINNER);
        } else {
            assert!(r.winner_rel < 4 || r.winner_rel == NO_WINNER);
        }
    }
}
