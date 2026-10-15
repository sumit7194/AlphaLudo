use ludo_core::*;

#[test]
fn test_initial_state_4p() {
    let s = GameState::new_4p();
    assert_eq!(s.current_player, 0);
    assert_eq!(s.scores, [0, 0, 0, 0]);
    assert_eq!(s.active_players, [true, true, true, true]);
    assert_eq!(s.is_terminal, false);
    assert_eq!(s.winner, -1);
    for p in 0..4 {
        for t in 0..4 {
            assert_eq!(s.positions[p][t], BASE_POS);
        }
    }
}

#[test]
fn test_base_spawn_requires_six() {
    let mut s = GameState::new_4p();
    
    // Rolling a 5: cannot move from base
    s.set_dice(5);
    let (_moves, count) = s.legal_moves();
    assert_eq!(count, 0);

    // Pass turn to player 1
    s.pass_turn();
    assert_eq!(s.current_player, 1);

    // Player 1 rolls a 6: all 4 base tokens are legal to spawn
    s.set_dice(6);
    let (moves, count) = s.legal_moves();
    assert_eq!(count, 4);
    assert_eq!(&moves[..4], &[0, 1, 2, 3]);

    // Apply move token 0
    let s2 = s.apply_move(0);
    assert_eq!(s2.positions[1][0], 0); // Spawned at pos 0
    // Rolling a 6 grants a bonus turn!
    assert_eq!(s2.current_player, 1);
}

#[test]
fn test_three_sixes_forfeit() {
    let mut s = GameState::new_4p();
    assert_eq!(s.current_player, 0);

    // Roll first 6
    s.set_dice(6);
    assert_eq!(s.consecutive_sixes[0], 1);
    assert_eq!(s.current_player, 0);

    // Roll second 6
    s.set_dice(6);
    assert_eq!(s.consecutive_sixes[0], 2);
    assert_eq!(s.current_player, 0);

    // Roll third 6: Forfeited!
    s.set_dice(6);
    assert_eq!(s.consecutive_sixes[0], 0);
    assert_eq!(s.current_player, 1); // Turn passed to Player 1
    assert_eq!(s.current_dice_roll, 0);
}

#[test]
fn test_capture_and_bonus_turn() {
    let mut s = GameState::new_4p();

    // P0 token 0 is at pos 2 (abs pos 2, unsafe)
    s.positions[0][0] = 2;
    // P3 token 0 is at pos 15 (P3 abs pos is (15 + 13*3) % 52 = 54 % 52 = 2)
    s.positions[3][0] = 15;
    assert_eq!(get_absolute_pos(0, 2), Some(2));
    assert_eq!(get_absolute_pos(3, 15), Some(2));
    assert_eq!(is_safe_pos(2), false);

    // Now P1's turn. P1 starts at abs pos 13.
    // To land on abs pos 2, P1 needs to move to relative pos (2 - 13) % 52 = 41.
    s.current_player = 1;
    s.positions[1][0] = 38; // relative pos 38
    s.set_dice(3); // moves from 38 -> 41 (abs pos = (41 + 13) % 52 = 2)

    let s2 = s.apply_move(0);

    // P1 moved to 41
    assert_eq!(s2.positions[1][0], 41);
    // P0 token was captured -> back to BASE_POS
    assert_eq!(s2.positions[0][0], BASE_POS);
    // P3 token was captured -> back to BASE_POS
    assert_eq!(s2.positions[3][0], BASE_POS);
    // Capture grants bonus turn! (P1 plays again even with roll 3)
    assert_eq!(s2.current_player, 1);
}

#[test]
fn test_blockade_immune_to_capture() {
    let mut s = GameState::new_4p();

    // P0 has 2 tokens at pos 2 (stack of 2 = blockade)
    s.positions[0][0] = 2;
    s.positions[0][1] = 2;

    // P1 lands on pos 41 (abs pos 2)
    s.current_player = 1;
    s.positions[1][0] = 38;
    s.set_dice(3); // lands on 41

    let s2 = s.apply_move(0);

    // P0 tokens are NOT captured because of blockade!
    assert_eq!(s2.positions[0][0], 2);
    assert_eq!(s2.positions[0][1], 2);
    // No bonus turn granted since roll was 3 and no capture
    assert_eq!(s2.current_player, 2);
}

#[test]
fn test_safe_square_immune_to_capture() {
    let mut s = GameState::new_4p();

    // Safe pos 0 on track: P0 pos 0 is abs pos 0
    s.positions[0][0] = 0;
    assert!(is_safe_pos(0));

    // P3 relative pos 39 maps to abs pos (39 + 39) % 52 = 78 % 52 = 26 (safe)
    // P3 relative pos 13 maps to abs pos (13 + 39) % 52 = 52 % 52 = 0
    s.current_player = 3;
    s.positions[3][0] = 9;
    s.set_dice(4); // lands on 13 (abs pos 0)

    let s2 = s.apply_move(0);

    // P0 token is SAFE -> NOT captured!
    assert_eq!(s2.positions[0][0], 0);
    assert_eq!(s2.positions[3][0], 13);
}

#[test]
fn test_exact_home_scoring_and_win() {
    let mut s = GameState::new_4p();
    s.positions[0][0] = 52;
    s.positions[0][1] = HOME_POS;
    s.positions[0][2] = HOME_POS;
    s.positions[0][3] = HOME_POS;
    s.scores[0] = 3;

    // Token 0 is at 52. Moving 4 steps lands exactly on 56 (HOME!)
    s.set_dice(4);
    let s2 = s.apply_move(0);

    assert_eq!(s2.positions[0][0], HOME_POS);
    assert_eq!(s2.scores[0], 4);
    assert_eq!(s2.is_terminal, true);
    assert_eq!(s2.winner, 0);

    // Test overshooting: roll 5 from pos 52 (52 + 5 = 57 > 56, illegal)
    s.set_dice(5);
    let (_, count) = s.legal_moves();
    assert_eq!(count, 0); // Cannot move token 0
}

#[test]
fn test_2player_game_transitions() {
    let mut s = GameState::new_2p();
    assert_eq!(s.current_player, 0);
    assert_eq!(s.active_players, [true, false, true, false]);

    // P0 rolls 1 (no legal moves from base) -> passes turn to P2
    s.set_dice(1);
    let (_, count) = s.legal_moves();
    assert_eq!(count, 0);
    s.pass_turn();
    assert_eq!(s.current_player, 2); // Directly to Player 2!

    // P2 rolls 1 -> passes turn back to P0
    s.set_dice(1);
    s.pass_turn();
    assert_eq!(s.current_player, 0); // Directly back to Player 0!

    // P0 spawns on 6
    s.set_dice(6);
    let s2 = s.apply_move(0);
    assert_eq!(s2.positions[0][0], 0);
    assert_eq!(s2.current_player, 0); // Bonus turn on 6

    // Next turn without bonus passes to P2
    let mut s3 = s2;
    s3.set_dice(3);
    let s4 = s3.apply_move(0);
    assert_eq!(s4.positions[0][0], 3);
    assert_eq!(s4.current_player, 2); // To P2!
}


/// Audit 2026-10-02: a third consecutive six must forfeit the turn to the
/// opponent. The old caller pattern (set_dice -> legal_moves -> pass_turn) passed
/// twice, handing the turn back to the player who rolled three sixes.
#[test]
fn three_six_forfeit_is_not_double_passed() {
    let mut s = GameState::new_2p();
    s.positions[0][0] = 10;
    for _ in 0..2 {
        assert!(s.roll(6));
        let (lm, _) = s.legal_moves();
        s = s.apply_move(lm[0]);
        assert_eq!(s.current_player, 0, "a six grants a bonus turn");
    }
    assert!(!s.roll(6), "third six is a forfeit, not a decision");
    assert_eq!(s.current_player, 2, "turn must pass to the opponent");
    assert_eq!(s.current_dice_roll, 0);
    assert_eq!(s.consecutive_sixes[0], 0, "counter resets after forfeit");
}

/// roll() passes exactly once when the roller has no legal move.
#[test]
fn roll_with_no_legal_move_passes_once() {
    let mut s = GameState::new_2p(); // everyone in base
    assert!(!s.roll(3));
    assert_eq!(s.current_player, 2);
    assert_eq!(s.current_dice_roll, 0);
}

/// 4-player: a forfeit must go to the NEXT seat, not skip it.
#[test]
fn three_six_forfeit_4p_goes_to_next_seat() {
    let mut s = GameState::new_4p();
    s.positions[0][0] = 10;
    for _ in 0..2 {
        assert!(s.roll(6));
        let (lm, _) = s.legal_moves();
        s = s.apply_move(lm[0]);
    }
    assert!(!s.roll(6));
    assert_eq!(s.current_player, 1);
}

/// A 6 rolled with no legal move ends the turn; the next turn starts a fresh
/// six streak (three-sixes forfeit counts sixes within one turn).
#[test]
fn six_with_no_legal_move_resets_streak() {
    let mut s = GameState::new_2p();
    // Player 0: all tokens home except one at relative 55 (needs exactly 1 to finish).
    s.positions[0] = [99, 99, 99, 55];
    s.scores[0] = 3;
    s.current_player = 0;
    assert!(!s.roll(6), "a 6 cannot move the token at 55");
    assert_eq!(s.consecutive_sixes[0], 0, "streak must not carry into the next turn");
    assert_eq!(s.current_player, 2);
}
