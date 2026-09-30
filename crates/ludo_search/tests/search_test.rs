use ludo_core::GameState;
use ludo_search::{AggressiveExpectimaxBot, Depth2ExpectimaxBot, ExpectimaxBot, MaxNMCTSBot, LudoBot};
use rand::Rng;

#[test]
fn test_bots_select_legal_move() {
    let mut s = GameState::new_4p();
    s.positions[0][0] = 5;
    s.positions[0][1] = 10;
    s.set_dice(4);

    let (legal, count) = s.legal_moves();
    assert_eq!(count, 2);

    let bot1 = ExpectimaxBot::new(Some(0));
    let a1 = bot1.select_move(&s);
    assert!(legal[..count].contains(&a1));

    let bot2 = AggressiveExpectimaxBot::new(Some(0));
    let a2 = bot2.select_move(&s);
    assert!(legal[..count].contains(&a2));

    let bot3 = Depth2ExpectimaxBot::new(Some(0));
    let a3 = bot3.select_move(&s);
    assert!(legal[..count].contains(&a3));

    let bot4 = MaxNMCTSBot::new(Some(0), 40);
    let a4 = bot4.select_move(&s);
    assert!(legal[..count].contains(&a4));
}

#[test]
fn test_full_game_simulation() {
    let mut rng = rand::thread_rng();
    let mut s = GameState::new_4p();

    let bots: [Box<dyn LudoBot>; 4] = [
        Box::new(Depth2ExpectimaxBot::new(Some(0))),
        Box::new(AggressiveExpectimaxBot::new(Some(1))),
        Box::new(Depth2ExpectimaxBot::new(Some(2))),
        Box::new(AggressiveExpectimaxBot::new(Some(3))),
    ];

    let mut moves = 0;
    while !s.is_terminal && moves < 600 {
        let cp = s.current_player as usize;
        let d = rng.gen_range(1..=6);
        s.set_dice(d);

        let (_, count) = s.legal_moves();
        if count == 0 {
            s.pass_turn();
            continue;
        }

        let a = bots[cp].select_move(&s);
        s = s.apply_move(a);
        moves += 1;
    }

    // Verify game progressed significantly
    assert!(moves > 20);
    println!("Game completed or progressed to {} moves (Terminal: {}, Winner: {})", moves, s.is_terminal, s.winner);
}
