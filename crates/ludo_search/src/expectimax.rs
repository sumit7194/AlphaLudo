//! High-speed Expectimax search algorithms (Depth-1, Aggressive, and True 4P Depth-2).

use ludo_core::GameState;
use crate::evaluator::{score_aggressive, score_position};

pub trait LudoBot: Send + Sync {
    fn name(&self) -> &'static str;
    fn select_move(&self, state: &GameState) -> u8;
}

/// Standard Depth-1 Expectimax Bot.
/// Simulates my move, then for each opponent dice roll (1..6) simulates the next
/// player's best greedy counter-move, and averages the resulting score.
pub struct ExpectimaxBot {
    pub player_id: Option<u8>,
}

impl ExpectimaxBot {
    pub fn new(player_id: Option<u8>) -> Self {
        Self { player_id }
    }
}

impl LudoBot for ExpectimaxBot {
    fn name(&self) -> &'static str {
        "Expectimax"
    }

    fn select_move(&self, state: &GameState) -> u8 {
        let (legal, count) = state.legal_moves();
        if count == 0 {
            return 0;
        }
        if count == 1 {
            return legal[0];
        }

        let me = self.player_id.unwrap_or(state.current_player);
        let mut best_action = legal[0];
        let mut best_expected = -f32::INFINITY;

        for &a in &legal[..count] {
            let after_mine = state.apply_move(a);

            if after_mine.is_terminal {
                let val = if after_mine.winner == me as i8 { 1e9 } else { 0.0 };
                if val > best_expected {
                    best_expected = val;
                    best_action = a;
                }
                continue;
            }

            let mut expected = 0.0;
            for d in 1..=6 {
                let mut sim = after_mine;
                sim.set_dice(d);
                let cp = sim.current_player;

                if cp == me {
                    // Bonus turn for me!
                    expected += score_position(me, &sim) / 6.0;
                    continue;
                }

                let (opp_legal, opp_count) = sim.legal_moves();
                if opp_count == 0 {
                    sim.pass_turn();
                    expected += score_position(me, &sim) / 6.0;
                    continue;
                }

                // Opponent picks greedy best move
                let mut best_opp_score = -f32::INFINITY;
                let mut best_oa = opp_legal[0];
                for &oa in &opp_legal[..opp_count] {
                    let after_opp = sim.apply_move(oa);
                    let s_opp = score_position(cp, &after_opp);
                    if s_opp > best_opp_score {
                        best_opp_score = s_opp;
                        best_oa = oa;
                    }
                }

                let after_opp = sim.apply_move(best_oa);
                expected += score_position(me, &after_opp) / 6.0;
            }

            if expected > best_expected {
                best_expected = expected;
                best_action = a;
            }
        }

        best_action
    }
}

/// Aggressive Expectimax Bot (#1 Elo Champion among single-ply bots).
/// Uses aggressive scoring to maximize knockouts and opponent pressure.
pub struct AggressiveExpectimaxBot {
    pub player_id: Option<u8>,
}

impl AggressiveExpectimaxBot {
    pub fn new(player_id: Option<u8>) -> Self {
        Self { player_id }
    }
}

impl LudoBot for AggressiveExpectimaxBot {
    fn name(&self) -> &'static str {
        "AggressiveExpectimax"
    }

    fn select_move(&self, state: &GameState) -> u8 {
        let (legal, count) = state.legal_moves();
        if count == 0 {
            return 0;
        }
        if count == 1 {
            return legal[0];
        }

        let me = self.player_id.unwrap_or(state.current_player);
        let mut best_action = legal[0];
        let mut best_expected = -f32::INFINITY;

        for &a in &legal[..count] {
            let after_mine = state.apply_move(a);

            if after_mine.is_terminal {
                let val = if after_mine.winner == me as i8 { 1e9 } else { 0.0 };
                if val > best_expected {
                    best_expected = val;
                    best_action = a;
                }
                continue;
            }

            let mut expected = 0.0;
            for d in 1..=6 {
                let mut sim = after_mine;
                sim.set_dice(d);
                let cp = sim.current_player;

                if cp == me {
                    expected += score_aggressive(me, &sim) / 6.0;
                    continue;
                }

                let (opp_legal, opp_count) = sim.legal_moves();
                if opp_count == 0 {
                    sim.pass_turn();
                    expected += score_aggressive(me, &sim) / 6.0;
                    continue;
                }

                // Opponent picks greedy response
                let mut best_opp_score = -f32::INFINITY;
                let mut best_oa = opp_legal[0];
                for &oa in &opp_legal[..opp_count] {
                    let after_opp = sim.apply_move(oa);
                    let s_opp = score_aggressive(cp, &after_opp);
                    if s_opp > best_opp_score {
                        best_opp_score = s_opp;
                        best_oa = oa;
                    }
                }

                let after_opp = sim.apply_move(best_oa);
                expected += score_aggressive(me, &after_opp) / 6.0;
            }

            if expected > best_expected {
                best_expected = expected;
                best_action = a;
            }
        }

        best_action
    }
}

/// True 4-Player Depth-2 Expectimax Bot.
/// Looks 2 full plies ahead: evaluates (my action -> opp_1 roll -> opp_1 move -> opp_2 roll -> opp_2 move).
/// Evaluates up to ~2,304 leaf paths per turn in microseconds.
pub struct Depth2ExpectimaxBot {
    pub player_id: Option<u8>,
}

impl Depth2ExpectimaxBot {
    pub fn new(player_id: Option<u8>) -> Self {
        Self { player_id }
    }
}

impl LudoBot for Depth2ExpectimaxBot {
    fn name(&self) -> &'static str {
        "Depth2Expectimax"
    }

    fn select_move(&self, state: &GameState) -> u8 {
        let (legal, count) = state.legal_moves();
        if count == 0 {
            return 0;
        }
        if count == 1 {
            return legal[0];
        }

        let me = self.player_id.unwrap_or(state.current_player);
        let mut best_action = legal[0];
        let mut best_q = -f32::INFINITY;

        for &a in &legal[..count] {
            let after_mine = state.apply_move(a);

            if after_mine.is_terminal {
                let val = if after_mine.winner == me as i8 { 1e9 } else { 0.0 };
                if val > best_q {
                    best_q = val;
                    best_action = a;
                }
                continue;
            }

            // Outer ply: Expectation over dice d1 (1..6)
            let mut outer_expected = 0.0;
            for d1 in 1..=6 {
                let mut sim1 = after_mine;
                sim1.set_dice(d1);
                let cp1 = sim1.current_player;

                if sim1.is_terminal {
                    outer_expected += (if sim1.winner == me as i8 { 1e9 } else { 0.0 }) / 6.0;
                    continue;
                }

                let (legal1, count1) = sim1.legal_moves();
                let after_opp1 = if count1 == 0 {
                    sim1.pass_turn();
                    sim1
                } else {
                    let mut best_score1 = -f32::INFINITY;
                    let mut best_a1 = legal1[0];
                    for &oa1 in &legal1[..count1] {
                        let cand = sim1.apply_move(oa1);
                        let s1 = score_position(cp1, &cand);
                        if s1 > best_score1 {
                            best_score1 = s1;
                            best_a1 = oa1;
                        }
                    }
                    sim1.apply_move(best_a1)
                };

                if after_opp1.is_terminal {
                    outer_expected += (if after_opp1.winner == me as i8 { 1e9 } else { 0.0 }) / 6.0;
                    continue;
                }

                // Inner ply: Expectation over dice d2 (1..6)
                let mut inner_expected = 0.0;
                for d2 in 1..=6 {
                    let mut sim2 = after_opp1;
                    sim2.set_dice(d2);
                    let cp2 = sim2.current_player;

                    if sim2.is_terminal {
                        inner_expected += (if sim2.winner == me as i8 { 1e9 } else { 0.0 }) / 6.0;
                        continue;
                    }

                    let (legal2, count2) = sim2.legal_moves();
                    let after_opp2 = if count2 == 0 {
                        sim2.pass_turn();
                        sim2
                    } else {
                        let mut best_score2 = -f32::INFINITY;
                        let mut best_a2 = legal2[0];
                        for &oa2 in &legal2[..count2] {
                            let cand = sim2.apply_move(oa2);
                            let s2 = score_position(cp2, &cand);
                            if s2 > best_score2 {
                                best_score2 = s2;
                                best_a2 = oa2;
                            }
                        }
                        sim2.apply_move(best_a2)
                    };

                    inner_expected += score_position(me, &after_opp2) / 6.0;
                }

                outer_expected += inner_expected / 6.0;
            }

            if outer_expected > best_q {
                best_q = outer_expected;
                best_action = a;
            }
        }

        best_action
    }
}
