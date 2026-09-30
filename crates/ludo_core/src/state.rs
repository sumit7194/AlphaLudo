//! High-performance, zero-allocation 32-byte GameState struct and rule transitions.

use crate::constants::{
    get_absolute_pos, is_safe_pos, BASE_POS, HOME_POS, NUM_PLAYERS, NUM_TOKENS,
};

/// A complete Ludo board state packed into exactly 32 bytes (fits in a single L1 cache line).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(C)]
pub struct GameState {
    /// Token positions for each player: positions[player][token]
    /// -1: in base, 0..50: main track, 51..55: home stretch, 99: home
    pub positions: [[i8; NUM_TOKENS]; NUM_PLAYERS],
    /// Number of tokens reached home per player (0..4)
    pub scores: [u8; NUM_PLAYERS],
    /// Number of consecutive sixes rolled per player (0..2)
    pub consecutive_sixes: [u8; NUM_PLAYERS],
    /// Which seats are active in the game
    pub active_players: [bool; NUM_PLAYERS],
    /// Current player turn (0..3)
    pub current_player: u8,
    /// Current dice roll (0 = unrolled, 1..6 = rolled)
    pub current_dice_roll: u8,
    /// Whether the game has finished
    pub is_terminal: bool,
    /// Winner seat (0..3), or -1 if game is ongoing
    pub winner: i8,
}

impl Default for GameState {
    fn default() -> Self {
        Self::new_4p()
    }
}

impl GameState {
    /// Creates a fresh 4-player game with all 4 seats active.
    pub fn new_4p() -> Self {
        Self {
            positions: [[BASE_POS; NUM_TOKENS]; NUM_PLAYERS],
            scores: [0; NUM_PLAYERS],
            consecutive_sixes: [0; NUM_PLAYERS],
            active_players: [true; NUM_PLAYERS],
            current_player: 0,
            current_dice_roll: 0,
            is_terminal: false,
            winner: -1,
        }
    }

    /// Creates a fresh 2-player game (Player 0 vs Player 2).
    pub fn new_2p() -> Self {
        let mut s = Self::new_4p();
        s.active_players[1] = false;
        s.active_players[3] = false;
        s
    }

    /// Advances `current_player` to the next active player.
    #[inline(always)]
    pub fn next_player(&self) -> u8 {
        let mut nxt = ((self.current_player as usize) + 1) % NUM_PLAYERS;
        while !self.active_players[nxt] {
            nxt = (nxt + 1) % NUM_PLAYERS;
        }
        nxt as u8
    }

    /// Computes the normalized progress / standings fraction (0.0 to 1.0)
    /// for each of the 4 seats in `pov_player` relative turn order (Me, Next, Opp, Prev).
    #[inline]
    pub fn normalized_standings(&self, pov_player: u8) -> [f32; 4] {
        let mut standings = [0.0f32; 4];
        for rel_idx in 0..4 {
            let p = ((pov_player as usize) + rel_idx) % NUM_PLAYERS;
            let mut total_prog = 0.0f32;
            for t in 0..NUM_TOKENS {
                let pos = self.positions[p][t];
                if pos == HOME_POS {
                    total_prog += 56.0;
                } else if pos == BASE_POS {
                    total_prog += 0.0;
                } else {
                    total_prog += pos as f32;
                }
            }
            standings[rel_idx] = (total_prog / 224.0).clamp(0.0, 1.0);
        }
        standings
    }

    /// Sets the current dice roll (1..6) and evaluates the 3-consecutive-six forfeit.
    /// If a player rolls 3 consecutive sixes, their turn is forfeited and passed to next player.
    #[inline]
    pub fn set_dice(&mut self, dice: u8) {
        debug_assert!((1..=6).contains(&dice));
        let p = self.current_player as usize;

        if dice == 6 {
            self.consecutive_sixes[p] += 1;
            if self.consecutive_sixes[p] >= 3 {
                // 3-six forfeit: reset counter, pass turn, dice=0
                self.consecutive_sixes[p] = 0;
                self.current_player = self.next_player();
                self.current_dice_roll = 0;
                return;
            }
        } else {
            self.consecutive_sixes[p] = 0;
        }

        self.current_dice_roll = dice;
    }

    /// Passes turn to the next active player when no legal moves are available.
    #[inline]
    pub fn pass_turn(&mut self) {
        self.current_player = self.next_player();
        self.current_dice_roll = 0;
    }

    /// Returns the legal moves as a fixed-size array and count without heap allocation.
    /// Returns `([token_0, token_1, ...], count)`.
    #[inline]
    pub fn legal_moves(&self) -> ([u8; 4], usize) {
        let mut moves = [0u8; 4];
        let mut count = 0;

        if self.is_terminal || self.current_dice_roll == 0 {
            return (moves, 0);
        }

        let p = self.current_player as usize;
        let roll = self.current_dice_roll as i8;

        for t in 0..NUM_TOKENS {
            let pos = self.positions[p][t];
            if pos == BASE_POS {
                if roll == 6 {
                    moves[count] = t as u8;
                    count += 1;
                }
            } else if pos == HOME_POS {
                // Already reached home, cannot move
                continue;
            } else {
                let target = pos + roll;
                if target <= 56 {
                    moves[count] = t as u8;
                    count += 1;
                }
            }
        }

        (moves, count)
    }

    /// Applies a move by token index (0..3) and returns the resulting state.
    #[inline]
    pub fn apply_move(&self, token_idx: u8) -> Self {
        let mut next = *self;
        let p = self.current_player as usize;
        let roll = self.current_dice_roll as i8;
        let slot = token_idx as usize;

        let cur_pos = next.positions[p][slot];
        let new_pos = if cur_pos == BASE_POS {
            0
        } else {
            cur_pos + roll
        };

        next.positions[p][slot] = new_pos;

        // Check if token reached Home (position 56)
        if new_pos == 56 {
            next.positions[p][slot] = HOME_POS;
            next.scores[p] += 1;
            if next.scores[p] == 4 {
                next.is_terminal = true;
                next.winner = p as i8;
                next.current_dice_roll = 0;
                return next;
            }
        }

        let mut bonus_turn = (roll == 6) || (new_pos == 56);

        // Capture detection on main track (positions 0..50)
        if new_pos <= 50 {
            if let Some(abs_pos) = get_absolute_pos(p as u8, new_pos) {
                if !is_safe_pos(abs_pos) {
                    for other_p in 0..NUM_PLAYERS {
                        if other_p == p || !next.active_players[other_p] {
                            continue;
                        }

                        // Count opponent tokens on this cell
                        let mut stack_count = 0;
                        for t in 0..NUM_TOKENS {
                            let op = next.positions[other_p][t];
                            if op != BASE_POS && op != HOME_POS && op <= 50 {
                                if get_absolute_pos(other_p as u8, op) == Some(abs_pos) {
                                    stack_count += 1;
                                }
                            }
                        }

                        // Single token on unsafe square is captured!
                        if stack_count == 1 {
                            for t in 0..NUM_TOKENS {
                                let op = next.positions[other_p][t];
                                if op != BASE_POS && op != HOME_POS && op <= 50 {
                                    if get_absolute_pos(other_p as u8, op) == Some(abs_pos) {
                                        next.positions[other_p][t] = BASE_POS;
                                        bonus_turn = true;
                                    }
                                }
                            }
                        }
                        // stack_count >= 2 is a blockade (immune to capture)
                    }
                }
            }
        }

        // Pass turn if no bonus granted
        if !bonus_turn {
            next.current_player = next.next_player();
        }

        next.current_dice_roll = 0;
        next
    }
}
