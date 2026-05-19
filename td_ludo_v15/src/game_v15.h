// V15 game engine — cell-based API, no token-IDs exposed.
//
// Rules are identical to the legacy `td_ludo_cpp` engine. Differences are
// purely API-shape:
//   * `apply_move_from_cell(state, row, col)` replaces `apply_move(state, token_id)`.
//   * `get_legal_source_cells(state)` replaces `get_legal_moves(state)`.
//   * `set_dice(state, d)` is a new entry that atomically handles the
//     3-consecutive-sixes forfeit (in legacy engine, forfeit was Python-side).
//   * Stripped: MCTS, all spatial encoders (V15 encoding is Python-side).
//   * Stripped: idle_counter / last_moved_token / streak — V15 doesn't use them.
//
// Internal token-IDs (array indices 0..3) still exist inside the GameState
// (a 4-position-per-player array is the natural storage), but the public
// API never returns or accepts them. State inspection helpers (read-only
// numpy views of `player_positions`) are kept for unit tests and the
// encoder, which iterates positions, not IDs.

#ifndef GAME_V15_H
#define GAME_V15_H

#include <array>
#include <cstdint>
#include <utility>
#include <vector>

namespace v15 {

// Constants
const int BOARD_SIZE = 15;
const int NUM_PLAYERS = 4;
const int NUM_TOKENS = 4;
const int PATH_LENGTH = 52;     // for absolute-position arithmetic (4 players × 13-cell offsets)
const int HOME_RUN_LENGTH = 5;
const int HOME_POS = 99;        // sentinel: token has scored
const int BASE_POS = -1;        // sentinel: token still in base

struct GameState {
  // Visualization board (-1 empty, 0-3 player ID). Topmost token wins ties; we
  // do not use this for game logic, only for debug rendering. Encoder ignores it.
  std::array<std::array<int8_t, BOARD_SIZE>, BOARD_SIZE> board;

  // Token positions: player_positions[p][slot] in {BASE_POS, 0..55, HOME_POS}.
  // Slot is an internal storage index — never exposed in the cell-based API.
  // The cell-based API treats slots-at-same-position as interchangeable.
  std::array<std::array<int8_t, NUM_TOKENS>, NUM_PLAYERS> player_positions;

  // Per-player scored-token counts (0..4). Winner is first to reach 4.
  std::array<int8_t, NUM_PLAYERS> scores;

  // active_players[p] == true if player p participates. 2P mode disables P1/P3.
  std::array<bool, NUM_PLAYERS> active_players;

  int8_t current_player;       // 0..3
  int8_t current_dice_roll;    // 1..6 (0 means awaiting a roll)
  bool is_terminal;

  // V15-only: tracks consecutive 6s for the current player. 3 in a row →
  // forfeit (handled atomically in `set_dice`). Reset on non-6 roll, on
  // forfeit, and on turn-pass.
  std::array<int8_t, NUM_PLAYERS> consecutive_sixes;
};

// ─── Construction ─────────────────────────────────────────────────────────
GameState create_initial_state();      // 4-player
GameState create_initial_state_2p();   // P0 vs P2 only

// ─── Dice & forfeit (atomic) ──────────────────────────────────────────────
// Sets the dice for the current player. If this is the 3rd consecutive 6,
// the move is forfeited: turn advances to next active player, dice is reset
// to 0, consecutive_sixes counter resets. Caller checks `current_dice_roll`
// in the returned state — if 0, forfeit happened.
GameState set_dice(const GameState& state, int dice_value);

// Advances the turn to the next active player. Resets dice to 0 and clears
// the current player's consecutive_sixes counter. Use when current player
// has no legal moves to play (e.g., all tokens at base + non-6 dice).
GameState pass_turn(const GameState& state);

// ─── Cell-based API (the only API V15 uses) ───────────────────────────────
// Returns unique source cells (row, col) where the current player has a
// movable token under the current dice. Two tokens stacked on the same cell
// yield ONE entry — they're action-equivalent.
std::vector<std::pair<int, int>> get_legal_source_cells(const GameState& state);

// Applies a move whose source is the given cell. Engine internally picks
// the lowest-slot legal token at that cell (deterministic tiebreak; the
// post-state is identical regardless of slot choice because stacked tokens
// are state-equivalent under apply_move).
//
// Returns the post-move state. If no legal token at (row, col), returns
// the input state unchanged (caller should check or filter via
// get_legal_source_cells first).
GameState apply_move_from_cell(const GameState& state, int row, int col);

// ─── Position queries (no token-IDs) ──────────────────────────────────────
// Returns the list of positions of the current player's tokens. Multiplicity
// is preserved — two tokens stacked at position 25 yield {25, 25} (plus any
// others). Order is implementation-defined and should not be relied upon.
std::vector<int> get_own_positions(const GameState& state);

// Same but for a specific opponent (by player id). Use for multi-opponent
// scenarios; in 2P only one opponent is active.
std::vector<int> get_opp_positions(const GameState& state, int opp_player);

// Convert a path position to its (row, col) cell in player `player`'s POV
// (after the appropriate rotation). Handles BASE (returns a designated home
// cell), main path (0..50), home stretch (51..55), and HOME (99).
//
// For BASE positions, all 4 base slots map to the SAME canonical "home
// counter" cell in player p's POV — `(2, 2)` after rotation. This matches
// V15's single-counter convention: multiple home tokens share one cell.
std::pair<int, int> position_to_cell(int pos, int player);

// ─── Terminal / winner ────────────────────────────────────────────────────
int get_winner(const GameState& state);  // -1 if no winner, else 0..3

// ─── Internal (exposed for tests + encoder iteration only, not for V15
// inference path) ──────────────────────────────────────────────────────────
std::vector<int> _internal_get_legal_token_slots(const GameState& state);
GameState _internal_apply_move_by_slot(const GameState& state, int slot);

} // namespace v15

#endif // GAME_V15_H
