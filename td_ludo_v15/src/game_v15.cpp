// V15 game engine implementation.
//
// Rule logic is copied verbatim from `td_ludo/src/game.cpp` (commit 1ff249f
// onwards — the post-BASE_COORDS-fix version). The cell-based API + atomic
// 3-six forfeit are new.
//
// Layout convention:
//   * PATH_COORDS_P0[51] — P0's main-track cells, indices 0..50.
//   * HOME_RUN_P0[5]     — P0's home stretch, indices 0..4 (positions 51..55).
//   * HOME_COORD_P0      — center cell (7, 6), P0's HOME.
//   * BASE_COORDS[p][slot] — POST-rotation cell for player p's slot in base.
//     Critical: assigned such that after k=current_player CCW-rotations are
//     applied (in get_board_coords), every player sees slot 0 → (2,2),
//     slot 1 → (2,3), etc. This was the encoder-symmetry fix in V13.5.

#include "game_v15.h"
#include <algorithm>
#include <cstring>

namespace v15 {

// ─────────────────────────────────────────────────────────────────────────
// Static tables — verbatim from legacy engine
// ─────────────────────────────────────────────────────────────────────────

// P0's 51-cell main track. Position i → (row, col).
static const int8_t PATH_COORDS_P0[51][2] = {
    {6, 1},  {6, 2},  {6, 3},  {6, 4},  {6, 5},           // 0-4
    {5, 6},  {4, 6},  {3, 6},  {2, 6},  {1, 6},  {0, 6},  // 5-10
    {0, 7},  {0, 8},                                      // 11-12
    {1, 8},  {2, 8},  {3, 8},  {4, 8},  {5, 8},           // 13-17
    {6, 9},  {6, 10}, {6, 11}, {6, 12}, {6, 13}, {6, 14}, // 18-23
    {7, 14}, {8, 14},                                     // 24-25
    {8, 13}, {8, 12}, {8, 11}, {8, 10}, {8, 9},           // 26-30
    {9, 8},  {10, 8}, {11, 8}, {12, 8}, {13, 8}, {14, 8}, // 31-36
    {14, 7}, {14, 6},                                     // 37-38
    {13, 6}, {12, 6}, {11, 6}, {10, 6}, {9, 6},           // 39-43
    {8, 5},  {8, 4},  {8, 3},  {8, 2},  {8, 1},  {8, 0},  // 44-49
    {7, 0}                                                // 50 (end of main)
};

// P0's home stretch (positions 51..55) and HOME (99).
static const int8_t HOME_RUN_P0[5][2] = {{7, 1}, {7, 2}, {7, 3}, {7, 4}, {7, 5}};
static const int8_t HOME_COORD_P0[2] = {7, 6};

// Base cells per (player, slot), pre-rotated so the same rotation applied to
// path cells maps slot s to the canonical home cell across all players.
// Verbatim from `td_ludo/src/game.cpp:41-46` (post-fix).
static const int8_t BASE_COORDS[4][4][2] = {
    {{2, 2},  {2, 3},  {3, 2},  {3, 3}},      // P0 (k=0)
    {{2, 12}, {3, 12}, {2, 11}, {3, 11}},     // P1 (k=1)
    {{12, 12},{12, 11},{11, 12},{11, 11}},    // P2 (k=2)
    {{12, 2}, {11, 2}, {12, 3}, {11, 3}}      // P3 (k=3)
};

// Safe-square indices on the 52-cell loop.
static const int SAFE_INDICES[] = {0, 8, 13, 21, 26, 34, 39, 47};

// ─────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────

static bool is_safe_pos(int abs_pos) {
  for (int s : SAFE_INDICES) {
    if (s == abs_pos) return true;
  }
  return false;
}

static int get_absolute_pos(int player, int relative_pos) {
  if (relative_pos > 50) return -1;  // home stretch isn't on the loop
  return (relative_pos + 13 * player) % PATH_LENGTH;
}

// Returns the (row, col) cell for a path position in player `player`'s POV.
// Replicated from legacy `get_board_coords` but with cleaner return type.
static void board_coords(int player, int pos, int& r, int& c, int slot_hint) {
  int local_r, local_c;

  if (pos == BASE_POS) {
    // Legacy used slot_hint to differentiate token slots in the home base.
    // V15 always returns the canonical "home counter" cell (slot 0) so the
    // public API is slot-agnostic. The slot_hint param is retained for the
    // internal update_board call that paints slot-specific home cells for
    // the debug board rendering.
    if (slot_hint >= 0 && slot_hint < NUM_TOKENS) {
      r = BASE_COORDS[player][slot_hint][0];
      c = BASE_COORDS[player][slot_hint][1];
    } else {
      r = BASE_COORDS[player][0][0];
      c = BASE_COORDS[player][0][1];
    }
    return;
  } else if (pos == HOME_POS) {
    local_r = HOME_COORD_P0[0];
    local_c = HOME_COORD_P0[1];
  } else if (pos > 50) {  // home stretch
    int idx = pos - 51;
    if (idx < HOME_RUN_LENGTH) {
      local_r = HOME_RUN_P0[idx][0];
      local_c = HOME_RUN_P0[idx][1];
    } else {
      local_r = HOME_COORD_P0[0];
      local_c = HOME_COORD_P0[1];
    }
  } else {  // main track
    local_r = PATH_COORDS_P0[pos][0];
    local_c = PATH_COORDS_P0[pos][1];
  }

  // Rotate k=player times CCW around (7, 7). Mirrors legacy convention.
  int tr = local_r, tc = local_c;
  for (int i = 0; i < player; ++i) {
    int new_r = tc;
    int new_c = 14 - tr;
    tr = new_r;
    tc = new_c;
  }
  r = tr;
  c = tc;
}

static void update_board(GameState& state) {
  for (int i = 0; i < BOARD_SIZE; ++i)
    for (int j = 0; j < BOARD_SIZE; ++j)
      state.board[i][j] = -1;

  for (int p = 0; p < NUM_PLAYERS; ++p) {
    for (int t = 0; t < NUM_TOKENS; ++t) {
      int pos = state.player_positions[p][t];
      int r, c;
      // For BASE, paint slot-specific cells (so visualization shows 4 distinct
      // base spots). For non-BASE, slot_hint is unused.
      board_coords(p, pos, r, c, t);
      if (r >= 0 && r < BOARD_SIZE && c >= 0 && c < BOARD_SIZE) {
        state.board[r][c] = p;
      }
    }
  }
}

// ─────────────────────────────────────────────────────────────────────────
// Initial state
// ─────────────────────────────────────────────────────────────────────────

GameState create_initial_state() {
  GameState s;
  std::memset(&s, 0, sizeof(GameState));
  for (int p = 0; p < NUM_PLAYERS; ++p) {
    for (int t = 0; t < NUM_TOKENS; ++t) {
      s.player_positions[p][t] = BASE_POS;
    }
    s.scores[p] = 0;
    s.active_players[p] = true;
    s.consecutive_sixes[p] = 0;
  }
  s.current_player = 0;
  s.current_dice_roll = 0;
  s.is_terminal = false;
  update_board(s);
  return s;
}

GameState create_initial_state_2p() {
  GameState s = create_initial_state();
  s.active_players[1] = false;
  s.active_players[3] = false;
  update_board(s);
  return s;
}

// ─────────────────────────────────────────────────────────────────────────
// Dice + 3-six forfeit (atomic)
// ─────────────────────────────────────────────────────────────────────────

GameState set_dice(const GameState& state, int dice_value) {
  GameState next = state;
  int p = next.current_player;

  if (dice_value == 6) {
    next.consecutive_sixes[p] += 1;
    if (next.consecutive_sixes[p] >= 3) {
      // Forfeit: reset counter, pass turn, dice=0.
      next.consecutive_sixes[p] = 0;
      int nxt = (p + 1) % NUM_PLAYERS;
      while (!next.active_players[nxt]) nxt = (nxt + 1) % NUM_PLAYERS;
      next.current_player = (int8_t)nxt;
      next.current_dice_roll = 0;
      return next;
    }
  } else {
    next.consecutive_sixes[p] = 0;
  }
  next.current_dice_roll = (int8_t)dice_value;
  return next;
}

// ─────────────────────────────────────────────────────────────────────────
// Pass turn (no legal moves to play)
// ─────────────────────────────────────────────────────────────────────────

GameState pass_turn(const GameState& state) {
  // Match legacy semantics: consecutive_sixes is NOT reset here. It is
  // updated only by set_dice (incremented on 6, reset on non-6, reset on
  // forfeit). The "no legal moves" path leaves the counter untouched.
  GameState next = state;
  int p = next.current_player;
  int nxt = (p + 1) % NUM_PLAYERS;
  while (!next.active_players[nxt]) nxt = (nxt + 1) % NUM_PLAYERS;
  next.current_player = (int8_t)nxt;
  next.current_dice_roll = 0;
  return next;
}

// ─────────────────────────────────────────────────────────────────────────
// Internal: legal slot enumeration + apply by slot
// ─────────────────────────────────────────────────────────────────────────

std::vector<int> _internal_get_legal_token_slots(const GameState& state) {
  std::vector<int> moves;
  if (state.is_terminal) return moves;
  int roll = state.current_dice_roll;
  if (roll == 0) return moves;
  int p = state.current_player;

  for (int t = 0; t < NUM_TOKENS; ++t) {
    int pos = state.player_positions[p][t];
    if (pos == BASE_POS) {
      if (roll == 6) moves.push_back(t);
    } else if (pos == HOME_POS) {
      continue;
    } else {
      int target = pos + roll;
      if (target <= 56) moves.push_back(t);
    }
  }
  return moves;
}

GameState _internal_apply_move_by_slot(const GameState& state, int slot) {
  GameState next = state;
  int p = state.current_player;
  int roll = state.current_dice_roll;
  int cur_pos = next.player_positions[p][slot];

  if (cur_pos == BASE_POS) {
    next.player_positions[p][slot] = 0;
  } else {
    next.player_positions[p][slot] += roll;
  }
  int new_pos = next.player_positions[p][slot];

  // Reach home?
  if (new_pos == 56) {
    next.player_positions[p][slot] = HOME_POS;
    next.scores[p]++;
    if (next.scores[p] == 4) {
      next.is_terminal = true;
      update_board(next);
      return next;
    }
  }

  bool bonus_turn = (roll == 6) || (new_pos == 56);

  // Capture detection — only on main track + non-safe + opponent has exactly 1.
  if (new_pos <= 50) {
    int abs_pos = get_absolute_pos(p, new_pos);
    if (!is_safe_pos(abs_pos)) {
      for (int other_p = 0; other_p < NUM_PLAYERS; ++other_p) {
        if (other_p == p || !next.active_players[other_p]) continue;
        int stack_count = 0;
        for (int t = 0; t < NUM_TOKENS; ++t) {
          int op = next.player_positions[other_p][t];
          if (op != BASE_POS && op != HOME_POS && op <= 50) {
            if (get_absolute_pos(other_p, op) == abs_pos) stack_count++;
          }
        }
        if (stack_count == 1) {
          for (int t = 0; t < NUM_TOKENS; ++t) {
            int op = next.player_positions[other_p][t];
            if (op != BASE_POS && op != HOME_POS && op <= 50) {
              if (get_absolute_pos(other_p, op) == abs_pos) {
                next.player_positions[other_p][t] = BASE_POS;
                bonus_turn = true;
              }
            }
          }
        }
        // stack_count >= 2 → blockade → no capture
      }
    }
  }

  update_board(next);

  if (!bonus_turn) {
    next.current_player = (int8_t)((p + 1) % NUM_PLAYERS);
    while (!next.active_players[next.current_player]) {
      next.current_player = (int8_t)((next.current_player + 1) % NUM_PLAYERS);
    }
    // Note: consecutive_sixes is NOT reset on turn-pass. set_dice manages
    // it (increment on 6, reset on non-6 or on 3-six forfeit). Reset on
    // turn-pass would cause divergence from legacy engine — see
    // tests/test_engine_parity.py for the regression case.
  }

  next.current_dice_roll = 0;
  return next;
}

// ─────────────────────────────────────────────────────────────────────────
// Public cell-based API
// ─────────────────────────────────────────────────────────────────────────

std::vector<std::pair<int, int>> get_legal_source_cells(const GameState& state) {
  std::vector<std::pair<int, int>> cells;
  if (state.is_terminal || state.current_dice_roll == 0) return cells;
  int p = state.current_player;
  auto legal_slots = _internal_get_legal_token_slots(state);

  for (int slot : legal_slots) {
    int pos = state.player_positions[p][slot];
    int r, c;
    // For BASE positions: V15 single-counter convention — all home tokens
    // map to slot-0's cell (canonical home counter).
    if (pos == BASE_POS) {
      board_coords(p, BASE_POS, r, c, /*slot_hint=*/0);
    } else {
      board_coords(p, pos, r, c, /*slot_hint=*/0);
    }
    std::pair<int, int> cell{r, c};
    if (std::find(cells.begin(), cells.end(), cell) == cells.end()) {
      cells.push_back(cell);
    }
  }
  return cells;
}

GameState apply_move_from_cell(const GameState& state, int row, int col) {
  if (state.is_terminal || state.current_dice_roll == 0) return state;
  int p = state.current_player;
  auto legal_slots = _internal_get_legal_token_slots(state);

  // Find the lowest-indexed legal slot whose cell matches (row, col).
  int chosen_slot = -1;
  for (int slot : legal_slots) {
    int pos = state.player_positions[p][slot];
    int r, c;
    if (pos == BASE_POS) {
      board_coords(p, BASE_POS, r, c, /*slot_hint=*/0);
    } else {
      board_coords(p, pos, r, c, /*slot_hint=*/0);
    }
    if (r == row && c == col) {
      chosen_slot = slot;
      break;  // lowest-slot tiebreak (legal_slots is in slot order)
    }
  }
  if (chosen_slot < 0) return state;  // no legal source at this cell
  return _internal_apply_move_by_slot(state, chosen_slot);
}

// ─────────────────────────────────────────────────────────────────────────
// Position queries
// ─────────────────────────────────────────────────────────────────────────

std::vector<int> get_own_positions(const GameState& state) {
  std::vector<int> out;
  int p = state.current_player;
  for (int t = 0; t < NUM_TOKENS; ++t) {
    out.push_back((int)state.player_positions[p][t]);
  }
  return out;
}

std::vector<int> get_opp_positions(const GameState& state, int opp_player) {
  std::vector<int> out;
  if (opp_player < 0 || opp_player >= NUM_PLAYERS) return out;
  for (int t = 0; t < NUM_TOKENS; ++t) {
    out.push_back((int)state.player_positions[opp_player][t]);
  }
  return out;
}

std::pair<int, int> position_to_cell(int pos, int player) {
  int r, c;
  board_coords(player, pos, r, c, /*slot_hint=*/0);
  return {r, c};
}

// ─────────────────────────────────────────────────────────────────────────
// Winner
// ─────────────────────────────────────────────────────────────────────────

int get_winner(const GameState& state) {
  if (!state.is_terminal) return -1;
  for (int p = 0; p < NUM_PLAYERS; ++p) {
    if (state.scores[p] == 4) return p;
  }
  return -1;
}

} // namespace v15
