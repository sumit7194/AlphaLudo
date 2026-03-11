#include "game.h"
#include <algorithm>
#include <cstring>
#include <iostream>

// Lookup tables for board coordinates (P0 view)
// We'll define the 52-square path for Player 0.
// (Row, Col) pairs.
const int8_t PATH_COORDS_P0[51][2] = {
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
    {7, 0} // 50 (End of main track)
};

// Home run coordinates for P0
const int8_t HOME_RUN_P0[5][2] = {{7, 1}, {7, 2}, {7, 3}, {7, 4}, {7, 5}};

const int8_t HOME_COORD_P0[2] = {7, 6};

const int8_t BASE_COORDS[4][4][2] = {
    {{2, 2}, {2, 3}, {3, 2}, {3, 3}},         // P0 (Top Left) - Red
    {{2, 11}, {2, 12}, {3, 11}, {3, 12}},     // P1 (Top Right) - Green
    {{11, 11}, {11, 12}, {12, 11}, {12, 12}}, // P2 (Bottom Right) - Yellow
    {{11, 2}, {11, 3}, {12, 2}, {12, 3}}      // P3 (Bottom Left) - Blue
};

// Safe squares (indices on 52-track, P0 view)
const int SAFE_INDICES[] = {0, 8, 13, 21, 26, 34, 39, 47};

bool is_safe(int abs_pos) {
  for (int s : SAFE_INDICES) {
    if (s == abs_pos)
      return true;
  }
  return false;
}

// Helper to get board coordinates
// Helper to get coordinates
void get_board_coords(int player, int pos, int &r, int &c, int token_index) {
  int local_r, local_c;

  if (pos == BASE_POS) {
    // Use token index to map to specific base spot
    if (token_index >= 0 && token_index < NUM_TOKENS) {
      r = BASE_COORDS[player][token_index][0];
      c = BASE_COORDS[player][token_index][1];
      return;
    }
    r = -1;
    c = -1;
    return;
  } else if (pos == HOME_POS) {
    // Map to home center? Or specific home spot?
    // Let's use the home run end.
    local_r = HOME_COORD_P0[0];
    local_c = HOME_COORD_P0[1];
  } else if (pos > 50) { // Home run (positions 51-55)
    int idx = pos - 51;
    if (idx < 5) {
      local_r = HOME_RUN_P0[idx][0];
      local_c = HOME_RUN_P0[idx][1];
    } else {
      local_r = HOME_COORD_P0[0];
      local_c = HOME_COORD_P0[1];
    }
  } else { // Main track
    local_r = PATH_COORDS_P0[pos][0];
    local_c = PATH_COORDS_P0[pos][1];
  }

  // Rotate based on player
  // P0: 0 deg, P1: 90 deg (Clockwise), P2: 180, P3: 270
  // Rotate (r, c) around (7, 7)
  // Actually, my manual rotation formula earlier was for 0->1.
  // Let's apply N rotations.

  int temp_r = local_r;
  int temp_c = local_c;

  for (int i = 0; i < player; ++i) {
    // Rotate 90 deg CW: (r, c) -> (c, 14-r)
    int new_r = temp_c;
    int new_c = 14 - temp_r;
    temp_r = new_r;
    temp_c = new_c;
  }
  r = temp_r;
  c = temp_c;
}

void update_board(GameState &state) {
  // Clear board
  for (int i = 0; i < BOARD_SIZE; ++i)
    for (int j = 0; j < BOARD_SIZE; ++j)
      state.board[i][j] = -1;

  // Place tokens
  for (int p = 0; p < NUM_PLAYERS; ++p) {
    for (int t = 0; t < NUM_TOKENS; ++t) {
      int pos = state.player_positions[p][t];
      int r, c;
      if (pos == BASE_POS) {
        // Use specific base coordinates
        r = BASE_COORDS[p][t][0];
        c = BASE_COORDS[p][t][1];
      } else {
        get_board_coords(p, pos, r, c);
      }

      if (r >= 0 && r < BOARD_SIZE && c >= 0 && c < BOARD_SIZE) {
        state.board[r][c] =
            p; // Overwrite if multiple (stacking not visualized in 2D grid)
      }
    }
  }
}

GameState create_initial_state() {
  GameState state;
  std::memset(&state, 0, sizeof(GameState));

  for (int p = 0; p < NUM_PLAYERS; ++p) {
    for (int t = 0; t < NUM_TOKENS; ++t) {
      state.player_positions[p][t] = BASE_POS;
      state.prev_player_positions[p][t] = BASE_POS; // Init history
    }
    state.scores[p] = 0;
  }
  state.current_player = 0;
  state.current_dice_roll = 0; // Waiting for roll
  state.is_terminal = false;

  // Default: all 4 players active
  for (int p = 0; p < NUM_PLAYERS; ++p)
    state.active_players[p] = true;

  update_board(state);
  return state;
}

GameState create_initial_state_2p() {
  GameState state = create_initial_state();
  // Only P0 and P2 are active (diagonal opposite)
  state.active_players[1] = false;
  state.active_players[3] = false;
  // P1/P3 tokens stay at BASE_POS (already set by create_initial_state)
  update_board(state);
  return state;
}

int get_absolute_pos(int player, int relative_pos) {
  if (relative_pos > 50)
    return -1; // Not on main track (0-50)
  return (relative_pos + 13 * player) % 52;
}

bool is_safe_pos(int abs_pos) {
  // Safe Indices: 0, 8, 13, 21, 26, 34, 39, 47
  static const int safe[] = {0, 8, 13, 21, 26, 34, 39, 47};
  for (int s : safe) {
    if (abs_pos == s)
      return true;
  }
  return false;
}

std::vector<int> get_legal_moves(const GameState &state) {
  std::vector<int> moves;
  if (state.is_terminal)
    return moves;

  int roll = state.current_dice_roll;
  if (roll == 0)
    return moves; // No moves if no roll
  int p = state.current_player;

  for (int t = 0; t < NUM_TOKENS; ++t) {
    int pos = state.player_positions[p][t];

    if (pos == BASE_POS) {
      // Standard Ludo: Need 6 to spawn
      if (roll == 6) {
        // Determine target spawn position
        // Standard spawn is pos 0 (relative)
        // Blockade Check Removed: Can spawn even if opponents are stacked on 0
        moves.push_back(t);
      }
    } else if (pos == HOME_POS) {
      continue;
    } else {
      // Check if move is valid
      int target = pos + roll;
      if (target <= 56) { // 56 is Home
        // Blockades do not block movement in this variation.
        // Just check basic bounds (already done).
        moves.push_back(t);
      }
    }
  }
  return moves;
}

GameState apply_move(const GameState &state, int token_index) {
  GameState next_state = state;

  // [Ghost Trail] Save history
  // We copy the *current* positions to *prev* before modifying
  // Optimized: memcpy or just copy
  std::memcpy(&next_state.prev_player_positions[0][0],
              &state.player_positions[0][0],
              NUM_PLAYERS * NUM_TOKENS * sizeof(int8_t));

  int p = state.current_player;
  int roll = state.current_dice_roll;
  int current_pos = next_state.player_positions[p][token_index];

  // Move logic
  // Move logic
  if (current_pos == BASE_POS) {
    // Standard Ludo: Spawn at Start (position 0)
    next_state.player_positions[p][token_index] = 0;
  } else {
    next_state.player_positions[p][token_index] += roll;
  }

  int new_pos = next_state.player_positions[p][token_index];

  // Check for Home
  if (new_pos == 56) {
    next_state.player_positions[p][token_index] = HOME_POS;
    next_state.scores[p]++;
    if (next_state.scores[p] == 4) {
      next_state.is_terminal = true;
      update_board(next_state);
      return next_state;
    }
    // DO NOT RETURN EARLY HERE!
    // We need to fall through to set bonus_turn = true (implicitly or
    // explicitly) and reset dice roll.
  }

  bool bonus_turn = (roll == 6) || (new_pos == 56); // 6 or Home gives bonus

  // Standard rules: 6 gives bonus. Reaching home usually gives bonus.
  // Cutting gives bonus.

  // Check for Cut
  // Check for Cut
  if (new_pos <= 50) { // Only on main track
    int abs_pos = get_absolute_pos(p, new_pos);
    if (!is_safe_pos(abs_pos)) {
      for (int other_p = 0; other_p < NUM_PLAYERS; ++other_p) {
        if (other_p == p || !next_state.active_players[other_p])
          continue;

        // First, count how many of this opponent's tokens are at each position
        // to detect stacks (blockades)
        int stack_count_at_pos = 0;
        int target_abs = abs_pos;

        for (int t = 0; t < NUM_TOKENS; ++t) {
          int other_pos = next_state.player_positions[other_p][t];
          if (other_pos != BASE_POS && other_pos != HOME_POS &&
              other_pos <= 50) {
            int other_abs = get_absolute_pos(other_p, other_pos);
            if (other_abs == target_abs) {
              stack_count_at_pos++;
            }
          }
        }

        // Only cut if opponent has exactly 1 token there (no blockade)
        if (stack_count_at_pos == 1) {
          for (int t = 0; t < NUM_TOKENS; ++t) {
            int other_pos = next_state.player_positions[other_p][t];
            if (other_pos != BASE_POS && other_pos != HOME_POS &&
                other_pos <= 50) {
              int other_abs = get_absolute_pos(other_p, other_pos);
              if (other_abs == abs_pos) {
                // Cut!
                next_state.player_positions[other_p][t] = BASE_POS;
                bonus_turn = true; // Cut gives bonus
              }
            }
          }
        }
        // If stack_count_at_pos >= 2, opponent has a blockade - cannot cut
      }
    }
  }

  update_board(next_state);

  // Update turn
  if (!bonus_turn) {
    next_state.current_player = (p + 1) % NUM_PLAYERS;
    // Skip inactive players
    while (!next_state.active_players[next_state.current_player]) {
      next_state.current_player = (next_state.current_player + 1) % NUM_PLAYERS;
    }
  }

  // Reset dice roll for next state (Chance Node)
  next_state.current_dice_roll = 0;

  return next_state;
}

int get_winner(const GameState &state) {
  if (!state.is_terminal)
    return -1;
  for (int p = 0; p < NUM_PLAYERS; ++p) {
    if (state.scores[p] == 4)
      return p;
  }
  return -1;
}

// --- Tensorization Helpers ---

// Precomputed Masks are skipped for now, using logic.

// Helper to fill buffer with 0
void clear_buffer(float *buffer, int size) {
  std::fill(buffer, buffer + size, 0.0f);
}

// Helper to write a value to a rotated position
// Channel c, Row r, Col c.
// Rotations k times CCW.
// Flattened index: ch * 225 + r * 15 + c
void write_tensor_val(float *buffer, int ch, int r, int c, float val,
                      int rotation_k, bool accumulate = true) {
  if (r < 0 || r >= BOARD_SIZE || c < 0 || c >= BOARD_SIZE)
    return;

  // Rotate (r, c) k times CCW to align visual perspective
  int tr = r;
  int tc = c;
  for (int i = 0; i < rotation_k; ++i) {
    // Rotate 90 CCW: (r, c) -> (14-c, r)
    int new_r = 14 - tc;
    int new_c = tr;
    tr = new_r;
    tc = new_c;
  }

  int idx = ch * (BOARD_SIZE * BOARD_SIZE) + tr * BOARD_SIZE + tc;
  if (accumulate) {
    buffer[idx] += val;
  } else {
    buffer[idx] = val;
  }
}

void write_state_tensor(const GameState &state, float *buffer) {
  // Cleaned 11-Channel Architecture: (11, 15, 15)
  // 0-3: My Tokens (Distinct Identity 0,1,2,3)
  // 4:   Opponent Pieces (Density, 0.25 per token, skip inactive)
  // 5:   Safe Zones (0.5)
  // 6:   My Home Path
  // 7:   Opponent Home Path (skip inactive)
  // 8:   Score Diff (broadcast)
  // 9:   My Locked (broadcast)
  // 10:  Opp Locked (broadcast)

  int num_channels = 17;
  int spatial_size = BOARD_SIZE * BOARD_SIZE;
  clear_buffer(buffer, num_channels * spatial_size);

  int current_p = state.current_player;
  int k = current_p; // Rotation steps (CCW)

  // --- CHANNELS 0-3: My Tokens (Distinct Identity) ---
  int p = current_p;
  for (int t = 0; t < NUM_TOKENS; ++t) {
    int pos = state.player_positions[p][t];
    int r, c;
    if (pos == BASE_POS)
      get_board_coords(p, pos, r, c, t);
    else
      get_board_coords(p, pos, r, c);

    if (r >= 0) {
      write_tensor_val(buffer, t, r, c, 1.0f, k);
    }
  }

  // --- CHANNEL 4: Opponent Pieces (Single Density Channel) ---
  for (int offset = 1; offset < 4; ++offset) {
    int opp_p = (current_p + offset) % 4;

    // Skip inactive players
    if (!state.active_players[opp_p])
      continue;

    for (int t = 0; t < NUM_TOKENS; ++t) {
      int pos = state.player_positions[opp_p][t];
      int r, c;
      if (pos == BASE_POS)
        get_board_coords(opp_p, pos, r, c, t);
      else
        get_board_coords(opp_p, pos, r, c);

      if (r >= 0) {
        write_tensor_val(buffer, 4, r, c, 0.25f, k);
      }
    }
  }

  // --- CHANNEL 5: Safe Zones ---
  for (int pl = 0; pl < NUM_PLAYERS; ++pl) {
    for (int s : SAFE_INDICES) {
      int r, c;
      get_board_coords(pl, s, r, c);
      if (r >= 0) {
        write_tensor_val(buffer, 5, r, c, 0.5f, k, false);
      }
    }
  }

  // --- CHANNEL 6: My Home Path ---
  for (int i = 0; i < 5; ++i) {
    int home_pos = 51 + i;
    int r, c;
    get_board_coords(current_p, home_pos, r, c);
    if (r >= 0) {
      write_tensor_val(buffer, 6, r, c, 1.0f, k, false);
    }
  }

  // --- CHANNEL 7: Opponent Home Path (skip inactive) ---
  for (int offset = 1; offset < 4; ++offset) {
    int opp_p = (current_p + offset) % 4;
    if (!state.active_players[opp_p])
      continue;

    for (int i = 0; i < 5; ++i) {
      int home_pos = 51 + i;
      int r, c;
      get_board_coords(opp_p, home_pos, r, c);
      if (r >= 0) {
        write_tensor_val(buffer, 7, r, c, 1.0f, k, false);
      }
    }
  }

  // --- CHANNEL 8: Score Diff ---
  int my_score = state.scores[current_p];
  int max_opp = 0;
  for (int pl = 0; pl < NUM_PLAYERS; ++pl) {
    if (pl != current_p)
      max_opp = std::max(max_opp, (int)state.scores[pl]);
  }
  float score_val = (float)(my_score - max_opp) / 4.0f;
  std::fill(buffer + (8 * spatial_size), buffer + (9 * spatial_size),
            score_val);

  // --- CHANNEL 9: My Locked ---
  int my_locked = 0;
  for (int t = 0; t < 4; ++t)
    if (state.player_positions[current_p][t] == BASE_POS)
      my_locked++;
  std::fill(buffer + (9 * spatial_size), buffer + (10 * spatial_size),
            (float)my_locked / 4.0f);

  // --- CHANNEL 10: Opp Locked ---
  int opp_locked = 0;
  int active_opp_tokens = 0;
  for (int pl = 0; pl < 4; ++pl) {
    if (pl == current_p)
      continue;
    if (!state.active_players[pl])
      continue;
    active_opp_tokens += 4;
    for (int t = 0; t < 4; ++t)
      if (state.player_positions[pl][t] == BASE_POS)
        opp_locked++;
  }
  float opp_locked_val = (active_opp_tokens > 0)
                             ? (float)opp_locked / (float)active_opp_tokens
                             : 0.0f;
  std::fill(buffer + (10 * spatial_size), buffer + (11 * spatial_size),
            opp_locked_val);
  // --- CHANNELS 11-16: Dice Roll (One-Hot) ---
  int roll = state.current_dice_roll;
  if (roll >= 1 && roll <= 6) {
    int channel_idx = 11 + (roll - 1);
    std::fill(buffer + (channel_idx * spatial_size),
              buffer + ((channel_idx + 1) * spatial_size), 1.0f);
  }
}
