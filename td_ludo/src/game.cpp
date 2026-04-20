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

// =============================================================================
// V9 Encoder: 14-Channel Architecture
// =============================================================================
// 0-3:  My Tokens (Distinct Identity 0,1,2,3)
// 4-7:  Opponent Tokens (Distinct Identity 0,1,2,3)
// 8:    Safe Zones (0.5)
// 9:    My Home Path
// 10:   Opponent Home Path (skip inactive)
// 11:   My Locked % (broadcast)
// 12:   Opp Locked % (broadcast)
// 13:   Dice Roll (single channel, value = roll / 6.0)

void write_state_tensor_v9(const GameState &state, float *buffer) {
  int num_channels = 14;
  int spatial_size = BOARD_SIZE * BOARD_SIZE;
  clear_buffer(buffer, num_channels * spatial_size);

  int current_p = state.current_player;
  int k = current_p; // Rotation steps (CCW)

  // --- CHANNELS 0-3: My Tokens (Distinct Identity) ---
  for (int t = 0; t < NUM_TOKENS; ++t) {
    int pos = state.player_positions[current_p][t];
    int r, c;
    if (pos == BASE_POS)
      get_board_coords(current_p, pos, r, c, t);
    else
      get_board_coords(current_p, pos, r, c);

    if (r >= 0) {
      write_tensor_val(buffer, t, r, c, 1.0f, k);
    }
  }

  // --- CHANNELS 4-7: Opponent Tokens (Distinct Identity) ---
  // In 2P mode: one active opponent. Find them and write their 4 tokens.
  // In 4P mode: use first active opponent's tokens (2P is primary use case).
  int opp_p = -1;
  for (int offset = 1; offset < 4; ++offset) {
    int candidate = (current_p + offset) % 4;
    if (state.active_players[candidate]) {
      opp_p = candidate;
      break;
    }
  }

  if (opp_p >= 0) {
    for (int t = 0; t < NUM_TOKENS; ++t) {
      int pos = state.player_positions[opp_p][t];
      int r, c;
      if (pos == BASE_POS)
        get_board_coords(opp_p, pos, r, c, t);
      else
        get_board_coords(opp_p, pos, r, c);

      if (r >= 0) {
        write_tensor_val(buffer, 4 + t, r, c, 1.0f, k);
      }
    }
  }

  // --- CHANNEL 8: Safe Zones ---
  for (int pl = 0; pl < NUM_PLAYERS; ++pl) {
    for (int s : SAFE_INDICES) {
      int r, c;
      get_board_coords(pl, s, r, c);
      if (r >= 0) {
        write_tensor_val(buffer, 8, r, c, 0.5f, k, false);
      }
    }
  }

  // --- CHANNEL 9: My Home Path ---
  for (int i = 0; i < 5; ++i) {
    int home_pos = 51 + i;
    int r, c;
    get_board_coords(current_p, home_pos, r, c);
    if (r >= 0) {
      write_tensor_val(buffer, 9, r, c, 1.0f, k, false);
    }
  }

  // --- CHANNEL 10: Opponent Home Path (skip inactive) ---
  for (int offset = 1; offset < 4; ++offset) {
    int opp = (current_p + offset) % 4;
    if (!state.active_players[opp])
      continue;

    for (int i = 0; i < 5; ++i) {
      int home_pos = 51 + i;
      int r, c;
      get_board_coords(opp, home_pos, r, c);
      if (r >= 0) {
        write_tensor_val(buffer, 10, r, c, 1.0f, k, false);
      }
    }
  }

  // --- CHANNEL 11: My Locked % (broadcast) ---
  int my_locked = 0;
  for (int t = 0; t < 4; ++t)
    if (state.player_positions[current_p][t] == BASE_POS)
      my_locked++;
  std::fill(buffer + (11 * spatial_size), buffer + (12 * spatial_size),
            (float)my_locked / 4.0f);

  // --- CHANNEL 12: Opp Locked % (broadcast) ---
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
  std::fill(buffer + (12 * spatial_size), buffer + (13 * spatial_size),
            opp_locked_val);

  // --- CHANNEL 13: Dice Roll (single value = roll / 6.0) ---
  int roll = state.current_dice_roll;
  if (roll >= 1 && roll <= 6) {
    float dice_val = (float)roll / 6.0f;
    std::fill(buffer + (13 * spatial_size), buffer + (14 * spatial_size),
              dice_val);
  }
}

// =============================================================================
// V6.1 Encoder: 24-Channel Strategic Architecture
// =============================================================================
// Channels 0-16: Identical to write_state_tensor (17ch V6)
// NEW:
// 17-20: Opponent Tokens (Distinct Identity 0,1,2,3)
// 21:    Danger Map (1.0 at own tokens with opponent 1-6 behind)
// 22:    Capture Opportunity Map (1.0 at positions capturable with current dice)
// 23:    Safe Landing Map (1.0 at safe positions reachable with current dice)

void write_state_tensor_v6(const GameState &state, float *buffer) {
  int num_channels = 24;
  int spatial_size = BOARD_SIZE * BOARD_SIZE;
  clear_buffer(buffer, num_channels * spatial_size);

  int current_p = state.current_player;
  int k = current_p;
  int dice = state.current_dice_roll;

  // === CHANNELS 0-16: Exact copy of write_state_tensor logic ===

  // Ch 0-3: My Tokens
  for (int t = 0; t < NUM_TOKENS; ++t) {
    int pos = state.player_positions[current_p][t];
    int r, c;
    if (pos == BASE_POS)
      get_board_coords(current_p, pos, r, c, t);
    else
      get_board_coords(current_p, pos, r, c);
    if (r >= 0)
      write_tensor_val(buffer, t, r, c, 1.0f, k);
  }

  // Ch 4: Opponent Density
  for (int offset = 1; offset < 4; ++offset) {
    int opp_p = (current_p + offset) % 4;
    if (!state.active_players[opp_p])
      continue;
    for (int t = 0; t < NUM_TOKENS; ++t) {
      int pos = state.player_positions[opp_p][t];
      int r, c;
      if (pos == BASE_POS)
        get_board_coords(opp_p, pos, r, c, t);
      else
        get_board_coords(opp_p, pos, r, c);
      if (r >= 0)
        write_tensor_val(buffer, 4, r, c, 0.25f, k);
    }
  }

  // Ch 5: Safe Zones
  for (int pl = 0; pl < NUM_PLAYERS; ++pl) {
    for (int s : SAFE_INDICES) {
      int r, c;
      get_board_coords(pl, s, r, c);
      if (r >= 0)
        write_tensor_val(buffer, 5, r, c, 0.5f, k, false);
    }
  }

  // Ch 6: My Home Path
  for (int i = 0; i < 5; ++i) {
    int r, c;
    get_board_coords(current_p, 51 + i, r, c);
    if (r >= 0)
      write_tensor_val(buffer, 6, r, c, 1.0f, k, false);
  }

  // Ch 7: Opponent Home Path
  for (int offset = 1; offset < 4; ++offset) {
    int opp_p = (current_p + offset) % 4;
    if (!state.active_players[opp_p])
      continue;
    for (int i = 0; i < 5; ++i) {
      int r, c;
      get_board_coords(opp_p, 51 + i, r, c);
      if (r >= 0)
        write_tensor_val(buffer, 7, r, c, 1.0f, k, false);
    }
  }

  // Ch 8: Score Diff
  int my_score = state.scores[current_p];
  int max_opp = 0;
  for (int pl = 0; pl < NUM_PLAYERS; ++pl)
    if (pl != current_p)
      max_opp = std::max(max_opp, (int)state.scores[pl]);
  std::fill(buffer + (8 * spatial_size), buffer + (9 * spatial_size),
            (float)(my_score - max_opp) / 4.0f);

  // Ch 9: My Locked %
  int my_locked = 0;
  for (int t = 0; t < 4; ++t)
    if (state.player_positions[current_p][t] == BASE_POS)
      my_locked++;
  std::fill(buffer + (9 * spatial_size), buffer + (10 * spatial_size),
            (float)my_locked / 4.0f);

  // Ch 10: Opp Locked %
  int opp_locked = 0;
  int active_opp_tokens = 0;
  for (int pl = 0; pl < 4; ++pl) {
    if (pl == current_p || !state.active_players[pl])
      continue;
    active_opp_tokens += 4;
    for (int t = 0; t < 4; ++t)
      if (state.player_positions[pl][t] == BASE_POS)
        opp_locked++;
  }
  std::fill(buffer + (10 * spatial_size), buffer + (11 * spatial_size),
            active_opp_tokens > 0 ? (float)opp_locked / (float)active_opp_tokens
                                  : 0.0f);

  // Ch 11-16: Dice One-Hot
  if (dice >= 1 && dice <= 6) {
    int ch = 11 + (dice - 1);
    std::fill(buffer + (ch * spatial_size), buffer + ((ch + 1) * spatial_size),
              1.0f);
  }

  // === CHANNELS 17-20: Individual Opponent Tokens ===
  int opp_p = -1;
  for (int offset = 1; offset < 4; ++offset) {
    int candidate = (current_p + offset) % 4;
    if (state.active_players[candidate]) {
      opp_p = candidate;
      break;
    }
  }
  if (opp_p >= 0) {
    for (int t = 0; t < NUM_TOKENS; ++t) {
      int pos = state.player_positions[opp_p][t];
      int r, c;
      if (pos == BASE_POS)
        get_board_coords(opp_p, pos, r, c, t);
      else
        get_board_coords(opp_p, pos, r, c);
      if (r >= 0)
        write_tensor_val(buffer, 17 + t, r, c, 1.0f, k);
    }
  }

  // === CHANNEL 21: Danger Map ===
  for (int t = 0; t < NUM_TOKENS; ++t) {
    int my_pos = state.player_positions[current_p][t];
    if (my_pos < 0 || my_pos > 50)
      continue;
    int my_abs = get_absolute_pos(current_p, my_pos);
    if (my_abs < 0)
      continue;
    if (is_safe_pos(my_abs))
      continue;
    bool stacked = false;
    for (int t2 = 0; t2 < NUM_TOKENS; ++t2) {
      if (t2 == t) continue;
      int op = state.player_positions[current_p][t2];
      if (op >= 0 && op <= 50 && get_absolute_pos(current_p, op) == my_abs) {
        stacked = true;
        break;
      }
    }
    if (stacked) continue;
    bool endangered = false;
    for (int op = 0; op < NUM_PLAYERS; ++op) {
      if (op == current_p || !state.active_players[op]) continue;
      for (int ot = 0; ot < NUM_TOKENS; ++ot) {
        int op_pos = state.player_positions[op][ot];
        if (op_pos < 0 || op_pos > 50) continue;
        int op_abs = get_absolute_pos(op, op_pos);
        if (op_abs < 0) continue;
        int dist = (my_abs - op_abs + 52) % 52;
        if (dist >= 1 && dist <= 6) { endangered = true; break; }
      }
      if (endangered) break;
    }
    if (endangered) {
      int r, c;
      get_board_coords(current_p, my_pos, r, c);
      if (r >= 0) write_tensor_val(buffer, 21, r, c, 1.0f, k, false);
    }
  }

  // === CHANNEL 22: Capture Opportunity Map ===
  if (dice >= 1 && dice <= 6) {
    for (int t = 0; t < NUM_TOKENS; ++t) {
      int my_pos = state.player_positions[current_p][t];
      int landing;
      if (my_pos == BASE_POS) {
        if (dice != 6) continue;
        landing = 0;
      } else if (my_pos >= 0) {
        landing = my_pos + dice;
      } else {
        continue;
      }
      if (landing < 0 || landing > 50) continue;
      int landing_abs = get_absolute_pos(current_p, landing);
      if (landing_abs < 0 || is_safe_pos(landing_abs)) continue;
      for (int op = 0; op < NUM_PLAYERS; ++op) {
        if (op == current_p || !state.active_players[op]) continue;
        int opp_count = 0;
        for (int ot = 0; ot < NUM_TOKENS; ++ot) {
          int op_pos = state.player_positions[op][ot];
          if (op_pos >= 0 && op_pos <= 50 &&
              get_absolute_pos(op, op_pos) == landing_abs)
            opp_count++;
        }
        if (opp_count == 1) {
          int r, c;
          get_board_coords(current_p, landing, r, c);
          if (r >= 0) write_tensor_val(buffer, 22, r, c, 1.0f, k, false);
        }
      }
    }
  }

  // === CHANNEL 23: Safe Landing Map ===
  if (dice >= 1 && dice <= 6) {
    for (int t = 0; t < NUM_TOKENS; ++t) {
      int my_pos = state.player_positions[current_p][t];
      int landing;
      if (my_pos == BASE_POS) {
        if (dice != 6) continue;
        landing = 0;
      } else if (my_pos >= 0) {
        landing = my_pos + dice;
      } else {
        continue;
      }
      if (landing > 56) continue;
      bool safe = false;
      if (landing == 56 || landing == HOME_POS) {
        safe = true;
      } else if (landing > 50 && landing < 56) {
        safe = true;
      } else if (landing >= 0 && landing <= 50) {
        int landing_abs = get_absolute_pos(current_p, landing);
        if (landing_abs >= 0) {
          if (is_safe_pos(landing_abs)) {
            safe = true;
          } else {
            for (int t2 = 0; t2 < NUM_TOKENS; ++t2) {
              if (t2 == t) continue;
              int op = state.player_positions[current_p][t2];
              if (op >= 0 && op <= 50 &&
                  get_absolute_pos(current_p, op) == landing_abs) {
                safe = true;
                break;
              }
            }
          }
        }
      }
      if (safe) {
        int r, c;
        get_board_coords(current_p, landing, r, c);
        if (r >= 0) write_tensor_val(buffer, 23, r, c, 1.0f, k, false);
      }
    }
  }
}

// ═══════════════════════════════════════════════════════════════
// V6.3: 27-channel encoding (V6 + bonus-turn awareness)
//
// Channels 0-23: identical to write_state_tensor_v6
// Channel 24:    bonus_turn_flag — broadcast 1.0 if current dice == 6
// Channel 25:    consecutive_sixes — broadcast normalized (0/0.5/1.0)
// Channel 26:    two_roll_capture_map — 1.0 at opponent positions
//                capturable in a 6+X two-roll sequence (7-12 ahead)
//
// The consecutive_sixes_count is passed from Python because GameState
// doesn't track it (it's managed by the game player's turn logic).
// ═══════════════════════════════════════════════════════════════

void write_state_tensor_v6_3(const GameState &state, float *buffer,
                              int consecutive_sixes_count) {
  int spatial_size = BOARD_SIZE * BOARD_SIZE;

  // Clear full 27-channel buffer, then write channels 0-23 via V6.
  // V6 internally clears channels 0-23 (harmless double-clear).
  clear_buffer(buffer, 27 * spatial_size);
  write_state_tensor_v6(state, buffer);

  int current_p = state.current_player;
  int k = current_p; // rotation key for write_tensor_val
  int dice = state.current_dice_roll;

  // === CHANNEL 24: bonus_turn_flag (broadcast) ===
  // 1.0 everywhere if current dice is 6 (player will get another turn).
  // The dice one-hot (ch 16) already encodes "dice == 6", but as one of
  // six planes. This broadcast makes the bonus-turn concept explicit as
  // a standalone feature the CNN can detect with a single 1x1 filter.
  if (dice == 6) {
    float *ch24 = buffer + 24 * spatial_size;
    std::fill(ch24, ch24 + spatial_size, 1.0f);
  }

  // === CHANNEL 25: consecutive_sixes (broadcast, normalized) ===
  // 0 sixes = 0.0, 1 six = 0.5, 2 sixes = 1.0.
  // At 2 consecutive sixes the model should play conservatively (one
  // more 6 = triple penalty = turn lost, tokens sent home in some
  // rule variants). Clamped to 2 because 3+ never occurs (penalty fires).
  int clamped = consecutive_sixes_count < 2 ? consecutive_sixes_count : 2;
  if (clamped > 0) {
    float csix_val = clamped * 0.5f;
    float *ch25 = buffer + 25 * spatial_size;
    std::fill(ch25, ch25 + spatial_size, csix_val);
  }

  // === CHANNEL 26: two_roll_capture_map (spatial) ===
  // For each own token, mark opponent positions that are capturable via
  // a two-roll sequence: first roll = 6 (bonus turn), second roll = 1-6.
  //
  // Main-track tokens (pos 0-50):
  //   Capturable range = pos+7 to pos+12 (6+1 to 6+6)
  //
  // Base tokens (pos == -1):
  //   Spawn on 6 lands at pos 0, then move 1-6 → landing at pos 1-6.
  //   Capturable range = 1 to 6.
  //
  // A position is capturable only if:
  //   - It's on the main track (not home run, not safe square)
  //   - Exactly 1 opponent token is there (no blockade)
  for (int t = 0; t < NUM_TOKENS; ++t) {
    int my_pos = state.player_positions[current_p][t];

    int range_start, range_end;
    if (my_pos == BASE_POS) {
      // Base token: spawn(6) → pos 0, then move 1-6 → pos 1..6
      range_start = 1;
      range_end = 6;
    } else if (my_pos >= 0 && my_pos <= 50) {
      // Main track: 6 + (1..6) = 7..12 ahead
      range_start = my_pos + 7;
      range_end = my_pos + 12;
    } else {
      continue; // home run or scored — can't capture from here
    }

    for (int target_rel = range_start; target_rel <= range_end;
         ++target_rel) {
      if (target_rel > 50)
        continue; // can't capture in home run

      int target_abs = get_absolute_pos(current_p, target_rel);
      if (target_abs < 0)
        continue;
      if (is_safe_pos(target_abs))
        continue; // can't capture on safe/globe square

      // Count opponent tokens at this absolute position.
      // Capturable only if exactly 1 (no blockade).
      for (int op = 0; op < NUM_PLAYERS; ++op) {
        if (op == current_p || !state.active_players[op])
          continue;
        int opp_count = 0;
        for (int ot = 0; ot < NUM_TOKENS; ++ot) {
          int op_pos = state.player_positions[op][ot];
          if (op_pos >= 0 && op_pos <= 50) {
            int op_abs = get_absolute_pos(op, op_pos);
            if (op_abs == target_abs)
              opp_count++;
          }
        }
        if (opp_count == 1) {
          int r, c;
          get_board_coords(current_p, target_rel, r, c);
          if (r >= 0)
            write_tensor_val(buffer, 26, r, c, 1.0f, k, false);
        }
      }
    }
  }
}


// ═══════════════════════════════════════════════════════════════
// V10: 28-channel encoding (V6.3 minus dead ch25 + 2 new strategic)
//
// Channels 0-23: identical to V6.1 / V6.3 (write_state_tensor_v6)
// Channel 24:    bonus_turn_flag      (same as V6.3 ch24)
// Channel 25:    two_roll_capture_map (same as V6.3 ch26 — promoted,
//                V6.3 ch25 was dropped because mech interp showed it unused)
// Channel 26 (NEW): non_home_tokens_frac
//                   broadcast (count_of_my_tokens_not_yet_home / 4)
//                   values: 0.0, 0.25, 0.5, 0.75, 1.0
//                   0.25 = "forced mode" (only 1 token left to score)
// Channel 27 (NEW): my_leader_progress
//                   broadcast (most_advanced_token_pos / 56), range [0, 1]
//                   1.0 = at least one token already home
// ═══════════════════════════════════════════════════════════════

// Progress helper: 0.0 for base, 1.0 for home, pos/56 otherwise
static inline float token_progress_01(int pos) {
  if (pos == BASE_POS) return 0.0f;
  if (pos == HOME_POS) return 1.0f;
  return (float)pos / 56.0f;
}

void write_state_tensor_v10(const GameState &state, float *buffer) {
  int spatial_size = BOARD_SIZE * BOARD_SIZE;

  // Clear full 28-channel buffer
  clear_buffer(buffer, 28 * spatial_size);

  // Channels 0-23: reuse V6.1 encoder
  write_state_tensor_v6(state, buffer);

  int current_p = state.current_player;
  int dice = state.current_dice_roll;
  int k = current_p; // rotation key

  // === CHANNEL 24: bonus_turn_flag ===
  if (dice == 6) {
    float *ch24 = buffer + 24 * spatial_size;
    std::fill(ch24, ch24 + spatial_size, 1.0f);
  }

  // === CHANNEL 25: two_roll_capture_map (spatial) ===
  // Same logic as V6.3 ch26, now at ch25.
  for (int t = 0; t < NUM_TOKENS; ++t) {
    int my_pos = state.player_positions[current_p][t];
    int range_start, range_end;
    if (my_pos == BASE_POS) {
      range_start = 1;
      range_end = 6;
    } else if (my_pos >= 0 && my_pos <= 50) {
      range_start = my_pos + 7;
      range_end = my_pos + 12;
    } else {
      continue;
    }
    for (int target_rel = range_start; target_rel <= range_end; ++target_rel) {
      if (target_rel > 50) continue;
      int target_abs = get_absolute_pos(current_p, target_rel);
      if (target_abs < 0) continue;
      if (is_safe_pos(target_abs)) continue;
      for (int op = 0; op < NUM_PLAYERS; ++op) {
        if (op == current_p || !state.active_players[op]) continue;
        int opp_count = 0;
        for (int ot = 0; ot < NUM_TOKENS; ++ot) {
          int op_pos = state.player_positions[op][ot];
          if (op_pos >= 0 && op_pos <= 50) {
            int op_abs = get_absolute_pos(op, op_pos);
            if (op_abs == target_abs) opp_count++;
          }
        }
        if (opp_count == 1) {
          int r, c;
          get_board_coords(current_p, target_rel, r, c);
          if (r >= 0)
            write_tensor_val(buffer, 25, r, c, 1.0f, k, false);
        }
      }
    }
  }

  // === CHANNEL 26 (NEW): non_home_tokens_frac (broadcast) ===
  // Count tokens NOT yet home (scored). fraction = count / 4.
  // Encodes "how much work is left" with emphasis on "1 = forced mode" state.
  int not_home_count = 0;
  for (int t = 0; t < NUM_TOKENS; ++t) {
    if (state.player_positions[current_p][t] != HOME_POS) not_home_count++;
  }
  float not_home_frac = (float)not_home_count / 4.0f;
  if (not_home_frac > 0.0f) {
    float *ch26 = buffer + 26 * spatial_size;
    std::fill(ch26, ch26 + spatial_size, not_home_frac);
  }

  // === CHANNEL 27 (NEW): my_leader_progress (broadcast) ===
  // Progress in [0, 1] of my most-advanced token.
  // 1.0 iff at least one of my tokens has already scored.
  float max_progress = 0.0f;
  for (int t = 0; t < NUM_TOKENS; ++t) {
    float p = token_progress_01(state.player_positions[current_p][t]);
    if (p > max_progress) max_progress = p;
  }
  if (max_progress > 0.0f) {
    float *ch27 = buffer + 27 * spatial_size;
    std::fill(ch27, ch27 + spatial_size, max_progress);
  }
}
