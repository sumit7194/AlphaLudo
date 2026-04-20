#ifndef GAME_H
#define GAME_H

#include <array>
#include <cstdint>
#include <vector>

// Constants
const int BOARD_SIZE = 15;
const int NUM_PLAYERS = 4;
const int NUM_TOKENS = 4;
const int PATH_LENGTH = 52; // Main path length
const int HOME_RUN_LENGTH = 5;
const int HOME_POS = 99; // Special value for Home
const int BASE_POS = -1; // Special value for Base

struct Move {
  int8_t token_index; // 0-3
  // We don't need 'to_pos' in the move struct for the move generation output,
  // just which token to move. The dice roll is part of the state context.
};

struct GameState {
  std::array<std::array<int8_t, BOARD_SIZE>, BOARD_SIZE>
      board; // -1 empty, 0-3 player ID
  std::array<std::array<int8_t, NUM_TOKENS>, NUM_PLAYERS>
      player_positions; // Current
  std::array<std::array<int8_t, NUM_TOKENS>, NUM_PLAYERS>
      prev_player_positions;              // Previous Turn (For Ghost Trail)
  std::array<int8_t, NUM_PLAYERS> scores; // Number of tokens home
  std::array<bool, NUM_PLAYERS> active_players; // true = player is in game
  int8_t current_player;                        // 0-3
  int8_t current_dice_roll;                     // 1-6
  bool is_terminal;
};

// Core Logic
std::vector<int>
get_legal_moves(const GameState &state); // Returns list of token indices (0-3)
GameState apply_move(const GameState &state, int token_index);
int get_winner(const GameState &state); // Returns -1 if none, 0-3 if winner

// Tensorization - 17 Channels (V6 encoding)
// Spatial: (17, 15, 15) -> writes 3825 floats
void write_state_tensor(const GameState &state, float *buffer);

// V6.1 Tensorization - 24 Channels (Strategic encoding)
// Spatial: (24, 15, 15) -> writes 5400 floats
void write_state_tensor_v6(const GameState &state, float *buffer);

// V9 Tensorization - 14 Channels (Optimized encoding)
// Spatial: (14, 15, 15) -> writes 3150 floats
void write_state_tensor_v9(const GameState &state, float *buffer);

// V6.3 Tensorization - 27 Channels (Bonus-turn awareness)
// Channels 0-23: identical to V6 (24ch strategic encoding)
// Channel 24: bonus_turn_flag (broadcast 1.0 if dice == 6)
// Channel 25: consecutive_sixes (broadcast 0.0/0.5/1.0 for 0/1/2 sixes)
// Channel 26: two_roll_capture_map (1.0 where opponents capturable in 6+X combo)
// Spatial: (27, 15, 15) -> writes 6075 floats
void write_state_tensor_v6_3(const GameState &state, float *buffer,
                              int consecutive_sixes_count);

// V10 encoder (28 channels):
// - Channels 0-23: same as V6.1/V6.3 (strategic channels)
// - Channel 24: bonus_turn_flag (broadcast if dice==6)
// - Channel 25: two_roll_capture_map (was V6.3 ch26, promoted)
// - Channel 26: non_home_tokens_frac (broadcast, count_not_yet_home / 4)
// - Channel 27: my_leader_progress (broadcast, most-advanced token pos / 56)
// Spatial: (28, 15, 15) -> writes 6300 floats.
// Note: V6.3's ch25 (consecutive_sixes) was dropped — mech interp showed it unused.
void write_state_tensor_v10(const GameState &state, float *buffer);

// Helper to reset state
GameState create_initial_state();    // 4-player
GameState create_initial_state_2p(); // 2-player (P0 vs P2)

// Helper to get board coordinates from linear path position
void get_board_coords(int player, int pos, int &r, int &c, int token_index = 0);

#endif // GAME_H
