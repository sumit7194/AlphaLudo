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

// Tensorization - 11 Channels (Cleaned 2P Afterstate)
// Spatial: (11, 15, 15) -> writes 2475 floats
void write_state_tensor(const GameState &state, float *buffer);

// Helper to reset state
GameState create_initial_state();    // 4-player
GameState create_initial_state_2p(); // 2-player (P0 vs P2)

// Helper to get board coordinates from linear path position
void get_board_coords(int player, int pos, int &r, int &c, int token_index = 0);

#endif // GAME_H
