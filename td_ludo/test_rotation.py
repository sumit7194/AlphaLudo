import td_ludo_cpp as ludo_cpp
import numpy as np

game = ludo_cpp.create_initial_state_2p()
game.current_dice_roll = 6

# Move P0 token 0 to start
m0 = ludo_cpp.get_legal_moves(game)[0]
ludo_cpp.apply_move(game, m0)

print(f"Current player: {game.current_player}")

# Encode from P0 perspective
state0 = ludo_cpp.encode_state(game)

# Now it's P2's turn (in a 2P game) or still P0 (since P0 rolled a 6).
# Let's force it to be P2's turn without changing the board.
game.current_player = 2
print(f"Current player: {game.current_player}")

# Encode from P2 perspective
state2 = ludo_cpp.encode_state(game)

# The state tensor has shape (11, 15, 15).
# Channel 0: Current player's pieces
# Channel 1: Next active player's pieces (which is P2 for P0, and P0 for P2)
# Since the game state is exactly the same, P0's pieces (currently at start) 
# should be in Channel 0 of state0, and Channel 1 of state2.
# Let's verify by printing the sum of the channels.
print("--- State 0 (P0 perspective) ---")
print(f"Ch 0 (P0 pieces) sum: {state0[0].sum()}")
print(f"Ch 1 (P2 pieces) sum: {state0[1].sum()}")

print("--- State 2 (P2 perspective) ---")
print(f"Ch 0 (P2 pieces) sum: {state2[0].sum()}")
print(f"Ch 1 (P0 pieces) sum: {state2[1].sum()}")
