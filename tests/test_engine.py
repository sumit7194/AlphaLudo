import ludo_cpp
import time
import sys
import random

def test_initial_state():
    state = ludo_cpp.create_initial_state()
    assert state.current_player == 0
    assert state.current_dice_roll == 0
    for p in range(4):
        for t in range(4):
            assert state.player_positions[p][t] == -1
        assert state.scores[p] == 0
    print("Initial state test passed.")

def test_start_rule():
    state = ludo_cpp.create_initial_state()
    state.current_dice_roll = 5
    moves = ludo_cpp.get_legal_moves(state)
    assert len(moves) == 0, "Should not move from base with 5"
    
    state.current_dice_roll = 6
    moves = ludo_cpp.get_legal_moves(state)
    assert len(moves) == 4, "Should be able to move any token from base with 6"
    
    next_state = ludo_cpp.apply_move(state, moves[0])
    assert next_state.player_positions[0][0] == 0, "Token should be at start (0)"
    assert next_state.current_player == 0, "Should get bonus turn for 6"
    print("Start rule test passed.")

def test_movement():
    state = ludo_cpp.create_initial_state()
    state.player_positions[0][0] = 0
    state.current_dice_roll = 5
    
    moves = ludo_cpp.get_legal_moves(state)
    assert 0 in moves
    
    next_state = ludo_cpp.apply_move(state, 0)
    assert next_state.player_positions[0][0] == 5
    assert next_state.current_player == 1, "Turn should pass"
    print("Movement test passed.")

def test_cut():
    state = ludo_cpp.create_initial_state()
    state.player_positions[0][0] = 1 # Abs 1, not safe
    state.player_positions[1][0] = 39 # Rel 39 for P1 -> Abs 1
    state.current_player = 1
    state.current_dice_roll = 1
    
    moves = ludo_cpp.get_legal_moves(state)
    assert 0 in moves
    
    next_state = ludo_cpp.apply_move(state, 0)
    assert next_state.player_positions[1][0] == 40
    assert next_state.player_positions[0][0] == -1, "P0 should be cut to base"
    assert next_state.current_player == 1, "Bonus turn for cut"
    print("Cut test passed.")

def test_home():
    state = ludo_cpp.create_initial_state()
    state.player_positions[0][0] = 50
    state.current_dice_roll = 6
    
    moves = ludo_cpp.get_legal_moves(state)
    assert 0 in moves
    
    next_state = ludo_cpp.apply_move(state, 0)
    assert next_state.player_positions[0][0] == 99, "Should be home (99)"
    assert next_state.scores[0] == 1
    assert next_state.current_player == 0, "Bonus turn for home"
    print("Home test passed.")

def test_safety_zone():
    state = ludo_cpp.create_initial_state()
    state.player_positions[0][0] = 8 # Abs 8 (Safe)
    state.player_positions[1][0] = 46 # Rel 46 for P1 -> Abs 8
    
    state.current_player = 1
    state.current_dice_roll = 1 # P1 moves from 46 -> 47 (Abs 8 + 1? No wait. 46 is pre-move)
    # P1 at 46. 46 + 13 = 59. 59 % 52 = 7. Not 8.
    # We want P1 to land on Abs 8.
    # Abs 8 -> (Rel + 13) % 52 = 8 -> Rel + 13 = 8 impossible. Rel + 13 = 60 -> Rel = 47.
    
    state.player_positions[1][0] = 46
    state.current_dice_roll = 1
    
    # 46 + 1 = 47. Abs(47) = (47+13)%52 = 60%52 = 8. Checking calc. Matches.
    
    next_state = ludo_cpp.apply_move(state, 0)
    
    # Check positions
    # P1 should be at Rel 47
    assert next_state.player_positions[1][0] == 47
    # P0 should STILL be at 8 (Safe)
    assert next_state.player_positions[0][0] == 8, "P0 should be safe at pos 8"
    assert next_state.current_player == 2, "Turn should pass (no cut)"
    print("Safety zone test passed.")

def test_overshoot():
    state = ludo_cpp.create_initial_state()
    state.player_positions[0][0] = 55 # 1 step from home (56)
    state.current_dice_roll = 2 # Overshoot
    
    moves = ludo_cpp.get_legal_moves(state)
    # Should be empty for this token
    assert 0 not in moves, "Should not allow overshoot move"
    print("Overshoot test passed.")

def test_manual_game_sim():
    # Simulate a few turns to verify state transitions
    state = ludo_cpp.create_initial_state()
    
    # Force some progress
    state.player_positions[0][0] = 50
    state.current_player = 0
    state.current_dice_roll = 6
    
    # P0 moves to home
    moves = ludo_cpp.get_legal_moves(state)
    state = ludo_cpp.apply_move(state, moves[0])
    
    assert state.scores[0] == 1
    assert state.current_player == 0 # Bonus
    
    # P0 rolls again, say 1. But no moves (all others in base, home token done)
    # Wait, other tokens are in base. 6 needed.
    state.current_dice_roll = 1
    moves = ludo_cpp.get_legal_moves(state)
    assert len(moves) == 0
    
    # Manual turn pass logic
    state.current_player = (state.current_player + 1) % 4
    state.current_dice_roll = 0 # Waiting for roll
    
    assert state.current_player == 1
    print("Manual game sim step passed.")

def benchmark():
    state = ludo_cpp.create_initial_state()
    state.player_positions[0][0] = 0
    state.current_dice_roll = 3
    
    start = time.time()
    iterations = 1000000
    for _ in range(iterations):
        ludo_cpp.get_legal_moves(state)
    end = time.time()
    
    print(f"Benchmark: {iterations} move generations in {end-start:.4f}s")
    print(f"Speed: {iterations/(end-start):.2f} ops/sec")

if __name__ == "__main__":
    test_initial_state()
    test_start_rule()
    test_movement()
    test_cut()
    test_home()
    test_safety_zone()
    test_overshoot()
    test_manual_game_sim()
    benchmark()
