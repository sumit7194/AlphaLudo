
import sys
import os
import random
import time

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

try:
    import ludo_cpp
except ImportError:
    print("Error: Could not import ludo_cpp. Make sure it is installed.")
    sys.exit(1)

def print_state(state):
    print(f"  Pos: {state.player_positions}")
    print(f"  Home: {state.home_tokens}")
    print(f"  Scores: {state.scores}")

def debug_game():
    print("Initializing GameState...")
    state = ludo_cpp.GameState()
    
    turn_count = 0
    max_turns = 300
    
    while not state.is_terminal and turn_count < max_turns:
        turn_count += 1
        current_p = state.current_player
        
        # 1. Roll Dice (if 0)
        roll = state.current_dice_roll
        if roll == 0:
            roll = random.randint(1, 6)
            state.current_dice_roll = roll
            print(f"\n[Turn {turn_count}] P{current_p} ROLLS {roll} 🎲")
        else:
            print(f"\n[Turn {turn_count}] P{current_p} (Bonus/Cont) ROLLS {roll} 🎲")

        # Details
        positions = state.player_positions[current_p]
        print(f"  Tokens: {positions} (Home Count: {state.scores[current_p]})")

        # 2. Get Legal Moves
        moves = ludo_cpp.get_legal_moves(state)
        print(f"  Legal Moves Indices: {moves}")
        
        # 3. Decision
        if len(moves) == 0:
            print(f"  ❌ NO MOVES. SKIP.")
            # Skip logic: Next player, Dice 0
            # Note: We must replicate exactly what vector_league does or check if game.cpp handles it?
            # game.cpp does NOT auto-skip in apply_move(token). 
            # We must set next player manually if python loop handles it.
            # BUT, vector_league manually sets state.current_player.
            # WE MUST SIMULATE THAT.
            
            state.current_player = (current_p + 1) % 4
            state.current_dice_roll = 0
            print(f"  -> Next Player: P{state.current_player}")
            
        else:
            # Random Choice
            move_idx = random.choice(moves)
            print(f"  ✅ Playing Token {move_idx}")
            
            # 4. Apply Move
            # apply_move returns new state object (by value/copy usually in loose wrapper, or modifies?)
            # ludo_cpp.GameState is a class. apply_move is a free function or method?
            # It is a method `state.apply_move(token, dice)` in some versions or `ludo_cpp.apply_move(state, token)`?
            # Checking game.cpp binding... usually `state.apply_move(move_idx)` if bound as method.
            # Or `next_state = ludo_cpp.apply_move(state, move_idx)`?
            
            # In vector_league: `state = ludo_cpp.apply_move(state, action)`
            state = ludo_cpp.apply_move(state, move_idx)
            
            # Print result
            # print_state(state)
            
            # Check turn continuity
            next_p = state.current_player
            if next_p == current_p:
                print("  🔄 BONUS TURN (6 or Home)")
            else:
                print(f"  -> Next Player: P{next_p}")

    if state.is_terminal:
        print(f"\n🏆 GAME OVER! Winner: {state.winner}")
    else:
        print("\n⚠️ Max turns reached.")

if __name__ == "__main__":
    debug_game()
