"""
AlphaLudo Super League Evaluation
---------------------------------
Runs a fair tournament between 8 agents (4 Models + 4 Bots).

Agents:
1. Ghost 120 (Model)
2. Ghost 140 (Model)
3. Ghost 160 (Model)
4. Kickstart Main / Cycle 169 (Model)
5. Random Bot
6. Aggressive Bot
7. Defensive Bot
8. Racing Bot

Format:
- Generates all unique 4-player combinations (70 combos).
- Plays N games for EACH combination.
- Rotates starting positions to ensure fairness.
- Tracks Win Rate, Elo-like score, and specific matchups.
"""

import os
import sys
import random
import time
import argparse
import itertools
import numpy as np
import torch
import json
from collections import defaultdict

# Add project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import ludo_cpp
from src.model_v3 import AlphaLudoV3
from src.heuristic_bot import HeuristicLudoBot, AggressiveBot, DefensiveBot, RacingBot
from src.tensor_utils_mastery import state_to_tensor_mastery

# --- Configuration ---
GHOSTS_DIR = "experiments/kickstart/ghosts"
MAIN_MODEL_PATH = "experiments/kickstart/model_kickstart.pt"

class Agent:
    def __init__(self, name, agent_type, model=None, bot_cls=None):
        self.name = name
        self.type = agent_type  # 'model' or 'bot'
        self.model = model      # PyTorch model (if type='model')
        self.bot_cls = bot_cls  # Bot class (if type='bot')
        self.bot_instance = None # Instantiated per game

    def select_move(self, state, legal_moves, device):
        if self.type == 'model':
            # Neural Net Inference
            state_tensor = state_to_tensor_mastery(state)
            input_tensor = state_tensor.unsqueeze(0).to(device=device, dtype=torch.float32)
            with torch.no_grad():
                policy, _ = self.model.forward_policy_value(input_tensor)
            
            probs = policy[0].cpu().numpy()
            masked_probs = np.zeros(4)
            for m in legal_moves:
                masked_probs[m] = probs[m]
            
            if masked_probs.sum() > 0:
                masked_probs /= masked_probs.sum()
                return np.argmax(masked_probs) # Greedy
            else:
                return random.choice(legal_moves)
        else:
            # Bot Logic
            if self.bot_instance is None:
                self.bot_instance = self.bot_cls(player_id=state.current_player)
            return self.bot_instance.select_move(state, legal_moves)

    def reset(self, player_id):
        if self.type == 'bot':
            self.bot_instance = self.bot_cls(player_id=player_id)

class RandomBotWrapper:
    def __init__(self, player_id): pass
    def select_move(self, state, legal_moves):
        return random.choice(legal_moves) if legal_moves else -1

def load_models(device):
    """Load all 4 model checkpoints."""
    models = {}
    
    # 1. Main Kickstart
    if os.path.exists(MAIN_MODEL_PATH):
        try:
            m = AlphaLudoV3().to(device)
            ckpt = torch.load(MAIN_MODEL_PATH, map_location=device)
            m.load_state_dict(ckpt['model_state_dict'])
            m.eval()
            models['Main (169)'] = m
            print(f"✅ Loaded Main Model (Cycle 169)")
        except Exception as e:
            print(f"❌ Failed to load Main Model: {e}")

    # 2. Ghosts
    ghost_cycles = [120, 140, 160]
    for c in ghost_cycles:
        path = os.path.join(GHOSTS_DIR, f"ghost_cycle_{c}.pt")
        if os.path.exists(path):
            try:
                m = AlphaLudoV3().to(device)
                ckpt = torch.load(path, map_location=device)
                m.load_state_dict(ckpt['model_state_dict'])
                m.eval()
                models[f"Ghost {c}"] = m
                print(f"✅ Loaded Ghost Cycle {c}")
            except Exception as e:
                print(f"❌ Failed to load Ghost {c}: {e}")
        else:
            print(f"⚠️  Ghost {c} not found at {path}")

    return models

def run_game(agents, device):
    """
    Run a single game with 4 specific agents assigned to P0-P3.
    Returns: winner_index (0-3), game_length
    """
    state = ludo_cpp.GameState()
    
    # Reset bots with assigned player IDs
    for pid, agent in enumerate(agents):
        agent.reset(pid)
        
    total_moves = 0
    max_moves = 1000
    consecutive_sixes = [0]*4

    while not state.is_terminal and total_moves < max_moves:
        current_player = state.current_player
        agent = agents[current_player]
        
        # Dice Roll
        if state.current_dice_roll == 0:
            roll = np.random.randint(1, 7)
            state.current_dice_roll = roll
            if roll == 6:
                consecutive_sixes[current_player] += 1
            else:
                consecutive_sixes[current_player] = 0
            
            if consecutive_sixes[current_player] >= 3:
                state.current_player = (state.current_player + 1) % 4
                state.current_dice_roll = 0
                consecutive_sixes[current_player] = 0
                continue
        
        legal = ludo_cpp.get_legal_moves(state)
        if not legal:
            state.current_player = (state.current_player + 1) % 4
            state.current_dice_roll = 0
            continue
            
        action = agent.select_move(state, legal, device)
        state = ludo_cpp.apply_move(state, action)
        total_moves += 1
        
        # Win Check
        if state.scores[current_player] == 4:
            return current_player, total_moves
            
    # Timeout/Draw -> Winner is highest score
    return np.argmax(state.scores), total_moves

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rounds', type=int, default=1, help='Rounds per combination (default: 1)')
    parser.add_argument('--full', action='store_true', help='Run full tournament (all 70 combos)')
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # --- 1. Initialize Agents ---
    agents_pool = []
    
    # Models
    loaded_models = load_models(device)
    for name, model in loaded_models.items():
        agents_pool.append(Agent(name, 'model', model=model))

    # Bots
    bot_configs = [
        ('Bot Random', RandomBotWrapper),
        ('Bot Aggressive', AggressiveBot),
        ('Bot Defensive', DefensiveBot),
        ('Bot Racing', RacingBot)
    ]
    for name, cls in bot_configs:
        agents_pool.append(Agent(name, 'bot', bot_cls=cls))

    print(f"\n🏆 League Agents ({len(agents_pool)}):")
    for i, a in enumerate(agents_pool):
        print(f"  {i+1}. {a.name} ({a.type})")
    
    if len(agents_pool) < 4:
        print("❌ Not enough agents to run a game!")
        return

    # --- 2. Generate Matchups ---
    # Case 1: Full Round Robin (All Combinations)
    # 8 choose 4 = 70 combinations.
    combinations = list(itertools.combinations(agents_pool, 4))
    print(f"\ngenerating {len(combinations)} unique matchups...")
    
    # Shuffle order of matchups
    random.shuffle(combinations)
    
    total_games = len(combinations) * args.rounds
    print(f"Total Games to Play: {total_games}")
    
    # Stats
    wins = defaultdict(int)
    games_played = defaultdict(int)
    head_to_head = defaultdict(lambda: defaultdict(int)) # wins[A][B] = times A beat B
    
    start_time = time.time()
    game_count = 0
    
    try:
        for matchup_idx, agents_combo in enumerate(combinations):
            # For each combination, play 'rounds' games
            # Rotate positions each round to be fair
            # 4 agents -> 24 permutations, but let's just rotate cyclically for simple 'rounds'
            
            # Start logic:
            # Round 1: A B C D
            # Round 2: D A B C ...
            
            current_order = list(agents_combo)
            
            for r in range(args.rounds):
                # Shuffle positions for every single game to be maximally fair
                # or rotate? Random shuffle is statistically fair over large N
                random.shuffle(current_order)
                
                game_count += 1
                winner_idx, moves = run_game(current_order, device)
                winner_agent = current_order[winner_idx]
                
                # Record Stats
                wins[winner_agent.name] += 1
                
                for agent in current_order:
                    games_played[agent.name] += 1
                    if agent != winner_agent:
                        head_to_head[winner_agent.name][agent.name] += 1

                print(f"Game {game_count}/{total_games} | Winner: {winner_agent.name:<15} | Moves: {moves}")

    except KeyboardInterrupt:
        print("\n🛑 Tournament Interrupted!")

    elapsed = time.time() - start_time

    # Save Results
    results_data = {
        'total_games': game_count,
        'elapsed_sec': elapsed,
        'games_played': dict(games_played),
        'wins': dict(wins),
        'head_to_head': {k: dict(v) for k, v in head_to_head.items()}
    }
    with open('league_results.json', 'w') as f:
        json.dump(results_data, f, indent=2)
    print(f"\n💾 Saved results to league_results.json")

    print(f"\n\n{'='*60}")
    print(f"🏆 LEAGUE STANDINGS ({game_count} games played)")
    print(f"{'='*60}")
    print(f"{'Rank':<4} {'Agent':<20} {'Win Rate':<10} {'Wins':<6} {'Games':<6}")
    print(f"{'-'*60}")
    
    # Sort by Win Rate
    sorted_stats = sorted(games_played.keys(), 
                         key=lambda x: wins[x]/games_played[x] if games_played[x]>0 else 0, 
                         reverse=True)
    
    for rank, name in enumerate(sorted_stats, 1):
        g = games_played[name]
        w = wins[name]
        wr = (w / g * 100) if g > 0 else 0
        print(f"{rank:<4} {name:<20} {wr:5.1f}%     {w:<6} {g:<6}")
    print(f"{'='*60}")
    
    print(f"\nSpeed: {game_count/elapsed:.1f} games/s")

if __name__ == "__main__":
    main()
