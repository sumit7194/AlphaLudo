import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import numpy as np
from collections import defaultdict
import random
import time
from tqdm import tqdm

# Add project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import td_ludo_cpp as ludo_cpp
from src.model import AlphaLudoV5
from src.heuristic_bot import get_bot, BOT_REGISTRY
from src.config import MAX_MOVES_PER_GAME

MODELS_CONFIG = [
    {
        "name": "V6 Big (SL Pre-trained)",
        "path": "checkpoints/ac_v5/model_sl.pt",
        "in_channels": 17,
        "num_blocks": 10,
        "num_channels": 128
    },
    {
        "name": "V5 Medium (Best RL 65.8%)",
        "path": "checkpoints/ac_v5/model_best_v5_64ch.pt",
        "in_channels": 17,
        "num_blocks": 5,
        "num_channels": 64
    },
    {
        "name": "V4 Small (Legacy 11ch)",
        "path": "checkpoints/ac_v5_11ch/model_latest.pt",
        "in_channels": 11,
        "num_blocks": 5,
        "num_channels": 64
    }
]

def run_evaluation(model_info, num_games=1000, device='mps'):
    name = model_info["name"]
    path = model_info["path"]
    in_channels = model_info["in_channels"]
    num_blocks = model_info.get("num_blocks", 5)
    num_channels_model = model_info.get("num_channels", 64)
    
    print(f"\n[Bench] Starting Evaluation: {name}")
    print(f"[Bench] Loading weights from: {path}")
    
    device = torch.device(device)
    model = AlphaLudoV5(in_channels=in_channels, num_res_blocks=num_blocks, num_channels=num_channels_model).to(device)
    
    checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    available_types = list(BOT_REGISTRY.keys())
    wins = 0
    total = 0
    per_bot = defaultdict(lambda: {'wins': 0, 'games': 0})
    
    for _ in tqdm(range(num_games), desc=f"Playing {name}"):
        model_player = random.choice([0, 2])
        opp_player = 2 if model_player == 0 else 0
        
        bot_type = random.choice(available_types)
        opp_bot = get_bot(bot_type, player_id=opp_player)
        
        state = ludo_cpp.create_initial_state_2p()
        move_count = 0
        consecutive_sixes = [0, 0, 0, 0]
        
        while not state.is_terminal and move_count < MAX_MOVES_PER_GAME:
            current_p = state.current_player
            
            # Skip inactive
            if not state.active_players[current_p]:
                next_p = (current_p + 1) % 4
                while not state.active_players[next_p]:
                    next_p = (next_p + 1) % 4
                state.current_player = next_p
                continue
                
            # Roll dice
            if state.current_dice_roll == 0:
                roll = random.randint(1, 6)
                state.current_dice_roll = roll
                if roll == 6:
                    consecutive_sixes[current_p] += 1
                else:
                    consecutive_sixes[current_p] = 0
                
                if consecutive_sixes[current_p] >= 3:
                    state.current_dice_roll = 0
                    consecutive_sixes[current_p] = 0
                    state.current_player = (current_p + 1) % 4
                    while not state.active_players[state.current_player]:
                        state.current_player = (state.current_player + 1) % 4
                    continue
            
            legal_moves = ludo_cpp.get_legal_moves(state)
            if not legal_moves:
                state.current_dice_roll = 0
                state.current_player = (current_p + 1) % 4
                while not state.active_players[state.current_player]:
                    state.current_player = (state.current_player + 1) % 4
                continue
            
            action = -1
            if current_p == model_player:
                # Model decision
                state_tensor = ludo_cpp.encode_state(state)
                # SLICE CHANNELS if legacy
                if in_channels == 11:
                    state_tensor = state_tensor[:11, :, :]
                
                s_t = torch.from_numpy(state_tensor).unsqueeze(0).to(device)
                
                # Apply mask (4 tokens)
                mask = np.zeros(4, dtype=np.float32)
                for m in legal_moves:
                    mask[m] = 1.0
                m_t = torch.from_numpy(mask).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    logits = model.forward_policy_only(s_t, m_t)
                    probs = F.softmax(logits, dim=1)
                    action = int(torch.argmax(probs, dim=1).item())
                
                # Safety
                if action not in legal_moves:
                    action = random.choice(legal_moves)
            else:
                # Bot decision
                action = opp_bot.select_move(state, legal_moves)
            
            # Apply move
            state = ludo_cpp.apply_move(state, action)
            move_count += 1
            
        # Outcome
        total += 1
        per_bot[bot_type]['games'] += 1
        winner = ludo_cpp.get_winner(state)
        if winner == model_player:
            wins += 1
            per_bot[bot_type]['wins'] += 1
            
    win_rate = (wins / total) * 100
    print(f"\n[Result] {name}: {win_rate:.1f}% Win Rate")
    for bt, stats in per_bot.items():
        wr = (stats['wins'] / stats['games']) * 100 if stats['games'] > 0 else 0
        print(f"  - vs {bt:<10}: {wr:5.1f}% ({stats['wins']}/{stats['games']})")
    
    return {
        "name": name,
        "win_rate": win_rate,
        "per_bot": dict(per_bot)
    }

def main():
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    results = []
    
    for cfg in MODELS_CONFIG:
        if not os.path.exists(cfg["path"]):
            print(f"Skipping {cfg['name']} (Path not found: {cfg['path']})")
            continue
        res = run_evaluation(cfg, num_games=1000, device=device)
        results.append(res)
    
    print("\n" + "="*50)
    print("FINAL BENCHMARK COMPARISON (1000 Games Each)")
    print("="*50)
    print(f"{'Model Name':<25} | {'Win Rate':<10}")
    print("-" * 40)
    for r in results:
        print(f"{r['name']:<25} | {r['win_rate']:>8.1f}%")
    print("="*50)

if __name__ == "__main__":
    main()
