import asyncio
import websockets
import json
import time
import torch
import numpy as np
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import ludo_cpp
from src.model_v3 import AlphaLudoV3
from src.tensor_utils_mastery import state_to_tensor_mastery, get_board_coords
from src.config import MAIN_CKPT_PATH

PORT = 8091
SLEEP_DELAY = 1.0  # Seconds between moves

# --- Helper to get Rich Logs ---
def get_rich_token_stats(state, p_idx):
    """
    Calculate rich context (SAFE/DANGER/CHASING) for the UI.
    Replicates logic from vector_league.py but for single GameState.
    """
    stats = []
    SAFE_INDICES = [0, 8, 13, 21, 26, 34, 39, 47]
    my_pos = state.player_positions[p_idx]
    
    # 1. Gather Opponent Tokens
    opp_tokens = []
    for opp_p in range(4):
        if opp_p == p_idx: continue
        for t in range(4):
            pos = int(state.player_positions[opp_p][t])
            if pos != -1 and pos != 99:
                r, c = get_board_coords(opp_p, pos, t if pos == -1 else 0)
                opp_tokens.append({'p': opp_p, 'r': r, 'c': c, 'pos': pos})

    # 2. Analyze My Tokens
    for t in range(4):
        pos = int(my_pos[t])
        info = {
            'id': t, 'pos': pos, 
            'safe': False, 'danger': False, 'chasing': False, 
            'chasing_who': [], 'danger_from': []
        }
        
        if pos == -1 or pos == 99:
            pass
        elif pos in SAFE_INDICES:
             info['safe'] = True
        else:
             # Check Danger
             r_me, c_me = get_board_coords(p_idx, pos, 0)
             for opp in opp_tokens:
                  for d in range(1, 7):
                      target_pos = opp['pos'] + d
                      if target_pos > 56: continue
                      tgt_r, tgt_c = get_board_coords(opp['p'], target_pos, 0)
                      if tgt_r == r_me and tgt_c == c_me:
                           info['danger'] = True
                           if opp['p'] not in info['danger_from']:
                               info['danger_from'].append(opp['p'])
        
        # Check Chasing
        if pos != -1 and pos != 99:
             for d in range(1, 7):
                  target_pos = pos + d
                  if target_pos > 56: continue
                  tgt_r, tgt_c = get_board_coords(p_idx, target_pos, 0)
                  for opp in opp_tokens:
                       if opp['pos'] in SAFE_INDICES: continue
                       if opp['r'] == tgt_r and opp['c'] == tgt_c:
                            info['chasing'] = True
                            if opp['p'] not in info['chasing_who']:
                                info['chasing_who'].append(opp['p'])
        
        stats.append(info)
    return stats

# --- JSON Sanitizer ---
def recursive_cast(obj):
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return recursive_cast(obj.tolist())
    elif isinstance(obj, dict):
        return {k: recursive_cast(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [recursive_cast(x) for x in obj]
    return obj

async def safe_send(websocket, payload):
    try:
        clean_payload = recursive_cast(payload)
        await websocket.send(json.dumps(clean_payload))
    except Exception as e:
        print(f"Send Error: {e}")

async def run_visual_eval(websocket):
    print(f"Client connected! Starting One-Off Evaluation Game...")
    
    # 1. Load Model (Mock or Real)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model = AlphaLudoV3().to(device)
    try:
        if os.path.exists(MAIN_CKPT_PATH):
            ckpt = torch.load(MAIN_CKPT_PATH, map_location=device)
            model.load_state_dict(ckpt['model_state_dict'])
            model.eval()
            print(f"Loaded Model: {MAIN_CKPT_PATH}")
    except: pass

    state = ludo_cpp.create_initial_state()
    # Start from fresh initial state (no debug override)
    
    loop_idx = 0
    max_moves = 500
    logs = []
    
    # 1. Send Identities
    await safe_send(websocket, {
        'type': 'identities',
        'game_id': 0,
        'data': ['AlphaLudo (P0)', 'Bot (P1)', 'Bot (P2)', 'Bot (P3)']
    })

    # Send Initial State
    print("Sending Initial State...")
    await send_update(websocket, state, logs, "Game Start", loop_idx)
    await asyncio.sleep(2)

    while not state.is_terminal and loop_idx < max_moves:
        loop_idx += 1
        current_p = int(state.current_player)
        
        # Roll
        if state.current_dice_roll == 0:
            state.current_dice_roll = np.random.randint(1, 7)
            # await send_update(websocket, state, logs, f"P{current_p} Rolled", loop_idx)
            # await asyncio.sleep(0.5)
        
        legal = ludo_cpp.get_legal_moves(state)
        
        if not legal:
            if state.current_dice_roll == 6:
                print(f"!!! CRITICAL BUG CAUGHT !!! P{current_p} Rolled 6 but No Moves!")
                print(f"Positions: {state.player_positions[current_p]}")
                print(f"Is Safe? {[ludo_cpp.is_safe(p) for p in state.player_positions[current_p]]}")
                # Print all positions to check for blockades
                print(f"All Pos: {state.player_positions}")
            
            log_msg = f"P{current_p} Rolled {state.current_dice_roll} - No Moves"
            logs.insert(0, log_msg)
            
            # Send 'move' with token -1 to indicate skip?
            await safe_send(websocket, {
                'type': 'move',
                'game_id': 0,
                'player': current_p,
                'dice': int(state.current_dice_roll),
                'token': -1
            })
            
            state.current_player = (state.current_player + 1) % 4
            state.current_dice_roll = 0
            await send_update(websocket, state, logs, log_msg, loop_idx)
            await asyncio.sleep(SLEEP_DELAY)
            continue

        move = -1
        token_stats = get_rich_token_stats(state, current_p)
        
        if current_p == 0: # Model
            tensor = state_to_tensor_mastery(state).to(device).unsqueeze(0)
            with torch.no_grad():
                pi, v, _ = model(tensor)
                pi_np = pi.cpu().numpy()[0]
                probs = np.array([pi_np[m] for m in legal])
                if probs.sum() > 0: probs /= probs.sum()
                else: probs = np.ones(len(legal))/len(legal)
                move = int(np.random.choice(legal, p=probs))
        else:
            move = int(np.random.choice(legal))

        pre_pos = int(state.player_positions[current_p][move])
        dice_used = int(state.current_dice_roll)
        
        state = ludo_cpp.apply_move(state, move)
        post_pos = int(state.player_positions[current_p][move])
        
        # Send Move
        await safe_send(websocket, {
            'type': 'move',
            'game_id': 0,
            'player': current_p,
            'dice': dice_used,
            'token': int(move),
            'from_pos': pre_pos,
            'to_pos': post_pos,
            'token_stats': token_stats
        })
        
        # Log
        log_txt = f"P{current_p} T{move}: {pre_pos} -> {post_pos}"
        print(f"Loop {loop_idx} | {log_txt}")
        logs.insert(0, log_txt)
        if len(logs) > 50: logs.pop()

        await send_update(websocket, state, logs, log_txt, loop_idx)
        await asyncio.sleep(SLEEP_DELAY)
    
    winner = ludo_cpp.get_winner(state)
    print(f"Game Over. Winner: {winner}")

async def send_update(websocket, state, logs, last_action, loop_idx):
    positions = []
    for p in range(4):
        positions.append([int(state.player_positions[p][t]) for t in range(4)])
    
    # Structure matching eval_game.html 'state' message
    # It expects: { type: 'state', state: { player_positions: ..., current_player: ... } }
    payload = {
        'type': 'state',
        'game_id': 0,
        'state': {
            'player_positions': positions,
            'current_player': int(state.current_player),
            'current_dice_roll': int(state.current_dice_roll),
            'scores': [0,0,0,0] 
        },
        'metrics': {'loop': loop_idx},
        'logs': logs
    }
    
    if isinstance(last_action, dict):
        payload['last_move'] = last_action
        
    await safe_send(websocket, payload)

async def main():
    print(f"Starting Visual Eval Server on port {PORT}...")
    async with websockets.serve(run_visual_eval, "0.0.0.0", PORT):
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    asyncio.run(main())
