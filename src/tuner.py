
import os
import time
import json
import random
import torch
import numpy as np
from datetime import datetime

import src.config as cfg
import ludo_cpp
from src.model_v3 import AlphaLudoV3
from src.tensor_utils_mastery import state_to_tensor_mastery

# --- TUNING CONSTANTS ---
EVAL_GAMES = 50           # Number of games per check
CHECK_INTERVAL_SEC = 1800 # 30 Minutes
HISTORY_FILE = "data/tuner_history.json"
CONFIG_FILE = "config.json"

class AutoTuner:
    def __init__(self):
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"[Tuner] Initialized on {self.device}")
        
        # Load Model
        self.model = AlphaLudoV3(num_res_blocks=10, num_channels=128)
        self.model.to(self.device)
        self.model.eval()
        
        # Ensure history dir exists
        os.makedirs("data", exist_ok=True)
        if not os.path.exists(HISTORY_FILE):
             with open(HISTORY_FILE, 'w') as f:
                 json.dump([], f)

    def load_latest_model(self):
        if os.path.exists(cfg.MAIN_CKPT_PATH):
            try:
                ckpt = torch.load(cfg.MAIN_CKPT_PATH, map_location=self.device)
                self.model.load_state_dict(ckpt['model_state_dict'])
                return True
            except Exception as e:
                print(f"[Tuner] Failed to load model: {e}")
                return False
        return False

    def calculate_entropy(self, probs):
        """Calculates Shannon entropy of a probability distribution."""
        # probs is numpy array
        # Add epsilon to avoid log(0)
        ent = -np.sum(probs * np.log(probs + 1e-10))
        return ent

    def run_evaluation(self, num_games):
        """Runs evaluation games and returns stats."""
        wins = 0
        total_entropy = 0.0
        entropy_samples = 0
        
        print(f"[Tuner] Running {num_games} eval games...")
        
        for g in range(num_games):
            state = ludo_cpp.GameState()
            # Model is Player 0
            model_pid = 0
            step = 0
            
            while not state.is_terminal and step < 500:
                pid = state.current_player
                moves = ludo_cpp.get_legal_moves(state)
                
                # Dice handling (simplified simulation)
                if state.current_dice_roll == 0:
                    state.current_dice_roll = random.randint(1, 6)
                
                if not moves:
                    state = ludo_cpp.apply_move(state, -1) # Pass
                    continue
                
                if pid == model_pid:
                    # Model Move
                    tens = state_to_tensor_mastery(state).unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        policy, _ = self.model.forward_policy_value(tens)
                    
                    probs = policy[0].cpu().numpy()
                    
                    # Log Entropy
                    # Only consider legal moves for entropy? 
                    # Actually, raw policy entropy is what we care about for collapse
                    # But if network predicts illegal moves, that's bad too.
                    # Let's take entropy of masked probs to be fair (actual usage)
                    
                    masked = np.zeros(4)
                    for m in moves:
                        masked[m] = probs[m]
                    
                    if masked.sum() > 0:
                        masked /= masked.sum()
                        ent = self.calculate_entropy(masked)
                        total_entropy += ent
                        entropy_samples += 1
                        action = np.argmax(masked)
                    else:
                        action = random.choice(moves)
                else:
                    # Random Opponent
                    action = random.choice(moves)
                
                state = ludo_cpp.apply_move(state, action)
                step += 1
            
            # Check winner
            if state.scores[model_pid] == 4:
                wins += 1
        
        win_rate = (wins / num_games) * 100
        avg_entropy = total_entropy / max(1, entropy_samples)
        return win_rate, avg_entropy

    def update_config(self, new_params):
        """Updates config.json safely."""
        try:
            with open(CONFIG_FILE, 'r') as f:
                conf = json.load(f)
            
            # Update all modes or just PROD? Let's update PROD and BACKGROUND
            # We assume active mode is PROD usually.
            
            changed = False
            for mode in ["PROD", "BACKGROUND"]:
                current = conf[mode]
                if (current.get("dirichlet_eps") != new_params["eps"] or 
                    current.get("c_puct") != new_params["cpuct"]):
                    
                    conf[mode]["dirichlet_eps"] = new_params["eps"]
                    conf[mode]["c_puct"] = new_params["cpuct"]
                    changed = True
            
            if changed:
                with open(CONFIG_FILE, 'w') as f:
                    json.dump(conf, f, indent=4)
                print(f"[Tuner] Updated Config: EPS={new_params['eps']}, CPUCT={new_params['cpuct']}")
                return True
        except Exception as e:
            print(f"[Tuner] Config Update Failed: {e}")
        return False

    def log_result(self, win_rate, entropy, params, state):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "win_rate": win_rate,
            "entropy": entropy,
            "params": params,
            "state": state
        }
        
        # Read existing data
        try:
            with open(HISTORY_FILE, 'r') as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            data = []
        
        # Append and write back
        data.append(entry)
        with open(HISTORY_FILE, 'w') as f:
            json.dump(data, f, indent=4)

    def run_loop(self):
        print("[Tuner] Starting Auto-Tuner Loop...")
        while True:
            if self.load_latest_model():
                wr, ent = self.run_evaluation(EVAL_GAMES)
                print(f"[Tuner] Result: WinRate={wr:.1f}%, Entropy={ent:.3f}")
                
                # --- STATE MACHINE LOGIC ---
                new_state = "UNKNOWN"
                params = {"eps": 0.25, "cpuct": 3.0} # Default
                
                # 1. Check Collapse (Entropy < 0.5 implies 90% bias on one move)
                # Or ridiculously low win rate
                if ent < 0.5 or wr < 10.0:
                    new_state = "COLLAPSED"
                    params = {"eps": 0.50, "cpuct": 4.0}
                
                # 2. Check Learning
                elif 10.0 <= wr <= 30.0:
                    new_state = "LEARNING"
                    params = {"eps": 0.25, "cpuct": 3.0}
                
                # 3. Check Refining
                else: # wr > 30.0
                    new_state = "REFINING"
                    params = {"eps": 0.15, "cpuct": 2.0}
                
                print(f"[Tuner] Diagnosis: {new_state}")
                
                # Apply
                self.update_config(params)
                self.log_result(wr, ent, params, new_state)
            
            else:
                print("[Tuner] Waiting for model checkpoint...")
            
            print(f"[Tuner] Sleeping for {CHECK_INTERVAL_SEC}s...")
            time.sleep(CHECK_INTERVAL_SEC)

if __name__ == "__main__":
    tuner = AutoTuner()
    tuner.run_loop()
