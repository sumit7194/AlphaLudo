import os
import sys
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque
import random
from functools import partial

# Force print to flush immediately so nohup.out updates in real-time
print = partial(print, flush=True)

# Add project root to path for td_ludo_cpp
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import td_ludo_cpp
from src.model import AlphaLudoV5
# No reward shaping in AlphaZero (pure win/loss)

# --- CONFIGURATION ---
BATCH_SIZE = 32              # Number of parallel games (Reduced for faster iterations)
MCTS_SIMULATIONS = 200       # Increased from 50 to 200 for better depth
REPLAY_BUFFER_SIZE = 100000  # Number of state-action-value tuples
TRAIN_BATCH_SIZE = 256
TRAIN_STEPS_PER_ITER = 100
LEARNING_RATE = 1e-4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
WARM_START_MODEL = "checkpoints/ac_v6_big/backups/model_latest_323k_shaped.pt"
CHECKPOINT_DIR = "checkpoints/mcts_v1"
EVAL_EVERY_N_ITERS = 5

import pickle

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state_tensor, policy, value):
        self.buffer.append((state_tensor, policy, value))
        
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, policies, values = zip(*batch)
        return torch.stack(states), torch.tensor(policies, dtype=torch.float32), torch.tensor(values, dtype=torch.float32)
    
    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(list(self.buffer), f)
            
    def load(self, path):
        if os.path.exists(path):
            with open(path, 'rb') as f:
                data = pickle.load(f)
                self.buffer.extend(data)
                
    def __len__(self):
        return len(self.buffer)


def play_mcts_games(model, env, mcts, num_games, histories):
    """
    Play one full step of batched games using MCTS.
    Returns collected transitions when games finish.
    """
    model.eval()
    transitions = []
        
    # 1. Get current python states
    states = [env.get_game(i) for i in range(num_games)]
    
    # 2. Set MCTS roots
    mcts.set_roots(states)
    
    # 3. MCTS Simulations
    for _ in range(MCTS_SIMULATIONS):
        # Selection phase
        leaf_states = mcts.select_leaves(1) # parallel_sims = 1
        
        # If no leaves to evaluate (all terminal/no moves), break early
        if not leaf_states:
            break
            
        # Get tensors from C++
        num_leaves = len(leaf_states)
        try:
            flat_tensors = mcts.get_leaf_tensors() # returns numpy array (batch, 17, 15, 15)
        except Exception as e:
            print(f"Error getting leaf tensors: {e}")
            break
            
        tensor_batch = torch.tensor(flat_tensors, dtype=torch.float32).to(DEVICE)
        
        # NN Evaluation
        with torch.no_grad():
            logits, values = model(tensor_batch)
            probs = F.softmax(logits, dim=1).cpu().numpy().tolist()
            values = values.squeeze(-1).cpu().numpy().tolist()
            
        # Backpropagate
        mcts.expand_and_backprop(probs, values)
        
    # 4. Extract target policies and select actions
    # Decay temperature over time or set to 1.0 for exploration, 0 for exploitation
    # For training we usually use temp=1.0 for first 30 moves, then 0.1
    # We will just use temp=1.0 for simplicity in self-play
    mcts_probs = mcts.get_action_probs(1.0) # List of lists (batch, 4)
    
    actions = []
    for i in range(num_games):
        state = states[i]
        probs = mcts_probs[i]
        
        # Save state tensor (before action)
        state_tensor = torch.tensor(td_ludo_cpp.encode_state(state), dtype=torch.float32)
        
        if not state.is_terminal:
            histories[i].append({
                'state_tensor': state_tensor,
                'pi': probs,
                'player': state.current_player
            })
            
            # Sample action
            legal = td_ludo_cpp.get_legal_moves(state)
            if legal:
                # Re-normalize over legal moves just in case MCTS gave non-zero to illegal
                legal_probs = [probs[m] if m in legal else 0.0 for m in range(4)]
                sum_p = sum(legal_probs)
                if sum_p > 0:
                    legal_probs = [p / sum_p for p in legal_probs]
                    action = np.random.choice(4, p=legal_probs)
                else:
                    action = random.choice(legal)
            else:
                action = 0 # Dummy
            actions.append(action)
        else:
            actions.append(0) # Dummy for terminal
            
    # 5. Step Environment
    next_states, rewards, dones, infos = env.step(actions)
    
    # 6. Process finished games
    for i in range(num_games):
        # Force a draw if a game takes more than 600 valid moves to prevent MCTS infinite loops
        if dones[i] or len(histories[i]) >= 600:
            if dones[i]:
                winner = infos[i]['winner']
            else:
                winner = -1 # Draw due to timeout
            # Assign rewards to history
            for turn in histories[i]:
                # Reward is +1 if the turn's player won, -1 if they lost, 0 for draw
                if winner == -1:
                    z = 0.0
                elif turn['player'] == winner:
                    z = 1.0
                else:
                    z = -1.0
                
                transitions.append((turn['state_tensor'], turn['pi'], z))
            
            # Reset the game in the C++ environment so it can start fresh
            env.reset_game(i)
            # Clear history
            histories[i] = []
            
    return transitions

def train():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    print(f"Initializing AlphaZero MCTS Training on {DEVICE}...")
    
    # 1. Model
    model = AlphaLudoV5(num_res_blocks=10, num_channels=128).to(DEVICE)
    if os.path.exists(WARM_START_MODEL):
        print(f"Loading warm start model: {WARM_START_MODEL}")
        ckpt = torch.load(WARM_START_MODEL, map_location=DEVICE, weights_only=True)
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        print("WARNING: Warm start model not found. Using random weights (Not recommended for MCTS).")
        
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    
    # 2. Replay Buffer
    buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)
    buffer_path = os.path.join(CHECKPOINT_DIR, "replay_buffer.pkl")
    if os.path.exists(buffer_path):
        print(f"Loading existing replay buffer from {buffer_path}")
        buffer.load(buffer_path)
        print(f"Loaded {len(buffer)} transitions.")
    
    # 3. Environment & MCTS
    # Initialize in 2-player mode directly (P0 vs P2)
    env = td_ludo_cpp.VectorGameState(BATCH_SIZE, True)
    env.reset() # Resets all games internally in C++
    
    histories = [[] for _ in range(BATCH_SIZE)]
    mcts = td_ludo_cpp.MCTSEngine(BATCH_SIZE, c_puct=1.5, dirichlet_alpha=0.3, dirichlet_eps=0.25)
    
    # 4. Training Loop
    iteration = 0
    total_games = 0
    
    try:
        while True:
            iteration += 1
            print(f"\n--- Iteration {iteration} ---")
            
            # Self Play Phase: Generate data until we have a chunk of finished games
            print("Generating self-play data via MCTS...")
            new_transitions = []
            games_completed = 0
            start_time = time.time()
            
            # Play until at least 10 games finish
            steps = 0
            while games_completed < 10:
                transitions = play_mcts_games(model, env, mcts, BATCH_SIZE, histories)
                new_transitions.extend(transitions)
                
                # Count how many games finished in this step
                if transitions:
                    # Roughly transitions / avg_game_length
                    games_completed += len([t for t in transitions if t[2] != 0.0]) / 200 # Appx
                
                steps += 1
                if steps % 10 == 0:
                    print(f"  Step {steps}... (Buffer adds pending)")
                    
                if len(new_transitions) > 2000:
                     break # Don't get stuck if no winners
                     
            for t in new_transitions:
                buffer.push(*t)
                
            time_taken = time.time() - start_time
            print(f"Gathered {len(new_transitions)} transitions in {time_taken:.1f}s. Buffer size: {len(buffer)}")
            
            # Optimization Phase
            if len(buffer) >= TRAIN_BATCH_SIZE:
                print("Training model...")
                model.train()
                total_v_loss = 0
                total_p_loss = 0
                
                for _ in range(TRAIN_STEPS_PER_ITER):
                    states, target_pis, target_vs = buffer.sample(TRAIN_BATCH_SIZE)
                    states, target_pis, target_vs = states.to(DEVICE), target_pis.to(DEVICE), target_vs.to(DEVICE)
                    
                    optimizer.zero_grad()
                    logits, values = model(states)
                    
                    # Value Loss (MSE)
                    v_loss = F.mse_loss(values.squeeze(-1), target_vs)
                    
                    # Policy Loss (Cross Entropy to target probabilities)
                    # target_pis is (B, 4), probabilities
                    log_probs = F.log_softmax(logits, dim=1)
                    p_loss = -(target_pis * log_probs).sum(dim=1).mean()
                    
                    loss = v_loss + p_loss
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    
                    total_v_loss += v_loss.item()
                    total_p_loss += p_loss.item()
                    
                print(f"  Avg V-Loss: {total_v_loss/TRAIN_STEPS_PER_ITER:.4f}")
                print(f"  Avg P-Loss: {total_p_loss/TRAIN_STEPS_PER_ITER:.4f}")
                
                # Save Checkpoint
                ckpt_path = os.path.join(CHECKPOINT_DIR, f"model_latest.pt")
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'iteration': iteration
                }, ckpt_path)
                print(f"Saved latest checkpoint: {ckpt_path}")
                
                if iteration % EVAL_EVERY_N_ITERS == 0:
                    history_path = os.path.join(CHECKPOINT_DIR, f"model_iter_{iteration}.pt")
                    os.system(f"cp {ckpt_path} {history_path}")
                    print(f"Saved historical checkpoint: {history_path}")
                    
                    # Run Evaluation (Optional, calls external script if needed)
                    # Since evaluating blocks training for minutes, we can just print a reminder
                    # or trigger a background eval. For now we will just log it.
                    print(f"--- Triggering Evaluation at Iteration {iteration} ---")
                    # os.system(f"python evaluate.py --model {ckpt_path}")
                    
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user (Ctrl+C).")
        print("Saving graceful shutdown checkpoints...")
        
        # Save Model
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"model_interrupted.pt")
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'iteration': iteration
        }, ckpt_path)
        print(f"Saved interrupted model state to: {ckpt_path}")
        
        # Save Replay Buffer
        buffer.save(buffer_path)
        print(f"Saved {len(buffer)} transitions to replay buffer.")
        print("Shutdown complete. You can safely close or restart.")

if __name__ == "__main__":
    train()
