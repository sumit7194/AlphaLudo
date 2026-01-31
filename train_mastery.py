
import os
import sys
import argparse
import time
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from model_mastery import AlphaLudoTopNet

from replay_buffer_mastery import ReplayBufferMastery
from src.visualizer import visualizer
from src.game_db import GameDB # NEW
from src.training_utils import EloTracker, TrainingMetrics # Explicit import for clarity
from src.config import LEARNING_RATE

import json

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def save_training_stats(path, total_games, effective_games):
    try:
        with open(path, 'w') as f:
            json.dump({
                'total_games': total_games,
                'effective_games': effective_games
            }, f)
    except Exception as e:
        print(f"Failed to save training stats: {e}")

def load_training_stats(path):
    if os.path.exists(path):
        try:
            with open(path, 'r') as f:
                data = json.load(f)
                return data.get('total_games', 0), data.get('effective_games', 0.0)
        except Exception as e:
            print(f"Failed to load training stats: {e}")
    return 0, 0.0

class TrainerMastery:
    def __init__(self, model, device, learning_rate=LEARNING_RATE):
        self.model = model
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        self.total_epochs = 0
        self.model.to(device)
        
    def train_step(self, states, target_policies, target_values):
        """
        18-Channel Async Plan training step.
        
        states: (B, 18, 15, 15) - Full spatial stack
        target_policies: (B, 225) - Sparse soft probabilities
        target_values: (B,) or (B, 1) - Win/Loss
        """
        self.model.train()
        states = states.to(self.device)
        target_policies = target_policies.to(self.device)
        target_values = target_values.to(self.device)
        
        if target_values.dim() == 1:
            target_values = target_values.unsqueeze(1)
        
        # --- Check Inputs ---
        if torch.isnan(states).any():
            print("[Trainer] CRITICAL: Input 'states' contains NaN!")
            return float('nan'), 0.0, 0.0
        if torch.isnan(target_policies).any():
            print("[Trainer] CRITICAL: Input 'target_policies' contains NaN!")
            return float('nan'), 0.0, 0.0
        if torch.isnan(target_values).any():
            print("[Trainer] CRITICAL: Input 'target_values' contains NaN!")
            return float('nan'), 0.0, 0.0

        # Forward pass with single input
        spatial_logits, predicted_values = self.model(states)
        
        # --- Check Outputs ---
        if torch.isnan(spatial_logits).any():
             print("[Trainer] CRITICAL: Model produced NaN logits!")
             # Debug weights?
             # print(self.model.conv_input.weight.mean())
             return float('nan'), 0.0, 0.0

        # --- Value Loss ---
        value_loss = F.mse_loss(predicted_values, target_values)
        
        # --- Policy Loss ---
        log_probs = F.log_softmax(spatial_logits, dim=1)
        policy_loss = -torch.sum(target_policies * log_probs, dim=1).mean()
        
        loss = value_loss + policy_loss
        
        if torch.isnan(loss):
            print(f"[Trainer] Warning: Loss is NaN. V_Loss={value_loss.item()}, P_Loss={policy_loss.item()}")
            return float('nan'), policy_loss.item(), value_loss.item()
            
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0) # Gradient Clipping
        self.optimizer.step()
        
        return loss.item(), policy_loss.item(), value_loss.item()
    
    def save_checkpoint(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'total_epochs': self.total_epochs
        }, path)

    def load_checkpoint(self, path):
        if not os.path.exists(path):
            return False
        try:
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.total_epochs = checkpoint['total_epochs']
            return True
        except Exception as e:
            print(f"Failed to load checkpoint {path}: {e}")
            return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-name', type=str, default='mastery_v1', help='Name of the run')
    parser.add_argument('--iterations', type=int, default=1000, help='Number of iterations (ignored if --continuous)')
    parser.add_argument('--continuous', action='store_true', help='Run indefinitely until Ctrl+C')
    parser.add_argument('--batch-size', type=int, default=16, help='Number of parallel games per batch')
    args = parser.parse_args()
    
    device = get_device()
    print(f"Mastery Training initialized on {device}")
    
    # Paths
    ckpt_dir = os.path.join("checkpoints_mastery", args.run_name)
    main_ckpt_path = os.path.join(ckpt_dir, "model_latest.pt")
    ghost_dir = os.path.join(ckpt_dir, "ghosts")
    os.makedirs(ghost_dir, exist_ok=True)
    
    # Ghost Pool: List of checkpoint paths
    ghost_pool = []
    # Populate existing ghosts if any
    if os.path.exists(ghost_dir):
        ghost_pool = [os.path.join(ghost_dir, f) for f in os.listdir(ghost_dir) if f.endswith(".pt")]
    
    
    # 1. Initialize Mastery Model
    main_model = AlphaLudoTopNet(num_res_blocks=10, num_channels=128)
    trainer = TrainerMastery(main_model, device)
    
    if trainer.load_checkpoint(main_ckpt_path):
         print(f"Resuming from {main_ckpt_path} (Epoch {trainer.total_epochs})")
    else:
         print("Starting fresh Mastery Model.")
    
    # 2. League: Pure win/loss signal (AlphaZero style)
    # No reward shaping - only game outcome matters
    
    # Probabilities: Multi-heuristic training distribution
    # 50% Self-play, 20% Ghost, 30% Bots (split among variants)
    probabilities = {
        'Main': 0.50,
        'Ghost': 0.20,
        'Heuristic': 0.10,
        'Aggressive': 0.07,
        'Defensive': 0.07,
        'Racing': 0.06
    }
    
    # 3. Components
    replay_buffer = ReplayBufferMastery(max_size=200000)
    
    # Elo Tracker for smart ghost selection
    from training_utils import EloTracker, augment_batch
    elo_path = f"checkpoints_mastery/{args.run_name}/elo_ratings.json"
    stats_path = f"checkpoints_mastery/{args.run_name}/training_stats.json"
    elo_tracker = EloTracker(save_path=elo_path)
    
    # VectorLeagueWorker with Ghost Support + Elo + Temperature Scheduling
    from vector_league import VectorLeagueWorker
    worker = VectorLeagueWorker(
        main_model=main_model, 
        probabilities=probabilities,
        mcts_simulations=200, 
        visualize=True, 
        ghost_pool=ghost_pool,
        elo_tracker=elo_tracker,
        temp_schedule='alphazero'  # Temperature annealing: 1.0 for first 30 moves, then 0.1
    )

    
    # --- Game Database ---
    game_db = GameDB("training_history.db")
    print(f"Game Database initialized at {game_db.db_path}")

    if visualizer:
        visualizer.start_server()
        visualizer.set_worker(worker) # Bridge Visualizer commands to Worker
        
        # Broadcast initial Elo data to UI
        main_elo = elo_tracker.get_rating('Main')
        
        # Fetch persistent stats from DB for initial rankings
        db_stats = game_db.get_all_stats()
        raw_rankings = elo_tracker.get_rankings()[:5]
        
        padded_rankings = []
        for name, elo in raw_rankings:
             if name == 'Main':
                 padded_rankings.append((name, elo, None))
             else:
                 stats = db_stats.get(name, {'wins': 0, 'games': 0})
                 wr = (stats['wins'] / stats['games'] * 100) if stats['games'] > 0 else 0.0
                 padded_rankings.append((name, elo, wr))
                 
        visualizer.broadcast_elo(main_elo, None, None, padded_rankings)
        visualizer.broadcast_elo_history(elo_tracker.history)


    # 4. Loop
    league_wins = {'Main': 0, 'Ghost': 0}
    league_played = {'Main': 0, 'Ghost': 0}
    
    # Time Tracking
    start_time = time.time()
    
    # Game Counters (Load if available)
    total_games, effective_games = load_training_stats(stats_path)
    print(f"Loaded Training Stats: {total_games} Games, {effective_games:.1f} Effective")
    
    # Choose iteration range: infinite if continuous, else fixed count
    start_iter = trainer.total_epochs
    num_iterations = args.iterations
    
    if args.continuous:
        print(f"Running CONTINUOUS training. Press Ctrl+C to stop (Buffer will be saved).")
        iteration_range = itertools.count(start_iter)
    else:
        iteration_range = range(start_iter, num_iterations)
    
    # --- Replay Buffer Persistence ---
    buffer_save_path = os.path.join(ckpt_dir, "replay_buffer.pkl")
    # replay_buffer already initialized at line 174 with max_size=200000
    
    # Load buffer if exists
    if os.path.exists(buffer_save_path):
        replay_buffer.load(buffer_save_path)
    
    
    # --- Winrate Tracking (Deprecated in favor of DB) ---
    # session_stats = {} # Now using game_db.get_all_stats()
    
    try:
        for i in iteration_range:
            iter_start_time = time.time()
            print(f"\n--- Iteration {i} ---")
            
            print(f"\nMastery Iteration {i} (Batch: {args.batch_size} Games, 25% Ghost)...")
            
            game_start_time = time.time()
            # Use a fixed temperature for self-play or annealing
            temp = 1.0 # Example: 1.0 for exploration, 0.1 for exploitation
            
            batch_data, metrics, batch_time = worker.play_batch(
                batch_size=args.batch_size,
                temperature=temp,
                epoch=i
            )
            
            # --- Stats Calculation ---
            iter_duration = time.time() - iter_start_time
            avg_game_time = batch_time / args.batch_size if args.batch_size > 0 else 0
            
            # Update Model Winrates & DB
            # results is list of {'winner': int, 'identities': [str]}
            results_list = metrics 
            
            # Save to DB
            for res in results_list:
                winner = res['winner']
                identities = res['identities']
                game_db.add_game(identities, winner)

            # Retrieve Persistent Stats from DB
            db_stats = game_db.get_all_stats()
            
            # Prepare Rankings with Winrate
            # Slice to top 10 to avoid flooding the frontend and ensuring consistency
            current_rankings = elo_tracker.get_rankings()[:10]
            augmented_rankings = []
            for name, elo in current_rankings:
                if name == 'Main':
                    augmented_rankings.append((name, elo, None)) # Main has global winrate logic below
                    continue
                
                # Fetch from DB stats
                stats = db_stats.get(name, {'wins': 0, 'games': 0})
                wr = (stats['wins'] / stats['games'] * 100) if stats['games'] > 0 else 0.0
                augmented_rankings.append((name, elo, wr))
            
            # Broadcast Stats
            if visualizer:
                 ghost_wr = 0.0 
                 
                 # Create stats object
                 # Use safely retrieved winrate for Main or default
                 main_wr = (league_wins.get('Main', 0) / league_played.get('Main', 1) * 100) if league_played.get('Main', 0) > 0 else 50.0
                 
                 stats_msg = {
                     'iteration': i,
                     'total_games': total_games,
                     'loss': 0.0, # Updated after training step
                     'win_rate': main_wr,
                     'buffer_size': len(replay_buffer),
                     'effective_games': effective_games,
                     'avg_game_time': avg_game_time,
                     'last_iter_time': iter_duration,
                     'ghost_winrate': 0.0 # Deprecated, see Elo table
                 }
                 
                 visualizer.broadcast_elo(elo_tracker.get_rating('Main'), None, None, augmented_rankings)
                 visualizer.broadcast_stats(**stats_msg)

            effective_games += (len(batch_data) / 200.0) # Approx
            total_games += args.batch_size
            
            # Add to buffer
            replay_buffer.add(batch_data)
            
            # Save Elo ratings and Stats periodically
            elo_tracker.save()
            save_training_stats(stats_path, total_games, effective_games)
            
            # Stats
            # The original league_wins/played logic is now handled by visualizer.game_results and elo_tracker
            # for res in results:
            #     winner = res.get('winner', -1)
            #     identities = res.get('identities', ['Main'] * 4)
            #     if winner != -1 and winner < len(identities):
            #         w_id = identities[winner]
            #         league_wins[w_id] = league_wins.get(w_id, 0) + 1
            #     for pid in identities:
            #         league_played[pid] = league_played.get(pid, 0) + 1
                
            print(f"  Iteration Completed in {iter_duration:.1f}s. (+{len(batch_data)} samples)")
            print(f"  Buffer: {len(replay_buffer)} | Elo Main: {elo_tracker.get_rating('Main'):.0f}")
            
            if visualizer:
                 # Broadcast metrics
                 visualizer.broadcast_metrics(i, 0.0, 0.0) # Will update loss later
                 
                 # Broadcast Elo ratings
                 ghost_name = worker.current_ghost_name
                 ghost_elo = elo_tracker.get_rating(ghost_name) if ghost_name else None
                 top_rankings = elo_tracker.get_rankings()[:5]  # Top 5
                 visualizer.broadcast_elo(
                     main_elo=elo_tracker.get_rating('Main'),
                     ghost_name=ghost_name,
                     ghost_elo=ghost_elo,
                     rankings=augmented_rankings[:5] # Send Top 5 with winrates
                 )
                 visualizer.broadcast_elo_history(elo_tracker.history)
                 
                 # Broadcast enhanced stats
                 visualizer.broadcast_stats(
                     iteration=i, 
                     game=len(replay_buffer), # Buffer size
                     total_games=total_games,
                     avg_game_time=avg_game_time,
                     last_iteration_time=iter_duration,
                     ghost_winrate=ghost_wr
                 )
            
            # Train
            # Training ratio: ~4:1 (train 4x as many samples as generated)
            # With 16 games × ~100 moves = ~1600 samples/iter
            # 50 batches × 128 = 6400 sample updates (reasonable ratio)
            
            if len(replay_buffer) >= 256:
                total_loss = 0
                train_steps = 50  # Reduced from 200 to prevent overfitting
                for _ in range(train_steps):
                    s, p, v = replay_buffer.sample(128) # Batch 128
                    loss, p_loss, v_loss = trainer.train_step(s, p, v)
                    total_loss += loss
                
                avg_loss = total_loss / train_steps
                print(f"  Avg Loss: {avg_loss:.4f}")
                
                if visualizer:
                    elapsed = time.time() - start_time
                    visualizer.broadcast_metrics(i, float(avg_loss), 0.5) # Winrate placeholder
                    visualizer.broadcast_stats(
                        iteration=i,
                        game=len(replay_buffer), # Buffer size
                        loss=float(avg_loss),
                        win_rate=0.0, # Placeholder, actual winrate is complex
                        buffer_size=len(replay_buffer),
                        winner=-1, # N/A for batch
                        elapsed_time=elapsed,
                        avg_game_time=avg_game_time,
                        eta=0,
                        total_games=total_games,
                        effective_games=effective_games,
                        last_iteration_time=iter_duration,
                        ghost_winrate=ghost_wr
                    )
                    
            # Save
            trainer.total_epochs += 1
            if i % 1 == 0:
                trainer.save_checkpoint(main_ckpt_path)
                print("  Saved Checkpoint.")
                
            # Save Ghost less frequently (every 100 iters for meaningful divergence)
            if i % 100 == 0:
                ghost_path = os.path.join(ghost_dir, f"ghost_{i}.pt")
                trainer.save_checkpoint(ghost_path)
                if ghost_path not in ghost_pool:
                    ghost_pool.append(ghost_path)
                print(f"  Saved Ghost: {os.path.basename(ghost_path)}")
            
            # Memory cleanup to prevent leaks
            del batch_data, metrics
            import gc
            gc.collect()
            if device.type == 'mps':
                torch.mps.empty_cache()
            elif device.type == 'cuda':
                torch.cuda.empty_cache()

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        print("Saving Checkpoint...")
        trainer.save_checkpoint(main_ckpt_path)
        elo_tracker.save()
        save_training_stats(stats_path, total_games, effective_games)
        replay_buffer.save(buffer_save_path)
        
        # Save Replay Buffer
        print("Saving Replay Buffer...")
        replay_buffer.save(buffer_save_path)
        print("Done.")
        sys.exit(0)
    
if __name__ == "__main__":
    main()

