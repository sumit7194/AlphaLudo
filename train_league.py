import os
import sys
import argparse
import time
import torch
import ludo_cpp
from collections import deque

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from model import AlphaLudoNet
from league import LeagueWorker
from replay_buffer import ReplayBuffer
from trainer import Trainer, get_device
from src.visualizer import visualizer, enable_visualization

def load_model(path, device):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}")
    model = AlphaLudoNet(num_res_blocks=8, num_channels=64)
    checkpoint = torch.load(path, map_location=device)
    # Handle both full checkpoint dict and direct state_dict
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # Assuming direct state_dict or raw save
        try:
           model.load_state_dict(checkpoint)
        except:
           print(f"Warning: Could not load {path} as standard state_dict.")
    model.to(device)
    model.eval()
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--main-model', type=str, required=True, help='Path to baseline model (will be updated)')
    parser.add_argument('--specialists', nargs='+', help='List of specialist paths (e.g. agg.pt rush.pt)')
    parser.add_argument('--iterations', type=int, default=50)
    args = parser.parse_args()
    
    device = get_device()
    print(f"League Training initialized on {device}")
    
    # 1. Load Main Model (Trainable)
    main_model = AlphaLudoNet(num_res_blocks=8, num_channels=64)
    trainer = Trainer(main_model, device=device)
    if not trainer.load_checkpoint(args.main_model):
        print("Starting Main Model from scratch (or failed load)")
    else:
        print(f"Loaded Main Model: {args.main_model} (Epoch {trainer.total_epochs})")
        
    # 2. Load Specialists (Frozen)
    specialist_pool = {}
    specialist_names = ['Aggressive', 'Rusher', 'Defensive', 'Blockade'] # Default mapping if 4 provided
    
    if args.specialists:
        for idx, path in enumerate(args.specialists):
            name = specialist_names[idx] if idx < len(specialist_names) else f"Spec_{idx}"
            print(f"Loading Specialist [{name}] from {path}...")
            spec_model = load_model(path, device)
            specialist_pool[name] = spec_model
            
    # 3. Setup Probabilities
    # 60% Main, 40% Specialists distributed evenly
    probabilities = {'Main': 0.6}
    if len(specialist_pool) > 0:
        spec_prob = 0.4 / len(specialist_pool)
        for name in specialist_pool:
            probabilities[name] = spec_prob
    else:
        probabilities['Main'] = 1.0 # Pure self play if no specialists
        
    print(f"Matchmaking Probabilities: {probabilities}")
    
    # 4. Components
    replay_buffer = ReplayBuffer(max_size=20000) # Larger buffer for variety
    worker = LeagueWorker(main_model, specialist_pool, probabilities, mcts_simulations=25, visualize=True)
    
    # Start Visualizer
    # Start Visualizer
    if visualizer:
        print(f"Main visualizer instance: {visualizer} (ID: {hex(id(visualizer))})")
        visualizer.start_server()
        print("Visualizer server started on ws://localhost:8765")
        
    # 5. Training Loop
    league_wins = {name: 0 for name in probabilities.keys()}
    league_played = {name: 0 for name in probabilities.keys()}
    
    start_iter = trainer.total_epochs
    for i in range(args.iterations):
        iter_num = trainer.total_epochs + 1
        print(f"\nLeague Iteration {iter_num}...")
        
        # Play Game
        # Play Game
        examples, winner, identities = worker.play_game(temperature=1.0) # Always explore a bit? Or decay?
        replay_buffer.add(examples)
        
        # Calculate Stats
        # Update Played
        for model_name in identities:
             league_played[model_name] = league_played.get(model_name, 0) + 1
        
        # Update Wins
        if winner != -1:
            winner_model = identities[winner]
            league_wins[winner_model] = league_wins.get(winner_model, 0) + 1
            
        print(f"  Game Result: Winner P{winner} ({identities[winner] if winner != -1 else 'Draw'})")
        
        # Broadcast League Stats
        if visualizer:
            stats_data = {}
            all_models = set(league_wins.keys()) | set(league_played.keys())
            for m in all_models:
                stats_data[m] = {
                    'wins': league_wins.get(m, 0),
                    'played': league_played.get(m, 0)
                }
            visualizer.broadcast_league_stats(stats_data)
        
        print(f"  Generated {len(examples)} examples. Buffer: {len(replay_buffer)}")
        
        # Train
        if len(replay_buffer) >= 64:
            total_loss = 0
            steps = 100
            for _ in range(steps):
                states, policies, values = replay_buffer.sample(64)
                values = values.unsqueeze(1)
                loss, p, v = trainer.train_step(states, policies, values)
                total_loss += loss
            
            avg_loss = total_loss / steps
            print(f"  Avg Loss: {avg_loss:.4f}")
            
            # Broadcast Stats (Training Progress + enables Session Wins)
            if visualizer:
                visualizer.broadcast_stats(
                    iteration=iter_num,
                    game=i + 1,
                    loss=float(avg_loss),
                    win_rate=0.0,
                    buffer_size=len(replay_buffer),
                    winner=winner
                )
                visualizer.broadcast_metrics(iter_num, float(avg_loss), 0.0)
        
        # Save
        trainer.total_epochs += 1
        if i % 5 == 0:
            trainer.save_checkpoint(args.main_model)
            print(f"  Saved checkpoint.")
    
    # Final save to ensure all progress is persisted
    trainer.save_checkpoint(args.main_model)
    print(f"\n=== Training Complete. Final checkpoint saved at epoch {trainer.total_epochs}. ===")

if __name__ == "__main__":
    main()
