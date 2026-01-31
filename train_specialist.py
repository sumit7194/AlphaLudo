import os
import sys
import argparse
import time
import torch
import ludo_cpp

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from model import AlphaLudoNet
from specialist import SpecialistWorker
from replay_buffer import ReplayBuffer
from trainer import Trainer, get_device

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint to train')
    parser.add_argument('--reward-type', type=str, choices=['aggressive', 'rusher', 'defensive', 'blockade'], required=True)
    parser.add_argument('--iterations', type=int, default=50)
    args = parser.parse_args()
    
    # Config
    if args.reward_type == 'aggressive':
        reward_config = {'cut': 1.0}
    elif args.reward_type == 'rusher':
        reward_config = {'home': 1.0} # Increased to match priority
    elif args.reward_type == 'defensive':
        reward_config = {'safe': 0.5} # Reward sitting on Globes
    elif args.reward_type == 'blockade':
        reward_config = {'blockade': 0.5} # Reward stacking
    else:
        reward_config = {}
        
    print(f"Training Specialist: {args.reward_type}")
    print(f"Base Model: {args.model}")
    print(f"Rewards: {reward_config}")
    
    device = get_device()
    
    # Initialize model
    model = AlphaLudoNet(num_res_blocks=8, num_channels=64) # Should match config
    trainer = Trainer(model, device=device)
    
    # Load checkpoint
    if not trainer.load_checkpoint(args.model):
        print(f"Error: Could not load checkpoint {args.model}")
        return
        
    print(f"Loaded model from epoch {trainer.total_epochs}")
    
    # Components
    replay_buffer = ReplayBuffer(max_size=10000)
    
    worker = SpecialistWorker(model, reward_config=reward_config, mcts_simulations=25)
    
    for i in range(args.iterations):
        iter_num = trainer.total_epochs + 1
        print(f"\nExample Iteration {iter_num} (Specialist)...")
        
        # 1. Self Play
        examples = worker.play_game(temperature=1.0)
        
        # SpecialistWorker returns list of (state, policy, value)
        replay_buffer.add(examples)
        print(f"  Generated {len(examples)} examples. Buffer: {len(replay_buffer)}")
        
        # 2. Train
        batch_size = 64
        num_steps = 100
        
        total_loss_sum = 0.0
        
        # Only train if we have enough data (min batch size)
        if len(replay_buffer) >= batch_size:
            for _ in range(num_steps):
                states, policies, values = replay_buffer.sample(batch_size)
                
                # Fix dimensions: values is (B), need (B, 1)
                values = values.unsqueeze(1)
                
                # train_step returns (total_loss, policy_loss, value_loss) as floats
                loss, p_loss, v_loss = trainer.train_step(states, policies, values)
                total_loss_sum += loss
                
            avg_loss = total_loss_sum / num_steps
            print(f"  Avg Loss: {avg_loss:.4f}")
        else:
            print("  Skipping training (buffer too small)")
        
        # 3. Save
        # Increment epoch manually since we are outside standard loop
        trainer.total_epochs += 1
        trainer.save_checkpoint(args.model)
        print(f"  Saved to {args.model}")

if __name__ == "__main__":
    main()
