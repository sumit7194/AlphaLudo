"""
AlphaLudo Training Pipeline

Main script that orchestrates:
1. Self-Play: Generate training data
2. Training: Update neural network
3. Evaluation: Validate improvement

Features:
- Apple MPS GPU acceleration
- Full checkpoint resume
- Optional visualization
- Progress tracking
"""

import os
import sys
import argparse
import time
import torch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from model import AlphaLudoNet
from self_play import SelfPlayWorker
from replay_buffer import ReplayBuffer
from trainer import Trainer, get_device
from evaluator import evaluate_model_vs_greedy

# Visualizer for stats broadcasting
viz = None


def generate_self_play_data_with_viz(model, num_games, mcts_sims, visualize=False, iteration=0, buffer_size=0):
    """Generate training data, optionally with visualization."""
    global viz
    worker = SelfPlayWorker(model, mcts_simulations=mcts_sims, visualize=visualize)
    all_examples = []
    
    for game_idx in range(num_games):
        # Broadcast stats if visualizer is active
        if viz:
            viz.broadcast_stats(iteration=iteration, game=game_idx+1, buffer_size=buffer_size + len(all_examples))
        
        examples = worker.play_game()
        all_examples.extend(examples)
        print(f"  Game {game_idx + 1}/{num_games}: {len(examples)} examples")
    
    return all_examples


def main(args):
    start_time = time.time()
    
    print("=" * 60)
    print("AlphaLudo Training Pipeline")
    print("=" * 60)
    
    # Initialize model
    device = get_device()
    model = AlphaLudoNet(num_res_blocks=args.res_blocks, num_channels=args.channels)
    print(f"Model: {args.res_blocks} ResBlocks, {args.channels} channels")
    print(f"Device: {device}")
    
    # Initialize components
    replay_buffer = ReplayBuffer(max_size=args.buffer_size)
    trainer = Trainer(model, learning_rate=args.lr, device=device)
    
    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.checkpoint_dir, "checkpoint.pt")
    model_path = os.path.join(args.checkpoint_dir, "best_model.pt")
    
    # Load checkpoint if exists
    start_iteration = 1
    if not args.fresh and os.path.exists(checkpoint_path):
        if trainer.load_checkpoint(checkpoint_path):
            start_iteration = trainer.total_epochs + 1
    
    # Optional visualization
    global viz
    if args.visualize:
        try:
            import visualizer as viz_module
            viz = viz_module.enable_visualization()
            print("Visualization server started at ws://localhost:8765")
        except ImportError:
            print("Warning: Could not start visualizer")
    
    # Training metrics
    best_win_rate = 0.0
    
    # Training loop
    for iteration in range(start_iteration, start_iteration + args.iterations):
        iter_start = time.time()
        
        print(f"\n{'=' * 60}")
        print(f"Iteration {iteration} | LR: {trainer.get_current_lr():.6f}")
        print("=" * 60)
        
        # 1. Self-Play
        print("\n[1/3] Self-Play...")
        model.eval()
        examples = generate_self_play_data_with_viz(
            model, 
            num_games=args.games_per_iter,
            mcts_sims=args.mcts_sims,
            visualize=args.visualize,
            iteration=iteration,
            buffer_size=len(replay_buffer)
        )
        replay_buffer.add(examples)
        print(f"  Buffer size: {len(replay_buffer)}")
        
        # 2. Training
        print("\n[2/3] Training...")
        avg_total, avg_policy, avg_value = trainer.train_epoch(
            replay_buffer,
            batch_size=args.batch_size,
            num_batches=args.train_batches
        )
        print(f"  Loss - Total: {avg_total:.4f}, Policy: {avg_policy:.4f}, Value: {avg_value:.4f}")
        
        # Save checkpoint after each iteration
        trainer.save_checkpoint(checkpoint_path)
        
        # 3. Evaluation (every N iterations)
        if iteration % args.eval_interval == 0:
            print("\n[3/3] Evaluation vs Greedy Bot...")
            model.eval()
            win_rate = evaluate_model_vs_greedy(
                model,
                num_games=args.eval_games,
                mcts_simulations=args.mcts_sims
            )
            print(f"  Win rate vs Greedy: {win_rate * 100:.1f}%")
            
            # Broadcast updated metrics
            if args.visualize and viz:
                # Use current losses (avg_total, etc.) from this iteration
                viz.update_metrics(iteration=iteration, loss=avg_total, win_rate=win_rate)
            
            # Save best model
            if win_rate > best_win_rate:
                best_win_rate = win_rate
                trainer.save_model(model_path)
                print(f"  New best model saved! ({win_rate * 100:.1f}%)")
        
        iter_time = time.time() - iter_start
        print(f"\n  Iteration time: {iter_time:.1f}s")
    
    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"Training complete! Total time: {total_time / 60:.1f} minutes")
    print(f"Best win rate: {best_win_rate * 100:.1f}%")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AlphaLudo Training")
    
    # Model
    parser.add_argument("--res-blocks", type=int, default=8, help="Number of residual blocks")
    parser.add_argument("--channels", type=int, default=64, help="Number of channels")
    
    # Training
    parser.add_argument("--iterations", type=int, default=100, help="Training iterations")
    parser.add_argument("--games-per-iter", type=int, default=10, help="Self-play games per iteration")
    parser.add_argument("--mcts-sims", type=int, default=50, help="MCTS simulations per move")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size")
    parser.add_argument("--train-batches", type=int, default=50, help="Training batches per iteration")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--buffer-size", type=int, default=50000, help="Replay buffer size")
    
    # Evaluation
    parser.add_argument("--eval-interval", type=int, default=10, help="Evaluate every N iterations")
    parser.add_argument("--eval-games", type=int, default=20, help="Games for evaluation")
    
    # Checkpoints
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Checkpoint directory")
    parser.add_argument("--fresh", action="store_true", help="Start fresh (ignore existing checkpoint)")
    
    # Visualization
    parser.add_argument("--visualize", action="store_true", help="Enable game visualization")
    
    args = parser.parse_args()
    main(args)
