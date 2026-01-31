#!/usr/bin/env python3
"""
Population-Based Training (PBT) for AlphaLudo.

Trains a population of agents in parallel, periodically evaluating and evolving them.
"""

import os
import sys
import argparse
import time
import itertools
import torch
import torch.optim as optim
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from model_mastery import AlphaLudoTopNet
from src.pbt_manager import PBTManager, PBTAgent
from replay_buffer_mastery import ReplayBufferMastery
from vector_league import VectorLeagueWorker
from training_utils import EloTracker, augment_batch


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def train_agent(agent: PBTAgent, worker: VectorLeagueWorker, 
                replay_buffer: ReplayBufferMastery, device, iterations: int = 10):
    """Train a single agent for some iterations."""
    
    agent.model.to(device)
    agent.model.train()
    
    optimizer = optim.Adam(
        agent.model.parameters(), 
        lr=agent.hyperparams['learning_rate'],
        weight_decay=1e-4
    )
    
    total_loss = 0.0
    
    for i in range(iterations):
        # Generate games
        examples, results = worker.play_batch(batch_size=16, temperature=agent.hyperparams['temperature'])
        
        # Augment
        augmented = augment_batch(examples, augment_probability=0.5)
        replay_buffer.add(augmented)
        
        # Track wins
        for res in results:
            winner = res.get('winner', -1)
            identities = res.get('identities', ['Main'] * 4)
            if winner >= 0 and identities[winner] == 'Main':
                agent.wins += 1
            agent.games_played += 1
        
        # Train
        if len(replay_buffer) >= 256:
            for _ in range(10):
                s, t_idx, p, v = replay_buffer.sample(128)
                s = s.to(device)
                t_idx = t_idx.to(device)
                p = p.to(device)
                v = v.to(device)
                
                spatial_logits, predicted_values = agent.model(s)
                
                # Value loss
                value_loss = F.mse_loss(predicted_values, v.unsqueeze(1))
                
                # Policy loss
                gathered_logits = torch.gather(spatial_logits, 1, t_idx)
                log_probs = F.log_softmax(gathered_logits, dim=1)
                policy_loss = -torch.sum(p * log_probs, dim=1).mean()
                
                loss = value_loss + policy_loss
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
    
    return total_loss / max(1, iterations * 10)


def main():
    parser = argparse.ArgumentParser(description='PBT Training for AlphaLudo')
    parser.add_argument('--population-size', type=int, default=4, help='Number of agents')
    parser.add_argument('--generations', type=int, default=100, help='Number of generations (ignored if --continuous)')
    parser.add_argument('--iterations-per-gen', type=int, default=20, help='Training iterations per generation')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints_pbt', help='Checkpoint directory')
    parser.add_argument('--continuous', action='store_true', help='Run indefinitely until Ctrl+C')
    args = parser.parse_args()
    
    device = get_device()
    print(f"PBT Training on {device}")
    
    # Initialize PBT Manager
    pbt = PBTManager(population_size=args.population_size, checkpoint_dir=args.checkpoint_dir)
    pbt.print_status()
    
    # Shared replay buffer
    replay_buffer = ReplayBufferMastery(max_size=100000)
    
    # Probabilities for training
    probabilities = {
        'Main': 0.50,
        'Ghost': 0.20,
        'Heuristic': 0.10,
        'Aggressive': 0.07,
        'Defensive': 0.07,
        'Racing': 0.06
    }
    
    
    # reward_config = {'cut': 0.10, 'home': 0.25, 'safe': 0.05} # REMOVED: Using pure win/loss
    
    # Choose generation range: infinite if continuous, else fixed count
    if args.continuous:
        print("Running in CONTINUOUS mode. Press Ctrl+C to stop.")
        generation_range = itertools.count()
    else:
        generation_range = range(args.generations)
    
    try:
        for gen in generation_range:
            print(f"\n{'='*50}")
            if args.continuous:
                print(f"Generation {gen + 1} (Continuous Mode)")
            else:
                print(f"Generation {gen + 1}/{args.generations}")
            print('='*50)
        
        # Train each agent
        for agent in pbt.agents:
            print(f"\nTraining Agent {agent.agent_id}...")
            
            # Create list of other agents' checkpoints for ghost pool
            # Use checkpoints from *previous* generation (or current if valid)
            other_agents = [a for a in pbt.agents if a.agent_id != agent.agent_id]
            ghost_pool = [a.checkpoint_path for a in other_agents if os.path.exists(a.checkpoint_path)]
            
            # Create worker with this agent's model
            worker = VectorLeagueWorker(
                main_model=agent.model,
                probabilities=probabilities,
                mcts_simulations=agent.hyperparams['mcts_simulations'],
                visualize=False,
                ghost_pool=ghost_pool,  # Enable ghosts!
                elo_tracker=None,
                temp_schedule='alphazero',
                c_puct=agent.hyperparams.get('c_puct', 1.0)
            )
            
            avg_loss = train_agent(
                agent, worker, replay_buffer, device,
                iterations=args.iterations_per_gen
            )
            
            print(f"  Agent {agent.agent_id}: Avg Loss = {avg_loss:.4f}, "
                  f"Games = {agent.games_played}, Wins = {agent.wins}")
        
        # Evaluate agents against each other
        print("\nEvaluating population...")
        win_rates = pbt.evaluate_round(games_per_pair=4)
        pbt.update_elos(win_rates)
        
        pbt.print_status()
        
        # Evolve: Replace worst with mutated copy of best
        if gen % 5 == 4:  # Every 5 generations
            pbt.evolve()
        
        # Save all agents
        pbt.save_all()

    except KeyboardInterrupt:
        print("\n\nPBT Training interrupted by user. Saving all agents...")
        pbt.save_all()
        print(f"All agents saved. Generation: {pbt.generation}")
    
    print("\n" + "="*50)
    print("PBT Training Complete!")
    print("="*50)
    
    best = pbt.get_best_agent()
    print(f"Best Agent: {best.agent_id} with ELO {best.elo:.0f}")
    print(f"Hyperparams: {best.hyperparams}")


if __name__ == "__main__":
    main()
