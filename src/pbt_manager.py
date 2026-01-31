"""
Population-Based Training (PBT) Manager for AlphaLudo.

Implements:
- Multiple parallel agents training
- Periodic evaluation between agents
- Mutation/selection mechanism
- Hyperparameter evolution
"""

import os
import json
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from model_mastery import AlphaLudoTopNet
import ludo_cpp


class PBTAgent:
    """Single agent in the population."""
    
    def __init__(self, agent_id: int, checkpoint_dir: str):
        self.agent_id = agent_id
        self.checkpoint_dir = checkpoint_dir
        self.model = AlphaLudoTopNet(num_res_blocks=10, num_channels=128)
        self.elo = 1500.0
        self.games_played = 0
        self.wins = 0
        
        # Hyperparameters that can be mutated
        self.hyperparams = {
            'learning_rate': 0.001,
            'temperature': 1.0,
            'mcts_simulations': 200,
            'c_puct': 1.0,  # Added exploration constant
        }
        
        self.checkpoint_path = os.path.join(checkpoint_dir, f'agent_{agent_id}.pt')
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    def save(self):
        """Save agent state."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'elo': self.elo,
            'games_played': self.games_played,
            'wins': self.wins,
            'hyperparams': self.hyperparams,
        }, self.checkpoint_path)
    
    def load(self):
        """Load agent state if exists."""
        if os.path.exists(self.checkpoint_path):
            checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.elo = checkpoint.get('elo', 1500.0)
            self.games_played = checkpoint.get('games_played', 0)
            self.wins = checkpoint.get('wins', 0)
            self.hyperparams = checkpoint.get('hyperparams', self.hyperparams)
            return True
        return False
    
    def copy_from(self, other: 'PBTAgent'):
        """Copy weights from another agent."""
        self.model.load_state_dict(other.model.state_dict())
        self.hyperparams = other.hyperparams.copy()
    
    def mutate(self, mutation_strength: float = 0.2):
        """Mutate hyperparameters."""
        for key in self.hyperparams:
            if np.random.random() < 0.5:  # 50% chance to mutate each param
                if key == 'learning_rate':
                    factor = np.random.choice([0.8, 1.25])  # Up or down by 20-25%
                    self.hyperparams[key] *= factor
                    self.hyperparams[key] = np.clip(self.hyperparams[key], 1e-5, 0.01)
                elif key == 'temperature':
                    self.hyperparams[key] += np.random.uniform(-0.2, 0.2)
                    self.hyperparams[key] = np.clip(self.hyperparams[key], 0.1, 2.0)
                elif key == 'mcts_simulations':
                    factor = np.random.choice([0.8, 1.25])
                    self.hyperparams[key] = int(self.hyperparams[key] * factor)
                    self.hyperparams[key] = max(50, min(400, self.hyperparams[key]))
                elif key == 'c_puct':
                    self.hyperparams[key] += np.random.uniform(-0.5, 0.5)
                    self.hyperparams[key] = np.clip(self.hyperparams[key], 0.1, 5.0)


class PBTManager:
    """
    Manages a population of agents for Population-Based Training.
    """
    
    def __init__(self, population_size: int = 4, checkpoint_dir: str = 'checkpoints_pbt'):
        self.population_size = population_size
        self.checkpoint_dir = checkpoint_dir
        self.agents: List[PBTAgent] = []
        
        # Initialize agents
        for i in range(population_size):
            agent = PBTAgent(i, checkpoint_dir)
            if not agent.load():
                print(f"Created new agent {i}")
            else:
                print(f"Loaded agent {i} (ELO: {agent.elo:.0f})")
            self.agents.append(agent)
        
        # Stats
        self.generation = 0
        self.stats_path = os.path.join(checkpoint_dir, 'pbt_stats.json')
        self.load_stats()
    
    def evaluate_round(self, games_per_pair: int = 4) -> Dict[int, float]:
        """
        Run evaluation games between all agent pairs using ACTUAL games.
        Returns dict of agent_id -> win_rate
        """
        from itertools import combinations
        from vector_league import VectorLeagueWorker # lazy import
        
        # Ensure latest checkpoints
        self.save_all()
        
        results = {i: {'wins': 0, 'games': 0} for i in range(self.population_size)}
        
        # reward_config = {'cut': 0.10, 'home': 0.25, 'safe': 0.05} # REMOVED: Pure win/loss
        
        # Determine device
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

        # Evaluate every pair
        for i, j in combinations(range(self.population_size), 2):
            # We play 2 rounds to swap roles creates a balanced eval:
            # Round 1: Main=i (3 copies), Ghost=j (1 copy)
            # Round 2: Main=j (3 copies), Ghost=i (1 copy)
            
            # Since VectorLeagueWorker runs batches, we use batch_size=games_per_pair (e.g., 4)
            # This results in 8 total games per pair (4 in each configuration)
            
            # --- Config A: i is Main, j is Ghost ---
            probs = {
                'Main': 0.0, 
                'Ghost': 1.0,
                'Heuristic': 0.0,
                'Aggressive': 0.0,
                'Defensive': 0.0,
                'Racing': 0.0
            }

            worker_a = VectorLeagueWorker(
                main_model=self.agents[i].model,
                probabilities=probs, # Force all games to have ghosts
                mcts_simulations=50, # Fast evaluation
                visualize=False,
                ghost_pool=[self.agents[j].checkpoint_path],
                elo_tracker=None
            )
            
            batch_size = max(4, games_per_pair)
            _, res_a = worker_a.play_batch(batch_size=batch_size, temperature=1.0)
            
            for r in res_a:
                w = r['winner']
                ids = r['identities'] # e.g. ['Main', 'agent_j', 'Main', 'Main']
                if w != -1:
                    winner_id = ids[w]
                    if winner_id == 'Main':
                        results[i]['wins'] += 1
                    else:
                        results[j]['wins'] += 1
                    results[i]['games'] += 1
                    results[j]['games'] += 1

            # --- Config B: j is Main, i is Ghost ---
            worker_b = VectorLeagueWorker(
                main_model=self.agents[j].model,
                probabilities=probs, # Reuse explicit probs
                mcts_simulations=50,
                visualize=False,
                ghost_pool=[self.agents[i].checkpoint_path],
                elo_tracker=None
            )
            
            _, res_b = worker_b.play_batch(batch_size=batch_size, temperature=1.0)
            
            for r in res_b:
                w = r['winner']
                ids = r['identities']
                if w != -1:
                    winner_id = ids[w]
                    if winner_id == 'Main':
                        results[j]['wins'] += 1
                    else:
                        results[i]['wins'] += 1
                    results[i]['games'] += 1
                    results[j]['games'] += 1
            
            print(f"  Eval {i} vs {j}: {results[i]['wins']} - {results[j]['wins']} (in {results[i]['games']} games)")

        # Calculate win rates
        win_rates = {}
        for agent_id, r in results.items():
            if r['games'] > 0:
                win_rates[agent_id] = r['wins'] / r['games']
            else:
                win_rates[agent_id] = 0.5
        
        return win_rates
    
    def evolve(self):
        """
        Replace worst agent with mutated copy of best.
        """
        # Rank by ELO
        sorted_agents = sorted(self.agents, key=lambda a: a.elo, reverse=True)
        
        best_agent = sorted_agents[0]
        worst_agent = sorted_agents[-1]
        
        if best_agent.agent_id != worst_agent.agent_id:
            print(f"Evolving: Agent {worst_agent.agent_id} (ELO {worst_agent.elo:.0f}) "
                  f"← Agent {best_agent.agent_id} (ELO {best_agent.elo:.0f})")
            
            # Copy weights from best to worst
            worst_agent.copy_from(best_agent)
            
            # Mutate hyperparameters
            worst_agent.mutate()
            
            # Reset stats
            worst_agent.elo = best_agent.elo - 50  # Start slightly lower
            worst_agent.games_played = 0
            worst_agent.wins = 0
        
        self.generation += 1
    
    def update_elos(self, win_rates: Dict[int, float]):
        """Update ELOs based on win rates."""
        for agent_id, win_rate in win_rates.items():
            # Simple ELO update based on performance
            expected = 0.5  # Average
            actual = win_rate
            k = 32
            
            self.agents[agent_id].elo += k * (actual - expected)
    
    def get_best_agent(self) -> PBTAgent:
        """Return the best performing agent."""
        return max(self.agents, key=lambda a: a.elo)
    
    def save_all(self):
        """Save all agents."""
        for agent in self.agents:
            agent.save()
        self.save_stats()
    
    def save_stats(self):
        """Save PBT statistics."""
        stats = {
            'generation': self.generation,
            'agents': [
                {
                    'id': a.agent_id,
                    'elo': a.elo,
                    'games_played': a.games_played,
                    'wins': a.wins,
                    'hyperparams': a.hyperparams,
                }
                for a in self.agents
            ]
        }
        with open(self.stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
    
    def load_stats(self):
        """Load PBT statistics."""
        if os.path.exists(self.stats_path):
            with open(self.stats_path, 'r') as f:
                stats = json.load(f)
            self.generation = stats.get('generation', 0)
    
    def print_status(self):
        """Print population status."""
        print(f"\n=== PBT Generation {self.generation} ===")
        for agent in sorted(self.agents, key=lambda a: a.elo, reverse=True):
            print(f"  Agent {agent.agent_id}: ELO={agent.elo:.0f}, "
                  f"LR={agent.hyperparams['learning_rate']:.5f}, "
                  f"Sims={agent.hyperparams['mcts_simulations']}, "
                  f"CPUCT={agent.hyperparams.get('c_puct', 1.0):.2f}")
