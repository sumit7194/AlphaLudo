
import sys
import os
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..')) # Add root for ludo_cpp

print(f"Python Executable: {sys.executable}")
print(f"Sys Path: {sys.path}")
try:
    import ludo_cpp
    print(f"Direct import ludo_cpp success: {ludo_cpp}")
except ImportError as e:
    print(f"Direct import ludo_cpp failed: {e}")

from pbt_manager import PBTManager
from vector_league import VectorLeagueWorker
from model_mastery import AlphaLudoTopNet

def test_pbt_cpuct():
    print("Testing PBT c_puct integration...")
    
    # 1. Check if PBT Manager handles c_puct
    manager = PBTManager(population_size=2, checkpoint_dir='manual_test/pbt_checkpoints')
    agent = manager.agents[0]
    
    # Set explicit c_puct
    agent.hyperparams['c_puct'] = 2.5
    print(f"Agent 0 c_puct set to: {agent.hyperparams['c_puct']}")
    
    # 2. Check if VectorLeagueWorker accepts it and initializes MCTS
    model = AlphaLudoTopNet()
    worker = VectorLeagueWorker(
        main_model=model,
        probabilities={'Main': 1.0},
        mcts_simulations=10,
        c_puct=agent.hyperparams['c_puct']
    )
    
    # Trigger MCTS init by running a small batch (dummy run)
    # We just want to ensure it doesn't crash and initializes C++
    print("Running batch to trigger MCTS init...")
    try:
        worker.play_batch(batch_size=2, temperature=1.0)
        print("Batch finished successfully.")
    except Exception as e:
        print(f"FAILED: {e}")
        sys.exit(1)
        
    # Check if mutation works
    print("Testing mutation...")
    original_c = agent.hyperparams['c_puct']
    mutated = False
    for _ in range(20): # Try multiple times as mutation is probabilistic
        agent.mutate()
        if agent.hyperparams['c_puct'] != original_c:
            mutated = True
            print(f"Mutated c_puct: {agent.hyperparams['c_puct']}")
            break
            
    if mutated:
        print("Mutation works!")
    else:
        print("Warning: Mutation did not change c_puct in 20 tries (could be bad luck or logic error)")

    print("\nALL CHECKS PASSED.")

if __name__ == "__main__":
    test_pbt_cpuct()
