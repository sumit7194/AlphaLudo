
import sys
import os
import torch
# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

from model_mastery import AlphaLudoTopNet
from vector_league import VectorLeagueWorker

def test_vector_run():
    print("Initializing Model...")
    model = AlphaLudoTopNet()
    device = torch.device('cpu') # Test on CPU
    model.to(device)
    
    print("Initializing Worker...")
    worker = VectorLeagueWorker(model, {}, mcts_simulations=10, visualize=False)
    
    print("Starting Batch Play (Size=4)...")
    examples = worker.play_batch(batch_size=4, temperature=1.0)
    
    print(f"Finished! Collected {len(examples)} examples.")
    assert len(examples) > 0
    print("Test Passed.")

if __name__ == "__main__":
    test_vector_run()
