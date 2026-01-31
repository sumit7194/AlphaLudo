import sys
import os
import time
import random
import threading

# Add src to path
sys.path.insert(0, os.path.abspath("src"))

from visualizer import visualizer

def simulate_training():
    print("Starting simulation...")
    visualizer.start_server(port=8765)
    
    # Simulate some initial history (mimic list of tuples from EloTracker)
    elo_history = {
        'Main': [[i, 1500 + i*10] for i in range(10)],
        'Ghost_v1': [[i, 1400 + i*5] for i in range(10)]
    }
    
    metrics_history = [
        {'iteration': i, 'loss': 0.5 - i*0.01, 'win_rate': 50 + i} 
        for i in range(10)
    ]
    
    print("Broadcasting initial history...")
    visualizer.broadcast_elo_history(elo_history)
    visualizer.metrics_history = metrics_history # Manually set for this test or use broadcast
    # broadcast_metrics appends, so let's just append loop
    visualizer.metrics_history = []
    for m in metrics_history:
        visualizer.broadcast_metrics(m['iteration'], m['loss'], m['win_rate']/100)
    
    print("Server running. Waiting for browser verification...")
    # Keep running
    while True:
        time.sleep(1)

if __name__ == "__main__":
    simulate_training()
