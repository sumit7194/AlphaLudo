"""
Demo script to test the visualizer.

Run this, then open visualizer.html in a browser and click 'Connect'.
"""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from model import AlphaLudoNet
from self_play import SelfPlayWorker
from visualizer import enable_visualization

print("Starting visualizer server...")
viz = enable_visualization()
time.sleep(1)

print("\n" + "=" * 50)
print("Open visualizer.html in your browser and click 'Connect'")
print("=" * 50 + "\n")

input("Press Enter when ready to start demo game...")

print("Playing demo game with visualization...")
model = AlphaLudoNet()
worker = SelfPlayWorker(model, mcts_simulations=5, visualize=True)
examples = worker.play_game()

print(f"\nGame finished! Generated {len(examples)} training examples.")
print("The visualizer should have shown the game progress.")
