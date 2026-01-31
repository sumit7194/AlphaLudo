import sys
import os
import unittest
import numpy as np

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import ludo_cpp
from specialist import SpecialistWorker

class MockModel:
    pass

class TestSpecialist(unittest.TestCase):
    def setUp(self):
        self.worker = SpecialistWorker(model=MockModel(), reward_config={'cut': 1.0, 'home': 0.5})

    def test_cut_detection(self):
        # Create dummy states
        # We need to manually set player positions.
        # ludo_cpp.GameState is mutable via bindings if exposed?
        # If not, we rely on having states where values differ.
        
        # Actually, creating states with specific positions might be hard if C++ doesn't expose setters.
        # But we can assume get_reward_shaping uses public properties.
        # We can mock the state objects!
        
        class MockState:
            def __init__(self, positions, current_player=0):
                self.player_positions = positions
                self.current_player = current_player
                
        # Prev: Opponent (Player 1) at pos 10
        prev_pos = [
            [-1, -1, -1, -1], # P0
            [10, -1, -1, -1], # P1 (Token 0 at 10)
            [-1, -1, -1, -1],
            [-1, -1, -1, -1]
        ]
        
        # Next: Opponent (Player 1) at pos -1 (Cut!)
        curr_pos = [
             [-1, -1, -1, -1],
             [-1, -1, -1, -1], # P1 (Token 0 reset)
             [-1, -1, -1, -1],
             [-1, -1, -1, -1]
        ]
        
        prev = MockState(prev_pos, current_player=0)
        curr = MockState(curr_pos, current_player=1) # Turn passed
        
        reward = self.worker.get_reward_shaping(prev, 0, curr, 0) # Player 0 acted
        self.assertEqual(reward, 1.0, "Should detect CUT and return 1.0")
        
    def test_home_detection(self):
        class MockState:
            def __init__(self, positions, current_player=0):
                self.player_positions = positions
                self.current_player = current_player

        # Prev: My token (Player 0) at 56 (close to home)
        prev_pos = [[56, -1, -1, -1], [-1]*4, [-1]*4, [-1]*4]
        
        # Next: My token at 99 (Home)
        curr_pos = [[99, -1, -1, -1], [-1]*4, [-1]*4, [-1]*4]
        
        prev = MockState(prev_pos, current_player=0)
        curr = MockState(curr_pos, current_player=1)
        
        reward = self.worker.get_reward_shaping(prev, 0, curr, 0)
        self.assertEqual(reward, 0.5, "Should detect HOME and return 0.5")

if __name__ == '__main__':
    unittest.main()
