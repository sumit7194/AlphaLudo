import unittest
import ludo_cpp
import numpy as np

class TestCPPMCTS(unittest.TestCase):
    def test_mcts_engine_basics(self):
        batch_size = 4
        engine = ludo_cpp.MCTSEngine(batch_size)
        
        # Create states
        states = []
        for _ in range(batch_size):
            s = ludo_cpp.create_initial_state()
            # Set mock dice roll
            s.current_dice_roll = 6
            states.append(s)
            
        engine.set_roots(states)
        
        # Step 1: Selection
        leaves = engine.select_leaves()
        self.assertEqual(len(leaves), batch_size)
        
        # Step 2: Expand
        # Fake Policy [batch, 4] and Value [batch]
        # Use simple policy: prefer move 0
        policies = np.zeros((batch_size, 4), dtype=np.float32)
        policies[:, 0] = 5.0 # High logit for move 0
        
        values = np.zeros(batch_size, dtype=np.float32) # Neutral value
        
        engine.expand_and_backprop(policies, values)
        
        # Check roots expanded
        stats = engine.get_root_stats()
        for visits, val in stats:
            self.assertEqual(visits, 1) # visited via backprop? 
            # root visited once during backprop.
            
        # Step 3: Another simulation
        leaves2 = engine.select_leaves()
        # Should now select children
        # Since we expanded, it should go deeper.
        # But wait, children are leaves now.
        # So it returns children of roots.
        self.assertEqual(len(leaves2), batch_size)
        
        # We need to provide values for children
        engine.expand_and_backprop(policies, values)
        
        stats2 = engine.get_root_stats()
        for visits, val in stats2:
            self.assertEqual(visits, 2)
            
        # Step 4: Action Probs
        probs = engine.get_action_probs(1.0)
        self.assertEqual(len(probs), batch_size)
        for p in probs:
            self.assertAlmostEqual(sum(p), 1.0, places=5)
            # Should prefer move 0 due to policy prior
            self.assertGreater(p[0], p[1])

if __name__ == '__main__':
    unittest.main()
