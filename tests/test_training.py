"""
Tests for Phase 4: Training Loop components.
"""

import sys
import os
import torch
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

import ludo_cpp
from model import AlphaLudoNet
from self_play import SelfPlayWorker
from replay_buffer import ReplayBuffer
from trainer import Trainer
from evaluator import GreedyBot, evaluate_model_vs_greedy


def test_greedy_bot():
    """Test that Greedy Bot can play a valid game."""
    bot = GreedyBot()
    state = ludo_cpp.create_initial_state()
    
    for _ in range(100):  # Play up to 100 moves
        if state.is_terminal:
            break
        
        state.current_dice_roll = np.random.randint(1, 7)
        legal_moves = ludo_cpp.get_legal_moves(state)
        
        if len(legal_moves) == 0:
            state.current_player = (state.current_player + 1) % 4
            state.current_dice_roll = 0
            continue
        
        action = bot.select_action(state)
        assert action is not None, "Greedy bot should return an action"
        assert action in legal_moves, f"Action {action} not in legal moves {legal_moves}"
        
        state = ludo_cpp.apply_move(state, action)
    
    print("✓ Greedy Bot test passed.")


def test_replay_buffer():
    """Test replay buffer add and sample."""
    buffer = ReplayBuffer(max_size=100)
    
    # Add some dummy examples
    for i in range(50):
        state = torch.randn(8, 15, 15)
        policy = torch.tensor([0.25, 0.25, 0.25, 0.25])
        value = torch.tensor([1.0 if i % 2 == 0 else -1.0])
        buffer.add([(state, policy, value)])
    
    assert len(buffer) == 50, f"Expected 50 items, got {len(buffer)}"
    
    # Sample batch
    states, policies, values = buffer.sample(16)
    assert states.shape == (16, 8, 15, 15)
    assert policies.shape == (16, 4)
    assert values.shape == (16, 1)
    
    print("✓ Replay Buffer test passed.")


def test_trainer():
    """Test single training step."""
    model = AlphaLudoNet()
    trainer = Trainer(model, learning_rate=0.001)
    
    # Create dummy batch
    states = torch.randn(4, 8, 15, 15)
    policies = torch.softmax(torch.randn(4, 4), dim=1)
    values = torch.randn(4, 1)
    
    # Run training step
    total_loss, policy_loss, value_loss = trainer.train_step(states, policies, values)
    
    assert total_loss > 0, "Loss should be positive"
    assert not np.isnan(total_loss), "Loss should not be NaN"
    
    print(f"✓ Trainer test passed. Loss: {total_loss:.4f}")


def test_self_play_short():
    """Test self-play generates valid examples (short game)."""
    model = AlphaLudoNet()
    worker = SelfPlayWorker(model, mcts_simulations=5, temperature_threshold=10)
    
    # Play one game
    examples = worker.play_game()
    
    assert len(examples) > 0, "Should generate some examples"
    
    # Check example structure
    state, policy, value = examples[0]
    assert state.shape == (8, 15, 15), f"State shape wrong: {state.shape}"
    assert len(policy) == 4, f"Policy length wrong: {len(policy)}"
    assert abs(policy.sum() - 1.0) < 0.01, "Policy should sum to 1"
    assert -1.0 <= value.item() <= 1.0, "Value should be in [-1, 1]"
    
    print(f"✓ Self-Play test passed. Generated {len(examples)} examples.")


def test_full_pipeline_mini():
    """Test the full pipeline with minimal settings."""
    model = AlphaLudoNet()
    buffer = ReplayBuffer(max_size=1000)
    trainer = Trainer(model)
    
    # Generate a tiny bit of data
    worker = SelfPlayWorker(model, mcts_simulations=5)
    examples = worker.play_game()
    buffer.add(examples)
    
    if len(buffer) >= 4:
        # Train one batch
        total, policy, value = trainer.train_step(*buffer.sample(4))
        print(f"✓ Full pipeline test passed. Loss: {total:.4f}")
    else:
        print(f"✓ Full pipeline test passed (not enough data for training).")


if __name__ == "__main__":
    print("Running Phase 4 Tests...\n")
    test_greedy_bot()
    test_replay_buffer()
    test_trainer()
    test_self_play_short()
    test_full_pipeline_mini()
    print("\n🎉 All Phase 4 tests passed!")
