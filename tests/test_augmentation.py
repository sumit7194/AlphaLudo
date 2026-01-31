"""
Comprehensive test for rotation augmentation in AlphaLudo.

This test verifies that:
1. State tensor rotation matches token index rotation
2. After rotation, the model sees the same relative positions
3. 4× rotation returns to original
"""

import torch
import numpy as np
import sys
sys.path.insert(0, 'src')

from training_utils import rotate_state_tensor, rotate_token_indices, rotate_channels, augment_batch
from tensor_utils import BOARD_SIZE

def test_token_index_rotation():
    """Test that token indices rotate correctly."""
    print("=" * 60)
    print("TEST 1: Token Index Rotation")
    print("=" * 60)
    
    # Test specific positions
    test_cases = [
        # (flat_idx, expected_after_90ccw)
        (0, 210),      # (0,0) -> (14,0)
        (14, 0),       # (0,14) -> (0,0)
        (224, 14),     # (14,14) -> (0,14)
        (210, 224),    # (14,0) -> (14,14)
        (112, 112),    # (7,7) center stays center
    ]
    
    all_passed = True
    for flat_idx, expected in test_cases:
        tensor = torch.tensor([flat_idx], dtype=torch.long)
        rotated = rotate_token_indices(tensor, k=1)
        actual = rotated[0].item()
        
        # Decode for display
        orig_r, orig_c = flat_idx // 15, flat_idx % 15
        rot_r, rot_c = actual // 15, actual % 15
        exp_r, exp_c = expected // 15, expected % 15
        
        passed = actual == expected
        status = "✓" if passed else "✗"
        print(f"  {status} ({orig_r},{orig_c}) -> ({rot_r},{rot_c}) [expected ({exp_r},{exp_c})]")
        
        if not passed:
            all_passed = False
    
    # Test 4× rotation = identity
    test_indices = torch.tensor([0, 14, 112, 210, 224], dtype=torch.long)
    rotated_4x = rotate_token_indices(test_indices, k=4)
    identity_passed = torch.equal(test_indices, rotated_4x)
    print(f"  {'✓' if identity_passed else '✗'} 4× rotation = identity")
    
    return all_passed and identity_passed


def test_state_tensor_and_token_consistency():
    """
    Test that rotating a state tensor and token indices gives consistent data.
    
    If we place a marker at token position, rotate both state and token index,
    the marker should still be at the token position.
    """
    print("\n" + "=" * 60)
    print("TEST 2: State Tensor ↔ Token Index Consistency")
    print("=" * 60)
    
    all_passed = True
    
    for test_idx in [0, 37, 112, 187, 224]:
        # Create a state tensor with a marker at position test_idx
        state = torch.zeros(12, 15, 15)
        r, c = test_idx // 15, test_idx % 15
        state[0, r, c] = 1.0  # Place marker in channel 0
        
        token_indices = torch.tensor([test_idx, 0, 0, 0], dtype=torch.long)
        
        for k in [1, 2, 3]:
            # Rotate both
            rot_state = rotate_channels(state, k)
            rot_indices = rotate_token_indices(token_indices, k)
            
            # Check: the marker should be at the new token position
            new_idx = rot_indices[0].item()
            new_r, new_c = new_idx // 15, new_idx % 15
            
            marker_value = rot_state[0, new_r, new_c].item()
            passed = marker_value == 1.0
            
            status = "✓" if passed else "✗"
            print(f"  {status} Marker at idx {test_idx} ({r},{c}), rotate {k*90}° CCW -> idx {new_idx} ({new_r},{new_c}), found={marker_value}")
            
            if not passed:
                all_passed = False
                # Find where marker actually is
                nonzero = torch.nonzero(rot_state[0])
                if len(nonzero) > 0:
                    actual_r, actual_c = nonzero[0].tolist()
                    actual_idx = actual_r * 15 + actual_c
                    print(f"      Marker actually at ({actual_r},{actual_c}) = idx {actual_idx}")
    
    return all_passed


def test_augment_batch():
    """Test the full augment_batch pipeline."""
    print("\n" + "=" * 60)
    print("TEST 3: Augment Batch Pipeline")
    print("=" * 60)
    
    # Create a fake training example with a marker
    state = torch.zeros(12, 15, 15)
    state[0, 3, 5] = 1.0  # Marker at (3, 5) = idx 50
    
    token_indices = torch.tensor([50, 0, 112, 224], dtype=torch.long)
    policy = torch.tensor([0.5, 0.2, 0.2, 0.1])
    value = torch.tensor(1.0)
    
    # Create batch with single example
    examples = [(state, token_indices, policy, value)]
    
    # Run augmentation with 100% probability to force rotation
    np.random.seed(42)  # For reproducibility
    augmented = augment_batch(examples, augment_probability=1.0)
    
    aug_state, aug_indices, aug_policy, aug_value = augmented[0]
    
    # Check consistency after augmentation
    # The marker should be at the new token position
    first_idx = aug_indices[0].item()
    first_r, first_c = first_idx // 15, first_idx % 15
    
    marker_value = aug_state[0, first_r, first_c].item()
    passed = marker_value == 1.0
    
    print(f"  Original: marker at (3,5), token idx 50")
    print(f"  Augmented: token idx {first_idx} ({first_r},{first_c}), marker value at that pos = {marker_value}")
    print(f"  {'✓' if passed else '✗'} Consistency maintained after augmentation")
    
    return passed


def test_policy_invariance():
    """Test that policy stays the same (since actions are relative to current player)."""
    print("\n" + "=" * 60)
    print("TEST 4: Policy Invariance")
    print("=" * 60)
    
    policy = torch.tensor([0.4, 0.3, 0.2, 0.1])
    
    from training_utils import rotate_policy
    
    all_passed = True
    for k in [1, 2, 3]:
        rotated = rotate_policy(policy, k)
        passed = torch.equal(policy, rotated)
        print(f"  {'✓' if passed else '✗'} Policy unchanged after {k*90}° rotation")
        if not passed:
            all_passed = False
            print(f"      Original: {policy.tolist()}")
            print(f"      Rotated:  {rotated.tolist()}")
    
    return all_passed


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("AUGMENTATION VERIFICATION TEST SUITE")
    print("=" * 60 + "\n")
    
    results = []
    results.append(("Token Index Rotation", test_token_index_rotation()))
    results.append(("State ↔ Token Consistency", test_state_tensor_and_token_consistency()))
    results.append(("Augment Batch Pipeline", test_augment_batch()))
    results.append(("Policy Invariance", test_policy_invariance()))
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {name}")
        if not passed:
            all_passed = False
    
    print()
    if all_passed:
        print("🎉 All tests passed! Augmentation is CORRECT.")
    else:
        print("❌ Some tests failed. Review the output above.")
    
    sys.exit(0 if all_passed else 1)
