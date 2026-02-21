
import numpy as np
import os

def inspect():
    path = 'checkpoints/td_prod/experience_buffer.npz'
    if not os.path.exists(path):
        print(f"No buffer found at {path}")
        return

    print(f"Loading {path}...")
    data = np.load(path)
    states = data['states']
    rewards = data['rewards']
    dones = data['dones']
    priorities = data['priorities']
    
    # Determine actual size (might be padded if using ring buffer implementation details, 
    # but .npz save usually slices :size. Let's check 'position' to be sure or just use len)
    # The save logic uses states[:self.size], so it is exact.
    size = len(states)
    print(f"Buffer Size: {size}")
    
    # 1. Reward Analysis
    print("\n=== Rewards ===")
    unique, counts = np.unique(rewards, return_counts=True)
    if len(unique) > 20:
        print(f"Unique rewards: {len(unique)}")
        print(f"Min: {unique.min()}, Max: {unique.max()}")
        print(f"Mean: {rewards.mean():.4f}")
    else:
        for val, count in zip(unique, counts):
            print(f"Value {val:.4f}: {count} ({count/size:.1%})")
            
    # 2. Priority Analysis
    print("\n=== Priorities (TD Error) ===")
    print(f"Min: {priorities.min():.6f}")
    print(f"Max: {priorities.max():.6f}")
    print(f"Mean: {priorities.mean():.6f}")
    print(f"std: {priorities.std():.6f}")
    
    # 3. State Analysis (Sample)
    print("\n=== State Samples (First 5) ===")
    for i in range(min(5, size)):
        s = states[i]
        
        # Decode Dice (Channels 12-17)
        dice = 0
        for ch in range(12, 18):
            if s[ch].max() > 0.5:
                dice = ch - 11
                break
        
        # Decode My Tokens (Channels 0-3)
        my_tokens = 0
        for ch in range(4):
            my_tokens += (s[ch] > 0.5).sum()
            
        # Decode Opp Tokens (Channels 4-6)
        opp_tokens_weighted = 0
        for ch in range(4, 7):
            opp_tokens_weighted += s[ch].sum()
        # Density is 0.25 per token
        opp_tokens = opp_tokens_weighted / 0.25
        
        # Decode Score Diff (Channel 18)
        score_diff = s[18].max() * 4.0
        
        print(f"Sample {i}: Dice={dice}, MyTokens={my_tokens}, OppTokens~{opp_tokens:.0f}, ScoreDiff={score_diff:.2f}, R={rewards[i]:.4f}, Done={dones[i]}, P={priorities[i]:.4f}")
        
    # 3b. Next State Dice Analysis
    print("\n=== Next State Dice Analysis (First 100) ===")
    next_states = data['next_states']
    dice_counts = {0: 0, "Non-Zero": 0}
    for i in range(min(100, size)):
        ns = next_states[i]
        has_dice = False
        for ch in range(12, 18):
            if ns[ch].max() > 0.5:
                has_dice = True
                break
        if has_dice:
            dice_counts["Non-Zero"] += 1
        else:
            dice_counts[0] += 1
    print(f"Next States Dice: {dice_counts}")

    # 4. Check for zeros (Empty states)
    print("\n=== Integrity Check ===")
    zero_states = 0
    for i in range(min(1000, size)):
        if states[i].sum() == 0:
            zero_states += 1
    print(f"Empty States (first 1000): {zero_states}")

if __name__ == "__main__":
    inspect()
