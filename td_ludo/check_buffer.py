import numpy as np

data = np.load('checkpoints/td_prod/experience_buffer.npz')
states = data['states']
ns = data['next_states']
r = data['rewards']
d = data['dones']
p = data['priorities']
size = len(states)
print(f"Buffer Size: {size}")

# Check dice in states
dice_zero = 0
dice_nonzero = 0
dice_dist = [0]*7  # 0-6
for i in range(min(500, size)):
    found = 0
    for ch in range(12, 18):
        if states[i][ch].max() > 0.5:
            found = ch - 11
            break
    dice_dist[found] += 1
    if found > 0:
        dice_nonzero += 1
    else:
        dice_zero += 1

print(f"\n=== States Dice Distribution (first 500) ===")
print(f"Dice=0: {dice_zero}  |  Dice>0: {dice_nonzero}")
for d_val in range(1, 7):
    print(f"  Dice={d_val}: {dice_dist[d_val]}")

# Check dice in next_states
ns_dice_zero = 0
ns_dice_nonzero = 0
for i in range(min(500, size)):
    found = False
    for ch in range(12, 18):
        if ns[i][ch].max() > 0.5:
            found = True
            break
    if found:
        ns_dice_nonzero += 1
    else:
        ns_dice_zero += 1

print(f"\n=== NextStates Dice (first 500) ===")
print(f"Dice=0: {ns_dice_zero}  |  Dice>0: {ns_dice_nonzero}")

# Rewards
print(f"\n=== Rewards ===")
print(f"Mean: {r.mean():.4f}, Std: {r.std():.4f}")
print(f"Min: {r.min():.4f}, Max: {r.max():.4f}")
print(f">0: {(r>0).sum()}, <0: {(r<0).sum()}, ==0: {(r==0).sum()}")

# Dones
print(f"\n=== Terminal States ===")
print(f"Terminal: {(d>0).sum()} ({(d>0).sum()/size*100:.1f}%)")

# Priorities
print(f"\n=== Priorities ===")
print(f"Mean: {p.mean():.4f}, Std: {p.std():.4f}")
print(f"Min: {p.min():.6f}, Max: {p.max():.4f}")
