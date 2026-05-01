"""Compare V13 (G=33K) policy/value vs V12.2 on the same state distribution.

Generates states by V13 self-play, evaluates both models on each state, and
reports:
  - Top-1 agreement rate (do they pick the same token?)
  - Mean KL(V13 || V12.2)
  - Mean abs(value_V13 - value_V12.2)
  - Per game-phase breakdown (early/mid/late)

Tells us:
  - How far V13 has drifted from V12.2 → predicts how aggressive KL anchoring needs to be.
  - Whether the value heads disagree → predicts if we should distill the value head too.
"""
import os, sys, random
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import td_ludo_cpp as cpp
from td_ludo.models.v12 import AlphaLudoV12

DEVICE = torch.device('cpu')

def load_v13(path):
    m = AlphaLudoV12(num_res_blocks=4, num_channels=96, num_attn_layers=2,
                    num_heads=4, ffn_ratio=4, dropout=0.0, in_channels=28)
    # V13 uses MinimalCNN14 — different arch.
    from experiments.distillation_14ch.model_14ch import MinimalCNN14
    m = MinimalCNN14(num_res_blocks=10, num_channels=128, in_channels=14)
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    sd = ckpt.get('model_state_dict', ckpt)
    sd = {k.replace('_orig_mod.', ''): v for k, v in sd.items()}
    m.load_state_dict(sd); m.eval(); return m

def load_v12_2(path):
    m = AlphaLudoV12(num_res_blocks=3, num_channels=128, num_attn_layers=2,
                     num_heads=4, ffn_ratio=4, dropout=0.0, in_channels=33)
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    sd = ckpt.get('model_state_dict', ckpt)
    sd = {k.replace('_orig_mod.', ''): v for k, v in sd.items()}
    m.load_state_dict(sd); m.eval(); return m

def encode_v14(state):
    return np.asarray(cpp.encode_state_v14_minimal(state))

def encode_v11(state):
    return np.asarray(cpp.encode_state_v11(state))

def collect_states(v13_model, n=2000, seed=0):
    """Run V13 self-play (greedy) and collect snapshots before each move."""
    rng = random.Random(seed)
    states = []
    while len(states) < n:
        s = cpp.create_initial_state_2p()
        moves = 0
        while not s.is_terminal and moves < 500:
            cp = s.current_player
            if s.current_dice_roll == 0:
                s.current_dice_roll = rng.randint(1, 6)
            legal = cpp.get_legal_moves(s)
            if not legal:
                nxt = (cp + 1) % 4
                while not s.active_players[nxt]: nxt = (nxt + 1) % 4
                s.current_player = nxt; s.current_dice_roll = 0
                continue
            # Snapshot before V13 picks
            if len(legal) > 1 and len(states) < n:
                states.append({'state_v14': encode_v14(s), 'state_v11': encode_v11(s),
                               'legal': legal, 'phase': _phase(s)})
            # V13 picks (greedy)
            with torch.no_grad():
                t = torch.from_numpy(encode_v14(s)).unsqueeze(0).float()
                mask = torch.zeros(1, 4)
                for m in legal: mask[0, m] = 1.0
                out = v13_model(t, mask)
                logits = out[0]
                logits = logits.masked_fill(mask == 0, -1e9)
                a = int(logits.argmax(1).item())
            s = cpp.apply_move(s, a)
            moves += 1
    return states[:n]

def _phase(s):
    pos = np.array(s.player_positions[s.current_player], dtype=int)
    n_out = int(np.sum((pos != -1) & (pos != 99)))
    if n_out <= 1: return 'early'
    if n_out == 2: return 'mid'
    return 'late'

def evaluate_pair(v13, v12_2, states):
    """For each state: get V13 + V12.2 policy and value, compare."""
    rows = []
    for st in states:
        legal = st['legal']
        mask_np = np.zeros(4, dtype=np.float32)
        for m in legal: mask_np[m] = 1.0
        mask = torch.from_numpy(mask_np).unsqueeze(0)
        with torch.no_grad():
            t13 = torch.from_numpy(st['state_v14']).unsqueeze(0).float()
            o13 = v13(t13, mask)
            p13 = torch.softmax(o13[0].masked_fill(mask == 0, -1e9), dim=1).squeeze().numpy()
            v13_v = float(o13[1].item())

            t12 = torch.from_numpy(st['state_v11']).unsqueeze(0).float()
            o12 = v12_2(t12, mask)
            p12 = torch.softmax(o12[0].masked_fill(mask == 0, -1e9), dim=1).squeeze().numpy()
            v12_v = float(o12[1].item())

        # Stable KL(V13 || V12.2) over legal moves
        legal_idx = [i for i in range(4) if mask_np[i] > 0]
        eps = 1e-9
        kl = float(sum(p13[i] * (np.log(p13[i] + eps) - np.log(p12[i] + eps)) for i in legal_idx))
        agree = int(np.argmax(p13)) == int(np.argmax(p12))
        rows.append({'phase': st['phase'], 'kl': kl, 'agree': agree,
                     'v13_v': v13_v, 'v12_v': v12_v, 'val_abs': abs(v13_v - v12_v)})
    return rows

def report(rows):
    n = len(rows)
    by = {'all': rows}
    for ph in ('early', 'mid', 'late'):
        by[ph] = [r for r in rows if r['phase'] == ph]
    print(f"\n{'group':8} {'n':>6} {'top1_agree':>11} {'mean_KL':>9} {'med_KL':>8} {'val_diff':>9}")
    for k, rs in by.items():
        if not rs: continue
        kls = [r['kl'] for r in rs]
        ag = sum(1 for r in rs if r['agree']) / len(rs) * 100
        v = sum(r['val_abs'] for r in rs) / len(rs)
        print(f"{k:8} {len(rs):>6} {ag:>10.1f}% {np.mean(kls):>9.3f} {np.median(kls):>8.3f} {v:>9.3f}")

if __name__ == '__main__':
    print('Loading V13 G=33K...')
    v13 = load_v13('/Users/sumit/Github/AlphaLudo/td_ludo/checkpoints/ac_v13_g33k/model.pt')
    print('Loading V12.2 latest...')
    v12 = load_v12_2('/Users/sumit/Github/AlphaLudo/td_ludo/play/model_weights/v12_2/model_latest.pt')
    print('Collecting 2000 multi-legal states from V13 self-play...')
    states = collect_states(v13, n=2000, seed=42)
    print(f'Got {len(states)} states. Evaluating both models...')
    rows = evaluate_pair(v13, v12, states)
    report(rows)
