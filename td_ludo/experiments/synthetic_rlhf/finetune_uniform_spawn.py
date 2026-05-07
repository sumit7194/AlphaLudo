"""POC: fine-tune V12.2 to output uniform [0.25, 0.25, 0.25, 0.25]
on the specific state {n_own_at_base=4 AND dice=6}.

Approach: targeted supervised fine-tune with KL(student || uniform) loss.
Single epoch over the 233K target-state rows from the 100K self-play DB.

Reconstructs state33 from stored own_pos / opp_pos / dice / current_player
on the fly per minibatch (cheaper than caching all 233K × 33 × 15 × 15
tensors in memory).

Verification at end:
  - On a held-out 5% of target rows, what's the new policy?
  - On a random sample of non-target rows, what's the new policy vs old?
    (Drift check.)

Usage:
  python -m experiments.synthetic_rlhf.finetune_uniform_spawn \\
      --teacher play/model_weights/v12_2/model_latest.pt \\
      --db experiments/synthetic_rlhf/v122_selfplay_100k.db \\
      --out experiments/synthetic_rlhf/v122_uniform_spawn.pt
"""
import argparse
import os
import random
import sqlite3
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import td_ludo_cpp as cpp
from td_ludo.models.v12 import AlphaLudoV12

BASE_POS = -1
HOME_POS = 99


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--teacher', default='play/model_weights/v12_2/model_latest.pt')
    p.add_argument('--db', default='experiments/synthetic_rlhf/v122_selfplay_100k.db')
    p.add_argument('--out', default='experiments/synthetic_rlhf/v122_uniform_spawn.pt')
    p.add_argument('--filter', default='n_own_at_base=4 AND dice=6',
                   help='SQL WHERE clause selecting target states')
    p.add_argument('--lr', type=float, default=1e-5)
    p.add_argument('--batch-size', type=int, default=256,
                   help='Target-state batch size. Anchor batch is the same size.')
    p.add_argument('--anchor-weight', type=float, default=1.0,
                   help='Weight on KL(student||frozen) anchor loss vs target uniform loss. '
                        '1.0 = equal weight. 0.0 = no anchor (will drift).')
    p.add_argument('--holdout-frac', type=float, default=0.05,
                   help='Fraction of target rows held out for after-training verification')
    p.add_argument('--n-non-target-verify', type=int, default=2000,
                   help='Random non-target rows to check post-train for drift')
    p.add_argument('--device', default='auto', choices=('auto', 'cpu', 'cuda', 'mps'))
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--log-every', type=int, default=50, help='Log every N batches')
    return p.parse_args()


def pick_device(name):
    if name != 'auto': return torch.device(name)
    if torch.cuda.is_available(): return torch.device('cuda')
    if torch.backends.mps.is_available(): return torch.device('mps')
    return torch.device('cpu')


def reconstruct_state(own_pos, opp_pos, current_player, dice):
    """Build a GameState and encode to 33ch using encode_state_v11."""
    g = cpp.create_initial_state_2p()
    pp = list(g.player_positions)
    pp[current_player] = list(own_pos)
    pp[(current_player + 2) % 4] = list(opp_pos)
    g.player_positions = pp
    g.current_player = current_player
    g.current_dice_roll = dice
    return np.asarray(cpp.encode_state_v11(g), dtype=np.float32)


def parse_pos(s):
    return [int(x) for x in s.split(',')]


def parse_mask(s):
    return [int(c) for c in s]


def fetch_rows(conn, where, fields=None):
    """Stream rows matching WHERE clause."""
    if fields is None:
        fields = 'own_pos, opp_pos, current_player, dice, legal_mask'
    cur = conn.execute(f'SELECT id, {fields} FROM decisions WHERE {where}')
    return cur


def encode_batch(rows, batch_size, device):
    """Yield batches of (state_tensor, legal_mask_tensor)."""
    states, masks = [], []
    for r in rows:
        rid, own_s, opp_s, cp, dice, mask_s = r[0], r[1], r[2], r[3], r[4], r[5]
        states.append(reconstruct_state(parse_pos(own_s), parse_pos(opp_s), cp, dice))
        masks.append(parse_mask(mask_s))
        if len(states) == batch_size:
            yield (
                torch.from_numpy(np.stack(states)).to(device),
                torch.tensor(masks, dtype=torch.float32, device=device),
            )
            states, masks = [], []
    if states:
        yield (
            torch.from_numpy(np.stack(states)).to(device),
            torch.tensor(masks, dtype=torch.float32, device=device),
        )


def main():
    args = parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = pick_device(args.device)

    print('=' * 70)
    print('TARGETED FINE-TUNE: V12.2 → uniform on {n_own_at_base=4 AND dice=6}')
    print('=' * 70)
    print(f'  device: {device}')
    print(f'  teacher checkpoint: {args.teacher}')
    print(f'  source DB: {args.db}')
    print(f'  output checkpoint: {args.out}')
    print(f'  filter: {args.filter}')
    print(f'  LR: {args.lr}  batch_size: {args.batch_size}  holdout: {args.holdout_frac}')
    print()

    # Load TWO copies of V12.2:
    #   model    = student (mutable, fine-tuned)
    #   frozen   = teacher anchor (no_grad, prevents drift on non-target states)
    print('[Load] V12.2 pre-search → student + frozen-anchor...')
    def _new_v122():
        m = AlphaLudoV12(num_res_blocks=3, num_channels=128, num_attn_layers=2,
                         num_heads=4, ffn_ratio=4, dropout=0.0, in_channels=33)
        ckpt = torch.load(args.teacher, map_location=device, weights_only=False)
        sd = ckpt.get('model_state_dict', ckpt)
        sd = {k.replace('_orig_mod.', ''): v for k, v in sd.items()}
        m.load_state_dict(sd)
        return m.to(device)
    model = _new_v122(); model.train()
    frozen = _new_v122(); frozen.eval()
    for p in frozen.parameters(): p.requires_grad = False
    n_params = sum(p.numel() for p in model.parameters())
    print(f'[Load] {n_params:,} params (frozen anchor: same)')

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Open DB + count target rows
    conn = sqlite3.connect(args.db)
    cur = conn.execute(f'SELECT COUNT(*) FROM decisions WHERE {args.filter}')
    n_total = cur.fetchone()[0]
    print(f'[Data] target rows matching filter: {n_total:,}')
    n_holdout = int(n_total * args.holdout_frac)
    n_train = n_total - n_holdout
    print(f'[Data] train: {n_train:,}  holdout: {n_holdout:,}')

    # Pull all row IDs and shuffle for train/holdout split
    cur = conn.execute(f'SELECT id FROM decisions WHERE {args.filter}')
    all_ids = [r[0] for r in cur.fetchall()]
    random.shuffle(all_ids)
    train_ids = all_ids[:n_train]
    holdout_ids = all_ids[n_train:]

    # ── Training loop ────────────────────────────────────────────────────
    print()
    print(f'[Train] starting single-epoch fine-tune on {n_train:,} target rows...')
    t_start = time.time()
    n_seen = 0
    losses = []
    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)

    # Anchor source: random non-target rows. Pull all non-target IDs once.
    cur = conn.execute(f'SELECT id FROM decisions WHERE NOT ({args.filter})')
    nontarget_ids = [r[0] for r in cur.fetchall()]
    print(f'[Anchor] non-target pool: {len(nontarget_ids):,} rows')

    def fetch_anchor_batch(n):
        ids = random.sample(nontarget_ids, n)
        placeholders = ','.join('?' * len(ids))
        cur = conn.execute(
            f'SELECT id, own_pos, opp_pos, current_player, dice, legal_mask '
            f'FROM decisions WHERE id IN ({placeholders})', ids)
        rows = list(cur.fetchall())
        states = np.stack([reconstruct_state(parse_pos(r[1]), parse_pos(r[2]), r[3], r[4]) for r in rows])
        masks = np.array([parse_mask(r[5]) for r in rows], dtype=np.float32)
        return (torch.from_numpy(states).to(device),
                torch.tensor(masks, dtype=torch.float32, device=device))

    # Process train_ids in chunks (avoid loading all into memory at once)
    CHUNK = 50000
    batch_idx = 0
    target_loss_avg = []
    anchor_loss_avg = []
    for chunk_start in range(0, n_train, CHUNK):
        chunk_ids = train_ids[chunk_start:chunk_start + CHUNK]
        placeholders = ','.join('?' * len(chunk_ids))
        cur = conn.execute(
            f'SELECT id, own_pos, opp_pos, current_player, dice, legal_mask '
            f'FROM decisions WHERE id IN ({placeholders})',
            chunk_ids,
        )
        rows = list(cur.fetchall())
        random.shuffle(rows)

        for states_t, masks_t in encode_batch(rows, args.batch_size, device):
            optimizer.zero_grad()
            # ── Target loss: push student → uniform on legal tokens ────
            policy_t, _, _ = model(states_t, masks_t)
            target = masks_t / masks_t.sum(dim=1, keepdim=True).clamp_min(1e-8)
            target_loss = -(target * torch.log(policy_t + 1e-8)).sum(dim=1).mean()

            # ── Anchor loss: keep student close to frozen V12.2 on non-target states ──
            anchor_states, anchor_masks = fetch_anchor_batch(states_t.shape[0])
            policy_a_student, _, _ = model(anchor_states, anchor_masks)
            with torch.no_grad():
                policy_a_frozen, _, _ = frozen(anchor_states, anchor_masks)
            # KL(frozen || student) = sum frozen * (log frozen - log student)
            # Equivalent up to constant to: cross-entropy with soft target
            anchor_loss = -(policy_a_frozen * torch.log(policy_a_student + 1e-8)).sum(dim=1).mean()

            loss = target_loss + args.anchor_weight * anchor_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            n_seen += states_t.shape[0]
            losses.append(loss.item())
            target_loss_avg.append(target_loss.item())
            anchor_loss_avg.append(anchor_loss.item())
            batch_idx += 1
            if batch_idx % args.log_every == 0:
                t = sum(target_loss_avg[-args.log_every:]) / args.log_every
                a = sum(anchor_loss_avg[-args.log_every:]) / args.log_every
                elapsed = time.time() - t_start
                rate = n_seen / max(elapsed, 1e-6)
                print(f'  batch {batch_idx:>5} | seen {n_seen:>7,}/{n_train:,} '
                      f'| target_loss {t:.3f} | anchor_loss {a:.3f} | {rate:.0f} rows/s',
                      flush=True)

    elapsed = time.time() - t_start
    print(f'[Train] done in {elapsed/60:.1f} min ({n_seen/elapsed:.0f} rows/s).')

    # ── Save fine-tuned weights ──────────────────────────────────────────
    torch.save({'model_state_dict': model.state_dict()}, args.out)
    print(f'[Save] {args.out}')

    # ── Verification: held-out target rows ───────────────────────────────
    print()
    print('=' * 70)
    print('VERIFICATION 1: held-out target rows (should be ~uniform)')
    print('=' * 70)
    model.eval()
    placeholders = ','.join('?' * len(holdout_ids))
    cur = conn.execute(
        f'SELECT id, own_pos, opp_pos, current_player, dice, legal_mask, '
        f'policy_t0, policy_t1, policy_t2, policy_t3 '
        f'FROM decisions WHERE id IN ({placeholders})',
        holdout_ids,
    )
    holdout_rows = list(cur.fetchall())
    sums_orig = np.zeros(4); sums_new = np.zeros(4); n_h = 0
    for batch in [holdout_rows[i:i+args.batch_size] for i in range(0, len(holdout_rows), args.batch_size)]:
        states = np.stack([reconstruct_state(parse_pos(r[1]), parse_pos(r[2]), r[3], r[4]) for r in batch])
        masks = np.array([parse_mask(r[5]) for r in batch], dtype=np.float32)
        with torch.no_grad():
            policy, _, _ = model(torch.from_numpy(states).to(device),
                                  torch.from_numpy(masks).to(device))
            policy_np = policy.cpu().numpy()
        for r, p_new in zip(batch, policy_np):
            p_orig = np.array(r[6:10])
            sums_orig += p_orig
            sums_new += p_new
            n_h += 1
    avg_orig = sums_orig / n_h
    avg_new = sums_new / n_h
    print(f'  Held-out n: {n_h:,}')
    print(f'  Pre-finetune avg policy: T0={avg_orig[0]:.3f} T1={avg_orig[1]:.3f} T2={avg_orig[2]:.3f} T3={avg_orig[3]:.3f}')
    print(f'  Post-finetune avg policy: T0={avg_new[0]:.3f} T1={avg_new[1]:.3f} T2={avg_new[2]:.3f} T3={avg_new[3]:.3f}')
    print(f'  Target uniform:          T0=0.250 T1=0.250 T2=0.250 T3=0.250')
    deviation = float(np.abs(avg_new - 0.25).sum())
    print(f'  Total absolute deviation from uniform: {deviation:.4f} (lower = closer to target)')

    # ── Verification 2: random non-target rows (drift check) ─────────────
    print()
    print('=' * 70)
    print(f'VERIFICATION 2: {args.n_non_target_verify} random non-target rows (drift check)')
    print('=' * 70)
    cur = conn.execute(
        f'SELECT id, own_pos, opp_pos, current_player, dice, legal_mask, '
        f'policy_t0, policy_t1, policy_t2, policy_t3 '
        f'FROM decisions WHERE NOT ({args.filter}) '
        f'ORDER BY RANDOM() LIMIT ?', (args.n_non_target_verify,),
    )
    nt_rows = list(cur.fetchall())
    abs_diffs = []
    kl_diffs = []
    for batch in [nt_rows[i:i+args.batch_size] for i in range(0, len(nt_rows), args.batch_size)]:
        states = np.stack([reconstruct_state(parse_pos(r[1]), parse_pos(r[2]), r[3], r[4]) for r in batch])
        masks = np.array([parse_mask(r[5]) for r in batch], dtype=np.float32)
        with torch.no_grad():
            policy, _, _ = model(torch.from_numpy(states).to(device),
                                  torch.from_numpy(masks).to(device))
            policy_np = policy.cpu().numpy()
        for r, p_new in zip(batch, policy_np):
            p_orig = np.array(r[6:10])
            abs_diffs.append(float(np.abs(p_new - p_orig).sum()))
            kl_diffs.append(float(np.sum(p_orig * np.log((p_orig + 1e-9) / (p_new + 1e-9)))))
    print(f'  Mean L1 |new-old|: {np.mean(abs_diffs):.4f}  (0 = no drift, 2 = complete swap)')
    print(f'  Mean KL(orig || new): {np.mean(kl_diffs):.4f}')
    print(f'  P95 L1: {np.percentile(abs_diffs, 95):.4f}')
    print(f'  Max L1: {np.max(abs_diffs):.4f}')

    conn.close()


if __name__ == '__main__':
    main()
