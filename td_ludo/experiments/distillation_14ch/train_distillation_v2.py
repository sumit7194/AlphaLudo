"""SL distillation V2 — V12.2 pre-search → MinimalCNN14 student (V13).

KEY DIFFERENCE FROM V1 (`train_distillation.py`):
  - Teacher is V12.2 PRE-SEARCH (the snapshot saved Apr 30 01:28, before Exp 24
    enabled MCTS-augmented training). The post-search V12.2 had a learned
    spatial-cell preference for the BR base corner that, due to the encoder
    asymmetry bug at the time, manifested as T2/T3 over-picking.
  - Encoder is now POST-FIX (commit 1ff249f). Both teacher's 33ch encoder and
    student's 14ch encoder produce canonical (P0-equivalent) views regardless
    of which seat is current. Self-play training data is now genuinely
    rotation-symmetric — student no longer has to learn each pattern twice.
  - Outputs to experiments/distillation_14ch/v2/ to preserve Distill14 v1.

Pipeline:
  - Both seats are V12.2 (self-play). Teacher samples from its policy to
    advance games; this also serves as the "diverse decisions" generator.
  - Per decision:
      • encode state in 33ch (teacher input) and 14ch (student input)
      • teacher forward → (policy, win_prob, moves_remaining)
      • student forward → same shape
      • losses: KL(student || teacher) policy + MSE win + SmoothL1 moves
      • optimizer step on student
  - Save student weights every 1M states processed.

Usage:
  python -m experiments.distillation_14ch.train_distillation_v2
  python -m experiments.distillation_14ch.train_distillation_v2 --target-states 1000000 --batch-size 256
  python -m experiments.distillation_14ch.train_distillation_v2 --teacher /path/to/other.pt --output-dir other/

Run from `td_ludo/` directory so relative paths resolve.
"""
import argparse
import os
import sys
import time
import random
import numpy as np
import torch
import torch.nn.functional as F

# Add repo root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import td_ludo_cpp as ludo_cpp
from td_ludo.models.v12 import AlphaLudoV12
from experiments.distillation_14ch.model_14ch import MinimalCNN14


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--teacher', default='play/model_weights/v12_2/model_latest.pt',
                   help='Path to V12.2 pre-search teacher checkpoint.')
    p.add_argument('--output-dir', default='experiments/distillation_14ch/v2',
                   help='Where to save student checkpoints (will be created).')
    p.add_argument('--output-prefix', default='student_14ch',
                   help='Filename prefix for student checkpoints.')
    p.add_argument('--target-states', type=int, default=5_000_000,
                   help='Total states to process before stopping.')
    p.add_argument('--batch-size', type=int, default=1024)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--save-every', type=int, default=1_000_000,
                   help='Save a checkpoint every N states.')
    p.add_argument('--log-every', type=int, default=10,
                   help='Print loss every N steps.')
    p.add_argument('--policy-coeff', type=float, default=1.0)
    p.add_argument('--value-coeff', type=float, default=1.0)
    p.add_argument('--moves-coeff', type=float, default=0.01)
    p.add_argument('--device', default='auto', choices=('auto', 'cpu', 'cuda', 'mps'))
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


def pick_device(name):
    if name == 'cpu':
        return torch.device('cpu')
    if name == 'cuda':
        return torch.device('cuda')
    if name == 'mps':
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def sanity_check_encoder():
    """Verify the encoder is post-fix symmetric (both 14ch and 33ch)."""
    g0 = ludo_cpp.create_initial_state_2p(); g0.current_player = 0; g0.current_dice_roll = 6
    g2 = ludo_cpp.create_initial_state_2p(); g2.current_player = 2; g2.current_dice_roll = 6
    for fn_name in ('encode_state_v11', 'encode_state_v14_minimal'):
        fn = getattr(ludo_cpp, fn_name)
        t0 = np.asarray(fn(g0)); t2 = np.asarray(fn(g2))
        d = float(np.abs(t0 - t2).sum())
        if d > 1e-6:
            raise RuntimeError(
                f'Encoder {fn_name} is NOT symmetric (sum_diff={d}). '
                'Did you forget to rebuild td_ludo_cpp after the BASE_COORDS fix? '
                'Run: cd td_ludo/ && ./td_env/bin/pip install -e . --no-deps --no-build-isolation'
            )
    return True


class OnTheFlyDistillationEnv:
    """Self-play game runner that yields one decision per game per call.

    Both seats use the teacher policy (V12.2). The student observes every
    decision via its own encoder. With the post-fix encoder, both 33ch and
    14ch tensors are in canonical (current_player POV) view, so the student
    sees a consistent representation regardless of seat.
    """
    def __init__(self, batch_size, max_game_len=400):
        self.batch_size = batch_size
        self.max_game_len = max_game_len
        self.games = [ludo_cpp.create_initial_state_2p() for _ in range(batch_size)]
        self.consec_sixes = np.zeros((batch_size, 4), dtype=np.int32)
        self.step_count = np.zeros(batch_size, dtype=np.int32)

    def _reset(self, i):
        self.games[i] = ludo_cpp.create_initial_state_2p()
        self.consec_sixes[i] = 0
        self.step_count[i] = 0

    def get_batch(self):
        decision_idxs = []
        batch33 = []
        batch14 = []
        batch_masks = []
        batch_legal = []
        for i in range(self.batch_size):
            while True:
                game = self.games[i]
                if game.is_terminal or self.step_count[i] >= self.max_game_len:
                    self._reset(i); game = self.games[i]
                cp = game.current_player
                if game.current_dice_roll == 0:
                    roll = random.randint(1, 6)
                    game.current_dice_roll = roll
                    if roll == 6: self.consec_sixes[i, cp] += 1
                    else: self.consec_sixes[i, cp] = 0
                    if self.consec_sixes[i, cp] >= 3:
                        nxt = (cp + 1) % 4
                        while not game.active_players[nxt]: nxt = (nxt + 1) % 4
                        game.current_player = nxt
                        game.current_dice_roll = 0
                        self.consec_sixes[i, cp] = 0
                        continue
                legal = ludo_cpp.get_legal_moves(game)
                if not legal:
                    nxt = (cp + 1) % 4
                    while not game.active_players[nxt]: nxt = (nxt + 1) % 4
                    game.current_player = nxt
                    game.current_dice_roll = 0
                    continue
                mask = np.zeros(4, dtype=np.float32)
                for m in legal: mask[m] = 1.0
                enc33 = np.array(ludo_cpp.encode_state_v11(game), dtype=np.float32)
                enc14 = np.array(ludo_cpp.encode_state_v14_minimal(game), dtype=np.float32)
                decision_idxs.append(i)
                batch33.append(enc33); batch14.append(enc14)
                batch_masks.append(mask); batch_legal.append(legal)
                break
        return (decision_idxs,
                np.stack(batch33), np.stack(batch14),
                np.stack(batch_masks), batch_legal)

    def apply_actions(self, decision_idxs, actions, batch_masks, batch_legal):
        for k, i in enumerate(decision_idxs):
            action = int(actions[k])
            if batch_masks[k][action] == 0:
                action = batch_legal[k][0]
            self.games[i] = ludo_cpp.apply_move(self.games[i], action)
            self.step_count[i] += 1


def main():
    args = parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    device = pick_device(args.device)
    print('=' * 70)
    print('SL DISTILLATION V2 — V12.2 PRE-SEARCH → MinimalCNN14')
    print('=' * 70)
    print(f'  device:        {device}')
    print(f'  teacher:       {args.teacher}')
    print(f'  output_dir:    {args.output_dir}')
    print(f'  output_prefix: {args.output_prefix}')
    print(f'  target_states: {args.target_states:,}')
    print(f'  batch_size:    {args.batch_size}')
    print(f'  lr:            {args.lr}')
    print(f'  loss coeffs:   policy={args.policy_coeff}  value={args.value_coeff}  moves={args.moves_coeff}')
    print('=' * 70)

    print('\n[Sanity] Verifying encoder symmetry (post-fix)...')
    sanity_check_encoder()
    print('[Sanity] OK — encode_state_v11 and encode_state_v14_minimal are seat-symmetric.\n')

    print(f'[Teacher] Loading {args.teacher}...')
    teacher = AlphaLudoV12(num_res_blocks=3, num_channels=128, num_attn_layers=2,
                           num_heads=4, ffn_ratio=4, dropout=0.0, in_channels=33)
    ckpt = torch.load(args.teacher, map_location=device, weights_only=False)
    sd = ckpt.get('model_state_dict', ckpt)
    sd = {k.replace('_orig_mod.', ''): v for k, v in sd.items()}
    teacher.load_state_dict(sd)
    teacher.to(device).eval()
    for p in teacher.parameters():
        p.requires_grad = False
    print(f'[Teacher] params: {sum(p.numel() for p in teacher.parameters()):,}\n')

    print('[Student] Initializing MinimalCNN14 (10 ResBlocks × 128ch, 14-ch input)...')
    student = MinimalCNN14(num_res_blocks=10, num_channels=128, in_channels=14)
    student.to(device).train()
    print(f'[Student] params: {sum(p.numel() for p in student.parameters()):,}\n')

    optimizer = torch.optim.Adam(student.parameters(), lr=args.lr)

    # Bulletproof output directory creation (resolve to absolute, create parents,
    # then verify writeability with a probe file). Past runs have died on the
    # first checkpoint save because of relative-path / missing-parent issues.
    abs_out = os.path.abspath(args.output_dir)
    os.makedirs(abs_out, exist_ok=True)
    probe = os.path.join(abs_out, '.write_probe')
    try:
        with open(probe, 'w') as f: f.write('ok')
        os.remove(probe)
    except Exception as e:
        raise RuntimeError(f'output_dir {abs_out} not writable: {e}')
    args.output_dir = abs_out  # use absolute everywhere downstream
    print(f'[Output] writable directory: {abs_out}')

    # Save the initial (random-init) student so we never lose a run, even if
    # the first training step crashes. Acts as the 0M baseline checkpoint.
    init_path = os.path.join(args.output_dir, f'{args.output_prefix}_0M.pt')
    torch.save(student.state_dict(), init_path)
    print(f'[checkpoint] initial random-init saved → {init_path}\n')

    env = OnTheFlyDistillationEnv(args.batch_size)
    print(f'[Run] starting on-the-fly distillation loop. Target: {args.target_states:,} states.\n')
    t_start = time.time()
    last_save = 0
    total = 0
    step = 0

    while total < args.target_states:
        decision_idxs, t33, t14, masks, legals = env.get_batch()
        t33_t = torch.from_numpy(t33).to(device)
        t14_t = torch.from_numpy(t14).to(device)
        masks_t = torch.from_numpy(masks).to(device)

        # Teacher inference (frozen, no_grad)
        with torch.no_grad():
            t_policy, t_win, t_moves = teacher(t33_t, masks_t)
            actions = torch.multinomial(t_policy, num_samples=1).squeeze(1).cpu().numpy()

        # Advance games using teacher's sampled actions
        env.apply_actions(decision_idxs, actions, masks, legals)

        # Student forward + distillation loss
        s_policy, s_win, s_moves = student(t14_t, masks_t)
        s_log_policy = torch.log(s_policy + 1e-8)
        loss_policy = F.kl_div(s_log_policy, t_policy, reduction='batchmean', log_target=False)
        loss_win    = F.mse_loss(s_win, t_win)
        loss_moves  = F.smooth_l1_loss(s_moves, t_moves)
        loss = (args.policy_coeff * loss_policy
                + args.value_coeff * loss_win
                + args.moves_coeff * loss_moves)

        optimizer.zero_grad(); loss.backward(); optimizer.step()
        step += 1
        total += args.batch_size

        if step % args.log_every == 0:
            elapsed = time.time() - t_start
            fps = total / max(1e-6, elapsed)
            print(f'step {step:>6} | states {total:>10,}/{args.target_states:,} '
                  f'| fps {fps:>7.0f} | loss {loss.item():.4f} '
                  f'(pol {loss_policy.item():.4f}, val {loss_win.item():.4f}, mov {loss_moves.item():.4f})',
                  flush=True)

        if total - last_save >= args.save_every:
            ckpt_path = os.path.join(args.output_dir,
                                       f'{args.output_prefix}_{total // 1_000_000}M.pt')
            torch.save(student.state_dict(), ckpt_path)
            print(f'[checkpoint] saved {ckpt_path}', flush=True)
            last_save = total

    final_path = os.path.join(args.output_dir, f'{args.output_prefix}_final.pt')
    torch.save(student.state_dict(), final_path)
    elapsed = time.time() - t_start
    print(f'\n[done] processed {total:,} states in {elapsed/60:.1f} min '
          f'({total/elapsed:.0f} states/sec). Final: {final_path}')


if __name__ == '__main__':
    main()
