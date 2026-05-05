"""V14_scalar SL distillation — V12.2-bias teacher → V14ScalarDeepSets student.

Differs from V13.2's trainer (train_v132_sl.py):

1. Student is **V14ScalarDeepSets** (~225K params, no CNN, no attention).
2. Student input is the V14_scalar non-spatial dict (per-token + global
   features) — not a 17-channel spatial tensor.
3. Same teacher (V12.2-bias, 33ch V11 encoder, ~1.36M params), unchanged.

Hypothesis: if V12.2's success was about its INPUT FEATURES (danger, capture,
idle, etc.) and not about spatial CNN structure, a DeepSets model with the
same scalar feature set should distill the teacher's policy/value just as
well — and reach a similar plateau in RL.

Usage:
    TD_LUDO_RUN_NAME=v14_scalar python train_v14_scalar_sl.py \\
        --teacher play/model_weights/v12_2/model_latest.pt \\
        --target-states 10_000_000 \\
        --port 8793
"""
import argparse
import functools
import json
import os
import random
import sys
import threading
import time
from http.server import HTTPServer, SimpleHTTPRequestHandler

import numpy as np
import torch
import torch.nn.functional as F

# Project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import td_ludo_cpp as ludo_cpp
from td_ludo.models.v12 import AlphaLudoV12
from td_ludo.models.v14_scalar import V14ScalarDeepSets
from td_ludo.game.encoder_v14_scalar import (
    encode_batch_v14_scalar, encode_state_v14_scalar,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--teacher', required=True,
                   help='Path to V12.2-bias teacher checkpoint.')
    p.add_argument('--run-name', default=None,
                   help='Override TD_LUDO_RUN_NAME (sets checkpoint dir).')
    p.add_argument('--target-states', type=int, default=10_000_000)
    p.add_argument('--batch-size', type=int, default=1024)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--lr-end', type=float, default=1e-4,
                   help='Cosine-decay LR target.')
    p.add_argument('--save-every', type=int, default=1_000_000)
    p.add_argument('--eval-every', type=int, default=250_000)
    p.add_argument('--eval-games', type=int, default=200)
    p.add_argument('--log-every', type=int, default=20)
    p.add_argument('--policy-coeff', type=float, default=1.0)
    p.add_argument('--value-coeff', type=float, default=0.5)
    p.add_argument('--moves-coeff', type=float, default=0.0,
                   help='No moves head in V14_scalar — defaults to 0.')
    # V14_scalar architecture sizes (defaults match v14_scalar.py)
    p.add_argument('--pos-emb-dim', type=int, default=32)
    p.add_argument('--token-hidden', type=int, default=64)
    p.add_argument('--trunk-hidden', type=int, default=256)
    p.add_argument('--head-hidden', type=int, default=64)
    p.add_argument('--device', default='auto', choices=('auto', 'cpu', 'cuda', 'mps'))
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--port', type=int, default=8793, help='Dashboard port.')
    p.add_argument('--no-dashboard', action='store_true')
    return p.parse_args()


def pick_device(name):
    if name in ('cuda', 'cpu', 'mps'): return torch.device(name)
    if torch.cuda.is_available(): return torch.device('cuda')
    if torch.backends.mps.is_available(): return torch.device('mps')
    return torch.device('cpu')


# ── On-the-fly distillation env ─────────────────────────────────────────
class OnTheFlyEnv:
    """Same logic as V13.2's env — produces (teacher_input, student_input,
    masks) per step. Teacher input is V11 33ch tensor; student input is
    V14_scalar dict (batched in apply_batch step)."""
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
        cps = []
        teacher_tensors = []
        student_dicts = []
        batch_masks = []
        batch_legal = []
        for i in range(self.batch_size):
            while True:
                game = self.games[i]
                if game.is_terminal or self.step_count[i] >= self.max_game_len:
                    self._reset(i); game = self.games[i]
                cp = int(game.current_player)
                if game.current_dice_roll == 0:
                    roll = random.randint(1, 6)
                    game.current_dice_roll = roll
                    if roll == 6: self.consec_sixes[i, cp] += 1
                    else: self.consec_sixes[i, cp] = 0
                    if self.consec_sixes[i, cp] >= 3:
                        nxt = (cp + 1) % 4
                        while not game.active_players[nxt]: nxt = (nxt + 1) % 4
                        game.current_player = nxt; game.current_dice_roll = 0
                        self.consec_sixes[i, cp] = 0
                        continue
                legal = ludo_cpp.get_legal_moves(game)
                if not legal:
                    nxt = (cp + 1) % 4
                    while not game.active_players[nxt]: nxt = (nxt + 1) % 4
                    game.current_player = nxt; game.current_dice_roll = 0
                    continue
                mask = np.zeros(4, dtype=np.float32)
                for m in legal: mask[m] = 1.0
                teacher_tensors.append(
                    np.array(ludo_cpp.encode_state_v11(game), dtype=np.float32)
                )
                student_dicts.append(encode_state_v14_scalar(game))
                decision_idxs.append(i)
                cps.append(cp)
                batch_masks.append(mask)
                batch_legal.append(legal)
                break

        # Stack into batched tensors / dict
        teacher_batch = np.stack(teacher_tensors)               # (B, 33, 15, 15)
        student_batch = {
            k: np.stack([d[k] for d in student_dicts], axis=0)
            for k in student_dicts[0].keys()
        }
        masks_batch = np.stack(batch_masks)                     # (B, 4)
        return decision_idxs, cps, teacher_batch, student_batch, masks_batch, batch_legal

    def apply_actions(self, decision_idxs, actions, batch_masks, batch_legal):
        for k, i in enumerate(decision_idxs):
            action = int(actions[k])
            if batch_masks[k][action] == 0:
                action = batch_legal[k][0]
            self.games[i] = ludo_cpp.apply_move(self.games[i], action)
            self.step_count[i] += 1


# ── Dashboard server ─────────────────────────────────────────────────────
class _SLHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, directory=None, sl_path=None, chain_path=None,
                 landing=None, **kwargs):
        self._sl_path = sl_path
        self._chain_path = chain_path
        self._landing = landing
        super().__init__(*args, directory=directory, **kwargs)

    def do_GET(self):
        if self.path == '/' or self.path == '':
            if self._landing:
                self.path = '/' + self._landing
            return super().do_GET()
        if self.path == '/api/sl_stats':
            return self._serve(self._sl_path)
        if self.path == '/api/chain':
            return self._serve(self._chain_path)
        super().do_GET()

    def _serve(self, p):
        try:
            with open(p) as f: data = f.read()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(data.encode())
        except FileNotFoundError:
            self.send_response(404); self.end_headers()


def start_dashboard(port, sl_path, chain_path, dashboard_dir):
    landing = None
    for cand in ('sl_dashboard.html', 'v13_dashboard.html', 'v12_dashboard.html', 'index.html'):
        if os.path.exists(os.path.join(dashboard_dir, cand)):
            landing = cand; break
    handler = functools.partial(
        _SLHandler, directory=dashboard_dir,
        sl_path=sl_path, chain_path=chain_path, landing=landing,
    )
    server = HTTPServer(('0.0.0.0', port), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f'[Dashboard] http://localhost:{port}/{landing or ""}')


# ── Eval ─────────────────────────────────────────────────────────────────
def quick_eval(student, device, n_games=200):
    """Greedy student vs random bot mix. Returns aggregate WR%."""
    from src.heuristic_bot import get_bot, BOT_REGISTRY
    from src.config import MAX_MOVES_PER_GAME
    bot_types = list(BOT_REGISTRY.keys())
    student.eval()
    wins = 0
    for g in range(n_games):
        model_player = random.choice([0, 2])
        opp_player = 2 if model_player == 0 else 0
        bot = get_bot(random.choice(bot_types), player_id=opp_player)
        state = ludo_cpp.create_initial_state_2p()
        csix = [0, 0, 0, 0]; mc = 0
        while not state.is_terminal and mc < MAX_MOVES_PER_GAME:
            cp = int(state.current_player)
            if not state.active_players[cp]:
                n = (cp + 1) % 4
                while not state.active_players[n]: n = (n + 1) % 4
                state.current_player = n; continue
            if state.current_dice_roll == 0:
                state.current_dice_roll = random.randint(1, 6)
                if state.current_dice_roll == 6: csix[cp] += 1
                else: csix[cp] = 0
                if csix[cp] >= 3:
                    n = (cp + 1) % 4
                    while not state.active_players[n]: n = (n + 1) % 4
                    state.current_player = n; state.current_dice_roll = 0
                    csix[cp] = 0; continue
            legal = ludo_cpp.get_legal_moves(state)
            if not legal:
                n = (cp + 1) % 4
                while not state.active_players[n]: n = (n + 1) % 4
                state.current_player = n; state.current_dice_roll = 0; continue
            if cp == model_player:
                if len(legal) == 1:
                    action = legal[0]
                else:
                    enc = encode_state_v14_scalar(state)
                    batch = {k: torch.from_numpy(v[None, ...]) for k, v in enc.items()}
                    batch['own_pos'] = batch['own_pos'].long()
                    batch['opp_pos'] = batch['opp_pos'].long()
                    batch = {k: v.to(device) for k, v in batch.items()}
                    mask = np.zeros(4, dtype=np.float32)
                    for m in legal: mask[m] = 1.0
                    m_t = torch.from_numpy(mask).unsqueeze(0).to(device)
                    with torch.no_grad():
                        logits = student.forward_policy_only(batch, m_t)
                        action = int(logits.argmax(dim=1).item())
                    if action not in legal: action = random.choice(legal)
            else:
                action = bot.select_move(state, list(legal))
            state = ludo_cpp.apply_move(state, int(action))
            mc += 1
        if state.is_terminal and ludo_cpp.get_winner(state) == model_player:
            wins += 1
    student.train()
    return 100 * wins / n_games


# ─────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = pick_device(args.device)

    if args.run_name:
        os.environ['TD_LUDO_RUN_NAME'] = args.run_name
    from src.config import CHECKPOINT_DIR
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    sl_stats_path = os.path.join(CHECKPOINT_DIR, 'sl_stats.json')
    chain_path = os.path.join(CHECKPOINT_DIR, 'chain_status.json')
    sl_log_path = os.path.join(CHECKPOINT_DIR, 'sl.log')

    print('=' * 70)
    print('V14_scalar SL DISTILLATION (V12.2-bias → V14ScalarDeepSets)')
    print('=' * 70)
    print(f'  device:        {device}')
    print(f'  teacher:       {args.teacher}')
    print(f'  checkpoint dir:{CHECKPOINT_DIR}')
    print(f'  arch:          DeepSets ('
          f'pos_emb={args.pos_emb_dim}, token_h={args.token_hidden}, '
          f'trunk={args.trunk_hidden})')
    print(f'  target_states: {args.target_states:,}')
    print(f'  batch_size:    {args.batch_size}')
    print(f'  lr:            {args.lr} → {args.lr_end} (cosine)')
    print(f'  eval_every:    {args.eval_every:,} states ({args.eval_games} games)')
    print('=' * 70)

    # Sanity: V11 encoder symmetry
    g0 = ludo_cpp.create_initial_state_2p(); g0.current_player = 0; g0.current_dice_roll = 6
    g2 = ludo_cpp.create_initial_state_2p(); g2.current_player = 2; g2.current_dice_roll = 6
    d = float(np.abs(np.asarray(ludo_cpp.encode_state_v11(g0))
                     - np.asarray(ludo_cpp.encode_state_v11(g2))).sum())
    if d > 1e-6:
        raise RuntimeError(f'V11 encoder not symmetric (sum_diff={d}).')
    print('[Sanity] V11 encoder symmetric ✓')

    # Teacher: V12.2-bias (3×128, 33ch input)
    print(f'[Teacher] Loading {args.teacher}...')
    teacher = AlphaLudoV12(num_res_blocks=3, num_channels=128, num_attn_layers=2,
                           num_heads=4, ffn_ratio=4, dropout=0.0, in_channels=33)
    ckpt = torch.load(args.teacher, map_location=device, weights_only=False)
    sd = ckpt.get('model_state_dict', ckpt)
    sd = {k.replace('_orig_mod.', ''): v for k, v in sd.items()}
    teacher.load_state_dict(sd)
    teacher.to(device).eval()
    for p in teacher.parameters(): p.requires_grad = False
    print(f'[Teacher] params: {sum(p.numel() for p in teacher.parameters()):,}')

    # Student: V14ScalarDeepSets
    print(f'[Student] Initializing V14ScalarDeepSets...')
    student = V14ScalarDeepSets(
        pos_emb_dim=args.pos_emb_dim,
        token_hidden=args.token_hidden,
        trunk_hidden=args.trunk_hidden,
        head_hidden=args.head_hidden,
    )
    student.to(device).train()
    print(f'[Student] params: {student.count_parameters():,}')

    optimizer = torch.optim.Adam(student.parameters(), lr=args.lr)

    # Save initial random-init weights so we never lose a run
    init_path = os.path.join(CHECKPOINT_DIR, 'sl_init.pt')
    torch.save(student.state_dict(), init_path)
    print(f'[Init] saved {init_path}')

    def write_chain(phase):
        with open(chain_path, 'w') as f:
            json.dump({'stage': 'SL', 'phase': phase,
                       'run_name': os.environ.get('TD_LUDO_RUN_NAME', '-'),
                       'arch': 'v14_scalar', 'ts': int(time.time())}, f)
    write_chain('initializing')

    if not args.no_dashboard:
        dashboard_dir = os.path.dirname(os.path.abspath(__file__))
        start_dashboard(args.port, sl_stats_path, chain_path, dashboard_dir)

    sl_log = open(sl_log_path, 'a')
    def log(msg):
        line = f'[{time.strftime("%H:%M:%S")}] {msg}'
        print(line, flush=True); sl_log.write(line + '\n'); sl_log.flush()

    env = OnTheFlyEnv(args.batch_size)
    write_chain('training')
    log(f'Starting SL: target {args.target_states:,} states')

    t_start = time.time()
    last_save = 0
    last_eval = 0
    total = 0
    step = 0
    recent_total, recent_pol, recent_val = [], [], []
    eval_history = []

    while total < args.target_states:
        decision_idxs, cps, teacher_t, student_b, masks, legals = env.get_batch()

        # Teacher input
        teacher_t_t = torch.from_numpy(teacher_t).to(device)
        masks_t = torch.from_numpy(masks).to(device)

        # Student input dict → torch
        student_b_t = {k: torch.from_numpy(v).to(device) for k, v in student_b.items()}
        student_b_t['own_pos'] = student_b_t['own_pos'].long()
        student_b_t['opp_pos'] = student_b_t['opp_pos'].long()

        # Teacher
        with torch.no_grad():
            t_policy, t_win, t_moves = teacher(teacher_t_t, masks_t)
            actions = torch.multinomial(t_policy, num_samples=1).squeeze(1).cpu().numpy()

        env.apply_actions(decision_idxs, actions, masks, legals)

        # LR cosine
        progress = total / args.target_states
        cur_lr = args.lr_end + 0.5 * (args.lr - args.lr_end) * (1 + np.cos(np.pi * progress))
        for g in optimizer.param_groups: g['lr'] = cur_lr

        # Student forward
        s_policy, s_win, _s_moves = student(student_b_t, masks_t)

        # Distillation losses (no moves head)
        s_log_policy = torch.log(s_policy + 1e-8)
        loss_policy = F.kl_div(s_log_policy, t_policy, reduction='batchmean', log_target=False)
        loss_win    = F.mse_loss(s_win, t_win)
        loss = (args.policy_coeff * loss_policy + args.value_coeff * loss_win)

        optimizer.zero_grad(); loss.backward(); optimizer.step()
        step += 1
        total += args.batch_size

        recent_total.append(loss.item()); recent_pol.append(loss_policy.item())
        recent_val.append(loss_win.item())
        if len(recent_total) > 200:
            recent_total.pop(0); recent_pol.pop(0); recent_val.pop(0)

        if step % args.log_every == 0:
            elapsed = time.time() - t_start
            fps = total / max(1e-6, elapsed)
            log(f'step {step:>6} | states {total:>10,}/{args.target_states:,} '
                f'| fps {fps:>6.0f} | lr {cur_lr:.1e} '
                f'| L {loss.item():.4f} '
                f'(pol {loss_policy.item():.3f} val {loss_win.item():.3f})')
            try:
                with open(sl_stats_path, 'w') as f:
                    json.dump({
                        'stage': 'SL', 'arch': 'v14_scalar',
                        'step': step, 'states': total,
                        'target': args.target_states, 'fps': fps,
                        'elapsed_sec': elapsed, 'lr': cur_lr,
                        'loss': float(np.mean(recent_total)),
                        'loss_policy': float(np.mean(recent_pol)),
                        'loss_value': float(np.mean(recent_val)),
                        'eval_history': eval_history,
                        'ts': int(time.time()),
                    }, f)
            except Exception as e:
                print(f'[stats] write failed: {e}')

        if total - last_save >= args.save_every:
            ckpt_path = os.path.join(CHECKPOINT_DIR, f'sl_{total // 1_000_000}M.pt')
            torch.save(student.state_dict(), ckpt_path)
            log(f'[checkpoint] {ckpt_path}')
            last_save = total

        if total - last_eval >= args.eval_every and total > 0:
            log(f'[eval] starting ({args.eval_games} games vs random bot mix)...')
            wr = quick_eval(student, device, n_games=args.eval_games)
            eval_history.append([total, wr])
            log(f'[eval] WR = {wr:.1f}% at {total:,} states')
            last_eval = total

    # Final save → model_sl.pt
    final_path = os.path.join(CHECKPOINT_DIR, 'model_sl.pt')
    torch.save(student.state_dict(), final_path)
    log(f'[done] processed {total:,} states. Final SL ckpt: {final_path}')
    write_chain('completed')
    sl_log.close()


if __name__ == '__main__':
    main()
