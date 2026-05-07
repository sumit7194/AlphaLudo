"""V13.3 SL distillation — V13.2-bias teacher → V133Temporal student.

Differs from train_v132_sl.py:
  - Teacher is V13.2 (`MinimalCNN14` with 17ch input, 10×128) instead of V12.2.
    V13.2 is now the strongest model in the codebase per the 3-way tournament.
  - Student is V133Temporal — small per-turn CNN + 2-layer transformer over
    K=8 frames of history.
  - Per-game history buffer: each parallel game keeps a deque of the last
    K-1 V17-encoded frames; on each new decision state, the (K-1) cached
    + 1 new frame form the student's input.

Hypothesis: temporal context across the last K turns lets the student learn
patterns the stateless V13.2 cannot — opponent style, recent dice luck,
"this opp keeps unlocking on 6s," "we just escaped a capture," etc.

Usage:
    TD_LUDO_RUN_NAME=v13_3 python train_v133_sl.py \\
        --teacher checkpoint_backups/v132_*/model_latest.pt \\
        --target-states 5_000_000 \\
        --port 8794
"""
from __future__ import annotations

import argparse
import collections
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import td_ludo_cpp as ludo_cpp
from td_ludo.game.encoder_v17 import encode_state_v17, V17_CHANNELS
from experiments.distillation_14ch.model_14ch import MinimalCNN14
from td_ludo.models.v13_3 import V133Temporal


HISTORY_K = 8


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher", required=True,
                   help="Path to V13.2 teacher checkpoint")
    p.add_argument("--run-name", default=None)
    p.add_argument("--target-states", type=int, default=5_000_000)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lr-end", type=float, default=1e-4)
    p.add_argument("--save-every", type=int, default=1_000_000)
    p.add_argument("--eval-every", type=int, default=250_000)
    p.add_argument("--eval-games", type=int, default=200)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--policy-coeff", type=float, default=1.0)
    p.add_argument("--value-coeff", type=float, default=0.5)
    p.add_argument("--moves-coeff", type=float, default=0.01)
    # V13.3 architecture
    p.add_argument("--cnn-blocks", type=int, default=4)
    p.add_argument("--cnn-channels", type=int, default=64)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--nhead", type=int, default=4)
    p.add_argument("--ffn-dim", type=int, default=256)
    p.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--port", type=int, default=8794)
    p.add_argument("--no-dashboard", action="store_true")
    return p.parse_args()


def pick_device(name):
    if name in ("cuda", "cpu", "mps"):
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ── Teacher loading ────────────────────────────────────────────────────────
def load_v132_teacher(path, device):
    print(f"[Teacher] Loading V13.2 from {path}...")
    model = MinimalCNN14(num_res_blocks=10, num_channels=128, in_channels=17)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    if any(k.startswith("_orig_mod.") for k in sd):
        sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    print(f"[Teacher] params: {sum(p.numel() for p in model.parameters()):,}")
    return model


# ── On-the-fly distillation env with per-game history ──────────────────────
class TemporalDistillEnv:
    """Per-game V17-frame history buffer. Each get_batch() returns a batch
    of (history_K, current_legal_mask, eventual_outcome_to_be_filled)."""
    def __init__(self, batch_size, history_k=HISTORY_K, max_game_len=400):
        self.batch_size = batch_size
        self.history_k = history_k
        self.max_game_len = max_game_len
        self.games = [ludo_cpp.create_initial_state_2p() for _ in range(batch_size)]
        # Per-game per-player history. Each game has TWO deques (one per
        # player in 2P); we push to the deque of the player to move so the
        # transformer's K-window is "this player's last K decision states"
        # — matching what an agent sees during inference.
        self.history = [
            {0: collections.deque(maxlen=history_k),
             2: collections.deque(maxlen=history_k)}
            for _ in range(batch_size)
        ]
        self.consec_sixes = np.zeros((batch_size, 4), dtype=np.int32)
        self.step_count = np.zeros(batch_size, dtype=np.int32)

    def _reset(self, i):
        self.games[i] = ludo_cpp.create_initial_state_2p()
        self.history[i][0].clear()
        self.history[i][2].clear()
        self.consec_sixes[i] = 0
        self.step_count[i] = 0

    def get_batch(self):
        """Spin every game forward to its next decision state. Encode the
        current frame, push into history, build (K, 17, 15, 15) tensor with
        history_mask. Returns batch + per-game decision context.
        """
        decision_idxs = []
        cps = []
        teacher_inputs = []     # current 17ch frame for V13.2 teacher (B, 17, 15, 15)
        student_histories = []  # K-frame stack for V13.3 student (B, K, 17, 15, 15)
        history_masks = []      # (B, K) bool — True where real frame
        legal_masks = []
        legal_lists = []

        for i in range(self.batch_size):
            while True:
                game = self.games[i]
                if game.is_terminal or self.step_count[i] >= self.max_game_len:
                    self._reset(i)
                    game = self.games[i]
                cp = int(game.current_player)
                if game.current_dice_roll == 0:
                    roll = random.randint(1, 6)
                    game.current_dice_roll = roll
                    if roll == 6:
                        self.consec_sixes[i, cp] += 1
                    else:
                        self.consec_sixes[i, cp] = 0
                    if self.consec_sixes[i, cp] >= 3:
                        nxt = (cp + 1) % 4
                        while not game.active_players[nxt]:
                            nxt = (nxt + 1) % 4
                        game.current_player = nxt
                        game.current_dice_roll = 0
                        self.consec_sixes[i, cp] = 0
                        continue
                legal = ludo_cpp.get_legal_moves(game)
                if not legal:
                    nxt = (cp + 1) % 4
                    while not game.active_players[nxt]:
                        nxt = (nxt + 1) % 4
                    game.current_player = nxt
                    game.current_dice_roll = 0
                    continue
                # decision state
                mask = np.zeros(4, dtype=np.float32)
                for m in legal:
                    mask[m] = 1.0

                # Encode current frame
                cur_frame = encode_state_v17(game)  # (17,15,15) float32

                # Push current frame to THIS PLAYER's history (per-player deques)
                self.history[i][cp].append(cur_frame)

                # Build K-frame stack from THIS PLAYER's history
                # (oldest first, padded with zeros at front)
                hist = list(self.history[i][cp])  # 1..K frames (current is last)
                pad = self.history_k - len(hist)
                if pad > 0:
                    zero = np.zeros((V17_CHANNELS, 15, 15), dtype=np.float32)
                    stack = np.stack([zero] * pad + hist, axis=0)
                    mask_arr = np.array([False] * pad + [True] * len(hist), dtype=bool)
                else:
                    stack = np.stack(hist, axis=0)
                    mask_arr = np.ones(self.history_k, dtype=bool)

                decision_idxs.append(i)
                cps.append(cp)
                teacher_inputs.append(cur_frame)
                student_histories.append(stack)
                history_masks.append(mask_arr)
                legal_masks.append(mask)
                legal_lists.append(legal)
                break

        return (
            decision_idxs,
            cps,
            np.stack(teacher_inputs, axis=0),   # (B, 17, 15, 15)
            np.stack(student_histories, axis=0),# (B, K, 17, 15, 15)
            np.stack(history_masks, axis=0),    # (B, K)
            np.stack(legal_masks, axis=0),      # (B, 4)
            legal_lists,
        )

    def apply_actions(self, decision_idxs, actions, legal_masks, legal_lists):
        for k, i in enumerate(decision_idxs):
            action = int(actions[k])
            if legal_masks[k][action] == 0:
                action = legal_lists[k][0]
            self.games[i] = ludo_cpp.apply_move(self.games[i], action)
            self.step_count[i] += 1


# ── Dashboard server ───────────────────────────────────────────────────────
class _SLHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, directory=None, sl_path=None, chain_path=None, landing=None, **kw):
        self._sl_path = sl_path
        self._chain_path = chain_path
        self._landing = landing
        super().__init__(*args, directory=directory, **kw)

    def do_GET(self):
        if self.path in ("/", ""):
            if self._landing:
                self.path = "/" + self._landing
            return super().do_GET()
        if self.path == "/api/sl_stats":
            return self._serve(self._sl_path)
        if self.path == "/api/chain":
            return self._serve(self._chain_path)
        super().do_GET()

    def _serve(self, p):
        try:
            with open(p) as f:
                data = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(data.encode())
        except FileNotFoundError:
            self.send_response(404)
            self.end_headers()


def start_dashboard(port, sl_path, chain_path, dashboard_dir):
    landing = None
    for cand in ("sl_dashboard.html", "v13_dashboard.html", "v12_dashboard.html", "index.html"):
        if os.path.exists(os.path.join(dashboard_dir, cand)):
            landing = cand
            break
    handler = functools.partial(
        _SLHandler, directory=dashboard_dir,
        sl_path=sl_path, chain_path=chain_path, landing=landing,
    )
    server = HTTPServer(("0.0.0.0", port), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"[Dashboard] http://localhost:{port}/{landing or ''}")


# ── Eval ───────────────────────────────────────────────────────────────────
def quick_eval(student, device, history_k=HISTORY_K, n_games=200):
    """Greedy student with maintained K-frame history vs random bot mix."""
    from src.heuristic_bot import get_bot, BOT_REGISTRY
    from src.config import MAX_MOVES_PER_GAME
    bot_types = list(BOT_REGISTRY.keys())
    student.eval()
    wins = 0
    for _ in range(n_games):
        model_player = random.choice([0, 2])
        opp_player = 2 if model_player == 0 else 0
        bot = get_bot(random.choice(bot_types), player_id=opp_player)
        state = ludo_cpp.create_initial_state_2p()
        history = collections.deque(maxlen=history_k)
        csix = [0, 0, 0, 0]
        mc = 0
        while not state.is_terminal and mc < MAX_MOVES_PER_GAME:
            cp = int(state.current_player)
            if not state.active_players[cp]:
                n = (cp + 1) % 4
                while not state.active_players[n]:
                    n = (n + 1) % 4
                state.current_player = n
                continue
            if state.current_dice_roll == 0:
                state.current_dice_roll = random.randint(1, 6)
                if state.current_dice_roll == 6:
                    csix[cp] += 1
                else:
                    csix[cp] = 0
                if csix[cp] >= 3:
                    n = (cp + 1) % 4
                    while not state.active_players[n]:
                        n = (n + 1) % 4
                    state.current_player = n
                    state.current_dice_roll = 0
                    csix[cp] = 0
                    continue
            legal = ludo_cpp.get_legal_moves(state)
            if not legal:
                n = (cp + 1) % 4
                while not state.active_players[n]:
                    n = (n + 1) % 4
                state.current_player = n
                state.current_dice_roll = 0
                continue

            # ALWAYS update history (even on opp turns? No — only when *we* see
            # the state from our POV). For SL we encoded only at decision
            # states the student would face. To keep it simple at eval,
            # we encode at every decision state (both players) but only use
            # the model on model_player decisions. Each player has its own
            # history... actually in self-play the model only plays as one
            # side, so this isn't a clean pattern.
            # Punt: encode only on model-player decisions, accept that
            # history reflects ONLY the model's view (which is what training
            # assumes too — the env builds history per game, not per side).
            cur_frame = encode_state_v17(state)
            if cp == model_player:
                history.append(cur_frame)
                if len(legal) == 1:
                    action = legal[0]
                else:
                    pad = history_k - len(history)
                    if pad > 0:
                        zero = np.zeros((V17_CHANNELS, 15, 15), dtype=np.float32)
                        stack = np.stack([zero] * pad + list(history), axis=0)
                        hmask = np.array([False] * pad + [True] * len(history), dtype=bool)
                    else:
                        stack = np.stack(list(history), axis=0)
                        hmask = np.ones(history_k, dtype=bool)
                    mask = np.zeros(4, dtype=np.float32)
                    for m in legal:
                        mask[m] = 1.0
                    with torch.no_grad():
                        x = torch.from_numpy(stack).unsqueeze(0).to(device)
                        h = torch.from_numpy(hmask).unsqueeze(0).to(device)
                        m_t = torch.from_numpy(mask).unsqueeze(0).to(device)
                        logits = student.forward_policy_only(x, m_t, h)
                        action = int(logits.argmax(dim=1).item())
                    if action not in legal:
                        action = random.choice(legal)
            else:
                action = bot.select_move(state, list(legal))
            state = ludo_cpp.apply_move(state, int(action))
            mc += 1
        if state.is_terminal and ludo_cpp.get_winner(state) == model_player:
            wins += 1
    student.train()
    return 100 * wins / n_games


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = pick_device(args.device)

    if args.run_name:
        os.environ["TD_LUDO_RUN_NAME"] = args.run_name
    from src.config import CHECKPOINT_DIR
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    sl_stats_path = os.path.join(CHECKPOINT_DIR, "sl_stats.json")
    chain_path = os.path.join(CHECKPOINT_DIR, "chain_status.json")
    sl_log_path = os.path.join(CHECKPOINT_DIR, "sl.log")

    print("=" * 70)
    print("V13.3 SL DISTILLATION (V13.2 → V133Temporal)")
    print("=" * 70)
    print(f"  device:        {device}")
    print(f"  teacher:       {args.teacher}")
    print(f"  checkpoint dir:{CHECKPOINT_DIR}")
    print(f"  arch:          V133Temporal (history_K={HISTORY_K}, "
          f"cnn={args.cnn_blocks}×{args.cnn_channels}, transformer={args.n_layers}L×{args.nhead}H d={args.d_model})")
    print(f"  target_states: {args.target_states:,}")
    print(f"  batch_size:    {args.batch_size}")
    print(f"  lr:            {args.lr} → {args.lr_end} (cosine)")
    print("=" * 70)

    # Sanity: V11 encoder symmetric (post-fix)
    g0 = ludo_cpp.create_initial_state_2p(); g0.current_player = 0; g0.current_dice_roll = 6
    g2 = ludo_cpp.create_initial_state_2p(); g2.current_player = 2; g2.current_dice_roll = 6
    d = float(np.abs(np.asarray(ludo_cpp.encode_state_v11(g0))
                     - np.asarray(ludo_cpp.encode_state_v11(g2))).sum())
    if d > 1e-6:
        raise RuntimeError(f"V11 encoder not symmetric (sum_diff={d}).")
    print("[Sanity] V11 encoder symmetric ✓")

    teacher = load_v132_teacher(args.teacher, device)

    print(f"[Student] Initializing V133Temporal...")
    student = V133Temporal(
        history_k=HISTORY_K,
        in_channels=V17_CHANNELS,
        cnn_blocks=args.cnn_blocks,
        cnn_channels=args.cnn_channels,
        d_model=args.d_model,
        nhead=args.nhead,
        n_layers=args.n_layers,
        ffn_dim=args.ffn_dim,
    )
    student.to(device).train()
    print(f"[Student] params: {student.count_parameters():,}")

    optimizer = torch.optim.Adam(student.parameters(), lr=args.lr)

    init_path = os.path.join(CHECKPOINT_DIR, "sl_init.pt")
    torch.save(student.state_dict(), init_path)
    print(f"[Init] saved {init_path}")

    def write_chain(phase):
        with open(chain_path, "w") as f:
            json.dump({"stage": "SL", "phase": phase, "arch": "v13_3",
                       "run_name": os.environ.get("TD_LUDO_RUN_NAME", "-"),
                       "ts": int(time.time())}, f)
    write_chain("training")

    if not args.no_dashboard:
        dash_dir = os.path.dirname(os.path.abspath(__file__))
        start_dashboard(args.port, sl_stats_path, chain_path, dash_dir)

    sl_log = open(sl_log_path, "a")
    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        sl_log.write(line + "\n")
        sl_log.flush()

    env = TemporalDistillEnv(args.batch_size)
    log(f"Starting SL: target {args.target_states:,} states.")

    t_start = time.time()
    last_save = 0
    last_eval = 0
    total = 0
    step = 0
    recent_total, recent_pol, recent_val = [], [], []
    eval_history = []

    while total < args.target_states:
        decision_idxs, cps, teacher_in, student_hist, history_mask, legal_masks, legal_lists = env.get_batch()

        teacher_t = torch.from_numpy(teacher_in).to(device, dtype=torch.float32)
        student_t = torch.from_numpy(student_hist).to(device, dtype=torch.float32)
        history_t = torch.from_numpy(history_mask).to(device)
        masks_t = torch.from_numpy(legal_masks).to(device, dtype=torch.float32)

        # Teacher (V13.2): single-frame
        with torch.no_grad():
            t_policy, t_win, t_moves = teacher(teacher_t, masks_t)
            actions = torch.multinomial(t_policy, num_samples=1).squeeze(1).cpu().numpy()

        env.apply_actions(decision_idxs, actions, legal_masks, legal_lists)

        progress = total / args.target_states
        cur_lr = args.lr_end + 0.5 * (args.lr - args.lr_end) * (1 + np.cos(np.pi * progress))
        for g in optimizer.param_groups:
            g["lr"] = cur_lr

        # Student forward (history)
        s_policy, s_win, s_moves = student(student_t, masks_t, history_t)

        # Distillation losses
        s_log = torch.log(s_policy + 1e-8)
        loss_policy = F.kl_div(s_log, t_policy, reduction="batchmean", log_target=False)
        loss_win = F.mse_loss(s_win, t_win)
        loss_moves = F.smooth_l1_loss(s_moves, t_moves)

        loss = (
            args.policy_coeff * loss_policy
            + args.value_coeff * loss_win
            + args.moves_coeff * loss_moves
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        step += 1
        total += args.batch_size

        recent_total.append(loss.item())
        recent_pol.append(loss_policy.item())
        recent_val.append(loss_win.item())
        for r in (recent_total, recent_pol, recent_val):
            if len(r) > 200:
                r.pop(0)

        if step % args.log_every == 0:
            elapsed = time.time() - t_start
            fps = total / max(1e-6, elapsed)
            log(f"step {step:>6} | states {total:>10,}/{args.target_states:,} "
                f"| fps {fps:>5.0f} | lr {cur_lr:.1e} "
                f"| L {loss.item():.4f} (pol {loss_policy.item():.3f} "
                f"val {loss_win.item():.3f} mov {loss_moves.item():.3f})")
            try:
                with open(sl_stats_path, "w") as f:
                    json.dump({
                        "stage": "SL", "arch": "v13_3",
                        "step": step, "states": total,
                        "target": args.target_states, "fps": fps,
                        "elapsed_sec": elapsed, "lr": cur_lr,
                        "loss": float(np.mean(recent_total)),
                        "loss_policy": float(np.mean(recent_pol)),
                        "loss_value": float(np.mean(recent_val)),
                        "eval_history": eval_history,
                        "ts": int(time.time()),
                    }, f)
            except Exception as e:
                print(f"[stats] write failed: {e}")

        if total - last_save >= args.save_every:
            ckpt_path = os.path.join(CHECKPOINT_DIR, f"sl_{total // 1_000_000}M.pt")
            torch.save(student.state_dict(), ckpt_path)
            log(f"[checkpoint] {ckpt_path}")
            last_save = total

        if total - last_eval >= args.eval_every and total > 0:
            log(f"[eval] starting ({args.eval_games} games)...")
            wr = quick_eval(student, device, n_games=args.eval_games)
            eval_history.append([total, wr])
            log(f"[eval] WR = {wr:.1f}% at {total:,} states")
            last_eval = total

    final_path = os.path.join(CHECKPOINT_DIR, "model_sl.pt")
    final_latest = os.path.join(CHECKPOINT_DIR, "model_latest.pt")
    torch.save(student.state_dict(), final_path)
    torch.save(student.state_dict(), final_latest)
    log(f"[done] processed {total:,} states.")
    log(f"[done] saved → {final_path} + {final_latest}")
    write_chain("completed")
    sl_log.close()


if __name__ == "__main__":
    main()
