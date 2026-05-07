"""V13.3 RL — self-play REINFORCE-with-baseline on V133Temporal.

Initialised from V13.3 SL student. The student plays both sides of a 2-player
Ludo game; trajectories are recorded with per-game K=8 history; at episode end
each step is labelled with G ∈ {+1, -1} from the POV of the player to move at
that step. We then take gradient steps on:

    L = -(G - V_old).detach() · log π(a|s)   [policy gradient w/ baseline]
        + value_coeff · MSE(V_θ(s), G)
        - entropy_coeff · H(π)
        + kl_anchor_coeff · KL(π_θ || π_v132_teacher)   [optional regulariser]

Design notes:
- Single-process, vectorised over B parallel games. Each game keeps its own
  collections.deque(maxlen=K) of V17 frames + a trajectory buffer of (stack,
  hist_mask, legal_mask, action, v_pred, cp).
- We collect data with grads OFF; once the trajectory pool reaches
  --train-chunk states, we do --train-epochs passes over it (with shuffle)
  then clear and continue.
- Eval = greedy student vs random heuristic-bot mix (same recipe as
  train_v133_sl.py:quick_eval).

Usage:
    TD_LUDO_RUN_NAME=v13_3_rl python train_v133_rl.py \\
        --init checkpoints/v13_3/model_sl.pt \\
        --target-states 2_000_000 \\
        --port 8795
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
from td_ludo.models.v13_3 import V133Temporal


HISTORY_K = 8


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--init", default=None,
                   help="Path to V13.3 SL checkpoint to start RL from (required if not --resume)")
    p.add_argument("--teacher", default=None,
                   help="Optional V13.2 teacher checkpoint for KL anchor (regulariser)")
    p.add_argument("--run-name", default=None)
    p.add_argument("--target-states", type=int, default=2_000_000)
    p.add_argument("--parallel-games", type=int, default=128,
                   help="Number of parallel games for self-play")
    p.add_argument("--train-chunk", type=int, default=4096,
                   help="Train when this many decision states have been collected")
    p.add_argument("--minibatch-size", type=int, default=512)
    p.add_argument("--train-epochs", type=int, default=2,
                   help="Passes over each chunk")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lr-end", type=float, default=2e-5)
    p.add_argument("--value-coeff", type=float, default=0.5)
    p.add_argument("--entropy-coeff", type=float, default=0.01)
    p.add_argument("--kl-anchor-coeff", type=float, default=0.0,
                   help=">0 enables KL anchor to V13.2 teacher (regulariser)")
    p.add_argument("--save-every", type=int, default=200_000,
                   help="Save every N states (ignored if --save-every-games > 0).")
    p.add_argument("--save-every-games", type=int, default=0,
                   help="Save every N completed self-play games. Overrides --save-every if > 0.")
    p.add_argument("--eval-every-games", type=int, default=0,
                   help="Eval every N completed self-play games. Overrides --eval-every if > 0.")
    p.add_argument("--eval-every", type=int, default=200_000)
    p.add_argument("--eval-games", type=int, default=200)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--max-game-len", type=int, default=400)
    # V13.3 architecture (must match init checkpoint)
    p.add_argument("--cnn-blocks", type=int, default=4)
    p.add_argument("--cnn-channels", type=int, default=64)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--nhead", type=int, default=4)
    p.add_argument("--ffn-dim", type=int, default=256)
    p.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--port", type=int, default=8795)
    p.add_argument("--no-dashboard", action="store_true")
    p.add_argument("--resume", action="store_true",
                   help="Resume from model_latest.pt in checkpoint dir (restores optimizer + counters)")
    return p.parse_args()


def pick_device(name):
    if name in ("cuda", "cpu", "mps"):
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ── Self-play env ──────────────────────────────────────────────────────────
class SelfPlayEnv:
    """B parallel games, each with its own K-frame history deque + trajectory
    buffer. spin_to_decision() pushes every game forward to its next decision
    state. apply_actions() applies actions and, on terminal, finalises the
    trajectory (assigning G to each step) and emits it."""

    def __init__(self, batch_size, history_k=HISTORY_K, max_game_len=400):
        self.batch_size = batch_size
        self.history_k = history_k
        self.max_game_len = max_game_len
        self.games = [ludo_cpp.create_initial_state_2p() for _ in range(batch_size)]
        # Per-game per-player history (2P only). Push to history[i][cp] when
        # player cp is to move; transformer sees ONLY that player's last K
        # decision states — matching inference-time agent memory.
        self.history = [
            {0: collections.deque(maxlen=history_k),
             2: collections.deque(maxlen=history_k)}
            for _ in range(batch_size)
        ]
        # trajectory[i] = list of (stack, hmask, lmask, action, v_pred, cp)
        self.trajectory = [[] for _ in range(batch_size)]
        self.consec_sixes = np.zeros((batch_size, 4), dtype=np.int32)
        self.step_count = np.zeros(batch_size, dtype=np.int32)
        self.games_played = 0
        self.game_lengths = []  # for telemetry

    def _reset(self, i):
        self.games[i] = ludo_cpp.create_initial_state_2p()
        self.history[i][0].clear()
        self.history[i][2].clear()
        self.trajectory[i] = []
        self.consec_sixes[i] = 0
        self.step_count[i] = 0

    def _finalize(self, i, winner):
        """Compute G per trajectory step; return list of finished tuples to feed
        into the training pool."""
        traj = self.trajectory[i]
        out = []
        for stack, hmask, lmask, action, v_pred, cp in traj:
            G = 1.0 if cp == winner else -1.0
            out.append((stack, hmask, lmask, action, v_pred, G))
        self.games_played += 1
        self.game_lengths.append(len(traj))
        if len(self.game_lengths) > 200:
            self.game_lengths.pop(0)
        return out

    def spin_to_decision(self):
        """Advance every game to its next decision state. Returns the batch
        of inputs needed to call the model."""
        decision_idxs = []
        cps = []
        student_histories = []  # (B, K, 17, 15, 15)
        history_masks = []      # (B, K)
        legal_masks = []        # (B, 4)
        legal_lists = []

        for i in range(self.batch_size):
            while True:
                game = self.games[i]
                if game.is_terminal or self.step_count[i] >= self.max_game_len:
                    # Terminal handling — caller already finalised in apply_actions.
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

                cur_frame = encode_state_v17(game)
                # Push to THIS PLAYER's deque (per-player history).
                self.history[i][cp].append(cur_frame)

                hist = list(self.history[i][cp])
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
                student_histories.append(stack)
                history_masks.append(mask_arr)
                legal_masks.append(mask)
                legal_lists.append(legal)
                break

        return (
            decision_idxs,
            cps,
            np.stack(student_histories, axis=0),
            np.stack(history_masks, axis=0),
            np.stack(legal_masks, axis=0),
            legal_lists,
        )

    def apply_actions(self, decision_idxs, cps, student_histories, history_masks,
                      legal_masks, legal_lists, actions, v_preds):
        """Apply actions, append to trajectory, finalise terminal games.
        Returns list of completed-trajectory tuples ready for the training
        pool."""
        finished = []
        for k, i in enumerate(decision_idxs):
            action = int(actions[k])
            if legal_masks[k][action] == 0:
                action = legal_lists[k][0]
            # Save for trajectory (we save all states, even single-legal — they
            # carry value-target signal even if their policy gradient is zero)
            self.trajectory[i].append((
                student_histories[k], history_masks[k], legal_masks[k],
                action, float(v_preds[k]), int(cps[k]),
            ))
            self.games[i] = ludo_cpp.apply_move(self.games[i], action)
            self.step_count[i] += 1

            game = self.games[i]
            if game.is_terminal:
                winner = int(ludo_cpp.get_winner(game))
                finished.extend(self._finalize(i, winner))
                # Don't reset here — spin_to_decision handles reset on next call.
            elif self.step_count[i] >= self.max_game_len:
                # Truncate as draw: assign G=0 to all (no winner reward signal).
                # Use winner = -1 so cp == winner is always False → all -1.
                # Actually flat 0 is more honest for a draw. We'll emit with G=0.
                traj = self.trajectory[i]
                for stack, hmask, lmask, act, v, c in traj:
                    finished.append((stack, hmask, lmask, act, v, 0.0))
                self.games_played += 1
                self.game_lengths.append(len(traj))
                if len(self.game_lengths) > 200:
                    self.game_lengths.pop(0)
                self.trajectory[i] = []
                # spin_to_decision will reset on next pass via step_count check.

        return finished


# ── Eval ───────────────────────────────────────────────────────────────────
def quick_eval(student, device, history_k=HISTORY_K, n_games=200):
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


# ── Dashboard ──────────────────────────────────────────────────────────────
class _RLHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, directory=None, rl_path=None, chain_path=None, landing=None, **kw):
        self._rl_path = rl_path
        self._chain_path = chain_path
        self._landing = landing
        super().__init__(*args, directory=directory, **kw)

    def do_GET(self):
        if self.path in ("/", ""):
            if self._landing:
                self.path = "/" + self._landing
            return super().do_GET()
        if self.path == "/api/rl_stats":
            return self._serve(self._rl_path)
        if self.path == "/api/sl_stats":  # alias for any dashboard expecting it
            return self._serve(self._rl_path)
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


def start_dashboard(port, rl_path, chain_path, dashboard_dir):
    landing = None
    for cand in ("rl_dashboard.html", "sl_dashboard.html", "v13_dashboard.html",
                 "v12_dashboard.html", "index.html"):
        if os.path.exists(os.path.join(dashboard_dir, cand)):
            landing = cand
            break
    handler = functools.partial(
        _RLHandler, directory=dashboard_dir,
        rl_path=rl_path, chain_path=chain_path, landing=landing,
    )
    server = HTTPServer(("0.0.0.0", port), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"[Dashboard] http://localhost:{port}/{landing or ''}")


# ── Training step ──────────────────────────────────────────────────────────
def train_on_chunk(student, optimizer, chunk, device, args, teacher=None):
    """chunk: list of (stack, hmask, lmask, action, v_pred_old, G).

    We re-run the model in train mode to get fresh logits + V; v_pred_old is
    from when the data was collected (used as baseline detached).
    """
    if not chunk:
        return None

    stacks = np.stack([c[0] for c in chunk], axis=0)
    hmasks = np.stack([c[1] for c in chunk], axis=0)
    lmasks = np.stack([c[2] for c in chunk], axis=0)
    actions = np.array([c[3] for c in chunk], dtype=np.int64)
    v_old = np.array([c[4] for c in chunk], dtype=np.float32)
    Gs = np.array([c[5] for c in chunk], dtype=np.float32)

    N = stacks.shape[0]
    metrics = {"loss": 0.0, "loss_pol": 0.0, "loss_val": 0.0, "entropy": 0.0,
               "loss_kl": 0.0, "n_steps": 0}

    for epoch in range(args.train_epochs):
        order = np.random.permutation(N)
        for s in range(0, N, args.minibatch_size):
            idx = order[s:s + args.minibatch_size]
            x = torch.from_numpy(stacks[idx]).to(device, dtype=torch.float32)
            hm = torch.from_numpy(hmasks[idx]).to(device)
            lm = torch.from_numpy(lmasks[idx]).to(device, dtype=torch.float32)
            a = torch.from_numpy(actions[idx]).to(device)
            vb = torch.from_numpy(v_old[idx]).to(device)
            G = torch.from_numpy(Gs[idx]).to(device)

            policy, win_prob, _moves = student(x, lm, hm)
            # Convert win_prob ∈ [0,1] → value v ∈ [-1,+1] for symmetric target.
            v = 2.0 * win_prob - 1.0

            # Multi-legal mask: only states with >1 legal moves contribute
            # gradient signal to policy / entropy heads. Single-legal states
            # have log_p(action) = 0 anyway (after legal-masking), so they
            # don't affect policy loss — but they DO drag the entropy mean
            # toward zero, making the entropy bonus useless. We filter them.
            multi = (lm.sum(dim=1) > 1).float()                     # (B,)
            multi_n = multi.sum().clamp(min=1.0)

            log_p = torch.log(policy.gather(1, a.unsqueeze(1)).squeeze(1) + 1e-8)
            advantage = (G - vb).detach()
            loss_pol = -(advantage * log_p * multi).sum() / multi_n

            loss_val = F.mse_loss(v, G)  # value loss uses ALL states

            entropy_per = -(policy * torch.log(policy + 1e-8)).sum(dim=1)  # (B,)
            entropy = (entropy_per * multi).sum() / multi_n
            loss_ent = -args.entropy_coeff * entropy

            loss = loss_pol + args.value_coeff * loss_val + loss_ent

            loss_kl_val = 0.0
            if teacher is not None and args.kl_anchor_coeff > 0:
                # Teacher = V13.2 single-frame: feed only the LAST frame (current decision state)
                cur = x[:, -1, :, :, :]  # (B, 17, 15, 15)
                with torch.no_grad():
                    t_pol, _, _ = teacher(cur, lm)
                # KL(student || teacher) as anchor
                loss_kl = F.kl_div(torch.log(policy + 1e-8), t_pol,
                                   reduction="batchmean", log_target=False)
                loss = loss + args.kl_anchor_coeff * loss_kl
                loss_kl_val = float(loss_kl.item())

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            optimizer.step()

            n = idx.shape[0]
            metrics["loss"]     += float(loss.item()) * n
            metrics["loss_pol"] += float(loss_pol.item()) * n
            metrics["loss_val"] += float(loss_val.item()) * n
            metrics["entropy"]  += float(entropy.item()) * n
            metrics["loss_kl"]  += loss_kl_val * n
            metrics["n_steps"]  += n

    if metrics["n_steps"]:
        for k in ("loss", "loss_pol", "loss_val", "entropy", "loss_kl"):
            metrics[k] /= metrics["n_steps"]
    return metrics


# ── Teacher loader (optional, for KL anchor) ──────────────────────────────
def load_v132_teacher(path, device):
    from experiments.distillation_14ch.model_14ch import MinimalCNN14
    model = MinimalCNN14(num_res_blocks=10, num_channels=128, in_channels=17)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    if any(k.startswith("_orig_mod.") for k in sd):
        sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


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
    rl_stats_path = os.path.join(CHECKPOINT_DIR, "rl_stats.json")
    chain_path = os.path.join(CHECKPOINT_DIR, "chain_status.json")
    rl_log_path = os.path.join(CHECKPOINT_DIR, "rl.log")

    # Resolve init path
    if args.resume:
        args.init = os.path.join(CHECKPOINT_DIR, "model_latest.pt")
        if not os.path.exists(args.init):
            print(f"ERROR: --resume but {args.init} not found"); sys.exit(1)
    elif not args.init:
        print("ERROR: either --init or --resume is required"); sys.exit(1)

    print("=" * 70)
    print("V13.3 RL — self-play REINFORCE")
    print("=" * 70)
    print(f"  device:          {device}")
    print(f"  init:            {args.init}")
    print(f"  checkpoint dir:  {CHECKPOINT_DIR}")
    print(f"  parallel_games:  {args.parallel_games}")
    print(f"  train_chunk:     {args.train_chunk}")
    print(f"  minibatch:       {args.minibatch_size}  × {args.train_epochs} epochs")
    print(f"  target_states:   {args.target_states:,}")
    print(f"  lr:              {args.lr} → {args.lr_end} (cosine)")
    print(f"  entropy_coeff:   {args.entropy_coeff}")
    print(f"  kl_anchor_coeff: {args.kl_anchor_coeff}")
    print("=" * 70)

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
    _ckpt = torch.load(args.init, map_location=device, weights_only=False)
    _resume_meta = None
    if isinstance(_ckpt, dict) and "model_state_dict" in _ckpt:
        sd = _ckpt["model_state_dict"]
        if args.resume:
            _resume_meta = _ckpt
    else:
        sd = _ckpt
    if any(k.startswith("_orig_mod.") for k in sd):
        sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    student.load_state_dict(sd)
    student.to(device).train()
    print(f"[Student] params: {student.count_parameters():,}  (loaded from {args.init})")

    teacher = None
    if args.teacher and args.kl_anchor_coeff > 0:
        teacher = load_v132_teacher(args.teacher, device)
        print(f"[Teacher] V13.2 loaded for KL anchor (coeff={args.kl_anchor_coeff})")

    optimizer = torch.optim.Adam(student.parameters(), lr=args.lr)
    if _resume_meta and "optimizer_state_dict" in _resume_meta:
        optimizer.load_state_dict(_resume_meta["optimizer_state_dict"])
        print(f"[Resume] Optimizer state restored")

    def write_chain(phase):
        with open(chain_path, "w") as f:
            json.dump({"stage": "RL", "phase": phase, "arch": "v13_3",
                       "run_name": os.environ.get("TD_LUDO_RUN_NAME", "-"),
                       "ts": int(time.time())}, f)
    write_chain("training")

    if not args.no_dashboard:
        dash_dir = os.path.dirname(os.path.abspath(__file__))
        start_dashboard(args.port, rl_stats_path, chain_path, dash_dir)

    rl_log = open(rl_log_path, "a")
    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        rl_log.write(line + "\n")
        rl_log.flush()

    env = SelfPlayEnv(args.parallel_games, history_k=HISTORY_K, max_game_len=args.max_game_len)
    log(f"Starting RL: target {args.target_states:,} states, init={args.init}")

    pool = []
    pool_size = 0
    total = 0
    step = 0
    last_save = 0
    last_eval = 0
    last_save_games = 0
    last_eval_games = 0
    t_start = time.time()
    eval_history = []
    last_metrics = None

    # Restore counters if resuming
    if _resume_meta:
        total = _resume_meta.get("total", 0)
        step = _resume_meta.get("step", 0)
        last_save = total
        last_eval = total
        eval_history = _resume_meta.get("eval_history", [])
        env.games_played = _resume_meta.get("games_played", 0)
        last_save_games = env.games_played
        last_eval_games = env.games_played
        log(f"[Resume] step={step}, states={total:,}, games={env.games_played}, evals={len(eval_history)}")

    while total < args.target_states:
        # 1. Spin all games to next decision
        decision_idxs, cps, hist_arr, hmask_arr, lmask_arr, legal_lists = env.spin_to_decision()

        # 2. Sample actions from current policy (no grads)
        with torch.no_grad():
            x = torch.from_numpy(hist_arr).to(device, dtype=torch.float32)
            hm = torch.from_numpy(hmask_arr).to(device)
            lm = torch.from_numpy(lmask_arr).to(device, dtype=torch.float32)
            policy, win_prob, _ = student(x, lm, hm)
            actions = torch.multinomial(policy, num_samples=1).squeeze(1).cpu().numpy()
            v_preds = (2.0 * win_prob - 1.0).cpu().numpy()

        # 3. Apply, collect finished trajectories
        per_step_data = list(zip(hist_arr, hmask_arr, lmask_arr))
        finished = env.apply_actions(
            decision_idxs, cps, hist_arr, hmask_arr, lmask_arr,
            legal_lists, actions, v_preds,
        )
        pool.extend(finished)
        pool_size += len(finished)

        # 4. If pool full, train
        if pool_size >= args.train_chunk:
            progress = total / max(1, args.target_states)
            cur_lr = args.lr_end + 0.5 * (args.lr - args.lr_end) * (1 + np.cos(np.pi * progress))
            for g in optimizer.param_groups:
                g["lr"] = cur_lr
            metrics = train_on_chunk(student, optimizer, pool, device, args, teacher=teacher)
            last_metrics = metrics
            total += pool_size
            step += 1
            pool = []
            pool_size = 0

            if step % max(1, args.log_every) == 0 and metrics is not None:
                elapsed = time.time() - t_start
                fps = total / max(1e-6, elapsed)
                avg_glen = float(np.mean(env.game_lengths)) if env.game_lengths else 0.0
                log(f"step {step:>5} | states {total:>9,}/{args.target_states:,} "
                    f"| fps {fps:>5.0f} | lr {cur_lr:.1e} | games {env.games_played} "
                    f"| avg_glen {avg_glen:.0f} "
                    f"| L {metrics['loss']:.4f} (pol {metrics['loss_pol']:+.4f} "
                    f"val {metrics['loss_val']:.4f} ent {metrics['entropy']:.3f}"
                    + (f" kl {metrics['loss_kl']:.3f}" if teacher is not None else "")
                    + ")")
                try:
                    with open(rl_stats_path, "w") as f:
                        json.dump({
                            "stage": "RL", "arch": "v13_3",
                            "step": step, "states": total,
                            "target": args.target_states, "fps": fps,
                            "elapsed_sec": elapsed, "lr": cur_lr,
                            "games_played": env.games_played,
                            "avg_game_len": avg_glen,
                            "loss": metrics["loss"],
                            "loss_pol": metrics["loss_pol"],
                            "loss_val": metrics["loss_val"],
                            "entropy": metrics["entropy"],
                            "loss_kl": metrics["loss_kl"],
                            "eval_history": eval_history,
                            "ts": int(time.time()),
                        }, f)
                except Exception as e:
                    print(f"[stats] write failed: {e}")

            _save_g = args.save_every_games > 0 and env.games_played - last_save_games >= args.save_every_games
            _save_s = args.save_every_games == 0 and total - last_save >= args.save_every
            if _save_g or _save_s:
                ckpt_path = os.path.join(CHECKPOINT_DIR, f"rl_{total // 1000}K.pt")
                save_dict = {
                    "model_state_dict": student.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "step": step, "total": total,
                    "games_played": env.games_played,
                    "eval_history": eval_history,
                }
                torch.save(save_dict, ckpt_path)
                latest_path = os.path.join(CHECKPOINT_DIR, "model_latest.pt")
                torch.save(save_dict, latest_path)
                log(f"[checkpoint] {ckpt_path} + model_latest.pt (games={env.games_played})")
                last_save = total
                last_save_games = env.games_played

            _eval_g = args.eval_every_games > 0 and env.games_played - last_eval_games >= args.eval_every_games
            _eval_s = args.eval_every_games == 0 and total - last_eval >= args.eval_every
            if (_eval_g or _eval_s) and total > 0:
                log(f"[eval] starting ({args.eval_games} games, at RL game {env.games_played})...")
                wr = quick_eval(student, device, n_games=args.eval_games)
                eval_history.append([total, wr])
                log(f"[eval] WR = {wr:.1f}% at {total:,} states ({env.games_played} games)")
                last_eval = total
                last_eval_games = env.games_played

    # Final save
    final_path = os.path.join(CHECKPOINT_DIR, "model_rl.pt")
    final_latest = os.path.join(CHECKPOINT_DIR, "model_latest.pt")
    save_dict = {
        "model_state_dict": student.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step, "total": total,
        "games_played": env.games_played,
        "eval_history": eval_history,
    }
    torch.save(save_dict, final_path)
    torch.save(save_dict, final_latest)
    log(f"[done] processed {total:,} states across {env.games_played} games.")
    log(f"[done] saved → {final_path} + {final_latest}")
    write_chain("completed")
    rl_log.close()


if __name__ == "__main__":
    main()
