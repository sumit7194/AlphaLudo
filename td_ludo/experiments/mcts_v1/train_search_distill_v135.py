"""Step 1 — Train V13.2-architecture student on search-augmented targets.

Reads the buffer produced by `generate_search_data.py`:
  states           (N, 17, 15, 15) float32  V17 encoded root states
  search_policies  (N, 4)          float32  soft policy targets (softmax(Q/τ))
  search_values    (N,)            float32  in [-1, +1] from root POV
  search_actions   (N,)            int8     argmax actions (unused; soft policy
                                              is the policy target)
  outcomes         (N,)            float32  0/1 from root POV

Trains a fresh MinimalCNN14 (10×128, 17ch input) student with loss:
  α_p · KL(student.π || search_policy)
  + α_v · MSE(student.V, search_value_to_winprob)
  + α_o · BCE(student.V, outcome)

where `search_value_to_winprob = (search_value + 1) / 2` converts from
[-1,+1] to [0,1] for direct comparison with the network's sigmoid output.

α_p=1.0, α_v=0.5, α_o=0.5 by default — search target on policy is the
primary signal; value gets dual supervision from search + outcome.

Output: `checkpoints/mcts_v1_step1_distill/model_latest.pt` (final),
`model_sl.pt` (alias for compatibility with V13.2 RL pipeline).

Usage:
    TD_LUDO_RUN_NAME=mcts_v1_step1_distill \\
    python -m experiments.mcts_v1.train_search_distill \\
        --buffer runs/mcts_v1_search_buffer.npz \\
        --epochs 5 \\
        --batch-size 1024 \\
        --lr 1e-3 --lr-end 1e-4 \\
        --port 8794
"""
from __future__ import annotations

import argparse
import functools
import json
import os
import random
import sys
import threading
import time
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import td_ludo_cpp as ludo_cpp
from td_ludo.game.encoder_v18_production import (
    encode_state_v18_production, V18_PROD_CHANNELS,
)
from td_ludo.models.v13_5_production import V135ProductionAdapter


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--buffer", required=True, help="Path to search-data .npz")
    p.add_argument("--init-from", default=None,
                   help="Optional V13.5 checkpoint to warm-start from. "
                        "Without this the student starts from random init.")
    p.add_argument("--run-name", default=None,
                   help="Override TD_LUDO_RUN_NAME (sets checkpoint dir).")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lr-end", type=float, default=1e-4,
                   help="Cosine-decay LR target")
    p.add_argument("--policy-coeff", type=float, default=1.0)
    p.add_argument("--search-value-coeff", type=float, default=0.5)
    p.add_argument("--outcome-coeff", type=float, default=0.5)
    p.add_argument("--save-every", type=int, default=2_000_000,
                   help="Save intermediate checkpoint every N states")
    p.add_argument("--eval-every", type=int, default=1_000_000)
    p.add_argument("--eval-games", type=int, default=200)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--num-res-blocks", type=int, default=10)
    p.add_argument("--num-channels", type=int, default=128)
    p.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--port", type=int, default=8794, help="Dashboard port")
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


# ── Dashboard server (same shape as train_v132_sl.py) ──────────────────────
class _SLHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, directory=None, sl_path=None, chain_path=None,
                 landing=None, **kwargs):
        self._sl_path = sl_path
        self._chain_path = chain_path
        self._landing = landing
        super().__init__(*args, directory=directory, **kwargs)

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
            self.send_response(404); self.end_headers()


def start_dashboard(port, sl_path, chain_path, dashboard_dir):
    landing = None
    for cand in ("sl_dashboard.html", "v13_dashboard.html",
                 "v12_dashboard.html", "index.html"):
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


# ── Eval (mirror of train_v132_sl.py's quick_eval) ─────────────────────────
def quick_eval(student, device, n_games=200):
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
            if cp == model_player:
                if len(legal) == 1:
                    action = legal[0]
                else:
                    enc = encode_state_v18_production(state)
                    mask = np.zeros(4, dtype=np.float32)
                    for m in legal:
                        mask[m] = 1.0
                    with torch.no_grad():
                        s_t = torch.from_numpy(enc).unsqueeze(0).to(device)
                        m_t = torch.from_numpy(mask).unsqueeze(0).to(device)
                        logits = student.forward_policy_only(s_t, m_t)
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


# ── Main ────────────────────────────────────────────────────────────────────
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
    print("MCTS_V1 STEP 1 — Search-augmented distillation")
    print("=" * 70)
    print(f"  device:        {device}")
    print(f"  buffer:        {args.buffer}")
    print(f"  checkpoint dir:{CHECKPOINT_DIR}")
    print(f"  arch:          {args.num_res_blocks} blocks × "
          f"{args.num_channels} ch (17ch input, V13.2 family)")
    print(f"  epochs:        {args.epochs}")
    print(f"  batch_size:    {args.batch_size}")
    print(f"  lr:            {args.lr} → {args.lr_end} (cosine)")
    print(f"  loss weights:  policy={args.policy_coeff} "
          f"search_value={args.search_value_coeff} outcome={args.outcome_coeff}")
    print("=" * 70)

    # Load buffer
    print(f"[Train] Loading buffer {args.buffer}...")
    d = np.load(args.buffer)
    states = d["states"]                    # (N, 17, 15, 15)
    search_policies = d["search_policies"]  # (N, 4)
    search_values = d["search_values"]      # (N,) in [-1, +1]
    outcomes = d["outcomes"]                # (N,) in {0, 1}
    N = states.shape[0]
    print(f"[Train] Buffer: {N:,} states, {N * args.epochs:,} train steps total")

    # Convert search_value [-1,+1] → win-prob [0,1] for MSE against sigmoid output
    search_winprob = (search_values + 1.0) / 2.0
    search_winprob = np.clip(search_winprob, 0.0, 1.0)

    # Move buffer to device-friendly tensors (kept on CPU; sliced per batch)
    states_t = torch.from_numpy(states)            # CPU; move per batch
    policies_t = torch.from_numpy(search_policies)
    swp_t = torch.from_numpy(search_winprob.astype(np.float32))
    outcomes_t = torch.from_numpy(outcomes.astype(np.float32))

    # Student model
    print(f"[Train] Initializing student (V135ProductionAdapter, "
          f"{args.num_res_blocks}×{args.num_channels}, {V18_PROD_CHANNELS}ch)...")
    student = V135ProductionAdapter(
        num_res_blocks=args.num_res_blocks,
        num_channels=args.num_channels,
    )
    # Optional warm-start from V13.5_exp ckpt — much better than fresh.
    # Without it the search-distilled student starts from random init.
    if args.init_from:
        ck = torch.load(args.init_from, map_location=device, weights_only=False)
        sd = ck.get("model_state_dict", ck) if isinstance(ck, dict) else ck
        if any(k.startswith("_orig_mod.") for k in sd):
            sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
        student.load_state_dict(sd, strict=False)
        print(f"[Train] WARM-STARTED from {args.init_from}")
    student.to(device).train()
    n_params = sum(p.numel() for p in student.parameters())
    print(f"[Train] Student params: {n_params:,}")

    optimizer = torch.optim.Adam(student.parameters(), lr=args.lr)
    init_path = os.path.join(CHECKPOINT_DIR, "sl_init.pt")
    torch.save(student.state_dict(), init_path)
    print(f"[Train] saved init weights → {init_path}")

    def write_chain(phase):
        with open(chain_path, "w") as f:
            json.dump({
                "stage": "SL", "phase": phase, "arch": "mcts_v1_step1",
                "run_name": os.environ.get("TD_LUDO_RUN_NAME", "-"),
                "ts": int(time.time()),
            }, f)
    write_chain("training")

    if not args.no_dashboard:
        dash_dir = os.path.dirname(os.path.abspath(__file__))
        # Use the project root dashboards if they exist
        for cand_dir in (
            dash_dir,
            os.path.abspath(os.path.join(dash_dir, "../..")),
        ):
            if any(os.path.exists(os.path.join(cand_dir, f))
                   for f in ("sl_dashboard.html", "v13_dashboard.html")):
                start_dashboard(args.port, sl_stats_path, chain_path, cand_dir)
                break

    sl_log = open(sl_log_path, "a")
    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        sl_log.write(line + "\n")
        sl_log.flush()

    log(f"Starting training: {args.epochs} epochs over {N:,}-state buffer.")
    t_start = time.time()
    total_steps = (N // args.batch_size) * args.epochs
    step = 0
    total_states = 0
    last_save = 0
    last_eval = 0
    recent_loss = []
    recent_pol = []
    recent_sval = []
    recent_oval = []
    eval_history = []

    for epoch in range(args.epochs):
        # Shuffle the buffer at the start of each epoch
        perm = np.random.permutation(N)
        for start in range(0, N - args.batch_size + 1, args.batch_size):
            idx = perm[start:start + args.batch_size]
            x = states_t[idx].to(device)
            p_target = policies_t[idx].to(device)
            v_search_target = swp_t[idx].to(device)
            v_outcome_target = outcomes_t[idx].to(device)

            # LR cosine over total steps
            progress = step / max(1, total_steps)
            cur_lr = args.lr_end + 0.5 * (args.lr - args.lr_end) * (1 + np.cos(np.pi * progress))
            for g in optimizer.param_groups:
                g["lr"] = cur_lr

            # Forward (use permissive mask — search policy is already over all 4)
            mask = torch.ones(x.size(0), 4, device=device)
            # V135ProductionAdapter returns 4-tuple (policy, win_prob, moves, progress).
            out = student(x, mask)
            policy, win_prob = out[0], out[1]

            # Losses
            log_policy = torch.log(policy + 1e-8)
            loss_policy = F.kl_div(
                log_policy, p_target, reduction="batchmean", log_target=False
            )
            loss_search_v = F.mse_loss(win_prob, v_search_target)
            loss_outcome = F.binary_cross_entropy(win_prob, v_outcome_target)

            loss = (
                args.policy_coeff * loss_policy
                + args.search_value_coeff * loss_search_v
                + args.outcome_coeff * loss_outcome
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step += 1
            total_states += args.batch_size
            recent_loss.append(loss.item())
            recent_pol.append(loss_policy.item())
            recent_sval.append(loss_search_v.item())
            recent_oval.append(loss_outcome.item())
            for r in (recent_loss, recent_pol, recent_sval, recent_oval):
                if len(r) > 200:
                    r.pop(0)

            if step % args.log_every == 0:
                elapsed = time.time() - t_start
                fps = total_states / max(1e-6, elapsed)
                log(f"epoch {epoch + 1}/{args.epochs} step {step:>6}/{total_steps} "
                    f"| states {total_states:>10,} | fps {fps:>5.0f} | lr {cur_lr:.1e} "
                    f"| L {loss.item():.4f} (pol {loss_policy.item():.3f} "
                    f"sval {loss_search_v.item():.3f} oval {loss_outcome.item():.3f})")
                try:
                    with open(sl_stats_path, "w") as f:
                        json.dump({
                            "stage": "SL", "arch": "mcts_v1_step1",
                            "step": step, "states": total_states,
                            "target": total_steps * args.batch_size,
                            "fps": fps, "elapsed_sec": elapsed, "lr": cur_lr,
                            "loss": float(np.mean(recent_loss)),
                            "loss_policy": float(np.mean(recent_pol)),
                            "loss_value": float(np.mean(recent_sval)),
                            "loss_outcome": float(np.mean(recent_oval)),
                            "eval_history": eval_history,
                            "ts": int(time.time()),
                        }, f)
                except Exception as e:
                    print(f"[stats] write failed: {e}")

            if total_states - last_save >= args.save_every:
                ckpt_path = os.path.join(
                    CHECKPOINT_DIR, f"sl_{total_states // 1_000_000}M.pt"
                )
                torch.save(student.state_dict(), ckpt_path)
                log(f"[checkpoint] {ckpt_path}")
                last_save = total_states

            if total_states - last_eval >= args.eval_every and total_states > 0:
                log(f"[eval] starting ({args.eval_games} games vs random bot mix)...")
                wr = quick_eval(student, device, n_games=args.eval_games)
                eval_history.append([total_states, wr])
                log(f"[eval] WR = {wr:.1f}% at {total_states:,} states")
                last_eval = total_states

    # Final save → both model_sl.pt and model_latest.pt
    final_sl = os.path.join(CHECKPOINT_DIR, "model_sl.pt")
    final_latest = os.path.join(CHECKPOINT_DIR, "model_latest.pt")
    torch.save(student.state_dict(), final_sl)
    torch.save(student.state_dict(), final_latest)
    log(f"[done] processed {total_states:,} states in {time.time() - t_start:.0f}s")
    log(f"[done] saved → {final_sl} and {final_latest}")
    write_chain("completed")
    sl_log.close()


if __name__ == "__main__":
    main()
