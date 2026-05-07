"""V13.5 SL distillation — V13.2 → V135Symmetric with token-ID permutation augmentation.

What's different vs train_v132_sl.py / train_v133_sl.py
-------------------------------------------------------
- Student is `V135Symmetric` (token-symmetric encoder, rank-indexed output).
- Teacher is V13.2 (`MinimalCNN14`, asymmetric per-token encoder/output).
- Per state, we sample a random permutation π of own token-IDs.
  - We feed V13.2 a token-permuted state (V17 encoding); V13.2 produces a
    per-token policy over PERMUTED token IDs.
  - We aggregate that per-token policy into a per-rank policy using the
    permuted state's rank mapping. The aggregation is invariant to π (proved
    by unit tests), so in expectation the aggregated target equals the
    symmetrized V13.2 policy. Random sampling of π gives unbiased gradients.
- Student receives the permutation-INVARIANT V18 encoding + rank masks
  computed from the permuted state (also invariant under π).

So the student input is unchanged across permutation samples; only the
target shifts. Over many samples, the student learns the symmetrized
V13.2 policy with V13.2's token-ID-specific biases averaged out.

Usage
-----
    TD_LUDO_RUN_NAME=v135 python train_v135_sl.py \\
        --teacher /path/to/v132/model_latest.pt \\
        --target-states 2_000_000 \\
        --port 8798
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
from td_ludo.game.encoder_v18_symmetric import encode_state_v18_symmetric, V18_CHANNELS
from td_ludo.game.rank_mapping import (
    HOME_POS,
    MAX_RANK_SLOTS,
    state_to_rank_mapping,
    aggregate_token_policy_to_ranks,
    legal_mask_per_rank,
    rank_to_token_id,
    permute_own_tokens,
)
from td_ludo.models.v13_5 import V135Symmetric, compute_rank_masks
from experiments.distillation_14ch.model_14ch import MinimalCNN14


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher", required=True, help="V13.2 checkpoint path")
    p.add_argument("--run-name", default=None)
    p.add_argument("--target-states", type=int, default=2_000_000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lr-end", type=float, default=1e-4)
    p.add_argument("--save-every", type=int, default=500_000)
    p.add_argument("--eval-every", type=int, default=200_000)
    p.add_argument("--eval-games", type=int, default=200)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--policy-coeff", type=float, default=1.0)
    p.add_argument("--value-coeff", type=float, default=0.5)
    p.add_argument("--moves-coeff", type=float, default=0.01)
    # V13.5 architecture
    p.add_argument("--num-res-blocks", type=int, default=6)
    p.add_argument("--num-channels", type=int, default=96)
    p.add_argument("--head-hidden", type=int, default=64)
    # Augmentation
    p.add_argument("--no-perm-augment", action="store_true",
                   help="Disable random token-ID permutation augmentation (debug)")
    p.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--port", type=int, default=8798)
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


# ── Teacher ────────────────────────────────────────────────────────────────
def load_v132_teacher(path, device):
    print(f"[Teacher] Loading V13.2 from {path}...")
    model = MinimalCNN14(num_res_blocks=10, num_channels=128, in_channels=17)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    if any(k.startswith("_orig_mod.") for k in sd):
        sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    print(f"[Teacher] params: {sum(p.numel() for p in model.parameters()):,}")
    return model


# ── Self-play env (no history; pure single-frame distillation) ────────────
class SymmetricDistillEnv:
    """Vectorised self-play env over B parallel games. Each get_batch()
    advances every game to its next decision state and returns:
        - student inputs: (B, V18_CH, 15, 15) symmetric encoder
        - student rank masks: (B, 4, 15, 15)
        - rank-indexed legal masks: (B, 4)
        - teacher inputs: (B, V17_CH, 15, 15) standard V13.2 encoder
          BUT with own token-IDs permuted per the per-game sampled π.
        - teacher legal masks: (B, 4) — token-ID-indexed AFTER permutation
          (so V13.2's output indexing is consistent with the permuted state)
        - per-game perm + rank_token_ids (for target aggregation)
    """

    def __init__(self, batch_size, max_game_len=400, perm_augment=True):
        self.batch_size = batch_size
        self.max_game_len = max_game_len
        self.perm_augment = perm_augment
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
        student_inputs = []
        rank_masks_list = []
        rank_legal_masks = []
        teacher_inputs = []
        teacher_legal_masks = []
        rank_token_ids_per_state = []
        legal_token_ids_per_state = []
        chosen_perms = []

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

                # Decision state. Build student inputs (permutation-invariant).
                pp = game.player_positions[cp]
                rank_positions, rank_tokens = state_to_rank_mapping(pp)
                student_x = encode_state_v18_symmetric(game)
                rank_masks = compute_rank_masks(game)
                rank_legal = legal_mask_per_rank(legal, rank_tokens)

                # Sample permutation for teacher augmentation
                if self.perm_augment:
                    perm = list(range(4))
                    random.shuffle(perm)
                else:
                    perm = [0, 1, 2, 3]

                # Build the permuted state for the teacher.
                if perm == [0, 1, 2, 3]:
                    teacher_state = game
                else:
                    teacher_state = permute_own_tokens(game, perm)

                # V17 encoding for the teacher
                teacher_x = encode_state_v17(teacher_state)

                # The teacher's policy is over token-IDs in the PERMUTED state.
                # Build teacher legal mask: legal in the permuted state.
                # Token-ID t in permuted state was OLD token perm[t]; legal
                # iff perm[t] was legal originally.
                t_legal_mask = np.zeros(4, dtype=np.float32)
                for t in range(4):
                    if perm[t] in legal:
                        t_legal_mask[t] = 1.0

                # Aggregation needs the permuted state's rank → token-IDs map
                # (rank ordering is identical to original since permutation
                # invariance, but token-IDs at each rank shift).
                _, rank_tokens_perm = state_to_rank_mapping(teacher_state.player_positions[cp])

                decision_idxs.append(i)
                cps.append(cp)
                student_inputs.append(student_x)
                rank_masks_list.append(rank_masks)
                rank_legal_masks.append(rank_legal)
                teacher_inputs.append(teacher_x)
                teacher_legal_masks.append(t_legal_mask)
                rank_token_ids_per_state.append(rank_tokens_perm)
                legal_token_ids_per_state.append(list(legal))
                chosen_perms.append(perm)
                break

        return (
            decision_idxs,
            cps,
            np.stack(student_inputs, axis=0),       # (B, 13, 15, 15)
            np.stack(rank_masks_list, axis=0),      # (B, 4, 15, 15)
            np.stack(rank_legal_masks, axis=0),     # (B, 4)
            np.stack(teacher_inputs, axis=0),       # (B, 17, 15, 15)
            np.stack(teacher_legal_masks, axis=0),  # (B, 4) token-id-indexed
            rank_token_ids_per_state,
            legal_token_ids_per_state,
            chosen_perms,
        )

    def apply_actions_by_rank(self, decision_idxs, chosen_ranks,
                              rank_token_ids_per_state, legal_token_ids_per_state,
                              rank_legal_masks):
        """Apply chosen ranks (sampled from teacher targets) to advance games.

        For each i: convert rank → token-id (using ORIGINAL state's rank
        mapping, since the game itself isn't permuted — only the teacher's
        view was permuted). The chosen_rank we sample is over the TEACHER's
        per-rank distribution, but rank ordering is identical between
        original and permuted state, so we use the original rank mapping.
        """
        # We need ORIGINAL rank mappings (not permuted). Re-derive on the fly.
        for k, i in enumerate(decision_idxs):
            game = self.games[i]
            cp = int(game.current_player)
            pp = game.player_positions[cp]
            _, rank_tokens_orig = state_to_rank_mapping(pp)
            legal = legal_token_ids_per_state[k]
            rank = int(chosen_ranks[k])
            # Safety: rank must be in legal mask
            if rank_legal_masks[k][rank] < 0.5:
                # Fallback: any legal rank
                legal_ranks = np.argwhere(rank_legal_masks[k] > 0.5).flatten().tolist()
                rank = int(legal_ranks[0]) if legal_ranks else 0
            tok = rank_to_token_id(rank, legal, rank_tokens_orig)
            self.games[i] = ludo_cpp.apply_move(self.games[i], int(tok))
            self.step_count[i] += 1


# ── Eval ───────────────────────────────────────────────────────────────────
def quick_eval(student, device, n_games=200):
    """Greedy V13.5 vs random heuristic-bot mix."""
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
                    pp = state.player_positions[cp]
                    _, rank_tokens = state_to_rank_mapping(pp)
                    rank_legal = legal_mask_per_rank(legal, rank_tokens)
                    enc = encode_state_v18_symmetric(state)
                    rm = compute_rank_masks(state)
                    with torch.no_grad():
                        x = torch.from_numpy(enc).unsqueeze(0).to(device, dtype=torch.float32)
                        rmt = torch.from_numpy(rm).unsqueeze(0).to(device, dtype=torch.float32)
                        lmt = torch.from_numpy(rank_legal).unsqueeze(0).to(device, dtype=torch.float32)
                        logits = student.forward_policy_only(x, rmt, lmt)
                        rank = int(logits.argmax(dim=1).item())
                    action = rank_to_token_id(rank, legal, rank_tokens)
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

    def log_message(self, *args):
        pass


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
    print("V13.5 SL DISTILLATION (V13.2 → V135Symmetric)")
    print("=" * 70)
    print(f"  device:           {device}")
    print(f"  teacher:          {args.teacher}")
    print(f"  checkpoint dir:   {CHECKPOINT_DIR}")
    print(f"  arch:             V135Symmetric ({args.num_res_blocks}x{args.num_channels})")
    print(f"  encoder:          V18 symmetric (13ch)")
    print(f"  output:           rank-indexed (4 logits)")
    print(f"  perm_augment:     {not args.no_perm_augment}")
    print(f"  target_states:    {args.target_states:,}")
    print(f"  batch_size:       {args.batch_size}")
    print("=" * 70)

    teacher = load_v132_teacher(args.teacher, device)

    student = V135Symmetric(
        num_res_blocks=args.num_res_blocks,
        num_channels=args.num_channels,
        in_channels=V18_CHANNELS,
        head_hidden=args.head_hidden,
    )
    student.to(device).train()
    print(f"[Student] V135Symmetric params: {student.count_parameters():,}")

    optimizer = torch.optim.Adam(student.parameters(), lr=args.lr)

    init_path = os.path.join(CHECKPOINT_DIR, "sl_init.pt")
    torch.save(student.state_dict(), init_path)
    print(f"[Init] saved {init_path}")

    def write_chain(phase):
        with open(chain_path, "w") as f:
            json.dump({"stage": "SL", "phase": phase, "arch": "v13_5",
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

    env = SymmetricDistillEnv(args.batch_size, perm_augment=not args.no_perm_augment)
    log(f"Starting SL: target {args.target_states:,} states.")

    t_start = time.time()
    last_save = 0
    last_eval = 0
    total = 0
    step = 0
    eval_history = []
    recent = collections.deque(maxlen=200)
    recent_pol = collections.deque(maxlen=200)
    recent_val = collections.deque(maxlen=200)

    while total < args.target_states:
        (decision_idxs, cps, student_in, rank_masks, rank_legal,
         teacher_in, teacher_legal, rank_token_ids_perm,
         legal_token_ids, perms) = env.get_batch()

        teacher_t = torch.from_numpy(teacher_in).to(device, dtype=torch.float32)
        student_t = torch.from_numpy(student_in).to(device, dtype=torch.float32)
        rmt = torch.from_numpy(rank_masks).to(device, dtype=torch.float32)
        rlt = torch.from_numpy(rank_legal).to(device, dtype=torch.float32)
        teacher_lmt = torch.from_numpy(teacher_legal).to(device, dtype=torch.float32)

        # Teacher forward (no grads). Teacher policy is over PERMUTED token-IDs.
        with torch.no_grad():
            t_policy, t_win, t_moves = teacher(teacher_t, teacher_lmt)
        t_policy_np = t_policy.cpu().numpy()  # (B, 4)

        # Aggregate teacher per-token policy → per-rank target (using
        # PERMUTED state's rank-token mapping)
        target_rank_probs = np.zeros((args.batch_size, MAX_RANK_SLOTS), dtype=np.float32)
        for k in range(args.batch_size):
            target_rank_probs[k] = aggregate_token_policy_to_ranks(
                t_policy_np[k], rank_token_ids_perm[k]
            )
        # Renormalize defensively (legal mask can affect the teacher's policy
        # so we ensure the rank target sums to 1 over legal ranks)
        for k in range(args.batch_size):
            s = target_rank_probs[k].sum()
            if s > 1e-6:
                target_rank_probs[k] /= s
        target_t = torch.from_numpy(target_rank_probs).to(device)

        # Sample teacher action by rank (used to advance the games)
        # We sample from the per-rank target distribution (over legal ranks).
        # Note: rank ordering in PERMUTED state == in ORIGINAL state, so the
        # rank index is consistent.
        chosen_ranks = torch.multinomial(target_t.clamp_min(1e-12), num_samples=1).squeeze(1).cpu().numpy()
        env.apply_actions_by_rank(
            decision_idxs, chosen_ranks, rank_token_ids_perm,
            legal_token_ids, rank_legal,
        )

        # LR schedule
        progress = total / args.target_states
        cur_lr = args.lr_end + 0.5 * (args.lr - args.lr_end) * (1 + np.cos(np.pi * progress))
        for g in optimizer.param_groups:
            g["lr"] = cur_lr

        # Student forward (with grads)
        s_policy, s_win, s_moves = student(student_t, rmt, rlt)

        # Distillation losses
        s_log = torch.log(s_policy + 1e-8)
        loss_policy = F.kl_div(s_log, target_t, reduction="batchmean", log_target=False)
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

        recent.append(loss.item())
        recent_pol.append(loss_policy.item())
        recent_val.append(loss_win.item())

        if step % args.log_every == 0:
            elapsed = time.time() - t_start
            fps = total / max(1e-6, elapsed)
            log(f"step {step:>5} | states {total:>9,}/{args.target_states:,} "
                f"| fps {fps:>5.0f} | lr {cur_lr:.1e} "
                f"| L {loss.item():.4f} (pol {loss_policy.item():.3f} "
                f"val {loss_win.item():.3f} mov {loss_moves.item():.3f})")
            try:
                with open(sl_stats_path, "w") as f:
                    json.dump({
                        "stage": "SL", "arch": "v13_5",
                        "step": step, "states": total,
                        "target": args.target_states, "fps": fps,
                        "elapsed_sec": elapsed, "lr": cur_lr,
                        "loss": float(np.mean(recent)),
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
