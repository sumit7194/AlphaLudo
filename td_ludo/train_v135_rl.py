"""V13.5 RL — self-play REINFORCE-with-baseline on V135Symmetric.

Mirrors train_v133_rl.py's recipe (KL anchor + multi-legal filter + cosine
LR + entropy bonus) with three structural differences:

1. **No history.** V13.5 is stateless single-frame; we drop the K=8 deque.
2. **Rank-indexed action space.** Model outputs 4 logits per canonical
   rank (rank 0 = most-advanced own token). At rollout time we sample
   a rank and map it to a legal token-ID via `rank_to_token_id`.
   Trajectories store the chosen rank, not the token-ID — gradients
   flow through rank logits.
3. **KL anchor target = V13.5_SL, not V13.2.** Anchoring to V13.2 would
   pull the policy back through the rank→token aggregation, defeating
   the symmetry constraint. V13.5_SL is the natural anchor since both
   live in the same per-rank policy space.

Optional H2H gating: every --h2h-gate-every states, run a 200-game H2H
vs V13.2 (in-memory, mirrored seeds, greedy). The result is logged to
rl_stats.json under "h2h_history" and printed; this is the only signal
that's not teacher-bound.

Usage
-----
    TD_LUDO_RUN_NAME=v135_rl python train_v135_rl.py \\
        --init checkpoints/v135_full/model_latest.pt \\
        --kl-teacher checkpoints/v135_full/model_latest.pt \\
        --h2h-opponent checkpoints/v132/model_latest.pt \\
        --target-states 20000000 \\
        --port 8799
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
    state_to_rank_mapping,
    legal_mask_per_rank,
    rank_to_token_id,
)
from td_ludo.models.v13_5 import V135Symmetric, compute_rank_masks
from experiments.distillation_14ch.model_14ch import MinimalCNN14


# ── Args ───────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--init", default=None,
                   help="Initial student checkpoint (V13.5_SL recommended)")
    p.add_argument("--resume", action="store_true",
                   help="Resume from <ckpt-dir>/model_latest.pt")
    p.add_argument("--kl-teacher", default=None,
                   help="V13.5_SL checkpoint for KL anchor (defaults to --init)")
    p.add_argument("--h2h-opponent", default=None,
                   help="V13.2 checkpoint for periodic H2H gating (optional)")
    p.add_argument("--run-name", default=None)
    p.add_argument("--target-states", type=int, default=20_000_000)
    p.add_argument("--max-game-len", type=int, default=400)
    p.add_argument("--parallel-games", type=int, default=64)
    p.add_argument("--train-chunk", type=int, default=2048)
    p.add_argument("--minibatch-size", type=int, default=256)
    p.add_argument("--train-epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--lr-end", type=float, default=5e-6)
    p.add_argument("--entropy-coeff", type=float, default=0.02)
    p.add_argument("--value-coeff", type=float, default=0.5)
    p.add_argument("--kl-anchor-coeff", type=float, default=0.1)
    p.add_argument("--save-every-games", type=int, default=20_000)
    p.add_argument("--save-every", type=int, default=500_000)
    p.add_argument("--eval-every-games", type=int, default=20_000)
    p.add_argument("--eval-every", type=int, default=200_000)
    p.add_argument("--eval-games", type=int, default=3000)
    p.add_argument("--h2h-gate-every", type=int, default=2_000_000,
                   help="Every N states, run H2H vs --h2h-opponent (0=disabled)")
    p.add_argument("--h2h-games", type=int, default=200)
    p.add_argument("--log-every", type=int, default=10)
    # V13.5 architecture (must match the SL checkpoint)
    p.add_argument("--num-res-blocks", type=int, default=10)
    p.add_argument("--num-channels", type=int, default=128)
    p.add_argument("--head-hidden", type=int, default=64)
    # V13.2 H2H opponent architecture
    p.add_argument("--v132-num-res-blocks", type=int, default=10)
    p.add_argument("--v132-num-channels", type=int, default=128)
    p.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--port", type=int, default=8799)
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


# ── Self-play env (no history; per-rank policy) ────────────────────────────
class SelfPlayEnv:
    """B parallel games. Each step computes V18 encoding + rank-masks +
    per-rank legal mask. Trajectories store (enc, rmasks, rlegal, chosen_rank,
    v_pred, cp) so the loss can replay through the per-rank policy head.
    """

    def __init__(self, batch_size, max_game_len=400):
        self.batch_size = batch_size
        self.max_game_len = max_game_len
        self.games = [ludo_cpp.create_initial_state_2p() for _ in range(batch_size)]
        # trajectory[i] = list of (enc, rmasks, rlegal, chosen_rank, v_pred, cp)
        self.trajectory = [[] for _ in range(batch_size)]
        self.consec_sixes = np.zeros((batch_size, 4), dtype=np.int32)
        self.step_count = np.zeros(batch_size, dtype=np.int32)
        self.games_played = 0
        self.game_lengths = []

    def _reset(self, i):
        self.games[i] = ludo_cpp.create_initial_state_2p()
        self.trajectory[i] = []
        self.consec_sixes[i] = 0
        self.step_count[i] = 0

    def _finalize(self, i, winner):
        out = []
        for enc, rmasks, rlegal, chosen_rank, v_pred, cp in self.trajectory[i]:
            G = 1.0 if cp == winner else -1.0
            out.append((enc, rmasks, rlegal, chosen_rank, v_pred, G))
        self.games_played += 1
        self.game_lengths.append(len(self.trajectory[i]))
        if len(self.game_lengths) > 200:
            self.game_lengths.pop(0)
        return out

    def spin_to_decision(self):
        decision_idxs = []
        cps = []
        encs = []          # (B, 13, 15, 15)
        rmasks_list = []   # (B, 4, 15, 15)
        rlegals = []       # (B, 4) — rank-indexed legal mask
        legal_lists = []   # raw token-IDs (for rank→token mapping)
        rank_token_ids_list = []  # per-state list-of-lists for rank_to_token_id

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

                # Decision state — compute V13.5 inputs
                pp = game.player_positions[cp]
                _, rank_tokens = state_to_rank_mapping(pp)
                rank_legal = legal_mask_per_rank(legal, rank_tokens)
                enc = encode_state_v18_symmetric(game).astype(np.float32)
                rmasks = compute_rank_masks(game).astype(np.float32)

                decision_idxs.append(i)
                cps.append(cp)
                encs.append(enc)
                rmasks_list.append(rmasks)
                rlegals.append(rank_legal.astype(np.float32))
                legal_lists.append(legal)
                rank_token_ids_list.append(rank_tokens)
                break

        return (
            decision_idxs,
            cps,
            np.stack(encs, axis=0),
            np.stack(rmasks_list, axis=0),
            np.stack(rlegals, axis=0),
            legal_lists,
            rank_token_ids_list,
        )

    def apply_actions(self, decision_idxs, cps, encs, rmasks, rlegals,
                      legal_lists, rank_token_ids_list, ranks, v_preds):
        """ranks: chosen rank per state (B,). Map to token-id, apply, store."""
        finished = []
        for k, i in enumerate(decision_idxs):
            chosen_rank = int(ranks[k])
            # Defensive: if rank_legal[k][chosen_rank] == 0, fall back to first legal rank
            if rlegals[k][chosen_rank] == 0:
                legal_ranks = np.where(rlegals[k] > 0)[0]
                chosen_rank = int(legal_ranks[0]) if len(legal_ranks) else 0
            token_id = rank_to_token_id(chosen_rank, legal_lists[k], rank_token_ids_list[k])
            if token_id < 0 or token_id not in legal_lists[k]:
                token_id = legal_lists[k][0]
                chosen_rank = 0  # mismatch fallback — shouldn't happen
            self.trajectory[i].append((
                encs[k], rmasks[k], rlegals[k],
                chosen_rank, float(v_preds[k]), int(cps[k]),
            ))
            self.games[i] = ludo_cpp.apply_move(self.games[i], int(token_id))
            self.step_count[i] += 1

            game = self.games[i]
            if game.is_terminal:
                winner = int(ludo_cpp.get_winner(game))
                finished.extend(self._finalize(i, winner))
            elif self.step_count[i] >= self.max_game_len:
                # Truncate as draw
                for enc, rm, rl, cr, v, c in self.trajectory[i]:
                    finished.append((enc, rm, rl, cr, v, 0.0))
                self.games_played += 1
                self.game_lengths.append(len(self.trajectory[i]))
                if len(self.game_lengths) > 200:
                    self.game_lengths.pop(0)
                self.trajectory[i] = []

        return finished


# ── Eval (V13.5 vs scripted bots) ──────────────────────────────────────────
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
                    pp = state.player_positions[cp]
                    _, rank_tokens = state_to_rank_mapping(pp)
                    rank_legal = legal_mask_per_rank(legal, rank_tokens)
                    enc = encode_state_v18_symmetric(state).astype(np.float32)
                    rm = compute_rank_masks(state).astype(np.float32)
                    with torch.no_grad():
                        x = torch.from_numpy(enc).unsqueeze(0).to(device)
                        rmt = torch.from_numpy(rm).unsqueeze(0).to(device)
                        lmt = torch.from_numpy(rank_legal.astype(np.float32)).unsqueeze(0).to(device)
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


# ── In-memory H2H vs V13.2 (mirrored seeds, greedy) ────────────────────────
def _v135_select(student, device, state, legal):
    if len(legal) == 1:
        return legal[0]
    pp = state.player_positions[int(state.current_player)]
    _, rank_tokens = state_to_rank_mapping(pp)
    rank_legal = legal_mask_per_rank(legal, rank_tokens)
    enc = encode_state_v18_symmetric(state).astype(np.float32)
    rm = compute_rank_masks(state).astype(np.float32)
    with torch.no_grad():
        x = torch.from_numpy(enc).unsqueeze(0).to(device)
        rmt = torch.from_numpy(rm).unsqueeze(0).to(device)
        lmt = torch.from_numpy(rank_legal.astype(np.float32)).unsqueeze(0).to(device)
        logits = student.forward_policy_only(x, rmt, lmt)
        rank = int(logits.argmax(dim=1).item())
    a = rank_to_token_id(rank, legal, rank_tokens)
    return a if a in legal else legal[0]


def _v132_select(v132_model, device, state, legal):
    if len(legal) == 1:
        return legal[0]
    enc = encode_state_v17(state).astype(np.float32)
    mask = np.zeros(4, dtype=np.float32)
    for a in legal:
        mask[a] = 1.0
    with torch.no_grad():
        x = torch.from_numpy(enc).unsqueeze(0).to(device)
        m = torch.from_numpy(mask).unsqueeze(0).to(device)
        policy, _, _ = v132_model(x, m)
        a = int(policy.argmax(dim=1).item())
    return a if a in legal else legal[0]


def _play_one(state, model_player, picks):
    """Run a game to terminal with the two selectors keyed by player id."""
    csix = [0, 0, 0, 0]
    mc = 0
    while not state.is_terminal and mc < 400:
        cp = int(state.current_player)
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
        action = picks[cp](state, list(legal))
        state = ludo_cpp.apply_move(state, int(action))
        mc += 1
    if state.is_terminal:
        return int(ludo_cpp.get_winner(state))
    return -1  # truncated


def quick_h2h_vs_v132(student, v132_model, device, n_games=200, seed_base=12345):
    """In-memory V13.5 vs V13.2 H2H, mirrored seeds (V13.5 plays each seed
    once as P0 and once as P2). Returns (v135_wins, v132_wins, draws).
    """
    student.eval()
    v135_pick = lambda s, l: _v135_select(student, device, s, l)
    v132_pick = lambda s, l: _v132_select(v132_model, device, s, l)
    v135_w = v132_w = draws = 0
    for i in range(n_games):
        seed = seed_base + (i // 2)
        random.seed(seed)
        np.random.seed(seed)
        v135_player = 0 if (i % 2 == 0) else 2
        v132_player = 2 if v135_player == 0 else 0
        state = ludo_cpp.create_initial_state_2p()
        picks = {v135_player: v135_pick, v132_player: v132_pick}
        winner = _play_one(state, v135_player, picks)
        if winner == v135_player:
            v135_w += 1
        elif winner == v132_player:
            v132_w += 1
        else:
            draws += 1
    student.train()
    return v135_w, v132_w, draws


# ── Dashboard ──────────────────────────────────────────────────────────────
class _RLHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, directory=None, rl_path=None, chain_path=None, landing=None, **kw):
        self._rl_path = rl_path
        self._chain_path = chain_path
        self._landing = landing
        super().__init__(*args, directory=directory, **kw)

    def log_message(self, *a, **kw):  # silence per-request stdout
        return

    def do_GET(self):
        if self.path in ("/", ""):
            if self._landing:
                self.path = "/" + self._landing
            return super().do_GET()
        if self.path == "/api/rl_stats":
            return self._serve(self._rl_path)
        if self.path == "/api/sl_stats":
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
    for cand in ("rl_dashboard.html", "v13_dashboard.html", "sl_dashboard.html", "index.html"):
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
def train_on_chunk(student, optimizer, chunk, device, args, kl_teacher=None):
    """chunk: list of (enc, rmasks, rlegal, chosen_rank, v_pred_old, G)."""
    if not chunk:
        return None
    encs = np.stack([c[0] for c in chunk], axis=0)
    rmasks = np.stack([c[1] for c in chunk], axis=0)
    rlegals = np.stack([c[2] for c in chunk], axis=0)
    ranks = np.array([c[3] for c in chunk], dtype=np.int64)
    v_old = np.array([c[4] for c in chunk], dtype=np.float32)
    Gs = np.array([c[5] for c in chunk], dtype=np.float32)

    N = encs.shape[0]
    metrics = {"loss": 0.0, "loss_pol": 0.0, "loss_val": 0.0, "entropy": 0.0,
               "loss_kl": 0.0, "n_steps": 0}

    for epoch in range(args.train_epochs):
        order = np.random.permutation(N)
        for s in range(0, N, args.minibatch_size):
            idx = order[s:s + args.minibatch_size]
            x = torch.from_numpy(encs[idx]).to(device, dtype=torch.float32)
            rm = torch.from_numpy(rmasks[idx]).to(device, dtype=torch.float32)
            rl = torch.from_numpy(rlegals[idx]).to(device, dtype=torch.float32)
            r = torch.from_numpy(ranks[idx]).to(device)
            vb = torch.from_numpy(v_old[idx]).to(device)
            G = torch.from_numpy(Gs[idx]).to(device)

            policy, win_prob, _moves = student(x, rm, rl)
            v = 2.0 * win_prob - 1.0

            multi = (rl.sum(dim=1) > 1).float()
            multi_n = multi.sum().clamp(min=1.0)

            log_p = torch.log(policy.gather(1, r.unsqueeze(1)).squeeze(1) + 1e-8)
            advantage = (G - vb).detach()
            loss_pol = -(advantage * log_p * multi).sum() / multi_n

            loss_val = F.mse_loss(v, G)

            entropy_per = -(policy * torch.log(policy + 1e-8)).sum(dim=1)
            entropy = (entropy_per * multi).sum() / multi_n
            loss_ent = -args.entropy_coeff * entropy

            loss = loss_pol + args.value_coeff * loss_val + loss_ent

            loss_kl_val = 0.0
            if kl_teacher is not None and args.kl_anchor_coeff > 0:
                with torch.no_grad():
                    t_pol, _, _ = kl_teacher(x, rm, rl)
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


# ── Loaders ────────────────────────────────────────────────────────────────
def _load_v135(path, args, device):
    model = V135Symmetric(
        num_res_blocks=args.num_res_blocks,
        num_channels=args.num_channels,
        head_hidden=args.head_hidden,
        in_channels=V18_CHANNELS,
    )
    ckpt = torch.load(path, map_location=device, weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    if any(k.startswith("_orig_mod.") for k in sd):
        sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    return model, ckpt


def load_v135_kl_teacher(path, args, device):
    model, _ = _load_v135(path, args, device)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def load_v132_h2h_opponent(path, args, device):
    model = MinimalCNN14(
        num_res_blocks=args.v132_num_res_blocks,
        num_channels=args.v132_num_channels,
        in_channels=17,
    )
    ckpt = torch.load(path, map_location=device, weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    if any(k.startswith("_orig_mod.") for k in sd):
        sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
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

    if args.resume:
        args.init = os.path.join(CHECKPOINT_DIR, "model_latest.pt")
        if not os.path.exists(args.init):
            print(f"ERROR: --resume but {args.init} not found"); sys.exit(1)
    elif not args.init:
        print("ERROR: either --init or --resume is required"); sys.exit(1)

    if args.kl_teacher is None and args.kl_anchor_coeff > 0:
        # Default: anchor to the same checkpoint we initialized from (= V13.5_SL)
        args.kl_teacher = args.init

    print("=" * 70)
    print("V13.5 RL — self-play REINFORCE (rank-indexed)")
    print("=" * 70)
    print(f"  device:           {device}")
    print(f"  init:             {args.init}")
    print(f"  KL teacher:       {args.kl_teacher}")
    print(f"  H2H opponent:     {args.h2h_opponent or '(none)'}")
    print(f"  checkpoint dir:   {CHECKPOINT_DIR}")
    print(f"  parallel_games:   {args.parallel_games}")
    print(f"  train_chunk:      {args.train_chunk}")
    print(f"  minibatch:        {args.minibatch_size}  × {args.train_epochs} epochs")
    print(f"  target_states:    {args.target_states:,}")
    print(f"  lr:               {args.lr} → {args.lr_end} (cosine)")
    print(f"  entropy_coeff:    {args.entropy_coeff}")
    print(f"  kl_anchor_coeff:  {args.kl_anchor_coeff}")
    print(f"  arch:             V135Symmetric ({args.num_res_blocks}x{args.num_channels})")
    print("=" * 70)

    student, _ckpt = _load_v135(args.init, args, device)
    _resume_meta = None
    if isinstance(_ckpt, dict) and "model_state_dict" in _ckpt and args.resume:
        _resume_meta = _ckpt
    student.to(device).train()
    print(f"[Student] V135Symmetric params: {sum(p.numel() for p in student.parameters()):,}  "
          f"(loaded from {args.init})")

    kl_teacher = None
    if args.kl_teacher and args.kl_anchor_coeff > 0:
        kl_teacher = load_v135_kl_teacher(args.kl_teacher, args, device)
        print(f"[KL teacher] V13.5_SL loaded for KL anchor (coeff={args.kl_anchor_coeff})")

    v132_h2h = None
    if args.h2h_opponent and args.h2h_gate_every > 0:
        v132_h2h = load_v132_h2h_opponent(args.h2h_opponent, args, device)
        print(f"[H2H] V13.2 opponent loaded ({args.h2h_games} games every "
              f"{args.h2h_gate_every:,} states)")

    optimizer = torch.optim.Adam(student.parameters(), lr=args.lr)
    if _resume_meta and "optimizer_state_dict" in _resume_meta:
        try:
            optimizer.load_state_dict(_resume_meta["optimizer_state_dict"])
            print("[Resume] Optimizer state restored")
        except Exception as e:
            print(f"[Resume] optimizer state mismatch ({e}); starting fresh optimizer")

    def write_chain(phase):
        with open(chain_path, "w") as f:
            json.dump({"stage": "RL", "phase": phase, "arch": "v13_5",
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

    env = SelfPlayEnv(args.parallel_games, max_game_len=args.max_game_len)
    log(f"Starting RL: target {args.target_states:,} states, init={args.init}")

    pool = []
    pool_size = 0
    total = 0
    step = 0
    last_save = 0
    last_eval = 0
    last_h2h = 0
    last_save_games = 0
    last_eval_games = 0
    t_start = time.time()
    eval_history = []
    h2h_history = []
    last_metrics = None

    if _resume_meta:
        total = _resume_meta.get("total", 0)
        step = _resume_meta.get("step", 0)
        last_save = total
        last_eval = total
        last_h2h = total
        eval_history = _resume_meta.get("eval_history", [])
        h2h_history = _resume_meta.get("h2h_history", [])
        env.games_played = _resume_meta.get("games_played", 0)
        last_save_games = env.games_played
        last_eval_games = env.games_played
        log(f"[Resume] step={step}, states={total:,}, games={env.games_played}, "
            f"evals={len(eval_history)}, h2h={len(h2h_history)}")

    while total < args.target_states:
        decision_idxs, cps, encs, rmasks, rlegals, legal_lists, rank_token_ids_list = \
            env.spin_to_decision()

        with torch.no_grad():
            x = torch.from_numpy(encs).to(device, dtype=torch.float32)
            rm = torch.from_numpy(rmasks).to(device, dtype=torch.float32)
            rl_ = torch.from_numpy(rlegals).to(device, dtype=torch.float32)
            policy, win_prob, _ = student(x, rm, rl_)
            ranks = torch.multinomial(policy, num_samples=1).squeeze(1).cpu().numpy()
            v_preds = (2.0 * win_prob - 1.0).cpu().numpy()

        finished = env.apply_actions(
            decision_idxs, cps, encs, rmasks, rlegals,
            legal_lists, rank_token_ids_list, ranks, v_preds,
        )
        pool.extend(finished)
        pool_size += len(finished)

        if pool_size >= args.train_chunk:
            progress = total / max(1, args.target_states)
            cur_lr = args.lr_end + 0.5 * (args.lr - args.lr_end) * (1 + np.cos(np.pi * progress))
            for g in optimizer.param_groups:
                g["lr"] = cur_lr
            metrics = train_on_chunk(student, optimizer, pool, device, args, kl_teacher=kl_teacher)
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
                    + (f" kl {metrics['loss_kl']:.3f}" if kl_teacher is not None else "")
                    + ")")
                try:
                    with open(rl_stats_path, "w") as f:
                        json.dump({
                            "stage": "RL", "arch": "v13_5",
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
                            "h2h_history": h2h_history,
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
                    "h2h_history": h2h_history,
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

            if v132_h2h is not None and total - last_h2h >= args.h2h_gate_every:
                log(f"[h2h-gate] starting ({args.h2h_games} games vs V13.2)...")
                v135_w, v132_w, draws = quick_h2h_vs_v132(
                    student, v132_h2h, device, n_games=args.h2h_games)
                n = max(1, v135_w + v132_w + draws)
                v135_wr = 100 * v135_w / n
                v132_wr = 100 * v132_w / n
                h2h_history.append([total, v135_wr, v132_wr, draws])
                log(f"[h2h-gate] V13.5 {v135_w}/{n}={v135_wr:.1f}%  vs  "
                    f"V13.2 {v132_w}/{n}={v132_wr:.1f}%  (draws {draws})")
                last_h2h = total

    # Final save
    final_path = os.path.join(CHECKPOINT_DIR, "model_rl.pt")
    final_latest = os.path.join(CHECKPOINT_DIR, "model_latest.pt")
    save_dict = {
        "model_state_dict": student.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step, "total": total,
        "games_played": env.games_played,
        "eval_history": eval_history,
        "h2h_history": h2h_history,
    }
    torch.save(save_dict, final_path)
    torch.save(save_dict, final_latest)
    log(f"[done] processed {total:,} states across {env.games_played} games.")
    log(f"[done] saved → {final_path} + {final_latest}")
    write_chain("completed")
    rl_log.close()


if __name__ == "__main__":
    main()
