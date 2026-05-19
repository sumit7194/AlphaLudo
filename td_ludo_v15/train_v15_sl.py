"""V15 SL distillation trainer — V13.5 teacher → V15 GraphTransformer student.

Mirrors `td_ludo/train_v135_sl.py` structure (continuous self-play state gen
via legacy engine + minibatch training + dashboard + checkpoints) but with:
  - V15 encoder (per-cell triplet, 8-frame stack, Option B spread-fill)
  - V15 model (GraphTransformer, ~3M params matching teacher capacity)
  - V15 cell-based action space (225-way, masked)
  - Cross-arch projection: V13.5 rank policy → V15 source-cell target
  - State generation: SAMPLES FROM TEACHER POLICY (not random play) — this
    keeps the training-state distribution aligned with the teacher's natural
    play distribution. Random play was the dominant bug in the first V15 SL
    run (plateau at 52% bot-eval vs teacher's 84%). Fixed 2026-05-14.

Usage on VM:
    cd /home/sumit/td_ludo_v15
    TD_LUDO_RUN_NAME=v15_sl_v2 python3 train_v15_sl.py \
        --teacher /home/sumit/td_ludo/checkpoints/v135_prod_rl_local/model_latest.pt \
        --target-states 20000000 \
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
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ─── Bridge to legacy code for V13.5 teacher + V18 encoder ─────────────────
_LEGACY_ROOT = Path(__file__).resolve().parent.parent / "td_ludo"
if str(_LEGACY_ROOT) not in sys.path:
    sys.path.insert(0, str(_LEGACY_ROOT))

from td_ludo.game.encoder_v18_symmetric import encode_state_v18_symmetric  # type: ignore
from td_ludo.game.rank_mapping import (  # type: ignore
    state_to_rank_mapping,
    legal_mask_per_rank,
    rank_to_token_id,
)
from td_ludo.models.v13_5 import V135Symmetric, compute_rank_masks  # type: ignore

import td_ludo_cpp as _legacy_cpp  # legacy engine for state gen + teacher
import td_ludo_v15_cpp as _v15_cpp  # constants only

from td_ludo_v15.game.cells import (
    NUM_BOARD_CELLS,
    cell_to_index,
    position_to_cell_in_pov,
)
from td_ludo_v15.game.encoder import encode_frame
from td_ludo_v15.models.v15 import V15GraphTransformer


_BASE_POS = _v15_cpp.BASE_POS
_HOME_POS = _v15_cpp.HOME_POS
NUM_PLAYERS = 4
# History/stack depth — derived from args.history_len at runtime. Modules
# in this file reference these globals via the configure_history() helper
# below so the encoding and the model agree on T.
HISTORY_LEN = 7          # past frames (TOTAL_FRAMES - 1)
TOTAL_FRAMES = 8         # current + past

def configure_history(total_frames: int):
    """Switch the module-level stack depth (default V15=8, V15.1=2)."""
    global HISTORY_LEN, TOTAL_FRAMES
    if total_frames < 1:
        raise ValueError(f"history-len must be >= 1, got {total_frames}")
    TOTAL_FRAMES = total_frames
    HISTORY_LEN = total_frames - 1


# ─── Args ───────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher", required=True,
                   help="V13.5 checkpoint path (model_latest.pt or model_sl.pt)")
    p.add_argument("--target-states", type=int, default=20_000_000)
    p.add_argument("--batch-size", type=int, default=256,
                   help="Logical mini-batch size (= --parallel-games for now)")
    p.add_argument("--parallel-games", type=int, default=256,
                   help="Number of games run in parallel for state collection. "
                        "Bumped from 64→256 (2026-05-14) to match V13.5 SL throughput.")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lr-end", type=float, default=1e-4)
    p.add_argument("--save-every", type=int, default=1_000_000)
    p.add_argument("--eval-every", type=int, default=1_000_000)
    p.add_argument("--eval-games", type=int, default=200)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--value-coeff", type=float, default=0.5)
    # Teacher-driven state generation knobs
    p.add_argument("--teacher-temperature", type=float, default=1.0,
                   help="Temperature for sampling from teacher policy to advance "
                        "games (1.0 = use teacher distribution as-is).")
    p.add_argument("--exploration-epsilon", type=float, default=0.05,
                   help="With this probability, override teacher sample with a "
                        "random legal move (state-distribution coverage).")
    # Model arch (defaults match new ~3M GT)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--n-layers", type=int, default=8)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--ffn-dim", type=int, default=512)
    # History window — V15=8 frames stack (1 current + 7 past), V15.1=2.
    p.add_argument("--history-len", type=int, default=8,
                   help="Total frames T in the stack (1 current + (T-1) past). "
                        "V15=8, V15.1=2. The student model's in_features is "
                        "derived as T*3.")
    p.add_argument("--baseline-teacher-eval", action="store_true",
                   help="Run an initial eval of the V13.5 teacher through the same "
                        "harness for fair comparison, before starting student SL.")
    p.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--port", type=int, default=8798)
    p.add_argument("--no-dashboard", action="store_true")
    p.add_argument("--run-name", default=None)
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def pick_device(name: str) -> torch.device:
    if name in ("cuda", "cpu", "mps"):
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ─── V13.5 teacher loader ──────────────────────────────────────────────────
def _probe_v135_arch(state_dict: dict) -> dict:
    conv_w = state_dict.get("conv_input.weight")
    if conv_w is None:
        raise RuntimeError("checkpoint missing conv_input.weight")
    num_channels = int(conv_w.shape[0])
    indices = set()
    for k in state_dict.keys():
        if k.startswith("res_blocks."):
            parts = k.split(".")
            if len(parts) >= 2 and parts[1].isdigit():
                indices.add(int(parts[1]))
    num_res_blocks = max(indices) + 1 if indices else 0
    return {"num_res_blocks": num_res_blocks, "num_channels": num_channels}


def load_v135_teacher(path: Path, device: torch.device) -> V135Symmetric:
    print(f"[teacher] loading V13.5 from {path}...", flush=True)
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    sd = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    if any(k.startswith("_orig_mod.") for k in sd):
        sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    # RL checkpoints wrap V135Symmetric inside an actor-critic class with an
    # "inner." prefix on all model keys. Strip it for compatibility.
    if any(k.startswith("inner.") for k in sd):
        sd = {k[len("inner."):]: v for k, v in sd.items() if k.startswith("inner.")}
    arch = _probe_v135_arch(sd)
    print(f"[teacher] arch: {arch}", flush=True)
    model = V135Symmetric(**arch)
    model.load_state_dict(sd, strict=False)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    n = sum(p.numel() for p in model.parameters())
    print(f"[teacher] V13.5 params: {n:,}", flush=True)
    return model


# ─── Cross-arch projection: V13.5 rank policy → V15 cell target ────────────
def project_v135_to_v15_cell_policy(
    state, legal_token_ids, v135_rank_policy: np.ndarray, rank_token_ids
) -> np.ndarray:
    """V13.5 (4,) rank policy → V15 (225,) cell target. Exact, not lossy."""
    cp = int(state.current_player)
    per_token = np.zeros(4, dtype=np.float32)
    for rank, tids in enumerate(rank_token_ids):
        n = len(tids)
        if n == 0:
            continue
        share = float(v135_rank_policy[rank]) / n
        for t in tids:
            per_token[t] += share
    target = np.zeros(NUM_BOARD_CELLS, dtype=np.float32)
    for t in legal_token_ids:
        pos = int(state.player_positions[cp][t])
        cell = position_to_cell_in_pov(_BASE_POS if pos == _BASE_POS else pos, cp, cp)
        target[cell_to_index(*cell)] += per_token[t]
    s = target.sum()
    if s > 1e-6:
        target /= s
    return target


# ─── Self-play env (B parallel games, legacy engine) ───────────────────────
class V15DistillEnv:
    """B parallel games on the LEGACY engine (parity-tested vs V15 engine).

    Each `get_batch` call returns B samples (one decision-state per game),
    each with everything needed for both teacher and student forward passes.

    Games are ADVANCED EXTERNALLY: the main loop runs the teacher on each
    decision state, samples an action from its policy, and calls
    `advance(chosen_token_ids)` to step every game forward. This makes the
    training-state distribution match the teacher's natural play distribution
    — critical for cross-arch distillation. (Random-play state generation
    was the dominant bug in the first V15 SL run.)
    """

    def __init__(self, batch_size: int, max_game_len: int = 400, seed: int = 42):
        self.batch_size = batch_size
        self.max_game_len = max_game_len
        self.games = [_legacy_cpp.create_initial_state_2p() for _ in range(batch_size)]
        self.consec_sixes = np.zeros((batch_size, NUM_PLAYERS), dtype=np.int32)
        self.step_count = np.zeros(batch_size, dtype=np.int32)
        # Per-game history: deque of past pre-decision GameStates.
        self.history = [collections.deque(maxlen=HISTORY_LEN) for _ in range(batch_size)]
        self.rng = random.Random(seed)
        self.games_played = 0
        # Captures the last decision states (used to advance after teacher
        # chooses actions outside this class).
        self._last_legal_per_game: list = [None] * batch_size

    def _reset(self, i: int):
        self.games[i] = _legacy_cpp.create_initial_state_2p()
        self.consec_sixes[i] = 0
        self.step_count[i] = 0
        self.history[i].clear()

    def get_batch(self):
        """Advance each game to its next decision-state. Returns dict of
        batched arrays. The caller is responsible for choosing actions and
        calling `advance(chosen_token_ids)` to step the games."""
        states_out = []
        v15_xs, v15_masks = [], []
        teacher_v18s, teacher_rmasks_list, teacher_lmasks = [], [], []
        legal_token_ids_list, rank_token_ids_list = [], []

        for i in range(self.batch_size):
            while True:
                state = self.games[i]
                if state.is_terminal or self.step_count[i] >= self.max_game_len:
                    self.games_played += 1
                    self._reset(i)
                    state = self.games[i]
                cp = int(state.current_player)
                # Roll dice
                d = self.rng.randint(1, 6)
                if d == 6:
                    self.consec_sixes[i, cp] += 1
                    if self.consec_sixes[i, cp] >= 3:
                        self.consec_sixes[i, cp] = 0
                        nxt = (cp + 1) % NUM_PLAYERS
                        while not state.active_players[nxt]:
                            nxt = (nxt + 1) % NUM_PLAYERS
                        state.current_player = nxt
                        state.current_dice_roll = 0
                        continue
                else:
                    self.consec_sixes[i, cp] = 0
                state.current_dice_roll = d
                legal_token_ids = _legacy_cpp.get_legal_moves(state)
                if not legal_token_ids:
                    nxt = (cp + 1) % NUM_PLAYERS
                    while not state.active_players[nxt]:
                        nxt = (nxt + 1) % NUM_PLAYERS
                    state.current_player = nxt
                    state.current_dice_roll = 0
                    continue

                # ── decision state — produce all the encodings ─────────────
                cp = int(state.current_player)
                # V13.5 teacher inputs
                v18_x = encode_state_v18_symmetric(state).astype(np.float32)
                rmasks = compute_rank_masks(state).astype(np.float32)
                pp = state.player_positions[cp]
                _, rank_token_ids = state_to_rank_mapping(pp)
                rank_legal = legal_mask_per_rank(legal_token_ids, rank_token_ids).astype(np.float32)
                # V15 student inputs
                v15_legal = np.zeros(NUM_BOARD_CELLS, dtype=np.float32)
                for t in legal_token_ids:
                    pos = int(state.player_positions[cp][t])
                    cell = position_to_cell_in_pov(_BASE_POS if pos == _BASE_POS else pos, cp, cp)
                    v15_legal[cell_to_index(*cell)] = 1.0
                # V15 history: pre-decision frames in pov of this current cp
                past_states = list(self.history[i])
                pad = HISTORY_LEN - len(past_states)
                v15_x = np.zeros((TOTAL_FRAMES, 15, 15, 3), dtype=np.float32)
                # Fill the non-padded slots with encoded frames
                real_frames = [None] * pad + past_states + [state]
                for t_idx, st in enumerate(real_frames):
                    if st is None:
                        continue
                    v15_x[t_idx] = encode_frame(st, pov_player=cp)

                # Collect
                states_out.append(state)
                v15_xs.append(v15_x)
                v15_masks.append(v15_legal)
                teacher_v18s.append(v18_x)
                teacher_rmasks_list.append(rmasks)
                teacher_lmasks.append(rank_legal)
                legal_token_ids_list.append(legal_token_ids)
                rank_token_ids_list.append(rank_token_ids)
                self._last_legal_per_game[i] = legal_token_ids
                break

        return {
            "states": states_out,
            "v15_x": np.stack(v15_xs, axis=0),
            "v15_mask": np.stack(v15_masks, axis=0),
            "teacher_v18": np.stack(teacher_v18s, axis=0),
            "teacher_rmasks": np.stack(teacher_rmasks_list, axis=0),
            "teacher_lmask": np.stack(teacher_lmasks, axis=0),
            "legal_token_ids": legal_token_ids_list,
            "rank_token_ids": rank_token_ids_list,
        }

    def advance(self, chosen_token_ids):
        """Advance every game by applying `chosen_token_ids[i]` to game i.

        Pushes the pre-decision state into history before applying.
        """
        if len(chosen_token_ids) != self.batch_size:
            raise ValueError(
                f"expected {self.batch_size} actions, got {len(chosen_token_ids)}"
            )
        for i, action in enumerate(chosen_token_ids):
            self.history[i].append(self.games[i])
            self.games[i] = _legacy_cpp.apply_move(self.games[i], int(action))
            self.step_count[i] += 1


# ─── Dashboard ──────────────────────────────────────────────────────────────
class _SLHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, directory=None, sl_path=None, landing=None, **kw):
        self._sl_path = sl_path
        self._landing = landing
        super().__init__(*args, directory=directory, **kw)

    def log_message(self, *a, **kw):
        return

    def do_GET(self):
        if self.path in ("/", ""):
            if self._landing:
                self.path = "/" + self._landing
            return super().do_GET()
        if self.path in ("/api/sl_stats", "/api/stats"):
            try:
                with open(self._sl_path) as f:
                    data = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(data.encode())
            except FileNotFoundError:
                self.send_response(404)
                self.end_headers()
            return
        if self.path == "/api/chain":
            # Synthesize chain status from current sl_stats so the dashboard's
            # Phase + Run cells populate. V15 has no chain pipeline (SL-only
            # right now), so phase is always "training" until done.
            try:
                with open(self._sl_path) as f:
                    sl = json.load(f)
                states = int(sl.get("states", 0))
                target = int(sl.get("target", 1))
                phase = "completed" if states >= target else "training"
                chain = {
                    "stage": sl.get("arch", "SL").upper().replace("V15", "SL"),
                    "phase": phase,
                    "run_name": sl.get("run_name", "v15_sl"),
                    "ts": sl.get("ts", int(time.time())),
                }
                payload = json.dumps(chain).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(payload)
            except (FileNotFoundError, json.JSONDecodeError):
                self.send_response(404)
                self.end_headers()
            return
        super().do_GET()


def start_dashboard(port: int, sl_path: str, dashboard_dir: str):
    landing = None
    for cand in ("sl_dashboard.html", "v13_dashboard.html", "v12_dashboard.html", "index.html"):
        if os.path.exists(os.path.join(dashboard_dir, cand)):
            landing = cand
            break
    handler = functools.partial(_SLHandler, directory=dashboard_dir,
                                sl_path=sl_path, landing=landing)
    server = HTTPServer(("0.0.0.0", port), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"[dashboard] http://localhost:{port}/{landing or ''}", flush=True)


# ─── V13.5 baseline eval through the SAME harness ─────────────────────────
def eval_v135_teacher_baseline(teacher, device, n_games: int = 200) -> float:
    """Plays V13.5 teacher (greedy, rank-indexed) vs the same bot mix
    `quick_eval` uses. Returns win-rate %.

    Purpose: give the V15 student a fair-comparison ceiling. The original
    V13.5 evals (e.g. 84% bot-eval) used a different harness; running the
    teacher through this code lets us compute the real student-vs-teacher
    gap on identical conditions.
    """
    from src.heuristic_bot import get_bot, BOT_REGISTRY  # legacy
    from src.config import MAX_MOVES_PER_GAME  # legacy
    bot_types = list(BOT_REGISTRY.keys())
    teacher.eval()
    wins = 0
    for _ in range(n_games):
        model_player = random.choice([0, 2])
        opp_player = 2 if model_player == 0 else 0
        bot = get_bot(random.choice(bot_types), player_id=opp_player)
        state = _legacy_cpp.create_initial_state_2p()
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
                d = random.randint(1, 6)
                if d == 6:
                    csix[cp] += 1
                    if csix[cp] >= 3:
                        csix[cp] = 0
                        n = (cp + 1) % 4
                        while not state.active_players[n]:
                            n = (n + 1) % 4
                        state.current_player = n
                        state.current_dice_roll = 0
                        continue
                else:
                    csix[cp] = 0
                state.current_dice_roll = d
            legal = _legacy_cpp.get_legal_moves(state)
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
                    rank_legal = legal_mask_per_rank(legal, rank_tokens).astype(np.float32)
                    enc = encode_state_v18_symmetric(state).astype(np.float32)
                    rm = compute_rank_masks(state).astype(np.float32)
                    with torch.no_grad():
                        x = torch.from_numpy(enc).unsqueeze(0).to(device, dtype=torch.float32)
                        rmt = torch.from_numpy(rm).unsqueeze(0).to(device, dtype=torch.float32)
                        lmt = torch.from_numpy(rank_legal).unsqueeze(0).to(device, dtype=torch.float32)
                        t_policy, _, _, _ = teacher(x, rmt, lmt)
                        rank = int(t_policy.argmax(dim=1).item())
                    action = rank_to_token_id(rank, legal, rank_tokens)
                    if action not in legal:
                        action = random.choice(legal)
            else:
                action = bot.select_move(state, list(legal))
            state = _legacy_cpp.apply_move(state, int(action))
            mc += 1
        if state.is_terminal and _legacy_cpp.get_winner(state) == model_player:
            wins += 1
    return 100.0 * wins / n_games


# ─── Quick eval (bot mix; reuses legacy heuristic_bot) ─────────────────────
def quick_eval(student, device, n_games: int = 200) -> float:
    """Plays V15 vs random heuristic-bot mix. Returns win-rate %."""
    from src.heuristic_bot import get_bot, BOT_REGISTRY  # legacy
    from src.config import MAX_MOVES_PER_GAME  # legacy
    bot_types = list(BOT_REGISTRY.keys())
    student.eval()
    wins = 0
    for _ in range(n_games):
        model_player = random.choice([0, 2])
        opp_player = 2 if model_player == 0 else 0
        bot = get_bot(random.choice(bot_types), player_id=opp_player)
        state = _legacy_cpp.create_initial_state_2p()
        history: collections.deque = collections.deque(maxlen=HISTORY_LEN)
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
                d = random.randint(1, 6)
                if d == 6:
                    csix[cp] += 1
                    if csix[cp] >= 3:
                        csix[cp] = 0
                        n = (cp + 1) % 4
                        while not state.active_players[n]:
                            n = (n + 1) % 4
                        state.current_player = n
                        state.current_dice_roll = 0
                        continue
                else:
                    csix[cp] = 0
                state.current_dice_roll = d
            legal = _legacy_cpp.get_legal_moves(state)
            if not legal:
                n = (cp + 1) % 4
                while not state.active_players[n]:
                    n = (n + 1) % 4
                state.current_player = n
                state.current_dice_roll = 0
                continue
            if cp == model_player:
                # Push current state into history before move
                # Build legal cell list + V15 input
                legal_cells = set()
                for t in legal:
                    pos = int(state.player_positions[cp][t])
                    cell = position_to_cell_in_pov(_BASE_POS if pos == _BASE_POS else pos, cp, cp)
                    legal_cells.add(cell)
                if len(legal_cells) == 1:
                    only_cell = next(iter(legal_cells))
                    # Pick any legal token at that cell
                    for t in legal:
                        pos = int(state.player_positions[cp][t])
                        c = position_to_cell_in_pov(_BASE_POS if pos == _BASE_POS else pos, cp, cp)
                        if c == only_cell:
                            action = t
                            break
                else:
                    # Build V15 input + run model
                    past = list(history)
                    pad = HISTORY_LEN - len(past)
                    v15_x = np.zeros((TOTAL_FRAMES, 15, 15, 3), dtype=np.float32)
                    real_frames = [None] * pad + past + [state]
                    for t_idx, st in enumerate(real_frames):
                        if st is None:
                            continue
                        v15_x[t_idx] = encode_frame(st, pov_player=cp)
                    v15_legal = np.zeros(NUM_BOARD_CELLS, dtype=np.float32)
                    for (r, c) in legal_cells:
                        v15_legal[cell_to_index(r, c)] = 1.0
                    with torch.no_grad():
                        xt = torch.from_numpy(v15_x).unsqueeze(0).to(device)
                        mt = torch.from_numpy(v15_legal).unsqueeze(0).to(device)
                        policy, _ = student(xt, mt)
                        chosen_idx = int(policy.argmax(dim=-1).item())
                    chosen_r, chosen_c = divmod(chosen_idx, 15)
                    # Map cell back to a legal token-id (lowest-token-id at that cell)
                    action = None
                    for t in sorted(legal):
                        pos = int(state.player_positions[cp][t])
                        c = position_to_cell_in_pov(_BASE_POS if pos == _BASE_POS else pos, cp, cp)
                        if c == (chosen_r, chosen_c):
                            action = t
                            break
                    if action is None:
                        action = legal[0]  # safety fallback
            else:
                action = bot.select_move(state, list(legal))
            history.append(state)
            state = _legacy_cpp.apply_move(state, int(action))
            mc += 1
        if state.is_terminal and _legacy_cpp.get_winner(state) == model_player:
            wins += 1
    student.train()
    return 100.0 * wins / n_games


# ─── Main training loop ────────────────────────────────────────────────────
def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    # Wire history depth before constructing env/model so both agree on T.
    configure_history(args.history_len)

    if args.run_name:
        os.environ["TD_LUDO_RUN_NAME"] = args.run_name
    # Use a local checkpoint dir under td_ludo_v15 (don't write into legacy area)
    run_name = os.environ.get("TD_LUDO_RUN_NAME", "v15_sl")
    CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints" / run_name
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    sl_stats_path = str(CHECKPOINT_DIR / "sl_stats.json")
    sl_log_path = str(CHECKPOINT_DIR / "sl.log")

    print("=" * 70)
    print(f"V15 SL DISTILLATION  (run={run_name})")
    print("=" * 70)
    print(f"  device:           {device}")
    print(f"  teacher:          {args.teacher}")
    print(f"  checkpoint dir:   {CHECKPOINT_DIR}")
    print(f"  target_states:    {args.target_states:,}")
    print(f"  parallel_games:   {args.parallel_games}")
    print(f"  batch_size:       {args.batch_size}")
    print(f"  lr:               {args.lr} → {args.lr_end} (cosine)")
    print("=" * 70)

    # Build models
    teacher = load_v135_teacher(Path(args.teacher), device)
    student = V15GraphTransformer(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        ffn_dim=args.ffn_dim,
        history_len=args.history_len,
    ).to(device)
    print(f"[student] V15 params: {student.count_parameters():,}", flush=True)
    print(f"[student] arch: d_model={args.d_model} n_heads={args.n_heads} "
          f"n_layers={args.n_layers} ffn_dim={args.ffn_dim} "
          f"history_len={args.history_len}", flush=True)
    optimizer = torch.optim.Adam(student.parameters(), lr=args.lr)

    # Optional resume
    total = 0
    step = 0
    eval_history = []
    if args.resume:
        latest = CHECKPOINT_DIR / "model_latest.pt"
        if latest.exists():
            print(f"[resume] loading {latest}...", flush=True)
            ck = torch.load(str(latest), map_location=device, weights_only=False)
            student.load_state_dict(ck["model_state_dict"])
            optimizer.load_state_dict(ck["optimizer_state_dict"])
            total = ck.get("total", 0)
            step = ck.get("step", 0)
            eval_history = ck.get("eval_history", [])
            print(f"[resume] from step={step} total={total:,} evals={len(eval_history)}", flush=True)

    # Dashboard
    if not args.no_dashboard:
        dash_dir = str(_LEGACY_ROOT)  # has sl_dashboard.html
        start_dashboard(args.port, sl_stats_path, dash_dir)

    # Optional: baseline teacher eval through the SAME harness so we have
    # a fair comparison ceiling (the V15 "32pp gap vs teacher" diagnosis
    # was apples-to-oranges without this).
    if args.baseline_teacher_eval:
        from src.heuristic_bot import get_bot, BOT_REGISTRY  # legacy  # noqa
        print(f"[baseline] running V13.5 teacher through quick_eval "
              f"({args.eval_games} games)...", flush=True)
        teacher_wr = eval_v135_teacher_baseline(teacher, device, n_games=args.eval_games)
        print(f"[baseline] V13.5 teacher WR through V15's eval harness = "
              f"{teacher_wr:.1f}%", flush=True)
        # Stash in checkpoint dir so the dashboard / journal can pick it up
        with open(CHECKPOINT_DIR / "baseline_teacher_wr.json", "w") as f:
            json.dump({"teacher_wr": teacher_wr, "n_games": args.eval_games,
                       "ts": int(time.time())}, f)

    # Env
    env = V15DistillEnv(args.parallel_games, seed=args.seed)
    sl_log = open(sl_log_path, "a")
    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        sl_log.write(line + "\n")
        sl_log.flush()

    log(f"starting SL: target {args.target_states:,} states")
    t_start = time.time()
    last_save = total
    last_eval = total

    while total < args.target_states:
        batch = env.get_batch()
        # Run teacher
        with torch.no_grad():
            tv18 = torch.from_numpy(batch["teacher_v18"]).to(device)
            trm = torch.from_numpy(batch["teacher_rmasks"]).to(device)
            tlm = torch.from_numpy(batch["teacher_lmask"]).to(device)
            t_policy, t_win, _, _ = teacher(tv18, trm, tlm)  # (B, 4), (B,)
            t_policy_np = t_policy.cpu().numpy()
            t_win_np = t_win.cpu().numpy()

        # Project to V15 cell-space targets (exact, lossless)
        targets = np.stack([
            project_v135_to_v15_cell_policy(
                batch["states"][i],
                batch["legal_token_ids"][i],
                t_policy_np[i],
                batch["rank_token_ids"][i],
            )
            for i in range(args.parallel_games)
        ], axis=0)

        # ── Teacher-driven state generation ────────────────────────────
        # Sample an action from the teacher's policy for each game, then
        # call env.advance to step every game forward. This keeps the
        # training-state distribution aligned with the teacher's natural
        # play distribution. ε-exploration mixes in random legal moves.
        chosen_token_ids = []
        rank_legal = batch["teacher_lmask"]  # (B, 4)
        for i in range(args.parallel_games):
            legal = batch["legal_token_ids"][i]
            if random.random() < args.exploration_epsilon:
                tok = random.choice(legal)
            else:
                # Sample rank from teacher policy (mask + temperature applied)
                rp = t_policy_np[i].astype(np.float64)
                rp = rp * rank_legal[i].astype(np.float64)
                s = rp.sum()
                if s < 1e-9:
                    tok = random.choice(legal)
                else:
                    rp = rp / s
                    if args.teacher_temperature != 1.0:
                        rp = np.power(rp, 1.0 / max(1e-3, args.teacher_temperature))
                        rp = rp / rp.sum()
                    rank = int(np.random.choice(4, p=rp))
                    rank_tids = batch["rank_token_ids"][i]
                    try:
                        tok = rank_to_token_id(rank, legal, rank_tids)
                    except Exception:
                        tok = random.choice(legal)
                    if tok not in legal:
                        tok = random.choice(legal)
            chosen_token_ids.append(int(tok))
        env.advance(chosen_token_ids)

        # Forward student
        sv = torch.from_numpy(batch["v15_x"]).to(device)
        sm = torch.from_numpy(batch["v15_mask"]).to(device)
        st_target = torch.from_numpy(targets).to(device)
        sv_win_target = torch.from_numpy(t_win_np).to(device)

        s_policy, s_win = student(sv, sm)
        log_p = torch.log(s_policy + 1e-9)
        loss_pol = -(st_target * log_p).sum(dim=-1).mean()
        loss_val = F.mse_loss(s_win, sv_win_target)
        loss = loss_pol + args.value_coeff * loss_val

        # Cosine LR
        progress = total / max(1, args.target_states)
        cur_lr = args.lr_end + 0.5 * (args.lr - args.lr_end) * (1 + np.cos(np.pi * progress))
        for g in optimizer.param_groups:
            g["lr"] = cur_lr

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
        optimizer.step()
        step += 1
        total += args.parallel_games  # one decision per parallel game

        if step % max(1, args.log_every) == 0:
            elapsed = time.time() - t_start
            fps = total / max(1e-6, elapsed)
            log(f"step {step:>5} | states {total:>9,}/{args.target_states:,} "
                f"| fps {fps:>5.0f} | lr {cur_lr:.1e} "
                f"| L {loss.item():.4f} (pol {loss_pol.item():.4f} val {loss_val.item():.4f})")
            try:
                with open(sl_stats_path, "w") as f:
                    json.dump({
                        "stage": "SL", "arch": "v15", "run_name": run_name,
                        "step": step, "states": total, "target": args.target_states,
                        "fps": fps, "elapsed_sec": elapsed, "lr": cur_lr,
                        "loss": loss.item(), "loss_policy": loss_pol.item(),
                        "loss_value": loss_val.item(),
                        "eval_history": eval_history,
                        "games_played": env.games_played,
                        "ts": int(time.time()),
                    }, f)
            except Exception as e:
                print(f"[stats] write failed: {e}", flush=True)

        if total - last_save >= args.save_every:
            ck_path = CHECKPOINT_DIR / f"sl_{total // 1_000_000}M.pt"
            save_dict = {
                "model_state_dict": student.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "step": step, "total": total, "eval_history": eval_history,
            }
            torch.save(save_dict, str(ck_path))
            torch.save(save_dict, str(CHECKPOINT_DIR / "model_latest.pt"))
            log(f"[checkpoint] {ck_path}")
            last_save = total

        if total - last_eval >= args.eval_every:
            log(f"[eval] starting ({args.eval_games} games)...")
            wr = quick_eval(student, device, n_games=args.eval_games)
            eval_history.append([total, wr])
            log(f"[eval] WR = {wr:.1f}% at {total:,} states")
            last_eval = total

    # Final save
    final_path = CHECKPOINT_DIR / "model_sl.pt"
    save_dict = {
        "model_state_dict": student.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step, "total": total, "eval_history": eval_history,
    }
    torch.save(save_dict, str(final_path))
    torch.save(save_dict, str(CHECKPOINT_DIR / "model_latest.pt"))
    log(f"[done] saved {final_path}")
    sl_log.close()


if __name__ == "__main__":
    main()
