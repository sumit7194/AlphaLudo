"""Batched, rule-correct evaluation for 2-player V15-architecture students.

Written in the 2026-10-02 audit to replace train_v153_sl.evaluate_model, which
  * played 15 games per bot (one eval point had a ±12 pp standard error, yet
    `model_best.pt` was chosen by its max — the "0.883" checkpoint scored
    38.2% ±1.5 head-to-head against the v15.2 champion),
  * omitted the three-consecutive-sixes forfeit that every training engine
    applies,
  * ran one batch-1 forward pass per move.

Here: K games run concurrently and every model decision in a round goes
through ONE forward pass, padded to a fixed batch size (variable shapes leak
MPS graph-cache memory). Dice come from a per-game RNG seeded by game index,
so successive evals replay the same dice (common random numbers: checkpoint
comparisons are much less noisy than independent samples of the same size).
The student takes seat 0 in even games and seat 2 in odd games.
"""
from __future__ import annotations

import math
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
for p in (str(ROOT_DIR / "td_ludo"), str(ROOT_DIR / "python"), str(ROOT_DIR / "td_ludo_v15")):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import torch

import alphaludo_rs
import td_ludo_cpp as ludo_cpp
import td_ludo_v15_cpp as v15_cpp
from td_ludo_v15.game.cells import NUM_BOARD_CELLS, cell_to_index, position_to_cell_in_pov
from td_ludo_v15.game.encoder import encode_frame

_BASE_POS = v15_cpp.BASE_POS
MAX_MOVES = 600
CHAMPION_PATH = ROOT_DIR / "td_ludo" / "play" / "model_weights" / "v15_2" / "model_best.pt"


# ── Players ──────────────────────────────────────────────────────────────────
class ModelPlayer:
    """Greedy (argmax) V15-architecture policy, batched."""

    batched = True

    def __init__(self, model: torch.nn.Module, device: torch.device, pad_to: int = 64, name: str = "model"):
        self.model = model
        self.device = device
        self.pad_to = pad_to
        self.name = name

    def choose_batch(self, items) -> List[int]:
        """items: list of (state, legal_tokens, consecutive_sixes)."""
        out: List[int] = []
        for lo in range(0, len(items), self.pad_to):
            chunk = items[lo:lo + self.pad_to]
            x = np.zeros((self.pad_to, 1, 15, 15, 3), dtype=np.float32)
            mask = np.zeros((self.pad_to, NUM_BOARD_CELLS), dtype=np.float32)
            mask[:, 0] = 1.0  # padding rows: one legal cell keeps softmax finite
            cell_maps = []
            for i, (state, legal, _) in enumerate(chunk):
                cp = int(state.current_player)
                x[i, 0] = encode_frame(state, pov_player=cp)
                mask[i, 0] = 0.0
                cmap = {}
                for t in legal:
                    pos = int(state.player_positions[cp][t])
                    idx = cell_to_index(*position_to_cell_in_pov(pos, cp, cp))
                    mask[i, idx] = 1.0
                    cmap.setdefault(idx, t)
                cell_maps.append(cmap)
            was_training = self.model.training
            self.model.train(False)
            with torch.no_grad():
                pol, _ = self.model(torch.from_numpy(x).to(self.device), torch.from_numpy(mask).to(self.device))
                choice = pol.argmax(dim=-1).cpu().numpy()
            self.model.train(was_training)
            for i, cmap in enumerate(cell_maps):
                out.append(cmap.get(int(choice[i]), chunk[i][1][0]))
        return out


class RustBotPlayer:
    """Native bot via alphaludo_rs.select_bot_move_2p ("depth2", "expectimax", "mcts")."""

    batched = False

    def __init__(self, bot: str, num_simulations: int = 3000):
        self.bot = bot
        self.name = bot
        self.num_simulations = num_simulations

    def choose(self, state, legal, csix: int) -> int:
        pos = np.ascontiguousarray(state.player_positions, dtype=np.int8)
        a = alphaludo_rs.select_bot_move_2p(
            self.bot, pos, int(state.current_player), int(state.current_dice_roll), int(csix), self.num_simulations
        )
        return a if a in legal else legal[0]


class ScriptedPlayer:
    """Python scripted bot (Expert, Heuristic, ...) from the champion's eval registry."""

    batched = False

    def __init__(self, bot_name: str):
        from td_ludo_v15.rich.v15_bot_eval import get_bot
        self._get_bot = get_bot
        self.name = bot_name
        self._bots: Dict[int, object] = {}

    def choose(self, state, legal, csix: int) -> int:
        seat = int(state.current_player)
        bot = self._bots.get(seat)
        if bot is None:
            bot = self._bots[seat] = self._get_bot(self.name, player_id=seat)
        return int(bot.select_move(state, list(legal)))


# ── Match runner ─────────────────────────────────────────────────────────────
class _Game:
    __slots__ = ("idx", "state", "csix", "rng", "moves", "student_seat", "players")

    def __init__(self, idx: int, seed: int, student, opponent):
        self.idx = idx
        self.state = ludo_cpp.create_initial_state_2p()
        self.csix = [0, 0, 0, 0]
        self.rng = random.Random(seed * 1_000_003 + idx)
        self.moves = 0
        self.student_seat = 0 if idx % 2 == 0 else 2
        opp_seat = 2 - self.student_seat
        self.players = {self.student_seat: student, opp_seat: opponent}

    def _pass(self, cp: int):
        n = (cp + 1) % 4
        while not self.state.active_players[n]:
            n = (n + 1) % 4
        self.state.current_player = n
        self.state.current_dice_roll = 0

    def advance(self):
        """Roll dice and play forced moves until a real decision or the end.
        Returns (player, legal) when a choice is needed, else None (finished)."""
        while not self.state.is_terminal and self.moves < MAX_MOVES:
            cp = int(self.state.current_player)
            if self.state.current_dice_roll == 0:
                d = self.rng.randint(1, 6)
                if d == 6:
                    self.csix[cp] += 1
                    if self.csix[cp] >= 3:  # three-six forfeit (same rule as Rust/training)
                        self.csix[cp] = 0
                        self._pass(cp)
                        continue
                else:
                    self.csix[cp] = 0
                self.state.current_dice_roll = d
            legal = list(ludo_cpp.get_legal_moves(self.state))
            if not legal:
                self._pass(cp)
                continue
            if len(legal) == 1:
                self.play(legal[0])
                continue
            return self.players[cp], legal
        return None

    def play(self, action: int):
        self.state = ludo_cpp.apply_move(self.state, int(action))
        self.moves += 1

    def student_won(self) -> bool:
        return bool(self.state.is_terminal) and int(ludo_cpp.get_winner(self.state)) == self.student_seat


def run_match(student, opponent, n_games: int, seed: int = 0, concurrency: int = 64) -> dict:
    t0 = time.time()
    next_idx = 0
    active: List[_Game] = []
    wins = 0
    finished = 0
    seat_wins = {0: 0, 2: 0}
    seat_games = {0: 0, 2: 0}
    unfinished = 0

    while finished < n_games:
        while len(active) < concurrency and next_idx < n_games:
            active.append(_Game(next_idx, seed, student, opponent))
            next_idx += 1

        requests: Dict[int, list] = {}
        still: List[_Game] = []
        for g in active:
            r = g.advance()
            if r is None:
                finished += 1
                seat_games[g.student_seat] += 1
                if not g.state.is_terminal:
                    unfinished += 1
                if g.student_won():
                    wins += 1
                    seat_wins[g.student_seat] += 1
                continue
            player, legal = r
            requests.setdefault(id(player), [player, []])[1].append((g, legal))
            still.append(g)
        active = still

        for player, reqs in requests.values():
            if player.batched:
                acts = player.choose_batch([(g.state, legal, g.csix[int(g.state.current_player)]) for g, legal in reqs])
            else:
                acts = [player.choose(g.state, legal, g.csix[int(g.state.current_player)]) for g, legal in reqs]
            for (g, _), a in zip(reqs, acts):
                g.play(a)

    p = wins / max(1, n_games)
    return {
        "opponent": opponent.name,
        "win_rate": p,
        "stderr": math.sqrt(max(p * (1 - p), 1e-9) / max(1, n_games)),
        "wins": wins,
        "games": n_games,
        "seat0_wr": seat_wins[0] / max(1, seat_games[0]),
        "seat2_wr": seat_wins[2] / max(1, seat_games[2]),
        "unfinished": unfinished,
        "seconds": time.time() - t0,
    }


def load_v15_model(path: Path, device: torch.device) -> torch.nn.Module:
    from td_ludo_v15.models.v15 import V15GraphTransformer
    m = V15GraphTransformer(d_model=128, n_heads=4, n_layers=4, ffn_dim=256, history_len=1).to(device)
    ck = torch.load(path, map_location=device, weights_only=False)
    m.load_state_dict(ck.get("model_state_dict", ck))
    m.train(False)
    return m


def evaluate_suite(
    model: torch.nn.Module,
    device: torch.device,
    champion: Optional[torch.nn.Module],
    games_champion: int = 400,
    games_bots: int = 400,
    games_depth2: int = 200,
    seed: int = 12345,
) -> Dict[str, dict]:
    """Head-to-head vs the champion, the champion's own Expert/Heuristic eval
    protocol (comparable to its 0.8465), and native Depth2."""
    student = ModelPlayer(model, device, name="student")
    out: Dict[str, dict] = {}
    if champion is not None and games_champion > 0:
        out["Champion_v15_2"] = run_match(student, ModelPlayer(champion, device, name="v15_2"), games_champion, seed)
    if games_bots > 0:
        half = games_bots // 2
        out["Expert"] = run_match(student, ScriptedPlayer("Expert"), half, seed + 1)
        out["Heuristic"] = run_match(student, ScriptedPlayer("Heuristic"), games_bots - half, seed + 2)
    if games_depth2 > 0:
        out["Depth2"] = run_match(student, RustBotPlayer("depth2"), games_depth2, seed + 3)
    return out


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Evaluate a V15-architecture checkpoint.")
    ap.add_argument("checkpoint")
    ap.add_argument("--games-champion", type=int, default=1000)
    ap.add_argument("--games-bots", type=int, default=1000)
    ap.add_argument("--games-depth2", type=int, default=400)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=777)
    a = ap.parse_args()
    dev = torch.device(a.device)
    res = evaluate_suite(
        load_v15_model(Path(a.checkpoint), dev), dev, load_v15_model(CHAMPION_PATH, dev),
        a.games_champion, a.games_bots, a.games_depth2, a.seed,
    )
    for k, r in res.items():
        print(f"{k:16s} {r['win_rate']:.3f} ±{r['stderr']:.3f}  ({r['wins']}/{r['games']}, "
              f"seat0 {r['seat0_wr']:.3f} seat2 {r['seat2_wr']:.3f}, {r['seconds']:.0f}s)")
