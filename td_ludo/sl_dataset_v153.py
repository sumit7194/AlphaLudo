"""Dataset loader for Rust `generate_sl_teacher` shards (formats v1 and v2).

FORMAT v1 (V15.3 run, no header) — 20-byte records:
    positions (4,4) int8 | current_player | dice_roll | action | target_value (±1)
  No record of WHICH bot moved. The mover is recovered from game boundaries:
  every game's first record has all 16 positions in base (an all-base state
  cannot occur mid-game: a capture leaves the capturing token on the board),
  and the teacher (MCTS) seat is 0 for even global game numbers, else 2, with
  global game = shard_index * shard_size + game index within the shard.

FORMAT v2 — 16-byte header b"ALSL" | u16 version | u16 record_size | u32 n | u32 0,
  then 32-byte records adding mover_bot, legal_count, consecutive_sixes, flags,
  the search's visit distribution pi (u8 x4, by token) and game_id.

Fixes from the 2026-10-02 audit:
  * `state.scores` was never set, so the encoder's MS/OS cells ("my scored" /
    "opp scored") were always 0 in training while real games fill them.
    Scores are now derived from positions == HOME_POS.
  * The data mixed MCTS and Depth2 moves ~49/51 with no way to tell them
    apart. `movers=` filters ("all" | "teacher" | "opponent" | "mcts" | "depth2").
  * Forced moves (one legal option) carry no policy information and inflated
    top-1 accuracy. `skip_forced=True` drops them (they still teach value, so
    it is a choice, not hard-coded).
  * Validation must be split by GAME, not by record: pass explicit
    `shard_files` (e.g. hold out whole shards).

Each item:
  x            (1, 15, 15, 3) float32 frame in mover POV
  target       int64 cell index of the label action
  target_pi    (225,) float32 policy target over cells (one-hot of `target`
               unless the record carries a visit distribution and
               policy_target == "pi")
  legal_mask   (225,) float32
  target_value float32 in {+1, -1, 0}
  legal_count  int64 number of legal tokens
"""
from __future__ import annotations

from pathlib import Path
import sys
from typing import Iterable, List, Optional

_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(_ROOT), str(_ROOT / "td_ludo"), str(_ROOT / "td_ludo_v15")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import torch
from torch.utils.data import Dataset

import td_ludo_cpp as ludo_cpp
import td_ludo_v15_cpp as v15_cpp
from td_ludo_v15.game.cells import NUM_BOARD_CELLS, cell_to_index, position_to_cell_in_pov
from td_ludo_v15.game.encoder import encode_frame

_BASE_POS = v15_cpp.BASE_POS
_HOME_POS = 99

MAGIC = b"ALSL"
HEADER_BYTES = 16

SL_RECORD_DTYPE = np.dtype([
    ("positions", np.int8, (4, 4)),
    ("current_player", np.int8),
    ("dice_roll", np.int8),
    ("action", np.int8),
    ("target_value", np.int8),
])

SL_RECORD_V2_DTYPE = np.dtype([
    ("positions", np.int8, (4, 4)),
    ("current_player", np.int8),
    ("dice_roll", np.int8),
    ("action", np.int8),
    ("target_value", np.int8),
    ("mover_bot", np.uint8),
    ("legal_count", np.uint8),
    ("consecutive_sixes", np.uint8),
    ("flags", np.uint8),
    ("pi", np.uint8, (4,)),
    ("game_id", "<u4"),
])
assert SL_RECORD_V2_DTYPE.itemsize == 32

BOT_MCTS, BOT_DEPTH2, BOT_EXPECTIMAX = 0, 1, 2
FLAG_HAS_PI, FLAG_TEACHER_SEAT, FLAG_FORCED, FLAG_EXPLORATORY = 1, 2, 4, 8

MOVER_CHOICES = ("all", "teacher", "opponent", "mcts", "depth2")


def open_shard(path: Path):
    """Returns (version, memmap of records)."""
    with open(path, "rb") as f:
        head = f.read(HEADER_BYTES)
    size = path.stat().st_size
    if head[:4] == MAGIC:
        version = int.from_bytes(head[4:6], "little")
        rec_size = int.from_bytes(head[6:8], "little")
        n = int.from_bytes(head[8:12], "little")
        if version != 2 or rec_size != 32:
            raise ValueError(f"{path}: unsupported shard format v{version} (record {rec_size} B)")
        if HEADER_BYTES + n * 32 > size:
            raise ValueError(f"{path}: truncated ({size} B for {n} records)")
        return 2, np.memmap(path, dtype=SL_RECORD_V2_DTYPE, mode="r", offset=HEADER_BYTES, shape=(n,))
    n = size // SL_RECORD_DTYPE.itemsize
    return 1, np.memmap(path, dtype=SL_RECORD_DTYPE, mode="r", shape=(n,))


def legal_counts(positions: np.ndarray, cp: np.ndarray, dice: np.ndarray) -> np.ndarray:
    """Vectorised legal-move count for records (mirrors GameState::legal_moves)."""
    own = positions[np.arange(len(cp)), cp.astype(np.int64)].astype(np.int16)  # (N, 4)
    d = dice.astype(np.int16)[:, None]
    in_base = own == _BASE_POS
    on_board = (~in_base) & (own != _HOME_POS)
    legal = (in_base & (d == 6)) | (on_board & (own + d <= 56))
    return legal.sum(axis=1)


def v1_teacher_seat(recs: np.ndarray, shard_index: int, shard_size: int) -> np.ndarray:
    """Teacher (MCTS) seat per record of a v1 shard, from game boundaries."""
    starts = np.all(recs["positions"].reshape(len(recs), 16) == _BASE_POS, axis=1)
    game_local = np.cumsum(starts) - 1
    n_games = int(starts.sum())
    if n_games != shard_size and n_games > 0:
        # A short final shard is legitimate; anything else means the boundary
        # rule failed and the labels would be wrong.
        if n_games > shard_size:
            raise ValueError(f"v1 shard {shard_index}: found {n_games} games > shard size {shard_size}")
    global_game = shard_index * shard_size + game_local
    return np.where(global_game % 2 == 0, 0, 2).astype(np.int8)


def _shard_index(path: Path) -> int:
    return int(path.stem.split("_")[-1])


class V153TeacherDataset(Dataset):
    """Memory-mapped dataset over teacher shards with mover / forced filtering."""

    def __init__(
        self,
        shard_dir: str | Path | None = None,
        max_shards: Optional[int] = None,
        shard_files: Optional[Iterable[str | Path]] = None,
        movers: str = "all",
        skip_forced: bool = False,
        policy_target: str = "action",
        v1_shard_size: int = 5000,
    ):
        if movers not in MOVER_CHOICES:
            raise ValueError(f"movers must be one of {MOVER_CHOICES}")
        if policy_target not in ("action", "pi"):
            raise ValueError("policy_target must be 'action' or 'pi'")
        self.movers = movers
        self.skip_forced = skip_forced
        self.policy_target = policy_target

        if shard_files is not None:
            self.shard_files = [Path(p) for p in shard_files]
        else:
            self.shard_files = sorted(Path(shard_dir).glob("shard_*.bin"))
            if max_shards is not None:
                self.shard_files = self.shard_files[:max_shards]

        self.shards: List[np.ndarray] = []
        self.versions: List[int] = []
        idx_shard, idx_off = [], []
        self.n_raw = 0
        for sf in self.shard_files:
            try:
                version, recs = open_shard(sf)
            except ValueError as e:
                print(f"[dataset] skipping {sf.name}: {e}")
                continue
            if len(recs) == 0:
                continue
            self.n_raw += len(recs)
            keep = np.ones(len(recs), dtype=bool)
            cp = recs["current_player"]

            if version == 2:
                lc = recs["legal_count"].astype(np.int64)
                teacher_seat_mover = (recs["flags"] & FLAG_TEACHER_SEAT) != 0
                bot = recs["mover_bot"]
            else:
                lc = legal_counts(recs["positions"], cp, recs["dice_roll"])
                teacher_seat_mover = cp == v1_teacher_seat(recs, _shard_index(sf), v1_shard_size)
                # V15.3 v1 data: teacher = MCTS, opponent = Depth2.
                bot = np.where(teacher_seat_mover, BOT_MCTS, BOT_DEPTH2)

            if movers == "teacher":
                keep &= teacher_seat_mover
            elif movers == "opponent":
                keep &= ~teacher_seat_mover
            elif movers == "mcts":
                keep &= bot == BOT_MCTS
            elif movers == "depth2":
                keep &= bot == BOT_DEPTH2
            if skip_forced:
                keep &= lc >= 2

            sel = np.nonzero(keep)[0].astype(np.int32)
            idx_shard.append(np.full(len(sel), len(self.shards), dtype=np.int32))
            idx_off.append(sel)
            self.shards.append(recs)
            self.versions.append(version)

        # Unfiltered datasets (every record kept) map an index to (shard,
        # offset) with a binary search over cumulative sizes instead of two
        # per-record arrays: on macOS every DataLoader worker is a spawned
        # process that receives its own copy of the dataset, so per-record
        # arrays cost ~8 bytes x records x (workers + 1).
        self.unfiltered = all(len(o) == len(r) for o, r in zip(idx_off, self.shards))
        if self.unfiltered:
            self.cum = np.cumsum([len(r) for r in self.shards]).astype(np.int64)
            self.idx_shard = self.idx_off = None
            self.total_records = int(self.cum[-1]) if len(self.cum) else 0
        else:
            self.cum = None
            self.idx_shard = np.concatenate(idx_shard) if idx_shard else np.zeros(0, np.int32)
            self.idx_off = np.concatenate(idx_off) if idx_off else np.zeros(0, np.int32)
            self.total_records = len(self.idx_shard)

    def __len__(self) -> int:
        return self.total_records

    def locate(self, idx: int):
        """(shard index, record offset) of dataset item `idx`."""
        if self.unfiltered:
            si = int(np.searchsorted(self.cum, idx, side="right"))
            return si, int(idx - (self.cum[si - 1] if si else 0))
        return int(self.idx_shard[idx]), int(self.idx_off[idx])

    def shard_items(self, si: int):
        """(dataset indices, record offsets) of the items that live in shard si."""
        if self.unfiltered:
            base = int(self.cum[si - 1]) if si else 0
            n = len(self.shards[si])
            return np.arange(base, base + n, dtype=np.int64), np.arange(n, dtype=np.int64)
        sel = np.nonzero(self.idx_shard == si)[0]
        return sel, self.idx_off[sel]

    def __getitem__(self, idx: int):
        si, off = self.locate(int(idx))
        rec = self.shards[si][off]
        version = self.versions[si]

        positions = np.asarray(rec["positions"], dtype=np.int8)
        cp = int(rec["current_player"])
        dice = int(rec["dice_roll"])
        act = int(rec["action"])
        val = float(rec["target_value"])

        state = ludo_cpp.create_initial_state_2p()
        state.player_positions = positions.tolist()
        state.scores = [int((positions[p] == _HOME_POS).sum()) for p in range(4)]
        state.current_player = cp
        state.current_dice_roll = dice

        def cell_of(token: int) -> int:
            pos = int(positions[cp, token])
            return int(cell_to_index(*position_to_cell_in_pov(pos, cp, cp)))

        target_cell_idx = cell_of(act)

        legal_mask = np.zeros(NUM_BOARD_CELLS, dtype=np.float32)
        legal_tokens = list(ludo_cpp.get_legal_moves(state))
        for t in legal_tokens:
            legal_mask[cell_of(int(t))] = 1.0
        if legal_mask[target_cell_idx] == 0.0:
            raise ValueError(f"record {idx}: label token {act} is not legal")

        target_pi = np.zeros(NUM_BOARD_CELLS, dtype=np.float32)
        if (
            self.policy_target == "pi"
            and version == 2
            and (int(rec["flags"]) & FLAG_HAS_PI)
        ):
            pi = np.asarray(rec["pi"], dtype=np.float32)
            for t in legal_tokens:
                # Tokens stacked on one cell share it: their mass adds up.
                target_pi[cell_of(int(t))] += pi[int(t)]
            s = target_pi.sum()
            if s > 0:
                target_pi /= s
            else:
                target_pi[target_cell_idx] = 1.0
        else:
            target_pi[target_cell_idx] = 1.0

        x = np.expand_dims(encode_frame(state, pov_player=cp).astype(np.float32), axis=0)

        return {
            "x": torch.from_numpy(x),
            "target": target_cell_idx,
            "target_pi": torch.from_numpy(target_pi),
            "legal_mask": torch.from_numpy(legal_mask),
            "target_value": torch.tensor(val, dtype=torch.float32),
            "legal_count": len(legal_tokens),
        }
