"""Common SL dataset loader for the V12.3 / V13.6 / V15.2 trainers.

Reads the sharded .npz dataset produced by `generate_sl_dataset.py`.
Each shard contains rows of (player_positions, current_player, dice_roll,
action) where `action` is the token-ID picked by the winning side.

The dataset reconstructs a cpp game state per row and encodes it using
the model-specific encoder. It also computes the legal mask and converts
the raw `action` (token-ID) into the model-specific action index:

  - V12.3 (token-indexed 4-token policy):  index = action (token-ID 0..3)
  - V13.6 (rank-indexed 4-token policy):   index = rank-position of
       the chosen token among own tokens (0=most-advanced, 3=least)
  - V15.2 (225-cell source-cell policy):   index = cell_to_index of the
       chosen token's current position in the current player's POV

Usage:
    from sl_dataset import BotGamesDataset
    ds = BotGamesDataset(
        shard_dir="checkpoints/sl_dataset_v1",
        variant="v15_2",  # or "v13_6", "v12_3"
        shard_indices=None,  # None = all shards
    )
    # ds[i] -> dict with model-specific 'x', 'action', 'legal_mask'

Memory plan: shards are mapped lazily (loaded on first access, kept
in memory once loaded). With ~75M total rows × 23 bytes raw = ~1.7 GB
in raw form. Encoded form (V15.2 = ~0.6KB per row) is too big to
preload; we encode on __getitem__.
"""
from __future__ import annotations

import os
import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


# ── State reconstruction (cheap; ~3μs per row) ──────────────────────────
_LUDO_CPP = None
_INITIAL_STATE_TEMPLATE = None


def _ensure_cpp():
    """Lazy import — keep top-level light for DataLoader workers."""
    global _LUDO_CPP, _INITIAL_STATE_TEMPLATE
    if _LUDO_CPP is None:
        import td_ludo_cpp as _cpp
        _LUDO_CPP = _cpp
        _INITIAL_STATE_TEMPLATE = _cpp.create_initial_state_2p()


def _reconstruct_state(pp_row, cp, dice):
    """Build a cpp game state from saved arrays.

    pp_row: (4, 4) int  — player_positions
    cp: int             — current_player (0 or 2)
    dice: int           — current_dice_roll (1..6)
    """
    _ensure_cpp()
    state = _LUDO_CPP.create_initial_state_2p()
    # Pybind exposes settable fields for all of these (verified).
    state.player_positions = pp_row.tolist()
    state.current_player = int(cp)
    state.current_dice_roll = int(dice)
    return state


# ── Variant-specific encoders ───────────────────────────────────────────

def _encode_v12_3(state, action_token_id):
    """V12.3: V17 encoder (17ch engineered features), token-indexed policy.

    Returns: (x (17,15,15) float32, action_index int, legal_mask (4,) float32)
    """
    from td_ludo.game.encoder_v17 import encode_state_v17
    enc = encode_state_v17(state).astype(np.float32)
    legal = _LUDO_CPP.get_legal_moves(state)
    legal_mask = np.zeros(4, dtype=np.float32)
    for a in legal:
        legal_mask[a] = 1.0
    return enc, int(action_token_id), legal_mask


def _encode_v12(state, action_token_id):
    """V12 (legibility A/B): V10 28-ch encoder, token-indexed 4-token policy.
    Own tokens = input channels 0-3, opp = 17-20 (one-hot per token).

    Returns: (x (28,15,15) float32, action_index int, legal_mask (4,) float32)
    """
    enc = np.asarray(_LUDO_CPP.encode_state_v10(state), dtype=np.float32)
    legal = _LUDO_CPP.get_legal_moves(state)
    legal_mask = np.zeros(4, dtype=np.float32)
    for a in legal:
        legal_mask[a] = 1.0
    return enc, int(action_token_id), legal_mask


def _encode_v13_6(state, action_token_id):
    """V13.6: V18-symmetric (13ch), rank-indexed 4-token policy.

    Returns: (x (13,15,15) float32, rank_masks (4,15,15) float32,
              action_rank int, rank_legal (4,) float32)
    """
    from td_ludo.game.encoder_v18_symmetric import encode_state_v18_symmetric
    from td_ludo.game.rank_mapping import (
        state_to_rank_mapping, legal_mask_per_rank,
    )
    from td_ludo.models.v13_5 import compute_rank_masks
    enc = encode_state_v18_symmetric(state).astype(np.float32)
    rmasks = compute_rank_masks(state).astype(np.float32)
    cp = int(state.current_player)
    pp = state.player_positions[cp]
    _, rank_tokens = state_to_rank_mapping(pp)
    legal = _LUDO_CPP.get_legal_moves(state)
    rank_legal = legal_mask_per_rank(legal, rank_tokens).astype(np.float32)

    # Convert chosen token-id to its rank.
    # NOTE: rank_tokens is List[List[int]] — each entry is the LIST of
    # token-ids sitting at that rank's position. So we check membership,
    # not equality. (The old `t == int(action_token_id)` bug compared a
    # list to an int → always False → fallback below always fired →
    # target became "lowest legal rank", which is trivially predictable
    # from legal_mask alone and gave the model 100% acc by step ~350.)
    action_rank = -1
    aid = int(action_token_id)
    for r, tokens_in_rank in enumerate(rank_tokens):
        if aid in tokens_in_rank:
            action_rank = r
            break
    if action_rank < 0:
        # Fallback: shouldn't happen because chosen action is always legal.
        # Use the first legal rank.
        legal_ranks = np.where(rank_legal > 0)[0]
        action_rank = int(legal_ranks[0]) if len(legal_ranks) else 0
    return enc, rmasks, int(action_rank), rank_legal


def _encode_v15_2(state, action_token_id):
    """V15.2: per-cell triplet, history_len=1 (single frame), 225-cell policy.

    Returns: (x (1,15,15,3) float32, action_cell_index int,
              legal_mask (225,) float32)
    """
    from td_ludo_v15.game.encoder import encode_frame
    from td_ludo_v15.game.cells import (
        NUM_BOARD_CELLS, cell_to_index, position_to_cell_in_pov,
    )
    import td_ludo_v15_cpp as v15_cpp
    BASE_POS = v15_cpp.BASE_POS

    cp = int(state.current_player)
    frame = encode_frame(state, pov_player=cp).astype(np.float32)
    x = frame[np.newaxis, ...]  # (1, 15, 15, 3)

    legal = _LUDO_CPP.get_legal_moves(state)
    legal_mask = np.zeros(225, dtype=np.float32)
    legal_cells = {}
    for t in legal:
        pos = int(state.player_positions[cp][t])
        c = position_to_cell_in_pov(BASE_POS if pos == BASE_POS else pos, cp, cp)
        idx = cell_to_index(*c)
        legal_mask[idx] = 1.0
        legal_cells[t] = idx
    # Action is the source-cell index of the chosen token.
    action_cell_index = legal_cells.get(int(action_token_id), -1)
    if action_cell_index < 0:
        # Fallback — shouldn't happen because action is always legal.
        legal_idxs = np.where(legal_mask > 0)[0]
        action_cell_index = int(legal_idxs[0]) if len(legal_idxs) else 0
    return x, int(action_cell_index), legal_mask


# ── Dataset class ───────────────────────────────────────────────────────

class BotGamesDataset(Dataset):
    """Sharded SL dataset. One variant at a time.

    The shards stay on disk; we mmap each shard the first time we touch
    a row in it. Encoding is done in __getitem__ to keep memory bounded.
    """

    SUPPORTED_VARIANTS = ("v12_3", "v13_6", "v15_2", "v12")

    def __init__(
        self,
        shard_dir: os.PathLike,
        variant: str,
        shard_indices: Optional[List[int]] = None,
        max_rows: Optional[int] = None,
        filter_winner_bots: Optional[List[str]] = ("Random",),
    ):
        """
        filter_winner_bots: list of bot names whose WINNER moves should be
            dropped. Default drops Random-won games (~0.4% of rows but pure
            noise — Random's moves are not strategy). Pass [] or None to
            disable filtering entirely.
        """
        if variant not in self.SUPPORTED_VARIANTS:
            raise ValueError(f"variant must be one of {self.SUPPORTED_VARIANTS}, got {variant}")
        self.variant = variant
        self.shard_dir = Path(shard_dir)
        self.filter_winner_bots = set(filter_winner_bots or [])

        shard_paths = sorted(self.shard_dir.glob("shard_*.npz"))
        if shard_indices is not None:
            shard_paths = [p for p in shard_paths
                           if int(p.stem.split("_")[1]) in set(shard_indices)]
        if not shard_paths:
            raise FileNotFoundError(f"No shards found in {self.shard_dir}")
        self.shard_paths = shard_paths

        # Build per-shard "valid row" index. For each shard we:
        #  1. Read meta to find which game_idx values were won by a
        #     filter_winner_bots bot (skip those games' rows).
        #  2. Open shard's game_idx column (~few hundred KB) to map row→game.
        #  3. Build `valid_rows[shard_idx] = np.array([row_idx, ...])` —
        #     only those row indices that survive filtering.
        # __getitem__ then looks up: global_idx → (shard_idx, local_idx)
        # → valid_rows[shard_idx][local_idx] → actual row in shard.
        self._row_offsets: List[int] = [0]
        self._valid_rows: List[np.ndarray] = []
        n_dropped = 0
        n_kept = 0
        for sp in shard_paths:
            meta_path = sp.with_suffix(".meta.json")
            if not meta_path.exists():
                # No meta sidecar — keep all rows, can't filter.
                with np.load(sp) as z:
                    n = z["action"].shape[0]
                self._valid_rows.append(np.arange(n, dtype=np.int32))
                self._row_offsets.append(self._row_offsets[-1] + n)
                continue
            with open(meta_path) as f:
                m = json.load(f)
            # Identify drop-target game indices within THIS shard.
            drop_game_idxs = set()
            for game_idx_in_shard, g in enumerate(m["games"]):
                winner_bot = g["p0"] if g["winner"] == 0 else g["p2"]
                if winner_bot in self.filter_winner_bots:
                    drop_game_idxs.add(game_idx_in_shard)
            if not drop_game_idxs:
                # Nothing to drop — keep all rows.
                n = m["n_rows"]
                self._valid_rows.append(np.arange(n, dtype=np.int32))
                self._row_offsets.append(self._row_offsets[-1] + n)
                n_kept += n
                continue
            # Load just the game_idx column to mask rows.
            with np.load(sp) as z:
                game_idxs = z["game_idx"][:]   # (N,) int32
            drop_arr = np.fromiter(drop_game_idxs, dtype=np.int32)
            keep_mask = ~np.isin(game_idxs, drop_arr)
            valid = np.where(keep_mask)[0].astype(np.int32)
            self._valid_rows.append(valid)
            self._row_offsets.append(self._row_offsets[-1] + len(valid))
            n_kept += len(valid)
            n_dropped += (len(game_idxs) - len(valid))
        self._total_rows = self._row_offsets[-1]
        if self.filter_winner_bots and n_dropped > 0:
            pct = 100.0 * n_dropped / max(1, n_dropped + n_kept)
            print(f"[BotGamesDataset] dropped {n_dropped:,} rows "
                  f"({pct:.2f}%) won by {sorted(self.filter_winner_bots)}; "
                  f"kept {n_kept:,}")
        if max_rows is not None:
            self._total_rows = min(self._total_rows, max_rows)

        # Shard caches (one np memmap per shard) — populated lazily.
        self._shard_cache: dict = {}

    def __len__(self):
        return self._total_rows

    def _locate(self, global_idx):
        """Map global row index to (shard_idx, row_in_shard)."""
        # Binary search on _row_offsets.
        import bisect
        s_idx = bisect.bisect_right(self._row_offsets, global_idx) - 1
        return s_idx, global_idx - self._row_offsets[s_idx]

    def _get_shard(self, shard_idx: int):
        if shard_idx not in self._shard_cache:
            self._shard_cache[shard_idx] = np.load(
                self.shard_paths[shard_idx], mmap_mode="r"
            )
        return self._shard_cache[shard_idx]

    def __getitem__(self, idx):
        if idx >= self._total_rows:
            raise IndexError(idx)
        shard_idx, local_idx = self._locate(idx)
        # local_idx is the position within this shard's VALID rows.
        # Resolve to the actual row index inside the shard's arrays.
        actual_row = int(self._valid_rows[shard_idx][local_idx])
        shard = self._get_shard(shard_idx)
        pp = np.asarray(shard["player_positions"][actual_row])
        cp = int(shard["current_player"][actual_row])
        dice = int(shard["dice_roll"][actual_row])
        action = int(shard["action"][actual_row])

        state = _reconstruct_state(pp, cp, dice)

        if self.variant == "v12":
            x, target, legal_mask = _encode_v12(state, action)
            return {
                "x": torch.from_numpy(x),
                "target": torch.tensor(target, dtype=torch.long),
                "legal_mask": torch.from_numpy(legal_mask),
            }
        if self.variant == "v12_3":
            x, target, legal_mask = _encode_v12_3(state, action)
            return {
                "x": torch.from_numpy(x),
                "target": torch.tensor(target, dtype=torch.long),
                "legal_mask": torch.from_numpy(legal_mask),
            }
        if self.variant == "v13_6":
            x, rmasks, target, rank_legal = _encode_v13_6(state, action)
            return {
                "x": torch.from_numpy(x),
                "rank_masks": torch.from_numpy(rmasks),
                "target": torch.tensor(target, dtype=torch.long),
                "legal_mask": torch.from_numpy(rank_legal),
            }
        if self.variant == "v15_2":
            x, target, legal_mask = _encode_v15_2(state, action)
            return {
                "x": torch.from_numpy(x),
                "target": torch.tensor(target, dtype=torch.long),
                "legal_mask": torch.from_numpy(legal_mask),
            }
        raise AssertionError(f"Unhandled variant {self.variant}")


def make_train_val_split(
    shard_dir: os.PathLike, val_fraction: float = 0.05, seed: int = 42,
) -> tuple[list[int], list[int]]:
    """Return (train_shard_indices, val_shard_indices) by held-out shard.

    Splitting by SHARD (not row) avoids leakage of within-game correlation.
    Each shard's games come from independent dice/bot streams.
    """
    shard_paths = sorted(Path(shard_dir).glob("shard_*.npz"))
    n = len(shard_paths)
    if n == 0:
        raise FileNotFoundError(f"No shards in {shard_dir}")
    rng = np.random.default_rng(seed)
    idxs = np.arange(n)
    rng.shuffle(idxs)
    n_val = max(1, int(round(n * val_fraction)))
    val = sorted(int(i) for i in idxs[:n_val])
    train = sorted(int(i) for i in idxs[n_val:])
    return train, val
