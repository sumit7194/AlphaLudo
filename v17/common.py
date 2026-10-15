"""Shared helpers for V17 training: losses, checkpoints, status files, stop files."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent


def pick_device(name: str) -> torch.device:
    if name == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def save_atomic(obj, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def write_json_atomic(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=float))
    os.replace(tmp, path)


def append_jsonl(path: Path, obj) -> None:
    with open(path, "a") as f:
        f.write(json.dumps({"time": time.time(), **obj}, default=float) + "\n")


def stop_requested(run_dir: Path) -> bool:
    return (run_dir / "stop").exists() or (ROOT / "stop").exists()


def dense_targets(cells: np.ndarray, pi: np.ndarray, n_cells: np.ndarray):
    """Sparse (cells, pi, n) -> dense legal mask (B,225) and policy target (B,225)."""
    B = len(cells)
    mask = np.zeros((B, 225), np.float32)
    target = np.zeros((B, 225), np.float32)
    rows = np.repeat(np.arange(B), 4)
    valid = (np.arange(4)[None, :] < n_cells[:, None]).reshape(-1)
    c = cells.reshape(-1).astype(np.int64)
    mask[rows[valid], c[valid]] = 1.0
    np.add.at(target, (rows[valid], c[valid]), pi.reshape(-1)[valid])
    s = target.sum(1, keepdims=True)
    target = np.where(s > 0, target / np.maximum(s, 1e-12), mask / np.maximum(mask.sum(1, keepdims=True), 1))
    return mask, target


def policy_value_losses(policy_logits, value_logits, target_pi, decision, winner_rel):
    """Soft cross-entropy on decisions (>=2 legal cells) + winner cross-entropy on finished games.
    Returns (policy_loss_sum, n_policy, value_loss_sum, n_value) so micro-batches can be
    normalised by the FULL batch counts (identical to one big batch)."""
    logp = F.log_softmax(policy_logits, dim=-1)
    pl = -(target_pi * logp).sum(-1)
    pl_sum = (pl * decision).sum()
    finished = winner_rel < 4
    vl = F.cross_entropy(value_logits, winner_rel.clamp(max=3).long(), reduction="none")
    vl_sum = (vl * finished).sum()
    return pl_sum, decision.sum(), vl_sum, finished.sum()
