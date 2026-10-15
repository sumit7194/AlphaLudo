"""Answers GameFarm network requests for any mix of models.

A Player wraps one network and returns, for a batch of farm requests, the policy
over the 225 source cells and the mover's win probability. Batches are padded to
a few fixed power-of-two sizes (8..pad): unbounded shape variety grows the MPS graph
cache and MLX recompiles per shape, but always padding to `pad` wastes most of the
compute on small batches.

Backends (all float32, verified equal to the PyTorch reference):
  v17 + "mlx"   : MlxV17 on the Apple GPU (fast inference)
  v17 + "torch" : V17Net on cpu/mps (e.g. the learner's current weights)
  v15 + "mlx"   : the v15.2 champion (reference opponent), MlxV15
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
CHAMPION_PATH = ROOT / "td_ludo" / "play" / "model_weights" / "v15_2" / "model_best.pt"


class Player:
    encoder = "unified"

    def __init__(self, pad: int = 256):
        self.pad = pad
        self.evals = 0

    def _forward(self, x: np.ndarray, mask: np.ndarray):
        raise NotImplementedError

    def bucket(self, k: int) -> int:
        """Smallest power-of-two batch (>= 8) holding k rows, capped at pad. A few fixed
        shapes keep the graph caches bounded; small batches (league snapshots answer
        ~5 rows) no longer pay for a full pad-sized forward."""
        b = 8
        while b < k and b < self.pad:
            b *= 2
        return min(b, self.pad)

    def _forward_full(self, x: np.ndarray, mask: np.ndarray):
        raise NotImplementedError

    def answer_full(self, x: np.ndarray, mask: np.ndarray):
        """Like answer(), but the value is the full (n, 4) win-probability vector over
        RELATIVE seats (k = k seats after the mover) — what Max^n search needs."""
        n = len(x)
        pol = np.empty((n, 225), np.float32)
        val = np.empty((n, 4), np.float32)
        for lo in range(0, n, self.pad):
            k = min(self.pad, n - lo)
            b = self.bucket(k)
            xb = np.zeros((b, x.shape[1]), np.int8)
            mb = np.zeros((b, 225), np.float32)
            mb[:, 0] = 1.0
            xb[:k], mb[:k] = x[lo:lo + k], mask[lo:lo + k]
            p, v = self._forward_full(xb, mb)
            pol[lo:lo + k], val[lo:lo + k] = p[:k], v[:k]
        self.evals += n
        return pol, val

    def answer(self, x: np.ndarray, mask: np.ndarray):
        n = len(x)
        pol = np.empty((n, 225), np.float32)
        val = np.empty(n, np.float32)
        for lo in range(0, n, self.pad):
            k = min(self.pad, n - lo)
            b = self.bucket(k)
            xb = np.zeros((b, x.shape[1]), np.int8)
            mb = np.zeros((b, 225), np.float32)
            mb[:, 0] = 1.0                     # padding rows: keep softmax finite
            xb[:k], mb[:k] = x[lo:lo + k], mask[lo:lo + k]
            p, v = self._forward(xb, mb)
            pol[lo:lo + k], val[lo:lo + k] = p[:k], v[:k]
        self.evals += n
        return pol, val


class V17Player(Player):
    """V17Net; value returned = P(mover wins) = softmax over active seats, index 0."""

    def __init__(self, net, backend: str = "mlx", device: str = "cpu", pad: int = 256):
        super().__init__(pad)
        self.backend = backend
        if backend == "mlx":
            from .mlx_model import MlxV17
            self.m = MlxV17(net)
        else:
            self.device = torch.device(device)
            self.m = net.to(self.device)
            self.m.train(False)

    def _forward_full(self, x, mask):
        if self.backend == "mlx":
            p, v = self.m(x, mask)
            return np.asarray(p), np.asarray(v)
        with torch.no_grad():
            p, v = self.m(torch.from_numpy(x).to(self.device), torch.from_numpy(mask).to(self.device))
            return p.float().cpu().numpy(), v.float().cpu().numpy()

    def _forward(self, x, mask):
        if self.backend == "mlx":
            p, v = self.m(x, mask)
            return p, v[:, 0]
        with torch.no_grad():
            p, v = self.m(torch.from_numpy(x).to(self.device), torch.from_numpy(mask).to(self.device))
            return p.float().cpu().numpy(), v[:, 0].float().cpu().numpy()


class V15Player(Player):
    """The v15.2 champion (v15 encoding, sigmoid value)."""

    encoder = "v15"

    def __init__(self, path: Path = CHAMPION_PATH, pad: int = 256):
        super().__init__(pad)
        sys.path.insert(0, str(ROOT / "python" / "alphaludo"))
        for p in (ROOT / "td_ludo", ROOT / "td_ludo_v15"):
            if str(p) not in sys.path:
                sys.path.insert(0, str(p))
        from net_farm import load_v15
        self.m = load_v15(str(path), torch.device("cpu"), backend="mlx")

    def _forward(self, x, mask):
        return self.m(x, mask)


def load_v17(path, device: str = "cpu"):
    from .model import V17Net
    ck = torch.load(path, map_location="cpu", weights_only=False)
    net = V17Net(**ck.get("model_cfg", {}))
    net.load_state_dict(ck["model_state_dict"])
    net.train(False)
    return net.to(device)


def serve(farm, players: Dict[int, Player]) -> int:
    """Plays the farm until it has no games left; returns network evaluations."""
    evals = 0
    while farm.collect():
        for m in farm.models_pending():
            x, mask = farm.batch(m)
            p, v = players[m].answer(x, mask)
            farm.submit(m, p, v)
            evals += len(x)
    return evals
