"""V17 encoder helper for V13.2 — V14_minimal (14ch) + 3 static channels.

The 3 static channels are extracted from V11 encoder channels 5/6/7:
  Ch 5: Safe Zones        (8 cells, value 0.5)
  Ch 6: My Home Path      (5 cells, value 1.0)
  Ch 7: Opp Home Path     (5 cells per opp, overlaid; value 1.0)

In a 2P game with both players always active and the post-fix canonical
encoder, these 3 channels are CONSTANT across all states + dice + cps.
We compute them once at import time from V11 and cache. Per-state encoding
becomes a cheap concat — no extra C++ calls beyond the V14 encode.

Functions:
    encode_state_v17(state) -> np.ndarray of shape (17, 15, 15)

Hardening:
    validate_static_channels()  — call periodically; raises if cache drifts.
"""
from __future__ import annotations

import numpy as np
import td_ludo_cpp as ludo_cpp

V14_CHANNELS = 14
STATIC_CHANNELS = 3   # V11 ch 5, 6, 7
V17_CHANNELS = V14_CHANNELS + STATIC_CHANNELS  # = 17


def _build_static_cache() -> np.ndarray:
    """Extract the 3 static channels from V11 encoder using a fresh state."""
    g = ludo_cpp.create_initial_state_2p()
    g.current_player = 0
    g.current_dice_roll = 6
    enc = np.asarray(ludo_cpp.encode_state_v11(g), dtype=np.float32)
    return enc[5:8].copy()  # (3, 15, 15)


# One-time compute at module import. ~negligible cost.
_STATIC_CACHE: np.ndarray = _build_static_cache()


def encode_state_v17(state) -> np.ndarray:
    """V17 encoding: V14_minimal (14ch) + 3 static V11 channels (safe + home paths).
    Returns (17, 15, 15) float32 array."""
    v14 = np.asarray(ludo_cpp.encode_state_v14_minimal(state), dtype=np.float32)
    # Concat along channel axis. _STATIC_CACHE shape (3, 15, 15).
    return np.concatenate([v14, _STATIC_CACHE], axis=0)


def validate_static_channels(state=None) -> None:
    """Assert the cached static channels still match a fresh V11 extract.
    Catches encoder drift / config changes (e.g. 4P game, encoder rebuild).
    Pass any state; defaults to a fresh 2P initial state."""
    if state is None:
        state = ludo_cpp.create_initial_state_2p()
        state.current_player = 0
        state.current_dice_roll = 6
    fresh = np.asarray(ludo_cpp.encode_state_v11(state), dtype=np.float32)[5:8]
    if not np.allclose(_STATIC_CACHE, fresh, atol=1e-6):
        raise RuntimeError(
            f"V17 static channel drift detected. "
            f"Cached safe={_STATIC_CACHE[0].sum():.1f} fresh safe={fresh[0].sum():.1f}. "
            f"Either the encoder changed, or this is a non-2P-active state. "
            f"V13.2 pipeline assumes 2-player setup with both players active."
        )


if __name__ == '__main__':
    print(f'V17 encoder: {V17_CHANNELS} channels (V14 {V14_CHANNELS} + static {STATIC_CHANNELS})')
    print(f'Static cache shape: {_STATIC_CACHE.shape}')
    print(f'  Ch5 (safe): {int(_STATIC_CACHE[0].sum())} cells × value 0.5')
    print(f'  Ch6 (my home path): {int(_STATIC_CACHE[1].sum())} cells × value 1.0')
    print(f'  Ch7 (opp home path): {int(_STATIC_CACHE[2].sum())} cells × value 1.0')

    g = ludo_cpp.create_initial_state_2p()
    enc = encode_state_v17(g)
    print(f'\nencode_state_v17 output shape: {enc.shape}')
    assert enc.shape == (17, 15, 15)

    # Validate static across a few cps + states
    for cp in (0, 2):
        for dice in (1, 6):
            g2 = ludo_cpp.create_initial_state_2p()
            g2.current_player = cp
            g2.current_dice_roll = dice
            validate_static_channels(g2)
    print('Static channel cp/dice invariance validated ✓')

    # Validate after some moves
    import random
    random.seed(7)
    for _ in range(20):
        if g.current_dice_roll == 0:
            g.current_dice_roll = random.randint(1, 6)
        legal = ludo_cpp.get_legal_moves(g)
        if not legal:
            nxt = (g.current_player + 1) % 4
            while not g.active_players[nxt]: nxt = (nxt + 1) % 4
            g.current_player = nxt; g.current_dice_roll = 0
            continue
        g = ludo_cpp.apply_move(g, random.choice(legal))
    validate_static_channels(g)
    print('Static channel post-move invariance validated ✓')
