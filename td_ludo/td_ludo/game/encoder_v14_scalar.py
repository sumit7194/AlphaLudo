"""V14_scalar encoder helper — flattens C++ dict into model-ready arrays.

The C++ binding `encode_state_v14_scalar(state)` returns a dict with 15 keys
(see `td_ludo/src/game.h::V14ScalarEncoding`). This module packs them into
exactly four arrays the V14ScalarDeepSets model expects:

  own_pos      (4,)   int64    — position embedding indices, 0..58
  own_features (4, 8) float32  — [in_danger, can_capture, can_score,
                                  can_land_safe, is_safe, at_base, at_home,
                                  idle_count]
  opp_pos      (4,)   int64    — position embedding indices, 0..58
  opp_features (4, 5) float32  — [in_my_danger, threatens_me, is_safe,
                                  at_base, at_home]
  globals      (13,)  float32  — [dice_one_hot(6), streak, my_lock,
                                  opp_lock, score_diff, leader_progress,
                                  non_home_frac, bonus_turn_flag]

Total per-state: 4*1 + 4*8 + 4*1 + 4*5 + 13 = 73 numbers.
"""
from __future__ import annotations

import numpy as np
import td_ludo_cpp as ludo_cpp

# Feature ordering — keep in sync with the model's input expectations.
OWN_FEAT_ORDER = (
    "own_in_danger",
    "own_can_capture",
    "own_can_score",
    "own_can_land_safe",
    "own_is_safe",
    "own_at_base",
    "own_at_home",
    "own_idle_count",
)
OPP_FEAT_ORDER = (
    "opp_in_my_danger",
    "opp_threatens_me",
    "opp_is_safe",
    "opp_at_base",
    "opp_at_home",
)
GLOBAL_SCALAR_ORDER = (
    "same_token_streak",
    "my_locked_frac",
    "opp_locked_frac",
    "score_diff",
    "leader_progress",
    "non_home_tokens_frac",
)
NUM_OWN_FEATS = len(OWN_FEAT_ORDER)        # 8
NUM_OPP_FEATS = len(OPP_FEAT_ORDER)        # 5
NUM_GLOBALS = 6 + len(GLOBAL_SCALAR_ORDER) + 1  # dice_one_hot(6) + 6 scalars + bonus = 13
NUM_POS_EMB = 60                            # 0..58 used; 59 reserved

# Total dim of the flat tensor used by the RL pipeline (CNN-compatible
# (FLAT_DIM, 1, 1) shape so the existing trainer batches it identically).
# Layout (slice indices):
#   [0,4)        own_pos        (int as float; pack/unpack via long())
#   [4, 36)      own_features   (4 tokens × 8 feats, flattened C-order)
#   [36, 40)    opp_pos
#   [40, 60)    opp_features   (4 tokens × 5 feats)
#   [60, 73)    globals
NUM_TOKENS = 4
FLAT_DIM = (
    NUM_TOKENS                            # own_pos
    + NUM_TOKENS * NUM_OWN_FEATS          # own_features
    + NUM_TOKENS                          # opp_pos
    + NUM_TOKENS * NUM_OPP_FEATS          # opp_features
    + NUM_GLOBALS                         # globals
)
SLICE_OWN_POS  = (0, 4)
SLICE_OWN_FEAT = (4, 4 + NUM_TOKENS * NUM_OWN_FEATS)         # (4, 36)
SLICE_OPP_POS  = (SLICE_OWN_FEAT[1], SLICE_OWN_FEAT[1] + NUM_TOKENS)  # (36, 40)
SLICE_OPP_FEAT = (SLICE_OPP_POS[1], SLICE_OPP_POS[1] + NUM_TOKENS * NUM_OPP_FEATS)  # (40, 60)
SLICE_GLOBALS  = (SLICE_OPP_FEAT[1], FLAT_DIM)               # (60, 73)


def encode_state_v14_scalar(state) -> dict:
    """Encode one state into model-ready numpy arrays."""
    raw = ludo_cpp.encode_state_v14_scalar(state)

    own_pos = raw["own_pos"].astype(np.int64)
    opp_pos = raw["opp_pos"].astype(np.int64)

    own_features = np.stack(
        [raw[k].astype(np.float32) for k in OWN_FEAT_ORDER], axis=-1
    )  # (4, 8)
    opp_features = np.stack(
        [raw[k].astype(np.float32) for k in OPP_FEAT_ORDER], axis=-1
    )  # (4, 5)

    # Globals: dice one-hot(6) + 6 scalars + bonus_turn_flag = 13
    dice_one_hot = np.zeros(6, dtype=np.float32)
    d = int(raw["dice"])
    if 1 <= d <= 6:
        dice_one_hot[d - 1] = 1.0
    scalars = np.array(
        [float(raw[k]) for k in GLOBAL_SCALAR_ORDER], dtype=np.float32
    )
    bonus = np.array([1.0 if raw["bonus_turn_flag"] else 0.0], dtype=np.float32)
    globals_vec = np.concatenate([dice_one_hot, scalars, bonus], axis=0)  # (13,)

    return {
        "own_pos": own_pos,
        "own_features": own_features,
        "opp_pos": opp_pos,
        "opp_features": opp_features,
        "globals": globals_vec,
    }


def encode_batch_v14_scalar(states) -> dict:
    """Encode a list of game states into batched arrays.

    Returns dict with leading batch dim added to every tensor.
    """
    encoded = [encode_state_v14_scalar(s) for s in states]
    return {
        k: np.stack([e[k] for e in encoded], axis=0) for k in encoded[0].keys()
    }


def encode_state_v14_scalar_flat(state) -> np.ndarray:
    """Same content as `encode_state_v14_scalar` but flattened into a single
    (FLAT_DIM, 1, 1) float32 ndarray so the existing RL trainer (which
    assumes encoder_fn returns a (C, H, W) tensor) can consume it without
    modification.

    The model's forward path detects this layout (4D batched input) and
    unpacks it back into the per-token + global structure internally.

    Position values (own_pos, opp_pos) are cast to float; positions are
    small ints (0..58) so float32 represents them exactly. The model
    casts back to long before the embedding lookup.
    """
    enc = encode_state_v14_scalar(state)
    flat = np.empty(FLAT_DIM, dtype=np.float32)
    a, b = SLICE_OWN_POS;  flat[a:b] = enc["own_pos"].astype(np.float32)
    a, b = SLICE_OWN_FEAT; flat[a:b] = enc["own_features"].reshape(-1)
    a, b = SLICE_OPP_POS;  flat[a:b] = enc["opp_pos"].astype(np.float32)
    a, b = SLICE_OPP_FEAT; flat[a:b] = enc["opp_features"].reshape(-1)
    a, b = SLICE_GLOBALS;  flat[a:b] = enc["globals"]
    return flat.reshape(FLAT_DIM, 1, 1)
