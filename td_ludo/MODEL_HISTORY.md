# AlphaLudo Model Architecture History

A chronological catalogue of every model architecture and input encoding
tried in this project, from the first commit to the current production
V12.2. Each entry covers what changed, why, and the input channels.

For training results and experiment outcomes see `td_ludo/training_journal.md`.

---

## V1 — `AlphaLudoNet` (initial commit `759f05b`, Feb 2026)

The original. Pure CNN, hand-rolled 8-channel encoder.

- **Input:** `(8, 15, 15)` spatial tensor.
- **Backbone:** Stem `Conv2d(8 → 128, 3×3) + BN + ReLU` → 10 ResidualBlocks (128 channels).
- **Heads:** policy `Conv2d(128, 32, 1×1) + Linear → 4` (per-token), value `Conv2d(128, 16, 1×1) + Linear → 1`.
- **File:** `src/model.py` (initial commit).

**Input channels (`tensor_utils.state_to_tensor`):**

| Ch | Encodes |
|---|---|
| 0–3 | Per-player position masks (one channel per player, 1.0 at each token) |
| 4 | Safe zones (constant board mask) |
| 5 | Home paths (constant board mask, all four players' home stretches) |
| 6 | Dice roll, broadcast as `roll / 6.0` |
| 7 | Turn indicator, broadcast as `(current_player + 1) * 0.25` |

**Notes:** No per-token identity (a player's 4 tokens collapse into one
mask), no opponent-specific channels, dice as a scalar rather than a
one-hot. The model couldn't distinguish "which of my four tokens is
where," forcing the policy head to break the tie purely from the legal-
move mask.

---

## V1.5 — `AlphaLudoTopNet` ("mastery", first commit `759f05b`)

A sibling to V1, lived in `src/model_mastery.py`. Wider encoder (18
channels), otherwise the same ResNet-10 × 128ch CNN. Channel layout
isn't documented in the source — this branch was experimental and
short-lived. Mentioned here for completeness.

- **Input:** `(18, 15, 15)`.
- **Backbone:** ResNet-10 × 128ch (same as V1).
- **File:** `src/model_mastery.py`.

---

## V3 — `td_v2_11ch` (commit `2ba5514`, Feb 2026)

First model where the encoder was overhauled. The one-hot dice planes
weren't here yet; this was the lean "physical state only" run.

- **Input:** `(11, 15, 15)`.
- **Backbone:** `AlphaLudoV3` — ResNet-10 × 128ch, `in_channels=11`.
- **Heads:** policy → 4 token logits, value → scalar, optional aux safety head.
- **File:** `td_ludo/src/model.py` (this commit set `in_channels=11`).

**Input channels (`write_state_tensor` 11ch variant):**

| Ch | Encodes |
|---|---|
| 0–3 | My tokens, **per-token identity** (one channel per own token; 1.0 at the token's cell) |
| 4 | Opponent tokens, single density plane (0.25 per token, summed; inactive players skipped) |
| 5 | Safe zones (0.5) |
| 6 | My home path |
| 7 | Opponent home path (skip inactive) |
| 8 | Score diff `(my_score − max_opp_score) / 4` (broadcast) |
| 9 | My locked fraction `count_in_base / 4` (broadcast) |
| 10 | Opp locked fraction (broadcast) |

**What changed vs V1:** per-token identity for own tokens (the model can
finally tell its four tokens apart), opponent rolled into a single
density plane, dice/turn dropped (held outside the encoder for now),
two scalar broadcasts capturing relative game state.

Checkpoint: `td_ludo/checkpoints/td_v2_11ch/model_latest.pt`.

---

## V3 / V4 / V5 — 17ch encoder family (`td_ludo/td_ludo/models/v5.py`)

Reused the 11ch core but added a **6-plane dice one-hot** (one channel
per dice value 1–6). All three architectures share the same encoder;
they differ only in CNN depth/width and head shape.

- **Input:** `(17, 15, 15)`.
- **Encoder:** original `write_state_tensor` (17ch).

| Variant | ResBlocks | Channels | Heads | Notes |
|---|---|---|---|---|
| V3 (`AlphaLudoV3`) | 10 | 128 | policy(4) + value(1) + aux safety(4) | Direct token selection — policy outputs 4 logits with legal-move mask before softmax |
| V4 (`AlphaLudoV4`) | 3 | 32 | same as V3 | Slim experiment to test capacity sensitivity |
| V5 (`AlphaLudoV5`) | 5 | 64 | same as V3 | Goldilocks midpoint between V3 and V4 |

**Input channels (17ch):**

| Ch | Encodes |
|---|---|
| 0–10 | Same as 11ch encoder above |
| 11–16 | Dice one-hot — 6 broadcast planes, exactly one is all-1.0 |

**What changed vs 11ch:** Dice as 6 disjoint spatial planes. CNNs can
form per-dice strategy filters cleanly.

Caveat: very early commits had `in_channels=21` floating around in
`model_v3.py` defaults; the encoder that produced 21 channels was
short-lived and pre-dates the 11ch cleanup. The "real" V3/V4/V5 line
is 17ch.

---

## V6.1 — 24ch strategic encoder (`write_state_tensor_v6`)

Encoder gets significantly richer. First time we add **per-token
opponent identity** and **derived "tactical maps"** computed from the
current dice.

- **Input:** `(24, 15, 15)`.
- **Backbone:** ResNet × 128ch, depth 5–10 (configurable).
- **File:** `td_ludo/src/game.cpp` `write_state_tensor_v6`.

**Input channels:**

| Ch | Encodes |
|---|---|
| 0–16 | Same as V5 17ch encoder |
| 17–20 | Opponent tokens, **per-token identity** (replaces ch 4's density plane — ch 4 still exists but is now overshadowed by these distinct-identity opponents) |
| 21 | **Danger map** — 1.0 at own tokens that have an opponent within 1–6 squares behind |
| 22 | **Capture-opportunity map** — 1.0 at opponent positions reachable with the current dice |
| 23 | **Safe-landing map** — 1.0 at safe positions reachable with the current dice |

**What changed vs V5:** Opponent finally has per-token identity (so the
model can target a specific opponent piece). Three derived planes
(21–23) act as pre-computed "tactical hints" that the model would
otherwise need extra depth to extract.

---

## V6.2 — `AlphaLudoV62` (model only, encoder unchanged)

Architecture refactor over V6.1 — same 24ch input. Lives in
`td_ludo/td_ludo/models/v6_2.py`.

- **Input:** `(24, 15, 15)` — V6.1 encoder, no change.
- **Backbone:** ResNet × 128ch (default `num_res_blocks` configurable).
- **Heads:** Same 3-head shape as V5/V6 (policy/value/aux).

Used as a stepping stone to V6.3.

---

## V6.3 — 27ch encoder (`write_state_tensor_v6_3`)

Adds three "bonus-turn awareness" channels. This was the first attempt
to inject Ludo-specific rule knowledge directly into the input.

- **Input:** `(27, 15, 15)`.
- **Backbone:** `AlphaLudoV63` — ResNet × 128ch, `in_channels=27`,
  default 10 ResBlocks.

**Input channels:**

| Ch | Encodes |
|---|---|
| 0–23 | Identical to V6.1 |
| 24 | `bonus_turn_flag` — broadcast 1.0 if `dice == 6` |
| 25 | `consecutive_sixes` — broadcast `0 / 0.5 / 1.0` for 0/1/2 prior 6s |
| 26 | `two_roll_capture_map` — 1.0 at opponent positions capturable in a 6-then-X two-roll sequence (target offsets 7–12 ahead) |

**What changed vs V6.1:** Explicit signals for the bonus-turn rule
(dice==6 → another action). The CNN no longer has to derive this from
the dice one-hot alone; it gets a single broadcast flag plus a
spatial map of who could be captured if a second roll happens.

---

## V7 — 1D transformer (`src/state_encoder_1d.py`, `src/model_v7.py`)

A complete pivot away from the spatial CNN encoder. Treats the game as
a 1D sequence designed for transformer input.

- **Input:** **18-dim 1D vector per turn** (no spatial structure):
  - 8 token positions: 4 self + 4 opponent, each in `[0, 58]` —
    `0` = locked in base, `1–51` = main track (player-relative),
    `53–57` = home stretch, `58` = scored.
  - 3 globals: `opp_locked_frac`, `my_locked_frac`, `score_diff`.
  - 6 dice one-hot for the current roll.
  - 1 historical action token: last action taken (`0–3` = token,
    `4` = pass/none).
- **Backbone:** Transformer encoder over a sequence of past turns.
- **File:** `td_ludo/src/model_v7.py` (commit `ec44058`,
  `td_ludo/V7_ARCHITECTURE.md`).

**What changed:** Drops the spatial board entirely. Token positions
become integers in a learned embedding. Designed to test whether
transformers can recover spatial structure from sequence-level signal
alone, and to give the model explicit access to multi-turn history.

V7 didn't beat the V6.x line and was eventually retired.

---

## V8 — V6 CNN + temporal transformer (`src/model_v8.py`)

Hybrid: keeps the V5 17ch spatial encoder but adds a temporal
transformer that attends over `K=16` past turns. Each turn's CNN
features get summed with a previous-action embedding before going
into the transformer.

- **Input per turn:** `(17, 15, 15)` — same encoder as V5.
- **Backbone per turn:** V5 CNN (128ch ResNet-10, frozen or trainable)
  → `(128,)` GAP feature → add action embedding `(128,)` → LayerNorm.
- **Temporal stack:** `K × (128,)` + temporal positional embeddings,
  4-layer transformer encoder with causal masking, 4 heads.
- **Heads:** policy `Linear(128, 128) → ReLU → Linear(128, 4)`,
  value `Linear(128, 128) → ReLU → Linear(128, 1)`.

**What changed:** First model with proper multi-turn history. The CNN
is reused as a per-frame feature extractor; the transformer composes
those features across time.

---

## V9 — slim CNN + temporal transformer, 14ch encoder

Designed informed by mech-interp on V6:
- Layer knockout showed all 10 V6 ResBlocks were individually
  removable → over-parameterised.
- CKA showed blocks 5–9 were nearly identical (>0.99).
- Channel ablation: score-diff plane lowest impact; the 6-plane dice
  one-hot was wasteful.

So V9 slimmed both the encoder and the backbone.

- **Input:** `(14, 15, 15)`.
- **CNN backbone:** Stem `Conv2d(14 → 80) + BN + ReLU`, 5 ResBlocks of
  80 channels each. ~750K params.
- **Temporal transformer:** 4 layers, 80-dim, 4 heads, FFN 320, GELU,
  norm-first, causal + padding mask. ~400K params.
- **Heads:** policy + value.

**Input channels (14ch encoder, `write_state_tensor_v9`):**

| Ch | Encodes |
|---|---|
| 0–3 | My tokens (per-token identity) |
| 4–7 | Opponent tokens (per-token identity, single primary opponent in 2P mode) |
| 8 | Safe zones (0.5) |
| 9 | My home path |
| 10 | Opponent home path (skip inactive) |
| 11 | My locked % (broadcast) |
| 12 | Opp locked % (broadcast) |
| 13 | Dice roll, single broadcast plane (`roll / 6.0`) |

**What changed vs V6:** Drops the 6-plane dice one-hot for a single
broadcast plane. Drops the opponent density plane (kept per-token
identity only). Drops the score-diff broadcast. Smaller backbone (80ch
× 5 blocks vs 128ch × 10).

This was the opposite direction from V6.3 (slimming, not expanding).
The V10 line eventually went back to expanding.

---
