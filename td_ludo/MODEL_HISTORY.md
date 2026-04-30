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
