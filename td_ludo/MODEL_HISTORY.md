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
