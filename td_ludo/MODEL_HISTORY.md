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
