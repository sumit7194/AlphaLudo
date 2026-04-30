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

## V10 — slim CNN, 28ch encoder, 3-head output (`td_ludo/td_ludo/models/v10.py`)

The V6.x line evolved further: drops V6.3's dead `consecutive_sixes`
channel, promotes the `two_roll_capture_map` into the main slot, and
adds two new "macro-state" broadcasts.

- **Input:** `(28, 15, 15)`.
- **Backbone:** `AlphaLudoV10` — 6 ResBlocks × 96 channels (~640K
  params).
- **Heads:** **three heads** — policy (4 logits), `win_prob` (sigmoid
  scalar trained with BCE), and `moves_remaining` (softplus scalar
  trained with SmoothL1).

**Input channels (`write_state_tensor_v10`):**

| Ch | Encodes |
|---|---|
| 0–23 | Identical to V6.1 |
| 24 | `bonus_turn_flag` (was V6.3 ch24) |
| 25 | `two_roll_capture_map` (was V6.3 ch26 — promoted; V6.3 ch25 `consecutive_sixes` was dropped because mech-interp said it was unused) |
| 26 | `non_home_tokens_frac` — broadcast `(non-home own tokens) / 4`, in {0, 0.25, 0.5, 0.75, 1.0}. 0.25 = "forced mode" (only one own token left to score) |
| 27 | `my_leader_progress` — broadcast `most_advanced_own_token / 56`, in `[0, 1]`; 1.0 iff at least one token already home |

**What changed vs V6.3:** Kept the bonus-turn flag and two-roll capture
map; dropped the dead `consecutive_sixes` channel. Added two macro-
state scalars (how much work is left, how far ahead is your leader).
Three-head output replaces V5/V6's policy+value+aux pattern.

---

## V11 — `AlphaLudoV11` (CNN + Transformer, same 28ch initially)

V11 added a token-attention transformer on top of the V10 CNN.
Initially used the V10 28ch encoder.

- **Input:** `(28, 15, 15)` initially; later upgraded to `(33, 15, 15)`
  with the V11 encoder (see below).
- **Backbone:** 4 ResBlocks × 96 channels (V11 default; smaller than
  V10) → token-attention transformer (default 2 layers, 4 heads,
  ffn_ratio 4) → 3 heads.
- **Token attention:** picks 4 cells corresponding to own tokens,
  attends across them with full (4 + 4 board-summary) context.
- **File:** `td_ludo/td_ludo/models/v11.py`.

The V11 encoder upgrade (33ch) below was the channel-side change.

---

## V11 encoder = V12.x encoder (33ch, `write_state_tensor_v11`)

Added in V12.1 to fix the same-token-stickiness pattern observed in
eval-lens analysis: V12 picked the same token across consecutive turns
at ~60% rate (vs 33% baseline). Per-token idle counters and a streak
broadcast give the model an explicit "this token has been ignored" /
"you've moved the same token N times in a row" signal.

- **Input:** `(33, 15, 15)`.

**Input channels:**

| Ch | Encodes |
|---|---|
| 0–27 | Identical to V10 |
| 28 | Idle counter for own token 0 — `idle_counter[me][0] / 20`, capped at 1.0, broadcast |
| 29 | Idle counter for own token 1 (same scaling) |
| 30 | Idle counter for own token 2 |
| 31 | Idle counter for own token 3 |
| 32 | Same-token streak — `streak[me] / 10`, capped at 1.0, broadcast |

`idle_counter[p][t]` increments each time `p` plays a turn but doesn't
move token `t`; resets to 0 on the moved token. `streak[p]` counts
consecutive turns `p` moved the same token (resets to 1 on a different
token).

---

## V12 — Token-entity attention model (`td_ludo/td_ludo/models/v12.py`)

V12 cleaned up V11's attention layout — own tokens are looked up at
their actual board cells (not at fixed slots), and the policy head is
per-token (Linear over the attended `(B, 4, C)` tensor) rather than
pooled.

- **Input:** `(33, 15, 15)` (V11 encoder).
- **Backbone:** 4 ResBlocks × 96 channels (V12 default).
- **Token attention:** `concat(GAP(CNN), mean(post-attn tokens)) → 192-dim`,
  feeds three heads.
- **Heads:** policy (per-token via `Linear(C, 64) → Linear(64, 1)` then
  squeeze), `win_prob` (sigmoid), `moves_remaining` (softplus).
- **Params:** ~951K.

V12 broke the V11.1 79.05% single-eval ceiling, peaking at 81.0% but
plateaued.

---

## V12.1 — V12 + eval-lens-driven surgery (cancelled at G=10K)

Built from the V12 checkpoint via parameter-surgery: drop the
`token_idx_emb` (was driving slot bias), rebuild the policy head into
a per-token form. Same V11 encoder (33ch) input.

- **Input:** `(33, 15, 15)`.
- **Backbone:** 4 ResBlocks × 96 channels (same shape as V12).
- **Heads:** per-token policy via `Linear(C, 64) → Linear(64, 1)`.
- **Params:** ~945K.

Cancelled mid-RL at G=10K because the eval-lens evidence and CKA
findings pointed at architecture redundancy beyond what the surgery
could fix. Pivoted to V12.2.

---

## V12.2 — fresh wider+shallower (current production)

Fresh-init from a V12-trained teacher. CKA on V12 showed >0.95
similarity across all 4 ResBlocks → most depth was wasted. So V12.2
goes shallower but wider.

- **Input:** `(33, 15, 15)` (V11 encoder).
- **Backbone:** **3 ResBlocks × 128 channels** (vs V12's 4×96).
- **Token attention:** 2 layers × 4 heads, attention dim 128
  (matches CNN width — V12 had 96).
- **Heads:** same 3-head shape as V12, per-token policy.
- **Params:** ~1.36M (about 1.4× V12).

V12.2 broke the V12 ceiling (81.0% → 82.65% at G=40K) and met the
plateau-break gate (≥80% for 3 consecutive evals at G=40-50-60K) for
the first time in project history.

Also paired with new training elements:
- Token-permutation augmentation (own tokens relabelled in the
  encoder) during SL warm-up.
- More self-play in the RL mix: 75% self-play / 15% Expert / 5%
  Heuristic / 3% Aggressive / 2% Defensive (vs V10's 40/25/15/10/10).

V12.2 is the current production model. Ongoing experiments use it
unchanged as the architecture; recent direction tries
**search-during-training** as an auxiliary policy target (Exp 24, see
`training_journal.md`).

---

## MinimalCNN14 — distillation experiment (Exp 25, COMPLETE — hypothesis confirmed)

A deliberately-stripped student trained from V12.2 to test the
"input richness explains depth collapse" hypothesis (see Exp 25 in
`training_journal.md`). **Mech-interp results confirm the hypothesis
on two independent measures.**

- **Input:** `(14, 15, 15)` — minimal raw inputs only.
- **Backbone:** Pure CNN, 10 ResBlocks × 128 channels. **No attention.**
  Deep-and-wide deliberately, to see whether depth re-engages when the
  model has to derive features that V12.2's encoder hands over.
- **Heads:** 3-head (policy 4-logits + win_prob sigmoid + moves_remaining
  softplus) — matches V12.2 for direct distillation.
- **Files:** `td_ludo/experiments/distillation_14ch/model_14ch.py`
  (`MinimalCNN14`), `td_ludo/experiments/distillation_14ch/train_distillation.py`.

**Input channels (`write_state_tensor_v14_minimal`, NEW C++ encoder):**

| Ch | Encodes |
|---|---|
| 0–3 | My tokens (per-token identity) |
| 4–7 | Opponent tokens (per-token identity, single-opponent 2P mode) |
| 8–13 | Dice 6-plane one-hot |

**Critically absent vs V12.2's 33ch:**
- No safe zones, no home paths (CNN must learn board geometry).
- No danger map, no capture map, no two-roll-capture map (must derive
  threats from raw token positions + dice).
- No bonus-turn flag (must learn dice=6 → bonus-turn from training games).
- No idle counters, no streak (history-derived; **architecturally
  unrecoverable** in a stateless single-frame model).
- No score broadcasts, no leader progress, no locked-fraction.

**Hypothesis:** with rich V12.2 inputs, layers 8-10 are redundant
(CKA > 0.95). With minimal inputs, layers 1-7 should specialize and
diverge in CKA — proving the depth-collapse pattern is input-driven,
not task-intrinsic. Final pol_acc projected at 76-82% (below V12.2's
88.4% due to genuine information loss). The headline result is the
per-block CKA matrix, not the eval skill.

**Result (mech-interp, 2026-05-01):**

| Measure | V12.2 (3×128, 33ch) | MinimalCNN14 (10×128, 14ch) |
|---|---|---|
| CKA spread (max-min off-diag) | 0.057 | **0.348** (6× wider) |
| Pairs with CKA > 0.95 | 2 / 3 | 20 / 45 |
| in_danger probe AUC, blk0 → last | 0.847 → 0.824 (down) | 0.767 → 0.926 (**+0.16**) |
| leader_progress probe R², blk0 → last | 0.995 → 0.990 | 0.79 → 0.94 (**+0.15**) |
| can_capture probe AUC, blk0 → last | 0.983 → 0.985 | 0.923 → 0.964 (+0.04) |

V12.2's blocks all read concepts off the input near-perfectly at block
0; later blocks add nothing or slightly degrade. MinimalCNN14 builds
those same concepts up across depth — by block 9 it computes a sharper
binary danger signal than V12.2 reads off its pre-baked graded plane.

See `training_journal.md` Exp 25 (results) and the analysis scripts at
`td_ludo/experiments/distillation_14ch/{cka_analysis.py, probe_analysis.py}`.

---

## Encoder progression at a glance

```
V1     →  8ch  raw board + dice scalar + turn scalar
V1.5   → 18ch  (mastery branch — short-lived)
V3 td_v2  → 11ch (per-token identity, score diff, locked frac)
V3/V4/V5 → 17ch (added 6-plane dice one-hot)
V6.1   → 24ch (per-token opponent + danger/capture/safe-landing maps)
V6.3   → 27ch (added bonus-turn flag + consecutive-sixes + two-roll-capture)
V7     → 18-dim 1D (transformer, no spatial encoding)
V8     → 17ch + temporal transformer over K=16 past turns
V9     → 14ch slim re-design (drops dice one-hot, opp density)
V10    → 28ch (V6.1 + bonus-turn + two-roll-capture + non_home_frac + leader_progress; V6.3 ch25 dropped)
V11+   → 33ch (V10 + per-token idle + streak — current)
v14_minimal → 14ch (4 own + 4 opp + 6 dice one-hot, NO derived features — Exp 25 distillation)
```

## CNN vs hybrid timeline

```
V1, V1.5, V3-V6.3   pure CNN
V7                  pure 1D transformer
V8                  V5 CNN + temporal transformer (multi-turn)
V9                  slim CNN + temporal transformer
V10                 pure CNN, 3-head output
V11                 CNN + token-attention transformer (single turn)
V12, V12.1, V12.2   CNN + token-attention transformer, per-token policy
```

The current line (V12.2) is the synthesis: V10's 3-head training, V11's
token attention, V12's per-token policy, plus V12.2's shallower-but-
wider CNN informed by V12 mech-interp. The V11 33-channel encoder is
shared across the whole V11/V12 family.
