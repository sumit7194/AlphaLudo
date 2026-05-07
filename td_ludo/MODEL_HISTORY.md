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

V12.2 was the production model through April 2026. It produced two
direct descendants via SL distillation (V13 line, V14_scalar) that
later matched and slightly exceeded V12.2 in head-to-head play —
see entries below.

---

## V13 — `MinimalCNN14` (14ch raw input, distilled from V12.2)

First architecture in the project that **drops every engineered feature**
in the encoder. Raw token positions + dice only. The hypothesis being
tested: can the network *learn* danger / capture / idle / streak
relationships from raw input + a competent teacher policy, instead of
having them hand-computed and broadcast?

- **Input:** `(14, 15, 15)` — 4 own-token planes (per-token identity)
  + 4 opp-token planes (per-token identity, single primary opponent)
  + 6 dice one-hot. **No** safe-zone map, no danger map, no capture
  map, no idle counter, no streak, no score-diff broadcast — none of
  V11's engineered channels.
- **Backbone:** Pure CNN. **10 ResBlocks × 128 channels**, no
  attention. ~3M params.
- **Heads:** policy (per-token spatial extraction → Linear → 1 logit
  per own token), `win_prob` (GAP → sigmoid), `moves_remaining`
  (GAP → softplus). Same 3-head shape as V10/V11/V12.
- **File:** `experiments/distillation_14ch/model_14ch.py`.

**Training recipe (became the V13.x template):**
1. **SL distillation from V12.2:** student trained on `(state, V12.2.policy, V12.2.win_prob, V12.2.moves)` triples generated by V12.2 self-play. Loss = `α_p · KL(student.π || teacher.π) + α_v · MSE(student.V, teacher.V) + α_m · SmoothL1(...)`. ~10M states, batch 1024, lr 1e-3 → 1e-4 cosine.
2. **RL on top of distilled weights** via `train_v12.py --resume --model-arch v13_minimal`. Standard PPO + curriculum, but using the V13 raw encoder instead of V11 33ch.

**Result:** SL distillation hit **79.5% vs heuristic-bot mix**, near
parity with V12.2's eval. RL initially degraded (G=35K dropped to ~43%
greedy strength). Diagnosis pointed to two issues: (a) encoder
symmetry bug — see `encoder_symmetry_bug_discovery.md` — and (b)
value-head drift under PPO without aux-trajectory training.

V13 was the first proof that the **input gap** (V12.2's 33ch hand-
computed features) wasn't load-bearing for SL distillation. The
plateau is in the recipe, not in the encoder.

---

## V13.1 — `MinimalCNN14Aux` (14ch + 2 static aux heads, 12×160)

Iteration on V13 that addressed two findings from V13's mech-interp:
(a) the network needed to internally reconstruct static board layout
(safe cells, home paths) from positions alone, which wasted backbone
capacity, and (b) V13 was modestly under-parameterised at 10×128.

- **Input:** same `(14, 15, 15)` as V13. No new input channels.
- **Backbone:** **12 ResBlocks × 160 channels**, ~5.6M params. Bigger
  than V13's 3M.
- **Heads:** policy + win_prob + moves (same as V13) PLUS two
  **auxiliary 1×1-conv prediction heads** trained to reproduce static
  V11 channels:
  - `aux_safe_conv`: predict the safe-cells map (V11 channel 5).
  - `aux_home_path_conv`: predict the home-path map (V11 channel 6).
  - The aux loss forces the backbone to encode these static layouts,
    which the CNN otherwise has to learn implicitly.
- **File:** `td_ludo/td_ludo/models/v13_1.py`.

**Aux head choice rationale:** earlier versions of V13.1 had four aux
heads (safe / danger / capture / home). Danger and capture were
**dropped** because they're state-dependent (sparse positive targets),
which gave the BCE-trained heads degenerate near-zero loss from start.
Only the two STATIC features (safe, home_path) survived.

**Result:** SL distillation reached eval band 78-84%, peak 84.5%. RL
was launched but later abandoned in favour of V13.2's input-side
approach (give the network the static features as INPUT, not as a
prediction target). V13.1 is preserved as a backup line — the bigger
backbone (12×160) may still be useful if V13.2 plateaus.

Mech-interp on the trained V13.1 (channel-activation + layer-knockout):
**all 160 channels active in every block** (zero dead channels), but
**blocks 8-11 are knockout-tolerant** (KL divergence < 0.03 per block
when removed). Suggests V13.1 has spare depth → 10 layers would
be enough, which informed V13.2's choice.

---

## V13.2 — `MinimalCNN14` with 17ch input (14 raw + 3 V11 static)

The synthesis: **V13's clean architecture + V13.1's static-layout
intuition, but provide the static features as input instead of as
aux training targets.**

- **Input:** `(17, 15, 15)` = V14_minimal 14ch (own + opp tokens +
  dice one-hot) **+ 3 static V11 channels** (safe-cells, my-home-path,
  opp-home-path). The 3 static channels are constant per-current-
  player (post-encoder-fix) and cached at module import. Encoded by
  `td_ludo/game/encoder_v17.py::encode_state_v17`.
- **Backbone:** **10 ResBlocks × 128 channels**, ~3M params. Same shape
  as V13. Smaller than V13.1.
- **Heads:** standard 3-head (policy + win_prob + moves). No aux heads.
- **Files:** `td_ludo/game/encoder_v17.py`,
  `experiments/distillation_14ch/model_14ch.py` (reused).

Critical insight: every dynamic strategic feature V12.2 had (danger
map, capture map, idle counter, streak, score-diff broadcast,
leader progress) is now derived implicitly by V13.2's CNN from raw
positions + dice. Only static features (safe cells, home paths) are
provided as input — and even those because they don't change per
state, so making the backbone re-derive them every forward pass is
purely wasted compute.

**Training recipe:**
- SL: `train_v132_sl.py` distills V12.2-bias → V13.2 student over
  10M states. Reaches 80-85% SL eval band.
- RL: `train_v12.py --resume --model-arch v132 --num-res-blocks 10
  --num-channels 128`. Curriculum gating (auto-swap from `v122` mix
  to `v122_hist_v2` when 3 consecutive evals ≥ 80%). Bias penalties
  active (`LUDO_BIAS_PENALTIES=1`).
- **Bias penalty bumps (2026-05-04/05):** P_LAGGARD_PER_CELL
  0.0005 → 0.0025 (5×); P_DANGER_ADVANCED_BASE 0.04 → 0.12 (3×).
  Earlier values were too small to overcome the value-head's
  preference for scoring tokens / leaving safe cells under
  bonus-turn discount.
- **Pipeline:** `run_v132_pipeline.sh` runs SL then auto-resumes RL
  from `model_sl.pt` via `--resume` flag.

**Result (current):** RL training reached **best eval 83.8%**, eval band
80-83%. Direct head-to-head vs V12.2-bias (10K games, greedy):
**V13.2 wins ~52.4% (95% CI 51.1-53.7%)**. Statistically significant
+17 Elo gap over V12.2-bias. *V13.2 is the new strongest model in
the codebase.*

The result is more striking than the eval-WR comparison suggests —
bot evals are saturated for both models, but H2H has no ceiling. The
~2.4pp edge in H2H is real signal, not noise. V13.2 with raw input
+ deeper CNN + small bias-penalty curriculum beats V12.2's hand-
engineered 33-channel encoder.

---

## V14_scalar — `V14ScalarDeepSets` (no CNN, no attention)

The bookend experiment to V13.2's "minimal input is enough" finding.
Tests whether **spatial CNN structure is necessary**, or whether the
information content is the only load-bearing component.

- **Input:** **non-spatial scalar dict** with the same information
  content as V12.2's 33ch encoder, repackaged per-token + globally:
  - Per-own-token (×4): position (int → embedding 0-58),
    in_danger (bool), can_capture (bool, dice-conditional),
    can_score (bool, dice-conditional), can_land_safe (bool,
    dice-conditional), is_safe (bool), at_base (bool), at_home (bool),
    idle_count (float).
  - Per-opp-token (×4): position, in_my_danger, threatens_me,
    is_safe, at_base, at_home.
  - Globals (×13): dice one-hot (6) + same-token-streak (1) +
    my_locked_frac + opp_locked_frac + score_diff + leader_progress
    + non_home_tokens_frac + bonus_turn_flag.
  - Total: 8+4 own per-token + 5+4 opp per-token + 13 globals.
  - Encoder: `td_ludo/game/encoder_v14_scalar.py` (C++ side at
    `td_ludo/src/game.cpp::write_state_v14_scalar`).
- **Backbone:** **DeepSets-style MLP architecture**, NO CNN, NO
  attention.
  - Per-own-token MLP (shared across 4 tokens): `40 → 64 → 64`.
  - Per-opp-token MLP (shared, same shape).
  - Global MLP: `13 → 64 → 64`.
  - Pooling: **sum + max** over each set (own / opp), permutation-
    invariant (DeepSets / Zaheer 2017).
  - Trunk: `320 → 256 → 256`.
  - Per-token policy head (shared, leave-one-out context):
    `448 → 64 → 1` per own token → softmax over 4.
  - Value head: `256 → 64 → 1` → sigmoid.
- **Total params:** ~226K. **6× smaller than V12.2 (1.36M), 13× smaller
  than V13.1 (5.6M).**
- **File:** `td_ludo/td_ludo/models/v14_scalar.py`.

Architectural design choices:
- **Position embedding** instead of one-hot: `nn.Embedding(60, 32)`,
  shared between own and opp tokens (positions are absolute board
  cells with same semantics).
- **DeepSets pooling** instead of attention: provably as expressive
  as attention for set inputs (universal approximator), but no
  learned attention weights. Sum + max captures "most-advanced
  token" / "stack count" / "any token in danger" / etc cleanly.
- **Leave-one-out per-token policy**: when scoring move-quality of
  token t, the policy head sees t's embedding + pooled context of
  the OTHER 3 own tokens + opp pool + globals + trunk. Avoids the
  "sum includes self" double-counting.
- **Dual-interface forward**: model accepts EITHER the dict batch
  (training/SL path) OR a flat `(B, 73, 1, 1)` tensor (RL pipeline
  path, so the existing `(C, H, W)` PPO trainer works unchanged).
  Encoder helper `encode_state_v14_scalar_flat` produces the flat
  form; `_unpack_flat` inside the model unpacks it back into the
  dict.

**Training recipe:**
- SL: `train_v14_scalar_sl.py` distills V12.2-bias → V14_scalar over
  10M states. Reaches 80-85% SL eval band — **same as V13.2**.
- RL: `train_v12.py --resume --model-arch v14_scalar`. Same
  curriculum + bias penalties as V13.2.
- **Pipeline:** `run_v14_scalar_pipeline.sh` (SL → RL auto-handoff).

**Result (current):** SL eval 80-85%, RL best eval 80%. In the 3-way
H2H tournament (10K per pair, greedy, seat-balanced), V14_scalar
sits competitive but slightly behind V13.2 / V12.2. Both V13.2 and
V14_scalar — three architectures, one encoder family, two with
spatial CNNs and one with pure MLPs — converge on the **same
~80-83% plateau**.

**The implication is clear:** the plateau is in the **task + reward
signal + opponent ladder**, not the architecture. Three independent
architectures (V12.2 attn-CNN, V13.2 deep-CNN, V14_scalar DeepSets)
arrive at the same ceiling because all are expressive enough to absorb
V12.2's policy and gain a small amount on top via RL. The architecture
isn't the bottleneck.

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
V11+   → 33ch (V10 + per-token idle + streak — V11/V12.x family)
V13 (= v14_minimal) → 14ch raw (4 own + 4 opp + 6 dice one-hot, NO derived features — Exp 25 distillation, pure CNN)
V13.1  → 14ch input + 2 static aux output heads (12×160 backbone)
V13.2  → 17ch (V13's 14ch + 3 V11 static channels: safe/my-home/opp-home)
V13.3 / V13.4 → 14ch (encoder_v17, V13 base) + K=8 turn history fed to a transformer
V13.5  → 13ch (encoder_v18 count-based, token-permutation-symmetric) + rank-indexed output, pure CNN
V14_scalar → non-spatial scalar dict (per-token + globals, ~73 floats)
```

## CNN vs hybrid timeline

```
V1, V1.5, V3-V6.3      pure CNN
V7                     pure 1D transformer
V8                     V5 CNN + temporal transformer (multi-turn)
V9                     slim CNN + temporal transformer
V10                    pure CNN, 3-head output
V11                    CNN + token-attention transformer (single turn)
V12, V12.1, V12.2      CNN + token-attention transformer, per-token policy
V13, V13.1, V13.2      pure CNN, 3-head — back to no attention
V14_scalar             no CNN at all — DeepSets MLP over scalar features
```

**The V12.2 → V13.x → V14_scalar arc:**
The progression went *richer* (V12.2's 33-channel engineered encoder +
attention), then *deliberately minimal* (V13's 14ch raw input + pure
CNN), then *minimal-but-with-static-context* (V13.2's 17ch), then
*architecture-free* (V14_scalar's pure MLP DeepSets). Each step
removed assumptions about what the model needed. Each step still
hit the same ~80-83% plateau. Final empirical conclusion: V12.2's
hand-engineered features and V13's spatial CNN structure were both
*sufficient but not necessary*. The plateau is in the supervised
signal (V12.2 teacher) + the bias-penalty reward shaping +
the opponent ladder, not the architecture.

---

## Post-V13.2 plateau-break attempts (2026-05-05 → 06)

After the 3-way tournament confirmed V13.2 as strongest (V13.2 53.1% >
V12.2 50.2% > V14_scalar 46.6%), four attempts were made to break the
~80-83% bot-eval / ±5pp H2H plateau.

### MCTS Step 1 — search-distillation from V13.2 (Python rewrite)

**Goal.** AlphaZero-lite: generate ~1M states with 2-ply expectimax
search using V13.2 as leaf evaluator, distill `(π_search, V_search,
outcome)` targets into a fresh V13.2-arch student.

**What we built:** `experiments/mcts_v1/{generate_search_data, train_search_distill}.py`
plus a 6-test unit-test suite that catches the two bugs that broke the
first attempt (state aliasing in the dice loop, bonus-turn sign error
in Q-aggregation). Atomic versioned save (.prev.npz failsafe) added
after a 11.7M-state buffer was corrupted by VM network failure.

**Result.** Step1_Distill loses **89.6% / 10.4%** to V13.2_latest in
H2H (25K games, capped early once verdict was clear). Bug fixes moved
the result from v1 (92.2/7.8) only to v2 (89.6/10.4) — the bugs were
real but not the dominant problem. **2-ply expectimax over the same
teacher cannot meaningfully improve the targets.**

Conclusion: shelved until we have a stronger leaf evaluator than V13.2.

### V13.3 — temporal transformer (mini, 418K params)

**Goal.** Test whether the model can use opponent-pattern signal across
K=8 past decision states. Architecture: per-frame CNN (4 res-blocks ×
64ch) + 2-layer transformer (d=64, h=4, ffn=256) over K=8 V17 frames.

**SL distillation result.** 5M states, V13.2 teacher. Final eval 82%
vs random heuristic-bot mix — exactly V13.2's band.

**RL result (vanilla self-play REINFORCE, v1).** Catastrophic policy
collapse: 82% → 31.5% → 30.0% over 600K states. Vanilla REINFORCE in
symmetric self-play has near-zero gradient signal at the population
level (advantage averages to zero across mirrored agents), and without
strong regularization the policy random-walks into garbage.

**RL v2 with patches** (KL anchor 0.1 to V13.2, multi-legal-move filter
on policy/entropy losses, LR 5e-5): drift slowed but did not stop —
82% → 77% → 70% over 300K states. Killed.

**4-way H2H tournament** (V13.2_latest vs V13.3_SL vs V13.3_RL_v2 vs
Step1_Distill, 500 games per pair, mirrored seeds, greedy):

| Rank | Model | Wins/Games | WR |
|---|---|---|---|
| 1 | V13.2_latest | 999/1500 | **66.6%** |
| 2 | V13.3_SL_82pct | 921/1500 | 61.4% |
| 3 | V13.3_RL_v2_DEGRADED | 905/1500 | 60.3% |
| 4 | Step1_Distill | 175/1500 | 11.7% |

Key findings:
- **V13.3 mini lost 43.4% / 56.6% to V13.2** — temporal arch at this
  size does not match teacher.
- **V13.3 RL "degradation" was a bot-eval illusion** — RL v2 was
  statistically tied with SL in H2H (52/48). Methodology lesson:
  bot-eval gets noisier as policy moves; trust H2H.
- The H2H gap to V13.2 has two confounders: 7× capacity gap AND a
  train/test history mismatch (see V13.4 below).

### V13.4 — temporal transformer at V13.2-comparable scale + bug fix

**Architecture.** Per-frame CNN 10×128 (matches V13.2 trunk) + 4-layer
transformer d=128, h=4, ffn=512. **3.79M params** (vs V13.2's 3.0M).
Same `V133Temporal` model class as V13.3, just bigger constructor args.

**Critical bug discovered & fixed.** The V13.3 SL/RL envs maintained ONE
deque per game and pushed every decision state regardless of which
player was to move — so the transformer's K=8 history was a mixed
sequence of own + opponent decision-state frames. But the encoder
(`encode_state_v17`) rotates the board to `current_player`'s POV, and
the inference-time agent in H2H only observes its own turns. Result:
**train/test distribution mismatch** — model trained on alternating-POV
sequences, asked to perform on own-POV-only sequences.

Fix (Option B, "per-player history"):
```python
# Before:
self.history = [collections.deque(maxlen=K) for _ in range(B)]
# After:
self.history = [{0: deque(maxlen=K), 2: deque(maxlen=K)} for _ in range(B)]
```
At a player p decision, push to `history[i][p]` and read from
`history[i][p]`. Inference matches automatically (each agent had its
own deque). Verified by **86 unit tests** in `experiments/v134/test_per_player_history.py`
covering init, push correctness, opp-leak isolation, reset, RL
trajectory snapshots, and encoder POV-pivot precondition.

**SL distillation** (10M states, V13.2 teacher, batch=256, lr 1e-3 → 1e-4,
~10 hrs on L4 GPU). Eval history:

| States | Eval WR |
|---|---|
| 1M | 53.5% |
| 4M | 81.5% |
| 6M | 82.5% (peak) |
| 9M | 81.5% |
| 10M (final) | — |

Lands at the same **80-82% bot-eval ceiling** as V13.2, V13.3-mini, and
V14_scalar. Bot eval cannot distinguish them — only H2H can.

**RL phase (chain-1).** Same recipe as V13.3 RL v2 (KL anchor 0.1,
multi-legal filter, lr 5e-5 → 5e-6, entropy 0.02). 1.5M states, 9,375
games. Bot-eval trajectory: 80→86.5→81→83→79.5→84→83 (peaks above the
80-82% SL band, but 200-game evals carry ±2-3pp SE so peaks are within
noise of mean).

**Chain-1 H2H (2026-05-07 11:24 UTC, 500 games per pair, mirrored seeds):**

| Rank | Model | WR (1000 games each) |
|---|---|---|
| 1 | V13.4_SL | 50.6% |
| 2 | V13.2_latest | 50.2% |
| 3 | V13.4_RL | 49.2% |

All three **statistically tied** (max delta 1.4pp, z=0.6). The bot-eval
peaks were variance, not real H2H lift. Confirms bot-eval is teacher-
bound and only H2H is the gate.

**RL was undersized.** 9.4K games is barely warm-up vs V14_scalar's 700K
or V12.2's 485K. Concluding "RL doesn't help" from this would be unfair.

**RL continuation (in progress, launched 2026-05-07 ~11:50 UTC).** Resumed
from `checkpoints/v134/model_latest.pt` (chain-1 end state). LR cosine
reset 5e-5 → 5e-6 over 100M states (effectively unbounded). Eval cadence
now game-based: every 20K games × 3K games per eval (±0.5pp SE — clean
signal). Other hyperparams unchanged. As of 2026-05-07 ~16:25 UTC: 2.01M
states / 12,678 games / fps 122 / ent 0.38 / kl 0.07 — healthy and
exploring. First eval at G=20K pending. Dashboard at `http://<vm-ip>:8792/`
(firewall whitelists user IP).

### V13.5 — symmetric encoder + symmetric output (POC 2026-05-07)

**Hypothesis.** The 4 own / 4 opp tokens are permutation-symmetric under
Ludo's rules, but per-token-channel encoders force the model to *learn*
this symmetry from data. Capture-and-return events break the empirical
"token-0 most advanced" correlation that the model ends up relying on.
Build the symmetry into the architecture instead.

**Two changes vs V13.2** (orthogonal axis to V13.4's temporal):
1. **Encoder V18 (13ch, count-based).** ch0/1: own/opp token-count per
   cell (sum of V14 ch0..3, with home cells zeroed since V14 leaks
   token-id via per-token home cells); ch2/3: own/opp at-home count
   (broadcast scalar /4); ch4..9: dice (unchanged); ch10..12: V11 statics
   (unchanged). 19/19 invariance unit tests pass under all 24 own- and
   opp-token permutations.
2. **Rank-indexed output.** Sort own-token positions descending (most-
   advanced = rank-0), output 4 logits per rank. Aggregate V13.2's per-
   token policy → per-rank target via summation. At play time, map
   chosen rank back to a legal token-ID.

No transformer, no temporal history. Pure CNN trunk in V13.2 style.

**POC (Mac MPS, 2026-05-07).** 6 ResBlocks × 96 ch = 1.03M params (1/3
V13.2). 2M states SL, V13.2 teacher, lr 1e-3 → 1e-4 cosine, batch 256,
~45 min wall. Two ablations:

| Variant | perm aug | Bot peak | H2H vs V13.2 (500 games) | z |
|---|---|---|---|---|
| Run A | ON | 82% | 47.0% | 1.34 |
| Run B | OFF | 85% | **47.8%** | **0.98 (tied)** |

**Findings.**
1. V13.5 at 1/3 V13.2's capacity statistically ties V13.2 in H2H
   (z=0.98). For comparison, V13.3 mini at 1/7 V13.2's capacity lost by
   13pp. **Per-param, the symmetric encoder is meaningfully more efficient.**
2. Permutation augmentation hurt H2H by 1.6pp. V13.2's residual token-id
   biases ("advance token-0 first") empirically align with good
   heuristics ("advance most-advanced first") because of training-time
   correlation. Random-permutation aug washes out useful bias along
   with the noise. **Default OFF for future runs.**
3. The 4-5pp residual gap at 1/3 capacity is plausibly capacity-bound,
   not a symmetry-doesn't-work signal.

**Pending (queued).** Full-size V13.5 SL on Mac MPS (in parallel with
V13.4 RL on VM): 10×128 ≈ 3M params (V13.2-matched), 5-10M states, perm
aug OFF. H2H gate after SL completes. If tied/winning, V13.5 + RL on VM
becomes the canonical follow-up.

**Files.** `td_ludo/game/encoder_v18_symmetric.py`, `td_ludo/game/rank_mapping.py`,
`td_ludo/models/v13_5.py`, `experiments/v135/test_v135_symmetry.py`,
`train_v135_sl.py`. Backups planned at `checkpoint_backups/v135_with_perm_*`
and `checkpoint_backups/v135_no_perm_*`.

### V14_scalar — RL run paused

The V14_scalar RL run resumed on Mac MPS (game 485,910 → 700,000). Eval
peaked at 80%, slowly drifted to 76.6%, ELO 1410 (down from peak ~1650).
Entropy collapsed to ~0 mid-run. Paused 2026-05-07 08:12 UTC, weights
backed up to `checkpoint_backups/v14_scalar_rl_20260507_081221/`.

### Open methodological learning

1. **Bot-eval ceilings around 80-83% across all architectures** — this
   metric saturates and cannot distinguish stronger models. H2H at 1K-25K
   games is the right tool above that band.
2. **SL distillation is teacher-bound.** A student trained on V13.2's
   outputs cannot exceed V13.2 in expectation, regardless of architecture.
   To break the plateau, RL must contribute beyond the SL initialization.
3. **Vanilla REINFORCE in symmetric self-play is too weak** to push the
   policy past teacher; needs either KL-anchored regularization (which
   slows drift but also slows improvement) or fundamentally different
   gradient signal (search-improved targets, league play, exploiter
   opponents).
4. **Train/test distribution mismatches in temporal models can silently
   cost percentage points.** The V13.3 mini's 43.4% H2H against V13.2
   was partly attributable to the per-player history bug, not just
   capacity. Always verify inference protocol matches training protocol
   for any stateful architecture.
