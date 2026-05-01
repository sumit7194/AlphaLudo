# V13 Investigation — State of Knowledge

## Project goal (user's framing)

Build a god-level Ludo player. Specifically: a model where, when the user plays against it, they cannot find a single bad move or weakness over any number of games. **No shipping pressure.** This is a learning project.

The strategic question for V13: **can a "minimal architecture" model (raw position + dice input only) match or surpass V12.2 ("glorified bot" with engineered features)?**

---

## Models in scope

| Model | Channels | Architecture | Params | Status |
|---|---|---|---|---|
| V12.2 | 33 (V11 encoder) | 3 ResBlocks × 128ch + 2 attn × 4 heads | 1.36M | Trained 1.14M games. Plateaued ~81-83%. Best 83.1%. |
| V13 SL (Distill14) | 14 (raw) | 10 ResBlocks × 128ch, pure CNN | 3M | SL-distilled from V12.2. Hits 79.5% vs bots. Parity with V12.2 in H2H. |
| V13 G=35K | 14 | same as SL | 3M | RL-degraded. Lost 22-33pp greedy strength. Bleeding ~3pp per 10K games. |

---

## Bot-sweep ground truth (500 games per bot, greedy, seat-balanced)

| Bot | V12.2 | Distill14 (SL frozen) | V13 G=35K (RL) |
|---|---|---|---|
| Expert | 77.8% | 76.2% | 44.6% |
| Heuristic | 80.6% | 74.2% | 41.8% |
| Aggressive | 79.0% | 77.8% | 44.6% |
| Defensive | 81.8% | 76.0% | 42.0% |
| Racing | 76.4% | 79.6% | 43.8% |
| Random | 93.4% | 93.0% | 79.8% |
| **Avg vs deterministic-5** | **79.1%** | **76.8%** | **43.4%** |

**Key facts:**
- V12.2 ≈ Distill14 within 2.3pp → SL distillation transferred ~97% of V12.2's strength to 14ch.
- V13 RL drops 33pp from SL baseline — RL is actively destroying the model, not slow-training.
- V13 still beats Random — basic positional play preserved; only strategic complexity collapses.

---

## Mech-interp findings on V12.2

### Channel ablation — global (random states) appeared to show engineered channels dead

| Channel | Global KL |
|---|---|
| Ch 21 Danger Map | 0.015 |
| Ch 22 Capture Opp Map | 0.007 |
| Ch 25 Two-Roll Capture | 0.002 |
| Ch 28-31 Idle counters | 0.003-0.006 |
| Ch 32 Streak | 0.003 |

This LOOKED like V12.2 ignoring the engineered features.

### Conditional ablation — completely flips the story

When restricted to states where the channel's signal is actually relevant:

| Channel | Global KL | Conditional KL (when relevant) | Bucket | Multiplier |
|---|---|---|---|---|
| **Ch 22 CaptureOpp** | 0.007 | **1.675** | `capture_available` | **240×** |
| **Ch 21 DangerMap** | 0.015 | **0.438** | `leading_token_in_danger` | **29×** |
| Ch 22 CaptureOpp | 0.007 | 0.187 | `capture_roll_3_only` | 27× |
| Ch 21 DangerMap | 0.015 | 0.149 | `capture_roll_3_only` | 10× |

**When captures are available, Ch 22 KL = 1.67 — higher than ANY token channel.** V12.2 reads "what to do" directly off the engineered map. This confirms the **glorified-bot hypothesis**: the bot logic is encoded into the input, the conv stack just confirms.

### Static geometry channels (Ch 5/6/7) — modest conditional effect

| Channel | KL when relevant | KL when irrelevant | Ratio |
|---|---|---|---|
| Ch 5 SafeZones | 0.0059 | 0.0014 | 4.3× |
| Ch 6 MyHomePath | 0.0332 | 0.0119 | 2.8× |
| Ch 7 OppHomePath | 0.0162 | 0.0056 | 2.9× |

Static geometry channels DO show higher KL when relevant, but **30-300× less impactful than dynamic Ch 21/22 even conditioned**. They're memorizable; the dynamic channels are computed inferences.

### Linear probes (V12.2 internal representations)

| Concept | Probe accuracy | Baseline | Verdict |
|---|---|---|---|
| can_capture_this_turn | 100% | 96.6% | ✅ derives perfectly |
| game_phase | 99.6% | 33% | ✅ |
| num_tokens_out | 98.8% | 33% | ✅ |
| home_stretch_count | 98.4% | 81% | ✅ (better on real states) |
| **leading_token_in_danger** | **97.2%** | **97.1%** | ❌ tied with baseline (danger blindness) |
| **closest_token_to_home** | 60.4% | 27.7% | ⚠️ moderate, not great |
| eventual_win | 67.8% | 51% | ⚠️ value head moderate at best |

**Persistent V12 defect:** danger detection at baseline. Probe can't beat majority class. Not fixed by adding Ch 21 graded danger.

### Layer redundancy (CKA similarity)

| Pair | Global CKA | Verdict |
|---|---|---|
| Stem → Block 0 | 0.922 | moderate |
| Block 0 → Block 1 | 0.949 | borderline redundant |
| **Block 1 → Block 2** | **0.976** | **REDUNDANT** (>0.95) |

**Per-phase:** in early and late game, ALL pairs >0.95 — V12.2's 3 res blocks act as ~1 functional layer for most game phases.

### Layer knockout

Knocking out any single block drops baseline WR (100% vs Random) by only 0.8-2.0pp. **V12.2 is over-parameterized in depth.**

### Channel activation

0 globally dead, 0 low-activity. All 384 conv channels (128 × 3 blocks) are alive — but redundantly computing similar features.

### Dice sensitivity (vs V12 broken, vs V13)

| Metric | V12 broken | V12.2 | V13 |
|---|---|---|---|
| `flip_any_roll` | 146 | **552** | 591 |
| `js_pairwise_mean` | 0.137 | 0.138 | 0.174 |

V12.2 is 3.7× more dice-responsive than V12 (genuine improvement). V13 is even more dice-responsive — likely over-relying on dice signal.

---

## Mech-interp findings on V13

### V13_SL behavioral test — token preference

Pick rate per slot when legal (3000 multi-legal random states):

| Model | T0 | T1 | T2 | T3 |
|---|---|---|---|---|
| V12.2 (calibrated baseline) | 30.6% | 36.8% | 35.4% | 35.1% |
| V13_SL (Distill14) | 32.7% | 30.1% | 32.0% | **43.1%** |
| V13_G35K (RL) | 32.4% | 27.7% | 32.8% | **45.1%** |

**Δ from V12.2 baseline:**

| Model | T0 | T1 | T2 | T3 |
|---|---|---|---|---|
| V13_SL | +2.1 | −6.7 | −3.3 | **+8.0** |
| V13_G35K | +1.8 | −9.1 | −2.5 | **+9.9** |

**Key facts:**
- T3 over-preference is REAL but **modest** (~10pp).
- Bias is **inherited from architecture** — present in V13_SL despite SL training to mimic V12.2's balanced policy.
- RL only mildly amplifies it (+8 → +10pp).
- The T3 over-pick mirrors a T1 under-pick. T0/T2 close to baseline.
- ~10pp slot bias **cannot fully explain** the 22pp greedy WR drop.

### V13 channel ablation (own-token channels)

| Channel | V13_SL global | V13_G35K global |
|---|---|---|
| Ch 0 (MyT0) | 0.014 | 0.027 |
| Ch 1 (MyT1) | 0.022 | 0.033 |
| Ch 2 (MyT2) | 0.019 | 0.032 |
| Ch 3 (MyT3) | 0.027 | 0.045 |

Ch 3 highest in both, RL roughly doubles all values uniformly.

---

## Channels V13 does/doesn't need (categorized)

### Tier 1 — V13 must derive from raw input
| Channel | Conditional KL | Notes |
|---|---|---|
| Ch 22 CaptureOpp | 1.67 in capture states | The single biggest signal V12.2 uses |
| Ch 21 DangerMap | 0.44 in danger states | Defensive role |

**Distill14 SL (79% vs bots) proves V13's conv stack CAN derive these.** Question: why does RL destroy the derivation?

### Tier 2 — V13 must memorize as conv weights
| Channel | What |
|---|---|
| Ch 5 SafeZones | 8 fixed positions |
| Ch 6 MyHomePath | 5 cells, player-rotated |
| Ch 7 OppHomePath | same for opp |

Static board geometry. Distill14 SL learned them. Stable under RL because constants.

### Tier 3 — V13 derives trivially
Ch 4 (opp density), Ch 9-10 (locked %), Ch 24 (bonus turn = dice==6), Ch 26 (non-home frac), Ch 27 (leader progress). All 1-layer scalar reductions.

### Tier 4 — Genuinely doesn't need
| Channel | KL | Why |
|---|---|---|
| Ch 8 ScoreDiff | 0.000 | Even V12.2 ignores it |
| Ch 23 SafeLanding | low | Redundant with Ch 21 |
| Ch 25 TwoRollCapture | low | V12.2 doesn't use it |

### Tier 5 — Architecturally impossible without history
Ch 28-31 (idle per token), Ch 32 (streak). Need stacked frames or recurrent state. V12.2 doesn't really use these either (KL ≤ 0.005 globally; no conditional bucket tested).

---

## Why V13 RL collapses (current best theory)

PPO at LR=5e-5 still has:
- `clip_fraction = 0.30` (5× normal)
- `approx_kl = 0.08` (5× normal)
- `avg_advantage = 0.000`

Hypothesis: **value head can't differentiate states from 14ch input** → noisy advantage estimates → PPO can't consistently reinforce the conv-stack chain that derives Ch 21/22 features → those derivations decay first → policy drifts toward simpler heuristics → 22pp greedy WR loss.

Note this is consistent with the channel ablation pattern: V13's per-channel KL values are uniformly small (0.01-0.05) — the network is using ALL its inputs in muddled combination rather than crisp dependencies.

---

## V13.1 proposal — current draft

**Two architectural changes, no engineered channels added** (keeps the minimal-architecture hypothesis honest):

| Change | Justification |
|---|---|
| Per-token shared MLP policy head (V12.2 surgery) | Mech-interp showed V13 has +8-10pp T3 over-pick that SL distillation couldn't flatten → architectural bias the per-token head should fix |
| Auxiliary loss: predict `can_capture` and `leading_token_in_danger` as side-tasks during RL | Forces conv stack to maintain Tier-1 derivations under PPO drift; doesn't change inputs |

Pipeline: SL re-distill from V12.2 → V13.1 with per-token head → RL with v122 mix + aux losses.

**V13.1 cleanly tests:** does RL drift come from (a) policy-head asymmetry, (b) dynamic-feature decay because nothing in the loss reinforces them, or (c) both?

Three outcome scenarios after 200K games:
- Trains stably to 80%+ → minimal architecture validated.
- Slower drift → value-head theory partially correct, need KL distillation as additional fix.
- Drifts identically → 14ch genuinely too lean for stable RL; rules out two specific theories with data.

**Cost:** ~1 day SL re-distill + 1-2 days RL.

---

## Open questions

1. **Idle/streak channels (28-32):** dead in V12.2 globally. No conditional bucket tested for "long-idle" states. If they would fire on rare same-token-streak states, V13 needs a memory architecture, not just a wider input.
2. **Two-roll capture (Ch 25):** similar — no conditional bucket. Likely V12.2 doesn't use it but we haven't proven it.
3. **V12.2 plateau at 83%:** if V13 distills from V12.2, it inherits this ceiling. Unrelated to V13's defects but bounds the experiment.
4. **Danger blindness (linear probe at baseline):** V12.2 still has it despite Ch 21 graded danger fix in V11. Not encoder-related — architectural.
5. **Value head miscalibration on real states:** V12.2's `eventual_win` probe drops from 78% (random states) to 68% (self-play states). The value head is least accurate exactly when it matters most.
