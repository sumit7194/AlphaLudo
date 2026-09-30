# V13.6 Play-Test Flaws — Diagnosis & Proposed Fixes (2026-06-14)

Source: first manual human-vs-AI game vs the V13.6 champion in `play/server.py`.
Target is **B** (decision quality / game understanding), NOT win rate — these
flaws don't move the eval (variance ceiling) but are visible in play.

Two systematic flaws observed (each a *general* pattern, one example given):

---

## Flaw 1 — Bad use of a 6: spawns instead of chasing/capturing

**Observation:** With 3 tokens in comfortable positions and a 6 available to
chase/capture an opponent's finishing token, the model instead spawned its
last base token.

**Root causes (all confirmed in code; user's "baked into bots/data" instinct
is correct, plus a reward-design cause):**

1. **Baked-in lineage bias.** The journal documented a **92% unlock-on-6
   bias** in V12.2 self-play; the SL data descends from that lineage.
2. **The dense reward actively pays it to spawn.** `dense_rewards.py`:
   `REWARD_SPAWN = 0.05`, `REWARD_FORWARD_STEP = 0.005`. So advancing a token
   6 cells = **+0.03**, but spawning = **+0.05**. On a 6 with no *immediate*
   capture, the shaped reward **prefers spawn over advancing/chasing.** Since
   at the variance ceiling the dense reward is the dominant learning signal,
   this tilt is a real driver, not a footnote. The model does what we pay it.
3. **Under-exploration.** Low entropy on 6-states → it locks onto the higher
   immediate-reward action (spawn) and never discovers chase → capture (+0.20).

**Why the existing mitigation doesn't catch it:** `bias_penalties.py` penalty
#1 (unlock-with-better-available) is **phase-gated OFF in the opening**
(`move_count <= 16` → scale 0), exactly where the user saw it, and "better"
only counts *immediate* finish/capture/escape-danger — **not "chase"**
(closing distance to capture next turn).

**Proposed fixes (need a retrain):**
- (a) **dice-6-conditional exploration** — raise sampling temperature/entropy
  only when `dice == 6` so it tries chase/capture and learns from the +0.20.
- (b) **Remove the reward tilt** — make spawn not out-reward advancing (drop
  spawn ≤ 0.03, or only award spawn when no advancing token could capture).
- (c) Extend the unlock-on-6 penalty to cover "chase" and enable it earlier.

---

## Flaw 2 — Neglects the trailing token (perceived as "T3")

**Observation:** Even when the trailing token is unsafe and an opponent is
slowly closing the gap, the model keeps playing other tokens and leaves it
exposed. General pattern.

**KEY REFRAME — this is NOT a representation problem (user history confirms):**
The team previously suspected token-identity and built V15 (graph transformer,
fully permutation-invariant) to fix it. But V13.6 is **rank-indexed**
(`rank_mapping.py`: output indexed by canonical rank — most-advanced → least)
and is *already* permutation-invariant over token-IDs; it **cannot** prefer
"T3-the-ID." The neglect **persists across BOTH symmetric architectures** →
near-proof it was never a representation problem. What's actually happening is
**laggard-rank neglect** — the model under-values the *trailing* token's
safety (whichever ID) — and that is a **value/reward gap**. No architecture
change fixes it; V15's graph transformer attacked the wrong layer.

**Root causes (confirmed in code):**
1. **The danger penalty only protects ADVANCED tokens.** `bias_penalties.py`
   penalty #5 fires only when `chosen_pre > 35`. A trailing token at pos ~20
   with an opponent closing gets **zero** penalty signal — by design.
2. **Safety/danger rewards are near-noise** — 300-game decomposition put them
   at win-loss delta +0.009–0.024 (negligible gradient).
3. **Flat capture penalty.** Got-captured is a flat **−0.20** regardless of
   the token's progress, so losing a token at pos 40 hurts the same as at pos
   5 → no signal that protecting a more-advanced/at-risk token matters more.
4. **Outcome signal is dice-drowned** — losing a laggard rarely flips a noisy
   game, so the value head never learns it matters either.

**Proposed fixes (need a retrain):**
- (a) Make the danger penalty fire by **value-at-risk** (progress lost if
  captured) for **all** tokens, not just `pos > 35`.
- (b) Scale the got-captured penalty by the token's progress (losing pos-40
  ≈ 8× the pain of pos-5) — gives a real gradient to protect mid/trailing
  tokens.

---

## Meta-conclusion (feeds the "different attack angle" discussion)

Both flaws are the SAME disease: **the model optimizes the local reward
surface rather than understanding the game.** It spawns on 6 because +0.05 >
+0.03; it abandons the laggard because protecting it pays ≈ 0. Reward-tuning
these (the fixes above) is whack-a-mole — fix one surface artifact, another
appears. The orthogonal fix is to make the model build a representation of
*consequences/risk* (a world model), not memorize reward-correlated patterns.
See the attack-angle discussion / next journal entries.

**Status:** documented, NOT being fixed now (per user, 2026-06-14). Return
here if a targeted v13.6 polish run is wanted later.
