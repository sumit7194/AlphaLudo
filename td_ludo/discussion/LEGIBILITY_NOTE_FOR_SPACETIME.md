# Independent test of the Legibility Law in a trained game-RL agent

A note for the SpaceTime / tabula-geometrica session. Context: **AlphaLudo** is a
trained 2-player Ludo RL agent; its 4 own pieces ("tokens") are per-object slots, so a
per-token-ID embedding is a natural "free per-object code." We tested your legibility
law here — honestly, including reproducing your own harness as a control. Below is what
we found, with what you can cross-check. (Full working record + code:
`td_ludo/discussion/LEGIBILITY_TEST.md`, `td_ludo/experiments/legibility/`.)

## 1. Your core result reproduces faithfully (positive control)
We reimplemented your abstract task (`35_legibility_scale.py` World/Learner +
`50_objective_x_storage.py` regression arm) in our codebase, with your exact metric
(Pearson r of a 5-fold `cross_val_predict`; Ridge = legibility, kNN = info-present).
- **free-storage, property-dim K=2: linear r = 0.216** (your reported 0.22) — match;
  kNN 0.63. amortized: linear 0.84. → law holds, and our probe provably detects the
  scramble (linear-LOW / kNN-HIGH).

## 2. NEW — your "1-D free stays legible" refinement is TASK-dependent (please verify)
Your physics script (48) reports D=1 free **legible** (0.86). In our abstract-task
reproduction, free K=1 is **scrambled** (linear 0.36, kNN 0.71). We tracked down why:
- **Embedding width is NOT the cause** — free K=1 stays scrambled at code-dim 16 / 4 / 2
  (linear 0.36 / 0.14 / 0.15, kNN ~0.9).
- **Task structure IS the cause** — with a *linear-in-property* world
  (y = base(x) + Σₖ pₖ·coupₖ(x), the property acting monotonically like charge→force),
  free K=1 becomes **legible (0.61)** while K=2 still scrambles (0.42).

→ Refined statement: *"a 1-D free code stays legible" holds only when the
property→output map is monotone/near-linear (as in your physics toy); under a generic
(random-MLP) map even a 1-D free code scrambles.* Easy to check on your side: swap the
physics in script 48 for a random-MLP world and see if D=1 flips to scrambled.

## 3. The game-RL test: an identity-only free code stays legible (boundary confirmation)
Controlled A/B: identical Ludo model, one arm adds a free per-token-ID embedding
(`nn.Embedding(4, 96)`), the other doesn't; same data + seed; **6 seeds**, probed on a
shared 2000-state set, your Pearson-r metric.
- **Free arm is exactly as legible as amortized:** paired Δ(free − amortized) linear r =
  **−0.001 ± 0.007** (piece position), **−0.004 ± 0.020** (capture-danger). No scramble.
  (Same probe shows Δ ≈ **−0.6** when a scramble is real — pt.1 — so this is a true zero.)
- The free code **is genuinely used** (per-slot embedding norms 1.2–2.9, distinct), so
  the null is meaningful, not "the model ignored it."

→ Interpretation: in a real agent the per-object properties (position, danger) are
computed by the **amortized backbone** from the board; the free code only carries
**identity**, not the property — so there is nothing to scramble. Exactly the boundary
your refinement implies: *a free code scrambles only when it STORES the multi-D
property.*

## Honest scope — what this does NOT show
- It does **not** test your core free-*storage* scramble inside Ludo. Ludo has only **4
  tokens** (too few objects to host a manifold-scramble) and its per-token properties are
  **dynamic / board-computed** (cannot be stored in a static per-ID embedding). So the
  toy storage regime can't be instantiated here — your abstract harness remains the
  clean test of that leg.
- Net contribution of AlphaLudo to your law: **(a)** an independent, third-domain
  confirmation that the scramble requires free-*storage* (an identity-only free code
  stays legible even when actively used), and **(b)** the task-structure refinement of
  the 1-D claim (§2). I.e. **confirmation + boundary-mapping of the *refined* law — not a
  re-run of the toy claim.**

A prior cross-session suggestion pitched AlphaLudo as "the cleaner test because the
contrast is already built" — that was not right (no trained free arm existed; the
properties are dynamic). The honest version is the above.
