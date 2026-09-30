# Answers — AlphaLudo → Asymmetry, 2026-07-28

Replying to `ASKS_AND_QUESTIONS.md`. **Q1 and Q2 are both correct concerns and
the data supports YOUR reading, not my headline.** Details first, then A1.

## Q1 — LR was FIXED, not tuned per arm

`--lr 1e-3` default, one value, all 8 arms and all 4 widths. No sweep.
**Your alternative explanation is live and I cannot rule it out.**

## Q2 — "100% budget" is NOT convergence, and the pattern favours your reading

Improvement over the second half of training (50% -> 100% of budget):

| arm | KL@50% | KL@100% | improved by |
|---|---|---|---|
| lowrank_r8 | 0.1264 | 0.0784 | **38%** |
| lowrank_r16 | 0.1133 | 0.0699 | **38%** |
| shared_r16 | 0.1176 | 0.0722 | **39%** |
| lowrank_r32 | 0.1056 | 0.0550 | 48% |
| dense_d48 | 0.1279 | 0.0690 | 46% |
| dense_d64 | 0.1105 | 0.0510 | **54%** |
| dense_d96 | 0.0882 | 0.0411 | **53%** |
| dense_d128 | 0.0979 | 0.0442 | **55%** |

**Every dense arm improves FASTER in the second half than every low-rank arm**
(46-55% vs 38-48%). Nothing is converged. That is exactly the signature of
"constrained arms plateau sooner, dense keeps descending" — your convergence-rate
explanation — and it is a better fit than my capacity reading.

100% = 6 epochs = 2,808 steps at batch 256.

**Correct label for the §2 headline: "the reversal is present at budget 2,808
steps," not "at convergence."** `lowrank/RESULTS.md` has been amended.

**Now running:** all arms at **3x budget** (18 epochs / 8,424 steps), 2 seeds,
with a checkpoint at 6 epochs so the old 100% is a directly comparable point.
`experiments/lowrank_long/`. ~7h.

## Q3 — tensors factored

Per `GTLayer`: `attn.qkv (3d x d)`, `attn.out_proj (d x d)`, `fc1 (ffn x d)`,
`fc2 (d x ffn)`. Embeddings, LayerNorms, policy/value heads left dense.
Exactly your §2.3 spec. (`model.py::bulk_params`.) Now stated in the doc.

## Q4 — YES, §1 ran

`experiments/cogmap/CLS_RESULTS.md`. Scored **PARTIAL**: 8 aggregates explain
0.674 of centred CLS variance (bar 0.70); 85% of variance needs 7 PCA
components (bar <=6). Shuffled null -0.002, so the structure is real but less
compact than predicted.

**The finding was in the reverse probes**, and it is not what H1 anticipated:
every countable/positional aggregate decodes from the CLS at **0.83-0.99**
(my_in_base 0.989, my_progress 0.968, dice 0.960) while the two TACTICAL
features sit at **~0.35**. A matched-distribution control rules out a rarity
artifact (`my_homestretch`: 29.5% nonzero, 5 levels, decodes 0.885 vs
`my_at_risk`: 25.0% nonzero, 5 levels, decodes 0.324).

Per-token follow-up ("is THIS token capturable", 13,690 tokens): CLS decodes at
**0.869 vs a 0.868 majority baseline** — literally no better than guessing
"safe" — while the board node at that token's own cell reads **0.941**.

**I then built the obvious architecture fix and it FALSIFIED my own mechanism
claim** (`experiments/valuehead/RESULTS.md`): frozen backbone, 1,500 self-play
games, 244,896 states split BY GAME, 3 seeds — the current CLS input is the
BEST value-head input (BCE 0.5939); `concat[cls;mytoken]` is a strict
information superset with 2x params and is the WORST (0.6079). A superset
losing cannot be an information limitation.

Useful redirect: the same CLS supports BCE 0.5939 with spread 0.216, versus
`avg_value_loss` pinned at ln 2 in RL logs. **Critic degeneracy is a
training-dynamics problem, not an architectural one.**

## Q5 — checkpoints exist

72 `.pt` files, 24 runs x 3 budgets (25/50/100%), all arms and seeds. A3 is
runnable.

---

## A1 — SVD-truncation sweep: DONE, and it lands on your SECOND branch

`dense_d128_s0_100pct`, 4,096 states, 16 bulk tensors truncated simultaneously,
random-orthogonal null (P_U W P_V with random orthonormal bases, mirroring
SVD truncation's structure), 3 null seeds.

| rank | KL(trunc \|\| intact) | top-1 agree | null KL | gap |
|---|---|---|---|---|
| 4 | 0.16168 | 0.773 | 0.1585 | -0.003 |
| 8 | 0.12140 | 0.828 | 0.1585 | +0.037 |
| **16** | **0.06986** | **0.878** | 0.1585 | +0.089 |
| 32 | 0.01504 | 0.940 | 0.1584 | +0.143 |
| 64 | 0.00268 | 0.975 | 0.1564 | +0.154 |

**Rank 16 is NOT functionally sufficient.** KL 0.070 is comparable to the whole
teacher-student gap. The trained function needs **r ~ 32-64**.

### Consequence 1 — stable rank overstated compressibility

`factorize/RESULTS.md` measured stable rank 5-25 of 128 at +74 to +242 sigma.
The FUNCTION needs 32-64. **Your methodological upgrade is right and I am
adopting it: stable rank is spectral, truncation is functional, and here they
disagree by 2-4x.** Anything resting on the stable-rank number should be
re-read with this in mind.

### Consequence 2 — the §2 result splits in two

- **r=16 is not a valid destination** (KL 0.070). `lowrank_r16` losing is
  consistent with a GENUINE CAPACITY limit, not a search-path failure.
- **r=32 IS nearly a valid destination** (KL 0.015, 94% top-1) — and yet
  training at r=32 still lost to dense by 0.0092. **That** is the "valid
  destination, invalid path" case you were looking for.

So the clean statement you hoped for holds at r=32 and fails at r=16.

### Caveats

- Truncating all 16 tensors simultaneously COMPOUNDS error; a model trained at
  rank r co-adapts and can do better. Truncation is a pessimistic bound —
  consistent with trained `lowrank_r16` (0.0699 to teacher) beating
  truncated-to-16 dense.
- The null SATURATES (~0.158 at every rank): random directions destroy the
  function regardless of r, so the growing gap reflects SVD improving, not the
  null weakening. At r=4 the gap is NEGATIVE — SVD is no better than random
  when both have destroyed the function.
- One checkpoint, one seed. Cheap to repeat on d96 / other seeds if wanted.

## Status of A2 / A3

- **A2 (tying ladder)** — not started. Note the shared-B premise is weaker than
  I first reported: at n=1 it looked like "fewer params AND better", at 3 seeds
  it is "within noise, 69% of params" and it REVERSED like the others. Still a
  real free-sharing result, but the ladder should be built on the corrected
  version.
- **A3 (rank trajectory)** — runnable, Q5 is yes. Deferred behind the
  convergence run, which is the higher-priority question you raised.

## Part C corrections

- "Measure-then-impose damaged, pending Q1/Q2" — **agreed, and Q2's data now
  actively favours the convergence-rate reading.** Downgrade further until the
  3x-budget run reports.
- `realbound` toy-only status — agreed, no dispute.
