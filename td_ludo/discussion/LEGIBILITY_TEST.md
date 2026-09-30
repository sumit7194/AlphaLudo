# AlphaLudo × the Legibility Law — honest test record (2026-06-21)

Testing whether the **Legibility Law** (from the SpaceTime / "tabula geometrica"
project, `~/Github/SpaceTime/writeups/legibility_law.md`) has anything to say in a
third domain: a trained game-RL agent. **Honesty/correctness is the only goal** —
a null or a "doesn't cleanly map" is a valid, recorded outcome.

## The law (1 paragraph)
A per-object property stored as a **free per-object embedding** (looked up by ID)
is linearly *un*-readable (scrambled: linear-decode low, kNN high) — while the same
property *inferred by a shared/amortized encoder* is linearly legible. Refinement:
the scramble needs a **multi-dimensional** latent (1-D free codes stay legible).
Probe ladder: ridge **linear-decode r** = legibility; kNN **r** = info presence.

## What AlphaLudo can and cannot honestly test
**Decisive finding (Explore audit, 2026-06-21):** the ONLY free-per-token-ID model,
`v12_legacy` (`td_ludo/td_ludo/models/v12_legacy.py:113`, `token_idx_emb =
nn.Embedding(4, 96)`), was **NEVER TRAINED** — it exists only as a weight-loading
scaffold (`scripts/generate_sl_data_v122.py:43` loads v12 weights `strict=False`,
ignoring the embedding). **Every TRAINED AlphaLudo model is amortized:** v12
(permutation-equivariant, no token-ID emb), v14_scalar (DeepSets), v13.5
(rank-symmetric), v15 (cell-based).

Consequences (honest scope):
1. **Cannot** run the decisive free→scramble contrast — no trained free arm exists.
2. **Cannot** cleanly instantiate the LITERAL law anyway: Ludo's per-token
   properties (position, danger) are DYNAMIC → board-inferred regardless of
   architecture; a static per-token-ID embedding couldn't store them. The natural
   free code (`token_idx_emb`) would carry low-D token IDENTITY, not a multi-D
   property — and the law says 1-D free codes stay legible.
3. **Can** test the "amortized → legible" leg directly (probe a trained amortized
   model for a multi-D per-token property).
4. **The absence itself is the contribution:** AlphaLudo, like the LLM cross-test,
   is **amortized-by-default** — the free regime wasn't even worth training. That
   is a SECOND real-world domain corroborating the writeup's strongest reframe:
   *"real trained systems are amortized → legible by default → that's WHY LRH
   holds; the free→scramble regime is a toy artifact."*

## Honest verdict on the prior-session suggestion
The other-session Claude (no context) pitched AlphaLudo as "the cleaner test
because the contrast is already built." That is **incorrect**: the free arm was
never trained, the existing models are a confounded multi-variable comparison, and
Ludo's dynamic properties don't instantiate the literal claim. The writeup's own
abstract synthetic harness (`curvature/scripts/35_legibility_scale.py`) is a
*cleaner* test of the free→scramble leg than dressing it in Ludo. Its "diagnostic
for the 80-83% plateau" angle is also moot — AlphaLudo's ~85% was independently
proven (6 ways) to be the variance/luck ceiling, not a representation problem.

## What we found when we tried (the honest outcome — no clean probe was run)
On inspecting the actual checkpoints, the "amortized → legible probe" I'd planned
turned out NOT to be cleanly interpretable, so — per the honesty directive — I did
not run it. What the inspection actually showed:

**(a) The checkpoints are transitional / ambiguous.** `play/model_weights/
model_v12.pt` (the only "v12" weights) contains BOTH `token_idx_emb` (4×96, the
free per-token-ID embedding, TRAINED) AND `policy_fc1` (the V12.1
permutation-equivariant head). It matches neither current class cleanly: live
`v12.py` dropped `token_idx_emb`; `v12_legacy` has it but a different (centralized)
policy head. So there is no checkpoint that cleanly IS "the free arm" or "the
amortized arm" — probing it could not be honestly attributed to free-vs-amortized.
A clean test would require *training* a controlled A/B from scratch (two minimal
models identical except `token_idx_emb`, same SL data) — a real, bounded effort,
not something the existing weights give us.

**(b) The genuinely honest, in-house finding — the "T2 blind spot" (MEASURED).**
The original v12, which DID have the free per-token-ID embedding (`token_idx_emb`),
developed a measured pathology: eval-lens analysis showed it **picked token T2 in
12% of decisions but in 0% of disagreements** (vs ~25% if uniform) — spurious
*slot-identity* leaking from the free embedding into the policy. V12.1 FIXED it by
**dropping `token_idx_emb`** and going permutation-equivariant (amortized). Source:
`td_ludo/td_ludo/models/v12.py:108-113`, `train_sl_v12.py:34,165` ("Default ON for
V12.1+ — actively breaks the T2 blind spot we measured"). This is a natural,
documented instance of *"a free per-object-ID code leaked spurious structure;
removing it (amortizing) fixed it"* — **consistent with the law's SPIRIT**, but it
is a *behavioral/policy* artifact, NOT the law's specific linear-legibility claim.
Cite as **illustration, not confirmation** (exactly the writeup's Othello-GPT
caveat).

**(c) Amortized-by-default corroboration.** The production lineage *converged on
amortized*: v12 → V12.1 (dropped free token-ID) → v13.5 (rank-symmetric) → v15
(cell-based, identity-erased). Nobody kept a free per-token-ID code because it hurt.
This corroborates the writeup's strongest reframe — *"real trained systems are
amortized → legible by default; the free regime is a toy artifact"* — in a SECOND
real-world domain (game-RL), after the LLM cross-test.

## Verdict
**AlphaLudo cannot, as-is, give a clean controlled test of the linear-legibility
claim** — checkpoints are ambiguous, the natural free code is low-D identity (the
law says 1-D stays legible), and Ludo's per-token properties are dynamic. Forcing a
probe would have produced an un-attributable number; honesty says don't. What it
DOES offer the law: (1) the T2-blind-spot illustration (free per-object-ID code →
pathology → fixed by amortizing — spirit-consistent, behavioral), and (2)
amortized-by-default corroboration of the "why LRH holds" reframe. Both are
**illustration/corroboration, not confirmation.** A real confirmation would need a
purpose-built one-variable A/B training run — and even then the abstract harness
(`curvature/scripts/35_legibility_scale.py`) tests the free→scramble leg more
cleanly than Ludo can.

## EXACT weight paths inspected
- v12 (transitional, has trained `token_idx_emb`): `td_ludo/play/model_weights/model_v12.pt`
  (28-ch V10 encoder, 4 res-blocks ×96, owner_emb 2×96, token_idx_emb 4×96, policy_fc1).
- v14_scalar (clean amortized/DeepSets): `td_ludo/checkpoints/v14_scalar/model_best.pt`.
- v13.5 (amortized rank-symmetric): `td_ludo/checkpoints/v135/model_sl.pt`.
- v15.2 (amortized cell-based): `checkpoint_backups/v152_gaeterminal_2026-06-19/...BEST...pt`.
- `v12_legacy` (clean free arm): class only, **NO trained checkpoint**.

## Controlled A/B — BUILT 2026-06-23 (code written; run when MPS/VM free)
Since the existing weights can't give the clean contrast, we built the one-variable
A/B from scratch. "Whatever we find — scramble, null, or boundary — is genuine
understanding." Methodology mirrors the writeup's `29_consensus_legibility.py`.

- **The one variable** (`experiments/legibility/model.py`): `LegibilityV12(free_token_id)`
  — identical V12 arch (V10 28-ch encoder → 4×96 CNN → per-token gather + owner_emb →
  2-layer token attention → per-token policy head); FREE arm ADDS
  `token_idx_emb=nn.Embedding(4,96)`, AMORTIZED arm does not. Same data, seed,
  hyperparams; NO own-token permutation augmentation (so the free embedding can
  develop per-slot structure — augmentation would neuter the contrast).
- **Task/data:** SL on the Ludo policy (token-indexed CE), data
  `checkpoints/sl_dataset_v1` re-encoded to V10 28-ch on the fly via the new
  `variant="v12"` added to `sl_dataset.py` (`_encode_v12` → `encode_state_v10`; the
  old pre-encoded `sl_data_v10` is GONE).
- **Probe** (`experiments/legibility/probe.py`): per-OWN-token rep = 8-token tensor
  AFTER attention (own=first 4) over fresh random-self-play states; multi-D property =
  (position, danger via `game/consequence_targets.py:capture_prob_next_turn`); ridge
  R² (legibility) vs kNN R² (info), same states both arms.
- **Outcomes:** FREE linear-LOW/kNN-HIGH + AMORTIZED legible → scramble in game-RL
  (real third-domain evidence). Both legible → honest NULL (free code is low-D
  identity, which the law says stays legible — consistent, not a refutation).

## v12.3 amortized BASELINE — RAN 2026-06-23 (no training; pipeline validation)
Probed the already-trained **v12.3** (`play/model_weights/v12_3/model_best.pt`,
`MinimalCNN14Aux` 8×96, V17 17-ch, fully amortized: ZERO embeddings) — per-own-token
rep = gathered cell feature (no token attention), 700 random-self-play states / 2800
token-samples. `experiments/legibility/probe_v123.py`.

| property | linear R² | kNN R² | read |
|---|---|---|---|
| **position** | **0.861** | **0.931** | **legible** ✓ |
| danger (`capture_prob_next_turn`) | 0.079 | 0.114 | unreliable target |

**Findings:**
- **Pipeline validated** + the easy leg confirmed: a trained amortized RL model keeps
  a per-token property (position) **linearly legible** (0.86). Expected, but real.
- **Danger is unusable as a probe target:** only **6.0% of tokens are nonzero**
  (mean 0.011, std 0.043) → R² is dominated by zeros; both decoders sit ~0. That is a
  TARGET-SPARSITY artifact, NOT a scramble (a scramble needs kNN-HIGH / linear-LOW).
- **Position is the clean target** (58 unique values, std 33.7, 27% base / 48% track /
  26% scored) → use it as the decisive A/B target; drop/replace danger.

**Scramble mechanism the matched A/B will test (on position):** the free
`token_idx_emb` adds a CONSTANT per-slot vector, offsetting each of the 4 own slots'
position-manifolds to different latent locations. Pooling all slots for the probe
then misaligns the global position axis → if the law bites, the FREE arm shows
linear-LOW / kNN-HIGH on position while the AMORTIZED arm stays linear-HIGH. (Null =
the constant offset doesn't disrupt linear readout — also a clean, honest result.)
NOTE: v12.3's gather-only rep is NOT a substitute for the matched A/B's post-attention
rep — different architecture; this run is only the amortized-legible anchor.

## MATCHED A/B RESULT — RAN 2026-06-23 (MPS) — the decisive run
Trained both arms identically (V10 28-ch, 600K rows, 3 epochs, seed 0, NO permute
aug; only `token_idx_emb` differs). **Both reached IDENTICAL fit: 94.0%
teacher-match.** Probe = same 800 random-self-play states / 3200 own-token samples.

| property | AMORTIZED linear | FREE linear | AMORTIZED kNN | FREE kNN |
|---|---|---|---|---|
| position | 0.920 | 0.901 | 0.961 | 0.949 |
| danger   | 0.590 | 0.573 | 0.688 | 0.673 |

**Result: NO SCRAMBLE.** A scramble = free linear-LOW + kNN-HIGH (linear collapses,
kNN preserved). Here linear AND kNN drop by the SAME tiny ~0.015 (noise floor) — a
uniform, non-divergent shift, not a scramble. Both arms stay legible on both props.
(Note: with the attention model, DANGER is now decodable — 0.59 vs v12.3's gather-only
0.08 — because token attention aggregates board context. Legible in both arms.)

**The null is MEANINGFUL, not trivial** (`arm_free.pt` inspection): the free code was
genuinely USED — per-slot `token_idx_emb` L2 norms **[2.77, 1.70, 1.16, 2.87]**
(LARGER than `owner_emb` 0.57/1.24), pairwise cosine −0.28 (4 distinct vectors). The
model actively learned a big, distinct free per-token-ID code **and the property
stayed legible regardless.**

### Interpretation (the actual contribution to the law)
**Scramble requires the free code to be the STORE of the property — not merely
present.** In the law's toy setup the free embedding *holds* the multi-D property
(looked up by ID), so its structure scatters → scramble. Here the property
(position, danger) is computed by the AMORTIZED backbone (CNN + attention) from the
board; the free `token_idx_emb` only adds per-slot IDENTITY *on top*. A distinct
constant per-slot offset makes 4 translated sub-clouds, but the position→latent
direction is shared across them, so a linear decoder still finds one global axis →
no scramble. The free code being multi-D (96-D) and used is NOT sufficient; it must
CARRY the property.

This **corroborates** the law's refinement (identity / non-property free codes stay
legible) and its "amortized-by-default → why LRH holds" reframe: real agents compute
properties amortized and use free codes only as identity tags, so they don't scramble
— now shown empirically with a free code that was *actively used*, not ignored.

**Honest limitation:** this does NOT test the law's CORE free-STORAGE scramble —
Ludo's per-token properties are dynamic/board-computed and can't be stored in a
static per-ID embedding, so we couldn't instantiate free-storage. It tests the
adjacent, *realistic* configuration (free identity tag alongside an amortized
property-computer), which is what real agents actually have. For the toy free-storage
scramble itself, the abstract harness (`35_legibility_scale.py`) remains the clean
test. Single seed; the ~0.015 Δ is at the noise floor — qualitative result (no
scramble, free code used) is robust; a multi-seed CI would tighten the Δ.

## DECISIVE STUDY pt.1 — POSITIVE CONTROL + dimensionality/task sweep (2026-06-23)
To make any Ludo NULL credible, we first proved our probe DETECTS a scramble when one
exists, by faithfully reimplementing the SpaceTime abstract task (`35_legibility_scale.py`
World/Learner + `50_objective_x_storage.py` regression arm) in
`experiments/legibility/positive_control.py`, SAME metric (Pearson r of 5-fold
cross_val_predict; Ridge=legible, kNN=info). Faithfulness check: our free-K=2 linear r =
**0.216** matches their reported **0.22** exactly.

**Core sweep (random-MLP world, code-dim 16), mean over 3 seeds:**
| K (property dim) | amortized linear | free linear | free kNN |
|---|---|---|---|
| 1 | 0.99 | 0.36 | 0.71 |
| 2 | 0.84 | 0.22 | 0.63 |
| 4 | 0.68 | 0.27 | 0.52 |
| 8 | 0.39 | 0.06 | 0.23 |

- **Core law CONFIRMED:** amortized ≫ free in linear legibility at every K; free shows the
  scramble signature (linear-LOW / kNN-HIGH = info present but illegible). **→ our probe
  works; a Ludo null is therefore meaningful.**
- (K=8 amortized is also low, but kNN≈linear → just task-too-hard, not a scramble.)

**Reconciling their "1-D free stays legible" refinement (NEW honest finding):**
Their physics script (48) reports 1-D free LEGIBLE (0.86); our abstract task gives 1-D free
SCRAMBLED (0.36). We tracked down why:
- Embedding width is NOT the cause — free K=1 stays scrambled at code-dim 16/4/2
  (linear 0.36/0.14/0.15, kNN ~0.9).
- TASK STRUCTURE is the cause. With a **linear-in-property** world (y = base(x) + Σ pₖ·coupₖ(x),
  i.e. the property acts monotonically like charge→force): free K=1 → **0.61 (legible)**,
  free K=2 → 0.42 (still scrambled). That reproduces their 1-D-legible / multi-D-scramble
  boundary **and** pins its precondition: *"1-D free stays legible" requires a
  monotone/near-linear property→output map; under a generic (random) map even a 1-D free
  code scrambles.* The general law (amortized ≫ free) holds across all conditions.

Artifacts: `positive_control.py` (`--ks`, `--cdim`, `--world {mlp,linear}`), `positive_control.log/json`.

## DECISIVE STUDY pt.2 — MULTI-SEED LUDO CI (2026-06-23) — the bulletproof null
6 seeds (0-5), each a fully paired amortized-vs-free pair (only `token_idx_emb`
differs; seeds 1-5 trained from a pre-encoded 300K cache, num_workers=0 for
robustness; all reached ~93% teacher-match). Probed on ONE shared 2000-state /
8000-token set with the law's Pearson-r metric. `probe_multiseed.py`.

| metric | amortized | free | PAIRED Δ (free−amortized) |
|---|---|---|---|
| position linear-r | 0.950±0.006 | 0.950±0.007 | **−0.001 ± 0.007** |
| position kNN-r    | 0.985±0.003 | 0.983±0.002 | −0.002 ± 0.003 |
| danger   linear-r | 0.769±0.016 | 0.766±0.012 | **−0.004 ± 0.020** |
| danger   kNN-r    | 0.825±0.017 | 0.826±0.007 | +0.001 ± 0.013 |

**NO SCRAMBLE, tight CI.** The free arm is exactly as legible as the amortized arm
on both properties; the paired Δ on linear legibility is ≈0 (±0.01) — and the SAME
probe shows Δlinear ≈ **−0.6** when a scramble is real (positive control, pt.1). The
free code is also genuinely USED (per-slot `token_idx_emb` norms 1.2–2.9, distinct).

### FINAL VERDICT (the whole study)
1. **Probe validated** (pt.1): faithfully reproduces the law — free-storage of a
   property scrambles (linear-LOW/kNN-HIGH), amortized stays legible; free-K2 = 0.216
   matches their 0.22. So our instrument detects scrambles.
2. **Refined the law's 1-D claim** (pt.1): "1-D free stays legible" is NOT universal —
   it needs a monotone property→output map (their physics); under a generic random map
   even 1-D free scrambles. Embedding width is irrelevant. (New, honest contribution.)
3. **Ludo null is real and tight** (pt.2): a free per-token-ID embedding in a trained
   game-RL agent does NOT scramble per-token properties (Δlinear ≈ 0 ± 0.01 over 6
   seeds), because the property is computed by the amortized backbone and the free code
   only carries IDENTITY — exactly the boundary the law's refinement predicts.

**What AlphaLudo honestly gives the legibility law:** (a) an independent third-domain
confirmation that the scramble needs the property to be free-STORED (an identity-only
free code stays legible even when actively used), and (b) a sharpening of the 1-D
refinement (it's task-structure-dependent, not width-dependent). Confirmation +
boundary-mapping of the REFINED law — not the toy claim, which Ludo can't host (only 4
tokens; properties are dynamic/board-computed, not free-storable).

## Artifacts
- This doc.  Source law: `~/Github/SpaceTime/writeups/legibility_law.md`.
- v12.3 baseline probe: `td_ludo/experiments/legibility/probe_v123.py`.
- Matched A/B: `td_ludo/experiments/legibility/{model,train_arm,probe}.py`;
  headline weights `arm_amortized.pt` / `arm_free.pt`; 6-seed `arm_{amortized,free}_s{1..5}.pt`.
- Decisive study: `positive_control.py` (pt.1, `--ks/--cdim/--world`) +
  `probe_multiseed.py` (pt.2, 6-seed CI) + `precompute_cache.py`. JSON: `probe_multiseed.json`.
- T2-blind-spot source: `td_ludo/td_ludo/models/v12.py:108-113`, `td_ludo/train_sl_v12.py`.
- Controlled-A/B code: `td_ludo/experiments/legibility/{model,train_arm,probe}.py` +
  `README.md`; data variant in `td_ludo/sl_dataset.py`. Trained arms →
  `arm_amortized.pt` / `arm_free.pt`. Run: see `experiments/legibility/README.md`.
