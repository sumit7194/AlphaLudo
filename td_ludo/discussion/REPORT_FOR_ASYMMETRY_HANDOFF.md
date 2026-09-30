# Report back to the Asymmetry handoff — what Ludo taught us

**To:** the Asymmetry session that wrote `/Users/sumit/Github/Asymmetry/LUDO_HANDOFF.md`
**From:** the AlphaLudo working session, 2026-07-21 → 2026-07-23
**Frame honoured:** §0 observation-not-improvement. Win rate was never a read-out.
Every claim below is labelled observational / controlled / in-progress, with n.

---

## 0. Executive summary

We ran G1 (registers) to a pre-registered null, killed G3-as-designed at the
premise check, and replaced it with something better the data handed us: a
**trajectory** measurement (your §1 complaint — "nobody watched the weights
move" — taken seriously) plus two live interventions. Main learnings:

1. **SL front-loads depth; RL redistributes it backwards.** Observational
   across two lineages, then confirmed by controlled intervention (n=1 run,
   79 trajectory points): starting from the exact SL checkpoint, RL began
   recruiting the idle middle layer within ~3K games.
2. **RL reallocates trained capacity but (so far) does not expand into
   brand-new capacity.** A provably-identity 5th layer inserted mid-RL took a
   ~5e-5 foothold within 500 games and then plateaued — 3,500x below the
   working layers — through 14.7K games. In-progress: pre-registered decision
   point is 40K games; kill condition on track to fire.
3. **Behaviour does not determine structure.** Students distilled to top-1
   ~0.90 agreement with a teacher reproduce NONE of its internal organisation
   — not its attention sink, not its layer profile — and organise differently
   every seed. This invalidated two of our experiments before we caught it and
   is the single most transferable warning in this report.
4. **Your G-family intuition was right in a way the design didn't anticipate:**
   growth-in-place (G3-style insertion) meets a network that *won't* use new
   capacity, while the same optimiser eagerly reuses *existing trained*
   capacity. "Growth produces distributed mush" (§4 expected outcome) looks
   wrong so far; the reality is sharper: growth produces **nothing**, while
   redistribution is fast and legible.

---

## 1. Findings in detail

### F1 — Layer-importance trajectory: SL front-loads, RL redistributes
*(observational n=9 checkpoints + n=4 endpoints; then controlled n=1 run)*

Layer knockout-KL (skip layer L, KL vs intact policy; eval sets fixed per
lineage; history_len=2 models measured on generated genuine frame-pairs, see
§3.4):

**v15.1 SL, 1M → 9M states (9 checkpoints, d128 L4 h2):**

| states | L0 | L1 | L2 | L3 |
|---|---|---|---|---|
| 1M | 0.305 | 0.490 | 0.036 | 0.013 |
| 5M | 0.720 | 0.540 | 0.062 | 0.035 |
| 9M | 0.987 | 0.777 | 0.063 | 0.040 |

Back half near-dead from the FIRST checkpoint and it never wakes; the
front/back imbalance *widens* with training (L0/L2 ratio 8.5x → 15.7x).
Structure is set early and intensifies — consistent with your §1 observation
that budget changes rankings, but here the *shape* is stable and only deepens.

**What RL does to finished SL models (same eval data within lineage):**

| | L0 | L1 | L2 | L3 |
|---|---|---|---|---|
| v15.1 SL 9M | 0.987 | 0.777 | 0.063 | 0.040 |
| v15.1 RL G826K | 0.253 | 0.192 | 0.027 | 0.085 |
| v15.2 SL 1ep | 0.419 | 0.165 | 0.102 | 0.186 |
| v15.2 RL champion | 0.427 | 0.296 | **0.422** | 0.194 |

Two independent lineages, same direction: mass moves off L0 toward the
middle/late layers. (Confound in this table alone: RL models also trained
longer. Resolved by F1b.)

**F1b — controlled confirmation (n=1, pre-registered):** we resumed RL *from
the exact v15.2 SL checkpoint* with per-5-min knockout profiling (79 points,
52K games). L2's share of total layer-KL: 0.117 → 0.224 (champion: 0.315), a
clean monotone climb with onset **within ~3K games**, while eval WR rose
48% → 82%. L0 share fell at −0.29/100K games; L2 rose at +0.21/100K. At G50K
L2 briefly overtook L1 (crossing not yet stable at cutoff). Not "just more
training": redistribution starts immediately. Also robust to opponent mix:
our pool (self 60 / scripted Expert 40) differs from the champion's neural
pool, same direction.

*Normalisation note that mattered:* early points showed ALL layers' absolute
KL drifting down together (global policy-sharpness change). Absolute KL
cannot distinguish that from redistribution — use each layer's **share of
total** as the redistribution observable.

### F2 — Growth: RL does not (so far) recruit born-blank capacity
*(controlled, in-progress: 14.7K of pre-registered 40K games; n=1)*

Your G3, executed against the live RL run rather than a static substrate.
Pre-registration is in
`AlphaLudo/td_ludo/experiments/depth/insert_layer_rl.py` (prediction, kill
condition, declared limits — locked before launch).

Method: state-dict surgery on the G52.7K RL checkpoint — insert a 5th GTLayer
at position 2 with `attn.out_proj` and final FFN linear zeroed. Pre-LN
residual ⇒ provable identity: measured end-to-end output delta **exactly
0.0**. Your §4/G2 zero-init-for-legibility preference, applied to depth: any
function the layer later carries was learned after insertion. Optimizer
restarted (param groups changed) — declared as a perturbation; the original
layers' unbroken trends are the control for it.

Inserted-layer knockout-KL (measurement noise floor demonstrated at ~1e-10 by
the at-insertion reading):

| post-insertion games | inserted layer KL |
|---|---|
| 0 | −8.5e-11 |
| ~500 | 1.8e-05 |
| 2,049 | 3.3e-05 |
| 8,705 | 7.9e-05 |
| 14,338 | 4.4e-05 |

**Immediate tiny foothold, then plateau.** No compounding. Meanwhile the four
original layers continue their pre-insertion trends (L3 kept climbing 0.197 →
0.224 — the F1b recruitment continuing straight through the surgery) and play
quality held. Contrast table:

| capacity type | RL's response |
|---|---|
| idle-but-TRAINED layer (SL-shaped weights) | recruited in <3K games → 0.17 abs KL by 50K |
| born-blank identity layer | ~5e-5 foothold, plateau, 3,500x below working layers |

Interpretation offered (carefully, n=1, 37% of decision window): **RL
repurposes latent structure that prior training built; it does not easily
create structure from nothing mid-run.** Nothing in the loss forces gradient
through the identity layer (the residual suffices), and evidently nothing
recruits it either. If the plateau holds to 40K the pre-registered kill fires:
"RL reallocates but does not expand." Relevant to your Grow hypothesis (§1
item 1): 'adding capacity to a live network and watching what it does' — what
it does, at least here, is *decline to use it*, which is a cleaner negative
than the "diffuse smear" your §4 expected-outcome paragraph priced in.

### F3 — G1 registers: NULL (complete, pre-registered kill condition met)

Full record: `AlphaLudo/td_ludo/experiments/registers/RESULTS.md`.

K=16 registers, arms control/born/grown x 3 seeds, 2 budgets, params matched
to 0.66%. Registers RECEIVE attention (~12% of CLS mass, born arm) but are
not load-bearing: per-register knockout-KL ~1e-4 vs a 1e-3 materiality bar;
never become the attention sink (0/6 runs); grown arm sits exactly on the
kill condition. **Attention received ≠ function performed** — measuring only
attention share would have called this a success. Echoes your ViT-registers
precedent in reverse: on this graph transformer, with this training, the
artifact the registers were meant to absorb doesn't transfer (see F4), and
the registers stay decorative.

### F4 — The substrate trap: distilled students don't inherit structure
*(3 seeds d64 + 1 seed d128 + teacher; the costliest lesson)*

Students distilled from v15.2 (KL to teacher policy over 120K cached states,
top-1 agreement ~0.90):

- **No stable attention sink.** Teacher: consistent sink on the OD cell,
  5.5x uniform, monotone in depth (0.74/1.11/2.66/5.52x across layers).
  Students: sink on a DIFFERENT node every seed, strength 16.8x ± 4.7 (d64);
  the size-matched d128 student sinks at **71x uniform on yet another node**
  while reading 0.26x on OD.
- **No stable layer profile.** d64 students: flat, overlapping within σ, one
  seed fully inverted. d128 student: monotone-decaying. Neither reproduces
  the teacher's shape.

Both G1's sink-migration prediction and G3-as-designed were unfalsifiable on
this substrate — there was no stable sink to migrate and no weak middle to
fill. **Two matrices were run or nearly run before this was caught.**

Transferable rule: *before any structural experiment, verify the target
structure exists in the substrate you will actually run on.* Costs minutes.
Also a finding in its own right for your frame: three systems with
near-identical input-output behaviour, three different internal organisations,
and (for students) organisation not even stable across seeds.

### F5 — Attention sinks: not an RL artifact, and not where you expect

Chronologically we passed through two WRONG conclusions here (kept in the
record): "students have no sink" (false — measured only the teacher's sink
cell) and "sinks are an RL artifact" (false — the distilled student sinks
harder than the teacher, elsewhere). Correct statement: **sink presence is
robust across training regimes; sink LOCATION and STRENGTH are not.**
Instrument accordingly: locate the sink (argmax over mean CLS attention),
then measure it — never hardcode the cell.

### F6 — Corrections to baselines you may be relying on

- The published layer-knockout U-shape
  (`AlphaLudo-MechInterp/.../layer_knockout_metrics.json`, L0 0.356 / L1
  0.106 / L2 0.076 / L3 0.179) belongs to **v15.2 SL** (`v152_sl_final.pt`).
  We reproduce it on that checkpoint (0.419/0.165/0.102/0.186 on our eval
  set — same ordering, same weakest layer). The **RL champion has a
  different profile entirely** (L2 0.422, second-strongest). Your §3 table
  cites the U-shape as "v15.2 numbers" — true only for the SL model. Anyone
  comparing against that baseline must say WHICH v15.2.
- Handoff §3 says edge biases ≈ noise ⇒ EdgeBiasedAttention ≈ vanilla — our
  work is consistent with this; nothing contradicted it.
- v15.3 (RL-from-random, the run §0 mentions): all four layers knockout-KL
  0.002-0.003 — at floor, uninformative per your §2.5, excluded from all
  claims above.

---

## 2. Methodology extensions to your §2 (paid for here)

Your five rules all held up. We'd add four:

6. **Premise check before matrix.** Verify the structure you're intervening on
   exists in the substrate, with the same instrument you'll use for the
   result. (F4 — would have saved ~5h of runs.)
7. **Materiality bars, not just comparisons.** "X > floor" is not "X
   matters." G1's registers scored 10x the board-node floor — both numbers
   round to zero. Fix an absolute bar in the pre-registration.
8. **Locate, then measure.** Any observable of the form "value at the cell
   where the phenomenon was" silently fails when the phenomenon moves. Two of
   our wrong conclusions trace to this.
9. **Share-of-total for redistribution claims.** Absolute importances move
   together under global sharpness changes; normalise before claiming
   reallocation.

And one operational one: **snapshot-and-measure as-you-go** (poll the rolling
checkpoint, copy + profile immediately, append JSON). Three power losses over
two nights; zero data lost, every partial run still usable. On this hardware,
treat interruption as the default execution mode.

Instrument details worth stealing: knockout = −inf on the attention *column*
pre-softmax with self-attention kept finite (NOT edge-type relabelling — type
0 is a real type; our first version severed nothing and would have called
every register dead). CKA(register, CLS) is ~0.97 at RANDOM INIT — the null
is high, not zero; always compute the untrained reference.

---

## 3. Status of your proposed lineup (§7 order)

| item | status |
|---|---|
| G3 depth | Superseded-and-answered-better: premise dead on distilled substrate; the real question ("is idle depth recruitable?") answered YES by F1b via RL; "is NEW depth recruitable?" = F2, in progress, leaning NO |
| G1 registers | DONE — null, kill condition met, full writeup |
| C2 modReLU | **DONE 2026-07-23 — double null, your conclusion STRENGTHENED.** 3 seeds x 2 budgets, params matched 0.66%, CLinear ported verbatim. (a) Fair-trial INVERTED: modReLU 0.0872±0.0136 is *worse* than split-GELU 0.0610±0.0128 (real: 0.0504±0.0037) — the "broken nonlinearity" objection (§2.3) is tested and retired; your complex nulls were not a split-GELU artifact. (b) The ring plot does not exist: pos-emb phases do not wind the main loop (all p 0.38-0.97, permutation null, real-arm placebo clean; NB the engine's loop is 51 cells, not 52), and weight phases stay uniform through training (resultant <0.01) — "phase is a nuisance dimension" reproduced. Bonus replication: complex arms ~3.5x seed-noisier than dense, matching your §2.3 stability note. Full record: `td_ludo/experiments/ring/` |
| G2 width | Not started. F2's result raises its stakes: does width-growth also get ignored, or is the reallocate-don't-expand pattern depth-specific? |
| C3, C1, B | Not started. For B, note our logit-lens/early-decision numbers replicate on SL models but the RL champion's profile differs — pick the substrate deliberately |

§9 open decisions, as resolved here: (1) register edge type — we ran "new"
types; "global" comparison arm dropped as moot after the null. (2) growth
trigger — fixed insertion at 50% (distill) / at-checkpoint (RL); signal-driven
never reached. (3) SL vs RL substrate — **our sharpest disagreement with the
handoff:** §9.3 recommends developing observables under SL/distillation
first. Half-right: develop the *instruments* there (cheap), but structural
*results* do not transfer from distilled proxies (F4), and SL-vs-RL is itself
the most interesting axis we found (F1). (4) 4-player — untouched, as ordered.

## 4. Data & code index (all under `/Users/sumit/Github/AlphaLudo/`)

| path | contents |
|---|---|
| `td_ludo/experiments/EXPERIMENTAL_ARCHITECTURES_LOG.md` | running honest log, includes retractions |
| `td_ludo/experiments/registers/` | G1: model, trainer, observer, RESULTS.md, 18 ckpts, observables.json |
| `td_ludo/experiments/depth/model.py` | V15Depth + verified identity insertion |
| `td_ludo/experiments/depth/trajectory.py` + `trajectory.json` | multi-checkpoint layer profiles |
| `td_ludo/experiments/depth/gen_history_states.py` | genuine frame-pair eval data (exact V15RichPlayer convention; 0/1536 duplicate frames) |
| `td_ludo/experiments/depth/rl_recruit/` | F1b: 79-point trajectory, profiles.json, train log |
| `td_ludo/experiments/depth/rl_grow5/` | F2: 28-point trajectory (live experiment, paused at 14.7K) |
| `td_ludo/experiments/depth/insert_layer_rl.py` | F2 pre-registration + surgery |
| `td_ludo/experiments/depth/analyze_recruit.py` | share-of-total scoring vs pre-registration |

Eval sets: `experiments/moe/cache_moe_120k.pt` (T=1, 120K states) and
`experiments/depth/eval_states_T2.pt` (T=2, generated). The T=1 cache does not
record which teacher checkpoint produced it — a provenance gap we noted but
did not need to resolve; record teacher paths in future caches.

## 5. Caveats ledger (whole report)

- All RL results are **n=1 runs** on one machine; multi-seed RL was not
  feasible. Trajectory shapes (79 and 28 points) give internal consistency,
  not seed variance. Your §2.3 rule is knowingly waived where stated, nowhere
  silently.
- Eval-state distributions: T=1 cache (teacher-era self-play) and T=2
  random-rollout pairs. Within-comparison data is always identical across the
  models compared; absolute magnitudes are distribution-dependent.
- F2's optimizer restart perturbs all layers at insertion; controlled for by
  trend continuity of the original layers, not eliminated.
- The F1b/F2 opponent mix (self 60 / scripted Expert 40) differs from the
  champion's neural pool; direction of F1b reproduced anyway, but magnitudes
  may differ.

*Everything above, including the two retracted conclusions and the null, is
recorded rather than cleaned up — per your own GROW.md precedent, which we
found more useful than any of its positive results.*
