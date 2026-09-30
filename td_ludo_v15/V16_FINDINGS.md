# V16 — two properties per connection: findings

**Status as of 2026-08-06.** `v16_routed` at ~215,000 games and still running.
`v16_mixed` (the control arm) has NOT run. Pre-registration:
`V16_PREREGISTRATION.md`. Mech-interp detail:
`AlphaLudo-MechInterp/V16_MECH_INTERP_SUMMARY.md`.

---

## 1. The idea

Owner's, verbatim:

> currently in a traditional NN we increase or decrease the connection strength
> based on if you want to boost the behaviour or suppress it... what if those
> connections have 2 properties... the scalar can be adjusted based on terminal
> win/loss signal, while the PHASE could change with OTHER SPARSE SIGNALS like
> cuts, or home, or others.

Implemented as `w = a·(1 + tanh(b))` on every bulk weight of the V15.2
GraphTransformer trunk, with

```
a  ←  gradient ONLY from the RL objective   (PPO policy + value)
b  ←  gradient ONLY from the AUX objective  (capture-available, in-danger)
```

Routing is two forward passes with opposite detaching — exact, not approximate.
Verified: MAIN loss → |∇a| 1e-4, |∇b| **0.0**; AUX loss → |∇a| **0.0**,
|∇b| 24.8.

---

## 2. Why the first attempt didn't count

`td_ludo/experiments/twosignal` ran this idea for 212,608 games. That run is
**uninterpretable** — the harness diverged from the v15.2 champion's pipeline in
~10 independent ways:

| | v15.2 champion | twosignal harness |
|---|---|---|
| opponents | 20% self, 65% strong neural, 15% depth-2 search | **100% self-play** |
| algorithm | PPO, clip 0.2, 2 epochs | REINFORCE, single pass |
| advantage | GAE λ0.95, γ0.999, normalised | r − V, no discount |
| entropy bonus | 0.03 | **none** |
| learning rate | 1e-5 | **3e-4** |
| value target | BCE to win/loss | MSE to {0,1} |
| whose moves train | student seat only | both seats |

Its measured outcome, for the record: it improved against weak scripted bots
(0.515 → 0.630 peak) while getting **worse** head-to-head against the v15.2
champion it was distilled from — 0.380 → 0.307 over 1,000 games (3.4σ). It
drifted; it did not learn. v15's own code predicts this in a comment written
long before:

> Weak bots … are harmful past the SL ceiling, model just learns to abuse their
> predictable mistakes instead of refining strong play.

**Lesson recorded:** a bot-eval gain is not evidence of strength. Every claim
here is anchored to an external fixed reference.

---

## 3. V16's design — one variable

v16 reuses `train_v15_rich.py` + `V15RichTrainer` as the **same code path**, not
a copy. `V16RichTrainer` overrides exactly two hooks (`_mode_main`,
`_aux_backward`). PPO/GAE/entropy/clipping/step cadence are inherited unmodified,
so the arms cannot drift apart.

**Init.** Both arms start from the champion's own stage-1 RL checkpoint
(`v152_rl_stage1_G261K/model_best.pt`, eval 0.796). Because `b`=0 ⇒ tanh(0)=0 ⇒
`w`=`a`, v16 begins **numerically identical** to it — verified
`max|v16 − v15| = 0.000e+00`, asserted at every launch. No init confound.

**Reference curve.** The champion went **0.796 → 0.8465** from this exact
checkpoint under this exact recipe. That is what "working" looks like.

**Curriculum** (owner's call, amendment in the pre-registration):
stage 1 = Expert 25 / Heuristic 20 / self 25 → stage 2 = the champion's strong
pool (v13.2 30 / v13.6 35 / self 20 / Depth2Expectimax 15).

---

## 4. Training result — routed does not reproduce the champion's gain

| | eval (2,000 games vs Expert+Heuristic) |
|---|---|
| init | 0.7960 |
| **champion from that init** | **0.8465** |
| v16 stage 1 (n=5, 10–50k) | 0.7926 |
| v16 stage 2 (n=16, 60–210k) | 0.7831 |
| v16 best ever | 0.8080 @ 60,001 games |
| v16 latest | 0.7680 @ 210,000 games |

```
stage-2 trend                  −1.81 pp per 100k games (n=16)
first 5 of stage 2 (60–100k)    0.7942
last  5 of stage 2 (170–210k)   0.7778
difference                     −1.64 pp, 1.88σ
5 of the last 6 evals below the init
```

**The 60k peak has not been beaten in 150,000 subsequent games**, over a span
where the champion was still climbing. v16 routed sits −2.8 pp below its own
starting point and −7.9 pp below the champion.

Diagnostics stayed healthy throughout (entropy ~0.37 held by the 0.03 bonus,
clip fraction ~0.02, KL ~0.002), so this is not instability — it is a slow,
consistent loss of ground.

---

## 5. Mech-interp — what the two weights actually do

Two new experiments, in `AlphaLudo-MechInterp` (`v16` is now a registered
variant, so all 12 existing experiments also run against it unchanged).

### Experiment 13 — anatomy (360 stratified states)

**`b` avoids the decision layer.** Mean |tanh b| by layer: 0.0356 / 0.0330 /
0.0395 / **0.0138**. Layer 3's FFN is barely touched (`ffn.3` = 0.0044, 11% of
gates moved). V15.1's layer-knockout identified layer 3 as the **decision
crystallizer** (100% logit-lens agreement, knockout KL 0.129). Nothing told the
aux signal to avoid it.

**`b` targets the weights RL made important.**
```
corr(tanh b, a)       +0.0007   → no sign preference
corr(|tanh b|, |a|)   +0.5148   → strongly targets LARGE-magnitude weights
```

**`b` owns 15.5% of all weight change** since init
(‖RL term‖ 65.7 vs ‖AUX term‖ 12.1, decomposing
`w_f − w_i = (a_f−a_i)(1+tanh b_f) + a_i·tanh b_f`). Peak 29.3% at
`layers.2.attn.out_proj`.

**`b` is behaviourally large and late-game.** Zeroing it flips **25% of
decisions** (top-1 agreement 0.753). KL 0.212 late-game vs 0.016 early — 13×.
Influence is front-loaded by layer (0 > 1 > 2 > 3) even though magnitude peaks
at layer 2.

*Not trusted:* the dose-response sweep is non-monotonic (KL at b×0.25 exceeds
b×0), which should not happen for a smooth effect. Likely noise at 360 states;
re-run at ~2,000 before reading the fine ordering.

### Experiment 14 — danger probe (6,000 real self-play states)

| CLS site | champion | v16 full | v16 `b`=0 | from `b` |
|---|---|---|---|---|
| in_danger | 0.8739 | **0.9985** | 0.7666 | **+0.2319** |
| capture_available | 0.7826 | **0.9505** | 0.6913 | **+0.2592** |
| *num_tokens_out* (control) | 0.8991 | 0.9192 | 0.9494 | −0.0302 |
| *my_progress* (control) | 0.9441 | 0.9699 | 0.9583 | +0.0116 |

**The routing mechanism works exactly as designed.** The tactical information
lives in `b`: remove it and danger decodability drops 23 points, below even the
champion. Controls do not move — `num_tokens_out` gets slightly *worse* with
`b` — so the gain is tactical-specific, not general representation quality.

**Caveat (stated, not buried):** v16 was trained with an aux head reading the
CLS to predict these exact labels, so "CLS linearly encodes danger" is close to
what the objective directly optimised; the champion never had that pressure.
What survives the circularity: the `b`=0 ablation localising the information to
the second property, the controls failing to move, and the dissociation below.

---

## 6. THE HEADLINE — representation and behaviour dissociate

| | champion | v16 routed |
|---|---|---|
| danger decodable from CLS (AUC) | 0.874 | **0.999** |
| eval win rate | **0.8465** | 0.768 |

Same init, same pipeline.

`cogmap/CLS_RESULTS.md` proposed that the champion is limited by its poor
capture-risk encoding (R²≈0.35). **This run falsifies that premise.** The
representation was fixed essentially completely and play got *worse*.

That is worth more than a win would have been: it rules out a whole line of
attack ("teach it to see danger and it will play better") rather than adding
one more inconclusive result.

### The generalisable lesson

Separating **gradients** does not separate **function**. `w = a·(1+tanh b)`
guarantees no aux gradient reaches `a` and no RL gradient reaches `b` — verified
to 0.0 — but in the forward pass `b` still *scales* `a`. The two objectives
compose multiplicatively no matter how cleanly the backward pass is split, and
Experiment 13 shows where that bites: `b` concentrates on exactly the
high-magnitude weights the primary objective depends on.

Any future version of this idea needs the second property to act on a subspace
the primary objective does not occupy — additive-and-orthogonal rather than
multiplicative-and-aligned.

---

## 7. Methodological finding (affects the whole MechInterp project)

`collect_states_stratified` produces states where **71% have ZERO opponent
tokens on the board** (mean 0.39). It buckets "phase" by the *current player's*
deployed tokens only, so a "late" state can have an opponent that never left
base. In-danger and capture-available become structurally impossible — the first
run of Experiment 14 returned base rates of 1.6% and **0.0%**, with probe AUCs
of 0.99+ on ~12 positives.

`run_danger_probe.py --collector selfplay` plays real 2p games and reproduces
the training distribution (2.17 opponent tokens, 29% danger, 5.8% capture).

**Any earlier experiment in that project that probed an opponent-dependent
concept should be re-checked against this.**

---

## 8. What is NOT yet established

**P1 in the pre-registration is untested.** It compares routed against
**mixed**, and `v16_mixed` has not run. Experiment 14 shows routing installs the
representation; it cannot say whether *routing* caused the play degradation
versus the auxiliary objective in general, or the two-property architecture
itself.

Three arms would close it:
1. `v16_mixed` — both losses reach both properties (already scripted:
   `./launch_v16_mixed.sh`)
2. `v16 --aux-coeff 0` — two properties, no aux signal at all; separates the
   architecture from the objective
3. re-run Experiments 13 + 14 on each; `mixed`'s gate anatomy should look
   materially different if routing does anything distinctive

**Also open:** the aux head's own readout was not measured (the MechInterp
adapter's forward does not pass `want_aux`), and the Experiment 13 dose-response
needs re-running at higher n.

---

## 9. Operational notes

- `best_eval_wr` was **not restored on resume** in `train_v15_rich.py` — after
  every restart the first eval trivially beat 0.0 and overwrote `model_best.pt`.
  Harmless on an uninterrupted VM run; destructive under frequent power cuts.
  Fixed 2026-08-05; `model_best.pt` now holds the genuine peak.
- `acquire_train_lock()` stores a pid and tests liveness with `kill(0)`, but a
  reboot recycles pids — so a stale lock can read as LIVE forever and silently
  refuse every restart. Cost ~40 minutes on 2026-08-05. Both launchers now clear
  stale locks before starting.
- Stage-1 endpoint archived as `model_stage1_final.pt` (54,476 games) so the
  curriculum boundary is recoverable.
