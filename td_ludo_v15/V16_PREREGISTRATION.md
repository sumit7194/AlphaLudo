# V16 — two properties per connection, two learning signals

**Locked 2026-08-04, before any v16 run.** Nothing below may be edited after
the first launch; corrections go in an amendment section at the bottom, dated.

## The idea (owner, verbatim)

> currently in a traditional NN we increase or decrease the connection strength
> based on if you want to boost the behaviour or suppress it... what if those
> connections have 2 properties... the scalar can be adjusted based on terminal
> win/loss signal, while the PHASE could change with OTHER SPARSE SIGNALS like
> cuts, or home, or others.

Formally, every bulk weight in the transformer trunk becomes

    w = a · (1 + tanh(b))

    a  ←  gradient ONLY from the RL objective   (PPO policy + value, win/loss)
    b  ←  gradient ONLY from the AUX objective  (capture-available, in-danger)

What makes this untested: every prior variant in this project family (complex
weights, phase/FHRR, Kronecker, Hadamard) used ONE loss updating BOTH parts.
The distinctive claim is that **different signals own different parameters**.

## Why this run exists (the previous attempt does not count)

`td_ludo/experiments/twosignal` ran the same idea for 212,608 games and is
**uninterpretable**, because the harness diverged from the v15.2 champion's
pipeline in ~10 independent ways: 100% self-play instead of a strong-opponent
mix, REINFORCE instead of PPO, no GAE, no discounting, no entropy bonus, lr 30×
too high, MSE-to-{0,1} value head instead of BCE, both seats trained, 300-move
cap instead of 400.

Its measured outcome, for the record: the model improved against weak scripted
bots (0.515 → 0.630 peak) while getting **worse** head-to-head against the
v15.2 champion it was distilled from — 0.380 → 0.307 over 1,000 games, 3.4σ.
It drifted; it did not learn. That is evidence about the harness, not about the
idea. v15's own code says why, in a comment predating all of this:

> Weak bots … are harmful past the SL ceiling, model just learns to abuse their
> predictable mistakes instead of refining strong play.

## Design

**One variable.** v16 changes the MODEL and nothing else. `train_v15_rich.py`
and `V15RichTrainer` are the same code path for both v15 and v16;
`V16RichTrainer` overrides exactly two hooks (`_mode_main`, `_aux_backward`).
The PPO/GAE math has one implementation, so the arms cannot drift apart.

**Arms** — identical architecture, parameter count, aux head, aux loss, init,
opponents and hyperparameters. Only the mode differs:

| arm | RL backward | aux backward | meaning |
|---|---|---|---|
| `v16_routed` | MAIN → `a` only | AUX → `b` only | the idea |
| `v16_mixed` | JOINT → both | JOINT → both | control |

`mixed` is what separates *routing signals to separate properties* from
*having two properties / an extra auxiliary loss*. Without it, any gap could
just be the parameters or the supervision.

**Init.** Both arms start from the champion's own stage-1 RL checkpoint
(`v152_rl_stage1_G261K/model_best.pt`, eval 0.796, 195k games). `b` = 0 ⇒
tanh(0) = 0 ⇒ w = a exactly, so v16 begins **numerically identical** to that
model. Verified: `max|v16 − v15| = 0.000e+00`. `train_v15_rich.py` asserts this
at startup and refuses to launch if it fails — an init bug would otherwise be
misread later as an effect of routing. There is therefore **no starting
confound**, unlike the twosignal run where routed began ~10 points ahead.

**Pipeline** — the champion's, verbatim: PPO (clip 0.2, 2 epochs, minibatch
256, buffer 64 games), GAE λ=0.95, γ=0.999, terminal-only ±1, entropy 0.03,
lr 1e-5, win-BCE value coeff 0.5, temperature 1.0, 400-move cap,
parallel-games 128. Opponent pool: **v13.2 30 / v13.6-champion 35 / self 20 /
Depth2Expectimax 15**, no weak scripted bots.

**Aux targets** (`rich/v16_aux.py`): `capture_available` and `in_danger`,
computed from the raw state at each student decision. Measured base rates
4.9% / 24.7% under random play and 3.2% / 13.7% in the teacher-game cache — the
rate MOVES as play strength changes, so the BCE positive weight is tracked with
an EMA rather than pinned, or the aux objective would silently re-weight itself
over the run.

**Eval**: 2,000 games vs Expert + Heuristic every 10,000 games — the v15
lineage's own metric, so v16 numbers are directly comparable to the champion's
recorded curve.

## Predictions

- **P1 (primary).** At matched games, `routed` ends with a higher eval win rate
  than `mixed`. **Bar: ≥ 2.0 pp, and ≥ 3 standard errors of the difference**
  computed on the last 5 evals of each arm.
- **P2 (external validity).** `routed`'s best checkpoint beats its own init in
  a 1,000-game H2H versus the v15.2 champion. **Bar: ≥ +2.0 pp over the init's
  score, ≥ 2 SE.** This exists because the twosignal run passed a bot metric
  while failing H2H; a bot-eval gain alone will not be accepted as success.
- **P3 (mechanism).** `gate_abs_mean` grows and then **converges**. Monotone
  unbounded growth is the twosignal failure signature — there, `b` grew
  linearly for 212k games while `aux_loss` flatlined after ~120k, i.e. drifting
  rather than learning.

## Kill conditions

- **K1.** `routed` ≤ `mixed` at matched games, or the gap is under the bar →
  routing adds nothing over simply having two properties + an aux loss. The
  idea reduces to "auxiliary losses help", which is already known.
- **K2.** `routed` fails P2 while passing P1 → we have reproduced the twosignal
  failure mode (better on the proxy, no better in real play) and the eval, not
  the model, is what improved.
- **K3.** `gate_abs_mean` stays ≈ 0 → the aux signal never reached the second
  property. Implementation failure, reported as such, not a result.
- **K4.** Either arm falls below its own init by ≥ 5 pp on the eval and stays
  there for 3 consecutive evals → the pipeline is degrading the champion's
  model and the run is stopped and diagnosed before continuing.

## Both outcomes are reachable

`b` is initialised at 0 and is unbounded, so the gate can stay dead or grow.
The routed−mixed gap is a difference of two win rates, free to take either
sign. Both arms start from an identical, already-strong model, so there is
room to move in either direction.

## Reading rules

1. **No conclusions from the early curve.** Point-to-point scatter on a
   2,000-game eval is ~1.1 pp from sampling alone, and the twosignal run showed
   real checkpoint-to-checkpoint fluctuation of about the same size again.
   Single-step moves under ~4 pp mean nothing.
2. **Judge at matched games**, not matched wall-clock.
3. **H2H is the arbiter**, not the bot eval. See P2.
4. Report negative and null results in full. Every result here, positive or
   negative, goes in `EXPERIMENTAL_ARCHITECTURES_LOG.md`.

## Amendments

### 2026-08-05 — staged opponent curriculum (owner's call, before any scored run)

The pipeline section above describes the champion's **stage-2** pool (v13.2 30 /
v13.6 35 / self 20 / Depth2Expectimax 15). v16 will instead run a curriculum:

- **Stage 1** — Expert 25 / Heuristic 20 / self 25. This is
  `train_v15_rich.py`'s own default pool minus the neural opponent. Run until
  the eval plateaus.
- **Stage 2** — introduce the strong pool (the champion's weights above),
  resuming from stage 1's checkpoint.

Rationale: the champion's launcher was itself a stage-2 run, initialised from a
model that had already had 195k games of RL. Starting v16 directly on the strong
pool skips the stage the champion actually had. The strong-opponent weights are
left at 0 in both launchers so stage 2 is a one-number change.

Ghosts are wired but off (`--opp-weight-ghost 0`). Trigger to enable: the eval
climbing while H2H against a fixed reference does not — the self-play drift
signature from the twosignal run.

**Effect on the predictions:** none. P1/P2/P3 and K1-K4 compare `routed` against
`mixed`, and both arms get the identical curriculum. The only change is that
"matched games" now means matched within a stage. A 1,604-game prefix trained
against the strong pool was discarded so stage 1 starts clean from the init.
