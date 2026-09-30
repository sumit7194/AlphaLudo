# Squeezing the last bit out of v15.2 — Research Synthesis (2026-06-20)

Codebase audit + fact-checked literature. Target: small, non-systematic
move-level suboptimalities ("could play a little better"), NOT a systematic flaw.
We're at the variance ceiling, so honest expected magnitude is **sub-percent to
low-single-digit** win-rate — gains show best as reduced *equity-lost-per-decision*.

## The coherent diagnosis (why the small suboptimalities persist)
The v15.2 **value head is trained ONLY on terminal win/loss (BCE)** — a single
sigmoid scalar off the CLS token. It never sees "this state is *slightly* better
than that one." So it's **near-flat in non-tactical positions** — which is the
single root cause of BOTH:
- the **2-ply search null** (a flat leaf evaluator gives search nothing to rank), and
- the **residual small suboptimalities** (the policy's move-ranking has no fine
  value signal to sharpen it).
Fix the value signal → potentially sharpen both. (codebase: v15_trainer.py:288
BCE on terminal z; models/v15.py:115 scalar value MLP, CLS-only.)

## THE prerequisite — a measurement harness (do this FIRST)
We literally cannot see sub-pp gains: current eval = 2 bots, independent dice,
±1.5pp noise → can only detect 2+pp. Detecting a 1% edge needs ~40K games
un-paired. The fix (each is unbiased, won't change true win-rate, just sharpens
the ruler):
1. **Common Random Numbers / paired dice** — same dice sequences for both agents,
   both sides. Variance of the *difference* collapses (backgammon: ~18-25× fewer
   games). We already used paired dice ad hoc in the H2H; make it the standard.
2. **Luck-subtraction control variate** — for each roll subtract (post-roll value −
   avg value over all 6 rolls); luck averages to 0. We HAVE a value net to compute
   it. Backgammon: ~25× per-game. Most transferable dice-game trick.
3. **Per-decision equity-loss metric** — score a model by mean equity lost per
   *unforced* decision vs a rollout-reference best move (how backgammon ranks
   near-equal bots without millions of games). EVERY decision is a datapoint →
   hundreds of games yield thousands of samples. **This directly operationalizes
   "could play a little better"** — it quantifies exactly how much v15.2 leaves on
   the table per move.
4. **SPRT** — sequential test, stop as soon as the result is clear.
Without this, every squeeze attempt is unfalsifiable. It may ALSO reveal that some
existing snapshots are genuinely (tiny-bit) stronger — invisible until now.

## The real squeeze — sharpen the value head's move-ranking
Best evidence-to-effort (Tesauro 5× error reduction in backgammon; Willemsen
"better + faster"; the current terminal-z target is "high variance AND high bias"):
1. **Richer value targets** — move from pure terminal-BCE to **TD(λ) / n-step
   bootstrapped** value targets (TD-Gammon used λ=0.7), or rollout-based value
   labels. Top pick.
2. **Decorrelate value training data** — train value on one/few positions per game
   (AlphaGo: correlated positions sharing one terminal label → value MSE 0.37; one
   position/game → 0.234). Nearly free; may fix a chunk of the flatness.
3. **(arch, optional) richer value output** — pool the 225 node embeddings into the
   value MLP (currently CLS-only / board-blind); or a W/D/L-margin **bucket** head
   (Lc0) that carries ordinal info the scalar averages away. NOTE distributional
   RL per se is NOT a clean fix (Lyle: = expected RL in tabular/linear, can hurt).
4. **(arch, strong direct evidence) dueling / per-action advantage head** — the one
   architecture result directly about ranking near-equal moves (benefit grows with
   #near-equal actions). Non-trivial to graft onto AlphaZero-style; consider later.

## Search — only AFTER a better value head, and rollouts not 2-ply
The reliable search payoff is **deep sampling/rollouts**, not 2-ply minimax
(backgammon 2→3 ply only +0.02-0.08 ppg; the big wins were Tesauro rollouts, 5×).
For a dice game: better value head + **shallow truncated rollouts** at inference.
Modest, compounding gain — not an AlphaGo step change.

## Mostly noise at the ceiling (cheap insurance only)
EMA/SWA weight averaging (~1% in vision, no game-Elo evidence), checkpoint merging
(decays fast), low-LR annealing tail (short-lived, overfit risk). Run only because
near-free; keep only what SPRT confirms. KataGo's aux-target tricks are training-
*efficiency* (reach the ceiling faster), not ceiling-raising — except aux value/
score heads which densify signal and CAN sharpen value (worth folding into #2).

## Recommended sequence
1. **Measurement harness** (CRN + luck-subtraction + per-decision equity-loss +
   SPRT). Prerequisite + quantifies "could play better." Cheap, local.
2. **Decorrelate value data** (nearly free) → measure.
3. **Richer value targets (TD(λ)/n-step)** → measure. Best shot at a real gain.
4. (optional) enrich value head (node-pool / bucket / dueling) → measure.
5. (if value sharpened) shallow rollouts at inference → measure.
Expect small but real reductions in equity-lost-per-decision that compound.

### Key papers
Tesauro&Galperin online MC search (NeurIPS'96, 5× error reduction);
Willemsen/Baier/Kaisers value-target study (NCA 2022); AlphaGo value-net
data-correlation (Nature'16, MSE 0.37→0.234); Lyle/Bellemare/Castro distributional
≡ expected (AAAI'19); Wang et al. dueling networks (ICML'16); CRN variance
reduction (Stout&Goldie); backgammon luck-subtraction (Montgomery/GammOnLine);
SPRT/Fishtest pentanomial; Jones scaling (NeurIPS'21, strength sigmoid in compute).
