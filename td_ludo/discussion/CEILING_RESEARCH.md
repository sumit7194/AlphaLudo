# Breaking the ~85% Ceiling — Research Synthesis (2026-06-20)

Three parallel research agents (codebase consolidation + 2× literature, fact-checked
vs primary sources). They converged hard. This doc is the durable record; the
discussion that follows it picks the path.

## 1. The ceiling is VARIANCE + ROSTER, not skill — confirmed by our data AND theory
- Our **Exp 49**: 6 snapshots over 2.74M games all within **4.2pp of 50%** H2H; the
  eval-PEAK (82.8%) was the H2H-WEAKEST (48.6%) → eval anti-correlates with strength
  near the ceiling. v15.2/v13.6/v13.5 tournament: all within ~3pp.
- Theory says this is the **expected fingerprint of near-optimal play in a high-variance
  dice race**, NOT a stall: Pig equilibrium head-to-head = 0.4999997 vs 0.5000003; Ludo
  mirror matchups 48–50%; top backgammon engines (XG≈GNUbg≈Sage) statistically
  indistinguishable. "All snapshots are coin-flips" = we climbed the curve.
- Two ceilings: **A (intrinsic dice ceiling)** — even a huge skill gap caps single-game
  win-rate ~60–75% in dice races (backgammon master vs beginner ≈75%/game; luck alone
  predicts 97.9% of match outcomes). **B (roster-exploitation ceiling)** — our 85% is vs
  a fixed roster = maxed exploitation of THOSE opponents. We're at B. B is movable.

## 2. THE key split — raises-ceiling vs only-reduces-variance
| RAISES win-rate (policy-improvement operators) | ONLY reduces variance (provably unbiased) |
|---|---|
| **Inference-time search** (expectimax/MCTS w/ net) | **AIVAT**, control variates, **paired-dice eval** |
| Opponent diversity / **exploiters** | common random numbers, more games |
| Lead-aware features (learns risk-modulation) | — |
- **This kills AIVAT (our parked "GAE stage 2") as a ceiling-breaker.** AIVAT is *provably
  unbiased* — by the authors' proof it is "truthful: a player cannot appear to do better
  by changing their play." It sharpens the ruler; it can't lengthen what's measured. (It
  IS great for cheaper/cleaner measurement — see lever 2.)

## 3. THE big untried lever — inference-time search (RAISES win-rate)
- **We NEVER tried net+search at play time.** Exp 24/45 used search as *training targets/
  distillation* (failed); we benchmarked net-vs-search-BOTS (net wins). The trained nets
  were only ever evaluated as raw argmax policies. Search is a *different, stronger
  operator* on the same net — AlphaGo Zero: raw net 3055 Elo → +MCTS 5185 (a 2130 gap).
- **Backgammon (closest analog) — search = BLUNDER ELIMINATION, modest but real:**
  TD-Gammon 2-ply −0.163 → 3-ply −0.050 ppg; blunders/game 0.20→0.04; gnubg raw net
  agrees with 2-ply ~97% of the time. Expect single-to-low-double-digit pts, concentrated
  in tactical spots, saturating after 1–2 plies (dice blur deeper states).
- **Ludo is EASY for this:** die has 6 faces → exact expectation per chance node (cheaper
  than backgammon's 21 roll-pairs). Use depth-limited **expectiminimax: our value net at
  the leaves + policy net as prior**. Don't need Stochastic MuZero's *learned* chance model
  (dice law is known). **Use a Gumbel-style root** (Gumbel MuZero) → search GUARANTEED to
  improve on the raw net even at few sims (plain PUCT w/ few sims is NOT guaranteed).
- **Honest tension:** our **Exp 45** found "search ≈ converged net, nothing to find, SNR
  ~0.25, search amplifies noise." That was 2-ply *distillation*, not inference rollouts +
  Gumbel root. AND v15.2's **token-disambiguation weakness** (same-cell tokens) is exactly
  a tactical blind spot search could fix. **Decisive cheap test: searched-net vs raw-net at
  identical weights.** Wins → free skill on the table. Doesn't → bottleneck is value-net
  accuracy, not absence of search.

## 4. Lever 2 — paired-dice (common-random-numbers) EVALUATION (fixes measurement)
- Replay IDENTICAL dice for both policies → shared dice cancel → resolve the same skill
  gap in FAR fewer games. A few lines of code, unbiased. Directly dissolves our
  "everything is 50/50" fog — it can't raise win-rate but it lets us SEE edges the
  independent-dice metric hides. Arguably do this FIRST (tells us if a gap even exists).

## 5. Lever 3 — single EXPLOITER diagnostic (resolves true-ceiling vs blind-spot)
- KataGo: a self-play-dominant net was beaten >99% by a dedicated adversary playing
  off-distribution, with <14% of training compute (and it's whack-a-mole). We STARTED this
  (Exp 50) but never finished. Train one exploiter (rewarded only for beating current best,
  allowed to play weird): crushes us → real headroom (exploitable hole); can't beat ~55% →
  genuinely near the transitive ceiling. One cheap experiment, big information.

## 6. Lever 4 — lead-aware features (modeling; NOT a risk objective)
- Decision theory is rigorous (Dubins–Savage bold-play: behind→gamble, ahead→safe;
  backgammon match-equity divergence). BUT for a BINARY win reward, EV-max ALREADY = win-
  prob-max, so a CVaR/risk objective is REDUNDANT. The real lever: ensure the encoder is
  CONDITIONED on **lead-margin** + **remaining-race-length** so the net LEARNS safe-ahead/
  bold-behind on its own. Ludo: a context-switching strategy beats pure strategies >90%.
  If our features under-represent lead/race-length, that's a concrete gap to close.
- Distributional RL (C51/QR-DQN/IQN): mostly a dead-end for win-rate (benefit is an
  auxiliary-task representation effect, can hurt; risk distortion = mixed/unexplained).

## DEAD ENDS (don't spend cycles here)
- AIVAT / any variance reduction as a *ceiling-breaker* (unbiased by proof).
- CVaR/risk objective on a binary win reward (redundant).
- Large diverse population / full league (defeats non-transitivity, which a near-transitive
  2-player perfect-info dice race mostly lacks; a SINGLE exploiter diagnostic is the keeper).

## Recommended sequence (to discuss)
1. **Paired-dice eval harness** (cheap, sharpens everything) — is there even a measurable gap?
2. **Inference-time expectimax + Gumbel root** on v15.2; **searched-vs-raw at identical
   weights** = the decisive test. Highest upside, directly attacks Ceiling B.
3. **One exploiter** — resolves "are we truly at the ceiling or is there a blind spot?"
4. (modeling check) audit whether the encoder sees lead-margin + remaining-race-length.

### Key papers
Stochastic MuZero (ICLR'22); Tesauro&Galperin online MC search (NIPS'96, the design
template); Tesauro backgammon (AI 134, 2002); Gumbel MuZero (ICLR'22); AlphaGo Zero
(Nature'17, "MCTS = powerful policy improvement operator"); AIVAT (AAAI'18, "provably
unbiased/truthful"); KataGo adversarial (2211.00241); AlphaStar league (Nature'19);
Spinning Tops (NeurIPS'20).
