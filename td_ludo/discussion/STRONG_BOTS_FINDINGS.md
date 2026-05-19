# Strong-Opponent Bots — Findings

Curated insights from the strong-bot development + benchmarking work.
Companion to:
- `STRONG_BOTS_PLAN.md` — the 7-phase roadmap (intent + design)
- `STRONG_BOTS_RESULTS.md` — raw numbers from every arena run

This doc is for distilled lessons: **what we learned about Ludo bot
design** that should inform the next iteration, RL training mixes, or
future bot-development decisions.

## Headlines (one line each)

1. **`Depth2Expectimax` is the only new bot that clearly beats your
   existing `ExpectimaxBot`** — 57.5% h2h, +7.5pp at 24× compute cost.
2. **`MCTSExpectimaxPrior` is the strongest non-trivial bot built**
   (77% vs Expert) but at ~1s/move it's too slow for production RL
   training; useful as a benchmark ceiling.
3. **Depth-1 expectimax + personality scoring gives real diversity at
   zero compute cost** — 3 of 4 personality variants beat base
   Expectimax head-to-head 52-54% in Phase 1.
4. **Phase-conditioned `AdaptiveExpectimax` is *worse* than coherent
   single-strategy** (60.7% WR vs 75.3% for base) — phase-switching
   loses more than it gains.
5. **Vote ensembles match but don't exceed** the strongest scorer in
   the ensemble. Useful for diversity (5 scorers mixed = hard to
   pattern-match), not for raw strength.
6. **More MCTS sims alone don't beat informed prior**. MCTSPure(50) vs
   MCTSExpertPrior(30) — the prior matters more than the search budget.
7. **Rule-based bots are weak but mutually independent**. The 4
   rule-bot variants are within ~5pp of each other but use entirely
   different decision rules — perfect for "low-difficulty varied
   training stimulus".

## What works

### Depth-2 expectimax

Real gain over depth-1, but with steeply diminishing returns.

| Metric | Depth-1 | Depth-2 | Δ |
|---|---|---|---|
| WR vs Expert | 67% | 75.5% | +8.5pp |
| WR vs depth-1 | — | 57.5% | +7.5pp |
| Compute per move | ~50ms | ~1.2s | 24× |

Depth-2 is your **practical strongest drop-in** — strictly stronger
than depth-1 Expectimax at acceptable inference cost.

### Informed-prior MCTS

The biggest single-bot improvement we measured. `MCTSExpectimaxPrior`
takes a small number of sims (30 default) but seeds each expansion with
Expectimax's choice as the prior — concentrating search on already-good
actions.

| Bot | WR vs Expert | Cost/move |
|---|---|---|
| MCTSPure (50 sims, uniform prior) | 63% | ~70ms |
| MCTSExpertPrior (30 sims) | 53.5% | ~150ms |
| MCTSExpectimaxPrior (30 sims) | **77%** | ~1s |

Note that **MCTSExpertPrior barely outperforms its prior** (Expert)
because the prior IS Expert. The real lift comes from using a
better-than-Expert prior (Expectimax) and amortizing search over it.

### Personality scoring (depth-1)

Same lookahead structure, different scoring functions — gives 4
opponents with the same compute cost but distinct decision preferences.

From the Phase 1 round-robin (200 g/pair):
- AggressiveExpectimax beats base 52%
- DefensiveExpectimax beats base 53.5%
- MinimaxExpectimax beats base 52.5%
- RacingExpectimax loses to base 35% (single-agent racer, exploitable)

Aggressive vs Defensive is asymmetric (55/45 — aggressive style edge),
which is what you want for RL training variety.

## What doesn't work

### Adaptive (phase-conditioned) scoring

**Hypothesis**: switch scoring based on game phase (early=race,
mid=aggressive, late=defensive) → opportunistic across the game.

**Reality**: 60.7% WR vs 75.3% for base Expectimax — adapting *loses*
~15pp. The early-game racing mode gives up engagement before positions
stabilize; by the time the bot switches to aggressive, opp tokens are
already too far ahead.

**Lesson**: a coherent single objective beats a clever-but-mistuned
phase switch. If we wanted to try this again, the phase boundaries
would need careful empirical tuning — the current `tokens-in-base` heuristic
is too crude.

### Borda-vote ensemble (VoteExpectimax)

**Hypothesis**: 5 different scorers vote → ensemble averages out
quirks, beats any single scorer.

**Reality**: VoteExpectimax matches base Expectimax (48% h2h, within
noise). The ensemble's average preference is no better than the best
single scorer. Cost is 5× per move.

**Lesson**: Ensembles work when individual members make uncorrelated
mistakes. Here all scorers see the same lookahead tree — their errors
are highly correlated. The vote provides diversity (harder to
pattern-match) but not strength.

### Depth-2 + aggressive scoring

**Hypothesis**: depth-2 lookahead + aggressive scoring = strongest
combo.

**Reality**: Depth2Aggressive vs base Expectimax = 50/50. Aggressive
scoring's added noise (heavier weight on opp's exposure estimate)
cancels the depth-2 gain.

**Lesson**: Search depth amplifies the scoring function's noise. Use
depth-2 only with the base (or defensive) scoring; don't combine with
aggressive personality.

### MCTS sims without prior

**Hypothesis**: doubling MCTS sims (50→100) closes the gap to
Expectimax.

**Reality** (preliminary, 30 g/pair = ~10pp std error):
MCTSHighSim (100 sims) lost to Expert 43% while MCTSPure (50 sims)
won 63%. Even allowing for noise, more sims didn't help meaningfully —
and Expectimax still beats both decisively (77% vs MCTSPure).

**Lesson**: For Ludo's branching factor (≤4) and game length, raw
random-rollout sims hit diminishing returns very fast. The informed
prior is doing the real work in `MCTSExpectimaxPrior`.

## Cost-vs-strength frontier

Plotting roughly (WR vs Expert on Y, ms/decision on X log-scale):

```
WR%
 80 ┤                                          ● MCTSExpectimaxPrior
    │                                       ● Depth2Expectimax
 75 ┤                                       ●  Depth2Defensive
    │                                  ● Expectimax
 70 ┤                              ● DefensiveExpectimax
    │                       ● Aggressive
 65 ┤                  ● MCTSPure(50)        ● Depth2Aggressive
    │                                        ● VoteExpectimax (slow)
 60 ┤            ● Adaptive
    │              ● Blockade
 55 ┤        ● MCTSExpertPrior   ● MCTSHighSim(100)
    │
 50 ┤  ● rule bots (35-50% vs Expert)
    └──────┬───────┬────────┬────────┬────────┬────────┬────
          0.1     1ms     10ms     50ms     200ms     1s    ms/move
```

**Pareto frontier** (best-strength-for-cost):
- ≤1ms: rule bots (MaxCapture, TwoStack, etc.) — for diversity only
- ~50ms: AggressiveExpectimax — best zero-extra-cost upgrade
- ~50ms: Expectimax (your existing) — coherent baseline
- ~200ms: Depth2Expectimax — best practical strong bot
- ~1s: MCTSExpectimaxPrior — strongest measured (research tier)

## Style diversity findings

Beyond raw strength, the bots show genuine style asymmetry — useful for
RL training because a model overfit to one style won't crush all of them.

| Bot | Distinct play texture | Exploit if model learns it |
|---|---|---|
| MaxCapture | Captures whenever possible | Bait it into bad trades |
| TwoStack | Builds blockades on safe squares | Avoid the blockade squares |
| HomeRush | Advances most-advanced token | Capture the lone leader |
| Racing | Never engages | Capture freely |
| Aggressive | Heavy capturing, low fear of retaliation | Set capture traps |
| Defensive | Refuses to advance, stalls | Forces draws / time-out |
| Blockade | Positional, slow | Out-race it |
| Vote | Ensemble of 5 scorers | Mixed, hard to pin down |

For RL training, the **diversity dimension matters more than the
strength dimension** — see the recommended mix in `STRONG_BOTS_RESULTS.md`.

## Recommendations for next iteration

### If you want a stronger single bot
Build `Depth3Expectimax` — Phase 5 was skipped, but pushing depth
further has clear precedent (Phase 3 worked). Cost: ~24× depth-2 =
~30s/move; only suitable for benchmark, not training.

### If you want broader playstyle coverage
Add `AntiPatternBot` — designed to punish observed model biases (e.g.,
"V13.5 has same-token stickiness; the bot exploits that"). Would
require analyzing actual model decisions first to identify exploitable
patterns. Most bespoke; highest ROI for RL.

### If you want compute efficiency
Port `Depth2Expectimax` to C++. The Python depth-2 takes ~1.2s/move;
C++ should be ~50-100ms, closing the gap with depth-1 cost while
keeping the strength advantage.

### What to NOT build
- More phase-conditioned variants — Adaptive failed clearly
- Bigger vote ensembles — Vote matched but didn't exceed
- Higher-sim MCTSPure variants — prior matters more than sims

## Reproducing the findings

```bash
cd td_ludo
# Phase 1: personality variants
python -m experiments.strong_bot_arena.run_round_robin \
  --bots Expert,Expectimax,AggressiveExpectimax,DefensiveExpectimax,RacingExpectimax,MinimaxExpectimax \
  --games-per-pair 200 --output runs/arena_phase1_200g.json

# Phase 6 focused (rule bots + Adaptive)
python -m experiments.strong_bot_arena.run_round_robin \
  --bots Expert,Expectimax,AggressiveExpectimax,AdaptiveExpectimax,MaxCapture,TwoStack,HomeRush,StackHomeRush \
  --games-per-pair 100 --output runs/arena_phase6_focused.json

# Bonus (Vote + Blockade)
python -m experiments.strong_bot_arena.run_round_robin \
  --bots Expert,Expectimax,AggressiveExpectimax,VoteExpectimax,BlockadeExpectimax \
  --games-per-pair 50 --output runs/arena_bonus.json

# Analyze any arena JSON
python -m experiments.strong_bot_arena.analyze runs/<file>.json
```

The Phase 3+4 arena (Depth2 + MCTS-prior) requires several hours due
to MCTS pair runtimes. Partial result at `runs/arena_phase3_4_200g.json`
(16/55 pairs) is resumable — re-run the same command and it skips
cached pairs.
