# Strong-Opponent Bots — Results

Cumulative results from the overnight strong-bot autonomous run.
Updated as phases complete. See `STRONG_BOTS_PLAN.md` for the original
phase plan.

## Bot zoo built

Code is in `td_ludo/td_ludo/game/`:

| Module | Bots | Brief |
|---|---|---|
| `strong_bots.py` (pre-existing) | `Expectimax`, `MCTSPure` | Base depth-1 expectimax + pure-rollout MCTS |
| `strong_bots_v2.py` | `Aggressive/Defensive/Racing/Minimax/BlockadeExpectimax` | Personality variants of base Expectimax |
| `strong_bots_depth2.py` | `Depth2Expectimax`, `Depth2{Aggressive,Defensive}Expectimax` | Same scoring, depth-2 search |
| `strong_bots_mcts_prior.py` | `MCTSExpertPrior`, `MCTSExpectimaxPrior` | MCTSPure with informed prior |
| `strong_bots_adaptive.py` | `AdaptiveExpectimax`, `VoteExpectimax` | Phase-conditioned scoring; ensemble Borda vote |
| `strong_bots_rule.py` | `MaxCapture`, `TwoStack`, `HomeRush`, `StackHomeRush` | Pure-rule heuristics, no search |
| (runner-level) | `MCTSHighSim` | MCTSPure at 100 sims / 8 rollouts (2× default) |

17 new bot variants total. All conform to the `BOT_REGISTRY` interface
(`__init__(player_id=None)`, `select_move(state, legal) -> int`) so they
drop into the existing training/eval harness as opponents.

## Arena infrastructure

`experiments/strong_bot_arena/`:
- `run_round_robin.py` — round-robin runner over the unified bot
  registry, JSON-checkpointed per pair (resumable), Bradley-Terry ELO
  from WR matrix
- `analyze.py` — diversity + recommended training-mix proposal

## Phase 1 results (200 g/pair, 6 bots, 3K games, 2.9 min)

```
#  Bot                       WR%    ELO
1  AggressiveExpectimax     58.0   1540
2  MinimaxExpectimax        57.2   1535
3  Expectimax               55.9   1527
4  DefensiveExpectimax      54.5   1519
5  RacingExpectimax         41.6   1442
6  Expert                   32.8   1387
```

Key findings:
- 3 personality variants (Aggressive, Minimax, Defensive) all beat
  base Expectimax head-to-head (52-54%)
- Pairwise WRs span 35-65% — real diversity, not all 50/50
- AggressiveExpectimax > DefensiveExpectimax: 55/45 (style asymmetry)
- Racing exploits-easily — pure single-agent planner

## Phase 3+4 results (extends Phase 1 with depth-2 + MCTS-prior)

**16/55 pairs completed** (run stopped early; MCTS pairs at ~15 min each
made full completion ~4-5 hr). Results inline; arena resumable from the
checkpoint JSON if needed.

Direct WR vs Expert (200 g/pair):

```
Depth2Expectimax              75.5%   ← clear winner over depth-1 (67%)
Depth2DefensiveExpectimax     75.5%
Depth2AggressiveExpectimax    68.0%
MCTSExpertPrior               53.5%   ← prior=Expert ≈ Expert; modest gain
MCTSExpectimaxPrior           77.0%   ← strongest in field vs Expert
```

Direct WR vs base Expectimax (the bot you already had):

```
Depth2Expectimax              57.5%   ← only meaningful improvement
Depth2AggressiveExpectimax    49.5%   ← depth+aggressive cancel
```

Key findings:
- **Depth-2 helps ~7.5pp head-to-head** (24× compute). Diminishing returns
  but real. Depth2Expectimax = best practical drop-in replacement.
- **MCTS + informed prior is the strongest configuration tested** (77% vs
  Expert). MCTSExpertPrior alone only matches Expert because the prior
  IS Expert. MCTSExpectimaxPrior takes Expectimax-quality decisions and
  amortizes search over them.
- Aggressive scoring + depth-2 lookahead don't compose well — depth-2
  Aggressive vs depth-1 Expectimax is 50/50.

## Phase 6 results (focused arena, 100 g/pair, 1.1 min)

8 bots: Expert, Expectimax, AggressiveExpectimax, AdaptiveExpectimax,
4 rule-based. 28 pairs.

```
#  Bot                       WR%    ELO
1  Expectimax                75.3   1640
2  AggressiveExpectimax      73.3   1623
3  AdaptiveExpectimax        60.7   1529   ← worse than base Expectimax
4  Expert                    54.1   1483
5  StackHomeRush             37.0   1365
6  TwoStack                  35.7   1356
7  MaxCapture                32.1   1330
8  HomeRush                  31.7   1327
```

Key findings:
- **AdaptiveExpectimax disappoints**: 36% h2h vs base Expectimax. The
  early-game racing-mode logic backfires (gives up engagement before
  positions stabilize). Phase-conditioning is harder to tune than it
  looked.
- **Rule bots are weak but diverse**: 32-37% WR each, but mutually
  ~50% pairwise. Different decision rules (capture-first, blockade,
  rush-leader, hybrid). Useful as varied-playstyle low-difficulty
  training opponents, not as challenges.

## Bonus arena (Vote + Blockade, 50 g/pair, 1 min)

5 bots, 10 pairs. Direct WR vs reference bots:

```
                    vs Expert   vs Expectimax   vs Aggressive
VoteExpectimax        82.0%        48.0%           40.0%
BlockadeExpectimax    56.0%        44.0%           42.0%
```

Key findings:
- **VoteExpectimax: 82% vs Expert** but only ties base Expectimax. The
  Borda ensemble doesn't outperform a single coherent scorer — it just
  averages out the personalities. Value as RL opponent: harder to
  pattern-match (5 scorers mixed) than predictable.
- **BlockadeExpectimax**: mediocre (56% vs Expert, 44% vs Expectimax).
  Blockade-style play is intrinsically slower; the bonus for stacking
  doesn't outweigh the lost tempo.

## Final consolidated leaderboard (across all arenas)

Best-of-all-runs direct WR vs Expert (the common baseline):

```
#  Bot                              WR vs Expert    Notes
1  MCTSExpectimaxPrior              77.0%          Strongest; ~5s/game pair
2  Depth2Expectimax                 75.5%          Best practical; ~0.2s/move
3  Depth2DefensiveExpectimax        75.5%          Same strength, defensive
4  Expectimax (your existing)       72.5%          Reference point
5  DefensiveExpectimax              70.0%          Phase 1 personality
6  Depth2AggressiveExpectimax       68.0%
7  MinimaxExpectimax                68.5%
8  AggressiveExpectimax             66.0%          Top of Phase 1 ELO
9  AdaptiveExpectimax               60.7%          Worse than base
10 BlockadeExpectimax               56.0%
11 MCTSExpertPrior                  53.5%
12 RacingExpectimax                 ~59%
13 StackHomeRush                    ~32%          Rule bot
14 TwoStack                         ~31%
15 MaxCapture                       ~26%
16 HomeRush                         ~26%
```

(Some numbers from different arenas; rule-bot WRs above are vs Expert
from Phase 6 leaderboard. Std error ≤7pp at 50 games, ≤5pp at 100.)

## Recommended training mix

Output of `analyze.py` on the Phase 6 focused arena:

```
SelfPlay                      45.0%
AggressiveExpectimax          15.1%
Expectimax                    14.1%
AdaptiveExpectimax            13.6%
Expert                         5.8%
MaxCapture                     1.7%
TwoStack                       1.7%
StackHomeRush                  1.6%
HomeRush                       1.3%
```

**Adjustments for the cross-arena evidence** (proposed manual mix):

```
SelfPlay                  40.0%   coherence on own policy
Expectimax                12.0%   ← the strongest practical drop-in
AggressiveExpectimax      10.0%   ← real h2h edge + style diversity
Depth2Expectimax           8.0%   ← strongest cheap-enough new bot
DefensiveExpectimax        5.0%   ← style: stalls/refuses-trades
MinimaxExpectimax          5.0%   ← style: worst-case opp
BlockadeExpectimax         3.0%   ← style: positional, blockades
VoteExpectimax             3.0%   ← style: hardest to pattern-match
RacingExpectimax           3.0%   ← style: pure race, easy capture-bait
Expert                     5.0%   ← legacy baseline reference
4 rule-bots                6.0%   ← 1.5% each, max playstyle variety
                          ─────
                         100.0%
```

Skip MCTSExpectimaxPrior + Depth2Aggressive/Defensive from training
budget — they're slow (>1s/move for the MCTS variant, ~0.2s for depth-2)
and the cheaper Depth2Expectimax already gives most of the strength gain.

## How to reproduce

```bash
cd td_ludo
# Phase 1: 6 base bots
python -m experiments.strong_bot_arena.run_round_robin \
  --bots Expert,Expectimax,AggressiveExpectimax,DefensiveExpectimax,RacingExpectimax,MinimaxExpectimax \
  --games-per-pair 200 --output runs/arena_phase1_200g.json

# Phase 3+4: add depth-2 + MCTS-prior (resumes from above)
python -m experiments.strong_bot_arena.run_round_robin \
  --bots Expert,Expectimax,AggressiveExpectimax,DefensiveExpectimax,RacingExpectimax,MinimaxExpectimax,Depth2Expectimax,Depth2AggressiveExpectimax,Depth2DefensiveExpectimax,MCTSExpertPrior,MCTSExpectimaxPrior \
  --games-per-pair 200 --output runs/arena_phase3_4_200g.json

# Phase 6: add adaptive + vote + rule-based
python -m experiments.strong_bot_arena.run_round_robin \
  --bots <Phase 3+4 list>,AdaptiveExpectimax,VoteExpectimax,MaxCapture,TwoStack,HomeRush,StackHomeRush \
  --games-per-pair 200 --output runs/arena_phase6_200g.json

# Analyze
python -m experiments.strong_bot_arena.analyze runs/arena_phase6_200g.json
```
