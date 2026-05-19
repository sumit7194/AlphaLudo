# Strong-Opponent Bots — Results

Cumulative results from the overnight strong-bot autonomous run.
Updated as phases complete. See `STRONG_BOTS_PLAN.md` for the original
phase plan.

## Bot zoo built

Code is in `td_ludo/td_ludo/game/`:

| Module | Bots | Brief |
|---|---|---|
| `strong_bots.py` (pre-existing) | `Expectimax`, `MCTSPure` | Base depth-1 expectimax + pure-rollout MCTS |
| `strong_bots_v2.py` | `Aggressive/Defensive/Racing/MinimaxExpectimax` | Personality variants of base Expectimax |
| `strong_bots_depth2.py` | `Depth2Expectimax`, `Depth2{Aggressive,Defensive}Expectimax` | Same scoring, depth-2 search |
| `strong_bots_mcts_prior.py` | `MCTSExpertPrior`, `MCTSExpectimaxPrior` | MCTSPure with informed prior |
| `strong_bots_adaptive.py` | `AdaptiveExpectimax`, `VoteExpectimax` | Phase-conditioned scoring; ensemble Borda vote |
| `strong_bots_rule.py` | `MaxCapture`, `TwoStack`, `HomeRush`, `StackHomeRush` | Pure-rule heuristics, no search |

15 new bot variants total. All conform to the `BOT_REGISTRY` interface
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

(pending — arena running)

## Phase 6 results (extends with adaptive + vote + 4 rule-based)

(pending — will run after 3+4 finishes)

## Recommended training mix

(pending — will be proposed once final arena completes)

Format will be:
```
SelfPlay         45%
<top-N bots>     55% distributed by ELO + diversity, weighted by speed
```

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
