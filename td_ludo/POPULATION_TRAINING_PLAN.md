# Population-Based Training — parked for later

## Context

Our central finding (May 2026): every neural model in the project is
descended from training against the same scripted bot set
(Heuristic, Expert, Aggressive, Defensive, Racing, Random). V13.5,
V13.2, V12.2, V11, V10 — all "siblings" that share the same blind
spots. Training V15.1 against any of them is recursively training
against "the policy that beats this bot set" — which converges to a
shared attractor at ~85% bot-WR.

H2H ranks them all within 3pp of each other across architectures
spanning 588K → 4.4M params. The ceiling is **the bot set itself**,
not the model. Cross-family H2H reveals this clearly: V13.5 (3M CNN)
slightly beats V15 (4.4M GraphTransformer), V15.1 (588K GraphTransformer)
matches V13.5 — but no model meaningfully exceeds the family.

## Why population-based training is the long-term answer

The "different opponent" idea has three tiers, in order of how
qualitatively different the opponents are:

1. **Other neural variants** (V13.5, V13.2, V11, etc.) — shared
   lineage, near-identical blind spots. **This is what we've been
   doing.** Doesn't break the ceiling.
2. **Handcrafted bots beyond the standard 6** (expectimax, MCTS-pure,
   programmatically-strong bots) — qualitatively different from
   neural pattern-matching. May break some blind spots. Cheap to
   build. **Try this first.**
3. **Population-based training (PBT)** — multiple agents trained in
   parallel from different random seeds + reward weights, playing
   each other. The Nash equilibrium of THAT pool is different from
   the V13.5 attractor.

Tier 3 is the AlphaStar / Capture-the-Flag approach. It's expensive
(8+ parallel training streams) but it's the only **principled** way
to escape the shared-lineage attractor.

## Proposed PBT recipe (for V15.x or successor architecture)

### Pool design
- 8 agents in parallel
- 2 "racers" — reward weight high on forward progress
- 2 "defenders" — reward weight high on capture-avoidance
- 2 "aggressors" — reward weight high on captures + forward
- 2 "balanced" — vanilla reward
- Each seeded from different random init OR different SL endpoints

### Training loop
- Each round: every agent plays N games vs every other agent (round-robin)
- After round: rank agents by win rate within pool
- Bottom 25% are "evicted": their weights replaced by a perturbed copy
  of a top-25% agent (PBT explore step)
- Continue until pool stabilizes (top agents stop being beaten by middle)

### Why this might work where individual training failed
- Each agent's training distribution is **the other 7 agents** —
  different reward weights produce qualitatively different
  policies → genuinely different blind spots in the pool
- Eviction prevents lineage convergence — the gene pool of policies
  stays diverse
- The "winners" are agents that can beat **many policy styles**, not
  just one

### Why we're deferring
- Engineering cost: ~1 week (8 parallel training streams + matchmaking
  infrastructure + eviction logic + cross-agent eval)
- Compute cost: 8× single-agent training. Needs VM time or A100.
- Wins only meaningful **after** we know Tier 2 (handcrafted bots)
  doesn't get us there

### Trigger to revisit
- When Tier 2 (expectimax bot + MCTS-pure bot) is exhausted as
  training opponents AND we still haven't broken the ceiling
- OR when we have a sponsor / a free week of A100 access

## Notes from earlier discussions

- The dice-variance ceiling is real but not yet measured. Should run
  V13.5 vs Expert 10K-game H2H to know the ceiling. Skipping for now
  because human play against the models reveals **clear mistakes**
  — the dice ceiling is not yet bottlenecking us.
- Mech-interp on V15.1 already shows: L2 redundant, prev frame ~3%
  useful, edge bias unused. The V15.1.2 arch shrink is worth doing
  AT THE SAME TIME as PBT (PBT amplifies arch wins, doesn't
  substitute for them).

## Files that would need to exist for PBT

- `td_ludo_v15/pbt/pool.py` — agent registry, matchmaking
- `td_ludo_v15/pbt/eviction.py` — bottom-N replacement with perturbation
- `td_ludo_v15/pbt/trainer.py` — per-agent training loop wrapper
- `td_ludo_v15/pbt/dashboard.py` — multi-agent stats display
- `td_ludo_v15/checkpoints/v15_1_pbt/<agent_N>/` — per-agent ckpt dirs

None of this exists yet. ~1 week of work.
