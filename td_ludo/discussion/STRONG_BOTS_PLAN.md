# Strong-Opponent Bots — Plan

**Purpose**: design + benchmark non-neural bots that are qualitatively
stronger or more diverse than the existing scripted-bot pool
(Heuristic/Aggressive/Defensive/Racing/Random/Expert/Expectimax/MCTSPure),
so the trained model has genuinely harder and more varied opponents
during RL.

Existing pool (commit `6b616d5`):
- 6 scripted bots
- `ExpectimaxBot` — depth-1 lookahead + dice expectation, beats Expert 68-32
- `MCTSPureBot` — PUCT + random rollouts; (30/4) loses to Expert; (50/8) beats Expert 55-45

Headline problem: every trained model (V6.x → V15.1) has only ever
optimized against the 6 scripted bots. They all plateau at ~85% vs that
mix. Need opponents the model has *never seen* and CAN'T converge to
defeating without genuinely different reasoning.

## Phase plan

### Phase 1 — Expectimax personalities (HIGH PRIORITY)

Subclass `ExpectimaxBot` 4 times, override only the scoring function.
Same depth-1 lookahead structure. Each creates a distinct policy
"personality" — same compute cost (~50ms/decision, 46 g/s), same
search structure, different *preferences*. The trained model will need
to learn 4 different counter-policies.

| Variant | Scoring change | Predicted "personality" |
|---|---|---|
| `AggressiveExpectimax` | exposure_penalty for OPP ×3; lower own exposure weight | favors captures and trade-downs; doesn't fear retaliation |
| `DefensiveExpectimax` | own exposure_penalty ×3; opp_progress weight doubled | risk-averse; refuses to advance into reach; stalls |
| `RacingExpectimax` | own progress only; ignore opp tokens entirely | pure race-to-home; never engages |
| `MinimaxExpectimax` | minimize opp's best score instead of max own | game-theoretic worst-case (different from maximax) |

Implementation: new file `td_ludo/td_ludo/game/strong_bots_v2.py`
inheriting from the base `ExpectimaxBot` for shared lookahead loop;
each subclass provides a `_score_position(player, state)` override.

Success criterion: each variant beats Expert ≥60% (proves the
scoring change didn't destroy strength) AND head-to-head WR between
variants is at least 5pp from 50/50 (proves they play differently).

### Phase 2 — Round-robin runner (PARALLEL with Phase 1)

A self-contained `experiments/strong_bot_arena/run_round_robin.py`
that:
- Reads bot list from a registry (or CLI)
- Plays N games per unordered pair (default 200)
- Per-pair JSON checkpointing so killed runs resume cleanly
- Computes ELO via iterative Bradley-Terry from the WR matrix
- Outputs: leaderboard, head-to-head matrix, per-bot games-per-second

This is the "cloud loop" infrastructure — once it exists, every new bot
just gets added to the registry and gets benchmarked automatically.

### Phase 3 — Depth-2 Expectimax (MEDIUM PRIORITY)

Same scoring as base ExpectimaxBot but search depth=2:
`my_move × opp_dice × opp_move × my_dice × my_move_again × terminal_score`.

Cost ~24× higher (~50ms → ~1.2s/decision, ~0.3–0.5 g/s). Acceptable
as training opponent if it adds genuine signal. Test against depth-1
ExpectimaxBot to see if depth helps or scoring noise dominates.

If depth-2 beats depth-1 by ≥10pp, also produce depth-2 variants of
Aggressive/Defensive/Racing/Minimax. If depth helps by <5pp, conclude
depth is not the bottleneck and skip.

### Phase 4 — MCTS with informed prior (MEDIUM PRIORITY)

Modify `MCTSPureBot` to use a non-uniform prior over legal actions.
Two variants:
- `MCTSExpertPrior` — use Expert bot's choice as the prior (one-hot
  argmax). Should converge much faster to good actions.
- `MCTSExpectimaxPrior` — use ExpectimaxBot's choice. Strongest combo.

Expected behaviour: at same compute budget, both should beat the
random-prior MCTSPureBot meaningfully. The Expectimax-prior version
should approach depth-2 ExpectimaxBot quality at maybe 3× lower cost.

### Phase 5 — Endgame solver (LOWER PRIORITY, time permitting)

When total tokens-not-yet-home ≤ 4, the remaining game tree is small.
Build a true expectiminimax with full game-tree expansion (no scoring
function, pure win/lose terminal eval). This gives "perfect endgame
play" for those positions.

Hybrid bot: use ExpectimaxBot in opening/midgame, switch to solver in
endgame. The trained model probably has weak endgame technique because
all the scripted bots also have weak endgame technique.

### Phase 6 — Adaptive / pattern-aware bots (TIME PERMITTING)

Lower priority because they're more bespoke and less general. Ideas:
- `AdaptiveExpectimax` — different scoring function per game phase
  (early/mid/late based on token count distribution)
- `BlockadeExpectimax` — penalize moves that break own 2-token stacks
- `AntiPatternBot` — designed to punish specific known model biases
  (e.g., V13.5's same-token-stickiness)

These produce variety more than raw strength. Useful if Phase 1-4
gives us a strong family but the trained model finds a generic
counter; bespoke bots can break the generic counter.

### Phase 7 — Final tournament + recommended training mix

Once we have all candidate bots from Phases 1–4 (and maybe 5–6), run a
big round-robin at high sample count (500 games/pair) to rank them
and identify which produce diverse strong play.

Output: recommended training-mix composition with bot weights, e.g.:

```
SelfPlay        45%
Expectimax       8%
AggressiveEx     8%
DefensiveEx      8%
RacingEx         5%
MinimaxEx        5%
Depth2Ex         5%  (if Phase 3 succeeds)
MCTSExpertPrior  6%  (if Phase 4 succeeds)
EndgameSolver    5%  (if Phase 5 succeeds)
Expert           5%
```

with rationale per slot.

## Autonomous-work guidelines (for the agent running this overnight)

1. **Stay on `main` branch.** Commit incrementally with informative
   messages. Don't open new branches unless something genuinely
   warrants isolation.
2. **Don't touch any L4-running code path.** That means: don't modify
   anything under `td_ludo_v15/` or any file the L4 trainer imports
   (the strong_bots module is fine since it's read-only there).
   Modifications stay in `experiments/strong_bot_arena/` and
   `td_ludo/td_ludo/game/strong_bots_v2.py`.
3. **Each phase ends with a commit + push.** Even partial-phase
   commits are fine — they preserve progress across any session
   interruption.
4. **Run things in background where useful.** Round-robins can take
   30–90 min; write a script, launch it, work on the next bot while
   it runs. Use `Bash` with `run_in_background=true` and check back
   periodically.
5. **Stop if anything destroys baseline.** If a "stronger" bot ends
   up losing 30% to Expert (when previous version won 70%), there's a
   bug — fix or revert, don't keep iterating on a broken base.
6. **Document each bot with a short docstring** describing the
   scoring philosophy + expected playstyle. Future readers (and the
   user) shouldn't have to reverse-engineer from code.
7. **Use Expectimax as the strength baseline** for all comparisons —
   it's the floor we're trying to exceed.
8. **Update this plan doc** at each phase boundary with actual
   measured numbers + decisions made.

## Success criteria for the autonomous run

End state I'm aiming for by morning:
- 4 Expectimax personality variants (Phase 1) implemented + benchmarked
- Round-robin runner (Phase 2) operational, reproducible, persistent
- Phase 3 attempted (depth-2): clear answer to "does depth help?"
- Phase 4 attempted (MCTS-prior): clear answer to "does informed prior help?"
- Final ELO leaderboard for all candidate bots + recommended
  training-mix composition
- All committed + pushed to `main` with clean commit history

## Measured-results log

(updated as phases complete — see commits)

| Phase | Outcome | Commit |
|---|---|---|
| 1 — Expectimax variants | ✓ 4 variants implemented; 3 beat base by 52-54% h2h | `834df41` |
| 2 — Round-robin runner | ✓ JSON-checkpointed, BT-ELO, resumable | `834df41` |
| 3 — Depth-2 | ✓ 3 depth-2 variants implemented; arena pending | `3ced262` |
| 4 — MCTS-prior | ✓ Expert + Expectimax priors implemented; arena running | `3ced262` |
| 5 — Endgame solver | skipped — defer to keep diversity-first focus | — |
| 6 — Adaptive + rule | ✓ 6 bots (Adaptive, Vote, 4 rule-based) implemented | `2cb3c44`, `73c6cd8` |
| 7 — Final tournament + mix | pending — depends on Phase 3+4+6 arena completion | — |

See `STRONG_BOTS_RESULTS.md` for measured numbers as they come in.
