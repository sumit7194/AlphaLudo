# Attack Angle: Consequence-Prediction World-Model Heads (target B)

Status: DESIGN SPEC for review (2026-06-14). Not yet implemented.
Goal: make the model *understand consequences* instead of pattern-matching
the reward surface — the root disease behind the plateau AND both play-test
flaws (see PLAYTEST_FLAWS_v136.md).

---

## Why this, why now (step 1 probe result)

Probe `probe_champion_vs_search.py` — V13.6 champion (greedy) vs the strongest
INDEPENDENT search bots, 300 games each:

| Opponent | Champion win% |
|---|---|
| Expectimax (depth-1) | 59.7% ±2.8 |
| Depth2Expectimax | 62.3% ±2.8 |
| MCTSExpectimaxPrior (77%-vs-Expert, your strongest) | 64.0% ±2.8 (192-108) |

**The champion beats every search bot we have.** So there is **no reachable
stronger policy to distill from** — distillation is dead (can't copy from
something weaker than you). The only way up is to **teach understanding the
net doesn't have.** This confirms the world-model route is the necessary
path, not one option among several.

Reconciles with the flaws: the champion is strong *on average* (beats search,
rides the variance ceiling) but has *specific blind spots* (spawn-on-6,
laggard neglect) that don't dominate win rate. Classic "pattern-matches well,
doesn't understand."

---

## The idea

Stop training the net ONLY to maximize outcome-reward (→ pattern-matching).
ALSO train it to **predict the consequences of the position** via auxiliary
supervised heads on the shared trunk. A trunk forced to predict "my laggard
is captured with prob 0.5 next opp turn, losing 22 cells" must *encode* that
risk — and the policy head reading that trunk can then act on it. Inspirations
(adapted, not copied): MuZero (learned model), UNREAL (auxiliary prediction
for representation). Ludo's chance is a clean 1/6, so targets are exact and
cheap — **no MCTS**, which is why this sidesteps the Exp-45 coherent-
equilibrium wall (we teach ground truth from the engine, not the net's own
value).

---

## The targets (computed exactly from the C++ engine, no search)

Per own token currently on the main track (rank-slot indexed, matching the
V135 symmetric head):

1. **P(captured next opponent turn)** ∈ [0,1].
   Compute: enumerate opp dice d∈{1..6}; for each, does ANY opp token have a
   legal move landing on this token's absolute cell (and the cell isn't safe)?
   `P = (# capturing dice) / 6`. Assumes opp captures when able — the relevant
   risk the human reasons about. ~6×4 cheap checks per token. (Reuses the
   capture/danger logic already in bias_penalties.py.)

2. **Expected cells-at-risk** = `P(captured) × token_progress`. A scalar
   regression target. Encodes "how much do I lose if this token dies" — the
   value-at-risk signal both flaws show the net lacks.

3. (Phase 2, optional) **Per-legal-action ΔÊ[total cells-at-risk]** — for each
   legal move, the predicted change in summed cells-at-risk after the move.
   Ties consequence directly to move choice. More expensive (one target per
   action); add only if heads 1–2 aren't enough.

K=1 (next opp turn) first — it's the dominant, cheapest risk. K=2 needs a
rollout; defer.

---

## Head + loss

- Attach a small MLP head off the V135Symmetric shared features (pre-policy).
- Outputs per rank-slot: capture_prob (sigmoid) + cells_at_risk (linear).
- Aux loss: `BCE(capture_prob, target)` + `SmoothL1(cells_at_risk, target)`,
  masked to valid (occupied, on-track) slots. Weight ~0.3–0.5 vs policy loss
  (tune so it shapes the trunk without dominating).
- The point is the SHARED TRUNK encoding risk; if the policy doesn't shift
  enough from trunk-shaping alone, feed the predicted risk back as an INPUT
  feature to the policy head (phase 2).

## Training plan

- **Cheapest test first:** fine-tune the existing champion with the aux heads
  added (continue RL + aux loss) for ~100–200K games. Does it fix laggard
  neglect in play? Fast hypothesis check; reuses champion weights.
- If promising → bake into a fresh SL+RL run.
- Note: this targets the **risk/consequence** flaws (laggard, danger). The
  **spawn-on-6** flaw is more reward-tilt-driven (+0.05 spawn > +0.03 fwd) —
  partly addressed if the action-Δ head (target 3) is added, but the reward
  fix in PLAYTEST_FLAWS_v136.md is the cleaner lever for that one.

## Validation (target B — NOT win rate)

1. **Head accuracy:** on held-out positions, is predicted P(capture) close to
   engine ground truth? A direct *understanding* metric.
2. **Play-test:** does it stop abandoning the laggard? (the original flaw).
3. **Decision-logs / eval-lens:** fewer "left token in danger" disagreements.
   Win-rate eval is expected to NOT move (variance ceiling) — that's fine.

## Open questions / risks

- Does trunk-shaping actually change POLICY behavior, or add an ignored head?
  (Shared trunk should propagate; verify via play + ablation.)
- Opp-capture assumption (greedy) for the target — is "max-capture" the right
  risk model, or should it weight by a real opp policy? Start greedy (cleanest).
- Compute/where: needs a retrain. VM is now the user's LLM box — sort out
  compute before the fine-tune step.
