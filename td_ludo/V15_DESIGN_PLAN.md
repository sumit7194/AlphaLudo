# V15 — Per-Cell Triplet Input + Graph Transformer Backbone

> Architecture design plan. **Not implemented yet.** This doc is the spec we settle on before any code is written. While V13.5 RL trains in the background, we lock the V15 design here.
>
> Author: Sumit (designing) + Claude (drafting). Date: 2026-05-14.

---

## TL;DR

**Replace V13.5's 21-channel spatial encoder with a per-cell-triplet encoding stacked across 8 chronological frames, processed by a Graph Transformer over the board's path topology.** Every cell on the 15×15 board carries the same 3-feature tuple. Global state (dice, scored counts) lives in specific repurposed cells. Temporal history is added as a frame-stack channel. The backbone is a Graph Transformer with edge biases reflecting Ludo's path structure — *not* a CNN.

**Input shape:** `(T, 15, 15, 3)` where `T = 8`.
**Float count:** `8 × 15 × 15 × 3 = 5,400` floats — comparable to V13.5's `4,725` per single frame, but carries 8× more state.

**Backbone:** Graph Transformer, full self-attention over all 225 cells + 1 CLS token, with learned edge biases encoding Ludo's path-graph topology.
**Output:** Source-cell policy (225-way masked softmax) + win_prob scalar. **No aux heads in first cut.**

---

## Per-cell tuple encoding rules

Every cell `(row, col)` of every frame is a length-3 vector `(a, b, c)`. The rules:

### Slot semantics

| Slot | Meaning | Range |
|---|---|---|
| `a` | My token count at this cell | `0..4` if I can be here · `-1` if I cannot |
| `b` | Opp token count at this cell | `0..4` if opp can be here · `-1` if opp cannot |
| `c` | Safety / playability flag (from my POV) | `1` safe & I can move here · `0` not safe & I can move here · `-1` I cannot move here |

### Special-cell overrides

These 4 cells use the tuple slots for **non-token global state** instead:

| Cell | Symbol | Tuple shape | Meaning |
|---|---|---|---|
| `(0, 0)` | `MD` | `(-1, -1, c)` | `c ∈ {1..6}` = my dice roll this frame · `c = -1` = not my turn this frame |
| `(14, 14)` | `OD` | `(-1, -1, c)` | symmetric: opp's dice this frame |
| `(7, 6)` | `MS` | `(a, -1, -1)` | `a ∈ {0..4}` = my scored-token count |
| `(7, 8)` | `OS` | `(a, -1, -1)` | symmetric: opp scored count |

These 4 cells violate the normal rule semantics. Position uniquely identifies them. The sign pattern of the tuple lets the network recognize them as "global" cells.

`MC` (move count) is **dropped** in V15 — game phase is implicit in scored counts + token positions.

### Sign patterns and what they classify

A useful invariant: every cell's sign pattern places it in exactly one category.

| Sign pattern | Category | Count (2P) |
|---|---|---|
| `(≥0, ≥0, ≥0)` | Shared-path cell — both players can be here | ~52 path cells |
| `(≥0, -1, ≥0)` | My-only territory — my home-counter `(2,2)` or any my-home-stretch cell | 1 home counter + 5 stretch = 6 |
| `(-1, ≥0, -1)` | Opp-only territory — opp home-counter `(11,11)` or any opp-stretch cell | 6 |
| `(-1, -1, 1..6)` | MD or OD active with dice value | 0..2 (only when that side just rolled) |
| `(0..4, -1, -1)` | MS or OS (scored count in slot 0) | 2 |
| `(-1, -1, -1)` | Inactive cell — 3 unused home corners (×2), MD/OD when not that side's turn, all dead corners | ~155 cells |

The 6-way classification means the network's first-layer node-MLP can roughly split cells into "what kind of cell am I looking at" via sign-pattern matching, then read the values within that category. In a Graph Transformer (per-node embedding before any aggregation), this classification is preserved cleanly through the trunk — no batch-norm statistics destroy the signal.

---

## Token-symmetry: universal rule, no special cases

V15 has no per-token input channels — every cell's `a` slot is simply *the count of my tokens currently at this cell*. The token-ID symmetry that V13.5 had to engineer comes for free: the model literally never sees token IDs at any layer.

This means home base, home stretch, scored cells, and main path cells all follow the **same rule**:

> `a` = number of my tokens at this exact cell (or `-1` if I cannot be here)

The cases break down naturally:

- **Main path cells**: `a = N` when `N` tokens stack there (often 0 or 1, sometimes 2+).
- **Home stretch cells** (5 per player): tokens occupy distinct stretch positions, so each cell shows `a ∈ {0, 1}` based on whether a token is at that specific position. Stacking is rare but possible.
- **Home base** (4 cells per player): tokens at home all share "position -1" in the engine — they're physically stacked in the home circle on the real board. So we designate **one cell** as the "home counter" and put `a = home_token_count` there. The other 3 cells in the home 2×2 are unused — signature `(-1, -1, -1)`.
- **Scored** (1 designated cell per player): `a = scored_count`. Other slots `-1`.

### Designated counter cells

| Counter | Cell | Slot | Range |
|---|---|---|---|
| My home base | `(2, 2)` | `a` | 0..4 |
| Opp home base | `(11, 11)` | `b` | 0..4 |
| My scored | `(7, 6)` (= MS) | `a` | 0..4 |
| Opp scored | `(7, 8)` (= OS) | `a` | 0..4 |

For my own home base counter at `(2, 2)`: tuple is `(home_count, -1, c)` where `c ∈ {0, 1, -1}` depending on whether unlocking from home is legal under the current dice. The unused 3 home cells at `(2,3), (3,2), (3,3)` carry `(-1, -1, -1)` — model learns they're inactive via positional embedding.

**Why this is cleaner than spread-fill:**

1. **One universal rule** — no special-case "first N cells in canonical order" bookkeeping in encoder or distillation projection.
2. **Matches the real board** — home tokens are physically stacked in the home circle, not spread across cells.
3. **Distillation projection becomes trivial**: V13.5 home-rank probability mass goes to one cell (the counter), no even-spread machinery needed.
4. **Symmetric by construction** — token IDs never appear anywhere in the input.

---

## Temporal stacking — chronological 8-frame history

Following AlphaZero's design for chess:

- Stack the **8 most-recent state snapshots** in chronological order. Frame index `t = 0` is the oldest, `t = 7` is the current state.
- Each frame is a complete `(15, 15, 3)` snapshot under the rules above.
- A frame represents the state **AT the time of a decision being evaluated** (or after a move resolved, for past frames).
- Bonus turns and dice-game asymmetry are naturally handled — frame stack is by *real-time order*, not by who-moved.

### Bonus-turn / consecutive-sixes encoded implicitly

Two consecutive frames with `MD ≠ -1` (i.e. both were my turns) means I just got a bonus turn. Three in a row = potential triple-six forfeit on the next roll. The model reads this from the MD pattern across frames — **no separate bonus-turn flag, no consecutive-sixes scalar.**

### Frame padding at game start

At move 0, there are no history frames. Fill all 7 past frames with **all-zero tensors** (every cell = `(0, 0, 0)`). The model will see "no game has happened yet" — same convention AlphaZero uses.

Optional: a "valid frame" flag per frame in a side-channel scalar. Not required for V15 — zero-padding works.

---

## Architecture choice — why Graph Transformer (and not CNN)

The input encoding is fixed; the backbone needs to match its semantics. The earlier draft of this doc proposed a 2D CNN over stacked frames, but two issues drove the pivot to a Graph Transformer:

### Why not CNN — the unnormalized-input problem

The per-cell triplet encoding deliberately uses **raw signed values** (`-1..4` for counts, `-1..1` for safety flags) so that the **sign pattern itself classifies cell type** (see "Sign patterns and what they classify" below — 6 categories from `(±, ±, ±)`).

In a CNN with BatchNorm, this is broken at layer 1:
- BN computes `mean / var` across `(batch × H × W)` — averaging over all ~225 cells per state.
- Most cells (~150 of 225) are inactive sentinels at `(-1, -1, -1)`, so the spatial mean is dominated by `-1`.
- BN shifts that toward `0` → the carefully-designed sign signal becomes zero-centered noise.
- The 4 special cells (MD/OD/MS/OS) are statistically a rounding error → distinctive features normalized away.

Workarounds (no-BN first conv, sign one-hot pre-projection, etc.) exist but add complexity. And path-topology is genuinely badly modeled by a 2D Conv kernel — Ludo's path is a 1D loop bent into 2D, so two cells adjacent on the grid aren't always adjacent on the path.

### Why Graph Transformer

A Graph Transformer processes nodes *first, independently*, then aggregates via attention. Three structural wins for this design:

1. **Per-node embedding preserves sign info.** Each cell's `(a, b, c)` is read by a shared 3→d MLP before any cross-cell mixing happens. The MLP learns the 6-category sign-pattern classification directly. No batch statistics destroy the signal.
2. **Normalization is per-node** (LayerNorm on each node's embedding) — no cross-cell averaging at any point.
3. **Path topology becomes an explicit attention bias.** Edges encode the *actual* graph structure of Ludo: "from cell X with dice d, where do I land?" becomes a single learned bias term in the attention matrix. The model doesn't need to figure out the path adjacency from grid geometry — it sees it as input.

The trade-off is one new code path (graph + attention) instead of reusing V13.5's ResNet trunk. Worth it given the input/output redesign already breaks V13.5 compatibility — we'd need new training scripts either way.

### Graph structure

**Nodes:** All 225 cells of the 15×15 board, kept as a 1:1 grid mapping. **Plus 1 CLS readout node** (virtual, not on the board). Total: 226 nodes.

Rationale: keeping the full 225 cells (rather than pruning to ~70 path-relevant cells) preserves the board's geometric essence — useful for debugging (visualize as 15×15), and gives 4-player adaptation a clean path (more cells become path cells, no graph reshape needed).

**Edges (5 types, all directed, all with learned 8-dim type embeddings used as attention bias):**

| Type | Count (2P) | Description |
|---|---|---|
| `path_step_d`  (d=1..6) | ~52 × 6 ≈ 312 | For each path cell `c`, edge `c → c+d` if `c+d` is on the path or home stretch. One edge type per dice value. |
| `path_back_d`  (d=1..6) | ~312 | Reverse of `path_step_d`. Lets each cell see "what's behind me" — crucial for danger evaluation. |
| `home_unlock`  (only dice=6 active) | 8 | Each home-base cell → spawn cell. Gated on dice=6 — model reads MD frame to know if active. |
| `stretch_to_score` | 2 | Home-stretch terminal cell → scored-state cell. Absorbing. |
| `global_broadcast` | 226 × 4 ≈ 904 | Each of MD/OD/MS/OS connects to **every** other node. Lets globals influence cell representations in one attention layer. |

**Total edges**: ~1,500. Each is just `(src, dst, type_id)`. The graph is **fully static** — same edge set across all games. Built once at startup.

In attention, edges act as **bias terms** added to attention logits between connected nodes: `attn_logit[i, j] += edge_bias[edge_type(i, j)]`. Unconnected pairs get no bias (so attention can still flow between them if useful, just less biased). This is softer than a hard mask and lets the model learn to ignore the prior when it doesn't help.

### Architecture spec

| Component | Spec |
|---|---|
| Node feature input | `(225, 24)` — flattened from `(15, 15, 8, 3)` with 8 frames stacked at the feature dim |
| Node embedding | 24 → 192 (shared MLP across all cells, 2-layer with GELU) |
| Positional embedding | Learned `(225, 192)` table, one per cell. Added to node embedding. |
| CLS token | 1 extra learned embedding `(1, 192)`. Prepended to the node sequence. |
| Edge type embedding | Learned `(num_edge_types, 8)` table. Projected to attention-bias scalars at each layer. |
| Trunk | 4 × Graph Transformer Encoder layer, each: pre-LN multi-head attention (8 heads, d=192, edge-bias) + pre-LN FFN (192 → 384 → 192, GELU) |
| Policy head | Per-cell MLP (192 → 64 → 1) applied to each of the 225 board nodes → `(225,)` logit vector → masked softmax over legal source cells. |
| Value head | CLS-token embedding → MLP (192 → 64 → 1) → sigmoid scalar |
| **Total params** | **~1.8M** (smaller than V13.5's 3M — attention is more expressive per layer than convolution) |

### Why this size

V13.5 is 3M params and ties V13.2 in matched-capacity H2H. The hypothesis is that the input/output redesign + graph topology gives V15 strictly more inductive bias per parameter, so smaller can suffice. If V15 plateaus, scaling to 6 layers × d=256 (~3M params) is a one-flag change.

---

## Output heads — per-source-cell policy + scalar value

V13.5's rank-indexed policy + gather-to-token-id machinery **cannot survive the V15 input redesign**. V13.5's input has per-token planes (V14 ch 0..3 mark each token-id individually); V15 deliberately destroys per-token identity at every cell. Without token-id in the input, the model has no anchor to output `logits[token_id]` — and forcing it to invent an ordering would reintroduce the asymmetry V15 was built to eliminate.

### The actual action space in Ludo

Given dice is already rolled, the **distinguishable** action is "pick a source cell with ≥1 of my tokens; advance by dice." Two of my tokens stacked on the same cell are **state-equivalent** — moving either one yields the identical resulting state. The true action set is "source cells with movable tokens," not "4 tokens." V18 already encoded this insight on the input side (home corner zeroed, count stored as scalar). V15 extends it to the output.

### V15 output heads (final)

| Head | Shape | Activation | Loss | Notes |
|---|---|---|---|---|
| **Policy** | `(15, 15)` → 225 logits | softmax + legal-mask | PPO clipped | "Move the token currently at cell (r, c)." Two stacked tokens collapse to one logit. |
| **Value (win_prob)** | scalar | sigmoid | BCE | Same as V13.5. Ludo has no draws, so 2-class is sufficient. |

That's it. **No aux heads in V15 first cut.**

### Why no aux heads

- **Per-rank progress (×4)** — same disease as rank-indexed policy. Requires the model to commit to a token-ID ordering, which V15's input doesn't support.
- **Moves remaining** — low-priority for Ludo (game-length variance is dominated by dice, not by play quality). Drop.
- **Threat-in-next-K-turns** — interesting but speculative. Skip in first cut; revisit only if V15 base performance plateaus and ablations point at value-head bottleneck.

The point of the V15 redesign is to clean up the input contract. Adding aux heads at the same time would confound the ablation.

### Architectural realization of the policy head

A per-node MLP (`192 → 64 → 1`) shared across all 225 board nodes maps each cell's final-layer embedding to a scalar logit. The CLS token is excluded — only the 225 board cells contribute policy logits. The 225-vector is then legal-masked and softmaxed.

This is the Graph-Transformer analogue of "1×1 conv from trunk features → per-cell logit." The per-cell triplet symmetry is preserved end-to-end: the model never sees a token-ID, only counts-per-cell + cell identity (via positional embedding).

### Bookkeeping: source-cell → token-id at the engine boundary

To actually `apply_move` in the engine you still need to nominate *some* token-id when multiple tokens share a source cell. Since the post-move state is identical regardless of which token-id from the stack you pick, the engine can deterministically pick (e.g. lowest token-id at the cell). The lossy-looking projection is exact.

---

## Training pipeline

1. **SL distillation from V13.5** — train a fresh V15 backbone to imitate V13.5's policy + win_prob on a corpus of games. Should get V15 to ~V13.5 level (~80% eval) before any RL.
2. **PPO RL on top** — same opponent mix as Phase L (V13_2 40% / V13_5_SL 30% / SelfPlay 20% / V12_2 10%), entropy 0.03, no search teacher.
3. **Evaluation cadence** — 10K interval / 2K games per eval, matching V13.5 Phase L.

### Cold-start: cross-architecture distillation

V13.5's input format is fundamentally different (21-channel single-frame); V13.5's output format is also different (4-logit rank-indexed). Distillation still works — both models see the same game state, just encoded differently, and the **target distribution is projected into V15's source-cell action space** before being matched.

**Target conversion (V13.5 → V15 source-cell space):**
1. Run V13.5 on the state → 4 rank-indexed logits → gather to per-token-id probabilities `p[token_id]`.
2. For each token-id, look up its current cell (using V15's cell mapping):
   - Token at main-path position `p` → its corresponding `(row, col)` per the canonical path order.
   - Token in home stretch at position `p` → its stretch cell.
   - Token at home (position -1) → the designated home-counter cell `(2, 2)`.
3. Sum probabilities of all token-ids mapping to the same cell:
   `q[cell] = Σ_{tid: cell(tid) = cell} p[tid]`
4. `q` is now a distribution over source cells — V15's policy target.

The sum-collapse is **exact, not lossy**: stacked tokens (and all home tokens collapsing to the home-counter cell) are state-equivalent, so summing their probabilities is the correct projection. Same approach applies for distilling from any rank- or token-indexed teacher.

Example: 3 of my tokens are home + 1 on path at position 10. V13.5 says `P[rank_0]=0.5` (the path token, most-advanced) and `P[rank_3]=0.5` (the home rank — split equally over the 3 home token-IDs). After expansion: `p[path_token] = 0.5`, `p[home_t1] = p[home_t2] = p[home_t3] = 0.167`. After sum-collapse: `q[path_cell] = 0.5`, `q[home_counter_cell] = 0.5`. V15 target is `{path_cell: 0.5, home_counter: 0.5}`. Clean.

---

## Input size accounting

| Quantity | V13.5 | V15 |
|---|---|---|
| Per-frame input | `21 × 15 × 15 = 4,725` floats | `15 × 15 × 3 = 675` floats |
| Temporal frames | 1 | 8 |
| Total per state | **4,725 floats** | **5,400 floats** |
| Effective info | 1 state's worth | 8 states' worth (chronological history) |
| Info density | ~12% real (88% broadcast/redundant) | ~50% real (homogeneous per-cell semantics) |

Marginally larger total input — but encodes 8× more game history.

---

## Design decisions — all resolved ✅

### Input
1. **Frame count = 8** ✅ — 7 historical frames + 1 current frame, AlphaZero convention.
2. **Multi-token home rule = single counter cell** ✅ — designated home-counter cell shows `a = home_token_count`; the other 3 home cells are inactive `(-1, -1, -1)`. Same rule for opp home base and scored cells. No spread-fill, matches the physical board.
3. **MC dropped from input** ✅ — game phase is implicit in scored counts + token positions.
4. **No bonus-turn / consecutive-sixes scalars** ✅ — implicit in the temporal MD pattern across the 8 frames; model learns it from training data.
5. **No input normalization** ✅ — feed raw `-1..4` and `-1..1` values straight to per-node embedding MLPs. Safe with GNN (per-node processing, no batch-norm-across-cells); would NOT have been safe with CNN+BN — that was the deciding factor for the backbone choice.
6. **Frame padding at game start = all-zeros** ✅ — AlphaZero/AlphaGo Zero/MuZero/Lc0/KataGo all use this convention.

### Backbone
7. **Backbone = Graph Transformer with edge biases** ✅ — 4 layers, d=192, 8 heads, ffn=384. Hand-rolled dense attention (no PyG dependency).
8. **All 225 cells as nodes + 1 CLS** ✅ — full board preserved (not pruned). Easier debugging, cleaner 4-player adaptation.
9. **5 edge types, all directed, all typed with 8-dim learned embeddings** ✅ — path-step ×6, path-back ×6, home-unlock, stretch-to-score, global-broadcast. Edge bias added to attention logits.
10. **Per-cell learned positional embeddings** ✅ — `(225, 192)` table, gives each cell stable identity.

### Output
11. **Output heads = source-cell policy (225-way) + win_prob (sigmoid) only** ✅ — no aux heads in first cut.
12. **Value head reads from CLS token** ✅ — virtual readout node attends over all cells.
13. **Policy head = per-cell MLP shared across board nodes** ✅ — CLS excluded from policy.

### Training
14. **Cross-arch distillation: V13.5 → V15** ✅ — exact target via rank → token-id → source-cell sum-collapse projection.
15. **Library: hand-rolled** ✅ — no PyG / DGL. Dense attention over 226 nodes is a single `torch.einsum`. Keeps MPS / CUDA compat trivial.

---

## What this doc deliberately leaves open

- **Backbone exact spec** (num_res_blocks, num_channels, attention heads, etc.) — defer to implementation
- **Exact normalization strategy** — defer to ablation experiments
- **Whether to add `threat_in_K` aux head later** — defer until base V15 is trained
- **Whether the input encoding handles 4-player mode** — explicitly **not** designed for 4P; this is 2P canonical only
- **Migration plan from V13.5 in production** — separate doc when V15 is trained

---

## Code organization — V15 lives entirely in new files

All existing V12.x / V13.x / V14_scalar / V13.5 code stays untouched. V15 is fully additive:

```
td_ludo/
  game/
    encoder_v19_triplet.py        # NEW — per-cell triplet + 8-frame chronological stack
    v15_graph.py                  # NEW — static graph topology (nodes, edges, types) built once
  models/
    v15.py                        # NEW — V15GraphTransformer model class
  train_v15_sl.py                 # NEW — SL distillation (V13.5 teacher → V15 student)
  train_v15_rl.py                 # NEW — RL self-play with H2H gating
  experiments/
    v15/
      test_v15_encoder.py         # NEW — encoder + graph + symmetry unit tests
      h2h_v15.py                  # NEW — V15 vs V13.5 / V13.2 H2H runner
      chain_full.sh               # NEW — SL → backup → H2H pipeline
```

Shared utilities imported but unchanged: `state_to_rank_mapping` (used in V13.5→V15 distillation projection), `heuristic_bot` (bot eval), `td_ludo_cpp` (engine).

---

## Implementation order — design locked, ready to plan

1. **Graph + encoder** — `v15_graph.py` (static graph build) + `encoder_v19_triplet.py` (per-state encoding). Both pure data transformations, fastest to test in isolation.
2. **Unit tests** — `test_v15_encoder.py`:
   - Sign-pattern category assignment correct for all 6 categories
   - Spread-fill rule produces N cells lit for N home tokens
   - Special cells (MD/OD/MS/OS) values correct under various dice/scored configs
   - Permutation-symmetry: shuffling token-IDs produces identical encoding
   - Graph has expected node/edge counts and types
3. **Model class** — `models/v15.py`:
   - Node embedding MLP
   - Positional embedding
   - CLS token
   - Edge-biased multi-head attention layer (custom, hand-rolled)
   - Stack of 4 layers
   - Policy + value heads
4. **Forward-pass sanity** — load random weights, feed a state, verify output shapes + masking works.
5. **SL distillation script** — `train_v15_sl.py`:
   - Target conversion: V13.5 rank policy → per-token-id → source-cell sum-collapse
   - Same recipe as V13.5 SL (Adam, batch 256, lr 1e-3 → 1e-4 cosine)
   - Eval at 1M-state intervals
6. **Parity validation**: extend the worked-example state from V13.5's verify script — dump V15's encoding, visualize as `(15, 15, 3)` per frame, confirm sign patterns + home-counter + special cells all look right.
7. **Smoke test** — 50K-state SL run on Mac MPS, verify trainer + eval + save end-to-end.
8. **Real SL run** — 5M states on Mac MPS, expect ~6-8h. Final eval + H2H vs V13.5 and V13.2.
9. **RL pipeline** — `train_v15_rl.py`: REINFORCE-with-baseline, KL anchor to V15_SL, multi-legal filter, H2H gating every 2M states.

Estimated effort to first trained V15-GNN: **~1.5 weeks of focused work** (encoder + graph + model class + tests), then 1-2 weeks training + evaluation.

---

## V16 preview (not in scope for V15)

V16 is now reserved for a follow-up if V15 ships and we want to push further. Candidates:

- **4-player support** — extend the graph (same 225 nodes, more path → more shared-path cells, all 4 home bases as nodes), retrain with 4P self-play.
- **Search-augmented RL** — V13.5 + V15 give us two diverse teachers. Use both as leaf evaluators for shallow expectimax search, distill the search-improved targets into V15. The MCTS Step-1 idea from Exp 32, but with a stronger and more diverse leaf evaluator.
- **Bigger Graph Transformer** — 8 layers × d=256, ~6M params. Only if base V15 plateaus and ablations point at capacity bottleneck.

Defer all until V15 is trained and evaluated.

---

## Implementation status (post-doc, 2026-05-14 → ongoing)

This design plan was the SPEC. The implementation deviated in two ways
from the original draft, both validated empirically:

1. **Backbone:** doc proposed CNN-over-stacked-frames (Candidate 1) as
   first cut. Implementation went with **GraphTransformer** (Candidate
   3's spiritual successor — edge-type-biased attention instead of pure
   message-passing GNN). 4.4M params, 8 layers × d_model=256 × 8 heads.
   Reason: a sibling Claude session shipped the GT variant before the
   CNN-first plan locked in; results justified keeping it (SL matched
   teacher).

2. **Trainer:** doc planned drop-in on `train_v12.py`. Reality: V15's
   8-frame-history + 225-cell action shape didn't drop into the V13.5
   trainer cleanly. Built a parallel rich pipeline in
   `td_ludo_v15/rich/` mirroring `train_v12.py`'s structure (ELO, GameDB,
   bot grid, v13_dashboard endpoints) but with V15-specific shapes.

**Empirical results to date** (see `MODEL_HISTORY.md` § V15 and
`training_journal.md` § Exp 42 for full detail):

| Phase | Result |
|---|---|
| SL v1 | 52% bot-eval (BROKEN — random state distribution + under-cap) |
| SL v2 | 83.0% bot-eval — **matches V13.5 teacher (82.0%)** at 20M states |
| RL initial | 81-82% — at parity with SL, policy barely moving (KL≈0) |

V13 input encoding is empirically dead — V15 SL matches V13.5 RL ceiling
purely from cross-architecture distillation with the new input/output
contract. Whether V15 RL can BREAK past V13.5's ceiling is the open
question being tested overnight 2026-05-15 → 2026-05-16.

---

## Final outcome (2026-05-17)

V15 work has concluded. See `MODEL_HISTORY.md § V15 — final outcome`
and `training_journal.md § Exp 42, 44` for the full record.

**One-line summary:** V15 SL successfully matched V13.5 (engineering
result) but V15 RL regressed despite 3 hyperparameter rotations
(scientific dead-end). The design as specified IS implementable and
matches V13.5 — but the new architecture didn't unlock plateau-breaking.

**V16 (GNN reservation) is paused** — paragraph here was "if V15-CNN
ships and we still want more headroom, GNN over the path-graph is the
right next step." Reality: V15-GT shipped (in SL form) at parity, not
above. GNN's value proposition was "match Ludo's actual topology" which
is an *architectural* hypothesis. We now have strong evidence (V15
results, MCTS coherent equilibrium, V13.5 RL Phase L only +0.5pp lift)
that **architecture isn't the bottleneck for this codebase at this
scale.** GNN deferred until there's a fundamentally different attack
plan (e.g., search-trained value head decoupled from policy, 4-player
mode, transformer at 10x compute).

**The V15 codebase is preserved** for any future work:
- `td_ludo_v15/td_ludo_v15/{game,models,rich}/` — encoder, model, RL pipeline
- `td_ludo_v15/checkpoints/v15_sl_v2/model_sl.pt` — final V15 SL ckpt
- `td_ludo_v15/tests/` — encoder + model + rich-pipeline tests (all pass)
