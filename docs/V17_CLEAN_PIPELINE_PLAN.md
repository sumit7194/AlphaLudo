# V17 — Clean From-Scratch Pipeline: Detailed Plan

*Draft for discussion · 2026-10-03 · nothing here has been built or run yet*

---

## 0. In one paragraph

Train **one unified 2P/4P graph network from random weights** in two stages, as decided in
the Gemini thread and confirmed 2026-10-03:
**(1) SL — 1M games imitating the fixed MCTS-3000 search bot** (now a strong teacher: 63% vs
Depth2, 48.7% vs v15.2 — the September teacher was the broken one that lost 85% to Depth2), then
**(2) RL — win/loss reward only, no shaping**, against a **progressive bot ladder that advances
when the model beats the current rung over consecutive evaluation windows**, plus its current
and past selves. The network must play **without search**. Training never "finishes": it is an
**infinite, safely interruptible run** (pause/resume, configurable threads and resources),
monitored by fixed evaluations against v15.2, Depth2, MCTS-3000 and the scripted bots. 2P first;
4P games are mixed into the same model later. If progress stalls, a later phase adds
search-based improvement, but only once a measured gate says search will help.

---

## 1. Why this plan looks the way it does (evidence)

Everything below comes from the training journal, MODEL_HISTORY, the Gemini thread, and
this session's experiments.

### 1.1 Nobody has ever done this cleanly here
Every strong model in the project traces back to **imitation of scripted bots**
(V6 SL on bot games → V6.1 → V10 → V12 → V12.2 → V13.x → V15) plus dense reward shaping,
bias penalties, KL anchors and scripted/frozen-model opponent pools. A run that is
*random init + self-play + win/loss only* has **never** been done (one accidental
random-init run in May was broken by a checkpoint bug).

### 1.2 The one shaping-free recipe that worked
"Terminal reward fails in Ludo" was the project's belief for months. The real cause was the
**pure Monte-Carlo return estimator**, not the terminal reward:

| run | estimator | result |
|---|---|---|
| Exp 54 — terminal-only, Monte-Carlo returns | MC | entropy ran away, eval drifted |
| **Exp 55b — terminal-only + GAE (λ 0.95)** | GAE | **85.5%**, beat its init 54.4% H2H |
| **Exp 58 — v15.2 stage 2, terminal-only + GAE** | GAE | **0.796 → 0.8465 → champion** |

So: **PPO + GAE + win/loss reward + BCE value head** is the validated core. It has only ever
been run from strong initial weights — whether it bootstraps from random is the open question
this plan tests.

### 1.3 Precedent that random-init self-play can work in a dice game
TD-Gammon (backgammon: dice, races, hitting — very Ludo-like) learned from random weights by
pure self-play with only the win/loss signal, reaching strong play after roughly 300k games and
expert play after roughly 1.5M games. Our pipeline is far more modern (PPO, GAE, transformer),
but the precedent sets realistic expectations: **hundreds of thousands to ~1.5M games**.

### 1.4 Known failure modes we must design against
| pathology | evidence | design response |
|---|---|---|
| Advantage → 0 in symmetric self-play ("coherent equilibrium") | Exp 33/44/48; V13.3 REINFORCE collapse 82%→30% | GAE (the fix that worked), league of past selves, measure KL/clip-fraction |
| Forgetting / cycles: losing to own snapshots | V6 decayed 77%→57%; its 170k snapshot beat its latest 60-40 | snapshot league; gate on H2H vs past checkpoints |
| Entropy bonus bleed in transformers | V15 RL −8 pp at entropy 0.01–0.03 | entropy only on multi-option states, normalised by log(#legal), small + decaying, entropy *target* not fixed bonus |
| Single-legal-move states dilute everything | 25.9% of states are forced | exclude forced moves from policy/entropy loss; keep for value |
| Value head problems | shaped returns inverted value; CLS encodes capture risk poorly (R²≈0.35); value can't rank siblings | win/loss only → BCE value; consider pooled value head (§4.3) |
| Exploiting weak opponents | twosignal: up vs bots, down vs champion | no scripted bots in training at all |
| Pipeline bugs masquerading as learning failures | lr=0 inherited, BN stats, double softmax, aux loss eating 97% of gradient, eval without three-six rule | invariants + tests (§7), logits-only losses, one rules engine (Rust) for everything |
| Eval noise and cherry-picking | 200-game gates ±3.5 pp; "best" = max of noisy evals | ≥1,000-game paired-dice evals, pre-registered metrics |
| Search with a weak judge makes play worse | 1-ply by value: 35% → 43% → 49% even after 400k games of value training | search only in Phase B, behind a measured gate |

### 1.5 What carries over from this session
- **Rust engine**, rules verified (C++ parity; three-six double-pass bug fixed; tests).
- **NetFarm**: many concurrent games in Rust, batched network calls from Python,
  records in shard format v2. ~1,100 games/min of champion self-play on the M4.
- **MLX fp32 inference** (bit-equivalent to PyTorch; 1,850 evals/s) + CPU worker.
- **Bit-exact Rust V15 encoder**; evaluation harness with paired dice; dashboard pattern.
- **Exact gradient accumulation** (memory), exact mid-epoch resume, atomic checkpoints.
- **The 1-ply-by-value test** — the metric that actually measures whether a value is good
  enough for search (aggregate log-loss/AUC do not).

---

## 2. Goals — no finish line, only measured progress

There is **no target and no stopping point**: every run is infinite and safely interruptible.
The aim is simply to keep getting stronger than anything before it. Progress is tracked with
fixed, pre-registered measurements (all greedy, seat-alternated, paired dice, ≥ 1,000 games):

| measurement | why |
|---|---|
| H2H vs **v15.2** (no search on either side) | the strongest previous model |
| vs **MCTS-3000** (fixed Rust search bot, = v15.2 level: 48.7% vs it) | strongest non-neural reference |
| vs **Depth2** (v15.2 scores 59.0%) | strong, cheap reference |
| Expert + Heuristic protocol (v15.2: 84.9%) | continuity with the whole project history |
| H2H vs own earlier checkpoints (ladder) | catches forgetting / cycles |
| 1-ply-by-value test | is the judgement good enough for search (Phase B gate) |

Diagnostic rule (not a stop rule): if H2H vs v15.2 hasn't moved by ≥ 2 pp over ~300k games,
we look at the logs and decide what single variable to change.

---

## 3. Algorithm — Stage 1 SL (fixed MCTS-3000 teacher) → Stage 2 RL (PPO + GAE, win/loss reward)

### 3.0 Stage 1 — SL from the fixed MCTS-3000 teacher (1M games)

- **Teacher**: Rust `TwoPlayerMCTSBot`, 3,000 simulations (post-audit: exact chance nodes,
  greedy-rollout + calibrated leaf; 63.1% vs Depth2, 48.7% vs v15.2). Search is used **only to
  make the labels**; the network never searches at play time.
- **Games**: **MCTS-3000 vs MCTS-3000** (your call, 2026-10-03). Variety comes from the dice;
  MCTS searches each position rather than following fixed rules. Both seats are the teacher, so
  **every move of both players is a label** — winner and loser alike. Not winner-only on purpose:
  winner-only data never shows how to play from behind and mostly selects lucky dice (see the
  4PW review).
- **Honest limits of the teacher**: 3,000 simulations look a few turns ahead (dice branching),
  not to the end; its leaves are scored by a short greedy playout + a calibrated evaluator fitted
  on Depth2 games (a little built-in knowledge). It is v15.2-level (48.7%), not perfect — which is
  why Stage 2 RL, from win/loss alone, must take it further.
- **Targets**: the teacher's **visit distribution π** (soft policy, richer than a one-hot move)
  and the game outcome for the value head (win/loss, BCE). Forced moves excluded from the policy
  loss. Logits-only losses (the V15.3 double-softmax bug cannot recur).
- **Streaming, no stored dataset** (2026-10-03): Rust generator threads (CPU only — MCTS needs
  no network) push finished games into an **in-memory shuffle buffer**; the GPU learner samples
  from it. Generation is throttled to the learner's consumption rate (back-pressure), so spare
  CPU stays free. Only a **fixed held-out validation set** is stored (~2,000 games, a few MB),
  so convergence is always measured on the same games.
- **Throughput reality**: generation ≈ 2,700 positions/s, training ≈ 600/s → training is the
  bottleneck. 1M games ≈ 160M positions; training on all would take ~3 days. Positions inside a
  game are highly correlated, so each game's positions are subsampled (~1 in 4) — ~20 h for the
  full 1M-game budget, and the auto-stop below usually ends SL earlier.
- **Auto-stop (tight and consistent)**:
  - checks every ~2M trained positions on the fixed validation set: soft cross-entropy vs the
    teacher's π, top-1 agreement on multi-option decisions, value BCE; plus a strength check
    (1,000 paired-dice games vs MCTS-3000 and vs v15.2);
  - **reduce-on-plateau first**: when the metrics stall, LR × 0.3 (max 2 drops) — a plateau can
    be an LR artefact;
  - **stop** only at the lowest LR, when *every* metric shows no improvement beyond noise /
    a small threshold (e.g. CE < 0.2% relative, top-1 < 0.2 pp, strength within its confidence
    interval) in **3 consecutive checks**;
  - guards: no stop before 300k games' worth of data; hard cap 1M games;
  - on stop: SL checkpoint saved, **RL starts automatically** (flag to pause for review instead).
- **Checks during SL**: held-out (by game) cross-entropy and top-1 on multi-option moves, and
  periodic 1,000-game evals vs Depth2 / v15.2 / MCTS-3000. Expect it to approach — not exceed —
  the teacher; exceeding it is RL's job.
- **Unified-model note**: SL uses 2P games only. 4P SL later uses the fixed 4P MaxN search
  (38.5% vs 3× Expectimax, fair share 25%) the same way.

### 3.1 Stage 2 — RL opponents and data generation

RL starts from the SL checkpoint. SL alone is bounded by its teacher (V15.3's buggy SL ended
at 38% vs v15.2); RL is what can go past it. Reward = win/loss only. Each game picks one of
three kinds of opponent:

1. **Current self (self-play).** Both seats are the learner → 2× data per game. Unbounded:
   it gets harder exactly as fast as the learner improves (AlphaZero / TD-Gammon).
2. **Past selves (league).** Snapshots saved every N updates (capped set on disk, sampled
   with a preference for recent ones). Prevents the forgetting/cycling seen in V6 (its 170k
   snapshot beat its latest 60–40).
3. **Progressive bot ladder (your decision).** Rungs, weakest → strongest:
   Heuristic → Expert → Depth2 → **MCTS-3000** (fixed Rust search bot — v15.2 level, ~1 ms/move
   on the CPU). Random is dropped (you called it wasted compute).
   - **Promotion**: move up a rung when the learner beats the current rung **≥ 60% in K
     consecutive evaluation windows** (e.g. K = 3 windows of 300 games, seat-alternated,
     paired dice); **demotion** if it falls below 45% in K consecutive windows. Thresholds,
     K and window size are flags. After SL it will likely start on the upper rungs.
   - At the top rung MCTS-3000 stays as a permanent sparring partner.
   - Only the learner's own decisions are trained on in bot games; bots are never imitated.

**Why not bots only:** training mostly against fixed bots exploits their specific weaknesses and
stops transferring ("twosignal" rose vs bots while falling vs the champion; bot evals saturate at
80–85%). The strongest bot is only v15.2-level, so a bot-only curriculum has a ceiling; self-play
doesn't.

Starting mix (every share is a flag, changeable on resume): 35% current ladder rung / 45% self /
20% past selves, shifting toward 15% / 60% / 25% once the top rung is reached.

**Other details.** Exploration = sampling from the policy (temperature 1); evaluation is
greedy. Rules = the Rust engine (`GameState::roll`), 2P configuration (seats 0 and 2). A move
cap (e.g. 600 decisions) — unfinished games carry no reward.

### 3.2 Learning
- **Reward**: +1 win, −1 loss, at game end only. Nothing else.
- **Advantage**: GAE, λ = 0.95, γ = 0.999 (v15.2's validated values), computed per player
  over that player's own decisions; advantages normalised per batch.
- **Policy loss**: PPO clipped (ε = 0.2) on logits (`forward_logits`, never on probabilities);
  forced moves excluded.
- **Value loss**: BCE of the value logit vs win/loss (the V10.2 lesson); all decisions.
- **Entropy**: computed only on multi-option states, normalised by log(#legal); a small
  coefficient with a schedule (or an entropy target), logged per phase of game.
- **PPO cadence**: buffer of ~512 games → 2 epochs → minibatch 512 (micro-batched for memory)
  → refresh actor weights. KL early stop per update (target ~0.01–0.02).
- **Learning rate**: from random init a higher LR is needed than v15.2's 1e-5 fine-tuning
  (planned 3e-4 → cosine decay, warmup); Phase 0 includes a short LR sanity sweep.

### 3.3 Execution layout on one Mac
The GPU can't train and serve games at full speed at the same time, so:
- **GPU**: the learner (PyTorch-MPS, or MLX if Phase 0 shows it faster and exact).
- **CPU worker(s)**: generate the next buffer of games *while* the GPU trains on the current
  one (~300 games/min on 6 threads ≈ one 512-game buffer per ~2 min).
- PPO tolerates one-buffer staleness (importance ratio corrects it).

### 3.4 Throughput budget
- Training ≈ 550–600 samples/s on MPS today (batch 512, fp32, micro-batched).
- One 2P self-play game ≈ 160 decisions; ~118 of them have ≥ 2 options.
- 2 PPO epochs → ~2.5 games/s → **~200k games/day**, or ~1.4M games/week.
- Phase 0 tries to raise this exactly (fused attention in training, MLX training).

---

## 4. Architecture — one graph model for 2P now, 4P later (your proposal)

### 4.1 Idea
Build the network for the **full 4-player board** from day one, but train it **only on 2-player
games first**. In 2P the two empty seats are simply inactive: their channels are empty and the
rules engine (already supporting `active_players`) drives a 2P game. Later, 4P games are mixed
in and the **same** network learns 4P — no new architecture, no re-training from scratch.

### 4.2 Concretely
- **Backbone**: V15 GraphTransformer, same size as v15.2 (d = 128, 4 layers, 4 heads, ffn 256,
  ~0.6M params, history = 1 frame). 226 nodes (225 cells + CLS) → **same compute as v15.2**.
- **Input per cell, 5 features** (exists in Rust `encode_frame_4p_into` and Python
  `encoder_4p.py`): me / next seat / opposite seat / previous seat / safety, all in
  relative turn order from the mover's view. In 2P the opponent is the opposite seat; next and
  previous are inactive (`-1` everywhere). Add an explicit **active-seat flag** on the special
  cells so the model is never guessing whether a seat is empty or just has no tokens out.
- **Graph edges**: the 4P edge matrix (`graph_4p.py`, 16 edge types).
- **Policy head**: 225 source cells (same as v15.2 — permutation-symmetric over tokens, the
  property that removed the "laggard/spawn" flaws).
- **Value head**: 4-way softmax "which seat wins" (relative seats), **inactive seats masked
  out** → in 2P it reduces exactly to P(I win). (Gemini's 4P design used this head.)
- **Optional later, outcome-derived auxiliary**: final placement/rank per seat (a real game
  outcome, not shaping) — only for 4P, off in Phase A.

**Reuse from the September 4P work (code review 2026-10-03):** the 5-feature encoders (Rust and
Python, rotation-consistent, in parity), the 16-type 4P edge matrix, and the 4-way relative-seat
value head design are sound and reusable. Fix before reuse: Rust maps base tokens to per-slot
base cells while the frame spread-fills by count and Python maps all base tokens to (2,2) —
pick the slot-agnostic (2,2) convention everywhere. Add the active-seat flag. Small graph gaps:
only the last home-stretch cell links to the scored cell, only the first base cell links to the
spawn cell. Do not reuse the old weights, the 4P PPO trainer (no value baseline; shaping that
rewards progress, never punishes) or the winner-only generator.

### 4.3 Open architectural choices (to decide in discussion, one variable at a time)
- Value head reads only CLS (v15.2) vs CLS + pooled board nodes (addresses the documented poor
  capture-risk encoding in CLS). Default: v15.2's CLS head; pooled head as a controlled A/B.
- History frames: keep 1 (champion), revisit later.
- Entity transformer (V12-style, 10–18 tokens instead of 226) is far cheaper and had the best
  SL accuracy ever (95.9%), but v15.2's cell design fixed real flaws. Not in Phase A.

### 4.4 Costs / risks of the unified model
- Slightly bigger input (5 vs 3 features per cell); negligible compute change.
- 2P-only training never exercises the next/previous channels → 4P later is real
  multi-task learning, not free. Mitigation: when 4P is added, mix 2P and 4P games and keep
  2P evals running so 2P strength can't silently drop.
- Benefit: one model, one pipeline, one encoder; 4P becomes an extension, not a rewrite.

---

## 5. Phase B — search-improved training (when self-play progress stalls)

- **Gate (must pass first)**: the 1-ply-by-value test — playing by the network's own value
  with exact dice expectations must score **≥ 50% against the network's own policy**
  (v15.2's value scored 35% here; that is why search on top of it failed).
- **Method**: expert iteration. Self-play games use network-guided search with **exact dice
  expectations** (`net_expecti`) and a policy-improvement rule that is safe with few
  simulations (Gumbel-style root selection); the policy is trained on the search's improved
  choices, the value on outcomes. **The final model still plays without search.**
- **Cost reality**: every network call is ~0.55 ms of M4 GPU; a search of S evaluations per
  move makes self-play ~S× slower. Budgets of 16–64 evals/move are the realistic range locally.

---

## 6. Phase C — 4-player

Once 2P success is reached (or Phase B concludes): mix in 4P self-play games (4 seats, all
learners/snapshots), 4-way value head fully active, optional placement auxiliary,
4P evaluation suite (seat-rotated, ≥ 1,000 games, paired dice) with 2P evals still running.

---

## 7. Engineering — built cleanly, verified before any long run

### 7.1 Components (Phase 0)
1. **Model** `ludo_v17` (unified 2P/4P): model + `forward_logits`; MLX inference port with an
   exactness test against PyTorch.
2. **Encoder**: Rust 5-feature encoder as the single source; Python reference used only in
   tests (bit-exact check on ≥ 100k real states, both seat counts).
3. **NetFarm extensions**: a `sample` seat (policy sampling, records the log-prob and value of
   every decision), league opponent selection, 4P-capable game setup.
4. **PPO learner**: logits-only losses, GAE per player, BCE value, normalised entropy, KL stop,
   micro-batching, exact resume (model, optimizer, step, league, RNG, metrics).
5. **Evaluator**: batched, paired dice, the fixed reference suite (v15.2, Depth2, Expert,
   Heuristic, MCTS-3000, own ladder); runs on the CPU between learner steps or on a schedule.
6. **Dashboard**: one page — games trained, throughput, losses, entropy/KL/clip-fraction,
   eval curves with confidence bands, league table.
7. **Launcher**: one `start_v17.sh` (idempotent), `touch stop`, everything detached.
8. **Resource control (flags, changeable on resume)**: number of CPU actor processes and their
   threads, GPU on/off for actors, learner batch / micro-batch, eval cadence and size, a
   "background mode" preset (few threads, for when you're working on the Mac) and a "full
   power" preset (overnight).

### 7.2 Invariants checked automatically (fail fast)
- Encoder parity Rust vs Python; rotation invariance across seats.
- Legal-mask correctness (sampled action always legal; forced moves flagged).
- Loss sanity on a fixed batch (CE on logits matches a reference implementation).
- Value target sign per player; GAE on a hand-computed toy trajectory.
- Learning-rate actually non-zero after resume; optimizer state restored.
- Eval harness reproduces known numbers (v15.2 vs Expert+Heuristic ≈ 84.9%).

### 7.3 Throughput experiments (exact only — no fp16)
- Fused attention (SDPA) in the training model: equal within float rounding? faster/leaner?
- MLX training vs PyTorch-MPS: speed, and gradient equality on a fixed batch.
- Training batch / micro-batch size vs memory and speed.

### 7.4 Repository hygiene
New, self-contained package (e.g. `v17/`) using the Rust crates; old pipelines untouched and
read-only. One journal section per milestone; MODEL_HISTORY entry at each gate.

---

## 8. Timeline

No end date — the run is infinite. Rough shape on the M4:

| step | what | wall time |
|---|---|---|
| 0 | build + test every component, throughput experiments, 1-hour smoke run | a few days of build |
| A | the infinite run starts (2P); first meaningful evals after ~25k games | continuous |
| B | (only if the diagnostic rule fires and the 1-ply gate passes) search-improved training | continuous |
| C | 4P games mixed into the same run | when you decide |

At ~200k games/day (full power), 1.5M games ≈ a week.

---

## 9. Risks and what we do about them

| risk | mitigation |
|---|---|
| Random-init self-play never takes off (no signal early) | sanity gate at ~10k games; random play still produces wins/losses every game; TD-Gammon precedent; if it fails, a stated fallback (tiny curriculum: shorter game cap / faster LR), never bot imitation |
| Progress stalls | diagnostic rule (§2) → change one variable, or Phase B (search), decided with data |
| Self-play cycles / forgetting | snapshot league, ladder evals |
| Entropy collapse or bleed | normalised entropy on multi-option states, schedule, monitored |
| Training too slow for 1.5M games | Phase 0 throughput work; CPU actors overlap; still ~1 week |
| Unified 2P/4P adds bugs | encoder/rotation tests; 2P evals stay on when 4P arrives |
| Mac memory | micro-batching (identical gradients), fixed shapes (MPS cache), measured |

---

## 10. Decisions

Decided (2026-10-03):
- Unified 2P/4P graph model (V15-style GraphTransformer, 5-feature 4P encoding), 2P first.
- No finish line: infinite, safely interruptible runs; resource flags.
- Months of local training are fine.
- Naming: V17 family (`v17-2p`, `v17-mix`).
- SL streams (no stored dataset) with a tight auto-stop (§3.0) and automatic hand-off to RL.
- Stage 1: up to 1M games of SL, MCTS-3000 vs MCTS-3000, every move of both seats labelled (visit
  distribution) + outcome for the value head; Stage 2: RL with a progressive bot
  ladder (consecutive-window promotion) plus current/past selves, win/loss reward only.

Open (sensible defaults chosen; all are flags or later A/Bs):
1. RL opponent mix and ladder thresholds in §3.1 (defaults: 35/45/20 → 15/60/25; promote
   ≥ 60% in 3 consecutive 300-game windows, demote < 45%).
2. Value head: v15.2-style CLS head (default); pooled head as a later controlled A/B.
