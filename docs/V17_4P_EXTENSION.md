# V17 → 2P + 4P: extending the same network to 4-player Ludo

Status: **closed 2026-10-15** at RL update 408 (208k RL games): RL plateaued and the search-headroom test
showed no gain. Final weights exported to `AlphaLudo-WebPlay/model_checkpoints/v17_final/`; see the training journal.

## Why this sequence

The V17 network already reads 4P boards (relative seats; absent seats are -1; the
4-way value head covers only the active seats), but 2P training never exercised the "next" and
"previous" seat channels. Zero-shot, the 2P model is poor at 4P: **9.9% vs 3×
Expectimax, 22.5% vs 3× Heuristic** (fair share 25%).

The sequence was chosen to give 4P most of the capacity at first without losing 2P:

| stage | 4P share | 2P share | ends when |
|---|---|---|---|
| SL A | 85% of positions | 15% anchor (same MCTS-3000 teacher as V17) | 4P validation plateau (≥ 4M positions in stage) |
| SL B | 50% | 50% | LR drops ×2 then plateau → `sl_final.pt` (6M–40M positions) |
| RL | 50% of training decisions | 50% | never (infinite, `touch stop`) |

(Changed 2026-10-08 from "RL A 80/20 → RL B 50/50": after SL A the model already played 4P at
teacher level and 15% 2P had let 2P slip. The 4P ladder starts at depth2. Per-run RL overrides go in
`<run>/rl_overrides.json`.)

- 2P never drops to 0%. Published work on forgetting (RL fine-tuning, replay) shows
  that skills in states which stop being visited get overwritten; a small replay share
  prevents it. Each stage changes only the ratio.
- Shares are measured in **positions / decisions, not games**. A 4P game has ~2.5–4×
  more decisions, so splitting by games would silently over-weight 4P.
- Warm start: weights from the frozen 2P model, fresh optimizer, LR 1e-4 (V17 used 3e-4)
  with a 1,000-step warm-up.

## The 4P teacher (and top 4P ladder rung)

Measured on the existing bots (1 bot vs 3 copies of another; 25% = equal strength):

- Expert = Heuristic in 4P (25% vs each other); both lose badly to the 1-ply
  Expectimax (9%).
- **Max^n MCTS did not scale**: 34% vs 3× Expectimax at 1k, 3k and 10k simulations
  alike. Its leaf evaluator was fitted on 2P games and, in 4P, compared "me" with the
  SUM of the three opponents. Its log-loss in 4P was **1.413, worse than a uniform
  guess (1.386)**.
- A 4P evaluator fitted on 4P games (`eval4.rs`: conditional logit over seats, per-seat
  features) is well calibrated (log-loss 1.105), but it ranks moves worse. Greedy on it
  wins 18% vs 3× Expectimax. Outcome prediction and move ranking are different skills.
- **Exact full-round search** (`round4.rs`) replaces sampling. For each candidate move,
  every dice face of every opponent until my next turn is enumerated (216+ lines, bonus
  rolls included). Each opponent answers greedily for itself, and the move's value is
  the exact expectation. There is no sampling noise, and it takes milliseconds in Rust.
- The leaf is a linear "me vs the others" utility on the per-seat features,
  `w·φ_me − Σ_others u·φ_j`, **tuned for playing strength** by SPSA
  (`v17/tune_round4.py`). All games run in Rust. Alone this gives 34% vs 3× Expectimax,
  29% vs 3× Depth2, but legacy MaxN-3000 still wins head to head.
- **Final teacher: Max^n-3000 with the tuned round search as its root prior.**
  - 39.9% vs 3× Expectimax and 35.2% vs 3× Depth2.
  - Head to head vs legacy MaxN-3000: 26.3% as the lone seat, vs 19.5% the other way.
  - Search scales again on the same seeds: prior alone 33.7% → MaxN-1000 37.8% →
    MaxN-3000 39.9%.
  - Its edge over Depth2 (strength ratio ≈ 1.6) is close to the 2P teacher's (≈ 1.7).
  - It produces soft visit-distribution targets.
  - Data rate is about 230 positions/s with 8 threads at an 85% 4P mix, so SL is
    teacher-bound; estimated 8–20 h.

Config file: `v17/teacher4.json` (single source of truth; read by the SL stream, the 4P
evals and the RL ladder; copied into the run dir at first start).

## Files

| file | what |
|---|---|
| `crates/ludo_search/src/eval4.rs` | per-seat 4P features + calibrated win-share evaluator |
| `crates/ludo_search/src/eval4_data.rs` | parallel bot-game dataset for fitting |
| `crates/ludo_search/src/round4.rs` | exact full-round expectimax bot (leaf: score / eval4 / linear) |
| `crates/ludo_search/src/mcts.rs` | Max^n now configurable (leaf, rollout, c, prior) + `search()` visit distribution |
| `crates/ludo_search/src/sl_stream.rs` | mixed 2P/4P teacher stream, `frac4` by positions, adjustable live |
| `crates/ludo_search/tests/v17_4p_test.rs` | rotation symmetry, distributions, blockades, round/MaxN/stream checks |
| `v17/fit_eval4.py`, `v17/tune_round4.py` | evaluator fit, SPSA tuning |
| `v17/sl_train.py`, `v17/rl_train.py` | `--init`, `--frac4-stages`, 4P validation/strength, 4P ladder (defaults = old 2P behaviour) |
| `v17/evaluate.py` | `4p:<bot>` references, `FOURP_SUITE` |
| `v17/dashboard.py` | 4P panels appear automatically for mixed runs; `--chain`, `--title` |
| `start_v17_4p.sh`, `run_v17_4p.sh` | launcher + chain (run dir `checkpoints/v17_4p`, dashboard :8791) |

## Operating it

```bash
touch ~/Github/AlphaLudo/checkpoints/v17/stop          # 1. pause the 2P run, wait for "[exit] clean"
cd ~/Github/AlphaLudo && ./start_v17_4p.sh             # 2. start (detached; PRESET=background for light use)
touch ~/Github/AlphaLudo/checkpoints/v17_4p/stop       # pause;  ./start_v17_4p.sh resumes
```

- At first start the 2P model is frozen as `checkpoints/v17/v17_2p_frozen.pt` (the
  start script refuses to run while the 2P run is training).
- 4P ladder thresholds are the Luce-model equivalents of the 2P ones: promote at ≥ 33.3%
  vs 3 copies (= 60% in 2P), demote below 21.4% (= 45%).
- "Don't forget 2P" check: the 2P eval suite runs unchanged in every full evaluation.
  Compare it with the frozen model's last 2P evals.
