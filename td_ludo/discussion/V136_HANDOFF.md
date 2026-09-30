# v13.6 — Handoff / Pick-up Notes (paused 2026-06-17)

We are **pausing the v13.6 line here** and pivoting to test whether the residual
play flaws are *architectural* (see "Strategic conclusion" + "Next: v15.2"). This
doc captures everything needed to resume v13.6 cold.

## Where v13.6 landed
- **Champion = `gaeterminal` best, eval 85.5%** — the strongest model in the
  family. Beats the old champion (v136_rl_parity) **54.4%** H2H; parity with
  v13.5_RL_best (51.0%). This is the top of the persistent **~85% variance
  ceiling** that has held across V6/V13.2/V13.5/V15 — touched, not broken.
- The 85% ceiling is a **roster/variance ceiling**, not a skill wall we can
  obviously beat with more of the same. All snapshots are ~50% coin-flips H2H.

## The winning pipeline (the thing worth reusing)
**Terminal-only (±1 win/loss) + GAE.** The breakthrough was realizing
"terminal rewards don't work in Ludo" was actually "*Monte-Carlo* terminal
rewards don't work" — the variance was the problem, not the sparsity.
- GAE (`--use-gae --gae-lambda 0.95`) bootstraps a value baseline →
  low-variance advantage → the sparse terminal signal becomes learnable.
- Code: `td_ludo/td_ludo/training/trainer_v10.py` — `_compute_gae` (lines
  ~200-227), `use_gae` flag, branch in `_ppo_update` (~403-408). λ=1 ≡ MC
  (correctness gate). Value head still BCE-trained on outcome; GAE only reads
  it as baseline.
- Launch ref: `td_ludo/launch_v136_gaeterminal.sh` (entropy 0.01, init champion).

## The two play flaws (target B) and what we tried
1. **Spawn-on-6**: with a token already in the open, a 6 spawns a fresh token
   instead of advancing/chasing.
2. **Laggard neglect**: ignores its own exposed token when an opponent is
   closing in, especially while chasing.

| Attempt | Result |
|---|---|
| World-model heads (Exp 52) | learned capture-risk @ 0.96 corr but behavior unchanged (knowledge ≠ action) |
| Risk-delta reward (Exp 53) | FAILED — potential-based = policy-invariant; hurt eval, fixed nothing |
| Terminal-only + GAE (Exp 55b → gaeterminal) | beat champion + touched 85.5%, but laggard probe 56.8% ≈ baseline (general gain, flaw not fixed) |
| **Full de-bias (Exp 56)** | **laggard 55.9→78.3%** (FIRST real fix) + spawn cut, BUT **−6-7pp eval** (converged ~78%, never recovered over 330K games) |
| Light de-bias (Exp 57) | half-strength laggard penalty, no spawn cut → settled ~79-80% @ 72K. **Halving the penalty did NOT buy back eval** → the eval cost is not magnitude-driven; it's the bias itself pulling off pure win-max. |

## Play-test verdict (human, 2026-06-17)
- Played `v13_6_gae` (gaeterminal, no fixes): old major problems gone; but
  spawn-on-6 and laggard-while-chasing both still visible. **Confirms the probe
  numbers** — gaeterminal has both flaws intact.
- Played `v13_6_debias` (full de-bias): "much better." Spawn-on-6 **softened
  not solved** (advanced the open token first, then spawned the new one *next*
  turn). Aggressive, chases well, knows when to chase. Laggard protection
  improved.

## Strategic conclusion (why we're pivoting)
The de-bias **softens** the flaws via reward shaping rather than the model
genuinely understanding them — and the cheaper light version showed the eval
cost is intrinsic to the bias, not tunable away. **Reward-engineering around
these flaws is not the right call.** Hypothesis: v13.6's **multi-turn input
channels** (stacked past frames) feed the "keep moving the same token /
momentum" behavior. The clean test is a **single-frame** architecture →
**v15.2**.

## Assets / exact paths
- **Backups** (VM + local): `checkpoint_backups/v136_gaeterminal_2026-06-17/`
  - `..._BEST_eval85.5pct_G200k_2026-06-16.pt` (champion)
  - `..._LATEST_G280k_eval82pct_2026-06-16.pt`
  - + `MANIFEST.md` with H2H verdicts + full eval trajectory.
- **De-bias checkpoint** (Exp 56, both fixes, ~78%): on VM at
  `checkpoints/v136_gae_debias/model_latest.pt`; pulled into play server.
  (NOT formally backed up — re-pull from VM if needed.)
- **Play server** (`td_ludo/play/server.py`): models wired —
  `LUDO_MODEL=v13_6_gae` (gaeterminal latest, default flaws),
  `v13_6_debias` (Exp 56, flaws fixed-ish). Each dir also has `model_best.pt`
  where applicable. Start: `LUDO_MODEL=<v> ... server.py` → http://localhost:5050.
- **De-bias knobs** (env-gated, canonical recipe untouched when off):
  - `LUDO_DEBIAS_DENSE=1` → broadened danger penalty (laggard), `dense_rewards.py`
    + `bias_penalties.py`.
  - `LUDO_DEBIAS_SPAWN=1` → spawn 0.05→0.02 (defaults to DENSE for back-compat).
  - `LUDO_DEBIAS_DANGER_MULT=0.5` → softens only the new laggard range.
  - Launch refs: `launch_v136_gae_debias.sh` (full), `launch_v136_gae_debias_light.sh`.
- Journal: through Exp 57. Dashboard reader: `rl_dashboard_standalone.py`
  (`RL_RUN_DIR=<run>`, `RL_DASH_PORT=8790`).

## If we come back to v13.6
- Ceiling-breaking would likely need a genuinely different lever (AIVAT
  dice-luck baseline = GAE stage 2, parked; or 4-player; or new representation),
  not more reward tweaks.
- The de-bias finding stands: we CAN fix the laggard via reward, it just costs
  ~6pp eval — bank that as a known tradeoff knob, not a free win.
