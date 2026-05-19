# Shaping-Only RL Experiment

**Hypothesis:** in stochastic games like Ludo, the terminal ±1 reward signal is too noisy (dice variance dominates over 80-150 plies). Pure local-event shaping might give a cleaner learning signal.

**Setup:**
- Init from V13.5 latest RL checkpoint (G=779K)
- Reward menu: v1 dense (`td_ludo/game/dense_rewards.py`)
  - Score token: +0.40
  - Capture enemy: +0.20
  - Got killed: −0.20
  - Home stretch entry: +0.10
  - Spawn: +0.05
  - Forward step: +0.005
- Terminal reward: **0** (removed)
- Opponent mix: `v122_hist` preset (SelfPlay + bots + Hist_V12_2/V10/V6.3/V6.1)
- Eval: 1000 games every 4000 games
- Device: MPS (Apple Silicon)
- Dashboard port: 8791

**Files touched (all behind env-var defaults so other training unaffected):**
- `td_ludo/game/dense_rewards.py` (new)
- `td_ludo/game/test_dense_rewards.py` (new, 12 tests)
- `td_ludo/game/players/v11.py` (env-var swap when `LUDO_REWARD_MENU=v1_dense`)
- `td_ludo/training/trainer_v10.py` (env-var scaling via `LUDO_TERMINAL_COEFF`)

**Run:** `bash experiments/shaping_only/run_local.sh`
