#!/bin/bash
# Shaping-only RL experiment — local Mac run.
#
# Runs V13.5 RL from latest-RL checkpoint with:
#   - v1 dense rewards (score / capture / kill / spawn / forward / home_stretch)
#   - Terminal scaled to ±0.1 (mixed mode — anchors objective without
#     drowning the dense signal under dice variance)
#   - v13_5_no_bots opp mix (V13_2 40% / V13_5_SL 30% / Self 20% / V12_2 10%)
#     — strong neural opps matching V13.5's Phase L. Drops bots entirely
#     since they're saturated at V13.5 strength.
#   - Eval every 4000 games × 1000 games
#   - Dashboard on port 8791
#
# Background-safe: stop with `pkill -f "shaping_exp"`.

set -e

cd "$(dirname "$0")/../.."  # cd to td_ludo/ root

# Env vars that activate the experiment's behavior (defaults preserve normal training)
export LUDO_REWARD_MENU=v1_dense       # switch v11.py to dense reward function
# Mixed-reward config: dense v1 rewards at full magnitude + scaled-down terminal.
# v1 dense per-game total ≈ +1 to +3; terminal at ±0.1 anchors the global
# objective without dominating local shaping. Matches journal Exp 2 "v1 era"
# recipe (the project's best training era) but with terminal toned down
# given Ludo's high dice variance.
export LUDO_TERMINAL_COEFF=0.1
export TD_LUDO_RUN_NAME=v135_shaping_exp

echo "================================================================"
echo "  V13.5 SHAPING-ONLY EXPERIMENT"
echo "================================================================"
echo "  Init:          checkpoints/v135_shaping_exp/model_latest.pt"
echo "  Reward menu:   v1_dense + terminal ±0.1 (MIXED)"
echo "  Opponent mix:  v13_5_no_bots (V13_2 40% / V13_5_SL 30% / Self 20% / V12_2 10%)"
echo "  Eval cadence:  every 4000 games × 1000 games"
echo "  Dashboard:     http://localhost:8791/v13_dashboard.html"
echo "================================================================"

"$(pwd)/td_env/bin/python" -u train_v12.py \
  --resume \
  --model-arch v13_5 \
  --num-res-blocks 10 \
  --num-channels 128 \
  --game-composition v13_5_no_bots \
  --entropy-coeff 0.01 \
  --eval-interval 4000 \
  --eval-games 1000 \
  --port 8791 \
  --device mps \
  --hours 0 \
  --games 0 \
  "$@"
