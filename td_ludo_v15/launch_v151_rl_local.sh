#!/usr/bin/env bash
# V15.1 RL training — LOCAL (Mac MPS) launcher.
#
# Strong-opponent recipe (2026-05-20):
#   - Init from V15.1 RL latest weights (preserves the ~270K games of prior
#     RL training — model weights, not trainer-dir metrics).
#   - Writes to a FRESH checkpoint dir (`v151_rl_strong/`) so the baseline
#     v151_rl/ run stays preserved as a comparison anchor.
#   - Opponent mix: 25 self / 35 Expectimax / 10 MCTSPure / 15 Expert / 15 Heuristic.
#     No V13.5-family neural opps — those are the lineage we're trying to
#     escape (every model from V6→V15.1 trained against the same Expert-family
#     mix and converged to the same ~85% bot-WR attractor).
#   - KL anchor still pulls toward V15.1 SL with a small weight (0.05) to
#     keep the policy from collapsing during early training against the
#     stronger opp pool.
#
# Behaviour:
#   - First call: --init from v151_rl/model_latest.pt → new v151_rl_strong/ dir
#   - Subsequent calls: --resume from v151_rl_strong/model_latest.pt
#   - Logs APPENDED so resume history is preserved
#   - Dashboard on http://localhost:8790/v13_dashboard.html

set -euo pipefail
cd "$(dirname "$0")"

V15_DIR=/Users/sumit/Github/AlphaLudo/td_ludo_v15
LEGACY_DIR=/Users/sumit/Github/AlphaLudo/td_ludo
PYBIN=$LEGACY_DIR/td_env/bin/python

RUN_NAME=v151_rl_strong
RUN_DIR=$V15_DIR/checkpoints/$RUN_NAME
mkdir -p "$RUN_DIR"
LOG=$RUN_DIR/console.log

# Init weights — V15.1 RL latest (~G=270K). Preserves prior training compute.
INIT_WEIGHTS=$V15_DIR/checkpoints/v151_rl/model_latest.pt
# Resume target inside THIS run dir (gets populated after the first save)
LATEST=$RUN_DIR/model_latest.pt

if [[ -f "$LATEST" ]]; then
  echo "[launcher] resuming v151_rl_strong from $LATEST"
  MODE_FLAG="--resume"
elif [[ -f "$INIT_WEIGHTS" ]]; then
  echo "[launcher] fresh strong-opp run, initializing from $INIT_WEIGHTS"
  echo "             (preserves v151_rl/'s G=270K RL weights as the starting point)"
  MODE_FLAG="--init $INIT_WEIGHTS"
else
  echo "ERROR: $INIT_WEIGHTS missing. Cannot start." >&2
  exit 1
fi

cd "$V15_DIR"
PYTHONPATH="$LEGACY_DIR:." \
  TD_LUDO_RUN_NAME=$RUN_NAME PYTHONUNBUFFERED=1 \
  "$PYBIN" -u train_v15_rich.py $MODE_FLAG \
    --kl-anchor "$V15_DIR/checkpoints/v151_sl/model_sl.pt" \
    --kl-anchor-coeff 0.05 \
    --history-len 2 \
    --d-model 128 --n-heads 4 --n-layers 4 --ffn-dim 256 \
    --opp-weight-self 25 \
    --opp-weight-expectimax 35 \
    --opp-weight-mcts-pure 10 \
    --mcts-pure-sims 50 --mcts-pure-rollouts 8 \
    --opp-weight-expert 15 --opp-weight-heuristic 15 \
    --opp-weight-v135-rl 0 --opp-weight-v135-sl 0 --opp-weight-v132 0 \
    --opp-weight-aggressive 0 --opp-weight-defensive 0 \
    --opp-weight-racing 0 --opp-weight-random 0 \
    --target-states 200000000 \
    --parallel-games 32 --ppo-buffer-games 32 \
    --ppo-minibatch-size 256 --ppo-epochs 2 --ppo-clip 0.2 \
    --lr 1e-5 --entropy-coeff 0.03 --win-bce-coeff 0.5 \
    --temperature 1.0 \
    --eval-interval 10000 --eval-games 2000 \
    --save-interval-sec 120 --log-every 5 \
    --device mps --opp-device cpu \
    --port 8790 \
    >> "$LOG" 2>&1
