#!/usr/bin/env bash
# V15.2 RL — WINNING-RECIPE PARITY build.
# train_v15_rich.py → V15RichTrainer (PPO, mirrors V13.5 ActorCriticTrainerV10:
# γ=0.999 + EMA return norm + win-prob BCE + ratio clamp). GraphTransformer
# 4×128 (history_len 1). Recipe parity:
#   - --use-shaped-reward 1  → DENSE (full v1 shaped) rewards (was score-only)
#   - STRONG-NEURAL opponent pool, NO scripted bots:
#       Hist_V13_2 40 / Hist_V13_5_SL 30 / Self 20 / Hist_V13_5_RL 10
#     (mirrors v13_5_no_bots; expert/heuristic defaults zeroed)
#   - no KL anchor (champion train_v12.py path had none)
# KNOWN PARITY GAP: V15 dense shaping has no bias-penalty hook yet (the
# td_ludo path uses LUDO_BIAS_PENALTIES=1). Negligible for GPM; close before
# the real training run.
# Init from the 1-epoch SL checkpoint (v152_sl_final.pt).
set -uo pipefail
cd /Users/sumit/Github/AlphaLudo/td_ludo_v15
export PYTHONPATH=.:../td_ludo
PYBIN=../td_ludo/td_env/bin/python

export TD_LUDO_RUN_NAME=v152_rl_parity
RUN_DIR=checkpoints/v152_rl_parity
mkdir -p "$RUN_DIR"
LOG=$RUN_DIR/console.log

PORT="${1:-8823}"
EVAL_INTERVAL="${2:-100000000}"

TD=/Users/sumit/Github/AlphaLudo/td_ludo/checkpoints

$PYBIN -u train_v15_rich.py \
  --init checkpoints/h2h_compare/v152_sl_final.pt \
  --d-model 128 --n-layers 4 --n-heads 4 --ffn-dim 256 --history-len 1 \
  --use-shaped-reward 1 \
  --opp-v132    "$TD/v132/model_latest.pt"             --opp-weight-v132 40 \
  --opp-v135-sl "$TD/v135_full/model_latest.pt"        --opp-weight-v135-sl 30 \
  --opp-v135-rl "$TD/v135_prod_rl_local/model_latest.pt" --opp-weight-v135-rl 10 \
  --opp-weight-self 20 \
  --opp-weight-expert 0 --opp-weight-heuristic 0 \
  --device mps --opp-device cpu --port "$PORT" \
  --eval-interval "$EVAL_INTERVAL" --eval-games 2000 \
  >> "$LOG" 2>&1
