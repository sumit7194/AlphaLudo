#!/usr/bin/env bash
# Exp 62 (2026-06-21) — RICHER VALUE TARGET retrain (the squeeze lever).
# Continue v15.2 with the value head trained on the GAE bootstrapped λ-return
# (MSE) instead of pure terminal win/loss (BCE). Hypothesis: lower-variance
# value target → better advantage estimates → policy reaches a marginally higher
# optimum. Judged by PAIRED-DICE A/B vs v15.2 (the only honest test at the
# ceiling). Same strong-neural pool as the gaeterminal champion run. Init from
# v15.2 best. Sub-percent shot — the definitive last 2P experiment.
set -uo pipefail
cd /home/sumit/AlphaLudo/td_ludo_v15
export PYTHONPATH=.:/home/sumit/AlphaLudo/td_ludo
PYBIN=python3
export TD_LUDO_RUN_NAME=v152_richvalue
RUN_DIR=checkpoints/v152_richvalue
mkdir -p "$RUN_DIR"
LOG=$RUN_DIR/console.log
TD=/home/sumit/AlphaLudo/td_ludo/checkpoints
V152=/home/sumit/AlphaLudo/td_ludo_v15/checkpoint_backups/v152_gaeterminal_2026-06-19/v152_gaeterminal_BEST_eval84.65pct_2026-06-19.pt
V136BEST=/home/sumit/AlphaLudo/td_ludo/checkpoint_backups/v136_gaeterminal_2026-06-17/v136_gaeterminal_BEST_eval85.5pct_G200k_2026-06-16.pt

RESUME_FLAG=""
if [[ -f "$RUN_DIR/model_latest.pt" ]]; then RESUME_FLAG="--resume"; fi

CMD="cd /home/sumit/AlphaLudo/td_ludo_v15 && PYTHONPATH=.:/home/sumit/AlphaLudo/td_ludo \
TD_LUDO_RUN_NAME=v152_richvalue PYTHONUNBUFFERED=1 $PYBIN -u train_v15_rich.py \
  --init $V152 \
  --d-model 128 --n-layers 4 --n-heads 4 --ffn-dim 256 --history-len 1 \
  --use-gae --gae-lambda 0.95 --terminal-only --use-shaped-reward 0 \
  --value-target lambda \
  --opp-v132    $TD/v132/model_latest.pt  --opp-weight-v132 30 \
  --opp-v135-rl $V136BEST                 --opp-weight-v135-rl 40 \
  --opp-weight-self 30 \
  --opp-weight-expert 0 --opp-weight-heuristic 0 \
  --entropy-coeff 0.03 --lr 1e-5 \
  --device cuda --opp-device cuda --port 8801 \
  --parallel-games 128 \
  --eval-interval 10000 --eval-games 2000 \
  $RESUME_FLAG \
  >> $LOG 2>&1"
setsid bash -c "$CMD" </dev/null >/dev/null 2>&1 &
disown
sleep 16
PID=$(ps -eo pid,args | grep "[t]rain_v15_rich" | awk '$2=="python3"{print $1}' | head -1)
if [[ -z "$PID" ]]; then echo "FAILED — log:"; tail -40 "$LOG"; exit 1; fi
echo "launched RICH-VALUE retrain: PID=$PID"
sleep 8
tail -20 "$LOG"
