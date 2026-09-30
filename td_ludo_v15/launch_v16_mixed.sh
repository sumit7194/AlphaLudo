#!/usr/bin/env bash
# V16 MIXED — the CONTROL arm. Two-property connections, NOT routed.
#
# Identical to launch_v16_routed.sh in every respect except `--routed 0`.
# Same architecture, same parameter count, same aux head, same aux loss, same
# two-backward structure, same init, same opponents, same hyperparameters.
#
# In mixed mode both backwards run in JOINT mode, and since w = a*(1+tanh(b))
# depends on both properties, EACH loss reaches BOTH of them. That is the whole
# point: it separates "routing signals to separate properties helps" from
# "having two properties / an auxiliary loss helps". Without this arm, any gap
# routed shows could just be the extra parameters or the extra supervision.
set -uo pipefail
cd /Users/sumit/Github/AlphaLudo/td_ludo_v15
export PYTHONPATH=.:/Users/sumit/Github/AlphaLudo/td_ludo
PYBIN=/Users/sumit/Github/AlphaLudo/td_ludo/td_env/bin/python

RUN=v16_mixed
export TD_LUDO_RUN_NAME=$RUN
RUN_DIR=checkpoints/$RUN
mkdir -p "$RUN_DIR"
LOG=$RUN_DIR/console.log
BK=/Users/sumit/Github/AlphaLudo/checkpoint_backups
V136=$BK/v136_gaeterminal_2026-06-17/v136_gaeterminal_BEST_eval85.5pct_G200k_2026-06-16.pt
V132=$BK/v132_20260505_230124/model_latest.pt
INIT=checkpoints/h2h_compare/v152_rl_stage1_G261K/model_best.pt

if pgrep -f "run-name $RUN" > /dev/null; then echo "$RUN already running"; exit 0; fi

# Clear a STALE lock before launching. acquire_train_lock() stores a pid and
# tests liveness with kill(0) — but a reboot recycles pids, so after a power cut
# the lock's pid often belongs to some unrelated process and the lock reads as
# LIVE forever. That silently refused two restarts on 2026-08-05 and training
# sat dead for ~40 minutes. Only remove the lock when no matching trainer is
# actually running.
LOCKF="$HOME/.alphaludo_locks/$RUN.lock"
if [[ -f "$LOCKF" ]] && ! pgrep -f "run-name $RUN" > /dev/null; then
  echo "clearing stale lock $LOCKF"
  rm -f "$LOCKF"
fi

RESUME=""
if [[ -f "$RUN_DIR/model_latest.pt" ]]; then RESUME="--resume"; fi

nohup $PYBIN -u train_v15_rich.py \
  --arch v16 --routed 0 --aux-coeff 1.0 \
  --init "$INIT" \
  --d-model 128 --n-layers 4 --n-heads 4 --ffn-dim 256 --history-len 1 \
  --use-gae --gae-lambda 0.95 --terminal-only --use-shaped-reward 0 \
  --opp-weight-self 20 \
  --opp-weight-expert 0 --opp-weight-heuristic 0 \
  --opp-v132    "$V132" --opp-weight-v132 30 \
  --opp-v135-rl "$V136" --opp-weight-v135-rl 35 \
  --opp-weight-depth2-expectimax 15 \
  --opp-weight-ghost 0 --ghost-interval 5000 --ghost-pool-size 5 \
  --entropy-coeff 0.03 --lr 1e-5 \
  --device mps --opp-device cpu \
  --parallel-games 128 --rollout-workers 6 \
  --eval-interval 10000 --eval-games 2000 \
  --save-interval-sec 600 \
  --port 8802 --run-name $RUN \
  $RESUME >> "$LOG" 2>&1 &
disown
echo "launched $RUN (pid $!)  log: $LOG"
