#!/usr/bin/env bash
# Exp 58 (2026-06-17) — V15.2 TERMINAL-ONLY + GAE (the v13.6 winning pipeline,
# ported to the single-frame GraphTransformer). Goal: see which of the v13.6
# play flaws (spawn-on-6, laggard-neglect) are ARCHITECTURAL — v15.2 has NO
# multi-turn input (history_len=1), the variable we suspect drives them.
#
# Pipeline parity with gaeterminal: terminal-only (±1, all step rewards zeroed)
# + GAE λ0.95. Entropy 0.03 = the v15.2 MC-baseline default (v15's 225-action
# entropy scale differs from v13.6's 4-action, so we keep v15's tuned value
# rather than copy v13.6's 0.01). Init from the best OLD v15.2 RL model
# (v152_rl_stage1_G261K/model_best.pt, eval 0.796) — same as gaeterminal which
# init'd from the v13.6 RL champion (not SL). Its value head is BCE-trained on
# win/loss (reward-agnostic) so it transfers cleanly to terminal-only + gives
# GAE a real baseline. Strong-neural opponent pool (no scripted bots),
# v136 champion substituted into the v135-rl slot (strongest available;
# auto-arch-probed). Baseline to beat: v152 MC stage1 eval 0.796.
set -uo pipefail
cd /home/sumit/AlphaLudo/td_ludo_v15
export PYTHONPATH=.:/home/sumit/AlphaLudo/td_ludo
PYBIN=python3

export TD_LUDO_RUN_NAME=v152_gaeterminal
RUN_DIR=checkpoints/v152_gaeterminal
mkdir -p "$RUN_DIR"
LOG=$RUN_DIR/console.log
TD=/home/sumit/AlphaLudo/td_ludo/checkpoints
# v13.6 BEST = gaeterminal 85.5% (current champion, beats v136_rl_parity 54.4%).
V136BEST=/home/sumit/AlphaLudo/td_ludo/checkpoint_backups/v136_gaeterminal_2026-06-17/v136_gaeterminal_BEST_eval85.5pct_G200k_2026-06-16.pt

RESUME_FLAG=""
if [[ -f "$RUN_DIR/model_latest.pt" ]]; then RESUME_FLAG="--resume"; fi

CMD="cd /home/sumit/AlphaLudo/td_ludo_v15 && PYTHONPATH=.:/home/sumit/AlphaLudo/td_ludo \
TD_LUDO_RUN_NAME=v152_gaeterminal PYTHONUNBUFFERED=1 $PYBIN -u train_v15_rich.py \
  --init checkpoints/h2h_compare/v152_rl_stage1_G261K/model_best.pt \
  --d-model 128 --n-layers 4 --n-heads 4 --ffn-dim 256 --history-len 1 \
  --use-gae --gae-lambda 0.95 --terminal-only --use-shaped-reward 0 \
  --opp-v132    $TD/v132/model_latest.pt  --opp-weight-v132 30 \
  --opp-v135-rl $V136BEST                 --opp-weight-v135-rl 35 \
  --opp-weight-self 20 \
  --opp-weight-depth2-expectimax 15 \
  --opp-weight-expert 0 --opp-weight-heuristic 0 \
  --entropy-coeff 0.03 --lr 1e-5 \
  --device cuda --opp-device cuda --port 8801 \
  --parallel-games 128 --rollout-workers 6 \
  --eval-interval 10000 --eval-games 2000 \
  $RESUME_FLAG \
  >> $LOG 2>&1"
setsid bash -c "$CMD" </dev/null >/dev/null 2>&1 &
disown
sleep 16
PID=$(ps -eo pid,args | grep "[t]rain_v15_rich" | awk '$2=="python3"{print $1}' | head -1)
if [[ -z "$PID" ]]; then echo "FAILED — log:"; tail -40 "$LOG"; exit 1; fi
echo "launched V15.2 TERMINAL-ONLY + GAE: PID=$PID"
sleep 8
tail -25 "$LOG"
