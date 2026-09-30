#!/bin/zsh
# Resume the 3x-budget low-rank run. Safe to run any time — it SKIPS completed
# runs (recorded in lowrank_long/results.json) and only trains what's missing.
# Detaches from the shell, so closing the terminal / app will not kill it.
cd /Users/sumit/Github/AlphaLudo
if pgrep -f train_long > /dev/null; then echo "already running (pid $(pgrep -f train_long|head -1))"; exit 0; fi
nohup env PYTHONPATH=td_ludo:td_ludo_v15 td_ludo/td_env/bin/python \
  td_ludo/experiments/lowrank_long/train_long.py \
  --device mps --out td_ludo/experiments/lowrank_long/results.json \
  >> td_ludo/experiments/lowrank_long/run.log 2>&1 &
echo "resumed, pid $!"
