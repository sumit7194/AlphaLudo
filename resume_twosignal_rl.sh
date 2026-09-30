#!/bin/zsh
# LONG two-signal RL: routed vs mixed, each from ITS OWN SL model, to 330k games.
# Resumable, skips finished arms, saves every 5k games. Safe to re-run any time.
cd /Users/sumit/Github/AlphaLudo
if pgrep -f train_twosignal_rl > /dev/null; then echo "already running (pid $(pgrep -f train_twosignal_rl|head -1))"; exit 0; fi
nohup env PYTHONPATH=td_ludo:td_ludo_v15 td_ludo/td_env/bin/python \
  td_ludo/experiments/twosignal/_run_long.py \
  >> td_ludo/experiments/twosignal/rl_long/run.log 2>&1 &
echo "launched pid $!"
