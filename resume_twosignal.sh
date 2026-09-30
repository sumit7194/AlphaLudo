#!/bin/zsh
# Resume the two-signal experiment. Skips completed runs; safe to re-run.
cd /Users/sumit/Github/AlphaLudo
if pgrep -f train_twosignal > /dev/null; then echo "already running (pid $(pgrep -f train_twosignal|head -1))"; exit 0; fi
nohup env PYTHONPATH=td_ludo:td_ludo_v15 td_ludo/td_env/bin/python \
  -m experiments.twosignal.train_twosignal --epochs 6 --seeds 0,1 --device mps \
  >> td_ludo/experiments/twosignal/run.log 2>&1 &
echo "resumed, pid $!"
