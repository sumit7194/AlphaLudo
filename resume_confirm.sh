#!/bin/zsh
# Resume the routing-confirmation sweep. Skips completed runs; safe to re-run.
cd /Users/sumit/Github/AlphaLudo
if pgrep -f confirm_routing > /dev/null; then echo "already running (pid $(pgrep -f confirm_routing|head -1))"; exit 0; fi
nohup env PYTHONPATH=td_ludo:td_ludo_v15 td_ludo/td_env/bin/python \
  -m experiments.twosignal.confirm_routing --epochs 6 --seeds 0,1 --device mps \
  >> td_ludo/experiments/twosignal/confirm/run.log 2>&1 &
echo "resumed, pid $!"
