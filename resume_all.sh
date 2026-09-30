#!/bin/zsh
# ============================================================
#  START / RESUME EVERYTHING  —  safe to run any number of times.
#  Skips whatever is already finished, refuses to double-start.
#  Run this after ANY power loss or restart.
# ============================================================
cd /Users/sumit/Github/AlphaLudo

# 1) the long two-signal RL run (routed vs mixed, 330k games each)
if pgrep -f train_twosignal_rl > /dev/null; then
  echo "RL        : already running (pid $(pgrep -f train_twosignal_rl | head -1))"
else
  nohup env PYTHONPATH=td_ludo:td_ludo_v15 td_ludo/td_env/bin/python \
    td_ludo/experiments/twosignal/_run_long.py \
    >> td_ludo/experiments/twosignal/rl_long/run.log 2>&1 &
  echo "RL        : started (pid $!)"
fi

# 2) dashboard refresher — regenerates the HTML every 2 minutes
if pgrep -f make_dashboard > /dev/null; then
  echo "dashboard : already running"
else
  nohup env PYTHONPATH=td_ludo:td_ludo_v15 zsh -c 'while true; do
      td_ludo/td_env/bin/python td_ludo/experiments/twosignal/make_dashboard.py > /dev/null 2>&1
      sleep 120
    done' >> td_ludo/experiments/twosignal/rl_long/dash.log 2>&1 &
  echo "dashboard : started (pid $!)"
fi

echo
echo "open the dashboard:"
echo "  open td_ludo/experiments/twosignal/rl_long/dashboard.html"
echo "check progress in the terminal:"
echo "  ./status.sh"
