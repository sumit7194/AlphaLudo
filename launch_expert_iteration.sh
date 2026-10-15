#!/bin/bash
# Expert iteration / AlphaZero loop from the v15.2 champion (detached, resumable).
#   RUN=checkpoints/expit_v1 SIMS=100 GAMES=1000 ./launch_expert_iteration.sh
# Stop cleanly: touch $RUN/stop   (re-run this script to resume)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
RUN="${RUN:-checkpoints/expit_v1}"; SIMS="${SIMS:-100}"; GAMES="${GAMES:-1000}"
DEVICE="${DEVICE:-mps}"; CPU_FRAC="${CPU_FRAC:-0.3}"
mkdir -p "$RUN"; rm -f "$RUN/stop"
if pgrep -f "expert_iteration.py.*$RUN" > /dev/null; then echo "already running: $(pgrep -f "expert_iteration.py.*$RUN")"; exit 1; fi
nohup ./td_ludo/td_env/bin/python -u python/alphaludo/expert_iteration.py \
    --run-dir "$RUN" --sims "$SIMS" --games-per-iter "$GAMES" --device "$DEVICE" --cpu-frac "$CPU_FRAC" \
    "$@" >> "$RUN/console.log" 2>&1 &
PID=$!; disown $PID; sleep 5
if kill -0 $PID 2>/dev/null; then echo "[OK] expert iteration pid $PID — tail -f $RUN/console.log"; else echo "[DEAD] see $RUN/console.log"; tail -20 "$RUN/console.log"; fi
