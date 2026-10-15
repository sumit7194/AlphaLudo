#!/bin/bash
# Start / resume V17 (idempotent). Everything runs detached — you can close Claude.
#   ./start_v17.sh                 full power (default)
#   PRESET=medium ./start_v17.sh       about half the CPU
#   PRESET=background ./start_v17.sh   few threads, for when you use the Mac
# Pause: touch checkpoints/v17/stop     Resume: ./start_v17.sh
# Dashboard: http://localhost:8790/
cd "$(dirname "${BASH_SOURCE[0]}")"
export RUN="${RUN:-checkpoints/v17}"
case "${PRESET:-full}" in
  full)       export SL_THREADS="${SL_THREADS:-10}" CONCURRENCY="${CONCURRENCY:-256}" RAYON_NUM_THREADS="${RAYON_NUM_THREADS:-10}" ;;
  medium)     export SL_THREADS="${SL_THREADS:-6}"  CONCURRENCY="${CONCURRENCY:-160}" RAYON_NUM_THREADS="${RAYON_NUM_THREADS:-6}" ;;
  background) export SL_THREADS="${SL_THREADS:-3}"  CONCURRENCY="${CONCURRENCY:-96}"  RAYON_NUM_THREADS="${RAYON_NUM_THREADS:-3}" ;;
  *) echo "unknown PRESET (full | medium | background)"; exit 1 ;;
esac
mkdir -p "$RUN"; rm -f "$RUN/stop" stop
P=./td_ludo/td_env/bin/python
if ! pgrep -f "v17.dashboard" > /dev/null; then
  nohup $P -u -m v17.dashboard --run-dir "$RUN" --port 8790 >> "$RUN/dashboard.log" 2>&1 & disown
  echo "[OK] dashboard http://localhost:8790/"
else echo "[--] dashboard already running"; fi
if pgrep -f "run_v17.sh" > /dev/null; then echo "[--] V17 chain already running"
else nohup ./run_v17.sh > /dev/null 2>&1 & disown; echo "[OK] V17 chain started (preset ${PRESET:-full})"; fi
sleep 5
for p in v17.dashboard run_v17.sh "v17.sl_train" "v17.rl_train"; do
  pgrep -f "$p" > /dev/null && echo "  alive: $p" || echo "  not running: $p"
done
