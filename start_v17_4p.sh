#!/bin/bash
# Start / resume the V17 2P+4P run (idempotent). Everything runs detached — you can close Claude.
#   ./start_v17_4p.sh                     full power (default)
#   PRESET=medium ./start_v17_4p.sh       about half the CPU
#   PRESET=background ./start_v17_4p.sh   few threads, for when you use the Mac
# Pause: touch checkpoints/v17_4p/stop     Resume: ./start_v17_4p.sh
# Dashboard: http://localhost:8791/
#
# First start: the 2P run (checkpoints/v17) must be PAUSED. Its latest RL checkpoint
# is frozen as checkpoints/v17/v17_2p_frozen.pt and becomes this run's init.
cd "$(dirname "${BASH_SOURCE[0]}")"
export RUN="${RUN:-checkpoints/v17_4p}"
SRC=checkpoints/v17
case "${PRESET:-full}" in
  full)       export SL_THREADS="${SL_THREADS:-10}" CONCURRENCY="${CONCURRENCY:-256}" RAYON_NUM_THREADS="${RAYON_NUM_THREADS:-10}" ;;
  medium)     export SL_THREADS="${SL_THREADS:-6}"  CONCURRENCY="${CONCURRENCY:-160}" RAYON_NUM_THREADS="${RAYON_NUM_THREADS:-6}" ;;
  background) export SL_THREADS="${SL_THREADS:-3}"  CONCURRENCY="${CONCURRENCY:-96}"  RAYON_NUM_THREADS="${RAYON_NUM_THREADS:-3}" ;;
  *) echo "unknown PRESET (full | medium | background)"; exit 1 ;;
esac

# One training run on the GPU at a time.
if pgrep -f "v17.(sl|rl)_train --run-dir $SRC " > /dev/null && [ -z "${ALLOW_PARALLEL:-}" ]; then
  echo "[!!] the 2P run ($SRC) is still training — pause it first:  touch $SRC/stop"; exit 1
fi

mkdir -p "$RUN"
if [ ! -f "$RUN/init_2p.pt" ]; then
  [ -f "$SRC/rl_latest.pt" ] || { echo "[!!] $SRC/rl_latest.pt not found"; exit 1; }
  if [ ! -f "$SRC/v17_2p_frozen.pt" ]; then
    cp -p "$SRC/rl_latest.pt" "$SRC/v17_2p_frozen.pt"
    echo "[OK] froze the 2P model: $SRC/v17_2p_frozen.pt"
  fi
  cp -p "$SRC/v17_2p_frozen.pt" "$RUN/init_2p.pt"
  cp -p v17/teacher4.json "$RUN/teacher4_at_start.json"
  [ -f "$SRC/sl_val.npz" ] && [ ! -f "$RUN/sl_val.npz" ] && cp -p "$SRC/sl_val.npz" "$RUN/sl_val.npz"
  echo "[OK] init: $RUN/init_2p.pt (from the frozen 2P model)"
fi

rm -f "$RUN/stop" stop
P=./td_ludo/td_env/bin/python
if ! pgrep -f "v17.dashboard --run-dir $RUN " > /dev/null; then
  nohup $P -u -m v17.dashboard --run-dir "$RUN" --port 8791 --chain run_v17_4p.sh \
      --title "V17 2P+4P — same network, warm-started from the frozen 2P model" >> "$RUN/dashboard.log" 2>&1 & disown
  echo "[OK] dashboard http://localhost:8791/"
else echo "[--] dashboard already running"; fi
if pgrep -f "bash ./run_v17_4p.sh" > /dev/null; then echo "[--] V17 2P+4P chain already running"
else nohup ./run_v17_4p.sh > /dev/null 2>&1 & disown; echo "[OK] V17 2P+4P chain started (preset ${PRESET:-full})"; fi
sleep 5
for p in "v17.dashboard --run-dir $RUN " "bash ./run_v17_4p.sh" "v17.sl_train --run-dir $RUN " "v17.rl_train --run-dir $RUN "; do
  pgrep -f "$p" > /dev/null && echo "  alive: $p" || echo "  not running: $p"
done
