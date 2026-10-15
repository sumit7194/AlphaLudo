#!/bin/bash
# Champion greedy self-play for value learning (detached, resumable).
#   OUT=data/selfplay_v152 GAMES=200000 ./launch_selfplay_v152.sh
# Stop: touch $OUT/stop   (re-run to resume: existing shards are kept)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
OUT="${OUT:-data/selfplay_v152}"; GAMES="${GAMES:-200000}"
mkdir -p "$OUT"; rm -f "$OUT/stop"
if pgrep -f "gen_selfplay.py.*$OUT" > /dev/null; then echo "already running"; exit 1; fi
nohup ./td_ludo/td_env/bin/python -u python/alphaludo/gen_selfplay.py --out "$OUT" --games "$GAMES" \
    --device mps --cpu-frac 0.4 "$@" >> "$OUT/console.log" 2>&1 &
PID=$!; disown $PID; sleep 8
if kill -0 $PID 2>/dev/null; then echo "[OK] self-play pid $PID — tail -f $OUT/console.log"; else echo "[DEAD]"; tail -20 "$OUT/console.log"; fi
