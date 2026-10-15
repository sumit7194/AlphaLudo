#!/bin/bash
# Start / resume value round 2 (idempotent; finished stages are skipped).
#   dashboard -> http://localhost:8790/   ·  pause: touch stop  ·  resume: ./start_value_round2.sh
cd "$(dirname "${BASH_SOURCE[0]}")"
rm -f stop
P=./td_ludo/td_env/bin/python
mkdir -p checkpoints/value_v2
pkill -f value_phase_dashboard.py 2>/dev/null; sleep 1
nohup $P -u python/alphaludo/value_phase_dashboard.py --round 2 --port 8790 >> checkpoints/value_v2/dashboard.log 2>&1 &
disown; echo "[OK] dashboard  http://localhost:8790/"
if pgrep -f chain_value_round2.sh > /dev/null; then echo "[--] round-2 chain already running"
else nohup ./chain_value_round2.sh > /dev/null 2>&1 & disown; echo "[OK] round-2 chain started"; fi
sleep 5; for p in value_phase_dashboard.py chain_value_round2.sh; do pgrep -f "$p" > /dev/null && echo "  alive: $p" || echo "  NOT running: $p"; done
