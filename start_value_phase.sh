#!/bin/bash
# ONE command to (re)start the whole value phase after a reboot. Idempotent:
# anything already running is left alone; finished work is never redone.
#   1. dashboard  -> http://localhost:8790/
#   2. self-play  -> data/selfplay_v152 (resumes from existing shards, 200k games)
#   3. chain      -> waits for self-play, trains the value net (resumes from
#                    value_latest.pt), then runs the decisive 1-ply test.
# Stop everything cleanly: touch stop   (then: rm stop; ./start_value_phase.sh to resume)
cd "$(dirname "${BASH_SOURCE[0]}")"
rm -f stop data/selfplay_v152/stop checkpoints/value_v1/stop
mkdir -p checkpoints/value_v1 data/selfplay_v152
P=./td_ludo/td_env/bin/python

if ! pgrep -f value_phase_dashboard.py > /dev/null; then
  nohup $P -u python/alphaludo/value_phase_dashboard.py --port 8790 >> checkpoints/value_v1/dashboard.log 2>&1 &
  disown; echo "[OK] dashboard  http://localhost:8790/"
else echo "[--] dashboard already running"; fi

# Self-play: two independent workers on one run (dynamic shard claiming).
#   GPU worker: MLX fp32 (argmax-identical to the PyTorch fp32 reference)
#   CPU worker: PyTorch fp32, 6 threads (shares the memory bus with the GPU;
#               measured +~300 games/min net of the slowdown it causes)
rm -f data/selfplay_v152/*.claim data/selfplay_v152/*.tmp
N=$(ls data/selfplay_v152/shard_*.bin 2>/dev/null | wc -l | tr -d ' ')
if [ "$N" -lt 40 ]; then
  if ! pgrep -f "gen_selfplay.py.*--backend mlx" > /dev/null; then
    nohup $P -u python/alphaludo/gen_selfplay.py --out data/selfplay_v152 --games 200000 \
        --concurrency 512 --backend mlx --tag "[gpu] " >> data/selfplay_v152/console_gpu.log 2>&1 &
    disown; echo "[OK] self-play GPU worker ($N/40 shards done)"
  else echo "[--] self-play GPU worker already running"; fi
  if ! pgrep -f "gen_selfplay.py.*--device cpu" > /dev/null; then
    nohup $P -u python/alphaludo/gen_selfplay.py --out data/selfplay_v152 --games 200000 \
        --concurrency 256 --device cpu --cpu-frac 0 --threads 6 --tag "[cpu] " >> data/selfplay_v152/console_cpu.log 2>&1 &
    disown; echo "[OK] self-play CPU worker"
  else echo "[--] self-play CPU worker already running"; fi
else echo "[--] self-play complete ($N/40 shards)"; fi

if ! pgrep -f chain_value_phase.sh > /dev/null; then
  if grep -q "test done" checkpoints/value_v1/chain.log 2>/dev/null; then echo "[--] chain already completed"
  else nohup ./chain_value_phase.sh > /dev/null 2>&1 & disown; echo "[OK] chain (train value net -> decisive test)"; fi
else echo "[--] chain already running"; fi

sleep 5
for p in value_phase_dashboard.py "gen_selfplay.py.*mlx" "gen_selfplay.py.*device cpu" chain_value_phase.sh; do
  pgrep -f "$p" > /dev/null && echo "  alive: $p" || echo "  not running: $p"
done
