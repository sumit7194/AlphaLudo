#!/bin/bash
# run_training.sh - Robustly start AlphaLudo training

# Environment Setup
export PATH="/opt/homebrew/bin:$PATH"
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

echo ">>> Checking Redis..."
if ! redis-cli ping > /dev/null 2>&1; then
    echo ">> Redis not running. Starting..."
    redis-server --daemonize yes
    sleep 1
fi

echo ">>> Stopping any existing training processes..."
# Kill main process and ALL spawned children
# Kill main process and ALL spawned children
pkill -9 -f train_async.py
pkill -9 -f "multiprocessing.spawn"
pkill -9 -f "spawn_main"
pkill -9 -f "multiprocessing.resource_tracker"
pkill -9 -f "src/learner.py"
pkill -9 -f "src/tuner.py"
pkill -9 -f "src/data_worker.py"
sleep 2

echo ">>> Waiting for port 8765 to free..."
for i in {1..10}; do
    if ! lsof -i :8765 >/dev/null 2>&1; then
        break
    fi
    echo "Port 8765 still in use, killing..."
    lsof -ti :8765 | xargs kill -9 2>/dev/null
    sleep 1
done

echo ">>> Cleaning stop signals..."
rm -f data/stop_signal
rm -f data/actor_stats.json
rm -rf data/cmds
mkdir -p data/cmds

echo ">>> Recompiling C++ Extensions..."
python3 setup.py build_ext --inplace

# echo ">>> Starting Auto-Tuner (Background)..."
# python3 -m src.tuner > data/tuner.log 2>&1 &
# TUNER_PID=$!
# echo "Tuner PID: $TUNER_PID"

# Trap to kill tuner on exit
# trap "kill $TUNER_PID" EXIT

echo ">>> Starting Async Training..."
# MPS Hardware Optimization Environment Variables
export PYTHONUNBUFFERED=1
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0   # Prevent memory fragmentation
export PYTORCH_ENABLE_MPS_FALLBACK=1          # Fallback for unsupported ops

python3 train_async.py
