#!/bin/bash
# Fully detached launcher for AlphaLudo V15-4PW Winner Distillation
# Safe to close terminal / Antigravity after launching.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT" || exit 1

PYTHON="$REPO_ROOT/td_ludo/td_env/bin/python"
LOG_FILE="$REPO_ROOT/winner_distill.log"
PID_FILE="$REPO_ROOT/checkpoints/v15_4pw.pid"

# Check if already running
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if ps -p "$OLD_PID" > /dev/null 2>&1; then
        echo "⚠️ Training is already running with PID $OLD_PID."
        echo "   To stop it safely: touch checkpoints/v15_4pw/stop"
        echo "   To view dashboard: http://localhost:8846"
        exit 0
    fi
fi

mkdir -p "$REPO_ROOT/checkpoints/v15_4pw"

echo "🚀 Launching detached V15-4PW Winner Distillation (MCTS + Depth-2)..."
nohup env PYTHONUNBUFFERED=1 "$PYTHON" python/alphaludo/train_winner.py --pace-ms 25 --port 8846 "$@" > "$LOG_FILE" 2>&1 &
TRAIN_PID=$!
echo "$TRAIN_PID" > "$PID_FILE"

sleep 2

if ps -p "$TRAIN_PID" > /dev/null 2>&1; then
    echo "✅ Successfully launched detached background process (PID: $TRAIN_PID)"
    echo "🌐 Live Dashboard: http://localhost:8846"
    echo "📄 Live Logs:      tail -f winner_distill.log"
    echo "🛑 To Pause:       touch checkpoints/v15_4pw/stop"
    echo "💡 You can now safely close Antigravity / your terminal window!"
else
    echo "❌ Failed to start. Check $LOG_FILE:"
    cat "$LOG_FILE"
fi
