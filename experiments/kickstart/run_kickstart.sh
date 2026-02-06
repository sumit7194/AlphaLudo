#!/bin/bash
# Streaming Kickstart Pipeline v2 Launcher
# - Activates virtual environment
# - Starts dashboard server
# - Runs stream_trainer.py
# - Handles graceful shutdown on Ctrl+C

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR/../.."

echo "=============================================="
echo "🚀 Streaming Kickstart Pipeline v2"
echo "=============================================="
echo "Press Ctrl+C to gracefully stop training."
echo "The model and checkpoint will be saved."
echo ""

# Activate virtual environment
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
    echo "✅ Virtual environment activated"
else
    echo "⚠️  No .venv found, using system Python"
fi

# Export production mode
export ALPHALUDO_MODE=PROD

# Start dashboard server in background
echo "🖥️  Starting dashboard server on port 8095..."
python3 -m http.server 8095 --bind 0.0.0.0 --directory experiments/kickstart > /dev/null 2>&1 &
PID_DASHBOARD=$!
echo "   Dashboard: http://localhost:8095/dashboard.html"
echo ""

# Cleanup function
cleanup() {
    echo -e "\n\n🔴 Shutting down..."
    kill -TERM $PID_TRAIN 2>/dev/null
    sleep 2
    kill $PID_DASHBOARD 2>/dev/null
    echo "✅ All processes stopped."
    exit 0
}

# Trap Ctrl+C and SIGTERM
trap cleanup SIGINT SIGTERM

# Start the trainer
python3 experiments/kickstart/stream_trainer.py &
PID_TRAIN=$!
echo "🏋️  Training started (PID: $PID_TRAIN)"
echo ""

# Wait for trainer to finish
wait $PID_TRAIN
EXIT_CODE=$?

# Cleanup dashboard
kill $PID_DASHBOARD 2>/dev/null

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✅ Training completed successfully!"
else
    echo ""
    echo "⚠️  Trainer exited with code $EXIT_CODE"
fi

exit $EXIT_CODE
