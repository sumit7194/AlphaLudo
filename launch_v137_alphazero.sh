#!/usr/bin/env bash
# ==============================================================================
# Launch Script for AlphaLudo V13.7 — Tabula Rasa AlphaZero 2-Player (N=3,000)
# ==============================================================================
# Features:
#   - 100% Pure AlphaZero tabula rasa (random weights init)
#   - Expecti-MCTS exploration at N=3,000 simulations/move in native multi-core Rust
#   - Terminal rewards only: z in {+1.0, -1.0}
#   - Fully detached daemon (nohup + disown) logging to v137_alphazero.log
#   - Live rich interactive dashboard at http://localhost:8790/v13_dashboard.html
#   - Gracefully interruptable anytime via 'touch stop'
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="$SCRIPT_DIR/td_ludo/td_env/bin/python"
LOG_FILE="$SCRIPT_DIR/v137_alphazero.log"
PORT=8790

# Ensure clean state: remove old stop file if present
rm -f "$SCRIPT_DIR/stop"

# Check if port 8790 is in use
if lsof -i :$PORT >/dev/null 2>&1; then
    echo "⚠️ Port $PORT is already in use by another process:"
    lsof -i :$PORT
    echo "Please stop the existing process before launching V13.7."
    exit 1
fi

echo "================================================================================"
echo "🚀 LAUNCHING ALPHALUDO V13.7 (ALPHAZERO 2-PLAYER TABULA RASA)"
echo "================================================================================"
echo "   Model:       V13.7 Minimal ResNet (6 ResBlocks x 128ch, ~1.8M params)"
echo "   Search:      TwoPlayerExpectiMCTS (N=3,000 simulations/move)"
echo "   Rewards:     Strict zero-sum terminal only: z in {-1.0, +1.0}"
echo "   Buffer:      60,000 states in RAM (zero disk I/O wear)"
echo "   Dashboard:   http://localhost:$PORT/v13_dashboard.html"
echo "   Logs:        $LOG_FILE"
echo "   Interrupt:   touch stop"
echo "================================================================================"

PYTHONUNBUFFERED=1 nohup "$PYTHON" -u python/alphaludo/train_v137_alphazero.py \
    --mcts-sims 3000 \
    --states-per-iter 2000 \
    --train-steps-per-iter 250 \
    --batch-size 256 \
    --replay-capacity 60000 \
    --temp-cutoff 20 \
    --lr 0.001 \
    --eval-every 20 \
    --eval-games 80 \
    --checkpoint-dir checkpoints/v13_7 \
    --port $PORT \
    --device mps > "$LOG_FILE" 2>&1 &

PID=$!
disown $PID

echo " Daemon running with PID: $PID"
echo " Live Dashboard: http://localhost:$PORT/v13_dashboard.html"
echo ""
echo "To monitor live training:"
echo "   tail -f $LOG_FILE"
echo ""
echo "To cleanly stop training without data loss:"
echo "   touch stop"
echo "================================================================================"
