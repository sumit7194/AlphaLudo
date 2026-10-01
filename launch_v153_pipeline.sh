#!/bin/bash
# ==============================================================================
# AlphaLudo V15.3 — 1,000,000 Game Supervised Learning & MCTS Distillation Launch
#
# Detached, background-resilient training pipeline:
# 1. Rust Generator: TwoPlayerMCTSBot (N=3,000 sims) vs Depth2ExpectimaxBot (1M games)
# 2. PyTorch Trainer: Tabula Rasa GraphTransformer (history_len=1, ~588K params)
# 3. Live Web Dashboard: http://localhost:8790/v13_dashboard.html
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "================================================================================"
echo "🚀 LAUNCHING ALPHALUDO V15.3 (1,000,000 GAME SEARCH DISTILLATION)"
echo "================================================================================"

# 1. Clean stop triggers
rm -f stop checkpoints/v15_3/stop

# 2. Verify directories
mkdir -p data/sl_teacher_v153 checkpoints/v15_3

# 3. Verify Rust binary
if [ ! -f "target/release/generate_sl_teacher" ]; then
    echo "⚠️ target/release/generate_sl_teacher not found! Building..."
    export PATH="$HOME/.cargo/bin:$PATH"
    cargo build --release --bin generate_sl_teacher
fi

# 4. Launch Rust Teacher Generator (nohup + disown)
echo "▶ Starting Rust MCTS (N=3,000) Teacher Generator..."
nohup ./target/release/generate_sl_teacher \
    --target-games 1000000 \
    --shard-size 5000 \
    --mcts-sims 3000 \
    --out-dir data/sl_teacher_v153 > sl_generator.log 2>&1 &
GEN_PID=$!
disown $GEN_PID
echo "  [OK] Rust Generator started with PID $GEN_PID (logging to sl_generator.log)"

# 5. Launch PyTorch SL Trainer (nohup + disown)
echo "▶ Starting PyTorch V15.3 Graph Transformer Trainer..."
nohup ./td_ludo/td_env/bin/python python/alphaludo/train_v153_sl.py \
    --shard-dir data/sl_teacher_v153 \
    --checkpoint-dir checkpoints/v15_3 \
    --batch-size 256 \
    --lr 3e-4 \
    --port 8790 \
    --device mps > v153_train.log 2>&1 &
TRAIN_PID=$!
disown $TRAIN_PID
echo "  [OK] PyTorch Trainer started with PID $TRAIN_PID (logging to v153_train.log)"

echo ""
echo "================================================================================"
echo "✅ PIPELINE RUNNING IN DETACHED BACKGROUND"
echo "   Rust Generator PID: $GEN_PID"
echo "   PyTorch Trainer PID: $TRAIN_PID"
echo "   Live Dashboard:     http://localhost:8790/v13_dashboard.html"
echo "   API Stats:          http://localhost:8790/api/stats"
echo "   Generator Log:      tail -f sl_generator.log"
echo "   Training Log:       tail -f v153_train.log"
echo "   Stop Pipeline:      touch stop"
echo "================================================================================"
