#!/bin/bash
# ==============================================================================
# AlphaLudo V15.3b — search distillation, post-audit (2026-10-02) pipeline.
#
#   1. Rust generator (format v2): fixed Expecti-MCTS teacher (63% vs Depth2,
#      was 15%) vs OPPONENT, labels per move, visit distributions, manifest.
#   2. PyTorch trainer: logits CE on the teacher's visit distribution, BCE
#      value, game-level validation, H2H-vs-champion eval, resumable.
#   3. Generator monitor for the dashboard: http://localhost:$PORT/v13_dashboard.html
#
# Everything is nohup + disown with logs on disk; stop with ./stop_v153b_pipeline.sh
# (or `touch stop`). Re-running this script RESUMES both stages.
#
# Config via env vars (defaults shown):
#   TEACHER=mcts OPPONENT=depth2 MOVERS=mcts SIMS=3000 GAMES=1000000
#   EXPLORE=0 DATA=data/sl_teacher_v2 CKPT=checkpoints/v15_3b PORT=8790 DEVICE=mps
# ==============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
export PATH="$HOME/.cargo/bin:$PATH"

TEACHER="${TEACHER:-mcts}"; OPPONENT="${OPPONENT:-depth2}"; MOVERS="${MOVERS:-mcts}"
SIMS="${SIMS:-3000}"; GAMES="${GAMES:-1000000}"; EXPLORE="${EXPLORE:-0}"
DATA="${DATA:-data/sl_teacher_v2}"; CKPT="${CKPT:-checkpoints/v15_3b}"
PORT="${PORT:-8790}"; DEVICE="${DEVICE:-mps}"
PY=./td_ludo/td_env/bin/python

rm -f stop "$CKPT/stop"
mkdir -p "$DATA" "$CKPT"

for pat in "generate_sl_teacher.*$DATA" "train_v153_sl.py.*$CKPT"; do
    if pgrep -f "$pat" > /dev/null; then
        echo "❌ already running: $pat (pid $(pgrep -f "$pat" | tr '\n' ' ')). Stop it first."; exit 1
    fi
done

echo "▶ Building Rust generator..."
cargo build --release --bin generate_sl_teacher 2>&1 | tail -1

echo "▶ Generator: $TEACHER vs $OPPONENT, $SIMS sims, $GAMES games -> $DATA"
nohup ./target/release/generate_sl_teacher \
    --teacher "$TEACHER" --opponent "$OPPONENT" --mcts-sims "$SIMS" \
    --target-games "$GAMES" --shard-size 5000 --explore-moves "$EXPLORE" \
    --out-dir "$DATA" --stop-file "$CKPT/stop" >> sl_generator_v2.log 2>&1 &
GEN_PID=$!; disown $GEN_PID

nohup $PY td_ludo/scripts/generator_monitor.py --shard-dir "$DATA" --log sl_generator_v2.log \
    --target-games "$GAMES" > /dev/null 2>&1 &
MON_PID=$!; disown $MON_PID

RESUME=""; [ -f "$CKPT/model_latest.pt" ] && RESUME="--resume"
echo "▶ Trainer: movers=$MOVERS -> $CKPT ${RESUME:+(resuming)}"
nohup $PY python/alphaludo/train_v153_sl.py $RESUME \
    --shard-dir "$DATA" --checkpoint-dir "$CKPT" --movers "$MOVERS" \
    --batch-size 256 --lr 3e-4 --port "$PORT" --device "$DEVICE" >> "$CKPT/train.log" 2>&1 &
TRAIN_PID=$!; disown $TRAIN_PID

sleep 5
for name_pid in "generator:$GEN_PID" "monitor:$MON_PID" "trainer:$TRAIN_PID"; do
    n="${name_pid%%:*}"; p="${name_pid##*:}"
    if kill -0 "$p" 2>/dev/null; then echo "  [OK]   $n pid $p"; else echo "  [DEAD] $n pid $p — check logs"; fi
done
echo "$GEN_PID $MON_PID $TRAIN_PID" > "$CKPT/pids"
echo "Logs: tail -f sl_generator_v2.log $CKPT/train.log | Dashboard: http://localhost:$PORT/v13_dashboard.html"
