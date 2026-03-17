#!/bin/bash
# =============================================================================
# V9 Full Training Pipeline
#
# Runs all steps sequentially:
#   1. Generate SL data (bots, 300K games)
#   2. SL training (behavioral cloning, 3 epochs)
#   3. PPO training (from SL checkpoint)
#
# Usage:
#   ./run_v9.sh              # Run full pipeline
#   ./run_v9.sh --skip-data  # Skip data gen (already have data)
#   ./run_v9.sh --skip-sl    # Skip data gen + SL (go straight to PPO)
#
# Each step saves checkpoints — if interrupted (Ctrl+C), re-run to resume.
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="./td_env/bin/python3"

# Check python exists
if [ ! -f "$PYTHON" ]; then
    echo "ERROR: Virtual environment not found at $PYTHON"
    echo "Create it first: python3 -m venv td_env && ./td_env/bin/pip install -e ."
    exit 1
fi

# Parse args
SKIP_DATA=false
SKIP_SL=false
SL_GAMES=300000
SL_EPOCHS=3
PPO_HOURS=0
PPO_GAMES=0

for arg in "$@"; do
    case $arg in
        --skip-data) SKIP_DATA=true ;;
        --skip-sl) SKIP_DATA=true; SKIP_SL=true ;;
        --sl-games=*) SL_GAMES="${arg#*=}" ;;
        --sl-epochs=*) SL_EPOCHS="${arg#*=}" ;;
        --ppo-hours=*) PPO_HOURS="${arg#*=}" ;;
        --ppo-games=*) PPO_GAMES="${arg#*=}" ;;
        --help)
            echo "Usage: ./run_v9.sh [options]"
            echo ""
            echo "Options:"
            echo "  --skip-data         Skip SL data generation"
            echo "  --skip-sl           Skip data gen + SL training (PPO only)"
            echo "  --sl-games=N        Number of games for SL data (default: 300000)"
            echo "  --sl-epochs=N       SL training epochs (default: 3)"
            echo "  --ppo-hours=N       PPO time limit in hours (default: unlimited)"
            echo "  --ppo-games=N       PPO game limit (default: unlimited)"
            exit 0
            ;;
    esac
done

echo "============================================================"
echo "  V9 Full Training Pipeline"
echo "============================================================"
echo ""

# =============================================================================
# Step 1: Generate SL Data
# =============================================================================
if [ "$SKIP_DATA" = false ]; then
    echo "=== Step 1/3: Generating SL Data ($SL_GAMES games) ==="
    echo ""
    $PYTHON generate_sl_data_v9.py --games "$SL_GAMES"
    echo ""
    echo "=== Step 1 Complete ==="
    echo ""
else
    echo "=== Step 1/3: Skipped (--skip-data) ==="
    echo ""
fi

# =============================================================================
# Step 2: SL Training
# =============================================================================
if [ "$SKIP_SL" = false ]; then
    echo "=== Step 2/3: SL Training ($SL_EPOCHS epochs) ==="
    echo ""
    $PYTHON train_sl_v9.py --epochs "$SL_EPOCHS" --resume
    echo ""
    echo "=== Step 2 Complete ==="
    echo ""
else
    echo "=== Step 2/3: Skipped (--skip-sl) ==="
    echo ""
fi

# =============================================================================
# Step 3: PPO Training
# =============================================================================
echo "=== Step 3/3: PPO Training ==="
echo ""

PPO_ARGS="--fresh"
if [ "$PPO_HOURS" != "0" ]; then
    PPO_ARGS="$PPO_ARGS --hours $PPO_HOURS"
fi
if [ "$PPO_GAMES" != "0" ]; then
    PPO_ARGS="$PPO_ARGS --games $PPO_GAMES"
fi

# Use fast multi-process trainer (N CPU actors + 1 MPS learner)
$PYTHON train_v9_fast.py $PPO_ARGS

echo ""
echo "============================================================"
echo "  V9 Pipeline Complete!"
echo "============================================================"
