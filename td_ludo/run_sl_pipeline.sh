#!/bin/bash
# =============================================================================
# TD-Ludo V7 — SL Pipeline Orchestration Script
#
# This script runs the full Supervised Learning pipeline:
#   1. Generate bot-vs-bot game data in V7 1D format
#   2. Train the V7 Transformer on this data (starting from current PPO weights)
#   3. Copy the SL-trained weights back to the V7 checkpoint directory
#   4. Resume PPO RL training with the improved baseline
#
# Usage:
#   cd td_ludo
#   bash run_sl_pipeline.sh [--games 50000] [--epochs 3] [--skip-datagen]
# =============================================================================

set -e  # Exit on error

# Defaults
GAMES=50000
EPOCHS=3
SKIP_DATAGEN=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --games)
            GAMES="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --skip-datagen)
            SKIP_DATAGEN=true
            shift
            ;;
        --help)
            echo "Usage: bash run_sl_pipeline.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --games N        Number of bot games to generate (default: 50000)"
            echo "  --epochs N       Number of SL training epochs (default: 3)"
            echo "  --skip-datagen   Skip data generation (use existing data)"
            echo "  --help           Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Paths
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CKPT_DIR="$SCRIPT_DIR/checkpoints/ac_v7_transformer"
PPO_LATEST="$CKPT_DIR/model_latest.pt"
SL_MODEL="$CKPT_DIR/model_sl.pt"
PPO_BACKUP="$CKPT_DIR/model_latest_pre_sl.pt"

# Detect Python — prefer td_env virtualenv
if [ -f "$SCRIPT_DIR/td_env/bin/python" ]; then
    PYTHON="$SCRIPT_DIR/td_env/bin/python"
elif [ -f "$SCRIPT_DIR/../venv/bin/python" ]; then
    PYTHON="$SCRIPT_DIR/../venv/bin/python"
else
    PYTHON="python3"
fi

echo "============================================================"
echo "  TD-Ludo V7 — Supervised Learning Pipeline"
echo "============================================================"
echo "  Python: $PYTHON"
echo "  Games:  $GAMES"
echo "  Epochs: $EPOCHS"
echo "  PPO Checkpoint: $PPO_LATEST"
echo "============================================================"
echo ""

# =============================================================================
# Step 1: Generate SL Data
# =============================================================================
if [ "$SKIP_DATAGEN" = true ]; then
    echo "[Step 1/4] SKIPPED — Using existing data in checkpoints/sl_data_v7/"
else
    echo "[Step 1/4] Generating bot-vs-bot game data ($GAMES games)..."
    echo ""
    cd "$SCRIPT_DIR"
    $PYTHON generate_sl_data_v7.py --games "$GAMES"
    echo ""
    echo "[Step 1/4] ✓ Data generation complete."
fi
echo ""

# =============================================================================
# Step 2: Train SL Model (starting from current PPO weights)
# =============================================================================
echo "[Step 2/4] Training V7 Transformer with Supervised Learning..."
echo ""

INIT_WEIGHTS_ARG=""
if [ -f "$PPO_LATEST" ]; then
    echo "  Using current PPO weights as starting point: $PPO_LATEST"
    INIT_WEIGHTS_ARG="--init-weights $PPO_LATEST"
else
    echo "  No PPO checkpoint found. Training from random weights."
fi

cd "$SCRIPT_DIR"
$PYTHON train_sl_v7.py $INIT_WEIGHTS_ARG --epochs "$EPOCHS" --fresh
echo ""
echo "[Step 2/4] ✓ SL training complete."
echo ""

# =============================================================================
# Step 3: Copy SL weights back to PPO checkpoint
# =============================================================================
echo "[Step 3/4] Installing SL weights into PPO checkpoint..."

if [ ! -f "$SL_MODEL" ]; then
    echo "  ERROR: SL model not found at $SL_MODEL"
    echo "  SL training may have failed. Aborting."
    exit 1
fi

# Backup current PPO weights
if [ -f "$PPO_LATEST" ]; then
    cp "$PPO_LATEST" "$PPO_BACKUP"
    echo "  Backed up PPO weights to: $PPO_BACKUP"
fi

echo "  SL model ready at: $SL_MODEL"
echo "  (train_v7.py --fresh will auto-load model_sl.pt)"
echo ""
echo "[Step 3/4] ✓ Weights ready."
echo ""

# =============================================================================
# Step 4: Resume PPO RL Training
# =============================================================================
echo "[Step 4/4] Starting PPO RL training with SL-bootstrapped weights..."
echo ""
echo "  The training will start fresh (resetting game counter and Elo)"
echo "  but using the SL-trained model weights as the starting point."
echo ""

cd "$SCRIPT_DIR"
$PYTHON train_v7.py --fresh

echo ""
echo "============================================================"
echo "  Pipeline Complete!"
echo "============================================================"
