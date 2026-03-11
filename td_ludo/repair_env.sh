#!/bin/bash
# repair_env.sh: Automates the recreation of the td_env virtual environment

set -e

PROJECT_ROOT="/Users/sumit/Github/AlphaLudo/td_ludo"
cd "$PROJECT_ROOT"

echo "🛠️ Detected broken td_env symlinks. Starting repair..."

# 1. Clean up old environment
if [ -d "td_env" ]; then
    echo "🗑️ Removing old td_env..."
    rm -rf td_env
fi

# 2. Create fresh virtual environment
echo "🌱 Creating new virtual environment with $(which python3)..."
python3 -m venv td_env

# 3. Install requirements
echo "📦 Installing Python dependencies (numpy, torch, pybind11, etc.)..."
./td_env/bin/pip install --upgrade pip
./td_env/bin/pip install -r requirements.txt

# 4. Build and install the C++ Ludo engine
echo "⚙️ Building and installing C++ engine (td_ludo_cpp)..."
./td_env/bin/pip install -e .

echo "✅ Environment repair complete!"
echo "🚀 You can now run training with:"
echo "TD_LUDO_RUN_NAME=td_v3_small td_env/bin/python train.py --resume --model v4"
