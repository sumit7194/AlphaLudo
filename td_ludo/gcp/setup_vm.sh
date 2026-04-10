#!/usr/bin/env bash
# setup_vm.sh - Set up a fresh Ubuntu 22.04 GCP VM with T4 GPU for AlphaLudo training
# Run as: bash setup_vm.sh
set -euo pipefail

echo "============================================"
echo "  AlphaLudo GCP VM Setup (Ubuntu 22.04 + T4)"
echo "============================================"

# -----------------------------------------------
# 1. System packages
# -----------------------------------------------
echo "[1/6] Installing system packages..."
sudo apt-get update -qq
sudo apt-get install -y -qq \
    build-essential \
    software-properties-common \
    wget curl git \
    linux-headers-$(uname -r) \
    screen tmux

# -----------------------------------------------
# 2. NVIDIA drivers + CUDA 12.1 toolkit
# -----------------------------------------------
echo "[2/6] Installing NVIDIA drivers and CUDA 12.1 toolkit..."

# Install NVIDIA driver from Ubuntu repos
if ! nvidia-smi &>/dev/null; then
    sudo apt-get install -y -qq nvidia-driver-535
    echo "NVIDIA driver installed. A reboot may be needed if nvidia-smi fails."
fi

# Install CUDA 12.1 toolkit
if [ ! -d /usr/local/cuda-12.1 ]; then
    CUDA_DEB="cuda-repo-ubuntu2204-12-1-local_12.1.1-530.30.02-1_amd64.deb"
    wget -q "https://developer.download.nvidia.com/compute/cuda/12.1.1/local_installers/${CUDA_DEB}" -O "/tmp/${CUDA_DEB}"
    sudo dpkg -i "/tmp/${CUDA_DEB}"
    sudo cp /var/cuda-repo-ubuntu2204-12-1-local/cuda-*-keyring.gpg /usr/share/keyrings/
    sudo apt-get update -qq
    sudo apt-get install -y -qq cuda-toolkit-12-1
    rm -f "/tmp/${CUDA_DEB}"
fi

# Set up CUDA environment
export PATH=/usr/local/cuda-12.1/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:${LD_LIBRARY_PATH:-}

# Persist CUDA paths
if ! grep -q 'cuda-12.1' ~/.bashrc 2>/dev/null; then
    cat >> ~/.bashrc << 'CUDA_EOF'

# CUDA 12.1
export PATH=/usr/local/cuda-12.1/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:${LD_LIBRARY_PATH:-}
CUDA_EOF
fi

echo "  NVIDIA driver: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null || echo 'reboot needed')"
echo "  CUDA toolkit:  $(nvcc --version 2>/dev/null | grep 'release' || echo 'installed')"

# -----------------------------------------------
# 3. Python 3.12
# -----------------------------------------------
echo "[3/6] Installing Python 3.12..."
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt-get update -qq
sudo apt-get install -y -qq python3.12 python3.12-venv python3.12-dev

# -----------------------------------------------
# 4. Virtual environment
# -----------------------------------------------
echo "[4/6] Creating virtualenv and installing Python packages..."
VENV_DIR="$HOME/td_ludo/venv"
python3.12 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

pip install --upgrade pip setuptools wheel
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install numpy pybind11 websockets psutil requests

# -----------------------------------------------
# 5. Build td_ludo_cpp C++ extension
# -----------------------------------------------
echo "[5/6] Compiling td_ludo_cpp C++ extension..."
cd "$HOME/td_ludo"
pip install -e .

# Verify the extension loads
python -c "import td_ludo_cpp; print(f'  td_ludo_cpp loaded: {td_ludo_cpp}')"

# -----------------------------------------------
# 6. Verify GPU access from PyTorch
# -----------------------------------------------
echo "[6/6] Verifying PyTorch CUDA..."
python -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available!'
print(f'  PyTorch {torch.__version__}')
print(f'  CUDA device: {torch.cuda.get_device_name(0)}')
print(f'  CUDA memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
"

echo ""
echo "============================================"
echo "  Setup complete!"
echo ""
echo "  To start training:"
echo "    cd ~/td_ludo"
echo "    bash gcp/start_v6.sh   # TD-learning v6"
echo "    bash gcp/start_v9.sh   # Fast v9 training"
echo "============================================"
