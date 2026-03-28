#!/usr/bin/env bash
# setup_a10.sh — Full environment setup for NVIDIA A10 (Ampere, CC 8.6)
#
# Run once after creating the machine:
#   bash src/models/cuda_exp_graphmixer_a10/setup_a10.sh
#
# What it does:
#   1. Installs NVIDIA driver 550 (proprietary, compatible with Ampere)
#   2. Installs CUDA toolkit 12.4
#   3. Creates Python venv with PyTorch 2.5.1+cu121
#   4. Clones repo and installs requirements
#   5. Compiles C++/CUDA extensions (TORCH_CUDA_ARCH_LIST="8.6")
#   6. Runs 30 tests to verify everything works
#
# A10 specs: Ampere architecture, compute capability 8.6, 24GB GDDR6

set -uo pipefail

echo "=== A10 Setup: $(date) ==="

# ── 1. NVIDIA driver ───────────────────────────────────────────────────────────
echo "[1/6] Checking NVIDIA driver..."
if nvidia-smi &>/dev/null; then
    echo "    Driver already working, skipping install."
else
    echo "    Driver not found, installing nvidia-driver-550-server..."
    sudo apt-get update -qq
    sudo apt-get install -y linux-headers-$(uname -r)
    sudo apt-get install -y nvidia-driver-550-server

    echo "    Rebooting to load driver — re-run script after reboot."
    sudo reboot
    exit 0
fi

# ── 2. Verify driver + CUDA ────────────────────────────────────────────────────
echo "[2/6] Verifying driver..."
nvidia-smi || { echo "ERROR: nvidia-smi failed."; exit 1; }
echo "    nvcc version:"
nvcc --version 2>/dev/null || echo "    (nvcc not found, will use PyTorch-bundled CUDA)"

# ── 3. Python environment ──────────────────────────────────────────────────────
echo "[3/6] Setting up Python venv..."
sudo apt-get install -y python3-venv python3-pip git tmux -qq

cd ~
if [ ! -d "payment-graph-forecasting" ]; then
    git clone https://github.com/kunikrubika05/payment-graph-forecasting.git
fi
cd payment-graph-forecasting
git pull

python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip -q

# ── 4. PyTorch 2.5.1 + CUDA 12.1 ──────────────────────────────────────────────
echo "[4/6] Installing PyTorch 2.5.1+cu121..."
pip install torch==2.5.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121 -q

echo "    Verifying PyTorch CUDA..."
python -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available!'
print(f'  PyTorch {torch.__version__}, CUDA {torch.version.cuda}')
print(f'  GPU: {torch.cuda.get_device_name(0)}')
print(f'  Compute capability: {torch.cuda.get_device_capability(0)}')
"

# ── 5. Project dependencies ────────────────────────────────────────────────────
echo "[5/6] Installing requirements..."
pip install -r requirements.txt -q

# ── 6. C++/CUDA extensions ────────────────────────────────────────────────────
echo "[6/6] Compiling C++/CUDA extensions (A10, CC 8.6)..."
export TORCH_CUDA_ARCH_LIST="8.6"
PYTHONPATH=. python src/models/build_ext.py --all

echo ""
echo "=== Running tests ==="
PYTHONPATH=. python -m pytest tests/test_temporal_sampler.py -v --tb=short
PYTHONPATH=. python -m pytest tests/test_models.py -v --tb=short -q

echo ""
echo "=== Setup complete! ==="
echo ""
echo "To run the benchmark:"
echo "  cd ~/payment-graph-forecasting"
echo "  tmux new -s cuda_exp"
echo "  source venv/bin/activate"
echo "  export YADISK_TOKEN='...'"
echo "  # Download stream graph first:"
echo "  PYTHONPATH=. python src/yadisk_utils.py download \\"
echo "      orbitaal_processed/stream_graph/2020-06-01__2020-08-31.parquet \\"
echo "      stream_graph/2020-06-01_2020-08-31.parquet"
echo "  # Run benchmark:"
echo "  bash src/models/cuda_exp_graphmixer_a10/run_experiment.sh"
echo ""
echo "Expected total time: ~18 min"
echo "Expected C++→CUDA speedup: ~3-4x"
