#!/usr/bin/env bash
# setup_a10.sh — Full environment setup for NVIDIA A10 (Ampere, CC 8.6)
#
# Run once after creating the machine (teslaa10-1.4.32.160 on immers.cloud):
#   bash src/models/cuda_exp_graphmixer_a10/setup_a10.sh
#
# What it does:
#   1. Checks NVIDIA driver (A10 on immers.cloud ships with 570, skips install)
#   2. Installs system deps: python3.12-dev, git, tmux, optuna
#   3. Creates Python venv with PyTorch 2.5.1+cu121
#   4. Clones repo and installs requirements
#   5. Compiles C++/CUDA extensions (TORCH_CUDA_ARCH_LIST="8.6")
#   6. Runs 30 tests to verify everything works
#
# A10 specs: Ampere CC 8.6, 24GB VRAM, 32GB RAM, 160GB SSD

set -uo pipefail

echo "=== A10 Setup: $(date) ==="

# ── 1. NVIDIA driver ───────────────────────────────────────────────────────────
echo "[1/6] Checking NVIDIA driver..."
if nvidia-smi &>/dev/null; then
    echo "    Driver already working ($(nvidia-smi --query-gpu=driver_version --format=csv,noheader)), skipping install."
else
    echo "    Driver not found, installing nvidia-driver-550-server..."
    sudo apt-get update -qq
    sudo apt-get install -y linux-headers-$(uname -r)
    sudo apt-get install -y nvidia-driver-550-server
    echo "    Rebooting to load driver — re-run script after reboot."
    sudo reboot
    exit 0
fi

# ── 2. System dependencies ─────────────────────────────────────────────────────
echo "[2/6] Installing system dependencies..."
sudo apt-get update -qq
sudo apt-get install -y python3-venv python3-pip python3-dev git tmux -qq

# ── 3. Python environment ──────────────────────────────────────────────────────
echo "[3/6] Setting up Python venv..."
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
pip install torch==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121 -q
pip install optuna -q

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
echo "Next steps:"
echo "  tmux new -s exp"
echo "  source venv/bin/activate"
echo "  export YADISK_TOKEN='...'"
echo ""
echo "  # Download stream graph (use variable to avoid line-break issues):"
echo "  P=\"orbitaal_processed/stream_graph/2020-06-01__2020-08-31.parquet\""
echo "  PYTHONPATH=. python scripts/slice_stream_graph.py --yadisk-path \"\$P\" --start 2020-07-01 --end 2020-07-07 --output stream_graph/week.parquet"
echo ""
echo "  # HPO (~3-4h):"
echo "  PYTHONPATH=. python src/models/GraphMixer/graphmixer_hpo.py --parquet-path stream_graph/week.parquet --n-trials 20 --hpo-epochs 10 --output /tmp/graphmixer_hpo 2>&1 | tee /tmp/hpo.log"
echo ""
echo "  # Final training with best params (see /tmp/graphmixer_hpo/best_train_command.sh):"
echo "  bash /tmp/graphmixer_hpo/best_train_command.sh"
