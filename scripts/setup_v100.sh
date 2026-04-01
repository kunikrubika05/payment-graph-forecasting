#!/bin/bash
###############################################################################
# setup_v100.sh — Full setup for V100 GPU dev machine
#
# Handles EVERYTHING from bare Ubuntu 24.04 + CUDA image to ready-to-run state:
#   - NVIDIA proprietary driver (open driver is incompatible with V100/Volta)
#   - System packages
#   - Python venv + PyTorch 2.5.1+cu121 + PyG + all dependencies
#   - C++ and CUDA extension builds
#   - Verification tests
#
# Tested on: immers.cloud Ubuntu 24.04 + CUDA 12.8, Tesla V100-PCIE-32GB, 32GB RAM
#
# Usage:
#   git clone https://github.com/kunikrubika05/payment-graph-forecasting.git
#   cd payment-graph-forecasting
#   bash scripts/setup_v100.sh 2>&1 | tee /tmp/setup_v100.log
#
# If the script reboots the machine (driver install), just re-run after reboot.
###############################################################################

log()  { echo "[$(date +%H:%M:%S)] $*"; }
warn() { echo "[$(date +%H:%M:%S)] WARNING: $*"; }
die()  { echo "[$(date +%H:%M:%S)] ERROR: $*"; exit 1; }

test "$(uname)" = "Linux" || die "Linux only"

###############################################################################
log "=== Step 1/7: NVIDIA driver ==="
###############################################################################

if nvidia-smi >/dev/null 2>&1; then
    log "NVIDIA driver OK"
    nvidia-smi
else
    log "NVIDIA driver not working — installing proprietary driver..."
    log "(V100 = Volta, needs closed driver, NOT nvidia-*-open)"

    sudo apt update -qq
    sudo apt install -y linux-headers-$(uname -r)

    sudo apt remove -y nvidia-driver-570-open nvidia-dkms-570-open \
        nvidia-kernel-source-570-open 2>/dev/null || true

    if ! sudo apt install -y nvidia-driver-570-server; then
        log "nvidia-driver-570-server failed, trying ubuntu-drivers..."
        sudo apt install -y ubuntu-drivers-common
        sudo ubuntu-drivers autoinstall
    fi

    sudo modprobe nvidia || true

    if nvidia-smi >/dev/null 2>&1; then
        log "Driver loaded without reboot"
        nvidia-smi
    else
        log "Reboot required. Re-run this script after reboot."
        sleep 3
        sudo reboot
    fi
fi

###############################################################################
log "=== Step 2/7: System packages ==="
###############################################################################

sudo apt update -qq
sudo apt install -y -qq python3-venv python3-dev git tmux
log "System packages OK"

###############################################################################
log "=== Step 3/7: Python venv ==="
###############################################################################

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"
log "Repo: $REPO_DIR"

if [ ! -d "venv" ]; then
    python3 -m venv venv
    log "Created venv"
else
    log "venv exists"
fi

. venv/bin/activate
pip install --upgrade pip -q
log "Python: $(python --version), pip: $(pip --version | awk '{print $2}')"

###############################################################################
log "=== Step 4/7: PyTorch + dependencies ==="
###############################################################################

TORCH_VER=$(python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "none")

if echo "$TORCH_VER" | grep -q "^2\.5\.1"; then
    log "PyTorch $TORCH_VER already installed"
else
    log "Installing PyTorch 2.5.1+cu121..."
    pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121 -q
fi

log "Installing Python dependencies..."
pip install numpy scipy pytest tqdm pandas pyarrow pybind11 requests matplotlib -q
python -c "import ninja" 2>/dev/null || pip install ninja -q
log "Installing project package dependencies..."
pip install -e ".[dev]" -q

log "Installing PyG (torch-geometric)..."
pip install torch-geometric torch-scatter torch-sparse \
    -q -f https://data.pyg.org/whl/torch-2.5.1+cu121.html

log "Verifying PyTorch + CUDA..."
python -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  Device: {torch.cuda.get_device_name(0)}')
    print(f'  CUDA runtime: {torch.version.cuda}')
    x = torch.randn(2, 2, device='cuda')
    print(f'  GPU tensor: OK')
else:
    raise RuntimeError('CUDA not available!')
"
python -c "import torch; assert torch.cuda.is_available()" || die "PyTorch CUDA check failed"
log "PyTorch + CUDA OK"
python -c "
import joblib
import yaml
import payment_graph_forecasting
print('  Base package deps: OK')
"
log "Project package dependencies OK"

###############################################################################
log "=== Step 5/7: Build C++ extension ==="
###############################################################################

if [ -f "src/models/csrc/temporal_sampling.cpp" ]; then
    log "Compiling C++ extension..."
    python src/models/build_ext.py 2>&1 | tail -3
    log "C++ extension OK"
else
    warn "temporal_sampling.cpp not found"
fi

###############################################################################
log "=== Step 6/7: Build CUDA extension ==="
###############################################################################

if [ -f "src/models/csrc/temporal_sampling.cu" ]; then
    NVCC_PATH=$(command -v nvcc 2>/dev/null || true)

    if [ -z "$NVCC_PATH" ]; then
        for d in /usr/local/cuda /usr/local/cuda-12 /usr/local/cuda-12.1; do
            if [ -f "$d/bin/nvcc" ]; then
                export CUDA_HOME="$d"
                export PATH="$CUDA_HOME/bin:$PATH"
                NVCC_PATH="$d/bin/nvcc"
                break
            fi
        done
    fi

    if [ -z "$NVCC_PATH" ]; then
        warn "nvcc not found — cannot build CUDA extension"
    else
        export TORCH_CUDA_ARCH_LIST="7.0"
        log "nvcc found, TORCH_CUDA_ARCH_LIST=7.0"
        log "Compiling CUDA extension (1-2 min)..."
        python src/models/build_ext.py --cuda 2>&1 | tail -3
        log "CUDA extension OK"
    fi
else
    warn "temporal_sampling.cu not found"
fi

###############################################################################
log "=== Step 7/7: Tests ==="
###############################################################################

PYTHONPATH=. python -m pytest tests/test_temporal_sampler.py -v --tb=short 2>&1 | tail -25

log "=============================================="
log "  SETUP COMPLETE"
log "=============================================="
echo ""
echo "Next:"
echo "  source venv/bin/activate"
echo "  export YADISK_TOKEN=\"...\""
echo "  bash scripts/run_cuda_comparison.sh 2>&1 | tee /tmp/cuda_comparison.log"
echo ""
