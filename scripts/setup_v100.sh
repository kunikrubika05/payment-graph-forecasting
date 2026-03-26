#!/bin/bash
###############################################################################
# setup_v100.sh — Full setup for V100 GPU dev machine (immers.cloud or similar)
#
# Tested on: Ubuntu 24.04 + CUDA 12.8 image, Tesla V100-PCIE-32GB
# PyTorch:   2.5.1+cu121 (last version supporting V100 compute capability 7.0)
#
# Usage:
#   ssh -i ~/Downloads/<key>.pem ubuntu@<ip>
#   git clone https://github.com/kunikrubika05/payment-graph-forecasting.git
#   cd payment-graph-forecasting
#   bash scripts/setup_v100.sh 2>&1 | tee /tmp/setup_v100.log
#
# After setup:
#   source venv/bin/activate
#   PYTHONPATH=. python scripts/bench_sampling.py --cuda 2>&1 | tee /tmp/bench.log
###############################################################################
set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log()  { echo -e "${GREEN}[$(date +%H:%M:%S)] $*${NC}"; }
warn() { echo -e "${YELLOW}[$(date +%H:%M:%S)] WARNING: $*${NC}"; }
err()  { echo -e "${RED}[$(date +%H:%M:%S)] ERROR: $*${NC}"; exit 1; }

###############################################################################
# 1. Pre-flight checks
###############################################################################
log "=== Step 1/7: Pre-flight checks ==="

if [[ "$(uname)" != "Linux" ]]; then
    err "This script is for Linux only (got: $(uname))"
fi

if ! command -v nvidia-smi &>/dev/null; then
    err "nvidia-smi not found. Is NVIDIA driver installed?"
fi

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
CUDA_VERSION=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9.]+' || echo "unknown")

log "GPU: $GPU_NAME"
log "Driver: $DRIVER_VERSION"
log "CUDA (driver): $CUDA_VERSION"

COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 2>/dev/null || echo "unknown")
log "Compute capability: $COMPUTE_CAP"

if [[ "$COMPUTE_CAP" == "7.0" ]] || echo "$GPU_NAME" | grep -qi "v100"; then
    log "V100 detected — will use PyTorch 2.5.1+cu121"
else
    warn "Not a V100 ($GPU_NAME, cc=$COMPUTE_CAP). Script may still work, but is optimized for V100."
fi

###############################################################################
# 2. System dependencies
###############################################################################
log "=== Step 2/7: System dependencies ==="

PYTHON_VERSION=$(python3 --version 2>&1 | grep -oP '[0-9]+\.[0-9]+')
log "Python: $PYTHON_VERSION"

PACKAGES_TO_INSTALL=""

if ! dpkg -s python3-venv &>/dev/null 2>&1; then
    PACKAGES_TO_INSTALL="$PACKAGES_TO_INSTALL python3-venv"
fi

PYTHON_DEV_PKG="python3-dev"
if dpkg -l 2>/dev/null | grep -q "python3\\.${PYTHON_VERSION#3.}-dev"; then
    log "python3-dev already installed"
else
    PYTHON_DEV_PKG="python${PYTHON_VERSION}-dev"
    if ! dpkg -s "$PYTHON_DEV_PKG" &>/dev/null 2>&1; then
        PACKAGES_TO_INSTALL="$PACKAGES_TO_INSTALL $PYTHON_DEV_PKG"
    fi
    if ! dpkg -s python3-dev &>/dev/null 2>&1; then
        PACKAGES_TO_INSTALL="$PACKAGES_TO_INSTALL python3-dev"
    fi
fi

if ! command -v git &>/dev/null; then
    PACKAGES_TO_INSTALL="$PACKAGES_TO_INSTALL git"
fi

if ! command -v tmux &>/dev/null; then
    PACKAGES_TO_INSTALL="$PACKAGES_TO_INSTALL tmux"
fi

if [[ -n "$PACKAGES_TO_INSTALL" ]]; then
    log "Installing: $PACKAGES_TO_INSTALL"
    sudo apt update -qq
    sudo apt install -y -qq $PACKAGES_TO_INSTALL
else
    log "All system packages already installed"
fi

###############################################################################
# 3. Python venv
###############################################################################
log "=== Step 3/7: Python venv ==="

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"
log "Working directory: $REPO_DIR"

if [[ ! -d "venv" ]]; then
    log "Creating venv..."
    python3 -m venv venv
else
    log "venv already exists"
fi

source venv/bin/activate
log "Activated venv: $(which python)"

pip install --upgrade pip -q

###############################################################################
# 4. PyTorch 2.5.1+cu121 (V100-compatible)
###############################################################################
log "=== Step 4/7: PyTorch installation ==="

TORCH_INSTALLED=$(python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "none")

if [[ "$TORCH_INSTALLED" == "2.5.1"* ]]; then
    log "PyTorch $TORCH_INSTALLED already installed"
else
    if [[ "$TORCH_INSTALLED" != "none" ]]; then
        warn "Existing PyTorch $TORCH_INSTALLED — replacing with 2.5.1+cu121"
    fi
    log "Installing PyTorch 2.5.1+cu121..."
    pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121 -q
fi

log "Installing other dependencies..."
pip install numpy pytest tqdm pandas pyarrow -q

if ! python -c "import ninja" 2>/dev/null; then
    log "Installing ninja (for fast C++/CUDA compilation)..."
    pip install ninja -q
fi

log "Verifying PyTorch + CUDA..."
python -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  CUDA device: {torch.cuda.get_device_name(0)}')
    print(f'  CUDA version (runtime): {torch.version.cuda}')
    x = torch.randn(2, 2, device='cuda')
    print(f'  GPU tensor test: OK ({x.device})')
"

if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    err "PyTorch cannot access CUDA. Check driver/PyTorch compatibility."
fi

###############################################################################
# 5. Build C++ extension
###############################################################################
log "=== Step 5/7: Build C++ extension ==="

if [[ -f "src/models/csrc/temporal_sampling.cpp" ]]; then
    log "Compiling temporal_sampling_cpp..."
    python src/models/build_ext.py 2>&1 | tail -3
    log "C++ extension built"
else
    warn "temporal_sampling.cpp not found — skipping C++ build"
fi

###############################################################################
# 6. Build CUDA extension
###############################################################################
log "=== Step 6/7: Build CUDA extension ==="

if [[ -f "src/models/csrc/temporal_sampling.cu" ]]; then
    NVCC_PATH=$(command -v nvcc 2>/dev/null || echo "")
    if [[ -z "$NVCC_PATH" ]]; then
        CUDA_HOMES=("/usr/local/cuda" "/usr/local/cuda-12" "/usr/local/cuda-12.1")
        for ch in "${CUDA_HOMES[@]}"; do
            if [[ -f "$ch/bin/nvcc" ]]; then
                export CUDA_HOME="$ch"
                export PATH="$CUDA_HOME/bin:$PATH"
                NVCC_PATH="$CUDA_HOME/bin/nvcc"
                log "Found nvcc at $NVCC_PATH"
                break
            fi
        done
    fi

    if [[ -z "$NVCC_PATH" ]]; then
        warn "nvcc not found — skipping CUDA build"
        warn "Install CUDA toolkit or set CUDA_HOME"
    else
        log "nvcc: $($NVCC_PATH --version | tail -1)"

        export TORCH_CUDA_ARCH_LIST="7.0"
        log "TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"

        log "Compiling temporal_sampling_cuda (may take 1-2 minutes)..."
        python src/models/build_ext.py --cuda 2>&1 | tail -3
        log "CUDA extension built"
    fi
else
    warn "temporal_sampling.cu not found — skipping CUDA build"
fi

###############################################################################
# 7. Verification
###############################################################################
log "=== Step 7/7: Verification ==="

log "Running tests..."
PYTHONPATH=. python -m pytest tests/test_temporal_sampler.py -v --tb=short 2>&1 | tail -20

echo ""
log "=============================================="
log "  SETUP COMPLETE"
log "=============================================="
echo ""
echo "Next steps:"
echo "  source venv/bin/activate"
echo ""
echo "  # Run benchmark"
echo "  PYTHONPATH=. python scripts/bench_sampling.py --cuda 2>&1 | tee /tmp/bench.log"
echo ""
echo "  # Run all tests"
echo "  PYTHONPATH=. python -m pytest tests/test_temporal_sampler.py -v"
echo ""
echo "  # Run DyGFormer training (when ready)"
echo "  YADISK_TOKEN=\"...\" PYTHONPATH=. python src/models/launcher.py \\"
echo "      --period mature_2020q2 --output /tmp/results 2>&1 | tee /tmp/train.log"
echo ""
