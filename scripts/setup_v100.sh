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

log()  { echo "[$(date +%H:%M:%S)] $*"; }
warn() { echo "[$(date +%H:%M:%S)] WARNING: $*"; }
die()  { echo "[$(date +%H:%M:%S)] ERROR: $*"; exit 1; }

###############################################################################
# 1. NVIDIA driver
###############################################################################
log "=== Step 1/8: NVIDIA driver ==="

test "$(uname)" = "Linux" || die "Linux only"

if nvidia-smi >/dev/null 2>&1; then
    log "NVIDIA driver already working"
else
    log "NVIDIA driver not responding — installing..."
    sudo apt update -qq
    sudo apt install -y -qq linux-headers-$(uname -r)
    sudo apt install -y nvidia-driver-550-server
    log "Driver installed. Loading kernel module..."
    sudo modprobe nvidia || true
    if ! nvidia-smi >/dev/null 2>&1; then
        log "modprobe didn't help — rebooting in 5 seconds..."
        log "After reboot, run this script again."
        sleep 5
        sudo reboot
    fi
fi

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo "unknown")
log "GPU: $GPU_NAME"

nvidia-smi

###############################################################################
# 2. System dependencies
###############################################################################
log "=== Step 2/8: System dependencies ==="

log "Python: $(python3 --version 2>&1)"

log "Installing system packages..."
sudo apt update -qq
sudo apt install -y -qq python3-venv python3-dev git tmux

###############################################################################
# 3. Python venv
###############################################################################
log "=== Step 3/8: Python venv ==="

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"
log "Working directory: $REPO_DIR"

if [ ! -d "venv" ]; then
    log "Creating venv..."
    python3 -m venv venv
else
    log "venv already exists"
fi

. venv/bin/activate
log "Activated venv: $(which python)"

pip install --upgrade pip -q

###############################################################################
# 4. PyTorch 2.5.1+cu121 (V100-compatible)
###############################################################################
log "=== Step 4/8: PyTorch installation ==="

TORCH_VER=$(python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "none")

if echo "$TORCH_VER" | grep -q "^2\.5\.1"; then
    log "PyTorch $TORCH_VER already installed"
else
    log "Installing PyTorch 2.5.1+cu121..."
    pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121 -q
fi

log "Installing other dependencies..."
pip install numpy pytest tqdm pandas pyarrow pybind11 -q

python -c "import ninja" 2>/dev/null || pip install ninja -q

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

python -c "import torch; assert torch.cuda.is_available()" || die "PyTorch cannot access CUDA"

###############################################################################
# 5. Build C++ extension
###############################################################################
log "=== Step 5/8: Build C++ extension ==="

if [ -f "src/models/csrc/temporal_sampling.cpp" ]; then
    log "Compiling temporal_sampling_cpp..."
    python src/models/build_ext.py 2>&1 | tail -5
    log "C++ extension built"
else
    warn "temporal_sampling.cpp not found — skipping"
fi

###############################################################################
# 6. Build CUDA extension
###############################################################################
log "=== Step 6/8: Build CUDA extension ==="

if [ -f "src/models/csrc/temporal_sampling.cu" ]; then
    NVCC_PATH=$(command -v nvcc 2>/dev/null || true)

    if [ -z "$NVCC_PATH" ]; then
        for d in /usr/local/cuda /usr/local/cuda-12 /usr/local/cuda-12.1; do
            if [ -f "$d/bin/nvcc" ]; then
                export CUDA_HOME="$d"
                export PATH="$CUDA_HOME/bin:$PATH"
                NVCC_PATH="$d/bin/nvcc"
                log "Found nvcc at $NVCC_PATH"
                break
            fi
        done
    fi

    if [ -z "$NVCC_PATH" ]; then
        warn "nvcc not found — skipping CUDA build"
    else
        log "nvcc: $($NVCC_PATH --version 2>&1 | tail -1)"
        export TORCH_CUDA_ARCH_LIST="7.0"
        log "TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"
        log "Compiling temporal_sampling_cuda (may take 1-2 minutes)..."
        python src/models/build_ext.py --cuda 2>&1 | tail -5
        log "CUDA extension built"
    fi
else
    warn "temporal_sampling.cu not found — skipping"
fi

###############################################################################
# 7. Verification
###############################################################################
log "=== Step 7/8: Verification ==="

log "Running tests..."
PYTHONPATH=. python -m pytest tests/test_temporal_sampler.py -v --tb=short 2>&1 | tail -20

log "=== Step 8/8: Done ==="
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
echo "  # Run CUDA comparison experiment"
echo "  bash scripts/run_cuda_comparison.sh 2>&1 | tee /tmp/cuda_comparison.log"
echo ""
