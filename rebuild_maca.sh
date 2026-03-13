#!/bin/bash
# Rebuild script for MACA PyTorch with correct environment

export MACA_PATH=/opt/maca
export CUDA_PATH=/opt/maca/tools/cu-bridge
export CUCC_PATH=/opt/maca/tools/cu-bridge
export PATH=$CUDA_PATH/bin:$PATH
export BUILD_TEST=0
export MAX_JOBS=16

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
cd "$SCRIPT_DIR"

# Full clean rebuild
rm -rf build

# Configure with explicit MACA settings
python3 setup.py develop 2>&1 | tee rebuild_maca_clean.log
