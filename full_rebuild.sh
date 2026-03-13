#!/bin/bash
# Full rebuild script with proper environment setup

# Set environment variables
export MACA_PATH=/opt/maca
export CUDA_PATH=/opt/maca/tools/cu-bridge
export CUCC_PATH=/opt/maca/tools/cu-bridge
export PATH=$CUDA_PATH/bin:$PATH
export BUILD_TEST=0
export MAX_JOBS=16

# Clean and rebuild
cd /root/mcPytorch-2.4
rm -rf build

# Run setup.py with all environment variables
python3 setup.py develop 2&1 | tee full_rebuild.log

echo "Build completed with exit code: $?"
