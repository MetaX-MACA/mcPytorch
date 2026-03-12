#!/bin/bash
# Build torch_cuda with proper environment

export MACA_PATH=/opt/maca
export CUDA_PATH=/opt/maca/tools/cu-bridge
export CUCC_PATH=/opt/maca/tools/cu-bridge
export PATH=$CUDA_PATH/bin:$PATH
export BUILD_TEST=0
export MAX_JOBS=16

cd /root/mcPytorch-2.4/build

# Build only torch_cuda target
make -j16 torch_cuda 2&>1 | tee -a ../build_torch_cuda.log
