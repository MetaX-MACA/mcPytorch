#!/bin/bash

export PATH=${MACA_PATH}/tools/cu-bridge/tools:${PATH}
export MACA_CLANG_PATH=${MACA_PATH}/mxgpu_llvm/bin
export LD_LIBRARY_PATH=$MACA_PATH/lib:${LD_LIBRARY_PATH}
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
