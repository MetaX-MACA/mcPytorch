#!/bin/bash
export MACA_PATH=${MACA_PATH:-/opt/maca/}
export PATH=${MACA_PATH}/tools/cu-bridge/tools:${PATH}
export LD_LIBRARY_PATH=$MACA_PATH/lib:$MACA_PATH/ompi/lib:${LD_LIBRARY_PATH}
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
