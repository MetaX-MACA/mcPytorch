#!/bin/bash
set -x

export MACA_PATH=${MACA_PATH:-"/opt/maca/"}
export CUCC_PATH=${MACA_PATH}/tools/cu-bridge/
export PATH=${CUCC_PATH}/tools/:${PATH}

cd $(dirname $0)

rm -rf build && mkdir build && cd build

cmake_maca -DCMAKE_PREFIX_PATH=`python -c 'import torch;print(torch.utils.cmake_prefix_path)'` ..
make_maca

./libtorch_demo

ret=$?

if [[ ${ret} != 0 ]]; then
  echo "Libtorch_demo failed"
else
  echo "Libtorch_demo passed"
fi

