#!/bin/bash
cd $(dirname $0)
pip uninstall -y torch_test_cpp_extension_cuda
rm -rf torch_test_cpp_extension_cuda.egg-info/ && rm -rf build/ && rm -rf dist/

export MACA_PATH=/opt/maca/
export LD_LIBRARY_PATH=$MACA_PATH/lib:$MACA_PATH/mxgpu_llvm/lib:$LD_LIBRARY_PATH

export CUDA_PATH=/opt/maca/tools/cu-bridge/
export CUCC_PATH=/opt/maca/tools/cu-bridge/
export PATH=${CUCC_PATH}/tools:${CUCC_PATH}/bin:${PATH}

python setup.py install

python test_cpp_extension_cuda.py

ret=$?

if [[ ${ret} == 1 ]]; then
  echo "cpp_extension_cuda_demo setuptools build failed"
  exit 1
elif [[ ${ret} == 2 ]]; then
  echo "cpp_extension_cuda_demo jit compile failed"
  exit 1
elif [[ ${ret} == 3 ]]; then
  echo "cpp_extension_cuda_demo inline jit compile failed"
  exit 1
elif [[ ${ret} == 4 ]]; then
  echo "cpp_extension_cuda_demo cudnn jit compile failed"
else
  echo "cpp_extension_cuda_demo passed"
  exit 0
fi
