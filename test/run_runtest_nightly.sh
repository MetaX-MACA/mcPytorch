#!/bin/bash

PYTORCH_ROOT="$(cd $(dirname $0);pwd)/.."
TEST_DIR=$PYTORCH_ROOT/test

if [[ $1 == "weekly" ]]; then
    test_list="test_ops test_profiler test_serialization test_testing test_utils test_jit_cuda_fuser test_ops_jit"
else
    # test_list="test_torch.py test_nn.py test_cuda.py test_optim.py test_reductions.py onnx/test_onnx_opset.py test_spectral_ops.py test_ops.py distributed/test_c10d_nccl.py distributed/test_c10d_gloo.py"
    # TODO(to fix): test_nn.py test_cuda.py onnx/test_onnx_opset.py 
    test_list="test_torch.py test_optim.py test_reductions.py test_spectral_ops.py test_ops.py distributed/test_c10d_nccl.py distributed/test_c10d_gloo.py"
fi

ERR=0
for test_file in ${test_list[@]}; do
    echo "Test start: "${test_file}
    python test/$test_file
    ret=$?
    if [[ $ret != 0 ]]; then
        ERR=$(expr ${ERR} + 1)
    fi
    echo "Test end: "${test_file}", exit "${ret}
done


if [[ ${ERR} != 0 ]]; then
    exit 1
else
    exit 0
fi

