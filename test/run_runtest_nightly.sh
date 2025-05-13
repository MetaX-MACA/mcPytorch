#!/bin/bash

PYTORCH_ROOT="$(cd $(dirname $0);pwd)/.."
TEST_DIR=$PYTORCH_ROOT/test

if [[ $1 == "weekly" ]]; then
    test_list="test_ops test_profiler test_serialization test_testing test_utils test_jit_cuda_fuser test_ops_jit"
#    test_list="test_ops"
else
    test_list="test_torch test_nn nn_test_convolution nn_test_dropout nn_test_embedding nn_test_multihead_attention nn_test_pooling test_autograd test_reductions functorch_test_vmap test_cusolver"
fi

ERR=0
for csv_file in ${test_list[@]}; do
    if [ -n "$2" ]; then
        bash $TEST_DIR/run_runtest.sh $csv_file $2
    else
        bash $TEST_DIR/run_runtest.sh $csv_file
    fi
    if [[ $? != 0 ]]; then
        ERR=$(expr ${ERR} + 1)
    fi
done


if [[ ${ERR} != 0 ]]; then
    exit 1
else
    exit 0
fi

