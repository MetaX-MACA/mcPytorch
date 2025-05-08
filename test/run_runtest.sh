#!/bin/bash

declare -A test_map=(
    ["test_torch"]="5"
    ["test_nn"]="11"
    ["nn_test_convolution"]="4"
    ["nn_test_dropout"]="1"
    ["nn_test_embedding"]="1"
    ["nn_test_multihead_attention"]="1"
    ["nn_test_pooling"]="1"
    ["test_autograd"]="2"
    ["test_reductions"]="17"
    ["test_jit_cuda_fuser"]="3"
    ["test_ops_jit"]="6"
    ["test_profiler"]="1"
    ["test_serialization"]="1"
    ["test_testing"]="3"
    ["test_utils"]="1"
    ["functorch_test_vmap"]="4"
    ["test_expanded_weights"]="1"
    ["test_ops"]="153"
)

PYTORCH_ROOT="$(cd $(dirname $0);pwd)/.."
cd "$(dirname $0)/.."
source ${PYTORCH_ROOT}/maca_tools/env/env_run_fast.sh

set_cmodel_env
startTime=`date +%Y%m%d-%H:%M:%S`
startTime_s=`date +%s`

num=${test_map[$1]}
ERR=0

echo $num
for((i=0; i<$num;i++))
do  
    filename=$1_seg$i.csv
    file=${PYTORCH_ROOT}/test_report/runtest/$1_seg$i.csv
    if [ -n "$2" ]; then
        if [[ $filename =~ ^functorch* ]]; then
            python ./test/run_test.py --functorch --use-pytest --run-specified-test-cases $file --export-report-path $2 --continue-through-error -v
        else
            python ./test/run_test.py --use-pytest --run-specified-test-cases $file --export-report-path $2 --continue-through-error -v
        fi
    else
        if [[ $filename =~ ^functorch* ]]; then
            python ./test/run_test.py --functorch --use-pytest --run-specified-test-cases $file --continue-through-error -v
        else
            python ./test/run_test.py --use-pytest --run-specified-test-cases $file --continue-through-error -v
        fi 
    fi
    if [[ $? != 0 ]]; then
        echo "Error found in ${file}"
#        ERR=$(expr ${ERR} + 1)
    fi
done


unset_cmodel_env

endTime=`date +%Y%m%d-%H:%M:%S`
endTime_s=`date +%s`

sumTime=$[ $endTime_s-$startTime_s ]
timeMinu=$[ $sumTime / 60 ]
echo "===== $startTime -----> $endTime Total run $timeMinu minutes"

if [[ $ERR != 0 ]];then
    exit 1
else
    exit 0
fi

