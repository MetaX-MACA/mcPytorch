#!/bin/bash

PYTORCH_ROOT="$(cd $(dirname $0);pwd)/.."
cd "$(dirname $0)/.."
source ${PYTORCH_ROOT}/maca_tools/env.sh

set_cmodel_env
startTime=`date +%Y%m%d-%H:%M:%S`
startTime_s=`date +%s`

python ./test/run_test.py --pytest --run-specified-test-cases ${PYTORCH_ROOT}/test_report/test_maca_case.csv --continue-through-error -v

status=$?

unset_cmodel_env

endTime=`date +%Y%m%d-%H:%M:%S`
endTime_s=`date +%s`

sumTime=$[ $endTime_s-$startTime_s ]
timeMinu=$[ $sumTime / 60 ]
echo "===== $startTime -----> $endTime Total run $timeMinu minutes"

if [[ $status != 0 ]];then
    exit 1
else
    exit 0
fi