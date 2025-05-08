#!/bin/bash
set -x


PYTORCH_ROOT="$(cd $(dirname $0);pwd)/.."
source ${PYTORCH_ROOT}/maca_tools/env.sh

set_cmodel_env
ERR=0

${PYTORCH_ROOT}/build/bin/test_maca_runtime run_test ${PYTORCH_ROOT}/test_report/cpp_maca_runtime_case.csv
ERR=$?

unset_cmodel_env

if [[ ${ERR} != 0 ]];then
    exit 1
else
    exit 0
fi