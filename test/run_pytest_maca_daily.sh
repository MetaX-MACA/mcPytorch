#!/bin/bash

pip install pytest pytest-xdist expecttest

declare -A test_map=(
    ["gloo"]="test_c10d_gloo_maca_daily.csv"
    ["mccl"]="test_c10d_mccl_maca_daily.csv"
    ["pipeline"]="test_pipeline_maca_daily.csv"
)

CUR_DIR="$(cd $(dirname $0);pwd)/"
cd ${CUR_DIR}

csv_file_list=${test_map[$1]}
ERR=0
for csv_file in ${csv_file_list[@]}; do
    python ./run_test.py --use-pytest --run-specified-test-cases ./maca_daily_test_report/${csv_file} --continue-through-error -v
    if [[ $? != 0 ]]; then
        ERR=$(expr ${ERR} + 1)
    fi
done
if [[ $ERR != 0 ]];then
    exit 1
else
    exit 0
fi
