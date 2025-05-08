#!/bin/bash


if [[ $1 == "" ]]; then
    echo "please give macatime log"
    exit 1
fi
macalog=$1

if [[ $2 == "" ]]; then
    echo "please give cudatime log"
    exit 1
fi
cudalog=$2

compare=compare.log
if [[ $3 != "" ]]; then
    compare=$3
fi



if [[ $4 != "" ]]; then
    python GetElemInfo.py --get_compare_time_log  \
                      --maca_time_log $macalog  \
                      --cuda_time_log $cudalog \
		              --compare_time_log $compare \
                      --filter_no_time
else
    python GetElemInfo.py --get_compare_time_log  \
                        --maca_time_log $macalog  \
                        --cuda_time_log $cudalog \
                        --compare_time_log $compare
fi
