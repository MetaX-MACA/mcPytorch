#!/bin/bash


if [[ $1 == "" ]]; then
    echo "please give inp log"
    exit 1
fi
inp=$1

out="device.log"
if [[ $2 != "" ]]; then
    out=$2
fi

python GetElemInfo.py --get_time_log \
                      --middle_log $inp \
		              --device_log $out
