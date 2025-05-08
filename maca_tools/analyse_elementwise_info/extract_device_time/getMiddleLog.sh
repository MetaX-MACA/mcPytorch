#!/bin/bash


if [[ $1 == "" ]]; then
    echo "please give inp log"
    exit 1
fi
inp=$1

out="middle.log"
if [[ $2 != "" ]]; then
    out=$2
fi

python GetElemInfo.py --get_middle_log  \
                      --origin_log $inp  \
                      --middle_log $out
