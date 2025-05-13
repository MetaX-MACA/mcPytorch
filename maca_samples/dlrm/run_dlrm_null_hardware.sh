#!/bin/bash

export USE_NULL_HARDWARE=ON

cd $(dirname $(realpath $0))

if [[ $1 == "" ]]; then
    echo "Error: Please input loop number."
    exit -1
fi
if [[ $2 == "" ]]; then
    echo "Found no batch_size option. Set batch_size to 1."
    batch_size=4
else
    batch_size=$2
fi

echo "Start to run dlrm null hardware (batch_size = $batch_size)"
python run_dlrm.py --batch_size $batch_size --null_hardware --loop_num $1
if [[ $? != 0 ]]; then
    echo "Error: Failed to run dlrm null hardware"
    exit 1
else
    echo "Finish running dlrm null hardware"
fi
