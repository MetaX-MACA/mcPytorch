#!/bin/bash

export USE_NULL_HARDWARE=ON

cd $(dirname $(realpath $0))

if [[ $1 == "" ]]; then
    echo "Error: Please input loop number."
    exit -1
fi

if [[ $2 == "" ]]; then
    echo "Found no batch_size option. Set batch_size to 4."
    batch_size=4
else
    batch_size=$2
fi

echo "Start to run resent50 null hardware (batch_size = $batch_size)"
python run_resnet50.py --batch_size $batch_size --null_hardware --loop_num $1
if [[ $? != 0 ]]; then
    echo "Error: Failed to run resent50 null hardware"
    exit 1
else
    echo "Finish running resent50 null hardware"
fi

