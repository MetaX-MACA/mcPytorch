#!/bin/bash

source /home/sw/pytorch/env.sh

cd $(dirname $(realpath $0))

if [[ $1 == "" ]]; then
    echo "Error: Please input batch size."
    exit -1
fi

echo "Start to run resent50 (batch_size = $1)"

python run_resnet50.py --batch_size $1 --mode c500_chip

if [[ $? != 0 ]];then
    echo "Error: Failed to check test data resent50 (batch_size = $1)"
    exit 1
else
    echo "Finish to check test data resent50 (batch_size = $1)"
fi

echo "Finish to run resent50 (batch_size = $1)"