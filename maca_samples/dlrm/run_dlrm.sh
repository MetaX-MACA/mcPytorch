#!/bin/bash

# set env
export USE_TDUMP=OFF
export TMEM_LOG=OFF
export DEBUG_ITRACE=0
export ISU_FASTMODEL=1

cd $(dirname $(realpath $0))

echo "Start to run dlrm (batch_size = 1)"
python run_dlrm.py --batch_size 1
if [[ $? != 0 ]];then
    echo "Error: Failed to run dlrm (batch_size = 1)"
    exit 1
else
    echo "Finish to run dlrm (batch_size = 1)"
    exit 0
fi
