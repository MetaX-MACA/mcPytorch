#!/bin/bash
# set -x

echo "****************************set environment variables***************************"
export USE_TDUMP=OFF
export TMEM_LOG=OFF
export DEBUG_ITRACE=0
export ISU_FASTMODEL=1

echo "****************************install required packages***************************"
pip install -r requirements.txt

if [[ $1 == "" ]]; then
    echo "Error: Please input epoch number."
    exit -1
fi
if [[ $2 == "" ]]; then
    echo "Found no batch_size option. Set batch_size to 1."
    batch_size=1
else
    batch_size=$2
fi

echo "Start to run bert amp under null_hardware mode..."
python bert_amp.py --null_hardware --epochs $1 --batch_size $batch_size --data_num $batch_size

if [[ $? != 0 ]];then
    echo "Error: AMP test run fail"
    exit 1
else
    echo "AMP test run success"
fi

