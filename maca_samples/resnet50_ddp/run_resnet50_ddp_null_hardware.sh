#!/bin/bash
# set -x

echo "****************************set environment variables***************************"
export USE_TDUMP=OFF
export TMEM_LOG=OFF
export DEBUG_ITRACE=0
export ISU_FASTMODEL=1

if [[ $1 == "" ]]; then
    echo "Error: Please input loop number."
    exit -1
fi
if [[ $2 == "" ]]; then
    echo "Found no batch_size option. Set batch_size to 1."
    batch_size=1
else
    batch_size=$2
fi

echo "****************************install required packages***************************"
pip install -r requirements.txt

echo "************************start running bert_ddp null hardware.*******************"
python run_resnet50_ddp.py --null_hardware --loop_num $1 --batch_size $batch_size

if [[ $? != 0 ]];then
    echo "Error: run fail"
    exit 1
else
    echo "run success"
fi
