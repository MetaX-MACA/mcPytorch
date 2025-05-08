#!/bin/bash
# set -x

echo "****************************set environment variables***************************"
export USE_TDUMP=OFF
export TMEM_LOG=OFF
export DEBUG_ITRACE=0
export ISU_FASTMODEL=1

echo "****************************install required packages***************************"
pip install -r requirements.txt

echo "******************************start running bert_ddp****************************"
python run_resnet50_ddp.py

if [[ $? != 0 ]];then
    echo "Error: run fail"
    exit 1
else
    echo "run success"
fi