#!/bin/bash
# set -x

echo "****************************set environment variables***************************"
export USE_TDUMP=OFF
export TMEM_LOG=OFF
export DEBUG_ITRACE=0
export ISU_FASTMODEL=1

echo "****************************install required packages***************************"
pip install -r requirements.txt

echo "Start to run bert amp..."
python bert_amp.py

if [[ $? != 0 ]];then
    echo "Error: AMP test run fail"
    exit 1
else
    echo "AMP test run success"
fi

