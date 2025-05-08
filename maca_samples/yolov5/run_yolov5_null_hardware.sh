#!/bin/bash

# set env
export USE_TDUMP=OFF
export TMEM_LOG=OFF
export DEBUG_ITRACE=0
export ISU_FASTMODEL=1

cd $(dirname $(realpath $0))

echo "****************************install required packages***************************"
pip install -r requirements.txt

if [[ $1 == "" ]]; then
    echo "Error: Please input loop number."
    exit -1
fi
if [[ $2 == "" ]]; then
    echo "Found no batch_size option. Set batch_size to 2."
    batch_size=2
else
    batch_size=$2
fi

echo "Start to run yolov5 null hardware(batch_size = 1)"
python train.py --batch_size $batch_size --mode null_hardware --loop_num $1

if [[ $? != 0 ]];then
    echo "Error: Failed to run yolov5 null hardware (batch_size = 1)"
    exit 1
else
    echo "Finish to run yolov5 null hardware (batch_size = 1)"
fi
