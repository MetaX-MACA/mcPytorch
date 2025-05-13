#!/bin/bash

# set env
export USE_TDUMP=OFF
export TMEM_LOG=OFF
export DEBUG_ITRACE=0
export ISU_FASTMODEL=1

cd $(dirname $(realpath $0))

echo "****************************install required packages***************************"
pip install -r requirements.txt

echo "Start to run yolov5 (batch_size = 1)"
echo "Start to create golden data yolov5 (batch_size = 1)"
python train.py --batch_size 1 --mode create_golden_data
if [[ $? != 0 ]];then
    echo "Error: Failed to create golden data yolov5 (batch_size = 1)"
    exit 1
else
    echo "Finish to create golden data yolov5 (batch_size = 1)"
fi
echo "Start to check test data yolov5 (batch_size = 1)"
python train.py --batch_size 1 --mode check_test_data
if [[ $? != 0 ]];then
    echo "Error: Failed to check test data yolov5 (batch_size = 1)"
    exit 1
else
    echo "Finish to check test data yolov5 (batch_size = 1)"
fi
echo "Finish to run yolov5 (batch_size = 1)"
