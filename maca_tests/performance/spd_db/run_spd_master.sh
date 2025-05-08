#!/bin/bash

cd $(dirname $(realpath $0))

python case_upload.py --branch master

if [[ $? != 0 ]];then
    echo " Failed run spd master case_upload.py"
    exit 1
else
    echo "Success run spd master case_upload.py"
    exit 0
fi