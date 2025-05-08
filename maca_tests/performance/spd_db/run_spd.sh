#!/bin/bash

cd $(dirname $(realpath $0))

python case_upload.py

if [[ $? != 0 ]];then
    echo " Failed run spd case_upload.py"
    exit 1
else
    echo "Success run spd case_upload.py"
    exit 0
fi