#!/bin/bash

cd $(dirname $(realpath $0))

python case_upload.py --branch release --project_name C500_FW

if [[ $? != 0 ]];then
    echo " Failed run fw release case_upload.py"
    exit 1
else
    echo "Success run fw release case_upload.py"
    exit 0
fi