#!/bin/bash

# set env
export USE_TDUMP=OFF
export TMEM_LOG=OFF
export DEBUG_ITRACE=0
export ISU_FASTMODEL=1


cd $(dirname $(realpath $0))
echo "Start to run resent50 amp (batch_size = 1)"
echo "Start to check test data resent50 amp (batch_size = 1)"


if [[ $1 != "debug" ]];then
    python resnet50_amp.py --batch_size 1 --run_type check_data --amp_mode amp
else
    python resnet50_amp.py --batch_size 1 --run_type check_data --amp_mode amp --debug --checkpoint_mode checkpoint_record --checkpoint_step 1
fi

if [[ $? != 0 ]];then
    echo "Error: Failed to check test data resent50 amp (batch_size = 1)"
    exit 1
else
    echo "Finish to check test data resent50 amp (batch_size = 1)"
fi

echo "Finish to run resent50 (batch_size = 1)"
