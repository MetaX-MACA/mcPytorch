#!/bin/bash

function set_cmodel_env() {
    export USE_TDUMP=OFF
    export TMEM_LOG=OFF
    export DEBUG_ITRACE=0
    export ISU_FASTMODEL=1
    export MXLOG_LEVEL=err  # verbose, debug, info, warn, err, critical, off
    export MCCL_P2P_DISABLE=1
}

function unset_cmodel_env() {
    unset USE_TDUMP
    unset TMEM_LOG
    unset DEBUG_ITRACE
    unset ISU_FASTMODEL
    unset MXLOG_LEVEL
    unset MCCL_P2P_DISABLE
}

export PATH=${MACA_PATH}/tools/cu-bridge/tools:${PATH}
export MACA_CLANG_PATH=${MACA_PATH}/mxgpu_llvm/bin
export LD_LIBRARY_PATH=$MACA_PATH/lib:$MACA_PATH/mxgpu_llvm/lib:${LD_LIBRARY_PATH}

unset_cmodel_env
set_cmodel_env