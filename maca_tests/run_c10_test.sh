#!/bin/bash

c10_tests=(
    # "c10_Array_test"
    "c10_Bitset_test"
    # "c10_C++17_test"
    "c10_CompileTimeFunctionPointer_test"
    "c10_ConstexprCrc_test"
    "c10_DeviceGuard_test"
    "c10_Device_test"
    "c10_DispatchKeySet_test"
    "c10_Half_test"
    "c10_InlineDeviceGuard_test"
    "c10_InlineStreamGuard_test"
    "c10_LeftRight_test"
    "c10_Metaprogramming_test"
    "c10_SizesAndStrides_test"
    "c10_StreamGuard_test"
    "c10_ThreadLocal_test"
    "c10_TypeIndex_test"
    "c10_TypeList_test"
    "c10_TypeTraits_test"
    "c10_accumulate_test"
    "c10_bfloat16_test"
    "c10_complex_math_test"
    "c10_complex_test"
    # "c10_either_test"
    # turn off c10_exception_test, at present, we need to print detailed error info
    # "c10_exception_test"
    "c10_flags_test"
    "c10_intrusive_ptr_test"
    "c10_irange_test"
    "c10_logging_test"
    "c10_cuda_CUDATest"
   "c10_optional_test"
    "c10_ordered_preserving_dict_test"
    "c10_registry_test"
    "c10_string_view_test"
    "c10_tempfile_test"
    "c10_typeid_test"
)

PYTORCH_ROOT="$(cd $(dirname $0);pwd)/.."
pytorch_test_path=${PYTORCH_TEST_PATH}
c10_test_path="$PYTORCH_ROOT/build/bin/"
if [[ -d "$pytorch_test_path" ]]; then
  if [[ ${USE_SLRUM} == 1 ]]; then
    echo "PYTORCH_TEST_PATH not supported slrum!"
    exit 1
  fi
  c10_test_path="$pytorch_test_path/torch/test/"
fi

if [[ ${PYTORCH_ZEBU_TEST} != 1 ]]; then
    source ${PYTORCH_ROOT}/maca_tools/env/env_run_fast.sh
fi

# currently run_c10_test.sh is fast enough.
USE_SLRUM=0

if [[ ${USE_SLRUM} == 1 ]]; then
    source ${PYTORCH_ROOT}/maca_tests/utils.sh
    set_cmodel_env
    cmd_args_str=$(IFS=\;;echo "${c10_tests[*]}")
    launch_and_check_slurm_jobs \
        "${PYTORCH_ROOT}/maca_tests/run_sbatch_job.sh" \
        "${PYTORCH_ROOT}/build/bin/" \
        "${cmd_args_str}" \
        "run_c10_test" \
        "0" \
        "$1"
    unset_cmodel_env
else
    if [[ ${PYTORCH_ZEBU_TEST} != 1 ]]; then
        set_cmodel_env
    fi
    ERR=0
    for file in ${c10_tests[@]}; do
        echo "Start test $file"
        if [ -n "$1" ]; then
            $c10_test_path/${file} --gtest_output=xml:"$1/${file}.xml"
        else
            $c10_test_path/${file}
        fi
        if [[ $? != 0 ]]; then
            ERR=$(expr ${ERR} + 1)
            echo "Error in tests of ${file}."
        else
            echo "Success in tests of ${file}."
        fi
    done
    if [[ ${PYTORCH_ZEBU_TEST} != 1 ]]; then
        unset_cmodel_env
    fi
    if [[ ${ERR} != 0 ]];then
        exit 1
    else
        exit 0
    fi
fi


