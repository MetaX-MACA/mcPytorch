#!/bin/bash

aten_test_names=(
    "CppSignature_test"
    "Dict_test"
    "Dimname_test"
    "KernelFunction_test"
    "List_test"
    "NamedTensor_test"
    # "TensorImpl_test" # fail
    "apply_utils_test"
    "atest"
    "backend_fallback_test"
    # todo rand
    # "basic"
    "broadcast_test"
    "cpu_generator_test"
    "cpu_profiling_allocator_test"
    "cpu_rng_test"
    # "maca_apply_test" # fail
    # "maca_atomic_ops_test"    # fail
    # "maca_complex_math_test"  # fail
    # "maca_complex_test"   # fail
    # "maca_cub_test"   # fail
    # "maca_distributions_test" # fail
#    "maca_dlconvertor_test"   # not support, need dlpack in tvm ecology
    # "maca_generator_test" # fail
    # "maca_half_test"  # fail
#    "maca_integer_divider_test"    # fail
    # "maca_optional_test"  # fail
    # "maca_packedtensoraccessor_test"  # fail
    # "maca_reportMemoryUsage_test" # fail
    # "maca_stream_test"    # fail
    # "maca_tensor_interop_test"    # fail
    # "maca_vectorized_test"    # fail
    "dlconvertor_test"
    "extension_backend_test"
    "half_test"
    "ivalue_test"
    "kernel_function_legacy_test"
    "kernel_function_test"
    "kernel_lambda_legacy_test"
    "kernel_lambda_test"
    "kernel_stackbased_test"
    "lazy_tensor_test"
    "make_boxed_from_unboxed_functor_test"
    "math_kernel_test"
    "memory_format_test"
    "memory_overlapping_test"
    "mobile_memory_cleanup"
    "native_test"
    "op_allowlist_test"
    "op_registration_test"
    "operators_test"
    "pow_test"
    "quantized_test"
    "reportMemoryUsage_test"
#   "scalar_tensor_test"
    "scalar_test"
    "tensor_iterator_test"
    "test_parallel"
    "thread_init_test"
    "type_test"
    "undefined_tensor_test"
    # "variant_test"
    # "vec_test_all_types_AVX2" # fail
    "vec_test_all_types_DEFAULT"
    "verify_api_visibility"
    # "vmap_test"   # fail
    "weakref_test"
    "wrapdim_test"
    # "test_mcrand_api" # fail
)

PYTORCH_ROOT="$(cd $(dirname $0);pwd)/.."

if [[ ${PYTORCH_ZEBU_TEST} != 1 ]]; then
    source ${PYTORCH_ROOT}/maca_tools/env/env_run_fast.sh
fi
cd ${PYTORCH_ROOT}/..

aten_test_path="${PYTORCH_ROOT}/build/bin/"
pytorch_test_path=${PYTORCH_TEST_PATH}
if [[ -d "$pytorch_test_path" ]]; then
  if [[ ${USE_SLRUM} == 1 ]]; then
    echo "PYTORCH_TEST_PATH not supported slrum!"
    exit 1
  fi
  aten_test_path="$pytorch_test_path/torch/test"
fi
cd ${aten_test_path}

USE_SLRUM=0

if [[ ${USE_SLRUM} == 1 ]]; then
    source ${PYTORCH_ROOT}/maca_tests/utils.sh
    set_cmodel_env
    cmd_args_str=$(IFS=\;;echo "${aten_test_names[*]}")
    launch_and_check_slurm_jobs \
        "${PYTORCH_ROOT}/maca_tests/run_sbatch_job.sh" \
        "${PYTORCH_ROOT}/build/bin/" \
        "${cmd_args_str}" \
        "run_aten_test" \
        "0" \
        "$1"
    ret=$?
    unset_cmodel_env
    exit ${ret}
else
    if [[ ${PYTORCH_ZEBU_TEST} != 1 ]]; then
        set_cmodel_env
    fi
    ERR=0
    err_files=()
    lost_files=()
    for file in ${aten_test_names[@]};do
        if [ ! -f "${file}" ]
        then
            ERR=$(expr ${ERR} + 1)
            lost_files[${#lost_files[*]}]=${file}
            echo "Lost:${lost_files[$i]}"
        else
            echo "Start test $file"
            if [ -n "$1" ]; then
                ./${file} --gtest_output=xml:"$1/${file}.xml"
            else
                ./${file}
            fi
            if [[ $? != 0 ]];then
                ERR=$(expr ${ERR} + 1)
                err_files[${#err_files[*]}]=${file}
                echo "Error in tests of ${file}."
            else
                echo "Success in tests of ${file}."
            fi
        fi
    done
    if [[ ${PYTORCH_ZEBU_TEST} != 1 ]]; then
        unset_cmodel_env
    fi
    if [[ ${ERR} != 0 ]];then
        for((i=0;i<${#err_files[@]};i++)); do
            echo ${err_files[$i]}
        done
        for((i=0;i<${#lost_files[@]};i++)); do
            echo ${lost_files[$i]}
        done
        exit 1
    else
        exit 0
    fi
fi
