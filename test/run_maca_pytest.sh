#!/bin/bash

declare -A test_map=(
    ["nn"]="test_nn_TestNNDeviceTypeCUDA_pass.csv test_nn_TestNNDeviceTypeCUDA_conv_fp32.csv test_nn_weekly_all_pass.csv"
    ["torch"]="test_torch_TestTorchDeviceTypeCUDA_pass.csv test_torch_TestDevicePrecisionCUDA.csv test_torch_TestVitalSignsCudaCUDA.csv test_torch_segment0.csv test_torch_segment1.csv test_torch_segment2.csv test_torch_segment3.csv test_torch_segment4.csv test_torch_segment5.csv test_torch_segment6.csv test_torch_segment7.csv"
    ["cuda"]="test_cuda_maca_pass.csv"
    ["optim"]="test_optim_case.csv"
    ["reduction"]="test_reductions_case_pass_cuda.csv"
    ["mccl_and_gloo_weekly"]="test_c10d_mccl_pass_weekly.csv test_c10d_gloo_pass_weekly.csv"
    ["mccl_and_gloo_checkin"]="test_c10d_mccl_pass_checkin.csv test_c10d_gloo_pass_checkin.csv"
    ["onnx"]="test_onnx_opset_pass.csv test_onnx_shape_inference_pass.csv"
    ["spectral"]="test_spectral_ops.csv"
    ["pipeline"]="test_pipe_pass.csv test_stash_pop_pass.csv test_verify_skippables_pass.csv"
    ["dynamo"]="test_dynamo/test_aot_autograd.csv test_dynamo/test_cudagraphs.csv test_dynamo/test_functions.csv test_dynamo/test_minifier.csv test_dynamo/test_modules.csv test_dynamo/test_python_autograd.csv test_dynamo/test_repros.cdynamo/test_sv test_dynamo/test_unspec.csv test_dynamo/test_backends.csv test_dynamo/test_export.csv test_dynamo/test_global.csv test_dynamo/test_misc.csv test_dynamo/test_nops.csv test_dynamo/test_recompile_ux.csv test_dynamo/test_skip_non_tensor.csv test_dynamo/test_verify_correctness.csv test_dynamo/test_comptime.csv test_dynamo/test_export_mutations.csv test_dynamo/test_interop.csv test_dynamo/test_model_output.csv test_dynamo/test_optimizers.csv test_dynamo/test_replay_record.csv test_dynamo/test_subgraphs.csv"
    ["inductor"]="test_inductor/test_config.csv test_inductor/test_minifier.csv test_inductor/test_pattern_matcher.csv test_inductor/test_perf.csv test_inductor/test_select_algorithm.csv test_inductor/test_smoke.csv test_inductor/test_torchinductor.csv test_inductor/test_torchinductor_opinfo.csv"
)

PYTORCH_ROOT="$(cd $(dirname $0);pwd)/.."
cd "$(dirname $0)/.."
source ${PYTORCH_ROOT}/maca_tools/env/env_run_fast.sh

set_cmodel_env
startTime=`date +%Y%m%d-%H:%M:%S`
startTime_s=`date +%s`

csv_file_list=${test_map[$1]}
ERR=0
for csv_file in ${csv_file_list[@]}; do
    if [ -n "$2" ]; then
        python ./test/run_test.py --run-specified-test-cases ${PYTORCH_ROOT}/test_report/${csv_file} --export-report-path $2 --continue-through-error -v
    else
        python ./test/run_test.py --run-specified-test-cases ${PYTORCH_ROOT}/test_report/${csv_file} --continue-through-error -v
    fi
    if [[ $? != 0 ]]; then
        ERR=$(expr ${ERR} + 1)
    fi
done

unset_cmodel_env

endTime=`date +%Y%m%d-%H:%M:%S`
endTime_s=`date +%s`

sumTime=$[ $endTime_s-$startTime_s ]
timeMinu=$[ $sumTime / 60 ]
echo "===== $startTime -----> $endTime Total run $timeMinu minutes"

if [[ $ERR != 0 ]];then
    exit 1
else
    exit 0
fi
