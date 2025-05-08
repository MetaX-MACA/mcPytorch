#!/bin/bash
cd $(dirname $0)
startTime=`date +%Y%m%d-%H:%M:%S`
startTime_s=`date +%s`

if [[ $1 == "checkin" ]]; then
    kernel_test_py=(
        "./test_AdaptiveAveragePooling.py"
        "./test_AdaptiveMaxPooling2d.py"
        "./test_AddAllDataType.py"
        "./test_batchnorm_precision.py"
        "./test_broadcast_copy.py"
        "./test_Broadcast.py --type=checkin"
        "./test_Bucketization.py"
        "./test_conv_depthwise2d_backward.py"
        "./test_convtranspose.py"
        "./test_cunnSoftmax.py --type=checkin"
        "./test_DilatedMaxPool2d.py"
        "./test_DistanceKernel.py"
        "./test_div.py"
        "./test_Dot_BHalf.py"
        "./test_driverApi.py"
        "./test_exponential.py"
        "./test_flip.py"
        "./test_FractionalMaxPool2d.py"
        "./test_group_norm_backward.py"
        "./test_group_norm.py"
        "./test_indexing_backward_kernel.py --type=checkin"
        "./test_layer_norm.py"
        "./test_Matmul_Batch_Half.py"
        "./test_MaxUnpooling.py"
        "./test_memory_layout.py"
        "./test_multi_output.py"
        "./test_Mv.py"
        "./test_Optimization.py"
        "./test_reduce_accuracy.py --type=checkin"
        "./test_Repeat.py"
        "./test_ReplicationPadding.py"
        "./test_Resize.py"
        "./test_runtime.py"
        # "./test_scaled_dot_attention.py" # gcc version problem
        "./test_Shape.py"
        "./test_sigmoid.py"
        "./test_Sorting.py"
        "./test_sort.py"
        "./test_sparse.py"
        "./test_SummaryOps.py"
        "./test_topk.py"
        "./test_TriangularOps.py"
        "./test_unrolled_elementwise.py"
        "./test_UpSampleNearest2d.py"
        "./test_WeightNorm.py"
        "./test_elementwise_copy_highdim.py"
        "./test_inductor_smoke.py"
        "./test_triton_device_assert.py"
	"./test_unrollelementwise_copy_cast.py"
    )
fi

if [[ $1 == "daily" || $1 == "device" ]]; then
    kernel_test_py=(
        "./test_AdaptiveAveragePooling3d.py"
        "./test_AdaptiveAveragePooling.py"
        "./test_AdaptiveMaxPooling2d.py"
        "./test_AdaptiveMaxPooling3d.py"
        "./test_AddAllDataType.py"
        "./test_add_unrolled_opt.py"
        "./test_AveragePool2d.py"
        "./test_AveragePool3d.py"
        "./test_avgpool2d.py"
        "./test_batchnorm_precision.py"
        "./test_bn_2_5_elementwise.py"
        "./test_bn_uncontiguous_backward.py"
        "./test_broadcast_copy.py"
        "./test_Broadcast.py --type=checkin"
        "./test_Broadcast.py --type=daily"
        "./test_Bucketization.py"
        "./test_cat_dim5_opt.py"
        "./test_cat_opt.py"
        "./test_conv_depthwise2d_backward.py"
        "./test_convtranspose.py"
        "./test_cunnSoftmax.py --type=checkin"
        "./test_cunnSoftmax.py --type=daily"
        "./test_DilatedMaxPool2d.py"
        "./test_DilatedMaxPool3d.py"
        "./test_DistanceKernel.py"
        "./test_div.py"
        "./test_Dot_BHalf.py"
        "./test_Dot_Half.py"
        "./test_driverApi.py"
        "./test_Dot.py"
        "./test_elem_2_2_arity1_uncontiue.py"
        "./test_element_3_2_cast_broadcast.py"
        "./test_element_3_2_cast.py"
        "./test_element_4_2_cast.py"
        "./test_element_5_1_lowdim_contiguous.py"
        "./test_elementwise_2_1_cp_cast_dim0_contiguous.py"
        "./test_elementwise_2_1_dim0_contiguous.py"
        "./test_elementwise_2_1_input_lowdim_contiguous.py"
        "./test_elementwise_2_2_align.py"
        "./test_elementwise_2_2_cast_broadcast.py --type=checkin"
        "./test_elementwise_2_2_cast_broadcast.py --type=daily"
        "./test_elementwise_2_2_dim0_contiguous_arg1_dim1_broadcast.py"
        "./test_elementwise_3_2_broadcast_dim0.py"
        "./test_elementwise_3_2_broadcast_dim1.py --type=checkin"
        "./test_elementwise_3_2_broadcast_dim1.py --type=daily"
        "./test_elementwise_3_2_broadcast_dim2_arg0_contiguous.py --type=checkin"
        "./test_elementwise_3_2_broadcast_dim2_arg0_contiguous.py --type=daily"
        "./test_elementwise_3_2_broadcast_dim2.py --type=checkin"
        "./test_elementwise_3_2_broadcast_dim2.py --type=daily"
        "./test_elementwise_3_2_dim0_contiguous_arg1_dim1_broadcast.py"
        "./test_elementwise_3_2_lowdim_contiguous.py"
        "./test_elementwise_3_2_arity2_transpose.py"
        "./test_elementwise_3_2_tile.py"
        "./test_elementwise_4_1_dim0_contiguous.py"
        "./test_elementwise_4_1_input_lowdim_continuous.py"
        "./test_elementwise_4_1_transpose.py"
        "./test_elementwise_4_2_broadcast_arg0_dim2_arg1_dim0.py"
        "./test_elementwise4_2_cast_broadcast.py"
        "./test_elementwise_broadcast_1_1.py --type=checkin"
        "./test_elementwise_broadcast_1_1.py --type=daily"
        "./test_elementwise_broadcast_3_1.py"
        "./test_elementwise_broadcast_3_2_arg0_dim2_arg1_dim0.py"
        "./test_elementwise_broadcast_3_3.py"
        "./test_elementwise_copy_3_1.py --type=checkin"
        "./test_elementwise_copy_3_1.py --type=daily"
        "./test_elementwise_copy_highdim.py"
        "./test_elementwise_copy_not_align.py"
        "./test_elementwise_cp51_42.py"
        "./test_elementwise_dilation_1_1.py"
        "./test_elementwise_dim4.py --type=checkin"
        "./test_elementwise_dim4.py --type=daily"
        "./test_elementwise_kernel_transpose_copy_64_uncontiguous.py"
        "./test_elementwise_transpose.py --type=daily"
        "./test_elementwise_n_1_dim0_pad.py"
        "./test_exponential.py"
        "./test_flip.py"
        "./test_foreach.py"
        "./test_FractionalMaxPool2d.py"
        "./test_FractionalMaxPool3d.py"
        "./test_group_norm_backward.py"
        "./test_group_norm.py"
        "./test_index_add_large.py"
        "./test_indexing_backward_kernel.py --type=checkin"
        "./test_indexing_backward_kernel.py --type=daily"
        "./test_index.py --type=checkin"
        "./test_index.py --type=daily"
        "./test_layer_norm.py"
        "./test_Matmul_Batch_BHalf.py"
        "./test_Matmul_Batch_Half.py"
        "./test_Matmul_BHalf.py"
        "./test_Matmul_Complex.py"
        "./test_Matmul_Half.py"
        "./test_max_bfloat16_accuracy.py"
        "./test_maxpool2d.py"
        "./test_maxpool3d.py"
        "./test_MaxUnpooling.py"
        "./test_mctx.py"
        "./test_memory_layout.py"
        "./test_MultiMarginLoss.py"
        "./test_multi_output.py"
        "./test_Mv.py"
        "./test_nllloss.py"
        "./test_Optimization.py"
        "./test_rand_byself.py"
        "./test_rand_rng_state.py"
        "./test_reduce_accuracy.py --type=daily"
        "./test_Repeat.py"
        "./test_ReplicationPadding.py"
        "./test_Resize.py"
        "./test_roll.py"
        "./test_runtime.py"
        # "./test_scaled_dot_attention.py" # gcc version problem
        "./test_scatter_gather_elementwise_kernel.py --type=checkin"
        "./test_indexing_backward_kernel.py --type=checkin"
        "./test_scatter_opt_pw.py"
        "./test_Shape.py"
        "./test_sigmoid.py"
        "./test_silu.py"
        "./test_softmax_backward_opt.py"
        "./test_softmax_dim_less_than_128.py"
        "./test_Sorting.py"
        "./test_sort.py"
        "./test_sparse.py"
        "./test_SummaryOps.py"
        "./test_topk.py"
        "./test_TriangularOps.py"
        "./test_triu_tril.py"
        "./test_unrolled_elementwise.py"
        "./test_upsample_bicubic.py"
        "./test_UpSampleBicubic2d.py"
        "./test_UpSampleBilinear2d.py"
        "./test_UpSampleLinear1d.py"
        "./test_UpSampleNearest1d.py"
        "./test_UpSampleNearest2d.py"
        "./test_UpSampleNearest3d.py"
        "./test_UpSampleTrilinear3d.py"
        "./test_upsample_trilinear3d.py"
        "./test_VDot.py"
        "./test_WeightNorm.py"
        # ############# bert related test #############
        "./bert/test_bert_ops.py --op_type=embedding"
        "./bert/test_bert_ops.py --op_type=expand"
        "./bert/test_bert_ops.py --op_type=view"
        "./bert/test_bert_ops.py --op_type=transpose"
        "./bert/test_bert_ops.py --op_type=permute"
        "./bert/test_bert_ops.py --op_type=zeros"
        "./bert/test_bert_ops.py --op_type=add"
        "./bert/test_bert_ops.py --op_type=dividescalar"
        "./bert/test_bert_ops.py --op_type=tanh"
        "./bert/test_bert_ops.py --op_type=gelu"
        "./bert/test_bert_ops.py --op_type=softmax"
        "./bert/test_bert_ops.py --op_type=contiguous"
        "./bert/test_bert_ops.py --op_type=arange"
        "./bert/test_bert_ops.py --op_type=transpose_matmul_combine"
        "./bert/test_bert_ops.py --op_type=linear_simple"
        "./bert/test_bert_ops.py --op_type=matmul_simple"
        "./bert/test_bert_ops.py --op_type=matmul_bw"
        "./bert/test_bert_Expand.py"
        "./bert/test_bert_View.py"
        "./bert/test_bert_Transpose.py"
        "./bert/test_bert_Permute.py"
        "./bert/test_bert_Zeros.py"
        "./bert/test_bert_Add.py"
        "./bert/test_bert_DivideScalar.py"
        # ############# resnet50 related test #############
        "./resnet50/test_resnet50_AdaptiveAveragePooling2d.py"
        "./resnet50/test_resnet50_Add.py"
        "./resnet50/test_resnet50_BatchNorm.py"
        "./resnet50/test_resnet50_DilatedMaxPool2d.py"
        "./resnet50/test_resnet50_Linear.py"
        "./resnet50/test_resnet50_Relu.py"
        # ############ dlrm related test #############
        "./dlrm/test_BCELoss.py"
        "./dlrm/test_Bmm.py"
        "./dlrm/test_Cat.py"
        "./dlrm/test_Clamp.py"
        "./dlrm/test_EmbeddingBag.py"
        "./dlrm/test_Linear.py"
        "./dlrm/test_MSELoss.py"
        "./dlrm/test_Relu.py"
        "./dlrm/test_Sigmoid.py"
        "./dlrm/test_Transpose.py"
    )

    if [[ ${PYTORCH_TEST_INTERNAL} == 1 ]]; then
        kernel_test_py+=("./bert/test_bert_ops.py --op_type=dropout")
        kernel_test_py+=("./test_import_accuracy.py")
        kernel_test_py+=("./test_math.py")
        kernel_test_py+=("./test_math_2.py")
        kernel_test_py+=("./test_rand_with_a100_golden.py")
        kernel_test_py+=("./test_silu_with_a100_golden.py")
    fi
fi

if [[ $1 == "daily" || $1 == "device" ]]; then
    kernel_test_py+=("./bert/test_bert_ops.py --op_type=linear")
    kernel_test_py+=("./bert/test_bert_ops.py --op_type=matmul")
    kernel_test_py+=("./resnet50/test_resnet50_ops.py --op_type conv2d")
    kernel_test_py+=("./resnet50/test_resnet50_ops.py --op_type batchnorm")
    kernel_test_py+=("./resnet50/test_resnet50_ops.py --op_type relu")
    kernel_test_py+=("./resnet50/test_resnet50_ops.py --op_type avgpool")
    kernel_test_py+=("./resnet50/test_resnet50_ops.py --op_type maxpool")
    kernel_test_py+=("./resnet50/test_resnet50_ops.py --op_type add")
    kernel_test_py+=("./resnet50/test_resnet50_ops.py --op_type misc")
    kernel_test_py+=("./bert/test_bert_ops.py --op_type=layernorm")
    kernel_test_py+=("./bert/test_bert_ops.py --op_type=layernorm_bww")
fi

# inductor test
if [[ $1 == "daily" || $1 == "device" ]]; then
    kernel_test_py+=("../test/inductor/test_codegen_triton.py")
    kernel_test_py+=("../test/inductor/test_codecache.py")
    kernel_test_py+=("../test/inductor/test_compile_worker.py")
    kernel_test_py+=("../test/inductor/test_config.py")
    kernel_test_py+=("../test/inductor/test_cuda_repro.py")
fi

if [[ ${#kernel_test_py[@]} == 0 ]]; then
    echo "Empty test list."
    exit 0
fi

ERR=0
PASS=0
err_files=()
res=""
for file in "${kernel_test_py[@]}";do
    echo "Start test $file"
    filename=$(basename "$file")
    if [ "$filename" = "test_memory_layout.py" ]; then
        export PYTORCH_DEFAULT_NDHWC=1
    fi
    if echo "$filename" | grep -q "test_indexing_backward_kernel.py"; then
        export PYTORCH_ENABLE_INDEXING_BACKWARD_KERNEL_OPT=1
    fi
    testStartTime=$(date +%s)
    python $file
    if [[ $? != 0 ]];then
        res+="${file},fail,$(($(date +%s) - ${testStartTime}))#"
        ERR=$(expr ${ERR} + 1)
        err_files[${#err_files[*]}]=${file}
        echo "Error in tests of ${file}."
    else
        PASS=$(expr ${PASS} + 1)
        res+="${file},pass,$(($(date +%s) - ${testStartTime}))#"
        echo "Success in tests of ${file}."
    fi
    unset PYTORCH_DEFAULT_NDHWC
    unset PYTORCH_ENABLE_INDEXING_BACKWARD_KERNEL_OPT
    echo "End test $file"
done

endTime=`date +%Y%m%d-%H:%M:%S`
endTime_s=`date +%s`
sumTime=$[ $endTime_s-$startTime_s ]
timeMinu=$[ $sumTime / 60 ]
echo "===== $startTime -----> $endTime Total run $timeMinu minutes"

if [[ $2 != "" ]]; then
    PYTORCH_ROOT="$(cd $(dirname $0);pwd)/.."
    source ${PYTORCH_ROOT}/maca_tools/utils.sh
    xml_date=`date +%Y%m%d-%H-%M-%S`
    xml_path=$2/${xml_date}-pytest.xml
    generate_xml ${xml_path} ${res// /_} $PASS $ERR
fi

if [[ ${ERR} != 0 ]];then
    echo "****** Below test failed. ******"
    for((i=0;i<${#err_files[@]};i++)); do
        echo ${err_files[$i]}
    done
    exit 1
else
    echo "****** All test passed. ******"
    exit 0
fi
