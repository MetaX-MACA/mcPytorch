#include <ATen/Config.h>

#include <ATen/Context.h>

#include <c10/core/CPUAllocator.h>

#include <algorithm>
#include <cctype>
#include <string>
#include <stdexcept>

#include <ATen/cpu/FlushDenormal.h>

#ifdef USE_FBGEMM
#include <fbgemm/Fbgemm.h>
#endif // USE_FBGEMM
#if defined(__aarch64__) && !defined(C10_MOBILE)
#include <cpuinfo.h>
#endif

namespace at {

namespace maca {

#define DEFINE_MACA_ENV_API_NE(api_name, env_name) \
bool api_name() {                                  \
  static char const* temp = std::getenv(env_name); \
  return temp != nullptr;                          \
}

#define DEFINE_MACA_ENV_API_E(api_name, env_name) \
bool api_name() {                                  \
  static char const* temp = std::getenv(env_name); \
  return temp == nullptr;                          \
}
DEFINE_MACA_ENV_API_NE(get_maca_allow_cuda_cudnn_tf32, "PYTORCH_ALLOW_CUDA_CUDNN_TF32")
DEFINE_MACA_ENV_API_NE(get_maca_check_anomaly_inf, "PYTORCH_CHECK_ANOMALY_INF")
DEFINE_MACA_ENV_API_NE(get_maca_default_nchw_memory_format, "PYTORCH_DEFAULT_NCHW")
DEFINE_MACA_ENV_API_NE(get_maca_default_ndhwc_memory_format, "PYTORCH_DEFAULT_NDHWC")
DEFINE_MACA_ENV_API_NE(get_maca_default_nlc_memory_format, "PYTORCH_DEFAULT_NLC")
DEFINE_MACA_ENV_API_NE(get_maca_disable_all_to_one_reduce_kernel, "PYTORCH_DISABLE_ALL_TO_ONE_REDUCE")
DEFINE_MACA_ENV_API_NE(get_maca_disable_avgpool_backward_opt, "PYTORCH_DISABLE_AVGPOOL_BACKWARD_OPT")
DEFINE_MACA_ENV_API_NE(get_maca_disable_batch_norm_backward_reduce_channels_last_kernel, "PYTORCH_DISABLE_BATCHNORM_BACKWARD_REDUCE_CHANNELS_LAST_KERNEL")
DEFINE_MACA_ENV_API_NE(get_maca_disable_cat, "PYTORCH_DISABLE_CAT")
DEFINE_MACA_ENV_API_NE(get_maca_disable_continuous_reduce_kernel, "PYTORCH_DISABLE_CONTINUOUS_REDUCE")
DEFINE_MACA_ENV_API_NE(get_maca_disable_continuous_reduce_transpose_kernel, "PYTORCH_DISABLE_CONTINUOUS_REDUCE_TRANSPOSE")
DEFINE_MACA_ENV_API_NE(get_maca_disable_continuous_reduce_kernel_use_offset_calculator, "PYTORCH_DISABLE_CONTINUOUS_REDUCE_USE_OFFSET_CALCULATOR")
DEFINE_MACA_ENV_API_NE(get_maca_disable_element_template_shape, "PYTORCH_DISABLE_ELEMENT_TEMPLATE_SHAPE")
DEFINE_MACA_ENV_API_NE(get_maca_disable_elementwise_1_2_broadcast_kernel, "PYTORCH_DISABLE_ELEMENTWISE_1_2_BROADCAST")
DEFINE_MACA_ENV_API_NE(get_maca_disable_elementwise_2_2_cast_broadcast_kernel, "PYTORCH_DISABLE_ELEMENTWISE_2_2_CAST_BROADCAST")
DEFINE_MACA_ENV_API_NE(get_maca_disable_elementwise_2_2_cast_template_kernel, "PYTORCH_DISABLE_ELEMENTWISE_2_2_CAST_TEMPLATE")
DEFINE_MACA_ENV_API_NE(get_maca_disable_elementwise_3_1_broadcast_kernel, "PYTORCH_DISABLE_ELEMENTWISE_3_1_BROADCAST")
DEFINE_MACA_ENV_API_NE(get_maca_disable_elementwise_3_1_copy_opt_kernel, "PYTORCH_DISABLE_ELEMENTWISE_3_1_COPY_OPT")
DEFINE_MACA_ENV_API_NE(get_maca_disable_elementwise_3_1_dim0_contiguous, "PYTORCH_DISABLE_ELEMENTWISE_3_1_DIM0_CONTIGUOUS")
DEFINE_MACA_ENV_API_NE(get_maca_disable_elementwise_3_1_transpose_half_copy_kernel, "PYTORCH_DISABLE_ELEMENTWISE_3_1_TRANSPOSE_HALF_COPY")
DEFINE_MACA_ENV_API_NE(get_maca_disable_elementwise_3_1_transpose02_half_copy_kernel, "PYTORCH_DISABLE_ELEMENTWISE_3_1_TRANSPOSE02_HALF_COPY")
DEFINE_MACA_ENV_API_NE(get_maca_disable_elementwise_3_2_arity2_transpose_kernel, "PYTORCH_DISABLE_ELEMENTWISE_3_2_ARITY2_TRANSPOSE")
DEFINE_MACA_ENV_API_NE(get_maca_disable_elementwise_3_2_arity2_transpose_dim02_kernel, "PYTORCH_DISABLE_ELEMENTWISE_3_2_ARITY2_TRANSPOSE_DIM02")
DEFINE_MACA_ENV_API_NE(get_maca_disable_elementwise_3_2_broadcast_arg0_dim2_arg1_dim0_kernel, "PYTORCH_DISABLE_ELEMENTWISE_3_2_BROADCAST_ARG0_DIM2_ARG1_DIM0")
DEFINE_MACA_ENV_API_NE(get_maca_disable_elementwise_3_2_broadcast_dim1_contiguous_kernel, "PYTORCH_DISABLE_ELEMENTWISE_3_2_BROADCAST_DIM1_CONTIGUOUS")
DEFINE_MACA_ENV_API_NE(get_maca_disable_elementwise_3_2_broadcast_dim1_uncontiguous_kernel, "PYTORCH_DISABLE_ELEMENTWISE_3_2_BROADCAST_DIM1_UNCONTIGUOUS")
DEFINE_MACA_ENV_API_NE(get_maca_disable_elementwise_3_2_broadcast_dim2_arg0_contiguous_kernel, "PYTORCH_DISABLE_ELEMENTWISE_3_2_BROADCAST_DIM2_ARG0_CONTIGUOUS")
DEFINE_MACA_ENV_API_NE(get_maca_disable_elementwise_3_2_broadcast_dim2_contiguous_kernel, "PYTORCH_DISABLE_ELEMENTWISE_3_2_BROADCAST_DIM2_CONTIGUOUS")
DEFINE_MACA_ENV_API_NE(get_maca_disable_elementwise_3_2_broadcast_dim2_uncontiguous_kernel, "PYTORCH_DISABLE_ELEMENTWISE_3_2_BROADCAST_DIM2_UNCONTIGUOUS")
DEFINE_MACA_ENV_API_NE(get_maca_disable_elementwise_3_2_broadcast_dim0_kernel, "PYTORCH_DISABLE_ELEMENTWISE_3_2_BROADCAST_DIM0")
DEFINE_MACA_ENV_API_NE(get_maca_disable_elementwise_3_2_dim0_contiguous_kernel, "PYTORCH_DISABLE_ELEMENTWISE_3_2_DIM0_CONTIGUOUS")
DEFINE_MACA_ENV_API_NE(get_maca_disable_elementwise_3_2_cast_broadcast_kernel, "PYTORCH_DISABLE_ELEMENTWISE_3_2_CAST_BROADCAST")
DEFINE_MACA_ENV_API_NE(get_maca_disable_elementwise_3_2_cast_broadcast_dim2_kernel, "PYTORCH_DISABLE_ELEMENTWISE_3_2_CAST_BROADCAST_DIM2")
DEFINE_MACA_ENV_API_NE(get_maca_disable_elementwise_3_2_dim0_contiguous_arg1_dim1_broadcast_kernel,"PYTORCH_DISABLE_ELEMENTWISE_3_2_DIM0_CONTIGUOUS_ARG1_DIM1_BROADCAST")
DEFINE_MACA_ENV_API_NE(get_maca_disable_elementwise_3_3_broadcast_kernel, "PYTORCH_DISABLE_ELEMENTWISE_3_3_BROADCAST")
DEFINE_MACA_ENV_API_NE(get_maca_disable_elementwise_4_1_copy_opt_kernel, "PYTORCH_DISABLE_ELEMENTWISE_4_1_COPY_OPT")
DEFINE_MACA_ENV_API_NE(get_maca_disable_elementwise_4_1_transpose12_copy_kernel, "PYTORCH_DISABLE_ELEMENTWISE_4_1_TRANSPOSE12_COPY")
DEFINE_MACA_ENV_API_NE(get_maca_disable_elementwise_4_2_broadcast_arg0_dim2_arg1_dim0_kernel, "PYTORCH_DISABLE_ELEMENTWISE_4_2_BROADCAST_ARG0_DIM2_ARG1_DIM0")
DEFINE_MACA_ENV_API_NE(get_maca_disable_elementwise_4_2_opt_kernel, "PYTORCH_DISABLE_ELEMENTWISE_4_2_OPT")
DEFINE_MACA_ENV_API_NE(get_maca_disable_elementwise_4_2_template_kernel, "PYTORCH_DISABLE_ELEMENTWISE_4_2_TEMPLATE_KERNEL")
DEFINE_MACA_ENV_API_NE(get_maca_disable_elementwise_4_2_uncontiguous_kernel, "PYTORCH_DISABLE_ELEMENTWISE_4_2_UNCONTIGUOUS_KERNEL")
DEFINE_MACA_ENV_API_NE(get_maca_disable_elementwise_arity2_cast_unroll, "PYTORCH_DISABLE_ELEMENTWISE_ARITY2_CAST_UNROLL")
DEFINE_MACA_ENV_API_NE(get_maca_disable_elementwise_kernel_1_1_broadcast, "PYTORCH_DISABLE_ELEMENTWISE_1_1_BROADCAST")
DEFINE_MACA_ENV_API_NE(get_maca_disable_elementwise_kernel_2_1_broadcast, "PYTORCH_DISABLE_ELEMENTWISE_2_1_BROADCAST")
DEFINE_MACA_ENV_API_NE(get_maca_disable_elementwise_kernel_2_1_dim0_contiguous, "PYTORCH_DISABLE_ELEMENTWISE_2_1_DIM0_CONTIGUOUS")
DEFINE_MACA_ENV_API_NE(get_maca_disable_elementwise_kernel_2_1_cp_cast_dim0_contiguous, "PYTORCH_DISABLE_ELEMENTWISE_2_1_CP_CAST_DIM0_CONTIGUOUS")
DEFINE_MACA_ENV_API_NE(get_maca_disable_elementwise_kernel_2_1_input_lowdim_contiuous, "PYTORCH_DISABLE_ELEMENTWISE_2_1_INPUT_LOWDIM_CONTIUOUS")
DEFINE_MACA_ENV_API_NE(get_maca_disable_elementwise_kernel_2_1_transpose_uncontiguous, "PYTORCH_DISABLE_ELEMENTWISE_2_1_TRANSPOSE_UNCONTIGUSOU")
DEFINE_MACA_ENV_API_NE(get_maca_disable_elementwise_kernel_2_2_align, "PYTORCH_DISABLE_ELEMENTWISE_2_2_ALIGN")
DEFINE_MACA_ENV_API_NE(get_maca_disable_elementwise_kernel_2_2_arity2_transpose, "PYTORCH_DISABLE_ELEMENTWISE_2_2_ARITY2_TRANPOSE")
DEFINE_MACA_ENV_API_NE(get_maca_disable_elementwise_kernel_2_2_broadcast, "PYTORCH_DISABLE_ELEMENTWISE_2_2_BROADCAST")
DEFINE_MACA_ENV_API_NE(get_maca_disable_elementwise_kernel_2_2_broadcast_arity1_dim0, "PYTORCH_DISABLE_ELEMENTWISE_2_2_BROADCAST_ARITY1_DIM0")
DEFINE_MACA_ENV_API_NE(get_maca_disable_elementwise_kernel_2_2_template, "PYTORCH_DISABLE_ELEMENTWISE_2_2_TEMPLATE")
DEFINE_MACA_ENV_API_NE(get_maca_disable_elementwise_kernel_4_1_input_lowdim_contiuous, "PYTORCH_DISABLE_ELEMENTWISE_4_1_INPUT_LOWDIM_CONTIUOUS")
DEFINE_MACA_ENV_API_NE(get_maca_disable_elementwise_kernel_4_2_cast_broadcast, "PYTORCH_DISABLE_ELEMENTWISE_4_2_CAST_BROADCAST")
DEFINE_MACA_ENV_API_NE(get_maca_enable_elementwise_kernel_highdim, "PYTORCH_ENABLE_ELEMENTWISE_HIGHDIM")
DEFINE_MACA_ENV_API_NE(get_maca_disable_elementwise_multi_outputs, "PYTORCH_DISABLE_ELEMENTWISE_MULTI_OUTPUTS")
DEFINE_MACA_ENV_API_NE(get_maca_disable_fast_fused_dropout_kernel, "PYTORCH_DISABLE_FAST_FUSED_DROPOUT")
DEFINE_MACA_ENV_API_NE(get_maca_disable_expontial_opt, "PYTORCH_DISABLE_EXPONTIAL_OPT")
DEFINE_MACA_ENV_API_NE(get_maca_disable_gatherTopK_opt, "PYTORCH_DISABLE_GATHERTOPK_OPT")
DEFINE_MACA_ENV_API_NE(get_maca_disable_group_norm_backward_opt_kernel, "PYTORCH_DISABLE_GROUP_NORM_BACKWARD_OPT_KERNEL")
DEFINE_MACA_ENV_API_NE(get_maca_disable_group_norm_fused_kernel, "PYTORCH_DISABLE_GROUP_NORM_FUSED_KERNEL")
DEFINE_MACA_ENV_API_NE(get_maca_disable_group_norm_opt_kernel, "PYTORCH_DISABLE_GROUP_NORM_OPT_KERNEL")
DEFINE_MACA_ENV_API_NE(get_maca_disable_imcontinuous_reduce_kernel, "PYTORCH_DISABLE_IMCONTINUOUS_REDUCE")
DEFINE_MACA_ENV_API_NE(get_maca_disable_index_elementwise_1_2_broadcast_kernel, "PYTORCH_DISABLE_INDEX_ELEMENTWISE_1_2_BROADCAST")
DEFINE_MACA_ENV_API_NE(get_maca_disable_max_pool_nhwc_opt, "PYTORCH_DISABLE_MAXPOOL_NHWC_OPT")
DEFINE_MACA_ENV_API_NE(get_maca_disable_mcdnn_bias_fusion, "PYTORCH_DISABLE_MCDNN_BIAS_FUSION")
DEFINE_MACA_ENV_API_NE(get_maca_disable_nllloss_bwd_2d, "PYTORCH_DISABLE_NLLLOSS_BWD_2D")
DEFINE_MACA_ENV_API_NE(get_maca_disable_nllloss_fwd_2d, "PYTORCH_DISABLE_NLLLOSS_FWD_2D")
DEFINE_MACA_ENV_API_NE(get_maca_enable_print_nllloss, "PYTORCH_ENABLE_PRINT_NLLLOSS");
DEFINE_MACA_ENV_API_NE(get_maca_disable_openblas, "PYTORCH_DISABLE_OPENBLAS")
DEFINE_MACA_ENV_API_NE(get_maca_disable_opt_cat_for_dim5, "PYTORCH_DISABLE_OPT_CAT_FOR_DIM5")
DEFINE_MACA_ENV_API_NE(get_maca_disable_opt_cat_for_not_allcontiguous, "PYTORCH_DISABLE_OPT_CAT_FOR_NOT_ALLCONTIGUOUS")
DEFINE_MACA_ENV_API_NE(get_maca_disable_opt_cat_for_not_allsamedtype, "PYTORCH_DISABLE_OPT_CAT_FOR_NOT_ALLSAMETYPE")
DEFINE_MACA_ENV_API_NE(get_maca_disable_opt_indexing, "PYTORCH_DISABLE_OPT_INDEXING")
DEFINE_MACA_ENV_API_NE(get_maca_disable_opt_layernorm_gammabeta, "PYTORCH_DISABLE_OPT_LAYERNORM_GAMMABETA")
DEFINE_MACA_ENV_API_NE(get_maca_disable_opt_layernorm_grad_input, "PYTORCH_DISABLE_OPT_LAYERNORM_GRAD_INPUT")
DEFINE_MACA_ENV_API_NE(get_maca_disable_opt_roll, "PYTORCH_DISABLE_OPT_ROLL")
DEFINE_MACA_ENV_API_NE(get_maca_disable_reduce_split_warp, "PYTORCH_DISABLE_SPLIT_WARP_REDUCE")
DEFINE_MACA_ENV_API_NE(get_maca_disable_scatter_gather_dim2_opt, "PYTORCH_DISABLE_SCATTER_GATHER_DIM2_OPT")
DEFINE_MACA_ENV_API_NE(get_maca_disable_scatter_gather_dim2_opt_assign, "PYTORCH_DISABLE_SCATTER_GATHER_DIM2_OPT_ASSIGN")
DEFINE_MACA_ENV_API_NE(get_maca_disable_scatter_gather_without_assert, "PYTORCH_DISABLE_SCATTER_GATHER_WITHOUT_ASSERT")
DEFINE_MACA_ENV_API_NE(get_maca_disable_sigmoid_opt, "PYTORCH_DISABLE_SIGMOID_OPT")
DEFINE_MACA_ENV_API_NE(get_maca_disable_softmax_backward, "PYTORCH_DISABLE_SOFTMAX_BACKWARD")
DEFINE_MACA_ENV_API_NE(get_maca_disable_softmax_opt, "PYTORCH_DISABLE_SOFTMAX_OPT")
DEFINE_MACA_ENV_API_NE(get_maca_disable_threshold_opt, "PYTORCH_DISABLE_THRESHOLD_OPT")
DEFINE_MACA_ENV_API_NE(get_maca_disable_triu_tril_opt, "PYTORCH_DISABLE_TRIU_TRIL_OPT")
DEFINE_MACA_ENV_API_NE(get_maca_disable_unroll_float_opt, "PYTORCH_DISABLE_UNROLL_FLOAT_OPT")
DEFINE_MACA_ENV_API_NE(get_maca_disable_upsample_bicubic2d_opt, "PYTORCH_DISABLE_UPSAMPLE_BICUBIC2D_OPT")
DEFINE_MACA_ENV_API_NE(get_maca_disable_upsample_trilinear3d_opt, "PYTORCH_DISABLE_UPSAMPLE_TRILINEAR3D_OPT")
DEFINE_MACA_ENV_API_NE(get_maca_disable_vectorized_elementwise_nullary_opt, "PYTORCH_DISABLE_VECTORIZED_ELEMENTWISE_NULLARY_OPT")
DEFINE_MACA_ENV_API_NE(get_maca_disable_vectorized_elementwise_unary_opt, "PYTORCH_DISABLE_VECTORIZED_ELEMENTWISE_UNARY_OPT")
DEFINE_MACA_ENV_API_NE(get_maca_disable_vectorized_layernorm_opt, "PYTORCH_DISABLE_VECTORIZED_LAYERNORM_OPT")
DEFINE_MACA_ENV_API_NE(get_maca_disable_blas_workspace_default_32m, "PYTORCH_DISABLE_BLAS_WORKSPACE_DEFAULT_32M")
DEFINE_MACA_ENV_API_NE(get_maca_elementwise_multi_outputs_shape, "PYTORCH_ELEMENTWISE_MULTI_OUTPUTS_SHAPE")
DEFINE_MACA_ENV_API_NE(get_maca_enable_avgpool_backward_info, "PYTORCH_ENABLE_AVGPOOL_BACKWARD_INFO")
DEFINE_MACA_ENV_API_NE(get_maca_enable_conv2d_nhwc, "PYTORCH_ENABLE_CONV2D_NHWC")
DEFINE_MACA_ENV_API_NE(get_maca_enable_conv_nhwc_c_1, "PYTORCH_ENABLE_CONV_NHWC_C_1")
DEFINE_MACA_ENV_API_NE(get_maca_disable_depwise_conv_opt, "PYTORCH_DISABLE_DEPWISE_CONV_OPT")
DEFINE_MACA_ENV_API_NE(get_maca_enable_cpu_topk_stable_sort, "PYTORCH_ENABLE_CPU_TOPK_STABLE_SORT")
DEFINE_MACA_ENV_API_NE(get_maca_enable_elementwise_kernel_1_1_dilation, "PYTORCH_ENABLE_ELEMENTWISE_1_1_DILATION")
DEFINE_MACA_ENV_API_NE(get_maca_disable_elementwise_kernel_5_1_lowdim_contiguous, "PYTORCH_DISABLE_ELEMENTWISE_5_1_LOWDIM_CONTIGUOUS")
DEFINE_MACA_ENV_API_NE(get_maca_enable_elementwise_kernel_info, "PYTORCH_ENABLE_ELEMENTWISE_KERNEL_INFO")
DEFINE_MACA_ENV_API_NE(get_maca_enable_elementwise_kernel_n_1_dim0_pad, "PYTORCH_ENABLE_ELEMENTWISE_N_1_DIM0_PAD")
DEFINE_MACA_ENV_API_NE(get_maca_disable_4_1_dim0_contiguous, "PYTORCH_DISALBE_4_1_DIM0_CONTIGUOUS")
DEFINE_MACA_ENV_API_NE(get_maca_enable_elementwise_without_assert, "PYTORCH_ENABLE_ELEMENTWISE_WITHOUT_ASSERT")
DEFINE_MACA_ENV_API_NE(get_maca_enable_fast_multitensorapply, "PYTORCH_ENABLE_FAST_MULTITENSORAPPLY")
DEFINE_MACA_ENV_API_NE(get_maca_enable_fast_multitensorapply_fused, "PYTORCH_ENABLE_FAST_MULTITENSORAPPLY_FUSED")
DEFINE_MACA_ENV_API_NE(get_maca_enable_gemm_api_info, "PYTORCH_ENABLE_GEMM_API_INFO")
DEFINE_MACA_ENV_API_NE(get_maca_enable_imcontinuous_reduce_kernel_v1, "PYTORCH_ENABLE_IMCONTINUOUS_REDUCE_V1")
DEFINE_MACA_ENV_API_NE(get_maca_enable_indexing_assert_kernel, "PYTORCH_ENABLE_INDEXING_ASSERT")
DEFINE_MACA_ENV_API_NE(get_maca_enable_indexing_backward_kernel_opt, "PYTORCH_ENABLE_INDEXING_BACKWARD_KERNEL_OPT")
DEFINE_MACA_ENV_API_NE(get_maca_disable_indexing_backward_kernel_opt1, "PYTORCH_DISABLE_INDEXING_BACKWARD_KERNEL_OPT1")
DEFINE_MACA_ENV_API_NE(get_maca_enable_print_indexing_backward_kernel, "PYTORCH_ENABLE_PRINT_INDEXING_BACKWARD_KERNEL")
DEFINE_MACA_ENV_API_NE(get_maca_enable_layernorm_kernel_info, "PYTORCH_ENABLE_LAYERNORM_KERNEL_INFO")
DEFINE_MACA_ENV_API_NE(get_maca_enable_memcpy_replace_memset_reduce_kernel, "PYTORCH_REDUCE_ENABLE_MEMCPY_REPLACE_MEMSET")
DEFINE_MACA_ENV_API_NE(get_maca_enable_mha_fusion, "PYTORCH_ENABLE_MHA_FUSION")
DEFINE_MACA_ENV_API_NE(get_maca_enable_opt_batchnorm_nhwc, "PYTORCH_ENABLE_OPT_BATCHNORM_NHWC")
DEFINE_MACA_ENV_API_NE(get_maca_enable_original_vectorized_layernorm, "PYTORCH_ENABLE_ORIGINAL_VECTORIZED_LAYERNORM")
DEFINE_MACA_ENV_API_NE(get_maca_enable_print_batchnorm_info, "PYTORCH_ENABLE_PRINT_BATCHNORM_INFO")
DEFINE_MACA_ENV_API_NE(get_maca_enable_print_conv_info, "PYTORCH_ENABLE_PRINT_CONV_INFO")
DEFINE_MACA_ENV_API_NE(get_maca_enable_print_indexing_info, "PYTORCH_ENABLE_PRINT_INDEXING_INFO")
DEFINE_MACA_ENV_API_NE(get_maca_enable_print_reduce_kernel, "PYTORCH_REDUCE_ENABLE_PRINT")
DEFINE_MACA_ENV_API_NE(get_maca_enable_print_upsample2d_info, "PYTORCH_ENABLE_PRINT_UPSAMPLE2D_INFO")
DEFINE_MACA_ENV_API_NE(get_maca_enable_print_upsample3d_info, "PYTORCH_ENABLE_PRINT_UPSAMPLE3D_INFO")
DEFINE_MACA_ENV_API_NE(get_maca_print_cat_shape, "PYTORCH_PRINT_CAT_SHAPE")
DEFINE_MACA_ENV_API_NE(get_maca_print_cub_info, "PYTORCH_PRINT_CUB_INFO")
DEFINE_MACA_ENV_API_NE(get_maca_print_group_norm_shape, "PYTORCH_PRINT_GROUP_NORM_SHAPE")
DEFINE_MACA_ENV_API_NE(get_maca_print_mha_shape, "PYTORCH_PRINT_MHA_SHAPE")
DEFINE_MACA_ENV_API_NE(get_maca_print_softmax_shape, "PYTORCH_PRINT_SOFTMAX_SHAPE")
DEFINE_MACA_ENV_API_NE(get_maca_push_conv_backend_grad_bias_on_cpu, "PYTORCH_ENABLE_PUSH_CONV_BACKEND_GRAD_BIAS_ON_CPU")
DEFINE_MACA_ENV_API_NE(get_maca_disable_element_kernel_opt_tile, "PYTORCH_DISABLE_ELEMENT_KERNEL_OPT_TILE")
DEFINE_MACA_ENV_API_NE(get_maca_disable_element_kernel_cast_opt_tile, "PYTORCH_DISABLE_ELEMENT_KERNEL_CAST_OPT_TILE")
DEFINE_MACA_ENV_API_NE(get_maca_enable_scatter_pw_opt, "PYTORCH_ENABLE_SCATTER_PW_OPT")
DEFINE_MACA_ENV_API_NE(get_maca_disable_alltoall_opt, "PYTORCH_DISABLE_ALLTOALL_OPT")
#undef DEFINE_MACA_ENV_API_NE
#undef DEFINE_MACA_ENV_API_E
}
Context::Context() = default;

// TODO: This could be bad juju if someone calls globalContext() in the
// destructor of an object with static lifetime.
Context& globalContext() {
  static Context globalContext_;
  return globalContext_;
}

// NB: This method is *purely* whether or not a user requested
// that CuDNN was enabled, it doesn't actually say anything about
// whether or not CuDNN is actually usable.
bool Context::userEnabledCuDNN() const {
  return enabled_cudnn;
}

void Context::setUserEnabledCuDNN(bool e) {
  enabled_cudnn = e;
}

bool Context::userEnabledMkldnn() const {
  return enabled_mkldnn;
}

void Context::setUserEnabledMkldnn(bool e) {
  enabled_mkldnn = e;
}

bool Context::deterministicCuDNN() const {
  return deterministic_cudnn;
}

void Context::setDeterministicCuDNN(bool b) {
  deterministic_cudnn = b;
}

bool Context::deterministicAlgorithms() const {
  return _deterministic_algorithms;
}

bool Context::deterministicAlgorithmsWarnOnly() const {
  return _deterministic_algorithms_warn_only;
}

void Context::setDeterministicAlgorithms(bool b, bool warn_only=false) {
  _deterministic_algorithms = b;
  _deterministic_algorithms_warn_only = warn_only;
}

bool Context::deterministicFillUninitializedMemory() const {
  return _deterministic_fill_uninitialized_memory;
}

void Context::setDeterministicFillUninitializedMemory(bool b) {
  _deterministic_fill_uninitialized_memory = b;
}

void Context::alertNotDeterministic(c10::string_view const& caller) {
  if (globalContext().deterministicAlgorithms()) {
    if (globalContext().deterministicAlgorithmsWarnOnly()) {
      TORCH_WARN(
        caller, " does not have a deterministic implementation, but you set "
        "'torch.use_deterministic_algorithms(True, warn_only=True)'. "
        "You can file an issue at https://github.com/pytorch/pytorch/issues "
        "to help us prioritize adding deterministic support for this operation.");
    } else {
      TORCH_CHECK(false,
        caller, " does not have a deterministic implementation, but you set "
        "'torch.use_deterministic_algorithms(True)'. You can turn off "
        "determinism just for this operation, or you can use the "
        "'warn_only=True' option, if that's acceptable for your application. "
        "You can also file an issue at https://github.com/pytorch/pytorch/issues "
        "to help us prioritize adding deterministic support for this operation.");
    }
  }
}

bool Context::userEnabledNNPACK() const {
  return enabled_nnpack;
}

void Context::setUserEnabledNNPACK(bool e) {
  enabled_nnpack = e;
}

bool Context::allowTF32CuDNN() const {
  return allow_tf32_cudnn;
}

void Context::setAllowTF32CuDNN(bool b) {
  allow_tf32_cudnn = b;
}

bool Context::userEnabledFlashSDP() const {
  return enabled_flashSDP;
}

void Context::setSDPUseFlash(bool e) {
  enabled_flashSDP = e;
}

bool Context::userEnabledMemEfficientSDP() const {
  return enabled_mem_efficientSDP;
}

void Context::setSDPUseMemEfficient(bool e) {
  enabled_mem_efficientSDP = e;
}

bool Context::userEnabledMathSDP() const {
  return enabled_mathSDP;
}

void Context::setSDPUseMath(bool e) {
  enabled_mathSDP = e;
}

bool Context::userEnabledCuDNNSDP() const {
  return enabled_cudnnSDP;
}

void Context::setSDPUseCuDNN(bool e) {
  enabled_cudnnSDP = e;
}


// NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
static const char cublas_config_var_name[] = "CUBLAS_WORKSPACE_CONFIG";
// NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
static const char* const cublas_deterministic_configs[] = { ":4096:8", ":16:8" };

bool Context::checkCuBLASConfigDeterministic() {
  bool cublas_config_deterministic = true;
  // If using CUDA 10.2 or greater, need to make sure CuBLAS workspace config
  // is set to deterministic setting
  if (hasCUDART() && (versionCUDART() >= 10020)) {
    char* workspace_config = std::getenv(cublas_config_var_name);
    cublas_config_deterministic = (workspace_config != nullptr) && (
      (strcmp(workspace_config, cublas_deterministic_configs[0]) == 0)
      || (strcmp(workspace_config, cublas_deterministic_configs[1]) == 0)
    );
  }
  return cublas_config_deterministic;
}

void Context::alertCuBLASConfigNotDeterministic() const {
  static bool cublas_config_deterministic = checkCuBLASConfigDeterministic();
  if (C10_LIKELY(!deterministicAlgorithms() || cublas_config_deterministic)) {
    return;
  }

  auto msg = c10::str(
    "Deterministic behavior was enabled with either `torch.use_deterministic_algorithms(True)` or ",
    "`at::Context::setDeterministicAlgorithms(true)`, but this operation is not deterministic because ",
    "it uses CuBLAS and you have CUDA >= 10.2. To enable deterministic behavior in this ",
    "case, you must set an environment variable before running your PyTorch application: ",
    cublas_config_var_name, "=", cublas_deterministic_configs[0], " or ",
    cublas_config_var_name, "=", cublas_deterministic_configs[1], ". For more information, go to ",
    "https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility"
  );

  if (deterministicAlgorithmsWarnOnly()) {
    TORCH_WARN(msg);
  } else {
    TORCH_CHECK(false, msg);
  }
}

bool Context::benchmarkCuDNN() const {
  return benchmark_cudnn;
}

void Context::setBenchmarkCuDNN(bool b) {
  benchmark_cudnn = b;
}

int Context::benchmarkLimitCuDNN() const {
  return benchmark_limit_cudnn;
}

void Context::setBenchmarkLimitCuDNN(int b) {
  benchmark_limit_cudnn = b;
}

bool Context::allowTF32CuBLAS() const {
  return float32_matmul_precision != at::Float32MatmulPrecision::HIGHEST;
}

void Context::setAllowTF32CuBLAS(bool b) {
  float32_matmul_precision = b ? at::Float32MatmulPrecision::HIGH : at::Float32MatmulPrecision::HIGHEST;
}

Float32MatmulPrecision Context::float32MatmulPrecision() const {
  return float32_matmul_precision;
}

void Context::setFloat32MatmulPrecision(Float32MatmulPrecision p) {
  float32_matmul_precision = p;
}

void Context::setFloat32MatmulPrecision(const std::string &s) {
  auto match = [this](const std::string & s_) {
    // TODO: consider if CuDNN field needs to also be set for potential future CuDNN ops like multi-headed attention
    if (s_ == "highest") {
      float32_matmul_precision = at::Float32MatmulPrecision::HIGHEST;
      return true;
    } else if (s_ == "high") {
      float32_matmul_precision = at::Float32MatmulPrecision::HIGH;
      return true;
    } else if (s_ == "medium") {
      float32_matmul_precision = at::Float32MatmulPrecision::MEDIUM;
      return true;
    }
    return false;
  };
  if (match(s)) { return; }
  std::string sl;
  std::transform(s.begin(), s.end(), sl.begin(),
                 [](unsigned char c) -> unsigned char { return std::tolower(c); });
  if (match(sl)) { return; }
  TORCH_WARN(s, " is not one of 'highest', 'high', or 'medium'; the current"
    "setFloat32MatmulPrecision call has no effect.");
}

at::LinalgBackend Context::linalgPreferredBackend() const {
  return linalg_preferred_backend;
}

void Context::setLinalgPreferredBackend(at::LinalgBackend b) {
  linalg_preferred_backend = b;
  TORCH_CHECK((b != at::LinalgBackend::Cusolver) || hasCuSOLVER(),
      "Cannot set preferred backend to cuSOLVER if PyTorch has not been compiled with cuSOLVER.");
  TORCH_CHECK((b != at::LinalgBackend::Magma) || hasMAGMA(),
      "Cannot set preferred backend to MAGMA if PyTorch has not been compiled with MAGMA.");
  if (b != at::LinalgBackend::Default) {
    TORCH_WARN_ONCE(
      "torch.backends.cuda.preferred_linalg_library is an experimental feature. "
      "If you see any error or unexpected behavior when this flag is set "
      "please file an issue on GitHub."
    );
  }
}

at::BlasBackend Context::blasPreferredBackend() const {
  return blas_preferred_backend;
}

void Context::setBlasPreferredBackend(at::BlasBackend b) {
#ifdef _MSC_VER
  TORCH_WARN_ONCE(
    "torch.backends.cuda.preferred_blas_library is an experimental feature. "
    "It is not supported on Windows."
  );
#else
  TORCH_CHECK((b != at::BlasBackend::Cublaslt) || hasCuBLASLt(),
      "Cannot set preferred backend to cuBLASLt if PyTorch has not been compiled with cuBLASLt.");
  if (b != at::BlasBackend::Cublas) {
    TORCH_WARN_ONCE(
      "torch.backends.cuda.preferred_blas_library is an experimental feature. "
      "If you see any error or unexpected behavior when this flag is set "
      "please file an issue on GitHub."
    );
  }
  blas_preferred_backend = b;
#endif
}

bool Context::allowFP16ReductionCuBLAS() const {
  return allow_fp16_reduction_cublas;
}

void Context::setAllowFP16ReductionCuBLAS(bool b) {
  allow_fp16_reduction_cublas = b;
}

bool Context::allowBF16ReductionCuBLAS() const {
  return allow_bf16_reduction_cublas;
}

void Context::setAllowBF16ReductionCuBLAS(bool b) {
  allow_bf16_reduction_cublas = b;
}


bool Context::hasMKL() {
#if AT_MKL_ENABLED()
  return true;
#else
  return false;
#endif
}

bool Context::hasMKLDNN() {
#if AT_MKLDNN_ENABLED()
  return true;
#else
  return false;
#endif
}

bool Context::hasOpenMP() {
#ifdef _OPENMP
  return true;
#else
  return false;
#endif
}

bool Context::hasLAPACK() {
#if AT_BUILD_WITH_LAPACK()
  return true;
#else
  return false;
#endif
}

at::QEngine Context::qEngine() const {
  static auto _quantized_engine = []() {
    at::QEngine qengine = at::kNoQEngine;
#if defined(C10_MOBILE) && defined(USE_PYTORCH_QNNPACK)
    qengine = at::kQNNPACK;
#endif

#if AT_MKLDNN_ENABLED()
    qengine = at::kONEDNN;
#endif

#ifdef USE_FBGEMM
    if (fbgemm::fbgemmSupportedCPU()) {
      /* X86 is enabled if and only if fbgemm is available.
       * It combines goodness of fbgemm and onednn by dispatching.
       * If onednn not available, always dispatch to fbgemm.
       * Make it default qengine for X86 CPU platforms.
      */
      qengine = at::kX86;
    }
#endif
    return qengine;
  }();
  return quantized_engine.value_or(_quantized_engine);
}

void Context::setQEngine(at::QEngine e) {
  const auto& qengines = supportedQEngines();
  if (std::find(qengines.begin(), qengines.end(), e) != qengines.end()) {
    quantized_engine = e;
    return;
  }
  TORCH_CHECK(false, "quantized engine ", toString(e), " is not supported");
}

const std::vector<at::QEngine>& Context::supportedQEngines() {
  static auto supported_qengines = []() {
    std::vector<at::QEngine> engines = {};
    // Engines are listed in priority order: later one wins
    // By default we prefer FBGEMM if we're running on server side
    // QNNPACK on server side has some issue, so we disable it by default.
#ifdef C10_MOBILE
    engines.push_back(at::kNoQEngine);
#ifdef USE_PYTORCH_QNNPACK
    engines.push_back(at::kQNNPACK);
#endif
#else  // C10_MOBILE
#ifdef USE_PYTORCH_QNNPACK
    engines.push_back(at::kQNNPACK);
#endif
    engines.push_back(at::kNoQEngine);
#endif // C10_MOBILE

#if AT_MKLDNN_ENABLED()
    engines.push_back(at::kONEDNN);
#endif

#ifdef USE_FBGEMM
    if (fbgemm::fbgemmSupportedCPU()) {
      engines.push_back(at::kX86);
      // The X86 qengine is available if and only if FBGEMM is available
      engines.push_back(at::kFBGEMM);
    }
#endif

    return engines;
  }();
  return supported_qengines;
}

bool Context::isXNNPACKAvailable() {
#ifdef USE_XNNPACK
  return true;
#else
  return false;
#endif
}

void Context::setCheckSparseTensorInvariants(bool e) {
  enable_sparse_tensor_invariant_checks = e;
}

bool Context::checkSparseTensorInvariants() const {
  return enable_sparse_tensor_invariant_checks;
}

bool Context::releaseWeightsWhenPrepacking() const {
  return release_original_weights;
}

void Context::setReleaseWeightsWhenPrepacking(bool e) {
  release_original_weights = e;
}

bool Context::setFlushDenormal(bool on) {
  return at::cpu::set_flush_denormal(on);
}

Allocator* getCPUAllocator() {
  return c10::GetCPUAllocator();
}

// override_allow_tf32_flag = true
//    means the allow_tf32 flags are overrided and tf32 is force disabled
// override_allow_tf32_flag = false
//    means the original allow_tf32 flags are followed
thread_local bool override_allow_tf32_flag = false;

NoTF32Guard::NoTF32Guard() {
  if (!override_allow_tf32_flag) {
    changed = true;
    override_allow_tf32_flag = true;
  }
}

NoTF32Guard::~NoTF32Guard() {
  if (changed) {
    override_allow_tf32_flag = false;
  }
}

bool NoTF32Guard::should_disable_tf32() {
  return override_allow_tf32_flag;
}

// Ops can query this flag to know they are in the backward pass.
// This information can be used, for example, to select implementations
// with different numerical or performance characteristics.
// See https://pytorch.org/docs/stable/notes/numerical_accuracy.html for details.
thread_local bool rocm_is_backward_pass;

ROCmBackwardPassGuard::ROCmBackwardPassGuard() {
  rocm_is_backward_pass = true;
}

ROCmBackwardPassGuard::~ROCmBackwardPassGuard() {
  rocm_is_backward_pass = false;
}

bool ROCmBackwardPassGuard::is_backward_pass() {
  return rocm_is_backward_pass;
}

bool Context::areVmapFallbackWarningsEnabled() const {
  return display_vmap_fallback_warnings_;
}

void Context::setDisplayVmapFallbackWarnings(bool enabled) {
  display_vmap_fallback_warnings_ = enabled;
}

void Context::setDefaultMobileCPUAllocator() {
  TORCH_CHECK(prev_allocator_ptr_ == nullptr,
      "Already within the scope of another non-default cpu allocator."
      "Cannot set another allocator.");
  // Setting the priority high to make sure no other allocator gets used instead of this.
  prev_allocator_ptr_ = c10::GetCPUAllocator();
  c10::SetCPUAllocator(c10::GetDefaultMobileCPUAllocator(), /*priority*/ 100);
}

void Context::unsetDefaultMobileCPUAllocator() {
  TORCH_CHECK(prev_allocator_ptr_ != nullptr,
      "setDefaultMobileCPUAllocator must have been called "
      "before unsetDefaultMobileCPUAllocator.");
  // Setting the priority high to make sure no other allocator gets used instead of this.
  c10::SetCPUAllocator(prev_allocator_ptr_ , /*priority*/ 100);
  prev_allocator_ptr_ = nullptr;
}

bool Context::allowFP16ReductionCPU() const {
  return allow_fp16_reduction_cpu;
}

void Context::setAllowFP16ReductionCPU(bool b) {
  if ( b && !allow_fp16_reduction_cpu) {
    // Check that CPU supports fp16 reductions
#if defined(__aarch64__) && !defined(C10_MOBILE)
    if (!cpuinfo_initialize() || !cpuinfo_has_arm_fp16_arith())
#else
    if (true)
#endif
      throw std::runtime_error("Float16 arithmetic is not supported by the CPU!");
  }
  allow_fp16_reduction_cpu = b;
}
} // namespace at
