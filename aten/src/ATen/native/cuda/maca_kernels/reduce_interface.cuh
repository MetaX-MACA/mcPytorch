#include <ATen/native/cuda/maca_kernels/reduce_continuous_kernels.cuh>
#include <ATen/native/cuda/maca_kernels/reduce_imcontinuous_kernels.cuh>
#include <ATen/native/cuda/maca_kernels/reduce_no_struct.cuh>
#include <ATen/native/cuda/maca_kernels/reduce_utils.cuh>
#include <iostream>

namespace at { namespace native {
struct ReduceConfig;
enum ReduceKernelType;

template <typename scalar_t, typename out_scalar_t, typename arg_t, int vt0>
static ReduceKernelType get_reduce_kernel_type(const TensorIterator& iter, bool is_not_over_i32_scope) {
  if (maca_likely(!at::maca::get_maca_disable_continuous_reduce_kernel()) && is_launch_continuous_reduce_kernel(iter, sizeof(scalar_t), sizeof(out_scalar_t))) {
    if (is_ndim_3_reduce_dims_2_case(iter, sizeof(scalar_t), sizeof(out_scalar_t))) {
      return ReduceContinuousMultiReduceDimsKernel;
    } else {
      return ReduceContinuousKernel;
    } 
  } else if (maca_likely(!at::maca::get_maca_disable_imcontinuous_reduce_kernel()) && is_not_over_i32_scope && is_launch_imcontinuous_reduce_kernel(iter, sizeof(scalar_t), sizeof(out_scalar_t))) {
    return ReduceImcontinuousKernel;
  } else {
    ReduceConfigCUDA config = setReduceConfigCUDA<arg_t, scalar_t, vt0>(iter);
    if (!config.should_global_reduce()) {
        return ReduceNoStructKernel;
    } else {
        return ReduceOriginKernel;
    }  
  }
}

template <typename scalar_t, typename out_scalar_t, typename arg_t, int vt0>
static bool is_launch_reduce_maca_kernel(const TensorIterator& tensor_iterator, bool is_over_i32_scope) {
  return get_reduce_kernel_type<scalar_t, out_scalar_t, arg_t, vt0>(tensor_iterator, is_over_i32_scope) != ReduceOriginKernel;
}

template <typename scalar_t, typename out_scalar_t, typename arg_t, int vt0, int max_num_threads, typename R>
static void launch_reduce_maca_kernel(const TensorIterator& tensor_iterator, const R& reduction, bool is_over_i32_scope) {
  ReduceKernelType reduce_kernel_type = get_reduce_kernel_type<scalar_t, out_scalar_t, arg_t, vt0>(tensor_iterator, is_over_i32_scope);
  ReduceConfigMaca config_maca = setReduceConfigMaca<arg_t, scalar_t, out_scalar_t, vt0>(tensor_iterator, reduce_kernel_type);
  if (maca_unlikely(at::maca::get_maca_enable_print_reduce_kernel())) {
    if (reduce_kernel_type == ReduceContinuousKernel ||reduce_kernel_type == ReduceImcontinuousKernel) {
        std::cout << "config maca: " << config_maca << std::endl;
    } else {
        ReduceConfigCUDA config_cuda = setReduceConfigCUDA<arg_t, scalar_t, vt0>(tensor_iterator);
        std::cout << "config cuda: " << config_cuda << std::endl;
    }
  }
  if (reduce_kernel_type == ReduceContinuousKernel) {
    launch_continuous_reduce_kernel<scalar_t, out_scalar_t, 
          max_num_threads, arg_t>(config_maca, reduction);
  } else if (reduce_kernel_type == ReduceImcontinuousKernel) {
    launch_imcontinuous_reduce_kernel<scalar_t, out_scalar_t, 
          max_num_threads, vt0, arg_t>(config_maca, reduction, tensor_iterator);
  } else if (reduce_kernel_type == ReduceContinuousMultiReduceDimsKernel) {
    launch_continuous_reduce_kernel_multi_reduce_dims<scalar_t, out_scalar_t, 
          max_num_threads, arg_t>(config_maca, reduction, tensor_iterator);
  } else if (reduce_kernel_type == ReduceNoStructKernel) {
    ReduceConfigCUDA config_cuda = setReduceConfigCUDA<arg_t, scalar_t, vt0>(tensor_iterator);
    launch_reduce_kernel_maca_opt<
        arg_t, scalar_t, int64_t, out_scalar_t, vt0,
        max_num_threads>(config_cuda, reduction);
  } else {
    // ReduceOriginKernel is impossible to be called when launch reduce maca kernel
    assert(false);
  }
}

template <typename scalar_t, typename out_scalar_t, typename arg_t>
static void print_reduce_info(
    ReduceKernelType reduce_type,
    const TensorIterator& iter) {
  std::stringstream ss;
  ss << "redcue_type: " << static_cast<typename std::underlying_type<ReduceKernelType>::type>(reduce_type);
  size_t ninputs = iter.ntensors() - 1;
  ss << " input strides: ";
  for (size_t i = 0; i < iter.strides(ninputs).size(); i++) {
    ss << iter.strides(ninputs)[i] << " ";
  }
  size_t noutputs = 0;
  ss << " output strides: ";
  for (size_t i = 0; i < iter.strides(noutputs).size(); i++) {
    ss << iter.strides(noutputs)[i] << " ";
  }
  ss << " shape: ";
  for (size_t i = 0; i < iter.shape().size(); i++) {
    ss << iter.shape()[i] << " ";
  } 
  scalar_t temp1;
  out_scalar_t temp2;
  arg_t temp3;
  ss << " scalar_t: " << typeid(temp1).name() <<  "out_scalar_t: " << typeid(temp2).name() << " arg_t: " << typeid(temp3).name();
  std::cout << ss.str() << std::endl;
}

}} // namespace at::native
