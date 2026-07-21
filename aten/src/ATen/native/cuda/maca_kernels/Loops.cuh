#pragma once

namespace at::native {

template <typename func_t>
void gpu_kernel_maca_arity0(TensorIteratorBase& iter, const func_t& f) {

  for (int arg = 0; arg < iter.ntensors(); arg++) {
    TORCH_INTERNAL_ASSERT(
      iter.device(arg).is_cuda(),
      "argument ", arg, ": expected a CUDA device but found ", iter.device(arg));
  }

  if (iter.numel() == 0) {
    return;
  }

  if (!iter.can_use_32bit_indexing()) {
    for (auto& sub_iter : iter.with_32bit_indexing()) {
      gpu_kernel_maca_arity0(sub_iter, f);
    }
    return;
  }

  gpu_kernel_impl_maca_arity0(iter, f);
}

template <typename func_t>
void gpu_kernel_maca_arity1(TensorIteratorBase& iter, const func_t& f) {

  for (int arg = 0; arg < iter.ntensors(); arg++) {
    TORCH_INTERNAL_ASSERT(
      iter.device(arg).is_cuda(),
      "argument ", arg, ": expected a CUDA device but found ", iter.device(arg));
  }

  if (iter.numel() == 0) {
    return;
  }

  if (!iter.can_use_32bit_indexing()) {
    for (auto& sub_iter : iter.with_32bit_indexing()) {
      gpu_kernel_maca_arity1(sub_iter, f);
    }
    return;
  }

  gpu_kernel_impl_maca_arity1(iter, f);
}

template <typename func_t>
void gpu_kernel_maca_arity2(TensorIteratorBase& iter, const func_t& f) {

  for (int arg = 0; arg < iter.ntensors(); arg++) {
    TORCH_INTERNAL_ASSERT(
      iter.device(arg).is_cuda(),
      "argument ", arg, ": expected a CUDA device but found ", iter.device(arg));
  }

  if (iter.numel() == 0) {
    return;
  }

  if (!iter.can_use_32bit_indexing()) {
    for (auto& sub_iter : iter.with_32bit_indexing()) {
      gpu_kernel_maca_arity2(sub_iter, f);
    }
    return;
  }

  gpu_kernel_impl_maca_arity2(iter, f);
}

template <typename func_t>
void gpu_kernel_maca_arity3(TensorIteratorBase& iter, const func_t& f) {

  for (int arg = 0; arg < iter.ntensors(); arg++) {
    TORCH_INTERNAL_ASSERT(
      iter.device(arg).is_cuda(),
      "argument ", arg, ": expected a CUDA device but found ", iter.device(arg));
  }

  if (iter.numel() == 0) {
    return;
  }

  if (!iter.can_use_32bit_indexing()) {
    for (auto& sub_iter : iter.with_32bit_indexing()) {
      gpu_kernel_maca_arity3(sub_iter, f);
    }
    return;
  }

  gpu_kernel_impl_maca_arity3(iter, f);
}

template <typename func_t>
void gpu_kernel_maca_arity5(TensorIteratorBase& iter, const func_t& f) {

  for (int arg = 0; arg < iter.ntensors(); arg++) {
    TORCH_INTERNAL_ASSERT(
      iter.device(arg).is_cuda(),
      "argument ", arg, ": expected a CUDA device but found ", iter.device(arg));
  }

  if (iter.numel() == 0) {
    return;
  }

  if (!iter.can_use_32bit_indexing()) {
    for (auto& sub_iter : iter.with_32bit_indexing()) {
      gpu_kernel_maca_arity5(sub_iter, f);
    }
    return;
  }

  gpu_kernel_impl_maca_arity5(iter, f);
}

template<typename arg1_t, typename arg2_t, typename return_t, typename func_t>
struct AUnaryFunctor;
template<typename arg1_t, typename arg2_t, typename return_t, typename func_t>
struct BUnaryFunctor;
template <typename arg1_t, typename arg2_t, typename return_t, typename func_t>
struct BinaryFunctor;
template <typename func_t>
void gpu_kernel_with_scalars(TensorIteratorBase& iter, const func_t& f);

template <typename arg1_t, typename arg2_t = arg1_t, typename return_t = arg1_t, typename func_t>
void opmath_gpu_kernel_with_scalars_arity2(TensorIteratorBase& iter, const func_t& f) {
  TORCH_INTERNAL_ASSERT(iter.ntensors() == 3);

  using traits = function_traits<func_t>;
  using opmath_arg1_t = typename traits::template arg<0>::type;
  using opmath_arg2_t = typename traits::template arg<1>::type;
  static_assert(
      traits::arity == 2,
      "gpu_kernel_with_scalars only supports two input arguments");

  if (iter.is_cpu_scalar(1)) {
    AUnaryFunctor<arg1_t, arg2_t, return_t, func_t> af(f, iter.scalar_value<opmath_arg1_t>(1));
    iter.remove_operand(1);
    // TODO: When all kernels that use gpu_kernel_with_scalars are
    // ported to structured, this device guard can be deleted.  This
    // works around incorrect device guard generation for pre-structured
    // kernels device guards, but structured kernels do it right and
    // we can assume the device is already set correctly
    const OptionalDeviceGuard device_guard(iter.device(1));
    gpu_kernel_maca_arity1(iter, af);
  } else if (iter.is_cpu_scalar(2)) {
    BUnaryFunctor<arg1_t, arg2_t, return_t, func_t> bf(f, iter.scalar_value<opmath_arg2_t>(2));
    iter.remove_operand(2);
    gpu_kernel_maca_arity1(iter, bf);
  } else {
    gpu_kernel_maca_arity2(iter, BinaryFunctor<arg1_t, arg2_t, return_t, func_t>(f));
  }
}


template <typename scalar_t, typename return_t = scalar_t, typename func_t>
void opmath_symmetric_gpu_kernel_with_scalars_arity2(TensorIteratorBase& iter, const func_t& f) {
  // Use symmetric property of the functor to reduce number of kernels,
  // requires f(a, b) == f(b, a)
  TORCH_INTERNAL_ASSERT(iter.ntensors() == 3);

  using traits = function_traits<func_t>;
  using opmath_arg_t = typename traits::template arg<0>::type;
  static_assert(
      traits::arity == 2,
      "gpu_kernel_with_scalars only supports two input arguments");
  static_assert(std::is_same<opmath_arg_t, typename traits::template arg<1>::type>::value,
                "f is not symmetric");

  OptionalDeviceGuard device_guard;
  opmath_arg_t scalar_val{};

  if (iter.is_cpu_scalar(1)) {
    scalar_val = iter.scalar_value<opmath_arg_t>(1);
    iter.remove_operand(1);

    // TODO: When all kernels that use gpu_kernel_with_scalars are
    // ported to structured, this device guard can be deleted.  This
    // works around incorrect device guard generation for pre-structured
    // kernels device guards, but structured kernels do it right and
    // we can assume the device is already set correctly
    device_guard.reset_device(iter.device(1));
  } else if (iter.is_cpu_scalar(2)) {
    scalar_val = iter.scalar_value<opmath_arg_t>(2);
    iter.remove_operand(2);
  }

  if (iter.ninputs() == 2) {
    gpu_kernel_maca_arity2(iter, BinaryFunctor<scalar_t, scalar_t, return_t, func_t>(f));
  } else {
    AUnaryFunctor<scalar_t, scalar_t, return_t, func_t> unary_f(f, scalar_val);
    gpu_kernel_maca_arity1(iter, unary_f);
  }
}


template <typename func_t>
void gpu_kernel_with_scalars_arity2(TensorIteratorBase& iter, const func_t& f) {
  bool disable_arity2_opt = at::maca::get_maca_disable_gpu_kernel_with_scalars_arity2();
  if(maca_likely(disable_arity2_opt)){
    gpu_kernel_with_scalars(iter, f);
    return;
  }

  using traits = function_traits<func_t>;
  static_assert(
      traits::arity == 2,
      "gpu_kernel_with_scalars_arity2 only supports two input arguments");
  using arg1_t = typename traits::template arg<0>::type;
  using arg2_t = typename traits::template arg<1>::type;
  using return_t = typename traits::result_type;
  opmath_gpu_kernel_with_scalars_arity2<arg1_t, arg2_t, return_t, func_t>(iter, f);
}

template <typename T> struct is_tuple: std::false_type {};

template <typename ...T> struct is_tuple<thrust::tuple<T...>>: std::true_type {};

template <int num_outputs, typename func_t, typename array_t, typename inp_calc_t, typename out_calc_t>
C10_LAUNCH_BOUNDS_1(num_threads())
__global__ void unrolled_elementwise_kernel_for_multi_outputs(int N, func_t f, array_t data, inp_calc_t ic, out_calc_t oc) {
  int remaining = N - block_work_size() * blockIdx.x;
  elementwise_kernel_helper(f, memory::policies::multi_outputs_unroll<array_t, inp_calc_t, out_calc_t, num_outputs>(data, remaining, ic, oc));
}

template <int num_outputs, typename func_t, typename array_t, typename inp_calc_t, typename out_calc_t>
static inline void launch_unrolled_kernel_for_multi_outputs(int64_t N, const func_t& f, array_t data, inp_calc_t ic, out_calc_t oc) {
  TORCH_INTERNAL_ASSERT(N > 0 && N <= std::numeric_limits<int32_t>::max());
  int64_t grid = (N + block_work_size() - 1) / block_work_size();
  auto stream = at::cuda::getCurrentCUDAStream();
  unrolled_elementwise_kernel_for_multi_outputs<num_outputs, func_t, array_t><<<grid, num_threads(), 0, stream>>>(N, f, data, ic, oc);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename func_t>
void gpu_kernel_multiple_outputs_impl(TensorIteratorBase& iter, const func_t& f) {
  using traits = function_traits<func_t>;
  using output_t = typename traits::result_type;
  static_assert(is_tuple<output_t>::value, "f's return type must be `thrust::tuple`");
  constexpr int num_outputs = thrust::tuple_size<output_t>::value;
  constexpr int num_inputs = traits::arity;
  constexpr int ntensors = num_outputs + num_inputs;

  TORCH_INTERNAL_ASSERT(iter.can_use_32bit_indexing());
  TORCH_INTERNAL_ASSERT(iter.ntensors() == ntensors);

  at::detail::Array<char*, ntensors> data;
  for (int i = 0; i < ntensors; i++) {
    data[i] = (char*)iter.data_ptr(i);
  }

  int64_t numel = iter.numel();
  int64_t ndim = iter.ndim();
  
  bool disable_elementwise_multi_outputs = at::maca::get_maca_disable_elementwise_multi_outputs();
  bool get_elementwise_multi_outputs_shape = at::maca::get_maca_elementwise_multi_outputs_shape();
  

  if (iter.is_contiguous()) {
    if(get_elementwise_multi_outputs_shape){
      std::cout<<"multi_outputs_contiguous(): numel is "<< numel <<std::endl;
    }

    if(!disable_elementwise_multi_outputs
       && ((num_outputs==2 && num_inputs==4) || (num_outputs==2 && num_inputs==1))){
      launch_contiguous_optim_unrolled_kernel_for_multi_outputs<num_outputs, num_inputs>()(numel, f, data);
    }else {
      auto input_calc = TrivialOffsetCalculator<num_inputs>();
      auto output_calc = TrivialOffsetCalculator<num_outputs>();
      launch_unrolled_kernel_for_multi_outputs<num_outputs>(numel, f, data, input_calc, output_calc);
    }
  } else {
    auto input_calc = make_input_offset_calculator<num_inputs>(iter);
    auto output_calc = make_output_offset_calculator<num_outputs>(iter);

    if(get_elementwise_multi_outputs_shape){
      print_multi_outputs(num_outputs, num_inputs, ndim, input_calc, output_calc);
    }

    if(!disable_elementwise_multi_outputs
       && ((num_outputs==2 && num_inputs==4 && ndim>0 && ndim <=6) || 
          (num_outputs==2 && num_inputs==1 && ndim>0 && ndim <=4))){
      launch_legacy_optim_unrolled_kernel_for_multi_outputs
           <num_outputs, num_inputs>()(numel, f, data, input_calc, output_calc, ndim);
    } else {
     launch_unrolled_kernel_for_multi_outputs<num_outputs>(numel, f, data, input_calc, output_calc);
    }
  }
}

}   // namespace at::native
