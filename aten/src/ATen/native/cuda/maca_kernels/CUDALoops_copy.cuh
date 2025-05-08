#pragma once

// This file provides two functions to help write GPU elementwise kernels:
//
//   gpu_kernel(TensorIterator iter, <lambda>)
//   gpu_kernel_with_scalars(TensorIterator iter, <lambda>)
//
// The gpu_kernel_with_scalars generates specializations that support a
// single scalar CPU argument, such as from `cuda_tensor + 5`. The CPU scalar
// is lifted to a kernel parameter instead of copying to device memory.
// This should be  used in conjunction with TensorIterator::allow_cpu_scalars_,
// which is the default for TensorIterator::binary_op. Otherwise, all inputs
// and the output must be on the GPU.
//
// For example, to write a reciprocal kernel for GPU float Tensors:
//
//   gpu_kernel(iter, []GPU_LAMBDA(float a) {
//    return 1.0f / a;
//   });
//
// To write a multiplication kernel for GPU float Tensors where one argument
// may be a CPU scalar:
//
//   gpu_kernel_with_scalars(iter, []GPU_LAMBDA(float a, float b) {
//     return a * b;
//   });
//
// See BinaryOpsKernel.cu for the complete implementation
//

#include <type_traits>
#include <tuple>
#include <iostream>

#include <ATen/cuda/CUDAContext.h>
#include <ATen/core/Array.h>
#include <ATen/native/cuda/MemoryAccess.cuh>
#include <ATen/detail/FunctionTraits.h>
#include <ATen/native/TensorIterator.h>
#include <c10/macros/Macros.h>
#include <c10/core/DynamicCast.h>
#include <c10/core/ScalarType.h>
#include <c10/util/TypeCast.h>
#include <c10/util/C++17.h>
#include "loop_utils.h"
#include "CUDALoops_maca_highdim.cuh"

#ifdef __NVCC__
#define ASSERT_HOST_DEVICE_LAMBDA(type) 
#if defined(TORCH_MACA_LAMBDA)
  static_assert(__nv_is_extended_host_device_lambda_closure_type(type), \
                #type " must be a __host__ __device__ lambda")
#endif  // TORCH_MACA_LAMBDA
#else
#define ASSERT_HOST_DEVICE_LAMBDA(type)
#endif

#define kTileDimMacaT 64
#define kTileDimMacaT_32 32
#define kBlockRowsMacaT 8

namespace {

template<typename T>
bool type_trait_is_half_maca_copy() { return false; }
template<>
bool type_trait_is_half_maca_copy<at::Half>() { return true; }

}

namespace at { namespace native {

template<int vec_size, typename func_t, typename array_t>
C10_LAUNCH_BOUNDS_1(num_threads())
__global__ void vectorized_elementwise_kernel(int N, func_t f, array_t data) {
  using traits = function_traits<func_t>;
  int remaining = N - block_work_size() * blockIdx.x;

  if (remaining < block_work_size()) {  // if this block handles the reminder, just do a naive unrolled loop
    auto input_calc = TrivialOffsetCalculator<traits::arity>();
    auto output_calc = TrivialOffsetCalculator<1>();
    auto loader = memory::LoadWithoutCast();
    auto storer = memory::StoreWithoutCast();
    auto policy = memory::policies::unroll<array_t, decltype(input_calc), decltype(output_calc),
                                           memory::LoadWithoutCast, memory::StoreWithoutCast>(
      data, remaining, input_calc, output_calc, loader, storer);
    elementwise_kernel_helper(f, policy);
  } else {  // if this block has a full `block_work_size` data to handle, use vectorized memory access
    elementwise_kernel_helper(f, memory::policies::vectorized<vec_size, array_t>(data));
  }
}

template<typename func_t, typename array_t, typename inp_calc_t, typename out_calc_t, typename loader_t, typename storer_t>
C10_LAUNCH_BOUNDS_1(num_threads())
__global__ void unrolled_elementwise_kernel(int N, func_t f, array_t data,
                                            inp_calc_t ic, out_calc_t oc, loader_t l, storer_t s)
{
  int remaining = N - block_work_size() * blockIdx.x;
  auto policy = memory::policies::unroll<array_t, inp_calc_t, out_calc_t, loader_t, storer_t>(data, remaining, ic, oc, l, s);
  elementwise_kernel_helper(f, policy);
}

template<typename func_t, typename size_t>
C10_LAUNCH_BOUNDS_1(num_threads())
__global__ void unrolled_elementwise_kernel_4_float_to_half(func_t f, char* data0, char* data1)
{
  elementwise_kernel_helper_4_float_to_half<func_t, size_t>(f, data0, data1);
}

template<typename func_t, typename size_t>
C10_LAUNCH_BOUNDS_1(num_threads())
__global__ void unrolled_elementwise_kernel_4_float_to_half_no_align(int64_t N, func_t f, char* data0, char* data1)
{
  elementwise_kernel_helper_4_float_to_half_no_align<func_t, size_t>(N, f, data0, data1);
}

template<typename func_t, typename size_t>
C10_LAUNCH_BOUNDS_1(num_threads())
__global__ void unrolled_elementwise_kernel_4_half_to_float(func_t f, char* data0, char* data1)
{
  elementwise_kernel_helper_4_half_to_float<func_t, size_t>(f, data0, data1);
}

template<typename func_t, typename size_t>
C10_LAUNCH_BOUNDS_1(num_threads())
__global__ void unrolled_elementwise_kernel_4_half_to_float_no_align(int64_t N, func_t f, char* data0, char* data1)
{
  elementwise_kernel_helper_4_half_to_float_no_align<func_t, size_t>(N, f, data0, data1);
}

template<typename func_t, typename size_t>
C10_LAUNCH_BOUNDS_1(num_threads())
__global__ void unrolled_elementwise_kernel_4_float_to_bfloat(func_t f, char* data0, char* data1)
{
  elementwise_kernel_helper_4_float_to_bfloat<func_t, size_t>(f, data0, data1);
}

template<typename func_t, typename size_t>
C10_LAUNCH_BOUNDS_1(num_threads())
__global__ void unrolled_elementwise_kernel_4_float_to_bfloat_no_align(int64_t N, func_t f, char* data0, char* data1)
{
  elementwise_kernel_helper_4_float_to_bfloat_no_align<func_t, size_t>(N, f, data0, data1);
}


template<typename func_t, typename size_t>
C10_LAUNCH_BOUNDS_1(num_threads())
__global__ void unrolled_elementwise_kernel_4_bfloat_to_float(func_t f, char* data0, char* data1)
{
  elementwise_kernel_helper_4_bfloat_to_float<func_t, size_t>(f, data0, data1);
}

template<typename func_t, typename size_t>
C10_LAUNCH_BOUNDS_1(num_threads())
__global__ void unrolled_elementwise_kernel_4_bfloat_to_float_no_align(int64_t N, func_t f, char* data0, char* data1)
{
  elementwise_kernel_helper_4_bfloat_to_float_no_align<func_t, size_t>(N, f, data0, data1);
}

// this function assume trivial 1d and no dynamic casting
template<typename func_t, typename array_t>
static inline void launch_vectorized_kernel(int64_t N, const func_t& f, array_t data) {
  TORCH_INTERNAL_ASSERT(N > 0 && N <= std::numeric_limits<int32_t>::max());
  using traits = function_traits<func_t>;
  int64_t grid = (N + block_work_size() - 1) / block_work_size();
  auto stream = at::cuda::getCurrentCUDAStream();
  int vec_size = memory::can_vectorize_up_to<func_t>(data);

  switch (vec_size) {
  case 4:
    vectorized_elementwise_kernel<4, func_t, array_t><<<grid, num_threads(), 0, stream>>>(N, f, data);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    break;
  case 2:
    vectorized_elementwise_kernel<2, func_t, array_t><<<grid, num_threads(), 0, stream>>>(N, f, data);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    break;
  case 1: {
    auto input_calc = TrivialOffsetCalculator<traits::arity>();
    auto output_calc = TrivialOffsetCalculator<1>();
    auto loader = memory::LoadWithoutCast();
    auto storer = memory::StoreWithoutCast();
    unrolled_elementwise_kernel<func_t, array_t><<<grid, num_threads(), 0, stream>>>(N, f, data, input_calc, output_calc, loader, storer);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    break;
  }
  default:
    TORCH_INTERNAL_ASSERT(false, "Unexpected vectorization size");
  }
}


template<typename func_t, typename array_t, typename inp_calc_t, typename out_calc_t, typename loader_t, typename storer_t>
static inline void launch_unrolled_kernel(int64_t N, const func_t& f, array_t data,
                                          inp_calc_t ic, out_calc_t oc, loader_t l, storer_t s)
{
  TORCH_INTERNAL_ASSERT(N > 0 && N <= std::numeric_limits<int32_t>::max());
  int64_t grid = (N + block_work_size() - 1) / block_work_size();
  auto stream = at::cuda::getCurrentCUDAStream();
  unrolled_elementwise_kernel<func_t, array_t><<<grid, num_threads(), 0, stream>>>(N, f, data, ic, oc, l, s);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}


template<typename func_t, typename size_t>
C10_LAUNCH_BOUNDS_1(num_threads())
__global__ void unrolled_elementwise_kernel_4_float_to_int(func_t f, char* data0, char* data1)
{
  using return_t = int;
  using args_t = float;
  const int vt = 4;

  int64_t idx = blockIdx.x;
  int thread_idx = threadIdx.x;

  using LoadT = at::native::memory::aligned_vector<args_t, vt>;
  using StoreT = at::native::memory::aligned_vector<return_t, vt>;

  args_t args[vt];
  return_t results[vt];

  LoadT* p_ld = reinterpret_cast<LoadT*>(&args);
  StoreT* p_sr = reinterpret_cast<StoreT*>(&results);

  size_t offset_ld = (thread_idx * vt + (blockDim.x * vt) * idx) * sizeof(args_t);
  size_t offset_sr = (thread_idx * vt + (blockDim.x * vt) * idx) * sizeof(return_t);
  *p_ld = *reinterpret_cast<LoadT*>(data1 + offset_ld);
  #pragma unroll vt
  for(int i = 0; i < vt; i++){
    results[i] = c10::convert<return_t>(args[i]);
  }
  StoreT* out = reinterpret_cast<StoreT*>(data0 + offset_sr);
  *out = *p_sr;
}
template<typename func_t, typename size_t>
C10_LAUNCH_BOUNDS_1(num_threads())
__global__ void unrolled_elementwise_kernel_4_float_to_int_no_align(int64_t N, func_t f, char* data0, char* data1)
{
  using return_t = int;
  using args_t = float;
  const int vt = 4;

  int64_t idx = blockIdx.x;
  int thread_idx = threadIdx.x;

  using LoadT = at::native::memory::aligned_vector<args_t, vt>;
  using StoreT = at::native::memory::aligned_vector<return_t, vt>;

  args_t args[vt];
  return_t results[vt];
  LoadT* p_ld = reinterpret_cast<LoadT*>(&args);
  StoreT* p_sr = reinterpret_cast<StoreT*>(&results);

  size_t offset=thread_idx * vt + (blockDim.x * vt) * idx;

  if(offset <= (N/vt-1)* vt ){
    *p_ld = *reinterpret_cast<LoadT*>(data1 + offset * sizeof(args_t));
    #pragma unroll vt
    for(int i = 0; i < vt; i++){
      results[i] = c10::convert<return_t>(args[i]);
    }
    StoreT* out = reinterpret_cast<StoreT*>(data0 + offset * sizeof(return_t));
    *out = *p_sr;
  }else if(offset == (N/vt)* vt){
    while(offset<N){
      return_t cur_result = c10::convert<return_t>(*reinterpret_cast<args_t*>(data1 + offset * sizeof(args_t)));
      return_t* cur_out = reinterpret_cast<return_t*>(data0 + offset * sizeof(return_t));
      *cur_out=cur_result;
      offset++;
    }
  }
}

template<typename func_t, typename size_t>
C10_LAUNCH_BOUNDS_1(num_threads())
__global__ void unrolled_elementwise_kernel_4_int_to_half(func_t f, char* data0, char* data1)
{
  using return_t = at::Half;
  using args_t = int;
  const int vt = 4;

  int64_t idx = blockIdx.x;
  int thread_idx = threadIdx.x;

  using LoadT = at::native::memory::aligned_vector<args_t, vt>;
  using StoreT = at::native::memory::aligned_vector<return_t, vt>;

  args_t args[vt];
  return_t results[vt];

  LoadT* p_ld = reinterpret_cast<LoadT*>(&args);
  StoreT* p_sr = reinterpret_cast<StoreT*>(&results);

  size_t offset_ld = (thread_idx * vt + (blockDim.x * vt) * idx) * sizeof(args_t);
  size_t offset_sr = (thread_idx * vt + (blockDim.x * vt) * idx) * sizeof(return_t);
  *p_ld = *reinterpret_cast<LoadT*>(data1 + offset_ld);
  #pragma unroll vt
  for(int i = 0; i < vt; i++){
    results[i] = c10::convert<return_t>(args[i]);
  }
  StoreT* out = reinterpret_cast<StoreT*>(data0 + offset_sr);
  *out = *p_sr;
}
template<typename func_t, typename size_t>
C10_LAUNCH_BOUNDS_1(num_threads())
__global__ void unrolled_elementwise_kernel_4_int_to_half_no_align(int64_t N, func_t f, char* data0, char* data1)
{
  using return_t = at::Half;
  using args_t = int;
  const int vt = 4;

  int64_t idx = blockIdx.x;
  int thread_idx = threadIdx.x;

  using LoadT = at::native::memory::aligned_vector<args_t, vt>;
  using StoreT = at::native::memory::aligned_vector<return_t, vt>;

  args_t args[vt];
  return_t results[vt];
  LoadT* p_ld = reinterpret_cast<LoadT*>(&args);
  StoreT* p_sr = reinterpret_cast<StoreT*>(&results);

  size_t offset=thread_idx * vt + (blockDim.x * vt) * idx;

  if(offset <= (N/vt-1)* vt ){
    *p_ld = *reinterpret_cast<LoadT*>(data1 + offset * sizeof(args_t));
    #pragma unroll vt
    for(int i = 0; i < vt; i++){
      results[i] = c10::convert<return_t>(args[i]);
    }
    StoreT* out = reinterpret_cast<StoreT*>(data0 + offset * sizeof(return_t));
    *out = *p_sr;
  }else if(offset == (N/vt)* vt){
    while(offset<N){
      return_t cur_result = c10::convert<return_t>(*reinterpret_cast<args_t*>(data1 + offset * sizeof(args_t)));
      return_t* cur_out = reinterpret_cast<return_t*>(data0 + offset * sizeof(return_t));
      *cur_out=cur_result;
      offset++;
    }
  }
}

template<typename func_t, typename size_t>
C10_LAUNCH_BOUNDS_1(num_threads())
__global__ void unrolled_elementwise_kernel_2_int_to_long(func_t f, char* data0, char* data1)
{
  using return_t = long;
  using args_t = int;
  const int vt = 2;

  int64_t idx = blockIdx.x;
  int thread_idx = threadIdx.x;

  using LoadT = at::native::memory::aligned_vector<args_t, vt>;
  using StoreT = at::native::memory::aligned_vector<return_t, vt>;

  args_t args[vt];
  return_t results[vt];

  LoadT* p_ld = reinterpret_cast<LoadT*>(&args);
  StoreT* p_sr = reinterpret_cast<StoreT*>(&results);

  size_t offset_ld = (thread_idx * vt + (blockDim.x * vt) * idx) * sizeof(args_t);
  size_t offset_sr = (thread_idx * vt + (blockDim.x * vt) * idx) * sizeof(return_t);
  *p_ld = *reinterpret_cast<LoadT*>(data1 + offset_ld);
  #pragma unroll vt
  for(int i = 0; i < vt; i++){
    results[i] = c10::convert<return_t>(args[i]);
  }
  StoreT* out = reinterpret_cast<StoreT*>(data0 + offset_sr);
  *out = *p_sr;
}
template<typename func_t, typename size_t>
C10_LAUNCH_BOUNDS_1(num_threads())
__global__ void unrolled_elementwise_kernel_2_int_to_long_no_align(int64_t N, func_t f, char* data0, char* data1)
{
  using return_t = long;
  using args_t = int;
  const int vt = 2;

  int64_t idx = blockIdx.x;
  int thread_idx = threadIdx.x;

  using LoadT = at::native::memory::aligned_vector<args_t, vt>;
  using StoreT = at::native::memory::aligned_vector<return_t, vt>;

  args_t args[vt];
  return_t results[vt];
  LoadT* p_ld = reinterpret_cast<LoadT*>(&args);
  StoreT* p_sr = reinterpret_cast<StoreT*>(&results);

  size_t offset=thread_idx * vt + (blockDim.x * vt) * idx;

  if(offset <= (N/vt-1)* vt ){
    *p_ld = *reinterpret_cast<LoadT*>(data1 + offset * sizeof(args_t));
    #pragma unroll vt
    for(int i = 0; i < vt; i++){
      results[i] = c10::convert<return_t>(args[i]);
    }
    StoreT* out = reinterpret_cast<StoreT*>(data0 + offset * sizeof(return_t));
    *out = *p_sr;
  }else if(offset == (N/vt)* vt){
    while(offset<N){
      return_t cur_result = c10::convert<return_t>(*reinterpret_cast<args_t*>(data1 + offset * sizeof(args_t)));
      return_t* cur_out = reinterpret_cast<return_t*>(data0 + offset * sizeof(return_t));
      *cur_out=cur_result;
      offset++;
    }
  }
}

template<typename func_t, typename size_t>
C10_LAUNCH_BOUNDS_1(num_threads())
__global__ void unrolled_elementwise_kernel_2_long_to_int(func_t f, char* data0, char* data1)
{
  using return_t = int;
  using args_t = long;
  const int vt = 2;

  int64_t idx = blockIdx.x;
  int thread_idx = threadIdx.x;

  using LoadT = at::native::memory::aligned_vector<args_t, vt>;
  using StoreT = at::native::memory::aligned_vector<return_t, vt>;

  args_t args[vt];
  return_t results[vt];

  LoadT* p_ld = reinterpret_cast<LoadT*>(&args);
  StoreT* p_sr = reinterpret_cast<StoreT*>(&results);

  size_t offset_ld = (thread_idx * vt + (blockDim.x * vt) * idx) * sizeof(args_t);
  size_t offset_sr = (thread_idx * vt + (blockDim.x * vt) * idx) * sizeof(return_t);
  *p_ld = *reinterpret_cast<LoadT*>(data1 + offset_ld);
  #pragma unroll vt
  for(int i = 0; i < vt; i++){
    results[i] = c10::convert<return_t>(args[i]);
  }
  StoreT* out = reinterpret_cast<StoreT*>(data0 + offset_sr);
  *out = *p_sr;
}
template<typename func_t, typename size_t>
C10_LAUNCH_BOUNDS_1(num_threads())
__global__ void unrolled_elementwise_kernel_2_long_to_int_no_align(int64_t N, func_t f, char* data0, char* data1)
{
  using return_t = int;
  using args_t = long;
  const int vt = 2;

  int64_t idx = blockIdx.x;
  int thread_idx = threadIdx.x;

  using LoadT = at::native::memory::aligned_vector<args_t, vt>;
  using StoreT = at::native::memory::aligned_vector<return_t, vt>;

  args_t args[vt];
  return_t results[vt];
  LoadT* p_ld = reinterpret_cast<LoadT*>(&args);
  StoreT* p_sr = reinterpret_cast<StoreT*>(&results);

  size_t offset=thread_idx * vt + (blockDim.x * vt) * idx;

  if(offset <= (N/vt-1)* vt ){
    *p_ld = *reinterpret_cast<LoadT*>(data1 + offset * sizeof(args_t));
    #pragma unroll vt
    for(int i = 0; i < vt; i++){
      results[i] = c10::convert<return_t>(args[i]);
    }
    StoreT* out = reinterpret_cast<StoreT*>(data0 + offset * sizeof(return_t));
    *out = *p_sr;
  }else if(offset == (N/vt)* vt){
    while(offset<N){
      return_t cur_result = c10::convert<return_t>(*reinterpret_cast<args_t*>(data1 + offset * sizeof(args_t)));
      return_t* cur_out = reinterpret_cast<return_t*>(data0 + offset * sizeof(return_t));
      *cur_out=cur_result;
      offset++;
    }
  }
}

template<typename func_t, typename size_t>
C10_LAUNCH_BOUNDS_1(num_threads())
__global__ void unrolled_elementwise_kernel_8_bfloat_to_half(func_t f, char* data0, char* data1)
{
  using return_t = at::Half;
  using args_t = at::BFloat16;
  const int vt = 8;

  int64_t idx = blockIdx.x;
  int thread_idx = threadIdx.x;

  using LoadT = at::native::memory::aligned_vector<args_t, vt>;
  using StoreT = at::native::memory::aligned_vector<return_t, vt>;

  args_t args[vt];
  return_t results[vt];

  LoadT* p_ld = reinterpret_cast<LoadT*>(&args);
  StoreT* p_sr = reinterpret_cast<StoreT*>(&results);

  size_t offset_ld = (thread_idx * vt + (blockDim.x * vt) * idx) * sizeof(args_t);
  size_t offset_sr = (thread_idx * vt + (blockDim.x * vt) * idx) * sizeof(return_t);
  *p_ld = *reinterpret_cast<LoadT*>(data1 + offset_ld);
  #pragma unroll vt
  for(int i = 0; i < vt; i++){
    results[i] = c10::convert<return_t>(args[i]);
  }
  StoreT* out = reinterpret_cast<StoreT*>(data0 + offset_sr);
  *out = *p_sr;
}
template<typename func_t, typename size_t>
C10_LAUNCH_BOUNDS_1(num_threads())
__global__ void unrolled_elementwise_kernel_8_bfloat_to_half_no_align(int64_t N, func_t f, char* data0, char* data1)
{
  using return_t = at::Half;
  using args_t = at::BFloat16;
  const int vt = 8;

  int64_t idx = blockIdx.x;
  int thread_idx = threadIdx.x;

  using LoadT = at::native::memory::aligned_vector<args_t, vt>;
  using StoreT = at::native::memory::aligned_vector<return_t, vt>;

  args_t args[vt];
  return_t results[vt];
  LoadT* p_ld = reinterpret_cast<LoadT*>(&args);
  StoreT* p_sr = reinterpret_cast<StoreT*>(&results);

  size_t offset=thread_idx * vt + (blockDim.x * vt) * idx;

  if(offset <= (N/vt-1)* vt ){
    *p_ld = *reinterpret_cast<LoadT*>(data1 + offset * sizeof(args_t));
    #pragma unroll vt
    for(int i = 0; i < vt; i++){
      results[i] = c10::convert<return_t>(args[i]);
    }
    StoreT* out = reinterpret_cast<StoreT*>(data0 + offset * sizeof(return_t));
    *out = *p_sr;
  }else if(offset == (N/vt)* vt){
    while(offset<N){
      return_t cur_result = c10::convert<return_t>(*reinterpret_cast<args_t*>(data1 + offset * sizeof(args_t)));
      return_t* cur_out = reinterpret_cast<return_t*>(data0 + offset * sizeof(return_t));
      *cur_out=cur_result;
      offset++;
    }
  }
}

template<typename func_t, typename size_t>
C10_LAUNCH_BOUNDS_1(num_threads())
__global__ void unrolled_elementwise_kernel_8_half_to_bfloat(func_t f, char* data0, char* data1)
{
  using return_t = at::BFloat16;
  using args_t = at::Half;
  const int vt = 8;

  int64_t idx = blockIdx.x;
  int thread_idx = threadIdx.x;

  using LoadT = at::native::memory::aligned_vector<args_t, vt>;
  using StoreT = at::native::memory::aligned_vector<return_t, vt>;

  args_t args[vt];
  return_t results[vt];

  LoadT* p_ld = reinterpret_cast<LoadT*>(&args);
  StoreT* p_sr = reinterpret_cast<StoreT*>(&results);

  size_t offset_ld = (thread_idx * vt + (blockDim.x * vt) * idx) * sizeof(args_t);
  size_t offset_sr = (thread_idx * vt + (blockDim.x * vt) * idx) * sizeof(return_t);
  *p_ld = *reinterpret_cast<LoadT*>(data1 + offset_ld);
  #pragma unroll vt
  for(int i = 0; i < vt; i++){
    results[i] = c10::convert<return_t>(args[i]);
  }
  StoreT* out = reinterpret_cast<StoreT*>(data0 + offset_sr);
  *out = *p_sr;
}
template<typename func_t, typename size_t>
C10_LAUNCH_BOUNDS_1(num_threads())
__global__ void unrolled_elementwise_kernel_8_half_to_bfloat_no_align(int64_t N, func_t f, char* data0, char* data1)
{
  using return_t = at::BFloat16;
  using args_t = at::Half;
  const int vt = 8;

  int64_t idx = blockIdx.x;
  int thread_idx = threadIdx.x;

  using LoadT = at::native::memory::aligned_vector<args_t, vt>;
  using StoreT = at::native::memory::aligned_vector<return_t, vt>;

  args_t args[vt];
  return_t results[vt];
  LoadT* p_ld = reinterpret_cast<LoadT*>(&args);
  StoreT* p_sr = reinterpret_cast<StoreT*>(&results);

  size_t offset=thread_idx * vt + (blockDim.x * vt) * idx;

  if(offset <= (N/vt-1)* vt ){
    *p_ld = *reinterpret_cast<LoadT*>(data1 + offset * sizeof(args_t));
    #pragma unroll vt
    for(int i = 0; i < vt; i++){
      results[i] = c10::convert<return_t>(args[i]);
    }
    StoreT* out = reinterpret_cast<StoreT*>(data0 + offset * sizeof(return_t));
    *out = *p_sr;
  }else if(offset == (N/vt)* vt){
    while(offset<N){
      return_t cur_result = c10::convert<return_t>(*reinterpret_cast<args_t*>(data1 + offset * sizeof(args_t)));
      return_t* cur_out = reinterpret_cast<return_t*>(data0 + offset * sizeof(return_t));
      *cur_out=cur_result;
      offset++;
    }
  }
}

template<typename func_t, typename size_t>
C10_LAUNCH_BOUNDS_1(num_threads())
__global__ void unrolled_elementwise_kernel_2_long_to_float(func_t f, char* data0, char* data1)
{
  using return_t = float;
  using args_t = long;
  const int vt = 2;

  int64_t idx = blockIdx.x;
  int thread_idx = threadIdx.x;

  using LoadT = at::native::memory::aligned_vector<args_t, vt>;
  using StoreT = at::native::memory::aligned_vector<return_t, vt>;

  args_t args[vt];
  return_t results[vt];

  LoadT* p_ld = reinterpret_cast<LoadT*>(&args);
  StoreT* p_sr = reinterpret_cast<StoreT*>(&results);

  size_t offset_ld = (thread_idx * vt + (blockDim.x * vt) * idx) * sizeof(args_t);
  size_t offset_sr = (thread_idx * vt + (blockDim.x * vt) * idx) * sizeof(return_t);
  *p_ld = *reinterpret_cast<LoadT*>(data1 + offset_ld);
  #pragma unroll vt
  for(int i = 0; i < vt; i++){
    results[i] = c10::convert<return_t>(args[i]);
  }
  StoreT* out = reinterpret_cast<StoreT*>(data0 + offset_sr);
  *out = *p_sr;
}
template<typename func_t, typename size_t>
C10_LAUNCH_BOUNDS_1(num_threads())
__global__ void unrolled_elementwise_kernel_2_long_to_float_no_align(int64_t N, func_t f, char* data0, char* data1)
{
  using return_t = float;
  using args_t = long;
  const int vt = 2;

  int64_t idx = blockIdx.x;
  int thread_idx = threadIdx.x;

  using LoadT = at::native::memory::aligned_vector<args_t, vt>;
  using StoreT = at::native::memory::aligned_vector<return_t, vt>;

  args_t args[vt];
  return_t results[vt];
  LoadT* p_ld = reinterpret_cast<LoadT*>(&args);
  StoreT* p_sr = reinterpret_cast<StoreT*>(&results);

  size_t offset=thread_idx * vt + (blockDim.x * vt) * idx;

  if(offset <= (N/vt-1)* vt ){
    *p_ld = *reinterpret_cast<LoadT*>(data1 + offset * sizeof(args_t));
    #pragma unroll vt
    for(int i = 0; i < vt; i++){
      results[i] = c10::convert<return_t>(args[i]);
    }
    StoreT* out = reinterpret_cast<StoreT*>(data0 + offset * sizeof(return_t));
    *out = *p_sr;
  }else if(offset == (N/vt)* vt){
    while(offset<N){
      return_t cur_result = c10::convert<return_t>(*reinterpret_cast<args_t*>(data1 + offset * sizeof(args_t)));
      return_t* cur_out = reinterpret_cast<return_t*>(data0 + offset * sizeof(return_t));
      *cur_out=cur_result;
      offset++;
    }
  }
}

template<typename func_t, typename size_t>
C10_LAUNCH_BOUNDS_1(num_threads())
__global__ void unrolled_elementwise_kernel_4_float_to_bool(func_t f, char* data0, char* data1)
{
  using return_t = bool;
  using args_t = float;
  const int vt = 4;

  int64_t idx = blockIdx.x;
  int thread_idx = threadIdx.x;

  using LoadT = at::native::memory::aligned_vector<args_t, vt>;
  using StoreT = at::native::memory::aligned_vector<return_t, vt>;

  args_t args[vt];
  return_t results[vt];

  LoadT* p_ld = reinterpret_cast<LoadT*>(&args);
  StoreT* p_sr = reinterpret_cast<StoreT*>(&results);

  size_t offset_ld = (thread_idx * vt + (blockDim.x * vt) * idx) * sizeof(args_t);
  size_t offset_sr = (thread_idx * vt + (blockDim.x * vt) * idx) * sizeof(return_t);
  *p_ld = *reinterpret_cast<LoadT*>(data1 + offset_ld);
  #pragma unroll vt
  for(int i = 0; i < vt; i++){
    results[i] = c10::convert<return_t>(args[i]);
  }
  StoreT* out = reinterpret_cast<StoreT*>(data0 + offset_sr);
  *out = *p_sr;
}
template<typename func_t, typename size_t>
C10_LAUNCH_BOUNDS_1(num_threads())
__global__ void unrolled_elementwise_kernel_4_float_to_bool_no_align(int64_t N, func_t f, char* data0, char* data1)
{
  using return_t = bool;
  using args_t = float;
  const int vt = 4;

  int64_t idx = blockIdx.x;
  int thread_idx = threadIdx.x;

  using LoadT = at::native::memory::aligned_vector<args_t, vt>;
  using StoreT = at::native::memory::aligned_vector<return_t, vt>;

  args_t args[vt];
  return_t results[vt];
  LoadT* p_ld = reinterpret_cast<LoadT*>(&args);
  StoreT* p_sr = reinterpret_cast<StoreT*>(&results);

  size_t offset=thread_idx * vt + (blockDim.x * vt) * idx;

  if(offset <= (N/vt-1)* vt ){
    *p_ld = *reinterpret_cast<LoadT*>(data1 + offset * sizeof(args_t));
    #pragma unroll vt
    for(int i = 0; i < vt; i++){
      results[i] = c10::convert<return_t>(args[i]);
    }
    StoreT* out = reinterpret_cast<StoreT*>(data0 + offset * sizeof(return_t));
    *out = *p_sr;
  }else if(offset == (N/vt)* vt){
    while(offset<N){
      return_t cur_result = c10::convert<return_t>(*reinterpret_cast<args_t*>(data1 + offset * sizeof(args_t)));
      return_t* cur_out = reinterpret_cast<return_t*>(data0 + offset * sizeof(return_t));
      *cur_out=cur_result;
      offset++;
    }
  }
}

template<typename func_t, typename size_t>
C10_LAUNCH_BOUNDS_1(num_threads())
__global__ void unrolled_elementwise_kernel_2_bool_to_long(func_t f, char* data0, char* data1)
{
  using return_t = long;
  using args_t = bool;
  const int vt = 2;

  int64_t idx = blockIdx.x;
  int thread_idx = threadIdx.x;

  using LoadT = at::native::memory::aligned_vector<args_t, vt>;
  using StoreT = at::native::memory::aligned_vector<return_t, vt>;

  args_t args[vt];
  return_t results[vt];

  LoadT* p_ld = reinterpret_cast<LoadT*>(&args);
  StoreT* p_sr = reinterpret_cast<StoreT*>(&results);

  size_t offset_ld = (thread_idx * vt + (blockDim.x * vt) * idx) * sizeof(args_t);
  size_t offset_sr = (thread_idx * vt + (blockDim.x * vt) * idx) * sizeof(return_t);
  *p_ld = *reinterpret_cast<LoadT*>(data1 + offset_ld);
  #pragma unroll vt
  for(int i = 0; i < vt; i++){
    results[i] = c10::convert<return_t>(args[i]);
  }
  StoreT* out = reinterpret_cast<StoreT*>(data0 + offset_sr);
  *out = *p_sr;
}
template<typename func_t, typename size_t>
C10_LAUNCH_BOUNDS_1(num_threads())
__global__ void unrolled_elementwise_kernel_2_bool_to_long_no_align(int64_t N, func_t f, char* data0, char* data1)
{
  using return_t = long;
  using args_t = bool;
  const int vt = 2;

  int64_t idx = blockIdx.x;
  int thread_idx = threadIdx.x;

  using LoadT = at::native::memory::aligned_vector<args_t, vt>;
  using StoreT = at::native::memory::aligned_vector<return_t, vt>;

  args_t args[vt];
  return_t results[vt];
  LoadT* p_ld = reinterpret_cast<LoadT*>(&args);
  StoreT* p_sr = reinterpret_cast<StoreT*>(&results);

  size_t offset=thread_idx * vt + (blockDim.x * vt) * idx;

  if(offset <= (N/vt-1)* vt ){
    *p_ld = *reinterpret_cast<LoadT*>(data1 + offset * sizeof(args_t));
    #pragma unroll vt
    for(int i = 0; i < vt; i++){
      results[i] = c10::convert<return_t>(args[i]);
    }
    StoreT* out = reinterpret_cast<StoreT*>(data0 + offset * sizeof(return_t));
    *out = *p_sr;
  }else if(offset == (N/vt)* vt){
    while(offset<N){
      return_t cur_result = c10::convert<return_t>(*reinterpret_cast<args_t*>(data1 + offset * sizeof(args_t)));
      return_t* cur_out = reinterpret_cast<return_t*>(data0 + offset * sizeof(return_t));
      *cur_out=cur_result;
      offset++;
    }
  }
}

template<typename res_t, typename arg0_t,typename func_t>
static inline void launch_unrolled_copy_cast_kernel(
      int64_t N, const func_t& f,
      ScalarType st0, ScalarType st1,
      char* data0, char* data1)
{
  TORCH_INTERNAL_ASSERT(N > 0 && N <= std::numeric_limits<int32_t>::max());
  int block = num_threads();
  auto stream = at::cuda::getCurrentCUDAStream();
  
  if (st0 == ScalarType::Half && st1 == ScalarType::Float) {
    int vt = 4;
    int64_t grid = (N + (block * vt) - 1) / (block * vt);
    if (N % (block * vt) == 0) {
      unrolled_elementwise_kernel_4_float_to_half<func_t, size_t><<<grid, block, 0, stream>>>(f, data0, data1);
    } else {
      unrolled_elementwise_kernel_4_float_to_half_no_align<func_t, size_t><<<grid, block, 0, stream>>>(N, f, data0, data1);
    }
  } else if (st0 == ScalarType::Float && st1 == ScalarType::Half) {
    int vt = 4;
    int64_t grid = (N + (block * vt) - 1) / (block * vt);
    if (N % (block * vt) == 0) {
      unrolled_elementwise_kernel_4_half_to_float<func_t, size_t><<<grid, block, 0, stream>>>(f, data0, data1);
    } else {
      unrolled_elementwise_kernel_4_half_to_float_no_align<func_t, size_t><<<grid, block, 0, stream>>>(N, f, data0, data1);
    }
  } else if (st0 == ScalarType::BFloat16 && st1 == ScalarType::Float) {
    int vt = 4;
    int64_t grid = (N + (block * vt) - 1) / (block * vt);
    if (N % (block * vt) == 0) {
      unrolled_elementwise_kernel_4_float_to_bfloat<func_t, size_t><<<grid, block, 0, stream>>>(f, data0, data1);
    } else {
      unrolled_elementwise_kernel_4_float_to_bfloat_no_align<func_t, size_t><<<grid, block, 0, stream>>>(N, f, data0, data1);
    }
  } else if (st0 == ScalarType::Float && st1 == ScalarType::BFloat16){
    int vt = 4;
    int64_t grid = (N + (block * vt) - 1) / (block * vt);
    if (N % (block * vt) == 0) {
      unrolled_elementwise_kernel_4_bfloat_to_float<func_t, size_t><<<grid, block, 0, stream>>>(f, data0, data1);
    } else {
      unrolled_elementwise_kernel_4_bfloat_to_float_no_align<func_t, size_t><<<grid, block, 0, stream>>>(N, f, data0, data1);
    }
  } else if (st0 == ScalarType::Int && st1 == ScalarType::Float) {
    int vt = 4;
    int64_t grid = (N + (block * vt) - 1) / (block * vt);
    if (N % (block * vt) == 0) {
      unrolled_elementwise_kernel_4_float_to_int<func_t, size_t><<<grid, block, 0, stream>>>(f, data0, data1);
    } else {
      unrolled_elementwise_kernel_4_float_to_int_no_align<func_t, size_t><<<grid, block, 0, stream>>>(N, f, data0, data1);
    }
  } else if (st0 == ScalarType::Half && st1 == ScalarType::Int) {
    int vt = 4;
    int64_t grid = (N + (block * vt) - 1) / (block * vt);
    if (N % (block * vt) == 0) {
      unrolled_elementwise_kernel_4_int_to_half<func_t, size_t><<<grid, block, 0, stream>>>(f, data0, data1);
    } else {
      unrolled_elementwise_kernel_4_int_to_half_no_align<func_t, size_t><<<grid, block, 0, stream>>>(N, f, data0, data1);
    }
  } else if (st0 == ScalarType::Long && st1 == ScalarType::Int) {
    int vt = 2;
    int64_t grid = (N + (block * vt) - 1) / (block * vt);
    if (N % (block * vt) == 0) {
      unrolled_elementwise_kernel_2_int_to_long<func_t, size_t><<<grid, block, 0, stream>>>(f, data0, data1);
    } else {
      unrolled_elementwise_kernel_2_int_to_long_no_align<func_t, size_t><<<grid, block, 0, stream>>>(N, f, data0, data1);
    }
  } else if (st0 == ScalarType::Int && st1 == ScalarType::Long) {
    int vt = 2;
    int64_t grid = (N + (block * vt) - 1) / (block * vt);
    if (N % (block * vt) == 0) {
      unrolled_elementwise_kernel_2_long_to_int<func_t, size_t><<<grid, block, 0, stream>>>(f, data0, data1);
    } else {
      unrolled_elementwise_kernel_2_long_to_int_no_align<func_t, size_t><<<grid, block, 0, stream>>>(N, f, data0, data1);
    }
  } else if (st0 == ScalarType::Half && st1 == ScalarType::BFloat16) {
    int vt = 8;
    int64_t grid = (N + (block * vt) - 1) / (block * vt);
    if (N % (block * vt) == 0) {
      unrolled_elementwise_kernel_8_bfloat_to_half<func_t, size_t><<<grid, block, 0, stream>>>(f, data0, data1);
    } else {
      unrolled_elementwise_kernel_8_bfloat_to_half_no_align<func_t, size_t><<<grid, block, 0, stream>>>(N, f, data0, data1);
    }
  } else if (st0 == ScalarType::BFloat16 && st1 == ScalarType::Half) {
    int vt = 8;
    int64_t grid = (N + (block * vt) - 1) / (block * vt);
    if (N % (block * vt) == 0) {
      unrolled_elementwise_kernel_8_half_to_bfloat<func_t, size_t><<<grid, block, 0, stream>>>(f, data0, data1);
    } else {
      unrolled_elementwise_kernel_8_half_to_bfloat_no_align<func_t, size_t><<<grid, block, 0, stream>>>(N, f, data0, data1);
    }
  } else if (st0 == ScalarType::Float && st1 == ScalarType::Long) {
    int vt = 2;
    int64_t grid = (N + (block * vt) - 1) / (block * vt);
    if (N % (block * vt) == 0) {
      unrolled_elementwise_kernel_2_long_to_float<func_t, size_t><<<grid, block, 0, stream>>>(f, data0, data1);
    } else {
      unrolled_elementwise_kernel_2_long_to_float_no_align<func_t, size_t><<<grid, block, 0, stream>>>(N, f, data0, data1);
    }
  } else if (st0 == ScalarType::Bool && st1 == ScalarType::Float) {
    int vt = 4;
    int64_t grid = (N + (block * vt) - 1) / (block * vt);
    if (N % (block * vt) == 0) {
      unrolled_elementwise_kernel_4_float_to_bool<func_t, size_t><<<grid, block, 0, stream>>>(f, data0, data1);
    } else {
      unrolled_elementwise_kernel_4_float_to_bool_no_align<func_t, size_t><<<grid, block, 0, stream>>>(N, f, data0, data1);
    }
  } else if (st0 == ScalarType::Long && st1 == ScalarType::Bool) {
    int vt = 2;
    int64_t grid = (N + (block * vt) - 1) / (block * vt);
    if (N % (block * vt) == 0) {
      unrolled_elementwise_kernel_2_bool_to_long<func_t, size_t><<<grid, block, 0, stream>>>(f, data0, data1);
    } else {
      unrolled_elementwise_kernel_2_bool_to_long_no_align<func_t, size_t><<<grid, block, 0, stream>>>(N, f, data0, data1);
    }
  } else {
    assert(0);
  }

  C10_CUDA_KERNEL_LAUNCH_CHECK();
}


template<int nt, int vt, typename func_t>
C10_LAUNCH_BOUNDS_2(nt, 4)
__global__ void elementwise_kernel(int N, func_t f) {
  int tid = threadIdx.x;
  int nv = nt * vt;
  int idx = nv * blockIdx.x + tid;
  #pragma unroll
  for (int i = 0; i < vt; i++) {
    if (idx < N) {
      f(idx);
      idx += nt;
    }
  }
}

template<int nt, int vt, typename res_t , typename arg0_t, typename func_t, typename index_t, typename stride_t>
C10_LAUNCH_BOUNDS_2(nt, 4)
__global__ void elementwise_kernel_1_1(
    int64_t N,
    char* data0, char* data1,
    index_t size0,
    stride_t stride00, stride_t stride01, 
    func_t f) {
  // ndim = 1, arity = 1, narg = 2
  int tid = threadIdx.x;
  int nv = nt * vt;
  int64_t idx = nv * blockIdx.x + tid;
  #pragma unroll
  for (int i = 0; i < vt; i++) {
    if (idx < N) {
      // ---------------- f(idx);
      // [=]GPU_LAMBDA(int idx) {
      //   auto offsets = offset_calc.get(idx);
      //   arg0_t* out = (arg0_t*)(data[0] + offsets[0]);
      //   *out = invoke(f, &data.data[1], &offsets.data[1], 1);
      // }
      // -----------------------
      int64_t offsets[2];
      auto linear_idx = idx;
      constexpr int NARGS = 2;
      constexpr int MAX_DIMS = 2;
      #pragma unroll
      for (int arg = 0; arg < NARGS; arg++) {
        offsets[arg] = 0;
      }
      // auto divmod_div = linear_idx / size0;
      auto divmod_mod = linear_idx % size0;
      // linear_idx = divmod_div;
      int arg = 0;
      offsets[arg] += divmod_mod * stride00;
      arg = 1;
      offsets[arg] += divmod_mod * stride01;

      res_t* out = (res_t*)(data0 + offsets[0]);
      *out = f(func_reinterpret_cast<arg0_t>(data1 + offsets[1]));

      idx += nt;
    }
  }
}

template<int nt, int vt, int vec, typename res_t , typename arg0_t, typename func_t, typename index_t, typename stride_t>
C10_LAUNCH_BOUNDS_2(nt, 4)
__global__ void elementwise_kernel_1_1_broadcast(
    int64_t N,
    char* data0, char* data1,
    index_t size0,
    stride_t stride00, stride_t stride01, 
    func_t f) {
  // ndim = 1, arity = 1, narg = 2
  int tid = threadIdx.x;
  using StoreT = at::native::memory::aligned_vector<res_t, vec>;

  res_t ld_out[vec];

  StoreT *p_ld_out = reinterpret_cast<StoreT*>(&ld_out);

  arg0_t val = *reinterpret_cast<arg0_t*>(data1);
  int64_t offset = ((blockDim.x * blockIdx.x) * vec + tid * vec) * stride00;
  StoreT* out = reinterpret_cast<StoreT*>(data0 + offset);
  #pragma unroll
  for(int v = 0; v < vec; ++v) {
    ld_out[v] = f(val);
  }
  *out = *p_ld_out;
}

template<int nt, int vt, typename res_t , typename arg0_t, typename func_t, typename index_t, typename stride_t>
C10_LAUNCH_BOUNDS_2(nt, 4)
__global__ void elementwise_kernel_1_1_dilation(
    int64_t N,
    char* data0, char* data1,
    index_t size0,
    stride_t stride00, stride_t stride01,
    func_t f) {

  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx * vt >= N) return;
  using LoadT = at::native::memory::aligned_vector<arg0_t, vt>;
  using StoreT = at::native::memory::aligned_vector<res_t, vt * 2>;

  LoadT val = (reinterpret_cast<LoadT*>(data1))[idx];
  StoreT out = (reinterpret_cast<StoreT*>(data0))[idx];

  #pragma unroll
  for (int i = 0; i < vt; i++) {
    out.val[i * 2] = val.val[i];
  }

  (reinterpret_cast<StoreT*>(data0))[idx] = out;
}


template<int nt, int vt, typename res_t , typename arg0_t, typename func_t, typename index_t, typename stride_t>
C10_LAUNCH_BOUNDS_2(nt, 4)
__global__ void elementwise_kernel_2_1(
    int64_t N,
    char* data0, char* data1,
    index_t size0, index_t size1,
    stride_t stride00, stride_t stride01, 
    stride_t stride10, stride_t stride11, 
    func_t f) {
  // ndim = 2, arity = 1, narg = 2
  int tid = threadIdx.x;
  int nv = nt * vt;
  int64_t idx = nv * blockIdx.x + tid;
  #pragma unroll
  for (int i = 0; i < vt; i++) {
    if (idx < N) {
      // ---------------- f(idx);
      // [=]GPU_LAMBDA(int idx) {
      //   auto offsets = offset_calc.get(idx);
      //   arg0_t* out = (arg0_t*)(data[0] + offsets[0]);
      //   *out = invoke(f, &data.data[1], &offsets.data[1], 1);
      // }
      // -----------------------
      int64_t offsets[2];
      int64_t linear_idx = idx;
      constexpr int NARGS = 2;
      constexpr int MAX_DIMS = 2;
      #pragma unroll
      for (int arg = 0; arg < NARGS; arg++) {
        offsets[arg] = 0;
      }
      auto divmod_div = linear_idx / size0;
      auto divmod_mod = linear_idx % size0;
      linear_idx = divmod_div;
      int arg = 0;
      offsets[arg] += divmod_mod * stride00;
      arg = 1;
      offsets[arg] += divmod_mod * stride01;
      // dim = 1
      divmod_div = linear_idx / size1;
      divmod_mod = linear_idx % size1;
      linear_idx = divmod_div;
      arg = 0;
      offsets[arg] += divmod_mod * stride10;
      arg = 1;
      offsets[arg] += divmod_mod * stride11;

      res_t* out = (res_t*)(data0 + offsets[0]);
      *out = f(func_reinterpret_cast<arg0_t>(data1 + offsets[1]));

      idx += nt;
    }
  }
}

template<int nt, int vt, typename res_t , typename arg0_t, typename func_t, typename index_t, typename stride_t>
C10_LAUNCH_BOUNDS_2(nt, 4)
__global__ void elementwise_kernel_2_1_dim0_contiguous(
    int64_t N,
    char* data0, char* data1,
    index_t size0, index_t size1,
    stride_t stride00, stride_t stride01,
    stride_t stride10, stride_t stride11,
    func_t f) {
  // ndim = 2, arity = 1, narg = 2
  int tid = threadIdx.x;
  int nv = nt * vt;
  int64_t linear_idx = nv * blockIdx.x + tid * vt;

  if (linear_idx < N) {
    int64_t offsets[2];
    constexpr int NARGS = 2;
    constexpr int MAX_DIMS = 2;
    #pragma unroll
    for (int arg = 0; arg < NARGS; arg++) {
      offsets[arg] = 0;
    }
    auto divmod_div = linear_idx / size0;
    auto divmod_mod = linear_idx % size0;
    linear_idx = divmod_div;
    int arg = 0;
    offsets[arg] += divmod_mod * stride00;
    arg = 1;
    offsets[arg] += divmod_mod * stride01;
    // dim = 1
    divmod_div = linear_idx / size1;
    divmod_mod = linear_idx % size1;
    linear_idx = divmod_div;
    arg = 0;
    offsets[arg] += divmod_mod * stride10;
    arg = 1;
    offsets[arg] += divmod_mod * stride11;

    arg0_t tmp_load[vt];
    using LoadT = at::native::memory::aligned_vector<arg0_t, vt>;
    LoadT* p_input1 = reinterpret_cast<LoadT*>(&tmp_load);
    *p_input1 = *reinterpret_cast<LoadT*>(data1 + offsets[1]);

    res_t tmp_store[vt];
    #pragma unroll
    for (int i = 0; i < vt; i++){
      tmp_store[i] = f(tmp_load[i]);
    }

    using StoreT = at::native::memory::aligned_vector<res_t, vt>;
    StoreT * p_store = reinterpret_cast<StoreT*>(&tmp_store);
    StoreT* out = reinterpret_cast<StoreT*>(data0 + offsets[0]);
    *out = *p_store;
  }
}

template<int nt, int vt, typename res_t , typename arg0_t, typename func_t, typename index_t, typename stride_t>
C10_LAUNCH_BOUNDS_2(nt, 4)
__global__ void elementwise_kernel_2_1_dim0_pad(
    int64_t N,
    char* data0, char* data1,
    index_t size0, index_t size1,
    stride_t stride00, stride_t stride01,
    stride_t stride10, stride_t stride11,
    func_t f, int tids_per_group) {
  // ndim = 2, arity = 1, narg = 2
  int tid = threadIdx.x;
  int64_t offsets[2];
  constexpr int NARGS = 2;

  // threads in a group padding to 128B data
  // 40 -> 8 threads
  // 80 -> 16 threads
  // int tids_per_group = ((size0 * sizeof(arg0_t) + 127) / 128) * 128 / (vt * sizeof(arg0_t));
  int num_groups = nt / tids_per_group;
  int group_idx = tid / tids_per_group;
  int tid_idx = tid % tids_per_group;
  int ld1_tid_num = size0 / vt; // threads that need to read data1

  // cal data0 offsets, at least 16B align
  int group_start_addr = (num_groups * blockIdx.x + group_idx) * stride10;
  int group_start_addr_align64 = (group_start_addr / 64) * 64;

  // threads padding to 64B align address, to avoid partial write
  int pad_tid_num = (group_start_addr - group_start_addr_align64) / (vt * sizeof(res_t));

  arg0_t tmp_load[vt];
  using LoadT = at::native::memory::aligned_vector<arg0_t, vt>;
  LoadT* p_input1 = reinterpret_cast<LoadT*>(&tmp_load);

  offsets[0] = group_start_addr_align64 + tid_idx * vt * stride00;

  if (tid_idx >= pad_tid_num && tid_idx < ld1_tid_num + pad_tid_num) {
    offsets[1] = (num_groups * blockIdx.x + group_idx) * stride11 + (tid_idx - pad_tid_num) * vt * stride01;
    *p_input1 = *reinterpret_cast<LoadT*>(data1 + offsets[1]);
  } else {
    *p_input1 = *reinterpret_cast<LoadT*>(data0 + offsets[0]);
  }

  res_t tmp_store[vt];
  #pragma unroll
  for (int i = 0; i < vt; i++){
    tmp_store[i] = f(tmp_load[i]);
  }

  // addr align to 64B, size align to 128B
  using StoreT = at::native::memory::aligned_vector<res_t, vt>;
  StoreT * p_store = reinterpret_cast<StoreT*>(&tmp_store);
  StoreT* out = reinterpret_cast<StoreT*>(data0 + offsets[0]);
  *out = *p_store;
}

template<int nt, int vt, typename res_t , typename arg0_t, typename func_t, typename index_t, typename stride_t>
C10_LAUNCH_BOUNDS_2(nt, 4)
__global__ void elementwise_kernel_2_1_dim0_pad_align(
    int64_t N,
    char* data0, char* data1,
    index_t size0, index_t size1,
    stride_t stride00, stride_t stride01,
    stride_t stride10, stride_t stride11,
    func_t f, int tids_per_group) {
  // ndim = 2, arity = 1, narg = 2
  int tid = threadIdx.x;
  int64_t offsets[2];
  constexpr int NARGS = 2;

  // threads in a group padding to 128B data
  // 40 -> 8 threads
  // 80 -> 16 threads
  // int tids_per_group = ((size0 * sizeof(arg0_t) + 127) / 128) * 128 / (vt * sizeof(arg0_t));
  int num_groups = nt / tids_per_group;
  int group_idx = tid / tids_per_group;
  int tid_idx = tid % tids_per_group;
  int ld1_tid_num = size0 / vt; // threads that need to read data1

  int global_offset = reinterpret_cast<uintptr_t>(data0) % 64;
  // cal data0 offsets, at least 16B align
  int group_start_addr = (num_groups * blockIdx.x + group_idx) * stride10;
  if (group_start_addr > 0) group_start_addr += global_offset;

  int group_start_addr_align64 = (group_start_addr / 64) * 64;

  // threads padding to 64B align address, to avoid partial write
  int pad_tid_num = (group_start_addr - group_start_addr_align64) / (vt * sizeof(res_t));

  arg0_t tmp_load[vt];
  using LoadT = at::native::memory::aligned_vector<arg0_t, vt>;
  LoadT* p_input1 = reinterpret_cast<LoadT*>(&tmp_load);

  offsets[0] = group_start_addr_align64 + tid_idx * vt * stride00;
  if (group_start_addr > 0) offsets[0] -= global_offset;

  if (tid_idx >= pad_tid_num && tid_idx < ld1_tid_num + pad_tid_num) {
    offsets[1] = (num_groups * blockIdx.x + group_idx) * stride11 + (tid_idx - pad_tid_num) * vt * stride01;
    *p_input1 = *reinterpret_cast<LoadT*>(data1 + offsets[1]);
  } else {
    *p_input1 = *reinterpret_cast<LoadT*>(data0 + offsets[0]);
  }

  res_t tmp_store[vt];
  #pragma unroll
  for (int i = 0; i < vt; i++){
    tmp_store[i] = f(tmp_load[i]);
  }

  // addr align to 64B, size align to 128B
  using StoreT = at::native::memory::aligned_vector<res_t, vt>;
  StoreT * p_store = reinterpret_cast<StoreT*>(&tmp_store);
  StoreT* out = reinterpret_cast<StoreT*>(data0 + offsets[0]);
  *out = *p_store;
}

template<int nt, int v_x, typename res_t , typename arg0_t, typename func_t, typename index_t, typename stride_t>
C10_LAUNCH_BOUNDS_2(nt, 4)
__global__ void elementwise_kernel_2_1_broadcast(
    int64_t N,
    char* data0, char* data1,
    index_t size0, index_t size1,
    stride_t stride00, stride_t stride01,
    stride_t stride10, stride_t stride11,
    func_t f, int y_t, int y_remain) {
  // ndim = 2, arity = 1, narg = 2
  // v_x: vectorized load n elems;
  // y_t: store columns for each thread;
  // load_num: load iterations for each thread;
  int tid = threadIdx.x;
  int64_t offsets[2];
  constexpr int NARGS = 2;
  constexpr int MAX_DIMS = 2;
  size_t x_remain = size0 % (blockDim.x * v_x);
  size_t tail = size0 % v_x;
  using LoadT = at::native::memory::aligned_vector<arg0_t, v_x>;
  using StoreT = at::native::memory::aligned_vector<res_t, v_x>;
  #pragma unroll
  for (int arg = 0; arg < NARGS; arg++) {
    offsets[arg] = 0;
  }
  // arg = 0: output; arg = 1: input1; arg = 2: input2;

  arg0_t ld_1[v_x];
  res_t ld_out[v_x];
  LoadT* p_ld_1 = reinterpret_cast<LoadT*>(&ld_1); 
  StoreT * p_ld_out = reinterpret_cast<StoreT*>(&ld_out); 

  size_t y_loop = y_t;
  if (y_remain != 0 && blockIdx.y == (gridDim.y - 1)) {
    y_loop = y_remain;
  }

  offsets[1] = (blockIdx.x * blockDim.x + tid) * v_x * stride01;
  size_t row_offset = (blockIdx.x * blockDim.x + tid) * v_x * stride00;
  *p_ld_1 = *reinterpret_cast<LoadT*>(data1 + offsets[1]);
  for (size_t y_idx = 0; y_idx < y_loop; y_idx++) {
    offsets[0] = (blockIdx.y * y_t + y_idx) * stride10 + row_offset;
    StoreT* out = reinterpret_cast<StoreT*>(data0 + offsets[0]);
    #pragma unroll
    for (int i = 0; i < v_x; i++) {
      ld_out[i] = f(ld_1[i]);
    }
    *out = *p_ld_out;
  }

  if (x_remain != 0 && (tid + 1) * v_x <= x_remain) {
    offsets[1] = (gridDim.x * blockDim.x + tid) * v_x * stride01;
    // TODO(liuyuxin): support or assert lambda function.
    *p_ld_1 = *reinterpret_cast<LoadT*>(data1 + offsets[1]);
    size_t row_offset = (gridDim.x * blockDim.x + tid) * v_x * stride00;
    for (int y_idx = 0; y_idx < y_loop; y_idx++) {
      offsets[0] = (blockIdx.y * y_t + y_idx) * stride10 + row_offset;
      StoreT* out = reinterpret_cast<StoreT*>(data0 + offsets[0]);
      #pragma unroll
      for (int i = 0; i < v_x; i++) {
        ld_out[i] = f(ld_1[i]);
      }
      *out = *p_ld_out;
    }
  }

  // handle tail columns( < v_x)
  if(tail && tid < tail){
    auto remain_offset = x_remain / v_x;
    offsets[1] = ((gridDim.x * blockDim.x + remain_offset) * v_x + tid)* stride01;
    int64_t row_offset = ((gridDim.x  * blockDim.x + remain_offset) * v_x + tid) * stride00 ;
    for (int y_idx = 0; y_idx < y_loop; y_idx++) {
      offsets[0] = (blockIdx.y * y_t + y_idx) * stride10 + row_offset;
      auto p0 = reinterpret_cast<res_t*>(data0 + offsets[0]);
      auto p1 = reinterpret_cast<arg0_t*>(data1 + offsets[1]);
      *p0 = f(*p1);
    }
  }
}

template<int nt, int vt, typename res_t , typename arg0_t, typename func_t, typename index_t, typename stride_t>
C10_LAUNCH_BOUNDS_2(nt, 4)
__global__ void elementwise_kernel_3_1(
    int64_t N,
    char* data0, char* data1,
    index_t size0, index_t size1, index_t size2,
    stride_t stride00, stride_t stride01, 
    stride_t stride10, stride_t stride11,
    stride_t stride20, stride_t stride21,
    func_t f) {
  // ndim = 3, arity = 1, narg = 2
  int tid = threadIdx.x;
  int nv = nt * vt;
  int idx = nv * blockIdx.x + tid;
  #pragma unroll
  for (int i = 0; i < vt; i++) {
    if (idx < N) {
      // ---------------- f(idx);
      // [=]GPU_LAMBDA(int idx) {
      //   auto offsets = offset_calc.get(idx);
      //   arg0_t* out = (arg0_t*)(data[0] + offsets[0]);
      //   *out = invoke(f, &data.data[1], &offsets.data[1], 1);
      // }
      // -----------------------
      stride_t offsets0 = 0;
      stride_t offsets1 = 0;
      auto linear_idx = idx;

      auto divmod_mod = linear_idx % size0;
      linear_idx = linear_idx / size0;
      offsets0 += divmod_mod * stride00;
      offsets1 += divmod_mod * stride01;
      // dim = 1
      divmod_mod = linear_idx % size1;
      linear_idx = linear_idx / size1;
      offsets0 += divmod_mod * stride10;
      offsets1 += divmod_mod * stride11;
      // dim = 2
      divmod_mod = linear_idx % size2;
      linear_idx = linear_idx / size2;
      offsets0 += divmod_mod * stride20;
      offsets1 += divmod_mod * stride21;

      res_t* out = (res_t*)(data0 + offsets0);
      *out = f(func_reinterpret_cast<arg0_t>(data1 + offsets1));

      idx += nt;
    }
  }
}

template<int nt, int vt, typename res_t , typename arg0_t, typename func_t, typename index_t, typename stride_t, int size0, int size1, int size2>
C10_LAUNCH_BOUNDS_2(nt, 4)
__global__ void elementwise_kernel_3_1_s(
    int64_t N,
    char* data0, char* data1,
    stride_t stride00, stride_t stride01, 
    stride_t stride10, stride_t stride11,
    stride_t stride20, stride_t stride21,
    func_t f) {
  // ndim = 3, arity = 1, narg = 2
  int tid = threadIdx.x;
  int nv = nt * vt;
  int64_t idx = nv * blockIdx.x + tid;
  #pragma unroll
  for (int i = 0; i < vt; i++) {
    if (idx < N) {
      // ---------------- f(idx);
      // [=]GPU_LAMBDA(int idx) {
      //   auto offsets = offset_calc.get(idx);
      //   arg0_t* out = (arg0_t*)(data[0] + offsets[0]);
      //   *out = invoke(f, &data.data[1], &offsets.data[1], 1);
      // }
      // -----------------------
      int64_t offsets0 = 0;
      int64_t offsets1 = 0;
      auto linear_idx = idx;

      auto divmod_mod = linear_idx % size0;
      linear_idx = linear_idx / size0;
      offsets0 += divmod_mod * stride00;
      offsets1 += divmod_mod * stride01;
      // dim = 1
      divmod_mod = linear_idx % size1;
      linear_idx = linear_idx / size1;
      offsets0 += divmod_mod * stride10;
      offsets1 += divmod_mod * stride11;
      // dim = 2
      divmod_mod = linear_idx % size2;
      linear_idx = linear_idx / size2;
      offsets0 += divmod_mod * stride20;
      offsets1 += divmod_mod * stride21;

      res_t* out = (res_t*)(data0 + offsets0);
      *out = f(func_reinterpret_cast<arg0_t>(data1 + offsets1));

      idx += nt;
    }
  }
}

template<int nt, int vt, typename res_t , typename arg0_t, typename func_t, typename index_t, typename stride_t>
C10_LAUNCH_BOUNDS_2(nt, 4)
__global__ void elementwise_kernel_4_1_dim0_contiuous(
    int64_t N,
    char* data0, char* data1,
    index_t size0, index_t size1, index_t size2, index_t size3,
    stride_t stride00, stride_t stride01, 
    stride_t stride10, stride_t stride11,
    stride_t stride20, stride_t stride21,
    stride_t stride30, stride_t stride31,
    func_t f) {

  int64_t linear_idx = (blockIdx.x * blockDim.x + threadIdx.x) * vt;
  if (linear_idx >= N) return;

  using vec_res = at::native::memory::aligned_vector<res_t, vt>;
  using vec_arg0 = at::native::memory::aligned_vector<arg0_t, vt>;
  int64_t offset0 = 0, offset1 = 0;

  auto divmod_div = linear_idx / size0;
  auto divmod_mod = linear_idx % size0;
  linear_idx = divmod_div;
  offset0 += divmod_mod * stride00;
  offset1 += divmod_mod * stride01;
  
  divmod_div = linear_idx / size1;
  divmod_mod = linear_idx % size1;
  linear_idx = divmod_div;
  offset0 += divmod_mod * stride10;
  offset1 += divmod_mod * stride11;

  divmod_div = linear_idx / size2;
  divmod_mod = linear_idx % size2;
  linear_idx = divmod_div;
  offset0 += divmod_mod * stride20;
  offset1 += divmod_mod * stride21;

  divmod_div = linear_idx / size3;
  divmod_mod = linear_idx % size3;
  linear_idx = divmod_div;
  offset0 += divmod_mod * stride30;
  offset1 += divmod_mod * stride31;

  vec_arg0 arg1_elems = *(reinterpret_cast<vec_arg0*>(data1 + offset1));
  *(reinterpret_cast<vec_res*>(data0 + offset0)) = arg1_elems;
}


template<int nt, int vt, typename res_t , typename arg0_t, typename func_t, typename index_t, typename stride_t>
C10_LAUNCH_BOUNDS_2(nt, 4)
__global__ void elementwise_kernel_4_1(
    int64_t N,
    char* data0, char* data1,
    index_t size0, index_t size1, index_t size2, index_t size3,
    stride_t stride00, stride_t stride01, 
    stride_t stride10, stride_t stride11,
    stride_t stride20, stride_t stride21,
    stride_t stride30, stride_t stride31,
    func_t f) {
  // ndim = 3, arity = 1, narg = 2
  int tid = threadIdx.x;
  int nv = nt * vt;
  int64_t idx = nv * blockIdx.x + tid;
  #pragma unroll
  for (int i = 0; i < vt; i++) {
    if (idx < N) {
      // ---------------- f(idx);
      // [=]GPU_LAMBDA(int idx) {
      //   auto offsets = offset_calc.get(idx);
      //   arg0_t* out = (arg0_t*)(data[0] + offsets[0]);
      //   *out = invoke(f, &data.data[1], &offsets.data[1], 1);
      // }
      // -----------------------
      int64_t offsets[2];
      auto linear_idx = idx;
      constexpr int NARGS = 2;
      constexpr int MAX_DIMS = 4;
      #pragma unroll
      for (int arg = 0; arg < NARGS; arg++) {
        offsets[arg] = 0;
      }

      auto divmod_div = linear_idx / size0;
      auto divmod_mod = linear_idx % size0;
      linear_idx = divmod_div;
      int arg = 0;
      offsets[arg] += divmod_mod * stride00;
      arg = 1;
      offsets[arg] += divmod_mod * stride01;
      // dim = 1
      divmod_div = linear_idx / size1;
      divmod_mod = linear_idx % size1;
      linear_idx = divmod_div;
      arg = 0;
      offsets[arg] += divmod_mod * stride10;
      arg = 1;
      offsets[arg] += divmod_mod * stride11;
      // dim = 2
      divmod_div = linear_idx / size2;
      divmod_mod = linear_idx % size2;
      linear_idx = divmod_div;
      arg = 0;
      offsets[arg] += divmod_mod * stride20;
      arg = 1;
      offsets[arg] += divmod_mod * stride21;
      // dim = 3
      divmod_div = linear_idx / size3;
      divmod_mod = linear_idx % size3;
      linear_idx = divmod_div;
      arg = 0;
      offsets[arg] += divmod_mod * stride30;
      arg = 1;
      offsets[arg] += divmod_mod * stride31;

      res_t* out = (res_t*)(data0 + offsets[0]);
      *out = f(func_reinterpret_cast<arg0_t>(data1 + offsets[1]));

      idx += nt;
    }
  }
}

template<int nt, int vt, typename res_t , typename arg0_t, typename func_t, typename index_t, typename stride_t>
C10_LAUNCH_BOUNDS_2(nt, 4)
__global__ void elementwise_kernel_4_1_input_lowdim_contiuous(
    int64_t N,
    char* data0, char* data1,
    index_t size0, index_t size1, index_t size2, index_t size3,
    stride_t stride00, stride_t stride01,
    stride_t stride10, stride_t stride11,
    stride_t stride20, stride_t stride21,
    stride_t stride30, stride_t stride31,
    func_t f) {
  using vec_t = at::native::memory::aligned_vector<arg0_t, vt>;

  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t linear_idx = idx * vt;

  if (linear_idx >= N) return;

  int64_t offset = 0;

  auto divmod_div = linear_idx / size0;
  auto divmod_mod = linear_idx % size0;
  linear_idx = divmod_div;
  offset += divmod_mod * stride01;

  divmod_div = linear_idx / size1;
  divmod_mod = linear_idx % size1;
  linear_idx = divmod_div;
  offset += divmod_mod * stride11;

  divmod_div = linear_idx / size2;
  divmod_mod = linear_idx % size2;
  linear_idx = divmod_div;
  offset += divmod_mod * stride21;

  divmod_div = linear_idx / size3;
  divmod_mod = linear_idx % size3;
  linear_idx = divmod_div;
  offset += divmod_mod * stride31;

  (reinterpret_cast<vec_t*>(data0))[idx] = *(reinterpret_cast<vec_t*>(data1 + offset));
}

template<int nt, int vt, typename res_t , typename arg0_t, typename func_t, typename index_t, typename stride_t>
C10_LAUNCH_BOUNDS_2(nt, 4)
__global__ void elementwise_kernel_4_1_dim0_pad(
    int64_t N,
    char* data0, char* data1,
    index_t size0, index_t size1, index_t size2, index_t size3,
    stride_t stride00, stride_t stride01, 
    stride_t stride10, stride_t stride11,
    stride_t stride20, stride_t stride21,
    stride_t stride30, stride_t stride31,
    func_t f, int tids_per_group) {
  // ndim = 4, arity = 1, narg = 2
  int tid = threadIdx.x;
  int64_t offsets[2];
  constexpr int NARGS = 2;
  int s3_idx = blockIdx.y / size2;
  int s2_idx = blockIdx.y % size2;

  // threads in a group padding to 128B data
  // 40 -> 8 threads
  // 80 -> 16 threads
  // int tids_per_group = ((size0 * sizeof(arg0_t) + 127) / 128) * 128 / (vt * sizeof(arg0_t));
  int num_groups = nt / tids_per_group;
  int group_idx = tid / tids_per_group;
  int tid_idx = tid % tids_per_group;
  int ld1_tid_num = size0 / vt; // threads that need to read data1  
  
  int group_start_addr = s3_idx * stride30 + s2_idx * stride20 + (num_groups * blockIdx.x + group_idx) * stride10;
  int group_start_addr_align64 = (group_start_addr / 64) * 64;
  // threads padding to 64B align address, to avoid partial write
  int pad_tid_num = (group_start_addr - group_start_addr_align64) / (vt * sizeof(res_t));

  arg0_t tmp_load[vt];
  using LoadT = at::native::memory::aligned_vector<arg0_t, vt>;
  LoadT* p_input1 = reinterpret_cast<LoadT*>(&tmp_load);

  offsets[0] = group_start_addr_align64 + tid_idx * vt * stride00;
  *p_input1 = *reinterpret_cast<LoadT*>(data0 + offsets[0]);

  if (tid_idx >= pad_tid_num && tid_idx < ld1_tid_num + pad_tid_num) {
    offsets[1] = s3_idx * stride31 + s2_idx * stride21 + (num_groups * blockIdx.x + group_idx) * stride11 + (tid_idx - pad_tid_num) * vt * stride01;
    *p_input1 = *reinterpret_cast<LoadT*>(data1 + offsets[1]);
  }

  res_t tmp_store[vt];
  #pragma unroll
  for (int i = 0; i < vt; i++) {
    tmp_store[i] = f(tmp_load[i]);
  }

  // addr align to 64B, size align to 128B
  using StoreT = at::native::memory::aligned_vector<res_t, vt>;
  StoreT * p_store = reinterpret_cast<StoreT*>(&tmp_store);
  StoreT* out = reinterpret_cast<StoreT*>(data0 + offsets[0]);
  *out = *p_store;  
}

template<int nt, int vt, typename res_t , typename arg0_t, typename func_t, typename index_t, typename stride_t>
C10_LAUNCH_BOUNDS_2(nt, 4)
__global__ void elementwise_kernel_4_1_dim0_pad_align(
    int64_t N,
    char* data0, char* data1,
    index_t size0, index_t size1, index_t size2, index_t size3,
    stride_t stride00, stride_t stride01, 
    stride_t stride10, stride_t stride11,
    stride_t stride20, stride_t stride21,
    stride_t stride30, stride_t stride31,
    func_t f, int tids_per_group) {
  // ndim = 4, arity = 1, narg = 2
  int tid = threadIdx.x;
  int64_t offsets[2];
  constexpr int NARGS = 2;
  int s3_idx = blockIdx.y / size2;
  int s2_idx = blockIdx.y % size2;

  // threads in a group padding to 128B data
  // 40 -> 8 threads
  // 80 -> 16 threads
  // int tids_per_group = ((size0 * sizeof(arg0_t) + 127) / 128) * 128 / (vt * sizeof(arg0_t));
  int num_groups = nt / tids_per_group;
  int group_idx = tid / tids_per_group;
  int tid_idx = tid % tids_per_group;
  int ld1_tid_num = size0 / vt; // threads that need to read data1

  int global_offset = reinterpret_cast<uintptr_t>(data0) % 64;
  int group_start_addr = s3_idx * stride30 + s2_idx * stride20 + (num_groups * blockIdx.x + group_idx) * stride10;
  if (group_start_addr > 0) group_start_addr += global_offset;

  int group_start_addr_align64 = (group_start_addr / 64) * 64;
  // threads padding to 64B align address, to avoid partial write
  int pad_tid_num = (group_start_addr - group_start_addr_align64) / (vt * sizeof(res_t));

  arg0_t tmp_load[vt];
  using LoadT = at::native::memory::aligned_vector<arg0_t, vt>;
  LoadT* p_input1 = reinterpret_cast<LoadT*>(&tmp_load);

  offsets[0] = group_start_addr_align64 + tid_idx * vt * stride00;
  if (group_start_addr > 0) offsets[0] -= global_offset;

  if (tid_idx >= pad_tid_num && tid_idx < ld1_tid_num + pad_tid_num) {
    offsets[1] = s3_idx * stride31 + s2_idx * stride21 + (num_groups * blockIdx.x + group_idx) * stride11 + (tid_idx - pad_tid_num) * vt * stride01;
    *p_input1 = *reinterpret_cast<LoadT*>(data1 + offsets[1]);
  } else {
    *p_input1 = *reinterpret_cast<LoadT*>(data0 + offsets[0]);
  }

  res_t tmp_store[vt];
  #pragma unroll
  for (int i = 0; i < vt; i++) {
    tmp_store[i] = f(tmp_load[i]);
  }

  // addr align to 64B, size align to 128B
  using StoreT = at::native::memory::aligned_vector<res_t, vt>;
  StoreT * p_store = reinterpret_cast<StoreT*>(&tmp_store);
  StoreT* out = reinterpret_cast<StoreT*>(data0 + offsets[0]);
  *out = *p_store;  
}

#define MACA_FETCH_AND_CAST_CASE_ARITY1(type, scalartype)                             \
  case ScalarType::scalartype:                                                        \
    result = f(c10::convert<arg0_t>(*reinterpret_cast<type*>(data1 + offsets[1])));

#define MACA_CAST_AND_STORE(type, scalartype)                             \
  case ScalarType::scalartype:                                                        \
    *(type*)out = c10::convert<type>(result);

template<int nt, int vt, typename res_t , typename arg0_t, typename func_t, typename index_t, typename stride_t>
C10_LAUNCH_BOUNDS_2(nt, 4)
__global__ void elementwise_kernel_1_1_cast(
    int64_t N,
    char* data0, char* data1,
    ScalarType st0, ScalarType st1,
    index_t size0,
    stride_t stride00, stride_t stride01, 
    func_t f) {
  // ndim = 1, arity = 1, narg = 2
  int tid = threadIdx.x;
  int nv = nt * vt;
  int64_t idx = nv * blockIdx.x + tid;
  #pragma unroll
  for (int i = 0; i < vt; i++) {
    if (idx < N) {
      // ---------------- f(idx);
      //    auto offsets = offset_calc.get(idx);
      //    void* out = data[0] + offsets[0];
      //    arg0_t result = invoke(f, &data.data[1], &offsets.data[1], &dtypes.data[1], 1);
      //    c10::cast_and_store<arg0_t>(dtypes[0], out, result);
      // -----------------------
      int64_t offsets[2];
      auto linear_idx = idx;
      constexpr int NARGS = 2;
      constexpr int MAX_DIMS = 2;
      #pragma unroll
      for (int arg = 0; arg < NARGS; arg++) {
        offsets[arg] = 0;
      }
      auto divmod_div = linear_idx / size0;
      auto divmod_mod = linear_idx % size0;
      linear_idx = divmod_div;
      int arg = 0;
      offsets[arg] += divmod_mod * stride00;
      arg = 1;
      offsets[arg] += divmod_mod * stride01;

      void* out = data0 + offsets[0];

      res_t result = f(c10::fetch_and_cast<arg0_t>(st1, data1 + offsets[1]));
      c10::cast_and_store<res_t>(st0, out, result);

      idx += nt;
    }
  }
}

template<int nt, int vt, typename res_t , typename arg0_t, typename func_t, typename index_t, typename stride_t>
C10_LAUNCH_BOUNDS_2(nt, 4)
__global__ void elementwise_kernel_2_1_cast(
    int64_t N,
    char* data0, char* data1,
    ScalarType st0, ScalarType st1,
    index_t size0, index_t size1,
    stride_t stride00, stride_t stride01,
    stride_t stride10, stride_t stride11,
    func_t f) {
  // ndim = 2, arity = 1, narg = 2
  int tid = threadIdx.x;
  int nv = nt * vt;
  int64_t idx = nv * blockIdx.x + tid;
  #pragma unroll
  for (int i = 0; i < vt; i++) {
    if (idx < N) {
      int64_t offsets[2];
      int64_t linear_idx = idx;
      constexpr int NARGS = 2;
      constexpr int MAX_DIMS = 2;
      #pragma unroll
      for (int arg = 0; arg < NARGS; arg++) {
        offsets[arg] = 0;
      }
      auto divmod_div = linear_idx / size0;
      auto divmod_mod = linear_idx % size0;
      linear_idx = divmod_div;
      int arg = 0;
      offsets[arg] += divmod_mod * stride00;
      arg = 1;
      offsets[arg] += divmod_mod * stride01;
      // dim = 1
      divmod_div = linear_idx / size1;
      divmod_mod = linear_idx % size1;
      linear_idx = divmod_div;
      arg = 0;
      offsets[arg] += divmod_mod * stride10;
      arg = 1;
      offsets[arg] += divmod_mod * stride11;

      void* out = data0 + offsets[0];

      res_t result = f(c10::fetch_and_cast<arg0_t>(st1, data1 + offsets[1]));
      c10::cast_and_store<res_t>(st0, out, result);

      idx += nt;
    }
  }
}

template<int nt, int vt, typename res_t , typename arg0_t, typename func_t, typename index_t, typename stride_t,
         typename res_dtype_t=float, typename arg0_dtype_t=float>
__global__ void elementwise_kernel_2_1_cast_dim0_contiguous(
    int64_t N,
    char* data0, char* data1,
    ScalarType st0, ScalarType st1,
    index_t size0, index_t size1,
    stride_t stride00, stride_t stride01,
    stride_t stride10, stride_t stride11,
    func_t f) {
  int tid = threadIdx.x;
  int nv = nt * vt;
  int64_t linear_idx = nv * blockIdx.x + tid * vt;

  if (linear_idx < N) {
    int64_t offsets[2];
    #pragma unroll
    for (int arg = 0; arg < 2; arg++) {
      offsets[arg] = 0;
    }
    auto divmod_div = linear_idx / size0;
    auto divmod_mod = linear_idx % size0;
    linear_idx = divmod_div;
    int arg = 0;
    offsets[arg] += divmod_mod * stride00;
    arg = 1;
    offsets[arg] += divmod_mod * stride01;
    // dim = 1
    divmod_div = linear_idx / size1;
    divmod_mod = linear_idx % size1;
    linear_idx = divmod_div;
    arg = 0;
    offsets[arg] += divmod_mod * stride10;
    arg = 1;
    offsets[arg] += divmod_mod * stride11;

    using LoadT = at::native::memory::aligned_vector<arg0_dtype_t, vt>;
    using StoreT = at::native::memory::aligned_vector<res_dtype_t, vt>;

    LoadT p_ld0 = *(reinterpret_cast<LoadT*>(data1 + offsets[1]));
    StoreT p_sr;
    #pragma unroll
    for (int i = 0; i < vt; i++) {
      p_sr.val[i] = c10::convert<res_dtype_t>(p_ld0.val[i]);
    }

    *(reinterpret_cast<StoreT*>(data0 + offsets[0])) = p_sr;
  }

}

template<int nt, int vt, typename func_t>
static void launch_legacy_kernel(int64_t N, const func_t& f) {
  TORCH_INTERNAL_ASSERT(N >= 0 && N <= std::numeric_limits<int32_t>::max());
  if (N == 0) {
    return;
  }
  dim3 block(nt);
  dim3 grid((N + block.x * vt - 1) / (block.x * vt));
  auto stream = at::cuda::getCurrentCUDAStream();
  elementwise_kernel<nt, vt, func_t><<<grid, block, 0, stream>>>(N, f);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename traits, typename func_t, typename index_t, size_t... INDEX>
C10_HOST_DEVICE typename traits::result_type
invoke_impl(const func_t &f, char *const C10_RESTRICT data[], const index_t strides[], int i,
            std::index_sequence<INDEX...>) {
  (void)strides;
  (void)i;
  return f(c10::load<typename traits::template arg<INDEX>::type>(data[INDEX] + i * strides[INDEX])...);
}

template <typename func_t, typename index_t, typename traits = function_traits<func_t>>
C10_HOST_DEVICE typename traits::result_type
invoke(const func_t &f, char *const C10_RESTRICT data[], const index_t strides[], int i) {
  using Indices = std::make_index_sequence<traits::arity>;
  return invoke_impl<traits>(f, data, strides, i, Indices{});
}

template <typename traits, typename func_t, typename index_t, size_t... I>
C10_HOST_DEVICE typename traits::result_type
invoke_impl(const func_t &f, char *const C10_RESTRICT data[], const index_t strides[], const ScalarType dtypes[], int i,
            std::index_sequence<I...>) {
  (void)strides;
  (void)i;
  return f(c10::fetch_and_cast<typename traits::template arg<I>::type>(dtypes[I], data[I] + i * strides[I])...);
}

template <typename func_t, typename index_t, typename traits = function_traits<func_t>>
C10_HOST_DEVICE typename traits::result_type
invoke(const func_t &f, char *const C10_RESTRICT data[], const index_t strides[], const ScalarType dtypes[], int i) {
  using Indices = std::make_index_sequence<traits::arity>;
  return invoke_impl<traits>(f, data, strides, dtypes, i, Indices{});
}


template <typename func_t>
void gpu_kernel_impl(TensorIteratorBase& iter, const func_t& f) {
  using traits = function_traits<func_t>;
  using arg0_t = typename traits::result_type;
  constexpr int ntensors = traits::arity + 1;

  TORCH_INTERNAL_ASSERT(iter.can_use_32bit_indexing());
  TORCH_INTERNAL_ASSERT(iter.ninputs() == traits::arity);
  TORCH_INTERNAL_ASSERT(iter.noutputs() == 1);

  at::detail::Array<char*, ntensors> data;
  at::detail::Array<ScalarType, ntensors> dtypes;
  for (int i = 0; i < ntensors; i++) {
    data[i] = (char*)iter.data_ptr(i);
    dtypes[i] = iter.dtype(i);
  }

  int64_t numel = iter.numel();
  constexpr int narity = traits::arity;
  int ndim = iter.ndim();

  bool contiguous = iter.is_contiguous();
  bool dynamic_casting = needs_dynamic_casting<func_t>::check(iter);
  auto offset_calc = ::make_offset_calculator<traits::arity + 1>(iter);
  constexpr int unroll_factor = sizeof(arg0_t) >= 4 ? 2 : 4;

  if (!dynamic_casting) {
    if (contiguous) {
      get_elementwise_info<narity + 1>(ndim, narity, offset_calc, dtypes, "p_e_noopt_cp_vec_1_2", f);
      launch_vectorized_kernel(numel, f, data);
    } else {
        get_elementwise_info<narity + 1>(ndim, narity, offset_calc, "p_e_noopt_cp_launch_legacy_kernel", f);
        launch_legacy_kernel<128,unroll_factor>(numel, [=]GPU_LAMBDA(int idx) {
        auto offsets = offset_calc.get(idx);
        arg0_t* out = (arg0_t*)(data[0] + offsets[0]);
        *out = invoke(f, &data.data[1], &offsets.data[1], 1);
        });
    }
  } else {
    if (contiguous) {
      auto loader = memory::LoadWithCast<traits::arity>(iter);
      auto storer = memory::StoreWithCast<1>(iter);
      auto input_offset_calculator = TrivialOffsetCalculator<traits::arity>();
      auto output_offset_calculator = TrivialOffsetCalculator<1>();
      get_elementwise_info<narity + 1>(ndim, narity, offset_calc, dtypes, "p_e_noopt_unroll_1_2", f);
      launch_unrolled_kernel(numel, f, data, input_offset_calculator, output_offset_calculator, loader, storer);
    } else {
      get_elementwise_info<narity + 1>(ndim, narity, offset_calc, dtypes, "p_e_noopt_cp_launch_legacy_kernel", f);
      launch_legacy_kernel<128, 4>(numel, [=]GPU_LAMBDA(int idx) {
        auto offsets = offset_calc.get(idx);
        void* out = data[0] + offsets[0];
        arg0_t result = invoke(f, &data.data[1], &offsets.data[1], &dtypes.data[1], 1);
        c10::cast_and_store<arg0_t>(dtypes[0], out, result);
      });
    }
  }
}

template<int nt, int vt, typename res_t , typename arg0_t, typename func_t, typename index_t, typename stride_t>
static void launch_legacy_kernel_maca_1_1(
    int64_t N,
    char* data0, char* data1,
    index_t size0,
    stride_t stride00, stride_t stride01, 
    const func_t& f) {
  // ndim = 1, arity = 1
  TORCH_INTERNAL_ASSERT(N >= 0 && N <= std::numeric_limits<int32_t>::max());
  if (N == 0) {
    return;
  }
  dim3 block(nt);
  dim3 grid((N + block.x * vt - 1) / (block.x * vt));
  auto stream = at::cuda::getCurrentCUDAStream();
  elementwise_kernel_1_1<nt, vt, res_t, arg0_t, func_t, index_t, stride_t><<<grid, block, 0, stream>>>(
      N, data0, data1, size0, stride00, stride01, f);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template<int nt, int vt, typename res_t , typename arg0_t, typename func_t, typename index_t, typename stride_t>
static void launch_legacy_kernel_maca_1_1_broadcast(
    int64_t N,
    char* data0, char* data1,
    index_t size0,
    stride_t stride00, stride_t stride01, 
    const func_t& f) {
  // ndim = 1, arity = 1
  TORCH_INTERNAL_ASSERT(N >= 0 && N <= std::numeric_limits<int32_t>::max());
  if (N == 0) {
    return;
  }

  int block_dim_x = get_block_size<index_t>(size0);
  size_t load_num = size0 / block_dim_x;
  TORCH_INTERNAL_ASSERT(load_num > 0);
  int vec = getVectorizedAlignment<res_t>((void*)data0, load_num);
  dim3 block(block_dim_x);
  size_t grid_dim_x = load_num / vec;
  dim3 grid(grid_dim_x);

  auto stream = at::cuda::getCurrentCUDAStream();
  if (vec == 8) {
    elementwise_kernel_1_1_broadcast<nt, vt, 8, res_t, arg0_t, func_t, index_t, stride_t><<<grid, block, 0, stream>>>(
      N, data0, data1, size0, stride00, stride01, f);
  } else if(vec == 4) {
    elementwise_kernel_1_1_broadcast<nt, vt, 4, res_t, arg0_t, func_t, index_t, stride_t><<<grid, block, 0, stream>>>(
      N, data0, data1, size0, stride00, stride01, f);
  } else if(vec == 2) {
    elementwise_kernel_1_1_broadcast<nt, vt, 2, res_t, arg0_t, func_t, index_t, stride_t><<<grid, block, 0, stream>>>(
      N, data0, data1, size0, stride00, stride01, f);
  } else {
    elementwise_kernel_1_1_broadcast<nt, vt, 1, res_t, arg0_t, func_t, index_t, stride_t><<<grid, block, 0, stream>>>(
      N, data0, data1, size0, stride00, stride01, f);
  }

  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template<int nt, int vt, typename res_t , typename arg0_t, typename func_t, typename index_t, typename stride_t>
static void launch_legacy_kernel_maca_1_1_dilation(
    int64_t N,
    char* data0, char* data1,
    index_t size0,
    stride_t stride00, stride_t stride01, 
    const func_t& f) {
  // ndim = 1, arity = 1
  TORCH_INTERNAL_ASSERT(N >= 0 && N <= std::numeric_limits<int32_t>::max());
  if (N == 0) {
    return;
  }

  dim3 block(nt);
  dim3 grid((N + block.x * vt - 1) / (block.x * vt));
  auto stream = at::cuda::getCurrentCUDAStream();
  elementwise_kernel_1_1_dilation<nt, vt, res_t, arg0_t, func_t, index_t, stride_t><<<grid, block, 0, stream>>>(
     N, data0, data1, size0, stride00, stride01, f
  );

  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template<int nt, int vt, typename res_t , typename arg0_t, typename func_t, typename index_t, typename stride_t>
static void launch_legacy_kernel_maca_1_1_cast(
    int64_t N,
    char* data0, char* data1,
    ScalarType st0, ScalarType st1,
    index_t size0,
    stride_t stride00, stride_t stride01, 
    const func_t& f) {
  // ndim = 1, arity = 1
  TORCH_INTERNAL_ASSERT(N >= 0 && N <= std::numeric_limits<int32_t>::max());
  if (N == 0) {
    return;
  }
  dim3 block(nt);
  dim3 grid((N + block.x * vt - 1) / (block.x * vt));
  auto stream = at::cuda::getCurrentCUDAStream();
  elementwise_kernel_1_1_cast<nt, vt, res_t, arg0_t, func_t, index_t, stride_t><<<grid, block, 0, stream>>>(
      N, data0, data1, st0, st1, size0, stride00, stride01, f);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template<int nt, int vt, typename res_t , typename arg0_t, typename func_t, typename index_t, typename stride_t>
static void launch_legacy_kernel_maca_2_1_cast(
    int64_t N,
    char* data0, char* data1,
    ScalarType st0, ScalarType st1,
    index_t size0, index_t size1,
    stride_t stride00, stride_t stride01,
    stride_t stride10, stride_t stride11,
    const func_t& f) {
  // ndim = 2, arity = 1
  TORCH_INTERNAL_ASSERT(N >= 0 && N <= std::numeric_limits<int32_t>::max());
  if (N == 0) {
    return;
  }
  dim3 block(nt);
  dim3 grid((N + block.x * vt - 1) / (block.x * vt));
  auto stream = at::cuda::getCurrentCUDAStream();
  elementwise_kernel_2_1_cast<nt, vt, res_t, arg0_t, func_t, index_t, stride_t><<<grid, block, 0, stream>>>(
    N,
    data0, data1,
    st0, st1,
    size0, size1,
    stride00, stride01,
    stride10, stride11,
    f);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template<int nt, int vt, typename res_t , typename arg0_t, typename func_t, typename index_t, typename stride_t>
static void launch_legacy_kernel_maca_2_1_cast_dim0_contiguous(
    int64_t N,
    char* data0, char* data1,
    ScalarType st0, ScalarType st1,
    index_t size0, index_t size1,
    stride_t stride00, stride_t stride01,
    stride_t stride10, stride_t stride11,
    const func_t& f) {
  // ndim = 2, arity = 1
  TORCH_INTERNAL_ASSERT(N >= 0 && N <= std::numeric_limits<int32_t>::max());
  if (N == 0) {
    return;
  }
  dim3 block(nt);
  dim3 grid((N + block.x * vt - 1) / (block.x * vt));
  auto stream = at::cuda::getCurrentCUDAStream();

  if (st0 == ScalarType::Float && st1 == ScalarType::Half) {
    elementwise_kernel_2_1_cast_dim0_contiguous<nt, vt, res_t, arg0_t, func_t, index_t, stride_t,float, at::Half>
    <<<grid, block, 0, stream>>>(
      N,
      data0, data1,
      st0, st1,
      size0, size1,
      stride00, stride01,
      stride10, stride11,
      f);
  } else if (st0 == ScalarType::Float && st1 == ScalarType::BFloat16) {
    elementwise_kernel_2_1_cast_dim0_contiguous<nt, vt, res_t, arg0_t, func_t, index_t, stride_t,float, at::BFloat16>
    <<<grid, block, 0, stream>>>(
      N,
      data0, data1,
      st0, st1,
      size0, size1,
      stride00, stride01,
      stride10, stride11,
      f);
  } else {
    assert(0);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template<int nt, int vt, typename res_t , typename arg0_t, typename func_t, typename index_t, typename stride_t>
static void launch_legacy_kernel_maca_2_1(
    int64_t N,
    char* data0, char* data1,
    index_t size0, index_t size1,
    stride_t stride00, stride_t stride01, 
    stride_t stride10, stride_t stride11, 
    const func_t& f) {
  // ndim = 2, arity = 1
  TORCH_INTERNAL_ASSERT(N >= 0 && N <= std::numeric_limits<int32_t>::max());
  if (N == 0) {
    return;
  }
  dim3 block(nt);
  dim3 grid((N + block.x * vt - 1) / (block.x * vt));
  auto stream = at::cuda::getCurrentCUDAStream();
  elementwise_kernel_2_1<nt, vt, res_t, arg0_t, func_t, index_t, stride_t><<<grid, block, 0, stream>>>(
      N, data0, data1, size0, size1, stride00, stride01, stride10, stride11, f);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template<int nt, int vt, typename res_t , typename arg0_t, typename func_t, typename index_t, typename stride_t>
static void launch_legacy_kernel_maca_2_1_dim0_pad(
    int64_t N,
    char* data0, char* data1,
    index_t size0, index_t size1,
    stride_t stride00, stride_t stride01, 
    stride_t stride10, stride_t stride11, 
    const func_t& f) {
  // ndim = 2, arity = 1
  TORCH_INTERNAL_ASSERT(N >= 0 && N <= std::numeric_limits<int32_t>::max());
  if (N == 0) {
    return;
  }

  int tids_per_group = ((size0 * sizeof(arg0_t) + 127) / 128) * 128 / (vt * sizeof(arg0_t));
  dim3 block(nt);
  dim3 grid(size1 / (nt / tids_per_group));
  auto stream = at::cuda::getCurrentCUDAStream();
  if (reinterpret_cast<uintptr_t>(data0) % 64 == 0) {
    elementwise_kernel_2_1_dim0_pad<nt, vt, res_t, arg0_t, func_t, index_t, stride_t><<<grid, block, 0, stream>>>(
        N, data0, data1, size0, size1, stride00, stride01, stride10, stride11, f, tids_per_group);
  } else {
    elementwise_kernel_2_1_dim0_pad_align<nt, vt, res_t, arg0_t, func_t, index_t, stride_t><<<grid, block, 0, stream>>>(
        N, data0, data1, size0, size1, stride00, stride01, stride10, stride11, f, tids_per_group);    
  }

  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template<int nt, int vt, typename res_t , typename arg0_t, typename func_t, typename index_t, typename stride_t>
static void launch_legacy_kernel_maca_2_1_dim0_contiguous(
    int64_t N,
    char* data0, char* data1,
    index_t size0, index_t size1,
    stride_t stride00, stride_t stride01,
    stride_t stride10, stride_t stride11,
    const func_t& f) {
  // ndim = 2, arity = 1
  TORCH_INTERNAL_ASSERT(N >= 0 && N <= std::numeric_limits<int32_t>::max());
  if (N == 0) {
    return;
  }
  auto stream = at::cuda::getCurrentCUDAStream();
  dim3 block(nt);
  dim3 grid;
  if (size0 % (nt * 8) == 0 && sizeof(arg0_t) != 4){
    grid = dim3((N + block.x * 8 - 1) / (block.x * 8));
    elementwise_kernel_2_1_dim0_contiguous<nt, 8, res_t, arg0_t, func_t, index_t, stride_t><<<grid, block, 0, stream>>>(
      N, data0, data1, size0, size1, stride00, stride01, stride10, stride11, f);
  }
  else if(size0 % (nt * 4) == 0){
    grid = dim3((N + block.x * 4 - 1) / (block.x * 4));
    elementwise_kernel_2_1_dim0_contiguous<nt, 4, res_t, arg0_t, func_t, index_t, stride_t><<<grid, block, 0, stream>>>(
      N, data0, data1, size0, size1, stride00, stride01, stride10, stride11, f);
  }
  else if(size0 % (nt * 2) == 0){
    grid = dim3((N + block.x * 2 - 1) / (block.x * 2));
    elementwise_kernel_2_1_dim0_contiguous<nt, 2, res_t, arg0_t, func_t, index_t, stride_t><<<grid, block, 0, stream>>>(
      N, data0, data1, size0, size1, stride00, stride01, stride10, stride11, f);
  }
  else if(size0 % (nt * 1) == 0){
    grid = dim3((N + block.x * 1 - 1) / (block.x * 1));
    elementwise_kernel_2_1_dim0_contiguous<nt, 1, res_t, arg0_t, func_t, index_t, stride_t><<<grid, block, 0, stream>>>(
      N, data0, data1, size0, size1, stride00, stride01, stride10, stride11, f);
  }
  // Add condition from size0 % vt == 0 to support more shape mod(64)!=0
  // TODO: may trigger partial write
  else if (size0 % 8 == 0 && sizeof(arg0_t) != 4){
    grid = dim3((N + block.x * 8 - 1) / (block.x * 8));
    elementwise_kernel_2_1_dim0_contiguous<nt, 8, res_t, arg0_t, func_t, index_t, stride_t><<<grid, block, 0, stream>>>(
      N, data0, data1, size0, size1, stride00, stride01, stride10, stride11, f);
  }
  else if(size0 % 4 == 0){
    grid = dim3((N + block.x * 4 - 1) / (block.x * 4));
    elementwise_kernel_2_1_dim0_contiguous<nt, 4, res_t, arg0_t, func_t, index_t, stride_t><<<grid, block, 0, stream>>>(
      N, data0, data1, size0, size1, stride00, stride01, stride10, stride11, f);
  }
  else if(size0 % 2 == 0){
    grid = dim3((N + block.x * 2 - 1) / (block.x * 2));
    elementwise_kernel_2_1_dim0_contiguous<nt, 2, res_t, arg0_t, func_t, index_t, stride_t><<<grid, block, 0, stream>>>(
      N, data0, data1, size0, size1, stride00, stride01, stride10, stride11, f);
  }
  else{
    TORCH_INTERNAL_ASSERT(0);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template<int nt, int vt, typename res_t, typename arg0_t, typename func_t, typename index_t, typename stride_t>
static void launch_legacy_kernel_maca_2_1_broadcast(
    int64_t N,
    char* data0, char* data1,
    index_t size0, index_t size1,
    stride_t stride00, stride_t stride01, 
    stride_t stride10, stride_t stride11, 
    const func_t& f) {
  TORCH_INTERNAL_ASSERT(N >= 0 && N <= std::numeric_limits<int32_t>::max());
  // TORCH_INTERNAL_ASSERT((sizeof(arg0_t) == 2 && sizeof(res_t) == 2) || (sizeof(arg0_t) == 4 && sizeof(res_t) == 4));

  if (N == 0) {
    return;
  }
  auto stream = at::cuda::getCurrentCUDAStream();
  constexpr int block_dim_x = 64;
  size_t load_num = size0 / block_dim_x;
  TORCH_INTERNAL_ASSERT(load_num > 0);
  int vec_data_0 = getVectorizedAlignment<res_t>((void*)data0, load_num);
  if (std::is_same<res_t, at::Half>::value || std::is_same<res_t, at::BFloat16>::value) {
    vec_data_0 = std::min(vec_data_0, getVectorizedAlignment<res_t>((void*)data0, stride10/sizeof(res_t)));
  }
  int vec_data_1 = getVectorizedAlignment<arg0_t>((void*)data1, load_num);
  // if (std::is_same<arg0_t, at::Half>::value || std::is_same<arg0_t, at::BFloat16>::value) {
  //   vec_data_1 = std::min(vec_data_1, getVectorizedAlignment<arg0_t>((void*)data1, stride01/sizeof(arg0_t)));
  // }
  int vec = std::min(vec_data_0, vec_data_1);
  if (sizeof(res_t) == 8) {
    vec = std::min(2, vec);
  }

  dim3 block(block_dim_x);
  int grid_dim_x = load_num / vec;
  TORCH_INTERNAL_ASSERT(grid_dim_x > 0);
  int grid_dim_y = getMaxGridSize(grid_dim_x, size1);
  TORCH_INTERNAL_ASSERT(grid_dim_y > 0);
  int y_t = size1 / grid_dim_y;
  grid_dim_y = std::ceil(float(size1)/float(y_t));
  // last block handle y_remain
  int y_remain = size1 - (grid_dim_y - 1) * y_t;
  dim3 grid(grid_dim_x, grid_dim_y, 1);
  TORCH_INTERNAL_ASSERT(y_t > 0);
  TORCH_INTERNAL_ASSERT(y_remain > 0);
  if (vec == 8) {
    elementwise_kernel_2_1_broadcast<nt, 8, res_t, arg0_t, func_t, index_t, stride_t><<<grid, block, 0, stream>>>(
        N, data0, data1, size0, size1, stride00, stride01, stride10, stride11, f, y_t, y_remain);
  } else if (vec == 4) {
    elementwise_kernel_2_1_broadcast<nt, 4, res_t, arg0_t, func_t, index_t, stride_t><<<grid, block, 0, stream>>>(
        N, data0, data1, size0, size1, stride00, stride01, stride10, stride11, f, y_t, y_remain);
  } else if (vec == 2) {
    elementwise_kernel_2_1_broadcast<nt, 2, res_t, arg0_t, func_t, index_t, stride_t><<<grid, block, 0, stream>>>(
        N, data0, data1, size0, size1, stride00, stride01, stride10, stride11, f, y_t, y_remain);
  } else {
    elementwise_kernel_2_1_broadcast<nt, 1, res_t, arg0_t, func_t, index_t, stride_t><<<grid, block, 0, stream>>>(
        N, data0, data1, size0, size1, stride00, stride01, stride10, stride11, f, y_t, y_remain);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// input layout: stride01==size0*sizeof(arg0_t), stride11==sizeof(arg0_t), stride21==0
// output layout: stride00==sizeof(res_t), stride10==size0*sizeof(res_t), stride20==size1*size0*sizeof(res_t)
template<int nt, int v_x, typename res_t , typename arg0_t, typename func_t, typename index_t, typename stride_t>
C10_LAUNCH_BOUNDS_2(nt, 4)
__global__ void elementwise_kernel_3_1_broadcast_dim_2(
    int64_t N,
    char* data0, char* data1,
    index_t size0, index_t size1, index_t size2,
    stride_t stride00, stride_t stride01,
    stride_t stride10, stride_t stride11,
    stride_t stride20, stride_t stride21,
    func_t f, int y_t) {
  // ndim = 2, arity = 2, narg = 3
  // v_x: vectorized load n elems;
  // y_t: store columns for each thread;
  // load_num: load iterations for each thread;
  int tid = threadIdx.x;
  constexpr int NARGS = 2;
  constexpr int MAX_DIMS = 3;
  int64_t offsets[NARGS];
  int64_t dim_offsets[MAX_DIMS];
  size_t x_remain = (size0 * size1) % (blockDim.x * v_x);
  size_t tail = (size0 * size1) % v_x;
  size_t y_remain = size2 % gridDim.y;
  using LoadT = at::native::memory::aligned_vector<arg0_t, v_x>;
  using StoreT = at::native::memory::aligned_vector<res_t, v_x>;
  #pragma unroll
  for (int arg = 0; arg < NARGS; arg++) {
    offsets[arg] = 0;
  }
  for (int arg = 0; arg < MAX_DIMS; arg++) {
    dim_offsets[arg] = 0;
  }
  // arg = 0: output; arg = 1: input1; arg = 2: input2;

  arg0_t ld_1[v_x];
  res_t ld_out[v_x];
  LoadT* p_ld_1 = reinterpret_cast<LoadT*>(&ld_1); 
  StoreT * p_ld_out = reinterpret_cast<StoreT*>(&ld_out); 

  size_t vec_x = (blockIdx.x * blockDim.x + tid) * v_x;
  dim_offsets[1] = vec_x / size0;
  dim_offsets[0] = vec_x % size0;
  #pragma unroll
  for (int i = 0; i < v_x; i++) {
    size_t offset0 = 0;
    size_t offset1 = 0;
    size_t offset_t = dim_offsets[0] + i;
    if(offset_t < size0){  // no need to cal 
      offset0 = dim_offsets[0] + i;
      offset1 = dim_offsets[1];
    }
    else{
      offset0 = offset_t % size0;
      offset1 = dim_offsets[1] + offset_t / size0;
    }
    offsets[1] = offset1 * stride11 + offset0 * stride01;
    ld_1[i] = *reinterpret_cast<arg0_t*>(data1 + offsets[1]);
  }
  size_t row_offset = vec_x * stride00;
  // *p_ld_2 = *reinterpret_cast<LoadT*>(data2 + offsets[2]);
  for (size_t y_idx = 0; y_idx < y_t; y_idx++) {
    offsets[0] = (blockIdx.y * y_t + y_idx) * stride20 + row_offset;
    StoreT* out = reinterpret_cast<StoreT*>(data0 + offsets[0]);
    #pragma unroll
    for (int i = 0; i < v_x; i++) {
      ld_out[i] = f(ld_1[i]);
    }
    *out = *p_ld_out;
  }
  if (y_remain != 0 && blockIdx.y == 0) {
    size_t row_offset = vec_x * stride00;
    // #pragma unroll
    for (int y_idx = 0; y_idx < y_remain; y_idx++) {
      offsets[0] = (gridDim.y * y_t + y_idx) * stride20 + row_offset;
      StoreT* out = reinterpret_cast<StoreT*>(data0 + offsets[0]);
      #pragma unroll
      for (int i = 0; i < v_x; i++) {
        ld_out[i] = f(ld_1[i]);
      }
      *out = *p_ld_out;
    }
  }

  if (x_remain && blockIdx.x == 0 && (tid + 1) * v_x <= x_remain) {
    vec_x = (gridDim.x * blockDim.x + tid) * v_x;
    dim_offsets[1] = vec_x / size0;
    dim_offsets[0] = vec_x % size0;
    #pragma unroll
    for (int i = 0; i < v_x; i++) {
      size_t offset0 = 0;
      size_t offset1 = 0;
      size_t offset_t = dim_offsets[0] + i;
      if(offset_t < size0){  // no need to cal 
        offset0 = dim_offsets[0] + i;
        offset1 = dim_offsets[1];
      }
      else{
        offset0 = offset_t % size0;
        offset1 = dim_offsets[1] + offset_t / size0;
      }
      offsets[1] = offset1 * stride11 + offset0 * stride01;
      ld_1[i] = *reinterpret_cast<arg0_t*>(data1 + offsets[1]);
    }
    size_t row_offset = vec_x * stride00;
    for (int y_idx = 0; y_idx < y_t; y_idx++) {
      offsets[0] = (blockIdx.y * y_t + y_idx) * stride20 + row_offset;
      StoreT* out = reinterpret_cast<StoreT*>(data0 + offsets[0]);
      #pragma unroll
      for (int i = 0; i < v_x; i++) {
        ld_out[i] = f(ld_1[i]);
      }
      *out = *p_ld_out;
    }
    if (y_remain != 0 && blockIdx.y == 0) {
      // #pragma unroll
      for (int y_idx = 0; y_idx < y_remain; y_idx++) {
        offsets[0] = (gridDim.y * y_t + y_idx) * stride20 + row_offset;
        StoreT* out = reinterpret_cast<StoreT*>(data0 + offsets[0]);
        #pragma unroll
        for (int i = 0; i < v_x; i++) {
          ld_out[i] = f(ld_1[i]);
        }
        *out = *p_ld_out;
      }
    }
  }

  // handle tail columns( < v_x)
  if(tail && blockIdx.x == 0 && tid < tail){
    auto remain_offset = x_remain / v_x;
    vec_x = (gridDim.x * blockDim.x + remain_offset) * v_x + tid;
    dim_offsets[1] = vec_x / size0;
    dim_offsets[0] = vec_x % size0;
    offsets[1] = dim_offsets[1] * stride11 + dim_offsets[0] * stride01;
    int64_t row_offset = ((gridDim.x  * blockDim.x + remain_offset) * v_x + tid) * stride00 ;
    for (int y_idx = 0; y_idx < y_t; y_idx++) {
      offsets[0] = (blockIdx.y * y_t + y_idx) * stride20 + row_offset;
      auto p0 = reinterpret_cast<res_t*>(data0 + offsets[0]);
      auto p1 = reinterpret_cast<arg0_t*>(data1 + offsets[1]);
      *p0 = f(*p1);
    }
    if (y_remain != 0 && blockIdx.y == 0) {
      // #pragma unroll
      for (int y_idx = 0; y_idx < y_remain; y_idx++) {
        offsets[0] = (gridDim.y * y_t + y_idx) * stride20 + row_offset;
        auto p0 = reinterpret_cast<res_t*>(data0 + offsets[0]);
        auto p1 = reinterpret_cast<arg0_t*>(data1 + offsets[1]);
        *p0 = f(*p1);
      }
    }
  }
}

template<int nt, int v_x, typename res_t , typename arg0_t, typename func_t, typename index_t, typename stride_t>
C10_LAUNCH_BOUNDS_2(nt, 4)
__global__ void elementwise_kernel_3_1_dim0_contiguous(
    int64_t N,
    char* data0, char* data1,
    index_t size0, index_t size1, index_t size2,
    stride_t stride00, stride_t stride01,
    stride_t stride10, stride_t stride11,
    stride_t stride20, stride_t stride21,
    func_t f) {
  int tid = threadIdx.x;
  int nv = blockDim.x * v_x;
  constexpr int NARGS = 2;
  constexpr int MAX_DIMS = 3;
  int64_t offsets[NARGS];
  int zid = blockIdx.y;
  int idx = blockIdx.x * nv + tid * v_x;
  using LoadT = at::native::memory::aligned_vector<arg0_t, v_x>;
  using StoreT = at::native::memory::aligned_vector<res_t, v_x>;
  offsets[0] = zid * stride20;
  offsets[1] = zid * stride21;

  arg0_t ld[v_x];
  res_t ld_out[v_x];
  LoadT *p_ld = reinterpret_cast<LoadT*>(&ld);
  StoreT *p_ld_out = reinterpret_cast<StoreT*>(&ld_out);

  if (idx < size0 * size1) {
    auto linear_idx = idx;

    auto divmod_mod = linear_idx % size0;
    linear_idx = linear_idx / size0;
    offsets[0] += divmod_mod * stride00;
    offsets[1] += divmod_mod * stride01;

    divmod_mod = linear_idx % size1;
    offsets[0] += divmod_mod * stride10;
    offsets[1] += divmod_mod * stride11;

    *p_ld = *reinterpret_cast<LoadT*>(data1 + offsets[1]);
    StoreT *out = reinterpret_cast<StoreT*>(data0 + offsets[0]);
    #pragma unroll
    for (int i=0; i<v_x; i++){
      ld_out[i] = f(ld[i]);
    }
    *out = *p_ld_out;
  }
}

template<int nt, int vt, typename res_t , typename arg0_t, typename func_t, typename index_t, typename stride_t>
static void launch_legacy_kernel_maca_3_1_broadcast(
    int64_t N,
    char* data0, char* data1,
    index_t size0, index_t size1, index_t size2,
    stride_t stride00, stride_t stride01,
    stride_t stride10, stride_t stride11,
    stride_t stride20, stride_t stride21,
    const func_t& f) {
  // ndim = 3, arity = 1
  TORCH_INTERNAL_ASSERT(N >= 0 && N <= std::numeric_limits<int32_t>::max());
  if (N == 0) {
    return;
  }
  auto stream = at::cuda::getCurrentCUDAStream();
  constexpr int block_dim_x = 64;
  if (stride21 == 0) {
    size_t load_num = size0 * size1 / block_dim_x;
    TORCH_INTERNAL_ASSERT(load_num > 0);
    // TODO: data1 does not use vector load, so no need alignment.
    int vec_data_0 = getVectorizedAlignment<res_t>((void*)data0, load_num);
    int vec_data_1 = getVectorizedAlignment<arg0_t>((void*)data1, load_num);
    if (std::is_same<res_t, at::Half>::value || std::is_same<res_t, at::BFloat16>::value) {
      vec_data_0 = std::min(vec_data_0, getVectorizedAlignment<res_t>((void*)data0, stride20/sizeof(res_t)));
    }
    int vec = std::min(vec_data_0, vec_data_1);
    dim3 block(block_dim_x);
    int grid_dim_x = load_num / vec;
    TORCH_INTERNAL_ASSERT(grid_dim_x > 0);
    int grid_dim_y = getMaxGridSize(grid_dim_x, size2);
    TORCH_INTERNAL_ASSERT(grid_dim_y > 0);
    dim3 grid(grid_dim_x, grid_dim_y, 1);
    int y_t = size2 / grid_dim_y;
    TORCH_INTERNAL_ASSERT(y_t > 0);

    // std::cout << "############" << std::endl;
    // std::cout << "grid_dim_x: " << grid_dim_x  << ", grid_dim_y: " << grid_dim_y << std::endl;
    // std::cout << "vec: " << vec << std::endl;
    // std::cout << "x_remain: " << (size0 * size1) % (block_dim_x * vec) << ", tail: " << (size0 * size1) % vec << std::endl;
    // std::cout << "y_remain: " << size2 % grid_dim_y << ", y_t: " << y_t << std::endl;
    if (vec == 8) {
      elementwise_kernel_3_1_broadcast_dim_2<nt, 8, res_t, arg0_t, func_t, index_t, stride_t><<<grid, block, 0, stream>>>(
          N, data0, data1, size0, size1, size2, stride00, stride01, stride10, stride11, stride20, stride21, f, y_t);
    } else if (vec == 4) {
      elementwise_kernel_3_1_broadcast_dim_2<nt, 4, res_t, arg0_t, func_t, index_t, stride_t><<<grid, block, 0, stream>>>(
          N, data0, data1, size0, size1, size2, stride00, stride01, stride10, stride11, stride20, stride21, f, y_t);
    } else if (vec == 2) {
      elementwise_kernel_3_1_broadcast_dim_2<nt, 2, res_t, arg0_t, func_t, index_t, stride_t><<<grid, block, 0, stream>>>(
          N, data0, data1, size0, size1, size2, stride00, stride01, stride10, stride11, stride20, stride21, f, y_t);
    } else {
      elementwise_kernel_3_1_broadcast_dim_2<nt, 1, res_t, arg0_t, func_t, index_t, stride_t><<<grid, block, 0, stream>>>(
          N, data0, data1, size0, size1, size2, stride00, stride01, stride10, stride11, stride20, stride21, f, y_t);
    }
  } else {
    TORCH_CHECK(false, "elementwise kernel 2_2 but not broadcast!!");
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template<int nt, int vt, typename res_t , typename arg0_t, typename func_t, typename index_t, typename stride_t>
static void launch_legacy_kernel_maca_3_1_dim0_contiguous(
    int64_t N,
    char* data0, char* data1,
    index_t size0, index_t size1, index_t size2,
    stride_t stride00, stride_t stride01,
    stride_t stride10, stride_t stride11,
    stride_t stride20, stride_t stride21,
    const func_t& f) {
  // ndim = 3, arity = 1
  TORCH_INTERNAL_ASSERT(N >= 0 && N <= std::numeric_limits<int32_t>::max());
  if (N == 0) {
    return;
  }
  auto stream = at::cuda::getCurrentCUDAStream();
  constexpr int block_dim_x = 64;
  dim3 block(block_dim_x);

  int vec = getVectorizedAlignment<arg0_t>((void*)data1, size0);
  dim3 grid = dim3((size0 * size1 + block.x * vec - 1) / (block.x * vec), size2);
  
  if (vec == 8) {
    elementwise_kernel_3_1_dim0_contiguous<block_dim_x, 8, res_t, arg0_t, func_t, index_t, stride_t><<<grid, block, 0, stream>>>(
        N, data0, data1, size0, size1, size2, stride00, stride01, stride10, stride11, stride20, stride21, f);
  } else if (vec == 4) {
    elementwise_kernel_3_1_dim0_contiguous<block_dim_x, 4, res_t, arg0_t, func_t, index_t, stride_t><<<grid, block, 0, stream>>>(
        N, data0, data1, size0, size1, size2, stride00, stride01, stride10, stride11, stride20, stride21, f);
  } else if (vec == 2) {
    elementwise_kernel_3_1_dim0_contiguous<block_dim_x, 2, res_t, arg0_t, func_t, index_t, stride_t><<<grid, block, 0, stream>>>(
        N, data0, data1, size0, size1, size2, stride00, stride01, stride10, stride11, stride20, stride21, f);       
  } else {
    elementwise_kernel_3_1_dim0_contiguous<block_dim_x, 1, res_t, arg0_t, func_t, index_t, stride_t><<<grid, block, 0, stream>>>(
        N, data0, data1, size0, size1, size2, stride00, stride01, stride10, stride11, stride20, stride21, f);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template<int nt, int vt, typename res_t , typename arg0_t, typename func_t, typename index_t, typename stride_t>
static void launch_legacy_kernel_maca_2_1_share(
    int64_t N,
    char* data0, char* data1,
    index_t size0, index_t size1,
    stride_t stride00, stride_t stride01, 
    stride_t stride10, stride_t stride11, 
    const func_t& f) {
  // ndim = 2, arity = 1
  TORCH_INTERNAL_ASSERT(N >= 0 && N <= std::numeric_limits<int32_t>::max());
  if (N == 0) {
    return;
  }
  dim3 block(nt);
  dim3 grid((N + block.x * vt - 1) / (block.x * vt));
  auto stream = at::cuda::getCurrentCUDAStream();
  elementwise_kernel_2_1<nt, vt, res_t, arg0_t, func_t, index_t, stride_t><<<grid, block, 60000, stream>>>(
      N, data0, data1, size0, size1, stride00, stride01, stride10, stride11, f);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template<int nt, int vt, typename res_t , typename arg0_t, typename func_t, typename index_t, typename stride_t>
static void launch_legacy_kernel_maca_3_1(
    int64_t N,
    char* data0, char* data1,
    index_t size0, index_t size1, index_t size2,
    stride_t stride00, stride_t stride01, 
    stride_t stride10, stride_t stride11, 
    stride_t stride20, stride_t stride21, 
    const func_t& f) {
  // ndim = 3, arity = 1
  TORCH_INTERNAL_ASSERT(N >= 0 && N <= std::numeric_limits<int32_t>::max());
  if (N == 0) {
    return;
  }
  dim3 block(nt);
  dim3 grid((N + block.x * vt - 1) / (block.x * vt));
  auto stream = at::cuda::getCurrentCUDAStream();
  // 3_1 1024 1280 2
  // 3_1 1024 320 2
  // 3_1 256 1280 2
  // 3_1 256 640 2
  // 3_1 4096 320 2
  // 3_1 4096 640 2
  // 3_1 64 1280 2
  // 3_1 64 16 77
  // 3_1 64 77 16
  if (size0 == 1024 && size1 == 1280 && size2 == 2) {
    elementwise_kernel_3_1_s<nt, vt, res_t, arg0_t, func_t, index_t, stride_t, 1024, 1280, 2><<<grid, block, 0, stream>>>(
        N, data0, data1, stride00, stride01, stride10, stride11, stride20, stride21, f);
  } else if (size0 == 1024 && size1 == 320 && size2 == 2) { 
    elementwise_kernel_3_1_s<nt, vt, res_t, arg0_t, func_t, index_t, stride_t, 1024, 320, 2><<<grid, block, 0, stream>>>(
        N, data0, data1, stride00, stride01, stride10, stride11, stride20, stride21, f);
  } else if (size0 == 256 && size1 == 1280 && size2 == 2) { 
    elementwise_kernel_3_1_s<nt, vt, res_t, arg0_t, func_t, index_t, stride_t, 256, 1280, 2><<<grid, block, 0, stream>>>(
        N, data0, data1, stride00, stride01, stride10, stride11, stride20, stride21, f);
  } else if (size0 == 256 && size1 == 640 && size2 == 2) { 
    elementwise_kernel_3_1_s<nt, vt, res_t, arg0_t, func_t, index_t, stride_t, 256, 640, 2><<<grid, block, 0, stream>>>(
        N, data0, data1, stride00, stride01, stride10, stride11, stride20, stride21, f);
  } else if (size0 == 4096 && size1 == 320 && size2 == 2) { 
    elementwise_kernel_3_1_s<nt, vt, res_t, arg0_t, func_t, index_t, stride_t, 4096, 320, 2><<<grid, block, 0, stream>>>(
        N, data0, data1, stride00, stride01, stride10, stride11, stride20, stride21, f);
  } else if (size0 == 4096 && size1 == 640 && size2 == 2) { 
    elementwise_kernel_3_1_s<nt, vt, res_t, arg0_t, func_t, index_t, stride_t, 4096, 640, 2><<<grid, block, 0, stream>>>(
        N, data0, data1, stride00, stride01, stride10, stride11, stride20, stride21, f);
  } else if (size0 == 64 && size1 == 1280 && size2 == 2) { 
    elementwise_kernel_3_1_s<nt, vt, res_t, arg0_t, func_t, index_t, stride_t, 64, 1280, 2><<<grid, block, 0, stream>>>(
        N, data0, data1, stride00, stride01, stride10, stride11, stride20, stride21, f);
  } else if (size0 == 64 && size1 == 16 && size2 == 77) { 
    elementwise_kernel_3_1_s<nt, vt, res_t, arg0_t, func_t, index_t, stride_t, 64, 16, 77><<<grid, block, 0, stream>>>(
        N, data0, data1, stride00, stride01, stride10, stride11, stride20, stride21, f);
  } else if (size0 == 64 && size1 == 77 && size2 == 16) { 
    elementwise_kernel_3_1_s<nt, vt, res_t, arg0_t, func_t, index_t, stride_t, 64, 77, 16><<<grid, block, 0, stream>>>(
        N, data0, data1, stride00, stride01, stride10, stride11, stride20, stride21, f);
  } else {
    elementwise_kernel_3_1<nt, vt, res_t, arg0_t, func_t, index_t, stride_t><<<grid, block, 0, stream>>>(
        N, data0, data1, size0, size1, size2, stride00, stride01, stride10, stride11, stride20, stride21, f);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template<typename res_t, typename arg0_t, typename TIndex, typename func_t, typename TData>
__global__ void elementwise_kernel_transpose_copy(
    char* out, char* in,
    TIndex H, TIndex W, TIndex dh, TIndex dw,
    func_t f) {
  // ndim = 3, arity = 1, narg = 2
  TData* Y = reinterpret_cast<TData*>(out);
  TData* X = reinterpret_cast<TData*>(in);

  __shared__ TData tile[kTileDimMacaT][kTileDimMacaT + 2];
  const TIndex n = blockIdx.z;
  const TIndex r = blockIdx.y;
  const TIndex c = blockIdx.x;
  const int64_t offset = n * H * W;
  int64_t x = c * kTileDimMacaT + threadIdx.x;
  int64_t y = r * kTileDimMacaT + threadIdx.y;
  if (x < W) {
    for (int i = 0; threadIdx.y + i < kTileDimMacaT && y + i < H; i += kBlockRowsMacaT) {
      tile[threadIdx.y + i][threadIdx.x] = (X[offset + (y + i) * W + x]);
    }
  }
  __syncthreads();
  x = r * kTileDimMacaT + threadIdx.x;
  y = c * kTileDimMacaT + threadIdx.y;
  if (x < H) {
    for (int i = 0; threadIdx.y + i < kTileDimMacaT && y + i < W; i += kBlockRowsMacaT) {
      Y[offset + (y + i) * H + x] = tile[threadIdx.x][threadIdx.y + i];
    }
  }
}

template<typename res_t, typename arg0_t, typename TIndex, typename func_t, typename TData>
__global__ void elementwise_kernel_transpose_half_copy_8(
    char* out, char* in,
    TIndex H, TIndex W, TIndex dh, TIndex dw,
    func_t f) {
  // ndim = 3, arity = 1, narg = 2
  // typedef arg0_t TData;
  TData* Y = reinterpret_cast<TData*>(out);
  TData* X = reinterpret_cast<TData*>(in);

  __shared__ TData tile[kTileDimMacaT][kTileDimMacaT + 2];
  const TIndex n = blockIdx.z;
  const TIndex r = blockIdx.y;
  const TIndex c = blockIdx.x;
  const int64_t offset = n * H * W;
  int64_t x = c * kTileDimMacaT + threadIdx.x * 8;
  int64_t y = r * kTileDimMacaT + threadIdx.y;

  if (x < W && y < H) {
    float4 tmp1 = *reinterpret_cast<float4*>(X + (offset + y * W + x));
    TData* tt = reinterpret_cast<TData*>(&tmp1);
    tile[threadIdx.y][threadIdx.x * 8 + 0] = tt[0];
    tile[threadIdx.y][threadIdx.x * 8 + 1] = tt[1];
    tile[threadIdx.y][threadIdx.x * 8 + 2] = tt[2];
    tile[threadIdx.y][threadIdx.x * 8 + 3] = tt[3];
    tile[threadIdx.y][threadIdx.x * 8 + 4] = tt[4];
    tile[threadIdx.y][threadIdx.x * 8 + 5] = tt[5];
    tile[threadIdx.y][threadIdx.x * 8 + 6] = tt[6];
    tile[threadIdx.y][threadIdx.x * 8 + 7] = tt[7];
  }
  __syncthreads();

  x = r * kTileDimMacaT + threadIdx.x * 8;
  y = c * kTileDimMacaT + threadIdx.y;
  if (x < H && y < W) {
    TData tmp[8];
    for (int i = 0; i < 8; ++i) {
      tmp[i] = tile[threadIdx.x * 8 + i][threadIdx.y];
    }
    *reinterpret_cast<float4*>(&Y[offset + y * H + x]) = *reinterpret_cast<float4*>(&tmp);
  }
}

template<typename res_t, typename arg0_t, typename TIndex, typename func_t, typename TData>
__global__ void elementwise_kernel_transpose_half_copy_64(
    char* out, char* in,
    TIndex H, TIndex W, TIndex dh, TIndex dw,
    func_t f) {
  // ndim = 3, arity = 1, narg = 2
  // typedef arg0_t TData;
  TData* Y = reinterpret_cast<TData*>(out);
  TData* X = reinterpret_cast<TData*>(in);

  __shared__ TData tile[kTileDimMacaT][kTileDimMacaT + 2];
  const TIndex n = blockIdx.z;
  const TIndex r = blockIdx.y;
  const TIndex c = blockIdx.x;
  const int64_t offset = n * H * W;
  int64_t x = c * kTileDimMacaT + threadIdx.x * 8;
  int64_t y = r * kTileDimMacaT + threadIdx.y;

  float4 tmp1 = *reinterpret_cast<float4*>(X + (offset + y * W + x));
  TData* tt = reinterpret_cast<TData*>(&tmp1);
  tile[threadIdx.y][threadIdx.x * 8 + 0] = tt[0];
  tile[threadIdx.y][threadIdx.x * 8 + 1] = tt[1];
  tile[threadIdx.y][threadIdx.x * 8 + 2] = tt[2];
  tile[threadIdx.y][threadIdx.x * 8 + 3] = tt[3];
  tile[threadIdx.y][threadIdx.x * 8 + 4] = tt[4];
  tile[threadIdx.y][threadIdx.x * 8 + 5] = tt[5];
  tile[threadIdx.y][threadIdx.x * 8 + 6] = tt[6];
  tile[threadIdx.y][threadIdx.x * 8 + 7] = tt[7];
  __syncthreads();

  x = r * kTileDimMacaT + threadIdx.x * 8;
  y = c * kTileDimMacaT + threadIdx.y;
  TData tmp[8];
  for (int i = 0; i < 8; ++i) {
    tmp[i] = tile[threadIdx.x * 8 + i][threadIdx.y];
  }
  *reinterpret_cast<float4*>(&Y[offset + y * H + x]) = *reinterpret_cast<float4*>(&tmp);
}

template<typename res_t, typename arg0_t, typename TIndex, typename func_t, typename TData>
__global__ void elementwise_kernel_transpose012_half_copy_64(
    char* out, char* in,
    TIndex H, TIndex W, TIndex dh, TIndex dw,
    func_t f) {
  // ndim = 3, arity = 1, narg = 2
  // typedef arg0_t TData;
  TData* Y = reinterpret_cast<TData*>(out);
  TData* X = reinterpret_cast<TData*>(in);

  __shared__ TData tile[kTileDimMacaT][kTileDimMacaT + 2];
  const TIndex n = blockIdx.z;
  const TIndex r = blockIdx.y;
  const TIndex c = blockIdx.x;
  // gridDim.z == batch
  const int64_t offset_input = n * gridDim.z;
  const int64_t offset_output = n * H;
  int64_t x = c * kTileDimMacaT + threadIdx.x * 8;
  int64_t y = r * kTileDimMacaT + threadIdx.y;

  float4 tmp1 = *reinterpret_cast<float4*>(X + (offset_input + y * H * gridDim.z + x));
  TData* tt = reinterpret_cast<TData*>(&tmp1);
  tile[threadIdx.y][threadIdx.x * 8 + 0] = tt[0];
  tile[threadIdx.y][threadIdx.x * 8 + 1] = tt[1];
  tile[threadIdx.y][threadIdx.x * 8 + 2] = tt[2];
  tile[threadIdx.y][threadIdx.x * 8 + 3] = tt[3];
  tile[threadIdx.y][threadIdx.x * 8 + 4] = tt[4];
  tile[threadIdx.y][threadIdx.x * 8 + 5] = tt[5];
  tile[threadIdx.y][threadIdx.x * 8 + 6] = tt[6];
  tile[threadIdx.y][threadIdx.x * 8 + 7] = tt[7];
  __syncthreads();

  x = r * kTileDimMacaT + threadIdx.x * 8;
  y = c * kTileDimMacaT + threadIdx.y;
  TData tmp[8];
  for (int i = 0; i < 8; ++i) {
    tmp[i] = tile[threadIdx.x * 8 + i][threadIdx.y];
  }
  *reinterpret_cast<float4*>(&Y[offset_output + y * H * gridDim.z + x]) = *reinterpret_cast<float4*>(&tmp);
}

template<typename res_t, typename arg0_t, typename TIndex, typename func_t,
typename std::enable_if<!std::is_same<arg0_t, at::Half>::value && !std::is_same<arg0_t, at::BFloat16>::value
                       && !std::is_same<arg0_t, float>::value, int>::type = 0>
__global__ void elementwise_kernel_transpose_copy_64_uncontiguous(
    char* out, char* in,
    TIndex H, TIndex W, TIndex dh, TIndex dw,
    int64_t s10, int64_t s01,
    func_t f) {
      assert(0);
    }

template<typename res_t, typename arg0_t, typename TIndex, typename func_t,
typename std::enable_if<std::is_same<arg0_t, at::Half>::value || std::is_same<arg0_t, at::BFloat16>::value
                       || std::is_same<arg0_t, float>::value, int>::type = 0>
__global__ void elementwise_kernel_transpose_copy_64_uncontiguous(
    char* out, char* in,
    TIndex H, TIndex W, TIndex dh, TIndex dw,
    int64_t s10, int64_t s01,
    func_t f) {
  __shared__ arg0_t tile[kTileDimMacaT][kTileDimMacaT + 2];
  const TIndex n = blockIdx.z;
  const TIndex r = blockIdx.y;
  const TIndex c = blockIdx.x;
  const int64_t offset = n * H * W;
  int64_t x = c * kTileDimMacaT + threadIdx.x * 8;
  int64_t y = r * kTileDimMacaT + threadIdx.y ;

  float4 tmp1 = *reinterpret_cast<float4*>(in + (offset + y * s01 + x * sizeof(arg0_t)));
  arg0_t* tt = reinterpret_cast<arg0_t*>(&tmp1);
  #pragma unroll
  for (int i = 0; i < 8; i++) {
    tile[threadIdx.y][threadIdx.x * 8 + i] = tt[i];
  }
  __syncthreads();

  x = r * kTileDimMacaT + threadIdx.x * 8;
  y = c * kTileDimMacaT + threadIdx.y;
  arg0_t tmp[8];
  for (int i = 0; i < 8; ++i) {
    tmp[i] = tile[threadIdx.x * 8 + i][threadIdx.y];
  }
  *reinterpret_cast<float4*>(out + (offset + y * s10 + x * sizeof(res_t))) = *reinterpret_cast<float4*>(&tmp);
}

template<typename res_t, typename arg0_t, typename TIndex>
__global__ void elementwise_kernel_transpose_half_copy_no_share(
    char *out, char *in,
    TIndex b, TIndex h, TIndex w) {
  typedef at::Half TData;
  TData* Y = reinterpret_cast<TData*>(out);
  TData* X = reinterpret_cast<TData*>(in);

  int64_t rr = blockIdx.y * 64 + threadIdx.y * 8;
  int64_t cc = blockIdx.x * 64 + threadIdx.x * 8;
  for (int i = 0; i < b; i++) {
    int offset = i * h * w;
    TData res0[8];
    TData res1[8];
    TData res2[8];
    TData res3[8];
    TData res4[8];
    TData res5[8];
    TData res6[8];
    TData res7[8];
    *(float4*)(&res0) = *reinterpret_cast<float4*>(&X[rr * w + cc + offset]);
    *(float4*)(&res1) = *reinterpret_cast<float4*>(&X[(rr+1) * w + cc + offset]);
    *(float4*)(&res2) = *reinterpret_cast<float4*>(&X[(rr+2) * w + cc + offset]);
    *(float4*)(&res3) = *reinterpret_cast<float4*>(&X[(rr+3) * w + cc + offset]);
    *(float4*)(&res4) = *reinterpret_cast<float4*>(&X[(rr+4) * w + cc + offset]);
    *(float4*)(&res5) = *reinterpret_cast<float4*>(&X[(rr+5) * w + cc + offset]);
    *(float4*)(&res6) = *reinterpret_cast<float4*>(&X[(rr+6) * w + cc + offset]);
    *(float4*)(&res7) = *reinterpret_cast<float4*>(&X[(rr+7) * w + cc + offset]);
    TData r0[8];
    TData r1[8];
    TData r2[8];
    TData r3[8];
    TData r4[8];
    TData r5[8];
    TData r6[8];
    TData r7[8];
    r0[0] = res0[0]; 
    r0[1] = res1[0]; 
    r0[2] = res2[0]; 
    r0[3] = res3[0]; 
    r0[4] = res4[0]; 
    r0[5] = res5[0]; 
    r0[6] = res6[0]; 
    r0[7] = res7[0]; 
    r1[0] = res0[1]; 
    r1[1] = res1[1]; 
    r1[2] = res2[1]; 
    r1[3] = res3[1]; 
    r1[4] = res4[1]; 
    r1[5] = res5[1]; 
    r1[6] = res6[1]; 
    r1[7] = res7[1]; 
    r2[0] = res0[2]; 
    r2[1] = res1[2]; 
    r2[2] = res2[2]; 
    r2[3] = res3[2]; 
    r2[4] = res4[2]; 
    r2[5] = res5[2]; 
    r2[6] = res6[2]; 
    r2[7] = res7[2]; 
    r3[0] = res0[3]; 
    r3[1] = res1[3]; 
    r3[2] = res2[3]; 
    r3[3] = res3[3]; 
    r3[4] = res4[3]; 
    r3[5] = res5[3]; 
    r3[6] = res6[3]; 
    r3[7] = res7[3]; 
    r4[0] = res0[4]; 
    r4[1] = res1[4]; 
    r4[2] = res2[4]; 
    r4[3] = res3[4]; 
    r4[4] = res4[4]; 
    r4[5] = res5[4]; 
    r4[6] = res6[4]; 
    r4[7] = res7[4]; 
    r5[0] = res0[5]; 
    r5[1] = res1[5]; 
    r5[2] = res2[5]; 
    r5[3] = res3[5]; 
    r5[4] = res4[5]; 
    r5[5] = res5[5]; 
    r5[6] = res6[5]; 
    r5[7] = res7[5]; 
    r6[0] = res0[6]; 
    r6[1] = res1[6]; 
    r6[2] = res2[6]; 
    r6[3] = res3[6]; 
    r6[4] = res4[6]; 
    r6[5] = res5[6]; 
    r6[6] = res6[6]; 
    r6[7] = res7[6]; 
    r7[0] = res0[7]; 
    r7[1] = res1[7]; 
    r7[2] = res2[7]; 
    r7[3] = res3[7]; 
    r7[4] = res4[7]; 
    r7[5] = res5[7]; 
    r7[6] = res6[7]; 
    r7[7] = res7[7]; 
    *(float4*)(&Y[rr      + (cc+0) *h + offset]) = *reinterpret_cast<float4*>(&r0);
    *(float4*)(&Y[rr      + (cc+1) *h + offset]) = *reinterpret_cast<float4*>(&r1);
    *(float4*)(&Y[rr      + (cc+2) *h + offset]) = *reinterpret_cast<float4*>(&r2);
    *(float4*)(&Y[rr      + (cc+3) *h + offset]) = *reinterpret_cast<float4*>(&r3);
    *(float4*)(&Y[rr      + (cc+4) *h + offset]) = *reinterpret_cast<float4*>(&r4);
    *(float4*)(&Y[rr      + (cc+5) *h + offset]) = *reinterpret_cast<float4*>(&r5);
    *(float4*)(&Y[rr      + (cc+6) *h + offset]) = *reinterpret_cast<float4*>(&r6);
    *(float4*)(&Y[rr      + (cc+7) *h + offset]) = *reinterpret_cast<float4*>(&r7);
  }
}

template<typename res_t, typename arg0_t, typename TIndex, typename func_t>
__global__ void elementwise_kernel_transpose_short_copy(
    char* out, char* in,
    TIndex H, TIndex W, TIndex dh, TIndex dw,
    func_t f) {
//  // ndim = 3, arity = 1, narg = 2
//  typedef int16_t TData;
//  TData* Y = reinterpret_cast<TData*>(out);
//  TData* X = reinterpret_cast<TData*>(in);
//
//  __shared__ TData tile[kTileDimMacaT][kTileDimMacaT + 2];
//  const TIndex n = blockIdx.x / (dh * dw);
//  const TIndex k = blockIdx.x % (dh * dw);
//  const TIndex r = k / dw;
//  const TIndex c = k % dw;
//  const TIndex offset = n * H * W;
//  int x = c * kTileDimMacaT + threadIdx.x;
//  int y = r * kTileDimMacaT + threadIdx.y;
//  if (x < W) {
//    for (int i = 0; threadIdx.y + i < kTileDimMacaT && y + i < H; i += kBlockRowsMacaT) {
//      tile[threadIdx.y + i][threadIdx.x] = (X[offset + (y + i) * W + x]);
//    }
//  }
//  __syncthreads();
//  x = r * kTileDimMacaT + threadIdx.x;
//  y = c * kTileDimMacaT + threadIdx.y;
//  if (x < H) {
//    for (int i = 0; threadIdx.y + i < kTileDimMacaT && y + i < W; i += kBlockRowsMacaT) {
//      Y[offset + (y + i) * H + x] = tile[threadIdx.x][threadIdx.y + i];
//    }
//  }
}

template<typename res_t, typename arg0_t, typename TIndex, typename func_t>
__global__ void elementwise_kernel_transpose_float_copy(
    char* out, char* in,
    TIndex H, TIndex W, TIndex dh, TIndex dw,
    func_t f) {
  // ndim = 3, arity = 1, narg = 2
  typedef float TData;
  TData* Y = reinterpret_cast<TData*>(out);
  TData* X = reinterpret_cast<TData*>(in);

  __shared__ TData tile[kTileDimMacaT][kTileDimMacaT + 2];
  const TIndex n = blockIdx.x / (dh * dw);
  const TIndex k = blockIdx.x % (dh * dw);
  const TIndex r = k / dw;
  const TIndex c = k % dw;
  const int64_t offset = n * H * W;
  int64_t x = c * kTileDimMacaT + threadIdx.x;
  int64_t y = r * kTileDimMacaT + threadIdx.y;
  if (x < W) {
    for (int i = 0; threadIdx.y + i < kTileDimMacaT && y + i < H; i += kBlockRowsMacaT) {
      tile[threadIdx.y + i][threadIdx.x] = (X[offset + (y + i) * W + x]);
    }
  }
  __syncthreads();
  x = r * kTileDimMacaT + threadIdx.x;
  y = c * kTileDimMacaT + threadIdx.y;
  if (x < H) {
    for (int i = 0; threadIdx.y + i < kTileDimMacaT && y + i < W; i += kBlockRowsMacaT) {
      Y[offset + (y + i) * H + x] = tile[threadIdx.x][threadIdx.y + i];
    }
  }
}

template<typename res_t, typename arg0_t, typename TIndex, typename func_t>
__global__ void elementwise_kernel_3_1_transpose12_half_copy_64(
      char*  out, char*  in, 
      TIndex size0, TIndex size1, TIndex size2,
      TIndex stride00, TIndex stride01,
      TIndex stride10, TIndex stride11,
      TIndex stride20, TIndex stride21,
      TIndex dh, TIndex dw,
      func_t f) {
  // ndim = 3, arity = 1, narg = 2
  const int64_t k = blockIdx.x % (dh * dw);   //size0*size1*size2
  const int64_t r = k / dw;                   //size1*size2
  const int64_t c = k % dw;                   //size0

  if((r * kTileDimMacaT + threadIdx.y) < (size1 * size2)){
    int s0 = c * kTileDimMacaT + threadIdx.x * 8;
    int s2 = (r * kTileDimMacaT + threadIdx.y) / size1;
    int s1 = (r * kTileDimMacaT + threadIdx.y) % size1;
    int64_t out_offset = s0 * stride00 + s1 * stride10 + s2 * stride20;
    int64_t in_offset = s0 * stride01 + s1 * stride11 + s2 * stride21;

    float4 tmp = *reinterpret_cast<float4*>(in + in_offset);
    *reinterpret_cast<float4*>(&out[out_offset]) = *reinterpret_cast<float4*>(&tmp);
  }
}

template<typename res_t, typename arg0_t, typename TIndex, typename func_t>
__global__ void elementwise_kernel_3_1_transpose12_half_copy_32(
      char*  out, char*  in, 
      TIndex size0, TIndex size1, TIndex size2,
      TIndex stride00, TIndex stride01,
      TIndex stride10, TIndex stride11,
      TIndex stride20, TIndex stride21,
      TIndex dh, TIndex dw,
      func_t f) {
  // ndim = 3, arity = 1, narg = 2
  const int64_t k = blockIdx.x % (dh * dw);   //size0*size1*size2
  const int64_t r = k / dw;                   //size1*size2
  const int64_t c = k % dw;                   //size0

  if((r * kTileDimMacaT_32 + threadIdx.y) < (size1 * size2)){
    int s0 = c * kTileDimMacaT_32 + threadIdx.x * 8;
    int s2 = (r * kTileDimMacaT_32 + threadIdx.y) / size1;
    int s1 = (r * kTileDimMacaT_32 + threadIdx.y) % size1;
    int64_t out_offset = s0 * stride00 + s1 * stride10 + s2 * stride20;
    int64_t in_offset = s0 * stride01 + s1 * stride11 + s2 * stride21;

    float4 tmp = *reinterpret_cast<float4*>(in + in_offset);
    *reinterpret_cast<float4*>(&out[out_offset]) = *reinterpret_cast<float4*>(&tmp);
  }
}

template<typename res_t, typename arg0_t, typename TIndex, typename func_t,
typename std::enable_if<!std::is_same<arg0_t, at::Half>::value && !std::is_same<arg0_t, at::BFloat16>::value
                       && !std::is_same<arg0_t, float>::value, int>::type = 0>
__global__ void elementwise_kernel_transpose_3_1_transpose02_half_copy(
    char* out, char* in,
    TIndex size0, TIndex size1, TIndex size2,
    int64_t s00, int64_t s01,
    int64_t s10, int64_t s11,
    int64_t s20, int64_t s21,
    func_t f) {
    assert(0);
    }

template<typename res_t, typename arg0_t, typename TIndex, typename func_t,
typename std::enable_if<std::is_same<arg0_t, at::Half>::value || std::is_same<arg0_t, at::BFloat16>::value
                       || std::is_same<arg0_t, float>::value, int>::type = 0>
__global__ void elementwise_kernel_transpose_3_1_transpose02_half_copy(
    char* out, char* in,
    TIndex size0, TIndex size1, TIndex size2,
    int64_t s00, int64_t s01,
    int64_t s10, int64_t s11,
    int64_t s20, int64_t s21,
    func_t f) {
  __shared__ arg0_t tile[kTileDimMacaT][kTileDimMacaT + 2];
  const TIndex n = blockIdx.z;
  const TIndex r = blockIdx.y;
  const TIndex c = blockIdx.x;
  
  int64_t offset_in = n * s11;
  int64_t offset_out = n * s10;

  int64_t x = c * kTileDimMacaT + threadIdx.x * 8;
  int64_t y = r * kTileDimMacaT + threadIdx.y ;

  float4 tmp1 = *reinterpret_cast<float4*>(in + (offset_in + y * s01 + x * sizeof(arg0_t)));
  arg0_t* tt = reinterpret_cast<arg0_t*>(&tmp1);
  #pragma unroll
  for (int i = 0; i < 8; i++) {
    tile[threadIdx.y][threadIdx.x * 8 + i] = tt[i];
  }
  __syncthreads();

  x = r * kTileDimMacaT + threadIdx.x * 8;
  y = c * kTileDimMacaT + threadIdx.y;
  res_t tmp[8];
  for (int i = 0; i < 8; ++i) {
    tmp[i] = tile[threadIdx.x * 8 + i][threadIdx.y];
  }
  *reinterpret_cast<float4*>(out + (offset_out + y * s20 + x * sizeof(res_t))) = *reinterpret_cast<float4*>(&tmp);
}

template<typename res_t, typename arg0_t, typename TIndex, typename func_t, int v>
__global__ void elementwise_kernel_3_1_copy_64_opt(
      char*  out, char*  in, 
      TIndex size0, TIndex size1, TIndex size2,
      TIndex stride00, TIndex stride01,
      TIndex stride10, TIndex stride11,
      TIndex stride20, TIndex stride21,
      TIndex dh, TIndex dw,
      func_t f) {
  // ndim = 3, arity = 1, narg = 2
  const int64_t k = blockIdx.x % (dh * dw);   //size0*size1*size2
  const int64_t r = k / dw;                   //size1*size2
  const int64_t c = k % dw;                   //size0

  if((r * kTileDimMacaT + threadIdx.y) < (size1 * size2)){
    int s0 = c * kTileDimMacaT + threadIdx.x * (16 / v);
    int s2 = (r * kTileDimMacaT + threadIdx.y) / size1;
    int s1 = (r * kTileDimMacaT + threadIdx.y) % size1;
    int64_t out_offset = s0 * stride00 + s1 * stride10 + s2 * stride20;
    int64_t in_offset = s0 * stride01 + s1 * stride11 + s2 * stride21;

    float4 tmp = *reinterpret_cast<float4*>(in + in_offset);
    *reinterpret_cast<float4*>(&out[out_offset]) = *reinterpret_cast<float4*>(&tmp);
  }
}

template<typename res_t, typename arg0_t, typename TIndex, typename func_t, int v>
__global__ void elementwise_kernel_3_1_copy_32_opt(
      char*  out, char*  in, 
      TIndex size0, TIndex size1, TIndex size2,
      TIndex stride00, TIndex stride01,
      TIndex stride10, TIndex stride11,
      TIndex stride20, TIndex stride21,
      TIndex dh, TIndex dw,
      func_t f) {
  // ndim = 3, arity = 1, narg = 2
  const int64_t k = blockIdx.x % (dh * dw);   //size0*size1*size2
  const int64_t r = k / dw;                   //size1*size2
  const int64_t c = k % dw;                   //size0

  if((r * kTileDimMacaT_32 + threadIdx.y) < (size1 * size2)){
    int s0 = c * kTileDimMacaT_32 + threadIdx.x * (16 / v);
    int s2 = (r * kTileDimMacaT_32 + threadIdx.y) / size1;
    int s1 = (r * kTileDimMacaT_32 + threadIdx.y) % size1;
    int64_t out_offset = s0 * stride00 + s1 * stride10 + s2 * stride20;
    int64_t in_offset = s0 * stride01 + s1 * stride11 + s2 * stride21;

    float4 tmp = *reinterpret_cast<float4*>(in + in_offset);
    *reinterpret_cast<float4*>(&out[out_offset]) = *reinterpret_cast<float4*>(&tmp);
  }
}

template<typename res_t, typename arg0_t, typename TIndex, typename func_t, int v>
__global__ void elementwise_kernel_4_1_transpose12_copy_64(
      char*  out, char*  in, 
      TIndex size0, TIndex size1, TIndex size2, TIndex size3,
      TIndex stride00, TIndex stride01,
      TIndex stride10, TIndex stride11,
      TIndex stride20, TIndex stride21,
      TIndex stride30, TIndex stride31,
      TIndex dh, TIndex dw,
      func_t f) {
  // ndim = 4, arity = 1, narg = 2
  const int64_t s3 = blockIdx.x / (dh * dw);  //size3
  const int64_t k = blockIdx.x % (dh * dw);   //size0*size1*size2
  const int64_t r = k / dw;                   //size1*size2
  const int64_t c = k % dw;                   //size0

  if((r * kTileDimMacaT + threadIdx.y) < (size1 * size2)){
    int s0 = c * kTileDimMacaT + threadIdx.x * (16 / v);
    int s2 = (r * kTileDimMacaT + threadIdx.y) / size1;
    int s1 = (r * kTileDimMacaT + threadIdx.y) % size1;
    int64_t out_offset = s0 * stride00 + s1 * stride10 + s2 * stride20 + s3 * stride30;
    int64_t in_offset = s0 * stride01 + s1 * stride11 + s2 * stride21 + s3 * stride31;

    float4 tmp = *reinterpret_cast<float4*>(in + in_offset);
    *reinterpret_cast<float4*>(&out[out_offset]) = *reinterpret_cast<float4*>(&tmp);
  }
}

template<typename res_t, typename arg0_t, typename TIndex, typename func_t, int v>
__global__ void elementwise_kernel_4_1_transpose12_copy_32(
      char*  out, char*  in, 
      TIndex size0, TIndex size1, TIndex size2, TIndex size3,
      TIndex stride00, TIndex stride01,
      TIndex stride10, TIndex stride11,
      TIndex stride20, TIndex stride21,
      TIndex stride30, TIndex stride31,
      TIndex dh, TIndex dw,
      func_t f) {
  // ndim = 4, arity = 1, narg = 2
  const int64_t s3 = blockIdx.x / (dh * dw);  //size3
  const int64_t k = blockIdx.x % (dh * dw);   //size0*size1*size2
  const int64_t r = k / dw;                   //size1*size2
  const int64_t c = k % dw;                   //size0

  if((r * kTileDimMacaT_32 + threadIdx.y) < (size1 * size2)){
    int s0 = c * kTileDimMacaT_32 + threadIdx.x * (16 / v);
    int s2 = (r * kTileDimMacaT_32 + threadIdx.y) / size1;
    int s1 = (r * kTileDimMacaT_32 + threadIdx.y) % size1;
    int64_t out_offset = s0 * stride00 + s1 * stride10 + s2 * stride20 + s3 * stride30;
    int64_t in_offset = s0 * stride01 + s1 * stride11 + s2 * stride21 + s3 * stride31;

    float4 tmp = *reinterpret_cast<float4*>(in + in_offset);
    *reinterpret_cast<float4*>(&out[out_offset]) = *reinterpret_cast<float4*>(&tmp);
  }
}

template<int vt, typename res_t, typename arg0_t, typename TIndex, typename func_t, int v>
__global__ void elementwise_kernel_4_1_transpose12_copy_8(
      int64_t N,
      char*  out, char*  in, 
      TIndex size0, TIndex size1, TIndex size2, TIndex size3,
      TIndex stride00, TIndex stride01,
      TIndex stride10, TIndex stride11,
      TIndex stride20, TIndex stride21,
      TIndex stride30, TIndex stride31,
      func_t f) {
  using vec_t = at::native::memory::aligned_vector<arg0_t, vt>;

  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t linear_idx = idx * vt;

  if (linear_idx >= N) return;

  int64_t offset = 0;

  auto divmod_div = linear_idx / size0;
  auto divmod_mod = linear_idx % size0;
  linear_idx = divmod_div;
  offset += divmod_mod * stride01;

  divmod_div = linear_idx / size1;
  divmod_mod = linear_idx % size1;
  linear_idx = divmod_div;
  offset += divmod_mod * stride11;

  divmod_div = linear_idx / size2;
  divmod_mod = linear_idx % size2;
  linear_idx = divmod_div;
  offset += divmod_mod * stride21;

  divmod_div = linear_idx / size3;
  divmod_mod = linear_idx % size3;
  linear_idx = divmod_div;
  offset += divmod_mod * stride31;

  (reinterpret_cast<vec_t*>(out))[idx] = *(reinterpret_cast<vec_t*>(in + offset));
}

template<typename res_t, typename arg0_t, typename func_t, typename index_t>
static void launch_legacy_kernel_maca_2_1_transpose(
    int64_t N,
    char* out, char* in,
    index_t H, index_t W,
    const func_t& f) {
  // ndim = 3, arity = 1

  TORCH_INTERNAL_ASSERT(N >= 0 && N <= std::numeric_limits<int32_t>::max());
  // TORCH_INTERNAL_ASSERT(type_trait_is_half_maca_copy<res_t>());
  // TORCH_INTERNAL_ASSERT(type_trait_is_half_maca_copy<arg0_t>());
  if (N == 0) return;
  const uint32_t dh = (H + kTileDimMacaT - 1) / kTileDimMacaT;
  const uint32_t dw = (W + kTileDimMacaT - 1) / kTileDimMacaT;
  auto stream = at::cuda::getCurrentCUDAStream();
  if (H % kTileDimMacaT == 0 && W % kTileDimMacaT == 0 && (reinterpret_cast<uintptr_t>(out) % 4) == 0 && (reinterpret_cast<uintptr_t>(in) % 4) == 0) {
    if(std::is_same<at::Half, arg0_t>::value){
      elementwise_kernel_transpose_half_copy_64<res_t, arg0_t, uint32_t, func_t, at::Half><<<dim3(dw, dh, 1), dim3(kBlockRowsMacaT, kTileDimMacaT), 0, stream>>>(
        out, in, (uint32_t)H, (uint32_t)W, (uint32_t)dh, (uint32_t)dw, f);
    }
    else if (std::is_same<at::BFloat16, arg0_t>::value)
    {
      elementwise_kernel_transpose_half_copy_64<res_t, arg0_t, uint32_t, func_t, at::BFloat16><<<dim3(dw, dh, 1), dim3(kBlockRowsMacaT, kTileDimMacaT), 0, stream>>>(
        out, in, (uint32_t)H, (uint32_t)W, (uint32_t)dh, (uint32_t)dw, f);
    }
    else{
      assert(0);
    }
  } else {
    if(std::is_same<at::Half, arg0_t>::value){
      elementwise_kernel_transpose_copy<res_t, arg0_t, uint32_t, func_t, at::Half><<<dim3(dw, dh, 1), dim3(kTileDimMacaT, kBlockRowsMacaT), 0, stream>>>(
        out, in, (uint32_t)H, (uint32_t)W, (uint32_t)dh, (uint32_t)dw, f);
    }
    else if (std::is_same<at::BFloat16, arg0_t>::value){
      elementwise_kernel_transpose_copy<res_t, arg0_t, uint32_t, func_t, at::BFloat16><<<dim3(dw, dh, 1), dim3(kTileDimMacaT, kBlockRowsMacaT), 0, stream>>>(
        out, in, (uint32_t)H, (uint32_t)W, (uint32_t)dh, (uint32_t)dw, f);
    }
    else{
      assert(0);
    }

  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template<typename res_t, typename arg0_t, typename func_t, typename index_t>
static void launch_legacy_kernel_maca_2_1_transpose_fp32(
    int64_t N,
    char* out, char* in,
    index_t H, index_t W,
    const func_t& f) {
  // ndim = 3, arity = 1

  TORCH_INTERNAL_ASSERT(N >= 0 && N <= std::numeric_limits<int32_t>::max());
  // TORCH_INTERNAL_ASSERT(type_trait_is_half_maca_copy<res_t>());
  // TORCH_INTERNAL_ASSERT(type_trait_is_half_maca_copy<arg0_t>());
  if (N == 0) return;
  const uint32_t dh = (H + kTileDimMacaT - 1) / kTileDimMacaT;
  const uint32_t dw = (W + kTileDimMacaT - 1) / kTileDimMacaT;
  auto stream = at::cuda::getCurrentCUDAStream();
  elementwise_kernel_transpose_copy<res_t, arg0_t, uint32_t, func_t, float><<<dim3(dw, dh, 1), dim3(kTileDimMacaT, kBlockRowsMacaT), 0, stream>>>(
    out, in, (uint32_t)H, (uint32_t)W, (uint32_t)dh, (uint32_t)dw, f);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template<typename res_t, typename arg0_t, typename func_t, typename index_t, typename stride_t>
static void launch_legacy_kernel_maca_2_1_transpose_uncontiguous(
    int64_t N,
    char* out, char* in,
    index_t H, index_t W,
    stride_t s10, stride_t s01,
    const func_t& f) {

  TORCH_INTERNAL_ASSERT(N >= 0 && N <= std::numeric_limits<int32_t>::max());
  if (N == 0) return;
  const uint32_t dh = (H + kTileDimMacaT - 1) / kTileDimMacaT;
  const uint32_t dw = (W + kTileDimMacaT - 1) / kTileDimMacaT;
  auto stream = at::cuda::getCurrentCUDAStream();

  elementwise_kernel_transpose_copy_64_uncontiguous<res_t, arg0_t, uint32_t, func_t>
  <<<dim3(dw, dh, 1), dim3(kBlockRowsMacaT, kTileDimMacaT), 0, stream>>>(
  out, in,
  (uint32_t)H, (uint32_t)W,
  (uint32_t)dh, (uint32_t)dw,
  s10, s01,
  f
  );

  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template<typename res_t, typename arg0_t, typename func_t, typename index_t>
static void launch_legacy_kernel_maca_3_1_transpose_half_copy(
    int64_t N,
    char* out, char* in,
    index_t batch, index_t H, index_t W,
    const func_t& f) {
  // ndim = 3, arity = 1
  TORCH_INTERNAL_ASSERT(N >= 0 && N <= std::numeric_limits<int32_t>::max());
  // TORCH_INTERNAL_ASSERT(type_trait_is_half_maca_copy<res_t>());
  // TORCH_INTERNAL_ASSERT(type_trait_is_half_maca_copy<arg0_t>());
  if (N == 0) return;
  const uint32_t dh = (H + kTileDimMacaT - 1) / kTileDimMacaT;
  const uint32_t dw = (W + kTileDimMacaT - 1) / kTileDimMacaT;
  auto stream = at::cuda::getCurrentCUDAStream();
  //printf("half_copy: %d, %d, %d, %d，%d\n", batch, dh, dw, H, W);
  if (H % kTileDimMacaT == 0 && W % kTileDimMacaT == 0 && (reinterpret_cast<uintptr_t>(out) % 4) == 0 && (reinterpret_cast<uintptr_t>(in) % 4) == 0) {
    if(std::is_same<at::Half, arg0_t>::value){
      elementwise_kernel_transpose_half_copy_64<res_t, arg0_t, uint32_t, func_t, at::Half><<<dim3(dw, dh, batch), dim3(kBlockRowsMacaT, kTileDimMacaT), 0, stream>>>(
          out, in, (uint32_t)H, (uint32_t)W, (uint32_t)dh, (uint32_t)dw, f);
    }
    else if(std::is_same<at::BFloat16, arg0_t>::value){
      elementwise_kernel_transpose_half_copy_64<res_t, arg0_t, uint32_t, func_t, at::BFloat16><<<dim3(dw, dh, batch), dim3(kBlockRowsMacaT, kTileDimMacaT), 0, stream>>>(
          out, in, (uint32_t)H, (uint32_t)W, (uint32_t)dh, (uint32_t)dw, f);
    }
    else{
      assert(0);
    }
  } else if (H % 8 == 0 && W % 8 == 0 && (reinterpret_cast<uintptr_t>(out) % 4) == 0 && (reinterpret_cast<uintptr_t>(in) % 4) == 0) {
    if (std::is_same<at::Half, arg0_t>::value) {
      elementwise_kernel_transpose_half_copy_8<res_t, arg0_t, uint32_t, func_t, at::Half><<<dim3(dw, dh, batch), dim3(kBlockRowsMacaT, kTileDimMacaT), 0, stream>>>(
          out, in, (uint32_t)H, (uint32_t)W, (uint32_t)dh, (uint32_t)dw, f);
    }
    else if (std::is_same<at::BFloat16, arg0_t>::value) {
      elementwise_kernel_transpose_half_copy_8<res_t, arg0_t, uint32_t, func_t, at::BFloat16><<<dim3(dw, dh, batch), dim3(kBlockRowsMacaT, kTileDimMacaT), 0, stream>>>(
          out, in, (uint32_t)H, (uint32_t)W, (uint32_t)dh, (uint32_t)dw, f);
    }
    else {
      assert(0);
    }
  } else {
    if(std::is_same<at::Half, arg0_t>::value){
      elementwise_kernel_transpose_copy<res_t, arg0_t, uint32_t, func_t, at::Half><<<dim3(dw, dh, batch), dim3(kTileDimMacaT, kBlockRowsMacaT), 0, stream>>>(
        out, in, (uint32_t)H, (uint32_t)W, (uint32_t)dh, (uint32_t)dw, f);
    }
    else if(std::is_same<at::BFloat16, arg0_t>::value){
      elementwise_kernel_transpose_copy<res_t, arg0_t, uint32_t, func_t, at::BFloat16><<<dim3(dw, dh, batch), dim3(kTileDimMacaT, kBlockRowsMacaT), 0, stream>>>(
        out, in, (uint32_t)H, (uint32_t)W, (uint32_t)dh, (uint32_t)dw, f);
    }
    else{
      assert(0);
    }
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template<typename res_t, typename arg0_t, typename func_t, typename index_t>
static void launch_legacy_kernel_maca_3_1_transpose012_half_copy(
    int64_t N,
    char* out, char* in,
    index_t batch, index_t H, index_t W,
    const func_t& f) {
  // ndim = 3, arity = 1
  TORCH_INTERNAL_ASSERT(N >= 0 && N <= std::numeric_limits<int32_t>::max());
  // TORCH_INTERNAL_ASSERT(type_trait_is_half_maca_copy<res_t>());
  // TORCH_INTERNAL_ASSERT(type_trait_is_half_maca_copy<arg0_t>());
  if (N == 0) return;
  const uint32_t dh = (H + kTileDimMacaT - 1) / kTileDimMacaT;
  const uint32_t dw = (W + kTileDimMacaT - 1) / kTileDimMacaT;
  auto stream = at::cuda::getCurrentCUDAStream();
  if(std::is_same<at::Half, arg0_t>::value){
    elementwise_kernel_transpose012_half_copy_64<res_t, arg0_t, uint32_t, func_t, at::Half><<<dim3(dw, dh, batch), dim3(kBlockRowsMacaT, kTileDimMacaT), 0, stream>>>(
        out, in, (uint32_t)H, (uint32_t)W, (uint32_t)dh, (uint32_t)dw, f);
  }
  else if(std::is_same<at::BFloat16, arg0_t>::value){
    elementwise_kernel_transpose012_half_copy_64<res_t, arg0_t, uint32_t, func_t, at::BFloat16><<<dim3(dw, dh, batch), dim3(kBlockRowsMacaT, kTileDimMacaT), 0, stream>>>(
        out, in, (uint32_t)H, (uint32_t)W, (uint32_t)dh, (uint32_t)dw, f);
  }
  else{
    assert(0);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template<typename res_t, typename arg0_t, typename func_t, typename index_t, typename stride_t>
static void launch_legacy_kernel_maca_3_1_transpose12_half_copy(
    int64_t N,
    char* data0, char* data1,
    index_t size0, index_t size1, index_t size2,
    stride_t stride00, stride_t stride01,
    stride_t stride10, stride_t stride11,
    stride_t stride20, stride_t stride21,
    const func_t& f) {
  // ndim = 3, arity = 1
  TORCH_INTERNAL_ASSERT(N >= 0 && N <= std::numeric_limits<int32_t>::max());
  // TORCH_INTERNAL_ASSERT(type_trait_is_half_maca_copy<res_t>());
  // TORCH_INTERNAL_ASSERT(type_trait_is_half_maca_copy<arg0_t>());
  if (N == 0) return;
  if (size0 % kTileDimMacaT == 0){
    const uint32_t dh = (size1 * size2 + kTileDimMacaT - 1) / kTileDimMacaT;
    const uint32_t dw = (size0 + kTileDimMacaT - 1) / kTileDimMacaT;
    auto stream = at::cuda::getCurrentCUDAStream();
    dim3 block(kBlockRowsMacaT, kTileDimMacaT);
    dim3 grid(dh * dw);
    elementwise_kernel_3_1_transpose12_half_copy_64<res_t, arg0_t, uint32_t, func_t><<<grid, block, 0, stream>>>(
    data0, data1, 
    (uint32_t)size0, (uint32_t)size1, (uint32_t)size2,
    (uint32_t)stride00, (uint32_t)stride01,
    (uint32_t)stride10, (uint32_t)stride11,
    (uint32_t)stride20, (uint32_t)stride21,
    dh, dw, 
    f);
  }
  else if(size0 % kTileDimMacaT_32 == 0){
    const uint32_t dh = (size1 * size2 + kTileDimMacaT_32 - 1) / kTileDimMacaT_32;
    const uint32_t dw = (size0 + kTileDimMacaT_32 - 1) / kTileDimMacaT_32;
    auto stream = at::cuda::getCurrentCUDAStream();
    dim3 block(kBlockRowsMacaT / 2, kTileDimMacaT_32);
    dim3 grid(dh * dw);
    elementwise_kernel_3_1_transpose12_half_copy_32<res_t, arg0_t, uint32_t, func_t><<<grid, block, 0, stream>>>(
    data0, data1, 
    (uint32_t)size0, (uint32_t)size1, (uint32_t)size2,
    (uint32_t)stride00, (uint32_t)stride01,
    (uint32_t)stride10, (uint32_t)stride11,
    (uint32_t)stride20, (uint32_t)stride21,
    dh, dw,
    f);
  }
  else{
    assert(0);
  }
  
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template<typename res_t, typename arg0_t, typename func_t, typename index_t, typename stride_t>
static void launch_legacy_kernel_maca_3_1_transpose02_half_copy(
    int64_t N,
    char* out, char* in,
    index_t size0, index_t size1, index_t size2,
    stride_t stride00, stride_t stride01,
    stride_t stride10, stride_t stride11,
    stride_t stride20, stride_t stride21,
    const func_t& f) {
  TORCH_INTERNAL_ASSERT(N >= 0 && N <= std::numeric_limits<int32_t>::max());
  if (N == 0) return;

  const uint32_t dh = (size0 + kTileDimMacaT - 1) / kTileDimMacaT;
  const uint32_t dw = (size2 + kTileDimMacaT - 1) / kTileDimMacaT;
  dim3 grid(dw, dh, size1);
  dim3 block(kBlockRowsMacaT, kTileDimMacaT);
  auto stream = at::cuda::getCurrentCUDAStream();
  elementwise_kernel_transpose_3_1_transpose02_half_copy<res_t, arg0_t, uint32_t, func_t>
  <<<grid, block, 0, stream>>>(
    out, in,
    size0, size1, size2,
    stride00, stride01,
    stride10, stride11,
    stride20, stride21,
    f
  );
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template<int v, typename res_t, typename arg0_t, typename func_t, typename index_t, typename stride_t>
static void launch_legacy_kernel_maca_3_1_copy_opt(
    int64_t N,
    char* data0, char* data1,
    index_t size0, index_t size1, index_t size2,
    stride_t stride00, stride_t stride01,
    stride_t stride10, stride_t stride11,
    stride_t stride20, stride_t stride21,
    const func_t& f) {
  // ndim = 3, arity = 1
  TORCH_INTERNAL_ASSERT(N >= 0 && N <= std::numeric_limits<int32_t>::max());
  // TORCH_INTERNAL_ASSERT(type_trait_is_half_maca_copy<res_t>());
  // TORCH_INTERNAL_ASSERT(type_trait_is_half_maca_copy<arg0_t>());
  if (N == 0) return;
  if (size0 % kTileDimMacaT == 0){
    const uint32_t dh = (size1 * size2 + kTileDimMacaT - 1) / kTileDimMacaT;
    const uint32_t dw = (size0 + kTileDimMacaT - 1) / kTileDimMacaT;
    auto stream = at::cuda::getCurrentCUDAStream();
    dim3 block(kBlockRowsMacaT * (v / 2), kTileDimMacaT);
    dim3 grid(dh * dw);
    elementwise_kernel_3_1_copy_64_opt<res_t, arg0_t, uint32_t, func_t, v><<<grid, block, 0, stream>>>(
    data0, data1, 
    (uint32_t)size0, (uint32_t)size1, (uint32_t)size2,
    (uint32_t)stride00, (uint32_t)stride01,
    (uint32_t)stride10, (uint32_t)stride11,
    (uint32_t)stride20, (uint32_t)stride21,
    dh, dw, 
    f);
  }
  else if(size0 % kTileDimMacaT_32 == 0){
    const uint32_t dh = (size1 * size2 + kTileDimMacaT_32 - 1) / kTileDimMacaT_32;
    const uint32_t dw = (size0 + kTileDimMacaT_32 - 1) / kTileDimMacaT_32;
    auto stream = at::cuda::getCurrentCUDAStream();
    dim3 block(kBlockRowsMacaT / 2 * (v / 2), kTileDimMacaT_32);
    dim3 grid(dh * dw);
    elementwise_kernel_3_1_copy_32_opt<res_t, arg0_t, uint32_t, func_t, v><<<grid, block, 0, stream>>>(
    data0, data1, 
    (uint32_t)size0, (uint32_t)size1, (uint32_t)size2,
    (uint32_t)stride00, (uint32_t)stride01,
    (uint32_t)stride10, (uint32_t)stride11,
    (uint32_t)stride20, (uint32_t)stride21,
    dh, dw,
    f);
  }
  else{
    assert(0);
  }
  
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template<typename res_t, typename arg0_t, typename func_t, typename index_t>
static void launch_legacy_kernel_maca_3_1_transpose_short_copy(
    int64_t N,
    char* out, char* in,
    index_t batch, index_t H, index_t W,
    const func_t& f) {
  // ndim = 3, arity = 1
  TORCH_INTERNAL_ASSERT(N >= 0 && N <= std::numeric_limits<int32_t>::max());
  if (N == 0) return;
  const uint32_t dh = (H + kTileDimMacaT - 1) / kTileDimMacaT;
  const uint32_t dw = (W + kTileDimMacaT - 1) / kTileDimMacaT;
  auto stream = at::cuda::getCurrentCUDAStream();
//  printf("short_copy: %d, %d, %d, %d，%d\n", batch, dh, dw, H, W);
  elementwise_kernel_transpose_short_copy<res_t, arg0_t, uint32_t, func_t><<<batch * dh * dw, dim3(kTileDimMacaT, kBlockRowsMacaT), 0, stream>>>(
      out, in, (uint32_t)H, (uint32_t)W, (uint32_t)dh, (uint32_t)dw, f);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template<typename res_t, typename arg0_t, typename func_t, typename index_t>
static void launch_legacy_kernel_maca_3_1_transpose_float_copy(
    int64_t N,
    char* out, char* in,
    index_t batch, index_t H, index_t W,
    const func_t& f) {
  // ndim = 3, arity = 1
  TORCH_INTERNAL_ASSERT(N >= 0 && N <= std::numeric_limits<int32_t>::max());
  if (N == 0) return;
  const uint32_t dh = (H + kTileDimMacaT - 1) / kTileDimMacaT;
  const uint32_t dw = (W + kTileDimMacaT - 1) / kTileDimMacaT;
  auto stream = at::cuda::getCurrentCUDAStream();
//  printf("float_copy: %d, %d, %d, %d，%d\n", batch, dh, dw, H, W);
  elementwise_kernel_transpose_float_copy<res_t, arg0_t, uint32_t><<<batch * dh * dw, dim3(kTileDimMacaT, kBlockRowsMacaT), 0, stream>>>(
      out, in, (uint32_t)H, (uint32_t)W, (uint32_t)dh, (uint32_t)dw, f);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}


template<int nt, int vt, typename res_t , typename arg0_t, typename func_t, typename index_t, typename stride_t>
static void launch_legacy_kernel_maca_4_1_dim0_contiuous(
    int64_t N,
    char* data0, char* data1,
    index_t size0, index_t size1, index_t size2, index_t size3,
    stride_t stride00, stride_t stride01, 
    stride_t stride10, stride_t stride11, 
    stride_t stride20, stride_t stride21, 
    stride_t stride30, stride_t stride31, 
    const func_t& f) {
  // ndim = 3, arity = 1
  TORCH_INTERNAL_ASSERT(N >= 0 && N <= std::numeric_limits<int32_t>::max());
  if (N == 0) {
    return;
  }
  int vec = 4;
  if (sizeof(arg0_t) == 2 && size0 >= 128) {
    vec = 8;
  }

  auto stream = at::cuda::getCurrentCUDAStream();
  dim3 block(nt);
  dim3 grid((N + block.x * vec - 1) / (block.x * vec));
  if (vec == 4) {
    elementwise_kernel_4_1_dim0_contiuous<nt, 4, res_t, arg0_t, func_t, index_t, stride_t><<<grid, block, 0, stream>>>(
      N, data0, data1, size0, size1, size2, size3, stride00, stride01, stride10, stride11, stride20, stride21, stride30, stride31, f);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  } else if (vec == 8) {
    elementwise_kernel_4_1_dim0_contiuous<nt, 8, res_t, arg0_t, func_t, index_t, stride_t><<<grid, block, 0, stream>>>(
      N, data0, data1, size0, size1, size2, size3, stride00, stride01, stride10, stride11, stride20, stride21, stride30, stride31, f);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
}

template<int nt, int vt, typename res_t , typename arg0_t, typename func_t, typename index_t, typename stride_t>
static void launch_legacy_kernel_maca_4_1(
    int64_t N,
    char* data0, char* data1,
    index_t size0, index_t size1, index_t size2, index_t size3,
    stride_t stride00, stride_t stride01, 
    stride_t stride10, stride_t stride11, 
    stride_t stride20, stride_t stride21, 
    stride_t stride30, stride_t stride31, 
    const func_t& f) {
  // ndim = 3, arity = 1
  TORCH_INTERNAL_ASSERT(N >= 0 && N <= std::numeric_limits<int32_t>::max());
  if (N == 0) {
    return;
  }
  dim3 block(nt);
  dim3 grid((N + block.x * vt - 1) / (block.x * vt));
  auto stream = at::cuda::getCurrentCUDAStream();
  elementwise_kernel_4_1<nt, vt, res_t, arg0_t, func_t, index_t, stride_t><<<grid, block, 0, stream>>>(
      N, data0, data1, size0, size1, size2, size3, stride00, stride01, stride10, stride11, stride20, stride21, stride30, stride31, f);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template<int nt, int vt, typename res_t , typename arg0_t, typename func_t, typename index_t, typename stride_t>
static void launch_legacy_kernel_maca_4_1_input_lowdim_contiuous(
    int64_t N,
    char* data0, char* data1,
    index_t size0, index_t size1, index_t size2, index_t size3,
    stride_t stride00, stride_t stride01,
    stride_t stride10, stride_t stride11,
    stride_t stride20, stride_t stride21,
    stride_t stride30, stride_t stride31,
    const func_t& f) {
  // ndim = 3, arity = 1
  TORCH_INTERNAL_ASSERT(N >= 0 && N <= std::numeric_limits<int32_t>::max());
  if (N == 0) {
    return;
  }
  dim3 block(nt);
  dim3 grid((N + block.x * vt - 1) / (block.x * vt));
  auto stream = at::cuda::getCurrentCUDAStream();
  elementwise_kernel_4_1_input_lowdim_contiuous<nt, vt, res_t, arg0_t, func_t, index_t, stride_t><<<grid, block, 0, stream>>>(
      N, data0, data1, size0, size1, size2, size3, stride00, stride01, stride10, stride11, stride20, stride21, stride30, stride31, f);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template<int nt, int vt, typename res_t , typename arg0_t, typename func_t, typename index_t, typename stride_t>
static void launch_legacy_kernel_maca_4_1_dim0_pad(
    int64_t N,
    char* data0, char* data1,
    index_t size0, index_t size1, index_t size2, index_t size3,
    stride_t stride00, stride_t stride01, 
    stride_t stride10, stride_t stride11, 
    stride_t stride20, stride_t stride21, 
    stride_t stride30, stride_t stride31, 
    const func_t& f) {
  // ndim = 4, arity = 1
  TORCH_INTERNAL_ASSERT(N >= 0 && N <= std::numeric_limits<int32_t>::max());
  if (N == 0) {
    return;
  }

  int tids_per_group = ((size0 * sizeof(arg0_t) + 127) / 128) * 128 / (vt * sizeof(arg0_t));
  dim3 block(nt);
  dim3 grid(size1 / (nt / tids_per_group), size2 * size3);
  auto stream = at::cuda::getCurrentCUDAStream();
  if (reinterpret_cast<uintptr_t>(data0) % 64 == 0) {
    elementwise_kernel_4_1_dim0_pad<nt, vt, res_t, arg0_t, func_t, index_t, stride_t><<<grid, block, 0, stream>>>(
        N, data0, data1, size0, size1, size2, size3, stride00, stride01, stride10, stride11, stride20, stride21, stride30, stride31, f, tids_per_group);
  } else {
    elementwise_kernel_4_1_dim0_pad_align<nt, vt, res_t, arg0_t, func_t, index_t, stride_t><<<grid, block, 0, stream>>>(
        N, data0, data1, size0, size1, size2, size3, stride00, stride01, stride10, stride11, stride20, stride21, stride30, stride31, f, tids_per_group);    
  }

  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template<int v, typename res_t , typename arg0_t, typename func_t, typename index_t, typename stride_t>
static void launch_legacy_kernel_maca_4_1_transpose12_copy(
    int64_t N,
    char* data0, char* data1,
    index_t size0, index_t size1, index_t size2, index_t size3,
    stride_t stride00, stride_t stride01,
    stride_t stride10, stride_t stride11,
    stride_t stride20, stride_t stride21,
    stride_t stride30, stride_t stride31,
    const func_t& f) {
  // ndim = 4, arity = 1
  TORCH_INTERNAL_ASSERT(N >= 0 && N <= std::numeric_limits<int32_t>::max());
  // TORCH_INTERNAL_ASSERT(type_trait_is_half_maca_copy<res_t>());
  // TORCH_INTERNAL_ASSERT(type_trait_is_half_maca_copy<arg0_t>());
  if (N == 0) return;
  if (size0 % kTileDimMacaT == 0) {
    const uint32_t dh = (size1 * size2 + kTileDimMacaT - 1) / kTileDimMacaT;
    const uint32_t dw = (size0 + kTileDimMacaT - 1) / kTileDimMacaT;
    auto stream = at::cuda::getCurrentCUDAStream();
    dim3 block(kBlockRowsMacaT * (v / 2), kTileDimMacaT);
    dim3 grid(size3 * dh * dw);
    elementwise_kernel_4_1_transpose12_copy_64<res_t, arg0_t, uint32_t, func_t, v><<<grid, block, 0, stream>>>(
    data0, data1, 
    (uint32_t)size0, (uint32_t)size1, (uint32_t)size2, (uint32_t)size3,
    (uint32_t)stride00, (uint32_t)stride01,
    (uint32_t)stride10, (uint32_t)stride11,
    (uint32_t)stride20, (uint32_t)stride21,
    (uint32_t)stride30, (uint32_t)stride31,
    dh, dw,
    f);
  } else if (size0 % kTileDimMacaT_32 == 0) {
    const uint32_t dh = (size1 * size2 + kTileDimMacaT_32 - 1) / kTileDimMacaT_32;
    const uint32_t dw = (size0 + kTileDimMacaT_32 - 1) / kTileDimMacaT_32;
    auto stream = at::cuda::getCurrentCUDAStream();
    dim3 block((kBlockRowsMacaT / 2) * (v / 2), kTileDimMacaT_32);
    dim3 grid(size3 * dh * dw);
    elementwise_kernel_4_1_transpose12_copy_32<res_t, arg0_t, uint32_t, func_t, v><<<grid, block, 0, stream>>>(
    data0, data1, 
    (uint32_t)size0, (uint32_t)size1, (uint32_t)size2, (uint32_t)size3,
    (uint32_t)stride00, (uint32_t)stride01,
    (uint32_t)stride10, (uint32_t)stride11,
    (uint32_t)stride20, (uint32_t)stride21,
    (uint32_t)stride30, (uint32_t)stride31,
    dh, dw,
    f);
  } else if (size0 % 8 == 0) {
    dim3 block(128);
    auto stream = at::cuda::getCurrentCUDAStream();
    if (v==2) {
      const int vt = 8;
      dim3 grid((N + block.x * vt - 1) / (block.x * vt));
      elementwise_kernel_4_1_transpose12_copy_8<vt, res_t, arg0_t, uint32_t, func_t, v><<<grid, block, 0, stream>>>(
        N,
        data0, data1, 
        (uint32_t)size0, (uint32_t)size1, (uint32_t)size2, (uint32_t)size3,
        (uint32_t)stride00, (uint32_t)stride01,
        (uint32_t)stride10, (uint32_t)stride11,
        (uint32_t)stride20, (uint32_t)stride21,
        (uint32_t)stride30, (uint32_t)stride31,
        f
      );
    } else if (v==4) {
      const int vt = 4;
      dim3 grid((N + block.x * vt - 1) / (block.x * vt));
      elementwise_kernel_4_1_transpose12_copy_8<vt, res_t, arg0_t, uint32_t, func_t, v><<<grid, block, 0, stream>>>(
        N,
        data0, data1, 
        (uint32_t)size0, (uint32_t)size1, (uint32_t)size2, (uint32_t)size3,
        (uint32_t)stride00, (uint32_t)stride01,
        (uint32_t)stride10, (uint32_t)stride11,
        (uint32_t)stride20, (uint32_t)stride21,
        (uint32_t)stride30, (uint32_t)stride31,
        f
      );
    } else {
      assert(0);
    }

  } else {
    assert(0);
  }
  
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template<int nt, int vt, typename res_t , typename arg0_t, typename func_t, typename index_t, typename stride_t>
C10_LAUNCH_BOUNDS_2(nt, 4)
__global__ void elementwise_kernel_5_1(
    int64_t N,
    char* data0, char* data1,
    index_t size0, index_t size1, index_t size2, index_t size3, index_t size4,
    stride_t stride00, stride_t stride01, 
    stride_t stride10, stride_t stride11,
    stride_t stride20, stride_t stride21,
    stride_t stride30, stride_t stride31,
    stride_t stride40, stride_t stride41,
    func_t f) {
  // ndim = 5, arity = 1, narg = 2
  int tid = threadIdx.x;
  int nv = nt * vt;
  int64_t idx = nv * blockIdx.x + tid;
  #pragma unroll
  for (int i = 0; i < vt; i++) {
    if (idx < N) {
      int64_t offsets[2];
      auto linear_idx = idx;
      constexpr int NARGS = 2;
      constexpr int MAX_DIMS = 5;
      #pragma unroll
      for (int arg = 0; arg < NARGS; arg++) {
        offsets[arg] = 0;
      }
      
      // dim = 0
      auto divmod_div = linear_idx / size0;
      auto divmod_mod = linear_idx % size0;
      linear_idx = divmod_div;
      int arg = 0;
      offsets[arg] += divmod_mod * stride00;
      arg = 1;
      offsets[arg] += divmod_mod * stride01;
      // dim = 1
      divmod_div = linear_idx / size1;
      divmod_mod = linear_idx % size1;
      linear_idx = divmod_div;
      arg = 0;
      offsets[arg] += divmod_mod * stride10;
      arg = 1;
      offsets[arg] += divmod_mod * stride11;
      // dim = 2
      divmod_div = linear_idx / size2;
      divmod_mod = linear_idx % size2;
      linear_idx = divmod_div;
      arg = 0;
      offsets[arg] += divmod_mod * stride20;
      arg = 1;
      offsets[arg] += divmod_mod * stride21;
      // dim = 3
      divmod_div = linear_idx / size3;
      divmod_mod = linear_idx % size3;
      linear_idx = divmod_div;
      arg = 0;
      offsets[arg] += divmod_mod * stride30;
      arg = 1;
      offsets[arg] += divmod_mod * stride31;
      // dim = 4
      divmod_div = linear_idx / size4;
      divmod_mod = linear_idx % size4;
      linear_idx = divmod_div;
      arg = 0;
      offsets[arg] += divmod_mod * stride40;
      arg = 1;
      offsets[arg] += divmod_mod * stride41;

      res_t* out = (res_t*)(data0 + offsets[0]);
      *out = f(func_reinterpret_cast<arg0_t>(data1 + offsets[1]));

      idx += nt;
    }
  }
}

template<int nt, int vt, typename res_t , typename arg0_t, typename func_t, typename index_t, typename stride_t>
C10_LAUNCH_BOUNDS_2(nt, 4)
__global__ void elementwise_kernel_5_1_lowdim_contiguous(
    int64_t N,
    char* data0, char* data1,
    index_t size0, index_t size1, index_t size2, index_t size3, index_t size4,
    stride_t stride00, stride_t stride01,
    stride_t stride10, stride_t stride11,
    stride_t stride20, stride_t stride21,
    stride_t stride30, stride_t stride31,
    stride_t stride40, stride_t stride41,
    func_t f) {
  int64_t linear_idx = (blockIdx.x * blockDim.x + threadIdx.x) * vt;
  int64_t res_idx = linear_idx * stride00;
  if (linear_idx >= N) return;

  int64_t offset = 0;
  //dim0
  auto divmod_div = linear_idx / size0;
  auto divmod_mod = linear_idx % size0;
  linear_idx = divmod_div;
  offset += divmod_mod * stride01;
  //dim1
  divmod_div = linear_idx / size1;
  divmod_mod = linear_idx % size1;
  linear_idx = divmod_div;
  offset += divmod_mod * stride11;
  //dim2
  divmod_div = linear_idx / size2;
  divmod_mod = linear_idx % size2;
  linear_idx = divmod_div;
  offset += divmod_mod * stride21;
  //dim3
  divmod_div = linear_idx / size3;
  divmod_mod = linear_idx % size3;
  linear_idx = divmod_div;
  offset += divmod_mod * stride31;
  //dim4
  divmod_div = linear_idx / size4;
  divmod_mod = linear_idx % size4;
  linear_idx = divmod_div;
  offset += divmod_mod * stride41;

  using vec_t = at::native::memory::aligned_vector<arg0_t, vt>;
  *(reinterpret_cast<vec_t*>(data0 + res_idx)) = *(reinterpret_cast<vec_t*>(data1 + offset));
}


template<int nt, int vt, typename res_t , typename arg0_t, typename func_t, typename index_t, typename stride_t>
static void launch_legacy_kernel_maca_5_1(
    int64_t N,
    char* data0, char* data1,
    index_t size0, index_t size1, index_t size2, index_t size3, index_t size4,
    stride_t stride00, stride_t stride01, 
    stride_t stride10, stride_t stride11, 
    stride_t stride20, stride_t stride21, 
    stride_t stride30, stride_t stride31, 
    stride_t stride40, stride_t stride41, 
    const func_t& f) {
  // ndim = 5, arity = 1
  TORCH_INTERNAL_ASSERT(N >= 0 && N <= std::numeric_limits<int32_t>::max());
  if (N == 0) {
    return;
  }
  dim3 block(nt);
  dim3 grid((N + block.x * vt - 1) / (block.x * vt));
  auto stream = at::cuda::getCurrentCUDAStream();
  elementwise_kernel_5_1<nt, vt, res_t, arg0_t, func_t, index_t, stride_t><<<grid, block, 0, stream>>>(
      N, data0, data1, 
      size0, size1, size2, size3, size4, 
      stride00, stride01, 
      stride10, stride11, 
      stride20, stride21,
      stride30, stride31, 
      stride40, stride41, 
  f);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}


template<int nt, int vt, typename res_t , typename arg0_t, typename func_t, typename index_t, typename stride_t>
static void launch_legacy_kernel_maca_5_1_lowdim_contiguous(
    int64_t N,
    char* data0, char* data1,
    index_t size0, index_t size1, index_t size2, index_t size3, index_t size4,
    stride_t stride00, stride_t stride01,
    stride_t stride10, stride_t stride11,
    stride_t stride20, stride_t stride21,
    stride_t stride30, stride_t stride31,
    stride_t stride40, stride_t stride41,
    const func_t& f) {
  // ndim = 5, arity = 1
  TORCH_INTERNAL_ASSERT(N >= 0 && N <= std::numeric_limits<int32_t>::max());
  if (N == 0) {
    return;
  }
  dim3 block(nt);
  dim3 grid((N + block.x * vt - 1) / (block.x * vt));
  auto stream = at::cuda::getCurrentCUDAStream();
  elementwise_kernel_5_1_lowdim_contiguous<nt, vt, res_t, arg0_t, func_t, index_t, stride_t><<<grid, block, 0, stream>>>(
      N, data0, data1,
      size0, size1, size2, size3, size4,
      stride00, stride01,
      stride10, stride11,
      stride20, stride21,
      stride30, stride31,
      stride40, stride41,
  f);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}


template <typename func_t>
void gpu_kernel_impl_maca_arity1_copy(TensorIteratorBase& iter, const func_t& f) {
  using traits = function_traits<func_t>;
  using arg0_t = typename traits::result_type;
  using arg1_t = typename traits::template arg<0>::type;
  constexpr int ntensors = traits::arity + 1;

  TORCH_INTERNAL_ASSERT(iter.can_use_32bit_indexing());
  TORCH_INTERNAL_ASSERT(iter.ninputs() == traits::arity);
  TORCH_INTERNAL_ASSERT(iter.noutputs() == 1);

  at::detail::Array<char*, ntensors> data;
  at::detail::Array<ScalarType, ntensors> dtypes;
  for (int i = 0; i < ntensors; i++) {
    data[i] = (char*)iter.data_ptr(i);
    dtypes[i] = iter.dtype(i);
  }

  int64_t numel = iter.numel();

  bool contiguous = iter.is_contiguous();
  bool dynamic_casting = needs_dynamic_casting<func_t>::check(iter);
  auto offset_calc = ::make_offset_calculator<traits::arity + 1>(iter);

  constexpr int unroll_factor = sizeof(arg0_t) >= 4 ? 2 : 4;
  constexpr int narity = traits::arity;
  int ndim = iter.ndim();
  assert(narity == 1);

  if (!dynamic_casting) {
    if (contiguous) {
      get_elementwise_info<narity + 1>(ndim, narity, offset_calc, dtypes, "p_e_cp_vec_1_1", f);
      launch_vectorized_kernel(numel, f, data);
    } else {
        if (ndim == 1) {
          bool can_vec4 = (reinterpret_cast<uint64_t>(data[0]) % 4 == 0) &&
                            (reinterpret_cast<uint64_t>(data[1]) % 4 == 0);
          bool disable_opt_1_1_broadcast = at::maca::get_maca_disable_elementwise_kernel_1_1_broadcast();
          bool enable_opt_1_1_dilation = at::maca::get_maca_enable_elementwise_kernel_1_1_dilation();
          if ((sizeof(arg0_t) == 2 || sizeof(arg0_t) == 4) && maca_likely(!disable_opt_1_1_broadcast) &&
            offset_calc.strides_[0][0] == sizeof(arg0_t) && offset_calc.strides_[0][1] == 0 &&
            offset_calc.sizes_[0].divisor % C10_WARP_SIZE == 0) {
            get_elementwise_info<narity + 1>(ndim, narity, offset_calc, dtypes, "p_e_cp_launch_legacy_kernel_maca_1_1_broadcast", f);
            launch_legacy_kernel_maca_1_1_broadcast<8*C10_WARP_SIZE, unroll_factor, arg0_t, typename traits::template arg<0>::type>(
              numel,
              data[0], data[1],  // data
              offset_calc.sizes_[0].divisor,
              offset_calc.strides_[0][0], offset_calc.strides_[0][1],
              f);
          } else if (sizeof(arg0_t) == 4 && maca_unlikely(enable_opt_1_1_dilation) &&
                    offset_calc.sizes_[0].divisor % C10_WARP_SIZE == 0 && can_vec4 &&
                    offset_calc.strides_[0][0] == 8 && offset_calc.strides_[0][1] == 4
                    ) {
              get_elementwise_info<narity + 1>(ndim, narity, offset_calc, dtypes, "p_e_cp_launch_legacy_kernel_maca_1_1_dilation", f);
              launch_legacy_kernel_maca_1_1_dilation<128, 4, arg0_t, typename traits::template arg<0>::type>(
              numel,
              data[0], data[1],  // data
              offset_calc.sizes_[0].divisor,
              offset_calc.strides_[0][0], offset_calc.strides_[0][1],
              f);
          } else {
            get_elementwise_info<narity + 1>(ndim, narity, offset_calc, dtypes, "p_e_cp_launch_legacy_kernel_maca_1_1", f);
            launch_legacy_kernel_maca_1_1<128, unroll_factor, arg0_t, typename traits::template arg<0>::type>(
              numel,
              data[0], data[1],  // data
              offset_calc.sizes_[0].divisor,
              offset_calc.strides_[0][0], offset_calc.strides_[0][1], 
              f);
          }
        } else if (ndim == 2) {
          // N :4194304
          // size0 :1024
          // size1 :4096
          // stride00:2
          // stride01:8192
          // stride10:2048
          // stride11:2
          bool disable_broadcast_opt = at::maca::get_maca_disable_elementwise_kernel_2_1_broadcast();
          bool disable_2_1_dim0_contiguous_opt = at::maca::get_maca_disable_elementwise_kernel_2_1_dim0_contiguous();
          bool enable_n_1_dim0_pad_opt = at::maca::get_maca_enable_elementwise_kernel_n_1_dim0_pad();
          if (((dtypes[0] == ScalarType::Half && dtypes[1] == ScalarType::Half) ||
              (dtypes[0] == ScalarType::BFloat16 && dtypes[1] == ScalarType::BFloat16)) &&
              offset_calc.strides_[0][0] == 2 && offset_calc.strides_[1][1] == 2 &&
              offset_calc.strides_[0][1] == 2 * offset_calc.sizes_[1].divisor &&
              offset_calc.strides_[1][0] == 2 * offset_calc.sizes_[0].divisor) {
            get_elementwise_info<narity + 1>(ndim, narity, offset_calc, dtypes, "p_e_cp_launch_legacy_kernel_maca_2_1_transpose", f);
            launch_legacy_kernel_maca_2_1_transpose<arg0_t, typename traits::template arg<0>::type>(
              numel,
              data[0], data[1],  // data
              offset_calc.sizes_[0].divisor, offset_calc.sizes_[1].divisor,
              f);
          } else if (dtypes[0] == ScalarType::Float && dtypes[1] == ScalarType::Float &&
              offset_calc.strides_[0][0] == 4 && offset_calc.strides_[1][1] == 4 &&
              offset_calc.strides_[0][1] == 4 * offset_calc.sizes_[1].divisor &&
              offset_calc.strides_[1][0] == 4 * offset_calc.sizes_[0].divisor) {
            get_elementwise_info<narity + 1>(ndim, narity, offset_calc, dtypes, "p_e_cp_launch_legacy_kernel_maca_2_1_transpose_fp32", f);
            launch_legacy_kernel_maca_2_1_transpose_fp32<arg0_t, typename traits::template arg<0>::type>(
              numel,
              data[0], data[1],  // data
              offset_calc.sizes_[0].divisor, offset_calc.sizes_[1].divisor,
              f);
          } else if (((dtypes[0] == ScalarType::Half && dtypes[1] == ScalarType::Half) ||
                     (dtypes[0] == ScalarType::BFloat16 && dtypes[1] == ScalarType::BFloat16)) &&
                     offset_calc.strides_[0][0] == 2 && offset_calc.strides_[1][1] == 2 &&
                     offset_calc.strides_[1][0] % 4 == 0 && offset_calc.strides_[0][1] % 4 == 0 &&
                     offset_calc.sizes_[0].divisor % kTileDimMacaT == 0 && offset_calc.sizes_[1].divisor % kTileDimMacaT == 0 &&
                     (reinterpret_cast<uintptr_t>(data[0]) % 4) == 0 && (reinterpret_cast<uintptr_t>(data[1]) % 4) == 0 &&
                     !at::maca::get_maca_disable_elementwise_kernel_2_1_transpose_uncontiguous()) {
            get_elementwise_info<narity + 1>(ndim, narity, offset_calc, dtypes, "p_e_cp_launch_legacy_kernel_maca_2_1_transpose_uncontiguous", f);
            launch_legacy_kernel_maca_2_1_transpose_uncontiguous<arg0_t, typename traits::template arg<0>::type>(
              numel,
              data[0], data[1],  // data
              offset_calc.sizes_[0].divisor, offset_calc.sizes_[1].divisor,
              offset_calc.strides_[1][0], offset_calc.strides_[0][1],
              f);
          } else if (((sizeof(arg0_t) == 2 && sizeof(arg1_t) == 2) || (sizeof(arg0_t) == 4 && sizeof(arg1_t) == 4) || (sizeof(arg0_t) == 8 && sizeof(arg1_t) == 8)) &&
              offset_calc.strides_[0][0] == sizeof(arg0_t) && offset_calc.strides_[0][1] == sizeof(arg1_t) &&
              offset_calc.strides_[1][0] == (offset_calc.sizes_[0].divisor * sizeof(arg0_t)) &&
              offset_calc.strides_[1][1] == 0 && offset_calc.sizes_[0].divisor >= C10_WARP_SIZE &&
              maca_likely(!disable_broadcast_opt) && !(offset_calc.sizes_[0].divisor == 30522 && offset_calc.sizes_[1].divisor == 16384)) {
            auto size0 = offset_calc.sizes_[0].divisor;
            auto size1 = offset_calc.sizes_[1].divisor;
            auto stride00 = offset_calc.strides_[0][0];
            auto stride01 = offset_calc.strides_[0][1];
            auto stride10 = offset_calc.strides_[1][0];
            auto stride11 = offset_calc.strides_[1][1];
            get_elementwise_info<narity + 1>(ndim, narity, offset_calc, dtypes, "p_e_cp_launch_legacy_kernel_maca_2_1_broadcast", f);
            launch_legacy_kernel_maca_2_1_broadcast<64, unroll_factor, arg0_t, typename traits::template arg<0>::type>(
              numel,
              data[0], data[1],  // data
              size0, size1, 
              stride00, stride01, 
              stride10, stride11, 
              f);
          } else if (maca_unlikely(enable_n_1_dim0_pad_opt) &&
              reinterpret_cast<uintptr_t>(data[0]) % 16 == 0 && reinterpret_cast<uintptr_t>(data[1]) % 16 == 0 &&
              // half & bfloat
              ((sizeof(arg0_t) == 2 && sizeof(arg1_t) == 2 && offset_calc.strides_[0][0] == 2 && offset_calc.strides_[0][1] == 2 &&
              (offset_calc.sizes_[1].divisor == 245760 || offset_calc.sizes_[1].divisor == 393408) &&
              ((offset_calc.sizes_[0].divisor == 40 && offset_calc.strides_[1][0] == 160 &&
              (offset_calc.strides_[1][1] == 80 || offset_calc.strides_[1][1] == 480)) ||
              (offset_calc.sizes_[0].divisor == 80 && offset_calc.strides_[1][0] == 480 && offset_calc.strides_[1][1] == 160))) || 
              // float
              (sizeof(arg0_t) == 4 && sizeof(arg1_t) == 4 && offset_calc.strides_[0][0] == 4 && offset_calc.strides_[0][1] == 4 &&
              (offset_calc.sizes_[0].divisor == 40 && offset_calc.strides_[1][0] == 320 &&
              (offset_calc.strides_[1][1] == 160 || offset_calc.strides_[1][1] == 320))))
            ) {
            // dim0 contiguous & padding
            // [shape, stride_out, stride_in] =
            // [(245760, 40), (80, 1), (40, 1)], [(245760, 80), (240, 1), (80, 1)], [(245760, 40), (80, 1), (240, 1)]
            get_elementwise_info<narity + 1>(ndim, narity, offset_calc, dtypes, "p_e_cp_launch_legacy_kernel_maca_2_1_dim0_pad", f);
            if (sizeof(arg0_t) == 2) {
              launch_legacy_kernel_maca_2_1_dim0_pad<64, 8, arg0_t, typename traits::template arg<0>::type>(
                numel,
                data[0], data[1],  // data
                offset_calc.sizes_[0].divisor, offset_calc.sizes_[1].divisor,
                offset_calc.strides_[0][0], offset_calc.strides_[0][1],
                offset_calc.strides_[1][0], offset_calc.strides_[1][1],
                f);
            } else if (sizeof(arg0_t) == 4) {
              launch_legacy_kernel_maca_2_1_dim0_pad<64, 4, arg0_t, typename traits::template arg<0>::type>(
                numel,
                data[0], data[1],  // data
                offset_calc.sizes_[0].divisor, offset_calc.sizes_[1].divisor,
                offset_calc.strides_[0][0], offset_calc.strides_[0][1],
                offset_calc.strides_[1][0], offset_calc.strides_[1][1],
                f);
            }
          } else if ((reinterpret_cast<uintptr_t>(data[0]) % 4) == 0 && reinterpret_cast<uintptr_t>(data[1]) % 4 == 0 &&
                    offset_calc.strides_[1][0] % 4 == 0 && offset_calc.strides_[1][1] % 4 == 0 &&
                    ((sizeof(arg0_t) == 2 && sizeof(arg1_t) == 2) || (sizeof(arg0_t) == 4 && sizeof(arg1_t) == 4)) &&
                    offset_calc.strides_[0][0] == sizeof(arg0_t) && offset_calc.strides_[0][1] == sizeof(arg1_t) &&
                    offset_calc.sizes_[0].divisor % 2 == 0 && maca_likely(!disable_2_1_dim0_contiguous_opt)){
            auto size0 = offset_calc.sizes_[0].divisor;
            auto size1 = offset_calc.sizes_[1].divisor;
            auto stride00 = offset_calc.strides_[0][0];
            auto stride01 = offset_calc.strides_[0][1];
            auto stride10 = offset_calc.strides_[1][0];
            auto stride11 = offset_calc.strides_[1][1];
            get_elementwise_info<narity + 1>(ndim, narity, offset_calc, dtypes, "p_e_cp_launch_legacy_kernel_maca_2_1_dim0_contiguous", f);
            launch_legacy_kernel_maca_2_1_dim0_contiguous<64, unroll_factor, arg0_t, typename traits::template arg<0>::type>(
              numel,
              data[0], data[1],  // data
              size0, size1,
              stride00, stride01,
              stride10, stride11,
              f);
          }
          else {
            get_elementwise_info<narity + 1>(ndim, narity, offset_calc, dtypes, "p_e_cp_launch_legacy_kernel_maca_2_1", f);
            launch_legacy_kernel_maca_2_1<128, unroll_factor, arg0_t, typename traits::template arg<0>::type>(
              numel,
              data[0], data[1],  // data
              offset_calc.sizes_[0].divisor, offset_calc.sizes_[1].divisor, 
              offset_calc.strides_[0][0], offset_calc.strides_[0][1], 
              offset_calc.strides_[1][0], offset_calc.strides_[1][1], 
              f);
          }
        } else if (ndim == 3) {
          auto s0 = offset_calc.sizes_[0].divisor;
          auto s1 = offset_calc.sizes_[1].divisor;
          auto s2 = offset_calc.sizes_[2].divisor;
          int transopose_3_1_dim_threshold = 20;
          bool disable_elementwise_3_1_copy_opt = at::maca::get_maca_disable_elementwise_3_1_copy_opt_kernel();
          bool disable_elementwise_3_1_transpose_half_copy = at::maca::get_maca_disable_elementwise_3_1_transpose_half_copy_kernel();
          if (((dtypes[0] == ScalarType::Half && dtypes[1] == ScalarType::Half) ||
              (dtypes[0] == ScalarType::BFloat16 && dtypes[1] == ScalarType::BFloat16)) &&
              offset_calc.sizes_[0].divisor > transopose_3_1_dim_threshold && offset_calc.sizes_[1].divisor > transopose_3_1_dim_threshold &&
              offset_calc.strides_[0][0] == 2 && offset_calc.strides_[1][0] == 2 * offset_calc.sizes_[0].divisor &&
              offset_calc.strides_[2][0] == 2 * offset_calc.sizes_[0].divisor * offset_calc.sizes_[1].divisor &&
              offset_calc.strides_[0][1] == 2 * offset_calc.sizes_[1].divisor && offset_calc.strides_[1][1] == 2 &&
              offset_calc.strides_[2][1] == offset_calc.strides_[2][0] && maca_likely(!disable_elementwise_3_1_transpose_half_copy)) {
            get_elementwise_info<narity + 1>(ndim, narity, offset_calc, dtypes, "p_e_cp_launch_legacy_kernel_maca_3_1_transpose_half_copy", f);
            launch_legacy_kernel_maca_3_1_transpose_half_copy<arg0_t, typename traits::template arg<0>::type>(
              numel,
              data[0], data[1],  // data
              offset_calc.sizes_[2].divisor, offset_calc.sizes_[0].divisor, offset_calc.sizes_[1].divisor,
              f);
         } else if (((dtypes[0] == ScalarType::Half && dtypes[1] == ScalarType::Half) ||
              (dtypes[0] == ScalarType::BFloat16 && dtypes[1] == ScalarType::BFloat16)) && 
              // example: shape = [128,4,64] stride_output = [2,256,1024] stride_input = [1024,8,2] half
              offset_calc.strides_[0][0] == 2 && offset_calc.strides_[1][0] == 2 * offset_calc.sizes_[0].divisor && 
              offset_calc.strides_[2][0] == 2 * offset_calc.sizes_[0].divisor * offset_calc.sizes_[1].divisor && 
              offset_calc.strides_[0][1] == 2 * offset_calc.sizes_[1].divisor * offset_calc.sizes_[0].divisor && 
              offset_calc.strides_[2][1] == 2 && offset_calc.strides_[1][1] == 2 * offset_calc.sizes_[1].divisor && 
              offset_calc.sizes_[0].divisor % kTileDimMacaT == 0 && offset_calc.sizes_[2].divisor % kTileDimMacaT == 0 && 
              offset_calc.sizes_[1].divisor % kTileDimMacaT == 0 && (reinterpret_cast<uintptr_t>(data[0]) % 4) == 0 && 
              (reinterpret_cast<uintptr_t>(data[1]) % 4) == 0) {
            // for float16 or bfloat16 offset_calc.sizes_[1].divisor should % 2 == 0, there % 64 == 0
            get_elementwise_info<narity + 1>(ndim, narity, offset_calc, dtypes, "p_e_cp_launch_legacy_kernel_maca_3_1_transpose012_half_copy", f);
            launch_legacy_kernel_maca_3_1_transpose012_half_copy<arg0_t, typename traits::template arg<0>::type>(
              numel,
              data[0], data[1],  // data
              offset_calc.sizes_[1].divisor, offset_calc.sizes_[0].divisor, offset_calc.sizes_[2].divisor, 
              f);
          } else if (dtypes[0] == ScalarType::Float && dtypes[1] == ScalarType::Float && 
              offset_calc.sizes_[0].divisor > transopose_3_1_dim_threshold && offset_calc.sizes_[1].divisor > transopose_3_1_dim_threshold &&
              offset_calc.strides_[0][0] == 4 && offset_calc.strides_[1][0] == 4 * offset_calc.sizes_[0].divisor &&
              offset_calc.strides_[2][0] == 4 * offset_calc.sizes_[0].divisor * offset_calc.sizes_[1].divisor &&
              offset_calc.strides_[0][1] == 4 * offset_calc.sizes_[1].divisor && offset_calc.strides_[1][1] == 4 &&
              offset_calc.strides_[2][1] == offset_calc.strides_[2][0]) {
            get_elementwise_info<narity + 1>(ndim, narity, offset_calc, dtypes, "p_e_cp_launch_legacy_kernel_maca_3_1_transpose_float_copy", f);
            launch_legacy_kernel_maca_3_1_transpose_float_copy<arg0_t, typename traits::template arg<0>::type>(
              numel,
              data[0], data[1],  // data
              offset_calc.sizes_[2].divisor, offset_calc.sizes_[0].divisor, offset_calc.sizes_[1].divisor,
              f);
          } else if (((dtypes[0] == ScalarType::Half && dtypes[1] == ScalarType::Half) ||
              (dtypes[0] == ScalarType::BFloat16 && dtypes[1] == ScalarType::BFloat16)) &&
              offset_calc.strides_[0][0] == 2 && offset_calc.strides_[1][0] == 2 * offset_calc.sizes_[0].divisor &&
              offset_calc.strides_[2][0] == 2 * offset_calc.sizes_[0].divisor * offset_calc.sizes_[1].divisor &&
              ((offset_calc.strides_[0][1] == 2 && offset_calc.strides_[1][1] == 2 * offset_calc.sizes_[0].divisor * offset_calc.sizes_[2].divisor &&
              offset_calc.strides_[2][1] == 2 * offset_calc.sizes_[0].divisor) ||
              (offset_calc.strides_[0][1] == 2 && offset_calc.strides_[1][1] % 4 == 0 &&
              offset_calc.strides_[1][1] > 2 * offset_calc.sizes_[0].divisor * offset_calc.sizes_[2].divisor &&
              offset_calc.strides_[2][1] == 2 * offset_calc.sizes_[0].divisor)) &&
              (offset_calc.sizes_[0].divisor % kTileDimMacaT == 0 || offset_calc.sizes_[0].divisor % kTileDimMacaT_32 == 0) && 
              (reinterpret_cast<uintptr_t>(data[0]) % 4) == 0 && (reinterpret_cast<uintptr_t>(data[1]) % 4) == 0){
                get_elementwise_info<narity + 1>(ndim, narity, offset_calc, dtypes, "p_e_cp_launch_legacy_kernel_maca_3_1_transpose12_half_copy", f);
                launch_legacy_kernel_maca_3_1_transpose12_half_copy<arg0_t, typename traits::template arg<0>::type>(
                  numel,
                  data[0], data[1],  // data
                  offset_calc.sizes_[0].divisor, offset_calc.sizes_[1].divisor, offset_calc.sizes_[2].divisor,
                  offset_calc.strides_[0][0], offset_calc.strides_[0][1], offset_calc.strides_[1][0], offset_calc.strides_[1][1], 
                  offset_calc.strides_[2][0], offset_calc.strides_[2][1],
                  f);
          } else if (((dtypes[0] == ScalarType::Half && dtypes[1] == ScalarType::Half) ||
                     (dtypes[0] == ScalarType::BFloat16 && dtypes[1] == ScalarType::BFloat16)) &&
                      s0 % kTileDimMacaT == 0 && s2 % kTileDimMacaT == 0 &&
                      offset_calc.strides_[0][0] == 2 &&  offset_calc.strides_[2][1] == 2 &&
                      offset_calc.strides_[1][0] % 4 == 0 && offset_calc.strides_[2][0] % 4 == 0 &&
                      offset_calc.strides_[0][1] % 4 == 0 && offset_calc.strides_[1][1] % 4 == 0 &&
                      (reinterpret_cast<uintptr_t>(data[0]) % 4) == 0 && (reinterpret_cast<uintptr_t>(data[1]) % 4) == 0 &&
                      !at::maca::get_maca_disable_elementwise_3_1_transpose02_half_copy_kernel()) {
              get_elementwise_info<narity + 1>(ndim, narity, offset_calc, dtypes, "p_e_cp_launch_legacy_kernel_maca_3_1_transpose02_half_copy", f);
              launch_legacy_kernel_maca_3_1_transpose02_half_copy<arg0_t, typename traits::template arg<0>::type>(
              numel,
              data[0], data[1],  // data
              offset_calc.sizes_[0].divisor, offset_calc.sizes_[1].divisor, offset_calc.sizes_[2].divisor,
              offset_calc.strides_[0][0], offset_calc.strides_[0][1],
              offset_calc.strides_[1][0], offset_calc.strides_[1][1],
              offset_calc.strides_[2][0], offset_calc.strides_[2][1],
              f);
          } else if (((dtypes[0] == ScalarType::Half && dtypes[1] == ScalarType::Half) ||
              (dtypes[0] == ScalarType::BFloat16 && dtypes[1] == ScalarType::BFloat16)) && 
              maca_likely(!disable_elementwise_3_1_copy_opt) && check_opt_dim_3(s1, offset_calc.strides_[1][0], offset_calc.strides_[2][0]) && 
              offset_calc.strides_[0][0] == 2 && offset_calc.strides_[1][0] == 2 * s0 * 4 && 
              (offset_calc.strides_[2][0] == 2 * s2 || offset_calc.strides_[2][0] == 2 * s2 * 2 || offset_calc.strides_[2][0] == 2 * s2 * 4) &&
              offset_calc.strides_[0][1] == 2 && offset_calc.strides_[1][1] == 2 * s0 * 2 && 
              (offset_calc.strides_[2][1] == 2 * s2 || offset_calc.strides_[2][1] == 2 * s2 * 2 || offset_calc.strides_[2][1] == 2 * s2 * 4) &&
              offset_calc.sizes_[0].divisor % kTileDimMacaT == 0 && 
              (reinterpret_cast<uintptr_t>(data[0]) % 4) == 0 && (reinterpret_cast<uintptr_t>(data[1]) % 4) == 0 && s2 % 2 == 0) {
                // stride10 != stride20 && stride20/stride10>=s1
                get_elementwise_info<narity + 1>(ndim, narity, offset_calc, dtypes, "p_e_cp_launch_legacy_kernel_maca_3_1_copy_opt", f);
                launch_legacy_kernel_maca_3_1_copy_opt<sizeof(arg0_t), arg0_t, typename traits::template arg<0>::type>(
                  numel,
                  data[0], data[1],  // data
                  offset_calc.sizes_[0].divisor, offset_calc.sizes_[1].divisor, offset_calc.sizes_[2].divisor,
                  offset_calc.strides_[0][0], offset_calc.strides_[0][1], offset_calc.strides_[1][0], offset_calc.strides_[1][1], 
                  offset_calc.strides_[2][0], offset_calc.strides_[2][1],
                  f);
          } else if ((sizeof(arg0_t) == 4 || sizeof(arg0_t) == 2) && 
              maca_likely(!disable_elementwise_3_1_copy_opt) && check_opt_dim_3(s1, offset_calc.strides_[1][0], offset_calc.strides_[2][0]) && 
              ((offset_calc.strides_[0][0] == sizeof(arg0_t) && offset_calc.strides_[1][0] == sizeof(arg0_t) * s0 && 
              (offset_calc.strides_[2][0] == sizeof(arg0_t) * s0 * s1 * 2 || offset_calc.strides_[2][0] == sizeof(arg0_t) * s0 * s1) && 
              offset_calc.strides_[0][1] == sizeof(arg0_t) && 
              (sizeof(arg0_t) == 4 && offset_calc.strides_[1][1] == sizeof(arg0_t) * s0 * s2 && offset_calc.strides_[2][1] == sizeof(arg0_t) * s0 || 
              s0 % 3 == 0 && offset_calc.strides_[1][1] == sizeof(arg0_t) * s0 / 3 && (offset_calc.strides_[1][1] == sizeof(arg0_t) * s0 / 3) * 4)) || 
              (sizeof(arg0_t) == 4 && offset_calc.strides_[0][0] == sizeof(arg0_t) && offset_calc.strides_[1][0] == sizeof(arg0_t) * 513 && 
              offset_calc.strides_[2][0] == sizeof(arg0_t) * s0 * 513 * 2 && offset_calc.strides_[0][1] == sizeof(arg0_t) && 
              offset_calc.strides_[1][1] == sizeof(arg0_t) * 513) && offset_calc.strides_[2][1] == sizeof(arg0_t) * s0 * 513 * 2) && 
              (offset_calc.sizes_[0].divisor % kTileDimMacaT == 0 || offset_calc.sizes_[0].divisor % kTileDimMacaT_32 == 0) && 
              (reinterpret_cast<uintptr_t>(data[0]) % 4) == 0 && (reinterpret_cast<uintptr_t>(data[1]) % 4) == 0 && s2 % 2 == 0) {
              get_elementwise_info<narity + 1>(ndim, narity, offset_calc, dtypes, "p_e_cp_launch_legacy_kernel_maca_3_1_copy_opt", f);
              launch_legacy_kernel_maca_3_1_copy_opt<sizeof(arg0_t), arg0_t, typename traits::template arg<0>::type>(
                numel,
                data[0], data[1],  // data
                offset_calc.sizes_[0].divisor, offset_calc.sizes_[1].divisor, offset_calc.sizes_[2].divisor,
                offset_calc.strides_[0][0], offset_calc.strides_[0][1], offset_calc.strides_[1][0], offset_calc.strides_[1][1], 
                offset_calc.strides_[2][0], offset_calc.strides_[2][1],
                f);
          } else if (maca_likely(!at::maca::get_maca_disable_elementwise_3_1_broadcast_kernel()) && offset_calc.strides_[0][0] == sizeof(arg0_t) &&
              (sizeof(arg0_t) == 2 || sizeof(arg0_t) == 4) &&
              offset_calc.strides_[1][0] == (offset_calc.sizes_[0].divisor * sizeof(arg0_t)) &&
              offset_calc.strides_[2][0] == (offset_calc.sizes_[0].divisor * offset_calc.sizes_[1].divisor * sizeof(arg0_t)) &&
              offset_calc.strides_[2][1] == 0 &&
              (offset_calc.sizes_[0].divisor * offset_calc.sizes_[1].divisor) >= C10_WARP_SIZE) {
            get_elementwise_info<narity + 1>(ndim, narity, offset_calc, dtypes, "p_e_cp_launch_legacy_kernel_maca_3_1_broadcast", f);
              // XXX(yuliu): one prerequisite for broadcast is divisible by warp_size, otherwise will cause
              // partial memory write and alignment fault.
              // 8 * C10_WARP_SIZE to be considered as best block size.
            launch_legacy_kernel_maca_3_1_broadcast<8*C10_WARP_SIZE, unroll_factor, arg0_t, typename traits::template arg<0>::type>(
              numel,
              data[0], data[1],  // data
              offset_calc.sizes_[0].divisor, offset_calc.sizes_[1].divisor, offset_calc.sizes_[2].divisor,
              offset_calc.strides_[0][0], offset_calc.strides_[0][1],
              offset_calc.strides_[1][0], offset_calc.strides_[1][1],
              offset_calc.strides_[2][0], offset_calc.strides_[2][1],
              f); 
          } else if ((sizeof(arg0_t) == 4 || sizeof(arg0_t) == 2) && (offset_calc.strides_[0][0] == sizeof(arg0_t) &&
              offset_calc.strides_[1][0] == sizeof(arg0_t) * s0 && offset_calc.strides_[2][0] > sizeof(arg0_t) * s0 * s1 &&
              offset_calc.strides_[2][0] % 4 == 0) && (offset_calc.strides_[0][1] == sizeof(arg0_t) &&
              offset_calc.strides_[1][1] % (sizeof(arg0_t) * s0) == 0 && offset_calc.strides_[1][1] > sizeof(arg0_t) * s0 &&
              offset_calc.strides_[2][1] % (offset_calc.strides_[1][1] * s1) == 0) && s0 * s1 >= C10_WARP_SIZE &&
              (reinterpret_cast<uintptr_t>(data[0]) % 4) == 0 && (reinterpret_cast<uintptr_t>(data[1]) % 4) == 0 &&
              maca_likely(!at::maca::get_maca_disable_elementwise_3_1_dim0_contiguous())) {
            get_elementwise_info<narity + 1>(ndim, narity, offset_calc, "p_dtypes, e_cp_launch_legacy_kernel_maca_3_1_dim0_contiguous", f);
            launch_legacy_kernel_maca_3_1_dim0_contiguous<8*C10_WARP_SIZE, unroll_factor, arg0_t, typename traits::template arg<0>::type>(
              numel,
              data[0], data[1],  // data
              offset_calc.sizes_[0].divisor, offset_calc.sizes_[1].divisor, offset_calc.sizes_[2].divisor,
              offset_calc.strides_[0][0], offset_calc.strides_[0][1],
              offset_calc.strides_[1][0], offset_calc.strides_[1][1],
              offset_calc.strides_[2][0], offset_calc.strides_[2][1],
              f);
          } else {
//            std::cout << "match2" << std::endl;
            get_elementwise_info<narity + 1>(ndim, narity, offset_calc, dtypes, "p_e_cp_launch_legacy_kernel_maca_3_1", f);
            launch_legacy_kernel_maca_3_1<128, 4, arg0_t, typename traits::template arg<0>::type>(
              numel,
              data[0], data[1],  // data
              offset_calc.sizes_[0].divisor, offset_calc.sizes_[1].divisor, offset_calc.sizes_[2].divisor,
              offset_calc.strides_[0][0], offset_calc.strides_[0][1], 
              offset_calc.strides_[1][0], offset_calc.strides_[1][1], 
              offset_calc.strides_[2][0], offset_calc.strides_[2][1], 
              f);
          }
        } else if (ndim == 4) {
          auto s0 = offset_calc.sizes_[0].divisor;
          auto s1 = offset_calc.sizes_[1].divisor;
          auto s2 = offset_calc.sizes_[2].divisor;
          size_t s01 = offset_calc.sizes_[0].divisor * offset_calc.sizes_[1].divisor;
          size_t s012 = offset_calc.sizes_[0].divisor * offset_calc.sizes_[1].divisor * offset_calc.sizes_[2].divisor;

          bool disable_elementwise_4_1_copy_opt = at::maca::get_maca_disable_elementwise_4_1_copy_opt_kernel();
          bool disable_elementwise_4_1_transpose12_copy = at::maca::get_maca_disable_elementwise_4_1_transpose12_copy_kernel();
          bool disable_4_1_input_lowdim_contiuous = at::maca::get_maca_disable_elementwise_kernel_4_1_input_lowdim_contiuous();
          bool enable_n_1_dim0_pad_opt = at::maca::get_maca_enable_elementwise_kernel_n_1_dim0_pad();
          bool disable_4_1_dim0_contiguous = at::maca::get_maca_disable_4_1_dim0_contiguous();

          //1 and 2 dim transpose
          if ((sizeof(arg0_t) == 2 || sizeof(arg0_t) == 4) && maca_likely(!disable_elementwise_4_1_transpose12_copy) &&
             offset_calc.strides_[0][0] == sizeof(arg0_t) &&
             offset_calc.strides_[1][0] == sizeof(arg0_t) * offset_calc.sizes_[0].divisor &&
             offset_calc.strides_[2][0] == sizeof(arg0_t) * offset_calc.sizes_[0].divisor * offset_calc.sizes_[1].divisor &&
             offset_calc.strides_[3][0] == sizeof(arg0_t) * offset_calc.sizes_[0].divisor * offset_calc.sizes_[1].divisor * offset_calc.sizes_[2].divisor &&
             offset_calc.strides_[0][1] == sizeof(arg0_t) &&
             offset_calc.strides_[1][1] == sizeof(arg0_t) * offset_calc.sizes_[0].divisor * offset_calc.sizes_[2].divisor &&
             offset_calc.strides_[2][1] == sizeof(arg0_t) * offset_calc.sizes_[0].divisor &&
             offset_calc.strides_[3][1] == offset_calc.strides_[3][0] && 
             offset_calc.sizes_[0].divisor % 8 == 0 && 
             (reinterpret_cast<uintptr_t>(data[0]) % 4) == 0 && (reinterpret_cast<uintptr_t>(data[1]) % 4) == 0
          ) {
            get_elementwise_info<narity + 1>(ndim, narity, offset_calc, dtypes, "p_e_cp_launch_legacy_kernel_maca_4_1_transpose12_copy", f);
            launch_legacy_kernel_maca_4_1_transpose12_copy<sizeof(arg0_t), arg0_t, typename traits::template arg<0>::type>(
              numel,
              data[0], data[1],  // data
              offset_calc.sizes_[0].divisor, offset_calc.sizes_[1].divisor, offset_calc.sizes_[2].divisor, offset_calc.sizes_[3].divisor,
              offset_calc.strides_[0][0], offset_calc.strides_[0][1], 
              offset_calc.strides_[1][0], offset_calc.strides_[1][1], 
              offset_calc.strides_[2][0], offset_calc.strides_[2][1], 
              offset_calc.strides_[3][0], offset_calc.strides_[3][1], 
              f);
          } else if((sizeof(arg0_t) == 2 || sizeof(arg0_t) == 4) && maca_likely(!disable_elementwise_4_1_copy_opt) && 
             check_opt_dim_4(s1, s2, offset_calc.strides_[1][0], offset_calc.strides_[2][0], offset_calc.strides_[3][0]) && 
             ((offset_calc.strides_[0][0] == sizeof(arg0_t) && offset_calc.strides_[1][0] == sizeof(arg0_t) * s0 &&
             offset_calc.strides_[2][0] == sizeof(arg0_t) * s0 * s1 && offset_calc.strides_[3][0] == sizeof(arg0_t) * s0 * s1 * s2 &&
             offset_calc.strides_[0][1] == sizeof(arg0_t) && offset_calc.strides_[1][1] == sizeof(arg0_t) * s0 * 348 &&
             offset_calc.strides_[2][1] == sizeof(arg0_t) * s0 && offset_calc.strides_[3][1] == sizeof(arg0_t) * s0 * s1 * 348) || 
             (offset_calc.strides_[0][1] == sizeof(arg0_t) && offset_calc.strides_[1][1] == sizeof(arg0_t) * s0 * s2 &&
             offset_calc.strides_[2][1] == sizeof(arg0_t) * s0 && offset_calc.strides_[3][1] == sizeof(arg0_t) * s0 * s1 * s2 &&
             offset_calc.strides_[0][0] == sizeof(arg0_t) && offset_calc.strides_[1][0] == sizeof(arg0_t) * s0 &&
             offset_calc.strides_[2][0] == sizeof(arg0_t) * s0 * 348 && offset_calc.strides_[3][0] == sizeof(arg0_t) * s0 * s2 * 348) || 
             (offset_calc.strides_[0][0] == sizeof(arg0_t) && offset_calc.strides_[1][0] == sizeof(arg0_t) * s0 * 2 &&
             offset_calc.strides_[2][0] == sizeof(arg0_t) * s0 * s1 * 2 && offset_calc.strides_[3][0] == sizeof(arg0_t) * s0 * s1 * s2 *2 &&
             offset_calc.strides_[0][1] == sizeof(arg0_t) && offset_calc.strides_[1][1] == sizeof(arg0_t) * s0 * s2 &&
             offset_calc.strides_[2][1] == sizeof(arg0_t) * s0 && offset_calc.strides_[3][1] == sizeof(arg0_t) * s0 * s1 * s2) || 
             (offset_calc.strides_[0][0] == sizeof(arg0_t) && offset_calc.strides_[1][0] == sizeof(arg0_t) * s0 * 2 &&
             offset_calc.strides_[2][0] == sizeof(arg0_t) * s0 * s1 * 2 && offset_calc.strides_[3][0] == sizeof(arg0_t) * s0 * s1 * s2 *2 &&
             offset_calc.strides_[0][1] == sizeof(arg0_t) && offset_calc.strides_[1][1] == sizeof(arg0_t) * s0 * s2 *2 &&
             offset_calc.strides_[2][1] == sizeof(arg0_t) * s0 *2 && offset_calc.strides_[3][1] == sizeof(arg0_t) * s0 * s1 * s2 * 2) || 
             (offset_calc.strides_[0][0] == sizeof(arg0_t) && offset_calc.strides_[1][0] == sizeof(arg0_t) * s0 &&
             offset_calc.strides_[2][0] == sizeof(arg0_t) * s0 * 348 && offset_calc.strides_[3][0] == sizeof(arg0_t) * s0 * s1 * 1392 &&
             offset_calc.strides_[0][1] == sizeof(arg0_t) && offset_calc.strides_[1][1] == sizeof(arg0_t) * s0 * 2048 &&
             offset_calc.strides_[2][1] == sizeof(arg0_t) * s0 && offset_calc.strides_[3][1] == sizeof(arg0_t) * s0 * s1 * 2048) || 
             (offset_calc.strides_[0][0] == sizeof(arg0_t) && offset_calc.strides_[1][0] == sizeof(arg0_t) * s0 &&
             offset_calc.strides_[2][0] == sizeof(arg0_t) * s0 * s1 && offset_calc.strides_[3][0] == sizeof(arg0_t) * s0 * s1 * s2 &&
             offset_calc.strides_[0][1] == sizeof(arg0_t) && offset_calc.strides_[1][1] == sizeof(arg0_t) * s0 * s2 * 3 &&
             offset_calc.strides_[2][1] == sizeof(arg0_t) * s0 && offset_calc.strides_[3][1] ==  sizeof(arg0_t) * s0 * s1 * s2 * 3)) && 
             (offset_calc.sizes_[0].divisor % kTileDimMacaT == 0 || offset_calc.sizes_[0].divisor % kTileDimMacaT_32 == 0) && 
             (reinterpret_cast<uintptr_t>(data[0]) % 4) == 0 && (reinterpret_cast<uintptr_t>(data[1]) % 4) == 0
          ) {
            get_elementwise_info<narity + 1>(ndim, narity, offset_calc, dtypes, "p_e_cp_launch_legacy_kernel_maca_4_1_transpose12_copy", f);
            launch_legacy_kernel_maca_4_1_transpose12_copy<sizeof(arg0_t), arg0_t, typename traits::template arg<0>::type>(
              numel,
              data[0], data[1],  // data
              offset_calc.sizes_[0].divisor, offset_calc.sizes_[1].divisor, offset_calc.sizes_[2].divisor, offset_calc.sizes_[3].divisor,
              offset_calc.strides_[0][0], offset_calc.strides_[0][1], 
              offset_calc.strides_[1][0], offset_calc.strides_[1][1], 
              offset_calc.strides_[2][0], offset_calc.strides_[2][1], 
              offset_calc.strides_[3][0], offset_calc.strides_[3][1], 
              f);
          } else if (((sizeof(arg0_t) == 2 || sizeof(arg0_t) == 4)) &&
                      offset_calc.strides_[0][0] == sizeof(arg0_t) && offset_calc.strides_[1][0] == sizeof(arg0_t) * s0 &&
                      offset_calc.strides_[2][0] == sizeof(arg0_t) * s01 && offset_calc.strides_[3][0] == sizeof(arg0_t) * s012 &&
                      offset_calc.strides_[0][1] == sizeof(arg0_t) && offset_calc.strides_[1][1] % 4 == 0 &&
                      offset_calc.strides_[2][1] % 4 == 0 && offset_calc.strides_[3][1] % 4 == 0 &&
                      s0 % 4 == 0 && maca_likely(!disable_4_1_input_lowdim_contiuous) &&
                      (reinterpret_cast<uintptr_t>(data[0]) % 4) == 0 && (reinterpret_cast<uintptr_t>(data[1]) % 4) == 0){
            get_elementwise_info<narity + 1>(ndim, narity, offset_calc, dtypes, "p_e_cp_launch_legacy_kernel_maca_4_1_input_lowdim_contiuous", f);
            launch_legacy_kernel_maca_4_1_input_lowdim_contiuous<128, 4, arg0_t, typename traits::template arg<0>::type>(
              numel,
              data[0], data[1],  // data
              offset_calc.sizes_[0].divisor, offset_calc.sizes_[1].divisor, offset_calc.sizes_[2].divisor, offset_calc.sizes_[3].divisor,
              offset_calc.strides_[0][0], offset_calc.strides_[0][1],
              offset_calc.strides_[1][0], offset_calc.strides_[1][1],
              offset_calc.strides_[2][0], offset_calc.strides_[2][1],
              offset_calc.strides_[3][0], offset_calc.strides_[3][1],
              f);
          } else if  ((sizeof(arg0_t) == 2 && sizeof(arg1_t) == 2) && maca_unlikely(enable_n_1_dim0_pad_opt) &&
              reinterpret_cast<uintptr_t>(data[0]) % 16 == 0 && reinterpret_cast<uintptr_t>(data[1]) % 16 == 0 &&
              s1 == 24 && s2 == 5 && offset_calc.sizes_[3].divisor == 2048 && offset_calc.strides_[0][0] == 2 &&
              ((s0 == 40 && offset_calc.strides_[1][0] == 160) || (s0 == 80 && offset_calc.strides_[1][0] == 480)) &&
              offset_calc.strides_[2][0] == s1 * offset_calc.strides_[1][0] && offset_calc.strides_[3][0] == s2 * offset_calc.strides_[2][0] &&
              offset_calc.strides_[0][1] == 2 &&
              (((s0 == 40 || s0 == 80) && offset_calc.strides_[1][1] == 160) || (s0 == 40 && offset_calc.strides_[1][1] == 80)) &&
              offset_calc.strides_[2][1] == offset_calc.sizes_[3].divisor * offset_calc.strides_[3][1] &&
              offset_calc.strides_[3][1] == s1 * offset_calc.strides_[1][1]) {
            // dim0 contiguous & padding
            // [shape, stride_out, stride_in] =
            // [(2048, 5, 24, 40), (9600, 1920, 80, 1), (1920, 3932160, 80, 1)], [(2048, 5, 24, 80), (28800, 5760, 240, 1), (1920, 3932160, 80, 1)],
            // [(2048, 5, 24, 40), (9600, 1920, 80, 1), (960, 1966080, 40, 1)]
            get_elementwise_info<narity + 1>(ndim, narity, offset_calc, dtypes, "p_e_cp_launch_legacy_kernel_maca_4_1_dim0_pad", f);
            launch_legacy_kernel_maca_4_1_dim0_pad<64, 8, arg0_t, typename traits::template arg<0>::type>(
              numel,
              data[0], data[1],  // data
              offset_calc.sizes_[0].divisor, offset_calc.sizes_[1].divisor, offset_calc.sizes_[2].divisor, offset_calc.sizes_[3].divisor,
              offset_calc.strides_[0][0], offset_calc.strides_[0][1],
              offset_calc.strides_[1][0], offset_calc.strides_[1][1],
              offset_calc.strides_[2][0], offset_calc.strides_[2][1],
              offset_calc.strides_[3][0], offset_calc.strides_[3][1],
              f);          
          } else if (((sizeof(arg0_t) == 2 || sizeof(arg0_t) == 4)) &&
                      offset_calc.strides_[0][0] == sizeof(arg0_t) && offset_calc.strides_[0][1] == sizeof(arg0_t) && 
                      offset_calc.strides_[1][0] % 4 == 0 && offset_calc.strides_[2][0] % 4 == 0 && offset_calc.strides_[3][0] % 4 == 0 &&
                      offset_calc.strides_[1][1] % 4 == 0 && offset_calc.strides_[2][1] % 4 == 0 && offset_calc.strides_[3][1] % 4 == 0 &&
                      s0 % C10_WARP_SIZE == 0 && maca_likely(!disable_4_1_dim0_contiguous) && numel >= 5120 &&
                      (reinterpret_cast<uintptr_t>(data[0]) % 4) == 0 &&
                      (reinterpret_cast<uintptr_t>(data[1]) % 4) == 0 ){
            get_elementwise_info<narity + 1>(ndim, narity, offset_calc, dtypes, "p_e_cp_launch_legacy_kernel_maca_4_1_dim0_contiguous", f);
            launch_legacy_kernel_maca_4_1_dim0_contiuous<128, 4, arg0_t, typename traits::template arg<0>::type>(
              numel,
              data[0], data[1],  // data
              offset_calc.sizes_[0].divisor, offset_calc.sizes_[1].divisor, offset_calc.sizes_[2].divisor, offset_calc.sizes_[3].divisor,
              offset_calc.strides_[0][0], offset_calc.strides_[0][1],
              offset_calc.strides_[1][0], offset_calc.strides_[1][1],
              offset_calc.strides_[2][0], offset_calc.strides_[2][1],
              offset_calc.strides_[3][0], offset_calc.strides_[3][1],
              f);
          } else {
            get_elementwise_info<narity + 1>(ndim, narity, offset_calc, dtypes, "p_e_cp_launch_legacy_kernel_maca_4_1", f);
            launch_legacy_kernel_maca_4_1<128, unroll_factor, arg0_t, typename traits::template arg<0>::type>(
              numel,
              data[0], data[1],  // data
              offset_calc.sizes_[0].divisor, offset_calc.sizes_[1].divisor, offset_calc.sizes_[2].divisor, offset_calc.sizes_[3].divisor,
              offset_calc.strides_[0][0], offset_calc.strides_[0][1],
              offset_calc.strides_[1][0], offset_calc.strides_[1][1],
              offset_calc.strides_[2][0], offset_calc.strides_[2][1],
              offset_calc.strides_[3][0], offset_calc.strides_[3][1],
              f);
          }
        } else if (ndim == 5) {
          size_t st00 = offset_calc.strides_[0][0]; size_t st01 = offset_calc.strides_[0][1];
          size_t st10 = offset_calc.strides_[1][0]; size_t st11 = offset_calc.strides_[1][1];
          size_t st20 = offset_calc.strides_[2][0]; size_t st21 = offset_calc.strides_[2][1];
          size_t st30 = offset_calc.strides_[3][0]; size_t st31 = offset_calc.strides_[3][1];
          size_t st40 = offset_calc.strides_[4][0]; size_t st41 = offset_calc.strides_[4][1];
          size_t s0 = offset_calc.sizes_[0].divisor;
          size_t s01 = offset_calc.sizes_[0].divisor * offset_calc.sizes_[1].divisor;
          size_t s012 = offset_calc.sizes_[0].divisor * offset_calc.sizes_[1].divisor * offset_calc.sizes_[2].divisor;
          size_t s0123 = offset_calc.sizes_[0].divisor * offset_calc.sizes_[1].divisor * 
                         offset_calc.sizes_[2].divisor * offset_calc.sizes_[3].divisor;
          bool disable_5_1_lowdim_contiguous = at::maca::get_maca_disable_elementwise_kernel_5_1_lowdim_contiguous();
          if ((sizeof(arg0_t) == 2 || sizeof(arg0_t) == 4) && s0 % 8 == 0 && maca_likely(!disable_5_1_lowdim_contiguous) &&
               st00 == sizeof(arg0_t) && st10 == s0 * sizeof(arg0_t) && st20 == s01 * sizeof(arg0_t) &&
               st30 == s012 * sizeof(arg0_t) && st40 == s0123 * sizeof(arg0_t) &&
               st01 == sizeof(arg0_t) && st11 % 4 == 0 && st21 % 4 == 0 && st31 % 4 == 0 && st41 % 4 == 0 &&
               (reinterpret_cast<uintptr_t>(data[0]) % 4) == 0 && (reinterpret_cast<uintptr_t>(data[1]) % 4) == 0) {
              get_elementwise_info<narity + 1>(ndim, narity, offset_calc, dtypes, "p_e_cp_launch_legacy_kernel_maca_5_1_lowdim_contiguous", f);
              launch_legacy_kernel_maca_5_1_lowdim_contiguous<128, 2, arg0_t, typename traits::template arg<0>::type>(
              numel,
              data[0], data[1],  // data
              offset_calc.sizes_[0].divisor, offset_calc.sizes_[1].divisor, offset_calc.sizes_[2].divisor, 
              offset_calc.sizes_[3].divisor, offset_calc.sizes_[4].divisor,
              offset_calc.strides_[0][0], offset_calc.strides_[0][1],
              offset_calc.strides_[1][0], offset_calc.strides_[1][1],
              offset_calc.strides_[2][0], offset_calc.strides_[2][1],
              offset_calc.strides_[3][0], offset_calc.strides_[3][1],
              offset_calc.strides_[4][0], offset_calc.strides_[4][1],
              f);
          } else {
           get_elementwise_info<narity + 1>(ndim, narity, offset_calc, dtypes, "p_e_cp_launch_legacy_kernel_maca_5_1", f);
           launch_legacy_kernel_maca_5_1<128, unroll_factor, arg0_t, typename traits::template arg<0>::type>(
              numel,
              data[0], data[1],  // data
              offset_calc.sizes_[0].divisor, offset_calc.sizes_[1].divisor, offset_calc.sizes_[2].divisor, 
              offset_calc.sizes_[3].divisor, offset_calc.sizes_[4].divisor,
              offset_calc.strides_[0][0], offset_calc.strides_[0][1],
              offset_calc.strides_[1][0], offset_calc.strides_[1][1],
              offset_calc.strides_[2][0], offset_calc.strides_[2][1],
              offset_calc.strides_[3][0], offset_calc.strides_[3][1],
              offset_calc.strides_[4][0], offset_calc.strides_[4][1],
              f);
          }
        } else if (gpu_kernel_impl_maca_copy_high_dim(iter, f)) {

        } else {
          get_elementwise_info<narity + 1>(ndim, narity, offset_calc, dtypes, "p_e_cp_launch_legacy_kernel", f);
          launch_legacy_kernel<128,unroll_factor>(numel, [=]GPU_LAMBDA(int idx) {
            auto offsets = offset_calc.get(idx);
            arg0_t* out = (arg0_t*)(data[0] + offsets[0]);
            *out = invoke(f, &data.data[1], &offsets.data[1], 1);
          });
        }
    }
  } else {
    if (contiguous) {
      auto loader = memory::LoadWithCast<traits::arity>(iter);
      auto storer = memory::StoreWithCast<1>(iter);
      auto input_offset_calculator = TrivialOffsetCalculator<traits::arity>();
      auto output_offset_calculator = TrivialOffsetCalculator<1>();
      bool enable_flag = !at::maca::get_maca_disable_unroll_float_opt() && numel >= 2048 &&
                         (uint64_t)data[0]%4 == 0 && (uint64_t)data[1]%4 == 0 &&
                         ((dtypes[0] == ScalarType::Half && dtypes[1] == ScalarType::Float) ||
                          (dtypes[0] == ScalarType::Float && dtypes[1] == ScalarType::Half) ||
                          (dtypes[0] == ScalarType::BFloat16 && dtypes[1] == ScalarType::Float) ||
                          (dtypes[0] == ScalarType::Float && dtypes[1] == ScalarType::BFloat16) ||
                          (dtypes[0] == ScalarType::Int && dtypes[1] == ScalarType::Float) ||
                          (dtypes[0] == ScalarType::Half && dtypes[1] == ScalarType::Int) ||
                          (dtypes[0] == ScalarType::Long && dtypes[1] == ScalarType::Int) ||
                          (dtypes[0] == ScalarType::Int && dtypes[1] == ScalarType::Long) ||
                          (dtypes[0] == ScalarType::Half && dtypes[1] == ScalarType::BFloat16) ||
                          (dtypes[0] == ScalarType::BFloat16 && dtypes[1] == ScalarType::Half) ||
                          (dtypes[0] == ScalarType::Float && dtypes[1] == ScalarType::Long) ||
                          (dtypes[0] == ScalarType::Bool && dtypes[1] == ScalarType::Float) ||
                          (dtypes[0] == ScalarType::Long && dtypes[1] == ScalarType::Bool));
      if (enable_flag) {
        get_elementwise_info<narity + 1>(ndim, narity, offset_calc, dtypes, "p_e_cp_unroll_1_1_cast", f);
        launch_unrolled_copy_cast_kernel<arg0_t, typename traits::template arg<0>::type>(
            numel, f,
            dtypes[0], dtypes[1],
            data[0], data[1]);
      } else {
        get_elementwise_info<narity + 1>(ndim, narity, offset_calc, dtypes, "p_e_cp_unroll_1_1", f);
        launch_unrolled_kernel(numel, f, data, input_offset_calculator, output_offset_calculator, loader, storer);
      }
    } else {
      if (ndim == 1) {
        get_elementwise_info<narity + 1>(ndim, narity, offset_calc, dtypes, "p_e_cp_launch_legacy_kernel_maca_1_1_cast", f);
        launch_legacy_kernel_maca_1_1_cast<128, 4, arg0_t, typename traits::template arg<0>::type>(
          numel,
          data[0], data[1],  // data
          dtypes[0], dtypes[1], 
          offset_calc.sizes_[0].divisor,
          offset_calc.strides_[0][0], offset_calc.strides_[0][1], 
          f);
      } else if (ndim == 2) {
        bool disable_elementwise_kernel_2_1_cp_cast_dim0_contiguous = at::maca::get_maca_disable_elementwise_kernel_2_1_cp_cast_dim0_contiguous();
        if (!disable_elementwise_kernel_2_1_cp_cast_dim0_contiguous && numel > 0 && offset_calc.sizes_[0].divisor % 4 == 0 &&
             offset_calc.sizes_[0].divisor >= 64 &&
            ((dtypes[0] == ScalarType::Float && dtypes[1] == ScalarType::Half) ||
             (dtypes[0] == ScalarType::Float && dtypes[1] == ScalarType::BFloat16)) &&
            (uint64_t)data[0] % 4 == 0 && (uint64_t)data[1] % 4 == 0 &&
            offset_calc.strides_[0][0] == 4  && offset_calc.strides_[0][1] == 2 &&
            offset_calc.strides_[1][0] % 4 == 0 && offset_calc.strides_[1][1] % 4 == 0
            ) {
          get_elementwise_info<narity + 1>(ndim, narity, offset_calc, dtypes, "p_e_cp_launch_legacy_kernel_maca_2_1_cast_dim0_contiguous", f);
          launch_legacy_kernel_maca_2_1_cast_dim0_contiguous<128, 4, arg0_t, typename traits::template arg<0>::type>(
            numel,
            data[0], data[1],  // data
            dtypes[0], dtypes[1],
            offset_calc.sizes_[0].divisor, offset_calc.sizes_[1].divisor,
            offset_calc.strides_[0][0], offset_calc.strides_[0][1],
            offset_calc.strides_[1][0], offset_calc.strides_[1][1],
            f);
        } else {
          get_elementwise_info<narity + 1>(ndim, narity, offset_calc, dtypes,  "p_e_cp_launch_legacy_kernel_maca_2_1_cast", f);
          launch_legacy_kernel_maca_2_1_cast<128, 4, arg0_t, typename traits::template arg<0>::type>(
            numel,
            data[0], data[1],  // data
            dtypes[0], dtypes[1],
            offset_calc.sizes_[0].divisor, offset_calc.sizes_[1].divisor,
            offset_calc.strides_[0][0], offset_calc.strides_[0][1],
            offset_calc.strides_[1][0], offset_calc.strides_[1][1],
            f);
        }
      } else {
        get_elementwise_info<narity + 1>(ndim, narity, offset_calc, dtypes,  "p_e_cp_launch_legacy_kernel", f);
        launch_legacy_kernel<128, 4>(numel, [=]GPU_LAMBDA(int idx) {
          auto offsets = offset_calc.get(idx);
          void* out = data[0] + offsets[0];
          arg0_t result = invoke(f, &data.data[1], &offsets.data[1], &dtypes.data[1], 1);
          c10::cast_and_store<arg0_t>(dtypes[0], out, result);
        });
      }
    }
  }
}

#include <ATen/native/cuda/maca_kernels/CUDALoops_maca_arity2.cuh>

#include <ATen/native/cuda/maca_kernels/CUDALoops_maca_arity3.cuh>

}} // namespace at::native
