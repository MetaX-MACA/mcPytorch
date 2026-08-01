#pragma once

#include <iostream>
#include <typeinfo>
#include <ATen/native/cuda/maca_kernels/LoopsUtils.h>


template<int nt, int vt, typename return_t, typename func_t, typename array_t, int narity,
typename std::enable_if<narity!=0, int>::type = 0>
C10_LAUNCH_BOUNDS_1(nt)
__global__ void vectorized_elementwise_kernel_nullary_opt(int N, func_t f, char* data0) {
  assert(0);
}

template<int nt, int vt, typename return_t, typename func_t, typename array_t, int narity,
typename std::enable_if<narity==0, int>::type = 0>
C10_LAUNCH_BOUNDS_1(nt)
__global__ void vectorized_elementwise_kernel_nullary_opt(int N, func_t f, char* data0) {
  using traits = function_traits<func_t>;
  using StoreT = memory::aligned_vector<return_t, vt>;

  return_t results[vt];

  // char* data0 = reinterpret_cast<char*>(data[0]);
  int st = sizeof(return_t);

  int tid = threadIdx.x;
  int nv = nt * vt;
  int64_t idx = nv * blockIdx.x + tid * vt;

  if (idx < N) {
    #pragma unroll
    for (int i = 0; i < vt; i++){
      results[i] = f();
    }

    StoreT* p_results = reinterpret_cast<StoreT*>(&results);
    StoreT* out = reinterpret_cast<StoreT*>(data0 + idx * st);
    *out = *p_results;
  }
}


template<typename func_t, typename array_t>
static inline void launch_vectorized_kernel_arity0(int64_t N, const func_t& f, array_t data) {
  TORCH_INTERNAL_ASSERT(N > 0 && N <= std::numeric_limits<int32_t>::max());
  using traits = function_traits<func_t>;
  using res_t = typename traits::result_type;
  constexpr int narity = traits::arity;
  bool disable_nullary_opt = at::maca::get_maca_disable_vectorized_elementwise_nullary_opt();

  // when arity == 0 and type is half or float
  if (narity == 0 && (sizeof(res_t) == 2 || sizeof(res_t) == 4) && maca_likely(!disable_nullary_opt) && N>=1024*64) {
    // fixed block size here
    constexpr int block_size = 1024;
    int vec_size = sizeof(res_t) > 2 ? 4 : 8;
    while (N % vec_size != 0){
      vec_size /= 2;
    }
    auto ip = reinterpret_cast<uintptr_t>(data[0]);
    while (vec_size > 1 && ip % (sizeof(res_t) * vec_size) != 0) {
      vec_size /= 2;
    }
    int64_t grid = (N + block_size * vec_size - 1) / (block_size * vec_size);
    auto stream = at::cuda::getCurrentCUDAStream();
    switch (vec_size) {
    case 8:
      vectorized_elementwise_kernel_nullary_opt<block_size, 8, res_t, func_t, array_t, narity><<<grid, block_size, 0, stream>>>(N, f, data[0]);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      break;
    case 4:
      vectorized_elementwise_kernel_nullary_opt<block_size, 4, res_t, func_t, array_t, narity><<<grid, block_size, 0, stream>>>(N, f, data[0]);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      break;
    case 2:
      vectorized_elementwise_kernel_nullary_opt<block_size, 2, res_t, func_t, array_t, narity><<<grid, block_size, 0, stream>>>(N, f, data[0]);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      break;
    case 1: {
      vectorized_elementwise_kernel_nullary_opt<block_size, 1, res_t, func_t, array_t, narity><<<grid, block_size, 0, stream>>>(N, f, data[0]);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      break;
    }
    default:
      TORCH_INTERNAL_ASSERT(false, "Unexpected vectorization size");
    }
  } else {
    launch_vectorized_kernel(N, f, data);
  }
}


template <typename func_t>
void gpu_kernel_impl_maca_arity0(TensorIteratorBase& iter, const func_t& f) {
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
  constexpr int unroll_factor = sizeof(arg0_t) >= 4 ? 2 : 4;
  constexpr int narity = traits::arity;
  int ndim = iter.ndim();

  bool contiguous = iter.is_contiguous();
  bool dynamic_casting = needs_dynamic_casting<func_t>::check(iter);
  auto offset_calc = ::make_offset_calculator<traits::arity + 1>(iter);

  if (!dynamic_casting) {
    if (contiguous) {
      get_elementwise_info<narity + 1>(ndim, narity, offset_calc, dtypes, "p_e_noopt_vec_1_2", f);
      launch_vectorized_kernel_arity0(numel, f, data);
    } else {
        get_elementwise_info<narity + 1>(ndim, narity, offset_calc, dtypes, "p_e_noopt_launch_legacy_kernel", f);
        launch_legacy_kernel<128,unroll_factor>(numel, [=]GPU_LAMBDA(int idx) {
        auto offsets = offset_calc.get(idx);
        arg0_t* out = (arg0_t*)(data[0] + offsets[0]);
        *out = invoke(f, &data.data[1], &offsets.data()[1], 1);
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
      get_elementwise_info<narity + 1>(ndim, narity, offset_calc, dtypes, "p_e_noopt_launch_legacy_kernel", f);
      launch_legacy_kernel<128, 4>(numel, [=]GPU_LAMBDA(int idx) {
        auto offsets = offset_calc.get(idx);
        void* out = data[0] + offsets[0];
        arg0_t result = invoke(f, &data.data[1], &offsets.data()[1], &dtypes.data[1], 1);
        c10::cast_and_store<arg0_t>(dtypes[0], out, result);
      });
    }
  }
}






