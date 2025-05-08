#pragma once

#include <iostream>
#include <typeinfo>
#include "loop_utils.h"

#define kTileDimMacaT 64
#define kBlockRowsMacaT 8

inline bool is_pow2(int a) {
  return (a & (a - 1)) == 0;
}

inline int get_log2(int n) {
  int i = -1;
  while (n) {
    n = n >> 1;
    ++i;
  }
  return i;
}


template<typename func_t, typename array_t>
static inline void launch_vectorized_kernel_arity1(int64_t N, const func_t& f, array_t data) {
  TORCH_INTERNAL_ASSERT(N > 0 && N <= std::numeric_limits<int32_t>::max());
  using traits = function_traits<func_t>;
  constexpr int narity = traits::arity;
  static_assert(narity==1);
  using res_t = typename traits::result_type;
  using arg0 = typename traits::template arg<0>::type;

  bool disable_onenary_opt = at::maca::get_maca_disable_vectorized_elementwise_unary_opt();

  if (sizeof(res_t) == sizeof(arg0) && (sizeof(res_t) == 2 || sizeof(res_t) == 4) && 
      (N == 64 * 1024 * 1024 || N == 1024* 1024 * 1024) && maca_likely(!disable_onenary_opt)) {
    constexpr int block_size = 512;
    int vec_size = sizeof(res_t) > 2 ? 4 : 8;
    while (N % vec_size != 0){
      vec_size /= 2;
    }
    auto ip0 = reinterpret_cast<uintptr_t>(data[0]);
    auto ip1 = reinterpret_cast<uintptr_t>(data[0]);

    while (vec_size > 1 && ((ip0 % (sizeof(res_t) * vec_size) != 0) || (ip1 % (sizeof(res_t) * vec_size) != 0))) {
      vec_size /= 2;
    }
    int64_t grid = (N + block_size * vec_size - 1) / (block_size * vec_size);
    auto stream = at::cuda::getCurrentCUDAStream();
    switch (vec_size) {
      case 8:
        vectorized_elementwise_kernel_unary_opt<block_size, 8, res_t, arg0, func_t, array_t, narity><<<grid, block_size, 0, stream>>>(N, f, data[0], data[1]);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        break;
      case 4:
        vectorized_elementwise_kernel_unary_opt<block_size, 4, res_t, arg0, func_t, array_t, narity><<<grid, block_size, 0, stream>>>(N, f, data[0], data[1]);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        break;
      case 2:
        vectorized_elementwise_kernel_unary_opt<block_size, 2, res_t, arg0, func_t, array_t, narity><<<grid, block_size, 0, stream>>>(N, f, data[0], data[1]);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        break;     
      case 1: 
        vectorized_elementwise_kernel_unary_opt<block_size, 1, res_t, arg0, func_t, array_t, narity><<<grid, block_size, 0, stream>>>(N, f, data[0], data[1]);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        break;
      default:
        TORCH_INTERNAL_ASSERT(false, "Unexpected vectorization size");
    }
  } else {
    launch_vectorized_kernel(N, f, data);
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
      auto divmod_div = linear_idx / size0;
      auto divmod_mod = linear_idx % size0;
      linear_idx = divmod_div;
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
  int64_t tid = threadIdx.x;
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
__global__ void elementwise_kernel_2_1_input_lowdim_contiuous(
    int64_t N,
    char* data0, char* data1,
    index_t size0, index_t size1,
    stride_t stride00, stride_t stride01,
    stride_t stride10, stride_t stride11,
    func_t f) {
  // ndim = 2, arity = 1, narg = 2
  using load_vec = at::native::memory::aligned_vector<arg0_t, vt>;
  using save_vec = at::native::memory::aligned_vector<res_t, vt>;
  int64_t out_idx = (blockIdx.x * nt + threadIdx.x) * vt;
  if (out_idx >= N) return;

  int64_t offsets = 0;
  auto linear_idx = out_idx;
  auto divmod_div = linear_idx / size0;
  auto divmod_mod = linear_idx % size0;
  linear_idx = divmod_div;
  offsets += divmod_mod * stride01;

  divmod_div = linear_idx / size1;
  divmod_mod = linear_idx % size1;
  linear_idx = divmod_div;
  offsets += divmod_mod * stride11;

  load_vec input_vec;
  save_vec out_vec;
  input_vec = *(reinterpret_cast<load_vec*>(data1 + offsets));

  #pragma unroll
  for (int i = 0; i < vt; i++) {
    out_vec.val[i] = f(input_vec.val[i]);
  }

  *(reinterpret_cast<save_vec*>(data0 + out_idx * stride00)) = out_vec;
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
      constexpr int MAX_DIMS = 3;
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

      res_t* out = (res_t*)(data0 + offsets[0]);
      *out = f(func_reinterpret_cast<arg0_t>(data1 + offsets[1]));

      idx += nt;
    }
  }
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

template<int nt, int vt, typename res_t , typename arg0_t, typename func_t, typename index_t, typename stride_t, int nn>
C10_LAUNCH_BOUNDS_2(nt, 4)
__global__ void elementwise_kernel_2_1_opt2(
    int64_t N,
    char* data0, char* data1,
    index_t size0, index_t size1_log2,
    stride_t stride00, stride_t stride01, 
    stride_t stride10, stride_t stride11, 
    func_t f) {
  // ndim = 2, arity = 1, narg = 2
  int tid = threadIdx.x;
  int nv = nt * vt;
  int64_t idx = nv * blockIdx.x + 8 * tid;

  auto linear_idx = idx;
  auto l00 = linear_idx % size0;
  linear_idx = linear_idx / nn;
  auto tmp = linear_idx >> size1_log2;
  auto l01 = linear_idx - (tmp << size1_log2);

  float4 tmpin = *reinterpret_cast<float4*>(data1 + l00 * stride01 + l01 * stride11);
  res_t o1 = f(*(reinterpret_cast<arg0_t*>(&tmpin)+0));
  res_t o2 = f(*(reinterpret_cast<arg0_t*>(&tmpin)+1));
  res_t o3 = f(*(reinterpret_cast<arg0_t*>(&tmpin)+2));
  res_t o4 = f(*(reinterpret_cast<arg0_t*>(&tmpin)+3));
  res_t o5 = f(*(reinterpret_cast<arg0_t*>(&tmpin)+4));
  res_t o6 = f(*(reinterpret_cast<arg0_t*>(&tmpin)+5));
  res_t o7 = f(*(reinterpret_cast<arg0_t*>(&tmpin)+6));
  res_t o8 = f(*(reinterpret_cast<arg0_t*>(&tmpin)+7));

  float4 tmp2;
  *(reinterpret_cast<res_t*>(&tmp2)+0) = o1;
  *(reinterpret_cast<res_t*>(&tmp2)+1) = o2;
  *(reinterpret_cast<res_t*>(&tmp2)+2) = o3;
  *(reinterpret_cast<res_t*>(&tmp2)+3) = o4;
  *(reinterpret_cast<res_t*>(&tmp2)+4) = o5;
  *(reinterpret_cast<res_t*>(&tmp2)+5) = o6;
  *(reinterpret_cast<res_t*>(&tmp2)+6) = o7;
  *(reinterpret_cast<res_t*>(&tmp2)+7) = o8;

  *(float4*)(data0 + l00 * stride00 + l01 * stride10) = tmp2;
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
static void launch_legacy_kernel_maca_2_1_merge_half(
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

  if (size0 == 1280) {
    elementwise_kernel_2_1_opt2<nt, vt, res_t, arg0_t, func_t, index_t, stride_t, 1280><<<grid, block, 0, stream>>>(
        N, data0, data1, size0, get_log2(size1), stride00, stride01, stride10, stride11, f);
  } else if (size0 == 2560) {
    elementwise_kernel_2_1_opt2<nt, vt, res_t, arg0_t, func_t, index_t, stride_t, 2560><<<grid, block, 0, stream>>>(
        N, data0, data1, size0, get_log2(size1), stride00, stride01, stride10, stride11, f);
  } else if (size0 == 5120) {
    elementwise_kernel_2_1_opt2<nt, vt, res_t, arg0_t, func_t, index_t, stride_t, 5120><<<grid, block, 0, stream>>>(
        N, data0, data1, size0, get_log2(size1), stride00, stride01, stride10, stride11, f);
  } else {
    assert(0);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}


template<int nt, int vt, typename res_t , typename arg0_t, typename func_t, typename index_t, typename stride_t>
static void launch_legacy_kernel_maca_2_1_input_lowdim_contiuous(
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

  elementwise_kernel_2_1_input_lowdim_contiuous<nt, vt, res_t, arg0_t, func_t, index_t, stride_t>
  <<<grid, block, 0, stream>>>(
      N, data0, data1,
      size0, size1,
      stride00, stride01,
      stride10, stride11,
  f);
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
  elementwise_kernel_3_1<nt, vt, res_t, arg0_t, func_t, index_t, stride_t><<<grid, block, 0, stream>>>(
      N, data0, data1, size0, size1, size2, stride00, stride01, stride10, stride11, stride20, stride21, f);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
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

template<typename res_t, typename arg0_t, typename TIndex, typename func_t>
__global__ void elementwise_kernel_transpose_half(
    char* out, char* in,
    TIndex H, TIndex W, TIndex dh, TIndex dw,
    func_t f) {
   // ndim = 3, arity = 1, narg = 2
   typedef at::Half TData;
   TData* Y = reinterpret_cast<TData*>(out);
   arg0_t* X = reinterpret_cast<arg0_t*>(in);
 
   __shared__ TData tile[kTileDimMacaT][kTileDimMacaT + 2];
   const TIndex n = blockIdx.x / (dh * dw);
   const TIndex k = blockIdx.x % (dh * dw);
   const TIndex r = k / dw;
   const TIndex c = k % dw;
   const int64_t offset = n * H * W;
   int x = c * kTileDimMacaT + threadIdx.x;
   int y = r * kTileDimMacaT + threadIdx.y;
   if (x < W) {
     for (int i = 0; threadIdx.y + i < kTileDimMacaT && y + i < H; i += kBlockRowsMacaT) {
        res_t tmp = f(X[offset + (y + i) * W + x]);
        tile[threadIdx.y + i][threadIdx.x] = *reinterpret_cast<TData*>(&tmp);
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

template<typename res_t, typename arg0_t, typename func_t, typename index_t>
static void launch_legacy_kernel_maca_3_1_transpose(
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
  elementwise_kernel_transpose_half<res_t, arg0_t, uint32_t, func_t><<<batch * dh * dw, dim3(kTileDimMacaT, kBlockRowsMacaT), 0, stream>>>(
      out, in, (uint32_t)H, (uint32_t)W, (uint32_t)dh, (uint32_t)dw, f);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename func_t>
void gpu_kernel_impl_maca_arity1(TensorIteratorBase& iter, const func_t& f) {
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

  bool contiguous = iter.is_contiguous();
  bool dynamic_casting = needs_dynamic_casting<func_t>::check(iter);
  auto offset_calc = ::make_offset_calculator<traits::arity + 1>(iter);

  constexpr int unroll_factor = sizeof(arg0_t) >= 4 ? 2 : 4;
  constexpr int narity = traits::arity;
  int ndim = iter.ndim();
  assert(narity == 1);

  if (!dynamic_casting) {
    if (contiguous) {
      get_elementwise_info<narity + 1>(ndim, narity, offset_calc, dtypes, "p_e_vec_1_1", f);
      launch_vectorized_kernel_arity1(numel, f, data);
    } else {
        if (ndim == 1) {
          get_elementwise_info<narity + 1>(ndim, narity, offset_calc, dtypes, "p_e_launch_legacy_kernel_maca_1_1", f);
          launch_legacy_kernel_maca_1_1<128, unroll_factor, arg0_t, typename traits::template arg<0>::type>(
            numel,
            data[0], data[1],  // data
            offset_calc.sizes_[0].divisor,
            offset_calc.strides_[0][0], offset_calc.strides_[0][1], 
            f);
        } else if (ndim == 2) {
          bool disable_2_1_input_lowdim_contiuous = at::maca::get_maca_disable_elementwise_kernel_2_1_input_lowdim_contiuous();
          if (dtypes[0] == ScalarType::Half && dtypes[1] == ScalarType::Half &&
              offset_calc.strides_[0][0] == 2 && offset_calc.strides_[0][1] == 2 &&
              (offset_calc.sizes_[0].divisor == 1280 || offset_calc.sizes_[0].divisor == 2560 || offset_calc.sizes_[0].divisor == 5120) &&
              offset_calc.sizes_[0].divisor * 2 == offset_calc.strides_[1][0] &&
              offset_calc.sizes_[0].divisor * 4 == offset_calc.strides_[1][1] &&
              reinterpret_cast<uintptr_t>(data[0]) % 4 == 0 &&
              reinterpret_cast<uintptr_t>(data[1]) % 4 == 0 &&
              is_pow2(offset_calc.sizes_[1].divisor)) {
            get_elementwise_info<narity + 1>(ndim, narity, offset_calc, dtypes, "p_e_launch_legacy_kernel_maca_2_1_merge_half", f);
            launch_legacy_kernel_maca_2_1_merge_half<128, 8, arg0_t, typename traits::template arg<0>::type>(
              numel,
              data[0], data[1],  // data
              offset_calc.sizes_[0].divisor, offset_calc.sizes_[1].divisor, 
              offset_calc.strides_[0][0], offset_calc.strides_[0][1], 
              offset_calc.strides_[1][0], offset_calc.strides_[1][1], 
              f);
          } else if((dtypes[0] == ScalarType::Half || dtypes[0] == ScalarType::BFloat16) && (dtypes[0] == dtypes[1]) &&
                    offset_calc.strides_[0][0] == 2 && offset_calc.strides_[0][1] == 2 &&
                    offset_calc.strides_[1][0] == 2 * offset_calc.sizes_[0].divisor &&
                    offset_calc.strides_[1][1] % 4 == 0 && offset_calc.sizes_[0].divisor % 8 == 0 &&
                    reinterpret_cast<uintptr_t>(data[0]) % 4 == 0 &&
                    reinterpret_cast<uintptr_t>(data[1]) % 4 == 0 &&
                    maca_likely(!disable_2_1_input_lowdim_contiuous)
          ) {
            get_elementwise_info<narity + 1>(ndim, narity, offset_calc, dtypes, "p_e_launch_legacy_kernel_maca_2_1_input_lowdim_contiuous", f);
            launch_legacy_kernel_maca_2_1_input_lowdim_contiuous<128, 8, arg0_t, typename traits::template arg<0>::type>(
              numel,
              data[0], data[1],  // data
              offset_calc.sizes_[0].divisor, offset_calc.sizes_[1].divisor,
              offset_calc.strides_[0][0], offset_calc.strides_[0][1],
              offset_calc.strides_[1][0], offset_calc.strides_[1][1],
              f);
          } else {
            get_elementwise_info<narity + 1>(ndim, narity, offset_calc, dtypes, "p_e_launch_legacy_kernel_maca_2_1", f);
            launch_legacy_kernel_maca_2_1<128, unroll_factor, arg0_t, typename traits::template arg<0>::type>(
              numel,
              data[0], data[1],  // data
              offset_calc.sizes_[0].divisor, offset_calc.sizes_[1].divisor, 
              offset_calc.strides_[0][0], offset_calc.strides_[0][1], 
              offset_calc.strides_[1][0], offset_calc.strides_[1][1], 
              f);
          }
        } else if (ndim == 3) {
          if (dtypes[0] == ScalarType::Half && dtypes[1] == ScalarType::Half && 
              // [4096, 320, 2],
              // (2,  8192， 2621440)
              // (640,   2， 2621440)
              offset_calc.strides_[0][0] == 2 && offset_calc.strides_[1][0] == 2 * offset_calc.sizes_[0].divisor &&
              offset_calc.strides_[2][0] == 2 * offset_calc.sizes_[0].divisor * offset_calc.sizes_[1].divisor &&
              offset_calc.strides_[0][1] == 2 * offset_calc.sizes_[1].divisor && offset_calc.strides_[1][1] == 2 &&
              offset_calc.strides_[2][1] == offset_calc.strides_[2][0]) {
            get_elementwise_info<narity + 1>(ndim, narity, offset_calc, dtypes, "p_e_launch_legacy_kernel_maca_3_1_transpose", f);
            launch_legacy_kernel_maca_3_1_transpose<arg0_t, typename traits::template arg<0>::type>(
              numel,
              data[0], data[1],  // data
              offset_calc.sizes_[2].divisor, offset_calc.sizes_[0].divisor, offset_calc.sizes_[1].divisor,
              f);
          } else {
            get_elementwise_info<narity + 1>(ndim, narity, offset_calc, dtypes, "p_e_launch_legacy_kernel_maca_3_1", f);
            launch_legacy_kernel_maca_3_1<128, unroll_factor, arg0_t, typename traits::template arg<0>::type>(
              numel,
              data[0], data[1],  // data
              offset_calc.sizes_[0].divisor, offset_calc.sizes_[1].divisor, offset_calc.sizes_[2].divisor,
              offset_calc.strides_[0][0], offset_calc.strides_[0][1], 
              offset_calc.strides_[1][0], offset_calc.strides_[1][1], 
              offset_calc.strides_[2][0], offset_calc.strides_[2][1], 
              f);
          }
        } else if (ndim == 4) {
          get_elementwise_info<narity + 1>(ndim, narity, offset_calc, dtypes, "p_e_launch_legacy_kernel_maca_4_1", f);
          launch_legacy_kernel_maca_4_1<128, unroll_factor, arg0_t, typename traits::template arg<0>::type>(
            numel,
            data[0], data[1],  // data
            offset_calc.sizes_[0].divisor, offset_calc.sizes_[1].divisor, offset_calc.sizes_[2].divisor, offset_calc.sizes_[3].divisor,
            offset_calc.strides_[0][0], offset_calc.strides_[0][1], 
            offset_calc.strides_[1][0], offset_calc.strides_[1][1], 
            offset_calc.strides_[2][0], offset_calc.strides_[2][1], 
            offset_calc.strides_[3][0], offset_calc.strides_[3][1], 
            f);
        } else {
          get_elementwise_info<narity + 1>(ndim, narity, offset_calc, dtypes, "p_e_launch_legacy_kernel", f);
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
      get_elementwise_info<narity + 1>(ndim, narity, offset_calc, dtypes, "p_e_unroll_1_1", f);
      launch_unrolled_kernel(numel, f, data, input_offset_calculator, output_offset_calculator, loader, storer);
    } else {
      if (ndim == 1) {
        get_elementwise_info<narity + 1>(ndim, narity, offset_calc, dtypes, "p_e_launch_legacy_kernel_maca_1_1_cast", f);
        launch_legacy_kernel_maca_1_1_cast<128, 4, arg0_t, typename traits::template arg<0>::type>(
          numel,
          data[0], data[1],  // data
          dtypes[0], dtypes[1], 
          offset_calc.sizes_[0].divisor,
          offset_calc.strides_[0][0], offset_calc.strides_[0][1], 
          f);
      } else {
        get_elementwise_info<narity + 1>(ndim, narity, offset_calc, dtypes, "p_e_launch_legacy_kernel", f);
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

