#pragma once

#include <iostream>
#include <typeinfo>
#include "loop_utils.h"

#define kTileDimMacaT 64
#define kTileDimMacaT_32 32
#define kBlockRowsMacaT 8

#define SWITCH_TEMPLATE_KERNEL_ARGS_3(nt, vt, vt_0, vt_1, vec, f, ...)  \
  if (vt == 8 && vt_0 == 8 && vt_1 == 8) {                           \
    f(nt, 8, 8, 8, 8, __VA_ARGS__);                                  \
  } else if (vt == 4 && vt_0 == 4 && vt_1 == 4) {                    \
    f(nt, 4, 4, 4, 4, __VA_ARGS__);                                  \
  } else if (vt == 2 && vt_0 == 2 && vt_1 == 2) {                    \
    f(nt, 2, 2, 2, 2, __VA_ARGS__);                                  \
  } else if (vt == 8 && vt_0 == 1 && vt_1 == 8) {                    \
    f(nt, 8, 1, 8, 8, __VA_ARGS__);                                  \
  } else if (vt == 8 && vt_0 == 8 && vt_1 == 1) {                    \
    f(nt, 8, 8, 1, 8, __VA_ARGS__);                                  \
  } else if (vt == 8 && vt_0 == 1 && vt_1 == 1) {                    \
    f(nt, 8, 1, 1, 8, __VA_ARGS__);                                  \
  } else if (vt == 4 && vt_0 == 1 && vt_1 == 4) {                    \
    f(nt, 4, 1, 4, 4, __VA_ARGS__);                                  \
  } else if (vt == 4 && vt_0 == 4 && vt_1 == 1) {                    \
    f(nt, 4, 4, 1, 4, __VA_ARGS__);                                  \
  } else if (vt == 4 && vt_0 == 1 && vt_1 == 1) {                    \
    f(nt, 4, 1, 1, 4, __VA_ARGS__);                                  \
  } else if (vt == 2 && vt_0 == 1 && vt_1 == 2) {                    \
    f(nt, 2, 1, 2, 2, __VA_ARGS__);                                  \
  } else if (vt == 2 && vt_0 == 2 && vt_1 == 1) {                    \
    f(nt, 2, 2, 1, 2, __VA_ARGS__);                                  \
  } else if (vt == 2 && vt_0 == 1 && vt_1 == 1) {                    \
    f(nt, 2, 1, 1, 2, __VA_ARGS__);                                  \
  } else if (vt == 1 && vt_0 == 8 && vt_1 == 8) {                    \
    f(nt, 1, 8, 8, 8, __VA_ARGS__);                                  \
  } else if (vt == 1 && vt_0 == 1 && vt_1 == 8) {                    \
    f(nt, 1, 1, 8, 8, __VA_ARGS__);                                  \
  } else if (vt == 1 && vt_0 == 8 && vt_1 == 1) {                    \
    f(nt, 1, 8, 1, 8, __VA_ARGS__);                                  \
  } else if (vt == 1 && vt_0 == 4 && vt_1 == 4) {                    \
    f(nt, 1, 4, 4, 4, __VA_ARGS__);                                  \
  } else if (vt == 1 && vt_0 == 1 && vt_1 == 4) {                    \
    f(nt, 1, 1, 4, 4, __VA_ARGS__);                                  \
  } else if (vt == 1 && vt_0 == 4 && vt_1 == 1) {                    \
    f(nt, 1, 4, 1, 4, __VA_ARGS__);                                  \
  } else if (vt == 1 && vt_0 == 2 && vt_1 == 2) {                    \
    f(nt, 1, 2, 2, 2, __VA_ARGS__);                                  \
  } else if (vt == 1 && vt_0 == 1 && vt_1 == 2) {                    \
    f(nt, 1, 1, 2, 2, __VA_ARGS__);                                  \
  } else if (vt == 1 && vt_0 == 2 && vt_1 == 1) {                    \
    f(nt, 1, 2, 1, 2, __VA_ARGS__);                                  \
  } else {                                                           \
    f(nt, 1, 1, 1, 1, __VA_ARGS__);                                  \
  }

template<int nt, int vt, typename res_t, typename arg0_t, typename arg1_t, typename func_t, typename index_t, typename stride_t>
C10_LAUNCH_BOUNDS_2(nt, 4)
__global__ void elementwise_kernel_1_2(
    int64_t N,
    char* data0, char* data1, char* data2,
    index_t size0,
    stride_t stride00, stride_t stride01,  stride_t stride02,
    func_t f) {
  // ndim = 1, arity = 2, narg = 3
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
      int64_t offsets[3];
      auto linear_idx = idx;
      constexpr int NARGS = 3;
      constexpr int MAX_DIMS = 1;
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
      arg = 2;
      offsets[arg] += divmod_mod * stride02;

      res_t* out = (res_t*)(data0 + offsets[0]);
      *out = f(func_reinterpret_cast<arg0_t>(data1 + offsets[1]),
               func_reinterpret_cast<arg1_t>(data2 + offsets[2]));

      idx += nt;
    }
  }
}

template<int nt, int vt, typename res_t, typename arg0_t, typename arg1_t, typename func_t, typename index_t, typename stride_t, bool swap_tensor=false>
C10_LAUNCH_BOUNDS_2(nt, 4)
__global__ void elementwise_kernel_1_2_broadcast(
    int64_t N,
    char* data0, char* data1, char* data2,
    index_t size0,
    stride_t stride00, stride_t stride01,  stride_t stride02,
    func_t f) {
  // ndim = 1, arity = 2
  int tid = threadIdx.x;
  int64_t offsets[3];
  constexpr int NARGS = 3;
  constexpr int MAX_DIMS = 2;
  size_t x_remain = size0 % (blockDim.x * vt);
  using LoadT = at::native::memory::aligned_vector<arg0_t, vt>;
  using StoreT = at::native::memory::aligned_vector<res_t, vt>;
  #pragma unroll
  for (int arg = 0; arg < NARGS; arg++) {
    offsets[arg] = 0;
  }
  // offsets[2] = 0
  arg0_t ld_1[vt];
  arg1_t ld_2;
  res_t ld_out[vt];
  LoadT* p_ld_1 = reinterpret_cast<LoadT*>(&ld_1);
  StoreT * p_ld_out = reinterpret_cast<StoreT*>(&ld_out);

  offsets[0] = (blockIdx.x * blockDim.x + tid) * vt * stride00;
  offsets[1] = (blockIdx.x * blockDim.x + tid) * vt * stride01;
  ld_2 = *(arg1_t*)data2;
  *p_ld_1 = *reinterpret_cast<LoadT*>(data1 + offsets[1]);
  StoreT* out = reinterpret_cast<StoreT*>(data0 + offsets[0]);
  #pragma unroll
  for (int i=0; i < vt; i++) {
    if (swap_tensor) {
      ld_out[i] = f(ld_2, ld_1[i]);
    } else {
      ld_out[i] = f(ld_1[i], ld_2);
    }
  }
  *out = *p_ld_out;

  if (x_remain && blockIdx.x == 0 && (tid + 1) * vt <= x_remain) {
    offsets[0] = (gridDim.x * blockDim.x + tid) * vt * stride00;
    offsets[1] = (gridDim.x * blockDim.x + tid) * vt * stride01;
    ld_2 = *(arg1_t*)data2;
    *p_ld_1 = *reinterpret_cast<LoadT*>(data1 + offsets[1]);
    StoreT* out = reinterpret_cast<StoreT*>(data0 + offsets[0]);
    #pragma unroll
    for (int i=0; i < vt; i++) {
      if (swap_tensor) {
        ld_out[i] = f(ld_2, ld_1[i]);
      } else {
        ld_out[i] = f(ld_1[i], ld_2);
      }
    }
    *out = *p_ld_out;
  }
}


template<int nt, int vt, typename res_t , typename arg0_t, typename arg1_t, typename func_t, typename index_t, typename stride_t>
C10_LAUNCH_BOUNDS_2(nt, 4)
__global__ void elementwise_kernel_2_2_broadcast_arity1_dim0(
    int64_t N,
    char* data0, char* data1, char* data2,
    index_t size0, index_t size1,
    stride_t stride00, stride_t stride01, stride_t stride02,
    stride_t stride10, stride_t stride11, stride_t stride12,
    func_t f) {

  int64_t linear_idx = (blockIdx.x * blockDim.x + threadIdx.x) * vt;
  int64_t res_idx = linear_idx * stride00;
  if (linear_idx >= N) return;

  using vec_res = at::native::memory::aligned_vector<res_t, vt>;
  using vec_arg0 = at::native::memory::aligned_vector<arg0_t, vt>;
  int64_t offset1 = 0;
  int64_t offset2 = 0;

  auto divmod_div = linear_idx / size0;
  auto divmod_mod = linear_idx % size0;
  linear_idx = divmod_div;
  offset1 += divmod_mod * stride01;
  offset2 += divmod_mod * stride02;

  divmod_div = linear_idx / size1;
  divmod_mod = linear_idx % size1;
  linear_idx = divmod_div;
  offset1 += divmod_mod * stride11;
  offset2 += divmod_mod * stride12;

  vec_arg0 arg1_elems = *(reinterpret_cast<vec_arg0*>(data1 + offset1));
  arg1_t arg2 = *(reinterpret_cast<arg1_t*>(data2 + offset2));
  vec_res  res_elems;
  #pragma unroll
  for (int ii = 0; ii < vt; ii++) {
    res_elems.val[ii] = f(arg1_elems.val[ii],arg2);
  }

  *(reinterpret_cast<vec_res*>(data0 + res_idx)) = res_elems;
}

template<int nt, int v_x, int KTile, typename res_t, typename arg0_t, typename arg1_t, typename func_t, typename TIndex, typename stride_t,
typename std::enable_if<!std::is_same<arg1_t, at::Half>::value && !std::is_same<arg1_t, at::BFloat16>::value
                       && !std::is_same<arg1_t, float>::value, int>::type = 0>
__global__ void elementwise_kernel_2_2_arity2_transpose(
  char* out, char* in0, char* in1,
  TIndex H, TIndex W, TIndex dh, TIndex dw,
  func_t f){
  assert(0);
}

template<int nt, int v_x, int KTile, typename res_t, typename arg0_t, typename arg1_t, typename func_t, typename TIndex, typename stride_t,
typename std::enable_if<std::is_same<arg1_t, at::Half>::value || std::is_same<arg1_t, at::BFloat16>::value
                       || std::is_same<arg1_t, float>::value, int>::type = 0>
__global__ void elementwise_kernel_2_2_arity2_transpose(
  char* out, char* in0, char* in1,
  TIndex H, TIndex W, TIndex dh, TIndex dw,
  func_t f) {
  const TIndex st = sizeof(arg1_t);
  using StoreT = at::native::memory::aligned_vector<res_t, v_x>;
  using Load0T = at::native::memory::aligned_vector<arg0_t, v_x>;
  using Load1T = at::native::memory::aligned_vector<arg1_t, v_x>;

  __shared__ arg1_t tile[KTile][KTile + 2];
  const TIndex r = blockIdx.y;
  const TIndex c = blockIdx.x;
  int64_t x = c * KTile + threadIdx.x * v_x;
  int64_t y = r * KTile + threadIdx.y;

  if (x < W && y < H) {
    Load1T tmp1 = *(reinterpret_cast<Load1T*>(in1 + (y * W + x) * st));
    arg1_t* tt = reinterpret_cast<arg1_t*>(&tmp1);

    #pragma unroll
    for (int i = 0; i < v_x; ++i) {
      tile[threadIdx.y][threadIdx.x * v_x + i] = tt[i];
    }
  }
  __syncthreads();

  x = r * KTile + threadIdx.x * v_x;
  y = c * KTile + threadIdx.y;
  if (x < H && y < W) {
    StoreT tmp_res;
    Load0T tmp_arg0;
    Load1T tmp_arg1;
    #pragma unroll
    for (int i = 0; i < v_x; ++i) {
      tmp_arg1.val[i] = tile[threadIdx.x * v_x + i][threadIdx.y];
    }
    tmp_arg0 = *reinterpret_cast<Load0T*>(in0 + (y * H + x) * st);
    #pragma unroll
    for (int i = 0; i < v_x; ++i) {
      tmp_res.val[i] = f(tmp_arg0.val[i], tmp_arg1.val[i]);
    }
    *reinterpret_cast<StoreT*>(out + (y * H + x) * st) = tmp_res;
  }
}


template<int nt, int vt, typename res_t , typename arg0_t, typename arg1_t, typename func_t, typename index_t, typename stride_t>
C10_LAUNCH_BOUNDS_2(nt, 4)
__global__ void elementwise_kernel_2_2(
    int64_t N,
    char* data0, char* data1, char* data2,
    index_t size0, index_t size1,
    stride_t stride00, stride_t stride01, stride_t stride02,
    stride_t stride10, stride_t stride11, stride_t stride12,
    func_t f) {
  // ndim = 2, arity = 2, narg = 3
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
      int64_t offsets[3];
      auto linear_idx = idx;
      constexpr int NARGS = 3;
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
      arg = 2;
      offsets[arg] += divmod_mod * stride02;
      // dim = 1
      divmod_div = linear_idx / size1;
      divmod_mod = linear_idx % size1;
      linear_idx = divmod_div;
      arg = 0;
      offsets[arg] += divmod_mod * stride10;
      arg = 1;
      offsets[arg] += divmod_mod * stride11;
      arg = 2;
      offsets[arg] += divmod_mod * stride12;

      res_t* out = (res_t*)(data0 + offsets[0]);
      *out = f(func_reinterpret_cast<arg0_t>(data1 + offsets[1]),
               func_reinterpret_cast<arg1_t>(data2 + offsets[2]));

      idx += nt;
    }
  }
}

template<int nt, int vt, int vt_0, int vt_1, int vec, typename res_t , typename arg0_t, typename arg1_t, typename func_t, typename index_t, typename stride_t>
C10_LAUNCH_BOUNDS_2(nt, 4)
__global__ void elementwise_kernel_2_2_template(
    int64_t N,
    char* data0, char* data1, char* data2,
    index_t size0, index_t size1,
    stride_t stride00, stride_t stride01, stride_t stride02,
    stride_t stride10, stride_t stride11, stride_t stride12,
    func_t f) {
  // ndim = 2, arity = 2, narg = 3
  int tid = threadIdx.x;
  int nv = nt * vec;
  int64_t linear_idx = nv * blockIdx.x + tid * vec;
  if (linear_idx >= N) return;
  int64_t offsets[3];
  constexpr int NARGS = 3;
  constexpr int MAX_DIMS = 2;
  #pragma unroll
  for (int arg = 0; arg < NARGS; arg++) {
    offsets[arg] = 0;
  }

  // load vec arg0
  arg0_t ld_1[vec];
  using LoadT_1 = at::native::memory::aligned_vector<arg0_t, vt_0>;
  int arg = 1;
  #pragma unroll
  for (int i = 0; i < vec; i += vt_0) {
    offsets[arg] = 0;
    auto linear_idx_1 = linear_idx + i;
    // dim = 0
    auto divmod_div_1 = linear_idx_1 / size0;
    auto divmod_mod_1 = linear_idx_1 % size0;
    linear_idx_1 = divmod_div_1;
    offsets[arg] += divmod_mod_1 * stride01;
    // dim = 1
    divmod_div_1 = linear_idx_1 / size1;
    divmod_mod_1 = linear_idx_1 % size1;
    linear_idx_1 = divmod_div_1;
    offsets[arg] += divmod_mod_1 * stride11;

    LoadT_1 val = *(reinterpret_cast<LoadT_1*>(data1 + offsets[arg]));
    #pragma unroll
    for (int j = 0; j < vt_0; ++j) {
      ld_1[i + j] = val.val[j];
    }
  }

  // load vec arg1
  arg1_t ld_2[vec];
  using LoadT_2 = at::native::memory::aligned_vector<arg1_t, vt_1>;
  arg = 2;
  #pragma unroll
  for (int i = 0; i < vec; i += vt_1) {
    offsets[arg] = 0;
    auto linear_idx_2 = linear_idx + i;
    // dim = 0
    auto divmod_div_2 = linear_idx_2 / size0;
    auto divmod_mod_2 = linear_idx_2 % size0;
    linear_idx_2 = divmod_div_2;
    offsets[arg] += divmod_mod_2 * stride02;
    // dim = 1
    divmod_div_2 = linear_idx_2 / size1;
    divmod_mod_2 = linear_idx_2 % size1;
    linear_idx_2 = divmod_div_2;
    offsets[arg] += divmod_mod_2 * stride12;

    LoadT_2 val = *(reinterpret_cast<LoadT_2*>(data2 + offsets[arg]));
    #pragma unroll
    for (int j = 0; j < vt_1; ++j) {
      ld_2[i + j] = val.val[j];
    }
  }

  // store output
  using Store_T = at::native::memory::aligned_vector<res_t, vt>;
  res_t output[vt];
  Store_T* p_store = reinterpret_cast<Store_T*>(&output);
  arg = 0;
  #pragma unroll
  for (int i = 0; i < vec; i += vt) {
    offsets[arg] = 0;
    auto linear_idx_0 = linear_idx + i;
    // dim = 0
    auto divmod_div_0 = linear_idx_0 / size0;
    auto divmod_mod_0 = linear_idx_0 % size0;
    linear_idx_0 = divmod_div_0;
    offsets[arg] += divmod_mod_0 * stride00;
    // dim = 1
    divmod_div_0 = linear_idx_0 / size1;
    divmod_mod_0 = linear_idx_0 % size1;
    linear_idx_0 = divmod_div_0;
    offsets[arg] += divmod_mod_0 * stride10;

    #pragma unroll
    for (int j = 0; j < vt; ++j) {
      output[j] = f(ld_1[i + j], ld_2[i + j]);
    }

    *(reinterpret_cast<Store_T*>(data0 + offsets[arg])) = *p_store;
  }
}


template<int nt, int vt, typename res_t , typename arg0_t, typename arg1_t, typename func_t, typename index_t, typename stride_t>
C10_LAUNCH_BOUNDS_2(nt, 4)
__global__ void elementwise_kernel_2_2_interval_arity1(
    int64_t N,
    char* data0, char* data1, char* data2,
    index_t size0, index_t size1,
    stride_t stride00, stride_t stride01, stride_t stride02,
    stride_t stride10, stride_t stride11, stride_t stride12,
    func_t f) {
  int tid = threadIdx.x;
  int idx = (blockIdx.x * nt + tid) * 8;

  if(idx < N){
    int64_t offsets[3];
    //out offset
    offsets[0] = idx * 2;
    //arity1 offset
    auto divmod_div = idx / size0;
    auto divmod_mod = idx % size0;
    offsets[1] = divmod_mod * 2 + divmod_div * stride11;
    //arity2 offset
    offsets[2] = offsets[0];

    float4 out;
    float4 tmp0 = *reinterpret_cast<float4*>(data1 + offsets[1]);
    float4 tmp1 = *reinterpret_cast<float4*>(data2 + offsets[2]);

    *(reinterpret_cast<res_t*>(&out)+0) =
      f(*(reinterpret_cast<arg0_t*>(&tmp0)+0),*(reinterpret_cast<arg1_t*>(&tmp1)+0));
    *(reinterpret_cast<res_t*>(&out)+1) =
      f(*(reinterpret_cast<arg0_t*>(&tmp0)+1),*(reinterpret_cast<arg1_t*>(&tmp1)+1));
    *(reinterpret_cast<res_t*>(&out)+2) =
      f(*(reinterpret_cast<arg0_t*>(&tmp0)+2),*(reinterpret_cast<arg1_t*>(&tmp1)+2));
    *(reinterpret_cast<res_t*>(&out)+3) =
      f(*(reinterpret_cast<arg0_t*>(&tmp0)+3),*(reinterpret_cast<arg1_t*>(&tmp1)+3));
    *(reinterpret_cast<res_t*>(&out)+4) =
      f(*(reinterpret_cast<arg0_t*>(&tmp0)+4),*(reinterpret_cast<arg1_t*>(&tmp1)+4));
    *(reinterpret_cast<res_t*>(&out)+5) =
      f(*(reinterpret_cast<arg0_t*>(&tmp0)+5),*(reinterpret_cast<arg1_t*>(&tmp1)+5));
    *(reinterpret_cast<res_t*>(&out)+6) =
      f(*(reinterpret_cast<arg0_t*>(&tmp0)+6),*(reinterpret_cast<arg1_t*>(&tmp1)+6));
    *(reinterpret_cast<res_t*>(&out)+7) =
      f(*(reinterpret_cast<arg0_t*>(&tmp0)+7),*(reinterpret_cast<arg1_t*>(&tmp1)+7));

    *reinterpret_cast<float4*>(data0 + offsets[0]) = out;
  }
}

// default output & input1 with same pattern, and stride12 >= stride10
template<int nt, int v_x, typename res_t , typename arg0_t, typename arg1_t, typename func_t, typename index_t, typename stride_t, bool swap_tensor=false>
C10_LAUNCH_BOUNDS_2(nt, 4)
__global__ void elementwise_kernel_2_2_align(
    int64_t N,
    char* data0, char* data1, char* data2,
    index_t size0, index_t size1,
    stride_t stride00, stride_t stride01, stride_t stride02,
    stride_t stride10, stride_t stride11, stride_t stride12,
    func_t f) {
  // ndim = 2, arity = 2, narg = 3
  // v_x: vectorized load n elems;
  // arg = 0: output; arg = 1: input1; arg = 2: input2;

  int64_t offsets[3];
  constexpr int NARGS = 3;
  constexpr int MAX_DIMS = 2;

  using LoadT = at::native::memory::aligned_vector<arg0_t, v_x>;
  using StoreT = at::native::memory::aligned_vector<res_t, v_x>;
  #pragma unroll
  for (int arg = 0; arg < NARGS; arg++) {
    offsets[arg] = 0;
  }
  int tid = threadIdx.x;
  size_t col_idx = (blockIdx.x * blockDim.x + tid) * v_x;
  size_t col_offset = gridDim.x * blockDim.x * v_x;
  size_t row_idx = blockIdx.y;
  size_t row_offset = gridDim.y;

  arg0_t ld_1[v_x];
  arg0_t ld_2[v_x];
  res_t ld_out[v_x];
  LoadT* p_ld_1 = reinterpret_cast<LoadT*>(&ld_1);
  LoadT* p_ld_2 = reinterpret_cast<LoadT*>(&ld_2);
  StoreT * p_ld_out = reinterpret_cast<StoreT*>(&ld_out);

  // as size0 is required multiple of 8, so here have no tail
  for (size_t r = row_idx; r < size1; r += row_offset){
    size_t r_offset0 = r * stride10;
    size_t r_offset1 = r * stride11;
    size_t r_offset2 = r * stride12;
    for (size_t c = col_idx; c < size0; c += col_offset){
      size_t c_offset = c * stride00;   // stride00 == stride01 ==stride02
      offsets[0] = r_offset0 + c_offset;
      offsets[1] = r_offset1 + c_offset;
      offsets[2] = r_offset2 + c_offset;
      *p_ld_1 = *reinterpret_cast<LoadT*>(data1 + offsets[1]);
      *p_ld_2 = *reinterpret_cast<LoadT*>(data2 + offsets[2]);
      StoreT* out = reinterpret_cast<StoreT*>(data0 + offsets[0]);
      #pragma unroll
      for (int i = 0; i < v_x; i++) {
        ld_out[i] = f(ld_1[i], ld_2[i]);
      }
      *out = *p_ld_out;
    }
  }
}

template<int nt, int v_x, typename res_t , typename arg0_t, typename arg1_t, typename func_t, typename index_t, typename stride_t, bool swap_tensor=false>
C10_LAUNCH_BOUNDS_2(nt, 4)
__global__ void elementwise_kernel_2_2_broadcast(
    int64_t N,
    char* data0, char* data1, char* data2,
    index_t size0, index_t size1,
    stride_t stride00, stride_t stride01, stride_t stride02,
    stride_t stride10, stride_t stride11, stride_t stride12,
    func_t f, int y_t, int y_remain) {
  // ndim = 2, arity = 2, narg = 3
  // v_x: vectorized load n elems;
  // y_t: store columns for each thread;
  // load_num: load iterations for each thread;
  int tid = threadIdx.x;
  int64_t offsets[3];
  constexpr int NARGS = 3;
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
  arg0_t ld_2[v_x];
  res_t ld_out[v_x];
  LoadT* p_ld_1 = reinterpret_cast<LoadT*>(&ld_1);
  LoadT* p_ld_2 = reinterpret_cast<LoadT*>(&ld_2);
  StoreT * p_ld_out = reinterpret_cast<StoreT*>(&ld_out);

  size_t y_loop = y_t;
  if (y_remain != 0 && blockIdx.y == (gridDim.y - 1)) {
    y_loop = y_remain;
  }

  offsets[2] = (blockIdx.x * blockDim.x + tid) * v_x * stride02;
  size_t row_offset = (blockIdx.x * blockDim.x + tid) * v_x;
  *p_ld_2 = *reinterpret_cast<LoadT*>(data2 + offsets[2]);
  for (size_t y_idx = 0; y_idx < y_loop; y_idx++) {
    offsets[0] = (blockIdx.y * y_t + y_idx) * stride10 + row_offset * stride00;
    offsets[1] = (blockIdx.y * y_t + y_idx) * stride11 + row_offset * stride01;
    *p_ld_1 = *reinterpret_cast<LoadT*>(data1 + offsets[1]);
    StoreT* out = reinterpret_cast<StoreT*>(data0 + offsets[0]);
    #pragma unroll
    for (int i = 0; i < v_x; i++) {
      if(swap_tensor){
        ld_out[i] = f(ld_2[i], ld_1[i]);
      }
      else{
        ld_out[i] = f(ld_1[i], ld_2[i]);
      }
    }
    *out = *p_ld_out;
  }

  if (x_remain && blockIdx.x == 0 && (tid + 1) * v_x <= x_remain) {
    offsets[2] = (gridDim.x * blockDim.x + tid) * v_x * stride02;
    // TODO(liuyuxin): support or assert lambda function.
    // *p_ld = f(*reinterpret_cast<LoadT*>(data1 + offsets[1]));
    *p_ld_2 = *reinterpret_cast<LoadT*>(data2 + offsets[2]);
    size_t row_offset = (gridDim.x * blockDim.x + tid) * v_x;
    for (int y_idx = 0; y_idx < y_loop; y_idx++) {
      offsets[0] = (blockIdx.y * y_t + y_idx) * stride10 + row_offset * stride00;
      offsets[1] = (blockIdx.y * y_t + y_idx) * stride11 + row_offset * stride01;
      *p_ld_1 = *reinterpret_cast<LoadT*>(data1 + offsets[1]);
      StoreT* out = reinterpret_cast<StoreT*>(data0 + offsets[0]);
      #pragma unroll
      for (int i = 0; i < v_x; i++) {
        if(swap_tensor){
          ld_out[i] = f(ld_2[i], ld_1[i]);
        }
        else{
          ld_out[i] = f(ld_1[i], ld_2[i]);
        }
      }
      *out = *p_ld_out;
    }
  }

  // handle tail columns( < v_x)
  if(tail && blockIdx.x == 0 && tid < tail) {
    auto remain_offset = x_remain / v_x;
    auto row_offset = (gridDim.x * blockDim.x + remain_offset) * v_x + tid;
    offsets[2] = row_offset * stride02;
    for (int y_idx = 0; y_idx < y_loop; y_idx++) {
      offsets[0] = (blockIdx.y * y_t + y_idx) * stride10 + row_offset * stride00;
      offsets[1] = (blockIdx.y * y_t + y_idx) * stride11 + row_offset * stride01;
      auto p0 = reinterpret_cast<res_t*>(data0 + offsets[0]);
      auto p1 = reinterpret_cast<arg0_t*>(data1 + offsets[1]);
      auto p2 = reinterpret_cast<arg0_t*>(data2 + offsets[2]);
      if (swap_tensor) {
        *p0 = f(*p2, *p1);
      } else {
        *p0 = f(*p1, *p2);
      }
    }
  }
}

template<int nt, int v_x, typename res_t , typename arg0_t, typename arg1_t, typename func_t, typename index_t, typename stride_t, typename res_dtype_t=float, typename arg0_dtype_t=float, typename arg1_dtype_t=float, bool swap_tensor=false>
C10_LAUNCH_BOUNDS_2(nt, 4)
__global__ void elementwise_kernel_2_2_cast_broadcast(
    int64_t N,
    char* data0, char* data1, char* data2,
    index_t size0, index_t size1,
    stride_t stride00, stride_t stride01, stride_t stride02,
    stride_t stride10, stride_t stride11, stride_t stride12,
    func_t f, int y_t, int y_remain) {
  // ndim = 2, arity = 2, narg = 3
  // v_x: vectorized load n elems;
  // y_t: store columns for each thread;
  // load_num: load iterations for each thread;
  int tid = threadIdx.x;
  int64_t offsets[3];
  constexpr int NARGS = 3;
  constexpr int MAX_DIMS = 2;
  size_t x_remain = size0 % (blockDim.x * v_x);
  size_t tail = size0 % v_x;
  // size_t y_remain = size1 % gridDim.y;
  using LoadT1 = at::native::memory::aligned_vector<arg0_dtype_t, v_x>;
  using LoadT2 = at::native::memory::aligned_vector<arg1_dtype_t, v_x>;
  using StoreT = at::native::memory::aligned_vector<res_dtype_t, v_x>;
  #pragma unroll
  for (int arg = 0; arg < NARGS; arg++) {
    offsets[arg] = 0;
  }
  // arg = 0: output; arg = 1: input1; arg = 2: input2;

  arg0_dtype_t ld_1[v_x];
  arg1_dtype_t ld_2[v_x];
  res_dtype_t ld_out[v_x];
  LoadT1* p_ld_1 = reinterpret_cast<LoadT1*>(&ld_1);
  LoadT2* p_ld_2 = reinterpret_cast<LoadT2*>(&ld_2);
  StoreT * p_ld_out = reinterpret_cast<StoreT*>(&ld_out);

  size_t y_loop = y_t;
  if (y_remain != 0 && blockIdx.y == (gridDim.y - 1)) {
    y_loop = y_remain;
  }

  offsets[2] = (blockIdx.x * blockDim.x + tid) * v_x * stride02;
  size_t row_offset0 = (blockIdx.x * blockDim.x + tid) * v_x * stride00;
  size_t row_offset1 = (blockIdx.x * blockDim.x + tid) * v_x * stride01;
  *p_ld_2 = *reinterpret_cast<LoadT2*>(data2 + offsets[2]);
  for (size_t y_idx = 0; y_idx < y_loop; y_idx++) {
    offsets[0] = (blockIdx.y * y_t + y_idx) * stride10 + row_offset0;
    offsets[1] = (blockIdx.y * y_t + y_idx) * stride11 + row_offset1;
    *p_ld_1 = *reinterpret_cast<LoadT1*>(data1 + offsets[1]);
    StoreT* out = reinterpret_cast<StoreT*>(data0 + offsets[0]);
    #pragma unroll
    for (int i = 0; i < v_x; i++) {
      if(swap_tensor){
        ld_out[i] = c10::convert<res_dtype_t>(f(c10::convert<arg1_t>(ld_2[i]), c10::convert<arg0_t>(ld_1[i])));
      }
      else{
        ld_out[i] = c10::convert<res_dtype_t>(f(c10::convert<arg0_t>(ld_1[i]), c10::convert<arg1_t>(ld_2[i])));
      }
    }
    *out = *p_ld_out;
  }

  if (x_remain && blockIdx.x == 0 && (tid + 1) * v_x <= x_remain) {
    offsets[2] = (gridDim.x * blockDim.x + tid) * v_x * stride02;
    // TODO(liuyuxin): support or assert lambda function.
    // *p_ld = f(*reinterpret_cast<LoadT*>(data1 + offsets[1]));
    *p_ld_2 = *reinterpret_cast<LoadT2*>(data2 + offsets[2]);
    size_t row_offset0 = (gridDim.x * blockDim.x + tid) * v_x * stride00;
    size_t row_offset1 = (gridDim.x * blockDim.x + tid) * v_x * stride01;
    for (int y_idx = 0; y_idx < y_loop; y_idx++) {
      offsets[0] = (blockIdx.y * y_t + y_idx) * stride10 + row_offset0;
      offsets[1] = (blockIdx.y * y_t + y_idx) * stride11 + row_offset1;
      *p_ld_1 = *reinterpret_cast<LoadT1*>(data1 + offsets[1]);
      StoreT* out = reinterpret_cast<StoreT*>(data0 + offsets[0]);
      #pragma unroll
      for (int i = 0; i < v_x; i++) {
        if(swap_tensor){
          ld_out[i] = c10::convert<res_dtype_t>(f(c10::convert<arg1_t>(ld_2[i]), c10::convert<arg0_t>(ld_1[i])));
        }
        else{
          ld_out[i] = c10::convert<res_dtype_t>(f(c10::convert<arg0_t>(ld_1[i]), c10::convert<arg1_t>(ld_2[i])));
        }
      }
      *out = *p_ld_out;
    }
  }

  // handle tail columns( < v_x)
  if(tail && blockIdx.x == 0 && tid < tail) {
    auto remain_offset = x_remain / v_x;
    offsets[2] = ((gridDim.x * blockDim.x + remain_offset) * v_x + tid)* stride02;
    int64_t row_offset0 = ((gridDim.x  * blockDim.x + remain_offset) * v_x + tid) * stride00;
    int64_t row_offset1 = ((gridDim.x  * blockDim.x + remain_offset) * v_x + tid) * stride01;
    for (int y_idx = 0; y_idx < y_loop; y_idx++) {
      offsets[0] = (blockIdx.y * y_t + y_idx) * stride10 + row_offset0;
      offsets[1] = (blockIdx.y * y_t + y_idx) * stride11 + row_offset1;
      auto p0 = reinterpret_cast<res_dtype_t*>(data0 + offsets[0]);
      auto p1 = reinterpret_cast<arg0_dtype_t*>(data1 + offsets[1]);
      auto p2 = reinterpret_cast<arg1_dtype_t*>(data2 + offsets[2]);
      *p0 = c10::convert<res_dtype_t>(f(c10::convert<arg0_t>(*p1), c10::convert<arg1_t>(*p2)));
    }
  }
}

template<int nt, int vt, int vt_0, int vt_1, int vec, typename res_t , typename arg0_t, typename arg1_t, typename func_t, typename index_t, typename stride_t, typename res_dtype_t=float, typename arg0_dtype_t=float, typename arg1_dtype_t=float>
C10_LAUNCH_BOUNDS_2(nt, 4)
__global__ void elementwise_kernel_2_2_cast_template(
    int64_t N,
    char* data0, char* data1, char* data2,
    index_t size0, index_t size1,
    stride_t stride00, stride_t stride01, stride_t stride02,
    stride_t stride10, stride_t stride11, stride_t stride12,
    func_t f) {
  // ndim = 2, arity = 2, narg = 3
  int tid = threadIdx.x;
  int nv = nt * vec;
  int64_t linear_idx = nv * blockIdx.x + tid * vec;
  if (linear_idx >= N) return;
  int64_t offsets[3];
  constexpr int NARGS = 3;
  constexpr int MAX_DIMS = 2;
  #pragma unroll
  for (int arg = 0; arg < NARGS; arg++) {
    offsets[arg] = 0;
  }

  arg0_dtype_t ld_1[vec];
  using LoadT_1 = at::native::memory::aligned_vector<arg0_dtype_t, vt_0>;
  int arg = 1;
  #pragma unroll
  for (int i = 0; i < vec; i += vt_0) {
    offsets[arg] = 0;
    auto linear_idx_1 = linear_idx + i;
    auto divmod_div_1 = linear_idx_1 / size0;
    auto divmod_mod_1 = linear_idx_1 % size0;
    linear_idx_1 = divmod_div_1;
    offsets[arg] += divmod_mod_1 * stride01;
    divmod_div_1 = linear_idx_1 / size1;
    divmod_mod_1 = linear_idx_1 % size1;
    linear_idx_1 = divmod_div_1;
    offsets[arg] += divmod_mod_1 * stride11;
    LoadT_1 val = *(reinterpret_cast<LoadT_1*>(data1 + offsets[arg]));
    #pragma unroll
    for (int j = 0; j < vt_0; ++j) {
      ld_1[i + j] = val.val[j];
    }
  }

  // load vec arg1
  arg1_dtype_t ld_2[vec];
  using LoadT_2 = at::native::memory::aligned_vector<arg1_dtype_t, vt_1>;
  arg = 2;
  #pragma unroll
  for (int i = 0; i < vec; i += vt_1) {
    offsets[arg] = 0;
    auto linear_idx_2 = linear_idx + i;
    // dim = 0
    auto divmod_div_2 = linear_idx_2 / size0;
    auto divmod_mod_2 = linear_idx_2 % size0;
    linear_idx_2 = divmod_div_2;
    offsets[arg] += divmod_mod_2 * stride02;
    // dim = 1
    divmod_div_2 = linear_idx_2 / size1;
    divmod_mod_2 = linear_idx_2 % size1;
    linear_idx_2 = divmod_div_2;
    offsets[arg] += divmod_mod_2 * stride12;

    LoadT_2 val = *(reinterpret_cast<LoadT_2*>(data2 + offsets[arg]));
    #pragma unroll
    for (int j = 0; j < vt_1; ++j) {
      ld_2[i + j] = val.val[j];
    }
  }

  // store output
  using Store_T = at::native::memory::aligned_vector<res_dtype_t, vt>;
  res_dtype_t output[vt];
  Store_T* p_store = reinterpret_cast<Store_T*>(&output);
  arg = 0;
  #pragma unroll
  for (int i = 0; i < vec; i += vt) {
    offsets[arg] = 0;
    auto linear_idx_0 = linear_idx + i;
    // dim = 0
    auto divmod_div_0 = linear_idx_0 / size0;
    auto divmod_mod_0 = linear_idx_0 % size0;
    linear_idx_0 = divmod_div_0;
    offsets[arg] += divmod_mod_0 * stride00;
    // dim = 1
    divmod_div_0 = linear_idx_0 / size1;
    divmod_mod_0 = linear_idx_0 % size1;
    linear_idx_0 = divmod_div_0;
    offsets[arg] += divmod_mod_0 * stride10;

    #pragma unroll
    for (int j = 0; j < vt; ++j) {
      output[j] = c10::convert<res_dtype_t>(f(c10::convert<arg0_t>(ld_1[i + j]), c10::convert<arg1_t>(ld_2[i + j])));
    }

    *(reinterpret_cast<Store_T*>(data0 + offsets[arg])) = *p_store;
  }
}

template<int nt, int v_x, typename res_t , typename arg0_t, typename arg1_t, typename func_t, typename index_t, typename stride_t, bool swap_tensor=false>
C10_LAUNCH_BOUNDS_2(nt, 4)
__global__ void elementwise_kernel_2_2_broadcast_uncontiguous(
    int64_t N,
    char* data0, char* data1, char* data2,
    index_t size0, index_t size1,
    stride_t stride00, stride_t stride01, stride_t stride02,
    stride_t stride10, stride_t stride11, stride_t stride12,
    func_t f, int y_t, int y_remain) {
  // ndim = 2, arity = 2, narg = 3
  // v_x: vectorized load n elems;
  // y_t: store columns for each thread;
  // load_num: load iterations for each thread;
  int tid = threadIdx.x;
  int64_t offsets[3];
  constexpr int NARGS = 3;
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

  size_t row_offset = (blockIdx.x * blockDim.x + tid) * v_x;
  size_t y_loop = y_t;
  if (y_remain != 0 && blockIdx.y == (gridDim.y - 1)) {
    y_loop = y_remain;
  }

  for (size_t y_idx = 0; y_idx < y_loop; y_idx++) {
    size_t y_offset = blockIdx.y * y_t + y_idx;
    offsets[2] = y_offset * stride12;
    auto p2 = *reinterpret_cast<arg0_t*>(data2 + offsets[2]);
    offsets[0] = y_offset * stride10 + row_offset * stride00;
    offsets[1] = y_offset * stride11 + row_offset * stride01;
    *p_ld_1 = *reinterpret_cast<LoadT*>(data1 + offsets[1]);
    StoreT* out = reinterpret_cast<StoreT*>(data0 + offsets[0]);
    #pragma unroll
    for (int i = 0; i < v_x; i++) {
      if(swap_tensor){
        ld_out[i] = f(p2, ld_1[i]);
      }
      else{
        ld_out[i] = f(ld_1[i], p2);
      }
    }
    *out = *p_ld_out;
  }

  if (x_remain != 0 && blockIdx.x == 0 && (tid + 1) * v_x <= x_remain) {
    size_t row_offset = (gridDim.x * blockDim.x + tid) * v_x;
    for (int y_idx = 0; y_idx < y_loop; y_idx++) {
      size_t y_offset = blockIdx.y * y_t + y_idx;
      offsets[2] = y_offset * stride12;
      auto p2 = *reinterpret_cast<arg0_t*>(data2 + offsets[2]);
      offsets[0] = y_offset * stride10 + row_offset * stride00;
      offsets[1] = y_offset * stride11 + row_offset * stride01;
      *p_ld_1 = *reinterpret_cast<LoadT*>(data1 + offsets[1]);
      StoreT* out = reinterpret_cast<StoreT*>(data0 + offsets[0]);
      #pragma unroll
      for (int i = 0; i < v_x; i++) {
        if(swap_tensor){
          ld_out[i] = f(p2, ld_1[i]);
        }
        else{
          ld_out[i] = f(ld_1[i], p2);
        }
      }
      *out = *p_ld_out;
    }
  }

  // handle tail columns( < v_x)
  if(tail && blockIdx.x == 0 && tid < tail){
    auto remain_offset = x_remain / v_x;
    int64_t row_offset = ((gridDim.x  * blockDim.x + remain_offset) * v_x + tid);
    for (int y_idx = 0; y_idx < y_loop; y_idx++) {
      size_t y_offset = blockIdx.y * y_t + y_idx;
      offsets[2] = y_offset * stride12;
      offsets[0] = y_offset * stride10 + row_offset * stride00;
      offsets[1] = y_offset * stride11 + row_offset * stride01;
      auto p0 = reinterpret_cast<res_t*>(data0 + offsets[0]);
      auto p1 = reinterpret_cast<arg0_t*>(data1 + offsets[1]);
      auto p2 = reinterpret_cast<arg0_t*>(data2 + offsets[2]);
      if(swap_tensor){
        *p0 = f(*p2, *p1);
      }
      else{
        *p0 = f(*p1, *p2);
      }
    }
  }
}

template<int nt, int v_x, typename res_t , typename arg0_t, typename arg1_t, typename func_t, typename index_t, typename stride_t, typename res_dtype_t=float, typename arg0_dtype_t=float, typename arg1_dtype_t=float, bool swap_tensor=false>
C10_LAUNCH_BOUNDS_2(nt, 4)
__global__ void elementwise_kernel_2_2_cast_broadcast_uncontiguous(
    int64_t N,
    char* data0, char* data1, char* data2,
    index_t size0, index_t size1,
    stride_t stride00, stride_t stride01, stride_t stride02,
    stride_t stride10, stride_t stride11, stride_t stride12,
    func_t f, int y_t, int y_remain) {
  // ndim = 2, arity = 2, narg = 3
  // v_x: vectorized load n elems;
  // y_t: store columns for each thread;
  // load_num: load iterations for each thread;
  int tid = threadIdx.x;
  int64_t offsets[3];
  constexpr int NARGS = 3;
  constexpr int MAX_DIMS = 2;
  size_t x_remain = size0 % (blockDim.x * v_x);
  size_t tail = size0 % v_x;
  // size_t y_remain = size1 % gridDim.y;
  using LoadT1 = at::native::memory::aligned_vector<arg0_dtype_t, v_x>;
  using LoadT2 = at::native::memory::aligned_vector<arg1_dtype_t, v_x>;
  using StoreT = at::native::memory::aligned_vector<res_dtype_t, v_x>;
  #pragma unroll
  for (int arg = 0; arg < NARGS; arg++) {
    offsets[arg] = 0;
  }
  // arg = 0: output; arg = 1: input1; arg = 2: input2;

  arg0_dtype_t ld_1[v_x];
  arg1_dtype_t ld_2[v_x];
  res_dtype_t ld_out[v_x];
  LoadT1* p_ld_1 = reinterpret_cast<LoadT1*>(&ld_1);
  LoadT2* p_ld_2 = reinterpret_cast<LoadT2*>(&ld_2);
  StoreT * p_ld_out = reinterpret_cast<StoreT*>(&ld_out);

  size_t row_offset0 = (blockIdx.x * blockDim.x + tid) * v_x * stride00;
  size_t row_offset1 = (blockIdx.x * blockDim.x + tid) * v_x * stride01;
  size_t y_loop = y_t;
  if (y_remain != 0 && blockIdx.y == (gridDim.y - 1)) {
    y_loop = y_remain;
  }

  for (size_t y_idx = 0; y_idx < y_loop; y_idx++) {
    size_t y_offset = blockIdx.y * y_t + y_idx;
    offsets[2] = y_offset * stride12;
    auto p2 = *reinterpret_cast<arg1_dtype_t*>(data2 + offsets[2]);
    offsets[0] = y_offset * stride10 + row_offset0;
    offsets[1] = y_offset * stride11 + row_offset1;
    *p_ld_1 = *reinterpret_cast<LoadT1*>(data1 + offsets[1]);
    StoreT* out = reinterpret_cast<StoreT*>(data0 + offsets[0]);
    #pragma unroll
    for (int i = 0; i < v_x; i++) {
      if(swap_tensor){
        ld_out[i] = c10::convert<res_dtype_t>(f(c10::convert<arg1_t>(p2), c10::convert<arg0_t>(ld_1[i])));
      }
      else{
        ld_out[i] = c10::convert<res_dtype_t>(f(c10::convert<arg0_t>(ld_1[i]), c10::convert<arg1_t>(p2)));
      }
    }
    *out = *p_ld_out;
  }

  if (x_remain != 0 && blockIdx.x == 0 && (tid + 1) * v_x <= x_remain) {
    size_t row_offset0 = (gridDim.x * blockDim.x + tid) * v_x * stride00;
    size_t row_offset1 = (gridDim.x * blockDim.x + tid) * v_x * stride01;
    for (int y_idx = 0; y_idx < y_loop; y_idx++) {
      size_t y_offset = blockIdx.y * y_t + y_idx;
      offsets[2] = y_offset * stride12;
      auto p2 = *reinterpret_cast<arg1_dtype_t*>(data2 + offsets[2]);
      offsets[0] = y_offset * stride10 + row_offset0;
      offsets[1] = y_offset * stride11 + row_offset1;
      *p_ld_1 = *reinterpret_cast<LoadT1*>(data1 + offsets[1]);
      StoreT* out = reinterpret_cast<StoreT*>(data0 + offsets[0]);
      #pragma unroll
      for (int i = 0; i < v_x; i++) {
        if(swap_tensor){
          ld_out[i] = c10::convert<res_dtype_t>(f(c10::convert<arg1_t>(p2), c10::convert<arg0_t>(ld_1[i])));
        }
        else{
          ld_out[i] = c10::convert<res_dtype_t>(f(c10::convert<arg0_t>(ld_1[i]), c10::convert<arg1_t>(p2)));
        }
      }
      *out = *p_ld_out;
    }
  }

  // handle tail columns( < v_x)
  if(tail && blockIdx.x == 0 && tid < tail){
    auto remain_offset = x_remain / v_x;
    int64_t row_offset0 = ((gridDim.x  * blockDim.x + remain_offset) * v_x + tid) * stride00;
    int64_t row_offset1 = ((gridDim.x  * blockDim.x + remain_offset) * v_x + tid) * stride01;
    for (int y_idx = 0; y_idx < y_loop; y_idx++) {
      size_t y_offset = blockIdx.y * y_t + y_idx;
      offsets[2] = y_offset * stride12;
      offsets[0] = y_offset * stride10 + row_offset0;
      offsets[1] = y_offset * stride11 + row_offset1;
      auto p0 = reinterpret_cast<res_dtype_t*>(data0 + offsets[0]);
      auto p1 = reinterpret_cast<arg0_dtype_t*>(data1 + offsets[1]);
      auto p2 = reinterpret_cast<arg1_dtype_t*>(data2 + offsets[2]);
      *p0 = c10::convert<res_dtype_t>(f(c10::convert<arg0_t>(*p1), c10::convert<arg1_t>(*p2)));
      if(swap_tensor){
        *p0 = c10::convert<res_dtype_t>(f(c10::convert<arg1_t>(*p2), c10::convert<arg0_t>(*p1)));
      }
      else{
        *p0 = c10::convert<res_dtype_t>(f(c10::convert<arg0_t>(*p1), c10::convert<arg1_t>(*p2)));
      }
    }
  }
}

template<int nt, int vt, typename res_t , typename arg0_t, typename arg1_t, typename func_t, typename index_t, typename stride_t, int size0, int size1>
C10_LAUNCH_BOUNDS_2(nt, 4)
__global__ void elementwise_kernel_2_2_s(
    int64_t N,
    char* data0, char* data1, char* data2,
    stride_t stride00, stride_t stride01, stride_t stride02,
    stride_t stride10, stride_t stride11, stride_t stride12,
    func_t f) {
  // ndim = 2, arity = 2, narg = 3
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
      int64_t offsets[3];
      auto linear_idx = idx;
      constexpr int NARGS = 3;
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
      arg = 2;
      offsets[arg] += divmod_mod * stride02;
      // dim = 1
      divmod_div = linear_idx / size1;
      divmod_mod = linear_idx % size1;
      linear_idx = divmod_div;
      arg = 0;
      offsets[arg] += divmod_mod * stride10;
      arg = 1;
      offsets[arg] += divmod_mod * stride11;
      arg = 2;
      offsets[arg] += divmod_mod * stride12;

      res_t* out = (res_t*)(data0 + offsets[0]);
      *out = f(func_reinterpret_cast<arg0_t>(data1 + offsets[1]),
               func_reinterpret_cast<arg1_t>(data2 + offsets[2]));

      idx += nt;
    }
  }
}

template<int nt, int vt, typename res_t , typename arg0_t, typename arg1_t, typename func_t, typename index_t, typename stride_t>
C10_LAUNCH_BOUNDS_2(nt, 4)
__global__ void elementwise_kernel_3_2_dim0_contiguous(
    int64_t N,
    char* data0, char* data1, char* data2,
    index_t size0, index_t size1, index_t size2,
    stride_t stride00, stride_t stride01, stride_t stride02,
    stride_t stride10, stride_t stride11, stride_t stride12,
    stride_t stride20, stride_t stride21, stride_t stride22,
    func_t f) {
  int64_t linear_idx = (blockIdx.x * blockDim.x + threadIdx.x) * vt;
  int64_t res_idx = linear_idx * stride00;
  if (linear_idx >= N) return;

  using vec_res = at::native::memory::aligned_vector<res_t, vt>;
  using vec_arg0 = at::native::memory::aligned_vector<arg0_t, vt>;
  using vec_arg1 = at::native::memory::aligned_vector<arg1_t, vt>;
  int64_t offset1 = 0, offset2 = 0;

  auto divmod_div = linear_idx / size0;
  auto divmod_mod = linear_idx % size0;
  linear_idx = divmod_div;
  offset1 += divmod_mod * stride01;
  offset2 += divmod_mod * stride02;
  
  divmod_div = linear_idx / size1;
  divmod_mod = linear_idx % size1;
  linear_idx = divmod_div;
  offset1 += divmod_mod * stride11;
  offset2 += divmod_mod * stride12;

  divmod_div = linear_idx / size2;
  divmod_mod = linear_idx % size2;
  linear_idx = divmod_div;
  offset1 += divmod_mod * stride21;
  offset2 += divmod_mod * stride22;

  vec_arg0 arg1_elems = *(reinterpret_cast<vec_arg0*>(data1 + offset1));
  vec_arg1 arg2_elems = *(reinterpret_cast<vec_arg1*>(data2 + offset2));
  vec_res  res_elems;
  #pragma unroll
  for (int ii = 0; ii < vt; ii++) {
    res_elems.val[ii] = f(arg1_elems.val[ii],arg2_elems.val[ii]);
  }

  *(reinterpret_cast<vec_res*>(data0 + res_idx)) = res_elems;
}

template<int nt, int vt, typename res_t , typename arg0_t, typename arg1_t, typename func_t, typename index_t, typename stride_t>
C10_LAUNCH_BOUNDS_2(nt, 4)
__global__ void elementwise_kernel_3_2_broadcast_dim0(
    int64_t N,
    char* data0, char* data1, char* data2,
    index_t size0, index_t size1, index_t size2,
    stride_t stride00, stride_t stride01, stride_t stride02,
    stride_t stride10, stride_t stride11, stride_t stride12,
    stride_t stride20, stride_t stride21, stride_t stride22,
    func_t f) {
  int64_t linear_idx = (blockIdx.x * blockDim.x + threadIdx.x) * vt;
  int64_t res_idx = linear_idx * stride00;
  if (linear_idx >= N) return;

  using vec_res = at::native::memory::aligned_vector<res_t, vt>;
  using vec_arg0 = at::native::memory::aligned_vector<arg0_t, vt>;
  int64_t offset2 = 0;

  auto divmod_div = linear_idx / size0;
  auto divmod_mod = linear_idx % size0;
  linear_idx = divmod_div;
  offset2 += divmod_mod * stride02;

  divmod_div = linear_idx / size1;
  divmod_mod = linear_idx % size1;
  linear_idx = divmod_div;
  offset2 += divmod_mod * stride12;

  divmod_div = linear_idx / size2;
  divmod_mod = linear_idx % size2;
  linear_idx = divmod_div;
  offset2 += divmod_mod * stride22;

  vec_arg0 arg1_elems = *(reinterpret_cast<vec_arg0*>(data1 + res_idx));
  arg1_t arg2 = *(reinterpret_cast<arg1_t*>(data2 + offset2));
  vec_res  res_elems;
  #pragma unroll
  for (int ii = 0; ii < vt; ii++) {
    res_elems.val[ii] = f(arg1_elems.val[ii],arg2);
  }

  *(reinterpret_cast<vec_res*>(data0 + res_idx)) = res_elems;
}

template<int nt, int vt, typename res_t , typename arg0_t, typename arg1_t, typename func_t, typename index_t, typename stride_t>
C10_LAUNCH_BOUNDS_2(nt, 4)
__global__ void elementwise_kernel_3_2_broadcast_arg0_dim2_arg1_dim0(
    int64_t N,
    char* data0, char* data1, char* data2,
    index_t size0, index_t size1, index_t size2,
    stride_t stride00, stride_t stride01, stride_t stride02,
    stride_t stride10, stride_t stride11, stride_t stride12,
    stride_t stride20, stride_t stride21, stride_t stride22,
    int64_t vec1, func_t f) {

  using StoreT = at::native::memory::aligned_vector<res_t, vt>;
  using arg0T = at::native::memory::aligned_vector<arg0_t, vt>;
  using arg1T = at::native::memory::aligned_vector<arg1_t, vt>;
  StoreT st;
  arg0T ld0;

  int64_t index01 = (blockIdx.x * blockDim.x + threadIdx.x) * vt;
  int64_t index0 = index01 % size0;
  int64_t index1 = index01 / size0;
  int offset0 = index0 * stride00 + index1 * stride10;
  int offset1 = index0 * stride01 + index1 * stride11;
  int offset2 =                     index1 * stride12;

  #pragma unroll
  for (int i = 0; i < vt; i++) {
    ld0.val[i] = *(reinterpret_cast<arg0_t*>(data1 + offset1 + i * stride01));
  }

  for(int64_t id = 0; id < vec1; id++) {
    int64_t index2 = blockIdx.y * vec1 + id;
    int64_t res_idx = offset0 + index2 * stride20;
    int64_t arg1_idx = offset2 + index2 * stride22;
    arg1_t tmp = *(reinterpret_cast<arg1_t*>(data2 + arg1_idx));

    #pragma unroll
    for (int i = 0; i < vt; i++) {
      st.val[i] = f(ld0.val[i], tmp);
    }

    *(reinterpret_cast<StoreT*>(data0 + res_idx)) = st;
  }
}

template<int nt, int vt, typename res_t , typename arg0_t, typename arg1_t, typename func_t, typename index_t, typename stride_t>
C10_LAUNCH_BOUNDS_2(nt, 4)
__global__ void elementwise_kernel_3_2_dim0_contiguous_arg1_dim1_broadcast(
    int64_t N,
    char* data0, char* data1, char* data2,
    index_t size0, index_t size1, index_t size2,
    stride_t stride00, stride_t stride01, stride_t stride02,
    stride_t stride10, stride_t stride11, stride_t stride12,
    stride_t stride20, stride_t stride21, stride_t stride22,
    func_t f) {
  using StoreT = at::native::memory::aligned_vector<res_t, vt>;
  using arg0T = at::native::memory::aligned_vector<arg0_t, vt>;
  using arg1T = at::native::memory::aligned_vector<arg1_t, vt>;

  int64_t linear_idx = (blockIdx.x * blockDim.x + threadIdx.x);
  if ((linear_idx * vt * vt) >= N) return;

  size0 = size0 / vt;
  size1 = size1 / vt;
  int64_t dim0 = (linear_idx % size0) * vt;
  int64_t div = linear_idx / size0;
  int64_t dim1 = (div % size1) * vt;
  int64_t dim2 = div / size1;

  int64_t offset0_tmp = dim0 * stride00  + dim2 * stride20;
  int64_t offset1_tmp = dim0 * stride01  + dim2 * stride21;
  int64_t offset2     = dim0 * stride02  + dim2 * stride22;

  arg1T arg1 = *(reinterpret_cast<arg1T*>(data2 + offset2));
  #pragma unroll
  for (int i = 0; i < vt; i++) {
    int64_t offset0 = offset0_tmp + stride10 * (i + dim1);
    int64_t offset1 = offset1_tmp + stride11 * (i + dim1);
    arg0T arg0 = *(reinterpret_cast<arg0T*>(data1 + offset1));

    StoreT st;
    #pragma unroll
    for (int j = 0; j < vt; j++) {
      st.val[j] = f(arg0.val[j], arg1.val[j]);
    }

    *(reinterpret_cast<StoreT*>(data0 + offset0)) = st;
  }

}

template<int nt, int vt, typename res_t , typename arg0_t, typename arg1_t, typename func_t, typename index_t, typename stride_t>
C10_LAUNCH_BOUNDS_2(nt, 4)
__global__ void elementwise_kernel_3_2(
    int64_t N,
    char* data0, char* data1, char* data2,
    index_t size0, index_t size1, index_t size2,
    stride_t stride00, stride_t stride01, stride_t stride02,
    stride_t stride10, stride_t stride11, stride_t stride12,
    stride_t stride20, stride_t stride21, stride_t stride22,
    func_t f) {
  // ndim = 3, arity = 2, narg = 3
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
      int64_t offsets[3];
      auto linear_idx = idx;
      constexpr int NARGS = 3;
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
      arg = 2;
      offsets[arg] += divmod_mod * stride02;
      // dim = 1
      divmod_div = linear_idx / size1;
      divmod_mod = linear_idx % size1;
      linear_idx = divmod_div;
      arg = 0;
      offsets[arg] += divmod_mod * stride10;
      arg = 1;
      offsets[arg] += divmod_mod * stride11;
      arg = 2;
      offsets[arg] += divmod_mod * stride12;
      // dim = 2
      divmod_div = linear_idx / size2;
      divmod_mod = linear_idx % size2;
      linear_idx = divmod_div;
      arg = 0;
      offsets[arg] += divmod_mod * stride20;
      arg = 1;
      offsets[arg] += divmod_mod * stride21;
      arg = 2;
      offsets[arg] += divmod_mod * stride22;

      res_t* out = (res_t*)(data0 + offsets[0]);
      *out = f(func_reinterpret_cast<arg0_t>(data1 + offsets[1]),
               func_reinterpret_cast<arg1_t>(data2 + offsets[2]));

      idx += nt;
    }
  }
}
template<typename func_t,typename index_t, typename stride_t,
          typename arg0_t, typename arg1_t,
          typename res_t,
          bool equal>
//C10_LAUNCH_BOUNDS_2(nt, 4)
__global__ void elementwise_kernel_3_2_opt_tile(
    int size0,int size,
    char* data0, char* data1, char* data2,
    stride_t stride00, stride_t stridea0, stride_t strideb0,
    stride_t stride01, stride_t stridea1, stride_t strideb1,
    stride_t stride02, stride_t stridea2, stride_t strideb2,
    func_t f) {
    // padding to avoid bank conflict
    constexpr int pad_size = 4 / sizeof(arg1_t);
    __shared__ arg1_t data_s[64][64 + pad_size];

    stride_t  offsets_start2=blockIdx.z*1*strideb2 + blockIdx.y*64*stridea2 + blockIdx.x*64*stride02;
    stride_t  offsets_start1=blockIdx.z*1*strideb1 + blockIdx.y*64*stridea1 + blockIdx.x*64*stride01;
    stride_t  offsets_start0=blockIdx.z*1*strideb0 + blockIdx.y*64*stridea0 + blockIdx.x*64*stride00;

    unsigned int inter_g_id = threadIdx.x /64;
    unsigned int intro_g_id = threadIdx.x %64;
    unsigned int mod_x = (size0 & 63)?(size0 & 63):64;
    unsigned int mod_y = (size & 63)?(size & 63):64;

    if(!equal){
      if (blockIdx.x != gridDim.x-1 && blockIdx.y != gridDim.y-1){
        // if stride02 == 0, only read once and store to row0 in shared mem
        if (stride02 == 0) {
          if (inter_g_id == 0) {
            stride_t offset=offsets_start2+intro_g_id*stridea2;
            data_s[0][intro_g_id]=func_reinterpret_cast<arg1_t>(data2 + offset);
          }
        } else {
          #pragma unroll
          for ( int i=0;i<8;i++)
          {
            stride_t offset=offsets_start2+(8*i+inter_g_id)*stride02+intro_g_id*stridea2;
            data_s[inter_g_id+8*i][intro_g_id]=func_reinterpret_cast<arg1_t>(data2 + offset);
          }
        }
      }
      else{
        if (stride02 == 0) {
            if((blockIdx.x==gridDim.x-1 && blockIdx.y!=gridDim.y-1 && inter_g_id < mod_x && inter_g_id == 0) ||
                (blockIdx.x!=gridDim.x-1 && blockIdx.y==gridDim.y-1 && intro_g_id < mod_y) ||
                (blockIdx.x==gridDim.x-1 && blockIdx.y==gridDim.y-1 && intro_g_id < mod_y && inter_g_id < mod_x && inter_g_id == 0)
              ){
              stride_t offset=offsets_start2+intro_g_id*stridea2;
              data_s[0][intro_g_id]=func_reinterpret_cast<arg1_t>(data2 + offset);
            }
        } else {
          #pragma unroll
          for ( int i=0;i<8;i++)
          {
            if( (blockIdx.x==gridDim.x-1 && blockIdx.y!=gridDim.y-1 && (inter_g_id+8*i) < mod_x) ||
                (blockIdx.x!=gridDim.x-1 && blockIdx.y==gridDim.y-1 && intro_g_id < mod_y) ||
                (blockIdx.x==gridDim.x-1 && blockIdx.y==gridDim.y-1 && intro_g_id < mod_y && (inter_g_id+8*i) < mod_x)
              ){
              stride_t offset=offsets_start2+(8*i+inter_g_id)*stride02+intro_g_id*stridea2;
              data_s[inter_g_id+8*i][intro_g_id]=func_reinterpret_cast<arg1_t>(data2 + offset);
            }
          }
        }
      }
    } else {
      if (blockIdx.x != gridDim.x-1 && blockIdx.y != gridDim.y-1){
        // if stridea2 == 0, only read once and store to row0 in shared mem
        if (stridea2 == 0) {
          if (inter_g_id == 0) {
            stride_t offset=offsets_start2+intro_g_id*stride02;
            data_s[0][intro_g_id]=func_reinterpret_cast<arg1_t>(data2 + offset);
          }
        } else {
          #pragma unroll
          for ( int i=0;i<8;i++)
          {
            stride_t offset=offsets_start2+(8*i+inter_g_id)*stridea2+intro_g_id*stride02;
            data_s[inter_g_id+8*i][intro_g_id]=func_reinterpret_cast<arg1_t>(data2 + offset);
          }
        }
      }
      else{
        if (stridea2 == 0) {
          if( (blockIdx.x==gridDim.x-1 && blockIdx.y!=gridDim.y-1 && intro_g_id < mod_x) ||
              (blockIdx.x!=gridDim.x-1 && blockIdx.y==gridDim.y-1 && inter_g_id < mod_y && inter_g_id == 0) ||
              (blockIdx.x==gridDim.x-1 && blockIdx.y==gridDim.y-1 && intro_g_id < mod_x && inter_g_id < mod_y && inter_g_id == 0)
            ){
            stride_t offset=offsets_start2+intro_g_id*stride02;
            data_s[0][intro_g_id]=func_reinterpret_cast<arg1_t>(data2 + offset);
          }
        } else {
          #pragma unroll
          for ( int i=0;i<8;i++)
          {
            if( (blockIdx.x==gridDim.x-1 && blockIdx.y!=gridDim.y-1 && intro_g_id < mod_x) ||
                (blockIdx.x!=gridDim.x-1 && blockIdx.y==gridDim.y-1 && (inter_g_id+i*8) < mod_y) ||
                (blockIdx.x==gridDim.x-1 && blockIdx.y==gridDim.y-1 && intro_g_id < mod_x && (inter_g_id+i*8) < mod_y)
              ){
              stride_t offset=offsets_start2+(8*i+inter_g_id)*stridea2+intro_g_id*stride02;
              data_s[inter_g_id+8*i][intro_g_id]=func_reinterpret_cast<arg1_t>(data2 + offset);
            }
          }
        }
      }
    }
    __syncthreads();
    if(!equal){
      if (blockIdx.x != gridDim.x-1 && blockIdx.y != gridDim.y-1){
        #pragma unroll
        for ( int i=0;i<8;i++)
        {
          stride_t offsets0=offsets_start0+(8*i+inter_g_id)*stridea0+intro_g_id*stride00;
          stride_t offset=offsets_start1+(8*i+inter_g_id)*stridea1+intro_g_id*stride01;
          res_t* out = (res_t*)(data0 + offsets0);
          if (stride02 == 0) *out = f(func_reinterpret_cast<arg0_t>(data1 + offset),data_s[0][8*i+inter_g_id]);
          else *out = f(func_reinterpret_cast<arg0_t>(data1 + offset),data_s[intro_g_id][8*i+inter_g_id]);
        }
      } else {
        #pragma unroll
        for ( int i=0;i<8;i++)
        {
          if( (blockIdx.x==gridDim.x-1 && blockIdx.y!=gridDim.y-1 && intro_g_id < mod_x) ||
              (blockIdx.x!=gridDim.x-1 && blockIdx.y==gridDim.y-1 && (inter_g_id+i*8) < mod_y)  ||
              (blockIdx.x==gridDim.x-1 && blockIdx.y==gridDim.y-1 && intro_g_id < mod_x && (inter_g_id+i*8) < mod_y)
            ){
            stride_t offsets0=offsets_start0+(8*i+inter_g_id)*stridea0+intro_g_id*stride00;
            stride_t offset=offsets_start1+(8*i+inter_g_id)*stridea1+intro_g_id*stride01;
            res_t* out = (res_t*)(data0 + offsets0);
            if (stride02 == 0) *out = f(func_reinterpret_cast<arg0_t>(data1 + offset),data_s[0][8*i+inter_g_id]);
            else *out = f(func_reinterpret_cast<arg0_t>(data1 + offset),data_s[intro_g_id][8*i+inter_g_id]);
          }
        }
      }
    } else {
      if (blockIdx.x != gridDim.x-1 && blockIdx.y != gridDim.y-1){
        #pragma unroll
        for ( int i=0;i<8;i++)
        {
          stride_t offsets0=offsets_start0+(8*i+inter_g_id)*stridea0+intro_g_id*stride00;
          stride_t offset=offsets_start1+(8*i+inter_g_id)*stridea1+intro_g_id*stride01;
          res_t* out = (res_t*)(data0 + offsets0);
          if (stridea2 == 0) *out = f(func_reinterpret_cast<arg0_t>(data1 + offset),data_s[0][intro_g_id]);
          else *out = f(func_reinterpret_cast<arg0_t>(data1 + offset),data_s[8*i+inter_g_id][intro_g_id]);
        }
      } else {
        #pragma unroll
        for ( int i=0;i<8;i++)
        {
          if( (blockIdx.x==gridDim.x-1 && blockIdx.y!=gridDim.y-1 && intro_g_id < mod_x) ||
              (blockIdx.x!=gridDim.x-1 && blockIdx.y==gridDim.y-1 && (inter_g_id+i*8) < mod_y)  ||
              (blockIdx.x==gridDim.x-1 && blockIdx.y==gridDim.y-1 && intro_g_id < mod_x && (inter_g_id+i*8) < mod_y)
            ){
            stride_t offsets0=offsets_start0+(8*i+inter_g_id)*stridea0+intro_g_id*stride00;
            stride_t offset=offsets_start1+(8*i+inter_g_id)*stridea1+intro_g_id*stride01;
            res_t* out = (res_t*)(data0 + offsets0);
            if (stridea2 == 0) *out = f(func_reinterpret_cast<arg0_t>(data1 + offset),data_s[0][intro_g_id]);
            else *out = f(func_reinterpret_cast<arg0_t>(data1 + offset),data_s[8*i+inter_g_id][intro_g_id]);
          }
        }
      }
    }

}
template<int nt, int v_x, typename res_t , typename arg0_t, typename arg1_t, typename func_t, typename index_t, typename stride_t>
C10_LAUNCH_BOUNDS_2(nt, 4)
__global__ void elementwise_kernel_3_2_broadcast_dim2_arg0_contiguous(
    int64_t N,
    char* data0, char* data1, char* data2,
    index_t size0, index_t size1, index_t size2,
    stride_t stride00, stride_t stride01, stride_t stride02,
    stride_t stride10, stride_t stride11, stride_t stride12,
    stride_t stride20, stride_t stride21, stride_t stride22,
    func_t f, int y_t, int y_remain) {
  // ndim = 3, arity = 2, narg = 3
  // load type arg0_dtype_t, compute type arg0_t, store type res_dtype_t
  // default arg1 is the broadcast
  int tid = threadIdx.x;
  int64_t offsets[3];
  constexpr int NARGS = 3;
  constexpr int MAX_DIMS = 2;
  size_t x_remain = size0 % (blockDim.x * v_x);
  size_t tail = size0 % v_x;

  using LoadT1 = at::native::memory::aligned_vector<arg0_t, v_x>;
  using StoreT = at::native::memory::aligned_vector<res_t, v_x>;
  #pragma unroll
  for (int arg = 0; arg < NARGS; arg++) {
    offsets[arg] = 0;
  }
  // arg = 0: output; arg = 1: input1; arg = 2: input2;

  arg0_t ld_1[v_x];
  arg1_t ld_2[v_x];
  res_t ld_out[v_x];
  LoadT1* p_ld_1 = reinterpret_cast<LoadT1*>(&ld_1);
  StoreT * p_ld_out = reinterpret_cast<StoreT*>(&ld_out);

  offsets[2] = (blockIdx.x * blockDim.x + tid) * v_x * stride02 + blockIdx.z * stride12;
  size_t row_offset0 = (blockIdx.x * blockDim.x + tid) * v_x * stride00 + blockIdx.z * stride10;
  size_t row_offset1 = (blockIdx.x * blockDim.x + tid) * v_x * stride01 + blockIdx.z * stride11;

  #pragma unroll
  for (int i = 0; i < v_x; i++) {
    ld_2[i] = *reinterpret_cast<arg1_t*>(data2 + offsets[2] + i * stride02);
  }

  for (size_t y_idx = 0; y_idx < y_t; y_idx++) {
    offsets[0] = (blockIdx.y * y_t + y_idx) * stride20 + row_offset0;
    // the layout of out is the same as arg0
    offsets[1] = (blockIdx.y * y_t + y_idx) * stride21 + row_offset1;
    *p_ld_1 = *reinterpret_cast<LoadT1*>(data1 + offsets[1]);
    StoreT* out = reinterpret_cast<StoreT*>(data0 + offsets[0]);
    #pragma unroll
    for (int i = 0; i < v_x; i++) {
      ld_out[i] = f(ld_1[i], ld_2[i]);
    }
    *out = *p_ld_out;
  }
  if (y_remain != 0 && blockIdx.y < y_remain) {
    size_t row_offset0 = (blockIdx.x * blockDim.x + tid) * v_x * stride00 + blockIdx.z * stride10;
    size_t row_offset1 = (blockIdx.x * blockDim.x + tid) * v_x * stride01 + blockIdx.z * stride11;
    offsets[0] = (gridDim.y * y_t + blockIdx.y) * stride20 + row_offset0;
    offsets[1] = (gridDim.y * y_t + blockIdx.y) * stride21 + row_offset1;
    *p_ld_1 = *reinterpret_cast<LoadT1*>(data1 + offsets[1]);
    StoreT* out = reinterpret_cast<StoreT*>(data0 + offsets[0]);
    #pragma unroll
    for (int i = 0; i < v_x; i++) {
      ld_out[i] = f(ld_1[i], ld_2[i]);
    }
    *out = *p_ld_out;
  }

  if (x_remain != 0 && (tid + 1) * v_x <= x_remain) {
    offsets[2] = (gridDim.x * blockDim.x + tid) * v_x * stride02 + blockIdx.z * stride12;
    // cache arg1
    #pragma unroll
    for (int i = 0; i < v_x; i++) {
      ld_2[i] = *reinterpret_cast<arg1_t*>(data2 + offsets[2] + i * stride02);
    }
    size_t row_offset0 = (gridDim.x * blockDim.x + tid) * v_x * stride00 + blockIdx.z * stride10;
    size_t row_offset1 = (gridDim.x * blockDim.x + tid) * v_x * stride01 + blockIdx.z * stride11;
    for (int y_idx = 0; y_idx < y_t; y_idx++) {
      offsets[0] = (blockIdx.y * y_t + y_idx) * stride20 + row_offset0;
      offsets[1] = (blockIdx.y * y_t + y_idx) * stride21 + row_offset1;
      *p_ld_1 = *reinterpret_cast<LoadT1*>(data1 + offsets[1]);
      StoreT* out = reinterpret_cast<StoreT*>(data0 + offsets[0]);
      #pragma unroll
      for (int i = 0; i < v_x; i++) {
        ld_out[i] = f(ld_1[i], ld_2[i]);
      }
      *out = *p_ld_out;
    }
    if (y_remain != 0 && blockIdx.y < y_remain) {
      offsets[0] = (gridDim.y * y_t + blockIdx.y) * stride20 + row_offset0;
      offsets[1] = (gridDim.y * y_t + blockIdx.y) * stride21 + row_offset1;
      *p_ld_1 = *reinterpret_cast<LoadT1*>(data1 + offsets[1]);
      StoreT* out = reinterpret_cast<StoreT*>(data0 + offsets[0]);
      #pragma unroll
      for (int i = 0; i < v_x; i++) {
        ld_out[i] = f(ld_1[i], ld_2[i]);
      }
      *out = *p_ld_out;
    }
  }

  // handle tail columns( < v_x)
  if(tail && tid < tail){
    auto remain_offset = x_remain / v_x;
    offsets[2] = ((gridDim.x * blockDim.x + remain_offset) * v_x + tid)* stride02 + blockIdx.z * stride12;
    int64_t row_offset0 = ((gridDim.x  * blockDim.x + remain_offset) * v_x + tid) * stride00 + blockIdx.z * stride10;
    int64_t row_offset1 = ((gridDim.x  * blockDim.x + remain_offset) * v_x + tid) * stride01 + blockIdx.z * stride11;
    for (int y_idx = 0; y_idx < y_t; y_idx++) {
      offsets[0] = (blockIdx.y * y_t + y_idx) * stride20 + row_offset0;
      offsets[1] = (blockIdx.y * y_t + y_idx) * stride21 + row_offset1;
      auto p0 = reinterpret_cast<res_t*>(data0 + offsets[0]);
      auto p1 = reinterpret_cast<arg0_t*>(data1 + offsets[1]);
      auto p2 = reinterpret_cast<arg1_t*>(data2 + offsets[2]);
      *p0 = f(*p1, *p2);
    }
    if (y_remain != 0 && blockIdx.y < y_remain) {
      offsets[0] = (gridDim.y * y_t + blockIdx.y) * stride20 + row_offset0;
      offsets[1] = (gridDim.y * y_t + blockIdx.y) * stride21 + row_offset1;
      auto p0 = reinterpret_cast<res_t*>(data0 + offsets[0]);
      auto p1 = reinterpret_cast<arg0_t*>(data1 + offsets[1]);
      auto p2 = reinterpret_cast<arg1_t*>(data2 + offsets[2]);
      *p0 = f(*p1, *p2);
    }
  }
}

template<int nt, int v_x, int KTile, typename res_t, typename arg0_t, typename arg1_t, typename func_t, typename TIndex, typename stride_t,
typename std::enable_if<!std::is_same<arg1_t, at::Half>::value && !std::is_same<arg1_t, at::BFloat16>::value
                       && !std::is_same<arg1_t, float>::value, int>::type = 0>
__global__ void elementwise_kernel_3_2_arity2_transpose(
  char* out, char* in0, char* in1,
  TIndex H, TIndex W, TIndex dh, TIndex dw,
  func_t f){
  assert(0);
}

template<int nt, int v_x, int KTile, typename res_t, typename arg0_t, typename arg1_t, typename func_t, typename TIndex, typename stride_t,
typename std::enable_if<std::is_same<arg1_t, at::Half>::value || std::is_same<arg1_t, at::BFloat16>::value
                       || std::is_same<arg1_t, float>::value, int>::type = 0>
__global__ void elementwise_kernel_3_2_arity2_transpose(
  char* out, char* in0, char* in1,
  TIndex H, TIndex W, TIndex dh, TIndex dw,
  func_t f) {
  const TIndex st = sizeof(arg1_t);
  using StoreT = at::native::memory::aligned_vector<res_t, v_x>;
  using Load0T = at::native::memory::aligned_vector<arg0_t, v_x>;
  using Load1T = at::native::memory::aligned_vector<arg1_t, v_x>;

  __shared__ arg1_t tile[KTile][KTile + 2];
  const TIndex n = blockIdx.z;
  const TIndex r = blockIdx.y;
  const TIndex c = blockIdx.x;
  const int64_t offset = n * H * W;
  int64_t x = c * KTile + threadIdx.x * v_x;
  int64_t y = r * KTile + threadIdx.y;

  if (x < W && y < H) {
    Load1T tmp1 = *(reinterpret_cast<Load1T*>(in1 + (offset + y * W + x) * st));
    arg1_t* tt = reinterpret_cast<arg1_t*>(&tmp1);

    #pragma unroll
    for (int i = 0; i < v_x; ++i) {
      tile[threadIdx.y][threadIdx.x * v_x + i] = tt[i];
    }
  }
  __syncthreads();
  x = r * KTile + threadIdx.x * v_x;
  y = c * KTile + threadIdx.y;
  if (x < H && y < W) {
    StoreT tmp_res;
    Load0T tmp_arg0;
    Load1T tmp_arg1;
    #pragma unroll
    for (int i = 0; i < v_x; ++i) {
      tmp_arg1.val[i] = tile[threadIdx.x * v_x + i][threadIdx.y];
    }
    tmp_arg0 = *reinterpret_cast<Load0T*>(in0 + (offset + y * H + x) * st);
    #pragma unroll
    for (int i = 0; i < v_x; ++i) {
      tmp_res.val[i] = f(tmp_arg0.val[i], tmp_arg1.val[i]);
    }
    *reinterpret_cast<StoreT*>(out + (offset + y * H + x) * st) = tmp_res;
  }

}

template<int nt, int v_x, int KTile, typename res_t, typename arg0_t, typename arg1_t, typename func_t, typename TIndex, typename stride_t,
typename std::enable_if<!std::is_same<arg1_t, at::Half>::value && !std::is_same<arg1_t, at::BFloat16>::value
                       && !std::is_same<arg1_t, float>::value, int>::type = 0>
__global__ void elementwise_kernel_3_2_arity2_transpose_dim02(
  char* out, char* in0, char* in1,
  TIndex size0, TIndex size1, TIndex size2,
  stride_t stride00, stride_t stride01, stride_t stride02,
  stride_t stride10, stride_t stride11, stride_t stride12,
  stride_t stride20, stride_t stride21, stride_t stride22,
  func_t f){
  assert(0);
}

template<int nt, int v_x, int KTile, typename res_t, typename arg0_t, typename arg1_t, typename func_t, typename TIndex, typename stride_t,
typename std::enable_if<std::is_same<arg1_t, at::Half>::value || std::is_same<arg1_t, at::BFloat16>::value
                       || std::is_same<arg1_t, float>::value, int>::type = 0>
__global__ void elementwise_kernel_3_2_arity2_transpose_dim02(
  char* out, char* in0, char* in1,
  TIndex size0, TIndex size1, TIndex size2,
  stride_t stride00, stride_t stride01, stride_t stride02,
  stride_t stride10, stride_t stride11, stride_t stride12,
  stride_t stride20, stride_t stride21, stride_t stride22,
  func_t f) {
  // size2 : contiguous dim in arg1
  // [size0, size2] transpose to [size2, size0]
  // const TIndex st = sizeof(arg1_t);
  using StoreT = at::native::memory::aligned_vector<res_t, v_x>;
  using Load0T = at::native::memory::aligned_vector<arg0_t, v_x>;
  using Load1T = at::native::memory::aligned_vector<arg1_t, v_x>;

  __shared__ arg1_t tile[KTile][KTile + 2];
  const TIndex n = blockIdx.z;  // the dim not in shared mem
  const TIndex r = blockIdx.y;
  const TIndex c = blockIdx.x;
  // const int64_t offset = n * H * W;
  int64_t x = c * KTile + threadIdx.x * v_x;
  int64_t y = r * KTile + threadIdx.y;

  if (x < size2 && y < size0) {
    // Load1T tmp1 = *(reinterpret_cast<Load1T*>(in1 + (offset + y * W + x) * st));
    const int64_t offset1 = x * stride22 + n * stride12 + y * stride02;
    Load1T tmp1 = *(reinterpret_cast<Load1T*>(in1 + offset1));
    arg1_t* tt = reinterpret_cast<arg1_t*>(&tmp1);

    #pragma unroll
    for (int i = 0; i < v_x; ++i) {
      tile[threadIdx.y][threadIdx.x * v_x + i] = tt[i];
    }
  }
  __syncthreads();
  x = r * KTile + threadIdx.x * v_x;
  y = c * KTile + threadIdx.y;
  if (x < size0 && y < size2) {
    StoreT tmp_res;
    Load0T tmp_arg0;
    Load1T tmp_arg1;
    #pragma unroll
    for (int i = 0; i < v_x; ++i) {
      tmp_arg1.val[i] = tile[threadIdx.x * v_x + i][threadIdx.y];
    }
    const int64_t offset_out = y * stride20 + n * stride10 + x * stride00;
    const int64_t offset0 = y * stride21 + n * stride11 + x * stride01;
    tmp_arg0 = *reinterpret_cast<Load0T*>(in0 + offset0);
    #pragma unroll
    for (int i = 0; i < v_x; ++i) {
      tmp_res.val[i] = f(tmp_arg0.val[i], tmp_arg1.val[i]);
    }
    *reinterpret_cast<StoreT*>(out + offset_out) = tmp_res;
  }
}

template<typename res_t, typename arg0_t, typename arg1_t, typename func_t, typename index_t, typename stride_t,
typename std::enable_if<!std::is_same<arg1_t, at::Half>::value && !std::is_same<arg1_t, at::BFloat16>::value
                       && !std::is_same<arg1_t, float>::value, int>::type = 0>
__global__ void elementwise_kernel_3_2_arity2_transpose_nonvec(
    char* data0, char* data1, char* data2,
    index_t H, index_t W, index_t dh, index_t dw,
    func_t f) {
    assert(0);
}

template<typename res_t, typename arg0_t, typename arg1_t, typename func_t, typename index_t, typename stride_t,
typename std::enable_if<std::is_same<arg1_t, at::Half>::value || std::is_same<arg1_t, at::BFloat16>::value
                       || std::is_same<arg1_t, float>::value, int>::type = 0>
__global__ void elementwise_kernel_3_2_arity2_transpose_nonvec(
    char* data0, char* data1, char* data2,
    index_t H, index_t W, index_t dh, index_t dw,
    func_t f) {
  // ndim = 3, arity = 2, narg = 3
  // res_t arg0_t arg1_t are same type
  stride_t st = sizeof(arg0_t);

  __shared__ arg1_t tile[kTileDimMacaT][kTileDimMacaT + 2];
  const index_t n = blockIdx.z;
  const index_t r = blockIdx.y;
  const index_t c = blockIdx.x;
  const int64_t offset = n * H * W;
  int64_t x = c * kTileDimMacaT + threadIdx.x;
  int64_t y = r * kTileDimMacaT + threadIdx.y;

  if (x < W) {
    for (int i = 0; threadIdx.y + i < kTileDimMacaT && y + i < H; i += kBlockRowsMacaT) {
      tile[threadIdx.y + i][threadIdx.x] = *(arg1_t*)(data2 + (offset + (y + i) * W + x) * st);
    }
  }
  __syncthreads();
  x = r * kTileDimMacaT + threadIdx.x;
  y = c * kTileDimMacaT + threadIdx.y;
  if (x < H) {
    for (int i = 0; threadIdx.y + i < kTileDimMacaT && y + i < W; i += kBlockRowsMacaT) {
      arg0_t arg0 = *(arg0_t*)(data1 + (offset + (y + i) * H + x) * st);
      arg1_t arg1 = tile[threadIdx.x][threadIdx.y + i];
      res_t res = f(arg0, arg1);
      *(res_t*)(data0 + (offset + (y + i) * H + x) * st) = res;
    }
  }
}

template<int nt, int v_x, typename res_t , typename arg0_t, typename arg1_t, typename func_t, typename index_t, typename stride_t>
C10_LAUNCH_BOUNDS_2(nt, 4)
__global__ void elementwise_kernel_3_2_broadcast_dim2_uncontiguous(
    int64_t N,
    char* data0, char* data1, char* data2,
    index_t size0, index_t size1, index_t size2,
    stride_t stride00, stride_t stride01, stride_t stride02,
    stride_t stride10, stride_t stride11, stride_t stride12,
    stride_t stride20, stride_t stride21, stride_t stride22,
    func_t f, int y_t, int z_t, int y_remain) {
  // ndim = 3, arity = 2, narg = 3
  // load type arg0_dtype_t, compute type arg0_t, store type res_dtype_t
  // default arg1 is the broadcast
  int tid = threadIdx.x;
  int64_t offsets[3];
  constexpr int NARGS = 3;
  constexpr int MAX_DIMS = 2;
  size_t x_remain = (size0 * z_t) % (blockDim.x * v_x);
  size_t tail = (size0 * z_t) % v_x;

  using StoreT = at::native::memory::aligned_vector<res_t, v_x>;
  #pragma unroll
  for (int arg = 0; arg < NARGS; arg++) {
    offsets[arg] = 0;
  }
  // arg = 0: output; arg = 1: input1; arg = 2: input2;

  res_t ld_out[v_x];
  arg1_t ld_2[v_x];

  StoreT *p_ld_out = reinterpret_cast<StoreT*>(&ld_out);

  auto z_offset_div = tid * v_x / size0;
  auto z_offset_mod = tid * v_x % size0;
  offsets[2] = blockIdx.x * blockDim.x * v_x * stride02 + z_offset_mod * stride02 + blockIdx.z * stride12 * z_t + z_offset_div * stride12;
  size_t row_offset0 = blockIdx.x * blockDim.x * v_x * stride00 + z_offset_mod * stride00 + blockIdx.z * stride10 * z_t + z_offset_div * stride10;
  size_t row_offset1 = blockIdx.x * blockDim.x * v_x * stride01 + z_offset_mod * stride01 + blockIdx.z * stride11 * z_t + z_offset_div * stride11;
  // cache arg1
  #pragma unroll
  for (int i = 0; i < v_x; i++) {
    ld_2[i] = *reinterpret_cast<arg1_t*>(data2 + offsets[2] + i * stride02);
  }
  for (size_t y_idx = 0; y_idx < y_t; y_idx++) {
    offsets[0] = (blockIdx.y * y_t + y_idx) * stride20 + row_offset0;
    offsets[1] = (blockIdx.y * y_t + y_idx) * stride21 + row_offset1;
    // vec store output
    StoreT* out = reinterpret_cast<StoreT*>(data0 + offsets[0]);
    auto ld_1 = *reinterpret_cast<arg0_t*>(data1 + offsets[1]);
    #pragma unroll
    for (int i = 0; i < v_x; i++) {
      // load arg0
      ld_1 = *reinterpret_cast<arg0_t*>(data1 + offsets[1] + i * stride01);
      ld_out[i] = f(ld_1, ld_2[i]);
    }
    *out = *p_ld_out;
  }
  if (y_remain != 0 && blockIdx.y < y_remain) {
    size_t row_offset0 = blockIdx.x * blockDim.x * v_x * stride00 + z_offset_mod * stride00 + blockIdx.z * stride10 * z_t + z_offset_div * stride10;
    size_t row_offset1 = blockIdx.x * blockDim.x * v_x * stride01 + z_offset_mod * stride01 + blockIdx.z * stride11 * z_t + z_offset_div * stride11;
    offsets[0] = (gridDim.y * y_t + blockIdx.y) * stride20 + row_offset0;
    offsets[1] = (gridDim.y * y_t + blockIdx.y) * stride21 + row_offset1;
    StoreT* out = reinterpret_cast<StoreT*>(data0 + offsets[0]);
    auto ld_1 = *reinterpret_cast<arg0_t*>(data1 + offsets[1]);
    #pragma unroll
    for (int i = 0; i < v_x; i++) {
      // load arg0
      ld_1 = *reinterpret_cast<arg0_t*>(data1 + offsets[1] + i * stride01);
      ld_out[i] = f(ld_1, ld_2[i]);
    }
    *out = *p_ld_out;
  }

  if (x_remain != 0 && (tid + 1) * v_x <= x_remain) {
    offsets[2] = gridDim.x * blockDim.x * v_x * stride02 + z_offset_mod * stride02 + blockIdx.z * stride12 * z_t + z_offset_div * stride12;
    // cache arg1
    #pragma unroll
    for (int i = 0; i < v_x; i++) {
      ld_2[i] = *reinterpret_cast<arg1_t*>(data2 + offsets[2] + i * stride02);
    }
    size_t row_offset0 = gridDim.x * blockDim.x * v_x * stride00 + z_offset_mod * stride00 + blockIdx.z * stride10 * z_t + z_offset_div * stride10;
    size_t row_offset1 = gridDim.x * blockDim.x * v_x * stride01 + z_offset_mod * stride01 + blockIdx.z * stride11 * z_t + z_offset_div * stride11;
    for (int y_idx = 0; y_idx < y_t; y_idx++) {
      offsets[0] = (blockIdx.y * y_t + y_idx) * stride20 + row_offset0;
      offsets[1] = (blockIdx.y * y_t + y_idx) * stride21 + row_offset1;
      StoreT* out = reinterpret_cast<StoreT*>(data0 + offsets[0]);
      auto ld_1 = *reinterpret_cast<arg0_t*>(data1 + offsets[1]);
      #pragma unroll
      for (int i = 0; i < v_x; i++) {
        // load arg0
        ld_1 = *reinterpret_cast<arg0_t*>(data1 + offsets[1] + i * stride01);
        ld_out[i] = f(ld_1, ld_2[i]);
      }
      *out = *p_ld_out;
    }
    if (y_remain != 0 && blockIdx.y < y_remain) {
      offsets[0] = (gridDim.y * y_t + blockIdx.y) * stride20 + row_offset0;
      offsets[1] = (gridDim.y * y_t + blockIdx.y) * stride21 + row_offset1;
      StoreT* out = reinterpret_cast<StoreT*>(data0 + offsets[0]);
      auto ld_1 = *reinterpret_cast<arg0_t*>(data1 + offsets[1]);
      #pragma unroll
      for (int i = 0; i < v_x; i++) {
        // load arg0
        ld_1 = *reinterpret_cast<arg0_t*>(data1 + offsets[1] + i * stride01);
        ld_out[i] = f(ld_1, ld_2[i]);
      }
      *out = *p_ld_out;
    }
  }

  // handle tail columns( < v_x)
  if(tail && tid < tail){
    auto remain_offset = x_remain / v_x;
    offsets[2] = (gridDim.x * blockDim.x + remain_offset) * v_x * stride02 + z_offset_mod * stride02 + blockIdx.z * stride12 * z_t + z_offset_div * stride12;
    int64_t row_offset0 = (gridDim.x  * blockDim.x + remain_offset) * v_x * stride00 + z_offset_mod * stride00 + blockIdx.z * stride10 * z_t + z_offset_div * stride10;
    int64_t row_offset1 = (gridDim.x  * blockDim.x + remain_offset) * v_x * stride01 + z_offset_mod * stride01 + blockIdx.z * stride11 * z_t + z_offset_div * stride11;
    for (int y_idx = 0; y_idx < y_t; y_idx++) {
      offsets[0] = (blockIdx.y * y_t + y_idx) * stride20 + row_offset0;
      offsets[1] = (blockIdx.y * y_t + y_idx) * stride21 + row_offset1;
      auto p0 = reinterpret_cast<res_t*>(data0 + offsets[0]);
      auto p1 = reinterpret_cast<arg0_t*>(data1 + offsets[1]);
      auto p2 = reinterpret_cast<arg1_t*>(data2 + offsets[2]);
      *p0 = f(*p1, *p2);
    }
    if (y_remain != 0 && blockIdx.y < y_remain) {
      offsets[0] = (gridDim.y * y_t + blockIdx.y) * stride20 + row_offset0;
      offsets[1] = (gridDim.y * y_t + blockIdx.y) * stride21 + row_offset1;
      auto p0 = reinterpret_cast<res_t*>(data0 + offsets[0]);
      auto p1 = reinterpret_cast<arg0_t*>(data1 + offsets[1]);
      auto p2 = reinterpret_cast<arg1_t*>(data2 + offsets[2]);
      *p0 = f(*p1, *p2);
    }
  }
}

template<int nt, int v_x, typename res_t , typename arg0_t, typename arg1_t, typename func_t, typename index_t, typename stride_t>
C10_LAUNCH_BOUNDS_2(nt, 4)
__global__ void elementwise_kernel_3_2_broadcast_dim2_contiguous(
    int64_t N,
    char* data0, char* data1, char* data2,
    index_t size0, index_t size1, index_t size2,
    stride_t stride00, stride_t stride01, stride_t stride02,
    stride_t stride10, stride_t stride11, stride_t stride12,
    stride_t stride20, stride_t stride21, stride_t stride22,
    func_t f, int y_t, int y_remain) {
  // ndim = 3, arity = 2, narg = 3
  // load type arg0_dtype_t, compute type arg0_t, store type res_dtype_t
  // default arg1 is the broadcast
  int tid = threadIdx.x;
  int64_t offsets[3];
  constexpr int NARGS = 3;
  constexpr int MAX_DIMS = 2;
  size_t x_remain = size0 % (blockDim.x * v_x);
  size_t tail = size0 % v_x;

  using LoadT1 = at::native::memory::aligned_vector<arg0_t, v_x>;
  using LoadT2 = at::native::memory::aligned_vector<arg1_t, v_x>;
  using StoreT = at::native::memory::aligned_vector<res_t, v_x>;
  #pragma unroll
  for (int arg = 0; arg < NARGS; arg++) {
    offsets[arg] = 0;
  }
  // arg = 0: output; arg = 1: input1; arg = 2: input2;

  arg0_t ld_1[v_x];
  arg1_t ld_2[v_x];
  res_t ld_out[v_x];
  LoadT1* p_ld_1 = reinterpret_cast<LoadT1*>(&ld_1);
  LoadT2* p_ld_2 = reinterpret_cast<LoadT2*>(&ld_2);
  StoreT * p_ld_out = reinterpret_cast<StoreT*>(&ld_out);

  offsets[2] = (blockIdx.x * blockDim.x + tid) * v_x * stride02 + blockIdx.z * stride12;
  size_t row_offset0 = (blockIdx.x * blockDim.x + tid) * v_x * stride00 + blockIdx.z * stride10;
  size_t row_offset1 = (blockIdx.x * blockDim.x + tid) * v_x * stride01 + blockIdx.z * stride11;
  *p_ld_2 = *reinterpret_cast<LoadT2*>(data2 + offsets[2]);
  for (size_t y_idx = 0; y_idx < y_t; y_idx++) {
    offsets[0] = (blockIdx.y * y_t + y_idx) * stride20 + row_offset0;
    // the layout of out is the same as arg0
    offsets[1] = (blockIdx.y * y_t + y_idx) * stride21 + row_offset1;
    *p_ld_1 = *reinterpret_cast<LoadT1*>(data1 + offsets[1]);
    StoreT* out = reinterpret_cast<StoreT*>(data0 + offsets[0]);
    #pragma unroll
    for (int i = 0; i < v_x; i++) {
      ld_out[i] = f(ld_1[i], ld_2[i]);
    }
    *out = *p_ld_out;
  }
  if (y_remain != 0 && blockIdx.y < y_remain) {
    size_t row_offset0 = (blockIdx.x * blockDim.x + tid) * v_x * stride00 + blockIdx.z * stride10;
    size_t row_offset1 = (blockIdx.x * blockDim.x + tid) * v_x * stride01 + blockIdx.z * stride11;
    offsets[0] = (gridDim.y * y_t + blockIdx.y) * stride20 + row_offset0;
    offsets[1] = (gridDim.y * y_t + blockIdx.y) * stride21 + row_offset1;
    *p_ld_1 = *reinterpret_cast<LoadT1*>(data1 + offsets[1]);
    StoreT* out = reinterpret_cast<StoreT*>(data0 + offsets[0]);
    #pragma unroll
    for (int i = 0; i < v_x; i++) {
      ld_out[i] = f(ld_1[i], ld_2[i]);
    }
    *out = *p_ld_out;
  }

  if (x_remain != 0 && (tid + 1) * v_x <= x_remain) {
    offsets[2] = (gridDim.x * blockDim.x + tid) * v_x * stride02 + blockIdx.z * stride12;
    *p_ld_2 = *reinterpret_cast<LoadT2*>(data2 + offsets[2]);
    size_t row_offset0 = (gridDim.x * blockDim.x + tid) * v_x * stride00 + blockIdx.z * stride10;
    size_t row_offset1 = (gridDim.x * blockDim.x + tid) * v_x * stride01 + blockIdx.z * stride11;
    for (int y_idx = 0; y_idx < y_t; y_idx++) {
      offsets[0] = (blockIdx.y * y_t + y_idx) * stride20 + row_offset0;
      offsets[1] = (blockIdx.y * y_t + y_idx) * stride21 + row_offset1;
      *p_ld_1 = *reinterpret_cast<LoadT1*>(data1 + offsets[1]);
      StoreT* out = reinterpret_cast<StoreT*>(data0 + offsets[0]);
      #pragma unroll
      for (int i = 0; i < v_x; i++) {
        ld_out[i] = f(ld_1[i], ld_2[i]);
      }
      *out = *p_ld_out;
    }
    if (y_remain != 0 && blockIdx.y < y_remain) {
      offsets[0] = (gridDim.y * y_t + blockIdx.y) * stride20 + row_offset0;
      offsets[1] = (gridDim.y * y_t + blockIdx.y) * stride21 + row_offset1;
      *p_ld_1 = *reinterpret_cast<LoadT1*>(data1 + offsets[1]);
      StoreT* out = reinterpret_cast<StoreT*>(data0 + offsets[0]);
      #pragma unroll
      for (int i = 0; i < v_x; i++) {
        ld_out[i] = f(ld_1[i], ld_2[i]);
      }
      *out = *p_ld_out;
    }
  }

  // handle tail columns( < v_x)
  if(tail && tid < tail){
    auto remain_offset = x_remain / v_x;
    offsets[2] = ((gridDim.x * blockDim.x + remain_offset) * v_x + tid)* stride02 + blockIdx.z * stride12;
    int64_t row_offset0 = ((gridDim.x  * blockDim.x + remain_offset) * v_x + tid) * stride00 + blockIdx.z * stride10;
    int64_t row_offset1 = ((gridDim.x  * blockDim.x + remain_offset) * v_x + tid) * stride01 + blockIdx.z * stride11;
    for (int y_idx = 0; y_idx < y_t; y_idx++) {
      offsets[0] = (blockIdx.y * y_t + y_idx) * stride20 + row_offset0;
      offsets[1] = (blockIdx.y * y_t + y_idx) * stride21 + row_offset1;
      auto p0 = reinterpret_cast<res_t*>(data0 + offsets[0]);
      auto p1 = reinterpret_cast<arg0_t*>(data1 + offsets[1]);
      auto p2 = reinterpret_cast<arg1_t*>(data2 + offsets[2]);
      *p0 = f(*p1, *p2);
    }
    if (y_remain != 0 && blockIdx.y < y_remain) {
      offsets[0] = (gridDim.y * y_t + blockIdx.y) * stride20 + row_offset0;
      offsets[1] = (gridDim.y * y_t + blockIdx.y) * stride21 + row_offset1;
      auto p0 = reinterpret_cast<res_t*>(data0 + offsets[0]);
      auto p1 = reinterpret_cast<arg0_t*>(data1 + offsets[1]);
      auto p2 = reinterpret_cast<arg1_t*>(data2 + offsets[2]);
      *p0 = f(*p1, *p2);
    }
  }
}

template<int nt, int v_x, typename res_t , typename arg0_t, typename arg1_t, typename func_t, typename index_t, typename stride_t>
C10_LAUNCH_BOUNDS_2(nt, 4)
__global__ void elementwise_kernel_3_2_broadcast_dim1_uncontiguous(
    int64_t N,
    char* data0, char* data1, char* data2,
    index_t size0, index_t size1, index_t size2,
    stride_t stride00, stride_t stride01, stride_t stride02,
    stride_t stride10, stride_t stride11, stride_t stride12,
    stride_t stride20, stride_t stride21, stride_t stride22,
    func_t f, int y_t, int z_t, int y_remain) {
  // ndim = 3, arity = 2, narg = 3
  // load type arg0_dtype_t, compute type arg0_t, store type res_dtype_t
  // default arg1 is the broadcast
  int tid = threadIdx.x;
  int64_t offsets[3];
  constexpr int NARGS = 3;
  constexpr int MAX_DIMS = 2;
  size_t x_remain = (size0 * z_t) % (blockDim.x * v_x);
  size_t tail = (size0 * z_t) % v_x;

  using StoreT = at::native::memory::aligned_vector<res_t, v_x>;
  #pragma unroll
  for (int arg = 0; arg < NARGS; arg++) {
    offsets[arg] = 0;
  }
  // arg = 0: output; arg = 1: input1; arg = 2: input2;

  res_t ld_out[v_x];
  arg1_t ld_2[v_x];

  StoreT *p_ld_out = reinterpret_cast<StoreT*>(&ld_out);

  auto z_offset_div = tid * v_x / size0;
  auto z_offset_mod = tid * v_x % size0;
  offsets[2] = blockIdx.x * blockDim.x * v_x * stride02 + z_offset_mod * stride02 + blockIdx.z * stride22 * z_t + z_offset_div * stride22;
  size_t row_offset0 = blockIdx.x * blockDim.x * v_x * stride00 + z_offset_mod * stride00 + blockIdx.z * stride20 * z_t + z_offset_div * stride20;
  size_t row_offset1 = blockIdx.x * blockDim.x * v_x * stride01 + z_offset_mod * stride01 + blockIdx.z * stride21 * z_t + z_offset_div * stride21;
  // cache arg1
  #pragma unroll
  for (int i = 0; i < v_x; i++) {
    ld_2[i] = *reinterpret_cast<arg1_t*>(data2 + offsets[2] + i * stride02);
  }
  for (size_t y_idx = 0; y_idx < y_t; y_idx++) {
    offsets[0] = (blockIdx.y * y_t + y_idx) * stride10 + row_offset0;
    offsets[1] = (blockIdx.y * y_t + y_idx) * stride11 + row_offset1;
    // vec store output
    StoreT* out = reinterpret_cast<StoreT*>(data0 + offsets[0]);
    auto ld_1 = *reinterpret_cast<arg0_t*>(data1 + offsets[1]);
    #pragma unroll
    for (int i = 0; i < v_x; i++) {
      // load arg0
      ld_1 = *reinterpret_cast<arg0_t*>(data1 + offsets[1] + i * stride01);
      ld_out[i] = f(ld_1, ld_2[i]);
    }
    *out = *p_ld_out;
  }
  if (y_remain != 0 && blockIdx.y < y_remain) {
    size_t row_offset0 = blockIdx.x * blockDim.x * v_x * stride00 + z_offset_mod * stride00 + blockIdx.z * stride20 * z_t + z_offset_div * stride20;
    size_t row_offset1 = blockIdx.x * blockDim.x * v_x * stride01 + z_offset_mod * stride01 + blockIdx.z * stride21 * z_t + z_offset_div * stride21;
    offsets[0] = (gridDim.y * y_t + blockIdx.y) * stride10 + row_offset0;
    offsets[1] = (gridDim.y * y_t + blockIdx.y) * stride11 + row_offset1;
    StoreT* out = reinterpret_cast<StoreT*>(data0 + offsets[0]);
    auto ld_1 = *reinterpret_cast<arg0_t*>(data1 + offsets[1]);
    #pragma unroll
    for (int i = 0; i < v_x; i++) {
      // load arg0
      ld_1 = *reinterpret_cast<arg0_t*>(data1 + offsets[1] + i * stride01);
      ld_out[i] = f(ld_1, ld_2[i]);
    }
    *out = *p_ld_out;
  }

  if (x_remain != 0 && (tid + 1) * v_x <= x_remain) {
    offsets[2] = gridDim.x * blockDim.x * v_x * stride02 + z_offset_mod * stride02 + blockIdx.z * stride22 * z_t + z_offset_div * stride22;
    // cache arg1
    #pragma unroll
    for (int i = 0; i < v_x; i++) {
      ld_2[i] = *reinterpret_cast<arg1_t*>(data2 + offsets[2] + i * stride02);
    }
    size_t row_offset0 = gridDim.x * blockDim.x * v_x * stride00 + z_offset_mod * stride00 + blockIdx.z * stride20 * z_t + z_offset_div * stride20;
    size_t row_offset1 = gridDim.x * blockDim.x * v_x * stride01 + z_offset_mod * stride01 + blockIdx.z * stride21 * z_t + z_offset_div * stride21;
    for (int y_idx = 0; y_idx < y_t; y_idx++) {
      offsets[0] = (blockIdx.y * y_t + y_idx) * stride10 + row_offset0;
      offsets[1] = (blockIdx.y * y_t + y_idx) * stride11 + row_offset1;
      StoreT* out = reinterpret_cast<StoreT*>(data0 + offsets[0]);
      auto ld_1 = *reinterpret_cast<arg0_t*>(data1 + offsets[1]);
      #pragma unroll
      for (int i = 0; i < v_x; i++) {
        // load arg0
        ld_1 = *reinterpret_cast<arg0_t*>(data1 + offsets[1] + i * stride01);
        ld_out[i] = f(ld_1, ld_2[i]);
      }
      *out = *p_ld_out;
    }
    if (y_remain != 0 && blockIdx.y < y_remain) {
      offsets[0] = (gridDim.y * y_t + blockIdx.y) * stride10 + row_offset0;
      offsets[1] = (gridDim.y * y_t + blockIdx.y) * stride11 + row_offset1;
      StoreT* out = reinterpret_cast<StoreT*>(data0 + offsets[0]);
      auto ld_1 = *reinterpret_cast<arg0_t*>(data1 + offsets[1]);
      #pragma unroll
      for (int i = 0; i < v_x; i++) {
        // load arg0
        ld_1 = *reinterpret_cast<arg0_t*>(data1 + offsets[1] + i * stride01);
        ld_out[i] = f(ld_1, ld_2[i]);
      }
      *out = *p_ld_out;
    }
  }

  // handle tail columns( < v_x)
  if(tail && tid < tail){
    auto remain_offset = x_remain / v_x;
    offsets[2] = (gridDim.x * blockDim.x + remain_offset) * v_x * stride02 + z_offset_mod * stride02 + blockIdx.z * stride22 * z_t + z_offset_div * stride22;
    int64_t row_offset0 = (gridDim.x  * blockDim.x + remain_offset) * v_x * stride00 + z_offset_mod * stride00 + blockIdx.z * stride20 * z_t + z_offset_div * stride20;
    int64_t row_offset1 = (gridDim.x  * blockDim.x + remain_offset) * v_x * stride01 + z_offset_mod * stride01 + blockIdx.z * stride21 * z_t + z_offset_div * stride21;
    for (int y_idx = 0; y_idx < y_t; y_idx++) {
      offsets[0] = (blockIdx.y * y_t + y_idx) * stride10 + row_offset0;
      offsets[1] = (blockIdx.y * y_t + y_idx) * stride11 + row_offset1;
      auto p0 = reinterpret_cast<res_t*>(data0 + offsets[0]);
      auto p1 = reinterpret_cast<arg0_t*>(data1 + offsets[1]);
      auto p2 = reinterpret_cast<arg1_t*>(data2 + offsets[2]);
      *p0 = f(*p1, *p2);
    }
    if (y_remain != 0 && blockIdx.y < y_remain) {
      offsets[0] = (gridDim.y * y_t + blockIdx.y) * stride10 + row_offset0;
      offsets[1] = (gridDim.y * y_t + blockIdx.y) * stride11 + row_offset1;
      auto p0 = reinterpret_cast<res_t*>(data0 + offsets[0]);
      auto p1 = reinterpret_cast<arg0_t*>(data1 + offsets[1]);
      auto p2 = reinterpret_cast<arg1_t*>(data2 + offsets[2]);
      *p0 = f(*p1, *p2);
    }
  }
}

template<int nt, int v_x, typename res_t , typename arg0_t, typename arg1_t, typename func_t, typename index_t, typename stride_t, bool swap_tensor=false>
C10_LAUNCH_BOUNDS_2(nt, 4)
__global__ void elementwise_kernel_3_2_broadcast_dim1_contiguous(
    int64_t N,
    char* data0, char* data1, char* data2,
    index_t size0, index_t size1, index_t size2,
    stride_t stride00, stride_t stride01, stride_t stride02,
    stride_t stride10, stride_t stride11, stride_t stride12,
    stride_t stride20, stride_t stride21, stride_t stride22,
    func_t f, int y_t, int y_remain) {
  // ndim = 3, arity = 2, narg = 3
  // load type arg0_dtype_t, compute type arg0_t, store type res_dtype_t
  // default arg1 is the broadcast
  int tid = threadIdx.x;
  int64_t offsets[3];
  constexpr int NARGS = 3;
  constexpr int MAX_DIMS = 2;
  size_t x_remain;
  size_t tail;
  if (size0 >= blockDim.x) {
    x_remain = size0 % (blockDim.x * v_x);
    tail = size0 % v_x;
  }
  else {  // deal  with size0 < blockDim.x (64)
    x_remain = 0;
    tail = 0;
  }


  using LoadT1 = at::native::memory::aligned_vector<arg0_t, v_x>;
  using LoadT2 = at::native::memory::aligned_vector<arg1_t, v_x>;
  using StoreT = at::native::memory::aligned_vector<res_t, v_x>;
  #pragma unroll
  for (int arg = 0; arg < NARGS; arg++) {
    offsets[arg] = 0;
  }
  // arg = 0: output; arg = 1: input1; arg = 2: input2;

  arg0_t ld_1[v_x];
  arg1_t ld_2[v_x];
  res_t ld_out[v_x];
  LoadT1* p_ld_1 = reinterpret_cast<LoadT1*>(&ld_1);
  LoadT2* p_ld_2 = reinterpret_cast<LoadT2*>(&ld_2);
  StoreT * p_ld_out = reinterpret_cast<StoreT*>(&ld_out);

  if ((blockIdx.x * blockDim.x + tid) * v_x < size0) {
    offsets[2] = (blockIdx.x * blockDim.x + tid) * v_x * stride02 + blockIdx.z * stride22;
    size_t row_offset0 = (blockIdx.x * blockDim.x + tid) * v_x * stride00 + blockIdx.z * stride20;
    size_t row_offset1 = (blockIdx.x * blockDim.x + tid) * v_x * stride01 + blockIdx.z * stride21;
    *p_ld_2 = *reinterpret_cast<LoadT2*>(data2 + offsets[2]);
    for (size_t y_idx = 0; y_idx < y_t; y_idx++) {
      offsets[0] = (blockIdx.y * y_t + y_idx) * stride10 + row_offset0;
      // the layout of out is the same as arg0
      offsets[1] = (blockIdx.y * y_t + y_idx) * stride11 + row_offset1;
      *p_ld_1 = *reinterpret_cast<LoadT1*>(data1 + offsets[1]);
      StoreT* out = reinterpret_cast<StoreT*>(data0 + offsets[0]);
      #pragma unroll
      for (int i = 0; i < v_x; i++) {
        if (swap_tensor) {
          ld_out[i] = f(ld_2[i], ld_1[i]);
        } else {
          ld_out[i] = f(ld_1[i], ld_2[i]);
        }
      }
      *out = *p_ld_out;
    }
    // y_remain = size1 % grid_dim_y so y_remain < grid_dim_y, blockIdx.y can handle all y_remain
    if (y_remain != 0 && blockIdx.y < y_remain) {
      size_t row_offset0 = (blockIdx.x * blockDim.x + tid) * v_x * stride00 + blockIdx.z * stride20;
      size_t row_offset1 = (blockIdx.x * blockDim.x + tid) * v_x * stride01 + blockIdx.z * stride21;
      offsets[0] = (gridDim.y * y_t + blockIdx.y) * stride10 + row_offset0;
      offsets[1] = (gridDim.y * y_t + blockIdx.y) * stride11 + row_offset1;
      *p_ld_1 = *reinterpret_cast<LoadT1*>(data1 + offsets[1]);
      StoreT* out = reinterpret_cast<StoreT*>(data0 + offsets[0]);
      #pragma unroll
      for (int i = 0; i < v_x; i++) {
        if (swap_tensor) {
          ld_out[i] = f(ld_2[i], ld_1[i]);
        } else {
          ld_out[i] = f(ld_1[i], ld_2[i]);
        }
      }
      *out = *p_ld_out;
    }
  }

  if (x_remain != 0 && (tid + 1) * v_x <= x_remain) {
    offsets[2] = (gridDim.x * blockDim.x + tid) * v_x * stride02 + blockIdx.z * stride22;
    *p_ld_2 = *reinterpret_cast<LoadT2*>(data2 + offsets[2]);
    size_t row_offset0 = (gridDim.x * blockDim.x + tid) * v_x * stride00 + blockIdx.z * stride20;
    size_t row_offset1 = (gridDim.x * blockDim.x + tid) * v_x * stride01 + blockIdx.z * stride21;
    for (int y_idx = 0; y_idx < y_t; y_idx++) {
      offsets[0] = (blockIdx.y * y_t + y_idx) * stride10 + row_offset0;
      offsets[1] = (blockIdx.y * y_t + y_idx) * stride11 + row_offset1;
      *p_ld_1 = *reinterpret_cast<LoadT1*>(data1 + offsets[1]);
      StoreT* out = reinterpret_cast<StoreT*>(data0 + offsets[0]);
      #pragma unroll
      for (int i = 0; i < v_x; i++) {
        if (swap_tensor) {
          ld_out[i] = f(ld_2[i], ld_1[i]);
        } else {
          ld_out[i] = f(ld_1[i], ld_2[i]);
        }
      }
      *out = *p_ld_out;
    }
    if (y_remain != 0 && blockIdx.y < y_remain) {
      offsets[0] = (gridDim.y * y_t + blockIdx.y) * stride10 + row_offset0;
      offsets[1] = (gridDim.y * y_t + blockIdx.y) * stride11 + row_offset1;
      *p_ld_1 = *reinterpret_cast<LoadT1*>(data1 + offsets[1]);
      StoreT* out = reinterpret_cast<StoreT*>(data0 + offsets[0]);
      #pragma unroll
      for (int i = 0; i < v_x; i++) {
        if (swap_tensor) {
          ld_out[i] = f(ld_2[i], ld_1[i]);
        } else {
          ld_out[i] = f(ld_1[i], ld_2[i]);
        }
      }
      *out = *p_ld_out;
    }
  }

  // handle tail columns( < v_x)
  if(tail && tid < tail){
    auto remain_offset = x_remain / v_x;
    offsets[2] = ((gridDim.x * blockDim.x + remain_offset) * v_x + tid)* stride02 + blockIdx.z * stride22;
    int64_t row_offset0 = ((gridDim.x  * blockDim.x + remain_offset) * v_x + tid) * stride00 + blockIdx.z * stride20;
    int64_t row_offset1 = ((gridDim.x  * blockDim.x + remain_offset) * v_x + tid) * stride01 + blockIdx.z * stride21;
    for (int y_idx = 0; y_idx < y_t; y_idx++) {
      offsets[0] = (blockIdx.y * y_t + y_idx) * stride10 + row_offset0;
      offsets[1] = (blockIdx.y * y_t + y_idx) * stride11 + row_offset1;
      auto p0 = reinterpret_cast<res_t*>(data0 + offsets[0]);
      auto p1 = reinterpret_cast<arg0_t*>(data1 + offsets[1]);
      auto p2 = reinterpret_cast<arg1_t*>(data2 + offsets[2]);
      if (swap_tensor){
        *p0 = f(*p2, *p1);
      } else {
        *p0 = f(*p1, *p2);
      }
    }
    if (y_remain != 0 && blockIdx.y < y_remain) {
      offsets[0] = (gridDim.y * y_t + blockIdx.y) * stride10 + row_offset0;
      offsets[1] = (gridDim.y * y_t + blockIdx.y) * stride11 + row_offset1;
      auto p0 = reinterpret_cast<res_t*>(data0 + offsets[0]);
      auto p1 = reinterpret_cast<arg0_t*>(data1 + offsets[1]);
      auto p2 = reinterpret_cast<arg1_t*>(data2 + offsets[2]);
      if (swap_tensor){
        *p0 = f(*p2, *p1);
      } else {
        *p0 = f(*p1, *p2);
      }
    }
  }
}

template<int nt, int v_x, typename res_t , typename arg0_t, typename arg1_t, typename func_t, typename index_t, typename stride_t>
C10_LAUNCH_BOUNDS_2(nt, 4)
__global__ void elementwise_kernel_3_2_broadcast_dim1_contiguous_s(
    int64_t N,
    char* data0, char* data1, char* data2,
    index_t size0, index_t size1, index_t size2,
    stride_t stride00, stride_t stride01, stride_t stride02,
    stride_t stride10, stride_t stride11, stride_t stride12,
    stride_t stride20, stride_t stride21, stride_t stride22,
    func_t f, int y_t, int z_t, int y_remain, int z_remain) {
  // ndim = 3, arity = 2, narg = 3
  // load type arg0_dtype_t, compute type arg0_t, store type res_dtype_t
  // default arg1 is the broadcast
  int tid = threadIdx.x;
  int64_t offsets[3];
  constexpr int NARGS = 3;
  constexpr int MAX_DIMS = 2;
  size_t x_remain = (size0 * z_t) % (blockDim.x * v_x);
  size_t tail = (size0 * z_t) % v_x;

  using LoadT1 = at::native::memory::aligned_vector<arg0_t, v_x>;
  using LoadT2 = at::native::memory::aligned_vector<arg1_t, v_x>;
  using StoreT = at::native::memory::aligned_vector<res_t, v_x>;
  #pragma unroll
  for (int arg = 0; arg < NARGS; arg++) {
    offsets[arg] = 0;
  }
  // arg = 0: output; arg = 1: input1; arg = 2: input2;

  arg0_t ld_1[v_x];
  arg1_t ld_2[v_x];
  res_t ld_out[v_x];
  LoadT1* p_ld_1 = reinterpret_cast<LoadT1*>(&ld_1);
  LoadT2* p_ld_2 = reinterpret_cast<LoadT2*>(&ld_2);
  StoreT * p_ld_out = reinterpret_cast<StoreT*>(&ld_out);

  auto z_offset_div = tid * v_x / size0;
  auto z_offset_mod = tid * v_x % size0;

  offsets[2] = blockIdx.x * blockDim.x * v_x * stride02 + z_offset_mod * stride02 + blockIdx.z * stride22 * z_t + z_offset_div * stride22;
  size_t row_offset0 = blockIdx.x * blockDim.x * v_x * stride00 + z_offset_mod * stride00 + blockIdx.z * stride20 * z_t + z_offset_div * stride20;
  size_t row_offset1 = blockIdx.x * blockDim.x * v_x * stride01 + z_offset_mod * stride01 + blockIdx.z * stride21 * z_t + z_offset_div * stride21;
  *p_ld_2 = *reinterpret_cast<LoadT2*>(data2 + offsets[2]);
  for (size_t y_idx = 0; y_idx < y_t; y_idx++) {
    offsets[0] = (blockIdx.y * y_t + y_idx) * stride10 + row_offset0;
    // the layout of out is the same as arg0
    offsets[1] = (blockIdx.y * y_t + y_idx) * stride11 + row_offset1;
    *p_ld_1 = *reinterpret_cast<LoadT1*>(data1 + offsets[1]);
    StoreT* out = reinterpret_cast<StoreT*>(data0 + offsets[0]);
    #pragma unroll
    for (int i = 0; i < v_x; i++) {
      ld_out[i] = f(ld_1[i], ld_2[i]);
    }
    *out = *p_ld_out;
  }
  // y_remain = size1 % grid_dim_y so y_remain < grid_dim_y, blockIdx.y can handle all y_remain
  if (y_remain != 0 && blockIdx.y < y_remain) {
    size_t row_offset0 = blockIdx.x * blockDim.x * v_x * stride00 + z_offset_mod * stride00 + blockIdx.z * stride20 * z_t + z_offset_div * stride20;
    size_t row_offset1 = blockIdx.x * blockDim.x * v_x * stride01 + z_offset_mod * stride01 + blockIdx.z * stride21 * z_t + z_offset_div * stride21;
    offsets[0] = (gridDim.y * y_t + blockIdx.y) * stride10 + row_offset0;
    offsets[1] = (gridDim.y * y_t + blockIdx.y) * stride11 + row_offset1;
    *p_ld_1 = *reinterpret_cast<LoadT1*>(data1 + offsets[1]);
    StoreT* out = reinterpret_cast<StoreT*>(data0 + offsets[0]);
    #pragma unroll
    for (int i = 0; i < v_x; i++) {
      ld_out[i] = f(ld_1[i], ld_2[i]);
    }
    *out = *p_ld_out;
  }

  if (z_remain != 0 && blockIdx.z == 0 && z_offset_div < z_remain) {
    auto offset = z_t * gridDim.z;
    offsets[2] = blockIdx.x * blockDim.x * v_x * stride02 + z_offset_mod * stride02 + blockIdx.z * stride22 * z_t + z_offset_div * stride22 + offset * stride22;
    size_t row_offset0 = blockIdx.x * blockDim.x * v_x * stride00 + z_offset_mod * stride00 + blockIdx.z * stride20 * z_t + z_offset_div * stride20 + offset * stride20;
    size_t row_offset1 = blockIdx.x * blockDim.x * v_x * stride01 + z_offset_mod * stride01 + blockIdx.z * stride21 * z_t + z_offset_div * stride21 + offset * stride21;
    *p_ld_2 = *reinterpret_cast<LoadT2*>(data2 + offsets[2]);
    for (size_t y_idx = 0; y_idx < y_t; y_idx++) {
      offsets[0] = (blockIdx.y * y_t + y_idx) * stride10 + row_offset0;
      // the layout of out is the same as arg0
      offsets[1] = (blockIdx.y * y_t + y_idx) * stride11 + row_offset1;
      *p_ld_1 = *reinterpret_cast<LoadT1*>(data1 + offsets[1]);
      StoreT* out = reinterpret_cast<StoreT*>(data0 + offsets[0]);
      #pragma unroll
      for (int i = 0; i < v_x; i++) {
        ld_out[i] = f(ld_1[i], ld_2[i]);
      }
      *out = *p_ld_out;
    }
    // y_remain = size1 % grid_dim_y so y_remain < grid_dim_y, blockIdx.y can handle all y_remain
    if (y_remain != 0 && blockIdx.y < y_remain) {
      size_t row_offset0 = blockIdx.x * blockDim.x * v_x * stride00 + z_offset_mod * stride00 + blockIdx.z * stride20 * z_t + z_offset_div * stride20 + offset * stride20;
      size_t row_offset1 = blockIdx.x * blockDim.x * v_x * stride01 + z_offset_mod * stride01 + blockIdx.z * stride21 * z_t + z_offset_div * stride21 + offset * stride21;
      offsets[0] = (gridDim.y * y_t + blockIdx.y) * stride10 + row_offset0;
      offsets[1] = (gridDim.y * y_t + blockIdx.y) * stride11 + row_offset1;
      *p_ld_1 = *reinterpret_cast<LoadT1*>(data1 + offsets[1]);
      StoreT* out = reinterpret_cast<StoreT*>(data0 + offsets[0]);
      #pragma unroll
      for (int i = 0; i < v_x; i++) {
        ld_out[i] = f(ld_1[i], ld_2[i]);
      }
      *out = *p_ld_out;
    }
  }
}

template<int nt, int v_x, typename res_t , typename arg0_t, typename arg1_t, typename func_t, typename index_t, typename stride_t>
C10_LAUNCH_BOUNDS_2(nt, 4)
__global__ void elementwise_kernel_3_2_broadcast_dim2_contiguous_s(
    int64_t N,
    char* data0, char* data1, char* data2,
    index_t size0, index_t size1, index_t size2,
    stride_t stride00, stride_t stride01, stride_t stride02,
    stride_t stride10, stride_t stride11, stride_t stride12,
    stride_t stride20, stride_t stride21, stride_t stride22,
    func_t f, int y_t, int z_t, int y_remain, int z_remain) {
  // ndim = 3, arity = 2, narg = 3
  // load type arg0_dtype_t, compute type arg0_t, store type res_dtype_t
  // default arg1 is the broadcast
  int tid = threadIdx.x;
  int64_t offsets[3];
  constexpr int NARGS = 3;
  constexpr int MAX_DIMS = 2;
  size_t x_remain = (size0 * z_t) % (blockDim.x * v_x);
  size_t tail = (size0 * z_t) % v_x;

  using LoadT1 = at::native::memory::aligned_vector<arg0_t, v_x>;
  using LoadT2 = at::native::memory::aligned_vector<arg1_t, v_x>;
  using StoreT = at::native::memory::aligned_vector<res_t, v_x>;
  #pragma unroll
  for (int arg = 0; arg < NARGS; arg++) {
    offsets[arg] = 0;
  }
  // arg = 0: output; arg = 1: input1; arg = 2: input2;

  arg0_t ld_1[v_x];
  arg1_t ld_2[v_x];
  res_t ld_out[v_x];
  LoadT1* p_ld_1 = reinterpret_cast<LoadT1*>(&ld_1);
  LoadT2* p_ld_2 = reinterpret_cast<LoadT2*>(&ld_2);
  StoreT * p_ld_out = reinterpret_cast<StoreT*>(&ld_out);

  auto z_offset_div = tid * v_x / size0;
  auto z_offset_mod = tid * v_x % size0;

  offsets[2] = blockIdx.x * blockDim.x * v_x * stride02 + z_offset_mod * stride02 + blockIdx.z * stride12 * z_t + z_offset_div * stride12;
  size_t row_offset0 = blockIdx.x * blockDim.x * v_x * stride00 + z_offset_mod * stride00 + blockIdx.z * stride10 * z_t + z_offset_div * stride10;
  size_t row_offset1 = blockIdx.x * blockDim.x * v_x * stride01 + z_offset_mod * stride01 + blockIdx.z * stride11 * z_t + z_offset_div * stride11;
  *p_ld_2 = *reinterpret_cast<LoadT2*>(data2 + offsets[2]);
  for (size_t y_idx = 0; y_idx < y_t; y_idx++) {
    offsets[0] = (blockIdx.y * y_t + y_idx) * stride20 + row_offset0;
    // the layout of out is the same as arg0
    offsets[1] = (blockIdx.y * y_t + y_idx) * stride21 + row_offset1;
    *p_ld_1 = *reinterpret_cast<LoadT1*>(data1 + offsets[1]);
    StoreT* out = reinterpret_cast<StoreT*>(data0 + offsets[0]);
    #pragma unroll
    for (int i = 0; i < v_x; i++) {
      ld_out[i] = f(ld_1[i], ld_2[i]);
    }
    *out = *p_ld_out;
  }
  // y_remain = size1 % grid_dim_y so y_remain < grid_dim_y, blockIdx.y can handle all y_remain
  if (y_remain != 0 && blockIdx.y < y_remain) {
    size_t row_offset0 = blockIdx.x * blockDim.x * v_x * stride00 + z_offset_mod * stride00 + blockIdx.z * stride10 * z_t + z_offset_div * stride10;
    size_t row_offset1 = blockIdx.x * blockDim.x * v_x * stride01 + z_offset_mod * stride01 + blockIdx.z * stride11 * z_t + z_offset_div * stride11;
    offsets[0] = (gridDim.y * y_t + blockIdx.y) * stride20 + row_offset0;
    offsets[1] = (gridDim.y * y_t + blockIdx.y) * stride21 + row_offset1;
    *p_ld_1 = *reinterpret_cast<LoadT1*>(data1 + offsets[1]);
    StoreT* out = reinterpret_cast<StoreT*>(data0 + offsets[0]);
    #pragma unroll
    for (int i = 0; i < v_x; i++) {
      ld_out[i] = f(ld_1[i], ld_2[i]);
    }
    *out = *p_ld_out;
  }

  if (z_remain != 0 && blockIdx.z == 0 && z_offset_div < z_remain) {
    auto offset = z_t * gridDim.z;
    offsets[2] = blockIdx.x * blockDim.x * v_x * stride02 + z_offset_mod * stride02 + blockIdx.z * stride12 * z_t + z_offset_div * stride12 + offset * stride12;
    size_t row_offset0 = blockIdx.x * blockDim.x * v_x * stride00 + z_offset_mod * stride00 + blockIdx.z * stride10 * z_t + z_offset_div * stride10 + offset * stride10;
    size_t row_offset1 = blockIdx.x * blockDim.x * v_x * stride01 + z_offset_mod * stride01 + blockIdx.z * stride11 * z_t + z_offset_div * stride11 + offset * stride11;
    *p_ld_2 = *reinterpret_cast<LoadT2*>(data2 + offsets[2]);
    for (size_t y_idx = 0; y_idx < y_t; y_idx++) {
      offsets[0] = (blockIdx.y * y_t + y_idx) * stride20 + row_offset0;
      // the layout of out is the same as arg0
      offsets[1] = (blockIdx.y * y_t + y_idx) * stride21 + row_offset1;
      *p_ld_1 = *reinterpret_cast<LoadT1*>(data1 + offsets[1]);
      StoreT* out = reinterpret_cast<StoreT*>(data0 + offsets[0]);
      #pragma unroll
      for (int i = 0; i < v_x; i++) {
        ld_out[i] = f(ld_1[i], ld_2[i]);
      }
      *out = *p_ld_out;
    }
    // y_remain = size1 % grid_dim_y so y_remain < grid_dim_y, blockIdx.y can handle all y_remain
    if (y_remain != 0 && blockIdx.y < y_remain) {
      size_t row_offset0 = blockIdx.x * blockDim.x * v_x * stride00 + z_offset_mod * stride00 + blockIdx.z * stride10 * z_t + z_offset_div * stride10 + offset * stride10;
      size_t row_offset1 = blockIdx.x * blockDim.x * v_x * stride01 + z_offset_mod * stride01 + blockIdx.z * stride11 * z_t + z_offset_div * stride11 + offset * stride11;
      offsets[0] = (gridDim.y * y_t + blockIdx.y) * stride20 + row_offset0;
      offsets[1] = (gridDim.y * y_t + blockIdx.y) * stride21 + row_offset1;
      *p_ld_1 = *reinterpret_cast<LoadT1*>(data1 + offsets[1]);
      StoreT* out = reinterpret_cast<StoreT*>(data0 + offsets[0]);
      #pragma unroll
      for (int i = 0; i < v_x; i++) {
        ld_out[i] = f(ld_1[i], ld_2[i]);
      }
      *out = *p_ld_out;
    }
  }
}

template<int nt, int vt, typename res_t , typename arg0_t, typename arg1_t, typename func_t, typename index_t, typename stride_t, int size0, int size1, int size2>
C10_LAUNCH_BOUNDS_2(nt, 4)
__global__ void elementwise_kernel_3_2_s(
    int64_t N,
    char* data0, char* data1, char* data2,
    stride_t stride00, stride_t stride01, stride_t stride02,
    stride_t stride10, stride_t stride11, stride_t stride12,
    stride_t stride20, stride_t stride21, stride_t stride22,
    func_t f) {
  // ndim = 3, arity = 2, narg = 3
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
      int64_t offsets[3];
      auto linear_idx = idx;
      constexpr int NARGS = 3;
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
      arg = 2;
      offsets[arg] += divmod_mod * stride02;
      // dim = 1
      divmod_div = linear_idx / size1;
      divmod_mod = linear_idx % size1;
      linear_idx = divmod_div;
      arg = 0;
      offsets[arg] += divmod_mod * stride10;
      arg = 1;
      offsets[arg] += divmod_mod * stride11;
      arg = 2;
      offsets[arg] += divmod_mod * stride12;
      // dim = 2
      divmod_div = linear_idx / size2;
      divmod_mod = linear_idx % size2;
      linear_idx = divmod_div;
      arg = 0;
      offsets[arg] += divmod_mod * stride20;
      arg = 1;
      offsets[arg] += divmod_mod * stride21;
      arg = 2;
      offsets[arg] += divmod_mod * stride22;

      res_t* out = (res_t*)(data0 + offsets[0]);
      *out = f(func_reinterpret_cast<arg0_t>(data1 + offsets[1]),
               func_reinterpret_cast<arg1_t>(data2 + offsets[2]));

      idx += nt;
    }
  }
}

template<int nt, int vt, typename res_t , typename arg0_t, typename arg1_t, typename func_t, typename index_t, typename stride_t>
static void launch_legacy_kernel_maca_1_2(
    int64_t N,
    char* data0, char* data1, char* data2,
    index_t size0,
    stride_t stride00, stride_t stride01, stride_t stride02,
    const func_t& f) {
  // ndim = 1, arity = 2
  TORCH_INTERNAL_ASSERT(N >= 0 && N <= std::numeric_limits<int32_t>::max());
  if (N == 0) {
    return;
  }
  dim3 block(nt);
  dim3 grid((N + block.x * vt - 1) / (block.x * vt));
  auto stream = at::cuda::getCurrentCUDAStream();
  elementwise_kernel_1_2<nt, vt, res_t, arg0_t, arg1_t, func_t, index_t, stride_t><<<grid, block, 0, stream>>>(
      N, data0, data1, data2, size0, stride00, stride01, stride02, f);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template<int nt, int vt, typename res_t , typename arg0_t, typename arg1_t, typename func_t, typename index_t, typename stride_t>
static void launch_legacy_kernel_maca_1_2_broadcast(
    int64_t N,
    char* data0, char* data1, char* data2,
    index_t size0,
    stride_t stride00, stride_t stride01, stride_t stride02,
    const func_t& f) {
  // ndim = 1, arity = 2, stride02 = 0 or stride01 = 0
  TORCH_INTERNAL_ASSERT(N >= 0 && N <= std::numeric_limits<int32_t>::max());
  if (N == 0) {
    return;
  }
  auto stream = at::cuda::getCurrentCUDAStream();
  const int block_dim_x = 64;
  size_t load_num = size0 / block_dim_x;
  TORCH_INTERNAL_ASSERT(load_num > 0);
  int vec_data_0 = getVectorizedAlignment<res_t>((void*)data0, load_num);
  if (std::is_same<res_t, at::Half>::value || std::is_same<res_t, at::BFloat16>::value) {
    vec_data_0 = std::min(vec_data_0, getVectorizedAlignment<res_t>((void*)data0, size0));
  }
  int vec_data_1 = getVectorizedAlignment<arg0_t>((void*)data1, load_num);
  int vec_data_2 = getVectorizedAlignment<arg1_t>((void*)data2, load_num);
  int vec = vec_data_0;
  if (stride01 != 0) {
    vec = std::min(vec_data_0, vec_data_1);
  }
  if (stride02 != 0) {
    vec = std::min(vec_data_0, vec_data_2);
  }
  dim3 block(block_dim_x);
  int grid_dim_x = load_num / vec;
  TORCH_INTERNAL_ASSERT(grid_dim_x > 0);
  dim3 grid(grid_dim_x);
  if (stride02 == 0) {
    if (vec == 8) {
      elementwise_kernel_1_2_broadcast<nt, 8, res_t, arg0_t, arg1_t, func_t, index_t, stride_t><<<grid, block, 0, stream>>>(
        N, data0, data1, data2, size0, stride00, stride01, stride02, f);
    } else if (vec == 4) {
      elementwise_kernel_1_2_broadcast<nt, 4, res_t, arg0_t, arg1_t, func_t, index_t, stride_t><<<grid, block, 0, stream>>>(
        N, data0, data1, data2, size0, stride00, stride01, stride02, f);
    } else if (vec == 2) {
      elementwise_kernel_1_2_broadcast<nt, 2, res_t, arg0_t, arg1_t, func_t, index_t, stride_t><<<grid, block, 0, stream>>>(
        N, data0, data1, data2, size0, stride00, stride01, stride02, f);
    } else {
      elementwise_kernel_1_2_broadcast<nt, 1, res_t, arg0_t, arg1_t, func_t, index_t, stride_t><<<grid, block, 0, stream>>>(
        N, data0, data1, data2, size0, stride00, stride01, stride02, f);
    }
  } else if (stride01 == 0) {
    if (vec == 8) {
      elementwise_kernel_1_2_broadcast<nt, 8, res_t, arg1_t, arg0_t, func_t, index_t, stride_t, true><<<grid, block, 0, stream>>>(
        N, data0, data2, data1, size0, stride00, stride02, stride01, f);
    } else if (vec == 4) {
      elementwise_kernel_1_2_broadcast<nt, 4, res_t, arg1_t, arg0_t, func_t, index_t, stride_t, true><<<grid, block, 0, stream>>>(
        N, data0, data2, data1, size0, stride00, stride02, stride01, f);
    } else if (vec == 2) {
      elementwise_kernel_1_2_broadcast<nt, 2, res_t, arg1_t, arg0_t, func_t, index_t, stride_t, true><<<grid, block, 0, stream>>>(
        N, data0, data2, data1, size0, stride00, stride02, stride01, f);
    } else {
      elementwise_kernel_1_2_broadcast<nt, 1, res_t, arg1_t, arg0_t, func_t, index_t, stride_t, true><<<grid, block, 0, stream>>>(
        N, data0, data2, data1, size0, stride00, stride02, stride01, f);
    }
  } else {
    TORCH_CHECK(false, "elementwise kernel 1_2 but not broadcast!");
  }
}

template<int nt, int vt, typename res_t , typename arg0_t, typename arg1_t, typename func_t, typename index_t, typename stride_t>
static void launch_legacy_kernel_maca_2_2_broadcast_arity1_dim0(
    int64_t N,
    char* data0, char* data1, char* data2,
    index_t size0, index_t size1,
    stride_t stride00, stride_t stride01, stride_t stride02,
    stride_t stride10, stride_t stride11, stride_t stride12,
    const func_t& f) {
  // ndim = 2, arity = 2
  TORCH_INTERNAL_ASSERT(N >= 0 && N <= std::numeric_limits<int32_t>::max());
  if (N == 0) {
    return;
  }

  int vec = 4;
  if(sizeof(arg0_t) == 2 && size0 % 128 == 0) {
    vec = 8;
  }

  auto stream = at::cuda::getCurrentCUDAStream();
  dim3 block(nt);
  dim3 grid((N + block.x * vec - 1) / (block.x * vec));
  if (vec == 4) {
    elementwise_kernel_2_2_broadcast_arity1_dim0<nt, 4, res_t, arg0_t, arg1_t, func_t, index_t, stride_t><<<grid, block, 0, stream>>>(
        N, data0, data1, data2, size0, size1, stride00, stride01, stride02, stride10, stride11, stride12, f);
  } else if (vec == 8) {
    elementwise_kernel_2_2_broadcast_arity1_dim0<nt, 8, res_t, arg0_t, arg1_t, func_t, index_t, stride_t><<<grid, block, 0, stream>>>(
        N, data0, data1, data2, size0, size1, stride00, stride01, stride02, stride10, stride11, stride12, f);
  } else {
    assert(0);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}


template<int nt, int vt, typename res_t , typename arg0_t, typename arg1_t, typename func_t, typename index_t, typename stride_t>
static void launch_legacy_kernel_maca_2_2_arity2_tranpose(
    int64_t N,
    char* data0, char* data1, char* data2,
    index_t size0, index_t size1,
    stride_t stride00, stride_t stride01, stride_t stride02,
    stride_t stride10, stride_t stride11, stride_t stride12,
    const func_t& f) {
  TORCH_INTERNAL_ASSERT(N >= 0 && N <= std::numeric_limits<int32_t>::max());
  if (N == 0) {
    return;
  }
  index_t H = size0;
  index_t W = size1;
  auto stream = at::cuda::getCurrentCUDAStream();

  int64_t vec = 4;
  if (size0 % 8 == 0 && size1 % 8 == 0 && sizeof(res_t) == 2) {
    vec = 8;
  }

  if (vec == 8) {
    const uint32_t dh = (H + kTileDimMacaT - 1) / kTileDimMacaT;
    const uint32_t dw = (W + kTileDimMacaT - 1) / kTileDimMacaT;
    dim3 grid(dw, dh);
    dim3 block(kBlockRowsMacaT, kTileDimMacaT);
    elementwise_kernel_2_2_arity2_transpose<nt, 8, kTileDimMacaT, res_t, arg0_t, arg1_t, func_t, index_t, stride_t>
          <<<grid, block, 0, stream>>>
          (data0, data1, data2, (uint32_t)H, (uint32_t)W, (uint32_t)dh, (uint32_t)dw, f);
  } else if(vec == 4) {
    const uint32_t dh = (H + kTileDimMacaT_32 - 1) / kTileDimMacaT_32;
    const uint32_t dw = (W + kTileDimMacaT_32 - 1) / kTileDimMacaT_32;
    dim3 grid(dw, dh);
    dim3 block(kBlockRowsMacaT, kTileDimMacaT_32);
    elementwise_kernel_2_2_arity2_transpose<nt, 4, kTileDimMacaT_32, res_t, arg0_t, arg1_t, func_t, index_t, stride_t>
          <<<grid, block, 0, stream>>>
          (data0, data1, data2, (uint32_t)H, (uint32_t)W, (uint32_t)dh, (uint32_t)dw, f);
  } else {
    assert(0);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}


template<int nt, int vt, typename res_t , typename arg0_t, typename arg1_t, typename func_t, typename index_t, typename stride_t>
static void launch_legacy_kernel_maca_2_2(
    int64_t N,
    char* data0, char* data1, char* data2,
    index_t size0, index_t size1,
    stride_t stride00, stride_t stride01, stride_t stride02,
    stride_t stride10, stride_t stride11, stride_t stride12,
    const func_t& f) {
  // ndim = 2, arity = 2
  TORCH_INTERNAL_ASSERT(N >= 0 && N <= std::numeric_limits<int32_t>::max());
  if (N == 0) {
    return;
  }
  dim3 block(nt);
  dim3 grid((N + block.x * vt - 1) / (block.x * vt));
  auto stream = at::cuda::getCurrentCUDAStream();

  // 2_2 1280 128
  // 2_2 1280 512
  // 2_2 1280 8192
  // 2_2 160 2
  // 2_2 2560 2048
  // 2_2 320 8192
  // 2_2 4096 4
  // 2_2 512 4096
  // 2_2 5120 128
  // 2_2 5120 512
  // 2_2 5929 16
  // 2_2 640 2048
  bool env = at::maca::get_maca_disable_element_template_shape();
  if(maca_unlikely(env)){
    if (size0 == 1280 && size1 == 128) {
        elementwise_kernel_2_2_s<nt, vt, res_t, arg0_t, arg1_t, func_t, index_t, stride_t, 1280, 128><<<grid, block, 0, stream>>>(
            N, data0, data1, data2, stride00, stride01, stride02, stride10, stride11, stride12, f);
    } else if (size0 == 1280 && size1 == 512){
      elementwise_kernel_2_2_s<nt, vt, res_t, arg0_t, arg1_t, func_t, index_t, stride_t, 1280, 512><<<grid, block, 0, stream>>>(
          N, data0, data1, data2, stride00, stride01, stride02, stride10, stride11, stride12, f);
    } else if (size0 == 1280 && size1 == 8192){
      elementwise_kernel_2_2_s<nt, vt, res_t, arg0_t, arg1_t, func_t, index_t, stride_t, 1280, 8192><<<grid, block, 0, stream>>>(
          N, data0, data1, data2, stride00, stride01, stride02, stride10, stride11, stride12, f);
    } else if (size0 == 160 && size1 == 2){
      elementwise_kernel_2_2_s<nt, vt, res_t, arg0_t, arg1_t, func_t, index_t, stride_t, 160, 2><<<grid, block, 0, stream>>>(
          N, data0, data1, data2, stride00, stride01, stride02, stride10, stride11, stride12, f);
    } else if (size0 == 2560 && size1 == 2048){
      elementwise_kernel_2_2_s<nt, vt, res_t, arg0_t, arg1_t, func_t, index_t, stride_t, 2560, 2048><<<grid, block, 0, stream>>>(
          N, data0, data1, data2, stride00, stride01, stride02, stride10, stride11, stride12, f);
    } else if (size0 == 320 && size1 == 8192){
      elementwise_kernel_2_2_s<nt, vt, res_t, arg0_t, arg1_t, func_t, index_t, stride_t, 320, 8192><<<grid, block, 0, stream>>>(
          N, data0, data1, data2, stride00, stride01, stride02, stride10, stride11, stride12, f);
    } else if (size0 == 4096 && size1 == 4){
      elementwise_kernel_2_2_s<nt, vt, res_t, arg0_t, arg1_t, func_t, index_t, stride_t, 4096, 4><<<grid, block, 0, stream>>>(
          N, data0, data1, data2, stride00, stride01, stride02, stride10, stride11, stride12, f);
    } else if (size0 == 512 && size1 == 4096){
      elementwise_kernel_2_2_s<nt, vt, res_t, arg0_t, arg1_t, func_t, index_t, stride_t, 512, 4096><<<grid, block, 0, stream>>>(
          N, data0, data1, data2, stride00, stride01, stride02, stride10, stride11, stride12, f);
    } else if (size0 == 5120 && size1 == 128){
      elementwise_kernel_2_2_s<nt, vt, res_t, arg0_t, arg1_t, func_t, index_t, stride_t, 5120, 128><<<grid, block, 0, stream>>>(
          N, data0, data1, data2, stride00, stride01, stride02, stride10, stride11, stride12, f);
    } else if (size0 == 5120 && size1 == 512){
      elementwise_kernel_2_2_s<nt, vt, res_t, arg0_t, arg1_t, func_t, index_t, stride_t, 5120, 512><<<grid, block, 0, stream>>>(
          N, data0, data1, data2, stride00, stride01, stride02, stride10, stride11, stride12, f);
    } else if (size0 == 640 && size1 == 2048){
      elementwise_kernel_2_2_s<nt, vt, res_t, arg0_t, arg1_t, func_t, index_t, stride_t, 640, 2048><<<grid, block, 0, stream>>>(
          N, data0, data1, data2, stride00, stride01, stride02, stride10, stride11, stride12, f);
    } else if (size0 == 5929 && size1 == 16){
      elementwise_kernel_2_2_s<nt, vt, res_t, arg0_t, arg1_t, func_t, index_t, stride_t, 5929, 16><<<grid, block, 0, stream>>>(
          N, data0, data1, data2, stride00, stride01, stride02, stride10, stride11, stride12, f);
    } else {
      elementwise_kernel_2_2<nt, vt, res_t, arg0_t, arg1_t, func_t, index_t, stride_t><<<grid, block, 0, stream>>>(
          N, data0, data1, data2, size0, size1, stride00, stride01, stride02, stride10, stride11, stride12, f);
    }
  }else{
    elementwise_kernel_2_2<nt, vt, res_t, arg0_t, arg1_t, func_t, index_t, stride_t><<<grid, block, 0, stream>>>(
        N, data0, data1, data2, size0, size1, stride00, stride01, stride02, stride10, stride11, stride12, f);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template<int nt, int vt, typename res_t , typename arg0_t, typename arg1_t, typename func_t, typename index_t, typename stride_t>
static void launch_legacy_kernel_maca_2_2_template(
    int64_t N,
    char* data0, char* data1, char* data2,
    index_t size0, index_t size1,
    stride_t stride00, stride_t stride01, stride_t stride02,
    stride_t stride10, stride_t stride11, stride_t stride12,
    const func_t& f) {
  // ndim = 2, arity = 2
  TORCH_INTERNAL_ASSERT(N >= 0 && N <= std::numeric_limits<int32_t>::max());
  if (N == 0) {
    return;
  }
  bool output_can_vec = (check_vec_template<res_t>(stride10) && (stride00 == sizeof(res_t)));
  bool arg0_can_vec = (check_vec_template<arg0_t>(stride11) && (stride01 == sizeof(arg0_t)));
  bool arg1_can_vec = (check_vec_template<arg1_t>(stride12) && (stride02 == sizeof(arg1_t)));

  int vec_data_0 = getVectorizedAlignment<res_t>((void*)data0, size0);
  int vec_data_1 = getVectorizedAlignment<arg0_t>((void*)data1, size0);
  int vec_data_2 = getVectorizedAlignment<arg1_t>((void*)data2, size0);

  if (!output_can_vec) {
    vec_data_0 = 1;
  }
  if (!arg0_can_vec) {
    vec_data_1 = 1;
  }
  if (!arg1_can_vec) {
    vec_data_2 = 1;
  }

#define ELEMENTWISE_KERNEL_TEMPLATE(nt, vt, vt_0, vt_1, vec, res_t, arg0_t, arg1_t, func_t, index_t, stride_t) \
  elementwise_kernel_2_2_template<nt, vt, vt_0, vt_1, vec, res_t, arg0_t, arg1_t, func_t, index_t, stride_t>   \
    <<<grid, block, 0, stream>>>(N, data0, data1, data2, size0, size1,                                         \
      stride00, stride01, stride02, stride10, stride11, stride12, f);                                          \
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  int vec = std::max(vec_data_0, std::max(vec_data_1, vec_data_2));
  dim3 block(nt);
  dim3 grid((N + block.x * vec - 1) / (block.x * vec));
  auto stream = at::cuda::getCurrentCUDAStream();

  SWITCH_TEMPLATE_KERNEL_ARGS_3(nt, vec_data_0, vec_data_1, vec_data_2, vec, ELEMENTWISE_KERNEL_TEMPLATE,
                                res_t, arg0_t, arg1_t, func_t, index_t, stride_t);

#undef ELEMENTWISE_KERNEL_TEMPLATE
}

template<int nt, int vt, typename res_t, typename arg0_t, typename arg1_t, typename func_t, typename index_t, typename stride_t>
static void launch_legacy_kernel_maca_2_2_interval_arity1(
  int64_t N,
  char* data0, char* data1, char* data2,
  index_t size0, index_t size1,
  stride_t stride00, stride_t stride01, stride_t stride02,
  stride_t stride10, stride_t stride11, stride_t stride12,
  const func_t& f
){
 // ndim = 2, arity = 2
  TORCH_INTERNAL_ASSERT(N >= 0 && N <= std::numeric_limits<int32_t>::max());
  if (N == 0) {
    return;
  }
  dim3 block(nt);
  dim3 grid((N + block.x * vt - 1) / (block.x * vt));
  auto stream = at::cuda::getCurrentCUDAStream();

  elementwise_kernel_2_2_interval_arity1<nt, vt, res_t, arg0_t, arg1_t, func_t, index_t, stride_t><<<grid, block, 0, stream>>>(
        N, data0, data1, data2, size0, size1, stride00, stride01, stride02, stride10, stride11, stride12, f);

  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template<int nt, int vt, typename res_t , typename arg0_t, typename arg1_t, typename func_t, typename index_t, typename stride_t>
static void launch_legacy_kernel_maca_2_2_align(
    int64_t N,
    char* data0, char* data1, char* data2,
    index_t size0, index_t size1,
    stride_t stride00, stride_t stride01, stride_t stride02,
    stride_t stride10, stride_t stride11, stride_t stride12,
    const func_t& f) {
  // ndim = 2, arity = 2
  TORCH_INTERNAL_ASSERT(N >= 0 && N <= std::numeric_limits<int32_t>::max());
  if (N == 0) {
    return;
  }
  auto stream = at::cuda::getCurrentCUDAStream();
  constexpr int block_dim_x = 64;
  // as size0 is multiple of 8, so here we can set alignment_size=8
  int vec_data_0 = getVectorizedAlignment<res_t>((void*)data0, 8);
  // Notice: offsets required vectorized alignment when Half or BFloat16.
  if (std::is_same<res_t, at::Half>::value || std::is_same<res_t, at::BFloat16>::value) {
    vec_data_0 = std::min(vec_data_0, getVectorizedAlignment<res_t>((void*)data0, stride10/sizeof(res_t)));
  }
  int vec_data_1 = getVectorizedAlignment<arg0_t>((void*)data1, 8);
  if (std::is_same<arg0_t, at::Half>::value || std::is_same<arg0_t, at::BFloat16>::value) {
    vec_data_1 = std::min(vec_data_1, getVectorizedAlignment<arg0_t>((void*)data1, stride11/sizeof(arg0_t)));
  }
  int vec_data_2 = getVectorizedAlignment<arg1_t>((void*)data2, 8);
  int vec = std::min(std::min(vec_data_0, vec_data_1), vec_data_2);
  dim3 block(block_dim_x);
  auto max_sm_size = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  int max_wave_num = max_sm_size * 8 * 4;
  int grid_dim_x = (size0 + vec * block_dim_x - 1) / (vec * block_dim_x);
  grid_dim_x = std::min(max_wave_num, grid_dim_x);
  TORCH_INTERNAL_ASSERT(grid_dim_x > 0);
  int grid_dim_y = std::max(int64_t(1), std::min(int64_t(max_wave_num - grid_dim_x), int64_t(size1)));
  TORCH_INTERNAL_ASSERT(grid_dim_y > 0);
  dim3 grid(grid_dim_x, grid_dim_y, 1);

  // std::cout << "vec: " << vec << std::endl;
  // std::cout << "grid_dim_x: " << grid_dim_x << std::endl;
  // std::cout << "grid_dim_y: " << grid_dim_y << std::endl;

  if (vec == 8) {
    elementwise_kernel_2_2_align<nt, 8, res_t, arg0_t, arg1_t, func_t, index_t, stride_t><<<grid, block, 0, stream>>>(
        N, data0, data1, data2, size0, size1, stride00, stride01, stride02,  stride10, stride11, stride12, f);
  } else if (vec == 4) {
    elementwise_kernel_2_2_align<nt, 4, res_t, arg0_t, arg1_t, func_t, index_t, stride_t><<<grid, block, 0, stream>>>(
        N, data0, data1, data2, size0, size1, stride00, stride01, stride02,  stride10, stride11, stride12, f);
  } else if (vec == 2) {
    elementwise_kernel_2_2_align<nt, 2, res_t, arg0_t, arg1_t, func_t, index_t, stride_t><<<grid, block, 0, stream>>>(
        N, data0, data1, data2, size0, size1, stride00, stride01, stride02,  stride10, stride11, stride12, f);
  } else {
    elementwise_kernel_2_2_align<nt, 1, res_t, arg0_t, arg1_t, func_t, index_t, stride_t><<<grid, block, 0, stream>>>(
        N, data0, data1, data2, size0, size1, stride00, stride01, stride02,  stride10, stride11, stride12, f);
  }
}

template<int nt, int vt, typename res_t , typename arg0_t, typename arg1_t, typename func_t, typename index_t, typename stride_t>
static void launch_legacy_kernel_maca_2_2_broadcast(
    int64_t N,
    char* data0, char* data1, char* data2,
    index_t size0, index_t size1,
    stride_t stride00, stride_t stride01, stride_t stride02,
    stride_t stride10, stride_t stride11, stride_t stride12,
    const func_t& f) {
  // ndim = 2, arity = 2
  TORCH_INTERNAL_ASSERT(N >= 0 && N <= std::numeric_limits<int32_t>::max());
  if (N == 0) {
    return;
  }
  auto stream = at::cuda::getCurrentCUDAStream();
  constexpr int block_dim_x = 64;
  size_t load_num = size0 / block_dim_x;
  TORCH_INTERNAL_ASSERT(load_num > 0);
  int vec_data_0 = getVectorizedAlignment<res_t>((void*)data0, load_num);
  // Notice: offsets required vectorized alignment when Half or BFloat16.
  if (std::is_same<res_t, at::Half>::value || std::is_same<res_t, at::BFloat16>::value) {
    vec_data_0 = std::min(vec_data_0, getVectorizedAlignment<res_t>((void*)data0, stride10/sizeof(res_t)));
  }
  int vec_data_1 = getVectorizedAlignment<arg0_t>((void*)data1, load_num);
  if ((std::is_same<arg0_t, at::Half>::value || std::is_same<arg0_t, at::BFloat16>::value) &&
      ((stride01 == 0 && stride11 != sizeof(arg0_t)) || (stride01 != 0 && stride11 != 0))) {
    vec_data_1 = std::min(vec_data_1, getVectorizedAlignment<arg0_t>((void*)data1, stride11/sizeof(arg0_t)));
  }
  int vec_data_2 = getVectorizedAlignment<arg1_t>((void*)data2, load_num);
  if ((std::is_same<arg1_t, at::Half>::value || std::is_same<arg1_t, at::BFloat16>::value) &&
      ((stride02 == 0 && stride12 != sizeof(arg1_t)) || (stride02 != 0 && stride12 != 0))) {
    vec_data_2 = std::min(vec_data_2, getVectorizedAlignment<arg1_t>((void*)data2, stride12/sizeof(arg1_t)));
  }
  int vec = std::min(std::min(vec_data_0, vec_data_1), vec_data_2);
  dim3 block(block_dim_x);
  int grid_dim_x = load_num / vec;
  TORCH_INTERNAL_ASSERT(grid_dim_x > 0);
  int grid_dim_y = getMaxGridSize(grid_dim_x, size1);
  TORCH_INTERNAL_ASSERT(grid_dim_y > 0);
  int y_t = size1 / grid_dim_y;
  if (y_t >= 32) {
    grid_dim_y = getMaxGridSize(grid_dim_x, size1, 4);
    TORCH_INTERNAL_ASSERT(grid_dim_y > 0);
    y_t = size1 / grid_dim_y;
  }
  grid_dim_y = std::ceil(float(size1)/float(y_t));
  // last block handle y_remain
  int y_remain = size1 - (grid_dim_y - 1) * y_t;
  dim3 grid(grid_dim_x, grid_dim_y, 1);
  TORCH_INTERNAL_ASSERT(y_t > 0);
  TORCH_INTERNAL_ASSERT(y_remain > 0);
  if (stride02 == 0) {
    if (vec == 8) {
      elementwise_kernel_2_2_broadcast_uncontiguous<nt, 8, res_t, arg0_t, arg1_t, func_t, index_t, stride_t><<<grid, block, 0, stream>>>(
          N, data0, data1, data2, size0, size1, stride00, stride01, stride02,  stride10, stride11, stride12, f, y_t, y_remain);
    } else if (vec == 4) {
      elementwise_kernel_2_2_broadcast_uncontiguous<nt, 4, res_t, arg0_t, arg1_t, func_t, index_t, stride_t><<<grid, block, 0, stream>>>(
          N, data0, data1, data2, size0, size1, stride00, stride01, stride02,  stride10, stride11, stride12, f, y_t, y_remain);
    } else if (vec == 2) {
      elementwise_kernel_2_2_broadcast_uncontiguous<nt, 2, res_t, arg0_t, arg1_t, func_t, index_t, stride_t><<<grid, block, 0, stream>>>(
          N, data0, data1, data2, size0, size1, stride00, stride01, stride02,  stride10, stride11, stride12, f, y_t, y_remain);
    } else {
      elementwise_kernel_2_2_broadcast_uncontiguous<nt, 1, res_t, arg0_t, arg1_t, func_t, index_t, stride_t><<<grid, block, 0, stream>>>(
          N, data0, data1, data2, size0, size1, stride00, stride01, stride02,  stride10, stride11, stride12, f, y_t, y_remain);
    }
  } else if (stride01 == 0) {
    // printf("broadcast uncontiguous swap half\n");
    if (vec == 8) {
      elementwise_kernel_2_2_broadcast_uncontiguous<nt, 8, res_t, arg1_t, arg0_t, func_t, index_t, stride_t, true><<<grid, block, 0, stream>>>(
          N, data0, data2, data1, size0, size1, stride00, stride02, stride01,  stride10, stride12, stride11, f, y_t, y_remain);
    } else if (vec == 4) {
      elementwise_kernel_2_2_broadcast_uncontiguous<nt, 4, res_t, arg1_t, arg0_t, func_t, index_t, stride_t, true><<<grid, block, 0, stream>>>(
          N, data0, data2, data1, size0, size1, stride00, stride02, stride01,  stride10, stride12, stride11, f, y_t, y_remain);
    } else if (vec == 2) {
      elementwise_kernel_2_2_broadcast_uncontiguous<nt, 2, res_t, arg1_t, arg0_t, func_t, index_t, stride_t, true><<<grid, block, 0, stream>>>(
          N, data0, data2, data1, size0, size1, stride00, stride02, stride01,  stride10, stride12, stride11, f, y_t, y_remain);
    } else {
      elementwise_kernel_2_2_broadcast_uncontiguous<nt, 1, res_t, arg1_t, arg0_t, func_t, index_t, stride_t, true><<<grid, block, 0, stream>>>(
          N, data0, data2, data1, size0, size1, stride00, stride02, stride01,  stride10, stride12, stride11, f, y_t, y_remain);
    }
  } else if (stride12 == 0) {
    if (vec == 8) {
      elementwise_kernel_2_2_broadcast<nt, 8, res_t, arg0_t, arg1_t, func_t, index_t, stride_t><<<grid, block, 0, stream>>>(
          N, data0, data1, data2, size0, size1, stride00, stride01, stride02,  stride10, stride11, stride12, f, y_t, y_remain);
    } else if (vec == 4) {
      elementwise_kernel_2_2_broadcast<nt, 4, res_t, arg0_t, arg1_t, func_t, index_t, stride_t><<<grid, block, 0, stream>>>(
          N, data0, data1, data2, size0, size1, stride00, stride01, stride02,  stride10, stride11, stride12, f, y_t, y_remain);
    } else if (vec == 2) {
      elementwise_kernel_2_2_broadcast<nt, 2, res_t, arg0_t, arg1_t, func_t, index_t, stride_t><<<grid, block, 0, stream>>>(
          N, data0, data1, data2, size0, size1, stride00, stride01, stride02,  stride10, stride11, stride12, f, y_t, y_remain);
    } else {
      elementwise_kernel_2_2_broadcast<nt, 1, res_t, arg0_t, arg1_t, func_t, index_t, stride_t><<<grid, block, 0, stream>>>(
          N, data0, data1, data2, size0, size1, stride00, stride01, stride02,  stride10, stride11, stride12, f, y_t, y_remain);
    }
  } else if (stride11 == 0) {
    // printf("broadcast swap half\n");
    if (vec == 8) {
      elementwise_kernel_2_2_broadcast<nt, 8, res_t, arg1_t, arg0_t, func_t, index_t, stride_t, true><<<grid, block, 0, stream>>>(
          N, data0, data2, data1, size0, size1, stride00, stride02, stride01,  stride10, stride12, stride11, f, y_t, y_remain);
    } else if (vec == 4) {
      elementwise_kernel_2_2_broadcast<nt, 4, res_t, arg1_t, arg0_t, func_t, index_t, stride_t, true><<<grid, block, 0, stream>>>(
          N, data0, data2, data1, size0, size1, stride00, stride02, stride01,  stride10, stride12, stride11, f, y_t, y_remain);
    } else if (vec == 2) {
      elementwise_kernel_2_2_broadcast<nt, 2, res_t, arg1_t, arg0_t, func_t, index_t, stride_t, true><<<grid, block, 0, stream>>>(
          N, data0, data2, data1, size0, size1, stride00, stride02, stride01,  stride10, stride12, stride11, f, y_t, y_remain);
    } else {
      elementwise_kernel_2_2_broadcast<nt, 1, res_t, arg1_t, arg0_t, func_t, index_t, stride_t, true><<<grid, block, 0, stream>>>(
          N, data0, data2, data1, size0, size1, stride00, stride02, stride01,  stride10, stride12, stride11, f, y_t, y_remain);
    }
  } else {
    TORCH_CHECK(false, "elementwise kernel 2_2 but not broadcast!!");
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template<int nt, int vt, typename res_t , typename arg0_t, typename arg1_t, typename func_t, typename index_t, typename stride_t>
static void launch_legacy_kernel_maca_2_2_cast_broadcast(
    int64_t N,
    char* data0, char* data1, char* data2,
    ScalarType st0, ScalarType st1, ScalarType st2,
    index_t size0, index_t size1,
    stride_t stride00, stride_t stride01, stride_t stride02,
    stride_t stride10, stride_t stride11, stride_t stride12,
    const func_t& f) {
  // ndim = 2, arity = 2
  TORCH_INTERNAL_ASSERT(N >= 0 && N <= std::numeric_limits<int32_t>::max());
  if (N == 0) {
    return;
  }
  auto stream = at::cuda::getCurrentCUDAStream();
  constexpr int block_dim_x = 64;
  size_t load_num = size0 / block_dim_x;
  TORCH_INTERNAL_ASSERT(load_num > 0);
  int vec_data_0 = getVectorizedAlignment<res_t>((void*)data0, load_num);
  int vec_data_1 = getVectorizedAlignment<arg0_t>((void*)data1, load_num);
  int vec_data_2 = getVectorizedAlignment<arg1_t>((void*)data2, load_num);
  int vec = std::min(std::min(vec_data_0, vec_data_1), vec_data_2);
  dim3 block(block_dim_x);
  int grid_dim_x = load_num / vec;
  TORCH_INTERNAL_ASSERT(grid_dim_x > 0);
  int grid_dim_y = getMaxGridSize(grid_dim_x, size1);
  TORCH_INTERNAL_ASSERT(grid_dim_y > 0);
  // dim3 grid(grid_dim_x, grid_dim_y, 1);
  int y_t = size1 / grid_dim_y;
  grid_dim_y = std::ceil(float(size1)/float(y_t));
  int y_remain = size1 - (grid_dim_y - 1) * y_t;
  dim3 grid(grid_dim_x, grid_dim_y, 1);
  TORCH_INTERNAL_ASSERT(y_t > 0);
  if (stride02 == 0 && st0 == ScalarType::Float && st1 == ScalarType::Half && st2 == ScalarType::Float) {
    if (vec == 8) {
      elementwise_kernel_2_2_cast_broadcast_uncontiguous<nt, 8, res_t, arg0_t, arg1_t, func_t, index_t, stride_t, float, c10::Half, float><<<grid, block, 0, stream>>>(
          N, data0, data1, data2, size0, size1, stride00, stride01, stride02,  stride10, stride11, stride12, f, y_t, y_remain);
    } else if (vec == 4) {
      elementwise_kernel_2_2_cast_broadcast_uncontiguous<nt, 4, res_t, arg0_t, arg1_t, func_t, index_t, stride_t, float, c10::Half, float><<<grid, block, 0, stream>>>(
          N, data0, data1, data2, size0, size1, stride00, stride01, stride02,  stride10, stride11, stride12, f, y_t, y_remain);
    } else if (vec == 2) {
      elementwise_kernel_2_2_cast_broadcast_uncontiguous<nt, 2, res_t, arg0_t, arg1_t, func_t, index_t, stride_t, float, c10::Half, float><<<grid, block, 0, stream>>>(
          N, data0, data1, data2, size0, size1, stride00, stride01, stride02,  stride10, stride11, stride12, f, y_t, y_remain);
    } else {
      elementwise_kernel_2_2_cast_broadcast_uncontiguous<nt, 1, res_t, arg0_t, arg1_t, func_t, index_t, stride_t, float, c10::Half, float><<<grid, block, 0, stream>>>(
          N, data0, data1, data2, size0, size1, stride00, stride01, stride02,  stride10, stride11, stride12, f, y_t, y_remain);
    }
  } else if (stride01 == 0 && stride11 == 2 && st0 == ScalarType::Float && st1 == ScalarType::Half && st2 == ScalarType::Float) {
    // printf("broadcast uncontiguous swap half\n");
    if (vec == 8) {
      elementwise_kernel_2_2_cast_broadcast_uncontiguous<nt, 8, res_t, arg1_t, arg0_t, func_t, index_t, stride_t, float, float, c10::Half, true><<<grid, block, 0, stream>>>(
          N, data0, data2, data1, size0, size1, stride00, stride02, stride01,  stride10, stride12, stride11, f, y_t, y_remain);
    } else if (vec == 4) {
      elementwise_kernel_2_2_cast_broadcast_uncontiguous<nt, 4, res_t, arg1_t, arg0_t, func_t, index_t, stride_t, float, float, c10::Half, true><<<grid, block, 0, stream>>>(
          N, data0, data2, data1, size0, size1, stride00, stride02, stride01,  stride10, stride12, stride11, f, y_t, y_remain);
    } else if (vec == 2) {
      elementwise_kernel_2_2_cast_broadcast_uncontiguous<nt, 2, res_t, arg1_t, arg0_t, func_t, index_t, stride_t, float, float, c10::Half, true><<<grid, block, 0, stream>>>(
          N, data0, data2, data1, size0, size1, stride00, stride02, stride01,  stride10, stride12, stride11, f, y_t, y_remain);
    } else {
      elementwise_kernel_2_2_cast_broadcast_uncontiguous<nt, 1, res_t, arg1_t, arg0_t, func_t, index_t, stride_t, float, float, c10::Half, true><<<grid, block, 0, stream>>>(
          N, data0, data2, data1, size0, size1, stride00, stride02, stride01,  stride10, stride12, stride11, f, y_t, y_remain);
    }
  } else if (stride12 == 0 && st0 == ScalarType::Float && st1 == ScalarType::Half && st2 == ScalarType::Float) {
    if (vec == 8) {
      elementwise_kernel_2_2_cast_broadcast<nt, 8, res_t, arg0_t, arg1_t, func_t, index_t, stride_t, float, c10::Half, float><<<grid, block, 0, stream>>>(
          N, data0, data1, data2, size0, size1, stride00, stride01, stride02,  stride10, stride11, stride12, f, y_t, y_remain);
    } else if (vec == 4) {
      elementwise_kernel_2_2_cast_broadcast<nt, 4, res_t, arg0_t, arg1_t, func_t, index_t, stride_t, float, c10::Half, float><<<grid, block, 0, stream>>>(
          N, data0, data1, data2, size0, size1, stride00, stride01, stride02,  stride10, stride11, stride12, f, y_t, y_remain);
    } else if (vec == 2) {
      elementwise_kernel_2_2_cast_broadcast<nt, 2, res_t, arg0_t, arg1_t, func_t, index_t, stride_t, float, c10::Half, float><<<grid, block, 0, stream>>>(
          N, data0, data1, data2, size0, size1, stride00, stride01, stride02,  stride10, stride11, stride12, f, y_t, y_remain);
    } else {
      elementwise_kernel_2_2_cast_broadcast<nt, 1, res_t, arg0_t, arg1_t, func_t, index_t, stride_t, float, c10::Half, float><<<grid, block, 0, stream>>>(
          N, data0, data1, data2, size0, size1, stride00, stride01, stride02,  stride10, stride11, stride12, f, y_t, y_remain);
    }
  } else if (stride12 == 0 && st0 == ScalarType::Float && st1 == ScalarType::Float && st2 == ScalarType::Bool) {
    if (vec == 8) {
      elementwise_kernel_2_2_cast_broadcast<nt, 8, res_t, arg0_t, arg1_t, func_t, index_t, stride_t, float, float, bool><<<grid, block, 0, stream>>>(
          N, data0, data1, data2, size0, size1, stride00, stride01, stride02,  stride10, stride11, stride12, f, y_t, y_remain);
    } else if (vec == 4) {
      elementwise_kernel_2_2_cast_broadcast<nt, 4, res_t, arg0_t, arg1_t, func_t, index_t, stride_t, float, float, bool><<<grid, block, 0, stream>>>(
          N, data0, data1, data2, size0, size1, stride00, stride01, stride02,  stride10, stride11, stride12, f, y_t, y_remain);
    } else if (vec == 2) {
      elementwise_kernel_2_2_cast_broadcast<nt, 2, res_t, arg0_t, arg1_t, func_t, index_t, stride_t, float, float, bool><<<grid, block, 0, stream>>>(
          N, data0, data1, data2, size0, size1, stride00, stride01, stride02,  stride10, stride11, stride12, f, y_t, y_remain);
    } else {
      elementwise_kernel_2_2_cast_broadcast<nt, 1, res_t, arg0_t, arg1_t, func_t, index_t, stride_t, float, float, bool><<<grid, block, 0, stream>>>(
          N, data0, data1, data2, size0, size1, stride00, stride01, stride02,  stride10, stride11, stride12, f, y_t, y_remain);
    }
  } else if (stride11 == 0 && st0 == ScalarType::Float && st1 == ScalarType::Half && st2 == ScalarType::Float) {
    // printf("broadcast swap half\n");
    if (vec == 8) {
      elementwise_kernel_2_2_cast_broadcast<nt, 8, res_t, arg1_t, arg0_t, func_t, index_t, stride_t, float, float, c10::Half, true><<<grid, block, 0, stream>>>(
          N, data0, data2, data1, size0, size1, stride00, stride02, stride01,  stride10, stride12, stride11, f, y_t, y_remain);
    } else if (vec == 4) {
      elementwise_kernel_2_2_cast_broadcast<nt, 4, res_t, arg1_t, arg0_t, func_t, index_t, stride_t, float, float, c10::Half, true><<<grid, block, 0, stream>>>(
          N, data0, data2, data1, size0, size1, stride00, stride02, stride01,  stride10, stride12, stride11, f, y_t, y_remain);
    } else if (vec == 2) {
      elementwise_kernel_2_2_cast_broadcast<nt, 2, res_t, arg1_t, arg0_t, func_t, index_t, stride_t, float, float, c10::Half, true><<<grid, block, 0, stream>>>(
          N, data0, data2, data1, size0, size1, stride00, stride02, stride01,  stride10, stride12, stride11, f, y_t, y_remain);
    } else {
      elementwise_kernel_2_2_cast_broadcast<nt, 1, res_t, arg1_t, arg0_t, func_t, index_t, stride_t, float, float, c10::Half, true><<<grid, block, 0, stream>>>(
          N, data0, data2, data1, size0, size1, stride00, stride02, stride01,  stride10, stride12, stride11, f, y_t, y_remain);
    }
  } else {
    TORCH_CHECK(false, "elementwise kernel 2_2 but not broadcast!!");
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template<int nt, int vt, typename res_t , typename arg0_t, typename arg1_t, typename func_t, typename index_t, typename stride_t, typename res_dtype_t=float, typename arg0_dtype_t=float, typename arg1_dtype_t=float>
static void launch_legacy_kernel_maca_2_2_cast_template(
    int64_t N,
    char* data0, char* data1, char* data2,
    ScalarType st0, ScalarType st1, ScalarType st2,
    index_t size0, index_t size1,
    stride_t stride00, stride_t stride01, stride_t stride02,
    stride_t stride10, stride_t stride11, stride_t stride12,
    const func_t& f) {
  // ndim = 2, arity = 2
  TORCH_INTERNAL_ASSERT(N >= 0 && N <= std::numeric_limits<int32_t>::max());
  if (N == 0) {
    return;
  }
  fetch_and_cast_check(st0);
  fetch_and_cast_check(st1);
  fetch_and_cast_check(st2);

  bool output_can_vec = check_cast_vec_template(st0, stride00, stride10);
  bool arg0_can_vec = check_cast_vec_template(st1, stride01, stride11);
  bool arg1_can_vec = check_cast_vec_template(st2, stride02, stride12);

  int vec_data_0 = output_can_vec ? getCastVectorizedAlignment(st0, (void*)data0, size0) : 1;
  int vec_data_1 = arg0_can_vec ? getCastVectorizedAlignment(st1, (void*)data1, size0) : 1;
  int vec_data_2 = arg1_can_vec ? getCastVectorizedAlignment(st2, (void*)data2, size0) : 1;

  // NOTICE: vec num will be 4, 4, 8, for the template kernel will loop to get output and arg0,
  //        this may cause wait in kernel, so modify 8 to 4 for one loop
  //        if we get the vec num 4, 1, 8, change to 4, 1, 4

  auto vec_min = 16;
  vec_min = (vec_data_0 > 1) ? std::min(vec_min, vec_data_0) : vec_min;
  vec_min = (vec_data_1 > 1) ? std::min(vec_min, vec_data_1) : vec_min;
  vec_min = (vec_data_2 > 1) ? std::min(vec_min, vec_data_2) : vec_min;

  vec_data_0 = (vec_data_0 > 1 && vec_data_0 > vec_min) ? vec_min : vec_data_0;
  vec_data_1 = (vec_data_1 > 1 && vec_data_1 > vec_min) ? vec_min : vec_data_1;
  vec_data_2 = (vec_data_2 > 1 && vec_data_2 > vec_min) ? vec_min : vec_data_2;

#define ELEMENTWISE_KERNEL_TEMPLATE(nt, vt, vt_0, vt_1, vec, res_t, arg0_t, arg1_t, func_t, index_t, stride_t, res_dtype_t, arg0_dtype_t, arg1_dtype_t) \
  elementwise_kernel_2_2_cast_template<nt, vt, vt_0, vt_1, vec, res_t, arg0_t, arg1_t, func_t, index_t, stride_t, res_dtype_t, arg0_dtype_t, arg1_dtype_t>   \
    <<<grid, block, 0, stream>>>(N, data0, data1, data2, size0, size1,                                         \
      stride00, stride01, stride02, stride10, stride11, stride12, f);                                          \
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  int vec = std::max(vec_data_0, std::max(vec_data_1, vec_data_2));
  dim3 block(nt);
  dim3 grid((N + block.x * vec - 1) / (block.x * vec));
  auto stream = at::cuda::getCurrentCUDAStream();

  if (st0 == ScalarType::Float && st1 == ScalarType::Char && st2 == ScalarType::Float) {
    SWITCH_TEMPLATE_KERNEL_ARGS_3(nt, vec_data_0, vec_data_1, vec_data_2, vec, ELEMENTWISE_KERNEL_TEMPLATE,
                                res_t, arg0_t, arg1_t, func_t, index_t, stride_t, float, int8_t, float);
  } else if (st0 == ScalarType::Float && st1 == ScalarType::Half && st2 == ScalarType::Float) {
    SWITCH_TEMPLATE_KERNEL_ARGS_3(nt, vec_data_0, vec_data_1, vec_data_2, vec, ELEMENTWISE_KERNEL_TEMPLATE,
                                res_t, arg0_t, arg1_t, func_t, index_t, stride_t, float, at::Half, float);
  } else if (st0 == ScalarType::BFloat16 && st1 == ScalarType::BFloat16 && st2 == ScalarType::Float) {
    SWITCH_TEMPLATE_KERNEL_ARGS_3(nt, vec_data_0, vec_data_1, vec_data_2, vec, ELEMENTWISE_KERNEL_TEMPLATE,
                                res_t, arg0_t, arg1_t, func_t, index_t, stride_t, at::BFloat16, at::BFloat16, float);
  } else {
    TORCH_CHECK(false, "2_2_cast template unsupported dtype!");
  }

#undef ELEMENTWISE_KERNEL_TEMPLATE
}


template<int nt, int vt, typename res_t , typename arg0_t, typename arg1_t, typename func_t, typename index_t, typename stride_t>
static void launch_legacy_kernel_maca_3_2_dim0_contiguous(
    int64_t N,
    char* data0, char* data1, char* data2,
    index_t size0, index_t size1, index_t size2,
    stride_t stride00, stride_t stride01, stride_t stride02,
    stride_t stride10, stride_t stride11, stride_t stride12,
    stride_t stride20, stride_t stride21, stride_t stride22,
    const func_t& f) {
  // ndim = 3, arity = 2
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
    elementwise_kernel_3_2_dim0_contiguous<nt, 4, res_t, arg0_t, arg1_t, func_t, index_t, stride_t><<<grid, block, 0, stream>>>(
      N, data0, data1, data2, size0, size1, size2, stride00, stride01, stride02, stride10, stride11, stride12, stride20, stride21, stride22, f);
  } else if (vec == 8) {
    elementwise_kernel_3_2_dim0_contiguous<nt, 8, res_t, arg0_t, arg1_t, func_t, index_t, stride_t><<<grid, block, 0, stream>>>(
      N, data0, data1, data2, size0, size1, size2, stride00, stride01, stride02, stride10, stride11, stride12, stride20, stride21, stride22, f);

  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template<int nt, int vt, typename res_t , typename arg0_t, typename arg1_t, typename func_t, typename index_t, typename stride_t>
static void launch_legacy_kernel_maca_3_2_broadcast_arg0_dim2_arg1_dim0(
    int64_t N,
    char* data0, char* data1, char* data2,
    index_t size0, index_t size1, index_t size2,
    stride_t stride00, stride_t stride01, stride_t stride02,
    stride_t stride10, stride_t stride11, stride_t stride12,
    stride_t stride20, stride_t stride21, stride_t stride22,
    const func_t& f) {
  // ndim = 3, arity = 2
  TORCH_INTERNAL_ASSERT(N >= 0 && N <= std::numeric_limits<int32_t>::max());
  if (N == 0) {
    return;
  }
  constexpr int64_t block_dim_x = 64;
  int64_t vec0 = 4;
  if (size0 < 4) vec0 = 2;
  int64_t vec1 = 4;

  auto stream = at::cuda::getCurrentCUDAStream();
  dim3 block(block_dim_x);
  dim3 grid((size0 * size1) / (vec0 * block_dim_x) , size2 / vec1);
  if (vec0 == 2) {
    elementwise_kernel_3_2_broadcast_arg0_dim2_arg1_dim0<nt, 2, res_t, arg0_t, arg1_t, func_t, index_t, stride_t><<<grid, block, 0, stream>>>(
      N, data0, data1, data2, size0, size1, size2, stride00, stride01, stride02, stride10, stride11, stride12, stride20, stride21, stride22, vec1, f);
  } else if (vec0 == 4) {
    elementwise_kernel_3_2_broadcast_arg0_dim2_arg1_dim0<nt, 4, res_t, arg0_t, arg1_t, func_t, index_t, stride_t><<<grid, block, 0, stream>>>(
      N, data0, data1, data2, size0, size1, size2, stride00, stride01, stride02, stride10, stride11, stride12, stride20, stride21, stride22, vec1, f);
  } else {
    assert(0);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}


template<int nt, int vt, typename res_t , typename arg0_t, typename arg1_t, typename func_t, typename index_t, typename stride_t>
static void launch_legacy_kernel_maca_3_2_broadcast_dim0(
    int64_t N,
    char* data0, char* data1, char* data2,
    index_t size0, index_t size1, index_t size2,
    stride_t stride00, stride_t stride01, stride_t stride02,
    stride_t stride10, stride_t stride11, stride_t stride12,
    stride_t stride20, stride_t stride21, stride_t stride22,
    const func_t& f) {
  // ndim = 3, arity = 2
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
    elementwise_kernel_3_2_broadcast_dim0<nt, 4, res_t, arg0_t, arg1_t, func_t, index_t, stride_t><<<grid, block, 0, stream>>>(
      N, data0, data1, data2, size0, size1, size2, stride00, stride01, stride02, stride10, stride11, stride12, stride20, stride21, stride22, f);
  } else if (vec == 8) {
    elementwise_kernel_3_2_broadcast_dim0<nt, 8, res_t, arg0_t, arg1_t, func_t, index_t, stride_t><<<grid, block, 0, stream>>>(
      N, data0, data1, data2, size0, size1, size2, stride00, stride01, stride02, stride10, stride11, stride12, stride20, stride21, stride22, f);

  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template<int nt, int vt, typename res_t , typename arg0_t, typename arg1_t, typename func_t, typename index_t, typename stride_t>
static void launch_legacy_kernel_maca_3_2_dim0_contiguous_arg1_dim1_broadcast(
    int64_t N,
    char* data0, char* data1, char* data2,
    index_t size0, index_t size1, index_t size2,
    stride_t stride00, stride_t stride01, stride_t stride02,
    stride_t stride10, stride_t stride11, stride_t stride12,
    stride_t stride20, stride_t stride21, stride_t stride22,
    const func_t& f) {
  // ndim = 3, arity = 2
  TORCH_INTERNAL_ASSERT(N >= 0 && N <= std::numeric_limits<int32_t>::max());
  if (N == 0) {
    return;
  }

  const int vec = 4;
  dim3 block(nt);
  dim3 grid((N + block.x * vec * vec - 1) / (block.x * vec * vec));

  auto stream = at::cuda::getCurrentCUDAStream();
  elementwise_kernel_3_2_dim0_contiguous_arg1_dim1_broadcast<nt, vec, res_t, arg0_t, arg1_t, func_t, index_t, stride_t><<<grid, block, 0, stream>>>(
      N, data0, data1, data2, size0, size1, size2, stride00, stride01, stride02, stride10, stride11, stride12, stride20, stride21, stride22, f);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}


template<int nt, int vt, typename res_t , typename arg0_t, typename arg1_t, typename func_t, typename index_t, typename stride_t>
static void launch_legacy_kernel_maca_3_2(
    int64_t N,
    char* data0, char* data1, char* data2,
    index_t size0, index_t size1, index_t size2,
    stride_t stride00, stride_t stride01, stride_t stride02,
    stride_t stride10, stride_t stride11, stride_t stride12,
    stride_t stride20, stride_t stride21, stride_t stride22,
    const func_t& f) {
  // ndim = 3, arity = 2
  TORCH_INTERNAL_ASSERT(N >= 0 && N <= std::numeric_limits<int32_t>::max());
  if (N == 0) {
    return;
  }

  auto stream = at::cuda::getCurrentCUDAStream();
  // 3_2 1024 640 2
  // 3_2 1280 256 2
  // 3_2 1280 64 2
  // 3_2 256 1280 2
  // 3_2 320 4096 2
  // 3_2 4096 320 2
  // 3_2 64 1280 2
  // 3_2 640 1024 2
  bool env = at::maca::get_maca_disable_element_template_shape();
  if(maca_unlikely(env)){
    dim3 block(nt);
    dim3 grid((N + block.x * vt - 1) / (block.x * vt));
    if (size0 == 1024 && size1 == 640 && size2 == 2) {
      elementwise_kernel_3_2_s<nt, vt, res_t, arg0_t, arg1_t, func_t, index_t, stride_t, 1024, 640, 2><<<grid, block, 0, stream>>>(
          N, data0, data1, data2, stride00, stride01, stride02, stride10, stride11, stride12, stride20, stride21, stride22, f);
    } else if (size0 == 1280 && size1 == 256 && size2 == 2) {
      elementwise_kernel_3_2_s<nt, vt, res_t, arg0_t, arg1_t, func_t, index_t, stride_t, 1280, 256, 2><<<grid, block, 0, stream>>>(
          N, data0, data1, data2, stride00, stride01, stride02, stride10, stride11, stride12, stride20, stride21, stride22, f);
    } else if (size0 == 1280 && size1 == 64 && size2 == 2) {
      elementwise_kernel_3_2_s<nt, vt, res_t, arg0_t, arg1_t, func_t, index_t, stride_t, 1280, 64, 2><<<grid, block, 0, stream>>>(
          N, data0, data1, data2, stride00, stride01, stride02, stride10, stride11, stride12, stride20, stride21, stride22, f);
    } else if (size0 == 256 && size1 == 1280 && size2 == 2) {
      elementwise_kernel_3_2_s<nt, vt, res_t, arg0_t, arg1_t, func_t, index_t, stride_t, 256, 1280, 2><<<grid, block, 0, stream>>>(
          N, data0, data1, data2, stride00, stride01, stride02, stride10, stride11, stride12, stride20, stride21, stride22, f);
    } else if (size0 == 320 && size1 == 4096 && size2 == 2) {
      elementwise_kernel_3_2_s<nt, vt, res_t, arg0_t, arg1_t, func_t, index_t, stride_t, 320, 4096, 2><<<grid, block, 0, stream>>>(
          N, data0, data1, data2, stride00, stride01, stride02, stride10, stride11, stride12, stride20, stride21, stride22, f);
    } else if (size0 == 4096 && size1 == 320 && size2 == 2) {
      elementwise_kernel_3_2_s<nt, vt, res_t, arg0_t, arg1_t, func_t, index_t, stride_t, 4096, 320, 2><<<grid, block, 0, stream>>>(
          N, data0, data1, data2, stride00, stride01, stride02, stride10, stride11, stride12, stride20, stride21, stride22, f);
    } else if (size0 == 64 && size1 == 1280 && size2 == 2) {
      elementwise_kernel_3_2_s<nt, vt, res_t, arg0_t, arg1_t, func_t, index_t, stride_t, 64, 1280, 2><<<grid, block, 0, stream>>>(
          N, data0, data1, data2, stride00, stride01, stride02, stride10, stride11, stride12, stride20, stride21, stride22, f);
    } else if (size0 == 640 && size1 == 1024 && size2 == 2) {
      elementwise_kernel_3_2_s<nt, vt, res_t, arg0_t, arg1_t, func_t, index_t, stride_t, 640, 1024, 2><<<grid, block, 0, stream>>>(
          N, data0, data1, data2, stride00, stride01, stride02, stride10, stride11, stride12, stride20, stride21, stride22, f);
    } else {
      elementwise_kernel_3_2<nt, vt, res_t, arg0_t, arg1_t, func_t, index_t, stride_t><<<grid, block, 0, stream>>>(
          N, data0, data1, data2, size0, size1, size2, stride00, stride01, stride02, stride10, stride11, stride12, stride20, stride21, stride22, f);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
  }else{
    if(maca_likely(!at::maca::get_maca_disable_element_kernel_opt_tile())){
      unsigned int dim_with_minimum_stride_arg0=0,dim_with_minimum_stride_arg1=0,dim_with_minimum_stride_arg2=0;
      unsigned int strides_arg0[3]={stride00,stride10,stride20};
      unsigned int strides_arg1[3]={stride01,stride11,stride21};
      unsigned int strides_arg2[3]={stride02,stride12,stride22};

      // Make sure min_stride != 0
      unsigned int *min_stride_arg0 = nullptr;
      for (unsigned int *it=strides_arg0; it!=(strides_arg0 + 3); it++) {
        if (min_stride_arg0 == nullptr && *it > 0) min_stride_arg0 = it;
        else if (*it > 0 && *it < *min_stride_arg0) min_stride_arg0 = it;
      }

      unsigned int *min_stride_arg1 = nullptr;
      for (unsigned int *it=strides_arg1; it!=(strides_arg1 + 3); it++) {
        if (min_stride_arg1 == nullptr && *it > 0) min_stride_arg1 = it;
        else if (*it > 0 && *it < *min_stride_arg1) min_stride_arg1 = it;
      }

      unsigned int *min_stride_arg2 = nullptr;
      for (unsigned int *it=strides_arg2; it!=(strides_arg2 + 3); it++) {
        if (min_stride_arg2 == nullptr && *it > 0) min_stride_arg2 = it;
        else if (*it > 0 && *it < *min_stride_arg2) min_stride_arg2 = it;
      }

      if((min_stride_arg0 && min_stride_arg1 && min_stride_arg2) && *min_stride_arg0<=40 && *min_stride_arg1<=40 && *min_stride_arg2<=40 && (size0*size1*size2>64*64*64) && size0 >= 64)
      {
        dim_with_minimum_stride_arg0 = std::distance(strides_arg0, min_stride_arg0);
        dim_with_minimum_stride_arg1 = std::distance(strides_arg1, min_stride_arg1);
        dim_with_minimum_stride_arg2 = std::distance(strides_arg2, min_stride_arg2);

        if(dim_with_minimum_stride_arg0==0 && dim_with_minimum_stride_arg1==0 && dim_with_minimum_stride_arg2==1 && size1 >= 16){//001
          dim3 grid ( ceilf((float)size0/(float)64),ceilf((float)(size1)/(float)64) ,size2);
          dim3 block (512)  ;
          elementwise_kernel_3_2_opt_tile<func_t,index_t,stride_t,arg0_t,arg1_t,res_t,false><<<grid, block, 0, stream>>>(
            size0,size1,data0, data1, data2, stride00,stride10, stride20,stride01,stride11, stride21, stride02,stride12, stride22, f);
        }else if(dim_with_minimum_stride_arg0==0 && dim_with_minimum_stride_arg1==0 && dim_with_minimum_stride_arg2==2 && size2 >= 16){//002
          dim3 grid ( ceilf((float)size0/(float)64),ceilf((float)(size2)/(float)64) ,size1);
          dim3 block (512)  ;
          elementwise_kernel_3_2_opt_tile<func_t,index_t,stride_t,arg0_t,arg1_t,res_t,false><<<grid, block, 0, stream>>>(
            size0,size2,data0, data1, data2, stride00,stride20, stride10,stride01,stride21, stride11, stride02,stride22, stride12, f);
        }else if(dim_with_minimum_stride_arg0==0 && dim_with_minimum_stride_arg1==0 && dim_with_minimum_stride_arg2==0 && size1 >= 16){
          //000 s0>=64 s1>=64, choose dim1 as y_idx in shared mem
          dim3 grid ( ceilf((float)size0/(float)64),ceilf((float)(size1)/(float)64) ,size2);
          dim3 block (512)  ;
          elementwise_kernel_3_2_opt_tile<func_t,index_t,stride_t,arg0_t,arg1_t,res_t,true><<<grid, block, 0, stream>>>(
            size0,size1,data0, data1, data2, stride00,stride10, stride20,stride01,stride11, stride21, stride02,stride12, stride22, f);
        }else if(dim_with_minimum_stride_arg0==0 && dim_with_minimum_stride_arg1==0 && dim_with_minimum_stride_arg2==0 && size2 >= 16){
          //000 s0>=64 s2>=64, choose dim2 as y_idx in shared mem
          dim3 grid ( ceilf((float)size0/(float)64),ceilf((float)(size2)/(float)64) ,size1);
          dim3 block (512)  ;
          elementwise_kernel_3_2_opt_tile<func_t,index_t,stride_t,arg0_t,arg1_t,res_t,true><<<grid, block, 0, stream>>>(
            size0,size2,data0, data1, data2, stride00,stride20, stride10,stride01,stride21, stride11, stride02,stride22, stride12, f);        
        } else{
          dim3 block(nt);
          dim3 grid((N + block.x * vt - 1) / (block.x * vt));
          elementwise_kernel_3_2<nt, vt, res_t, arg0_t, arg1_t, func_t, index_t, stride_t><<<grid, block, 0, stream>>>(
            N, data0, data1, data2, size0, size1, size2, stride00, stride01, stride02, stride10, stride11, stride12, stride20, stride21, stride22, f);
        }
      }
      else{
          dim3 block(nt);
          dim3 grid((N + block.x * vt - 1) / (block.x * vt));
          elementwise_kernel_3_2<nt, vt, res_t, arg0_t, arg1_t, func_t, index_t, stride_t><<<grid, block, 0, stream>>>(
            N, data0, data1, data2, size0, size1, size2, stride00, stride01, stride02, stride10, stride11, stride12, stride20, stride21, stride22, f);
      }
    }
    else{
      dim3 block(nt);
      dim3 grid((N + block.x * vt - 1) / (block.x * vt));
      elementwise_kernel_3_2<nt, vt, res_t, arg0_t, arg1_t, func_t, index_t, stride_t><<<grid, block, 0, stream>>>(
          N, data0, data1, data2, size0, size1, size2, stride00, stride01, stride02, stride10, stride11, stride12, stride20, stride21, stride22, f);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
}


template<int nt, int vt, typename res_t , typename arg0_t, typename arg1_t, typename func_t, typename index_t, typename stride_t>
static void launch_legacy_kernel_maca_3_2_broadcast_dim2_arg0_contiguous(
    int64_t N,
    char* data0, char* data1, char* data2,
    index_t size0, index_t size1, index_t size2,
    stride_t stride00, stride_t stride01, stride_t stride02,
    stride_t stride10, stride_t stride11, stride_t stride12,
    stride_t stride20, stride_t stride21, stride_t stride22,
    const func_t& f) {
  // ndim = 3, arity = 2
  TORCH_INTERNAL_ASSERT(N >= 0 && N <= std::numeric_limits<int32_t>::max());
  if (N == 0) {
    return;
  }
  constexpr int block_dim_x = 64;
  size_t load_num = size0 / block_dim_x;
  TORCH_INTERNAL_ASSERT(load_num > 0);
  int vec_data_0 = getVectorizedAlignment<res_t>((void*)data0, load_num);
  // data0 + stride10 should alignment
  int vec_data_0_z = getVectorizedAlignment<res_t>((void*)(data0 + stride10), load_num);
  vec_data_0 = std::min(vec_data_0, vec_data_0_z);
  int vec_data_1 = getVectorizedAlignment<arg0_t>((void*)data1, load_num);
  int vec_data_1_z = getVectorizedAlignment<arg0_t>((void*)(data1 + stride11), load_num);
  vec_data_1 = std::min(vec_data_1, vec_data_1_z);
  int vec = std::min(vec_data_0, vec_data_1);
  dim3 block(block_dim_x);
  int grid_dim_x = load_num / vec;
  TORCH_INTERNAL_ASSERT(grid_dim_x > 0);
  int grid_dim_y = getMaxGridSize(grid_dim_x * size1, size2);
  TORCH_INTERNAL_ASSERT(grid_dim_y > 0);
  int y_t = size2 / grid_dim_y;
  grid_dim_y = std::ceil(float(size2)/float(y_t));
  y_t = size2 / grid_dim_y;
  int y_remain = size2 % grid_dim_y;
  dim3 grid(grid_dim_x, grid_dim_y, size1);
  TORCH_INTERNAL_ASSERT(y_t > 0);
  auto stream = at::cuda::getCurrentCUDAStream();
  if (vec == 8) {
    elementwise_kernel_3_2_broadcast_dim2_arg0_contiguous<nt, 8, res_t, arg0_t, arg1_t, func_t, index_t, stride_t><<<grid, block, 0, stream>>>(
      N, data0, data1, data2, size0, size1, size2, stride00, stride01, stride02, stride10, stride11, stride12, stride20, stride21, stride22, f, y_t, y_remain);
  } else if (vec == 4) {
    elementwise_kernel_3_2_broadcast_dim2_arg0_contiguous<nt, 4, res_t, arg0_t, arg1_t, func_t, index_t, stride_t><<<grid, block, 0, stream>>>(
      N, data0, data1, data2, size0, size1, size2, stride00, stride01, stride02, stride10, stride11, stride12, stride20, stride21, stride22, f, y_t, y_remain);
  } else if (vec == 2) {
    elementwise_kernel_3_2_broadcast_dim2_arg0_contiguous<nt, 2, res_t, arg0_t, arg1_t, func_t, index_t, stride_t><<<grid, block, 0, stream>>>(
      N, data0, data1, data2, size0, size1, size2, stride00, stride01, stride02, stride10, stride11, stride12, stride20, stride21, stride22, f, y_t, y_remain);
  } else {
    elementwise_kernel_3_2_broadcast_dim2_arg0_contiguous<nt, 1, res_t, arg0_t, arg1_t, func_t, index_t, stride_t><<<grid, block, 0, stream>>>(
      N, data0, data1, data2, size0, size1, size2, stride00, stride01, stride02, stride10, stride11, stride12, stride20, stride21, stride22, f, y_t, y_remain);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template<int nt, int vt, typename res_t , typename arg0_t, typename arg1_t, typename func_t, typename index_t, typename stride_t>
static void launch_legacy_kernel_maca_3_2_broadcast_dim1_contiguous(
    int64_t N,
    char* data0, char* data1, char* data2,
    index_t size0, index_t size1, index_t size2,
    stride_t stride00, stride_t stride01, stride_t stride02,
    stride_t stride10, stride_t stride11, stride_t stride12,
    stride_t stride20, stride_t stride21, stride_t stride22,
    const func_t& f) {
  // ndim = 3, arity = 2
  TORCH_INTERNAL_ASSERT(N >= 0 && N <= std::numeric_limits<int32_t>::max());
  if (N == 0) {
    return;
  }
  constexpr int block_dim_x = 64;
  size_t load_num = size0 / block_dim_x;
  // deal with size0 < block_dim_x
  load_num = std::max(load_num, size_t(1));
  TORCH_INTERNAL_ASSERT(load_num > 0);
  int vec_data_0 = getVectorizedAlignment<res_t>((void*)data0, load_num);
  // data0 + stride20 should alignment
  int vec_data_0_z = getVectorizedAlignment<res_t>((void*)(data0 + stride20), load_num);
  vec_data_0 = std::min(vec_data_0, vec_data_0_z);
  int vec_data_1 = getVectorizedAlignment<arg0_t>((void*)data1, load_num);
  int vec_data_1_z = getVectorizedAlignment<arg0_t>((void*)(data1 + stride21), load_num);
  vec_data_1 = std::min(vec_data_1, vec_data_1_z);
  int vec_data_2 = getVectorizedAlignment<arg1_t>((void*)data2, load_num);
  int vec_data_2_z = getVectorizedAlignment<arg1_t>((void*)(data2 + stride22), load_num);
  vec_data_2 = std::min(vec_data_2, vec_data_2_z);
  int vec = std::min(std::min(vec_data_0, vec_data_1), vec_data_2);
  dim3 block(block_dim_x);
  int grid_dim_x = load_num / vec;
  TORCH_INTERNAL_ASSERT(grid_dim_x > 0);
  int grid_dim_y = getMaxGridSize(grid_dim_x * size2, size1);
  TORCH_INTERNAL_ASSERT(grid_dim_y > 0);
  // adjust y_remain
  int y_t = size1 / grid_dim_y;
  grid_dim_y = std::ceil(float(size1)/float(y_t));
  y_t = size1 / grid_dim_y;
  int y_remain = size1 % grid_dim_y;
  dim3 grid(grid_dim_x, grid_dim_y, size2);
  TORCH_INTERNAL_ASSERT(y_t > 0);
  auto stream = at::cuda::getCurrentCUDAStream();
  if (stride12 == 0) {  // broadcast arg2
    if (vec == 8) {
      elementwise_kernel_3_2_broadcast_dim1_contiguous<nt, 8, res_t, arg0_t, arg1_t, func_t, index_t, stride_t><<<grid, block, 0, stream>>>(
        N, data0, data1, data2, size0, size1, size2, stride00, stride01, stride02, stride10, stride11, stride12, stride20, stride21, stride22, f, y_t, y_remain);
    } else if (vec == 4) {
      elementwise_kernel_3_2_broadcast_dim1_contiguous<nt, 4, res_t, arg0_t, arg1_t, func_t, index_t, stride_t><<<grid, block, 0, stream>>>(
        N, data0, data1, data2, size0, size1, size2, stride00, stride01, stride02, stride10, stride11, stride12, stride20, stride21, stride22, f, y_t, y_remain);
    } else if (vec == 2) {
      elementwise_kernel_3_2_broadcast_dim1_contiguous<nt, 2, res_t, arg0_t, arg1_t, func_t, index_t, stride_t><<<grid, block, 0, stream>>>(
        N, data0, data1, data2, size0, size1, size2, stride00, stride01, stride02, stride10, stride11, stride12, stride20, stride21, stride22, f, y_t, y_remain);
    } else {
      elementwise_kernel_3_2_broadcast_dim1_contiguous<nt, 1, res_t, arg0_t, arg1_t, func_t, index_t, stride_t><<<grid, block, 0, stream>>>(
        N, data0, data1, data2, size0, size1, size2, stride00, stride01, stride02, stride10, stride11, stride12, stride20, stride21, stride22, f, y_t, y_remain);
    }
  }
  else if (stride11 == 0) { // broadcast arg1
    if (vec == 8) {
      elementwise_kernel_3_2_broadcast_dim1_contiguous<nt, 8, res_t, arg0_t, arg1_t, func_t, index_t, stride_t, true><<<grid, block, 0, stream>>>(
        N, data0, data2, data1, size0, size1, size2, stride00, stride02, stride01, stride10, stride12, stride11, stride20, stride22, stride21, f, y_t, y_remain);
    } else if (vec == 4) {
      elementwise_kernel_3_2_broadcast_dim1_contiguous<nt, 4, res_t, arg0_t, arg1_t, func_t, index_t, stride_t, true><<<grid, block, 0, stream>>>(
        N, data0, data2, data1, size0, size1, size2, stride00, stride02, stride01, stride10, stride12, stride11, stride20, stride22, stride21, f, y_t, y_remain);
    } else if (vec == 2) {
      elementwise_kernel_3_2_broadcast_dim1_contiguous<nt, 2, res_t, arg0_t, arg1_t, func_t, index_t, stride_t, true><<<grid, block, 0, stream>>>(
        N, data0, data2, data1, size0, size1, size2, stride00, stride02, stride01, stride10, stride12, stride11, stride20, stride22, stride21, f, y_t, y_remain);
    } else {
      elementwise_kernel_3_2_broadcast_dim1_contiguous<nt, 1, res_t, arg0_t, arg1_t, func_t, index_t, stride_t, true><<<grid, block, 0, stream>>>(
        N, data0, data2, data1, size0, size1, size2, stride00, stride02, stride01, stride10, stride12, stride11, stride20, stride22, stride21, f, y_t, y_remain);
    }
  }
  else {
    TORCH_CHECK(false, "elementwise kernel 3_2 but not dim1 broadcast!!");
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template<int nt, int vt, typename res_t , typename arg0_t, typename arg1_t, typename func_t, typename index_t, typename stride_t>
static void launch_legacy_kernel_maca_3_2_broadcast_dim1_contiguous_s(
    int64_t N,
    char* data0, char* data1, char* data2,
    index_t size0, index_t size1, index_t size2,
    stride_t stride00, stride_t stride01, stride_t stride02,
    stride_t stride10, stride_t stride11, stride_t stride12,
    stride_t stride20, stride_t stride21, stride_t stride22,
    const func_t& f) {
  // ndim = 3, arity = 2
  TORCH_INTERNAL_ASSERT(N >= 0 && N <= std::numeric_limits<int32_t>::max());
  if (N == 0) {
    return;
  }
  constexpr int block_dim_x = 64;
  int z_t = getSplitGridZS<res_t>(size0, size1, size2);
  int grid_dim_z = size2 / z_t;
  size_t load_num = size0 * z_t / block_dim_x;
  TORCH_INTERNAL_ASSERT(load_num > 0);
  int vec_data_0 = getVectorizedAlignment<res_t>((void*)data0, load_num);
  // data0 + stride20 should alignment
  int vec_data_0_z = getVectorizedAlignment<res_t>((void*)(data0 + stride20), load_num);
  vec_data_0 = std::min(vec_data_0, vec_data_0_z);
  int vec_data_1 = getVectorizedAlignment<arg0_t>((void*)data1, load_num);
  int vec_data_1_z = getVectorizedAlignment<arg0_t>((void*)(data1 + stride21), load_num);
  vec_data_1 = std::min(vec_data_1, vec_data_1_z);
  int vec_data_2 = getVectorizedAlignment<arg1_t>((void*)data2, load_num);
  int vec_data_2_z = getVectorizedAlignment<arg1_t>((void*)(data2 + stride22), load_num);
  vec_data_2 = std::min(vec_data_2, vec_data_2_z);
  int vec = std::min(std::min(vec_data_0, vec_data_1), vec_data_2);
  dim3 block(block_dim_x);
  int grid_dim_x = load_num / vec;
  TORCH_INTERNAL_ASSERT(grid_dim_x > 0);
  int grid_dim_y = getMaxGridSize(grid_dim_x * grid_dim_z, size1);
  TORCH_INTERNAL_ASSERT(grid_dim_y > 0);
  // adjust y_remain
  int y_t = size1 / grid_dim_y;
  grid_dim_y = std::ceil(float(size1)/float(y_t));
  y_t = size1 / grid_dim_y;
  size_t y_remain = size1 % grid_dim_y;
  size_t z_remain = size2 % z_t;
  dim3 grid(grid_dim_x, grid_dim_y, grid_dim_z);
  TORCH_INTERNAL_ASSERT(y_t > 0);
  auto stream = at::cuda::getCurrentCUDAStream();
  if (vec == 8) {
    elementwise_kernel_3_2_broadcast_dim1_contiguous_s<nt, 8, res_t, arg0_t, arg1_t, func_t, index_t, stride_t><<<grid, block, 0, stream>>>(
      N, data0, data1, data2, size0, size1, size2, stride00, stride01, stride02, stride10, stride11, stride12, stride20, stride21, stride22, f, y_t, z_t, y_remain, z_remain);
  } else if (vec == 4) {
    elementwise_kernel_3_2_broadcast_dim1_contiguous_s<nt, 4, res_t, arg0_t, arg1_t, func_t, index_t, stride_t><<<grid, block, 0, stream>>>(
      N, data0, data1, data2, size0, size1, size2, stride00, stride01, stride02, stride10, stride11, stride12, stride20, stride21, stride22, f, y_t, z_t, y_remain, z_remain);
  } else if (vec == 2) {
    elementwise_kernel_3_2_broadcast_dim1_contiguous_s<nt, 2, res_t, arg0_t, arg1_t, func_t, index_t, stride_t><<<grid, block, 0, stream>>>(
      N, data0, data1, data2, size0, size1, size2, stride00, stride01, stride02, stride10, stride11, stride12, stride20, stride21, stride22, f, y_t, z_t, y_remain, z_remain);
  } else {
    elementwise_kernel_3_2_broadcast_dim1_contiguous_s<nt, 1, res_t, arg0_t, arg1_t, func_t, index_t, stride_t><<<grid, block, 0, stream>>>(
      N, data0, data1, data2, size0, size1, size2, stride00, stride01, stride02, stride10, stride11, stride12, stride20, stride21, stride22, f, y_t, z_t, y_remain, z_remain);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template<int nt, int vt, typename res_t , typename arg0_t, typename arg1_t, typename func_t, typename index_t, typename stride_t>
static void launch_legacy_kernel_maca_3_2_broadcast_dim2_contiguous_s(
    int64_t N,
    char* data0, char* data1, char* data2,
    index_t size0, index_t size1, index_t size2,
    stride_t stride00, stride_t stride01, stride_t stride02,
    stride_t stride10, stride_t stride11, stride_t stride12,
    stride_t stride20, stride_t stride21, stride_t stride22,
    const func_t& f) {
  // ndim = 3, arity = 2
  TORCH_INTERNAL_ASSERT(N >= 0 && N <= std::numeric_limits<int32_t>::max());
  if (N == 0) {
    return;
  }
  constexpr int block_dim_x = 64;
  int z_t = getSplitGridZS<res_t>(size0, size2, size1);
  int grid_dim_z = size1 / z_t;
  size_t load_num = size0 * z_t / block_dim_x;
  TORCH_INTERNAL_ASSERT(load_num > 0);
  int vec_data_0 = getVectorizedAlignment<res_t>((void*)data0, load_num);
  // data0 + stride10 should alignment
  int vec_data_0_z = getVectorizedAlignment<res_t>((void*)(data0 + stride10), load_num);
  vec_data_0 = std::min(vec_data_0, vec_data_0_z);
  int vec_data_1 = getVectorizedAlignment<arg0_t>((void*)data1, load_num);
  int vec_data_1_z = getVectorizedAlignment<arg0_t>((void*)(data1 + stride11), load_num);
  vec_data_1 = std::min(vec_data_1, vec_data_1_z);
  int vec_data_2 = getVectorizedAlignment<arg1_t>((void*)data2, load_num);
  int vec_data_2_z = getVectorizedAlignment<arg1_t>((void*)(data2 + stride12), load_num);
  vec_data_2 = std::min(vec_data_2, vec_data_2_z);
  int vec = std::min(std::min(vec_data_0, vec_data_1), vec_data_2);
  dim3 block(block_dim_x);
  int grid_dim_x = load_num / vec;
  TORCH_INTERNAL_ASSERT(grid_dim_x > 0);
  int grid_dim_y = getMaxGridSize(grid_dim_x * grid_dim_z, size2);
  TORCH_INTERNAL_ASSERT(grid_dim_y > 0);
  // adjust y_remain
  int y_t = size2 / grid_dim_y;
  grid_dim_y = std::ceil(float(size2)/float(y_t));
  y_t = size2 / grid_dim_y;
  size_t y_remain = size2 % grid_dim_y;
  size_t z_remain = size1 % z_t;
  dim3 grid(grid_dim_x, grid_dim_y, grid_dim_z);
  TORCH_INTERNAL_ASSERT(y_t > 0);
  auto stream = at::cuda::getCurrentCUDAStream();
  if (vec == 8) {
    elementwise_kernel_3_2_broadcast_dim2_contiguous_s<nt, 8, res_t, arg0_t, arg1_t, func_t, index_t, stride_t><<<grid, block, 0, stream>>>(
      N, data0, data1, data2, size0, size1, size2, stride00, stride01, stride02, stride10, stride11, stride12, stride20, stride21, stride22, f, y_t, z_t, y_remain, z_remain);
  } else if (vec == 4) {
    elementwise_kernel_3_2_broadcast_dim2_contiguous_s<nt, 4, res_t, arg0_t, arg1_t, func_t, index_t, stride_t><<<grid, block, 0, stream>>>(
      N, data0, data1, data2, size0, size1, size2, stride00, stride01, stride02, stride10, stride11, stride12, stride20, stride21, stride22, f, y_t, z_t, y_remain, z_remain);
  } else if (vec == 2) {
    elementwise_kernel_3_2_broadcast_dim2_contiguous_s<nt, 2, res_t, arg0_t, arg1_t, func_t, index_t, stride_t><<<grid, block, 0, stream>>>(
      N, data0, data1, data2, size0, size1, size2, stride00, stride01, stride02, stride10, stride11, stride12, stride20, stride21, stride22, f, y_t, z_t, y_remain, z_remain);
  } else {
    elementwise_kernel_3_2_broadcast_dim2_contiguous_s<nt, 1, res_t, arg0_t, arg1_t, func_t, index_t, stride_t><<<grid, block, 0, stream>>>(
      N, data0, data1, data2, size0, size1, size2, stride00, stride01, stride02, stride10, stride11, stride12, stride20, stride21, stride22, f, y_t, z_t, y_remain, z_remain);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template<int nt, int vt, typename res_t , typename arg0_t, typename arg1_t, typename func_t, typename index_t, typename stride_t>
static void launch_legacy_kernel_maca_3_2_arity2_transpose(
    int64_t N,
    char* data0, char* data1, char* data2,
    index_t size0, index_t size1, index_t size2,
    stride_t stride00, stride_t stride01, stride_t stride02,
    stride_t stride10, stride_t stride11, stride_t stride12,
    stride_t stride20, stride_t stride21, stride_t stride22,
    const func_t& f) {
  TORCH_INTERNAL_ASSERT(N >= 0 && N <= std::numeric_limits<int32_t>::max());
  if (N == 0) {
    return;
  }
  index_t batch = size2;
  index_t H = size0;
  index_t W = size1;
  auto stream = at::cuda::getCurrentCUDAStream();

  if((size0 % 8 == 0) && (size1 % 8 == 0)) {
    const uint32_t dh = (H + kTileDimMacaT - 1) / kTileDimMacaT;
    const uint32_t dw = (W + kTileDimMacaT - 1) / kTileDimMacaT;
    dim3 grid(dw, dh, batch);
    dim3 block(kBlockRowsMacaT, kTileDimMacaT);
    if (std::is_same<at::Half, res_t>::value && std::is_same<at::Half, arg0_t>::value && std::is_same<at::Half, arg1_t>::value) {
      elementwise_kernel_3_2_arity2_transpose<nt, 8, kTileDimMacaT, res_t, arg0_t, arg1_t, func_t, index_t, stride_t>
          <<<grid, block, 0, stream>>>
          (data0, data1, data2, (uint32_t)H, (uint32_t)W, (uint32_t)dh, (uint32_t)dw, f);
    } else if (std::is_same<at::BFloat16, res_t>::value && std::is_same<at::BFloat16, arg0_t>::value && std::is_same<at::BFloat16, arg1_t>::value) {
      elementwise_kernel_3_2_arity2_transpose<nt, 8, kTileDimMacaT, res_t, arg0_t, arg1_t, func_t, index_t, stride_t>
          <<<grid, block, 0, stream>>>
          (data0, data1, data2, (uint32_t)H, (uint32_t)W, (uint32_t)dh, (uint32_t)dw, f);
    } else if (std::is_same<float, res_t>::value && std::is_same<float, arg0_t>::value && std::is_same<float, arg1_t>::value) {
      elementwise_kernel_3_2_arity2_transpose<nt, 8, kTileDimMacaT, res_t, arg0_t, arg1_t, func_t, index_t, stride_t>
          <<<grid, block, 0, stream>>>
          (data0, data1, data2, (uint32_t)H, (uint32_t)W, (uint32_t)dh, (uint32_t)dw, f);
    } else {
      assert(0);
    }
  } else if ((size0 % 4 == 0) && (size1 % 4 == 0)) {
    // Using Tile size = kTileDimMacaT_32 * kTileDimMacaT_32 when size % 4 ==0
    // Threads per block = kBlockRowsMacaT * kTileDimMacaT_32 = 256
    const uint32_t dh = (H + kTileDimMacaT_32 - 1) / kTileDimMacaT_32;
    const uint32_t dw = (W + kTileDimMacaT_32 - 1) / kTileDimMacaT_32;
    dim3 grid(dw, dh, batch);
    dim3 block(kBlockRowsMacaT, kTileDimMacaT_32);
    if (std::is_same<at::Half, res_t>::value && std::is_same<at::Half, arg0_t>::value && std::is_same<at::Half, arg1_t>::value) {
      elementwise_kernel_3_2_arity2_transpose<nt, 4, kTileDimMacaT_32, res_t, arg0_t, arg1_t, func_t, index_t, stride_t>
          <<<grid, block, 0, stream>>>
          (data0, data1, data2, (uint32_t)H, (uint32_t)W, (uint32_t)dh, (uint32_t)dw, f);
    } else if (std::is_same<at::BFloat16, res_t>::value && std::is_same<at::BFloat16, arg0_t>::value && std::is_same<at::BFloat16, arg1_t>::value) {
      elementwise_kernel_3_2_arity2_transpose<nt, 4, kTileDimMacaT_32, res_t, arg0_t, arg1_t, func_t, index_t, stride_t>
          <<<grid, block, 0, stream>>>
          (data0, data1, data2, (uint32_t)H, (uint32_t)W, (uint32_t)dh, (uint32_t)dw, f);
    } else if (std::is_same<float, res_t>::value && std::is_same<float, arg0_t>::value && std::is_same<float, arg1_t>::value) {
      elementwise_kernel_3_2_arity2_transpose<nt, 4, kTileDimMacaT_32, res_t, arg0_t, arg1_t, func_t, index_t, stride_t>
          <<<grid, block, 0, stream>>>
          (data0, data1, data2, (uint32_t)H, (uint32_t)W, (uint32_t)dh, (uint32_t)dw, f);
    } else {
      assert(0);
    }
  } else {
    // Non-vector version, support size % 2 == 0 Or size is odd number
    const uint32_t dh = (H + kTileDimMacaT - 1) / kTileDimMacaT;
    const uint32_t dw = (W + kTileDimMacaT - 1) / kTileDimMacaT;
    dim3 grid(dw, dh, batch);
    dim3 block(kTileDimMacaT, kBlockRowsMacaT);
    if (std::is_same<at::Half, res_t>::value && std::is_same<at::Half, arg0_t>::value && std::is_same<at::Half, arg1_t>::value) {
      elementwise_kernel_3_2_arity2_transpose_nonvec<res_t, arg0_t, arg1_t, func_t, index_t, stride_t>
          <<<grid, block, 0, stream>>>
          (data0, data1, data2, (uint32_t)H, (uint32_t)W, (uint32_t)dh, (uint32_t)dw, f);
    } else if (std::is_same<at::BFloat16, res_t>::value && std::is_same<at::BFloat16, arg0_t>::value && std::is_same<at::BFloat16, arg1_t>::value) {
      elementwise_kernel_3_2_arity2_transpose_nonvec<res_t, arg0_t, arg1_t, func_t, index_t, stride_t>
          <<<grid, block, 0, stream>>>
          (data0, data1, data2, (uint32_t)H, (uint32_t)W, (uint32_t)dh, (uint32_t)dw, f);
    } else if (std::is_same<float, res_t>::value && std::is_same<float, arg0_t>::value && std::is_same<float, arg1_t>::value) {
      elementwise_kernel_3_2_arity2_transpose_nonvec<res_t, arg0_t, arg1_t, func_t, index_t, stride_t>
          <<<grid, block, 0, stream>>>
          (data0, data1, data2, (uint32_t)H, (uint32_t)W, (uint32_t)dh, (uint32_t)dw, f);
    }
    else {
      assert(0);
    }
  }

  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template<int nt, int vt, typename res_t , typename arg0_t, typename arg1_t, typename func_t, typename index_t, typename stride_t>
static void launch_legacy_kernel_maca_3_2_arity2_transpose_dim02(
    int64_t N,
    char* data0, char* data1, char* data2,
    index_t size0, index_t size1, index_t size2,
    stride_t stride00, stride_t stride01, stride_t stride02,
    stride_t stride10, stride_t stride11, stride_t stride12,
    stride_t stride20, stride_t stride21, stride_t stride22,
    const func_t& f) {
  TORCH_INTERNAL_ASSERT(N >= 0 && N <= std::numeric_limits<int32_t>::max());
  if (N == 0) {
    return;
  }
  // out & arg0: contiguous
  // arg1: contiguous in dim2
  // dim1 : batch dimension
  // shared[size0][size2] = shared[H][W]
  index_t batch = size1;
  // index_t H = size0;
  // index_t W = size1;
  auto stream = at::cuda::getCurrentCUDAStream();
  if((size0 % 8 == 0) && (size2 % 8 == 0)) {
    // const uint32_t dh = (H + kTileDimMacaT - 1) / kTileDimMacaT;
    // const uint32_t dw = (W + kTileDimMacaT - 1) / kTileDimMacaT;
    const uint32_t dy = (size0 + kTileDimMacaT - 1) / kTileDimMacaT;
    const uint32_t dx = (size2 + kTileDimMacaT - 1) / kTileDimMacaT;
    dim3 grid(dx, dy, batch);
    dim3 block(kBlockRowsMacaT, kTileDimMacaT);
    if (std::is_same<at::Half, res_t>::value && std::is_same<at::Half, arg0_t>::value && std::is_same<at::Half, arg1_t>::value) {
      elementwise_kernel_3_2_arity2_transpose_dim02<nt, 8, kTileDimMacaT, res_t, arg0_t, arg1_t, func_t, index_t, stride_t>
          <<<grid, block, 0, stream>>>
          (data0, data1, data2, size0, size1, size2, stride00, stride01, stride02,
          stride10, stride11, stride12, stride20, stride21, stride22, f);
    } else if (std::is_same<at::BFloat16, res_t>::value && std::is_same<at::BFloat16, arg0_t>::value && std::is_same<at::BFloat16, arg1_t>::value) {
      elementwise_kernel_3_2_arity2_transpose_dim02<nt, 8, kTileDimMacaT, res_t, arg0_t, arg1_t, func_t, index_t, stride_t>
          <<<grid, block, 0, stream>>>
          (data0, data1, data2, size0, size1, size2, stride00, stride01, stride02,
          stride10, stride11, stride12, stride20, stride21, stride22, f);
    } else if (std::is_same<float, res_t>::value && std::is_same<float, arg0_t>::value && std::is_same<float, arg1_t>::value) {
      elementwise_kernel_3_2_arity2_transpose_dim02<nt, 8, kTileDimMacaT, res_t, arg0_t, arg1_t, func_t, index_t, stride_t>
          <<<grid, block, 0, stream>>>
          (data0, data1, data2, size0, size1, size2, stride00, stride01, stride02,
          stride10, stride11, stride12, stride20, stride21, stride22, f);
    } else {
      assert(0);
    }
  } else {
    assert(0);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template<int nt, int vt, typename res_t , typename arg0_t, typename arg1_t, typename func_t, typename index_t, typename stride_t>
static void launch_legacy_kernel_maca_3_2_broadcast_dim2_uncontiguous(
    int64_t N,
    char* data0, char* data1, char* data2,
    index_t size0, index_t size1, index_t size2,
    stride_t stride00, stride_t stride01, stride_t stride02,
    stride_t stride10, stride_t stride11, stride_t stride12,
    stride_t stride20, stride_t stride21, stride_t stride22,
    const func_t& f) {
  // ndim = 3, arity = 2
  TORCH_INTERNAL_ASSERT(N >= 0 && N <= std::numeric_limits<int32_t>::max());
  if (N == 0) {
    return;
  }
  constexpr int block_dim_x = 64;
  // split in size1, blockDim.z along the size1
  int z_t = getSplitGridZ<res_t>(size0, size2, size1);
  size_t load_num = size0 * z_t / block_dim_x;
  int grid_dim_z = size1 / z_t;
  TORCH_INTERNAL_ASSERT(load_num > 0);
  int vec = getVectorizedAlignment<res_t>((void*)data0, load_num);
  dim3 block(block_dim_x);
  int grid_dim_x = load_num / vec;
  TORCH_INTERNAL_ASSERT(grid_dim_x > 0);
  int grid_dim_y = getMaxGridSize(grid_dim_x * grid_dim_z, size2);
  TORCH_INTERNAL_ASSERT(grid_dim_y > 0);
  int y_t = size2 / grid_dim_y;
  grid_dim_y = std::ceil(float(size2)/float(y_t));
  y_t = size2 / grid_dim_y;
  int y_remain = size2 % grid_dim_y;
  dim3 grid(grid_dim_x, grid_dim_y, grid_dim_z);
  TORCH_INTERNAL_ASSERT(y_t > 0);
  auto stream = at::cuda::getCurrentCUDAStream();
  if (vec == 8) {
    elementwise_kernel_3_2_broadcast_dim2_uncontiguous<nt, 8, res_t, arg0_t, arg1_t, func_t, index_t, stride_t><<<grid, block, 0, stream>>>(
      N, data0, data1, data2, size0, size1, size2, stride00, stride01, stride02, stride10, stride11, stride12, stride20, stride21, stride22, f, y_t, z_t, y_remain);
  } else if (vec == 4) {
    elementwise_kernel_3_2_broadcast_dim2_uncontiguous<nt, 4, res_t, arg0_t, arg1_t, func_t, index_t, stride_t><<<grid, block, 0, stream>>>(
      N, data0, data1, data2, size0, size1, size2, stride00, stride01, stride02, stride10, stride11, stride12, stride20, stride21, stride22, f, y_t, z_t, y_remain);
  } else if (vec == 2) {
    elementwise_kernel_3_2_broadcast_dim2_uncontiguous<nt, 2, res_t, arg0_t, arg1_t, func_t, index_t, stride_t><<<grid, block, 0, stream>>>(
      N, data0, data1, data2, size0, size1, size2, stride00, stride01, stride02, stride10, stride11, stride12, stride20, stride21, stride22, f, y_t, z_t, y_remain);
  } else {
    elementwise_kernel_3_2_broadcast_dim2_uncontiguous<nt, 1, res_t, arg0_t, arg1_t, func_t, index_t, stride_t><<<grid, block, 0, stream>>>(
      N, data0, data1, data2, size0, size1, size2, stride00, stride01, stride02, stride10, stride11, stride12, stride20, stride21, stride22, f, y_t, z_t, y_remain);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template<int nt, int vt, typename res_t , typename arg0_t, typename arg1_t, typename func_t, typename index_t, typename stride_t>
static void launch_legacy_kernel_maca_3_2_broadcast_dim2_contiguous(
    int64_t N,
    char* data0, char* data1, char* data2,
    index_t size0, index_t size1, index_t size2,
    stride_t stride00, stride_t stride01, stride_t stride02,
    stride_t stride10, stride_t stride11, stride_t stride12,
    stride_t stride20, stride_t stride21, stride_t stride22,
    const func_t& f) {
  // ndim = 3, arity = 2
  TORCH_INTERNAL_ASSERT(N >= 0 && N <= std::numeric_limits<int32_t>::max());
  if (N == 0) {
    return;
  }
  constexpr int block_dim_x = 64;
  size_t load_num = size0 / block_dim_x;
  TORCH_INTERNAL_ASSERT(load_num > 0);
  int vec_data_0 = getVectorizedAlignment<res_t>((void*)data0, load_num);
  // data0 + stride10 should alignment
  int vec_data_0_z = getVectorizedAlignment<res_t>((void*)(data0 + stride10), load_num);
  vec_data_0 = std::min(vec_data_0, vec_data_0_z);
  int vec_data_1 = getVectorizedAlignment<arg0_t>((void*)data1, load_num);
  int vec_data_1_z = getVectorizedAlignment<arg0_t>((void*)(data1 + stride11), load_num);
  vec_data_1 = std::min(vec_data_1, vec_data_1_z);
  int vec_data_2 = getVectorizedAlignment<arg1_t>((void*)data2, load_num);
  int vec_data_2_z = getVectorizedAlignment<arg1_t>((void*)(data2 + stride12), load_num);
  vec_data_2 = std::min(vec_data_2, vec_data_2_z);
  int vec = std::min(std::min(vec_data_0, vec_data_1), vec_data_2);
  dim3 block(block_dim_x);
  int grid_dim_x = load_num / vec;
  TORCH_INTERNAL_ASSERT(grid_dim_x > 0);
  int grid_dim_y = getMaxGridSize(grid_dim_x * size1, size2);
  TORCH_INTERNAL_ASSERT(grid_dim_y > 0);
  int y_t = size2 / grid_dim_y;
  grid_dim_y = std::ceil(float(size2)/float(y_t));
  y_t = size2 / grid_dim_y;
  int y_remain = size2 % grid_dim_y;
  dim3 grid(grid_dim_x, grid_dim_y, size1);
  TORCH_INTERNAL_ASSERT(y_t > 0);
  auto stream = at::cuda::getCurrentCUDAStream();
  if (vec == 8) {
    elementwise_kernel_3_2_broadcast_dim2_contiguous<nt, 8, res_t, arg0_t, arg1_t, func_t, index_t, stride_t><<<grid, block, 0, stream>>>(
      N, data0, data1, data2, size0, size1, size2, stride00, stride01, stride02, stride10, stride11, stride12, stride20, stride21, stride22, f, y_t, y_remain);
  } else if (vec == 4) {
    elementwise_kernel_3_2_broadcast_dim2_contiguous<nt, 4, res_t, arg0_t, arg1_t, func_t, index_t, stride_t><<<grid, block, 0, stream>>>(
      N, data0, data1, data2, size0, size1, size2, stride00, stride01, stride02, stride10, stride11, stride12, stride20, stride21, stride22, f, y_t, y_remain);
  } else if (vec == 2) {
    elementwise_kernel_3_2_broadcast_dim2_contiguous<nt, 2, res_t, arg0_t, arg1_t, func_t, index_t, stride_t><<<grid, block, 0, stream>>>(
      N, data0, data1, data2, size0, size1, size2, stride00, stride01, stride02, stride10, stride11, stride12, stride20, stride21, stride22, f, y_t, y_remain);
  } else {
    elementwise_kernel_3_2_broadcast_dim2_contiguous<nt, 1, res_t, arg0_t, arg1_t, func_t, index_t, stride_t><<<grid, block, 0, stream>>>(
      N, data0, data1, data2, size0, size1, size2, stride00, stride01, stride02, stride10, stride11, stride12, stride20, stride21, stride22, f, y_t, y_remain);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template<int nt, int vt, typename res_t , typename arg0_t, typename arg1_t, typename func_t, typename index_t, typename stride_t>
static void launch_legacy_kernel_maca_3_2_broadcast_dim1_uncontiguous(
    int64_t N,
    char* data0, char* data1, char* data2,
    index_t size0, index_t size1, index_t size2,
    stride_t stride00, stride_t stride01, stride_t stride02,
    stride_t stride10, stride_t stride11, stride_t stride12,
    stride_t stride20, stride_t stride21, stride_t stride22,
    const func_t& f) {
  // ndim = 3, arity = 2
  TORCH_INTERNAL_ASSERT(N >= 0 && N <= std::numeric_limits<int32_t>::max());
  if (N == 0) {
    return;
  }
  constexpr int block_dim_x = 64;
  // split in size2
  int z_t = getSplitGridZ<res_t>(size0, size1, size2);
  size_t load_num = size0 * z_t / block_dim_x;
  int grid_dim_z = size2 / z_t;
  TORCH_INTERNAL_ASSERT(load_num > 0);
  int vec = getVectorizedAlignment<res_t>((void*)data0, load_num);
  dim3 block(block_dim_x);
  int grid_dim_x = load_num / vec;
  TORCH_INTERNAL_ASSERT(grid_dim_x > 0);
  int grid_dim_y = getMaxGridSize(grid_dim_x * grid_dim_z, size1);
  TORCH_INTERNAL_ASSERT(grid_dim_y > 0);
  int y_t = size1 / grid_dim_y;
  grid_dim_y = std::ceil(float(size1)/float(y_t));
  y_t = size1 / grid_dim_y;
  int y_remain = size1 % grid_dim_y;
  dim3 grid(grid_dim_x, grid_dim_y, grid_dim_z);
  TORCH_INTERNAL_ASSERT(y_t > 0);
  auto stream = at::cuda::getCurrentCUDAStream();
  if (vec == 8) {
    elementwise_kernel_3_2_broadcast_dim1_uncontiguous<nt, 8, res_t, arg0_t, arg1_t, func_t, index_t, stride_t><<<grid, block, 0, stream>>>(
      N, data0, data1, data2, size0, size1, size2, stride00, stride01, stride02, stride10, stride11, stride12, stride20, stride21, stride22, f, y_t, z_t, y_remain);
  } else if (vec == 4) {
    elementwise_kernel_3_2_broadcast_dim1_uncontiguous<nt, 4, res_t, arg0_t, arg1_t, func_t, index_t, stride_t><<<grid, block, 0, stream>>>(
      N, data0, data1, data2, size0, size1, size2, stride00, stride01, stride02, stride10, stride11, stride12, stride20, stride21, stride22, f, y_t, z_t, y_remain);
  } else if (vec == 2) {
    elementwise_kernel_3_2_broadcast_dim1_uncontiguous<nt, 2, res_t, arg0_t, arg1_t, func_t, index_t, stride_t><<<grid, block, 0, stream>>>(
      N, data0, data1, data2, size0, size1, size2, stride00, stride01, stride02, stride10, stride11, stride12, stride20, stride21, stride22, f, y_t, z_t, y_remain);
  } else {
    elementwise_kernel_3_2_broadcast_dim1_uncontiguous<nt, 1, res_t, arg0_t, arg1_t, func_t, index_t, stride_t><<<grid, block, 0, stream>>>(
      N, data0, data1, data2, size0, size1, size2, stride00, stride01, stride02, stride10, stride11, stride12, stride20, stride21, stride22, f, y_t, z_t, y_remain);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template<int nt, int vt, typename res_t , typename arg0_t, typename arg1_t, typename func_t, typename index_t, typename stride_t, int v, int v_x>
C10_LAUNCH_BOUNDS_2(nt, 4)
__global__ void elementwise_kernel_4_2_opt_64(
    int64_t N,
    char* data0, char* data1, char* data2,
    index_t size0, index_t size1, index_t size2, index_t size3,
    stride_t stride00, stride_t stride01, stride_t stride02,
    stride_t stride10, stride_t stride11, stride_t stride12,
    stride_t stride20, stride_t stride21, stride_t stride22,
    stride_t stride30, stride_t stride31, stride_t stride32,
    func_t f, index_t dh, index_t dw) {

    using LoadTx = at::native::memory::aligned_vector<arg0_t, v_x>;
    using LoadTy = at::native::memory::aligned_vector<arg1_t, v_x>;
    using StoreT = at::native::memory::aligned_vector<res_t, v_x>;
    arg0_t ld_1[v_x];
    arg1_t ld_2[v_x];
    res_t ld_out[v_x];
    LoadTx* p_ld_1 = reinterpret_cast<LoadTx*>(&ld_1);
    LoadTy* p_ld_2 = reinterpret_cast<LoadTy*>(&ld_2);
    StoreT* p_ld_out = reinterpret_cast<StoreT*>(&ld_out);

    const int64_t s3 = blockIdx.x / (dh * dw);  //size3
    const int64_t k = blockIdx.x % (dh * dw);   //size0*size1*size2
    const int64_t r = k / dw;                   //size1*size2
    const int64_t c = k % dw;                   //size0

    if((r * kTileDimMacaT + threadIdx.y) < (size1 * size2)) {
      int s0 = c * kTileDimMacaT + threadIdx.x * (16 / v);
      int s2 = (r * kTileDimMacaT + threadIdx.y) / size1;
      int s1 = (r * kTileDimMacaT + threadIdx.y) % size1;
      int64_t out_offset = s0 * stride00 + s1 * stride10 + s2 * stride20 + s3 * stride30;
      int64_t in_offset_1 = s0 * stride01 + s1 * stride11 + s2 * stride21 + s3 * stride31;
      int64_t in_offset_2 = s0 * stride02 + s1 * stride12 + s2 * stride22 + s3 * stride32;

      *p_ld_1 = *reinterpret_cast<LoadTx*>(data1 + in_offset_1);
      *p_ld_2 = *reinterpret_cast<LoadTy*>(data2 + in_offset_2);
      StoreT* out = reinterpret_cast<StoreT*>(data0 + out_offset);
      #pragma unroll
      for (int i=0; i < v_x; ++i) {
        ld_out[i] = f(ld_1[i], ld_2[i]);
      }
      *out = *p_ld_out;
    }
}

template<int nt, int vt, typename res_t , typename arg0_t, typename arg1_t, typename func_t, typename index_t, typename stride_t>
C10_LAUNCH_BOUNDS_2(nt, 4)
__global__ void elementwise_kernel_4_2_broadcast_arg0_dim2_arg1_dim0(
    int64_t N,
    char* data0, char* data1, char* data2,
    index_t size0, index_t size1, index_t size2, index_t size3,
    stride_t stride00, stride_t stride01, stride_t stride02,
    stride_t stride10, stride_t stride11, stride_t stride12,
    stride_t stride20, stride_t stride21, stride_t stride22,
    stride_t stride30, stride_t stride31, stride_t stride32,
    int64_t vec1, func_t f) {
  using StoreT = at::native::memory::aligned_vector<res_t, vt>;
  using arg0T = at::native::memory::aligned_vector<arg0_t, vt>;
  using arg1T = at::native::memory::aligned_vector<arg1_t, vt>;
  StoreT st;
  arg0T ld0;

  int64_t index01 = (blockIdx.x * blockDim.x + threadIdx.x) * vt;
  int64_t index0 = index01 % size0;
  int64_t index1 = index01 / size0;
  int offset0 = index0 * stride00 + index1 * stride10 + blockIdx.z * stride30;
  int offset1 = index0 * stride01 + index1 * stride11 + blockIdx.z * stride31;
  int offset2 =                     index1 * stride12 + blockIdx.z * stride32;

  #pragma unroll
  for (int i = 0; i < vt; i++) {
    ld0.val[i] = *(reinterpret_cast<arg0_t*>(data1 + offset1 + i * stride01));
  }

  for(int64_t id = 0; id < vec1; id++) {
    int64_t index2 = blockIdx.y * vec1 + id;
    int64_t res_idx = offset0 + index2 * stride20;
    int64_t arg1_idx = offset2 + index2 * stride22;
    arg1_t tmp = *(reinterpret_cast<arg1_t*>(data2 + arg1_idx));

    #pragma unroll
    for (int i = 0; i < vt; i++) {
      st.val[i] = f(ld0.val[i], tmp);
    }

    *(reinterpret_cast<StoreT*>(data0 + res_idx)) = st;
  }
  
}

template<int nt, int vt, typename res_t , typename arg0_t, typename arg1_t, typename func_t, typename index_t, typename stride_t>
C10_LAUNCH_BOUNDS_2(nt, 4)
__global__ void elementwise_kernel_4_2(
    int64_t N,
    char* data0, char* data1, char* data2,
    index_t size0, index_t size1, index_t size2, index_t size3,
    stride_t stride00, stride_t stride01, stride_t stride02,
    stride_t stride10, stride_t stride11, stride_t stride12,
    stride_t stride20, stride_t stride21, stride_t stride22,
    stride_t stride30, stride_t stride31, stride_t stride32,
    func_t f) {
  // ndim = 4, arity = 2, narg = 3
  int tid = threadIdx.x;
  int nv = nt * vt;
  int64_t idx = nv * blockIdx.x + tid;
  #pragma unroll
  for (int i = 0; i < vt; i++) {
    if (idx < N) {
      int64_t offsets[3];
      auto linear_idx = idx;
      constexpr int NARGS = 3;
      constexpr int MAX_DIMS = 4;
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
      arg = 2;
      offsets[arg] += divmod_mod * stride02;
      // dim = 1
      divmod_div = linear_idx / size1;
      divmod_mod = linear_idx % size1;
      linear_idx = divmod_div;
      arg = 0;
      offsets[arg] += divmod_mod * stride10;
      arg = 1;
      offsets[arg] += divmod_mod * stride11;
      arg = 2;
      offsets[arg] += divmod_mod * stride12;
      // dim = 2
      divmod_div = linear_idx / size2;
      divmod_mod = linear_idx % size2;
      linear_idx = divmod_div;
      arg = 0;
      offsets[arg] += divmod_mod * stride20;
      arg = 1;
      offsets[arg] += divmod_mod * stride21;
      arg = 2;
      offsets[arg] += divmod_mod * stride22;
      // dim = 3
      divmod_div = linear_idx / size3;
      divmod_mod = linear_idx % size3;
      linear_idx = divmod_div;
      arg = 0;
      offsets[arg] += divmod_mod * stride30;
      arg = 1;
      offsets[arg] += divmod_mod * stride31;
      arg = 2;
      offsets[arg] += divmod_mod * stride32;

      res_t* out = (res_t*)(data0 + offsets[0]);
      *out = f(func_reinterpret_cast<arg0_t>(data1 + offsets[1]),
               func_reinterpret_cast<arg1_t>(data2 + offsets[2]));

      idx += nt;
    }
  }
}

template<int nt, int vt, int vt_0, int vt_1, int vec, typename res_t , typename arg0_t, typename arg1_t, typename func_t, typename index_t, typename stride_t>
C10_LAUNCH_BOUNDS_2(nt, 4)
__global__ void elementwise_kernel_4_2_template(
    int64_t N,
    char* data0, char* data1, char* data2,
    index_t size0, index_t size1, index_t size2, index_t size3,
    stride_t stride00, stride_t stride01, stride_t stride02,
    stride_t stride10, stride_t stride11, stride_t stride12,
    stride_t stride20, stride_t stride21, stride_t stride22,
    stride_t stride30, stride_t stride31, stride_t stride32,
    func_t f) {
  int tid = threadIdx.x;
  int nv = nt * vec;
  int64_t linear_idx = nv * blockIdx.x + tid * vec;
  if (linear_idx >= N) return;
  int64_t offsets[3];
  constexpr int NARGS = 3;
  constexpr int MAX_DIMS = 4;
  #pragma unroll
  for (int arg = 0; arg < NARGS; arg++) {
    offsets[arg] = 0;
  }

  // load vec arg0
  arg0_t ld_1[vec];
  using LoadT_1 = at::native::memory::aligned_vector<arg0_t, vt_0>;
  int arg = 1;
  #pragma unroll
  for (int i = 0; i < vec; i += vt_0) {
    offsets[arg] = 0;
    auto linear_idx_1 = linear_idx + i;
    // dim = 0
    auto divmod_div_1 = linear_idx_1 / size0;
    auto divmod_mod_1 = linear_idx_1 % size0;
    linear_idx_1 = divmod_div_1;
    offsets[arg] += divmod_mod_1 * stride01;
    // dim = 1
    divmod_div_1 = linear_idx_1 / size1;
    divmod_mod_1 = linear_idx_1 % size1;
    linear_idx_1 = divmod_div_1;
    offsets[arg] += divmod_mod_1 * stride11;
    // dim = 2
    divmod_div_1 = linear_idx_1 / size2;
    divmod_mod_1 = linear_idx_1 % size2;
    linear_idx_1 = divmod_div_1;
    offsets[arg] += divmod_mod_1 * stride21;
    // dim = 3
    divmod_div_1 = linear_idx_1 / size3;
    divmod_mod_1 = linear_idx_1 % size3;
    linear_idx_1 = divmod_div_1;
    offsets[arg] += divmod_mod_1 * stride31;

    LoadT_1 val = *(reinterpret_cast<LoadT_1*>(data1 + offsets[arg]));
    #pragma unroll
    for (int j = 0; j < vt_0; ++j) {
      ld_1[i + j] = val.val[j];
    }
  }

  // load vec arg1
  arg1_t ld_2[vec];
  using LoadT_2 = at::native::memory::aligned_vector<arg1_t, vt_1>;
  arg = 2;
  #pragma unroll
  for (int i = 0; i < vec; i += vt_1) {
    offsets[arg] = 0;
    auto linear_idx_2 = linear_idx + i;
    // dim = 0
    auto divmod_div_2 = linear_idx_2 / size0;
    auto divmod_mod_2 = linear_idx_2 % size0;
    linear_idx_2 = divmod_div_2;
    offsets[arg] += divmod_mod_2 * stride02;
    // dim = 1
    divmod_div_2 = linear_idx_2 / size1;
    divmod_mod_2 = linear_idx_2 % size1;
    linear_idx_2 = divmod_div_2;
    offsets[arg] += divmod_mod_2 * stride12;
    // dim = 2
    divmod_div_2 = linear_idx_2 / size2;
    divmod_mod_2 = linear_idx_2 % size2;
    linear_idx_2 = divmod_div_2;
    offsets[arg] += divmod_mod_2 * stride22;
    // dim = 3
    divmod_div_2 = linear_idx_2 / size3;
    divmod_mod_2 = linear_idx_2 % size3;
    linear_idx_2 = divmod_div_2;
    offsets[arg] += divmod_mod_2 * stride32;

    LoadT_2 val = *(reinterpret_cast<LoadT_2*>(data2 + offsets[arg]));
    #pragma unroll
    for (int j = 0; j < vt_1; ++j) {
      ld_2[i + j] = val.val[j];
    }
  }

  // store output
  using Store_T = at::native::memory::aligned_vector<res_t, vt>;
  res_t output[vt];
  Store_T* p_store = reinterpret_cast<Store_T*>(&output);
  arg = 0;
  #pragma unroll
  for (int i = 0; i < vec; i += vt) {
    offsets[arg] = 0;
    auto linear_idx_0 = linear_idx + i;
    // dim = 0
    auto divmod_div_0 = linear_idx_0 / size0;
    auto divmod_mod_0 = linear_idx_0 % size0;
    linear_idx_0 = divmod_div_0;
    offsets[arg] += divmod_mod_0 * stride00;
    // dim = 1
    divmod_div_0 = linear_idx_0 / size1;
    divmod_mod_0 = linear_idx_0 % size1;
    linear_idx_0 = divmod_div_0;
    offsets[arg] += divmod_mod_0 * stride10;
    // dim = 2
    divmod_div_0 = linear_idx_0 / size2;
    divmod_mod_0 = linear_idx_0 % size2;
    linear_idx_0 = divmod_div_0;
    offsets[arg] += divmod_mod_0 * stride20;
    // dim = 3
    divmod_div_0 = linear_idx_0 / size3;
    divmod_mod_0 = linear_idx_0 % size3;
    linear_idx_0 = divmod_div_0;
    offsets[arg] += divmod_mod_0 * stride30;

    #pragma unroll
    for (int j = 0; j < vt; ++j) {
      output[j] = f(ld_1[i + j], ld_2[i + j]);
    }

    *(reinterpret_cast<Store_T*>(data0 + offsets[arg])) = *p_store;
  }
}

template<int nt, int vt_0, int vt_1, typename res_t , typename arg0_t, typename arg1_t, typename func_t, typename index_t, typename stride_t>
C10_LAUNCH_BOUNDS_2(nt, 4)
__global__ void elementwise_kernel_4_2_uncontiguous(
    int64_t N,
    char* data0, char* data1, char* data2,
    index_t size0, index_t size1, index_t size2, index_t size3,
    stride_t stride00, stride_t stride01, stride_t stride02,
    stride_t stride10, stride_t stride11, stride_t stride12,
    stride_t stride20, stride_t stride21, stride_t stride22,
    stride_t stride30, stride_t stride31, stride_t stride32,
    func_t f) {

  int tid = threadIdx.x;
  int64_t offsets[3];
  constexpr int NARGS = 3;
  constexpr int MAX_DIMS = 4;

  int idx_0 = blockIdx.x;
  int idx_1 = blockIdx.y;
  int idx_2 = blockIdx.z % size2;
  int idx_3 = blockIdx.z / size2;

  #pragma unroll
  for (int arg = 0; arg < NARGS; arg++) {
    offsets[arg] = 0;
  }
  using LoadT1 = at::native::memory::aligned_vector<arg0_t, vt_0>;
  using LoadT2 = at::native::memory::aligned_vector<arg1_t, vt_1>;
  using StoreT = at::native::memory::aligned_vector<res_t, vt_0>;

  res_t ld_out[vt_1 * vt_0];
  arg0_t ld_1[vt_1 * vt_0];
  arg1_t ld_2[vt_1];
  LoadT1* p_ld_1 = reinterpret_cast<LoadT1*>(&ld_1);
  LoadT2* p_ld_2 = reinterpret_cast<LoadT2*>(&ld_2);
  StoreT * p_ld_out = reinterpret_cast<StoreT*>(&ld_out);

  int offset = idx_3 * stride30 + idx_2 * stride20 + (idx_0 * blockDim.x * vt_0 + tid * vt_0) * stride00;
  offsets[2] = idx_3 * stride32 + idx_2 * stride22 + idx_1 * vt_1 * stride12;
  *p_ld_2 = *reinterpret_cast<LoadT2*>(data2 + offsets[2]);

  #pragma unroll
  for(int i = 0; i < vt_1; i++){
    offsets[0] = offset + (idx_1 * vt_1 + i) * stride10;
    offsets[1] = offset + (idx_1 * vt_1 + i) * stride11;
    *(p_ld_1 + i) = *reinterpret_cast<LoadT1*>(data1 + offsets[1]);
    StoreT* out = reinterpret_cast<StoreT*>(data0 + offsets[0]);
    #pragma unroll
    for (int j=0; j < vt_0; j++){
      ld_out[i * vt_0 + j] = f(ld_1[i * vt_0 + j], ld_2[i]);
    }
    *out = *(p_ld_out + i);
  }
 
}

template<int nt, int vt, typename res_t , typename arg0_t, typename arg1_t, typename func_t, typename index_t, typename stride_t>
static void launch_legacy_kernel_maca_4_2_broadcast_arg0_dim2_arg1_dim0(
    int64_t N,
    char* data0, char* data1, char* data2,
    index_t size0, index_t size1, index_t size2, index_t size3,
    stride_t stride00, stride_t stride01, stride_t stride02,
    stride_t stride10, stride_t stride11, stride_t stride12,
    stride_t stride20, stride_t stride21, stride_t stride22,
    stride_t stride30, stride_t stride31, stride_t stride32,
    const func_t& f) {
  // ndim = 4, arity = 2
  TORCH_INTERNAL_ASSERT(N >= 0 && N <= std::numeric_limits<int32_t>::max());
  if (N == 0) {
    return;
  }
  constexpr int64_t block_dim_x = 64;
  int64_t vec0 = 4;
  if (size0 < 4) vec0 = 2;
  int64_t vec1 = 4;

  auto stream = at::cuda::getCurrentCUDAStream();
  dim3 block(block_dim_x);
  dim3 grid((size0 * size1) / (vec0 * block_dim_x) , size2 / vec1, size3);

  if (vec0 == 2) {
    elementwise_kernel_4_2_broadcast_arg0_dim2_arg1_dim0<nt, 2, res_t, arg0_t, arg1_t, func_t, index_t, stride_t><<<grid, block, 0, stream>>>(
    N, data0, data1, data2,
    size0, size1, size2, size3,
    stride00, stride01, stride02,
    stride10, stride11, stride12,
    stride20, stride21, stride22,
    stride30, stride31, stride32,
    vec1, f);
  } else if (vec0 == 4) {
    elementwise_kernel_4_2_broadcast_arg0_dim2_arg1_dim0<nt, 4, res_t, arg0_t, arg1_t, func_t, index_t, stride_t><<<grid, block, 0, stream>>>(
    N, data0, data1, data2,
    size0, size1, size2, size3,
    stride00, stride01, stride02,
    stride10, stride11, stride12,
    stride20, stride21, stride22,
    stride30, stride31, stride32,
    vec1, f);
  } else {
    assert(0);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template<int nt, int vt, typename res_t , typename arg0_t, typename arg1_t, typename func_t, typename index_t, typename stride_t>
static void launch_legacy_kernel_maca_4_2(
    int64_t N,
    char* data0, char* data1, char* data2,
    index_t size0, index_t size1, index_t size2, index_t size3,
    stride_t stride00, stride_t stride01, stride_t stride02,
    stride_t stride10, stride_t stride11, stride_t stride12,
    stride_t stride20, stride_t stride21, stride_t stride22,
    stride_t stride30, stride_t stride31, stride_t stride32,
    const func_t& f) {
  // ndim = 3, arity = 2
  TORCH_INTERNAL_ASSERT(N >= 0 && N <= std::numeric_limits<int32_t>::max());
  if (N == 0) {
    return;
  }
  dim3 block(nt);
  dim3 grid((N + block.x * vt - 1) / (block.x * vt));
  auto stream = at::cuda::getCurrentCUDAStream();

  elementwise_kernel_4_2<nt, vt, res_t, arg0_t, arg1_t, func_t, index_t, stride_t><<<grid, block, 0, stream>>>(
    N, data0, data1, data2,
    size0, size1, size2, size3,
    stride00, stride01, stride02,
    stride10, stride11, stride12,
    stride20, stride21, stride22,
    stride30, stride31, stride32,
    f);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template<int v, int nt, int vt, typename res_t , typename arg0_t, typename arg1_t, typename func_t, typename index_t, typename stride_t>
static void launch_legacy_kernel_maca_4_2_opt(
    int64_t N,
    char* data0, char* data1, char* data2,
    index_t size0, index_t size1, index_t size2, index_t size3,
    stride_t stride00, stride_t stride01, stride_t stride02,
    stride_t stride10, stride_t stride11, stride_t stride12,
    stride_t stride20, stride_t stride21, stride_t stride22,
    stride_t stride30, stride_t stride31, stride_t stride32,
    const func_t& f) {
  // ndim = 3, arity = 2
  TORCH_INTERNAL_ASSERT(N >= 0 && N <= std::numeric_limits<int32_t>::max());
  if (N == 0) {
    return;
  }
  const index_t dh = (size1 * size2 + kTileDimMacaT - 1) / kTileDimMacaT;
  const index_t dw = (size0 + kTileDimMacaT - 1) / kTileDimMacaT;
  auto stream = at::cuda::getCurrentCUDAStream();
  dim3 block(kBlockRowsMacaT * (v / 2), kTileDimMacaT);
  dim3 grid(size3 * dh * dw);

  elementwise_kernel_4_2_opt_64<nt, vt, res_t, arg0_t, arg1_t, func_t, index_t, stride_t, sizeof(arg0_t), 16 / sizeof(arg0_t)><<<grid, block, 0, stream>>>(
    N, data0, data1, data2,
    size0, size1, size2, size3,
    stride00, stride01, stride02,
    stride10, stride11, stride12,
    stride20, stride21, stride22,
    stride30, stride31, stride32,
    f, dh, dw);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template<int nt, int vt, typename res_t , typename arg0_t, typename arg1_t, typename func_t, typename index_t, typename stride_t>
static void launch_legacy_kernel_maca_4_2_template(
    int64_t N,
    char* data0, char* data1, char* data2,
    index_t size0, index_t size1, index_t size2, index_t size3,
    stride_t stride00, stride_t stride01, stride_t stride02,
    stride_t stride10, stride_t stride11, stride_t stride12,
    stride_t stride20, stride_t stride21, stride_t stride22,
    stride_t stride30, stride_t stride31, stride_t stride32,
    const func_t& f) {
  // ndim = 3, arity = 2
  TORCH_INTERNAL_ASSERT(N >= 0 && N <= std::numeric_limits<int32_t>::max());
  if (N == 0) {
    return;
  }

  bool output_can_vec = (check_vec_template<res_t>(stride10, stride20, stride30) && (stride00 == sizeof(res_t)));
  bool arg0_can_vec = (check_vec_template<arg0_t>(stride11, stride21, stride31) && (stride01 == sizeof(arg0_t)));
  bool arg1_can_vec = (check_vec_template<arg1_t>(stride12, stride22, stride32) && (stride02 == sizeof(arg1_t)));

  int vec_data_0 = output_can_vec ? getVectorizedAlignment<res_t>((void*)data0, size0) : 1;
  int vec_data_1 = arg0_can_vec ? getVectorizedAlignment<arg0_t>((void*)data1, size0) : 1;
  int vec_data_2 = arg1_can_vec ? getVectorizedAlignment<arg1_t>((void*)data2, size0) : 1;

#define ELEMENTWISE_KERNEL_TEMPLATE(nt, vt, vt_0, vt_1, vec, res_t, arg0_t, arg1_t, func_t, index_t, stride_t) \
  elementwise_kernel_4_2_template<nt, vt, vt_0, vt_1, vec, res_t, arg0_t, arg1_t, func_t, index_t, stride_t>   \
    <<<grid, block, 0, stream>>>(N, data0, data1, data2, size0, size1, size2, size3,                           \
      stride00, stride01, stride02, stride10, stride11, stride12,                                              \
      stride20, stride21, stride22, stride30, stride31, stride32, f);                                          \
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  int vec = std::max(vec_data_0, std::max(vec_data_1, vec_data_2));
  dim3 block(nt);
  dim3 grid((N + block.x * vec - 1) / (block.x * vec));
  auto stream = at::cuda::getCurrentCUDAStream();

  SWITCH_TEMPLATE_KERNEL_ARGS_3(nt, vec_data_0, vec_data_1, vec_data_2, vec, ELEMENTWISE_KERNEL_TEMPLATE,
                                res_t, arg0_t, arg1_t, func_t, index_t, stride_t);

#undef ELEMENTWISE_KERNEL_TEMPLATE

}

template<int nt, int vt, typename res_t , typename arg0_t, typename arg1_t, typename func_t, typename index_t, typename stride_t>
static void launch_legacy_kernel_maca_4_2_uncontiguous(
    int64_t N,
    char* data0, char* data1, char* data2,
    index_t size0, index_t size1, index_t size2, index_t size3,
    stride_t stride00, stride_t stride01, stride_t stride02,
    stride_t stride10, stride_t stride11, stride_t stride12,
    stride_t stride20, stride_t stride21, stride_t stride22,
    stride_t stride30, stride_t stride31, stride_t stride32,
    const func_t& f) {
  // ndim = 4, arity = 2, stride02==0
  TORCH_INTERNAL_ASSERT(N >= 0 && N <= std::numeric_limits<int32_t>::max());
  if (N == 0) {
    return;
  }
  // size0 == n * block_dim_x, so no x_remains
  const int block_dim_x = 64;
  dim3 block(block_dim_x);
  size_t load_num = size0 / block_dim_x;

  //  vec in dim0
  int vec_data_0 = getVectorizedAlignment<res_t>((void*)data0, load_num);
  int vec_data_1 = getVectorizedAlignment<arg0_t>((void*)data1, load_num);
  int vec0 = std::min(vec_data_0, vec_data_1);
  int grid_dim_x = load_num / vec0;

  //  vec in dim1
  int vec1 = getVectorizedAlignment<arg1_t>((void*)data2, size1);
  int grid_dim_y = size1 / vec1;

  int grid_dim_z = size2 * size3;
  dim3 grid(grid_dim_x, grid_dim_y, grid_dim_z);
  auto stream = at::cuda::getCurrentCUDAStream();

  if (vec0 == 4) {
    if (vec1 == 4) {
      elementwise_kernel_4_2_uncontiguous<nt, 4, 4, res_t, arg0_t, arg1_t, func_t, index_t, stride_t><<<grid, block, 0, stream>>>(
        N, data0, data1, data2, size0, size1, size2, size3,
        stride00, stride01, stride02, stride10, stride11, stride12, 
        stride20, stride21, stride22, stride30, stride31, stride32, f);
    } else if (vec1 == 2) {
      elementwise_kernel_4_2_uncontiguous<nt, 4, 2, res_t, arg0_t, arg1_t, func_t, index_t, stride_t><<<grid, block, 0, stream>>>(
        N, data0, data1, data2, size0, size1, size2, size3,
        stride00, stride01, stride02, stride10, stride11, stride12, 
        stride20, stride21, stride22, stride30, stride31, stride32, f);
    } else {
      elementwise_kernel_4_2_uncontiguous<nt, 4, 1, res_t, arg0_t, arg1_t, func_t, index_t, stride_t><<<grid, block, 0, stream>>>(
        N, data0, data1, data2, size0, size1, size2, size3,
        stride00, stride01, stride02, stride10, stride11, stride12, 
        stride20, stride21, stride22, stride30, stride31, stride32, f);    
    }
  } else if (vec0 == 2) {
    if (vec1 == 4) {
      elementwise_kernel_4_2_uncontiguous<nt, 2, 4, res_t, arg0_t, arg1_t, func_t, index_t, stride_t><<<grid, block, 0, stream>>>(
        N, data0, data1, data2, size0, size1, size2, size3,
        stride00, stride01, stride02, stride10, stride11, stride12, 
        stride20, stride21, stride22, stride30, stride31, stride32, f);    
    } else if (vec1 == 2) {
      elementwise_kernel_4_2_uncontiguous<nt, 2, 2, res_t, arg0_t, arg1_t, func_t, index_t, stride_t><<<grid, block, 0, stream>>>(
        N, data0, data1, data2, size0, size1, size2, size3,
        stride00, stride01, stride02, stride10, stride11, stride12, 
        stride20, stride21, stride22, stride30, stride31, stride32, f);  
    } else {
      elementwise_kernel_4_2_uncontiguous<nt, 2, 1, res_t, arg0_t, arg1_t, func_t, index_t, stride_t><<<grid, block, 0, stream>>>(
        N, data0, data1, data2, size0, size1, size2, size3,
        stride00, stride01, stride02, stride10, stride11, stride12, 
        stride20, stride21, stride22, stride30, stride31, stride32, f);          
    }
  } else {
    if (vec1 == 4) {
      elementwise_kernel_4_2_uncontiguous<nt, 1, 4, res_t, arg0_t, arg1_t, func_t, index_t, stride_t><<<grid, block, 0, stream>>>(
        N, data0, data1, data2, size0, size1, size2, size3,
        stride00, stride01, stride02, stride10, stride11, stride12, 
        stride20, stride21, stride22, stride30, stride31, stride32, f);    
    } else if (vec1 == 2) {
      elementwise_kernel_4_2_uncontiguous<nt, 1, 2, res_t, arg0_t, arg1_t, func_t, index_t, stride_t><<<grid, block, 0, stream>>>(
        N, data0, data1, data2, size0, size1, size2, size3,
        stride00, stride01, stride02, stride10, stride11, stride12, 
        stride20, stride21, stride22, stride30, stride31, stride32, f);  
    } else {
      elementwise_kernel_4_2_uncontiguous<nt, 1, 1, res_t, arg0_t, arg1_t, func_t, index_t, stride_t><<<grid, block, 0, stream>>>(
        N, data0, data1, data2, size0, size1, size2, size3,
        stride00, stride01, stride02, stride10, stride11, stride12, 
        stride20, stride21, stride22, stride30, stride31, stride32, f);          
    }
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}


#define AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_MACA(_) \
  _(uint8_t, Byte)                             \
  _(int8_t, Char)                              \
  _(int16_t, Short)                            \
  _(int, Int)                                  \
  _(int64_t, Long)                             \
  _(at::Half, Half)                            \
  _(float, Float)                              \
  _(double, Double)                            \
  _(c10::complex<c10::Half>, ComplexHalf)      \
  _(c10::complex<float>, ComplexFloat)         \
  _(c10::complex<double>, ComplexDouble)       \
  _(bool, Bool)                                \
  _(at::BFloat16, BFloat16)


#define MACA_FETCH_AND_CAST_CASE_ARITY2_ARG0(type, scalartype)                        \
  case ScalarType::scalartype:                                                        \
    ld_arg0 = c10::convert<arg0_t>(*reinterpret_cast<type*>(data1 + offsets[1]));

#define MACA_FETCH_AND_CAST_CASE_ARITY2_ARG1(type, scalartype)                        \
  case ScalarType::scalartype:                                                        \
    ld_arg1 = c10::convert<arg1_t>(*reinterpret_cast<type*>(data2 + offsets[2]));

template<int nt, int vec_size, typename res_t, typename arg0_t, typename arg1_t, typename func_t,
                       typename res_dtype_t=float, typename arg0_dtype_t=float, typename arg1_dtype_t=float>
C10_LAUNCH_BOUNDS_2(nt, 4)
__global__ void elementwise_unrolled_arity2_cast(
    int64_t N, const func_t f,
    int64_t stride0, int64_t stride1, int64_t stride2,
    char* data0, char* data1, char* data2) {
    using StoreT = at::native::memory::aligned_vector<res_dtype_t, vec_size>;
    using Load0T = at::native::memory::aligned_vector<arg0_dtype_t, vec_size>;
    using Load1T = at::native::memory::aligned_vector<arg1_dtype_t, vec_size>;

    int linear_idx = (blockIdx.x * blockDim.x + threadIdx.x) * vec_size;
    if (linear_idx >= N) return;
    arg0_dtype_t args0[vec_size];
    arg1_dtype_t args1[vec_size];
    res_dtype_t results[vec_size];
    Load0T* p_ld0 = reinterpret_cast<Load0T*>(&args0);
    Load1T* p_ld1 = reinterpret_cast<Load1T*>(&args1);
    StoreT* p_sr = reinterpret_cast<StoreT*>(&results);
    *p_ld0 = *(reinterpret_cast<Load0T*>(data1 + linear_idx * stride1));
    *p_ld1 = *(reinterpret_cast<Load1T*>(data2 + linear_idx * stride2));
    #pragma unroll
    for (int i = 0; i < vec_size; i++) {
        results[i] = c10::convert<res_dtype_t>(f(c10::convert<arg0_t>(args0[i]), c10::convert<arg1_t>(args1[i])));
    }
    StoreT* out = reinterpret_cast<StoreT*>(data0 + linear_idx * stride0);
    *out = *p_sr;

}

template<int vec_size, typename res_t, typename arg0_t, typename arg1_t, typename func_t>
static inline void launch_unrolled_arity2_cast_kernel(
      int64_t N, const func_t& f,
      ScalarType st0, ScalarType st1, ScalarType st2,
      char* data0, char* data1, char* data2)
{
  TORCH_INTERNAL_ASSERT(N > 0 && N <= std::numeric_limits<int32_t>::max());

  const int64_t nt = 256;
  dim3 block(nt);
  dim3 grid((N + block.x * vec_size - 1) / (block.x * vec_size));
  auto stream = at::cuda::getCurrentCUDAStream();
  if (st0 == ScalarType::Float && st1 == ScalarType::Half && st2 == ScalarType::Float) {
     elementwise_unrolled_arity2_cast<nt, vec_size, res_t, arg0_t, arg1_t, func_t, float, at::Half, float>
                        <<<grid, block, 0, stream>>>(N, f, 4, 2, 4, data0, data1, data2);

  } else if (st0 == ScalarType::Float && st1 == ScalarType::BFloat16 && st2 == ScalarType::Float) {
     elementwise_unrolled_arity2_cast<nt, vec_size, res_t, arg0_t, arg1_t, func_t, float, at::BFloat16, float>
                        <<<grid, block, 0, stream>>>(N, f, 4, 2, 4, data0, data1, data2);

  } else if (st0 == ScalarType::Float && st1 == ScalarType::Float && st2 == ScalarType::Half) {
     elementwise_unrolled_arity2_cast<nt, vec_size, res_t, arg0_t, arg1_t, func_t, float, float, at::Half>
                        <<<grid, block, 0, stream>>>(N, f, 4, 4, 2, data0, data1, data2);

  } else if (st0 == ScalarType::Float && st1 == ScalarType::Float && st2 == ScalarType::BFloat16) {
     elementwise_unrolled_arity2_cast<nt, vec_size, res_t, arg0_t, arg1_t, func_t, float, float, at::BFloat16>
                        <<<grid, block, 0, stream>>>(N, f, 4, 4, 2, data0, data1, data2);

  } else {
    assert(0);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template<int nt, int vt, typename res_t, typename arg0_t, typename arg1_t, typename func_t, typename index_t, typename stride_t>
C10_LAUNCH_BOUNDS_2(nt, 4)
__global__ void elementwise_kernel_1_2_cast(
    int64_t N,
    char* data0, char* data1, char* data2,
    ScalarType st0, ScalarType st1, ScalarType st2,
    index_t size0,
    stride_t stride00, stride_t stride01, stride_t stride02,
    func_t f) {
  // ndim = 1, arity = 2, narg = 3
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
      int64_t offsets[3];
      auto linear_idx = idx;
      constexpr int NARGS = 3;
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
      arg = 2;
      offsets[arg] += divmod_mod * stride02;

      void* out = data0 + offsets[0];

      res_t result = f(c10::fetch_and_cast_without_assert<arg0_t>(st1, data1 + offsets[1]),
                       c10::fetch_and_cast_without_assert<arg1_t>(st2, data2 + offsets[2]));
      c10::cast_and_store_without_assert<res_t>(st0, out, result);
      idx += nt;
    }
  }
}

template<int nt, int vt, typename res_t, typename arg0_t, typename arg1_t, typename func_t, typename index_t, typename stride_t>
C10_LAUNCH_BOUNDS_2(nt, 4)
__global__ void elementwise_kernel_2_2_cast(
    int64_t N,
    char* data0, char* data1, char* data2,
    ScalarType st0, ScalarType st1, ScalarType st2,
    index_t size0, index_t size1,
    stride_t stride00, stride_t stride01, stride_t stride02,
    stride_t stride10, stride_t stride11, stride_t stride12,
    func_t f) {
  // ndim = 2, arity = 2, narg = 3
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
      int64_t offsets[3];
      auto linear_idx = idx;
      constexpr int NARGS = 3;
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
      arg = 2;
      offsets[arg] += divmod_mod * stride02;
      // dim = 1
      divmod_div = linear_idx / size1;
      divmod_mod = linear_idx % size1;
      linear_idx = divmod_div;
      arg = 0;
      offsets[arg] += divmod_mod * stride10;
      arg = 1;
      offsets[arg] += divmod_mod * stride11;
      arg = 2;
      offsets[arg] += divmod_mod * stride12;

      void* out = data0 + offsets[0];

      res_t result = f(c10::fetch_and_cast_without_assert<arg0_t>(st1, data1 + offsets[1]),
                       c10::fetch_and_cast_without_assert<arg1_t>(st2, data2 + offsets[2]));
      c10::cast_and_store_without_assert<res_t>(st0, out, result);
      idx += nt;
    }
  }
}

template<int nt, int vt, typename res_t , typename arg0_t, typename arg1_t, typename func_t, typename index_t, typename stride_t>
C10_LAUNCH_BOUNDS_2(nt, 4)
__global__ void elementwise_kernel_3_2_cast(
    int64_t N,
    char* data0, char* data1, char* data2,
    ScalarType st0, ScalarType st1, ScalarType st2,
    index_t size0, index_t size1, index_t size2,
    stride_t stride00, stride_t stride01, stride_t stride02,
    stride_t stride10, stride_t stride11, stride_t stride12,
    stride_t stride20, stride_t stride21, stride_t stride22,
    func_t f) {
  // ndim = 3, arity = 2, narg = 3
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
      int64_t offsets[3];
      auto linear_idx = idx;
      constexpr int NARGS = 3;
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
      arg = 2;
      offsets[arg] += divmod_mod * stride02;
      // dim = 1
      divmod_div = linear_idx / size1;
      divmod_mod = linear_idx % size1;
      linear_idx = divmod_div;
      arg = 0;
      offsets[arg] += divmod_mod * stride10;
      arg = 1;
      offsets[arg] += divmod_mod * stride11;
      arg = 2;
      offsets[arg] += divmod_mod * stride12;
      // dim = 2
      divmod_div = linear_idx / size2;
      divmod_mod = linear_idx % size2;
      linear_idx = divmod_div;
      arg = 0;
      offsets[arg] += divmod_mod * stride20;
      arg = 1;
      offsets[arg] += divmod_mod * stride21;
      arg = 2;
      offsets[arg] += divmod_mod * stride22;

      void* out = data0 + offsets[0];

      res_t result = f(c10::fetch_and_cast_without_assert<arg0_t>(st1, data1 + offsets[1]),
                       c10::fetch_and_cast_without_assert<arg1_t>(st2, data2 + offsets[2]));
      c10::cast_and_store_without_assert<res_t>(st0, out, result);
      idx += nt;

      // res_t* out = (res_t*)(data0 + offsets[0]);
      // *out = f(*reinterpret_cast<arg0_t*>(data1 + offsets[1]),
      //          *reinterpret_cast<arg1_t*>(data2 + offsets[2]));

      // idx += nt;
    }
  }
}

template<int nt, int vt, typename res_t , typename arg0_t, typename arg1_t, typename func_t, typename index_t, typename stride_t>
C10_LAUNCH_BOUNDS_2(nt, 4)
__global__ void elementwise_kernel_3_2_cast_without_assert(
    int64_t N,
    char* data0, char* data1, char* data2,
    ScalarType st0, ScalarType st1, ScalarType st2,
    index_t size0, index_t size1, index_t size2,
    stride_t stride00, stride_t stride01, stride_t stride02,
    stride_t stride10, stride_t stride11, stride_t stride12,
    stride_t stride20, stride_t stride21, stride_t stride22,
    func_t f) {
  // ndim = 3, arity = 2, narg = 3
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
      int64_t offsets[3];
      auto linear_idx = idx;
      constexpr int NARGS = 3;
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
      arg = 2;
      offsets[arg] += divmod_mod * stride02;
      // dim = 1
      divmod_div = linear_idx / size1;
      divmod_mod = linear_idx % size1;
      linear_idx = divmod_div;
      arg = 0;
      offsets[arg] += divmod_mod * stride10;
      arg = 1;
      offsets[arg] += divmod_mod * stride11;
      arg = 2;
      offsets[arg] += divmod_mod * stride12;
      // dim = 2
      divmod_div = linear_idx / size2;
      divmod_mod = linear_idx % size2;
      linear_idx = divmod_div;
      arg = 0;
      offsets[arg] += divmod_mod * stride20;
      arg = 1;
      offsets[arg] += divmod_mod * stride21;
      arg = 2;
      offsets[arg] += divmod_mod * stride22;

      void* out = data0 + offsets[0];

      res_t result = f(c10::fetch_and_cast_without_assert<arg0_t>(st1, data1 + offsets[1]),
                       c10::fetch_and_cast_without_assert<arg1_t>(st2, data2 + offsets[2]));
      c10::cast_and_store_without_assert<res_t>(st0, out, result);
      idx += nt;

      // res_t* out = (res_t*)(data0 + offsets[0]);
      // *out = f(*reinterpret_cast<arg0_t*>(data1 + offsets[1]),
      //          *reinterpret_cast<arg1_t*>(data2 + offsets[2]));

      // idx += nt;
    }
  }
}
template<typename func_t,typename index_t, typename stride_t,
          typename arg0_t, typename arg1_t,
          typename res_t,
          bool equal>
//C10_LAUNCH_BOUNDS_2(nt, 4)
__global__ void elementwise_kernel_3_2_cast_without_assert_opt_tile(
    int size0,int size,
    char* data0, char* data1, char* data2,
    ScalarType st0, ScalarType st1, ScalarType st2,
    stride_t stride00, stride_t stridea0, stride_t strideb0,
    stride_t stride01, stride_t stridea1, stride_t strideb1,
    stride_t stride02, stride_t stridea2, stride_t strideb2,
    func_t f) {

    __shared__ arg1_t data_s[64][64];

    stride_t  offsets_start2=blockIdx.z*1*strideb2 + blockIdx.y*64*stridea2 + blockIdx.x*64*stride02;
    stride_t  offsets_start1=blockIdx.z*1*strideb1 + blockIdx.y*64*stridea1 + blockIdx.x*64*stride01;
    stride_t  offsets_start0=blockIdx.z*1*strideb0 + blockIdx.y*64*stridea0 + blockIdx.x*64*stride00;

    unsigned int inter_g_id = threadIdx.x /64;
    unsigned int intro_g_id = threadIdx.x %64;
    unsigned int mod_x = (size0 & 63)?(size0 & 63):64;
    unsigned int mod_y = (size & 63)?(size & 63):64;

    if(!equal){
      if (blockIdx.x != gridDim.x-1 && blockIdx.y != gridDim.y-1){
        #pragma unroll
        for ( int i=0;i<8;i++)
        {
          stride_t offset=offsets_start2+(8*i+inter_g_id)*stride02+intro_g_id*stridea2;
          data_s[inter_g_id+8*i][intro_g_id]=c10::fetch_and_cast_without_assert<arg1_t>(st2, data2 + offset);
        }
      }
      else{
        #pragma unroll
        for ( int i=0;i<8;i++)
        {
          if( (blockIdx.x==gridDim.x-1 && blockIdx.y!=gridDim.y-1 && (inter_g_id+8*i) < mod_x) ||
              (blockIdx.x!=gridDim.x-1 && blockIdx.y==gridDim.y-1 && intro_g_id < mod_y) ||
              (blockIdx.x==gridDim.x-1 && blockIdx.y==gridDim.y-1 && intro_g_id < mod_y && (inter_g_id+8*i) < mod_x)
            ){
            stride_t offset=offsets_start2+(8*i+inter_g_id)*stride02+intro_g_id*stridea2;
            data_s[inter_g_id+8*i][intro_g_id]=c10::fetch_and_cast_without_assert<arg1_t>(st2, data2 + offset);

          }
        }
      }
    } else {
      if (blockIdx.x != gridDim.x-1 && blockIdx.y != gridDim.y-1){
        #pragma unroll
        for ( int i=0;i<8;i++)
        {
          stride_t offset=offsets_start2+(8*i+inter_g_id)*stridea2+intro_g_id*stride02;
          data_s[inter_g_id+8*i][intro_g_id]=c10::fetch_and_cast_without_assert<arg1_t>(st2, data2 + offset);

        }
      }
      else{
        #pragma unroll
        for ( int i=0;i<8;i++)
        {
          if( (blockIdx.x==gridDim.x-1 && blockIdx.y!=gridDim.y-1 && intro_g_id < mod_x) ||
              (blockIdx.x!=gridDim.x-1 && blockIdx.y==gridDim.y-1 && (inter_g_id+i*8) < mod_y) ||
              (blockIdx.x==gridDim.x-1 && blockIdx.y==gridDim.y-1 &&  intro_g_id < mod_x && (inter_g_id+i*8) < mod_y)
            ){
            stride_t offset=offsets_start2+(8*i+inter_g_id)*stridea2+intro_g_id*stride02;
            data_s[inter_g_id+8*i][intro_g_id]=c10::fetch_and_cast_without_assert<arg1_t>(st2, data2 + offset);

          }
        }
      }
    }
    __syncthreads();
    if(!equal){
      if (blockIdx.x != gridDim.x-1 && blockIdx.y != gridDim.y-1){
        #pragma unroll
        for ( int i=0;i<8;i++)
        {
          stride_t offsets0=offsets_start0+(8*i+inter_g_id)*stridea0+intro_g_id*stride00;
          stride_t offset=offsets_start1+(8*i+inter_g_id)*stridea1+intro_g_id*stride01;
          void* out = data0 + offsets0;
          res_t result  = f(c10::fetch_and_cast_without_assert<arg0_t>(st1, data1 + offset),data_s[intro_g_id][8*i+inter_g_id]);
          c10::cast_and_store_without_assert<res_t>(st0, out, result);
        }
      } else {
        #pragma unroll
        for ( int i=0;i<8;i++)
        {
          if( (blockIdx.x==gridDim.x-1 && blockIdx.y!=gridDim.y-1 && intro_g_id < mod_x) ||
              (blockIdx.x!=gridDim.x-1 && blockIdx.y==gridDim.y-1 && (inter_g_id+i*8) < mod_y)  ||
              (blockIdx.x==gridDim.x-1 && blockIdx.y==gridDim.y-1 && intro_g_id < mod_x && (inter_g_id+i*8) < mod_y)
            ){
              stride_t offsets0=offsets_start0+(8*i+inter_g_id)*stridea0+intro_g_id*stride00;
              stride_t offset=offsets_start1+(8*i+inter_g_id)*stridea1+intro_g_id*stride01;
              void* out = data0 + offsets0;
              res_t result  = f(c10::fetch_and_cast_without_assert<arg0_t>(st1, data1 + offset),data_s[intro_g_id][8*i+inter_g_id]);
              c10::cast_and_store_without_assert<res_t>(st0, out, result);
          }
        }
      }
    } else {
      if (blockIdx.x != gridDim.x-1 && blockIdx.y != gridDim.y-1){
        #pragma unroll
        for ( int i=0;i<8;i++)
        {
          stride_t offsets0=offsets_start0+(8*i+inter_g_id)*stridea0+intro_g_id*stride00;
          stride_t offset=offsets_start1+(8*i+inter_g_id)*stridea1+intro_g_id*stride01;
          void* out = data0 + offsets0;
          res_t result  = f(c10::fetch_and_cast_without_assert<arg0_t>(st1, data1 + offset),data_s[8*i+inter_g_id][intro_g_id]);
          c10::cast_and_store_without_assert<res_t>(st0, out, result);
        }
      } else {
        #pragma unroll
        for ( int i=0;i<8;i++)
        {
          if( (blockIdx.x==gridDim.x-1 && blockIdx.y!=gridDim.y-1 && intro_g_id < mod_x) ||
              (blockIdx.x!=gridDim.x-1 && blockIdx.y==gridDim.y-1 && (inter_g_id+i*8) < mod_y)  ||
              (blockIdx.x==gridDim.x-1 && blockIdx.y==gridDim.y-1 && intro_g_id < mod_x && (inter_g_id+i*8) < mod_y)
            ){
              stride_t offsets0=offsets_start0+(8*i+inter_g_id)*stridea0+intro_g_id*stride00;
              stride_t offset=offsets_start1+(8*i+inter_g_id)*stridea1+intro_g_id*stride01;
              void* out = data0 + offsets0;
              res_t result  = f(c10::fetch_and_cast_without_assert<arg0_t>(st1, data1 + offset),data_s[8*i+inter_g_id][intro_g_id]);
              c10::cast_and_store_without_assert<res_t>(st0, out, result);
          }
        }
      }
    }
}
template<int nt, int vt, typename res_t , typename arg0_t, typename arg1_t, typename func_t, typename index_t, typename stride_t,
        typename res_dtype_t=float, typename arg0_dtype_t=float, typename arg1_dtype_t=float>
C10_LAUNCH_BOUNDS_2(nt, 4)
__global__ void elementwise_kernel_4_2_cast_broadcast(
    int64_t N,
    char* data0, char* data1, char* data2,
    ScalarType st0, ScalarType st1, ScalarType st2,
    index_t size0, index_t size1, index_t size2, index_t size3,
    stride_t stride00, stride_t stride01, stride_t stride02,
    stride_t stride10, stride_t stride11, stride_t stride12,
    stride_t stride20, stride_t stride21, stride_t stride22,
    stride_t stride30, stride_t stride31, stride_t stride32,
    func_t f) {
  //out continuous, inp0 dilation, inp1 broadcast

  int tid = threadIdx.x;
  int linear_idx = (blockIdx.x * blockDim.x + tid) * vt;
  int res_idx = linear_idx;
  if (linear_idx >= N) return;

  using LoadT0 = at::native::memory::aligned_vector<arg0_dtype_t, vt>;
  using LoadT1 = at::native::memory::aligned_vector<arg1_dtype_t, vt>;
  using StoreT = at::native::memory::aligned_vector<res_dtype_t, vt>;

  // arg0_dtype_t ld_0[vt];
  LoadT1 ld_1;
  StoreT ld_out;

  int offset2 = 0;
  auto divmod_div = linear_idx / size0;
  auto divmod_mod = linear_idx % size0;
  linear_idx = divmod_div;
  offset2 += divmod_mod * stride02;

  divmod_div = linear_idx / size1;
  divmod_mod = linear_idx % size1;
  linear_idx = divmod_div;
  offset2 += divmod_mod * stride12;

  divmod_div = linear_idx / size2;
  divmod_mod = linear_idx % size2;
  linear_idx = divmod_div;
  offset2 += divmod_mod * stride22;

  divmod_div = linear_idx / size3;
  divmod_mod = linear_idx % size3;
  linear_idx = divmod_div;
  offset2 += divmod_mod * stride32;
  ld_1 = *(reinterpret_cast<LoadT1*>(data2 + offset2));

  #pragma unroll
  for (int i = 0; i < vt; i++) {
    int offset1 = (res_idx + i) * stride01;
    arg0_dtype_t item0 = *(reinterpret_cast<arg0_dtype_t*>(data1 + offset1));
    ld_out.val[i] = c10::convert<res_dtype_t>(f(c10::convert<arg0_t>(item0), c10::convert<arg1_t>(ld_1.val[i])));
  }
  *(reinterpret_cast<StoreT*>(data0 + res_idx * stride00)) = ld_out;
}


template<int nt, int vt, typename res_t , typename arg0_t, typename arg1_t, typename func_t, typename index_t, typename stride_t>
C10_LAUNCH_BOUNDS_2(nt, 4)
__global__ void elementwise_kernel_4_2_cast_without_assert(
    int64_t N,
    char* data0, char* data1, char* data2,
    ScalarType st0, ScalarType st1, ScalarType st2,
    index_t size0, index_t size1, index_t size2, index_t size3,
    stride_t stride00, stride_t stride01, stride_t stride02,
    stride_t stride10, stride_t stride11, stride_t stride12,
    stride_t stride20, stride_t stride21, stride_t stride22,
    stride_t stride30, stride_t stride31, stride_t stride32,
    func_t f) {
  // ndim = 3, arity = 2, narg = 3
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
      int64_t offsets[3];
      auto linear_idx = idx;
      constexpr int NARGS = 3;
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
      arg = 2;
      offsets[arg] += divmod_mod * stride02;
      // dim = 1
      divmod_div = linear_idx / size1;
      divmod_mod = linear_idx % size1;
      linear_idx = divmod_div;
      arg = 0;
      offsets[arg] += divmod_mod * stride10;
      arg = 1;
      offsets[arg] += divmod_mod * stride11;
      arg = 2;
      offsets[arg] += divmod_mod * stride12;
      // dim = 2
      divmod_div = linear_idx / size2;
      divmod_mod = linear_idx % size2;
      linear_idx = divmod_div;
      arg = 0;
      offsets[arg] += divmod_mod * stride20;
      arg = 1;
      offsets[arg] += divmod_mod * stride21;
      arg = 2;
      offsets[arg] += divmod_mod * stride22;
      // dim = 3
      divmod_div = linear_idx / size3;
      divmod_mod = linear_idx % size3;
      linear_idx = divmod_div;
      arg = 0;
      offsets[arg] += divmod_mod * stride30;
      arg = 1;
      offsets[arg] += divmod_mod * stride31;
      arg = 2;
      offsets[arg] += divmod_mod * stride32;

      void* out = data0 + offsets[0];

      res_t result = f(c10::fetch_and_cast_without_assert<arg0_t>(st1, data1 + offsets[1]),
                       c10::fetch_and_cast_without_assert<arg1_t>(st2, data2 + offsets[2]));
      c10::cast_and_store_without_assert<res_t>(st0, out, result);
      idx += nt;

    }
  }
}



template<int nt, int v_x, typename res_t , typename arg0_t, typename arg1_t, typename func_t, typename index_t, typename stride_t,
        typename res_dtype_t=float, typename arg0_dtype_t=float, typename arg1_dtype_t=float>
C10_LAUNCH_BOUNDS_2(nt, 4)
__global__ void elementwise_kernel_3_2_cast_broadcast(
    int64_t N,
    char* data0, char* data1, char* data2,
    ScalarType st0, ScalarType st1, ScalarType st2,
    index_t size0, index_t size1, index_t size2,
    stride_t stride00, stride_t stride01, stride_t stride02,
    stride_t stride10, stride_t stride11, stride_t stride12,
    stride_t stride20, stride_t stride21, stride_t stride22,
    func_t f, int y_t) {
  // ndim = 3, arity = 2, narg = 3
  // load type arg0_dtype_t, compute type arg0_t, store type res_dtype_t
  // default arg1 is the broadcast
  int tid = threadIdx.x;
  int64_t offsets[3];
  constexpr int NARGS = 3;
  constexpr int MAX_DIMS = 2;
  size_t x_remain = size0 % (blockDim.x * v_x);
  size_t tail = size0 % v_x;
  size_t y_remain = size1 % gridDim.y;
  using LoadT1 = at::native::memory::aligned_vector<arg0_dtype_t, v_x>;
  using LoadT2 = at::native::memory::aligned_vector<arg1_dtype_t, v_x>;
  using StoreT = at::native::memory::aligned_vector<res_dtype_t, v_x>;
  #pragma unroll
  for (int arg = 0; arg < NARGS; arg++) {
    offsets[arg] = 0;
  }
  // arg = 0: output; arg = 1: input1; arg = 2: input2;

  arg0_dtype_t ld_1[v_x];
  arg1_dtype_t ld_2[v_x];
  res_dtype_t ld_out[v_x];
  LoadT1* p_ld_1 = reinterpret_cast<LoadT1*>(&ld_1);
  LoadT2* p_ld_2 = reinterpret_cast<LoadT2*>(&ld_2);
  StoreT * p_ld_out = reinterpret_cast<StoreT*>(&ld_out);

  offsets[2] = (blockIdx.x * blockDim.x + tid) * v_x * stride02 + blockIdx.z * stride22;
  size_t row_offset0 = (blockIdx.x * blockDim.x + tid) * v_x * stride00 + blockIdx.z * stride20;
  size_t row_offset1 = (blockIdx.x * blockDim.x + tid) * v_x * stride01 + blockIdx.z * stride21;
  *p_ld_2 = *reinterpret_cast<LoadT2*>(data2 + offsets[2]);
  for (size_t y_idx = 0; y_idx < y_t; y_idx++) {
    offsets[0] = (blockIdx.y * y_t + y_idx) * stride10 + row_offset0;
    // the layout of out is the same as arg0
    offsets[1] = (blockIdx.y * y_t + y_idx) * stride11 + row_offset1;
    *p_ld_1 = *reinterpret_cast<LoadT1*>(data1 + offsets[1]);
    StoreT* out = reinterpret_cast<StoreT*>(data0 + offsets[0]);
    #pragma unroll
    for (int i = 0; i < v_x; i++) {
      ld_out[i] = c10::convert<res_dtype_t>(f(c10::convert<arg0_t>(ld_1[i]), c10::convert<arg1_t>(ld_2[i])));
    }
    *out = *p_ld_out;
  }
  if (y_remain != 0 && blockIdx.y < y_remain) {
    size_t row_offset0 = (blockIdx.x * blockDim.x + tid) * v_x * stride00 + blockIdx.z * stride20;
    size_t row_offset1 = (blockIdx.x * blockDim.x + tid) * v_x * stride01 + blockIdx.z * stride21;
    offsets[0] = (gridDim.y * y_t + blockIdx.y) * stride10 + row_offset0;
    offsets[1] = (gridDim.y * y_t + blockIdx.y) * stride11 + row_offset1;
    *p_ld_1 = *reinterpret_cast<LoadT1*>(data1 + offsets[1]);
    StoreT* out = reinterpret_cast<StoreT*>(data0 + offsets[0]);
    #pragma unroll
    for (int i = 0; i < v_x; i++) {
      ld_out[i] = c10::convert<res_dtype_t>(f(c10::convert<arg0_t>(ld_1[i]), c10::convert<arg1_t>(ld_2[i])));
    }
    *out = *p_ld_out;
  }

  if (x_remain != 0 && (tid + 1) * v_x <= x_remain) {
    offsets[2] = (gridDim.x * blockDim.x + tid) * v_x * stride02 + blockIdx.z * stride22;
    *p_ld_2 = *reinterpret_cast<LoadT2*>(data2 + offsets[2]);
    size_t row_offset0 = (gridDim.x * blockDim.x + tid) * v_x * stride00 + blockIdx.z * stride20;
    size_t row_offset1 = (gridDim.x * blockDim.x + tid) * v_x * stride01 + blockIdx.z * stride21;
    for (int y_idx = 0; y_idx < y_t; y_idx++) {
      offsets[0] = (blockIdx.y * y_t + y_idx) * stride10 + row_offset0;
      offsets[1] = (blockIdx.y * y_t + y_idx) * stride11 + row_offset1;
      *p_ld_1 = *reinterpret_cast<LoadT1*>(data1 + offsets[1]);
      StoreT* out = reinterpret_cast<StoreT*>(data0 + offsets[0]);
      #pragma unroll
      for (int i = 0; i < v_x; i++) {
        ld_out[i] = c10::convert<res_dtype_t>(f(c10::convert<arg0_t>(ld_1[i]), c10::convert<arg1_t>(ld_2[i])));
      }
      *out = *p_ld_out;
    }
    if (y_remain != 0 && blockIdx.y == 0) {
      // #pragma unroll
      for (int y_idx = 0; y_idx < y_remain; y_idx++) {
        offsets[0] = (gridDim.y * y_t + y_idx) * stride10 + row_offset0;
        offsets[1] = (gridDim.y * y_t + y_idx) * stride11 + row_offset1;
        *p_ld_1 = *reinterpret_cast<LoadT1*>(data1 + offsets[1]);
        StoreT* out = reinterpret_cast<StoreT*>(data0 + offsets[0]);
        #pragma unroll
        for (int i = 0; i < v_x; i++) {
          ld_out[i] = c10::convert<res_dtype_t>(f(c10::convert<arg0_t>(ld_1[i]), c10::convert<arg1_t>(ld_2[i])));
        }
        *out = *p_ld_out;
      }
    }
  }

  // handle tail columns( < v_x)
  if(tail && tid < tail){
    auto remain_offset = x_remain / v_x;
    offsets[2] = ((gridDim.x * blockDim.x + remain_offset) * v_x + tid)* stride02 + blockIdx.z * stride22;
    int64_t row_offset0 = ((gridDim.x  * blockDim.x + remain_offset) * v_x + tid) * stride00 + blockIdx.z * stride20;
    int64_t row_offset1 = ((gridDim.x  * blockDim.x + remain_offset) * v_x + tid) * stride01 + blockIdx.z * stride21;
    for (int y_idx = 0; y_idx < y_t; y_idx++) {
      offsets[0] = (blockIdx.y * y_t + y_idx) * stride10 + row_offset0;
      offsets[1] = (blockIdx.y * y_t + y_idx) * stride11 + row_offset1;
      auto p0 = reinterpret_cast<res_dtype_t*>(data0 + offsets[0]);
      auto p1 = reinterpret_cast<arg0_dtype_t*>(data1 + offsets[1]);
      auto p2 = reinterpret_cast<arg1_dtype_t*>(data2 + offsets[2]);
      *p0 = c10::convert<res_dtype_t>(f(c10::convert<arg0_t>(*p1), c10::convert<arg1_t>(*p2)));
    }
    if (y_remain != 0 && blockIdx.y == 0) {
      // #pragma unroll
      for (int y_idx = 0; y_idx < y_remain; y_idx++) {
        offsets[0] = (gridDim.y * y_t + y_idx) * stride10 + row_offset0;
        offsets[1] = (gridDim.y * y_t + y_idx) * stride11 + row_offset1;
        auto p0 = reinterpret_cast<res_dtype_t*>(data0 + offsets[0]);
        auto p1 = reinterpret_cast<arg0_dtype_t*>(data1 + offsets[1]);
        auto p2 = reinterpret_cast<arg1_dtype_t*>(data2 + offsets[2]);
        *p0 = c10::convert<res_dtype_t>(f(c10::convert<arg0_t>(*p1), c10::convert<arg1_t>(*p2)));
      }
    }
  }
}

template<int nt, int v_x, typename res_t , typename arg0_t, typename arg1_t, typename func_t, typename index_t, typename stride_t,
        typename res_dtype_t=float, typename arg0_dtype_t=float, typename arg1_dtype_t=float>
C10_LAUNCH_BOUNDS_2(nt, 4)
__global__ void elementwise_kernel_3_2_cast_broadcast_dim2(
    int64_t N,
    char* data0, char* data1, char* data2,
    ScalarType st0, ScalarType st1, ScalarType st2,
    index_t size0, index_t size1, index_t size2,
    stride_t stride00, stride_t stride01, stride_t stride02,
    stride_t stride10, stride_t stride11, stride_t stride12,
    stride_t stride20, stride_t stride21, stride_t stride22,
    func_t f, int y_t) {
  // ndim = 3, arity = 2, narg = 3
  // load type arg0_dtype_t, compute type arg0_t, store type res_dtype_t
  // default arg1 is the broadcast
  int tid = threadIdx.x;
  int64_t offsets[3];
  constexpr int NARGS = 3;
  constexpr int MAX_DIMS = 2;
  size_t x_remain = size0 % (blockDim.x * v_x);
  size_t tail = size0 % v_x;
  size_t y_remain = size2 % gridDim.y;

  using LoadT1 = at::native::memory::aligned_vector<arg0_dtype_t, v_x>;
  using StoreT = at::native::memory::aligned_vector<res_dtype_t, v_x>;
  #pragma unroll
  for (int arg = 0; arg < NARGS; arg++) {
    offsets[arg] = 0;
  }

  arg0_dtype_t ld_1[v_x];
  arg1_dtype_t ld_2;
  res_dtype_t ld_out[v_x];
  LoadT1* p_ld_1 = reinterpret_cast<LoadT1*>(&ld_1);
  StoreT * p_ld_out = reinterpret_cast<StoreT*>(&ld_out);

  offsets[2] = blockIdx.z * stride12;
  size_t row_offset0 = (blockIdx.x * blockDim.x + tid) * v_x * stride00 + blockIdx.z * stride10;
  size_t row_offset1 = (blockIdx.x * blockDim.x + tid) * v_x * stride01 + blockIdx.z * stride11;
  ld_2 = *(arg1_dtype_t*)(data2 + offsets[2]);
  for (size_t y_idx = 0; y_idx < y_t; y_idx++) {
    offsets[0] = (blockIdx.y * y_t + y_idx) * stride20 + row_offset0;
    // the layout of out is the same as arg0
    offsets[1] = (blockIdx.y * y_t + y_idx) * stride21 + row_offset1;
    *p_ld_1 = *reinterpret_cast<LoadT1*>(data1 + offsets[1]);
    StoreT* out = reinterpret_cast<StoreT*>(data0 + offsets[0]);
    #pragma unroll
    for (int i = 0; i < v_x; i++) {
      ld_out[i] = c10::convert<res_dtype_t>(f(c10::convert<arg0_t>(ld_1[i]), c10::convert<arg1_t>(ld_2)));
    }
    *out = *p_ld_out;
  }
  if (y_remain != 0 && blockIdx.y < y_remain) {
    offsets[0] = (gridDim.y * y_t + blockIdx.y) * stride20 + row_offset0;
    offsets[1] = (gridDim.y * y_t + blockIdx.y) * stride21 + row_offset1;
    *p_ld_1 = *reinterpret_cast<LoadT1*>(data1 + offsets[1]);
    StoreT* out = reinterpret_cast<StoreT*>(data0 + offsets[0]);
    #pragma unroll
    for (int i = 0; i < v_x; i++) {
      ld_out[i] = c10::convert<res_dtype_t>(f(c10::convert<arg0_t>(ld_1[i]), c10::convert<arg1_t>(ld_2)));
    }
    *out = *p_ld_out;
  }

  if (x_remain != 0 && (tid + 1) * v_x <= x_remain) {
    offsets[2] = blockIdx.z * stride12;
    ld_2 = *(arg1_dtype_t*)(data2 + offsets[2]);
    size_t row_offset0 = (gridDim.x * blockDim.x + tid) * v_x * stride00 + blockIdx.z * stride10;
    size_t row_offset1 = (gridDim.x * blockDim.x + tid) * v_x * stride01 + blockIdx.z * stride11;
    for (int y_idx = 0; y_idx < y_t; y_idx++) {
      offsets[0] = (blockIdx.y * y_t + y_idx) * stride20 + row_offset0;
      offsets[1] = (blockIdx.y * y_t + y_idx) * stride21 + row_offset1;
      *p_ld_1 = *reinterpret_cast<LoadT1*>(data1 + offsets[1]);
      StoreT* out = reinterpret_cast<StoreT*>(data0 + offsets[0]);
      #pragma unroll
      for (int i = 0; i < v_x; i++) {
        ld_out[i] = c10::convert<res_dtype_t>(f(c10::convert<arg0_t>(ld_1[i]), c10::convert<arg1_t>(ld_2)));
      }
      *out = *p_ld_out;
    }
    if (y_remain != 0 && blockIdx.y < y_remain) {
      offsets[0] = (gridDim.y * y_t + blockIdx.y) * stride20 + row_offset0;
      offsets[1] = (gridDim.y * y_t + blockIdx.y) * stride21 + row_offset1;
      *p_ld_1 = *reinterpret_cast<LoadT1*>(data1 + offsets[1]);
      StoreT* out = reinterpret_cast<StoreT*>(data0 + offsets[0]);
      #pragma unroll
      for (int i = 0; i < v_x; i++) {
        ld_out[i] = c10::convert<res_dtype_t>(f(c10::convert<arg0_t>(ld_1[i]), c10::convert<arg1_t>(ld_2)));
      }
      *out = *p_ld_out;
    }
  }

  // handle tail columns( < v_x)
  if(tail && tid < tail){
    auto remain_offset = x_remain / v_x;
    offsets[2] = blockIdx.z * stride12;
    int64_t row_offset0 = ((gridDim.x  * blockDim.x + remain_offset) * v_x + tid) * stride00 + blockIdx.z * stride10;
    int64_t row_offset1 = ((gridDim.x  * blockDim.x + remain_offset) * v_x + tid) * stride01 + blockIdx.z * stride11;
    for (int y_idx = 0; y_idx < y_t; y_idx++) {
      offsets[0] = (blockIdx.y * y_t + y_idx) * stride20 + row_offset0;
      offsets[1] = (blockIdx.y * y_t + y_idx) * stride21 + row_offset1;
      auto p0 = reinterpret_cast<res_dtype_t*>(data0 + offsets[0]);
      auto p1 = reinterpret_cast<arg0_dtype_t*>(data1 + offsets[1]);
      auto p2 = reinterpret_cast<arg1_dtype_t*>(data2 + offsets[2]);
      *p0 = c10::convert<res_dtype_t>(f(c10::convert<arg0_t>(*p1), c10::convert<arg1_t>(*p2)));
    }
    if (y_remain != 0 && blockIdx.y < y_remain) {
      offsets[0] = (gridDim.y * y_t + blockIdx.y) * stride20 + row_offset0;
      offsets[1] = (gridDim.y * y_t + blockIdx.y) * stride21 + row_offset1;
      auto p0 = reinterpret_cast<res_dtype_t*>(data0 + offsets[0]);
      auto p1 = reinterpret_cast<arg0_dtype_t*>(data1 + offsets[1]);
      auto p2 = reinterpret_cast<arg1_dtype_t*>(data2 + offsets[2]);
      *p0 = c10::convert<res_dtype_t>(f(c10::convert<arg0_t>(*p1), c10::convert<arg1_t>(*p2)));
    }
  }
}

template<int nt, int vt, typename res_t , typename arg0_t, typename arg1_t, typename func_t, typename index_t, typename stride_t>
static void launch_legacy_kernel_maca_1_2_cast(
    int64_t N,
    char* data0, char* data1, char* data2,
    ScalarType st0, ScalarType st1, ScalarType st2,
    index_t size0,
    stride_t stride00, stride_t stride01, stride_t stride02,
    const func_t& f) {
  // ndim = 1, arity = 1
  TORCH_INTERNAL_ASSERT(N >= 0 && N <= std::numeric_limits<int32_t>::max());
  if (N == 0) {
    return;
  }
  dim3 block(nt);
  dim3 grid((N + block.x * vt - 1) / (block.x * vt));
  auto stream = at::cuda::getCurrentCUDAStream();
  fetch_and_cast_check(st0);
  fetch_and_cast_check(st1);
  fetch_and_cast_check(st2);
  elementwise_kernel_1_2_cast<nt, vt, res_t, arg0_t, arg1_t, func_t, index_t, stride_t><<<grid, block, 0, stream>>>(
      N, data0, data1, data2, st0, st1, st2, size0, stride00, stride01, stride02, f);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template<int nt, int vt, typename res_t , typename arg0_t, typename arg1_t, typename func_t, typename index_t, typename stride_t>
static void launch_legacy_kernel_maca_2_2_cast(
    int64_t N,
    char* data0, char* data1, char* data2,
    ScalarType st0, ScalarType st1, ScalarType st2,
    index_t size0, index_t size1,
    stride_t stride00, stride_t stride01, stride_t stride02,
    stride_t stride10, stride_t stride11, stride_t stride12,
    const func_t& f) {
  // ndim = 1, arity = 1
  TORCH_INTERNAL_ASSERT(N >= 0 && N <= std::numeric_limits<int32_t>::max());
  if (N == 0) {
    return;
  }
  dim3 block(nt);
  dim3 grid((N + block.x * vt - 1) / (block.x * vt));
  auto stream = at::cuda::getCurrentCUDAStream();
  fetch_and_cast_check(st0);
  fetch_and_cast_check(st1);
  fetch_and_cast_check(st2);
  elementwise_kernel_2_2_cast<nt, vt, res_t, arg0_t, arg1_t, func_t, index_t, stride_t><<<grid, block, 0, stream>>>(
      N, data0, data1, data2, st0, st1, st2, size0, size1, stride00, stride01, stride02, stride10, stride11, stride12, f);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template<int nt, int vt, typename res_t , typename arg0_t, typename arg1_t, typename func_t, typename index_t, typename stride_t>
static void launch_legacy_kernel_maca_3_2_cast_broadcast(
    int64_t N,
    char* data0, char* data1, char* data2,
    ScalarType st0, ScalarType st1, ScalarType st2,
    index_t size0, index_t size1, index_t size2,
    stride_t stride00, stride_t stride01, stride_t stride02,
    stride_t stride10, stride_t stride11, stride_t stride12,
    stride_t stride20, stride_t stride21, stride_t stride22,
    const func_t& f) {
  // ndim = 3, arity = 2
  TORCH_INTERNAL_ASSERT(N >= 0 && N <= std::numeric_limits<int32_t>::max());
  // check type out of kernel to prevent call the assert func in device to reduce private memory
  fetch_and_cast_check(st0);
  fetch_and_cast_check(st1);
  fetch_and_cast_check(st2);
  if (N == 0) {
    return;
  }
  constexpr int block_dim_x = 64;
  size_t load_num = size0 / block_dim_x;
  TORCH_INTERNAL_ASSERT(load_num > 0);
  int vec_data_0 = getVectorizedAlignment<res_t>((void*)data0, load_num);
  int vec_data_1 = getVectorizedAlignment<arg0_t>((void*)data1, load_num);
  int vec_data_2 = getVectorizedAlignment<arg1_t>((void*)data2, load_num);
  int vec = std::min(std::min(vec_data_0, vec_data_1), vec_data_2);
  dim3 block(block_dim_x);
  int grid_dim_x = load_num / vec;
  TORCH_INTERNAL_ASSERT(grid_dim_x > 0);
  int grid_dim_y = getMaxGridSize(grid_dim_x, size1);
  TORCH_INTERNAL_ASSERT(grid_dim_y > 0);
  dim3 grid(grid_dim_x, grid_dim_y, size2);
  int y_t = size1 / grid_dim_y;
  TORCH_INTERNAL_ASSERT(y_t > 0);
  auto stream = at::cuda::getCurrentCUDAStream();
  // double check
  if (st0 == ScalarType::Float && st1 == ScalarType::Half && st2 == ScalarType::Float) {
    if (vec == 8) {
      elementwise_kernel_3_2_cast_broadcast<nt, 8, res_t, arg0_t, arg1_t, func_t, index_t, stride_t, float, c10::Half, float><<<grid, block, 0, stream>>>(
        N, data0, data1, data2, st0, st1, st2, size0, size1, size2, stride00, stride01, stride02, stride10, stride11, stride12, stride20, stride21, stride22, f, y_t);
    } else if (vec == 4) {
      elementwise_kernel_3_2_cast_broadcast<nt, 4, res_t, arg0_t, arg1_t, func_t, index_t, stride_t, float, c10::Half, float><<<grid, block, 0, stream>>>(
        N, data0, data1, data2, st0, st1, st2, size0, size1, size2, stride00, stride01, stride02, stride10, stride11, stride12, stride20, stride21, stride22, f, y_t);
    } else if (vec == 2) {
      elementwise_kernel_3_2_cast_broadcast<nt, 2, res_t, arg0_t, arg1_t, func_t, index_t, stride_t, float, c10::Half, float><<<grid, block, 0, stream>>>(
        N, data0, data1, data2, st0, st1, st2, size0, size1, size2, stride00, stride01, stride02, stride10, stride11, stride12, stride20, stride21, stride22, f, y_t);
    } else {
      elementwise_kernel_3_2_cast_broadcast<nt, 1, res_t, arg0_t, arg1_t, func_t, index_t, stride_t, float, c10::Half, float><<<grid, block, 0, stream>>>(
        N, data0, data1, data2, st0, st1, st2, size0, size1, size2, stride00, stride01, stride02, stride10, stride11, stride12, stride20, stride21, stride22, f, y_t);
    }
  } else {
    TORCH_CHECK(false, "elementwise kernel 3_2 cast not supported!");
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template<int nt, int vt, typename res_t , typename arg0_t, typename arg1_t, typename func_t, typename index_t, typename stride_t>
static void launch_legacy_kernel_maca_3_2_cast_broadcast_dim2(
    int64_t N,
    char* data0, char* data1, char* data2,
    ScalarType st0, ScalarType st1, ScalarType st2,
    index_t size0, index_t size1, index_t size2,
    stride_t stride00, stride_t stride01, stride_t stride02,
    stride_t stride10, stride_t stride11, stride_t stride12,
    stride_t stride20, stride_t stride21, stride_t stride22,
    const func_t& f) {
  // ndim = 3, arity = 2
  TORCH_INTERNAL_ASSERT(N >= 0 && N <= std::numeric_limits<int32_t>::max());
  // check type out of kernel to prevent call the assert func in device to reduce private memory
  fetch_and_cast_check(st0);
  fetch_and_cast_check(st1);
  fetch_and_cast_check(st2);
  if (N == 0) {
    return;
  }
  constexpr int block_dim_x = 64;
  size_t load_num = size0 / block_dim_x;
  TORCH_INTERNAL_ASSERT(load_num > 0);
  int vec_data_0 = getVectorizedAlignment<res_t>((void*)data0, load_num);
  int vec_data_1 = getVectorizedAlignment<arg0_t>((void*)data1, load_num);
  int vec_data_2 = getVectorizedAlignment<arg1_t>((void*)data2, load_num);
  int vec = std::min(std::min(vec_data_0, vec_data_1), vec_data_2);
  // same as launch_legacy_kernel_maca_3_2_broadcast_dim2_contiguous
  dim3 block(block_dim_x);
  int grid_dim_x = load_num / vec;
  TORCH_INTERNAL_ASSERT(grid_dim_x > 0);
  int grid_dim_y = getMaxGridSize(grid_dim_x * size1, size2);
  TORCH_INTERNAL_ASSERT(grid_dim_y > 0);
  int y_t = size2 / grid_dim_y;
  grid_dim_y = std::ceil(float(size2)/float(y_t));
  y_t = size2 / grid_dim_y;
  // int y_remain = size2 % grid_dim_y;
  dim3 grid(grid_dim_x, grid_dim_y, size1);
  TORCH_INTERNAL_ASSERT(y_t > 0);
  auto stream = at::cuda::getCurrentCUDAStream();
  // double check
  if (st0 == ScalarType::Float && st1 == ScalarType::Half && st2 == ScalarType::Float) {
    if (vec == 8) {
      elementwise_kernel_3_2_cast_broadcast_dim2<nt, 8, res_t, arg0_t, arg1_t, func_t, index_t, stride_t, float, c10::Half, float><<<grid, block, 0, stream>>>(
        N, data0, data1, data2, st0, st1, st2, size0, size1, size2, stride00, stride01, stride02, stride10, stride11, stride12, stride20, stride21, stride22, f, y_t);
    } else if (vec == 4) {
      elementwise_kernel_3_2_cast_broadcast_dim2<nt, 4, res_t, arg0_t, arg1_t, func_t, index_t, stride_t, float, c10::Half, float><<<grid, block, 0, stream>>>(
        N, data0, data1, data2, st0, st1, st2, size0, size1, size2, stride00, stride01, stride02, stride10, stride11, stride12, stride20, stride21, stride22, f, y_t);
    } else if (vec == 2) {
      elementwise_kernel_3_2_cast_broadcast_dim2<nt, 2, res_t, arg0_t, arg1_t, func_t, index_t, stride_t, float, c10::Half, float><<<grid, block, 0, stream>>>(
        N, data0, data1, data2, st0, st1, st2, size0, size1, size2, stride00, stride01, stride02, stride10, stride11, stride12, stride20, stride21, stride22, f, y_t);
    } else {
      elementwise_kernel_3_2_cast_broadcast_dim2<nt, 1, res_t, arg0_t, arg1_t, func_t, index_t, stride_t, float, c10::Half, float><<<grid, block, 0, stream>>>(
        N, data0, data1, data2, st0, st1, st2, size0, size1, size2, stride00, stride01, stride02, stride10, stride11, stride12, stride20, stride21, stride22, f, y_t);
    }
  } else {
    TORCH_CHECK(false, "elementwise kernel 3_2 cast not supported!");
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
template<int nt, int vt, typename res_t , typename arg0_t, typename arg1_t, typename func_t, typename index_t, typename stride_t>
static void launch_legacy_kernel_maca_3_2_cast(
    int64_t N,
    char* data0, char* data1, char* data2,
    ScalarType st0, ScalarType st1, ScalarType st2,
    index_t size0, index_t size1, index_t size2,
    stride_t stride00, stride_t stride01, stride_t stride02,
    stride_t stride10, stride_t stride11, stride_t stride12,
    stride_t stride20, stride_t stride21, stride_t stride22,
    const func_t& f) {
  // ndim = 3, arity = 2
  TORCH_INTERNAL_ASSERT(N >= 0 && N <= std::numeric_limits<int32_t>::max());
  // check type out of kernel to prevent call the assert func in device to reduce private memory
  fetch_and_cast_check(st0);
  fetch_and_cast_check(st1);
  fetch_and_cast_check(st2);
  if (N == 0) {
    return;
  }

  auto stream = at::cuda::getCurrentCUDAStream();
  if( maca_likely(!at::maca::get_maca_disable_element_kernel_cast_opt_tile())){
    unsigned int dim_with_minimum_stride_arg0=0,dim_with_minimum_stride_arg1=0,dim_with_minimum_stride_arg2=0;
    unsigned int strides_arg0[3]={stride00,stride10,stride20};
    unsigned int strides_arg1[3]={stride01,stride11,stride21};
    unsigned int strides_arg2[3]={stride02,stride12,stride22};

    auto min_stride_arg0=std::min_element(strides_arg0, strides_arg0 + 3);

    auto min_stride_arg1=std::min_element(strides_arg1, strides_arg1 + 3);
    auto min_stride_arg2=std::min_element(strides_arg2, strides_arg2 + 3);

    if(*min_stride_arg0<=40 && *min_stride_arg1<=40 && *min_stride_arg2<=40 && (size0*size1*size2>64*64*64))
    {

      dim_with_minimum_stride_arg0 = std::distance(strides_arg0, min_stride_arg0);
      dim_with_minimum_stride_arg1 = std::distance(strides_arg1, min_stride_arg1);
      dim_with_minimum_stride_arg2 = std::distance(strides_arg2, min_stride_arg2);

      if(dim_with_minimum_stride_arg0==0 && dim_with_minimum_stride_arg1==0 && dim_with_minimum_stride_arg2==1){//001
        dim3 grid ( ceilf((float)size0/(float)64),ceilf((float)(size1)/(float)64) ,size2);
        dim3 block (512)  ;

        elementwise_kernel_3_2_cast_without_assert_opt_tile<func_t,index_t,stride_t,
        arg0_t,arg1_t,res_t,false><<<grid, block, 0, stream>>>(
          size0,size1,data0, data1, data2, st0,st1,st2,stride00,stride10, stride20,stride01,stride11, stride21,stride02,stride12, stride22,  f);
      }else if(dim_with_minimum_stride_arg0==0 && dim_with_minimum_stride_arg1==0 && dim_with_minimum_stride_arg2==2){//002
        dim3 grid ( ceilf((float)size0/(float)64),ceilf((float)(size2)/(float)64) ,size1);
        dim3 block (512)  ;
        elementwise_kernel_3_2_cast_without_assert_opt_tile<func_t,index_t,stride_t,
        arg0_t,arg1_t,res_t,false><<<grid, block, 0, stream>>>(
          size0,size2,data0, data1, data2, st0,st1,st2,stride00,stride20, stride10,stride01,stride21, stride11,stride02,stride22, stride12,  f);
      }else{
        dim3 block(nt);
        dim3 grid((N + block.x * vt - 1) / (block.x * vt));
        elementwise_kernel_3_2_cast_without_assert<nt, vt, res_t, arg0_t, arg1_t, func_t, index_t, stride_t><<<grid, block, 0, stream>>>(
        N, data0, data1, data2, st0, st1, st2, size0, size1, size2, stride00, stride01, stride02, stride10, stride11, stride12, stride20, stride21, stride22, f);
      }
    }else{
      dim3 block(nt);
      dim3 grid((N + block.x * vt - 1) / (block.x * vt));
      elementwise_kernel_3_2_cast_without_assert<nt, vt, res_t, arg0_t, arg1_t, func_t, index_t, stride_t><<<grid, block, 0, stream>>>(
        N, data0, data1, data2, st0, st1, st2, size0, size1, size2, stride00, stride01, stride02, stride10, stride11, stride12, stride20, stride21, stride22, f);
    }
  }else{
      dim3 block(nt);
      dim3 grid((N + block.x * vt - 1) / (block.x * vt));
      elementwise_kernel_3_2_cast_without_assert<nt, vt, res_t, arg0_t, arg1_t, func_t, index_t, stride_t><<<grid, block, 0, stream>>>(
        N, data0, data1, data2, st0, st1, st2, size0, size1, size2, stride00, stride01, stride02, stride10, stride11, stride12, stride20, stride21, stride22, f);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}


template<int nt, int vt, typename res_t , typename arg0_t, typename arg1_t, typename func_t, typename index_t, typename stride_t,
         typename res_dtype_t=float, typename arg0_dtype_t=float, typename arg1_dtype_t=float>
static void launch_legacy_kernel_maca_4_2_cast_broadcast(
    int64_t N,
    char* data0, char* data1, char* data2,
    ScalarType st0, ScalarType st1, ScalarType st2,
    index_t size0, index_t size1, index_t size2, index_t size3,
    stride_t stride00, stride_t stride01, stride_t stride02,
    stride_t stride10, stride_t stride11, stride_t stride12,
    stride_t stride20, stride_t stride21, stride_t stride22,
    stride_t stride30, stride_t stride31, stride_t stride32,
    const func_t& f) {
  // ndim = 3, arity = 2
  TORCH_INTERNAL_ASSERT(N >= 0 && N <= std::numeric_limits<int32_t>::max());
  // check type out of kernel to prevent call the assert func in device to reduce private memory
  fetch_and_cast_check(st0);
  fetch_and_cast_check(st1);
  fetch_and_cast_check(st2);
  if (N == 0) {
    return;
  }
  dim3 block(nt);
  dim3 grid((N + block.x * vt - 1) / (block.x * vt));
  auto stream = at::cuda::getCurrentCUDAStream();

  if (st0 == ScalarType::Float && st1 == ScalarType::Float && st2 == ScalarType::BFloat16) {
     elementwise_kernel_4_2_cast_broadcast<nt, vt, res_t, arg0_t, arg1_t, func_t, index_t, stride_t,
     float, float, at::BFloat16>
     <<<grid, block, 0, stream>>>(
      N, data0, data1, data2, st0, st1, st2,
      size0, size1, size2, size3,
      stride00, stride01, stride02,
      stride10, stride11, stride12,
      stride20, stride21, stride22,
      stride30, stride31, stride32,
      f);
  } else if (st0 == ScalarType::Float && st1 == ScalarType::Float && st2 == ScalarType::Half) {
     elementwise_kernel_4_2_cast_broadcast<nt, vt, res_t, arg0_t, arg1_t, func_t, index_t, stride_t,
     float, float, at::Half>
     <<<grid, block, 0, stream>>>(
      N, data0, data1, data2, st0, st1, st2,
      size0, size1, size2, size3,
      stride00, stride01, stride02,
      stride10, stride11, stride12,
      stride20, stride21, stride22,
      stride30, stride31, stride32,
      f);
  } else {
    TORCH_CHECK(false, "elementwise kernel 4_2 cast broadcast not supported!");
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}


template<int nt, int vt, typename res_t , typename arg0_t, typename arg1_t, typename func_t, typename index_t, typename stride_t>
static void launch_legacy_kernel_maca_4_2_cast(
    int64_t N,
    char* data0, char* data1, char* data2,
    ScalarType st0, ScalarType st1, ScalarType st2,
    index_t size0, index_t size1, index_t size2, index_t size3,
    stride_t stride00, stride_t stride01, stride_t stride02,
    stride_t stride10, stride_t stride11, stride_t stride12,
    stride_t stride20, stride_t stride21, stride_t stride22,
    stride_t stride30, stride_t stride31, stride_t stride32,
    const func_t& f) {
  // ndim = 3, arity = 2
  TORCH_INTERNAL_ASSERT(N >= 0 && N <= std::numeric_limits<int32_t>::max());
  // check type out of kernel to prevent call the assert func in device to reduce private memory
  fetch_and_cast_check(st0);
  fetch_and_cast_check(st1);
  fetch_and_cast_check(st2);
  if (N == 0) {
    return;
  }
  dim3 block(nt);
  dim3 grid((N + block.x * vt - 1) / (block.x * vt));
  auto stream = at::cuda::getCurrentCUDAStream();
  elementwise_kernel_4_2_cast_without_assert<nt, vt, res_t, arg0_t, arg1_t, func_t, index_t, stride_t><<<grid, block, 0, stream>>>(
      N, data0, data1, data2, st0, st1, st2,
      size0, size1, size2, size3,
      stride00, stride01, stride02,
      stride10, stride11, stride12,
      stride20, stride21, stride22,
      stride30, stride31, stride32,
      f);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename func_t>
void gpu_kernel_impl_maca_arity2(TensorIteratorBase& iter, const func_t& f) {
  using traits = function_traits<func_t>;
  using arg0_t = typename traits::result_type;
  using arg1_t = typename traits::template arg<0>::type;
  using arg2_t = typename traits::template arg<1>::type;
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
  assert(narity == 2);

  bool can_vectorize4 = true;
  can_vectorize4 =  can_vectorize4 && (reinterpret_cast<uint64_t>(data[0])%4==0);
  can_vectorize4 =  can_vectorize4 && (reinterpret_cast<uint64_t>(data[1])%4==0);
  can_vectorize4 =  can_vectorize4 && (reinterpret_cast<uint64_t>(data[2])%4==0);

  if (!dynamic_casting) {
    if (contiguous) {
      get_elementwise_info<narity + 1>(ndim, narity, offset_calc, dtypes, "p_e_vec_1_2", f);
      launch_vectorized_kernel(numel, f, data);
    } else {
        if (ndim == 1) {
          bool disable_1_2_broadcast = at::maca::get_maca_disable_elementwise_1_2_broadcast_kernel();
          if(((sizeof(arg0_t) == 2 && sizeof(arg1_t) == 2 && sizeof(arg2_t) == 2) || (sizeof(arg0_t) == 4 && sizeof(arg1_t) == 4 && sizeof(arg2_t) == 4) ||
            (sizeof(arg0_t) == 2 && sizeof(arg1_t) == 2 && sizeof(arg2_t) == 1) || (sizeof(arg0_t) == 4 && sizeof(arg1_t) == 4 && sizeof(arg2_t) == 1)) &&
            offset_calc.strides_[0][0] == sizeof(arg0_t) && ((offset_calc.strides_[0][1] == sizeof(arg1_t) && offset_calc.strides_[0][2] == 0) || 
            (offset_calc.strides_[0][1] == 0 && offset_calc.strides_[0][2] == sizeof(arg2_t))) &&
            offset_calc.sizes_[0].divisor % C10_WARP_SIZE == 0 && maca_likely(!disable_1_2_broadcast)){
            get_elementwise_info<narity + 1>(ndim, narity, offset_calc, dtypes, "p_e_launch_legacy_kernel_maca_1_2_broadcast", f);
            launch_legacy_kernel_maca_1_2_broadcast<8*C10_WARP_SIZE, unroll_factor, arg0_t, typename traits::template arg<0>::type, typename traits::template arg<1>::type>(
              numel,
              data[0], data[1], data[2], // data
              offset_calc.sizes_[0].divisor,
              offset_calc.strides_[0][0], offset_calc.strides_[0][1], offset_calc.strides_[0][2],
              f
            );
          } else {
            get_elementwise_info<narity + 1>(ndim, narity, offset_calc, dtypes, "p_e_launch_legacy_kernel_maca_1_2", f);
            launch_legacy_kernel_maca_1_2<128, unroll_factor, arg0_t, typename traits::template arg<0>::type, typename traits::template arg<1>::type>(
              numel,
              data[0], data[1], data[2], // data
              offset_calc.sizes_[0].divisor,
              offset_calc.strides_[0][0], offset_calc.strides_[0][1], offset_calc.strides_[0][2],
              f);
          }
        } else if (ndim == 2) {
          auto s0 = offset_calc.sizes_[0].divisor;
          auto s1 = offset_calc.sizes_[1].divisor;
          size_t st00 = offset_calc.strides_[0][0], st10 = offset_calc.strides_[1][0];
          size_t st01 = offset_calc.strides_[0][1], st11 = offset_calc.strides_[1][1];
          size_t st02 = offset_calc.strides_[0][2], st12 = offset_calc.strides_[1][2];

          bool disable_opt = at::maca::get_maca_disable_elementwise_kernel_2_2_broadcast();
          bool disable_align = at::maca::get_maca_disable_elementwise_kernel_2_2_align();
          bool disable_2_2_template = at::maca::get_maca_disable_elementwise_kernel_2_2_template();
          bool disable_2_2_broadcast_arity1_dim0 = at::maca::get_maca_disable_elementwise_kernel_2_2_broadcast_arity1_dim0();
          bool disable_2_2_arity2_transpose = at::maca::get_maca_disable_elementwise_kernel_2_2_arity2_transpose();

          if(((sizeof(arg0_t) == 2 && sizeof(arg1_t) == 2 && sizeof(arg2_t) == 2) || (sizeof(arg0_t) == 4 && sizeof(arg1_t) == 4 && sizeof(arg2_t) == 4)) &&
              offset_calc.strides_[0][0] == sizeof(arg0_t) && offset_calc.strides_[0][1] == sizeof(arg1_t) && offset_calc.strides_[0][2] == sizeof(arg2_t) &&
              offset_calc.strides_[1][0] == offset_calc.sizes_[0].divisor * sizeof(arg0_t) && offset_calc.strides_[1][0] % 4 ==0 &&
              offset_calc.strides_[1][1] > 0 && offset_calc.strides_[1][1] % 4 ==0 && offset_calc.strides_[1][2] > 0 && offset_calc.strides_[1][2] % 4 ==0 &&
              offset_calc.sizes_[0].divisor % 8 == 0 && maca_likely(!disable_align)){
            get_elementwise_info<narity + 1>(ndim, narity, offset_calc, dtypes, "p_e_launch_legacy_kernel_maca_2_2_align", f);
            launch_legacy_kernel_maca_2_2_align<8*C10_WARP_SIZE, unroll_factor, arg0_t, typename traits::template arg<0>::type, typename traits::template arg<1>::type>(
              numel,
              data[0], data[1], data[2],  // data
              offset_calc.sizes_[0].divisor, offset_calc.sizes_[1].divisor,
              offset_calc.strides_[0][0], offset_calc.strides_[0][1], offset_calc.strides_[0][2],
              offset_calc.strides_[1][0], offset_calc.strides_[1][1], offset_calc.strides_[1][2],
              f);
          } else if (((sizeof(arg0_t) == 2 && sizeof(arg1_t) == 2 && sizeof(arg2_t) == 2) || (sizeof(arg0_t) == 4 && sizeof(arg1_t) == 4 && sizeof(arg2_t) == 4)) &&
              offset_calc.strides_[0][0] == sizeof(arg0_t) && (offset_calc.strides_[0][1] == sizeof(arg1_t) ||
              offset_calc.strides_[0][2] == sizeof(arg2_t)) &&
              (((offset_calc.strides_[0][2] == sizeof(arg2_t) && offset_calc.strides_[1][2] == 0) ||  // broadcast arg1
                (offset_calc.strides_[0][2] == 0 && offset_calc.strides_[1][2] == sizeof(arg2_t))) ||   // broacast arg1 uncontiguous
               ((offset_calc.strides_[0][1] == sizeof(arg1_t) && offset_calc.strides_[1][1] == 0) ||  // broadcast arg2
                (offset_calc.strides_[0][1] == 0 && offset_calc.strides_[1][1] == sizeof(arg1_t)))) &&  // broadcast arg2 uncontiguous
              offset_calc.strides_[1][0] == (offset_calc.sizes_[0].divisor * sizeof(arg0_t)) &&
              offset_calc.sizes_[0].divisor >= C10_WARP_SIZE && maca_likely(!disable_opt)){ // float, float16, bfloat16
              // XXX(): one prerequisite for broadcast is divisible by warp_size, otherwise will cause
              // partial memory write and alignment fault.
              // 8 * C10_WARP_SIZE to be considered as best block size.
              // pattern
              // shape = [s0, s1] with s0 >= C10_WARP_SIZE
              // output_stride = [sizeoof(arg0_t), sizeof(arg0_t) * s0]
              // arg0_stride = [sizeof(arg1_t), 0] or [0, sizeof(arg1_t)] with arg1_stride = [sizeof(arg2_t), *]
              // arg0_stride = [sizeof(arg1_t), *] with arg1_stride = [sizeof(arg2_t), 0] or [0, sizeof(arg2_t)]
            get_elementwise_info<narity + 1>(ndim, narity, offset_calc, dtypes, "p_e_launch_legacy_kernel_maca_2_2_broadcast", f);
            launch_legacy_kernel_maca_2_2_broadcast<8*C10_WARP_SIZE, unroll_factor, arg0_t, typename traits::template arg<0>::type, typename traits::template arg<1>::type>(
              numel,
              data[0], data[1], data[2],  // data
              offset_calc.sizes_[0].divisor, offset_calc.sizes_[1].divisor,
              offset_calc.strides_[0][0], offset_calc.strides_[0][1], offset_calc.strides_[0][2],
              offset_calc.strides_[1][0], offset_calc.strides_[1][1], offset_calc.strides_[1][2],
              f);
          } else if( can_vectorize4 &&
                    (sizeof(arg0_t) == 2 && sizeof(arg1_t) == 2 && sizeof(arg2_t) == 2) &&
                    offset_calc.strides_[0][0] == 2 && offset_calc.strides_[1][0] == (offset_calc.sizes_[0].divisor * 2) &&
                    offset_calc.strides_[0][2] == 2 && offset_calc.strides_[1][2] == offset_calc.strides_[1][0] &&
                    offset_calc.strides_[0][1] == 2 && offset_calc.strides_[1][1] % (offset_calc.sizes_[0].divisor*2) == 0 &&
                    offset_calc.strides_[1][1] > (offset_calc.sizes_[0].divisor*2) &&
                    offset_calc.sizes_[0].divisor % C10_WARP_SIZE == 0){
            get_elementwise_info<narity + 1>(ndim, narity, offset_calc, dtypes, "p_e_launch_legacy_kernel_maca_2_2_interval", f);
            launch_legacy_kernel_maca_2_2_interval_arity1<128, 8, arg0_t, typename traits::template arg<0>::type, typename traits::template arg<1>::type>(
              numel,
              data[0], data[1], data[2],  // data
              offset_calc.sizes_[0].divisor, offset_calc.sizes_[1].divisor,
              offset_calc.strides_[0][0], offset_calc.strides_[0][1], offset_calc.strides_[0][2],
              offset_calc.strides_[1][0], offset_calc.strides_[1][1], offset_calc.strides_[1][2],
              f
            );
          } else if (((sizeof(arg0_t) == 2 && sizeof(arg1_t) == 2 && sizeof(arg2_t) == 2) ||
                      (sizeof(arg0_t) == 4 && sizeof(arg1_t) == 4 && sizeof(arg2_t) == 4) ||
                      (sizeof(arg0_t) == 2 && sizeof(arg1_t) == 2 && sizeof(arg2_t) == 1) ||
                      (sizeof(arg0_t) == 4 && sizeof(arg1_t) == 4 && sizeof(arg2_t) == 1)) &&
                       st00 == sizeof(arg0_t) && st10 == s0 * sizeof(arg0_t)  &&
                       st01 == sizeof(arg1_t) && st11 % 4 == 0 &&
                       st02 == 0              &&
                       (reinterpret_cast<uintptr_t>(data[0]) % 4) == 0 &&
                       (reinterpret_cast<uintptr_t>(data[1]) % 4) == 0 &&
                       (reinterpret_cast<uintptr_t>(data[2]) % 4) == 0 &&
                       s0 >= C10_WARP_SIZE && s0 % 4 == 0 &&
                       maca_likely(!disable_2_2_broadcast_arity1_dim0)) {
              // out: contiguous
              // arg0: dim0 contiguous
              // arg1: dim0 broadcast
              get_elementwise_info<narity + 1>(ndim, narity, offset_calc, dtypes, "p_e_launch_legacy_kernel_maca_2_2_broadcast_arity1_dim0", f);
              launch_legacy_kernel_maca_2_2_broadcast_arity1_dim0<128, unroll_factor, arg0_t, typename traits::template arg<0>::type, typename traits::template arg<1>::type>(
                numel,
              data[0], data[1], data[2],  // data
              offset_calc.sizes_[0].divisor, offset_calc.sizes_[1].divisor,
              offset_calc.strides_[0][0], offset_calc.strides_[0][1], offset_calc.strides_[0][2],
              offset_calc.strides_[1][0], offset_calc.strides_[1][1], offset_calc.strides_[1][2],
              f);
           } else if (((sizeof(arg0_t) == 2 && sizeof(arg1_t) == 2 && sizeof(arg2_t) == 2) ||
                      (sizeof(arg0_t) == 4 && sizeof(arg1_t) == 4 && sizeof(arg2_t) == 4)) &&
                      st00 == sizeof(arg0_t) && st10 == s0 * sizeof(arg0_t)  &&
                      st01 == sizeof(arg1_t) && st11 == s0 * sizeof(arg1_t)  &&
                      st02 == s1 * sizeof(arg2_t) && st12 == sizeof(arg2_t)  &&
                      s0 >= C10_WARP_SIZE && s0 % 4 == 0 &&
                      s1 >= C10_WARP_SIZE && s1 % 4 == 0 &&
                      (reinterpret_cast<uintptr_t>(data[0]) % 4) == 0 &&
                      (reinterpret_cast<uintptr_t>(data[1]) % 4) == 0 &&
                      (reinterpret_cast<uintptr_t>(data[2]) % 4) == 0 &&
                      maca_likely(!disable_2_2_arity2_transpose)) {
              // out:  contiguous
              // arg0: contiguous
              // arg1: transpose
              get_elementwise_info<narity + 1>(ndim, narity, offset_calc, "p_e_launch_legacy_kernel_maca_2_2_arity2_tranpose", f);
              launch_legacy_kernel_maca_2_2_arity2_tranpose<128, unroll_factor, arg0_t, typename traits::template arg<0>::type, typename traits::template arg<1>::type>(
                numel,
              data[0], data[1], data[2],  // data
              offset_calc.sizes_[0].divisor, offset_calc.sizes_[1].divisor,
              offset_calc.strides_[0][0], offset_calc.strides_[0][1], offset_calc.strides_[0][2],
              offset_calc.strides_[1][0], offset_calc.strides_[1][1], offset_calc.strides_[1][2],
              f);
          } else {
            if (maca_likely(!disable_2_2_template) && ((sizeof(arg0_t) == 2 && sizeof(arg1_t) == 2 && sizeof(arg2_t) == 2) ||
                (sizeof(arg0_t) == 4 && sizeof(arg1_t) == 4 && sizeof(arg2_t) == 4))) {
              get_elementwise_info<narity + 1>(ndim, narity, offset_calc, dtypes, "p_e_launch_legacy_kernel_maca_2_2_template", f);
              launch_legacy_kernel_maca_2_2_template<128, unroll_factor, arg0_t, typename traits::template arg<0>::type, typename traits::template arg<1>::type>(
                numel,
                data[0], data[1], data[2],  // data
                offset_calc.sizes_[0].divisor, offset_calc.sizes_[1].divisor,
                offset_calc.strides_[0][0], offset_calc.strides_[0][1], offset_calc.strides_[0][2],
                offset_calc.strides_[1][0], offset_calc.strides_[1][1], offset_calc.strides_[1][2],
                f);
            } else {
              get_elementwise_info<narity + 1>(ndim, narity, offset_calc, dtypes, "p_e_launch_legacy_kernel_maca_2_2", f);
              launch_legacy_kernel_maca_2_2<128, unroll_factor, arg0_t, typename traits::template arg<0>::type, typename traits::template arg<1>::type>(
                numel,
                data[0], data[1], data[2],  // data
                offset_calc.sizes_[0].divisor, offset_calc.sizes_[1].divisor,
                offset_calc.strides_[0][0], offset_calc.strides_[0][1], offset_calc.strides_[0][2],
                offset_calc.strides_[1][0], offset_calc.strides_[1][1], offset_calc.strides_[1][2],
                f);
            }
          }
        } else if (ndim == 3) {
          auto s0 = offset_calc.sizes_[0].divisor;
          auto s1 = offset_calc.sizes_[1].divisor;
          auto s2 = offset_calc.sizes_[2].divisor;
          size_t st00 = offset_calc.strides_[0][0], st01 = offset_calc.strides_[0][1], st02 = offset_calc.strides_[0][2];
          size_t st10 = offset_calc.strides_[1][0], st11 = offset_calc.strides_[1][1], st12 = offset_calc.strides_[1][2];
          size_t st20 = offset_calc.strides_[2][0], st21 = offset_calc.strides_[2][1], st22 = offset_calc.strides_[2][2];
          bool disable_opt_3_2_arity2_transpose = at::maca::get_maca_disable_elementwise_3_2_arity2_transpose_kernel();
          bool disable_opt_3_2_arity2_transpose_dim02 = at::maca::get_maca_disable_elementwise_3_2_arity2_transpose_dim02_kernel();

          if (((sizeof(arg0_t) == 2 && sizeof(arg1_t) == 2 && sizeof(arg2_t) == 2) ||
              (sizeof(arg0_t) == 4 && sizeof(arg1_t) == 4 && sizeof(arg2_t) == 4)) &&
              st00 == sizeof(arg0_t) && st10 == sizeof(arg0_t) * s0 && st20 == sizeof(arg0_t) * s0 * s1 &&
              st01 == st00 && st11 == st10 && st21 == st20 &&
              st02 == sizeof(arg2_t) * s1 && st12 == sizeof(arg2_t) && st22 == sizeof(arg2_t) * s0 * s1 &&
              (reinterpret_cast<uintptr_t>(data[0]) % 4) == 0 &&
              (reinterpret_cast<uintptr_t>(data[1]) % 4) == 0 &&
              (reinterpret_cast<uintptr_t>(data[2]) % 4) == 0 &&
              s0 >= 20 && s1 >= 20 && maca_likely(!disable_opt_3_2_arity2_transpose)) {
              // 3-2 arity2 transpose: support output shape=(0, 1, 2), input shape=(0, 1, 2) & (1, 0, 2)
              get_elementwise_info<narity + 1>(ndim, narity, offset_calc, dtypes, "p_e_launch_legacy_kernel_maca_3_2_arity2_transpose", f);
              launch_legacy_kernel_maca_3_2_arity2_transpose<128, 8, arg0_t, typename traits::template arg<0>::type, typename traits::template arg<1>::type>(
                numel,
                data[0], data[1], data[2], // data
                offset_calc.sizes_[0].divisor, offset_calc.sizes_[1].divisor, offset_calc.sizes_[2].divisor,
                offset_calc.strides_[0][0], offset_calc.strides_[0][1], offset_calc.strides_[0][2],
                offset_calc.strides_[1][0], offset_calc.strides_[1][1], offset_calc.strides_[1][2],
                offset_calc.strides_[2][0], offset_calc.strides_[2][1], offset_calc.strides_[2][2],
                f);
          } else if (((sizeof(arg0_t) == 2 && sizeof(arg1_t) == 2 && sizeof(arg2_t) == 2) || (sizeof(arg0_t) == 4 && sizeof(arg1_t) == 4 && sizeof(arg2_t) == 4)) &&
              st00 == sizeof(arg0_t) && st10 == sizeof(arg0_t) * s0 && st20 == sizeof(arg0_t) * s0 * s1 &&
              st01 == st00 && st11 == st10 && st21 == st20 && st02 == s2 * sizeof(arg2_t) && st22 == sizeof(arg2_t) &&
              s0 >= 20 && s2 >= 20 && s0 % 8 == 0 && s2 % 8 == 0 && maca_likely(!disable_opt_3_2_arity2_transpose_dim02)) {
              get_elementwise_info<narity + 1>(ndim, narity, offset_calc, dtypes, "p_e_launch_legacy_kernel_maca_3_2_arity2_transpose_dim02", f);
              launch_legacy_kernel_maca_3_2_arity2_transpose_dim02<128, 8, arg0_t, typename traits::template arg<0>::type, typename traits::template arg<1>::type>(
                numel,
                data[0], data[1], data[2], // data
                offset_calc.sizes_[0].divisor, offset_calc.sizes_[1].divisor, offset_calc.sizes_[2].divisor,
                offset_calc.strides_[0][0], offset_calc.strides_[0][1], offset_calc.strides_[0][2],
                offset_calc.strides_[1][0], offset_calc.strides_[1][1], offset_calc.strides_[1][2],
                offset_calc.strides_[2][0], offset_calc.strides_[2][1], offset_calc.strides_[2][2],
                f);
          } else {
            bool disable_opt_3_2_broadcast_dim2_arg0_contiguous = at::maca::get_maca_disable_elementwise_3_2_broadcast_dim2_arg0_contiguous_kernel();
            bool disable_opt_3_2_broadcast_dim1_contiguous = at::maca::get_maca_disable_elementwise_3_2_broadcast_dim1_contiguous_kernel();
            bool disable_opt_3_2_broadcast_dim1_uncontiguous = at::maca::get_maca_disable_elementwise_3_2_broadcast_dim1_uncontiguous_kernel();
            bool disable_opt_3_2_broadcast_dim2_uncontiguous = at::maca::get_maca_disable_elementwise_3_2_broadcast_dim2_uncontiguous_kernel();
            bool disable_opt_3_2_broadcast_dim2_contiguous = at::maca::get_maca_disable_elementwise_3_2_broadcast_dim2_contiguous_kernel();
            bool disable_opt_3_2_dim0_contiguous = at::maca::get_maca_disable_elementwise_3_2_dim0_contiguous_kernel();
            bool disable_3_2_broadcast_arg0_dim2_arg1_dim0 = at::maca::get_maca_disable_elementwise_3_2_broadcast_arg0_dim2_arg1_dim0_kernel();
            bool disable_opt_3_2_broadcast_dim0 = at::maca::get_maca_disable_elementwise_3_2_broadcast_dim0_kernel();
            bool disable_3_2_dim0_contiguous_arg1_dim1_broadcast = at::maca::get_maca_disable_elementwise_3_2_dim0_contiguous_arg1_dim1_broadcast_kernel();

            if (((sizeof(arg0_t) == 2 && sizeof(arg1_t) == 2 && sizeof(arg2_t) == 2) || (sizeof(arg0_t) == 4 && sizeof(arg1_t) == 4 && sizeof(arg2_t) == 4)) &&
                maca_likely(!disable_opt_3_2_broadcast_dim2_arg0_contiguous) && offset_calc.strides_[0][0] == sizeof(arg0_t) &&
                offset_calc.strides_[1][0] == s0 * sizeof(arg0_t) && offset_calc.strides_[2][0] == s0 * s1 * sizeof(arg0_t) &&
                offset_calc.strides_[0][1] == sizeof(arg0_t) && offset_calc.strides_[1][1] == s0 * sizeof(arg0_t) &&
                offset_calc.strides_[2][1] == s0 * s1 * sizeof(arg0_t) && offset_calc.strides_[0][2] != sizeof(arg0_t) &&
                offset_calc.strides_[1][2] != s0 * sizeof(arg0_t) && offset_calc.strides_[2][2] == 0 && s0 % C10_WARP_SIZE == 0) {
                // 3_2_broadcast in dim2 kernel
                // out: contiguous
                // arg0: contiguous
                // arg1: uncontiguou in dim0, broadcast in dim2
                // s0 divided by WARP_SIZE
                // TODO(): support s0 < WARP_SIZE as 3_2_broadcast_dim1_uncontiguous kernel
                // TODO(): check if support offset_calc.strides_[1][2] > 0 but not only offset_calc.strides_[1][2] % (s0 * sizeof(arg0_t)) == 0
                get_elementwise_info<narity + 1>(ndim, narity, offset_calc, dtypes, "p_e_launch_legacy_kernel_maca_3_2_broadcast_dim2_arg0_contiguous", f);
                launch_legacy_kernel_maca_3_2_broadcast_dim2_arg0_contiguous<128, unroll_factor, arg0_t, typename traits::template arg<0>::type, typename traits::template arg<1>::type>(
                  numel,
                  data[0], data[1], data[2], // data
                  offset_calc.sizes_[0].divisor, offset_calc.sizes_[1].divisor, offset_calc.sizes_[2].divisor,
                  offset_calc.strides_[0][0], offset_calc.strides_[0][1], offset_calc.strides_[0][2],
                  offset_calc.strides_[1][0], offset_calc.strides_[1][1], offset_calc.strides_[1][2],
                  offset_calc.strides_[2][0], offset_calc.strides_[2][1], offset_calc.strides_[2][2],
                  f);
            } else if(((sizeof(arg0_t) == 2 && sizeof(arg1_t) == 2 && sizeof(arg2_t) == 2) || (sizeof(arg0_t) == 4 && sizeof(arg1_t) == 4 && sizeof(arg2_t) == 4) ||
                (sizeof(arg0_t) == 4 && sizeof(arg1_t) == 4 && sizeof(arg2_t) == 1)) &&
                maca_likely(!disable_opt_3_2_broadcast_dim1_contiguous) && offset_calc.strides_[0][0] == sizeof(arg0_t) &&
                offset_calc.strides_[1][0] == s0 * sizeof(arg0_t) && offset_calc.strides_[2][0] == s0 * s1 * sizeof(arg0_t) &&
                ((offset_calc.strides_[0][1] == sizeof(arg1_t) && offset_calc.strides_[1][1] == s0 * sizeof(arg1_t) &&
                offset_calc.strides_[2][1] == s0 * s1 * sizeof(arg1_t) && offset_calc.strides_[0][2] == sizeof(arg2_t) &&
                offset_calc.strides_[1][2] == 0 && offset_calc.strides_[2][2] % (s0 * sizeof(arg2_t)) == 0) ||
                (offset_calc.strides_[0][1] == sizeof(arg1_t) && offset_calc.strides_[1][1] == 0 &&
                offset_calc.strides_[2][1] % (s0 * sizeof(arg1_t)) == 0 && offset_calc.strides_[0][2] == sizeof(arg2_t) &&
                offset_calc.strides_[1][2] == s0 * sizeof(arg2_t) && offset_calc.strides_[2][2] == s0 * s1 * sizeof(arg2_t)))
                && (s0 % C10_WARP_SIZE == 0 || (s0 < C10_WARP_SIZE && (s0 & (s0 -1)) != 0 && s0 % 8 == 0))){
                // 3_2_broadcast in dim1 contiguous kernel
                // out: contiguous
                // arg0: contiguous
                // arg1: contiguous, broadcast in dim1
                // or
                // out: contiguous
                // arg0: contiguous, broadcast in dim1
                // arg1: contiguous
                // s0 divided by WARP_SIZE
                // support s0 < WARP_SIZE and s0 is not the power of 2
                // TODO(): check if support offset_calc.strides_[2][2] > 0 but not only offset_calc.strides_[2][2] % (s0 * sizeof(arg0_t)) == 0
                get_elementwise_info<narity + 1>(ndim, narity, offset_calc, dtypes, "p_e_launch_legacy_kernel_maca_3_2_broadcast_dim1_contiguous", f);
                launch_legacy_kernel_maca_3_2_broadcast_dim1_contiguous<128, unroll_factor, arg0_t, typename traits::template arg<0>::type, typename traits::template arg<1>::type>(
                  numel,
                  data[0], data[1], data[2], // data
                  offset_calc.sizes_[0].divisor, offset_calc.sizes_[1].divisor, offset_calc.sizes_[2].divisor,
                  offset_calc.strides_[0][0], offset_calc.strides_[0][1], offset_calc.strides_[0][2],
                  offset_calc.strides_[1][0], offset_calc.strides_[1][1], offset_calc.strides_[1][2],
                  offset_calc.strides_[2][0], offset_calc.strides_[2][1], offset_calc.strides_[2][2],
                  f);
            } else if (((sizeof(arg0_t) == 2 && sizeof(arg1_t) == 2 && sizeof(arg2_t) == 2) || (sizeof(arg0_t) == 4 && sizeof(arg1_t) == 4 && sizeof(arg2_t) == 4)) &&
                maca_likely(!disable_opt_3_2_broadcast_dim1_contiguous) && offset_calc.strides_[0][0] == sizeof(arg0_t) &&
                offset_calc.strides_[1][0] == s0 * sizeof(arg0_t) && offset_calc.strides_[2][0] == s0 * s1 * sizeof(arg0_t) &&
                offset_calc.strides_[0][1] == sizeof(arg0_t) && offset_calc.strides_[1][1] == s0 * sizeof(arg0_t) && offset_calc.strides_[2][1] == 0 &&
                offset_calc.strides_[0][2] == sizeof(arg0_t) && offset_calc.strides_[1][2] == 0 && offset_calc.strides_[2][2] == s0 * sizeof(arg0_t) &&
                offset_calc.strides_[2][2] % (s0 * sizeof(arg0_t)) == 0 && s0 * s2 >= C10_WARP_SIZE && (s0 & (s0 -1)) == 0 && s0 <= C10_WARP_SIZE) {
                  get_elementwise_info<narity + 1>(ndim, narity, offset_calc, dtypes, "p_e_launch_legacy_kernel_maca_3_2_broadcast_dim1_contiguous_s", f);
                  launch_legacy_kernel_maca_3_2_broadcast_dim1_contiguous_s<128, unroll_factor, arg0_t, typename traits::template arg<0>::type, typename traits::template arg<1>::type>(
                    numel,
                    data[0], data[1], data[2], // data
                    offset_calc.sizes_[0].divisor, offset_calc.sizes_[1].divisor, offset_calc.sizes_[2].divisor,
                    offset_calc.strides_[0][0], offset_calc.strides_[0][1], offset_calc.strides_[0][2],
                    offset_calc.strides_[1][0], offset_calc.strides_[1][1], offset_calc.strides_[1][2],
                    offset_calc.strides_[2][0], offset_calc.strides_[2][1], offset_calc.strides_[2][2],
                    f);
            } else if (((sizeof(arg0_t) == 2 && sizeof(arg1_t) == 2 && sizeof(arg2_t) == 2) || (sizeof(arg0_t) == 4 && sizeof(arg1_t) == 4 && sizeof(arg2_t) == 4)) &&
                maca_likely(!disable_opt_3_2_broadcast_dim2_contiguous) && offset_calc.strides_[0][0] == sizeof(arg0_t) &&
                offset_calc.strides_[1][0] == s0 * sizeof(arg0_t) && offset_calc.strides_[2][0] == s0 * s1 * sizeof(arg0_t) &&
                offset_calc.strides_[0][1] == sizeof(arg0_t) && offset_calc.strides_[1][1] == 0 && offset_calc.strides_[2][1] == s0 * sizeof(arg0_t) * 2 &&
                offset_calc.strides_[0][2] == sizeof(arg0_t) && offset_calc.strides_[1][2] == s0 * sizeof(arg0_t) * 2 && offset_calc.strides_[2][2] == 0 &&
                offset_calc.strides_[1][2] % (s0 * sizeof(arg0_t)) == 0 && s0 * s1 >= C10_WARP_SIZE && (s0 & (s0 -1)) == 0 && s0 <= C10_WARP_SIZE) {
                  get_elementwise_info<narity + 1>(ndim, narity, offset_calc, dtypes, "p_e_launch_legacy_kernel_maca_3_2_broadcast_dim2_contiguous_s", f);
                  launch_legacy_kernel_maca_3_2_broadcast_dim2_contiguous_s<128, unroll_factor, arg0_t, typename traits::template arg<0>::type, typename traits::template arg<1>::type>(
                    numel,
                    data[0], data[1], data[2], // data
                    offset_calc.sizes_[0].divisor, offset_calc.sizes_[1].divisor, offset_calc.sizes_[2].divisor,
                    offset_calc.strides_[0][0], offset_calc.strides_[0][1], offset_calc.strides_[0][2],
                    offset_calc.strides_[1][0], offset_calc.strides_[1][1], offset_calc.strides_[1][2],
                    offset_calc.strides_[2][0], offset_calc.strides_[2][1], offset_calc.strides_[2][2],
                    f);
            } else if (((sizeof(arg0_t) == 2 && sizeof(arg1_t) == 2 && sizeof(arg2_t) == 2) || (sizeof(arg0_t) == 4 && sizeof(arg1_t) == 4 && sizeof(arg2_t) == 4)) &&
              maca_likely(!disable_opt_3_2_broadcast_dim1_uncontiguous) && offset_calc.strides_[0][0] == sizeof(arg0_t) &&
              offset_calc.strides_[1][0] == s0 * sizeof(arg0_t) && offset_calc.strides_[2][0] == s0 * s1 * sizeof(arg0_t) &&
              offset_calc.strides_[0][1] != sizeof(arg0_t) && offset_calc.strides_[0][1] != 0 && offset_calc.strides_[1][1] != 0 &&
              offset_calc.strides_[2][1] != 0 && offset_calc.strides_[0][2] != sizeof(arg0_t) && offset_calc.strides_[0][2] != 0 &&
              offset_calc.strides_[1][2] == 0 && offset_calc.strides_[2][2] != 0 &&
              (s0 % C10_WARP_SIZE == 0 || ((s0 & (s0 -1)) == 0 && s2 % (C10_WARP_SIZE / s0) == 0))) {
              // 3_2_broadcast in dim1 uncontiguous kernel
              // out: contiguous
              // arg0: uncontiguous
              // arg1: uncontiguous, broadcast in dim1
              // s0 divided by WARP_SIZE if s0 >= WARP_SIZE or s0 is power of two and s2 % (C10_WARP_SIZE / s0) == 0
              // shape like [32, 65, 32], [16, 56, 64], [8, *, 128] etc
              get_elementwise_info<narity + 1>(ndim, narity, offset_calc, dtypes, "p_e_launch_legacy_kernel_maca_3_2_broadcast_dim1_uncontiguous", f);
              launch_legacy_kernel_maca_3_2_broadcast_dim1_uncontiguous<128, unroll_factor, arg0_t, typename traits::template arg<0>::type, typename traits::template arg<1>::type>(
                numel,
                data[0], data[1], data[2], // data
                offset_calc.sizes_[0].divisor, offset_calc.sizes_[1].divisor, offset_calc.sizes_[2].divisor,
                offset_calc.strides_[0][0], offset_calc.strides_[0][1], offset_calc.strides_[0][2],
                offset_calc.strides_[1][0], offset_calc.strides_[1][1], offset_calc.strides_[1][2],
                offset_calc.strides_[2][0], offset_calc.strides_[2][1], offset_calc.strides_[2][2],
                f);
            } else if(((sizeof(arg0_t) == 2 && sizeof(arg1_t) == 2 && sizeof(arg2_t) == 2) || (sizeof(arg0_t) == 4 && sizeof(arg1_t) == 4 && sizeof(arg2_t) == 4)) &&
                maca_likely(!disable_opt_3_2_broadcast_dim2_contiguous) && offset_calc.strides_[0][0] == sizeof(arg0_t) &&
                offset_calc.strides_[1][0] == s0 * sizeof(arg0_t) && offset_calc.strides_[2][0] == s0 * s1 * sizeof(arg0_t) &&
                offset_calc.strides_[0][1] == sizeof(arg0_t) && offset_calc.strides_[1][1] == s0 * sizeof(arg0_t) &&
                offset_calc.strides_[2][1] == s0 * s1 * sizeof(arg0_t) && offset_calc.strides_[0][2] == sizeof(arg0_t) &&
                offset_calc.strides_[1][2] % (s0 * sizeof(arg0_t)) == 0 && offset_calc.strides_[2][2] == 0 && s0 % C10_WARP_SIZE == 0) {
                // 3_2_broadcast in dim2 kernel
                // out: contiguous
                // arg0: contiguous
                // arg1: contiguous in dim0, broadcast in dim2
                // s0 divided by WARP_SIZE
                // TODO(): support s0 < WARP_SIZE as 3_2_broadcast_dim1_uncontiguous kernel
                // TODO(): check if support offset_calc.strides_[1][2] > 0 but not only offset_calc.strides_[1][2] % (s0 * sizeof(arg0_t)) == 0
                get_elementwise_info<narity + 1>(ndim, narity, offset_calc, dtypes, "p_e_launch_legacy_kernel_maca_3_2_broadcast_dim2_contiguous", f);
                launch_legacy_kernel_maca_3_2_broadcast_dim2_contiguous<128, unroll_factor, arg0_t, typename traits::template arg<0>::type, typename traits::template arg<1>::type>(
                  numel,
                  data[0], data[1], data[2], // data
                  offset_calc.sizes_[0].divisor, offset_calc.sizes_[1].divisor, offset_calc.sizes_[2].divisor,
                  offset_calc.strides_[0][0], offset_calc.strides_[0][1], offset_calc.strides_[0][2],
                  offset_calc.strides_[1][0], offset_calc.strides_[1][1], offset_calc.strides_[1][2],
                  offset_calc.strides_[2][0], offset_calc.strides_[2][1], offset_calc.strides_[2][2],
                  f);
            } else if(((sizeof(arg0_t) == 2 && sizeof(arg1_t) == 2 && sizeof(arg2_t) == 2) || (sizeof(arg0_t) == 4 && sizeof(arg1_t) == 4 && sizeof(arg2_t) == 4)) &&
              maca_likely(!disable_opt_3_2_broadcast_dim2_uncontiguous) && offset_calc.strides_[0][0] == sizeof(arg0_t) &&
              offset_calc.strides_[1][0] == s0 * sizeof(arg0_t) &&
              offset_calc.strides_[2][0] == s0 * s1 * sizeof(arg0_t) && offset_calc.strides_[0][1] != sizeof(arg0_t) &&
              offset_calc.strides_[0][1] != 0 && offset_calc.strides_[1][1] != 0 && offset_calc.strides_[2][1] != 0 &&
              offset_calc.strides_[0][2] != sizeof(arg0_t) && offset_calc.strides_[0][2] != 0 && offset_calc.strides_[1][2] != 0 &&
              offset_calc.strides_[2][2] == 0 && (s0 % C10_WARP_SIZE == 0 || ((s0 & (s0 -1)) == 0 && s1 % (C10_WARP_SIZE / s0) == 0))) {
              // 3_2_broadcast in dim2 uncontiguous kernel
              // out: contiguous
              // arg0: uncontiguous
              // arg1: uncontiguou, broadcast in dim2
              // s0 divided by WARP_SIZE if s0 >= WARP_SIZE or s0 is power of two and s1 % (C10_WARP_SIZE / s0) == 0
              get_elementwise_info<narity + 1>(ndim, narity, offset_calc, dtypes, "p_e_launch_legacy_kernel_maca_3_2_broadcast_dim2_uncontiguous", f);
              launch_legacy_kernel_maca_3_2_broadcast_dim2_uncontiguous<128, unroll_factor, arg0_t, typename traits::template arg<0>::type, typename traits::template arg<1>::type>(
                numel,
                data[0], data[1], data[2], // data
                offset_calc.sizes_[0].divisor, offset_calc.sizes_[1].divisor, offset_calc.sizes_[2].divisor,
                offset_calc.strides_[0][0], offset_calc.strides_[0][1], offset_calc.strides_[0][2],
                offset_calc.strides_[1][0], offset_calc.strides_[1][1], offset_calc.strides_[1][2],
                offset_calc.strides_[2][0], offset_calc.strides_[2][1], offset_calc.strides_[2][2],
                f);
            } else if (((sizeof(arg0_t) == 2 && sizeof(arg1_t) == 2 && sizeof(arg2_t) == 2) || (sizeof(arg0_t) == 4 && sizeof(arg1_t) == 4 && sizeof(arg2_t) == 4)) &&
              st00 == sizeof(arg0_t) && st10 == sizeof(arg0_t) * s0 && st20 == sizeof(arg0_t) * s0 * s1 &&
              st01 == sizeof(arg1_t) && st02 == sizeof(arg2_t) &&
              s0 % C10_WARP_SIZE == 0 && maca_likely(!disable_opt_3_2_dim0_contiguous) && numel >= 5120 &&
              (reinterpret_cast<uintptr_t>(data[0]) % 4) == 0 &&
              (reinterpret_cast<uintptr_t>(data[1]) % 4) == 0 &&
              (reinterpret_cast<uintptr_t>(data[2]) % 4) == 0 &&
              st11 % 4 == 0 && st21 % 4 == 0 && st12 % 4 == 0 && st22 % 4 == 0) {
              get_elementwise_info<narity + 1>(ndim, narity, offset_calc, dtypes, "p_e_launch_legacy_kernel_maca_3_2_dim0_contiguous", f);
              launch_legacy_kernel_maca_3_2_dim0_contiguous<128, unroll_factor, arg0_t, typename traits::template arg<0>::type, typename traits::template arg<1>::type>(
                numel,
                data[0], data[1], data[2], // data
                offset_calc.sizes_[0].divisor, offset_calc.sizes_[1].divisor, offset_calc.sizes_[2].divisor,
                offset_calc.strides_[0][0], offset_calc.strides_[0][1], offset_calc.strides_[0][2],
                offset_calc.strides_[1][0], offset_calc.strides_[1][1], offset_calc.strides_[1][2],
                offset_calc.strides_[2][0], offset_calc.strides_[2][1], offset_calc.strides_[2][2],
                f);
            } else if (((sizeof(arg0_t) == 2 && sizeof(arg1_t) == 2 && sizeof(arg2_t) == 2) || (sizeof(arg0_t) == 4 && sizeof(arg1_t) == 4 && sizeof(arg2_t) == 4)) &&
                       st00 == sizeof(arg0_t) && st10 == s0 * sizeof(arg0_t) && st20 == s0 * s1 * sizeof(arg0_t) &&
                       st01 == sizeof(arg1_t) && st11 == s0 * sizeof(arg1_t) && st21 == s0 * s1 * sizeof(arg1_t) &&
                       st02 == 0 && st12 % 4 ==0 && st22 % 4 ==0 &&
                       (reinterpret_cast<uintptr_t>(data[0]) % 4) == 0 &&
                       (reinterpret_cast<uintptr_t>(data[1]) % 4) == 0 &&
                       (reinterpret_cast<uintptr_t>(data[2]) % 4) == 0 &&
                       s0 % C10_WARP_SIZE == 0 && maca_likely(!disable_opt_3_2_broadcast_dim0)) {
              // out: contiguous
              // arg0: contiguous
              // arg1: dim0 broadcast
              get_elementwise_info<narity + 1>(ndim, narity, offset_calc, dtypes, "p_e_launch_legacy_kernel_maca_3_2_broadcast_dim0", f);
              launch_legacy_kernel_maca_3_2_broadcast_dim0<128, unroll_factor, arg0_t, typename traits::template arg<0>::type, typename traits::template arg<1>::type>(
                numel,
                data[0], data[1], data[2], // data
                offset_calc.sizes_[0].divisor, offset_calc.sizes_[1].divisor, offset_calc.sizes_[2].divisor,
                offset_calc.strides_[0][0], offset_calc.strides_[0][1], offset_calc.strides_[0][2],
                offset_calc.strides_[1][0], offset_calc.strides_[1][1], offset_calc.strides_[1][2],
                offset_calc.strides_[2][0], offset_calc.strides_[2][1], offset_calc.strides_[2][2],
                f);
            } else if (((sizeof(arg0_t) == 2 && sizeof(arg1_t) == 2 && sizeof(arg2_t) == 2) || (sizeof(arg0_t) == 4 && sizeof(arg1_t) == 4 && sizeof(arg2_t) == 4)) &&
                       st00 == sizeof(arg0_t) && st10 == s0 * sizeof(arg0_t) && st20 == s0 * s1 * sizeof(arg0_t) &&
                       st21 == 0 && st02 == 0 && s0 % 2 == 0 && (s0 * s1) % (C10_WARP_SIZE * 4) == 0 && s2 % 4 == 0 &&
                       (reinterpret_cast<uintptr_t>(data[0]) % 4) == 0 &&
                       (reinterpret_cast<uintptr_t>(data[1]) % 4) == 0 &&
                       (reinterpret_cast<uintptr_t>(data[2]) % 4) == 0 &&
                       maca_likely(!disable_3_2_broadcast_arg0_dim2_arg1_dim0)) {
              //out: contiguous
              //arg0: dim2 broadcast
              //arg1: dim0 broadcast
              get_elementwise_info<narity + 1>(ndim, narity, offset_calc, dtypes, "p_e_launch_legacy_kernel_maca_3_2_broadcast_arg0_dim2_arg1_dim0", f);
              launch_legacy_kernel_maca_3_2_broadcast_arg0_dim2_arg1_dim0<128, unroll_factor, arg0_t, typename traits::template arg<0>::type, typename traits::template arg<1>::type>(
                numel,
                data[0], data[1], data[2], // data
                offset_calc.sizes_[0].divisor, offset_calc.sizes_[1].divisor, offset_calc.sizes_[2].divisor,
                offset_calc.strides_[0][0], offset_calc.strides_[0][1], offset_calc.strides_[0][2],
                offset_calc.strides_[1][0], offset_calc.strides_[1][1], offset_calc.strides_[1][2],
                offset_calc.strides_[2][0], offset_calc.strides_[2][1], offset_calc.strides_[2][2],
                f);
            } else if (((sizeof(arg0_t) == 2 && sizeof(arg1_t) == 2 && sizeof(arg2_t) == 2) || 
                        (sizeof(arg0_t) == 4 && sizeof(arg1_t) == 4 && sizeof(arg2_t) == 4) ||
                        (sizeof(arg0_t) == 2 && sizeof(arg1_t) == 2 && sizeof(arg2_t) == 1) ||
                        (sizeof(arg0_t) == 4 && sizeof(arg1_t) == 4 && sizeof(arg2_t) == 1)) &&
                       st00 == sizeof(arg0_t) && st10 == s0 * sizeof(arg0_t) && st20 == s0 * s1 * sizeof(arg0_t) &&
                       st01 == sizeof(arg1_t) && st02 == sizeof(arg2_t) &&
                       st12 == 0 && s0 >= C10_WARP_SIZE && s0 % 4 == 0 && s1 % 4 == 0 &&
                       (reinterpret_cast<uintptr_t>(data[0]) % 4) == 0 &&
                       (reinterpret_cast<uintptr_t>(data[1]) % 4) == 0 &&
                       (reinterpret_cast<uintptr_t>(data[2]) % 4) == 0 &&
                       st11 % 4 == 0 && st21 % 4 ==0 && st12 % 4 ==0 && st22 % 4 == 0 &&
                       maca_likely(!disable_3_2_dim0_contiguous_arg1_dim1_broadcast)) {
              //out: contiguous
              //arg0: dim0 contiguous
              //arg1: dim0 contiguous dim1 broadcast
              get_elementwise_info<narity + 1>(ndim, narity, offset_calc, dtypes, "p_e_launch_legacy_kernel_maca_3_2_dim0_contiguous_arg1_dim1_broadcast", f);
              launch_legacy_kernel_maca_3_2_dim0_contiguous_arg1_dim1_broadcast<128, unroll_factor, arg0_t, typename traits::template arg<0>::type, typename traits::template arg<1>::type>(
                numel,
                data[0], data[1], data[2], // data
                offset_calc.sizes_[0].divisor, offset_calc.sizes_[1].divisor, offset_calc.sizes_[2].divisor,
                offset_calc.strides_[0][0], offset_calc.strides_[0][1], offset_calc.strides_[0][2],
                offset_calc.strides_[1][0], offset_calc.strides_[1][1], offset_calc.strides_[1][2],
                offset_calc.strides_[2][0], offset_calc.strides_[2][1], offset_calc.strides_[2][2],
                f);
            } else {
              get_elementwise_info<narity + 1>(ndim, narity, offset_calc, dtypes, "p_e_launch_legacy_kernel_maca_3_2", f);
              launch_legacy_kernel_maca_3_2<128, unroll_factor, arg0_t, typename traits::template arg<0>::type, typename traits::template arg<1>::type>(
                numel,
                data[0], data[1], data[2], // data
                offset_calc.sizes_[0].divisor, offset_calc.sizes_[1].divisor, offset_calc.sizes_[2].divisor,
                offset_calc.strides_[0][0], offset_calc.strides_[0][1], offset_calc.strides_[0][2],
                offset_calc.strides_[1][0], offset_calc.strides_[1][1], offset_calc.strides_[1][2],
                offset_calc.strides_[2][0], offset_calc.strides_[2][1], offset_calc.strides_[2][2],
                f);
            }
          }
        } else if (ndim == 4) {
          auto s0 = offset_calc.sizes_[0].divisor;
          auto s1 = offset_calc.sizes_[1].divisor;
          auto s2 = offset_calc.sizes_[2].divisor;
          auto s3 = offset_calc.sizes_[3].divisor;
          
          size_t st00 = offset_calc.strides_[0][0], st01 = offset_calc.strides_[0][1], st02 = offset_calc.strides_[0][2];
          size_t st10 = offset_calc.strides_[1][0], st11 = offset_calc.strides_[1][1], st12 = offset_calc.strides_[1][2];
          size_t st20 = offset_calc.strides_[2][0], st21 = offset_calc.strides_[2][1], st22 = offset_calc.strides_[2][2];
          size_t st30 = offset_calc.strides_[3][0], st31 = offset_calc.strides_[3][1], st32 = offset_calc.strides_[3][2];

          bool disable_elementwise_4_2_opt = at::maca::get_maca_disable_elementwise_4_2_opt_kernel();
          bool disable_elementwise_4_2_template = at::maca::get_maca_disable_elementwise_4_2_template_kernel();
          bool disable_elementwise_4_2_uncontiguous = at::maca::get_maca_disable_elementwise_4_2_uncontiguous_kernel();
          bool disable_elementwise_4_2_broadcast_arg0_dim2_arg1_dim0 = at::maca::get_maca_disable_elementwise_4_2_broadcast_arg0_dim2_arg1_dim0_kernel();

          if ((sizeof(arg0_t) == 2 || sizeof(arg0_t) == 4) && maca_likely(!disable_elementwise_4_2_opt) &&
          check_opt_dim_4(s1, s2, offset_calc.strides_[1][0], offset_calc.strides_[2][0], offset_calc.strides_[3][0]) &&
          offset_calc.strides_[0][0] == sizeof(arg0_t) && offset_calc.strides_[1][0] == sizeof(arg0_t) * s0 &&
          offset_calc.strides_[2][0] == sizeof(arg0_t) * s0 * s1 && offset_calc.strides_[3][0] == sizeof(arg0_t) * s0 * s1 * s2 &&
          offset_calc.strides_[0][1] == sizeof(arg1_t) && offset_calc.strides_[1][1] == sizeof(arg1_t) * s0 &&
          offset_calc.strides_[2][1] == sizeof(arg1_t) * s0 * s1 && offset_calc.strides_[3][1] == sizeof(arg1_t) * s0 * s1 * s2 &&
          offset_calc.strides_[0][2] == sizeof(arg2_t) && offset_calc.strides_[1][2] == sizeof(arg2_t) * s0 * s2 &&
          offset_calc.strides_[2][2] == sizeof(arg2_t) * s0 && offset_calc.strides_[3][2] == sizeof(arg2_t) * s0 * s1 * s2 &&
          s0 % C10_WARP_SIZE == 0 && (reinterpret_cast<uintptr_t>(data[0]) % 4) == 0 && (reinterpret_cast<uintptr_t>(data[1]) % 4) == 0 &&
          (reinterpret_cast<uintptr_t>(data[2]) % 4) == 0) {
            get_elementwise_info<narity + 1>(ndim, narity, offset_calc, dtypes, "p_e_launch_legacy_kernel_maca_4_2_opt", f);
              launch_legacy_kernel_maca_4_2_opt<sizeof(arg0_t), 8*C10_WARP_SIZE, unroll_factor, arg0_t, typename traits::template arg<0>::type, typename traits::template arg<1>::type>(
                numel,
                data[0], data[1], data[2], // data
                offset_calc.sizes_[0].divisor, offset_calc.sizes_[1].divisor, offset_calc.sizes_[2].divisor, offset_calc.sizes_[3].divisor,
                offset_calc.strides_[0][0], offset_calc.strides_[0][1], offset_calc.strides_[0][2],
                offset_calc.strides_[1][0], offset_calc.strides_[1][1], offset_calc.strides_[1][2],
                offset_calc.strides_[2][0], offset_calc.strides_[2][1], offset_calc.strides_[2][2],
                offset_calc.strides_[3][0], offset_calc.strides_[3][1], offset_calc.strides_[3][2],
                f);
          } else if (maca_likely(!disable_elementwise_4_2_uncontiguous) && (sizeof(arg0_t) == 4 && sizeof(arg1_t) == 4 && sizeof(arg2_t) == 4) &&
              offset_calc.strides_[0][0] == sizeof(arg0_t) && offset_calc.strides_[1][0] == sizeof(arg0_t) * s0 &&
              offset_calc.strides_[2][0] == sizeof(arg0_t) * s0 * s1 && offset_calc.strides_[3][0] == sizeof(arg0_t) * s0 * s1 * s2 && 
              offset_calc.strides_[0][1] == sizeof(arg1_t) && offset_calc.strides_[1][1] == sizeof(arg1_t) * s0 &&
              offset_calc.strides_[2][1] == sizeof(arg1_t) * s0 * s1 && offset_calc.strides_[3][1] == sizeof(arg1_t) * s0 * s1 * s2 &&
              offset_calc.strides_[0][2] == 0 && offset_calc.strides_[1][2] == sizeof(arg2_t) && 
              offset_calc.strides_[2][2] == sizeof(arg2_t) * s1 * s3 && offset_calc.strides_[3][2] == sizeof(arg2_t) * s1 &&
              s0 % C10_WARP_SIZE == 0 && s1 % 2 == 0 && (reinterpret_cast<uintptr_t>(data[0]) % 4) == 0 && (reinterpret_cast<uintptr_t>(data[1]) % 4) == 0) {
              // arg0 & arg1 contiguous, arg2 dim0 broadcast & dim1 contiguous, shape=[12,64,4096,64], stride=[4096,49152,1,0]
            get_elementwise_info<narity + 1>(ndim, narity, offset_calc, dtypes, "p_e_launch_legacy_kernel_maca_4_2_uncontiguous", f);
            launch_legacy_kernel_maca_4_2_uncontiguous<128, unroll_factor, arg0_t, typename traits::template arg<0>::type, typename traits::template arg<1>::type>(
                numel,
                data[0], data[1], data[2], // data
                offset_calc.sizes_[0].divisor, offset_calc.sizes_[1].divisor, offset_calc.sizes_[2].divisor, offset_calc.sizes_[3].divisor,
                offset_calc.strides_[0][0], offset_calc.strides_[0][1], offset_calc.strides_[0][2],
                offset_calc.strides_[1][0], offset_calc.strides_[1][1], offset_calc.strides_[1][2],
                offset_calc.strides_[2][0], offset_calc.strides_[2][1], offset_calc.strides_[2][2],
                offset_calc.strides_[3][0], offset_calc.strides_[3][1], offset_calc.strides_[3][2],
                f);    
            } else if (((sizeof(arg0_t) == 2 && sizeof(arg1_t) == 2 && sizeof(arg2_t) == 2) || (sizeof(arg0_t) == 4 && sizeof(arg1_t) == 4 && sizeof(arg2_t) == 4)) &&
                       st00 == sizeof(arg0_t) && st10 == s0 * sizeof(arg0_t) && st20 == s0 * s1 * sizeof(arg0_t) && st30 == s0 * s1 * s2 * sizeof(arg0_t) &&
                       st21 == 0 && st02 == 0 && s0 % 2 == 0 && s1 % C10_WARP_SIZE == 0 && s2 % 4 == 0 &&
                       (reinterpret_cast<uintptr_t>(data[0]) % 4) == 0 &&
                       (reinterpret_cast<uintptr_t>(data[1]) % 4) == 0 &&
                       (reinterpret_cast<uintptr_t>(data[2]) % 4) == 0 &&
                       maca_likely(!disable_elementwise_4_2_broadcast_arg0_dim2_arg1_dim0)) {
            get_elementwise_info<narity + 1>(ndim, narity, offset_calc, dtypes, "p_e_launch_legacy_kernel_maca_4_2_broadcast_arg0_dim2_arg1_dim0", f);
            launch_legacy_kernel_maca_4_2_broadcast_arg0_dim2_arg1_dim0<128, unroll_factor, arg0_t, typename traits::template arg<0>::type, typename traits::template arg<1>::type>(
                numel,
                data[0], data[1], data[2], // data
                offset_calc.sizes_[0].divisor, offset_calc.sizes_[1].divisor, offset_calc.sizes_[2].divisor, offset_calc.sizes_[3].divisor,
                offset_calc.strides_[0][0], offset_calc.strides_[0][1], offset_calc.strides_[0][2],
                offset_calc.strides_[1][0], offset_calc.strides_[1][1], offset_calc.strides_[1][2],
                offset_calc.strides_[2][0], offset_calc.strides_[2][1], offset_calc.strides_[2][2],
                offset_calc.strides_[3][0], offset_calc.strides_[3][1], offset_calc.strides_[3][2],
                f);        
          } else if ((sizeof(arg0_t) == 2 || sizeof(arg0_t) == 4) && maca_likely(!disable_elementwise_4_2_template) &&
                (reinterpret_cast<uintptr_t>(data[0]) % 4) == 0 && (reinterpret_cast<uintptr_t>(data[1]) % 4) == 0 &&
                (reinterpret_cast<uintptr_t>(data[2]) % 4) == 0) {
            get_elementwise_info<narity + 1>(ndim, narity, offset_calc, dtypes, "p_e_launch_legacy_kernel_maca_4_2_template", f);
            launch_legacy_kernel_maca_4_2_template<128, unroll_factor, arg0_t, typename traits::template arg<0>::type, typename traits::template arg<1>::type>(
                numel,
                data[0], data[1], data[2], // data
                offset_calc.sizes_[0].divisor, offset_calc.sizes_[1].divisor, offset_calc.sizes_[2].divisor, offset_calc.sizes_[3].divisor,
                offset_calc.strides_[0][0], offset_calc.strides_[0][1], offset_calc.strides_[0][2],
                offset_calc.strides_[1][0], offset_calc.strides_[1][1], offset_calc.strides_[1][2],
                offset_calc.strides_[2][0], offset_calc.strides_[2][1], offset_calc.strides_[2][2],
                offset_calc.strides_[3][0], offset_calc.strides_[3][1], offset_calc.strides_[3][2],
                f);
          } else {
            get_elementwise_info<narity + 1>(ndim, narity, offset_calc, dtypes, "p_e_launch_legacy_kernel_maca_4_2", f);
              launch_legacy_kernel_maca_4_2<128, unroll_factor, arg0_t, typename traits::template arg<0>::type, typename traits::template arg<1>::type>(
                numel,
                data[0], data[1], data[2], // data
                offset_calc.sizes_[0].divisor, offset_calc.sizes_[1].divisor, offset_calc.sizes_[2].divisor, offset_calc.sizes_[3].divisor,
                offset_calc.strides_[0][0], offset_calc.strides_[0][1], offset_calc.strides_[0][2],
                offset_calc.strides_[1][0], offset_calc.strides_[1][1], offset_calc.strides_[1][2],
                offset_calc.strides_[2][0], offset_calc.strides_[2][1], offset_calc.strides_[2][2],
                offset_calc.strides_[3][0], offset_calc.strides_[3][1], offset_calc.strides_[3][2],
                f);
          }
        }else {
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

      bool disable_elementwise_arity2_cast_unroll = at::maca::get_maca_disable_elementwise_arity2_cast_unroll();
      const int vec_unroll = 4;
      get_elementwise_info<narity + 1>(ndim, narity, offset_calc, dtypes, "p_e_unroll_1_2", f);
      if (!disable_elementwise_arity2_cast_unroll && numel > 0 && numel % vec_unroll == 0 &&
          ((dtypes[0] == ScalarType::Float && dtypes[1] == ScalarType::Half && dtypes[2] == ScalarType::Float) ||
           (dtypes[0] == ScalarType::Float && dtypes[1] == ScalarType::BFloat16 && dtypes[2] == ScalarType::Float) ||
           (dtypes[0] == ScalarType::Float && dtypes[1] == ScalarType::Float && dtypes[2] == ScalarType::Half) ||
           (dtypes[0] == ScalarType::Float && dtypes[1] == ScalarType::Float && dtypes[2] == ScalarType::BFloat16)) &&
          (uint64_t)data[0] % 4 == 0 && (uint64_t)data[1] % 4 == 0 && (uint64_t)data[2] % 4 == 0) {
        launch_unrolled_arity2_cast_kernel<vec_unroll, arg0_t, typename traits::template arg<0>::type, typename traits::template arg<1>::type>(
            numel, f,
            dtypes[0], dtypes[1], dtypes[2],
            data[0], data[1], data[2]);
      } else {
        launch_unrolled_kernel(numel, f, data, input_offset_calculator, output_offset_calculator, loader, storer);
      }
    } else {
      if (ndim == 1) {
        get_elementwise_info<narity + 1>(ndim, narity, offset_calc, dtypes, "p_e_launch_legacy_kernel_maca_1_2_cast", f);
        launch_legacy_kernel_maca_1_2_cast<128, 4, arg0_t, typename traits::template arg<0>::type, typename traits::template arg<1>::type>(
          numel,
          data[0], data[1], data[2], // data
          dtypes[0], dtypes[1], dtypes[2],
          offset_calc.sizes_[0].divisor,
          offset_calc.strides_[0][0], offset_calc.strides_[0][1], offset_calc.strides_[0][2],
          f);
      } else if (ndim == 2) {
        size_t s0 = offset_calc.sizes_[0].divisor;
        bool disable_opt = at::maca::get_maca_disable_elementwise_2_2_cast_broadcast_kernel();
        bool disable_2_2_cast_template = at::maca::get_maca_disable_elementwise_2_2_cast_template_kernel();
        if (((dtypes[0] == ScalarType::Float && dtypes[1] == ScalarType::Half && dtypes[2] == ScalarType::Float &&
            offset_calc.strides_[0][0] == 4 && offset_calc.strides_[1][0] == s0 * 4 &&
            offset_calc.strides_[0][1] == 2 && offset_calc.strides_[1][1] == 0 &&
            offset_calc.strides_[0][2] == 4 && offset_calc.strides_[1][2] == s0 * 4) ||
            (dtypes[0] == ScalarType::Float && dtypes[1] == ScalarType::Half && dtypes[2] == ScalarType::Float &&
            offset_calc.strides_[0][0] == 4 && offset_calc.strides_[1][0] == s0 * 4 &&
            offset_calc.strides_[0][1] == 2 && offset_calc.strides_[1][1] == s0 * 2 &&
            offset_calc.strides_[0][2] == 0 && offset_calc.strides_[1][2] == 4) ||
            (dtypes[0] == ScalarType::Float && dtypes[1] == ScalarType::Float && dtypes[2] == ScalarType::Bool &&
            offset_calc.strides_[0][0] == 4 && offset_calc.strides_[1][0] == s0 * 4 &&
            offset_calc.strides_[0][1] == 4 && offset_calc.strides_[1][1] == s0 * 4 &&
            offset_calc.strides_[0][2] == 1 && offset_calc.strides_[1][2] == 0)) &&
            s0 >= C10_WARP_SIZE && maca_likely(!disable_opt)){ // float, float16, bfloat16
              // XXX(yuliu): one prerequisite for broadcast is divisible by warp_size, otherwise will cause
              // partial memory write and alignment fault.
              // 8 * C10_WARP_SIZE to be considered as best block size.
          get_elementwise_info<narity + 1>(ndim, narity, offset_calc, dtypes, "p_e_launch_legacy_kernel_maca_2_2_cast_broadcast", f);
          launch_legacy_kernel_maca_2_2_cast_broadcast<8*C10_WARP_SIZE, unroll_factor, arg0_t, typename traits::template arg<0>::type, typename traits::template arg<1>::type>(
            numel,
            data[0], data[1], data[2],  // data
            dtypes[0], dtypes[1], dtypes[2],
            offset_calc.sizes_[0].divisor, offset_calc.sizes_[1].divisor,
            offset_calc.strides_[0][0], offset_calc.strides_[0][1], offset_calc.strides_[0][2],
            offset_calc.strides_[1][0], offset_calc.strides_[1][1], offset_calc.strides_[1][2],
            f);
        } else if (((dtypes[0] == ScalarType::Float && (dtypes[1] == ScalarType::Char || dtypes[1] == ScalarType::Half) &&
          dtypes[2] == ScalarType::Float) || (dtypes[0] == ScalarType::BFloat16 && dtypes[1] == ScalarType::BFloat16 &&
          dtypes[2] == ScalarType::Float)) && maca_likely(!disable_2_2_cast_template)) {
          get_elementwise_info<narity + 1>(ndim, narity, offset_calc, dtypes, "p_e_launch_legacy_kernel_maca_2_2_cast_template", f);
          launch_legacy_kernel_maca_2_2_cast_template<2*C10_WARP_SIZE, unroll_factor, arg0_t, typename traits::template arg<0>::type, typename traits::template arg<1>::type>(
            numel,
            data[0], data[1], data[2],  // data
            dtypes[0], dtypes[1], dtypes[2],
            offset_calc.sizes_[0].divisor, offset_calc.sizes_[1].divisor,
            offset_calc.strides_[0][0], offset_calc.strides_[0][1], offset_calc.strides_[0][2],
            offset_calc.strides_[1][0], offset_calc.strides_[1][1], offset_calc.strides_[1][2],
            f);
        } else {
          get_elementwise_info<narity + 1>(ndim, narity, offset_calc, dtypes, "p_e_launch_legacy_kernel_maca_2_2_cast", f);
          launch_legacy_kernel_maca_2_2_cast<128, 4, arg0_t, typename traits::template arg<0>::type, typename traits::template arg<1>::type>(
            numel,
            data[0], data[1], data[2], // data
            dtypes[0], dtypes[1], dtypes[2],
            offset_calc.sizes_[0].divisor, offset_calc.sizes_[1].divisor,
            offset_calc.strides_[0][0], offset_calc.strides_[0][1], offset_calc.strides_[0][2],
            offset_calc.strides_[1][0], offset_calc.strides_[1][1], offset_calc.strides_[1][2],
            f);
        }
      } else if (ndim == 3) {
        size_t s0 = offset_calc.sizes_[0].divisor;
        size_t s0_s1 = offset_calc.sizes_[0].divisor * offset_calc.sizes_[1].divisor;
        bool disable_opt = at::maca::get_maca_disable_elementwise_3_2_cast_broadcast_kernel();
        bool disable_3_2_broadcast_dim2_opt = at::maca::get_maca_disable_elementwise_3_2_cast_broadcast_dim2_kernel();
        if (dtypes[0] == ScalarType::Float && dtypes[1] == ScalarType::Half && dtypes[2] == ScalarType::Float &&
            offset_calc.strides_[0][0] == 4 && offset_calc.strides_[1][0] == s0 * 4 && offset_calc.strides_[2][0] == s0_s1 * 4 &&
            offset_calc.strides_[0][1] == 2 && offset_calc.strides_[1][1] % (s0 * 2) == 0 && offset_calc.strides_[2][1] % (s0_s1 * 2) == 0 &&
            offset_calc.strides_[0][2] == 4 && offset_calc.strides_[1][2] == 0 && offset_calc.strides_[2][2] == s0 * 4 &&
            s0 % C10_WARP_SIZE == 0 && maca_likely(!disable_opt)) {
          get_elementwise_info<narity + 1>(ndim, narity, offset_calc, dtypes, "p_e_launch_legacy_kernel_maca_3_2_cast_broadcast", f);
          launch_legacy_kernel_maca_3_2_cast_broadcast<128, 4, arg0_t, typename traits::template arg<0>::type, typename traits::template arg<1>::type>(
            numel,
            data[0], data[1], data[2], // data
            dtypes[0], dtypes[1], dtypes[2],
            offset_calc.sizes_[0].divisor, offset_calc.sizes_[1].divisor, offset_calc.sizes_[2].divisor,
            offset_calc.strides_[0][0], offset_calc.strides_[0][1], offset_calc.strides_[0][2],
            offset_calc.strides_[1][0], offset_calc.strides_[1][1], offset_calc.strides_[1][2],
            offset_calc.strides_[2][0], offset_calc.strides_[2][1], offset_calc.strides_[2][2],
            f);
        } else if (dtypes[0] == ScalarType::Float && dtypes[1] == ScalarType::Half && dtypes[2] == ScalarType::Float &&
            offset_calc.strides_[0][0] == 4 && offset_calc.strides_[1][0] == s0 * 4 && offset_calc.strides_[2][0] == s0_s1 * 4 &&
            offset_calc.strides_[0][1] == 2 && offset_calc.strides_[1][1] == s0 * 2 && offset_calc.strides_[2][1] == s0_s1 * 2 &&
            offset_calc.strides_[0][2] == 0 && offset_calc.strides_[1][2] != 0 && offset_calc.strides_[2][2] == 0 &&
            s0 % C10_WARP_SIZE == 0 && maca_likely(!disable_3_2_broadcast_dim2_opt)){
          // 3_2_cast_broadcast in dim0 & dim2
          // arg0: contiguous float
          // arg1: contiguous half
          // arg2: broadcast in dim0 & dim2, contiguous in dim1
          // s0 % WARP SIZE == 0
          get_elementwise_info<narity + 1>(ndim, narity, offset_calc, dtypes, "p_e_launch_legacy_kernel_maca_3_2_cast_broadcast_dim2", f);
          launch_legacy_kernel_maca_3_2_cast_broadcast_dim2<128, 4, arg0_t, typename traits::template arg<0>::type, typename traits::template arg<1>::type>(
            numel,
            data[0], data[1], data[2], // data
            dtypes[0], dtypes[1], dtypes[2],
            offset_calc.sizes_[0].divisor, offset_calc.sizes_[1].divisor, offset_calc.sizes_[2].divisor,
            offset_calc.strides_[0][0], offset_calc.strides_[0][1], offset_calc.strides_[0][2],
            offset_calc.strides_[1][0], offset_calc.strides_[1][1], offset_calc.strides_[1][2],
            offset_calc.strides_[2][0], offset_calc.strides_[2][1], offset_calc.strides_[2][2],
            f);
        } else {
          get_elementwise_info<narity + 1>(ndim, narity, offset_calc, dtypes, "p_e_launch_legacy_kernel_maca_3_2_cast", f);
          launch_legacy_kernel_maca_3_2_cast<128, 4, arg0_t, typename traits::template arg<0>::type, typename traits::template arg<1>::type>(
            numel,
            data[0], data[1], data[2], // data
            dtypes[0], dtypes[1], dtypes[2],
            offset_calc.sizes_[0].divisor, offset_calc.sizes_[1].divisor, offset_calc.sizes_[2].divisor,
            offset_calc.strides_[0][0], offset_calc.strides_[0][1], offset_calc.strides_[0][2],
            offset_calc.strides_[1][0], offset_calc.strides_[1][1], offset_calc.strides_[1][2],
            offset_calc.strides_[2][0], offset_calc.strides_[2][1], offset_calc.strides_[2][2],
            f);
        }
      } else if (ndim == 4) {
         bool disable_opt_4_2_cast = at::maca::get_maca_disable_elementwise_kernel_4_2_cast_broadcast();
         size_t s0 = offset_calc.sizes_[0].divisor;
         size_t s01 = offset_calc.sizes_[0].divisor * offset_calc.sizes_[1].divisor;
         size_t s012 = offset_calc.sizes_[0].divisor * offset_calc.sizes_[1].divisor * offset_calc.sizes_[2].divisor;

         if (dtypes[0] == ScalarType::Float && dtypes[1] == ScalarType::Float &&
             (dtypes[2] == ScalarType::Half || dtypes[2] == ScalarType::BFloat16) &&
             offset_calc.strides_[0][0] == 4 && offset_calc.strides_[1][0] == 4 * s0 &&
             offset_calc.strides_[2][0] == 4 * s01 && offset_calc.strides_[3][0] == 4 * s012 &&
             offset_calc.strides_[0][1] == 8 && offset_calc.strides_[1][1] == 8 * s0 &&
             offset_calc.strides_[2][1] == 8 * s01 && offset_calc.strides_[3][1] == 8 * s012 &&
             offset_calc.strides_[0][2] == 2 && offset_calc.strides_[1][2] == 0 &&
             offset_calc.strides_[2][2] == 2 * s0 && offset_calc.strides_[3][2] == 0 &&
             s0 % 4 == 0 && !disable_opt_4_2_cast && (reinterpret_cast<uintptr_t>(data[0]) % 4) == 0 &&
             (reinterpret_cast<uintptr_t>(data[1]) % 4) == 0 && (reinterpret_cast<uintptr_t>(data[2]) % 4) == 0
            ){
            get_elementwise_info<narity + 1>(ndim, narity, offset_calc, dtypes, "p_e_launch_legacy_kernel_maca_4_2_cast_broadcast", f);
            launch_legacy_kernel_maca_4_2_cast_broadcast<128, 4, arg0_t, typename traits::template arg<0>::type, typename traits::template arg<1>::type>(
                numel,
                data[0], data[1], data[2], // data
                dtypes[0], dtypes[1], dtypes[2],
                offset_calc.sizes_[0].divisor, offset_calc.sizes_[1].divisor, offset_calc.sizes_[2].divisor, offset_calc.sizes_[3].divisor,
                offset_calc.strides_[0][0], offset_calc.strides_[0][1], offset_calc.strides_[0][2],
                offset_calc.strides_[1][0], offset_calc.strides_[1][1], offset_calc.strides_[1][2],
                offset_calc.strides_[2][0], offset_calc.strides_[2][1], offset_calc.strides_[2][2],
                offset_calc.strides_[3][0], offset_calc.strides_[3][1], offset_calc.strides_[3][2],
                f);
         } else {
            get_elementwise_info<narity + 1>(ndim, narity, offset_calc, dtypes, "p_e_launch_legacy_kernel_maca_4_2_cast", f);
            launch_legacy_kernel_maca_4_2_cast<128, 4, arg0_t, typename traits::template arg<0>::type, typename traits::template arg<1>::type>(
                numel,
                data[0], data[1], data[2], // data
                dtypes[0], dtypes[1], dtypes[2],
                offset_calc.sizes_[0].divisor, offset_calc.sizes_[1].divisor, offset_calc.sizes_[2].divisor, offset_calc.sizes_[3].divisor,
                offset_calc.strides_[0][0], offset_calc.strides_[0][1], offset_calc.strides_[0][2],
                offset_calc.strides_[1][0], offset_calc.strides_[1][1], offset_calc.strides_[1][2],
                offset_calc.strides_[2][0], offset_calc.strides_[2][1], offset_calc.strides_[2][2],
                offset_calc.strides_[3][0], offset_calc.strides_[3][1], offset_calc.strides_[3][2],
                f);
         }

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
