#pragma once 
#include "loop_utils.h"

template<int nt, int vt, typename res_t , typename arg0_t, typename arg1_t, typename arg2_t, typename func_t, typename index_t, typename stride_t>
C10_LAUNCH_BOUNDS_2(nt, 4)
__global__ void elementwise_kernel_2_3(
    int64_t N,
    char* data0, char* data1, char* data2, char* data3,
    index_t size0, index_t size1,
    stride_t stride00, stride_t stride01, stride_t stride02, stride_t stride03,
    stride_t stride10, stride_t stride11, stride_t stride12, stride_t stride13,
    func_t f) {
  // ndim = 2, arity = 3, narg = 4
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
      int64_t offsets[4];
      auto linear_idx = idx;
      constexpr int NARGS = 4;
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
      arg = 3;
      offsets[arg] += divmod_mod * stride03;
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
      arg = 3;
      offsets[arg] += divmod_mod * stride13;

      res_t* out = (res_t*)(data0 + offsets[0]);
      *out = f(func_reinterpret_cast<arg0_t>(data1 + offsets[1]),
               func_reinterpret_cast<arg1_t>(data2 + offsets[2]),
               func_reinterpret_cast<arg2_t>(data3 + offsets[3]));
      idx += nt;
    }
  }
}

template<int nt, int vt, typename res_t , typename arg0_t, typename arg1_t, typename arg2_t, typename func_t, typename index_t, typename stride_t>
C10_LAUNCH_BOUNDS_2(nt, 4)
__global__ void elementwise_kernel_3_3(
    int64_t N,
    char* data0, char* data1, char* data2, char* data3,
    index_t size0, index_t size1, index_t size2,
    stride_t stride00, stride_t stride01, stride_t stride02, stride_t stride03,
    stride_t stride10, stride_t stride11, stride_t stride12, stride_t stride13,
    stride_t stride20, stride_t stride21, stride_t stride22, stride_t stride23,
    func_t f) {
  // ndim = 3, arity = 3, narg = 4
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
      int64_t offsets[4];
      auto linear_idx = idx;
      constexpr int NARGS = 4;
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
      arg = 3;
      offsets[arg] += divmod_mod * stride03;
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
      arg = 3;
      offsets[arg] += divmod_mod * stride13;
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
      arg = 3;
      offsets[arg] += divmod_mod * stride23;

      res_t* out = (res_t*)(data0 + offsets[0]);
      *out = f(func_reinterpret_cast<arg0_t>(data1 + offsets[1]),
               func_reinterpret_cast<arg1_t>(data2 + offsets[2]),
               func_reinterpret_cast<arg2_t>(data3 + offsets[3]));
      idx += nt;
    }
  }
}

template<int nt, int v_x, typename res_t , typename arg0_t, typename arg1_t, typename arg2_t, typename func_t, typename index_t, typename stride_t>
C10_LAUNCH_BOUNDS_2(nt, 4)
__global__ void elementwise_kernel_3_3_broadcast(
    int64_t N,
    char* data0, char* data1, char* data2, char* data3,
    index_t size0, index_t size1, index_t size2,
    stride_t stride00, stride_t stride01, stride_t stride02, stride_t stride03,
    stride_t stride10, stride_t stride11, stride_t stride12, stride_t stride13,
    stride_t stride20, stride_t stride21, stride_t stride22, stride_t stride23,
    func_t f, int iter_x, int remain_x, int tail_x, int iter_y, int remain_y, 
    int iter_z, int remain_z) {
  // ndim = 3, arity = 3, narg = 4
  int tid = threadIdx.x;
  constexpr int NARGS = 4;
  constexpr int MAX_DIMS = 3;
  int64_t offsets[NARGS];

  using LoadT = at::native::memory::aligned_vector<arg0_t, v_x>;
  using LoadT1 = at::native::memory::aligned_vector<arg1_t, v_x>;
  using StoreT = at::native::memory::aligned_vector<res_t, v_x>;
  #pragma unroll
  for (int arg = 0; arg < NARGS; arg++) {
    offsets[arg] = 0;
  }
  // arg = 0: output; arg = 1: input1; arg = 2: input2; arg = 3: input3;

  res_t ld_out[v_x * v_x]; 
  arg0_t ld_1[v_x * v_x]; 
  arg1_t ld_2[v_x]; 
  arg2_t ld_3[v_x]; 

  LoadT1* p_ld_2 = reinterpret_cast<LoadT1*>(&ld_2);
  LoadT1* p_ld_3 = reinterpret_cast<LoadT1*>(&ld_3);
  for(size_t y_idx = 0; y_idx < iter_y; ++y_idx){
    size_t row_idx = blockIdx.y * iter_y + y_idx;
    offsets[2] = row_idx * v_x * stride12;
    offsets[3] = offsets[2];
    *p_ld_2 = *reinterpret_cast<LoadT1*>(data2 + offsets[2]);
    *p_ld_3 = *reinterpret_cast<LoadT1*>(data3 + offsets[3]);
    for(size_t z_idx = 0; z_idx < iter_z; z_idx++){
      size_t z_offset = (blockIdx.z * iter_z + z_idx) * stride20;
      for (size_t x_idx = 0; x_idx < iter_x; x_idx++) {
        size_t col_idx = ((gridDim.x * x_idx + blockIdx.x) * blockDim.x + tid) * v_x;
        #pragma unroll
        for (int ii = 0; ii < v_x; ii++) {
          offsets[1] = z_offset + (col_idx + ii) * stride01 + row_idx * v_x * stride11;
          LoadT* p_ld_1 = reinterpret_cast<LoadT*>(&(ld_1[ii * v_x])); 
          *p_ld_1 = *reinterpret_cast<LoadT*>(data1 + offsets[1]);
        }
        #pragma unroll
        for (int ii = 0; ii < v_x; ii++) {
          offsets[0] = z_offset + (row_idx * v_x + ii)* stride10 + col_idx * stride00;
          StoreT * p_ld_out = reinterpret_cast<StoreT*>(&(ld_out[ii * v_x]));
          StoreT* out = reinterpret_cast<StoreT*>(data0 + offsets[0]);
          #pragma unroll
          for (int i = 0; i < v_x; i++) {
              auto p1 = reinterpret_cast<res_t*>(data1 + offsets[1] + i * stride01);
              ld_out[ii * v_x + i] = f(ld_1[ii + i * v_x], ld_2[ii], ld_3[ii]);
          }
          *out = *p_ld_out;
        }
      }
    }
  }

  if (remain_y != 0 && (blockIdx.y + 1) * v_x <= remain_y) {
    size_t row_idx = gridDim.y * iter_y + blockIdx.y;
    offsets[2] = row_idx * v_x * stride12;
    offsets[3] = offsets[2];
    *p_ld_2 = *reinterpret_cast<LoadT1*>(data2 + offsets[2]);
    *p_ld_3 = *reinterpret_cast<LoadT1*>(data3 + offsets[3]);
    for(size_t z_idx = 0; z_idx < iter_z; z_idx++){
      size_t z_offset = (blockIdx.z * iter_z + z_idx) * stride20;
      for (size_t x_idx = 0; x_idx < iter_x; x_idx++) {
        size_t col_idx = ((gridDim.x * x_idx + blockIdx.x) * blockDim.x + tid) * v_x;
        #pragma unroll
        for (int ii = 0; ii < v_x; ii++) {
          offsets[1] = z_offset + (col_idx + ii) * stride01 + row_idx * v_x * stride11;
          LoadT* p_ld_1 = reinterpret_cast<LoadT*>(&(ld_1[ii * v_x])); 
          *p_ld_1 = *reinterpret_cast<LoadT*>(data1 + offsets[1]);
        }
        #pragma unroll
        for (int ii = 0; ii < v_x; ii++) {
          offsets[0] = z_offset + (row_idx * v_x + ii)* stride10 + col_idx * stride00;
          StoreT * p_ld_out = reinterpret_cast<StoreT*>(&(ld_out[ii * v_x]));
          StoreT* out = reinterpret_cast<StoreT*>(data0 + offsets[0]);
          #pragma unroll
          for (int i = 0; i < v_x; i++) {
              auto p1 = reinterpret_cast<res_t*>(data1 + offsets[1] + i * stride01);
              ld_out[ii * v_x + i] = f(ld_1[ii + i * v_x], ld_2[ii], ld_3[ii]);
          }
          *out = *p_ld_out;
        }
      }
    }
  }
  
  if (remain_x != 0 && (blockIdx.x * blockDim.x + tid + 1) * v_x <= remain_x) {
    for(size_t y_idx = 0; y_idx < iter_y; ++y_idx){
      size_t row_idx = blockIdx.y * iter_y + y_idx;
      offsets[2] = row_idx * v_x * stride12;
      offsets[3] = offsets[2];
      *p_ld_2 = *reinterpret_cast<LoadT1*>(data2 + offsets[2]);
      *p_ld_3 = *reinterpret_cast<LoadT1*>(data3 + offsets[3]);
      for(size_t z_idx = 0; z_idx < iter_z; z_idx++){
        size_t z_offset = (blockIdx.z * iter_z + z_idx) * stride20;
        size_t col_idx = ((gridDim.x * iter_x + blockIdx.x) * blockDim.x + tid) * v_x;
        #pragma unroll
        for (int ii = 0; ii < v_x; ii++) {
          offsets[1] = z_offset + (col_idx + ii) * stride01 + row_idx * v_x * stride11;
          LoadT* p_ld_1 = reinterpret_cast<LoadT*>(&(ld_1[ii * v_x])); 
          *p_ld_1 = *reinterpret_cast<LoadT*>(data1 + offsets[1]);
        }
        #pragma unroll
        for (int ii = 0; ii < v_x; ii++) {
          offsets[0] = z_offset + (row_idx * v_x + ii)* stride10 + col_idx * stride00;
          StoreT * p_ld_out = reinterpret_cast<StoreT*>(&(ld_out[ii * v_x]));
          StoreT* out = reinterpret_cast<StoreT*>(data0 + offsets[0]);
          #pragma unroll
          for (int i = 0; i < v_x; i++) {
              auto p1 = reinterpret_cast<res_t*>(data1 + offsets[1] + i * stride01);
              ld_out[ii * v_x + i] = f(ld_1[ii + i * v_x], ld_2[ii], ld_3[ii]);
          }
          *out = *p_ld_out;
        }
      }
    }
    if (remain_y != 0 && (blockIdx.y + 1) * v_x <= remain_y) {
      size_t row_idx = gridDim.y * iter_y + blockIdx.y;
      offsets[2] = row_idx * v_x * stride12;
      offsets[3] = offsets[2];
      *p_ld_2 = *reinterpret_cast<LoadT1*>(data2 + offsets[2]);
      *p_ld_3 = *reinterpret_cast<LoadT1*>(data3 + offsets[3]);
      for(size_t z_idx = 0; z_idx < iter_z; z_idx++){
        size_t z_offset = (blockIdx.z * iter_z + z_idx) * stride20;
        size_t col_idx = ((gridDim.x * iter_x + blockIdx.x) * blockDim.x + tid) * v_x;
        #pragma unroll
        for (int ii = 0; ii < v_x; ii++) {
          offsets[1] = z_offset + (col_idx + ii) * stride01 + row_idx * v_x * stride11;
          LoadT* p_ld_1 = reinterpret_cast<LoadT*>(&(ld_1[ii * v_x])); 
          *p_ld_1 = *reinterpret_cast<LoadT*>(data1 + offsets[1]);
        }
        #pragma unroll
        for (int ii = 0; ii < v_x; ii++) {
          offsets[0] = z_offset + (row_idx * v_x + ii)* stride10 + col_idx * stride00;
          StoreT * p_ld_out = reinterpret_cast<StoreT*>(&(ld_out[ii * v_x]));
          StoreT* out = reinterpret_cast<StoreT*>(data0 + offsets[0]);
          #pragma unroll
          for (int i = 0; i < v_x; i++) {
              auto p1 = reinterpret_cast<res_t*>(data1 + offsets[1] + i * stride01);
              ld_out[ii * v_x + i] = f(ld_1[ii + i * v_x], ld_2[ii], ld_3[ii]);
          }
          *out = *p_ld_out;
        }
      }
    }
  }
}

template<int nt, int vt, typename res_t , typename arg0_t, typename arg1_t, typename arg2_t, typename func_t, typename index_t, typename stride_t>
C10_LAUNCH_BOUNDS_2(nt, 4)
__global__ void elementwise_kernel_3_3_cast(
    int64_t N,
    char* data0, char* data1, char* data2, char* data3,
    ScalarType st0, ScalarType st1, ScalarType st2, ScalarType st3,
    index_t size0, index_t size1, index_t size2,
    stride_t stride00, stride_t stride01, stride_t stride02, stride_t stride03,
    stride_t stride10, stride_t stride11, stride_t stride12, stride_t stride13,
    stride_t stride20, stride_t stride21, stride_t stride22, stride_t stride23,
    func_t f) {
  // ndim = 3, arity = 3, narg = 4
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
      int64_t offsets[4];
      auto linear_idx = idx;
      constexpr int NARGS = 4;
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
      arg = 3;
      offsets[arg] += divmod_mod * stride03;
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
      arg = 3;
      offsets[arg] += divmod_mod * stride13;
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
      arg = 3;
      offsets[arg] += divmod_mod * stride23;

      void* out = data0 + offsets[0];
      res_t result = f(c10::fetch_and_cast<arg0_t>(st1, data1 + offsets[1]),
               c10::fetch_and_cast<arg1_t>(st2, data2 + offsets[2]),
               c10::fetch_and_cast<arg2_t>(st3, data3 + offsets[3]));
      c10::cast_and_store<res_t>(st0, out, result);
      idx += nt;
    }
  }
}

template<int nt, int vt, typename res_t , typename arg0_t, typename arg1_t, typename arg2_t, typename func_t, typename index_t, typename stride_t>
C10_LAUNCH_BOUNDS_2(nt, 4)
__global__ void elementwise_kernel_3_3_cast_without_assert(
    int64_t N,
    char* data0, char* data1, char* data2, char* data3,
    ScalarType st0, ScalarType st1, ScalarType st2, ScalarType st3,
    index_t size0, index_t size1, index_t size2,
    stride_t stride00, stride_t stride01, stride_t stride02, stride_t stride03,
    stride_t stride10, stride_t stride11, stride_t stride12, stride_t stride13,
    stride_t stride20, stride_t stride21, stride_t stride22, stride_t stride23,
    func_t f) {
  // ndim = 3, arity = 3, narg = 4
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
      int64_t offsets[4];
      auto linear_idx = idx;
      constexpr int NARGS = 4;
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
      arg = 3;
      offsets[arg] += divmod_mod * stride03;
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
      arg = 3;
      offsets[arg] += divmod_mod * stride13;
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
      arg = 3;
      offsets[arg] += divmod_mod * stride23;

      void* out = data0 + offsets[0];
      res_t result = f(c10::fetch_and_cast_without_assert<arg0_t>(st1, data1 + offsets[1]),
               c10::fetch_and_cast_without_assert<arg1_t>(st2, data2 + offsets[2]),
               c10::fetch_and_cast_without_assert<arg2_t>(st3, data3 + offsets[3]));
      c10::cast_and_store_without_assert<res_t>(st0, out, result);
      idx += nt;
    }
  }
}

template<int nt, int vt, typename res_t , typename arg0_t, typename arg1_t, typename arg2_t, typename func_t, typename index_t, typename stride_t>
static void launch_legacy_kernel_maca_2_3(
    int64_t N,
    char* data0, char* data1, char* data2, char* data3,
    index_t size0, index_t size1,
    stride_t stride00, stride_t stride01, stride_t stride02, stride_t stride03,
    stride_t stride10, stride_t stride11, stride_t stride12, stride_t stride13,
    const func_t& f) {
  // ndim = 2, arity = 3
  TORCH_INTERNAL_ASSERT(N >= 0 && N <= std::numeric_limits<int32_t>::max());
  if (N == 0) {
    return;
  }
  dim3 block(nt);
  dim3 grid((N + block.x * vt - 1) / (block.x * vt));
  auto stream = at::cuda::getCurrentCUDAStream();
  elementwise_kernel_2_3<nt, vt, res_t, arg0_t, arg1_t, arg2_t, func_t, index_t, stride_t><<<grid, block, 0, stream>>>(
      N, data0, data1, data2, data3, size0, size1, stride00, stride01, stride02, stride03, stride10, stride11, stride12, stride13, f);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template<int nt, int vt, typename res_t , typename arg0_t, typename arg1_t, typename arg2_t, typename func_t, typename index_t, typename stride_t>
static void launch_legacy_kernel_maca_3_3(
    int64_t N,
    char* data0, char* data1, char* data2, char* data3,
    index_t size0, index_t size1, index_t size2,
    stride_t stride00, stride_t stride01, stride_t stride02, stride_t stride03,
    stride_t stride10, stride_t stride11, stride_t stride12, stride_t stride13,
    stride_t stride20, stride_t stride21, stride_t stride22, stride_t stride23,
    const func_t& f) {
  // ndim = 3, arity = 3
  TORCH_INTERNAL_ASSERT(N >= 0 && N <= std::numeric_limits<int32_t>::max());
  if (N == 0) {
    return;
  }
  dim3 block(nt);
  dim3 grid((N + block.x * vt - 1) / (block.x * vt));
  auto stream = at::cuda::getCurrentCUDAStream();
  elementwise_kernel_3_3<nt, vt, res_t, arg0_t, arg1_t, arg2_t, func_t, index_t, stride_t><<<grid, block, 0, stream>>>(
      N, data0, data1, data2, data3, size0, size1, size2, stride00, stride01, stride02, stride03, stride10, stride11, stride12, stride13, stride20, stride21, stride22, stride23, f);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template<int nt, int vt, typename res_t , typename arg0_t, typename arg1_t, typename arg2_t, typename func_t, typename index_t, typename stride_t>
static void launch_legacy_kernel_maca_3_3_cast(
    int64_t N,
    char* data0, char* data1, char* data2, char* data3,
    ScalarType st0, ScalarType st1, ScalarType st2, ScalarType st3,
    index_t size0, index_t size1, index_t size2,
    stride_t stride00, stride_t stride01, stride_t stride02, stride_t stride03,
    stride_t stride10, stride_t stride11, stride_t stride12, stride_t stride13,
    stride_t stride20, stride_t stride21, stride_t stride22, stride_t stride23,
    const func_t& f) {
  // ndim = 3, arity = 3
  TORCH_INTERNAL_ASSERT(N >= 0 && N <= std::numeric_limits<int32_t>::max());
  if (N == 0) {
    return;
  }
  dim3 block(nt);
  dim3 grid((N + block.x * vt - 1) / (block.x * vt));
  auto stream = at::cuda::getCurrentCUDAStream();
  bool without_assert = at::maca::get_maca_enable_elementwise_without_assert();
  if (without_assert) {
    elementwise_kernel_3_3_cast_without_assert<nt, vt, res_t, arg0_t, arg1_t, arg2_t, func_t, index_t, stride_t><<<grid, block, 0, stream>>>(
        N, data0, data1, data2, data3, st0, st1, st2, st3, size0, size1, size2, stride00, stride01, stride02, stride03, stride10, 
        stride11, stride12, stride13, stride20, stride21, stride22, stride23, f);
  } else {
    elementwise_kernel_3_3_cast<nt, vt, res_t, arg0_t, arg1_t, arg2_t, func_t, index_t, stride_t><<<grid, block, 0, stream>>>(
        N, data0, data1, data2, data3, st0, st1, st2, st3, size0, size1, size2, stride00, stride01, stride02, stride03, stride10, 
        stride11, stride12, stride13, stride20, stride21, stride22, stride23, f);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

/*current only support below pattern
  output: shape(s0, s1, s2)， stride(1, s0, s0*s1)
  input1: stride(s1, 1, s0*s1)
  input2: stride(0, 1, 0)
  input3: stride(0, 1, 0)*/
template<int nt, int vt, typename res_t , typename arg0_t, typename arg1_t, typename arg2_t, typename func_t, typename index_t, typename stride_t>
static void launch_legacy_kernel_maca_3_3_broadcast(
    int64_t N,
    char* data0, char* data1, char* data2, char* data3,
    index_t size0, index_t size1, index_t size2,
    stride_t stride00, stride_t stride01, stride_t stride02, stride_t stride03,
    stride_t stride10, stride_t stride11, stride_t stride12, stride_t stride13,
    stride_t stride20, stride_t stride21, stride_t stride22, stride_t stride23,
    const int vec, const func_t& f) {
  // ndim = 3, arity = 3
  TORCH_INTERNAL_ASSERT(N >= 0 && N <= std::numeric_limits<int32_t>::max());
  if (N == 0) {
    return;
  }
  auto stream = at::cuda::getCurrentCUDAStream();
  constexpr int block_dim_x = 64;
  // int vec0 = getVectorizedAlignment<res_t>((void*)data0, size0);
  // int vec1 = getVectorizedAlignment<arg0_t>((void*)data1, size1);
  // int vec2 = getVectorizedAlignment<arg1_t>((void*)data2, size1);
  // int vec3 = getVectorizedAlignment<arg2_t>((void*)data3, size1);
  // int vec = std::min(std::min(std::min(vec0, vec1), vec2), vec3);

  dim3 block(block_dim_x);
  int max_handle_num_per_block = block_dim_x * vec * 16;
  int grid_dim_x = max(size0 / max_handle_num_per_block, 1);
  TORCH_INTERNAL_ASSERT(grid_dim_x > 0);
  int iter_x = size0 / (block_dim_x * vec * grid_dim_x); // iterations of whole block in x direction
  // TORCH_INTERNAL_ASSERT(iter_x > 0);
  int remain_x = size0 % (block_dim_x * vec * grid_dim_x);  // threads to handle remain cols in x direction
  int tail_x = size0 % vec; // the cols cannot be handle by thread*vec
  TORCH_INTERNAL_ASSERT(tail_x == 0);
  int grid_dim_y = getMaxGridSize(grid_dim_x, size1/vec);
  TORCH_INTERNAL_ASSERT(grid_dim_y > 0);
  // int grid_dim_z = getMaxGridSize(grid_dim_x * grid_dim_y, size2);
  // as z handle batchsize, blow config have higher efficient, even when size2 is 1024
  int grid_dim_z = size2;
  TORCH_INTERNAL_ASSERT(grid_dim_z > 0);
  dim3 grid(grid_dim_x, grid_dim_y, grid_dim_z);
  int iter_y = size1 / (grid_dim_y * vec);
  int remain_y = size1 % (grid_dim_y * vec);
  int tail_y = size1 % vec;
  TORCH_INTERNAL_ASSERT(tail_y == 0); // as size1 is multiple of 8, so tail_y should always be 0
  int iter_z = size2 / grid_dim_z;
  int remain_z = size2 % grid_dim_z;
  // TORCH_INTERNAL_ASSERT(iter_y > 0);
  // TORCH_INTERNAL_ASSERT(iter_z > 0);
  // printf("grid: %d, %d, %d\n", grid_dim_x, grid_dim_y, grid_dim_z);
  // printf("iter_x: %d, remain_x: %d, tail_x: %d\n", iter_x, remain_x, tail_x);
  // printf("iter_y: %d, remain_y: %d\n", iter_y, remain_y);
  // printf("iter_z: %d, remain_z: %d\n", iter_z, remain_z);
  // printf("vec: %d\n", vec);
  if (vec == 8) {
    elementwise_kernel_3_3_broadcast<nt, 8, res_t, arg0_t, arg1_t, arg2_t, func_t, index_t, stride_t><<<grid, block, 0, stream>>>(
        N, data0, data1, data2, data3, size0, size1, size2, stride00, stride01, stride02, stride03, stride10, stride11, stride12, 
        stride13, stride20, stride21, stride22, stride23, f, iter_x, remain_x, tail_x, iter_y, remain_y, iter_z, remain_z);
  } else if (vec == 4) {
    elementwise_kernel_3_3_broadcast<nt, 4, res_t, arg0_t, arg1_t, arg2_t, func_t, index_t, stride_t><<<grid, block, 0, stream>>>(
        N, data0, data1, data2, data3, size0, size1, size2, stride00, stride01, stride02, stride03, stride10, stride11, stride12, 
        stride13, stride20, stride21, stride22, stride23, f, iter_x, remain_x, tail_x, iter_y, remain_y, iter_z, remain_z);
  } else if (vec == 2) {
    elementwise_kernel_3_3_broadcast<nt, 2, res_t, arg0_t, arg1_t, arg2_t, func_t, index_t, stride_t><<<grid, block, 0, stream>>>(
        N, data0, data1, data2, data3, size0, size1, size2, stride00, stride01, stride02, stride03, stride10, stride11, stride12, 
        stride13, stride20, stride21, stride22, stride23, f, iter_x, remain_x, tail_x, iter_y, remain_y, iter_z, remain_z);
  } else {
    elementwise_kernel_3_3_broadcast<nt, 1, res_t, arg0_t, arg1_t, arg2_t, func_t, index_t, stride_t><<<grid, block, 0, stream>>>(
        N, data0, data1, data2, data3, size0, size1, size2, stride00, stride01, stride02, stride03, stride10, stride11, stride12, 
        stride13, stride20, stride21, stride22, stride23, f, iter_x, remain_x, tail_x, iter_y, remain_y, iter_z, remain_z);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename func_t>
void gpu_kernel_impl_maca_arity3(TensorIteratorBase& iter, const func_t& f) {
  using traits = function_traits<func_t>;
  using arg0_t = typename traits::result_type;
  using arg1_t = typename traits::template arg<0>::type;
  using arg2_t = typename traits::template arg<1>::type;
  using arg3_t = typename traits::template arg<2>::type;

  constexpr int ntensors = traits::arity + 1;

  TORCH_INTERNAL_ASSERT(iter.can_use_32bit_indexing());
  TORCH_INTERNAL_ASSERT(iter.ninputs() == traits::arity);
  TORCH_INTERNAL_ASSERT(iter.noutputs() == 1);

  at::detail::Array<char*, ntensors> data;
  for (int i = 0; i < ntensors; i++) {
    data[i] = (char*)iter.data_ptr(i);
  }

  int64_t numel = iter.numel();

  bool contiguous = iter.is_contiguous();
  bool dynamic_casting = needs_dynamic_casting<func_t>::check(iter);
  int ndim = iter.ndim();
  constexpr int narity = traits::arity;
  
  if (!dynamic_casting) {
    if (contiguous) {
      launch_vectorized_kernel(numel, f, data);
    } else {
        auto offset_calc = ::make_offset_calculator<traits::arity + 1>(iter);
        constexpr int unroll_factor = sizeof(arg0_t) >= 4 ? 2 : 4;
        if (ndim == 2 && narity == 3) {
          get_elementwise_info<narity + 1>(ndim, narity, offset_calc, "p_e_launch_legacy_kernel_maca_2_3",f);
          launch_legacy_kernel_maca_2_3<128, unroll_factor, arg0_t,
                                        typename traits::template arg<0>::type,
                                        typename traits::template arg<1>::type,
                                        typename traits::template arg<2>::type>(
            numel,
            data[0], data[1], data[2], data[3], // data
            offset_calc.sizes_[0].divisor, offset_calc.sizes_[1].divisor,
            offset_calc.strides_[0][0], offset_calc.strides_[0][1], offset_calc.strides_[0][2], offset_calc.strides_[0][3],
            offset_calc.strides_[1][0], offset_calc.strides_[1][1], offset_calc.strides_[1][2], offset_calc.strides_[1][3],
            f);
        } else if (ndim == 3 && narity == 3) {
          int vec0 = getVectorizedAlignment<arg0_t>((void*)data[0], offset_calc.sizes_[0].divisor);
          int vec1 = getVectorizedAlignment<arg1_t>((void*)data[1], offset_calc.sizes_[1].divisor);
          int vec2 = getVectorizedAlignment<arg2_t>((void*)data[2], offset_calc.sizes_[1].divisor);
          int vec3 = getVectorizedAlignment<arg3_t>((void*)data[3], offset_calc.sizes_[1].divisor);
          int vec = std::min(std::min(std::min(vec0, vec1), vec2), vec3);

          if(maca_likely(!at::maca::get_maca_disable_elementwise_3_3_broadcast_kernel()) && ((sizeof(arg0_t) == 4 && sizeof(arg1_t) == 4 &&
             sizeof(arg2_t) == 4 && sizeof(arg3_t) == 4) || (sizeof(arg0_t) == 2 && sizeof(arg1_t) == 2 && sizeof(arg2_t) == 2 
             && sizeof(arg3_t) == 2) || (sizeof(arg0_t) == 2  && sizeof(arg1_t) == 2 && sizeof(arg2_t) == 4 && sizeof(arg3_t) == 4)) &&   // correspond to float, half and amp mode
             offset_calc.strides_[0][2] == 0 && offset_calc.strides_[2][2] == 0 && offset_calc.strides_[1][2] == sizeof(arg2_t) &&
             offset_calc.strides_[0][3] == 0 && offset_calc.strides_[2][3] == 0 && offset_calc.strides_[1][3] == sizeof(arg3_t) &&
             offset_calc.strides_[1][1] == sizeof(arg1_t) && offset_calc.strides_[0][1] == offset_calc.sizes_[1].divisor * sizeof(arg1_t) &&
              offset_calc.strides_[2][1] == offset_calc.strides_[0][1] * offset_calc.sizes_[0].divisor &&
             offset_calc.strides_[0][0] == sizeof(arg0_t) && offset_calc.strides_[1][0] == offset_calc.sizes_[0].divisor * sizeof(arg0_t) &&
              offset_calc.strides_[2][0] == offset_calc.strides_[1][0] * offset_calc.sizes_[1].divisor &&
             vec > 1 && offset_calc.sizes_[0].divisor % vec == 0 && offset_calc.sizes_[1].divisor % vec == 0){  // as max vec is 8, so multiple of 8 keep memory alignment 
            get_elementwise_info<narity + 1>(ndim, narity, offset_calc, "p_e_launch_legacy_kernel_maca_3_3_broadcast", f);
            launch_legacy_kernel_maca_3_3_broadcast<8*C10_WARP_SIZE, unroll_factor, arg0_t, typename traits::template arg<0>::type, 
                                                    typename traits::template arg<1>::type, typename traits::template arg<2>::type>(
              numel,
              data[0], data[1], data[2], data[3], // data
              offset_calc.sizes_[0].divisor, offset_calc.sizes_[1].divisor, offset_calc.sizes_[2].divisor,
              offset_calc.strides_[0][0], offset_calc.strides_[0][1], offset_calc.strides_[0][2], offset_calc.strides_[0][3],
              offset_calc.strides_[1][0], offset_calc.strides_[1][1], offset_calc.strides_[1][2], offset_calc.strides_[1][3],
              offset_calc.strides_[2][0], offset_calc.strides_[2][1], offset_calc.strides_[2][2], offset_calc.strides_[2][3],
              vec, f); 
          }
          else{
            get_elementwise_info<narity + 1>(ndim, narity, offset_calc, "p_e_launch_legacy_kernel_maca_3_3", f);
            launch_legacy_kernel_maca_3_3<128, unroll_factor, arg0_t,
                                          typename traits::template arg<0>::type,
                                          typename traits::template arg<1>::type,
                                          typename traits::template arg<2>::type>(
              numel,
              data[0], data[1], data[2], data[3], // data
              offset_calc.sizes_[0].divisor, offset_calc.sizes_[1].divisor, offset_calc.sizes_[2].divisor,
              offset_calc.strides_[0][0], offset_calc.strides_[0][1], offset_calc.strides_[0][2], offset_calc.strides_[0][3], 
              offset_calc.strides_[1][0], offset_calc.strides_[1][1], offset_calc.strides_[1][2], offset_calc.strides_[1][3], 
              offset_calc.strides_[2][0], offset_calc.strides_[2][1], offset_calc.strides_[2][2], offset_calc.strides_[2][3], 
              f);
          }
        } else {
          get_elementwise_info<narity + 1>(ndim, narity, offset_calc, "p_e_launch_legacy_kernel",f);
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
      launch_unrolled_kernel(numel, f, data, input_offset_calculator, output_offset_calculator, loader, storer);
    } else {
      at::detail::Array<ScalarType, ntensors> dtypes;
      for (int i = 0; i < ntensors; i++) {
        dtypes[i] = iter.dtype(i);
      }
      auto offset_calc = ::make_offset_calculator<traits::arity + 1>(iter);
      constexpr int unroll_factor = sizeof(arg0_t) >= 4 ? 2 : 4;
      if(ndim == 3 && narity == 3){
        get_elementwise_info<narity + 1>(ndim, narity, offset_calc, "p_e_launch_legacy_kernel_maca_3_3_cast",f);
        launch_legacy_kernel_maca_3_3_cast<128, unroll_factor, arg0_t,
                                      typename traits::template arg<0>::type,
                                      typename traits::template arg<1>::type,
                                      typename traits::template arg<2>::type>(
          numel,
          data[0], data[1], data[2], data[3], // data
          dtypes[0], dtypes[1], dtypes[2], dtypes[3],
          offset_calc.sizes_[0].divisor, offset_calc.sizes_[1].divisor, offset_calc.sizes_[2].divisor, 
          offset_calc.strides_[0][0], offset_calc.strides_[0][1], offset_calc.strides_[0][2], offset_calc.strides_[0][3], 
          offset_calc.strides_[1][0], offset_calc.strides_[1][1], offset_calc.strides_[1][2], offset_calc.strides_[1][3], 
          offset_calc.strides_[2][0], offset_calc.strides_[2][1], offset_calc.strides_[2][2], offset_calc.strides_[2][3], 
          f);
      }else{
        get_elementwise_info<narity + 1>(ndim, narity, offset_calc, "p_e_launch_legacy_kernel",f);
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
