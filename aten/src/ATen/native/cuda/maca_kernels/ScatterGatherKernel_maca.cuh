#pragma once
#include <c10/macros/Macros.h>
#include "loop_utils.h"

template<int nt, int vt, int v_x, bool is_scatter_like, typename scalar_t, typename func_t, typename index_t, typename stride_t>
C10_LAUNCH_BOUNDS_2(nt, 4)
__global__ void _gather_elementwise_kernel_dim2_opt(
    int64_t numel, int64_t index_size, int64_t index_stride, 
    char* self_ptr, char* src_ptr, char* index_ptr,
    index_t size0, index_t size1,
    stride_t stride00, stride_t stride01, stride_t stride02,
    stride_t stride10, stride_t stride11, stride_t stride12,
    func_t f, int y_t, int y_remain) {

  int64_t offsets[3];
  constexpr int NARGS = 3;

  using StoreT = at::native::memory::aligned_vector<scalar_t, v_x>;
  #pragma unroll
  for (int arg = 0; arg < NARGS; arg++) {
    offsets[arg] = 0;
  }
  // arg = 0: output; arg = 1: input1; arg = 2: input2;

  scalar_t ld_out[v_x];
  StoreT* p_ld_out = reinterpret_cast<StoreT*>(&ld_out);

  int tid = threadIdx.x;

  for (int y_idx = 0; y_idx < y_t; ++y_idx) {
    size_t row_offset = (blockIdx.y * y_t + y_idx);
    offsets[2] = (blockIdx.x * blockDim.x + tid) * stride02 + row_offset * stride12;
    int64_t idx_dim = *(int64_t*)(index_ptr + offsets[2]);
    for (int v = 0; v < v_x; ++v) {
      offsets[1] = (blockIdx.x * blockDim.x + tid) * v_x * stride01 + v * stride01 + row_offset * stride11;
      // ld_out[v] = *((scalar_t*)(src_ptr + offsets[1]) + idx_dim);
      f(
        (ld_out + v),
        is_scatter_like ? idx_dim * index_stride : 0,
        numel,
        (scalar_t*)(src_ptr + offsets[1]) + (is_scatter_like ? 0 : idx_dim * index_stride)
      );
    }
    offsets[0] = (blockIdx.x * blockDim.x + tid) * v_x * stride00 + row_offset * stride10;
    StoreT* out = reinterpret_cast<StoreT*>(self_ptr + offsets[0]);
    *out = *p_ld_out;
  }
}

template <int nt, int vt, bool is_scatter_like, typename scalar_t, typename func_t, typename index_t, typename stride_t>
static void _launch_gather_kernel_dim2_opt(
  int64_t N, int64_t index_size, int64_t index_stride, 
  char* self_ptr, char* src_ptr, char* index_ptr,
  index_t size0, index_t size1,
  stride_t stride00, stride_t stride01, stride_t stride02,
  stride_t stride10, stride_t stride11, stride_t stride12,
  const func_t& f) {
  if (N == 0) {
    return;
  }

  constexpr int block_dim_x = 64;
  size_t load_num = size0 / block_dim_x;
  int vec = getVectorizedAlignment<scalar_t>((void*)self_ptr, load_num);

  int grid_dim_x = load_num / vec;
  // maybe need y_t, for shape [8739, 5120], adjust y_t can get just a little performance improvement, but
  // y_t=1 alse very fast, keep y_t for more shapes
  int y_t = 1;
  int grid_dim_y = size1 / y_t;
  // no use now
  int y_remain = size1 % y_t;

  dim3 block(block_dim_x);
  dim3 grid(grid_dim_x, grid_dim_y);
  const auto stream = at::cuda::getCurrentCUDAStream();

  if (vec == 8) {
    _gather_elementwise_kernel_dim2_opt<nt, vt, 8, is_scatter_like, scalar_t, func_t><<<grid, block, 0, stream>>>(
      N, index_size, index_stride, 
      self_ptr, src_ptr, index_ptr, 
      size0, size1, 
      stride00, stride01, stride02, 
      stride10, stride11, stride12, 
      f, y_t, y_remain);
  } else if(vec == 4) {
    _gather_elementwise_kernel_dim2_opt<nt, vt, 4, is_scatter_like, scalar_t, func_t><<<grid, block, 0, stream>>>(
      N, index_size, index_stride, 
      self_ptr, src_ptr, index_ptr, 
      size0, size1, 
      stride00, stride01, stride02, 
      stride10, stride11, stride12, 
      f, y_t, y_remain);
  } else if(vec == 2) {
    _gather_elementwise_kernel_dim2_opt<nt, vt, 2, is_scatter_like, scalar_t, func_t><<<grid, block, 0, stream>>>(
      N, index_size, index_stride, 
      self_ptr, src_ptr, index_ptr, 
      size0, size1, 
      stride00, stride01, stride02, 
      stride10, stride11, stride12, 
      f, y_t, y_remain);
  } else {
    _gather_elementwise_kernel_dim2_opt<nt, vt, 1, is_scatter_like, scalar_t, func_t><<<grid, block, 0, stream>>>(
      N, index_size, index_stride, 
      self_ptr, src_ptr, index_ptr, 
      size0, size1, 
      stride00, stride01, stride02, 
      stride10, stride11, stride12, 
      f, y_t, y_remain);
  }

  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template<int nt, int vt, int v_x, bool is_scatter_like, typename scalar_t, typename func_t, typename index_t, typename stride_t>
C10_LAUNCH_BOUNDS_2(nt, 4)
__global__ void _scatter_elementwise_kernel_dim2_opt(
    int64_t numel, int64_t index_size, int64_t index_stride, 
    char* self_ptr, char* src_ptr, char* index_ptr,
    index_t size0, index_t size1,
    stride_t stride00, stride_t stride01, stride_t stride02,
    stride_t stride10, stride_t stride11, stride_t stride12,
    func_t f) {

  int64_t offsets[3];
  constexpr int NARGS = 3;

  using LoadT = at::native::memory::aligned_vector<scalar_t, v_x>;
  #pragma unroll
  for (int arg = 0; arg < NARGS; arg++) {
    offsets[arg] = 0;
  }
  // arg = 0: output; arg = 1: input1; arg = 2: input2;

  scalar_t ld_val[v_x];
  LoadT* p_ld_val = reinterpret_cast<LoadT*>(&ld_val);

  int tid = threadIdx.x;

  size_t row_offset = blockIdx.y;
  offsets[2] = (blockIdx.x * blockDim.x + tid) * stride02 + row_offset * stride12;
  int64_t idx_dim = *(int64_t*)(index_ptr + offsets[2]);
  offsets[1] = (blockIdx.x * blockDim.x + tid) * v_x * stride01 + row_offset * stride11;
  *p_ld_val = *reinterpret_cast<LoadT*>(src_ptr + offsets[1]);
  for (int v = 0; v < v_x; ++v) {
    offsets[0] = (blockIdx.x * blockDim.x + tid) * v_x * stride00 + v * stride00 + row_offset * stride10;
    // ld_out[v] = *((scalar_t*)(src_ptr + offsets[1]) + idx_dim);
    f(
      (scalar_t*)(self_ptr + offsets[0]),
      idx_dim * index_stride,
      numel,
      // (scalar_t*)(src_ptr + offsets[1]) + (is_scatter_like ? 0 : idx_dim * index_stride)
      (ld_val + v)
    );
  }
}

template <int nt, int vt, bool is_scatter_like, typename scalar_t, typename func_t, typename index_t, typename stride_t>
static void _launch_scatter_kernel_dim2_opt(
  int64_t N, int64_t index_size, int64_t index_stride, 
  char* self_ptr, char* src_ptr, char* index_ptr,
  index_t size0, index_t size1,
  stride_t stride00, stride_t stride01, stride_t stride02,
  stride_t stride10, stride_t stride11, stride_t stride12,
  const func_t& f) {
  if (N == 0) {
    return;
  }

  constexpr int block_dim_x = 64;
  size_t load_num = size0 / block_dim_x;
  int vec = std::min(getVectorizedAlignment<scalar_t>((void*)self_ptr, load_num), 2);
  // adjust vec num
  if ((load_num / vec) * size1 < getMaxWaveNum()) {
    vec = std::min(vec, 1);
  }

  int grid_dim_x = load_num / vec;

  dim3 block(block_dim_x);
  dim3 grid(grid_dim_x, size1);
  const auto stream = at::cuda::getCurrentCUDAStream();

  if (vec == 8) {
    _scatter_elementwise_kernel_dim2_opt<nt, vt, 8, is_scatter_like, scalar_t, func_t><<<grid, block, 0, stream>>>(
      N, index_size, index_stride, 
      self_ptr, src_ptr, index_ptr, 
      size0, size1, 
      stride00, stride01, stride02, 
      stride10, stride11, stride12, 
      f);
  } else if(vec == 4) {
    _scatter_elementwise_kernel_dim2_opt<nt, vt, 4, is_scatter_like, scalar_t, func_t><<<grid, block, 0, stream>>>(
      N, index_size, index_stride, 
      self_ptr, src_ptr, index_ptr, 
      size0, size1, 
      stride00, stride01, stride02, 
      stride10, stride11, stride12, 
      f);
  } else if(vec == 2) {
    _scatter_elementwise_kernel_dim2_opt<nt, vt, 2, is_scatter_like, scalar_t, func_t><<<grid, block, 0, stream>>>(
      N, index_size, index_stride, 
      self_ptr, src_ptr, index_ptr, 
      size0, size1, 
      stride00, stride01, stride02, 
      stride10, stride11, stride12, 
      f);
  } else {
    _scatter_elementwise_kernel_dim2_opt<nt, vt, 1, is_scatter_like, scalar_t, func_t><<<grid, block, 0, stream>>>(
      N, index_size, index_stride, 
      self_ptr, src_ptr, index_ptr, 
      size0, size1, 
      stride00, stride01, stride02, 
      stride10, stride11, stride12, 
      f);
  }

  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
