#pragma once

template<int nt, int vt, typename func_t, typename index_t, typename stride_t>
C10_LAUNCH_BOUNDS_2(nt, launch_bound2)
__global__ void index_elementwise_kernel_3_1(
    int n,
    char* out_ptr, char* in_ptr,
    int64_t s0, int64_t s1, int64_t s2, int64_t st0,  int64_t st1, int64_t st2, char* index_ptr0, char* index_ptr1, char* index_ptr2,
    index_t size0,
    stride_t stride00, stride_t stride01, stride_t stride02,
    func_t f) {
  int tid = threadIdx.x;
  int nv = nt * vt;
  int64_t idx = nv * blockIdx.x + tid;
  #pragma unroll
  for (int i = 0; i < vt; i++) {
    if (idx < n) {
      // -----------------------------------------------
      // auto offsets = offset_calc.get(idx);
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
      // -----------------------------------------------
      // char* out_data = out_ptr + offsets[0];
      // char* in_data = in_ptr + offsets[1];
      // int64_t offset = 0;
      // #pragma unroll
      // for (int i = 0; i < num_indices; i++) {
      //   int64_t index = *(int64_t*)(index_ptrs[i] + offsets[2]);
      //   // CUDA_KERNEL_ASSERT(index >= -sizes[i] && index < sizes[i] && "index out of bounds");
      //   if (index < 0) {
      //     index += sizes[i];
      //   }
      //   offset += index * strides[i];
      // }
      // f(out_data, in_data, offset);
      char* out_data = out_ptr + offsets[0];
      char* in_data = in_ptr + offsets[1];
      int64_t offset = 0;

      //int64_t s0, int64_t s1, int64_t st0, int64_t st1, char* index_ptr0, char* index_ptr1,
      int64_t index = *(int64_t*)(index_ptr0 + offsets[2]);
      // CUDA_KERNEL_ASSERT(index >= -s0 && index < s0 && "index out of bounds");
      if (index < 0) {
        index += s0;
      }
      offset += index * st0;
      index = *(int64_t*)(index_ptr1 + offsets[2]);
      // CUDA_KERNEL_ASSERT(index >= -s1 && index < s1 && "index out of bounds");
      if (index < 0) {
        index += s1;
      }
      offset += index * st1;
      index = *(int64_t*)(index_ptr2 + offsets[2]);
      // CUDA_KERNEL_ASSERT(index >= -s2 && index < s2 && "index out of bounds");
      if (index < 0) {
        index += s2;
      }
      offset += index * st2;

      f(out_data, in_data, offset);
      // -----------------------------------------------
      idx += nt;
    }
  }
}

template<int nt, int vt, typename func_t, typename index_t, typename stride_t>
C10_LAUNCH_BOUNDS_2(nt, launch_bound2)
__global__ void index_elementwise_kernel_3_2(
    int n,
    char* out_ptr, char* in_ptr,
    int64_t s0, int64_t s1, int64_t s2, int64_t st0,  int64_t st1, int64_t st2, char* index_ptr0, char* index_ptr1, char* index_ptr2,
    index_t size0,
    index_t size1,
    stride_t stride00, stride_t stride01, stride_t stride02,
    stride_t stride10, stride_t stride11, stride_t stride12,
    func_t f) {
  int tid = threadIdx.x;
  int nv = nt * vt;
  int64_t idx = nv * blockIdx.x + tid;
  #pragma unroll
  for (int i = 0; i < vt; i++) {
    if (idx < n) {
      // -----------------------------------------------
      // auto offsets = offset_calc.get(idx);
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
      divmod_div = linear_idx / size1;
      divmod_mod = linear_idx % size1;
      linear_idx = divmod_div;
      arg = 0;
      offsets[arg] += divmod_mod * stride10;
      arg = 1;
      offsets[arg] += divmod_mod * stride11;
      arg = 2;
      offsets[arg] += divmod_mod * stride12;
      // -----------------------------------------------
      // char* out_data = out_ptr + offsets[0];
      // char* in_data = in_ptr + offsets[1];
      // int64_t offset = 0;
      // #pragma unroll
      // for (int i = 0; i < num_indices; i++) {
      //   int64_t index = *(int64_t*)(index_ptrs[i] + offsets[2]);
      //   // CUDA_KERNEL_ASSERT(index >= -sizes[i] && index < sizes[i] && "index out of bounds");
      //   if (index < 0) {
      //     index += sizes[i];
      //   }
      //   offset += index * strides[i];
      // }
      // f(out_data, in_data, offset);
      char* out_data = out_ptr + offsets[0];
      char* in_data = in_ptr + offsets[1];
      int64_t offset = 0;

      //int64_t s0, int64_t s1, int64_t st0, int64_t st1, char* index_ptr0, char* index_ptr1,
      int64_t index = *(int64_t*)(index_ptr0 + offsets[2]);
      // CUDA_KERNEL_ASSERT(index >= -s0 && index < s0 && "index out of bounds");
      if (index < 0) {
        index += s0;
      }
      offset += index * st0;
      index = *(int64_t*)(index_ptr1 + offsets[2]);
      // CUDA_KERNEL_ASSERT(index >= -s1 && index < s1 && "index out of bounds");
      if (index < 0) {
        index += s1;
      }
      offset += index * st1;
      index = *(int64_t*)(index_ptr2 + offsets[2]);
      // CUDA_KERNEL_ASSERT(index >= -s2 && index < s2 && "index out of bounds");
      if (index < 0) {
        index += s2;
      }
      offset += index * st2;

      f(out_data, in_data, offset);
      // -----------------------------------------------
      idx += nt;
    }
  }
}

template<int nt, int vt, typename func_t, typename index_t, typename stride_t>
C10_LAUNCH_BOUNDS_2(nt, launch_bound2)
__global__ void index_elementwise_kernel_3_3(
    int n,
    char* out_ptr, char* in_ptr,
    int64_t s0, int64_t s1, int64_t s2, int64_t st0,  int64_t st1, int64_t st2, char* index_ptr0, char* index_ptr1, char* index_ptr2,
    index_t size0,
    index_t size1,
    index_t size2,
    stride_t stride00, stride_t stride01, stride_t stride02,
    stride_t stride10, stride_t stride11, stride_t stride12,
    stride_t stride20, stride_t stride21, stride_t stride22,
    func_t f) {
  int tid = threadIdx.x;
  int nv = nt * vt;
  int64_t idx = nv * blockIdx.x + tid;
  #pragma unroll
  for (int i = 0; i < vt; i++) {
    if (idx < n) {
      // -----------------------------------------------
      // auto offsets = offset_calc.get(idx);
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
      divmod_div = linear_idx / size1;
      divmod_mod = linear_idx % size1;
      linear_idx = divmod_div;
      arg = 0;
      offsets[arg] += divmod_mod * stride10;
      arg = 1;
      offsets[arg] += divmod_mod * stride11;
      arg = 2;
      offsets[arg] += divmod_mod * stride12;
      divmod_div = linear_idx / size2;
      divmod_mod = linear_idx % size2;
      linear_idx = divmod_div;
      arg = 0;
      offsets[arg] += divmod_mod * stride20;
      arg = 1;
      offsets[arg] += divmod_mod * stride21;
      arg = 2;
      offsets[arg] += divmod_mod * stride22;
      // -----------------------------------------------
      // char* out_data = out_ptr + offsets[0];
      // char* in_data = in_ptr + offsets[1];
      // int64_t offset = 0;
      // #pragma unroll
      // for (int i = 0; i < num_indices; i++) {
      //   int64_t index = *(int64_t*)(index_ptrs[i] + offsets[2]);
      //   // CUDA_KERNEL_ASSERT(index >= -sizes[i] && index < sizes[i] && "index out of bounds");
      //   if (index < 0) {
      //     index += sizes[i];
      //   }
      //   offset += index * strides[i];
      // }
      // f(out_data, in_data, offset);
      char* out_data = out_ptr + offsets[0];
      char* in_data = in_ptr + offsets[1];
      int64_t offset = 0;

      //int64_t s0, int64_t s1, int64_t st0, int64_t st1, char* index_ptr0, char* index_ptr1,
      int64_t index = *(int64_t*)(index_ptr0 + offsets[2]);
      // CUDA_KERNEL_ASSERT(index >= -s0 && index < s0 && "index out of bounds");
      if (index < 0) {
        index += s0;
      }
      offset += index * st0;
      index = *(int64_t*)(index_ptr1 + offsets[2]);
      // CUDA_KERNEL_ASSERT(index >= -s1 && index < s1 && "index out of bounds");
      if (index < 0) {
        index += s1;
      }
      offset += index * st1;
      index = *(int64_t*)(index_ptr2 + offsets[2]);
      // CUDA_KERNEL_ASSERT(index >= -s2 && index < s2 && "index out of bounds");
      if (index < 0) {
        index += s2;
      }
      offset += index * st2;

      f(out_data, in_data, offset);
      // -----------------------------------------------
      idx += nt;
    }
  }
}

template<int nt, int vt, typename func_t, typename index_t, typename stride_t>
C10_LAUNCH_BOUNDS_2(nt, launch_bound2)
__global__ void index_elementwise_kernel_1_1(
    int n,
    char* out_ptr, char* in_ptr,
    int64_t s0, int64_t st0, char* index_ptr0,
    index_t size0,
    stride_t stride00, stride_t stride01, stride_t stride02,
    func_t f) {
  int tid = threadIdx.x;
  int nv = nt * vt;
  int64_t idx = nv * blockIdx.x + tid;
  #pragma unroll
  for (int i = 0; i < vt; i++) {
    if (idx < n) {
      // -----------------------------------------------
      // auto offsets = offset_calc.get(idx);
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
      // -----------------------------------------------
      // char* out_data = out_ptr + offsets[0];
      // char* in_data = in_ptr + offsets[1];
      // int64_t offset = 0;
      // #pragma unroll
      // for (int i = 0; i < num_indices; i++) {
      //   int64_t index = *(int64_t*)(index_ptrs[i] + offsets[2]);
      //   // CUDA_KERNEL_ASSERT(index >= -sizes[i] && index < sizes[i] && "index out of bounds");
      //   if (index < 0) {
      //     index += sizes[i];
      //   }
      //   offset += index * strides[i];
      // }
      // f(out_data, in_data, offset);
      char* out_data = out_ptr + offsets[0];
      char* in_data = in_ptr + offsets[1];
      int64_t offset = 0;

      //int64_t s0, int64_t s1, int64_t st0, int64_t st1, char* index_ptr0, char* index_ptr1,
      int64_t index = *(int64_t*)(index_ptr0 + offsets[2]);
      // CUDA_KERNEL_ASSERT(index >= -s0 && index < s0 && "index out of bounds");
      if (index < 0) {
        index += s0;
      }
      offset += index * st0;

      f(out_data, in_data, offset);
      // -----------------------------------------------
      idx += nt;
    }
  }
}

template<int nt, int vt, typename func_t, typename index_t, typename stride_t>
C10_LAUNCH_BOUNDS_2(nt, launch_bound2)
__global__ void index_elementwise_kernel_1_2(
    int n,
    char* out_ptr, char* in_ptr,
    int64_t s0, int64_t st0, char* index_ptr0,
    index_t size0,
    index_t size1,
    stride_t stride00, stride_t stride01, stride_t stride02,
    stride_t stride10, stride_t stride11, stride_t stride12,
    func_t f) {
  int tid = threadIdx.x;
  int nv = nt * vt;
  int64_t idx = nv * blockIdx.x + tid;
  #pragma unroll
  for (int i = 0; i < vt; i++) {
    if (idx < n) {
      // -----------------------------------------------
      // auto offsets = offset_calc.get(idx);
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
      divmod_div = linear_idx / size1;
      divmod_mod = linear_idx % size1;
      linear_idx = divmod_div;
      arg = 0;
      offsets[arg] += divmod_mod * stride10;
      arg = 1;
      offsets[arg] += divmod_mod * stride11;
      arg = 2;
      offsets[arg] += divmod_mod * stride12;
      // -----------------------------------------------
      // char* out_data = out_ptr + offsets[0];
      // char* in_data = in_ptr + offsets[1];
      // int64_t offset = 0;
      // #pragma unroll
      // for (int i = 0; i < num_indices; i++) {
      //   int64_t index = *(int64_t*)(index_ptrs[i] + offsets[2]);
      //   // CUDA_KERNEL_ASSERT(index >= -sizes[i] && index < sizes[i] && "index out of bounds");
      //   if (index < 0) {
      //     index += sizes[i];
      //   }
      //   offset += index * strides[i];
      // }
      // f(out_data, in_data, offset);
      char* out_data = out_ptr + offsets[0];
      char* in_data = in_ptr + offsets[1];
      int64_t offset = 0;

      //int64_t s0, int64_t s1, int64_t st0, int64_t st1, char* index_ptr0, char* index_ptr1,
      int64_t index = *(int64_t*)(index_ptr0 + offsets[2]);
      // CUDA_KERNEL_ASSERT(index >= -s0 && index < s0 && "index out of bounds");
      if (index < 0) {
        index += s0;
      }
      offset += index * st0;

      f(out_data, in_data, offset);
      // -----------------------------------------------
      idx += nt;
    }
  }
}

template<int nt, int v_x, typename arg_t, typename idx_t, typename func_t, typename index_t, typename stride_t>
C10_LAUNCH_BOUNDS_2(nt, launch_bound2)
__global__ void index_elementwise_kernel_1_2_broadcast(
    int n,
    char* out_ptr, char* in_ptr,
    int64_t s0, int64_t st0, char* index_ptr0,
    index_t size0,
    index_t size1,
    stride_t stride00, stride_t stride01, stride_t stride02,
    stride_t stride10, stride_t stride11, stride_t stride12,
    func_t f, int y_t, int y_remain) {
  // vec store output, vec load index and broadcast index, single load value
  int tid = threadIdx.x;
  int64_t offsets[3];
  constexpr int NARGS = 3;

  using StoreT = at::native::memory::aligned_vector<arg_t, v_x>;
  using IndexT = at::native::memory::aligned_vector<idx_t, v_x>;
  #pragma unroll
  for (int arg = 0; arg < NARGS; arg++) {
    offsets[arg] = 0;
  }
  // arg = 0: output; arg = 1: input1; arg = 2: input2;

  idx_t ld_idx[v_x];
  arg_t ld_out[v_x];

  IndexT* p_ld_idx = reinterpret_cast<IndexT*>(&ld_idx); 
  StoreT* p_ld_out = reinterpret_cast<StoreT*>(&ld_out); 

  size_t y_loop = y_t;
  if (y_remain != 0 && blockIdx.y == (gridDim.y - 1)) {
    y_loop = y_remain;
  }

  size_t row_offset = (blockIdx.x * blockDim.x + tid) * v_x;
  if (row_offset >= size0) return;

  // broadcast index
  offsets[2] = row_offset * stride02;
  *p_ld_idx = *reinterpret_cast<IndexT*>(index_ptr0 + offsets[2]);
  #pragma unroll
  for (int i = 0; i < v_x; i++) {
    int64_t offset = 0;
    int64_t index = (int64_t)ld_idx[i];
    if (index < 0) {
      index += s0;
    }
    offset += index * st0;
    ld_idx[i] = offset;
  }

  for (size_t y_idx = 0; y_idx < y_loop; y_idx++) {
    offsets[0] = (blockIdx.y * y_t + y_idx) * stride10 + row_offset * stride00;
    StoreT* out = reinterpret_cast<StoreT*>(out_ptr + offsets[0]);
    #pragma unroll
    for (int i = 0; i < v_x; i++) {
      offsets[1] = (blockIdx.y * y_t + y_idx) * stride11 + (row_offset + i) * stride01 + ld_idx[i];
      ld_out[i] = *reinterpret_cast<arg_t*>(in_ptr + offsets[1]);
    }
    *out = *p_ld_out;
  }

}

template<int nt, int v_x, int v_y, typename arg_t, typename idx_t, typename func_t, typename index_t, typename stride_t>
C10_LAUNCH_BOUNDS_2(nt, launch_bound2)
__global__ void index_elementwise_kernel_1_2_broadcast_transpose(
    int n,
    char* out_ptr, char* in_ptr,
    int64_t s0, int64_t st0, char* index_ptr0,
    index_t size0,
    index_t size1,
    stride_t stride00, stride_t stride01, stride_t stride02,
    stride_t stride10, stride_t stride11, stride_t stride12,
    func_t f) {
  // vec store output, vec load value and broadcast value, single load value
  int tid = threadIdx.x;
  int64_t offsets[3];
  constexpr int NARGS = 3;

  using StoreT = at::native::memory::aligned_vector<arg_t, v_x>;
  using IndexT = at::native::memory::aligned_vector<idx_t, v_y>;
  #pragma unroll
  for (int arg = 0; arg < NARGS; arg++) {
    offsets[arg] = 0;
  }
  // arg = 0: output; arg = 1: input1; arg = 2: input2;

  idx_t ld_idx[v_y];
  arg_t ld_out[v_x];

  IndexT* p_ld_idx = reinterpret_cast<IndexT*>(&ld_idx); 
  StoreT* p_ld_out = reinterpret_cast<StoreT*>(&ld_out); 

  size_t row_offset = (blockIdx.x * blockDim.x + tid) * v_x;
  size_t col_offset = blockIdx.y * v_y;
  if (row_offset >= size0 || col_offset >= size1) return;

  // broadcast index
  offsets[2] = col_offset * stride12;
  *p_ld_idx = *reinterpret_cast<IndexT*>(index_ptr0 + offsets[2]);
  #pragma unroll
  for (int i = 0; i < v_y; i++) {
    int64_t offset = 0;
    int64_t index = (int64_t)ld_idx[i];
    if (index < 0) {
      index += s0;
    }
    offset += index * st0;
    ld_idx[i] = offset;
  }

  #pragma unroll
  for (int j = 0; j < v_y; ++j) {
    offsets[0] = (blockIdx.y * v_y + j) * stride10 + row_offset * stride00;
    StoreT* out = reinterpret_cast<StoreT*>(out_ptr + offsets[0]);
    #pragma unroll
    for (int i = 0; i < v_x; i++) {
      offsets[1] = (blockIdx.y * v_y + j) * stride11 + (row_offset + i) * stride01 + ld_idx[j];
      ld_out[i] = *reinterpret_cast<arg_t*>(in_ptr + offsets[1]);
    }
    *out = *p_ld_out;
  }

  // for (size_t y_idx = 0; y_idx < y_loop; y_idx++) {
  //   offsets[0] = (blockIdx.y * y_t + y_idx) * stride10 + row_offset * stride00;
  //   StoreT* out = reinterpret_cast<StoreT*>(out_ptr + offsets[0]);
  //   #pragma unroll
  //   for (int i = 0; i < v_x; i++) {
  //     offsets[1] = (blockIdx.y * y_t + y_idx) * stride11 + row_offset * stride01 + ld_idx[i];
  //     ld_out[i] = *reinterpret_cast<arg_t*>(in_ptr + offsets[1]);
  //   }
  //   *out = *p_ld_out;
  // }

}

template<int nt, int vt, typename func_t, typename index_t, typename stride_t>
C10_LAUNCH_BOUNDS_2(nt, launch_bound2)
__global__ void index_elementwise_kernel_1_3(
    int n,
    char* out_ptr, char* in_ptr,
    int64_t s0, int64_t st0, char* index_ptr0,
    index_t size0,
    index_t size1,
    index_t size2,
    stride_t stride00, stride_t stride01, stride_t stride02,
    stride_t stride10, stride_t stride11, stride_t stride12,
    stride_t stride20, stride_t stride21, stride_t stride22,
    func_t f) {
  int tid = threadIdx.x;
  int nv = nt * vt;
  int64_t idx = nv * blockIdx.x + tid;
  #pragma unroll
  for (int i = 0; i < vt; i++) {
    if (idx < n) {
      // -----------------------------------------------
      // auto offsets = offset_calc.get(idx);
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
      divmod_div = linear_idx / size1;
      divmod_mod = linear_idx % size1;
      linear_idx = divmod_div;
      arg = 0;
      offsets[arg] += divmod_mod * stride10;
      arg = 1;
      offsets[arg] += divmod_mod * stride11;
      arg = 2;
      offsets[arg] += divmod_mod * stride12;
      divmod_div = linear_idx / size2;
      divmod_mod = linear_idx % size2;
      linear_idx = divmod_div;
      arg = 0;
      offsets[arg] += divmod_mod * stride20;
      arg = 1;
      offsets[arg] += divmod_mod * stride21;
      arg = 2;
      offsets[arg] += divmod_mod * stride22;
      // -----------------------------------------------
      // char* out_data = out_ptr + offsets[0];
      // char* in_data = in_ptr + offsets[1];
      // int64_t offset = 0;
      // #pragma unroll
      // for (int i = 0; i < num_indices; i++) {
      //   int64_t index = *(int64_t*)(index_ptrs[i] + offsets[2]);
      //   // CUDA_KERNEL_ASSERT(index >= -sizes[i] && index < sizes[i] && "index out of bounds");
      //   if (index < 0) {
      //     index += sizes[i];
      //   }
      //   offset += index * strides[i];
      // }
      // f(out_data, in_data, offset);
      char* out_data = out_ptr + offsets[0];
      char* in_data = in_ptr + offsets[1];
      int64_t offset = 0;

      //int64_t s0, int64_t s1, int64_t st0, int64_t st1, char* index_ptr0, char* index_ptr1,
      int64_t index = *(int64_t*)(index_ptr0 + offsets[2]);
      // CUDA_KERNEL_ASSERT(index >= -s0 && index < s0 && "index out of bounds");
      if (index < 0) {
        index += s0;
      }
      offset += index * st0;

      f(out_data, in_data, offset);
      // -----------------------------------------------
      idx += nt;
    }
  }
}

template<int nt, int vt, typename func_t, typename index_t, typename stride_t>
C10_LAUNCH_BOUNDS_2(nt, launch_bound2)
__global__ void index_elementwise_kernel_2_1(
    int n,
    char* out_ptr, char* in_ptr,
    int64_t s0, int64_t s1, int64_t st0, int64_t st1, char* index_ptr0, char* index_ptr1,
    index_t size0,
    stride_t stride00, stride_t stride01, stride_t stride02,
    func_t f) {
  int tid = threadIdx.x;
  int nv = nt * vt;
  int64_t idx = nv * blockIdx.x + tid;
  constexpr int num_indices = 2;
  #pragma unroll
  for (int i = 0; i < vt; i++) {
    if (idx < n) {
      // -----------------------------------------------
      // auto offsets = offset_calc.get(idx);
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
      // -----------------------------------------------
      // char* out_data = out_ptr + offsets[0];
      // char* in_data = in_ptr + offsets[1];
      // int64_t offset = 0;
      // #pragma unroll
      // for (int i = 0; i < num_indices; i++) {
      //   int64_t index = *(int64_t*)(index_ptrs[i] + offsets[2]);
      //   // CUDA_KERNEL_ASSERT(index >= -sizes[i] && index < sizes[i] && "index out of bounds");
      //   if (index < 0) {
      //     index += sizes[i];
      //   }
      //   offset += index * strides[i];
      // }
      // f(out_data, in_data, offset);
      char* out_data = out_ptr + offsets[0];
      char* in_data = in_ptr + offsets[1];
      int64_t offset = 0;

      //int64_t s0, int64_t s1, int64_t st0, int64_t st1, char* index_ptr0, char* index_ptr1,
      int64_t index = *(int64_t*)(index_ptr0 + offsets[2]);
      // CUDA_KERNEL_ASSERT(index >= -s0 && index < s0 && "index out of bounds");
      if (index < 0) {
        index += s0;
      }
      offset += index * st0;
      
      index = *(int64_t*)(index_ptr1 + offsets[2]);
      // CUDA_KERNEL_ASSERT(index >= -s1 && index < s1 && "index out of bounds");
      if (index < 0) {
        index += s1;
      }
      offset += index * st1;

      f(out_data, in_data, offset);
      // -----------------------------------------------
      idx += nt;
    }
  }
}

template<int nt, int vt, typename func_t, typename index_t, typename stride_t>
C10_LAUNCH_BOUNDS_2(nt, launch_bound2)
__global__ void index_elementwise_kernel_2_2(
    int n,
    char* out_ptr, char* in_ptr,
    int64_t s0, int64_t s1, int64_t st0, int64_t st1, char* index_ptr0, char* index_ptr1,
    index_t size0,
    index_t size1,
    stride_t stride00, stride_t stride01, stride_t stride02,
    stride_t stride10, stride_t stride11, stride_t stride12,
    func_t f) {
  int tid = threadIdx.x;
  int nv = nt * vt;
  int64_t idx = nv * blockIdx.x + tid;
  constexpr int num_indices = 2;
  #pragma unroll
  for (int i = 0; i < vt; i++) {
    if (idx < n) {
      // -----------------------------------------------
      // auto offsets = offset_calc.get(idx);
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
      divmod_div = linear_idx / size1;
      divmod_mod = linear_idx % size1;
      linear_idx = divmod_div;
      arg = 0;
      offsets[arg] += divmod_mod * stride10;
      arg = 1;
      offsets[arg] += divmod_mod * stride11;
      arg = 2;
      offsets[arg] += divmod_mod * stride12;
      // -----------------------------------------------
      // char* out_data = out_ptr + offsets[0];
      // char* in_data = in_ptr + offsets[1];
      // int64_t offset = 0;
      // #pragma unroll
      // for (int i = 0; i < num_indices; i++) {
      //   int64_t index = *(int64_t*)(index_ptrs[i] + offsets[2]);
      //   // CUDA_KERNEL_ASSERT(index >= -sizes[i] && index < sizes[i] && "index out of bounds");
      //   if (index < 0) {
      //     index += sizes[i];
      //   }
      //   offset += index * strides[i];
      // }
      // f(out_data, in_data, offset);
      char* out_data = out_ptr + offsets[0];
      char* in_data = in_ptr + offsets[1];
      int64_t offset = 0;

      //int64_t s0, int64_t s1, int64_t st0, int64_t st1, char* index_ptr0, char* index_ptr1,
      int64_t index = *(int64_t*)(index_ptr0 + offsets[2]);
      // CUDA_KERNEL_ASSERT(index >= -s0 && index < s0 && "index out of bounds");
      if (index < 0) {
        index += s0;
      }
      offset += index * st0;
      
      index = *(int64_t*)(index_ptr1 + offsets[2]);
      // CUDA_KERNEL_ASSERT(index >= -s1 && index < s1 && "index out of bounds");
      if (index < 0) {
        index += s1;
      }
      offset += index * st1;

      f(out_data, in_data, offset);
      // -----------------------------------------------
      idx += nt;
    }
  }
}

template<int nt, int vt, typename func_t, typename index_t, typename stride_t>
C10_LAUNCH_BOUNDS_2(nt, launch_bound2)
__global__ void index_elementwise_kernel_2_3(
    int n,
    char* out_ptr, char* in_ptr,
    int64_t s0, int64_t s1, int64_t st0, int64_t st1, char* index_ptr0, char* index_ptr1,
    index_t size0,
    index_t size1,
    index_t size2,
    stride_t stride00, stride_t stride01, stride_t stride02,
    stride_t stride10, stride_t stride11, stride_t stride12,
    stride_t stride20, stride_t stride21, stride_t stride22,
    func_t f) {
  int tid = threadIdx.x;
  int nv = nt * vt;
  int64_t idx = nv * blockIdx.x + tid;
  constexpr int num_indices = 2;
  #pragma unroll
  for (int i = 0; i < vt; i++) {
    if (idx < n) {
      // -----------------------------------------------
      // auto offsets = offset_calc.get(idx);
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
      divmod_div = linear_idx / size1;
      divmod_mod = linear_idx % size1;
      linear_idx = divmod_div;
      arg = 0;
      offsets[arg] += divmod_mod * stride10;
      arg = 1;
      offsets[arg] += divmod_mod * stride11;
      arg = 2;
      offsets[arg] += divmod_mod * stride12;
      divmod_div = linear_idx / size2;
      divmod_mod = linear_idx % size2;
      linear_idx = divmod_div;
      arg = 0;
      offsets[arg] += divmod_mod * stride20;
      arg = 1;
      offsets[arg] += divmod_mod * stride21;
      arg = 2;
      offsets[arg] += divmod_mod * stride22;
      // -----------------------------------------------
      // char* out_data = out_ptr + offsets[0];
      // char* in_data = in_ptr + offsets[1];
      // int64_t offset = 0;
      // #pragma unroll
      // for (int i = 0; i < num_indices; i++) {
      //   int64_t index = *(int64_t*)(index_ptrs[i] + offsets[2]);
      //   // CUDA_KERNEL_ASSERT(index >= -sizes[i] && index < sizes[i] && "index out of bounds");
      //   if (index < 0) {
      //     index += sizes[i];
      //   }
      //   offset += index * strides[i];
      // }
      // f(out_data, in_data, offset);
      char* out_data = out_ptr + offsets[0];
      char* in_data = in_ptr + offsets[1];
      int64_t offset = 0;

      //int64_t s0, int64_t s1, int64_t st0, int64_t st1, char* index_ptr0, char* index_ptr1,
      int64_t index = *(int64_t*)(index_ptr0 + offsets[2]);
      // CUDA_KERNEL_ASSERT(index >= -s0 && index < s0 && "index out of bounds");
      if (index < 0) {
        index += s0;
      }
      offset += index * st0;
      
      index = *(int64_t*)(index_ptr1 + offsets[2]);
      // CUDA_KERNEL_ASSERT(index >= -s1 && index < s1 && "index out of bounds");
      if (index < 0) {
        index += s1;
      }
      offset += index * st1;

      f(out_data, in_data, offset);
      // -----------------------------------------------
      idx += nt;
    }
  }
}



template<int nt, int vt, typename func_t, typename index_t, typename stride_t>
static void launch_kernel_index_2_1(
    int64_t N,
    char* out_ptr, char* in_ptr,
    int64_t s0, int64_t s1, int64_t st0, int64_t st1, char* index_ptr0, char* index_ptr1,
    index_t size0,
    stride_t stride00, stride_t stride01, stride_t stride02,
    const func_t& f
) {
  TORCH_INTERNAL_ASSERT(N >= 0 && N <= std::numeric_limits<int32_t>::max());
  if (N == 0) {
    return;
  }
  dim3 block(nt);
  dim3 grid((N + block.x * vt - 1) / (block.x * vt));
  auto stream = at::cuda::getCurrentCUDAStream();
  index_elementwise_kernel_2_1<nt, vt, func_t><<<grid, block, 0, stream>>>(
    N,
    out_ptr, in_ptr,
    s0, s1, st0, st1, index_ptr0, index_ptr1,
    size0,
    stride00, stride01, stride02,
    f
  );
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template<int nt, int vt, typename func_t, typename index_t, typename stride_t>
static void launch_kernel_index_2_2(
    int64_t N,
    char* out_ptr, char* in_ptr,
    int64_t s0, int64_t s1, int64_t st0, int64_t st1, char* index_ptr0, char* index_ptr1,
    index_t size0,
    index_t size1,
    stride_t stride00, stride_t stride01, stride_t stride02,
    stride_t stride10, stride_t stride11, stride_t stride12,
    const func_t& f
) {
  TORCH_INTERNAL_ASSERT(N >= 0 && N <= std::numeric_limits<int32_t>::max());
  if (N == 0) {
    return;
  }
  dim3 block(nt);
  dim3 grid((N + block.x * vt - 1) / (block.x * vt));
  auto stream = at::cuda::getCurrentCUDAStream();
  index_elementwise_kernel_2_2<nt, vt, func_t><<<grid, block, 0, stream>>>(
    N,
    out_ptr, in_ptr,
    s0, s1, st0, st1, index_ptr0, index_ptr1,
    size0,
    size1,
    stride00, stride01, stride02,
    stride10, stride11, stride12,
    f
  );
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template<int nt, int vt, typename func_t, typename index_t, typename stride_t>
static void launch_kernel_index_2_3(
    int64_t N,
    char* out_ptr, char* in_ptr,
    int64_t s0, int64_t s1, int64_t st0, int64_t st1, char* index_ptr0, char* index_ptr1,
    index_t size0,
    index_t size1,
    index_t size2,
    stride_t stride00, stride_t stride01, stride_t stride02,
    stride_t stride10, stride_t stride11, stride_t stride12,
    stride_t stride20, stride_t stride21, stride_t stride22,
    const func_t& f
) {
  TORCH_INTERNAL_ASSERT(N >= 0 && N <= std::numeric_limits<int32_t>::max());
  if (N == 0) {
    return;
  }
  dim3 block(nt);
  dim3 grid((N + block.x * vt - 1) / (block.x * vt));
  auto stream = at::cuda::getCurrentCUDAStream();
  index_elementwise_kernel_2_3<nt, vt, func_t><<<grid, block, 0, stream>>>(
    N,
    out_ptr, in_ptr,
    s0, s1, st0, st1, index_ptr0, index_ptr1,
    size0,
    size1,
    size2,
    stride00, stride01, stride02,
    stride10, stride11, stride12,
    stride20, stride21, stride22,
    f
  );
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template<int nt, int vt, typename func_t, typename index_t, typename stride_t>
static void launch_kernel_index_1_1(
    int64_t N,
    char* out_ptr, char* in_ptr,
    int64_t s0, int64_t st0, char* index_ptr0,
    index_t size0,
    stride_t stride00, stride_t stride01, stride_t stride02,
    const func_t& f
) {
  TORCH_INTERNAL_ASSERT(N >= 0 && N <= std::numeric_limits<int32_t>::max());
  if (N == 0) {
    return;
  }
  dim3 block(nt);
  dim3 grid((N + block.x * vt - 1) / (block.x * vt));
  auto stream = at::cuda::getCurrentCUDAStream();
  index_elementwise_kernel_1_1<nt, vt, func_t><<<grid, block, 0, stream>>>(
    N,
    out_ptr, in_ptr,
    s0, st0, index_ptr0,
    size0,
    stride00, stride01, stride02,
    f
  );
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template<int nt, int vt, typename func_t, typename index_t, typename stride_t>
static void launch_kernel_index_1_2(
    int64_t N,
    char* out_ptr, char* in_ptr,
    int64_t s0, int64_t st0, char* index_ptr0,
    index_t size0,
    index_t size1,
    stride_t stride00, stride_t stride01, stride_t stride02,
    stride_t stride10, stride_t stride11, stride_t stride12,
    const func_t& f
) {
  TORCH_INTERNAL_ASSERT(N >= 0 && N <= std::numeric_limits<int32_t>::max());
  if (N == 0) {
    return;
  }
  dim3 block(nt);
  dim3 grid((N + block.x * vt - 1) / (block.x * vt));
  auto stream = at::cuda::getCurrentCUDAStream();
  index_elementwise_kernel_1_2<nt, vt, func_t><<<grid, block, 0, stream>>>(
    N,
    out_ptr, in_ptr,
    s0, st0, index_ptr0,
    size0,
    size1,
    stride00, stride01, stride02,
    stride10, stride11, stride12,
    f
  );
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template<int nt, int vt, typename func_t, typename index_t, typename stride_t>
static void launch_kernel_index_1_2_broadcast(
    int64_t N,
    char* out_ptr, char* in_ptr,
    int64_t s0, int64_t st0, char* index_ptr0,
    index_t size0,
    index_t size1,
    stride_t stride00, stride_t stride01, stride_t stride02,
    stride_t stride10, stride_t stride11, stride_t stride12,
    const func_t& f, ScalarType arg_t, ScalarType idx_t
) {
  // pattern: s0,s1,1,s0,0,s0,1,0
  TORCH_INTERNAL_ASSERT(N >= 0 && N <= std::numeric_limits<int32_t>::max());
  if (N == 0) {
    return;
  }
  auto stream = at::cuda::getCurrentCUDAStream();
  int block_dim_x = 64;
  int vec_data_0 = getCastVectorizedAlignment(arg_t, (void*)out_ptr, size0);
  // Notice: offsets required vectorized alignment when Half or BFloat16.
  if (arg_t == ScalarType::Half || arg_t == ScalarType::BFloat16) {
    vec_data_0 = std::min(vec_data_0, getCastVectorizedAlignment(arg_t, (void*)(out_ptr + stride10), size0));
  }
  // TODO: modify getVectorizedAlignment func for long type
  int vec_data_idx = getCastVectorizedAlignment(idx_t, (void*)index_ptr0, size0);

  // for most size0, which are not multiples of warp_size, vec=1 works better
  // int vec = std::min(vec_data_0, vec_data_idx);
  int vec = 1;
  dim3 block(block_dim_x);

  int grid_dim_x = std::ceil(float(size0)/float(block_dim_x * vec));
  TORCH_INTERNAL_ASSERT(grid_dim_x > 0);
  int grid_dim_y = getMaxGridSize(grid_dim_x, size1);
  TORCH_INTERNAL_ASSERT(grid_dim_y > 0);
  // if y_t is too large(for example > 32), this kernel will take a long time
  int y_t = std::min(int(size1 / grid_dim_y), 32);

  grid_dim_y = std::ceil(float(size1)/float(y_t));
  // last block handle y_remain
  int y_remain = size1 - (grid_dim_y - 1) * y_t;
  dim3 grid(grid_dim_x, grid_dim_y, 1);
  TORCH_INTERNAL_ASSERT(y_t > 0);
  TORCH_INTERNAL_ASSERT(y_remain > 0);
#define SWITCH_INDEX_ELEMENTWISE_KERNEL(nt, v_x, f, ...)    \
  if (v_x == 8) {                                           \
    f(nt, 8, __VA_ARGS__);                                  \
  } else if (v_x == 4) {                                    \
    f(nt, 4, __VA_ARGS__);                                  \
  } else if (v_x == 2) {                                    \
    f(nt, 2, __VA_ARGS__);                                  \
  } else {                                                  \
    f(nt, 1, __VA_ARGS__);                                  \
  }

#define INDEX_ELEMENTWISE_KERNEL(nt, v_x, arg_t, idx_t, func_t)                                      \
  index_elementwise_kernel_1_2_broadcast<nt, v_x, arg_t, idx_t, func_t><<<grid, block, 0, stream>>>( \
      N, out_ptr, in_ptr, s0, st0, index_ptr0, size0, size1,                                         \
      stride00, stride01, stride02, stride10, stride11, stride12,                                    \
      f, y_t, y_remain);                                                                             \
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  if (arg_t == ScalarType::Float) {
    SWITCH_INDEX_ELEMENTWISE_KERNEL(nt, vec, INDEX_ELEMENTWISE_KERNEL, float, int64_t, func_t);
  } else if (arg_t == ScalarType::Half) {
    SWITCH_INDEX_ELEMENTWISE_KERNEL(nt, vec, INDEX_ELEMENTWISE_KERNEL, at::Half, int64_t, func_t);
  } else if (arg_t == ScalarType::BFloat16) {
    SWITCH_INDEX_ELEMENTWISE_KERNEL(nt, vec, INDEX_ELEMENTWISE_KERNEL, at::BFloat16, int64_t, func_t);
  } else if (arg_t == ScalarType::Char) {
    SWITCH_INDEX_ELEMENTWISE_KERNEL(nt, vec, INDEX_ELEMENTWISE_KERNEL, int8_t, int64_t, func_t);
  } else {
    TORCH_CHECK(false, "unspoorted dtype in index elementwise kernel 1_2 broadcast!");
  }

#undef INDEX_ELEMENTWISE_KERNEL
#undef SWITCH_INDEX_ELEMENTWISE_KERNEL
}

template<int nt, int vt, typename func_t, typename index_t, typename stride_t>
static void launch_kernel_index_1_2_broadcast_transpose(
    int64_t N,
    char* out_ptr, char* in_ptr,
    int64_t s0, int64_t st0, char* index_ptr0,
    index_t size0,
    index_t size1,
    stride_t stride00, stride_t stride01, stride_t stride02,
    stride_t stride10, stride_t stride11, stride_t stride12,
    const func_t& f, ScalarType arg_t, ScalarType idx_t
) {
  // pattern: s0,s1,1,s0,1,s0,0,1
  TORCH_INTERNAL_ASSERT(N >= 0 && N <= std::numeric_limits<int32_t>::max());
  if (N == 0) {
    return;
  }
  auto stream = at::cuda::getCurrentCUDAStream();
  constexpr int block_dim_x = 64;
  int vec_data_0 = getCastVectorizedAlignment(arg_t, (void*)out_ptr, size0);
  // Notice: offsets required vectorized alignment when Half or BFloat16.
  if (arg_t == ScalarType::Half || arg_t == ScalarType::BFloat16) {
    vec_data_0 = std::min(vec_data_0, getCastVectorizedAlignment(arg_t, (void*)(out_ptr + stride10), size0));
  }
  // TODO: modify getVectorizedAlignment func for long type
  int vec_data_idx = getCastVectorizedAlignment(idx_t, (void*)index_ptr0, size1);
  TORCH_INTERNAL_ASSERT(vec_data_idx <= 4);

  dim3 block(block_dim_x);

  int grid_dim_x = std::ceil(float(size0)/float(block_dim_x * vec_data_0));
  TORCH_INTERNAL_ASSERT(grid_dim_x > 0);
  int grid_dim_y = size1 / vec_data_idx;
  TORCH_INTERNAL_ASSERT(grid_dim_y > 0);
  dim3 grid(grid_dim_x, grid_dim_y, 1);
#define SWITCH_INDEX_ELEMENTWISE_KERNEL(nt, v_x, v_y, f, ...)   \
  if (v_x == 8 && v_y == 4) {                                   \
    f(nt, 8, 4, __VA_ARGS__);                                   \
  } else if (v_x == 8 && v_y == 2) {                            \
    f(nt, 8, 2, __VA_ARGS__);                                   \
  } else if (v_x == 8 && v_y == 1) {                            \
    f(nt, 8, 1, __VA_ARGS__);                                   \
  } else if (v_x == 4 && v_y == 4) {                            \
    f(nt, 4, 4, __VA_ARGS__);                                   \
  } else if (v_x == 4 && v_y == 2) {                            \
    f(nt, 4, 2, __VA_ARGS__);                                   \
  } else if (v_x == 4 && v_y == 1) {                            \
    f(nt, 4, 1, __VA_ARGS__);                                   \
  } else if (v_x == 2 && v_y == 4) {                            \
    f(nt, 2, 4, __VA_ARGS__);                                   \
  } else if (v_x == 2 && v_y == 2) {                            \
    f(nt, 2, 2, __VA_ARGS__);                                   \
  } else if (v_x == 2 && v_y == 1) {                            \
    f(nt, 2, 1, __VA_ARGS__);                                   \
  } else if (v_x == 1 && v_y == 4) {                            \
    f(nt, 1, 4, __VA_ARGS__);                                   \
  } else if (v_x == 1 && v_y == 2) {                            \
    f(nt, 1, 2, __VA_ARGS__);                                   \
  } else {                                                      \
    f(nt, 1, 1, __VA_ARGS__);                                   \
  }

#define INDEX_ELEMENTWISE_KERNEL(nt, v_x, v_y, arg_t, idx_t, func_t)                                      \
  index_elementwise_kernel_1_2_broadcast_transpose<nt, v_x, v_y, arg_t, idx_t, func_t><<<grid, block, 0, stream>>>( \
      N, out_ptr, in_ptr, s0, st0, index_ptr0, size0, size1,                                         \
      stride00, stride01, stride02, stride10, stride11, stride12,                                    \
      f);                                                                             \
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  if (arg_t == ScalarType::Float) {
    SWITCH_INDEX_ELEMENTWISE_KERNEL(nt, vec_data_0, vec_data_idx, INDEX_ELEMENTWISE_KERNEL, float, int64_t, func_t);
  } else if (arg_t == ScalarType::Half) {
    SWITCH_INDEX_ELEMENTWISE_KERNEL(nt, vec_data_0, vec_data_idx, INDEX_ELEMENTWISE_KERNEL, at::Half, int64_t, func_t);
  } else if (arg_t == ScalarType::BFloat16) {
    SWITCH_INDEX_ELEMENTWISE_KERNEL(nt, vec_data_0, vec_data_idx, INDEX_ELEMENTWISE_KERNEL, at::BFloat16, int64_t, func_t);
  } else if (arg_t == ScalarType::Char) {
    SWITCH_INDEX_ELEMENTWISE_KERNEL(nt, vec_data_0, vec_data_idx, INDEX_ELEMENTWISE_KERNEL, int8_t, int64_t, func_t);
  } else {
    TORCH_CHECK(false, "unspoorted dtype in index elementwise kernel 1_2 broadcast transpose!");
  }

#undef INDEX_ELEMENTWISE_KERNEL
#undef SWITCH_INDEX_ELEMENTWISE_KERNEL
}

template<int nt, int vt, typename func_t, typename index_t, typename stride_t>
static void launch_kernel_index_1_3(
    int64_t N,
    char* out_ptr, char* in_ptr,
    int64_t s0, int64_t st0, char* index_ptr0,
    index_t size0,
    index_t size1,
    index_t size2,
    stride_t stride00, stride_t stride01, stride_t stride02,
    stride_t stride10, stride_t stride11, stride_t stride12,
    stride_t stride20, stride_t stride21, stride_t stride22,
    const func_t& f
) {
  TORCH_INTERNAL_ASSERT(N >= 0 && N <= std::numeric_limits<int32_t>::max());
  if (N == 0) {
    return;
  }
  dim3 block(nt);
  dim3 grid((N + block.x * vt - 1) / (block.x * vt));
  auto stream = at::cuda::getCurrentCUDAStream();
  index_elementwise_kernel_1_3<nt, vt, func_t><<<grid, block, 0, stream>>>(
    N,
    out_ptr, in_ptr,
    s0, st0, index_ptr0,
    size0,
    size1,
    size2,
    stride00, stride01, stride02,
    stride10, stride11, stride12,
    stride20, stride21, stride22,
    f
  );
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template<int nt, int vt, typename func_t, typename index_t, typename stride_t>
static void launch_kernel_index_3_1(
    int64_t N,
    char* out_ptr, char* in_ptr,
    int64_t s0, int64_t s1, int64_t s2, int64_t st0,  int64_t st1, int64_t st2, char* index_ptr0, char* index_ptr1, char* index_ptr2,
    index_t size0,
    stride_t stride00, stride_t stride01, stride_t stride02,
    const func_t& f
) {
  TORCH_INTERNAL_ASSERT(N >= 0 && N <= std::numeric_limits<int32_t>::max());
  if (N == 0) {
    return;
  }
  dim3 block(nt);
  dim3 grid((N + block.x * vt - 1) / (block.x * vt));
  auto stream = at::cuda::getCurrentCUDAStream();
  index_elementwise_kernel_3_1<nt, vt, func_t><<<grid, block, 0, stream>>>(
    N,
    out_ptr, in_ptr,
    s0, s1, s2, st0, st1, st2, index_ptr0, index_ptr1, index_ptr2,
    size0,
    stride00, stride01, stride02,
    f
  );
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template<int nt, int vt, typename func_t, typename index_t, typename stride_t>
static void launch_kernel_index_3_2(
    int64_t N,
    char* out_ptr, char* in_ptr,
    int64_t s0, int64_t s1, int64_t s2, int64_t st0,  int64_t st1, int64_t st2, char* index_ptr0, char* index_ptr1, char* index_ptr2,
    index_t size0,
    index_t size1,
    stride_t stride00, stride_t stride01, stride_t stride02,
    stride_t stride10, stride_t stride11, stride_t stride12,
    const func_t& f
) {
  TORCH_INTERNAL_ASSERT(N >= 0 && N <= std::numeric_limits<int32_t>::max());
  if (N == 0) {
    return;
  }
  dim3 block(nt);
  dim3 grid((N + block.x * vt - 1) / (block.x * vt));
  auto stream = at::cuda::getCurrentCUDAStream();
  index_elementwise_kernel_3_2<nt, vt, func_t><<<grid, block, 0, stream>>>(
    N,
    out_ptr, in_ptr,
    s0, s1, s2, st0, st1, st2, index_ptr0, index_ptr1, index_ptr2,
    size0,
    size1,
    stride00, stride01, stride02,
    stride10, stride11, stride12,
    f
  );
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template<int nt, int vt, typename func_t, typename index_t, typename stride_t>
static void launch_kernel_index_3_3(
    int64_t N,
    char* out_ptr, char* in_ptr,
    int64_t s0, int64_t s1, int64_t s2, int64_t st0,  int64_t st1, int64_t st2, char* index_ptr0, char* index_ptr1, char* index_ptr2,
    index_t size0,
    index_t size1,
    index_t size2,
    stride_t stride00, stride_t stride01, stride_t stride02,
    stride_t stride10, stride_t stride11, stride_t stride12,
    stride_t stride20, stride_t stride21, stride_t stride22,
    const func_t& f
) {
  TORCH_INTERNAL_ASSERT(N >= 0 && N <= std::numeric_limits<int32_t>::max());
  if (N == 0) {
    return;
  }
  dim3 block(nt);
  dim3 grid((N + block.x * vt - 1) / (block.x * vt));
  auto stream = at::cuda::getCurrentCUDAStream();
  index_elementwise_kernel_3_3<nt, vt, func_t><<<grid, block, 0, stream>>>(
    N,
    out_ptr, in_ptr,
    s0, s1, s2, st0, st1, st2, index_ptr0, index_ptr1, index_ptr2,
    size0,
    size1,
    size2,
    stride00, stride01, stride02,
    stride10, stride11, stride12,
    stride20, stride21, stride22,
    f
  );
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template<int nt, int vt, typename scalar_t, typename func_t, typename index_t, typename stride_t>
C10_LAUNCH_BOUNDS_2(nt, launch_bound2)
__global__ void flip_elementwise_kernel_1_2(
    int n,
    char* const __restrict__ out_ptr, const char* const __restrict__ in_ptr,
    index_t size0,
    index_t size1,
    stride_t stride00, stride_t stride01, 
    stride_t stride10, stride_t stride11, 
    func_t f) {
  int tid = threadIdx.x;
  int nv = nt * vt;
  int64_t idx = nv * blockIdx.x + tid;
  #pragma unroll
  for (int i = 0; i < vt; i++) {
    if (idx < n) {
      // -----------------------------------------------
      // const auto offset_calc = make_offset_calculator<2, /*signed_strides=*/true>(iter);
      // auto offsets = offset_calc.get(idx);
      int64_t offsets[2];
      auto linear_idx = idx;
      constexpr int NARGS = 2;
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
      divmod_div = linear_idx / size1;
      divmod_mod = linear_idx % size1;
      linear_idx = divmod_div;
      arg = 0;
      offsets[arg] += divmod_mod * stride10;
      arg = 1;
      offsets[arg] += divmod_mod * stride11;
      // -----------------------------------------------
      // offsets can be negative here, but it's fine
      // scalar_t* const __restrict__ out_data = reinterpret_cast<scalar_t*>(out_ptr + offsets[0]);
      // const scalar_t* const __restrict__ in_data = reinterpret_cast<const scalar_t*>(in_ptr + offsets[1]);
      // *out_data = *in_data;
      scalar_t* const __restrict__ out_data = reinterpret_cast<scalar_t*>(out_ptr + offsets[0]);
      const scalar_t* const __restrict__ in_data = reinterpret_cast<const scalar_t*>(in_ptr + offsets[1]);
      *out_data = *in_data;
      // -----------------------------------------------
      idx += nt;
    }
  }
}

template<int nt, int vt, typename scalar_t, typename func_t, typename index_t, typename stride_t>
static void launch_kernel_flip_1_2(
    int64_t N,
    char* const __restrict__ out_ptr, const char* const __restrict__ in_ptr,
    index_t size0, index_t size1,
    stride_t stride00, stride_t stride01,
    stride_t stride10, stride_t stride11,
    const func_t& f
) {
  // one input, ndim=2
  TORCH_INTERNAL_ASSERT(N >= 0 && N <= std::numeric_limits<int32_t>::max());
  if (N == 0) {
    return;
  }
  dim3 block(nt);
  dim3 grid((N + block.x * vt - 1) / (block.x * vt));
  auto stream = at::cuda::getCurrentCUDAStream();
  flip_elementwise_kernel_1_2<nt, vt, scalar_t, func_t><<<grid, block, 0, stream>>>(
    N,
    out_ptr, in_ptr,
    size0,
    size1,
    stride00, stride01, 
    stride10, stride11, 
    f
  );
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename func_t>
void gpu_index_kernel_maca(TensorIteratorBase& iter, IntArrayRef index_size, IntArrayRef index_stride, const func_t& f, bool is_index_put=false) {
  int num_indices = index_size.size();
  AT_ASSERT(num_indices == index_stride.size());
  AT_ASSERT(num_indices == iter.ntensors() - 2);

  if (iter.numel() == 0) {
    return;
  }

  if (!iter.can_use_32bit_indexing()) {
    for (auto& sub_iter : iter.with_32bit_indexing()) {
      gpu_index_kernel_maca(sub_iter, index_size, index_stride, f, is_index_put);
    }
    return;
  }

  auto sizes = at::detail::Array<int64_t, MAX_DIMS>(0);
  auto strides = at::detail::Array<int64_t, MAX_DIMS>(0);
  auto index_ptrs = at::detail::Array<char*, MAX_DIMS>(nullptr);
  for (int i = 0; i < num_indices; i++) {
    sizes[i] = index_size[i];
    strides[i] = index_stride[i];
    index_ptrs[i] = (char*)iter.data_ptr(i + 2);
  }

  char* out_ptr = (char*)iter.data_ptr(0);
  char* in_ptr = (char*)iter.data_ptr(1);

  auto offset_calc = make_offset_calculator<3>(iter);

  if (num_indices == 2 && iter.ndim() == 1) {
    launch_kernel_index_2_1<launch_size_nd, launch_bound2>(
      iter.numel(),
      out_ptr, in_ptr,
      sizes[0], sizes[1], strides[0], strides[1], index_ptrs[0], index_ptrs[1],
      offset_calc.sizes_[0].divisor,
      offset_calc.strides_[0][0], offset_calc.strides_[0][1], offset_calc.strides_[0][2],
      f
    );
  } else if (num_indices == 2 && iter.ndim() == 2) {
    launch_kernel_index_2_2<launch_size_nd, launch_bound2>(
      iter.numel(),
      out_ptr, in_ptr,
      sizes[0], sizes[1], strides[0], strides[1], index_ptrs[0], index_ptrs[1],
      offset_calc.sizes_[0].divisor,
      offset_calc.sizes_[1].divisor,
      offset_calc.strides_[0][0], offset_calc.strides_[0][1], offset_calc.strides_[0][2],
      offset_calc.strides_[1][0], offset_calc.strides_[1][1], offset_calc.strides_[1][2],
      f
    );
  } else if (num_indices == 2 && iter.ndim() == 3) {
    launch_kernel_index_2_3<launch_size_nd, launch_bound2>(
      iter.numel(),
      out_ptr, in_ptr,
      sizes[0], sizes[1], strides[0], strides[1], index_ptrs[0], index_ptrs[1],
      offset_calc.sizes_[0].divisor,
      offset_calc.sizes_[1].divisor,
      offset_calc.sizes_[2].divisor,
      offset_calc.strides_[0][0], offset_calc.strides_[0][1], offset_calc.strides_[0][2],
      offset_calc.strides_[1][0], offset_calc.strides_[1][1], offset_calc.strides_[1][2],
      offset_calc.strides_[2][0], offset_calc.strides_[2][1], offset_calc.strides_[2][2],
      f
    );
  } else if (num_indices == 1 && iter.ndim() == 1) {
    launch_kernel_index_1_1<launch_size_nd, launch_bound2>(
      iter.numel(),
      out_ptr, in_ptr,
      sizes[0], strides[0], index_ptrs[0],
      offset_calc.sizes_[0].divisor,
      offset_calc.strides_[0][0], offset_calc.strides_[0][1], offset_calc.strides_[0][2],
      f
    );
  } else if (num_indices == 1 && iter.ndim() == 2) {
    bool disable_index_1_2_broadcast = at::maca::get_maca_disable_index_elementwise_1_2_broadcast_kernel();
    auto s0 = offset_calc.sizes_[0].divisor;
    auto s1 = offset_calc.sizes_[1].divisor;
    const int narity = 2;
    if (maca_likely(!disable_index_1_2_broadcast) && !is_index_put && 
      ((iter.dtype(0)== ScalarType::BFloat16 && iter.dtype(1)== ScalarType::BFloat16 && offset_calc.strides_[0][0] == 2 && 
      offset_calc.strides_[1][0] == 2 * s0) || (iter.dtype(0)== ScalarType::Half && iter.dtype(1)== ScalarType::Half && 
      offset_calc.strides_[0][0] == 2 && offset_calc.strides_[1][0] == 2 * s0) || (iter.dtype(0)== ScalarType::Float && 
      iter.dtype(1)== ScalarType::Float && offset_calc.strides_[0][0] == 4 && offset_calc.strides_[1][0] == 4 * s0) || 
      (iter.dtype(0)== ScalarType::Char && iter.dtype(1)== ScalarType::Char && offset_calc.strides_[0][0] == 1 && offset_calc.strides_[1][0] == s0)) 
      && offset_calc.strides_[0][2] == 8 && offset_calc.strides_[1][2] == 0) {
      get_elementwise_info<narity + 1>(2, narity, offset_calc, "p_e_launch_legacy_index_broadcast_kernel_maca_1_2", f);
      launch_kernel_index_1_2_broadcast<launch_size_nd, launch_bound2>(
        iter.numel(),
        out_ptr, in_ptr,
        sizes[0], strides[0], index_ptrs[0],
        offset_calc.sizes_[0].divisor,
        offset_calc.sizes_[1].divisor,
        offset_calc.strides_[0][0], offset_calc.strides_[0][1], offset_calc.strides_[0][2],
        offset_calc.strides_[1][0], offset_calc.strides_[1][1], offset_calc.strides_[1][2],
        f, iter.dtype(0), iter.dtype(1)
      );
    } else if (maca_likely(!disable_index_1_2_broadcast) && !is_index_put && 
      ((iter.dtype(0)== ScalarType::BFloat16 && iter.dtype(1)== ScalarType::BFloat16 && offset_calc.strides_[0][0] == 2 && 
      offset_calc.strides_[1][0] == 2 * s0) || (iter.dtype(0)== ScalarType::Half && iter.dtype(1)== ScalarType::Half && 
      offset_calc.strides_[0][0] == 2 && offset_calc.strides_[1][0] == 2 * s0) || (iter.dtype(0)== ScalarType::Float && 
      iter.dtype(1)== ScalarType::Float && offset_calc.strides_[0][0] == 4 && offset_calc.strides_[1][0] == 4 * s0) || 
      (iter.dtype(0)== ScalarType::Char && iter.dtype(1)== ScalarType::Char && offset_calc.strides_[0][0] == 1 && offset_calc.strides_[1][0] == s0)) 
      && offset_calc.strides_[0][2] == 0 && offset_calc.strides_[1][2] == 8) {
      get_elementwise_info<narity + 1>(2, narity, offset_calc, "p_e_launch_legacy_index_broadcast_transpose_kernel_maca_1_2", f);
      launch_kernel_index_1_2_broadcast_transpose<launch_size_nd, launch_bound2>(
        iter.numel(),
        out_ptr, in_ptr,
        sizes[0], strides[0], index_ptrs[0],
        offset_calc.sizes_[0].divisor,
        offset_calc.sizes_[1].divisor,
        offset_calc.strides_[0][0], offset_calc.strides_[0][1], offset_calc.strides_[0][2],
        offset_calc.strides_[1][0], offset_calc.strides_[1][1], offset_calc.strides_[1][2],
        f, iter.dtype(0), iter.dtype(2)
      );
    } else {
      get_elementwise_info<narity + 1>(2, narity, offset_calc, "p_e_launch_legacy_index_kernel_maca_1_2", f);
      launch_kernel_index_1_2<launch_size_nd, launch_bound2>(
        iter.numel(),
        out_ptr, in_ptr,
        sizes[0], strides[0], index_ptrs[0],
        offset_calc.sizes_[0].divisor,
        offset_calc.sizes_[1].divisor,
        offset_calc.strides_[0][0], offset_calc.strides_[0][1], offset_calc.strides_[0][2],
        offset_calc.strides_[1][0], offset_calc.strides_[1][1], offset_calc.strides_[1][2],
        f
      );
    }
  } else if (num_indices == 1 && iter.ndim() == 3) {
    launch_kernel_index_1_3<launch_size_nd, launch_bound2>(
      iter.numel(),
      out_ptr, in_ptr,
      sizes[0], strides[0], index_ptrs[0],
      offset_calc.sizes_[0].divisor,
      offset_calc.sizes_[1].divisor,
      offset_calc.sizes_[2].divisor,
      offset_calc.strides_[0][0], offset_calc.strides_[0][1], offset_calc.strides_[0][2],
      offset_calc.strides_[1][0], offset_calc.strides_[1][1], offset_calc.strides_[1][2],
      offset_calc.strides_[2][0], offset_calc.strides_[2][1], offset_calc.strides_[2][2],
      f
    );
  } else if (num_indices == 3 && iter.ndim() == 1) {
    launch_kernel_index_3_1<launch_size_nd, launch_bound2>(
      iter.numel(),
      out_ptr, in_ptr,
      sizes[0], sizes[1], sizes[2], strides[0], strides[1], strides[2], index_ptrs[0], index_ptrs[1], index_ptrs[2],
      offset_calc.sizes_[0].divisor,
      offset_calc.strides_[0][0], offset_calc.strides_[0][1], offset_calc.strides_[0][2],
      f
    );
  } else if (num_indices == 3 && iter.ndim() == 2) {
    launch_kernel_index_3_2<launch_size_nd, launch_bound2>(
      iter.numel(),
      out_ptr, in_ptr,
      sizes[0], sizes[1], sizes[2], strides[0], strides[1], strides[2], index_ptrs[0], index_ptrs[1], index_ptrs[2],
      offset_calc.sizes_[0].divisor,
      offset_calc.sizes_[1].divisor,
      offset_calc.strides_[0][0], offset_calc.strides_[0][1], offset_calc.strides_[0][2],
      offset_calc.strides_[1][0], offset_calc.strides_[1][1], offset_calc.strides_[1][2],
      f
    );
  } else if (num_indices == 3 && iter.ndim() == 3) {
    launch_kernel_index_3_3<launch_size_nd, launch_bound2>(
      iter.numel(),
      out_ptr, in_ptr,
      sizes[0], sizes[1], sizes[2], strides[0], strides[1], strides[2], index_ptrs[0], index_ptrs[1], index_ptrs[2],
      offset_calc.sizes_[0].divisor,
      offset_calc.sizes_[1].divisor,
      offset_calc.sizes_[2].divisor,
      offset_calc.strides_[0][0], offset_calc.strides_[0][1], offset_calc.strides_[0][2],
      offset_calc.strides_[1][0], offset_calc.strides_[1][1], offset_calc.strides_[1][2],
      offset_calc.strides_[2][0], offset_calc.strides_[2][1], offset_calc.strides_[2][2],
      f
    );
  } else {
    launch_kernel<launch_size_nd, launch_bound2>(iter.numel(), [=]__device__(int idx) {
      auto offsets = offset_calc.get(idx);
      char* out_data = out_ptr + offsets[0];
      char* in_data = in_ptr + offsets[1];

      int64_t offset = 0;
      #pragma unroll
      for (int i = 0; i < num_indices; i++) {
        int64_t index = *(int64_t*)(index_ptrs[i] + offsets[2]);
        CUDA_KERNEL_ASSERT(index >= -sizes[i] && index < sizes[i] && "index out of bounds");
        if (index < 0) {
          index += sizes[i];
        }
        offset += index * strides[i];
      }

      f(out_data, in_data, offset);
    });
  }
}

template <typename scalar_t>
void index_kernel_impl_maca(TensorIteratorBase& iter, IntArrayRef index_size, IntArrayRef index_stride, bool is_index_put=false) {
  gpu_index_kernel_maca(iter, index_size, index_stride, []C10_DEVICE(char* out_data, char* in_data, int64_t offset) {
    *(scalar_t*)out_data = *(scalar_t*)(in_data + offset);
  }, is_index_put);
}
