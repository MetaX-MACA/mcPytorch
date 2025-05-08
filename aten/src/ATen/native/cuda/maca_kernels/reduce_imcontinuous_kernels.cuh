#pragma once
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cuda/maca_kernels/reduce_utils.cuh>

namespace at { namespace native {
static bool is_launch_imcontinuous_reduce_kernel(const TensorIterator& iter, const size_t &input_element_size, const size_t &output_element_size) {
  // Currently imcontiunous reduce kernel support 3 cases
  // 1. input shape is 2d (dim0, dim1), reduce along dim0
  // 2. input shape is 3d (dim0, dim1, dim2) and not reduce along dim0, dim1 or both dim0 and dim1
  // 3. input shape is 2d (dim0, dim1), reduce along dim0 and stride(dim0) == n * shape(dim1), n >= 2
  // 4. input shape is 2d(1, dim1), reduce along dim1 or 2d(dim0, 1), reduce along dim0
  // The fastest moving dim (dim1 in case 1 or dim2 in case 2) 
  // should be continuous and fit the align demand for 
  // ldg command. The reduce dim should be imcontinuous

  // case 2 judge
  if (is_ndim_3_reduce_not_continuous(iter, input_element_size, output_element_size)) {
    return true;
  }

  // case 3 judge
  if (is_ndim_2_dim0_stride_not_continuous(iter, input_element_size, output_element_size)) {
    return true;
  }

  // case 4 judge
  // In this case, there is actually no need to perform a reduce operation,
  // so it is processed by default in the imcontinuous kernel.
  if (is_ndim_1_reduce_not_continuous(iter, input_element_size, output_element_size)) {
    return true;
  }

  // case1 judge
  bool dims_cond = iter.num_reduce_dims() == 1 && iter.ndim() == 2;
  if (!dims_cond) {
    return false;
  }
  size_t input_index = iter.ntensors() - 1;
  size_t output_index = 0;
  bool reduce_dim_imcontinuous_cond = iter.strides(input_index)[iter.num_reduce_dims() - 1] >= iter.strides(input_index)[iter.num_reduce_dims()];
  bool input_element_continuous_cond = iter.strides(input_index)[iter.num_reduce_dims()] == input_element_size && 
                                      iter.strides(input_index)[0] == iter.shape()[iter.num_reduce_dims()] * iter.strides(input_index)[iter.num_reduce_dims()];
  bool output_element_continuous_cond = iter.strides(output_index)[iter.num_reduce_dims()] == output_element_size;
  size_t output_vec_size = get_vec_size(iter.shape()[1], input_element_size);
  bool align_cond = iter.strides(input_index)[iter.num_reduce_dims() - 1] % sizeof(int32_t) == 0 && iter.shape()[iter.num_reduce_dims()] % output_vec_size == 0;
  return reduce_dim_imcontinuous_cond && input_element_continuous_cond && output_element_continuous_cond && align_cond;
}

using InputCalculator = reduce::OffsetCalculator<1, uint32_t>;
using OutputCalculator = reduce::OffsetCalculator<2, uint32_t>;

template <
    typename scalar_t,
    typename out_scalar_t,
    typename arg_t,
    int vt0,
    int output_vec_size,
    bool enable_offset,
    typename ops_t>
__global__ typename std::enable_if<std::is_same<arg_t, thrust::pair<scalar_t, int64_t>>::value&& \
                                  !std::is_same<arg_t, thrust::pair<int64_t, int64_t>>::value, void>::type \
                                  InputPerOutputImcontinuousReduceKernel(
    const scalar_t* input_data,
    ops_t ops,
    arg_t ident,
    int input_size,
    int output_size,
    const bool enable_base_map,
    const bool enable_step_map,
    InputCalculator input_calc,
    OutputCalculator output_calc,
    int noutputs,
    const bool should_block_y_reduce,
    void* cta_buf,
    int* semaphores,
    const char* dst0,
    const char* dst1) {
  using vec_t = aligned_vector<scalar_t, output_vec_size>;
  size_t input_start = threadIdx.y + blockDim.y * blockIdx.y;
  size_t input_step = blockDim.y * gridDim.y;
  size_t output_start = threadIdx.x + blockDim.x * blockIdx.x;
  if (!should_block_y_reduce) {
    input_start = blockIdx.y;
    input_step = gridDim.y;
    output_start = threadIdx.x + threadIdx.y * blockDim.x + blockIdx.x * blockDim.x * blockDim.y;
  }
  const size_t output_idx = output_start * output_vec_size;
  if (output_idx >= output_size) {
    return;
  }

  auto input_offset = output_size / output_vec_size;
  auto base_offset0 = output_idx * sizeof(out_scalar_t);
  auto base_offset1 = output_idx * sizeof(scalar_t);
  if (enable_base_map) {
    base_offset0 = output_calc.get(output_idx)[0];
    base_offset1 = output_calc.get(output_idx)[1];
  }
  if (enable_step_map) {
    input_offset = input_calc.get(1)[0] / sizeof(scalar_t) / output_vec_size;
  }

  // single thread reduce
  using vec_index_t = aligned_vector<int64_t, output_vec_size>;
  typedef  decltype(ident.first) acc_t;
  using vec_acc_t = aligned_vector<acc_t, output_vec_size>;

  vec_acc_t thread_vals[vt0];
  vec_index_t thread_index[vt0];
  #pragma unroll
  for (size_t i = 0; i < vt0; i++) {
    #pragma unroll
    for (size_t j = 0; j < output_vec_size; j++) {
      thread_vals[i].val[j] = ident.first;
      thread_index[i].val[j] = ident.second;
    }
  }

  size_t input_idx = input_start;
  vec_t input_vals[vt0];
  const scalar_t* input_slice = (const scalar_t*)((const char*)input_data + base_offset1);
  while (input_idx + (vt0 - 1) * input_step < input_size) {
    #pragma unroll
    for (size_t i = 0; i < vt0; i++) {
      const scalar_t* input_load_addr;
      size_t input_load_offset;
      if (enable_offset) {
        size_t remapped_input_idx = input_calc.get(input_idx + input_step * i)[0];
        input_load_addr = (const scalar_t*)((const char *)input_slice + remapped_input_idx);
        input_load_offset = 0;
      } else {
        input_load_addr = input_slice;
        input_load_offset = input_offset * (input_idx + input_step * i);
      }
      input_vals[i] = load_vector<output_vec_size>(input_load_addr, input_load_offset);
    }
    #pragma unroll
    for (size_t i = 0; i < vt0; i++) {
      #pragma unroll
      for (size_t j = 0; j < output_vec_size; j++) {
        ops.reduce_no_struct(thread_vals[i].val[j],
                             thread_index[i].val[j],
                             input_vals[i].val[j],
                             input_idx + input_step * i,
                             &(thread_vals[i].val[j]),
                             &(thread_index[i].val[j]));
      }
    }
    input_idx += (vt0 * input_step);
  }
  // process tail
  size_t idx_ = input_idx;
  #pragma unroll
  for (size_t i = 0; i < vt0; i++) {
    if (input_idx >= input_size) {
      break;
    }
    const scalar_t* input_load_addr;
    size_t input_load_offset;
    if (enable_offset) {
      size_t remapped_input_idx = input_calc.get(input_idx)[0];
      input_load_addr = (const scalar_t*)((const char *)input_slice + remapped_input_idx);
      input_load_offset = 0;
    } else {
      input_load_addr = input_slice;
      input_load_offset = input_offset * input_idx;
    }
    input_vals[i] = load_vector<output_vec_size>(input_load_addr,  input_load_offset);;
    input_idx += input_step;
  }
  input_idx = idx_;
  #pragma unroll
  for (size_t i = 0; i < vt0; i++) {
    if (input_idx >= input_size) {
      break;
    }
    #pragma unroll
    for (size_t j = 0; j < output_vec_size; j++) {
      ops.reduce_no_struct(thread_vals[i].val[j],
                           thread_index[i].val[j],
                           input_vals[i].val[j],
                           input_idx,
                           &(thread_vals[i].val[j]),
                           &(thread_index[i].val[j]));
    }
    input_idx += input_step;
  }  

  // accumulate along vt0
  vec_acc_t cur_thread_vals = thread_vals[0];
  vec_index_t cur_thread_index = thread_index[0];
  #pragma unroll
  for (size_t i = 1; i < vt0; i++) {
    #pragma unroll
    for (size_t j = 0; j < output_vec_size; j++) {
      ops.combine_no_struct(cur_thread_vals.val[j],
                            cur_thread_index.val[j],
                            thread_vals[i].val[j],
                            thread_index[i].val[j],
                            &(cur_thread_vals.val[j]),
                            &(cur_thread_index.val[j]));
    }
  }
  if (!should_block_y_reduce) {
    #pragma unroll
    for (size_t j = 0; j < output_vec_size; j++) {
      arg_t result = {cur_thread_vals.val[j], cur_thread_index.val[j]};
      set_results1<out_scalar_t>(
          ops.project(result),
          base_offset0 + j * sizeof(out_scalar_t),
          noutputs,
          dst0,
          dst1);
    }
    return;
  }
  __syncthreads();

  // inter warp reduce
  extern __shared__ char shared_memory[];
  vec_acc_t* shared_val = reinterpret_cast<vec_acc_t*>(shared_memory);
  vec_index_t* shared_index =
      reinterpret_cast<vec_index_t*>(shared_memory + blockDim.y * blockDim.x * output_vec_size * sizeof(scalar_t) / 2);
  const size_t block_size = blockDim.x * blockDim.y;
  size_t wave_size = C10_WARP_SIZE;
  if (block_size < C10_WARP_SIZE) wave_size = block_size;
  if (blockDim.x > wave_size) wave_size = blockDim.x;
  const size_t warp_y_lane = wave_size / blockDim.x;
  const size_t warp_id = threadIdx.y / warp_y_lane;

  for (size_t offset  = (blockDim.y >> 1); offset >= warp_y_lane; offset >>= 1) {
    if (threadIdx.y >= offset && threadIdx.y < 2 * offset) {
      size_t shared_idx = (threadIdx.y - offset) * blockDim.x + threadIdx.x;
      shared_val[shared_idx] = cur_thread_vals;
      shared_index[shared_idx] = cur_thread_index;
    }
    __syncthreads();
    if (threadIdx.y < offset) {
      size_t shared_read_idx = (threadIdx.y) * blockDim.x + threadIdx.x;
      vec_acc_t other_thread_val = shared_val[shared_read_idx];
      vec_index_t other_thread_index = shared_index[shared_read_idx];
      #pragma unroll
      for (size_t j = 0; j < output_vec_size; j++) {
        ops.combine_no_struct(cur_thread_vals.val[j],
                              cur_thread_index.val[j],
                              other_thread_val.val[j],
                              other_thread_index.val[j],
                              &(cur_thread_vals.val[j]),
                              &(cur_thread_index.val[j]));
      }
    }
    __syncthreads();
  }

  //intra warp reduce
  if (warp_id == 0) {
    for (size_t offset = (wave_size >> 1); offset >= blockDim.x; offset >>= 1) {
      #pragma unroll
      for (size_t j = 0; j < output_vec_size; j++) {
        acc_t other_thread_val;
        int64_t other_thread_idx;
        ops.warp_shfl_down_no_struct(cur_thread_vals.val[j],
                                     cur_thread_index.val[j],
                                     &other_thread_val,
                                     &other_thread_idx,
                                     offset);
        ops.combine_no_struct(cur_thread_vals.val[j],
                              cur_thread_index.val[j],
                              other_thread_val,
                              other_thread_idx,
                              &(cur_thread_vals.val[j]),
                              &(cur_thread_index.val[j]));
      }
    }
  }

  vec_acc_t *reduce_buffer = (vec_acc_t *)cta_buf;
  const size_t global_index_offset = gridDim.x * gridDim.y * blockDim.x * output_vec_size * sizeof(scalar_t);
  vec_index_t *reduce_index_buffer = (vec_index_t *)((char *)cta_buf + global_index_offset);
  if (threadIdx.y == 0) {
    if (gridDim.y == 1) {
      #pragma unroll
      for (size_t j = 0; j < output_vec_size; j++) {
        arg_t result = {cur_thread_vals.val[j], cur_thread_index.val[j]};
        set_results1<out_scalar_t>(
            ops.project(result),
            base_offset0 + j * sizeof(out_scalar_t),
            noutputs,
            dst0,
            dst1);
      }
      return;
    }
    // every blocks save block reduce result to cta_buf
    size_t global_index = blockIdx.x * gridDim.y * blockDim.x  + blockIdx.y *  blockDim.x + threadIdx.x;
    reduce_buffer[global_index] = cur_thread_vals;
    reduce_index_buffer[global_index] = cur_thread_index;
  }

  if (gridDim.y == 1) {
    return;
  }
  __threadfence(); // make sure writes are globally visible
  __syncthreads(); // if multiple warps in this block wrote to staging, make sure they're all done

  // inter grid reduce
  bool is_last_block_done = mark_block_finished1(semaphores);   
  if (is_last_block_done) {
    // read and reduce from other blocks
    vec_acc_t grid_value;
    vec_index_t grid_index;
    #pragma unroll
    for (size_t i = 0; i < output_vec_size; i++) {
      grid_value.val[i] = ident.first;
      grid_index.val[i] = ident.second;
    }
    size_t grid_x_offset = blockIdx.x * gridDim.y * blockDim.x  + threadIdx.x;
    for (size_t i = threadIdx.y; i < gridDim.y; i+=blockDim.y) {
      size_t global_read_index = grid_x_offset + i * blockDim.x; 
      vec_acc_t other_grid_value = reduce_buffer[global_read_index];
      vec_index_t other_grid_index = reduce_index_buffer[global_read_index];
      #pragma unroll
      for (size_t j = 0; j < output_vec_size; j++) {
        ops.combine_no_struct(grid_value.val[j],
                              grid_index.val[j],
                              other_grid_value.val[j],
                              other_grid_index.val[j],
                              &(grid_value.val[j]),
                              &(grid_index.val[j]));
      }        
    }

    //inter warp reduce
    for (size_t offset  = (blockDim.y >> 1); offset >= warp_y_lane; offset >>= 1) {
      if (threadIdx.y >= offset && threadIdx.y < 2 * offset) {
        size_t shared_idx = (threadIdx.y - offset) * blockDim.x + threadIdx.x;
        shared_val[shared_idx] = grid_value;
        shared_index[shared_idx] = grid_index;
      }
      __syncthreads();
      if (threadIdx.y < offset) {
        size_t shared_read_idx = (threadIdx.y) * blockDim.x + threadIdx.x;
        vec_acc_t other_thread_val = shared_val[shared_read_idx];
        vec_index_t other_thread_index = shared_index[shared_read_idx];
        #pragma unroll
        for (size_t j = 0; j < output_vec_size; j++) {
          ops.combine_no_struct(grid_value.val[j],
                                grid_index.val[j],
                                other_thread_val.val[j],
                                other_thread_index.val[j],
                                &(grid_value.val[j]),
                                &(grid_index.val[j]));
        }
      }
      __syncthreads();
    }

    //intra warp reduce
    if (warp_id == 0) {
      for (size_t offset = (C10_WARP_SIZE >> 1); offset >= blockDim.x; offset >>= 1) {
        #pragma unroll
        for (size_t j = 0; j < output_vec_size; j++) {
          acc_t other_thread_val;
          int64_t other_thread_idx;
          ops.warp_shfl_down_no_struct(grid_value.val[j],
                                       grid_index.val[j],
                                       &other_thread_val,
                                       &other_thread_idx,
                                       offset);
          ops.combine_no_struct(grid_value.val[j],
                                grid_index.val[j],
                                other_thread_val,
                                other_thread_idx,
                                &(grid_value.val[j]),
                                &(grid_index.val[j]));
        }
      }
    }

    if (threadIdx.y == 0) {
      #pragma unroll
      for (size_t j = 0; j < output_vec_size; j++) {
        arg_t result = {grid_value.val[j], grid_index.val[j]};
        set_results1<out_scalar_t>(
            ops.project(result),
            base_offset0 + j * sizeof(out_scalar_t),
            noutputs,
            dst0,
            dst1);
      }
      return;
    }
  }   
}


template <
    typename scalar_t,
    typename out_scalar_t,
    typename arg_t,
    int vt0,
    int output_vec_size,
    bool enable_offset,
    typename ops_t>
__global__ typename std::enable_if<!std::is_same<arg_t, thrust::pair<scalar_t, int64_t>>::value || \
                                  std::is_same<arg_t, thrust::pair<int64_t, int64_t>>::value, void>::type \
                                  InputPerOutputImcontinuousReduceKernel(
    const scalar_t* input_data,
    ops_t ops,
    arg_t ident,
    int input_size,
    int output_size,
    const bool enable_base_map,
    const bool enable_step_map,
    InputCalculator input_calc,
    OutputCalculator output_calc,
    int noutputs,
    const bool should_block_y_reduce,
    void* cta_buf,
    int* semaphores,
    const char* dst0,
    const char* dst1) {
  using vec_t = aligned_vector<scalar_t, output_vec_size>;
  size_t input_start = threadIdx.y + blockDim.y * blockIdx.y;
  size_t input_step = blockDim.y * gridDim.y;
  size_t output_start = threadIdx.x + blockDim.x * blockIdx.x;
  if (!should_block_y_reduce) {
    input_start = blockIdx.y;
    input_step = gridDim.y;
    output_start = threadIdx.x + threadIdx.y * blockDim.x + blockIdx.x * blockDim.x * blockDim.y;
  }

  const size_t output_idx = output_start * output_vec_size;
  if (output_idx >= output_size) {
    return;
  }

  auto input_offset = output_size / output_vec_size;;
  auto base_offset0 = output_idx * sizeof(out_scalar_t);
  auto base_offset1 = output_idx * sizeof(scalar_t);
  if (enable_base_map) {
    base_offset0 = output_calc.get(output_idx)[0];
    base_offset1 = output_calc.get(output_idx)[1];
  }
  if (enable_step_map) {
    input_offset = input_calc.get(1)[0] / sizeof(scalar_t) / output_vec_size;
  }

  // single thread reduce
  using vec_arg_t = aligned_vector<arg_t, output_vec_size>;
  vec_arg_t thread_val;
  #pragma unroll
  for (size_t j = 0; j < output_vec_size; j++) {
    thread_val.val[j] = ident;
  }

  size_t input_idx = input_start;
  const scalar_t* input_slice = (const scalar_t*)((const char*)input_data + base_offset1);
  for (size_t i = input_start; i < input_size; i+=input_step) {
    const scalar_t* input_load_addr;
    size_t input_load_offset;
    if (enable_offset) {
      size_t remapped_input_idx = input_calc.get(i)[0];
      input_load_addr = (const scalar_t*)((const char *)input_slice + remapped_input_idx);
      input_load_offset = 0;
    } else {
      input_load_addr = input_slice;
      input_load_offset = i * input_offset;
    }
    vec_t data = load_vector<output_vec_size>(input_load_addr, input_load_offset);

    #pragma unroll
    for (size_t j = 0; j < output_vec_size; j++) {
      thread_val.val[j] = ops.reduce(thread_val.val[j], data.val[j], i);
    }
  }

  if (!should_block_y_reduce) {
    #pragma unroll
    for (size_t j = 0; j < output_vec_size; j++) {
      set_results1<out_scalar_t>(
          ops.project(thread_val.val[j]),
          base_offset0 + j * sizeof(out_scalar_t),
          noutputs,
          dst0,
          dst1);
    }
    return;
  }
  __syncthreads();
  // inter warp reduce
  extern __shared__ char shared_memory[];
  vec_arg_t* shared_val = reinterpret_cast<vec_arg_t*>(shared_memory);
  const size_t block_size = blockDim.x * blockDim.y;
  size_t wave_size = C10_WARP_SIZE;
  if (block_size < C10_WARP_SIZE) wave_size = block_size;
  if (blockDim.x > wave_size) wave_size = blockDim.x;
  const size_t warp_y_lane = wave_size / blockDim.x;
  const size_t warp_id = threadIdx.y / warp_y_lane;

  for (size_t offset = (blockDim.y >> 1); offset >= warp_y_lane; offset >>= 1) {
    if (threadIdx.y >= offset && threadIdx.y < 2 * offset) {
      size_t shared_idx = (threadIdx.y - offset) * blockDim.x + threadIdx.x;
      shared_val[shared_idx] = thread_val;
    }
    __syncthreads();
    if (threadIdx.y < offset) {
      size_t shared_read_idx = (threadIdx.y) * blockDim.x + threadIdx.x;
      vec_arg_t other_thread_val = shared_val[shared_read_idx];
      #pragma unroll
      for (size_t j = 0; j < output_vec_size; j++) {
        thread_val.val[j] = ops.combine(thread_val.val[j], other_thread_val.val[j]);
      }
    }
    __syncthreads();
  }

  //intra warp reduce
  if (warp_id == 0) {
    for (size_t offset = (wave_size >> 1); offset >= blockDim.x; offset >>= 1) {
      #pragma unroll
      for (size_t j = 0; j < output_vec_size; j++) {
        arg_t data = thread_val.val[j];
        arg_t other_thread_val = ops.warp_shfl_down(data, offset);
        thread_val.val[j] = ops.combine(thread_val.val[j], other_thread_val);
      }
    }
  }
  __syncthreads();

  vec_arg_t *reduce_buffer = (vec_arg_t *)cta_buf;
  if (threadIdx.y == 0) {
    if (gridDim.y == 1) {
      #pragma unroll
      for (size_t j = 0; j < output_vec_size; j++) {
        set_results1<out_scalar_t>(
            ops.project(thread_val.val[j]),
            base_offset0 + j * sizeof(out_scalar_t),
            noutputs,
            dst0,
            dst1);
      }
      return;      
    }

    // every blocks save block reduce result to cta_buf
    size_t global_index = blockIdx.x * gridDim.y * blockDim.x  + blockIdx.y *  blockDim.x + threadIdx.x;
    reduce_buffer[global_index] = thread_val;
  }

  if (gridDim.y == 1) {
    return;
  }
  __threadfence(); // make sure writes are globally visible
  __syncthreads(); // if multiple warps in this block wrote to staging, make sure they're all done

  // inter grid reduce
  bool is_last_block_done = mark_block_finished1(semaphores);   
  if (is_last_block_done) {
    // read and reduce from other blocks
    vec_arg_t grid_value;
    #pragma unroll
    for (size_t i = 0; i < output_vec_size; i++) {
      grid_value.val[i] = ident;
    }
    size_t grid_x_offset = blockIdx.x * gridDim.y * blockDim.x  + threadIdx.x;
    for (size_t i = threadIdx.y; i < gridDim.y; i += blockDim.y) {
      size_t global_read_index = grid_x_offset + i * blockDim.x; 
      vec_arg_t other_grid_value = reduce_buffer[global_read_index];
      #pragma unroll
      for (size_t j = 0; j < output_vec_size; j++) {
        grid_value.val[j] = ops.combine(grid_value.val[j], other_grid_value.val[j]);
      }        
    }

    for (size_t offset  = (blockDim.y >> 1); offset >= warp_y_lane; offset >>= 1) {
      if (threadIdx.y >= offset && threadIdx.y < 2 * offset) {
        size_t shared_idx = (threadIdx.y - offset) * blockDim.x + threadIdx.x;
        shared_val[shared_idx] = grid_value;
      }
      __syncthreads();
      if (threadIdx.y < offset) {
        size_t shared_read_idx = (threadIdx.y) * blockDim.x + threadIdx.x;
        vec_arg_t other_thread_val = shared_val[shared_read_idx];
        #pragma unroll
        for (size_t j = 0; j < output_vec_size; j++) {
          grid_value.val[j] = ops.combine(grid_value.val[j], other_thread_val.val[j]);
        }
      }
      __syncthreads();
    }

    if (warp_id == 0) {
      for (size_t offset = (C10_WARP_SIZE >> 1); offset >= blockDim.x; offset >>= 1) {
        #pragma unroll
        for (size_t j = 0; j < output_vec_size; j++) {
          arg_t data = grid_value.val[j];
          arg_t other_thread_val = ops.warp_shfl_down(data, offset);
          grid_value.val[j] = ops.combine(grid_value.val[j], other_thread_val);
        }
      }
    }

    if (threadIdx.y == 0) {
      #pragma unroll
      for (size_t j = 0; j < output_vec_size; j++) {
        set_results1<out_scalar_t>(
            ops.project(grid_value.val[j]),
            base_offset0 + j * sizeof(out_scalar_t),
            noutputs,
            dst0,
            dst1);
      }
      return;
    }
  } 
}

template<typename scalar_t, typename out_scalar_t, int max_threads, int vt0, bool enable_input_idx_map, typename arg_t, typename R>
static void launch_imcontinuous_reduce_kernel_impl(const ReduceConfigMaca& config, const R& reduction, const TensorIterator& tensor_iterator) {
  dim3 block = config.block();
  dim3 grid = config.grid();
  int shared_memory = config.shared_memory_size();
  auto stream = at::cuda::getCurrentCUDAStream();
  int *semaphores_ptr = nullptr;
  void *global_buffer_ptr = nullptr;
  bool enable_base_map = tensor_iterator.ndim() > 2 && (tensor_iterator.ndim() - tensor_iterator.num_reduce_dims()) > 1;
  bool enable_step_map = is_ndim_2_dim0_stride_not_continuous(tensor_iterator, sizeof(scalar_t), sizeof(out_scalar_t)) || 
                        (tensor_iterator.num_reduce_dims() == 1 && tensor_iterator.ndim() > 2);
  at::DataPtr buffer;
  at::DataPtr semaphores;
  if (grid.y > 1) {
    auto& allocator = *c10::cuda::CUDACachingAllocator::get();
    buffer = allocator.allocate(config.global_memory_size());
    semaphores = allocator.allocate(config.semaphore_size());
    auto reduce_memcpy_src_ptr = c10::cuda::CUDACachingAllocator::getReduceAsyncMemcpySrc();
    // The semaphore_size threshold is set to c10::cuda::getReduceAsyncMemcpyByteSize().
    if (config.semaphore_size() <= c10::cuda::CUDACachingAllocator::getReduceAsyncMemcpyByteSize() &&
        reduce_memcpy_src_ptr != nullptr &&
        maca_likely(at::maca::get_maca_enable_memcpy_replace_memset_reduce_kernel())) {
      AT_CUDA_CHECK(cudaMemcpyAsync(semaphores.get(),
                    reduce_memcpy_src_ptr,
                    config.semaphore_size(),
                    cudaMemcpyHostToDevice,
                    stream));
    } else {
      AT_CUDA_CHECK(cudaMemsetAsync(semaphores.get(), 0, config.semaphore_size(), stream));
    }

    semaphores_ptr = (int *)semaphores.get();
    global_buffer_ptr = (void *)buffer.get();
  }  
  switch (config.output_vec_size)
  {
  case 8:
    InputPerOutputImcontinuousReduceKernel<scalar_t, out_scalar_t, arg_t, vt0, 8, enable_input_idx_map><<<grid, block, shared_memory, stream>>>(
      (const scalar_t *)reduction.src,
      reduction.ops,
      reduction.ident,
      config.num_inputs,
      config.num_outputs,
      enable_base_map,
      enable_step_map,
      reduce::make_input_calculator<uint32_t>(tensor_iterator),
      reduce::make_output_calculator<uint32_t>(tensor_iterator),
      reduction.noutputs,
      config.should_block_y_reduce(),
      global_buffer_ptr,
      semaphores_ptr, 
      reduction.dst[0],
      reduction.dst[1]
    );
    break;
  
  case 4:
    InputPerOutputImcontinuousReduceKernel<scalar_t, out_scalar_t, arg_t, vt0, 4, enable_input_idx_map><<<grid, block, shared_memory, stream>>>(
      (const scalar_t *)reduction.src,
      reduction.ops,
      reduction.ident,
      config.num_inputs,
      config.num_outputs,
      enable_base_map,
      enable_step_map,
      reduce::make_input_calculator<uint32_t>(tensor_iterator),
      reduce::make_output_calculator<uint32_t>(tensor_iterator),
      reduction.noutputs,
      config.should_block_y_reduce(),
      global_buffer_ptr,
      semaphores_ptr, 
      reduction.dst[0],
      reduction.dst[1]
    );
    break;
  
  case 2:
    InputPerOutputImcontinuousReduceKernel<scalar_t, out_scalar_t, arg_t, vt0, 2, enable_input_idx_map><<<grid, block, shared_memory, stream>>>(
      (const scalar_t *)reduction.src,
      reduction.ops,
      reduction.ident,
      config.num_inputs,
      config.num_outputs,
      enable_base_map,
      enable_step_map,
      reduce::make_input_calculator<uint32_t>(tensor_iterator),
      reduce::make_output_calculator<uint32_t>(tensor_iterator),
      reduction.noutputs,
      config.should_block_y_reduce(),
      global_buffer_ptr,
      semaphores_ptr, 
      reduction.dst[0],
      reduction.dst[1]
    );  
    break;
  
  default:
    InputPerOutputImcontinuousReduceKernel<scalar_t, out_scalar_t, arg_t, vt0, 1, enable_input_idx_map><<<grid, block, shared_memory, stream>>>(
      (const scalar_t *)reduction.src,
      reduction.ops,
      reduction.ident,
      config.num_inputs,
      config.num_outputs,
      enable_base_map,
      enable_step_map,
      reduce::make_input_calculator<uint32_t>(tensor_iterator),
      reduce::make_output_calculator<uint32_t>(tensor_iterator),
      reduction.noutputs,
      config.should_block_y_reduce(),
      global_buffer_ptr,
      semaphores_ptr, 
      reduction.dst[0],
      reduction.dst[1]
    ); 
    break;
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template<typename scalar_t, typename out_scalar_t, int max_threads, int vt0, typename arg_t, typename R>
static void launch_imcontinuous_reduce_kernel(const ReduceConfigMaca& config, const R& reduction, const TensorIterator& tensor_iterator) {
  bool enable_input_idx_map = tensor_iterator.num_reduce_dims() > 1;
  if (enable_input_idx_map) {
    launch_imcontinuous_reduce_kernel_impl<scalar_t, out_scalar_t, max_threads, vt0, true, arg_t>(config, reduction, tensor_iterator);
  } else {
    launch_imcontinuous_reduce_kernel_impl<scalar_t, out_scalar_t, max_threads, vt0, false, arg_t>(config, reduction, tensor_iterator);
  }
}
}} // namespace at::native
