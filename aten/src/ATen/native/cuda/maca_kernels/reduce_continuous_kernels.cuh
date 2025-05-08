#pragma once
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cuda/maca_kernels/reduce_utils.cuh>

namespace at {
namespace native {

static bool is_launch_continuous_reduce_kernel(const TensorIterator& iter, const size_t &input_element_size, const size_t &output_element_size) {
  // Currently contiunous reduce kernel support 4 cases
  // 1. input shape is 2d (dim0, dim1), reduce along dim1
  // 2. input shape is 3d (dim0, dim1, dim2) reduce along dim0 and dim2
  // 3. reduce along fastest moving stride dim, but other dims are reordered by as_strided etc
  // 4. output num is 1

  if (is_all_to_one_kernel(iter, input_element_size)) {
    return true;
  }
  if (is_ndim_3_reduce_dims_2_case(iter, input_element_size, output_element_size)) {
    return true;
  }
  if (maca_likely(!at::maca::get_maca_disable_continuous_reduce_kernel_use_offset_calculator()) && is_continuous_input_output_reorder_stride_case(iter, input_element_size, output_element_size)) {
    return true;
  }
  bool dims_cond = iter.num_reduce_dims() == 1 && iter.ndim() == 2;
  if (!dims_cond) {
    return false;
  }
  size_t input_index = iter.ntensors() - 1;
  size_t output_index = 0;
  bool reduce_dim_continuous_cond = iter.strides(input_index)[0] < iter.strides(input_index)[iter.num_reduce_dims()];
  bool input_element_continuous_cond = iter.strides(input_index)[0] == input_element_size &&
                                     iter.strides(input_index)[iter.num_reduce_dims()] == iter.shape()[0] * input_element_size;
  bool output_element_continuous_cond = iter.strides(output_index)[iter.num_reduce_dims()] == output_element_size;
    // align conditon
  int input_vec_size = get_vec_size(iter.shape()[0], input_element_size);
  int input_vec_byte_size = input_vec_size * input_element_size;
  int align_byte_size = input_vec_byte_size > sizeof(int32_t) ? sizeof(int32_t):input_vec_byte_size;
  bool align_cond = iter.strides(input_index)[iter.num_reduce_dims()] % input_vec_byte_size == 0;
  return reduce_dim_continuous_cond && input_element_continuous_cond && output_element_continuous_cond && align_cond;
}


template <
    typename scalar_t,
    typename out_scalar_t,
    typename arg_t,
    int input_vec_size,
    bool can_accumulate_in_output,
    bool use_offset_calculator,
    typename ops_t,
    typename InputCalculator,
    typename OutputCalculator>
__global__ typename std::enable_if<(std::is_same<arg_t, thrust::pair<scalar_t, int64_t>>::value || std::is_same<arg_t, thrust::pair<out_scalar_t, int64_t>>::value || std::is_same<arg_t, thrust::pair<float, int64_t>>::value) && !std::is_same<arg_t, thrust::pair<int64_t, int64_t>>::value, void>::type InputPerOutputContinuousReduceKernel(
    const scalar_t* input_data,
    ops_t ops,
    arg_t ident,
    int input_size,
    int output_size,
    InputCalculator input_calc,
    OutputCalculator output_calc,
    int noutputs,
    const bool should_block_y_reduce,
    const char* dst0,
    const char* dst1,
    void* cta_buf,
    int* semaphores,
    void* acc_buf,
    int64_t base_idx,
    bool accumulate,
    bool final_output) {
  using vec_t = aligned_vector<scalar_t, input_vec_size>;
  size_t step_x = gridDim.y * blockDim.x;
  size_t row_idx = blockIdx.x * blockDim.y + threadIdx.y;
  size_t col_idx = threadIdx.x + blockDim.x * blockIdx.y;
  if (should_block_y_reduce) {
    step_x = gridDim.y * blockDim.x * blockDim.y;
    row_idx = blockIdx.x;
    col_idx = threadIdx.x + threadIdx.y * blockDim.x + blockDim.x * blockDim.y * blockIdx.y;
  }

  if (row_idx >= output_size) {
    return;
  }

  auto base_offsets1 = row_idx * input_size;
  if (use_offset_calculator) {
    base_offsets1 = output_calc.get(row_idx)[1] / sizeof(scalar_t);
  }
  const scalar_t* cur_row_input = input_data + base_offsets1;
  const vec_t* input_vec = reinterpret_cast<const vec_t*>(cur_row_input);
  const int64_t n_vec_to_read = input_size / input_vec_size;
  typedef  decltype(ident.first) acc_t;
  acc_t val = ident.first;
  int64_t idx = ident.second;

  // single thread process
  for (int i = col_idx; i < n_vec_to_read; i += step_x) {
    vec_t data = load_vector<input_vec_size>(cur_row_input, i);
    int64_t index = i * input_vec_size;
#pragma unroll
    for (int j = 0; j < input_vec_size; j++) {
      ops.reduce_no_struct(val, idx, data.val[j], index + j, &val, &idx);
    }
  }

  // process remainer
  if (col_idx == 0) {
    for (int i = n_vec_to_read * input_vec_size; i < input_size; i++) {
      ops.reduce_no_struct(val, idx, load(cur_row_input, i), i, &val, &idx);
    }
  }

  __syncthreads();

  // block x reduce
  size_t dim_x = blockDim.x;
  if (dim_x > C10_WARP_SIZE) {
    extern __shared__ char shared_memory[];
    acc_t* shared_val = reinterpret_cast<acc_t*>(shared_memory);
    int64_t* shared_index = reinterpret_cast<int64_t*>(shared_memory + dim_x * blockDim.y * sizeof(acc_t));
    size_t shared_address = threadIdx.x + threadIdx.y * dim_x;
    shared_val[shared_address] = val;
    shared_index[shared_address] = idx;
    for (size_t offset = dim_x >> 1; offset >= C10_WARP_SIZE; offset >>= 1) {
       __syncthreads();
      if (threadIdx.x < offset) {
        size_t shared_read_address = threadIdx.x + offset + threadIdx.y * dim_x;
        acc_t other_val = shared_val[shared_read_address];
        int64_t other_index = shared_index[shared_read_address];
        ops.combine_no_struct(val, idx, other_val, other_index, &val, &idx);
        shared_val[shared_address] = val;
        shared_index[shared_address] = idx;
      }
    }
    dim_x = C10_WARP_SIZE;
  }
  __syncthreads();
  for (int offset = 1; offset < dim_x; offset <<= 1) {
    acc_t other_val;
    int64_t other_index;
    ops.warp_shfl_down_no_struct(val, idx, &other_val, &other_index, offset);
    ops.combine_no_struct(val, idx, other_val, other_index, &val, &idx);
  }
  __syncthreads();

  // block y reduce
  if (should_block_y_reduce) {
    extern __shared__ char shared_memory[];
    acc_t* shared_val = reinterpret_cast<acc_t*>(shared_memory);
    int64_t* shared_index = reinterpret_cast<int64_t*>(shared_memory + blockDim.y * sizeof(acc_t));

    if (threadIdx.x == 0) {
      shared_val[threadIdx.y] = val;
      shared_index[threadIdx.y] = idx;
      for (int offset = blockDim.y >> 1; offset >= 1 ; offset >>= 1) {
        __syncthreads();
        if (threadIdx.y < offset) {
          acc_t other_val = shared_val[threadIdx.y + offset];
          int64_t other_index = shared_index[threadIdx.y + offset];
          ops.combine_no_struct(val, idx, other_val, other_index, &val, &idx);
          shared_val[threadIdx.y] = val;
          shared_index[threadIdx.y] = idx;
        }
      }
    }
  }

  // inter block process
  bool is_global_reduce_output_thread = false;
  if (gridDim.y > 1) {
    acc_t *cta_value = reinterpret_cast<acc_t *>(cta_buf);
    int64_t *cta_idx = reinterpret_cast<int64_t *>(cta_value + gridDim.x * gridDim.y);
    // It is impossible that gridDim.y > 1 and block_y is used for different rows
    const bool is_first_thread_in_block = (threadIdx.x == 0 && threadIdx.y == 0);
    if (is_first_thread_in_block) {
      size_t block_idx = blockIdx.y + gridDim.y * blockIdx.x;
      cta_value[block_idx] = val;
      cta_idx[block_idx] = idx;
    }

    __threadfence();
    __syncthreads();
    bool is_last_block_done = mark_block_finished1(semaphores);
    if (is_last_block_done) {
      val = ident.first;
      idx = ident.second;
      size_t thread_id = threadIdx.x + threadIdx.y * blockDim.x;
      size_t reduce_buffer_offset = gridDim.y * blockIdx.x;
      if (thread_id < C10_WARP_SIZE) {
        for (size_t i = thread_id; i < gridDim.y; i+=C10_WARP_SIZE) {
          acc_t other_cta_value = cta_value[reduce_buffer_offset+i];
          int64_t other_cta_idx = cta_idx[reduce_buffer_offset+i];
          ops.combine_no_struct(val, idx, other_cta_value, other_cta_idx, &val, &idx);
        }

        for (int offset = 1; offset < C10_WARP_SIZE; offset <<= 1) {
          acc_t other_val;
          int64_t other_index;
          ops.warp_shfl_down_no_struct(val, idx, &other_val, &other_index, offset);
          ops.combine_no_struct(val, idx, other_val, other_index, &val, &idx);
        }
      }
      is_global_reduce_output_thread = is_first_thread_in_block;
    }
  }

  if ((gridDim.y == 1 && col_idx == 0) ||
      (gridDim.y > 1 && is_global_reduce_output_thread)) {
    arg_t thread_val = {val, idx};
    size_t base_offset = row_idx * sizeof(out_scalar_t);
    if (use_offset_calculator) {
      base_offset = output_calc.get(row_idx)[0];
    }
    arg_t* acc = nullptr;
    if (acc_buf != nullptr) {
      size_t numerator = sizeof(arg_t);
      size_t denominator = sizeof(out_scalar_t);
      reduce_fraction(numerator, denominator);
      acc = (arg_t*)((char*)acc_buf + (base_offset * numerator / denominator));
    }

    out_scalar_t *out = (out_scalar_t*)(dst0 + base_offset);
    if (accumulate) {
      thread_val = ops.translate_idx(thread_val, base_idx);
    }

    if (acc == nullptr) {
      if (accumulate) {
        thread_val = accumulate_in_output_once<can_accumulate_in_output>(out, thread_val, ops);
      }
      if (final_output) {
        set_results1<out_scalar_t>(
          ops.project(thread_val),
          base_offset,
          noutputs,
          dst0,
          dst1);
      } else {
        *(out) = get_accumulated_output<can_accumulate_in_output>(out, thread_val);
      }
    } else {
      if (accumulate) {
        ops.combine_no_struct(thread_val.first,
                              thread_val.second,
                              acc->first,
                              acc->second,
                              &thread_val.first,
                              &thread_val.second);
      }
      if (final_output) {
        set_results1<out_scalar_t>(
          ops.project(thread_val),
          base_offset,
          noutputs,
          dst0,
          dst1);
      } else {
        acc->first = thread_val.first;
        acc->second = thread_val.second;
      }
    }
  }
}

template <
    typename scalar_t,
    typename out_scalar_t,
    typename arg_t,
    int input_vec_size,
    bool can_accumulate_in_output,
    bool use_offset_calculator,
    typename ops_t,
    typename InputCalculator,
    typename OutputCalculator>
__global__ typename std::enable_if<(!std::is_same<arg_t, thrust::pair<scalar_t, int64_t>>::value && !std::is_same<arg_t, thrust::pair<out_scalar_t, int64_t>>::value && !std::is_same<arg_t, thrust::pair<float, int64_t>>::value ) || std::is_same<arg_t, thrust::pair<int64_t, int64_t>>::value, void>::type InputPerOutputContinuousReduceKernel(
    const scalar_t* input_data,
    ops_t ops,
    arg_t ident,
    int input_size,
    int output_size,
    InputCalculator input_calc,
    OutputCalculator output_calc,
    int noutputs,
    const bool should_block_y_reduce,
    const char* dst0,
    const char* dst1,
    void* cta_buf,
    int* semaphores,
    void* acc_buf,
    int64_t base_idx,
    bool accumulate,
    bool final_output) {
  using vec_t = aligned_vector<scalar_t, input_vec_size>;
  size_t step_x = gridDim.y * blockDim.x;
  size_t row_idx = blockIdx.x * blockDim.y + threadIdx.y;
  size_t col_idx = threadIdx.x + blockDim.x * blockIdx.y;
  if (should_block_y_reduce) {
    step_x = gridDim.y * blockDim.x * blockDim.y;
    row_idx = blockIdx.x;
    col_idx = threadIdx.x + threadIdx.y * blockDim.x + blockDim.x * blockDim.y * blockIdx.y;
  }
  if (row_idx >= output_size) {
    return;
  }

  auto base_offsets1 = row_idx * input_size;
  if (use_offset_calculator) {
    base_offsets1 = output_calc.get(row_idx)[1] / sizeof(scalar_t);
  }
  const scalar_t* cur_row_input = input_data + base_offsets1;
  const vec_t* input_vec = reinterpret_cast<const vec_t*>(cur_row_input);
  const int64_t n_vec_to_read = input_size / input_vec_size;
  arg_t thread_val = ident;
  // single thread process
  for (int i = col_idx; i < n_vec_to_read; i += step_x) {
    vec_t data = load_vector<input_vec_size>(cur_row_input, i);
    int64_t index = i * input_vec_size;
#pragma unroll
    for (int j = 0; j < input_vec_size; j++) {
      thread_val = ops.reduce(thread_val, data.val[j], index + j);
    }
  }

  // process remainer
  if (col_idx == 0) {
    for (int i = n_vec_to_read * input_vec_size; i < input_size; i++) {
      thread_val = ops.reduce(thread_val, load(cur_row_input, i), i);
    }
  }
  __syncthreads();

  // block x reduce
  size_t dim_x = blockDim.x;
  if (dim_x > C10_WARP_SIZE) {
    extern __shared__ char shared_memory[];
    arg_t* shared_val = reinterpret_cast<arg_t*>(shared_memory);
    size_t shared_address = threadIdx.x + threadIdx.y * dim_x;
    shared_val[shared_address] = thread_val;
    for (size_t offset = dim_x >> 1; offset >= C10_WARP_SIZE; offset >>= 1) {
       __syncthreads();
      if (threadIdx.x < offset) {
        size_t shared_read_address = threadIdx.x + offset + threadIdx.y * dim_x;
        arg_t other_val = shared_val[shared_read_address];
        thread_val = ops.combine(thread_val, other_val);
        shared_val[shared_address] = thread_val;
      }
    }
    dim_x = C10_WARP_SIZE;
  }
  __syncthreads();
  for (int offset = 1; offset < dim_x; offset <<= 1) {
    arg_t other_thread_val = ops.warp_shfl_down(thread_val, offset);
    thread_val = ops.combine(thread_val, other_thread_val);
  }
  __syncthreads();

  // block y reduce
  if (should_block_y_reduce) {
    extern __shared__ char shared_memory[];
    arg_t* shared_val = reinterpret_cast<arg_t*>(shared_memory);

    if (threadIdx.x == 0) {
      shared_val[threadIdx.y] = thread_val;
      for (int offset = blockDim.y >> 1; offset >= 1 ; offset >>= 1) {
        __syncthreads();
        if (threadIdx.y < offset) {
          arg_t other_val = shared_val[threadIdx.y + offset];
          thread_val = ops.combine(thread_val, other_val);
          shared_val[threadIdx.y] = thread_val;
        }
      }
    }
  }

  // inter block process
  bool is_global_reduce_output_thread = false;
  if (gridDim.y > 1) {
    arg_t *reduce_buffer = (arg_t *)cta_buf;
    // It is impossible that gridDim.y > 1 and block_y is used for different rows
    const bool is_first_thread_in_block = (threadIdx.x == 0 && threadIdx.y == 0);
    if (is_first_thread_in_block) {
      size_t block_idx = blockIdx.y + gridDim.y * blockIdx.x;
      reduce_buffer[block_idx] = thread_val;
    }

    __threadfence();
    __syncthreads();
    bool is_last_block_done = mark_block_finished1(semaphores);
    if (is_last_block_done) {
      thread_val = ident;
      size_t thread_id = threadIdx.x + threadIdx.y * blockDim.x;
      size_t reduce_buffer_offset = gridDim.y * blockIdx.x;
      if (thread_id < C10_WARP_SIZE) {
        for (size_t i = thread_id; i < gridDim.y; i+=C10_WARP_SIZE) {
          thread_val = ops.combine(thread_val, reduce_buffer[reduce_buffer_offset+i]);
        }

        for (int offset = 1; offset < C10_WARP_SIZE; offset <<= 1) {
          arg_t other = ops.warp_shfl_down(thread_val, offset);
          thread_val = ops.combine(thread_val, other);
        }
      }
      is_global_reduce_output_thread = is_first_thread_in_block;
    }
  }

  if ((gridDim.y == 1 && col_idx == 0) ||
      (gridDim.y > 1 && is_global_reduce_output_thread)) {
    size_t base_offset = row_idx * sizeof(out_scalar_t);
    if (use_offset_calculator) {
      base_offset = output_calc.get(row_idx)[0];
    }
    arg_t* acc = nullptr;
    if (acc_buf != nullptr) {
      size_t numerator = sizeof(arg_t);
      size_t denominator = sizeof(out_scalar_t);
      reduce_fraction(numerator, denominator);
      acc = (arg_t*)((char*)acc_buf + (base_offset * numerator / denominator));
    }

    out_scalar_t *out = (out_scalar_t*)(dst0 + base_offset);
    if (accumulate) {
      thread_val = ops.translate_idx(thread_val, base_idx);
    }

    if (acc == nullptr) {
      if (accumulate) {
        thread_val = accumulate_in_output_once<can_accumulate_in_output>(out, thread_val, ops);
      }
      if (final_output) {
        set_results1<out_scalar_t>(
          ops.project(thread_val),
          base_offset,
          noutputs,
          dst0,
          dst1);
      } else {
        *(out) = get_accumulated_output<can_accumulate_in_output>(out, thread_val);
      }
    } else {
      if (accumulate) {
        thread_val = ops.combine((*acc), thread_val);
      }
      if (final_output) {
        set_results1<out_scalar_t>(
          ops.project(thread_val),
          base_offset,
          noutputs,
          dst0,
          dst1);
      } else {
        *acc = thread_val;
      }
    }
  }
}

template <
    typename scalar_t,
    typename out_scalar_t,
    typename arg_t,
    int input_vec_size,
    bool can_accumulate_in_output,
    typename ops_t>
__global__ void InputPerOutputContinuousReduceKernelMultiReducDims(
    const scalar_t* input_data,
    ops_t ops,
    arg_t ident,
    int input_size,
    int output_size,
    int fastest_stride_len,
    int noutputs,
    const bool should_block_y_reduce,
    const char* dst0,
    const char* dst1,
    void* cta_buf,
    int* semaphores,
    void* acc_buf,
    int64_t base_idx,
    bool accumulate,
    bool final_output) {
  using vec_t = aligned_vector<scalar_t, input_vec_size>;
  const size_t step_x = should_block_y_reduce ? (gridDim.y * blockDim.x * blockDim.y) : (gridDim.y * blockDim.x);
  const size_t row_idx = should_block_y_reduce ? blockIdx.x : (blockIdx.x * blockDim.y + threadIdx.y);
  if (row_idx >= output_size) {
    return;
  }
  const size_t col_idx = should_block_y_reduce ? (threadIdx.x + threadIdx.y * blockDim.x + blockDim.x * blockDim.y * blockIdx.y) : (threadIdx.x + blockDim.x * blockIdx.y);
  const scalar_t* cur_row_input = input_data + fastest_stride_len * row_idx;
  const vec_t* input_vec = reinterpret_cast<const vec_t*>(input_data);
  const int64_t n_vec_to_read = input_size / input_vec_size;
  arg_t thread_val = ident;
  // single thread process
  size_t x_index = col_idx * input_vec_size;
  size_t x_div = x_index / fastest_stride_len;
  size_t x_mod = x_index % fastest_stride_len;
  size_t x_update = step_x * input_vec_size;
  for (int i = col_idx; i < n_vec_to_read; i += step_x) {
    size_t x_offset = x_div * fastest_stride_len * output_size + fastest_stride_len * row_idx + x_mod;
    x_index = i * input_vec_size;
    vec_t data = load_vector<input_vec_size>(input_data, x_offset / input_vec_size);
    #pragma unroll
    for (int j = 0; j < input_vec_size; j++) {
      thread_val = ops.reduce(thread_val, data.val[j], x_index + j);
    }
    if (x_mod + x_update < fastest_stride_len) {
      x_mod += x_update;
    } else {
      size_t mod_add = x_mod + x_update;
      x_mod = (mod_add % fastest_stride_len);
      x_div += (mod_add / fastest_stride_len);
    }
  }

  __syncthreads();

  // block x reduce
  size_t dim_x = blockDim.x;
  if (dim_x > C10_WARP_SIZE) {
    extern __shared__ char shared_memory[];
    arg_t* shared_val = reinterpret_cast<arg_t*>(shared_memory);
    size_t shared_address = threadIdx.x + threadIdx.y * dim_x;
    shared_val[shared_address] = thread_val;
    for (size_t offset = dim_x >> 1; offset >= C10_WARP_SIZE; offset >>= 1) {
       __syncthreads();
      if (threadIdx.x < offset) {
        size_t shared_address = threadIdx.x + offset + threadIdx.y * dim_x;
        arg_t other_val = shared_val[shared_address];
        thread_val = ops.combine(thread_val, other_val);
        shared_val[shared_address] = thread_val;
      }
    }
    dim_x = C10_WARP_SIZE;
  }

  for (int offset = 1; offset < dim_x; offset <<= 1) {
    arg_t other_thread_val = ops.warp_shfl_down(thread_val, offset);
    thread_val = ops.combine(thread_val, other_thread_val);
  }
  __syncthreads();

  // block y reduce
  if (should_block_y_reduce) {
    extern __shared__ char shared_memory[];
    arg_t* shared_val = reinterpret_cast<arg_t*>(shared_memory);

    if (threadIdx.x == 0) {
      shared_val[threadIdx.y] = thread_val;
      for (int offset = blockDim.y >> 1; offset >= 1 ; offset >>= 1) {
        __syncthreads();
        if (threadIdx.y < offset) {
          arg_t other_val = shared_val[threadIdx.y + offset];
          thread_val = ops.combine(thread_val, other_val);
          shared_val[threadIdx.y] = thread_val;
        }
      }
    }
  }

  // inter block process
  bool is_global_reduce_output_thread = false;
  if (gridDim.y > 1) {
    arg_t *reduce_buffer = (arg_t *)cta_buf;
    // It is impossible that gridDim.y > 1 and block_y is used for different rows
    const bool is_first_thread_in_block = (threadIdx.x == 0 && threadIdx.y == 0);
    if (is_first_thread_in_block) {
      size_t block_idx = blockIdx.y + gridDim.y * blockIdx.x;
      reduce_buffer[block_idx] = thread_val;
    }

    __threadfence();
    __syncthreads();
    bool is_last_block_done = mark_block_finished1(semaphores);
    if (is_last_block_done && is_first_thread_in_block) {
      thread_val = ident;
      size_t reduce_buffer_offset = gridDim.y * blockIdx.x;
      for (size_t i = 0; i < gridDim.y; i++) {
        thread_val = ops.combine(thread_val, reduce_buffer[reduce_buffer_offset+i]);
      }
      is_global_reduce_output_thread = true;
    }
  }

  if ((gridDim.y == 1 && col_idx == 0) ||
      (gridDim.y > 1 && is_global_reduce_output_thread)) {
    size_t base_offset = row_idx * sizeof(out_scalar_t);
    arg_t* acc = nullptr;
    if (acc_buf != nullptr) {
      size_t numerator = sizeof(arg_t);
      size_t denominator = sizeof(out_scalar_t);
      reduce_fraction(numerator, denominator);
      acc = (arg_t*)((char*)acc_buf + (base_offset * numerator / denominator));
    }

    out_scalar_t *out = (out_scalar_t*)(dst0 + base_offset);
    if (accumulate) {
      thread_val = ops.translate_idx(thread_val, base_idx);
    }

    if (acc == nullptr) {
      if (accumulate) {
        thread_val = accumulate_in_output_once<can_accumulate_in_output>(out, thread_val, ops);
      }
      if (final_output) {
        set_results1<out_scalar_t>(
          ops.project(thread_val),
          base_offset,
          noutputs,
          dst0,
          dst1);
      } else {
        *(out) = get_accumulated_output<can_accumulate_in_output>(out, thread_val);
      }
    } else {
      if (accumulate) {
        thread_val = ops.combine((*acc), thread_val);
      }
      if (final_output) {
        set_results1<out_scalar_t>(
          ops.project(thread_val),
          base_offset,
          noutputs,
          dst0,
          dst1);
      } else {
        *acc = thread_val;
      }
    }
  }
}

template <
    typename scalar_t,
    typename out_scalar_t,
    typename arg_t,
    int input_vec_size,
    bool can_accumulate_in_output,
    typename ops_t>
__global__ void InputPerOutputContinuousReduceKernelTranspose(
    const scalar_t* input_data,
    ops_t ops,
    arg_t ident,
    int input_size,
    int output_size,
    int output_dim0,
    int output_dim1,
    int noutputs,
    const char* dst0,
    const char* dst1,
    void* cta_buf,
    int* semaphores,
    void* acc_buf,
    int64_t base_idx,
    bool accumulate,
    bool final_output){
  using vec_t = aligned_vector<scalar_t, input_vec_size>;
  size_t row_idx = blockDim.y * blockIdx.y + threadIdx.y;
  size_t col_idx = blockDim.x * blockIdx.x + threadIdx.x;
  arg_t thread_val = ident;
  extern __shared__ char shared_memory[];
  arg_t* shared_val = reinterpret_cast<arg_t*>(shared_memory);

  // This kernel transpose matrix after ops.reduce
  int input_dim0 = output_dim1;
  int input_dim1 = output_dim0;

  size_t input_offset = row_idx * input_dim1+col_idx;

  if(row_idx < input_dim0 && col_idx < input_dim1){
    vec_t data = load_vector<input_vec_size>(input_data, input_offset, input_size);
    #pragma unroll
    for (int i = 0; i < input_vec_size; i++) {
      thread_val = ops.reduce(thread_val, data.val[i],i);
    }
    shared_val[threadIdx.y*(blockDim.x+1)+threadIdx.x] = thread_val;
  } else{
    shared_val[threadIdx.y*(blockDim.x+1)+threadIdx.x] = ident;
  }
  __syncthreads();

  size_t output_row_idx = blockDim.x * blockIdx.x + threadIdx.y;
  size_t output_col_idx = blockDim.y * blockIdx.y + threadIdx.x;

  if(output_row_idx < output_dim0 && output_col_idx < output_dim1){
    size_t output_offset = (output_row_idx * output_dim1 + output_col_idx) * sizeof(out_scalar_t);
    thread_val = shared_val[threadIdx.x*(blockDim.x+1)+threadIdx.y];
    set_results1<out_scalar_t>(
          ops.project(thread_val),
          output_offset,
          noutputs,
          dst0,
          dst1);
  }

  return;
}

template<typename scalar_t, typename out_scalar_t, typename arg_t, typename R>
static void launch_continuous_transpose_kernel(const ReduceConfigMaca& config, const R& reduction, const TensorIterator& tensor_iterator){
  dim3 block = config.block();
  dim3 grid = config.grid();
  int shared_memory = config.shared_memory_size();
  auto stream = at::cuda::getCurrentCUDAStream();
  int *semaphores_ptr = nullptr;
  void *global_buffer_ptr = nullptr;
  at::DataPtr buffer;
  at::DataPtr semaphores;

  static constexpr bool can_accumulate_in_output =
    std::is_convertible<arg_t, out_scalar_t>::value
    && std::is_convertible<out_scalar_t, arg_t>::value;

  // output_dim1 is the fastest dim in output
  int output_dim0, output_dim1;
  size_t output_index = 0;
  if(tensor_iterator.strides(output_index)[1] > tensor_iterator.strides(output_index)[2]){
    output_dim0 = tensor_iterator.shape()[1];
    output_dim1 = tensor_iterator.shape()[2];
  }
  else{
    output_dim0 = tensor_iterator.shape()[2];
    output_dim1 = tensor_iterator.shape()[1];
  }

  switch (config.input_vec_size)
  {
    case 4:
      InputPerOutputContinuousReduceKernelTranspose<scalar_t, out_scalar_t, arg_t, 4, can_accumulate_in_output><<<grid, block, shared_memory, stream>>>(
        (const scalar_t *)reduction.src,
        reduction.ops,
        reduction.ident,
        config.num_inputs,
        config.num_outputs,
        output_dim0,
        output_dim1,
        reduction.noutputs,
        reduction.dst[0],
        reduction.dst[1],
        global_buffer_ptr,
        semaphores_ptr,
        reduction.acc_buf,
        reduction.base_idx,
        reduction.accumulate,
        reduction.final_output
      );
      break;

    case 3:
      InputPerOutputContinuousReduceKernelTranspose<scalar_t, out_scalar_t, arg_t, 3, can_accumulate_in_output><<<grid, block, shared_memory, stream>>>(
        (const scalar_t *)reduction.src,
        reduction.ops,
        reduction.ident,
        config.num_inputs,
        config.num_outputs,
        output_dim0,
        output_dim1,
        reduction.noutputs,
        reduction.dst[0],
        reduction.dst[1],
        global_buffer_ptr,
        semaphores_ptr,
        reduction.acc_buf,
        reduction.base_idx,
        reduction.accumulate,
        reduction.final_output
      );
      break;

    case 2:
      InputPerOutputContinuousReduceKernelTranspose<scalar_t, out_scalar_t, arg_t, 2, can_accumulate_in_output><<<grid, block, shared_memory, stream>>>(
        (const scalar_t *)reduction.src,
        reduction.ops,
        reduction.ident,
        config.num_inputs,
        config.num_outputs,
        output_dim0,
        output_dim1,
        reduction.noutputs,
        reduction.dst[0],
        reduction.dst[1],
        global_buffer_ptr,
        semaphores_ptr,
        reduction.acc_buf,
        reduction.base_idx,
        reduction.accumulate,
        reduction.final_output
      );
      break;

    default:
      InputPerOutputContinuousReduceKernelTranspose<scalar_t, out_scalar_t, arg_t, 1, can_accumulate_in_output><<<grid, block, shared_memory, stream>>>(
        (const scalar_t *)reduction.src,
        reduction.ops,
        reduction.ident,
        config.num_inputs,
        config.num_outputs,
        output_dim0,
        output_dim1,
        reduction.noutputs,
        reduction.dst[0],
        reduction.dst[1],
        global_buffer_ptr,
        semaphores_ptr,
        reduction.acc_buf,
        reduction.base_idx,
        reduction.accumulate,
        reduction.final_output
      );
      break;
  }
}

template<typename scalar_t, typename out_scalar_t, int max_threads, typename arg_t, typename R>
static void launch_continuous_reduce_kernel_multi_reduce_dims(const ReduceConfigMaca& config, const R& reduction, const TensorIterator& tensor_iterator) {
  dim3 block = config.block();
  dim3 grid = config.grid();
  int shared_memory = config.shared_memory_size();
  auto stream = at::cuda::getCurrentCUDAStream();
  int *semaphores_ptr = nullptr;
  void *global_buffer_ptr = nullptr;
  at::DataPtr buffer;
  at::DataPtr semaphores;
  if (grid.y > 1) {
    auto& allocator = *c10::cuda::CUDACachingAllocator::get();
    buffer = allocator.allocate(config.global_memory_size());
    semaphores = allocator.allocate(config.semaphore_size());
    AT_CUDA_CHECK(cudaMemsetAsync(semaphores.get(), 0, config.semaphore_size(), stream));
    semaphores_ptr = (int *)semaphores.get();
    global_buffer_ptr = (void *)buffer.get();
  }
  static constexpr bool can_accumulate_in_output =
    std::is_convertible<arg_t, out_scalar_t>::value
    && std::is_convertible<out_scalar_t, arg_t>::value;
  switch (config.input_vec_size)
  {
  case 8:
    InputPerOutputContinuousReduceKernelMultiReducDims<scalar_t, out_scalar_t, arg_t, 8, can_accumulate_in_output><<<grid, block, shared_memory, stream>>>(
      (const scalar_t *)reduction.src,
      reduction.ops,
      reduction.ident,
      config.num_inputs,
      config.num_outputs,
      tensor_iterator.shape()[0],
      reduction.noutputs,
      config.should_block_y_reduce(),
      reduction.dst[0],
      reduction.dst[1],
      global_buffer_ptr,
      semaphores_ptr,
      reduction.acc_buf,
      reduction.base_idx,
      reduction.accumulate,
      reduction.final_output
    );
    break;

  case 4:
    InputPerOutputContinuousReduceKernelMultiReducDims<scalar_t, out_scalar_t, arg_t, 4, can_accumulate_in_output><<<grid, block, shared_memory, stream>>>(
      (const scalar_t *)reduction.src,
      reduction.ops,
      reduction.ident,
      config.num_inputs,
      config.num_outputs,
      tensor_iterator.shape()[0],
      reduction.noutputs,
      config.should_block_y_reduce(),
      reduction.dst[0],
      reduction.dst[1],
      global_buffer_ptr,
      semaphores_ptr,
      reduction.acc_buf,
      reduction.base_idx,
      reduction.accumulate,
      reduction.final_output
    );
    break;

  case 2:
    InputPerOutputContinuousReduceKernelMultiReducDims<scalar_t, out_scalar_t, arg_t, 2, can_accumulate_in_output><<<grid, block, shared_memory, stream>>>(
      (const scalar_t *)reduction.src,
      reduction.ops,
      reduction.ident,
      config.num_inputs,
      config.num_outputs,
      tensor_iterator.shape()[0],
      reduction.noutputs,
      config.should_block_y_reduce(),
      reduction.dst[0],
      reduction.dst[1],
      global_buffer_ptr,
      semaphores_ptr,
      reduction.acc_buf,
      reduction.base_idx,
      reduction.accumulate,
      reduction.final_output
    );
    break;

  default:
    InputPerOutputContinuousReduceKernelMultiReducDims<scalar_t, out_scalar_t, arg_t, 1, can_accumulate_in_output><<<grid, block, shared_memory, stream>>>(
      (const scalar_t *)reduction.src,
      reduction.ops,
      reduction.ident,
      config.num_inputs,
      config.num_outputs,
      tensor_iterator.shape()[0],
      reduction.noutputs,
      config.should_block_y_reduce(),
      reduction.dst[0],
      reduction.dst[1],
      global_buffer_ptr,
      semaphores_ptr,
      reduction.acc_buf,
      reduction.base_idx,
      reduction.accumulate,
      reduction.final_output
    );
    break;
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template<typename scalar_t, typename out_scalar_t, bool use_offset_calculator, typename arg_t, typename R>
static void launch_continuous_reduce_kernel_impl(const ReduceConfigMaca& config, const R& reduction, const TensorIterator& tensor_iterator) {
  dim3 block = config.block();
  dim3 grid = config.grid();
  int shared_memory = config.shared_memory_size();
  auto stream = at::cuda::getCurrentCUDAStream();
  int *semaphores_ptr = nullptr;
  void *global_buffer_ptr = nullptr;
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
  static constexpr bool can_accumulate_in_output =
    std::is_convertible<arg_t, out_scalar_t>::value
    && std::is_convertible<out_scalar_t, arg_t>::value;
  switch (config.input_vec_size)
  {
  case 8:
    InputPerOutputContinuousReduceKernel<scalar_t, out_scalar_t, arg_t, 8, can_accumulate_in_output, use_offset_calculator><<<grid, block, shared_memory, stream>>>(
      (const scalar_t *)reduction.src,
      reduction.ops,
      reduction.ident,
      config.num_inputs,
      config.num_outputs,
      reduce::make_input_calculator<uint32_t>(tensor_iterator),
      reduce::make_output_calculator<uint32_t>(tensor_iterator),
      reduction.noutputs,
      config.should_block_y_reduce(),
      reduction.dst[0],
      reduction.dst[1],
      global_buffer_ptr,
      semaphores_ptr,
      reduction.acc_buf,
      reduction.base_idx,
      reduction.accumulate,
      reduction.final_output
    );
    break;

  case 4:
    InputPerOutputContinuousReduceKernel<scalar_t, out_scalar_t, arg_t, 4, can_accumulate_in_output, use_offset_calculator><<<grid, block, shared_memory, stream>>>(
      (const scalar_t *)reduction.src,
      reduction.ops,
      reduction.ident,
      config.num_inputs,
      config.num_outputs,
      reduce::make_input_calculator<uint32_t>(tensor_iterator),
      reduce::make_output_calculator<uint32_t>(tensor_iterator),
      reduction.noutputs,
      config.should_block_y_reduce(),
      reduction.dst[0],
      reduction.dst[1],
      global_buffer_ptr,
      semaphores_ptr,
      reduction.acc_buf,
      reduction.base_idx,
      reduction.accumulate,
      reduction.final_output
    );
    break;

  case 3:
    InputPerOutputContinuousReduceKernel<scalar_t, out_scalar_t, arg_t, 3, can_accumulate_in_output, use_offset_calculator><<<grid, block, shared_memory, stream>>>(
      (const scalar_t *)reduction.src,
      reduction.ops,
      reduction.ident,
      config.num_inputs,
      config.num_outputs,
      reduce::make_input_calculator<uint32_t>(tensor_iterator),
      reduce::make_output_calculator<uint32_t>(tensor_iterator),
      reduction.noutputs,
      config.should_block_y_reduce(),
      reduction.dst[0],
      reduction.dst[1],
      global_buffer_ptr,
      semaphores_ptr,
      reduction.acc_buf,
      reduction.base_idx,
      reduction.accumulate,
      reduction.final_output
    );
    break;

  case 2:
    InputPerOutputContinuousReduceKernel<scalar_t, out_scalar_t, arg_t, 2, can_accumulate_in_output, use_offset_calculator><<<grid, block, shared_memory, stream>>>(
      (const scalar_t *)reduction.src,
      reduction.ops,
      reduction.ident,
      config.num_inputs,
      config.num_outputs,
      reduce::make_input_calculator<uint32_t>(tensor_iterator),
      reduce::make_output_calculator<uint32_t>(tensor_iterator),
      reduction.noutputs,
      config.should_block_y_reduce(),
      reduction.dst[0],
      reduction.dst[1],
      global_buffer_ptr,
      semaphores_ptr,
      reduction.acc_buf,
      reduction.base_idx,
      reduction.accumulate,
      reduction.final_output
    );
    break;

  default:
    InputPerOutputContinuousReduceKernel<scalar_t, out_scalar_t, arg_t, 1, can_accumulate_in_output, use_offset_calculator><<<grid, block, shared_memory, stream>>>(
      (const scalar_t *)reduction.src,
      reduction.ops,
      reduction.ident,
      config.num_inputs,
      config.num_outputs,
      reduce::make_input_calculator<uint32_t>(tensor_iterator),
      reduce::make_output_calculator<uint32_t>(tensor_iterator),
      reduction.noutputs,
      config.should_block_y_reduce(),
      reduction.dst[0],
      reduction.dst[1],
      global_buffer_ptr,
      semaphores_ptr,
      reduction.acc_buf,
      reduction.base_idx,
      reduction.accumulate,
      reduction.final_output
    );
    break;
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}


template<typename scalar_t, typename out_scalar_t, int max_threads, typename arg_t, typename R>
static void launch_continuous_reduce_kernel(const ReduceConfigMaca& config, const R& reduction, const TensorIterator& tensor_iterator) {
  bool use_offset_calculator = is_continuous_input_output_reorder_stride_case(tensor_iterator, sizeof(scalar_t), sizeof(out_scalar_t));
  if (use_offset_calculator) {
    launch_continuous_reduce_kernel_impl<scalar_t, out_scalar_t, true, arg_t>(config, reduction, tensor_iterator);
  } else {
    launch_continuous_reduce_kernel_impl<scalar_t, out_scalar_t, false, arg_t>(config, reduction, tensor_iterator);
  }

}
} // namespace native
} // namespace at
