#pragma once
#include <ATen/native/cuda/maca_kernels/reduce_utils.cuh>

namespace at { namespace native {
template <int output_vec_size, typename arg_t, typename ops_t>
C10_DEVICE at::detail::Array<arg_t, output_vec_size> block_y_reduce(
    at::detail::Array<arg_t, output_vec_size> value, char* shared_memory,
    ReduceConfigCUDA config, ops_t ops) {
  using args_vec_t = at::detail::Array<arg_t, output_vec_size>;
  args_vec_t* shared = (args_vec_t*)shared_memory;
  shared[config.shared_memory_offset(0)] = value;
  for (int offset = blockDim.y / 2; offset > 0; offset >>= 1) {
    __syncthreads();
    if (threadIdx.y < offset && threadIdx.y + offset < blockDim.y) {
      args_vec_t other = shared[config.shared_memory_offset(offset)];
      #pragma unroll
      for (int i = 0; i < output_vec_size; i++) {
        value[i] = ops.combine(value[i], other[i]);
      }
      shared[config.shared_memory_offset(0)] = value;
    }
  }
  return value;
}

template <int output_vec_size, typename arg_t, typename ops_t>
C10_DEVICE at::detail::Array<arg_t, output_vec_size> block_x_reduce(
    at::detail::Array<arg_t, output_vec_size> value, char* shared_memory,
    ops_t ops) {
  using args_vec_t = at::detail::Array<arg_t, output_vec_size>;
  int dim_x = blockDim.x;
  args_vec_t* shared = (args_vec_t*)shared_memory;
  if (dim_x > warpSize) {
    int address_base = threadIdx.x + threadIdx.y*blockDim.x;
    shared[address_base] = value;
    for (int offset = dim_x/2; offset >= warpSize; offset >>= 1) {
      __syncthreads();
      if (threadIdx.x < offset && threadIdx.x + offset < blockDim.x) {
        args_vec_t other = shared[address_base + offset];
        #pragma unroll
        for (int i = 0; i < output_vec_size; i++) {
          value[i] = ops.combine(value[i], other[i]);
        }
        shared[address_base] = value;
      }
    }
    dim_x = warpSize;
  }

  __syncthreads();

  for (int offset = 1; offset < dim_x; offset <<= 1) {
    #pragma unroll
    for (int i = 0; i < output_vec_size; i++) {
      arg_t other = ops.warp_shfl_down(value[i], offset);
      value[i] = ops.combine(value[i], other);
    }
  }
  return value;
}

template<bool can_accumulate_in_output, int input_vec_size,
         typename scalar_t, typename index_t, typename out_scalar_t, int vt0, int nt, int output_vec_size,
         typename ops_t, typename arg_t, typename InputCalculator, typename OutputCalculator>
C10_LAUNCH_BOUNDS_2(nt, 4)
__global__ void reduce_kernel_maca(
    ops_t ops,
    arg_t ident,
    ReduceConfigCUDA config,
    InputCalculator input_calc,
    OutputCalculator output_calc,
    const void* src,
    const char* dst0, //it accepts at most two destinations
    const char* dst1, //it accepts at most two destinations
    // acc_buf used for accumulation among sub Tensor Iterator when accumulation on
    // output is not permissible
    void* acc_buf,
    // cta_buf used for accumulation between blocks during global reduction
    void* cta_buf,
    int* semaphores,
    int64_t base_idx,
    bool accumulate,
    bool final_output,
    int noutputs) {

  extern __shared__ char shared_memory[];
  index_t output_idx = config.output_idx<output_vec_size>();
  index_t input_idx = config.input_idx();
  auto base_offsets1 = output_calc.get(output_idx)[1];

  using arg_vec_t = at::detail::Array<arg_t, output_vec_size>;
  arg_vec_t value;

  if (output_idx < config.num_outputs && input_idx < config.num_inputs) {
    // ------ thread_reduce ------
    if (config.vectorize_input) {
      const scalar_t* input_slice = (const scalar_t*)((const char*)src + base_offsets1);
      assert(output_vec_size == 1);
      // reduce at the header of input_slice where memory is not aligned,
      // so that thread_reduce will have an aligned memory to work on.
      // ------- input_vectorized_thread_reduce_impl -------------
      index_t end = config.num_inputs;
      // Handle the head of input slice where data is not aligned
      arg_t value_local = ident;
      constexpr int align_bytes = alignof(at::native::memory::aligned_vector<scalar_t, input_vec_size>);
      constexpr int align_elements = align_bytes / sizeof(scalar_t);
      int shift = ((uint64_t)input_slice) % align_bytes / sizeof(scalar_t);
      if (shift > 0) {
        input_slice -= shift;
        end += shift;
        if(threadIdx.x >= shift && threadIdx.x < align_elements && config.should_reduce_tail()){
          value_local = ops.reduce(value_local, c10::load(input_slice + threadIdx.x), threadIdx.x - shift);
        }
        end -= align_elements;
        input_slice += align_elements;
        shift = align_elements - shift;
      }
      
      // Do the vectorized reduction
      using load_t = at::native::memory::aligned_vector<scalar_t, input_vec_size>;
      
      index_t idx = config.input_idx();
      const index_t stride = config.step_input;
      
      // Multiple accumulators to remove dependency between unrolled loops.
      arg_t value_list[input_vec_size];
      value_list[0] = value_local;
      
      #pragma unroll
      for (int i = 1; i < input_vec_size; i++) {
        value_list[i] = ident;
      }
      
      while (idx * input_vec_size + input_vec_size - 1 < end) {
        const auto values_vec = memory::load_vector<input_vec_size>(input_slice, idx);
        #pragma unroll
        for (index_t i = 0; i < input_vec_size; i++) {
          value_list[i] = ops.reduce(value_list[i], values_vec.val[i], shift + idx * input_vec_size + i);
        }
        idx += stride;
      }
      
      // tail
      index_t tail_start = end - end % input_vec_size;
      if (config.should_reduce_tail()) {
        int idx = tail_start + threadIdx.x;
        if (idx < end) {
          const auto value_local2 = c10::load(input_slice + idx);
          value_list[0] = ops.reduce(value_list[0], value_local2, idx + shift);
        }
      }
      
      // combine accumulators
      #pragma unroll
      for (int i = 1; i < input_vec_size; i++) {
        value_list[0] = ops.combine(value_list[0], value_list[i]);
      }
      value = {value_list[0]};
      // ------------ input_vectorized_thread_reduce_impl -------------
    } else {
      index_t element_stride = input_calc.strides_[0][0] / sizeof(scalar_t);
      bool is_contiguous = (input_calc.dims == 1 && element_stride == 1);
      const scalar_t* input_slice = (const scalar_t*)((const char*)src + base_offsets1);
      if (is_contiguous) {
          // ---------- thread_reduce_impl 1-----------
          // value = thread_reduce_impl<output_vec_size>(input_slice, [&](index_t idx) { return idx; });
          index_t idx = config.input_idx();
          const index_t end = config.num_inputs;
          const index_t stride = config.step_input;
        
          using arg_vec_t = at::detail::Array<arg_t, output_vec_size>;
          using load_t = at::native::memory::aligned_vector<scalar_t, output_vec_size>;
        
          // Multiple accumulators to remove dependency between unrolled loops.
          arg_vec_t value_list[vt0];
        
          #pragma unroll
          for (int i = 0; i < vt0; i++) {
            #pragma unroll
            for (int j = 0; j < output_vec_size; j++) {
              value_list[i][j] = ident;
            }
          }
        
          load_t values[vt0];
        
          while (idx + (vt0 - 1) * stride < end) {
            #pragma unroll
            for (index_t i = 0; i < vt0; i++) {
              const auto offset = (idx + i * stride) / output_vec_size;
              values[i] = memory::load_vector<output_vec_size>(input_slice, offset);
            }
            #pragma unroll
            for (index_t i = 0; i < vt0; i++) {
              #pragma unroll
              for (index_t j = 0; j < output_vec_size; j++) {
                value_list[i][j] = ops.reduce(value_list[i][j], values[i].val[j], idx + i * stride);
              }
            }
            idx += stride * vt0;
          }
        
          // tail
          int idx_ = idx;
          #pragma unroll
          for (index_t i = 0; i < vt0; i++) {
            if (idx >= end) {
              break;
            }
            const auto offset = (idx) / output_vec_size;
            values[i] = memory::load_vector<output_vec_size>(input_slice, offset);
            idx += stride;
          }
          idx = idx_;
          #pragma unroll
          for (index_t i = 0; i < vt0; i++) {
            if (idx >= end) {
              break;
            }
            #pragma unroll
            for (index_t j = 0; j < output_vec_size; j++) {
              value_list[i][j] = ops.reduce(value_list[i][j], values[i].val[j], idx);
            }
            idx += stride;
          }
        
          // combine accumulators
          #pragma unroll
          for (int i = 1; i < vt0; i++) {
            #pragma unroll
            for (index_t j = 0; j < output_vec_size; j++) {
              value_list[0][j] = ops.combine(value_list[0][j], value_list[i][j]);
            }
          }
          value = value_list[0];
          // ---------- thread_reduce_impl 1-----------
      } else if (input_calc.dims == 1) {
          // ---------- thread_reduce_impl 2-----------
          // value = thread_reduce_impl<output_vec_size>(input_slice, [&](index_t idx) { return idx * element_stride; });
          index_t idx = config.input_idx();
          const index_t end = config.num_inputs;
          const index_t stride = config.step_input;

          using arg_vec_t = at::detail::Array<arg_t, output_vec_size>;
          using load_t = at::native::memory::aligned_vector<scalar_t, output_vec_size>;

          // Multiple accumulators to remove dependency between unrolled loops.
          arg_vec_t value_list[vt0];

          #pragma unroll
          for (int i = 0; i < vt0; i++) {
            #pragma unroll
            for (int j = 0; j < output_vec_size; j++) {
              value_list[i][j] = ident;
            }
          }

          load_t values[vt0];

          while (idx + (vt0 - 1) * stride < end) {
            #pragma unroll
            for (index_t i = 0; i < vt0; i++) {
              const auto offset = (idx + i * stride) * element_stride / output_vec_size;
              values[i] = memory::load_vector<output_vec_size>(input_slice, offset);
            }
            #pragma unroll
            for (index_t i = 0; i < vt0; i++) {
              #pragma unroll
              for (index_t j = 0; j < output_vec_size; j++) {
                value_list[i][j] = ops.reduce(value_list[i][j], values[i].val[j], idx + i * stride);
              }
            }
            idx += stride * vt0;
          }

          // tail
          int idx_ = idx;
          #pragma unroll
          for (index_t i = 0; i < vt0; i++) {
            if (idx >= end) {
              break;
            }
            const auto offset = (idx) * element_stride / output_vec_size;
            values[i] = memory::load_vector<output_vec_size>(input_slice, offset);
            idx += stride;
          }
          idx = idx_;
          #pragma unroll
          for (index_t i = 0; i < vt0; i++) {
            if (idx >= end) {
              break;
            }
            #pragma unroll
            for (index_t j = 0; j < output_vec_size; j++) {
              value_list[i][j] = ops.reduce(value_list[i][j], values[i].val[j], idx);
            }
            idx += stride;
          }

          // combine accumulators
          #pragma unroll
          for (int i = 1; i < vt0; i++) {
            #pragma unroll
            for (index_t j = 0; j < output_vec_size; j++) {
              value_list[0][j] = ops.combine(value_list[0][j], value_list[i][j]);
            }
          }
          value = value_list[0];
          // ---------- thread_reduce_impl 2-----------
      } else {
          // ---------- thread_reduce_impl 3-----------
          // value = thread_reduce_impl<output_vec_size>(input_slice, [&](index_t idx) { return input_calc.get(idx)[0] / sizeof(scalar_t); });
          index_t idx = config.input_idx();
          const index_t end = config.num_inputs;
          const index_t stride = config.step_input;

          using arg_vec_t = at::detail::Array<arg_t, output_vec_size>;
          using load_t = at::native::memory::aligned_vector<scalar_t, output_vec_size>;

          // Multiple accumulators to remove dependency between unrolled loops.
          arg_vec_t value_list[vt0];

          #pragma unroll
          for (int i = 0; i < vt0; i++) {
            #pragma unroll
            for (int j = 0; j < output_vec_size; j++) {
              value_list[i][j] = ident;
            }
          }

          load_t values[vt0];

          while (idx + (vt0 - 1) * stride < end) {
            #pragma unroll
            for (index_t i = 0; i < vt0; i++) {
              const auto offset = input_calc.get(idx + i * stride)[0] / sizeof(scalar_t) / output_vec_size;
              values[i] = memory::load_vector<output_vec_size>(input_slice, offset);
            }
            #pragma unroll
            for (index_t i = 0; i < vt0; i++) {
              #pragma unroll
              for (index_t j = 0; j < output_vec_size; j++) {
                value_list[i][j] = ops.reduce(value_list[i][j], values[i].val[j], idx + i * stride);
              }
            }
            idx += stride * vt0;
          }

          // tail
          int idx_ = idx;
          #pragma unroll
          for (index_t i = 0; i < vt0; i++) {
            if (idx >= end) {
              break;
            }
            const auto offset = input_calc.get(idx)[0] / sizeof(scalar_t) / output_vec_size;
            values[i] = memory::load_vector<output_vec_size>(input_slice, offset);
            idx += stride;
          }
          idx = idx_;
          #pragma unroll
          for (index_t i = 0; i < vt0; i++) {
            if (idx >= end) {
              break;
            }
            #pragma unroll
            for (index_t j = 0; j < output_vec_size; j++) {
              value_list[i][j] = ops.reduce(value_list[i][j], values[i].val[j], idx);
            }
            idx += stride;
          }

          // combine accumulators
          #pragma unroll
          for (int i = 1; i < vt0; i++) {
            #pragma unroll
            for (index_t j = 0; j < output_vec_size; j++) {
              value_list[0][j] = ops.combine(value_list[0][j], value_list[i][j]);
            }
          }
          value = value_list[0];
          // ---------- thread_reduce_impl 3-----------
      }
    }
  }
  // ---------- thread_reduce -----------

  if (config.should_block_y_reduce()) {
    value = block_y_reduce<output_vec_size>(value, shared_memory, config, ops);
  }

  if (config.should_block_x_reduce()) {
    value = block_x_reduce<output_vec_size>(value, shared_memory, ops);
  }

  using out_ptr_vec_t = at::detail::Array<out_scalar_t*, output_vec_size>;
  using offset_vec_t = at::detail::Array<index_t, output_vec_size>;
  offset_vec_t base_offsets;
  out_ptr_vec_t out;

  #pragma unroll
  for (int i = 0; i < output_vec_size; i++) {
    base_offsets[i] = output_calc.get(output_idx + i)[0];
    out[i] = (out_scalar_t*)((char*)dst0 + base_offsets[i]);
  }

  arg_vec_t* acc = nullptr;
  if (acc_buf != nullptr) {
    size_t numerator = sizeof(arg_t);
    size_t denominator = sizeof(out_scalar_t);
    reduce_fraction(numerator, denominator);
    acc = (arg_vec_t*)((char*)acc_buf + (base_offsets[0] * numerator / denominator));
  }

  
  if (config.should_global_reduce()) {
    //value = global_reduce<output_vec_size>(value, acc, shared_memory);
  } else if (config.should_store(output_idx)) {
    if (accumulate) {
      #pragma unroll
      for (int i = 0; i < output_vec_size; i++) {
        value[i] = ops.translate_idx(value[i], base_idx);
      }
    }

    if (acc == nullptr) {
      if (accumulate) {
        value = accumulate_in_output<output_vec_size, can_accumulate_in_output, arg_t, out_scalar_t>(out, value, ops);
      }
      if (final_output) {
        #pragma unroll
        for (int i = 0; i < output_vec_size; i++) {
          set_results<out_scalar_t>(ops.project(value[i]), base_offsets[i], noutputs, dst0, dst1);
        }
      } else {
        #pragma unroll
        for (int i = 0; i < output_vec_size; i++) {
          *(out[i]) = get_accumulated_output<can_accumulate_in_output, out_scalar_t, arg_t>(out[i], value[i]);
        }
      }
    } else {
      if (accumulate) {
        #pragma unroll
        for (int i = 0; i < output_vec_size; i++) {
          value[i] = ops.combine((*acc)[i], value[i]);
        }
      }
      if (final_output) {
        #pragma unroll
        for (int i = 0; i < output_vec_size; i++) {
          set_results<out_scalar_t>(ops.project(value[i]), base_offsets[i], noutputs, dst0, dst1);
        }
      } else {
        *acc = value;
      }
    }
  }
}

template<typename arg_t, typename scalar_t, typename index_t, typename out_scalar_t, int vt0, int max_threads, typename R>
static void launch_reduce_kernel_maca_opt(const ReduceConfigCUDA& config, const R& reduction) {
  dim3 block = config.block();
  dim3 grid = config.grid();

  auto stream = at::cuda::getCurrentCUDAStream();
  int shared_memory = config.shared_memory_size();

  static constexpr bool can_accumulate_in_output =
    std::is_convertible<arg_t, out_scalar_t>::value
    && std::is_convertible<out_scalar_t, arg_t>::value;

  
  switch(config.output_vec_size) {
  case 4:
    reduce_kernel_maca<
        can_accumulate_in_output, 4,
        scalar_t, index_t, out_scalar_t, vt0, max_threads / 4, 4><<<grid, block, shared_memory, stream>>>(
      reduction.ops,
      reduction.ident,
      config,
      reduction.input_calc,
      reduction.output_calc,
      reduction.src,
      reduction.dst[0],
      reduction.dst[1],
      reduction.acc_buf,
      reduction.cta_buf,
      reduction.semaphores,
      reduction.base_idx,
      reduction.accumulate,
      reduction.final_output,
      reduction.noutputs);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    break;
  case 2:
    reduce_kernel_maca<
        can_accumulate_in_output, 4,
        scalar_t, index_t, out_scalar_t, vt0, max_threads / 2, 2><<<grid, block, shared_memory, stream>>>(
      reduction.ops,
      reduction.ident,
      config,
      reduction.input_calc,
      reduction.output_calc,
      reduction.src,
      reduction.dst[0],
      reduction.dst[1],
      reduction.acc_buf,
      reduction.cta_buf,
      reduction.semaphores,
      reduction.base_idx,
      reduction.accumulate,
      reduction.final_output,
      reduction.noutputs);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    break;
  default:
    reduce_kernel_maca<
        can_accumulate_in_output, 4,
        scalar_t, index_t, out_scalar_t, vt0, max_threads / 1, 1><<<grid, block, shared_memory, stream>>>(
      reduction.ops,
      reduction.ident,
      config,
      reduction.input_calc,
      reduction.output_calc,
      reduction.src,
      reduction.dst[0],
      reduction.dst[1],
      reduction.acc_buf,
      reduction.cta_buf,
      reduction.semaphores,
      reduction.base_idx,
      reduction.accumulate,
      reduction.final_output,
      reduction.noutputs);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
}

}} // namespace at::native
