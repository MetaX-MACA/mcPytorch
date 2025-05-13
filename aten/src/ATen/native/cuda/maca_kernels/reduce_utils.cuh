#pragma once
#include <ATen/cuda/detail/OffsetCalculator.cuh>
#include <ATen/native/TensorIterator.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/DeviceUtils.cuh>
#include <thrust/pair.h>

namespace at { namespace native {
template<typename T>
struct mnt_wrapper;

namespace reduce{
static inline int64_t div_up(int64_t a, int64_t b) {
  return (a + b - 1) / b;
}

// returns floor(log2(n))
static inline int last_pow2(int n) {
  n |= (n >>  1);
  n |= (n >>  2);
  n |= (n >>  4);
  n |= (n >>  8);
  n |= (n >> 16);
  return std::max(1, n - (n >> 1));
}
}
C10_HOST_DEVICE static void reduce_fraction(size_t &numerator, size_t &denominator);

template <int output_vec_size, bool can_acc, typename arg_t, typename out_scalar_t, typename ops_t>
C10_DEVICE at::detail::Array<arg_t, output_vec_size> accumulate_in_output(
  at::detail::Array<out_scalar_t*, output_vec_size> out,
  at::detail::Array<arg_t, output_vec_size> value,
  ops_t ops,
  typename std::enable_if<can_acc>::type* = nullptr
) {
  at::detail::Array<arg_t, output_vec_size> ret;
  #pragma unroll
  for (int i = 0; i < output_vec_size; i++) {
    ret[i] = ops.combine(*(out[i]), value[i]);
  }
  return ret;
}

template <bool can_acc, typename out_scalar_t, typename arg_t>
C10_DEVICE out_scalar_t get_accumulated_output(
  out_scalar_t* out, arg_t value,
  typename std::enable_if<can_acc>::type* = nullptr
) {
  // assert(!final_output);
  return (out_scalar_t)value;
}

// This function should never be called --
// it's the version of `accumulate_in_output`
// when accumulation in the output is not possible.
template <int output_vec_size, bool can_acc, typename arg_t, typename out_scalar_t, typename ops_t>
C10_DEVICE at::detail::Array<arg_t, output_vec_size> accumulate_in_output(
  at::detail::Array<out_scalar_t*, output_vec_size>,
  at::detail::Array<arg_t, output_vec_size>,
  ops_t,
  typename std::enable_if<!can_acc>::type* = nullptr
) {
  assert(false); // can't use AT_ASSERT in Cuda.
  return arg_t {};
}

// This function should never be called --
// it's the version of `get_accumulated_output`
// when accumulation in the output is not possible.
template <bool can_acc, typename out_scalar_t, typename arg_t>
C10_DEVICE out_scalar_t get_accumulated_output(
  out_scalar_t* out, arg_t value,
  typename std::enable_if<!can_acc>::type* = nullptr
) {
  assert(false);
  return *out;
}

template<class out_scalar_t, class T, class index_t>
C10_DEVICE void set_results(const T x, const index_t base_offset, int noutputs, const char* dst0, const char* dst1) {
  assert(noutputs == 1);
  auto res = (out_scalar_t*)((char*)dst0 + base_offset);
  *res = x;
}

//Currently implemented for max of two outputs
template<class out_scalar_t, class T1, class T2, class index_t>
C10_DEVICE void set_results(const thrust::pair<T1, T2> x, const index_t base_offset, int noutputs, const char* dst0, const char* dst1) {
  if (noutputs >= 1) {
    auto res0 = (T1*)((char*)dst0 + base_offset);
    *res0 = x.first;
  }
  if (noutputs >= 2) {
    // base offset is computed assuming element size being sizeof(T1), so we need to make a
    // correction to obtain the correct base offset
    auto res1 = (T2*) ((char *) dst1 + base_offset / sizeof(T1) * sizeof(T2));
    *res1 = x.second;
  }
}

static int get_output_vec_size(const size_t &continuous_dim_len, const size_t &element_size) {
  constexpr int ldg128_bytes = 16;
  int vec_size = ldg128_bytes / element_size;
  int min_vec_size = 4;
  vec_size = std::max(std::min(vec_size, min_vec_size), 1);
  auto update_vec_size = [&vec_size](uint64_t n) {
    while(n % vec_size != 0) {
      vec_size /= 2;
    }
  };
  update_vec_size(continuous_dim_len);
  return vec_size;
}

static int get_vec_size(const size_t &continuous_dim_len, const size_t &element_size) {
  constexpr int ldg128_bytes = 16;
  int vec_size = ldg128_bytes / element_size;
  vec_size = std::max(std::min(vec_size, 8), 1);
  auto update_vec_size = [&vec_size](uint64_t n) {
    while(n % vec_size != 0) {
      vec_size /= 2;
    }
  };
  update_vec_size(continuous_dim_len);
  return vec_size;
}

template <typename scalar_t>
int get_output_vec_size_cuda(const TensorIterator &iter) {
  int vec_size = 4;
  auto update_vec_size = [&vec_size](uint64_t n) {
    while(n % vec_size != 0) {
      vec_size /= 2;
    }
  };

  uint64_t base_address = reinterpret_cast<uint64_t>(iter.data_ptr(iter.noutputs())) / sizeof(scalar_t);
  update_vec_size(base_address);

  const int output_index = iter.num_reduce_dims();
  update_vec_size(iter.shape()[output_index]);

  int j = 0;
  for(auto i : iter.strides(iter.noutputs())) {
    if (j != output_index) {
      update_vec_size(i / sizeof(scalar_t));
    }
    j++;
  }
  return vec_size;
}

struct ReduceConfigCUDA {
  static constexpr int BLOCK_X = 0;
  static constexpr int BLOCK_Y = 1;
  static constexpr int CTA = 2;

  static constexpr int input_vec_size = 4;

  ReduceConfigCUDA(int element_size_bytes, int num_outputs, int num_inputs)
    : element_size_bytes(element_size_bytes)
    , num_inputs(num_inputs)
    , num_outputs(num_outputs) {}
  int element_size_bytes;
  int num_inputs;
  int num_outputs;
  int step_input = 1;
  int step_output = 1;
  int ctas_per_output = 1;
  int input_mult[3] = {0, 0, 0};
  int output_mult[2] = {0, 0};

  int block_width;
  int block_height;
  int num_threads;

  bool vectorize_input = false;
  int output_vec_size = 1;

  template <typename T>
  void set_block_dimension(int64_t dim0, int64_t dim1) {
    const int max_num_threads = mnt_wrapper<T>::MAX_NUM_THREADS / output_vec_size;
    int dim0_pow2 = dim0 < max_num_threads ? static_cast<int>(reduce::last_pow2(dim0)) : max_num_threads;
    int dim1_pow2 = dim1 < max_num_threads ? static_cast<int>(reduce::last_pow2(dim1)) : max_num_threads;
    block_width = std::min(dim0_pow2, int(at::cuda::warp_size()));
    block_height = std::min(dim1_pow2, int(max_num_threads / block_width));
    block_width = std::min(dim0_pow2, int(max_num_threads / block_height));
    num_threads = block_width * block_height;
  }

  int split_input(int parallelism) {
    int step = step_input;
    step_input *= parallelism;
    return step;
  }

  int split_output(int parallelism) {
    int step = step_output;
    step_output *= parallelism;
    return step;
  }

  dim3 block() const {
    return dim3(block_width, block_height);
  }

  dim3 grid() const {
    return dim3(reduce::div_up(num_outputs / output_vec_size, step_output), ctas_per_output);
  }

  C10_HOST_DEVICE bool should_block_x_reduce() const {
    return input_mult[BLOCK_X] != 0;
  }

  C10_HOST_DEVICE bool should_block_y_reduce() const {
    return input_mult[BLOCK_Y] != 0;
  }

  C10_HOST_DEVICE bool should_global_reduce() const {
    return input_mult[CTA] != 0;
  }

  C10_DEVICE bool should_store(int output_idx) const {
    return output_idx < num_outputs &&
      (!should_block_x_reduce() || threadIdx.x == 0) &&
      (!should_block_y_reduce() || threadIdx.y == 0);
  }

  C10_DEVICE bool should_reduce_tail() const {
    return (!should_block_y_reduce() || threadIdx.y == 0) &&
      (!should_global_reduce() || blockIdx.y == 0);
  }

  C10_HOST_DEVICE int input_idx() const {
    int lane = threadIdx.x;
    int warp = threadIdx.y;
    int cta2 = blockIdx.y;
    return (lane * input_mult[BLOCK_X] +
            warp * input_mult[BLOCK_Y] +
            cta2 * input_mult[CTA]);
  }

  template <int output_vec_size>
  C10_HOST_DEVICE int output_idx() const {
    int lane = threadIdx.x;
    int warp = threadIdx.y;
    int cta1 = blockIdx.x;
    return (lane * output_mult[BLOCK_X] +
            warp * output_mult[BLOCK_Y] +
            cta1 * step_output) * output_vec_size;
  }

  C10_DEVICE int shared_memory_offset(int offset) const {
    return threadIdx.x + (threadIdx.y + offset) * blockDim.x;
  }

  C10_DEVICE int staging_memory_offset(int cta2) const {
    int offset = cta2 + blockIdx.x * gridDim.y;
    if (!should_block_x_reduce()) {
      offset = threadIdx.x + offset * blockDim.x;
    }
    return offset;
  }

  int shared_memory_size() const {
    if (!should_block_y_reduce() &&
        (!should_block_x_reduce() ||
         block_width <= at::cuda::warp_size())) {
      return 0;
    }
    return element_size_bytes * num_threads * output_vec_size;
  }

  int64_t global_memory_size() const {
    if (!should_global_reduce()) {
      return 0;
    }
    auto size = (int64_t)element_size_bytes * num_outputs * ctas_per_output;
    if (!should_block_x_reduce()) {
      size *= block().x * output_vec_size;
    }
    return size;
  }

  int semaphore_size() const {
    if (!should_global_reduce()) {
      return 0;
    }
    return sizeof(int) * grid().x;
  }

  int values_per_thread() const {
    return reduce::div_up(num_inputs, step_input);
  }
};

template<typename arg_t, typename scalar_t, int vt0>
ReduceConfigCUDA setReduceConfigCUDA(const TensorIterator& iter){
  // Start by assuming that each thread handles a single output and all
  // the inputs for that output.
  int64_t num_outputs = iter.num_output_elements();
  int64_t inputs_per_output = iter.numel() / num_outputs;
  int input_index = iter.ntensors() - 1;

  auto config = ReduceConfigCUDA(sizeof(arg_t), num_outputs, inputs_per_output);

  int64_t dim0;
  int64_t dim1;
  int64_t fastest_moving_stride;
  bool reduction_on_fastest_striding_dimension;

  if (iter.ndim() > 0) {
    // Adjust block size to map block width to fastest changing dimension of input
    // tensor. This grants the best possible memory accessing pattern, given that
    // for non-contiguous tensor with space in between, we cannot have perfect
    // memory coalescing.
    reduction_on_fastest_striding_dimension =
        (iter.num_reduce_dims() == iter.ndim()) ||
        (iter.strides(/*arg=*/input_index)[0] <
        iter.strides(/*arg=*/input_index)[iter.num_reduce_dims()]);
    // Notice that dim0 & dim1 does NOT guarantee any launch configuration here!
    // dim0 & dim1 are more like the upper bound of the block dimension. The
    // actual launch config and reduction scheme is determined by setting values
    // to `config.input_mult` and `config.output_mult`.
    // We try to max out dim1 so that we have enough threads per CTA to deliver
    // performance for larger problem size.
    if (reduction_on_fastest_striding_dimension) {
      // Map block.x to the fastest reducing dimension. It implies:
      //   1. block_x_reduce is required.
      //   2. block.y now max out to num_outputs.
      dim0 = inputs_per_output;
      dim1 = num_outputs;
      fastest_moving_stride = iter.strides(/*arg=*/input_index)[0];
    } else {
      // Map block.x to the fastest non reducing dimension. It implies:
      //   1. block_x_reduce is turned off.
      //   2. block.y now max out to inputs_per_output.
      dim0 = num_outputs;
      dim1 = inputs_per_output;
      fastest_moving_stride = iter.strides(/*arg=*/input_index)[iter.num_reduce_dims()];
    }
  } else {
    reduction_on_fastest_striding_dimension = true;
    fastest_moving_stride = sizeof(scalar_t);
    dim0 = 1;
    dim1 = 1;
  }

  // We do vectorization to gain better memory access, there are two cases which we call
  // "vectorize along input" and "vectorize along output". Note that the "input/output"
  // here does not mean we are vectorizing load/store instructions. We always only vectorize
  // load instructions.
  //
  // Case 1: "vectorize along input"
  // This case happens when we are reducing along fastest moving dimesion. In such case, threads
  // with the same threadIdx.y works on the same reduction cooperatively and will produce results
  // for the same ouput. In such case, values in each loaded vector always correspond to the same ouput.
  //
  // Case 2: "vectorize along output"
  // This case happens when the fastest moving dimesion is not the dimension of reduction. In such case,
  // threads with different threadIdx.x are independent and will produce results for different outputs.
  // In such case, values in each loaded vector always correspond to different outputs.
  if (fastest_moving_stride == sizeof(scalar_t)) {
    if (reduction_on_fastest_striding_dimension && dim0 > 128 && iter.num_reduce_dims() == 1 && vt0 >= ReduceConfigCUDA::input_vec_size) {
      // Case 1: "vectorize along input"
      // Note that if vt0 < ReduceConfig::vec_size, then this means the register pressure could be high, in such case,
      // we should avoid vectorization.
      config.vectorize_input = true;
      dim0 /= config.input_vec_size;
    } else if (!reduction_on_fastest_striding_dimension) {
      // Case 2: "vectorize along output"
      config.output_vec_size = get_output_vec_size_cuda<scalar_t>(iter);
      dim0 /= config.output_vec_size;
    }
  }

  // Adjust block_width and block_height
  config.set_block_dimension<scalar_t>(dim0, dim1);

  int block_width = config.block_width;
  int block_height = config.block_height;

  if (iter.ndim() == 0 || reduction_on_fastest_striding_dimension) {
    // Split the input across lanes if the input is contiguous in the reduced
    // dimension. This will require reduction between threads using warp
    // shuffle instructions and shared memory (if block_width > warpSize).
    config.input_mult[0] = config.split_input(block_width);
  } else {
    // Otherwise split the output across lanes in a warp.
    config.output_mult[0] = config.split_output(block_width);
  }

  constexpr int min_values_per_thread = 16;
  constexpr int max_values_per_thread = 256;

  if (config.values_per_thread() >= block_height * 16 || config.values_per_thread() >= max_values_per_thread) {
    // Divide the input across warps in a thread-block, if that leaves at least
    // 16 elements to be summed by each thread. This will require inter-warp
    // reduction using shared memory.
    config.input_mult[1] = config.split_input(block_height);
  } else {
    // Otherwise, each warp handles a separate output.
    config.output_mult[1] = config.split_output(block_height);
  }

  const int blocks_per_sm = at::cuda::getCurrentDeviceProperties()->maxThreadsPerMultiProcessor / config.num_threads;
  const int num_mp = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  const int target_grid_size = num_mp * blocks_per_sm;
  int grid = config.grid().x;
  if (config.input_mult[1] != 0 && config.values_per_thread() >= max_values_per_thread && grid <= target_grid_size) {
    // Divide the input across thread-blocks if the amount of work per-thread
    // is large enough and the size of the output is small enough. This will
    // require a reduction using global memory.
    // If we decide to split input across blocks, as long as we can get enough
    // number of blocks (`target_grid_size`) to balance SM, we should still
    // make the number of values per thread large for best performance.
    int ctas_per_output1 = reduce::div_up(target_grid_size, grid);
    int ctas_per_output2 = reduce::div_up(config.values_per_thread(), min_values_per_thread);
    int ctas_per_output3 = reduce::div_up(config.values_per_thread(), max_values_per_thread);
    // We want the minimum of ctas_per_output1 and ctas_per_output2, so that each thread can have
    // a large number of values to deal with. But we don't want values_per_thread to be larger than
    // max_values_per_thread
    config.ctas_per_output = std::max(std::min<int>(ctas_per_output1, ctas_per_output2), ctas_per_output3);
    if (config.ctas_per_output > 1) {
      config.input_mult[2] = config.split_input(config.ctas_per_output);
    }
  }
  return config;
};

enum ReduceKernelType {
  ReduceImcontinuousKernel = 0,
  ReduceContinuousKernel = 1,
  ReduceContinuousMultiReduceDimsKernel = 2,
  ReduceNoStructKernel = 3,
  ReduceOriginKernel = 4
};

struct ReduceConfigMaca {
  static constexpr int BLOCK_X = 0;
  static constexpr int BLOCK_Y = 1;
  static constexpr int CTA = 2;

  ReduceConfigMaca(int element_size_bytes, int num_outputs, int num_inputs)
    : element_size_bytes(element_size_bytes)
    , num_inputs(num_inputs)
    , num_outputs(num_outputs) {}
  int element_size_bytes;
  int num_inputs;
  int num_outputs;
  int step_input = 1;
  int step_output = 1;
  int ctas_per_output = 1;
  int input_mult[3] = {0, 0, 0};
  int output_mult[2] = {0, 0};

  int block_width;
  int block_height;
  int num_threads;

  bool vectorize_input = false;
  int output_vec_size = 1;
  int input_vec_size = 1;
  ReduceKernelType reduce_type;
  bool is_split_warp = false;

  int split_input(int parallelism) {
    int step = step_input;
    step_input *= parallelism;
    return step;
  }

  int split_output(int parallelism) {
    int step = step_output;
    step_output *= parallelism;
    return step;
  }

  dim3 block() const {
    return dim3(block_width, block_height);
  }

  dim3 grid() const {
    return dim3(reduce::div_up(num_outputs / output_vec_size, step_output), ctas_per_output);
  }

  C10_HOST_DEVICE bool should_block_x_reduce() const {
    return block_width > 1 && input_mult[BLOCK_X] != 0;
  }

  C10_HOST_DEVICE bool should_block_y_reduce() const {
    return block_height > 1 && input_mult[BLOCK_Y] != 0;
  }

  C10_HOST_DEVICE bool should_global_reduce() const {
    return input_mult[CTA] != 0;
  }

  C10_DEVICE bool should_store(int output_idx) const {
    return output_idx < num_outputs &&
      (!should_block_x_reduce() || threadIdx.x == 0) &&
      (!should_block_y_reduce() || threadIdx.y == 0);
  }

  C10_DEVICE bool should_reduce_tail() const {
    return (!should_block_y_reduce() || threadIdx.y == 0) &&
      (!should_global_reduce() || blockIdx.y == 0);
  }

  C10_HOST_DEVICE int input_idx() const {
    int lane = threadIdx.x;
    int warp = threadIdx.y;
    int cta2 = blockIdx.y;
    return (lane * input_mult[BLOCK_X] +
            warp * input_mult[BLOCK_Y] +
            cta2 * input_mult[CTA]);
  }

  template <int output_vec_size>
  C10_HOST_DEVICE int output_idx() const {
    int lane = threadIdx.x;
    int warp = threadIdx.y;
    int cta1 = blockIdx.x;
    return (lane * output_mult[BLOCK_X] +
            warp * output_mult[BLOCK_Y] +
            cta1 * step_output) * output_vec_size;
  }

  C10_DEVICE int shared_memory_offset(int offset) const {
    return threadIdx.x + (threadIdx.y + offset) * blockDim.x;
  }

  C10_DEVICE int staging_memory_offset(int cta2) const {
    int offset = cta2 + blockIdx.x * gridDim.y;
    if (!should_block_x_reduce()) {
      offset = threadIdx.x + offset * blockDim.x;
    }
    return offset;
  }

  template <typename T>
  void set_block_dimension(int64_t dim0, int64_t dim1) {
    const int max_num_threads = mnt_wrapper<T>::MAX_NUM_THREADS;
    int dim0_pow2 = dim0 < max_num_threads ? static_cast<int>(reduce::last_pow2(dim0)) : max_num_threads;
    int dim1_pow2 = dim1 < max_num_threads ? static_cast<int>(reduce::last_pow2(dim1)) : max_num_threads;
    if (is_split_warp) {
      // In this case, we split block_width and block_height more reasonably
      // to achieve better performance.
      int reduce_warp_size_div = 4;
      block_width = std::min(dim0_pow2, int(at::cuda::warp_size() / reduce_warp_size_div));
    } else {
      block_width = std::min(dim0_pow2, int(at::cuda::warp_size()));
    }
    block_height = std::min(dim1_pow2, int(max_num_threads / block_width));
    block_width = std::min(dim0_pow2, int(max_num_threads / block_height));
    while (shared_memory_size() >= at::cuda::getCurrentDeviceProperties()->sharedMemPerBlock && block_height > 1) {
      block_height /= 2;
    }
    // while (shared_memory_size() >= at::cuda::getCurrentDeviceProperties()->sharedMemPerBlock && block_width > 1) {
    //   block_width /= 2;
    // }
    num_threads = block_width * block_height;
  }

  int64_t global_memory_size() const {
    auto size = (int64_t)element_size_bytes * grid().x * ctas_per_output;
    if (!should_block_x_reduce()) {
      size *= block().x * output_vec_size;
    }
    return size;
  }

  int64_t shared_memory_size() const {
    if (!should_block_y_reduce() &&
        (!should_block_x_reduce() ||
         block_width <= at::cuda::warp_size())) {
      return 0;
    }
    // When the reduce type is imcontinuous, reduce the use of shared memory
    // in order to increase the usage of waves as much as possible,
    // ultimately achieving better performance.
    if (reduce_type == ReduceImcontinuousKernel) {
      return element_size_bytes * (block_width * block_height / 2) * output_vec_size;
    }
    return element_size_bytes * (block_width * block_height) * output_vec_size;
  }

  int semaphore_size() const {
    if (!should_global_reduce()) {
      return 0;
    }
    return sizeof(int) * grid().x;
  }

  int values_per_thread() const {
    return reduce::div_up(num_inputs, step_input);
  }
};

std::ostream& operator<<(std::ostream& out, const ReduceConfigMaca& config);
std::ostream& operator<<(std::ostream& out, const ReduceConfigCUDA& config);

static bool is_ndim_3_reduce_dim1(const TensorIterator& iter, const size_t &input_element_size, const size_t &output_element_size) {
  // input shape is 3d (dim0, dim1, dim2) reduce along dim1
  bool dims_cond = iter.num_reduce_dims() == 1 && iter.ndim() == 3;
  if (!dims_cond) {
    return false;
  }
  size_t input_index = iter.ntensors() - 1;
  size_t output_index = 0;
  bool reduce_dim_imcontinuous_cond = iter.strides(input_index)[iter.num_reduce_dims() - 1] >= iter.strides(input_index)[iter.num_reduce_dims()] &&
                                      iter.strides(input_index)[iter.num_reduce_dims() - 1] <= iter.strides(input_index)[iter.num_reduce_dims()+1];
  bool input_element_continuous_cond = iter.strides(input_index)[iter.num_reduce_dims()] == input_element_size && 
                                      iter.strides(input_index)[0] == iter.shape()[iter.num_reduce_dims()] * iter.strides(input_index)[iter.num_reduce_dims()] &&
                                      iter.strides(input_index)[iter.num_reduce_dims() + 1] == iter.shape()[0] * iter.strides(input_index)[0];
  bool output_element_continuous_cond = iter.strides(output_index)[iter.num_reduce_dims()] == output_element_size &&
                                        iter.strides(output_index)[iter.num_reduce_dims() + 1] == iter.strides(output_index)[iter.num_reduce_dims()] * iter.shape()[iter.num_reduce_dims()];
  size_t output_vec_size = get_output_vec_size(iter.shape()[1], iter.element_size(input_index));
  AT_ASSERT(output_vec_size > 0);
  bool align_cond = iter.strides(input_index)[iter.num_reduce_dims() - 1] % sizeof(int32_t) == 0 && iter.shape()[iter.num_reduce_dims()] % output_vec_size == 0;
  return reduce_dim_imcontinuous_cond && input_element_continuous_cond && output_element_continuous_cond && align_cond;  
}

static bool is_ndim_2_dim0_stride_not_continuous(const TensorIterator& iter, const size_t &input_element_size, const size_t &output_element_size) {
  // input shape is 2d (dim0, dim1), reduce along dim0 and stride(dim0) == n * shape(dim1), n >= 2
  bool dims_cond = iter.num_reduce_dims() == 1 && iter.ndim() == 2;
  if (!dims_cond) {
    return false;
  }
  size_t input_index = iter.ntensors() - 1;
  size_t output_index = 0;
  bool reduce_dim_imcontinuous_cond = iter.strides(input_index)[iter.num_reduce_dims() - 1] >= iter.strides(input_index)[iter.num_reduce_dims()];
  size_t normal_dim0_stride = iter.shape()[iter.num_reduce_dims()] * iter.strides(input_index)[iter.num_reduce_dims()];
  if (normal_dim0_stride == 0) {
    return false;
  }
  bool dim0_stride_cond = iter.strides(input_index)[0] % normal_dim0_stride == 0 && iter.strides(input_index)[0] / normal_dim0_stride >= 2;
  bool dim1_stride_cond = iter.strides(input_index)[1] == input_element_size;
  bool output_stride_cond = iter.strides(output_index)[1] == output_element_size;
  size_t output_vec_size = get_output_vec_size(iter.shape()[1], input_element_size);
  bool align_cond = iter.strides(input_index)[iter.num_reduce_dims() - 1] % sizeof(int32_t) == 0 && iter.shape()[iter.num_reduce_dims()] % output_vec_size == 0;
  return reduce_dim_imcontinuous_cond && dim0_stride_cond && dim1_stride_cond && output_stride_cond && align_cond;
}

static bool is_all_to_one_kernel(const TensorIterator& iter, const size_t &input_element_size) {
  size_t num_outputs = iter.num_output_elements();
  size_t inputs_per_output = iter.numel() / num_outputs;
  bool reduce_one = num_outputs == 1 && iter.num_reduce_dims() == iter.ndim();
  size_t input_index = iter.ntensors() - 1;
  size_t expect_stride = input_element_size;
  for (size_t i = 0; i < iter.strides(input_index).size(); i++) {
    if (expect_stride != iter.strides(input_index)[i]) {
      return false;
    }
    expect_stride *= iter.shape()[i];
  }
  if (num_outputs == 1 && inputs_per_output == 1) {
    return true;
  }
  bool align_cond = (inputs_per_output * iter.element_size(input_index)) % sizeof(int32_t) == 0;
  return reduce_one && align_cond;
}


static bool is_ndim_1_reduce_not_continuous(const TensorIterator& iter, const size_t &input_element_size, const size_t &output_element_size) {
  // if the input shape is (1, dim1) or (dim0, 1),
  // after coalesce dimension, the shape becomes (dim1) or (dim0)，so the dimension size is 1.
  if (iter.ndim() != 1) return false;
  size_t num_outputs = iter.num_output_elements();
  size_t inputs_per_output = iter.numel() / num_outputs;
  
  bool config_maca_cond = inputs_per_output == 1 && num_outputs == iter.shape()[0];
  size_t input_index = iter.ntensors() - 1;
  size_t output_index = 0;
  bool dim0_stride_cond = iter.strides(input_index)[0] == input_element_size;
  bool output_stride_cond = iter.strides(output_index)[0] == output_element_size;

  return config_maca_cond && dim0_stride_cond && output_stride_cond;
}

static bool is_ndim_3_reduce_dims_2_case(const TensorIterator& iter, const size_t &input_element_size, const size_t &output_element_size) {
  // input shape is 3d (dim0, dim1, dim2) reduce along dim0 and dim2
  bool dims_3_cond = iter.ndim() == 3;
  bool reduce_dims_2_cond = iter.num_reduce_dims() == 2;
  size_t input_index = iter.ntensors() - 1;
  size_t output_index = 0;
  bool input_element_continuous_cond = iter.strides(input_index)[0] == input_element_size &&
                                       iter.strides(input_index)[2] == iter.strides(input_index)[0] * iter.shape()[0] && 
                                       iter.strides(input_index)[1] == iter.strides(input_index)[2] * iter.shape()[2];
  bool output_element_continuous_cond = iter.strides(output_index)[2] == output_element_size;
  return dims_3_cond && reduce_dims_2_cond && input_element_continuous_cond && output_element_continuous_cond;
}

template<typename arg_t, typename scalar_t, typename out_scalar_t, int vt0>
ReduceConfigMaca setReduceConfigMaca(const TensorIterator& iter, ReduceKernelType reduce_type){
  int64_t num_outputs = iter.num_output_elements();
  int64_t inputs_per_output = iter.numel() / num_outputs;
  int64_t dim0;
  int64_t dim1;
  int64_t fastest_moving_stride;
  bool reduction_on_fastest_striding_dimension;  
  auto config = ReduceConfigMaca(sizeof(arg_t), num_outputs, inputs_per_output);
  int input_index = iter.ntensors() - 1;

  config.reduce_type = reduce_type;
     
  // In this case, we split block_width and block_height more reasonably
  // to achieve better performance.
  // For example, iter.shape()[0] <= 256 and iter.shape()[1] >= 4096.
  config.is_split_warp =
      maca_likely(!at::maca::get_maca_disable_reduce_split_warp()) && iter.ndim() == 2 &&
      ((reduce_type == ReduceImcontinuousKernel && iter.shape()[0] <= 512) ||
      (reduce_type == ReduceContinuousKernel && iter.shape()[0] <= 512 && iter.shape()[1] >= 4096));

  if (iter.ndim() > 0) {
    // Adjust block size to map block width to fastest changing dimension of input
    // tensor. This grants the best possible memory accessing pattern, given that
    // for non-contiguous tensor with space in between, we cannot have perfect
    // memory coalescing.
    reduction_on_fastest_striding_dimension =
        (iter.num_reduce_dims() == iter.ndim()) ||
        (iter.strides(/*arg=*/input_index)[0] <
        iter.strides(/*arg=*/input_index)[iter.num_reduce_dims()]);
    // Notice that dim0 & dim1 does NOT guarantee any launch configuration here!
    // dim0 & dim1 are more like the upper bound of the block dimension. The
    // actual launch config and reduction scheme is determined by setting values
    // to `config.input_mult` and `config.output_mult`.
    // We try to max out dim1 so that we have enough threads per CTA to deliver
    // performance for larger problem size.
    if (reduction_on_fastest_striding_dimension) {
      // Map block.x to the fastest reducing dimension. It implies:
      //   1. block_x_reduce is required.
      //   2. block.y now max out to num_outputs.
      dim0 = inputs_per_output;
      dim1 = num_outputs;
      fastest_moving_stride = iter.strides(/*arg=*/input_index)[0];
    } else {
      // Map block.x to the fastest non reducing dimension. It implies:
      //   1. block_x_reduce is turned off.
      //   2. block.y now max out to inputs_per_output.
      dim0 = num_outputs;
      dim1 = inputs_per_output;
      fastest_moving_stride = iter.strides(/*arg=*/input_index)[iter.num_reduce_dims()];
    }
  } else {
    reduction_on_fastest_striding_dimension = true;
    fastest_moving_stride = sizeof(scalar_t);
    dim0 = 1;
    dim1 = 1;
  }

  // We do vectorization to gain better memory access, there are two cases which we call
  // "vectorize along input" and "vectorize along output". Note that the "input/output"
  // here does not mean we are vectorizing load/store instructions. We always only vectorize
  // load instructions.
  //
  // Case 1: "vectorize along input"
  // This case happens when we are reducing along fastest moving dimesion. In such case, threads
  // with the same threadIdx.y works on the same reduction cooperatively and will produce results
  // for the same ouput. In such case, values in each loaded vector always correspond to the same ouput.
  // 
  // Case 2: "vectorize along fastest striding dimension and there are other reduce dims"
  // This case happens when the fastest moving dimesion is one of the dimensions of reduction.
  // For instance: input shape is (dim0, dim1, dim2), and reduce dims are dim0 and dim2, in this case dim2 can be vectorized 
  // and dim to should be divisible for vec_size to make sure there are no stride step across any vectors.
  //
  // Case 3: "vectorize along output"
  // This case happens when the fastest moving dimesion is not the dimension of reduction. In such case,
  // threads with different threadIdx.x are independent and will produce results for different outputs.
  // In such case, values in each loaded vector always correspond to different outputs.
  if (fastest_moving_stride == sizeof(scalar_t)) {
    if (reduction_on_fastest_striding_dimension && dim0 > 128 && iter.num_reduce_dims() == 1) {
      // Case 1: "vectorize along input"
      // Note that if vt0 < ReduceConfig::vec_size, then this means the register pressure could be high, in such case,
      // we should avoid vectorization.
      config.vectorize_input = true;
      config.input_vec_size = get_vec_size(dim0, sizeof(scalar_t));
      dim0 /= config.input_vec_size;
    } else if (reduction_on_fastest_striding_dimension && is_ndim_3_reduce_dims_2_case(iter, sizeof(scalar_t), sizeof(out_scalar_t))) {
      // Case 2: "vectorize along fastest striding dimension and there are other reduce dims"
      config.input_vec_size = get_output_vec_size(iter.shape()[0], sizeof(scalar_t));
      dim0 /= config.input_vec_size;
    } else if (!reduction_on_fastest_striding_dimension) {
      // Case 3: "vectorize along output"
      if (is_ndim_3_reduce_dim1(iter, sizeof(scalar_t), sizeof(out_scalar_t))) {
        config.output_vec_size = get_output_vec_size(iter.shape()[iter.num_reduce_dims()], sizeof(scalar_t));
      } else {
        config.output_vec_size = get_output_vec_size(dim0, sizeof(scalar_t));
      }
      dim0 /= config.output_vec_size;
    }
  }

  // Adjust block_width and block_height
  config.set_block_dimension<scalar_t>(dim0, dim1);

  int block_width = config.block_width;
  int block_height = config.block_height;

  if (iter.ndim() == 0 || reduction_on_fastest_striding_dimension) {
    // Split the input across lanes if the input is contiguous in the reduced
    // dimension. This will require reduction between threads using warp
    // shuffle instructions and shared memory (if block_width > warpSize).
    config.input_mult[0] = config.split_input(block_width);
  } else {
    // Otherwise split the output across lanes in a warp.
    config.output_mult[0] = config.split_output(block_width);
  }

  int min_values_per_thread = 16;
  int max_values_per_thread = 256;
  int height_multiple = 16;

  // To achieve better performance.
  if (reduce_type == ReduceImcontinuousKernel && iter.ndim() == 2) {
    min_values_per_thread = 8;
    max_values_per_thread = 64;

    // When shape[0] is less than or equal to 256,
    // split finer granularity to achieve better performance.
    if (iter.shape()[0] <= 256) {
      height_multiple = 1;
    }
  }
  int block_height_multiple = block_height * height_multiple;
  if (config.values_per_thread() >= block_height_multiple || config.values_per_thread() >= max_values_per_thread) {    // Divide the input across warps in a thread-block, if that leaves at least
    // 16 elements to be summed by each thread. This will require inter-warp
    // reduction using shared memory.
    config.input_mult[1] = config.split_input(block_height);
  } else {
    // Otherwise, each warp handles a separate output.
    config.output_mult[1] = config.split_output(block_height);
  }

  const int max_sm_size = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  const int blocks_per_sm = 4;
  const int target_grid_size = max_sm_size * blocks_per_sm;
  int grid = config.grid().x;
  if (config.input_mult[1] != 0 && config.values_per_thread() >= max_values_per_thread && grid <= target_grid_size) {
    // Divide the input across thread-blocks if the amount of work per-thread
    // is large enough and the size of the output is small enough. This will
    // require a reduction using global memory.
    // If we decide to split input across blocks, as long as we can get enough
    // number of blocks (`target_grid_size`) to balance SM, we should still
    // make the number of values per thread large for best performance.
    int ctas_per_output1 = reduce::div_up(target_grid_size, grid);
    int ctas_per_output2 = reduce::div_up(config.values_per_thread(), min_values_per_thread);
    int ctas_per_output3 = reduce::div_up(config.values_per_thread(), max_values_per_thread);
    // We want the minimum of ctas_per_output1 and ctas_per_output2, so that each thread can have
    // a large number of values to deal with. But we don't want values_per_thread to be larger than
    // max_values_per_thread
    config.ctas_per_output = std::max(std::min<int>(ctas_per_output1, ctas_per_output2), ctas_per_output3);
    if (config.ctas_per_output > 1) {
      config.input_mult[2] = config.split_input(config.ctas_per_output);
    }
  }
  return config;
}

template <typename scalar_t, int vec_size>
struct alignas(sizeof(scalar_t) * vec_size) aligned_vector {
  scalar_t val[vec_size];
};

template <typename scalar_t>
__device__ __forceinline__ scalar_t load(const scalar_t *base_ptr, uint32_t offset) {
  return base_ptr[offset];
}

template <>
__device__ __forceinline__ bool load(const bool *base_ptr, uint32_t offset) {
  static_assert(sizeof(bool) == sizeof(char), "");
  // NOTE: [Loading boolean values]
  // Protect against invalid boolean values by loading as a byte
  // first, then converting to bool (see gh-54789).
  return *reinterpret_cast<const unsigned char*>(base_ptr+offset);
}

template <int vec_size, typename scalar_t, typename = std::enable_if_t<!std::is_same<scalar_t, bool>()>>
__device__ __forceinline__ aligned_vector<scalar_t, vec_size> load_vector(const scalar_t* base_ptr, uint32_t offset) {
  using vec_t = aligned_vector<scalar_t, vec_size>;
  auto *from = reinterpret_cast<const vec_t *>(base_ptr);
  return from[offset];
}

template <int vec_size>
__device__ __forceinline__ aligned_vector<bool, vec_size> load_vector(const bool *base_ptr, uint32_t offset) {
  // See NOTE [Loading boolean values]
  auto tmp = load_vector<vec_size>(reinterpret_cast<const uint8_t*>(base_ptr), offset);
  aligned_vector<bool, vec_size> ret;
  for (int i = 0; i < vec_size; ++i) {
    ret.val[i] = bool(tmp.val[i]);
  }
  return ret;
}

template <bool can_acc, typename arg_t, typename out_scalar_t, typename ops_t>
C10_DEVICE __forceinline__ arg_t accumulate_in_output_once(
  out_scalar_t* out,
  arg_t value,
  ops_t ops,
  typename std::enable_if<can_acc>::type* = nullptr) {
  arg_t ret;
  ret = ops.combine(*out, value);
  return ret;
}

template <bool can_acc, typename arg_t, typename out_scalar_t, typename ops_t>
C10_DEVICE __forceinline__ arg_t accumulate_in_output_once(
  out_scalar_t* out,
  arg_t value,
  ops_t ops,
  typename std::enable_if<!can_acc>::type* = nullptr) {
  assert(false);
  arg_t ret;
  return ret;
}

template <typename out_scalar_t, class T, typename index_t>
C10_DEVICE __forceinline__ void set_results1(
    const T x,
    const index_t base_offset,
    int noutputs,
    const char* dst0,
    const char* dst1) {
  assert(noutputs == 1);
  auto res = (out_scalar_t*)((char*)dst0 + base_offset);
  *res = x;
}

// Currently implemented for max of two outputs
template <typename out_scalar_t, class T1, class T2, typename index_t>
C10_DEVICE __forceinline__ void set_results1(
    const thrust::pair<T1, T2> x,
    const index_t base_offset,
    int noutputs,
    const char* dst0,
    const char* dst1) {
  if (noutputs >= 1) {
    auto res0 = (T1*)((char*)dst0 + base_offset);
    *res0 = x.first;
  }
  if (noutputs >= 2) {
    // base offset is computed assuming element size being sizeof(T1), so we
    // need to make a correction to obtain the correct base offset
    auto res1 = (T2*)((char*)dst1 + base_offset / sizeof(T1) * sizeof(T2));
    *res1 = x.second;
  }
}

C10_DEVICE __forceinline__ bool mark_block_finished(int* semaphores) {
  __shared__ bool is_last_block_done_shared;

  __syncthreads();
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    int prev_blocks_finished = atomicAdd(&semaphores[0], 1);
    is_last_block_done_shared = (prev_blocks_finished == gridDim.x - 1);
  }

  __syncthreads();

  return is_last_block_done_shared;
}

C10_DEVICE __forceinline__ bool mark_block_finished1(int* semaphores) {
  __shared__ bool is_last_block_done_shared;

  __syncthreads();
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    int prev_blocks_finished = atomicAdd(&semaphores[blockIdx.x], 1);
    is_last_block_done_shared = (prev_blocks_finished == gridDim.y - 1);
  }

  __syncthreads();

  return is_last_block_done_shared;
}

}} // namespace at::native
