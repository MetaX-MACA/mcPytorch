
#pragma once

#include<cstdlib>
std::tuple<uint64_t, dim3, dim3> calc_execution_policy_opt(int64_t total_elements) {
  const uint64_t numel = static_cast<uint64_t>(total_elements);

  if (numel < (128*8192)) {
    const uint32_t vec = curand4_engine_calls;
    const uint32_t block = block_size_bound;
    dim3 dim_block(block);
    dim3 grid((numel + block * vec - 1) / (block * vec));
    return std::make_tuple(vec, grid, dim_block);
  }

  const uint32_t block_size = block_size_bound;
  const uint32_t unroll = curand4_engine_calls;
  dim3 dim_block(block_size);
  dim3 grid((numel + block_size * unroll - 1) / (block_size * unroll));
  uint32_t blocks_per_sm = 0;

  blocks_per_sm = at::cuda::getCurrentDeviceProperties()->maxThreadsPerMultiProcessor / block_size;
  grid.x = std::min(
      static_cast<uint32_t>(at::cuda::getCurrentDeviceProperties()->multiProcessorCount) * blocks_per_sm,
      grid.x);

  uint64_t counter_offset = ((numel - 1) / (block_size * grid.x * unroll) + 1)
                            * curand4_engine_calls;
  return std::make_tuple(counter_offset, grid, dim_block);
}

// grid stride loop kernel for distributions
template<typename scalar_t, typename accscalar_t, int unroll_factor, typename dist_t, typename transform_t>
C10_LAUNCH_BOUNDS_2(block_size_bound, grid_size_bound)
__global__ void distribution_elementwise_grid_stride_kernel_opt(
                                                            scalar_t* out_data,
                                                            int numel,
                                                            PhiloxCudaState philox_args,
                                                            const dist_t dist_func,
                                                            const transform_t transform_func) {
  using StoreT = at::native::memory::aligned_vector<scalar_t, unroll_factor>;
  auto seeds = at::cuda::philox::unpack(philox_args);
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  curandStatePhilox4_32_10_t state;
  curand_init(std::get<0>(seeds),
              idx,
              std::get<1>(seeds),
              &state);
  StoreT st;
  for(int linear_index = idx * unroll_factor; linear_index < numel; linear_index += blockDim.x * gridDim.x * unroll_factor) {
    auto rand = dist_func(&state);
    #pragma unroll
    for (int ii = 0; ii < unroll_factor; ii++) {
      st.val[ii] = transform_func(static_cast<accscalar_t>((&rand.x)[ii]));
    }
    *(reinterpret_cast<StoreT*>(out_data + linear_index)) = st;
  }

}

