#include <ATen/cuda/CUDAApplyUtils.cuh>

namespace at::native {

template <typename scalar_t, typename IndexType, bool upper>
C10_LAUNCH_BOUNDS_1(cuda::getApplyBlockSize())
__global__ void triu_tril_kernel_opt_dim2(
    IndexType self_info_sizes_0, IndexType self_info_strides_0, IndexType result_info_strides_0,
    IndexType self_info_sizes_1, IndexType self_info_strides_1, IndexType result_info_strides_1,
    scalar_t *result_info_data, scalar_t *self_info_data,
    const int64_t k,
    const int64_t N) {
  int64_t linear_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (linear_idx >= N) {
    return;
  }

  IndexType self_offset = 0, result_offset = 0;
  // Compute column index and corresponding offset
  IndexType col = linear_idx % self_info_sizes_1;
  linear_idx /= self_info_sizes_1;
  self_offset += self_info_strides_1 * col;
  result_offset += result_info_strides_1 * col;

  // Compute row index and corresponding offset
  IndexType row = linear_idx % self_info_sizes_0;
  linear_idx /= self_info_sizes_0;
  self_offset += self_info_strides_0 * row;
  result_offset += result_info_strides_0 * row;

  bool mask = upper ? (col - row >= k) : (col - row <= k);
  *(result_info_data + result_offset) = mask ? *(self_info_data + self_offset) : scalar_t(0);
}

}
