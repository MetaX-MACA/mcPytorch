#include <ATen/native/cuda/ReduceMaxValuesKernel.cuh>

namespace at::native {

void max_launch_kernel(TensorIterator& iter) {
  AT_DISPATCH_ALL_TYPES_AND3(
      kBFloat16, kHalf, kBool, iter.input_dtype(), "max_cuda", [&]() {
        gpu_reduce_kernel<scalar_t, scalar_t, 1>(
            iter,
            MaxOps<scalar_t>{},
            thrust::pair<scalar_t, int64_t>(
                at::numeric_limits<scalar_t>::lower_bound(), 0));
      });
}

} // namespace at::native
