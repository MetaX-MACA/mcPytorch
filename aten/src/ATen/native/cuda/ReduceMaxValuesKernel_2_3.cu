#include <ATen/native/cuda/ReduceMaxValuesKernel.cuh>

namespace at::native {

void max_all_launch_kernel(TensorIterator &iter) {
  AT_DISPATCH_ALL_TYPES_AND3(kBFloat16, kHalf, kBool, iter.input_dtype(), "max_all_cuda", [&] {
    max_values_kernel_cuda_impl<scalar_t>(iter);
  });
}

} // namespace at::native
