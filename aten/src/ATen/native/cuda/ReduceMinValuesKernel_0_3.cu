#include <ATen/native/cuda/ReduceMinValuesKernel.cuh>

namespace at::native {

void min_all_launch_kernel(TensorIterator &iter) {
  AT_DISPATCH_ALL_TYPES_AND3(kBFloat16, kHalf, kBool, iter.input_dtype(), "min_all_cuda", [&] {
    min_values_kernel_cuda_impl<scalar_t>(iter);
  });
}

} // namespace at::native
