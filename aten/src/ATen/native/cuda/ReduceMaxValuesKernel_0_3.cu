#include <ATen/native/cuda/ReduceMaxValuesKernel.cuh>

namespace at::native {

void max_values_kernel_cuda(TensorIterator& iter) {
  AT_DISPATCH_ALL_TYPES_AND3(
      kBFloat16, kHalf, kBool, iter.dtype(), "max_values_cuda", [&]() {
        max_values_kernel_cuda_impl<scalar_t>(iter);
      });
}

REGISTER_DISPATCH(max_values_stub, &max_values_kernel_cuda);

} // namespace at::native
