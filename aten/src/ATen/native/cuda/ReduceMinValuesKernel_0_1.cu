#include <ATen/native/cuda/ReduceMinValuesKernel.cuh>

namespace at::native {

void min_values_kernel_cuda(TensorIterator& iter) {
  AT_DISPATCH_ALL_TYPES_AND3(kBFloat16, kHalf, kBool, iter.dtype(), "min_values_cuda", [&]() {
    min_values_kernel_cuda_impl<scalar_t>(iter);
  });
}

REGISTER_DISPATCH(min_values_stub, &min_values_kernel_cuda);

} // namespace at::native
