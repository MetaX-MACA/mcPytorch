#include <ATen/native/cuda/SparseBinaryOpIntersectionKernel.cuh>

namespace at::native {

namespace {

void sparse_mask_intersection_out_cuda_kernel(
    Tensor& result,
    const Tensor& x,
    const Tensor& y) {
  using CUDAValueLhsProjKernel = CUDAValueSelectionIntersectionKernel<LhsProjOp>;
  _sparse_binary_op_intersection_kernel_out<CUDAKernelLauncher, CUDAValueLhsProjKernel>(
      result, x, y, true
  );
}

}

REGISTER_CUDA_DISPATCH(sparse_mask_intersection_out_stub, &sparse_mask_intersection_out_cuda_kernel);

} // namespace at::native