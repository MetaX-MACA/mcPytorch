#include <ATen/native/cuda/SparseBinaryOpIntersectionKernel.cuh>

namespace at::native {

namespace {

void mul_sparse_sparse_out_cuda_kernel(
    Tensor& result,
    const Tensor& x,
    const Tensor& y) {
  using CUDAValueSelectionMulKernel = CUDAValueSelectionIntersectionKernel<MulOp>;
  _sparse_binary_op_intersection_kernel_out<CUDAKernelLauncher, CUDAValueSelectionMulKernel>(
      result, x, y
  );
}

}

REGISTER_CUDA_DISPATCH(mul_sparse_sparse_out_stub, &mul_sparse_sparse_out_cuda_kernel);

} // namespace at::native
