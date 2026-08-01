#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/custom_fuse_rsm_kernel.h>

#include <type_traits>
#include <thrust/tuple.h>
#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <iostream>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like_native.h>
#include <ATen/ops/zeros_like_native.h>
#endif

#include <c10/cuda/CUDAMathCompat.h>
#include <maca_bfloat16.h>
#include <ATen/native/cuda/maca_kernels/custom_fused_rsm_kernels.cuh>

namespace at::native {

std::tuple<Tensor, Tensor, Tensor> custom_fused_rsm_cuda(
    const Tensor& x,
    const Tensor& variance,
    const Tensor& weight) {

    int batch_size = x.size(0);
    int hidden_dim = 768;

    auto options = TensorOptions().dtype(at::kBFloat16).device(x.device());
    Tensor out = at::empty({batch_size, hidden_dim}, options);

    customFuseRsmKernelImpl(&x, &variance, &weight, &out);

    //return std::make_tuple(std::move(out), std::move(variance), std::move(x));
    return std::make_tuple(std::move(out), std::move(variance), std::move(x.slice(1, 0, 768))); // return slice x[16384. 768] in graph
}


REGISTER_DISPATCH(customFuseRsmKernel, &customFuseRsmKernelImpl)
} // namespace at::native