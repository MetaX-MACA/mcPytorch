#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/Dispatch.h>
#include <ATen/NumericUtils.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/ReduceAllOps.h>
#include <ATen/native/ReduceOps.h>
#include <ATen/native/SharedReduceOps.h>
#include <ATen/native/TensorCompare.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cuda/ReduceOps.h>
#include <ATen/cuda/NumericLimits.cuh>
#include <ATen/native/cuda/Reduce.cuh>

#include <ATen/Dispatch.h>
#include <ATen/NumericUtils.h>
#include <ATen/cuda/NumericLimits.cuh>

namespace at::native {

template <typename acc_t>
struct MaxNanFunctor {
  __device__ __forceinline__ acc_t operator()(acc_t a, acc_t b) const {
    return (at::_isnan(a) || a > b) ? a : b;
  }
};

template <typename scalar_t, typename acc_t = scalar_t>
void max_values_kernel_cuda_impl(TensorIterator& iter) {
  gpu_reduce_kernel<scalar_t, scalar_t>(
      iter,
      func_wrapper<acc_t>(MaxNanFunctor<acc_t>()),
      at::numeric_limits<acc_t>::lower_bound());
}

} // namespace at::native
