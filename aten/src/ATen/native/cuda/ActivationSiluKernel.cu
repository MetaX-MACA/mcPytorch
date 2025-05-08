#define TORCH_ASSERT_NO_OPERATORS
#define _USE_MATH_DEFINES

#include <ATen/native/Activation.h>

#include <cmath>

#include <thrust/tuple.h>

#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/core/TensorBase.h>
#include <c10/core/Scalar.h>
#include <c10/cuda/CUDAMathCompat.h>
#include <ATen/cuda/ApplyGridUtils.cuh>
#include <ATen/cuda/detail/OffsetCalculator.cuh>
#include <ATen/native/cuda/Loops.cuh>
#include <c10/util/complex.h>

namespace at::native {
namespace {

void silu_kernel(TensorIteratorBase& iter) {
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2_FLOAT_AND_HALF_SPECIAL(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.dtype(),
      "silu_cuda",
      [&]() {
        gpu_kernel_maca_arity1(iter, [] GPU_LAMBDA(scalar_t x) -> scalar_t {
          using opmath_t = at::opmath_type<scalar_t>;
          const opmath_t x_acc = static_cast<opmath_t>(x);
          return x_acc *  __builtin_mxc_rcpf(opmath_t(1) + __expf(-x_acc));
        });
      },
      [&]() {
        gpu_kernel_maca_arity1(iter, [] GPU_LAMBDA(scalar_t x) -> scalar_t {
          using opmath_t = at::opmath_type<scalar_t>;
          const opmath_t x_acc = static_cast<opmath_t>(x);
          return x_acc / (opmath_t(1) + ::exp(-x_acc));
        });
      }
    );
}
  

void silu_backward_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2_DOUBLE_SPECIAL(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.dtype(),
      "silu_backward_cuda",
      [&]() {
        gpu_kernel_maca_arity2(iter, [] GPU_LAMBDA(scalar_t dy, scalar_t x) -> scalar_t {
          using opmath_t = at::opmath_type<scalar_t>;
          const opmath_t dy_acc = static_cast<opmath_t>(dy);
          const opmath_t x_acc = static_cast<opmath_t>(x);
          const opmath_t s_acc =
              opmath_t(1) / (opmath_t(1) + __expf(-x_acc));
          return dy_acc * s_acc * (opmath_t(1) + x_acc * (opmath_t(1) - s_acc));
        });
      },
      [&]() {
        gpu_kernel_maca_arity2(iter, [] GPU_LAMBDA(scalar_t dy, scalar_t x) -> scalar_t {
          using opmath_t = at::opmath_type<scalar_t>;
          const opmath_t dy_acc = static_cast<opmath_t>(dy);
          const opmath_t x_acc = static_cast<opmath_t>(x);
          const opmath_t s_acc =
              opmath_t(1) / (opmath_t(1) + c10::cuda::compat::exp(-x_acc));
          return dy_acc * s_acc * (opmath_t(1) + x_acc * (opmath_t(1) - s_acc));
        });
      }
  );
}
} // namespace

REGISTER_DISPATCH(silu_stub, &silu_kernel);
REGISTER_DISPATCH(silu_backward_stub, &silu_backward_kernel);

} // namespace at::native