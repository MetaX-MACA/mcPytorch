#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cuda/BinaryInternal.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAMathCompat.h>
#include <c10/util/TypeSafeSignMath.h>
#include <ATen/native/cuda/JitLoops.cuh>
#include <ATen/native/cuda/Loops.cuh>

#include <type_traits>

// NOTE: CUDA on Windows requires that the enclosing function
// of a __device__ lambda not have internal linkage.

namespace at::native {

template <typename scalar_t, typename return_t = scalar_t, typename func_t>
void opmath_symmetric_gpu_kernel_with_scalars_arity3(TensorIteratorBase& iter, const func_t& f) {
  // Use symmetric property of the functor to reduce number of kernels,
  // requires f(a, b) == f(b, a)
  TORCH_INTERNAL_ASSERT(iter.ntensors() == 3);

  using traits = function_traits<func_t>;
  using opmath_arg_t = typename traits::template arg<0>::type;
  static_assert(
      traits::arity == 2,
      "gpu_kernel_with_scalars only supports two input arguments");
  static_assert(std::is_same<opmath_arg_t, typename traits::template arg<1>::type>::value,
                "f is not symmetric");

  OptionalDeviceGuard device_guard;
  opmath_arg_t scalar_val{};

  if (iter.is_cpu_scalar(1)) {
    scalar_val = iter.scalar_value<opmath_arg_t>(1);
    iter.remove_operand(1);

    // TODO: When all kernels that use gpu_kernel_with_scalars are
    // ported to structured, this device guard can be deleted.  This
    // works around incorrect device guard generation for pre-structured
    // kernels device guards, but structured kernels do it right and
    // we can assume the device is already set correctly
    device_guard.reset_device(iter.device(1));
  } else if (iter.is_cpu_scalar(2)) {
    scalar_val = iter.scalar_value<opmath_arg_t>(2);
    iter.remove_operand(2);
  }

  if (iter.ninputs() == 2) {
    gpu_kernel_maca_arity2(iter, BinaryFunctor<scalar_t, scalar_t, return_t, func_t>(f));
  } else {
    AUnaryFunctor<scalar_t, scalar_t, return_t, func_t> unary_f(f, scalar_val);
    gpu_kernel_maca_arity1(iter, unary_f);
  }
}

CONSTEXPR_EXCEPT_WIN_CUDA char mul_name[] = "mul_kernel";
void mul_kernel_cuda(TensorIteratorBase& iter) {
  auto common_dtype = iter.common_dtype();
  if (common_dtype == kComplexHalf) {
    using scalar_t = c10::complex<at::Half>;
#if AT_USE_JITERATOR()
    static const auto mul_string = jiterator_stringify(
        template <typename T> T mul_kernel(T a, T b) { return a * b; });
    opmath_jitted_gpu_kernel_with_scalars<mul_name, scalar_t, scalar_t>(
        iter, mul_string);
#else
    using opmath_t = at::opmath_type<scalar_t>;
    opmath_symmetric_gpu_kernel_with_scalars<scalar_t>(
        iter, binary_internal::MulFunctor<opmath_t>());
#endif
  } else {
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
        kHalf, kBFloat16, kBool, iter.common_dtype(), "mul_cuda", [&]() {
          using opmath_t = at::opmath_type<scalar_t>;
          opmath_symmetric_gpu_kernel_with_scalars_arity3<scalar_t>(
              iter, binary_internal::MulFunctor<opmath_t>());
        });
  }
}

REGISTER_DISPATCH(mul_stub, &mul_kernel_cuda);

} // namespace at::native
