#pragma once

#include <ATen/detail/FunctionTraits.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/TensorIteratorDynamicCasting.h>
#include <ATen/cuda/detail/OffsetCalculator.cuh>
#include <ATen/OpMathType.h>
#include <ATen/native/cuda/thread_constants.h>

#include <thrust/tuple.h>

#include <ATen/native/cuda/MemoryAccess.cuh>

// Note:
// CUDA and ROCm get diverged in this PR:
//   https://github.com/pytorch/pytorch/pull/32383
// Because for some reason trying to enable vectorized
// memory access introduce regression on ROCm.

#define USE_MACA_COPY

#if !defined(USE_ROCM)
  #include <ATen/native/cuda/Loops.cuh>
  #include <ATen/native/cuda/maca_kernels/CUDALoopsCopy.cuh>
#else
  #include <ATen/native/cuda/ROCmLoops.cuh>
#endif

#undef USE_MACA_COPY

namespace at { namespace native {

template <typename func_t>
void gpu_kernel_maca_arity1_copy(TensorIteratorBase& iter, const func_t& f) {

  for (int arg = 0; arg < iter.ntensors(); arg++) {
    TORCH_INTERNAL_ASSERT(
      iter.device(arg).is_cuda(),
      "argument ", arg, ": expected a CUDA device but found ", iter.device(arg));
  }

  if (iter.numel() == 0) {
    return;
  }

  if (!iter.can_use_32bit_indexing()) {
    for (auto& sub_iter : iter.with_32bit_indexing()) {
      gpu_kernel_maca_arity1_copy(sub_iter, f);
    }
    return;
  }

  at::native::gpu_kernel_impl_maca_arity1_copy(iter, f);
}

}} //namespace at::native
