#include <ATen/ceil_div.h>
#include <ATen/cuda/Atomic.cuh>
#include <ATen/cuda/DeviceUtils.cuh>
#include <ATen/cuda/AsmUtils.cuh>

namespace at {
namespace native {

template <typename scalar_t, typename bitwise_t, typename index_t>
__device__ void radixSelect_largest(
    scalar_t* data,
    index_t k,
    index_t sliceSize,
    index_t withinSliceStride,
    int* smem,
    scalar_t* topK) {
  // Per-thread buckets into which we accumulate digit counts in our
  // radix
  int counts[RADIX_SIZE];

  // We only consider elements x such that (x & desiredMask) == desired
  // Initially, we consider all elements of the array, so the above
  // statement is true regardless of input.
  bitwise_t desired = 0;
  bitwise_t desiredMask = 0;

  // We are looking for the top kToFind-th element when iterating over
  // digits; this count gets reduced by elimination when counting
  // successive digits
  int kToFind = k;

  // We start at the most significant digit in our radix, scanning
  // through to the least significant digit
  for (int digitPos = sizeof(scalar_t) * 8 - RADIX_BITS; digitPos >= 0;
       digitPos -= RADIX_BITS) {
    // Count radix distribution for the current position and reduce
    // across all threads
    countRadixUsingMask<
        scalar_t,
        bitwise_t,
        index_t,
        int,
        RADIX_SIZE,
        RADIX_BITS>(
        counts,
        smem,
        desired,
        desiredMask,
        digitPos,
        sliceSize,
        withinSliceStride,
        data);

    auto found_unique = [&](int i, int count) -> bool {
      /* All threads have the same value in counts here, so all */
      /* threads will return from the function. */
      if (count == 1 && kToFind == 1) {
        /* There is a unique answer. */
        desired = at::cuda::Bitfield<bitwise_t>::setBitfield(
            desired, i, digitPos, RADIX_BITS);
        desiredMask = at::cuda::Bitfield<bitwise_t>::setBitfield(
            desiredMask, RADIX_MASK, digitPos, RADIX_BITS);

        /* The answer is now the unique element v such that: */
        /* (v & desiredMask) == desired */
        /* However, we do not yet know what the actual element is. We */
        /* need to perform a search through the data to find the */
        /* element that matches this pattern. */
        *topK = findPattern<scalar_t, bitwise_t, index_t>(
            (scalar_t*)smem,
            data,
            sliceSize,
            withinSliceStride,
            desired,
            desiredMask);
        return true;
      }
      return false;
    };
    auto found_non_unique = [&](int i, int count) -> bool {
      if (count >= kToFind) {
        desired =
            at::cuda::Bitfield<bitwise_t>::setBitfield(
                desired, i, digitPos, RADIX_BITS);
        desiredMask = at::cuda::Bitfield<bitwise_t>::setBitfield(
            desiredMask, RADIX_MASK, digitPos, RADIX_BITS);

        /* The top-Kth element v must now be one such that: */
        /* (v & desiredMask == desired) */
        /* but we haven't narrowed it down; we must check the next */
        /* least-significant digit */
        return true;
      }
      kToFind -= count;
      return false; // continue the loop
    };

    // All threads participate in the comparisons below to know the
    // final result
#pragma unroll
    for (int i = RADIX_SIZE - 1; i >= 0; --i) {
      int count = counts[i];
      if (found_unique(i, count)) {
        return;
      }
      if (found_non_unique(i, count)) {
        break;
      }
    }
  } // end digitPos for

  // There is no unique result, but there is a non-unique result
  // matching `desired` exactly
  *topK = TopKTypeConfig<scalar_t>::deconvert(desired);
}

} // namespace native
} // namespace at