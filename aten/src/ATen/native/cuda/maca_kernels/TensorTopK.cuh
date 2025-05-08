#include <ATen/native/cuda/TensorTopK.h>
#include <ATen/core/TensorBase.h>
#include <ATen/ceil_div.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/detail/TensorInfo.cuh>
#include <ATen/cuda/detail/OffsetCalculator.cuh>
#include <ATen/cuda/ScanUtils.cuh>
#include <ATen/cuda/AsmUtils.cuh>
#include <ATen/cuda/DeviceUtils.cuh>
#include <ATen/native/cuda/SortingCommon.cuh>
#include <ATen/native/cuda/maca_kernels/SortingRadixSelect.cuh>
#include <ATen/cuda/cub.cuh>
#include <c10/cuda/CUDACachingAllocator.h>
#include <ATen/cuda/detail/KernelUtils.h>

#include <c10/macros/Macros.h>

using namespace at::native;

namespace at::native {

template <typename T>
struct AddOp {
  __device__ __forceinline__ T operator()(T const &lhs, T const &rhs) {
    return (lhs + rhs);
  }
};

template <typename T, typename IndexType, int Dim>
C10_LAUNCH_BOUNDS_1(1024)
__global__ void gatherTopK_opt(at::cuda::detail::TensorInfo<T, IndexType> input,
                           IndexType inputSliceSize,
                           IndexType outputSliceSize, // aka `k`

                           IndexType numInputSlices,
                           IndexType inputWithinSliceStride,

                           at::cuda::detail::TensorInfo<T, IndexType> topK,
                           IndexType topKWithinSliceStride,

                           at::cuda::detail::TensorInfo<int64_t, IndexType> indices,
                           IndexType indicesWithinSliceStride,
                           T* kthValues) {
  // Indices are limited to integer fp precision, so counts can fit in
  // int32, regardless of IndexType
#if defined(USE_ROCM) || defined(USE_MACA)
  __shared__ int smem[64];
#else
  __shared__ int smem[32]; // one per each warp, up to warp limit
#endif
  IndexType slice = getLinearBlockId<IndexType>();
  if (slice >= numInputSlices) {
    return;
  }

  // Find the start offset for our slice
  IndexType sliceStartIndex =
    at::cuda::detail::IndexToOffset<T, IndexType, Dim>::get(slice, input);
  IndexType topKSliceStartIndex =
    at::cuda::detail::IndexToOffset<T, IndexType, Dim>::get(slice, topK);
  IndexType indicesSliceStartIndex =
    at::cuda::detail::IndexToOffset<int64_t, IndexType, Dim>::get(slice, indices);

  T* inputSliceStart = &input.data[sliceStartIndex];
  T* topKSliceStart = &topK.data[topKSliceStartIndex];
  int64_t* indicesSliceStart = &indices.data[indicesSliceStartIndex];

  // Find the k-th highest element in our input
  T topKValue = static_cast<T>(0);
  radixSelect_largest<T, typename TopKTypeConfig<T>::RadixType, IndexType>(
    inputSliceStart, outputSliceSize,
    inputSliceSize, inputWithinSliceStride,
    smem, &topKValue);
  const auto topKConverted = at::native::TopKTypeConfig<T>::convert(topKValue);

  // Every value that is strictly less/greater than `pattern`
  // (depending on sort dir) in sorted int format is in the top-K.
  // The top-K value itself might not be unique.
  //
  // Since there are a variable number of elements that we see that
  // are within the top-k, we don't know at what index to write out
  // the resulting values.
  // In order to get this, we perform an exclusive prefix sum of
  // `hasTopK`. This will return the resulting index into which we
  // need to write the result, if a thread has a result.

  // All threads need to participate in the loop and the prefix sum,
  // but not necessarily in the load; hence loop bounds being rounded
  // up to a multiple of the block dim.
  IndexType numIterations = round_up(inputSliceSize, (IndexType) blockDim.x);
  IndexType writeIndexStart = 0;

  for (IndexType i = threadIdx.x; i < numIterations; i += blockDim.x) {
    bool inRange = (i < inputSliceSize);
    T v = static_cast<T>(0);
    if (inRange) {
      v = doLdg(&inputSliceStart[i * inputWithinSliceStride]);
    }
    const auto convertedV = at::native::TopKTypeConfig<T>::convert(v);
    bool hasTopK = inRange && (convertedV > topKConverted);

    int index;
    int carry;
    at::cuda::exclusiveBinaryPrefixScan<int, true>(
        smem, hasTopK, &index, &carry, AddOp<int>());

    if (hasTopK) {
      int writeIndex = writeIndexStart + index;

      IndexType topKOffset = writeIndex * topKWithinSliceStride;
      IndexType indexOffset = writeIndex * indicesWithinSliceStride;

      topKSliceStart[topKOffset] = v;
      indicesSliceStart[indexOffset] = i;
    }

    writeIndexStart += carry;
  }

  // We need to fill in the rest with actual == top-K values.
  // The number that we need is outputSliceSize -
  // writeIndexStart. There might be more than that number available,
  // in which case we have to choose the first seen set. We do this
  // via a prefix sum to calculate indices for writing results.
  // CUDA_KERNEL_ASSERT(outputSliceSize >= writeIndexStart);
  IndexType topKRemaining = (outputSliceSize - writeIndexStart);

  for (IndexType i = threadIdx.x; i < numIterations; i += blockDim.x) {
    bool inRange = (i < inputSliceSize);
    T v =static_cast<T>(0);
    if (inRange) {
      v = doLdg(&inputSliceStart[i * inputWithinSliceStride]);
    }
    const auto convertedV = at::native::TopKTypeConfig<T>::convert(v);
    bool hasTopK = inRange && (convertedV == topKConverted);

    int index;
    int carry;
    at::cuda::exclusiveBinaryPrefixScan<int, true>(
        smem, hasTopK, &index, &carry, AddOp<int>());

    if (hasTopK && index < topKRemaining) {
      int writeIndex = writeIndexStart + index;

      IndexType topKOffset = writeIndex * topKWithinSliceStride;
      IndexType indexOffset = writeIndex * indicesWithinSliceStride;

      topKSliceStart[topKOffset] = v;
      indicesSliceStart[indexOffset] = i;
    }

    if (carry >= topKRemaining) {
      break;
    }

    topKRemaining -= carry;
    writeIndexStart += carry;
  }

};

} // at::native
