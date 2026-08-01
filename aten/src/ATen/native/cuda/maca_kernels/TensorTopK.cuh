#pragma once

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
__global__ void gatherTopK_opt(at::cuda::detail::TensorInfo<const T, IndexType> input,
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
    at::cuda::detail::IndexToOffset<const T, IndexType, Dim>::get(slice, input);
  IndexType topKSliceStartIndex =
    at::cuda::detail::IndexToOffset<T, IndexType, Dim>::get(slice, topK);
  IndexType indicesSliceStartIndex =
    at::cuda::detail::IndexToOffset<int64_t, IndexType, Dim>::get(slice, indices);

  const T* inputSliceStart = &input.data[sliceStartIndex];
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

template<class scalar_t>
__device__ __forceinline__ scalar_t get_weight(const int32_t& v) {
    const scalar_t* idx_and_weight = (const scalar_t*)&v;
    return idx_and_weight[0];
}

template<class scalar_t>
__device__ __forceinline__ void set_weight(int32_t& v, scalar_t& val) {
    scalar_t* idx_and_weight = (scalar_t*)&v;
    idx_and_weight[0] = val;
}

template<class scalar_t, int WARP_SIZE=32, uint64_t MASK=0xffffffff>
__device__ __forceinline__ void warpSortDescending(int32_t& idx_and_weight, int tid) {
    int32_t val = idx_and_weight;
    for (int width = 2; width <= WARP_SIZE; width <<=1) {
        for (int step = width >> 1; step > 0; step >>=1) {
            const bool is_not_final_phase = (width != WARP_SIZE);
            const uint32_t bitmask = (tid & width);
            const bool direction = is_not_final_phase & (bitmask == 0);
            int32_t other_temp_val = __shfl_xor_sync(MASK, val, step);
            int other_tid = tid ^ step;

            scalar_t current_weight_bits = get_weight<scalar_t>(val);
            scalar_t other_weight_bits = get_weight<scalar_t>(other_temp_val);
            int current_index = val >> 16;
            int other_index = other_temp_val >> 16;

            bool weight_gt = other_weight_bits > current_weight_bits;
            bool weight_eq = other_weight_bits == current_weight_bits;
            bool index_lt = other_index < current_index;
            bool cond = (tid < other_tid) ^ direction;

            bool swap = (cond & (weight_gt | (weight_eq & index_lt))) |
                        (!cond & ((other_weight_bits < current_weight_bits) | (weight_eq & (other_index > current_index))));

            val = swap ? other_temp_val : val;
        }
    }
    idx_and_weight = val;
}


template<typename scalar_t, typename index, int WARP, uint64_t MASK>
__global__ void gather_topk_kernel_wave(
  scalar_t* self_ptr,
  scalar_t* values_ptr,
  index*    indices_ptr,
  int64_t k,
  int64_t dim,
  int64_t dim_size,
  bool largest
) {
  self_ptr    = self_ptr    + blockIdx.x * dim_size;
  values_ptr  = values_ptr  + blockIdx.x * k;
  indices_ptr = indices_ptr + blockIdx.x * k;

  int tid = threadIdx.x;
  if (tid >= WARP) return;

  int32_t idx_and_weight = 0;
  set_weight(idx_and_weight, self_ptr[tid]);
  idx_and_weight |= (tid << 16);
  warpSortDescending<scalar_t, WARP, MASK>(idx_and_weight, tid);

  if (tid < k) {
    scalar_t val_i = get_weight<scalar_t>(idx_and_weight);
    int      tid_i = idx_and_weight >> 16;
    values_ptr[tid] = val_i;
    indices_ptr[tid] = tid_i;
  }
}


template<typename scalar_t, typename index, int WARP, uint64_t MASK>
__global__ void gather_topk_kernel_block(
  scalar_t* self_ptr,
  scalar_t* values_ptr,
  index*    indices_ptr,
  int64_t k,
  int64_t dim,
  int64_t dim_size,
  bool largest
) {
  self_ptr    = self_ptr    + blockIdx.x * dim_size;
  values_ptr  = values_ptr  + blockIdx.x * k;
  indices_ptr = indices_ptr + blockIdx.x * k;

  int tid = threadIdx.x;
  int32_t idx_and_weight = 0;
  set_weight(idx_and_weight, self_ptr[tid]);
  idx_and_weight |= (tid << 16);
  warpSortDescending<scalar_t, 64, 0xffffffffffffffff>(idx_and_weight, tid);

  __shared__ int32_t idx_and_weight_all[WARP];
  int warp = tid / C10_WARP_SIZE;
  int tid_in_warp = tid - warp * C10_WARP_SIZE;
  if (tid_in_warp < k) {
    idx_and_weight_all[warp * k + tid_in_warp] = idx_and_weight;
  }
  __syncthreads();

  if (tid < WARP) {
    idx_and_weight = idx_and_weight_all[tid];
    warpSortDescending<scalar_t, WARP, MASK>(idx_and_weight, tid);
    if (tid < k) {
      scalar_t val_i = get_weight<scalar_t>(idx_and_weight);
      int      tid_i = idx_and_weight >> 16;
      values_ptr[tid] = val_i;
      indices_ptr[tid] = tid_i;
    }
  }

}


template<typename scalar_t>
bool launch_gather_topk_kernel_dtype(
    const TensorBase& self, int64_t k, int64_t dim, bool largest,
    const TensorBase& values, const TensorBase& indices) {

  bool is_cond0 = self.numel() > 0 && values.numel() > 0 && indices.numel() > 0 &&
                  self.dim() == values.dim() && self.dim() == indices.dim() &&
                  (dim == self.dim() - 1) && k==values.size(dim) && k==indices.size(dim) && k <= self.size(dim) &&
                  values.is_contiguous() && indices.is_contiguous() && largest &&
                  !at::maca::get_maca_disable_gatherTopK_opt();
  if (!is_cond0) return false;
  bool is_opt = false;

  auto input = self.contiguous();
  int dim_size = self.size(dim);
  int block = ((dim_size + C10_WARP_SIZE - 1) / C10_WARP_SIZE) * C10_WARP_SIZE;
  int grid  = self.numel() / dim_size;

  if (dim_size == 8) {
    is_opt = true;
    gather_topk_kernel_wave<scalar_t, int64_t, 8, 0xff><<<grid, block, 0, c10::cuda::getCurrentCUDAStream()>>>(
      input.data_ptr<scalar_t>(),
      values.data_ptr<scalar_t>(),
      indices.data_ptr<int64_t>(),
      k,
      dim,
      dim_size,
      largest
    );
  } else if (dim_size == 16) {
    is_opt = true;
    gather_topk_kernel_wave<scalar_t, int64_t, 16, 0xffff><<<grid, block, 0, c10::cuda::getCurrentCUDAStream()>>>(
      input.data_ptr<scalar_t>(),
      values.data_ptr<scalar_t>(),
      indices.data_ptr<int64_t>(),
      k,
      dim,
      dim_size,
      largest
    );
  } else if (dim_size == 32) {
    is_opt = true;
    gather_topk_kernel_wave<scalar_t, int64_t, 32, 0xffffffff><<<grid, block, 0, c10::cuda::getCurrentCUDAStream()>>>(
      input.data_ptr<scalar_t>(),
      values.data_ptr<scalar_t>(),
      indices.data_ptr<int64_t>(),
      k,
      dim,
      dim_size,
      largest
    );
  } else if (dim_size == 64) {
    is_opt = true;
    gather_topk_kernel_wave<scalar_t, int64_t, 64, 0xffffffffffffffff><<<grid, block, 0, c10::cuda::getCurrentCUDAStream()>>>(
      input.data_ptr<scalar_t>(),
      values.data_ptr<scalar_t>(),
      indices.data_ptr<int64_t>(),
      k,
      dim,
      dim_size,
      largest
    );
  } else if (dim_size == 128 && k == 2) {
    is_opt = true;
    gather_topk_kernel_block<scalar_t, int64_t, 4, 0xf><<<grid, block, 0, c10::cuda::getCurrentCUDAStream()>>>(
      input.data_ptr<scalar_t>(),
      values.data_ptr<scalar_t>(),
      indices.data_ptr<int64_t>(),
      k,
      dim,
      dim_size,
      largest
    );
  } else if (dim_size == 128 && k == 4) {
    is_opt = true;
    gather_topk_kernel_block<scalar_t, int64_t, 8, 0xff><<<grid, block, 0, c10::cuda::getCurrentCUDAStream()>>>(
      input.data_ptr<scalar_t>(),
      values.data_ptr<scalar_t>(),
      indices.data_ptr<int64_t>(),
      k,
      dim,
      dim_size,
      largest
    );
  } else if (dim_size == 128 && k == 8) {
    is_opt = true;
    gather_topk_kernel_block<scalar_t, int64_t, 16, 0xffff><<<grid, block, 0, c10::cuda::getCurrentCUDAStream()>>>(
      input.data_ptr<scalar_t>(),
      values.data_ptr<scalar_t>(),
      indices.data_ptr<int64_t>(),
      k,
      dim,
      dim_size,
      largest
    );
  } else if (dim_size == 256 && k == 2) {
    is_opt = true;
    gather_topk_kernel_block<scalar_t, int64_t, 8, 0xff><<<grid, block, 0, c10::cuda::getCurrentCUDAStream()>>>(
      input.data_ptr<scalar_t>(),
      values.data_ptr<scalar_t>(),
      indices.data_ptr<int64_t>(),
      k,
      dim,
      dim_size,
      largest
    );
  } else if (dim_size == 256 && k == 4) {
    is_opt = true;
    gather_topk_kernel_block<scalar_t, int64_t, 16, 0xffff><<<grid, block, 0, c10::cuda::getCurrentCUDAStream()>>>(
      input.data_ptr<scalar_t>(),
      values.data_ptr<scalar_t>(),
      indices.data_ptr<int64_t>(),
      k,
      dim,
      dim_size,
      largest
    );
  } else if (dim_size == 256 && k == 8) {
    is_opt = true;
    gather_topk_kernel_block<scalar_t, int64_t, 32, 0xffffffff><<<grid, block, 0, c10::cuda::getCurrentCUDAStream()>>>(
      input.data_ptr<scalar_t>(),
      values.data_ptr<scalar_t>(),
      indices.data_ptr<int64_t>(),
      k,
      dim,
      dim_size,
      largest
    );
  }

  return is_opt;

}


bool launch_gather_topk_kernel_opt(
    const TensorBase& self, int64_t k, int64_t dim, bool largest,
    const TensorBase& values, const TensorBase& indices) {

  bool is_opt_half = false;
  bool is_opt_bhalf = false;
  if (self.scalar_type() == ScalarType::Half) {
    is_opt_half = launch_gather_topk_kernel_dtype<at::Half>(self, k, dim, largest, values, indices);
  } else if (self.scalar_type() == ScalarType::BFloat16) {
    is_opt_bhalf = launch_gather_topk_kernel_dtype<at::BFloat16>(self, k, dim, largest, values, indices);
  }

  return is_opt_half || is_opt_bhalf;
}

} // at::native
