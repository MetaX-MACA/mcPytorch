#pragma once

#include <thrust/tuple.h>

#include <ATen/native/SharedReduceOps.h>
#include <ATen/cuda/DeviceUtils.cuh>

namespace at {
namespace native {
namespace cuda_utils {

constexpr int kCUDABlockReduceNumThreads = 512;
// Algorithmic limitation: BlockReduce does two WarpReduce calls, each
// of which reduces C10_WARP_SIZE elements. So, at most
// C10_WARP_SIZE**2 elements can be reduced at a time.
// NOTE: This is >= the max block size on current hardware anyway (1024).
constexpr int kCUDABlockReduceMaxThreads = C10_WARP_SIZE * C10_WARP_SIZE;

// Sums `val` across all threads in a warp.
//
// Assumptions:
//   - The size of each block should be a multiple of `C10_WARP_SIZE`
template <typename T>
__inline__ __device__ T WarpReduceSum(T val) {
#pragma unroll
  for (int offset = (C10_WARP_SIZE >> 1); offset > 0; offset >>= 1) {
    val += WARP_SHFL_DOWN(val, offset);
  }
  return val;
}

#ifdef USE_MACA
template <typename T>
__inline__ __device__ T WarpXReduceSum(T val) {
  for (int offset = (blockDim.x >> 1); offset > 0; offset >>= 1) {
    val += WARP_SHFL_DOWN(val, offset);
  }
  return val;
}
#endif

// Picks the maximum `val` across all threads in a warp.
//
// Assumptions:
//   - The size of each block should be a multiple of `C10_WARP_SIZE`
template <typename T>
__inline__ __device__ T WarpReduceMax(T val) {
#pragma unroll
  for (int offset = (C10_WARP_SIZE >> 1); offset > 0; offset >>= 1) {
    val = max_propagate_nan(val, WARP_SHFL_DOWN(val, offset));
  }
  return val;
}

struct Block1D {
    static __forceinline__ __device__ int Tid() { return threadIdx.x; }

    static __forceinline__ __device__ int Warps() {
        return blockDim.x / C10_WARP_SIZE;
    }
};

struct Block2D {
    static __forceinline__ __device__ int Tid() {
        return threadIdx.x + threadIdx.y * blockDim.x;
    }

    static __forceinline__ __device__ int Warps() {
        return blockDim.x * blockDim.y / C10_WARP_SIZE;
    }
};


#ifdef USE_MACA
template <typename T>
__inline__ __device__ T BlockXReduceSum(T val, T* shared) {
  const int tid = threadIdx.x;
  const int yid = threadIdx.y;
  const int lid = tid % C10_WARP_SIZE;
  const int wid = tid / C10_WARP_SIZE;
  const int x_warps = blockDim.x / C10_WARP_SIZE;

  const int sid = yid * x_warps + wid;
  val = WarpReduceSum(val);
  __syncthreads(); // prevent races when BlockReduces are called in a row.
  if (lid == 0) {
    shared[sid] = val;
  }
  __syncthreads();
  const int srid = yid * x_warps + lid;
  val = (tid < x_warps) ? shared[srid] : T(0);
  if (wid == 0) {
    val = WarpReduceSum(val);
  }
  return val;
}
#else
static_assert(0);
#endif

// Sums `val` across all threads in a block.
//
// Warning: the return value is only valid for thread 0.
// Assumptions:
//   - The size of each block should be a multiple of `C10_WARP_SIZE`
//   - `shared` should be a pointer to shared memory with size of, at least,
//     `sizeof(T) * number_of_warps`
template <typename T, typename B = Block1D>
__inline__ __device__ T BlockReduceSum(T val, T* shared) {
  const int tid = B::Tid();
  const int lid = tid % C10_WARP_SIZE;
  const int wid = tid / C10_WARP_SIZE;
  val = WarpReduceSum(val);
  __syncthreads(); // prevent races when BlockReduces are called in a row.
  if (lid == 0) {
    shared[wid] = val;
  }
  __syncthreads();
  val = (tid < B::Warps()) ? shared[lid] : T(0);
  if (wid == 0) {
    val = WarpReduceSum(val);
  }
  return val;
}

// Picks out the maximum `val` across all threads in a block.
//
// Warning: the return value is only valid for thread 0.
// Assumptions:
//   - The size of each block should be a multiple of `C10_WARP_SIZE`
//   - `shared` should be a pointer to shared memory with size of, at least,
//     `sizeof(T) * number_of_warps`
template <typename T, typename B = Block1D>
__inline__ __device__ T BlockReduceMax(T val, T* shared) {
  const int tid = B::Tid();
  const int lid = tid % C10_WARP_SIZE;
  const int wid = tid / C10_WARP_SIZE;
  val = WarpReduceMax(val);
  __syncthreads(); // prevent races when BlockReduces are called in a row.
  if (lid == 0) {
    shared[wid] = val;
  }
  __syncthreads();
  val = (tid < B::Warps()) ? shared[lid] : T(std::numeric_limits<T>::lowest());
  if (wid == 0) {
    val = WarpReduceMax(val);
  }
  return val;
}

template <typename T, class ReduceOp>
__inline__ __device__ T WarpReduce(T val, const ReduceOp& op) {
#pragma unroll
  for (int offset = (C10_WARP_SIZE >> 1); offset > 0; offset >>= 1) {
    val = op.combine(val, op.warp_shfl_down(val, offset));
  }
  return val;
}

#ifdef USE_MACA
template <typename T, class ReduceOp>
__inline__ __device__ T WarpXReduce(T val, const ReduceOp& op) {
  for (int offset = (blockDim.x >> 1); offset > 0; offset >>= 1) {
    val = op.combine(val, op.warp_shfl_down(val, offset));
  }
  return val;
}
#endif

template <typename T, class ReduceOp, typename B = Block1D>
__inline__ __device__ T
BlockReduce(T val, const ReduceOp& op, const T& identity_element, T* shared) {
  const int tid = B::Tid();
  const int lid = tid % C10_WARP_SIZE;
  const int wid = tid / C10_WARP_SIZE;
  val = WarpReduce(val, op);
  __syncthreads(); // prevent races when BlockReduces are called in a row.
  if (lid == 0) {
    shared[wid] = val;
  }
  __syncthreads();
  val = (tid < B::Warps()) ? shared[lid] : identity_element;
  if (wid == 0) {
    val = WarpReduce(val, op);
  }
  return val;
}

#ifdef USE_MACA
template <typename T, class ReduceOp>
__inline__ __device__ T
BlockXReduce(T val, const ReduceOp& op, const T& identity_element, T* shared) {
  const int tid = threadIdx.x;
  const int lid = tid % C10_WARP_SIZE;
  const int wid = tid / C10_WARP_SIZE;
  const int tyid = threadIdx.y;
  const int wsize = blockDim.x / C10_WARP_SIZE;
  val = WarpReduce(val, op);
  __syncthreads(); // prevent races when BlockReduces are called in a row.
  if (lid == 0) {
    shared[tyid * wsize + wid] = val;
  }
  __syncthreads();
  val = (tid < wsize) ? shared[tyid * wsize + lid] : identity_element;
  if (wid == 0) {
    val = WarpReduce(val, op);
  }
  return val;
}
#endif

} // namespace cuda_utils
} // namespace native
} // namespace at
