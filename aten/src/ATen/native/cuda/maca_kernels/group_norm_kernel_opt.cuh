#pragma once
// #include <ATen/native/cuda/block_reduce.cuh>
#include <ATen/AccumulateType.h>
// #include <ATen/native/SharedReduceOps.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/empty.h>
#endif

namespace at::native {

// namespace group_norm{

constexpr int kVecSize8 = 8;
size_t get_vec_size(size_t DxHxW, size_t element_size) {
  constexpr int ldg128_bytes = 16;
  int vec_size = ldg128_bytes / element_size;
  vec_size = std::max(std::min(vec_size, 8), 1);
  auto update_vec_size = [&vec_size](uint64_t n) {
    while(vec_size > 1 && n % vec_size != 0) {
      vec_size /= 2;
    }
  };
  update_vec_size(DxHxW);
  return vec_size;
}

size_t get_forward_fused_kernel_block_size(size_t DxHxW, size_t vec_size) {
  size_t block_size = C10_WARP_SIZE;
  while (block_size < 512 && block_size * vec_size * 2 < DxHxW) {
    block_size <<= 1;
  }
  return block_size;
}

dim3 get_backward_fused_kernel_block_size(size_t D, size_t HxW, size_t vec_size) {
  size_t block_x_size = C10_WARP_SIZE;
  size_t block_y_size = D;
  while (block_y_size * block_x_size < 512 && block_x_size * vec_size * 2 < HxW) {
    block_x_size <<= 1;
  }
  return dim3(block_x_size, block_y_size);
}

size_t get_forward_fused_kernel_shared_memory_size(size_t block_size, size_t D, size_t welford_element_size, size_t t_acc_size) {
   size_t reduce_smem_size = block_size / C10_WARP_SIZE * welford_element_size;
   size_t ab_smem_size = D * t_acc_size * 2;
   return std::max(reduce_smem_size, ab_smem_size);
}

size_t get_backward_fused_kernel_shared_memory_size(size_t D, size_t acc_elem_size) {
  return D * 2 * acc_elem_size;
}

bool backward_match_fused_kernel(size_t  D, size_t HxW, size_t element_size) {
  // 1. HxW reduce thread nums only support over 1 warp.
  size_t expected_vec_size = get_vec_size(HxW, element_size);
  bool cond1 = C10_WARP_SIZE * expected_vec_size <= HxW;
  // 2. block thread num should less than 512.
  dim3 expected_block_size = get_backward_fused_kernel_block_size(D, HxW, expected_vec_size);
  bool cond2 = expected_block_size.x * expected_block_size.y <= 512;
  return cond1 && cond2;
}

// returns 2^floor(log2(n))
static inline int last_pow2(int n) {
  n |= (n >> 1);
  n |= (n >> 2);
  n |= (n >> 4);
  n |= (n >> 8);
  n |= (n >> 16);
  return std::max(1, n - (n >> 1));
}

dim3 get_vectorized_rowwise_moment_kernel_block_size(size_t NxG, size_t DxHxW, size_t vec_size) {
  uint32_t dim0 = DxHxW / vec_size, dim1 = NxG;
  const uint32_t max_num_threads = 512;
  uint32_t dim0_pow2 = dim0 < max_num_threads ? static_cast<uint32_t>(last_pow2(dim0)) : max_num_threads;
  uint32_t dim1_pow2 = dim1 < max_num_threads ? static_cast<uint32_t>(last_pow2(dim1)) : max_num_threads;
  uint32_t block_width = dim0_pow2;
  uint32_t block_height = std::min(dim1_pow2, uint32_t(max_num_threads / block_width));
  block_width = std::min(dim0_pow2, uint32_t(max_num_threads / block_height));
  return {block_width, block_height};
}

template <typename T, class ReduceOp, typename B = cuda_utils::Block1D>
__device__ __forceinline__ T
BlockReduceSize256(T val, const ReduceOp& op, const T& identity_element, T* shared) {
  const int tid = B::Tid();
  const int lid = tid % C10_WARP_SIZE;
  const int wid = tid / C10_WARP_SIZE;
  val = cuda_utils::WarpReduce(val, op);
  __syncthreads(); // prevent races when BlockReduces are called in a row.
  if (lid == 0) {
    shared[wid] = val;
  }
  __syncthreads();
  if (tid < 8) {
  #pragma unroll
      for (int offset = 8; offset > 0; offset >>= 1) {
        if (tid < offset) {
          shared[tid] = op.combine(shared[tid], shared[tid + offset]);
        }
      }
  }
  return shared[0];
}

template <typename T, class ReduceOp, typename B = cuda_utils::Block1D>
__device__ __forceinline__ T
BlockReduceSize8(T val, const ReduceOp& op, const T& identity_element, T* shared) {
  const int tid = B::Tid();
  const int lid = tid % C10_WARP_SIZE;
  const int wid = tid / C10_WARP_SIZE;
  val = cuda_utils::WarpReduce(val, op);
  __syncthreads(); // prevent races when BlockReduces are called in a row.
  if (lid == 0) {
    shared[wid] = val;
  }
  __syncthreads();
  if (tid < 4) {
  #pragma unroll
      for (int offset = 4; offset > 0; offset >>= 1) {
        if (tid < offset) {
          shared[tid] = op.combine(shared[tid], shared[tid + offset]);
        }
      }
  }
  __syncthreads();
  return shared[0];
}

template <typename T, typename T_ACC>
__device__ __forceinline__ void VectorizedComputeResult(
  int64_t N,
  int64_t C,
  int64_t G,
  int64_t HxW,
  T eps,
  const T* X,
  const T* gamma,
  const T* beta,
  T* Y,
  T_ACC* a,
  T_ACC* b,
  T mean,
  T rsqrt
) {
  const int64_t D = C / G;
  const int64_t n = blockIdx.x / G;
  const int channel_offset = blockIdx.x % G * D;
  const int64_t i = blockIdx.x;
  if (threadIdx.x < D) {
    const int64_t compute_c = channel_offset  + threadIdx.x;
    T_ACC scale = (gamma == nullptr)
        ? static_cast<T_ACC>(rsqrt)
        : static_cast<T_ACC>(rsqrt) * static_cast<T_ACC>(gamma[compute_c]);
    a[threadIdx.x] = scale;
    b[threadIdx.x] = -scale * static_cast<T_ACC>(mean) +
        ((beta == nullptr) ? 0 : static_cast<T_ACC>(beta[compute_c]));    
  }

  __syncthreads();
  const int hxw_step = HxW / kVecSize8;
  const int x_offset0 = (n *C + channel_offset) * hxw_step;
  using T_VEC8 = memory::aligned_vector<T, kVecSize8>;
  const T_VEC8* x_vec8 = reinterpret_cast<const T_VEC8*>(X);
  T_VEC8* y_vec8 = reinterpret_cast<T_VEC8*>(Y);

  // before using, make sure there are no vec walk across different img
  const int64_t block_data = C / G * HxW;
  const int64_t n_vec_to_read = block_data / kVecSize8;
  if (hxw_step > blockDim.x) {
    for (int64_t j = threadIdx.x; j < n_vec_to_read; j += blockDim.x) {
      const int64_t index = i * n_vec_to_read + j;
      T_VEC8 x_data = x_vec8[index];
      T_VEC8 y_data;
      const int64_t d = j / hxw_step;
      T_ACC a_data = a[d];
      T_ACC b_data = b[d];
  #pragma unroll
      for (int jj = 0; jj < kVecSize8; jj++) {
        y_data.val[jj] = a_data * static_cast<T_ACC>(x_data.val[jj]) + b_data;
      }
      y_vec8[index] = y_data;
    }   
    return; 
  }

  if (threadIdx.x < hxw_step) {
#pragma unroll
    for (int ii = 0; ii < D; ii++) {
      T_ACC a_data = a[ii];
      T_ACC b_data = b[ii];
      const int x_offset1 = x_offset0 + ii * hxw_step + threadIdx.x ;
      T_VEC8 x_data = x_vec8[x_offset1];
      T_VEC8 y_data;
      #pragma unroll
      for (int jj = 0; jj < kVecSize8; jj++) {
        y_data.val[jj] = a_data * static_cast<T_ACC>(x_data.val[jj]) + b_data;
      }
      y_vec8[x_offset1] = y_data;
    }
  }
}

template <typename T>
C10_LAUNCH_BOUNDS_1(1024)
__global__ void block40960VectorizedRowwiseMomentsCUDAKernel(
    int64_t N,
    int64_t C,
    int64_t G,
    int64_t HxW,
    T eps,
    const T* X,
    const T* gamma,
    const T* beta,
    T* Y,
    bool fuse_result,
    T* mean_data,
    T* rsqrt_data) {
    using T_ACC = acc_type<T, true>;
    using T_VEC8 = memory::aligned_vector<T, kVecSize8>;
    using WelfordType = WelfordData<T_ACC, int64_t>;
    using WelfordOp =
        WelfordOps<T_ACC, T_ACC, int64_t, thrust::pair<T_ACC, T_ACC>>;
    const int64_t i = blockIdx.x;
    const int64_t j = threadIdx.x;
    const int64_t index1 = i * blockDim.x + j;
    const int64_t index2 = index1 * 5;

    WelfordOp welford_op = {/*correction=*/0, /*take_sqrt=*/false};
    WelfordType val(0, 0, 0, 0);
    const T_VEC8* X_vec8 = reinterpret_cast<const T_VEC8*>(X);
    #pragma unroll
    for (int ii = 0; ii < 5; ii++) {
    T_VEC8 data = X_vec8[index2 + ii];
    #pragma unroll
    for (int jj = 0; jj < kVecSize8; jj++) {
        val = welford_op.reduce(
            val, static_cast<T_ACC>(data.val[jj]), 0);
    }
    }

    // There will be a warning if we declare a __shared__ WelfordType array.
    // https://github.com/pytorch/pytorch/pull/13967
    __shared__ typename std::aligned_storage<
        sizeof(WelfordType),
        alignof(WelfordType)>::type val_shared[C10_WARP_SIZE];
    WelfordType* val_shared_ptr = reinterpret_cast<WelfordType*>(val_shared);
    val = BlockReduceSize256(
        val,
        welford_op,
        /*identity_element=*/WelfordType(0, 0, 0, 0),
        val_shared_ptr);
    __shared__ T mean;
    __shared__ T rsqrt;
    if (threadIdx.x == 0) {
    T_ACC m1;
    T_ACC m2;
    thrust::tie(m2, m1) = welford_op.project(val);
    mean_data[i] =  mean = m1;
    rsqrt_data[i] = rsqrt = rsqrtf(m2 + static_cast<T_ACC>(eps));
    }
    __syncthreads();

    if (fuse_result) {
    const int64_t D = C / G;
    extern __shared__ char shared_memory[];
    T_ACC *a = reinterpret_cast<T_ACC *>(shared_memory);
    T_ACC *b = a + D;
    VectorizedComputeResult(N, C, G, HxW, eps, X, gamma, beta, Y, a, b, mean, rsqrt);
    }
}

template <typename scalar_t, typename acc_scalar_t, typename index_t, typename res_t>
struct WelfordOpsOpt {
  index_t correction;
  bool take_sqrt;
 public:
  using acc_t = WelfordData<acc_scalar_t, index_t>;
  inline C10_DEVICE acc_t reduce(acc_t acc, scalar_t data, index_t /*idx*/) const {
    // We accumulate n in index_t to avoid cumulative rounding error, but still
    // need nf for use in combine where int32 may overflow.
    index_t new_n = acc.n + 1;
    acc_scalar_t new_nf = static_cast<acc_scalar_t>(new_n);
    acc_scalar_t delta = data - acc.mean;
    acc_scalar_t rcp_new_nf = __builtin_mxc_rcpf(new_nf);
    acc_scalar_t new_mean = acc.mean + delta * rcp_new_nf;
    acc_scalar_t new_delta = data - new_mean;
    return {
      new_mean,
      acc.m2 + delta * new_delta,
      new_n,
      new_nf,
    };
  }
  inline C10_DEVICE acc_t combine(acc_t a, acc_t b) const {
    if (a.nf == 0) {
      return b;
    }
    if (b.nf == 0) {
      return a;
    }
    acc_scalar_t delta = b.mean - a.mean;
    acc_scalar_t new_count = a.nf + b.nf;
    acc_scalar_t rcp_new_count = __builtin_mxc_rcpf(new_count);
    acc_scalar_t nb_over_n = b.nf * rcp_new_count;
    return {
      a.mean + delta * nb_over_n,
      a.m2 + b.m2 + delta * delta * a.nf * nb_over_n,
      // setting acc.n as -1 since acc.n might not be able to represent the count
      // correctly within its range, setting it to -1 to avoid confusion
      -1,
      new_count
    };
  }
  inline C10_DEVICE res_t project(acc_t acc) const __ubsan_ignore_float_divide_by_zero__ {
    const auto mean = static_cast<scalar_t>(acc.mean);
    const auto divisor = acc.nf > correction ? acc.nf - correction : 0;
    acc_scalar_t rcp_divisor = __builtin_mxc_rcpf(divisor);
    const auto var = acc.m2 * rcp_divisor;
    res_t results(take_sqrt ? __builtin_mxc_sqrtf(var) : var, mean);
    return results;
  }

  static C10_DEVICE acc_t translate_idx(acc_t acc, int64_t /*base_idx*/) {
    return acc;
  }

#if defined(__CUDACC__) || defined(__HIPCC__)
  inline __device__ acc_t warp_shfl_down(acc_t acc, int offset) const {
    return {
      WARP_SHFL_DOWN(acc.mean, offset)
      , WARP_SHFL_DOWN(acc.m2, offset)
      , WARP_SHFL_DOWN(acc.n, offset)
      , WARP_SHFL_DOWN(acc.nf, offset)
    };
  }
#endif
  C10_HOST_DEVICE WelfordOpsOpt(index_t correction, bool take_sqrt)
      : correction(correction), take_sqrt(take_sqrt) {}
};

template <typename T, int64_t D, int64_t n_vec_load>
C10_LAUNCH_BOUNDS_1(512)
__global__ void block512VectorizedRowwiseMomentsCUDAKernel(
    int64_t N,
    int64_t C,
    int64_t G,
    int64_t HxW,
    T eps,
    const T* X,
    const T* gamma,
    const T* beta,
    T* Y,
    bool fuse_result,
    T* mean_data,
    T* rsqrt_data) {
  // block = 512, grid = 64;
  // D = n_vec_load = 10;
  using T_ACC = acc_type<T, true>;
  using T_VEC8 = memory::aligned_vector<T, kVecSize8>;
  using WelfordType = WelfordData<T_ACC, int64_t>;
  using WelfordOp =
      WelfordOpsOpt<T_ACC, T_ACC, int64_t, thrust::pair<T_ACC, T_ACC>>;
  const int64_t i = blockIdx.x;
  const int64_t j = threadIdx.x;
  const int64_t index = i * (blockDim.x * n_vec_load) + j;
  const int64_t step = HxW / kVecSize8;

  WelfordOp welford_op = {/*correction=*/0, /*take_sqrt=*/false};
  WelfordType val(0, 0, 0, 0);
  const T_VEC8* X_vec8 = reinterpret_cast<const T_VEC8*>(X);
  T_VEC8 x_vec[n_vec_load];
#pragma unroll
  for (int ii = 0; ii < n_vec_load; ii++) {
    x_vec[ii] = X_vec8[index + ii * step];
#pragma unroll
    for (int jj = 0; jj < kVecSize8; jj++) {
      val = welford_op.reduce(
          val, static_cast<T_ACC>(x_vec[ii].val[jj]), 0);
    }
  }

  // There will be a warning if we declare a __shared__ WelfordType array.
  // https://github.com/pytorch/pytorch/pull/13967
  extern __shared__ unsigned char val_shared[]; // warpnum * sizeof(WelfordType) warpnum=8;
  WelfordType* val_shared_ptr = reinterpret_cast<WelfordType*>(val_shared);
  val = BlockReduceSize8(
      val,
      welford_op,
      /*identity_element=*/WelfordType(0, 0, 0, 0),
      val_shared_ptr);
  T mean;
  T rsqrt;
  thrust::tie(rsqrt, mean) = welford_op.project(val);
  rsqrt = rsqrtf(rsqrt + static_cast<T_ACC>(eps));
  if (threadIdx.x == 0) {
    mean_data[i] = mean;
    rsqrt_data[i] = rsqrt;
  }
  if (fuse_result) {
    const int channel_offset = blockIdx.x % G * D;
    T_ACC a[D];
    T_ACC b[D];
#pragma unroll
    for (int d_idx = 0; d_idx < D; d_idx++) {
        const int64_t compute_c = channel_offset + d_idx;
        T_ACC scale = (gamma == nullptr)
            ? static_cast<T_ACC>(rsqrt)
            : static_cast<T_ACC>(rsqrt) * static_cast<T_ACC>(gamma[compute_c]);
        a[d_idx] = scale;
        b[d_idx] = -scale * static_cast<T_ACC>(mean) +
            ((beta == nullptr) ? 0 : static_cast<T_ACC>(beta[compute_c]));    
    }
    T_VEC8* p_y = reinterpret_cast<T_VEC8*>(Y);
#pragma unroll
    for (int ii = 0; ii < n_vec_load; ii++) {
      T_VEC8 y_vec;
#pragma unroll
      for (int jj = 0; jj < kVecSize8; jj++) {
          y_vec.val[jj] = a[ii] * static_cast<T_ACC>(x_vec[ii].val[jj]) + b[ii];
      }
      p_y[index + ii * step] = y_vec;
    }
  }
}

template <typename T, size_t vec_size>
__global__ void VectorizedRowwiseMomentsCUDAKernel(
    int64_t NxG,
    int64_t DxHxW,
    T eps,
    const T* X,
    T* mean,
    T* rstd) {
  using T_ACC = acc_type<T, true>;
  using T_VEC = memory::aligned_vector<T, vec_size>;
  using WelfordType = WelfordData<T_ACC, int64_t>;
  using WelfordOp =
      WelfordOps<T_ACC, T_ACC, int64_t, thrust::pair<T_ACC, T_ACC>>;
  const int64_t n_vec_to_read = DxHxW / vec_size;

  WelfordOp welford_op = {/*correction=*/0, /*take_sqrt=*/false};
  WelfordType val(0, 0, 0, 0);
  const int output_idx = blockIdx.x * blockDim.y + threadIdx.y;
  if (output_idx >= NxG) {
    return;
  }
  const int64_t row_offset =  DxHxW * output_idx;
  const T_VEC* X_vec = reinterpret_cast<const T_VEC*>(X + row_offset);
  for (int64_t i = threadIdx.x; i < n_vec_to_read; i += blockDim.x) {
    T_VEC data = X_vec[i];
#pragma unroll
    for (int j = 0; j < vec_size; j++) {
      val = welford_op.reduce(
          val, static_cast<T_ACC>(data.val[j]), i * vec_size + j);
    }
  }
  if (blockDim.x <= C10_WARP_SIZE) {
    val = cuda_utils::WarpXReduce(val, welford_op);
  } else {
    // There will be a warning if we declare a __shared__ WelfordType array.
    // https://github.com/pytorch/pytorch/pull/13967
    __shared__ typename std::aligned_storage<
        sizeof(WelfordType),
        alignof(WelfordType)>::type val_shared[C10_WARP_SIZE];
    WelfordType* val_shared_ptr = reinterpret_cast<WelfordType*>(val_shared);
    val = cuda_utils::BlockXReduce(
        val,
        welford_op,
        /*identity_element=*/WelfordType(0, 0, 0, 0),
        val_shared_ptr);
  }
  if (threadIdx.x == 0) {
    T_ACC m1;
    T_ACC m2;
    thrust::tie(m2, m1) = welford_op.project(val);
    mean[output_idx] = m1;
    rstd[output_idx] = c10::cuda::compat::rsqrt(m2 + static_cast<T_ACC>(eps));
  }
}

template <typename T, size_t vt0, size_t vec_size>
C10_LAUNCH_BOUNDS_1(512)
__global__ void FusedGroupNormCUDAKernel(
    int64_t N,
    int64_t C,
    int64_t G,
    int64_t HxW,
    T eps,
    const T* X,
    const T* gamma,
    const T* beta,
    T* Y,
    T* mean_data,
    T* rsqrt_data) {
  using T_ACC = acc_type<T, true>;
  using T_VEC = memory::aligned_vector<T, vec_size>;
  using WelfordType = WelfordData<T_ACC, int64_t>; 
  using WelfordOp =
      WelfordOpsOpt<T_ACC, T_ACC, int64_t, thrust::pair<T_ACC, T_ACC>>;
  const int64_t D = C / G;
  const int64_t reduce_size = D * HxW;
  const size_t block_idx = blockIdx.x;
  const T_VEC *input = reinterpret_cast<const T_VEC *>(X + block_idx * reduce_size);
  const int64_t n_vec_to_read = reduce_size / vec_size;
  size_t read_idx = threadIdx.x;
  size_t read_step = blockDim.x;
  WelfordType val(0, 0, 0, 0);
  WelfordOp welford_op = {/*correction=*/0, /*take_sqrt=*/false};
  while (read_idx + (vt0 - 1) * read_step < n_vec_to_read) {
    T_VEC x_vec[vt0];
  #pragma unroll
    for (size_t i = 0; i < vt0; i++) {
      x_vec[i] = input[read_idx];
    #pragma unroll
      for (size_t j = 0; j < vec_size; j++) {
        val = welford_op.reduce(
          val, static_cast<T_ACC>(x_vec[i].val[j]), 0);
      }
      read_idx += read_step;
    }
  }
  // process remainers
  while (read_idx < n_vec_to_read) {
    T_VEC x_vec = input[read_idx];
  #pragma unroll
    for (size_t j = 0; j < vec_size; j++) {
      val = welford_op.reduce(
        val, static_cast<T_ACC>(x_vec.val[j]), 0);
    }
    read_idx += read_step;  
  }

  // There will be a warning if we declare a __shared__ WelfordType array.
  // https://github.com/pytorch/pytorch/pull/13967
  extern __shared__ unsigned char val_shared[]; // warpnum * sizeof(WelfordType) warpnum=8;
  WelfordType* val_shared_ptr = reinterpret_cast<WelfordType*>(val_shared);
  val = cuda_utils::BlockReduce(
      val,
      welford_op,
      /*identity_element=*/WelfordType(0, 0, 0, 0),
      val_shared_ptr);

  T mean;
  T rsqrt;
  thrust::tie(rsqrt, mean) = welford_op.project(val);
  rsqrt = rsqrtf(rsqrt + static_cast<T_ACC>(eps));
  const int channel_offset = block_idx % G * D;
  T_ACC *a = reinterpret_cast<T_ACC*>(val_shared);
  T_ACC *b = a + D;
  if (threadIdx.x == 0) {
    mean_data[block_idx] = mean;
    rsqrt_data[block_idx] = rsqrt;

    for (int d_idx = 0; d_idx < D; d_idx++) {
        const int64_t compute_c = channel_offset + d_idx;
        T_ACC scale = (gamma == nullptr)
            ? static_cast<T_ACC>(rsqrt)
            : static_cast<T_ACC>(rsqrt) * static_cast<T_ACC>(gamma[compute_c]);
        a[d_idx] = scale;
        b[d_idx] = -scale * static_cast<T_ACC>(mean) +
            ((beta == nullptr) ? 0 : static_cast<T_ACC>(beta[compute_c]));    
    }
  }
  __syncthreads();

  // compute result
  read_idx = threadIdx.x;
  const size_t n_vec_hxw = HxW / vec_size;
  size_t d_idx = read_idx / n_vec_hxw;
  size_t hxw_idx = read_idx % n_vec_hxw;
  const size_t read_step_cross_hxw_nums = read_step / n_vec_hxw;
  const size_t read_step_cross_hxw_remainers = read_step % n_vec_hxw;
  T_VEC *output = reinterpret_cast<T_VEC *>(Y + block_idx * reduce_size);
  while (read_idx + (vt0 - 1) * read_step < n_vec_to_read) {
    T_VEC x_vec[vt0];
  #pragma unroll
    for (size_t i = 0; i < vt0; i++) {
      x_vec[i] = input[read_idx];
      T_VEC y_vec;
    #pragma unroll
      for (size_t j = 0; j < vec_size; j++) {
        y_vec.val[j] = a[d_idx] * static_cast<T_ACC>(x_vec[i].val[j]) + b[d_idx];
      }
      output[read_idx] = y_vec;
      read_idx += read_step;
      d_idx += read_step_cross_hxw_nums;
      hxw_idx += read_step_cross_hxw_remainers;
      if (hxw_idx >= n_vec_hxw) {
        d_idx++;
        hxw_idx-=n_vec_hxw;
      }
    }
  }
  // process remainers
  while (read_idx < n_vec_to_read) {
    T_VEC x_vec = input[read_idx];
    T_VEC y_vec;
  #pragma unroll
    for (size_t j = 0; j < vec_size; j++) {
      y_vec.val[j] = a[d_idx] * static_cast<T_ACC>(x_vec.val[j]) + b[d_idx];
    }
    output[read_idx] = y_vec;
    read_idx += read_step;
    d_idx += read_step_cross_hxw_nums;
    hxw_idx += read_step_cross_hxw_remainers;
    if (hxw_idx >= n_vec_hxw) {
        d_idx++;
        hxw_idx-=n_vec_hxw;
    } 
  }
}

template <typename T, int vec_size>
__global__ void FusedGroupNormBackwardKernel(
    int64_t N,
    int64_t C,
    int64_t group,
    int64_t HxW,
    const T* dY,
    const T* X,
    const T* mean,
    const T* rstd,
    const T* gamma,
    acc_type<T, true>* ds_g,
    acc_type<T, true>* db_g,
    acc_type<T, true>* c1_g,
    acc_type<T, true>* c2_g,
    acc_type<T, true>* c3_g,
    T* dx) {
  // step1: ComputeInternalGradientsCUDAKernel
  //         N * C * HxW => N * C reduce along block_x
  const size_t D = blockDim.y;
  const size_t n_vec_hxw = HxW / vec_size;
  const size_t vec_hxw_idx = threadIdx.x;
  const size_t d = threadIdx.y;
  const size_t n = blockIdx.y;
  const size_t g = blockIdx.x;
  size_t c = g * D + d;
  const size_t ng = n * group + g;
  const size_t nc = n * C + c;
  using T_ACC = acc_type<T, true>;
  using T_VEC = memory::aligned_vector<T, vec_size>;
  T_ACC sum1 = 0;
  T_ACC sum2 = 0;
  const T_VEC *dy_nc = reinterpret_cast<const T_VEC *>(dY + nc * HxW);
  const T_VEC *x_nc = reinterpret_cast<const T_VEC *>(X + nc * HxW);
  for (size_t i = vec_hxw_idx; i < n_vec_hxw; i+=blockDim.x) {
    const T_VEC dy_vec = dy_nc[i];
    const T_VEC x_vec = x_nc[i];
    #pragma unroll
    for (size_t j = 0; j < vec_size; j++) {
      sum1 += static_cast<T_ACC>(dy_vec.val[j]) * static_cast<T_ACC>(x_vec.val[j]);
      sum2 += static_cast<T_ACC>(dy_vec.val[j]);
    }
  }
  if (blockDim.x <= C10_WARP_SIZE) {
    // todo: WarpReduceSum don't support blockDim.x < C10_WARP_SIZE actually
    sum1 = cuda_utils::WarpReduceSum<T_ACC>(sum1);
    sum2 = cuda_utils::WarpReduceSum<T_ACC>(sum2);
  } else {
    __shared__ T_ACC ds_shared[C10_WARP_SIZE*8];
    __shared__ T_ACC db_shared[C10_WARP_SIZE*8];
    sum1 = cuda_utils::BlockXReduceSum<T_ACC>(sum1, ds_shared);
    sum2 = cuda_utils::BlockXReduceSum<T_ACC>(sum2, db_shared);
  }
  extern __shared__ unsigned char shared_memory[]; // warpnum * sizeof(WelfordType) warpnum=8;
  T_ACC* ds = reinterpret_cast<T_ACC*>(shared_memory);
  T_ACC* db = reinterpret_cast<T_ACC*>(shared_memory) + D;
  if (threadIdx.x == 0) {
    ds[d] = sum1;
    db[d] = sum2;
    ds_g[nc] = sum1;
    db_g[nc] = sum2;
  }

  // step 2: Compute C1
  const T_ACC c1 = gamma == nullptr ? static_cast<T_ACC>(rstd[ng]) : static_cast<T_ACC>(rstd[ng]) * static_cast<T_ACC>(gamma[c]);
  if (threadIdx.x == 0) {
    c1_g[nc] = c1;
  }

  // step 3: ComputeBackwardFusedParamsCUDAKernel(C2, C3)
  //         N * G * D => N * G 
  __syncthreads();
  __shared__ T_ACC c2;
  __shared__ T_ACC c3;
  T_ACC sum3 = 0;
  T_ACC sum4 = 0;
  for (size_t i = threadIdx.x; i < D; i+= blockDim.x) {
    const int64_t c = g * D + i;
    const T_ACC gamma_v =
      gamma == nullptr ? T_ACC(1) : static_cast<T_ACC>(gamma[c]);
    sum3 += (ds[i] * gamma_v);
    sum4 += (db[i] * gamma_v);
  }
  if (blockDim.x <= C10_WARP_SIZE) {
    sum3 = cuda_utils::WarpReduceSum<T_ACC>(sum3);
    sum4 = cuda_utils::WarpReduceSum<T_ACC>(sum4);
  } else {
    __shared__ T_ACC ds_shared[C10_WARP_SIZE*8];
    __shared__ T_ACC db_shared[C10_WARP_SIZE*8];
    sum3 = cuda_utils::BlockXReduceSum<T_ACC>(sum3, ds_shared);
    sum4 = cuda_utils::BlockXReduceSum<T_ACC>(sum4, db_shared);
  }
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    const T_ACC s = T_ACC(1) / static_cast<T_ACC>(D * HxW);
    const T_ACC x = (sum4 * static_cast<T_ACC>(mean[ng]) - sum3) *
        static_cast<T_ACC>(rstd[ng]) * static_cast<T_ACC>(rstd[ng]) *
        static_cast<T_ACC>(rstd[ng]) * s;
    c2 = x;
    c3 = -x * static_cast<T_ACC>(mean[ng]) -
        sum4 * static_cast<T_ACC>(rstd[ng]) * s;
    c2_g[ng] = c2;
    c3_g[ng] = c3;
  }

  // step 4: Compute dx
  __syncthreads();
  T_VEC *dx_nc = reinterpret_cast<T_VEC *>(dx + nc * HxW);
  for (size_t i = vec_hxw_idx; i < n_vec_hxw; i+=blockDim.x) {
    T_VEC dx_vec;
    const T_VEC dy_vec = dy_nc[i];
    const T_VEC x_vec = x_nc[i];
    #pragma unroll
    for (size_t j = 0; j < vec_size; j++) {
      dx_vec.val[j] = c1 * static_cast<T_ACC>(dy_vec.val[j]) + c2 * static_cast<T_ACC>(x_vec.val[j]) + c3;
    }
    dx_nc[i] = dx_vec;
  }
}

template<typename T>
bool is_launch_group_norm_forward_opt_kernel(const Tensor& X) {
    return  maca_likely(!at::maca::get_maca_disable_group_norm_opt_kernel()) && 
            (std::is_same<T, float>::value || std::is_same<T, at::Half>::value ||
            std::is_same<T, at::BFloat16>::value) && X.is_contiguous();
}

template<typename T>
bool is_launch_group_norm_backward_opt_kernel(const Tensor& X, const Tensor& dX, size_t  D, size_t HxW, size_t element_size) {
    return  maca_likely(!at::maca::get_maca_disable_group_norm_backward_opt_kernel()) && 
            (std::is_same<T, float>::value || std::is_same<T, at::Half>::value ||
            std::is_same<T, at::BFloat16>::value) && X.is_contiguous() && dX.defined() && dX.is_contiguous() &&
            backward_match_fused_kernel(D, HxW, element_size);
}

template <typename T>
bool is_forward_fused_kernel(
    int64_t HxW,
    Tensor& Y,
    const int vec_size) {
    auto can_vectorize = [&](const T* ptr, int alignment) {
    uint64_t addr = reinterpret_cast<uint64_t>(ptr);
    return addr % alignment == 0;
    }; 
    T *y_data = Y.data_ptr<T>();
    #ifdef USE_MACA
    int alignment = sizeof(T) < 4 ? 4 : sizeof(T);
    #else
    int alignment = vec_size * sizeof(T);
    #endif
    return Y.is_contiguous() && HxW % vec_size == 0 && can_vectorize(y_data, alignment); 
}

template <typename T>
bool launch_group_norm_forward_kernel(
    int64_t N,
    int64_t C,
    int64_t G,
    int64_t HxW,
    T eps,
    const Tensor& X,
    const T* gamma_data,
    const T* beta_data,
    Tensor& Y,
    T* mean_data,
    T* rstd_data
) {
    int64_t D = C / G;
    int64_t DxHxW = D * HxW;
    const T* X_data = X.data_ptr<T>();
    T* Y_data = Y.data_ptr<T>();
    size_t vec_size = get_vec_size(DxHxW, sizeof(T));
    bool fuse_result = is_forward_fused_kernel<T>(HxW, Y, vec_size);
    cudaStream_t cuda_stream = cuda::getCurrentCUDAStream();
    using T_ACC = acc_type<T, true>;
    using WelfordType = WelfordData<T_ACC, int64_t>;
    if (DxHxW == 40960 && HxW == 4096 && fuse_result) {
        // warp_num = 8 per block; D = 10; n_vec_load = 10;
        int64_t elementPerThread = 80;
        int64_t vectorized_block_size = DxHxW / elementPerThread;
        int64_t warp_num = vectorized_block_size / C10_WARP_SIZE;
        int shared_memory_size = warp_num * sizeof(WelfordType);  
        block512VectorizedRowwiseMomentsCUDAKernel<T, 10, 10>
            <<<N * G, vectorized_block_size, shared_memory_size, cuda_stream>>>(N, C, G, HxW, eps, X_data, gamma_data, beta_data, Y_data, fuse_result, mean_data, rstd_data);
        AT_CUDA_CHECK(cudaGetLastError());
        return fuse_result;
    } else if (maca_likely(!at::maca::get_maca_disable_group_norm_fused_kernel()) && fuse_result) {
        size_t block_size = get_forward_fused_kernel_block_size(DxHxW, vec_size);
        size_t shared_memory_size = get_forward_fused_kernel_shared_memory_size(block_size, D, sizeof(WelfordType), sizeof(T_ACC));
        switch(vec_size) {
        case 8:
            FusedGroupNormCUDAKernel<T, 4, 8>
                <<<N * G, block_size, shared_memory_size, cuda_stream>>>(N, C, G, HxW, eps, X_data, gamma_data, beta_data, Y_data, mean_data, rstd_data);
            break;
        case 4:
            FusedGroupNormCUDAKernel<T, 4, 4>
                <<<N * G, block_size, shared_memory_size, cuda_stream>>>(N, C, G, HxW, eps, X_data, gamma_data, beta_data, Y_data, mean_data, rstd_data);
            break;
        case 2:
            FusedGroupNormCUDAKernel<T, 4, 2>
                <<<N * G, block_size, shared_memory_size, cuda_stream>>>(N, C, G, HxW, eps, X_data, gamma_data, beta_data, Y_data, mean_data, rstd_data);
            break;
        default:
            FusedGroupNormCUDAKernel<T, 4, 1>
                <<<N * G, block_size, shared_memory_size, cuda_stream>>>(N, C, G, HxW, eps, X_data, gamma_data, beta_data, Y_data, mean_data, rstd_data);
            break;
        }
        AT_CUDA_CHECK(cudaGetLastError());
        return true;
    } else {
        size_t NxG = N * G;
        dim3 block_size = get_vectorized_rowwise_moment_kernel_block_size(N * G, D * HxW, vec_size);
        size_t grid_size = (NxG + block_size.y - 1) / block_size.y;
        switch(vec_size) {
            case 8:
            VectorizedRowwiseMomentsCUDAKernel<T, 8><<<grid_size, block_size, 0, cuda_stream>>>(NxG, D * HxW, eps, X_data, mean_data, rstd_data);
            break;
            case 4:
            VectorizedRowwiseMomentsCUDAKernel<T, 4><<<grid_size, block_size, 0, cuda_stream>>>(NxG, D * HxW, eps, X_data, mean_data, rstd_data);
            break;
            case 2:
            VectorizedRowwiseMomentsCUDAKernel<T, 2><<<grid_size, block_size, 0, cuda_stream>>>(NxG, D * HxW, eps, X_data, mean_data, rstd_data);
            break;
            default:
            VectorizedRowwiseMomentsCUDAKernel<T, 1><<<grid_size, block_size, 0, cuda_stream>>>(NxG, D * HxW, eps, X_data, mean_data, rstd_data);
            break;
        }
        AT_CUDA_CHECK(cudaGetLastError());
        return false;
    }
}

template <typename T>
void launch_backward_fused_kernel(
    const Tensor& dY,
    const Tensor& X,
    const Tensor& mean,
    const Tensor& rstd,
    const Tensor& gamma,
    int64_t N,
    int64_t C,
    int64_t HxW,
    int64_t G,
    Tensor& dX,
    Tensor& ds,
    Tensor& db
) {
    using T_ACC = acc_type<T, true>;
    size_t vec_size = get_vec_size(HxW, sizeof(T));
    T* dX_data = dX.data_ptr<T>();
    const T* dY_data = dY.data_ptr<T>();
    const T* X_data = X.data_ptr<T>();
    const T* gamma_data = gamma.defined() ? gamma.data_ptr<T>() : nullptr;
    const T* rstd_data = rstd.data_ptr<T>();
    const T* mean_data = mean.data_ptr<T>();
    int64_t D = C / G;
    dim3 grid = dim3(G, N);
    dim3 block = get_backward_fused_kernel_block_size(D, HxW, vec_size);
    size_t shared_memory_size = get_backward_fused_kernel_shared_memory_size(D, sizeof(T_ACC));
    const auto kAccType =
      (X.scalar_type() == kHalf || X.scalar_type() == kBFloat16)
      ? kFloat
      : X.scalar_type();
    Tensor c1 = at::empty({N, C}, X.options().dtype(kAccType));
    Tensor c2 = at::empty({N, G}, X.options().dtype(kAccType));
    Tensor c3 = at::empty({N, G}, X.options().dtype(kAccType));
    T_ACC* c1_data = c1.data_ptr<T_ACC>();
    T_ACC* c2_data = c2.data_ptr<T_ACC>();
    T_ACC* c3_data = c3.data_ptr<T_ACC>();
    T_ACC* ds_data = ds.data_ptr<T_ACC>();
    T_ACC* db_data = db.data_ptr<T_ACC>();
    cudaStream_t cuda_stream = cuda::getCurrentCUDAStream();
    switch (vec_size)
    {
    case 8:
      FusedGroupNormBackwardKernel<T, 8><<<grid, block, shared_memory_size, cuda_stream>>>(N, C, G, HxW, dY_data, X_data, mean_data, rstd_data, gamma_data, ds_data, db_data, c1_data, c2_data, c3_data, dX_data);
      break;
    case 4:
      FusedGroupNormBackwardKernel<T, 4><<<grid, block, shared_memory_size, cuda_stream>>>(N, C, G, HxW, dY_data, X_data, mean_data, rstd_data, gamma_data, ds_data, db_data, c1_data, c2_data, c3_data, dX_data);
      break;
    case 2:
      FusedGroupNormBackwardKernel<T, 2><<<grid, block, shared_memory_size, cuda_stream>>>(N, C, G, HxW, dY_data, X_data, mean_data, rstd_data, gamma_data, ds_data, db_data, c1_data, c2_data, c3_data, dX_data);
      break;
    default:
      FusedGroupNormBackwardKernel<T, 1><<<grid, block, shared_memory_size, cuda_stream>>>(N, C, G, HxW, dY_data, X_data, mean_data, rstd_data, gamma_data, ds_data, db_data, c1_data, c2_data, c3_data, dX_data);
      break;
    }
    return;  
}

template<typename T>
void print_group_norm_info(
    int64_t N,
    int64_t C,
    int64_t HxW,
    int64_t G,
    std::string kernel_type) {
  T temp;
  std::cout << "N:" << N << "C: " << C << "HxW: " << HxW << "G: " << G << " scalar_t: " << typeid(temp).name() << "kernel_type: " << kernel_type << std::endl;
}

}
