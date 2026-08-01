#pragma once


namespace at::native {
namespace layernorm {

constexpr int vec_size = 8;
constexpr unsigned int kWarpSize = 32;

inline int last_pow2(int n) {
  n |= (n >>  1);
  n |= (n >>  2);
  n |= (n >>  4);
  n |= (n >>  8);
  n |= (n >> 16);
  return std::max(1, n - (n >> 1));
}

template<typename scalar_t, int vec_size>
struct alignas(sizeof(scalar_t) * vec_size) aligned_vector {
  scalar_t val[vec_size];
};

struct WelfordDataLN{
  float mean;
  float sigma2;
  float count;
  C10_HOST_DEVICE WelfordDataLN(): mean(0.f), sigma2(0.f), count(0.f){}
  C10_HOST_DEVICE WelfordDataLN(float mean, float sigma2, float count): mean(mean), sigma2(sigma2), count(count) {}
};

template<typename U> __device__
WelfordDataLN cuWelfordOnlineSum(
  const U val,
  const WelfordDataLN& curr_sum)
{
  U delta = val - curr_sum.mean;
  U new_count = curr_sum.count + 1.f;
  #ifdef USE_MACA
    auto coef =  __builtin_mxc_rcpf(new_count);
  #else
    auto coef = 1.f/new_count; //NB we don't use --use_fast_math, but this is emulation, 1./count goes to intrinsic, `* coef` is multiplication, instead of slow fp division
  #endif
  U new_mean = curr_sum.mean + delta * coef; //proper division is slow, this is less accurate but noticeably faster
  return {new_mean, curr_sum.sigma2 + delta * (val - new_mean), new_count};
}

__device__
WelfordDataLN cuWelfordCombine(
  const WelfordDataLN dataB,
  const WelfordDataLN dataA
) {
  using U = decltype(dataB.count);
  U delta = dataB.mean - dataA.mean;
  U count = dataA.count + dataB.count;
  U mean = U(0);
  U sigma2 = U(0);;
  if (count > decltype(dataB.count){0}) {
  #ifdef USE_MACA
    auto coef =  __builtin_mxc_rcpf(count);
  #else
    auto coef = 1.f/count; //NB we don't use --use_fast_math, but this is emulation, 1./count goes to intrinsic, `* coef` is multiplication, instead of slow fp division
  #endif
    auto nA = dataA.count * coef;
    auto nB = dataB.count * coef;
    mean = nA*dataA.mean + nB*dataB.mean;
    sigma2 = dataA.sigma2 + dataB.sigma2 + delta * delta * dataA.count * nB;
  }
  return {mean, sigma2, count};
}

int min_pow2(int n){
  assert(n > 0 && n <= 64);
  if(n <= 2 ) return n;
  else if(n > 2 && n <= 4) return 4;
  else if(n > 4 && n <= 8) return 8;
  else if(n > 8 && n <= 16) return 16;
  else if(n > 16 && n <=32) return 32;
  else return 64;
}


template <typename T> 
__device__ WelfordDataLN compute_stats_maca(
    const T* __restrict__ X,
    const int N,
    const int last_pow2_size,
    float* buf) {
  // X points to the row to read
  using vec_t = aligned_vector<T, vec_size>;
  using acc_t = acc_type<T, true>;
  const vec_t* X_vec = reinterpret_cast<const vec_t*>(X);
  const int thrx = threadIdx.x + threadIdx.y * blockDim.x;
  const int numx = blockDim.x * blockDim.y;
  const int n_vec_to_read = N / vec_size;
  WelfordDataLN wd(0.f, 0.f, 0.f);
  // no tail, we check that N is multiple of vec_size
  for (int i = thrx; i < n_vec_to_read; i += numx) {
    vec_t data = X_vec[i];
#pragma unroll
    for (int ii = 0; ii < vec_size; ii++) {
      wd = cuWelfordOnlineSum(static_cast<acc_t>(data.val[ii]), wd);
    }
  }

  // intra-warp reduction
  for (int offset = (C10_WARP_SIZE >> 1); offset > 0; offset >>= 1) {
    WelfordDataLN wdB{
        WARP_SHFL_DOWN(wd.mean, offset),
        WARP_SHFL_DOWN(wd.sigma2, offset),
        WARP_SHFL_DOWN(wd.count, offset)};
    wd = cuWelfordCombine(wd, wdB);
  }
  __syncthreads();
  // threadIdx.x == 0 has correct values for each warp
  // inter-warp reductions
  if (blockDim.y > 1) {
    const int last_pow2_dimy = last_pow2_size;
    const int remaining_size = blockDim.y - last_pow2_dimy;
    float* meansigmabuf = buf;
    float* countbuf = buf + last_pow2_dimy;
    float* remainingbuf = countbuf + last_pow2_dimy / 2;
    float* remaining_countbuf = remainingbuf + remaining_size * 2;
    // threads id over 2^n store wd to shared memory.
    if (threadIdx.x == 0 && threadIdx.y >= last_pow2_dimy) {
      const int remaining_idx = threadIdx.y - last_pow2_dimy;
      remainingbuf[2 * remaining_idx] = wd.mean;
      remainingbuf[2 * remaining_idx + 1] = wd.sigma2;
      remaining_countbuf[remaining_idx] = wd.count;
    }
    __syncthreads();
    // threads id less than remaining_size read wd from shared memory
    // and combine it.
    if (threadIdx.x == 0 && threadIdx.y < remaining_size) {
      WelfordDataLN wdB{
          remainingbuf[2 * threadIdx.y],
          remainingbuf[2 * threadIdx.y + 1],
          remaining_countbuf[threadIdx.y]};
      wd = cuWelfordCombine(wd, wdB);
    }
    __syncthreads();

    for (int offset = last_pow2_dimy / 2; offset > 0; offset /= 2) {
      // upper half of warps write to shared
      if (threadIdx.x == 0 && threadIdx.y >= offset &&
          threadIdx.y < 2 * offset) {
        const int wrt_y = threadIdx.y - offset;
        meansigmabuf[2 * wrt_y] = wd.mean;
        meansigmabuf[2 * wrt_y + 1] = wd.sigma2;
        countbuf[wrt_y] = wd.count;
      }
      __syncthreads();
      // lower half merges
      if (threadIdx.x == 0 && threadIdx.y < offset) {
        WelfordDataLN wdB{
            meansigmabuf[2 * threadIdx.y],
            meansigmabuf[2 * threadIdx.y + 1],
            countbuf[threadIdx.y]};
        wd = cuWelfordCombine(wd, wdB);
      }
      __syncthreads();
    }

#ifdef USE_MACA
      auto coef =  __builtin_mxc_rcpf(float(N));
#else
      auto coef = 1.f/float(N); //NB we don't use --use_fast_math, but this is emulation, 1./count goes to intrinsic, `* coef` is multiplication, instead of slow fp division
#endif
    if (threadIdx.x == 0 && threadIdx.y == 0) {
      meansigmabuf[0] = wd.mean;
      meansigmabuf[1] = wd.sigma2 * coef;
    }
    __syncthreads();
    return WelfordDataLN{meansigmabuf[0], meansigmabuf[1], 0.f};

  } else {
    return WelfordDataLN{
        WARP_SHFL(wd.mean, 0), WARP_SHFL(wd.sigma2, 0) / float(N), 0.f};
  }
}


template <typename T, typename T_ACC,
typename std::enable_if<!std::is_same<T, double>::value, int>::type = 0>
__device__ __inline__ void vectorized_layer_norm_kernel_maca_impl(
    const int N,
    const int last_pow2_size,
    T_ACC eps,
    const T* __restrict__ X,
    const T* gamma,
    const T* beta,
    T_ACC* mean,
    T_ACC* rstd,
    T* Y) {
  extern __shared__ float s_data[]; // if we made smem WelfordDataLN type, there
                                    // would be bank conflicts,
  // as one thread would have to write 3 consecutive floats
  auto i1 = blockIdx.x;
  const T* block_row = X + i1 * N;
  WelfordDataLN wd = compute_stats_maca(block_row, N, last_pow2_size, s_data);
  using vec_t = aligned_vector<T, vec_size>;
  const vec_t* X_vec = reinterpret_cast<const vec_t*>(block_row);
  vec_t* Y_vec = reinterpret_cast<vec_t*>(Y + i1 * N);
  const int numx = blockDim.x * blockDim.y;
  const int thrx = threadIdx.x + threadIdx.y * blockDim.x;
  const int n_vec_to_read = N / vec_size;
  T_ACC rstd_val = c10::cuda::compat::rsqrt(wd.sigma2 + eps);
  // no tail, N is guaranteed to be multiple of vec size
  for (int i = thrx; i < n_vec_to_read; i += numx) {
    vec_t data = X_vec[i];
    vec_t out;
    // computation is performed in T_ACC, X is cast to T_ACC and result is
    // implicitly cast to T
    if (gamma != nullptr && beta != nullptr) {
#pragma unroll
      for (int ii = 0; ii < vec_size; ii++) {
        out.val[ii] = static_cast<T_ACC>(gamma[i * vec_size + ii]) *
                (rstd_val * (static_cast<T_ACC>(data.val[ii]) - wd.mean)) +
            static_cast<T_ACC>(beta[i * vec_size + ii]);
      }
    } else if (gamma != nullptr) {
#pragma unroll
      for (int ii = 0; ii < vec_size; ii++) {
        out.val[ii] = static_cast<T_ACC>(gamma[i * vec_size + ii]) *
            (rstd_val * (static_cast<T_ACC>(data.val[ii]) - wd.mean));
      }
    } else if (beta != nullptr) {
#pragma unroll
      for (int ii = 0; ii < vec_size; ii++) {
        out.val[ii] =
            (rstd_val * (static_cast<T_ACC>(data.val[ii]) - wd.mean)) +
            static_cast<T_ACC>(beta[i * vec_size + ii]);
      }
    } else {
#pragma unroll
      for (int ii = 0; ii < vec_size; ii++) {
        out.val[ii] = rstd_val * (static_cast<T_ACC>(data.val[ii]) - wd.mean);
      }
    }
    Y_vec[i] = out;
  }

  if (thrx == 0) {
    mean[i1] = wd.mean;
    rstd[i1] = rstd_val;
  }
}


template <typename T, typename T_ACC,
typename std::enable_if<std::is_same<T, double>::value, int>::type = 0>
__device__ __inline__ void vectorized_layer_norm_kernel_maca_impl(
  const int /*N*/,
  const int /*last_pow2_size*/,
  T_ACC /*eps*/,
  const  T* __restrict__ /*X*/,
  const  T* /*gamma*/,
  const  T* /*beta*/,
  T_ACC* /*mean*/,
  T_ACC* /*rstd*/,
  T* /*Y*/){
    CUDA_KERNEL_ASSERT(false && "doesn't work with double");
  }

template <typename T, typename T_ACC>
__global__ __inline__ void vectorized_layer_norm_kernel_opt_discard(
  const int N,
  const int last_pow2_size,
  T_ACC eps,
  const  T* __restrict__ X,
  const  T* gamma,
  const  T* beta,
  T_ACC* mean,
  T_ACC* rstd,
  T* Y){
  vectorized_layer_norm_kernel_maca_impl(N, last_pow2_size, eps, X, gamma, beta, mean, rstd, Y);
}


template <int vec_maca, typename T, typename T_ACC,
typename std::enable_if<!std::is_same<T, double>::value, int>::type = 0>
__device__ __inline__ void vectorized_layer_norm_kernel_maca_impl_opt(
    const int N,
    const int pow2,
    T_ACC eps,
    const T* __restrict__ X,
    const T* gamma,
    const T* beta,
    T_ACC* mean,
    T_ACC* rstd,
    T* Y) {
  
  extern __shared__ float s_data[];
  auto i1 = blockIdx.x;
  const T* block_row = X + i1 * N;

  using vec_t = aligned_vector<T, vec_maca>;
  using acc_t = acc_type<T, true>;
  const vec_t* X_vec = reinterpret_cast<const vec_t*>(block_row);
  const int thrx = threadIdx.x + threadIdx.y * blockDim.x;
  const int numx = blockDim.x * blockDim.y;
  const int n_vec_to_read = N / vec_maca;
  WelfordDataLN wd(0.f, 0.f, 0.f);
  
  // thread reduce
  for (int i = thrx; i < n_vec_to_read; i += numx) {
    vec_t data = X_vec[i];
    #pragma unroll
    for (int ii = 0; ii < vec_maca; ii++) {
      wd = cuWelfordOnlineSum(static_cast<acc_t>(data.val[ii]), wd);
    }
  }
  
  // warp reduce
  for (int offset = (C10_WARP_SIZE >> 1); offset > 0; offset >>= 1) {
    WelfordDataLN wdB{
        WARP_SHFL_DOWN(wd.mean, offset),
        WARP_SHFL_DOWN(wd.sigma2, offset),
        WARP_SHFL_DOWN(wd.count, offset)};
    wd = cuWelfordCombine(wd, wdB);
  }
  __syncthreads();
  
  if (blockDim.y > 1) {
    float * meanbuf = s_data;
    float * sigmabuf = s_data + pow2;
    float * countbuf = s_data + pow2 * 2;

    using B = cuda_utils::Block2D;
    const int tid = B::Tid();
    const int lid = tid % C10_WARP_SIZE;
    const int wid = tid / C10_WARP_SIZE;
    
    if (lid == 0) {
      meanbuf[wid] =  wd.mean;
      sigmabuf[wid] = wd.sigma2;
      countbuf[wid] = wd.count;
    }

    __syncthreads();
    if (wid == 0) {
      wd.mean = T_ACC(0);
      wd.sigma2 = T_ACC(0);
      wd.count = T_ACC(0);
      if (tid < B::Warps()) {
        wd.mean = meanbuf[tid];
        wd.sigma2 = sigmabuf[tid];
        wd.count = countbuf[tid];
      }
      #pragma unroll
      for (int offset = (pow2 >> 1); offset > 0; offset >>= 1) {
        WelfordDataLN wdB{WARP_SHFL_DOWN(wd.mean, offset),
                          WARP_SHFL_DOWN(wd.sigma2, offset),
                          WARP_SHFL_DOWN(wd.count, offset)};
        wd = cuWelfordCombine(wd, wdB);
      }
    }

    if (tid == 0) {
      meanbuf[0] =  wd.mean;
      sigmabuf[0] = wd.sigma2;
      countbuf[0] = wd.count;
    }
    __syncthreads();
    wd.mean = meanbuf[0];
    wd.sigma2 = sigmabuf[0] * __builtin_mxc_rcpf(float(N));
    wd.count = countbuf[0];
  } else {
    wd = WelfordDataLN{WARP_SHFL(wd.mean, 0), 
                       WARP_SHFL(wd.sigma2, 0) * __builtin_mxc_rcpf(float(N)), 
                       0.f};
  }
  
  vec_t* Y_vec = reinterpret_cast<vec_t*>(Y + i1 * N);
  T_ACC rstd_val = c10::cuda::compat::rsqrt(wd.sigma2 + eps);
  const vec_t * gamma_vec = reinterpret_cast<const vec_t*>(gamma);
  const vec_t * beta_vec = reinterpret_cast<const vec_t*>(beta);
  for (int i = thrx; i < n_vec_to_read; i += numx) {
    vec_t data = X_vec[i];
    vec_t out;
    if (gamma != nullptr && beta != nullptr) {
      vec_t gamma_data = gamma_vec[i];
      vec_t beta_data = beta_vec[i];
      #pragma unroll
      for (int ii=0; ii < vec_maca; ii++) {
        out.val[ii] = static_cast<T_ACC>(gamma_data.val[ii]) * (rstd_val * (static_cast<T_ACC>(data.val[ii]) - wd.mean))
        + static_cast<T_ACC>(beta_data.val[ii]);
      }
    } else if (gamma != nullptr) {
      vec_t gamma_data = gamma_vec[i];
      #pragma unroll
      for (int ii=0; ii < vec_maca; ii++) {
        out.val[ii] = static_cast<T_ACC>(gamma_data.val[ii]) * (rstd_val * (static_cast<T_ACC>(data.val[ii]) - wd.mean));
      }
    } else if (beta != nullptr) {
      vec_t beta_data = beta_vec[i];
      #pragma unroll
      for (int ii=0; ii < vec_maca; ii++) {
        out.val[ii] = (rstd_val * (static_cast<T_ACC>(data.val[ii]) - wd.mean))+ static_cast<T_ACC>(beta_data.val[ii]);
      }
    } else {
      #pragma unroll
      for (int ii=0; ii < vec_maca; ii++) {
        out.val[ii] = rstd_val * (static_cast<T_ACC>(data.val[ii]) - wd.mean);
      }
    }
    Y_vec[i] = out;
  }

  if (thrx == 0) {
    mean[i1] = wd.mean;
    rstd[i1] = rstd_val;
  }
}


template <int vec_maca, typename T, typename T_ACC,
typename std::enable_if<std::is_same<T, double>::value, int>::type = 0>
__device__ __inline__ void vectorized_layer_norm_kernel_maca_impl_opt(
  const int /*N*/,
  const int /*pow2*/,
  T_ACC /*eps*/,
  const  T* __restrict__ /*X*/,
  const  T* /*gamma*/,
  const  T* /*beta*/,
  T_ACC* /*mean*/,
  T_ACC* /*rstd*/,
  T* /*Y*/){
    CUDA_KERNEL_ASSERT(false && "doesn't work with double");
  }


//to avoid windows SFINAE errors
template <int vec_maca, typename T, typename T_ACC>
__global__ __inline__ void vectorized_layer_norm_kernel_opt(
  const int N,
  const int pow2,
  T_ACC eps,
  const  T* __restrict__ X,
  const  T* gamma,
  const  T* beta,
  T_ACC* mean,
  T_ACC* rstd,
  T* Y){
  vectorized_layer_norm_kernel_maca_impl_opt<vec_maca>(N, pow2, eps, X, gamma, beta, mean, rstd, Y);
}



template <int vec_maca, typename T, typename T_ACC,
typename std::enable_if<std::is_same<T, double>::value, int>::type = 0>
__global__ __inline__ void vectorized_layer_norm_kernel_opt_wave1(
  const int N,
  const int pow2_x,
  T_ACC eps,
  const  T* __restrict__ X,
  const  T* gamma,
  const  T* beta,
  T_ACC* mean,
  T_ACC* rstd,
  T* Y){ CUDA_KERNEL_ASSERT(false && "doesn't work with double"); }


template <int vec_maca, typename T, typename T_ACC,
typename std::enable_if<!std::is_same<T, double>::value, int>::type = 0>
__global__ __inline__ void vectorized_layer_norm_kernel_opt_wave1(
  const int N,
  const int pow2_x,
  T_ACC eps,
  const  T* __restrict__ X,
  const  T* gamma,
  const  T* beta,
  T_ACC* mean,
  T_ACC* rstd,
  T* Y){
  using vec_t = aligned_vector<T, vec_maca>;
  using acc_t = acc_type<T, true>;

  auto i1 = blockIdx.x;
  const T* block_row = X + i1 * N;
  const vec_t* X_vec = reinterpret_cast<const vec_t*>(block_row);
  const int thrx = threadIdx.x;
  WelfordDataLN wd(0.f, 0.f, 0.f);

  vec_t data = X_vec[thrx];

  // thread reduce
  #pragma unroll
  for (int ii = 0; ii < vec_maca; ii++) {
    wd = cuWelfordOnlineSum(static_cast<acc_t>(data.val[ii]), wd);
  }
  
  // warp reduce
  for (int offset = (pow2_x >> 1); offset > 0; offset >>= 1) {
    WelfordDataLN wdB{
        WARP_SHFL_DOWN(wd.mean, offset),
        WARP_SHFL_DOWN(wd.sigma2, offset),
        WARP_SHFL_DOWN(wd.count, offset)};
    wd = cuWelfordCombine(wd, wdB);
  }
  __syncthreads();
  
  wd = WelfordDataLN{ WARP_SHFL(wd.mean, 0), 
                      WARP_SHFL(wd.sigma2, 0) * __builtin_mxc_rcpf(float(N)), 
                      0.f};
  
  vec_t* Y_vec = reinterpret_cast<vec_t*>(Y + i1 * N);
  T_ACC rstd_val = c10::cuda::compat::rsqrt(wd.sigma2 + eps);
  const vec_t * gamma_vec = reinterpret_cast<const vec_t*>(gamma);
  const vec_t * beta_vec = reinterpret_cast<const vec_t*>(beta);
  // vec_t data = X_vec[thrx];
  vec_t out;
  if (gamma != nullptr && beta != nullptr) {
    vec_t gamma_data = gamma_vec[thrx];
    vec_t beta_data = beta_vec[thrx];
    #pragma unroll
    for (int ii=0; ii < vec_maca; ii++) {
      out.val[ii] = static_cast<T_ACC>(gamma_data.val[ii]) * (rstd_val * (static_cast<T_ACC>(data.val[ii]) - wd.mean))
      + static_cast<T_ACC>(beta_data.val[ii]);
    }
  } else if (gamma != nullptr) {
    vec_t gamma_data = gamma_vec[thrx];
    #pragma unroll
    for (int ii=0; ii < vec_maca; ii++) {
      out.val[ii] = static_cast<T_ACC>(gamma_data.val[ii]) * (rstd_val * (static_cast<T_ACC>(data.val[ii]) - wd.mean));
    }
  } else if (beta != nullptr) {
    vec_t beta_data = beta_vec[thrx];
    #pragma unroll
    for (int ii=0; ii < vec_maca; ii++) {
      out.val[ii] = (rstd_val * (static_cast<T_ACC>(data.val[ii]) - wd.mean))+ static_cast<T_ACC>(beta_data.val[ii]);
    }
  } else {
    #pragma unroll
    for (int ii=0; ii < vec_maca; ii++) {
      out.val[ii] = rstd_val * (static_cast<T_ACC>(data.val[ii]) - wd.mean);
    }
  }
  Y_vec[thrx] = out;

  if (thrx == 0) {
    mean[i1] = wd.mean;
    rstd[i1] = rstd_val;
  }
}


template <int vec_maca, typename T, typename T_ACC,
typename std::enable_if<std::is_same<T, double>::value, int>::type = 0>
__global__ __inline__ void vectorized_layer_norm_kernel_opt_pass1(
  const int N,
  const int pow2_x,
  T_ACC eps,
  const  T* __restrict__ X,
  const  T* gamma,
  const  T* beta,
  T_ACC* mean,
  T_ACC* rstd,
  T* Y){ CUDA_KERNEL_ASSERT(false && "doesn't work with double"); }


template <int vec_maca, typename T, typename T_ACC,
typename std::enable_if<!std::is_same<T, double>::value, int>::type = 0>
__global__ __inline__ void vectorized_layer_norm_kernel_opt_pass1(
  const int N,
  const int pow2_y,
  T_ACC eps,
  const  T* __restrict__ X,
  const  T* gamma,
  const  T* beta,
  T_ACC* mean,
  T_ACC* rstd,
  T* Y) {
  const int thrx = threadIdx.x + threadIdx.y * blockDim.x;
  if (thrx * vec_maca >= N) return;

  extern __shared__ float s_data[];
  auto i1 = blockIdx.x;
  const T* block_row = X + i1 * N;

  using vec_t = aligned_vector<T, vec_maca>;
  using acc_t = acc_type<T, true>;
  const vec_t* X_vec = reinterpret_cast<const vec_t*>(block_row);
  WelfordDataLN wd(0.f, 0.f, 0.f);

  vec_t data = X_vec[thrx];
  //thread reduce
  #pragma unroll
  for (int ii = 0; ii < vec_maca; ii++) {
    wd = cuWelfordOnlineSum(static_cast<acc_t>(data.val[ii]), wd);
  }

  // warp reduce
  for (int offset = (C10_WARP_SIZE >> 1); offset > 0; offset >>= 1) {
    WelfordDataLN wdB{
        WARP_SHFL_DOWN(wd.mean, offset),
        WARP_SHFL_DOWN(wd.sigma2, offset),
        WARP_SHFL_DOWN(wd.count, offset)};
    wd = cuWelfordCombine(wd, wdB);
  }
  __syncthreads();

  //block reduce, blockdim.y >1
  float * meanbuf = s_data;
  float * sigmabuf = s_data + pow2_y;
  float * countbuf = s_data + pow2_y * 2;

  using B = cuda_utils::Block2D;
  const int tid = B::Tid();
  const int lid = tid % C10_WARP_SIZE;
  const int wid = tid / C10_WARP_SIZE;
  if (lid == 0) {
    meanbuf[wid] =  wd.mean;
    sigmabuf[wid] = wd.sigma2;
    countbuf[wid] = wd.count;
  }
  __syncthreads();
  if (wid == 0) {
    wd.mean = T_ACC(0);
    wd.sigma2 = T_ACC(0);
    wd.count = T_ACC(0);
    if (tid < B::Warps()) {
      wd.mean = meanbuf[tid];
      wd.sigma2 = sigmabuf[tid];
      wd.count = countbuf[tid];
    }
    #pragma unroll
    for (int offset = (pow2_y >> 1); offset > 0; offset >>= 1) {
      WelfordDataLN wdB{WARP_SHFL_DOWN(wd.mean, offset),
                        WARP_SHFL_DOWN(wd.sigma2, offset),
                        WARP_SHFL_DOWN(wd.count, offset)};
      wd = cuWelfordCombine(wd, wdB);
    }
  }

  if (tid == 0) {
    meanbuf[0] =  wd.mean;
    sigmabuf[0] = wd.sigma2;
    countbuf[0] = wd.count;
  }
  __syncthreads();
  wd.mean = meanbuf[0];
  wd.sigma2 = sigmabuf[0] * __builtin_mxc_rcpf(float(N));
  wd.count = countbuf[0];

  //compute and save
  vec_t* Y_vec = reinterpret_cast<vec_t*>(Y + i1 * N);
  T_ACC rstd_val = c10::cuda::compat::rsqrt(wd.sigma2 + eps);
  const vec_t * gamma_vec = reinterpret_cast<const vec_t*>(gamma);
  const vec_t * beta_vec = reinterpret_cast<const vec_t*>(beta);
  
  vec_t out;
  if (gamma != nullptr && beta != nullptr) {
    vec_t gamma_data = gamma_vec[thrx];
    vec_t beta_data = beta_vec[thrx];
    #pragma unroll
    for (int ii=0; ii < vec_maca; ii++) {
      out.val[ii] = static_cast<T_ACC>(gamma_data.val[ii]) * (rstd_val * (static_cast<T_ACC>(data.val[ii]) - wd.mean))
      + static_cast<T_ACC>(beta_data.val[ii]);
    }
  } else if (gamma != nullptr) {
    vec_t gamma_data = gamma_vec[thrx];
    #pragma unroll
    for (int ii=0; ii < vec_maca; ii++) {
      out.val[ii] = static_cast<T_ACC>(gamma_data.val[ii]) * (rstd_val * (static_cast<T_ACC>(data.val[ii]) - wd.mean));
    }
  } else if (beta != nullptr) {
    vec_t beta_data = beta_vec[thrx];
    #pragma unroll
    for (int ii=0; ii < vec_maca; ii++) {
      out.val[ii] = (rstd_val * (static_cast<T_ACC>(data.val[ii]) - wd.mean))+ static_cast<T_ACC>(beta_data.val[ii]);
    }
  } else {
    #pragma unroll
    for (int ii=0; ii < vec_maca; ii++) {
      out.val[ii] = rstd_val * (static_cast<T_ACC>(data.val[ii]) - wd.mean);
    }
  }
  Y_vec[thrx] = out;

  if (thrx == 0) {
    mean[i1] = wd.mean;
    rstd[i1] = rstd_val;
  }
} 


template<int vec_bac, typename T, typename T_ACC>
__global__ void vectorized_layer_norm_grad_input_kernel(
  const T* __restrict__ dY,
  const T* __restrict__ X,
  const T_ACC* __restrict__ mean,
  const T_ACC* __restrict__ rstd,
  const T* __restrict__ gamma,
  T*  dX,
  const int N){
    alignas(sizeof(double)) extern __shared__ char s_data1[];
    T_ACC * buf = reinterpret_cast<T_ACC*>(&s_data1);

    const auto i1 = blockIdx.x;
    const T_ACC mean_val = mean[i1];
    const T_ACC rstd_val = rstd[i1];
    T_ACC stats_x1{0}, stats_x2{0};
    
    const T *dY_row = dY + i1 * N;
    const T *X_row = X + i1 * N;
    T * dX_row = dX + i1 * N;
    using vec_t = aligned_vector<T, vec_bac>;
    const vec_t* dY_vec = reinterpret_cast<const vec_t*>(dY_row);
    const vec_t* X_vec = reinterpret_cast<const vec_t*>(X_row);
    vec_t* dX_vec = reinterpret_cast<vec_t*>(dX_row);

    const int n_vec_to_read = N / vec_bac;
    const int begin = threadIdx.x;
    const int step = blockDim.x;
    if (gamma != nullptr) {
      const vec_t* gamma_vec = reinterpret_cast<const vec_t*>(gamma);
      for(int i = begin; i < n_vec_to_read; i+=step){
        vec_t dY_data = dY_vec[i];
        vec_t X_data = X_vec[i];
        vec_t gamma_data = gamma_vec[i];
        #pragma unroll
        for (int ii = 0; ii < vec_bac; ii++) {
          T_ACC gamma_val = static_cast<T_ACC>(gamma_data.val[ii]);
          const T_ACC c_h = static_cast<T_ACC>(X_data.val[ii]);
          const T_ACC c_loss = static_cast<T_ACC>(dY_data.val[ii]);
          stats_x1 += c_loss * gamma_val;
          stats_x2 += c_loss * gamma_val * (c_h - mean_val) * rstd_val;
        }
      }
    } else {
      for (int i = begin; i < n_vec_to_read; i+=step) {
        vec_t dY_data = dY_vec[i];
        vec_t X_data = X_vec[i];
        #pragma unroll
        for (int ii = 0; ii < vec_bac; ii++) {
          const T_ACC c_h = static_cast<T_ACC>(X_data.val[ii]);
          const T_ACC c_loss = static_cast<T_ACC>(dY_data.val[ii]);
          stats_x1 += c_loss;
          stats_x2 += c_loss * (c_h - mean_val) * rstd_val;
        }
      }
    }
    
    stats_x1 = cuda_utils::BlockReduceSum(stats_x1, buf);
    stats_x2 = cuda_utils::BlockReduceSum(stats_x2, buf);
   
    if (threadIdx.x == 0) {
      buf[0] = stats_x1;
      buf[1] = stats_x2;
    }
    __syncthreads();
    stats_x1 = buf[0];
    stats_x2 = buf[1];
    T_ACC fH = N;

  #ifdef USE_MACA
    auto coef =  __builtin_mxc_rcpf(T_ACC(fH));
  #else
    auto coef = (T_ACC(1) / fH);
  #endif
    T_ACC term1 = coef * rstd_val;
    
    vec_t tmp;
    if (gamma != nullptr) {
      const vec_t* gamma_vec = reinterpret_cast<const vec_t*>(gamma);
      for(int i = begin; i < n_vec_to_read; i+=step){
        vec_t dY_data = dY_vec[i];
        vec_t X_data = X_vec[i];
        vec_t gamma_data = gamma_vec[i];
        
        #pragma unroll
        for (int ii = 0; ii < vec_bac; ii++) {
          const T_ACC dy = static_cast<T_ACC>(dY_data.val[ii]);
          const T_ACC x = static_cast<T_ACC>(X_data.val[ii]);
          T_ACC gamma_val = static_cast<T_ACC>(gamma_data.val[ii]);
          T_ACC f_grad_input = fH * gamma_val * dy;
          f_grad_input -= (x - mean_val) * rstd_val * stats_x2;
          f_grad_input -= stats_x1;
          f_grad_input *= term1;
          tmp.val[ii] = f_grad_input;
        }
        dX_vec[i]=tmp;
      }
    } else {
      for (int i = begin; i < n_vec_to_read; i+=step) {
        vec_t dY_data = dY_vec[i];
        vec_t X_data = X_vec[i];
        #pragma unroll
        for (int ii = 0; ii < vec_bac; ii++) {
          const T_ACC dy = static_cast<T_ACC>(dY_data.val[ii]);
          const T_ACC x = static_cast<T_ACC>(X_data.val[ii]);
          T_ACC f_grad_input = fH * dy;
          f_grad_input -= (x - mean_val) * rstd_val * stats_x2;
          f_grad_input -= stats_x1;
          f_grad_input *= term1;
          tmp.val[ii] = f_grad_input;
        }
        dX_vec[i]=tmp;
      }
    }

  }


template<int vec_bac, typename T, typename T_ACC>
__global__ void vectorized_layer_norm_grad_input_kernel_fp_256(
  const T* __restrict__ dY,
  const T* __restrict__ X,
  const T_ACC* __restrict__ mean,
  const T_ACC* __restrict__ rstd,
  const T* __restrict__ gamma,
  T*  dX,
  const int N){
    alignas(sizeof(double)) extern __shared__ char s_data1[];
    T_ACC * buf = reinterpret_cast<T_ACC*>(&s_data1);

    const auto i1 = blockIdx.x;
    const T_ACC mean_val = mean[i1];
    const T_ACC rstd_val = rstd[i1];
    T_ACC stats_x1{0}, stats_x2{0};

    const T *dY_row = dY + i1 * N;
    const T *X_row = X + i1 * N;
    T * dX_row = dX + i1 * N;
    using vec_t = aligned_vector<T, vec_bac>;
    const vec_t* dY_vec = reinterpret_cast<const vec_t*>(dY_row);
    const vec_t* X_vec = reinterpret_cast<const vec_t*>(X_row);
    vec_t* dX_vec = reinterpret_cast<vec_t*>(dX_row);

    const int index = threadIdx.x;
    vec_t dY_data = dY_vec[index];;
    vec_t X_data =  X_vec[index];
    vec_t gamma_data;
    if (gamma != nullptr) {
      gamma_data = reinterpret_cast<const vec_t*>(gamma)[index];
      #pragma unroll
      for (int ii = 0; ii < vec_bac; ii++) {
        T_ACC gamma_val = static_cast<T_ACC>(gamma_data.val[ii]);
        const T_ACC c_h = static_cast<T_ACC>(X_data.val[ii]);
        const T_ACC c_loss = static_cast<T_ACC>(dY_data.val[ii]);
        stats_x1 += c_loss * gamma_val;
        stats_x2 += c_loss * gamma_val * (c_h - mean_val) * rstd_val;
      }
    } else {
      #pragma unroll
      for (int ii = 0; ii < vec_bac; ii++) {
        const T_ACC c_h = static_cast<T_ACC>(X_data.val[ii]);
        const T_ACC c_loss = static_cast<T_ACC>(dY_data.val[ii]);
        stats_x1 += c_loss;
        stats_x2 += c_loss * (c_h - mean_val) * rstd_val;
      }
    }

    #pragma unroll
    for (int offset = (C10_WARP_SIZE >> 1); offset > 0; offset >>= 1) {
      stats_x1 += WARP_SHFL_DOWN(stats_x1, offset);
      stats_x2 += WARP_SHFL_DOWN(stats_x2, offset);
    }

    if (threadIdx.x == 0) {
      buf[0] = stats_x1;
      buf[1] = stats_x2;
    }
    __syncthreads();
    stats_x1 = buf[0];
    stats_x2 = buf[1];
    T_ACC fH = N;

    #ifdef USE_MACA
      auto coef =  __builtin_mxc_rcpf(T_ACC(fH));
    #else
      auto coef = (T_ACC(1) / fH);
    #endif
      T_ACC term1 = coef * rstd_val;

    vec_t tmp;
    if (gamma != nullptr) {
      #pragma unroll
      for (int ii = 0; ii < vec_bac; ii++) {
        const T_ACC dy = static_cast<T_ACC>(dY_data.val[ii]);
        const T_ACC x = static_cast<T_ACC>(X_data.val[ii]);
        T_ACC gamma_val = static_cast<T_ACC>(gamma_data.val[ii]);
        T_ACC f_grad_input = fH * gamma_val * dy;
        f_grad_input -= (x - mean_val) * rstd_val * stats_x2;
        f_grad_input -= stats_x1;
        f_grad_input *= term1;
        tmp.val[ii] = f_grad_input;
      }
      dX_vec[index]=tmp;
    } else {
      #pragma unroll
      for (int ii = 0; ii < vec_bac; ii++) {
        const T_ACC dy = static_cast<T_ACC>(dY_data.val[ii]);
        const T_ACC x = static_cast<T_ACC>(X_data.val[ii]);
        T_ACC f_grad_input = fH * dy;
        f_grad_input -= (x - mean_val) * rstd_val * stats_x2;
        f_grad_input -= stats_x1;
        f_grad_input *= term1;
        tmp.val[ii] = f_grad_input;
      }
      dX_vec[index]=tmp;
    }

  }



template<int vec_bac, typename T, typename T_ACC>
__global__ void vectorized_layer_norm_grad_input_kernel_wave1(
  const T* __restrict__ dY,
  const T* __restrict__ X,
  const T_ACC* __restrict__ mean,
  const T_ACC* __restrict__ rstd,
  const T* __restrict__ gamma,
  T*  dX,
  const int N){
    const int index = threadIdx.x;
    if (index * vec_bac >= N) return;

    alignas(sizeof(double)) extern __shared__ char s_data1[];
    T_ACC * buf = reinterpret_cast<T_ACC*>(&s_data1);

    const auto i1 = blockIdx.x;
    const T_ACC mean_val = mean[i1];
    const T_ACC rstd_val = rstd[i1];
    T_ACC stats_x1{0}, stats_x2{0};

    const T *dY_row = dY + i1 * N;
    const T *X_row = X + i1 * N;
    T * dX_row = dX + i1 * N;
    using vec_t = aligned_vector<T, vec_bac>;
    const vec_t* dY_vec = reinterpret_cast<const vec_t*>(dY_row);
    const vec_t* X_vec = reinterpret_cast<const vec_t*>(X_row);
    vec_t* dX_vec = reinterpret_cast<vec_t*>(dX_row);

    vec_t dY_data = dY_vec[index];;
    vec_t X_data =  X_vec[index];
    vec_t gamma_data;
    if (gamma != nullptr) {
      gamma_data = reinterpret_cast<const vec_t*>(gamma)[index];
      #pragma unroll
      for (int ii = 0; ii < vec_bac; ii++) {
        T_ACC gamma_val = static_cast<T_ACC>(gamma_data.val[ii]);
        const T_ACC c_h = static_cast<T_ACC>(X_data.val[ii]);
        const T_ACC c_loss = static_cast<T_ACC>(dY_data.val[ii]);
        stats_x1 += c_loss * gamma_val;
        stats_x2 += c_loss * gamma_val * (c_h - mean_val) * rstd_val;
      }
    } else {
      #pragma unroll
      for (int ii = 0; ii < vec_bac; ii++) {
        const T_ACC c_h = static_cast<T_ACC>(X_data.val[ii]);
        const T_ACC c_loss = static_cast<T_ACC>(dY_data.val[ii]);
        stats_x1 += c_loss;
        stats_x2 += c_loss * (c_h - mean_val) * rstd_val;
      }
    }

    #pragma unroll
    for (int offset = (C10_WARP_SIZE >> 1); offset > 0; offset >>= 1) {
      stats_x1 += WARP_SHFL_DOWN(stats_x1, offset);
      stats_x2 += WARP_SHFL_DOWN(stats_x2, offset);
    }

    if (threadIdx.x == 0) {
      buf[0] = stats_x1;
      buf[1] = stats_x2;
    }
    __syncthreads();
    stats_x1 = buf[0];
    stats_x2 = buf[1];
    T_ACC fH = N;

    #ifdef USE_MACA
      auto coef =  __builtin_mxc_rcpf(T_ACC(fH));
    #else
      auto coef = (T_ACC(1) / fH);
    #endif
      T_ACC term1 = coef * rstd_val;

    vec_t tmp;
    if (gamma != nullptr) {
      #pragma unroll
      for (int ii = 0; ii < vec_bac; ii++) {
        const T_ACC dy = static_cast<T_ACC>(dY_data.val[ii]);
        const T_ACC x = static_cast<T_ACC>(X_data.val[ii]);
        T_ACC gamma_val = static_cast<T_ACC>(gamma_data.val[ii]);
        T_ACC f_grad_input = fH * gamma_val * dy;
        f_grad_input -= (x - mean_val) * rstd_val * stats_x2;
        f_grad_input -= stats_x1;
        f_grad_input *= term1;
        tmp.val[ii] = f_grad_input;
      }
      dX_vec[index]=tmp;
    } else {
      #pragma unroll
      for (int ii = 0; ii < vec_bac; ii++) {
        const T_ACC dy = static_cast<T_ACC>(dY_data.val[ii]);
        const T_ACC x = static_cast<T_ACC>(X_data.val[ii]);
        T_ACC f_grad_input = fH * dy;
        f_grad_input -= (x - mean_val) * rstd_val * stats_x2;
        f_grad_input -= stats_x1;
        f_grad_input *= term1;
        tmp.val[ii] = f_grad_input;
      }
      dX_vec[index]=tmp;
    }

  }


template<int vec_bac, typename T, typename T_ACC>
__global__ void vectorized_layer_norm_grad_input_kernel_pass1(
  const T* __restrict__ dY,
  const T* __restrict__ X,
  const T_ACC* __restrict__ mean,
  const T_ACC* __restrict__ rstd,
  const T* __restrict__ gamma,
  T*  dX,
  const int N,
  const int pow2){
    const int index = threadIdx.x;
    if (index * vec_bac >= N) return;

    alignas(sizeof(double)) extern __shared__ char s_data1[];
    T_ACC * buf = reinterpret_cast<T_ACC*>(&s_data1);
    T_ACC * buf1 = buf + pow2;

    const auto i1 = blockIdx.x;
    const T_ACC mean_val = mean[i1];
    const T_ACC rstd_val = rstd[i1];
    T_ACC stats_x1{0}, stats_x2{0};

    const T *dY_row = dY + i1 * N;
    const T *X_row = X + i1 * N;
    T * dX_row = dX + i1 * N;
    using vec_t = aligned_vector<T, vec_bac>;
    const vec_t* dY_vec = reinterpret_cast<const vec_t*>(dY_row);
    const vec_t* X_vec = reinterpret_cast<const vec_t*>(X_row);
    vec_t* dX_vec = reinterpret_cast<vec_t*>(dX_row);

    vec_t dY_data = dY_vec[index];;
    vec_t X_data =  X_vec[index];
    vec_t gamma_data;
    if (gamma != nullptr) {
      gamma_data = reinterpret_cast<const vec_t*>(gamma)[index];
      #pragma unroll
      for (int ii = 0; ii < vec_bac; ii++) {
        T_ACC gamma_val = static_cast<T_ACC>(gamma_data.val[ii]);
        const T_ACC c_h = static_cast<T_ACC>(X_data.val[ii]);
        const T_ACC c_loss = static_cast<T_ACC>(dY_data.val[ii]);
        stats_x1 += c_loss * gamma_val;
        stats_x2 += c_loss * gamma_val * (c_h - mean_val) * rstd_val;
      }
    } else {
      #pragma unroll
      for (int ii = 0; ii < vec_bac; ii++) {
        const T_ACC c_h = static_cast<T_ACC>(X_data.val[ii]);
        const T_ACC c_loss = static_cast<T_ACC>(dY_data.val[ii]);
        stats_x1 += c_loss;
        stats_x2 += c_loss * (c_h - mean_val) * rstd_val;
      }
    }

    using B = cuda_utils::Block1D;
    const int tid = B::Tid();
    const int lid = tid % C10_WARP_SIZE;
    const int wid = tid / C10_WARP_SIZE;
    //warp reduce
    #pragma unroll
    for (int offset = (C10_WARP_SIZE >> 1); offset > 0; offset >>= 1) {
      stats_x1 += WARP_SHFL_DOWN(stats_x1, offset);
      stats_x2 += WARP_SHFL_DOWN(stats_x2, offset);
    }
    //block reduce
     __syncthreads();
    if (lid == 0) {
      buf[wid] = stats_x1;
      buf1[wid] = stats_x2;
    }
    __syncthreads();
    if (wid == 0) {
      stats_x1 = T_ACC(0);
      stats_x2 = T_ACC(0);
      if (tid < B::Warps()) {
        stats_x1 = buf[tid];
        stats_x2 = buf1[tid];
      }
      #pragma unroll
      for (int offset = (pow2 >> 1); offset > 0; offset >>= 1) {
        stats_x1 += WARP_SHFL_DOWN(stats_x1, offset);
        stats_x2 += WARP_SHFL_DOWN(stats_x2, offset);
      }
    }

    if (threadIdx.x == 0) {
      buf[0] = stats_x1;
      buf[1] = stats_x2;
    }
    __syncthreads();
    stats_x1 = buf[0];
    stats_x2 = buf[1];
    T_ACC fH = N;

    #ifdef USE_MACA
      auto coef =  __builtin_mxc_rcpf(T_ACC(fH));
    #else
      auto coef = (T_ACC(1) / fH);
    #endif
      T_ACC term1 = coef * rstd_val;

    vec_t tmp;
    if (gamma != nullptr) {
      #pragma unroll
      for (int ii = 0; ii < vec_bac; ii++) {
        const T_ACC dy = static_cast<T_ACC>(dY_data.val[ii]);
        const T_ACC x = static_cast<T_ACC>(X_data.val[ii]);
        T_ACC gamma_val = static_cast<T_ACC>(gamma_data.val[ii]);
        T_ACC f_grad_input = fH * gamma_val * dy;
        f_grad_input -= (x - mean_val) * rstd_val * stats_x2;
        f_grad_input -= stats_x1;
        f_grad_input *= term1;
        tmp.val[ii] = f_grad_input;
      }
      dX_vec[index]=tmp;
    } else {
      #pragma unroll
      for (int ii = 0; ii < vec_bac; ii++) {
        const T_ACC dy = static_cast<T_ACC>(dY_data.val[ii]);
        const T_ACC x = static_cast<T_ACC>(X_data.val[ii]);
        T_ACC f_grad_input = fH * dy;
        f_grad_input -= (x - mean_val) * rstd_val * stats_x2;
        f_grad_input -= stats_x1;
        f_grad_input *= term1;
        tmp.val[ii] = f_grad_input;
      }
      dX_vec[index]=tmp;
    }

  }


template<int vec_bac, typename T, typename T_ACC>
__global__ void vectorized_layer_norm_grad_input_kernel_pass2(
  const T* __restrict__ dY,
  const T* __restrict__ X,
  const T_ACC* __restrict__ mean,
  const T_ACC* __restrict__ rstd,
  const T* __restrict__ gamma,
  T*  dX,
  const int N,
  const int pow2){
    alignas(sizeof(double)) extern __shared__ char s_data1[];
    T_ACC * buf = reinterpret_cast<T_ACC*>(&s_data1);
    T_ACC * buf1 = buf + pow2;

    const auto i1 = blockIdx.x;
    const T_ACC mean_val = mean[i1];
    const T_ACC rstd_val = rstd[i1];
    T_ACC stats_x1{0}, stats_x2{0};

    const T *dY_row = dY + i1 * N;
    const T *X_row = X + i1 * N;
    T * dX_row = dX + i1 * N;
    using vec_t = aligned_vector<T, vec_bac>;
    const vec_t* dY_vec = reinterpret_cast<const vec_t*>(dY_row);
    const vec_t* X_vec = reinterpret_cast<const vec_t*>(X_row);
    vec_t* dX_vec = reinterpret_cast<vec_t*>(dX_row);

    const int index0 = threadIdx.x;
    const int index1 = index0 + blockDim.x;
    const bool in_bound = ((index1 * vec_bac) < N);

    vec_t dY_data0 = dY_vec[index0];
    vec_t X_data0 =  X_vec[index0];
    vec_t gamma_data0;
    if (gamma != nullptr) {
      gamma_data0 = reinterpret_cast<const vec_t*>(gamma)[index0];
    }
    vec_t dY_data1;
    vec_t X_data1;
    vec_t gamma_data1;
    if (in_bound) {
      dY_data1 = dY_vec[index1];
      X_data1 = X_vec[index1];
      if (gamma != nullptr) {
        gamma_data1 = reinterpret_cast<const vec_t*>(gamma)[index1];
      }
    }

    if (gamma != nullptr) {
      #pragma unroll
      for (int ii = 0; ii < vec_bac; ii++) {
        T_ACC gamma_val = static_cast<T_ACC>(gamma_data0.val[ii]);
        const T_ACC c_h = static_cast<T_ACC>(X_data0.val[ii]);
        const T_ACC c_loss = static_cast<T_ACC>(dY_data0.val[ii]);
        stats_x1 += c_loss * gamma_val;
        stats_x2 += c_loss * gamma_val * (c_h - mean_val) * rstd_val;
      }
      if (in_bound) {
        #pragma unroll
        for (int ii = 0; ii < vec_bac; ii++) {
          T_ACC gamma_val = static_cast<T_ACC>(gamma_data1.val[ii]);
          const T_ACC c_h = static_cast<T_ACC>(X_data1.val[ii]);
          const T_ACC c_loss = static_cast<T_ACC>(dY_data1.val[ii]);
          stats_x1 += c_loss * gamma_val;
          stats_x2 += c_loss * gamma_val * (c_h - mean_val) * rstd_val;
        }
      }
    } else {
      #pragma unroll
      for (int ii = 0; ii < vec_bac; ii++) {
        const T_ACC c_h = static_cast<T_ACC>(X_data0.val[ii]);
        const T_ACC c_loss = static_cast<T_ACC>(dY_data0.val[ii]);
        stats_x1 += c_loss;
        stats_x2 += c_loss * (c_h - mean_val) * rstd_val;
      }
      if (in_bound) {
        #pragma unroll
        for (int ii = 0; ii < vec_bac; ii++) {
          const T_ACC c_h = static_cast<T_ACC>(X_data1.val[ii]);
          const T_ACC c_loss = static_cast<T_ACC>(dY_data1.val[ii]);
          stats_x1 += c_loss;
          stats_x2 += c_loss * (c_h - mean_val) * rstd_val;
        }
      }
    }

    using B = cuda_utils::Block1D;
    const int tid = B::Tid();
    const int lid = tid % C10_WARP_SIZE;
    const int wid = tid / C10_WARP_SIZE;
    //warp reduce
    #pragma unroll
    for (int offset = (C10_WARP_SIZE >> 1); offset > 0; offset >>= 1) {
      stats_x1 += WARP_SHFL_DOWN(stats_x1, offset);
      stats_x2 += WARP_SHFL_DOWN(stats_x2, offset);
    }
    //block reduce
     __syncthreads();
    if (lid == 0) {
      buf[wid] = stats_x1;
      buf1[wid] = stats_x2;
    }
    __syncthreads();
    if (wid == 0) {
      stats_x1 = T_ACC(0);
      stats_x2 = T_ACC(0);
      if (tid < B::Warps()) {
        stats_x1 = buf[tid];
        stats_x2 = buf1[tid];
      }
      #pragma unroll
      for (int offset = (pow2 >> 1); offset > 0; offset >>= 1) {
        stats_x1 += WARP_SHFL_DOWN(stats_x1, offset);
        stats_x2 += WARP_SHFL_DOWN(stats_x2, offset);
      }
    }

    if (threadIdx.x == 0) {
      buf[0] = stats_x1;
      buf[1] = stats_x2;
    }
    __syncthreads();
    stats_x1 = buf[0];
    stats_x2 = buf[1];
    T_ACC fH = N;

    #ifdef USE_MACA
      auto coef =  __builtin_mxc_rcpf(T_ACC(fH));
    #else
      auto coef = (T_ACC(1) / fH);
    #endif
      T_ACC term1 = coef * rstd_val;

    vec_t tmp;
    if (gamma != nullptr) {
      #pragma unroll
      for (int ii = 0; ii < vec_bac; ii++) {
        const T_ACC dy = static_cast<T_ACC>(dY_data0.val[ii]);
        const T_ACC x = static_cast<T_ACC>(X_data0.val[ii]);
        T_ACC gamma_val = static_cast<T_ACC>(gamma_data0.val[ii]);
        T_ACC f_grad_input = fH * gamma_val * dy;
        f_grad_input -= (x - mean_val) * rstd_val * stats_x2;
        f_grad_input -= stats_x1;
        f_grad_input *= term1;
        tmp.val[ii] = f_grad_input;
      }
      dX_vec[index0]=tmp;
      if (in_bound) {
        #pragma unroll
        for (int ii = 0; ii < vec_bac; ii++) {
          const T_ACC dy = static_cast<T_ACC>(dY_data1.val[ii]);
          const T_ACC x = static_cast<T_ACC>(X_data1.val[ii]);
          T_ACC gamma_val = static_cast<T_ACC>(gamma_data1.val[ii]);
          T_ACC f_grad_input = fH * gamma_val * dy;
          f_grad_input -= (x - mean_val) * rstd_val * stats_x2;
          f_grad_input -= stats_x1;
          f_grad_input *= term1;
          tmp.val[ii] = f_grad_input;
        }
        dX_vec[index1]=tmp;
      }
    } else {
      #pragma unroll
      for (int ii = 0; ii < vec_bac; ii++) {
        const T_ACC dy = static_cast<T_ACC>(dY_data0.val[ii]);
        const T_ACC x = static_cast<T_ACC>(X_data0.val[ii]);
        T_ACC f_grad_input = fH * dy;
        f_grad_input -= (x - mean_val) * rstd_val * stats_x2;
        f_grad_input -= stats_x1;
        f_grad_input *= term1;
        tmp.val[ii] = f_grad_input;
      }
      dX_vec[index0]=tmp;
      if (in_bound) {
        #pragma unroll
        for (int ii = 0; ii < vec_bac; ii++) {
          const T_ACC dy = static_cast<T_ACC>(dY_data1.val[ii]);
          const T_ACC x = static_cast<T_ACC>(X_data1.val[ii]);
          T_ACC f_grad_input = fH * dy;
          f_grad_input -= (x - mean_val) * rstd_val * stats_x2;
          f_grad_input -= stats_x1;
          f_grad_input *= term1;
          tmp.val[ii] = f_grad_input;
        }
        dX_vec[index1]=tmp;
      }
    }

  }


template <typename T, typename T_ACC>
__global__ void GammaBetaBackwardCUDAKernel(
    int64_t M,
    int64_t N,
    const T* dY,
    const T* X,
    const T_ACC* mean,
    const T_ACC* rstd,
    T* dg,
    T* db) {
  alignas(sizeof(double)) extern __shared__ char s_data1[];
  T_ACC* s_data_typed = reinterpret_cast<T_ACC*>(&s_data1);
  T_ACC* s_dg;
  T_ACC* s_db;

  const int64_t j = blockIdx.x * blockDim.x + threadIdx.x;

  T_ACC dg_sum = 0;
  T_ACC db_sum = 0;

  if (j < N) {
    constexpr int unroll_factor = 8;

    T_ACC mean_reg;
    T_ACC rstd_reg;
    T dY_reg;
    T X_reg;

    // Main Loop
    int bcounter;
    for (bcounter = 0; bcounter < M / (blockDim.y * unroll_factor); bcounter++){
      int offset = (bcounter * blockDim.y + threadIdx.y) * unroll_factor;

      #pragma unroll
      for (int ii = 0; ii < unroll_factor; ++ii) {
        dY_reg = dY[(offset + ii) * N + j];
        X_reg = X[(offset + ii) * N + j];
        mean_reg = mean[offset + ii];
        rstd_reg = rstd[offset + ii];
        dg_sum += dY_reg * (X_reg - mean_reg) * rstd_reg;
        db_sum += dY_reg;
      }
    }

    // Remainder loop
    int offset = (bcounter * blockDim.y + threadIdx.y) * unroll_factor;
    for (int ii = 0; ii < unroll_factor; ii++ ){
      if ((offset + ii) < M) {
        dY_reg = dY[(offset + ii) * N + j ];
        X_reg = X[(offset + ii) * N + j];
        mean_reg = mean[offset + ii];
        rstd_reg = rstd[offset + ii];
        dg_sum += dY_reg * (X_reg - mean_reg) * rstd_reg;
        db_sum += dY_reg;
      }
    }

    // Do the final reduction in shared memory
    s_dg = s_data_typed;
    s_db = s_data_typed + blockDim.x * blockDim.y;
    s_dg[threadIdx.y * blockDim.x + threadIdx.x] = dg_sum;
    s_db[threadIdx.y * blockDim.x + threadIdx.x] = db_sum;
    __syncthreads();

    for (int offset = blockDim.y / 2; offset >= 1; offset /= 2) {
      if (threadIdx.y < offset) {
        s_dg[threadIdx.y * blockDim.x + threadIdx.x] +=
            s_dg[(threadIdx.y + offset) * blockDim.x + threadIdx.x];
        s_db[threadIdx.y * blockDim.x + threadIdx.x] +=
            s_db[(threadIdx.y + offset) * blockDim.x + threadIdx.x];
        }
      __syncthreads();
    }

    if (threadIdx.y == 0) {
      if (dg) {
        dg[j] = s_dg[threadIdx.x];
      }
      if (db) {
        db[j] = s_db[threadIdx.x];
      }
    }
  }
}



template <typename T, typename T_ACC>
__global__ void GammaBetaBackwardCUDAKernel_32x32(
    int64_t M,
    int64_t N,
    const T* dY,
    const T* X,
    const T_ACC* mean,
    const T_ACC* rstd,
    T* dg,
    T* db) {
  alignas(sizeof(double)) extern __shared__ char s_data1[];
  T_ACC* s_data_typed = reinterpret_cast<T_ACC*>(&s_data1);
  T_ACC* s_dg;
  T_ACC* s_db;

  T_ACC dg_sum = 0;
  T_ACC db_sum = 0;

  const int64_t j = blockIdx.x * blockDim.x + threadIdx.x;

  if (j < N) {
    constexpr int unroll_factor = 8;
    int laneId = threadIdx.x & 0x1f;

    T_ACC mean_reg, mean_reg_tmp;
    T_ACC rstd_reg, rstd_reg_tmp;
    T dY_reg;
    T X_reg;

    // Main loop
    int bcounter;
    for (bcounter = 0; bcounter < M / (blockDim.y * unroll_factor);
         bcounter++) {
      int offset = (bcounter * blockDim.y + threadIdx.y) * unroll_factor;

      if (laneId < unroll_factor) {
        mean_reg_tmp = mean[offset + laneId];
        rstd_reg_tmp = rstd[offset + laneId];
      }
      WARP_SYNC();

      #pragma unroll
      for (int ii = 0; ii < unroll_factor; ++ii) {
        dY_reg = dY[(offset + ii) * N + j];
        X_reg = X[(offset + ii) * N + j];
        mean_reg = WARP_SHFL(mean_reg_tmp, ii, kWarpSize);
        rstd_reg = WARP_SHFL(rstd_reg_tmp, ii, kWarpSize);
        dg_sum += dY_reg * (X_reg - mean_reg) * rstd_reg;
        db_sum += dY_reg;
      }
    }

    // Remainder loop
    int offset = (bcounter * blockDim.y + threadIdx.y) * unroll_factor;
    for (int ii = 0; ii < unroll_factor; ii++) {
      if ((offset + ii) < M) {
        mean_reg = mean[offset + ii];
        rstd_reg = rstd[offset + ii];
        dY_reg = dY[(offset + ii) * N + j];
        X_reg = X[(offset + ii) * N + j];
        dg_sum += dY_reg * (X_reg - mean_reg) * rstd_reg;
        db_sum += dY_reg;
      }
    }

    // This kernel uses a block of (C10_WARP_SIZE x C10_WARP_SIZE) and
    // gets called when M; N divide by 32. We can use warp shuffles
    // for the final reduction step. This removes 4 shmem loads and
    // stores with their corresponding __syncthreads()

    // This greatly reduces bank conflicts at the expense of a little
    // extra shared memory. It does not impact occupancy
    int padded_bx = (1 + blockDim.x);

    s_dg = s_data_typed;
    s_db = s_data_typed + (padded_bx * blockDim.y);
    s_dg[threadIdx.y * padded_bx + threadIdx.x] = dg_sum;
    s_db[threadIdx.y * padded_bx + threadIdx.x] = db_sum;
    __syncthreads();

    // Load transposed so that a warp holds an entire column
    T_ACC reg_dg = s_dg[threadIdx.x * padded_bx + threadIdx.y];
    T_ACC reg_db = s_db[threadIdx.x * padded_bx + threadIdx.y];
    for (int delta = 16; delta >= 1; delta /= 2) {
      reg_dg += WARP_SHFL_XOR(reg_dg, delta, kWarpSize);
      reg_db += WARP_SHFL_XOR(reg_db, delta, kWarpSize);
    }

    if (threadIdx.x == 0) {
      const int64_t j = blockIdx.x * blockDim.x + threadIdx.y;
      if (dg) {
        dg[j] = reg_dg;
      }
      if (db) {
        db[j] = reg_db;
      }
    }
  }
}



template <int vec, typename T, typename T_ACC>
__global__ void GammaBetaBackwardCUDAKernel_32x32_opt(
    int64_t M,
    int64_t N,
    const T* dY,
    const T* X,
    const T_ACC* mean,
    const T_ACC* rstd,
    T* dg,
    T* db,
    T_ACC* dg_sum_data,
    T_ACC* db_sum_data,
    int* semaphores) {

  alignas(sizeof(double)) extern __shared__ char s_data1[];
  T_ACC* s_data_typed = reinterpret_cast<T_ACC*>(&s_data1);
  int padded_bx = (1 + blockDim.x * vec);
  T_ACC* s_dg = s_data_typed;
  T_ACC* s_db = s_data_typed + (padded_bx * blockDim.y);

  using vec_tcc = aligned_vector<T_ACC, vec>;
  using vec_t = aligned_vector<T, vec>;

  vec_tcc dg_sum;
  vec_tcc db_sum;
  #pragma unroll
  for (int ii = 0; ii < vec; ii++) {
    dg_sum.val[ii] = 0;
    db_sum.val[ii] = 0;
  }

  int64_t j = (blockIdx.x * blockDim.x + threadIdx.x);
  if ((j * vec) < N) {
    int m_offset = blockIdx.y * blockDim.y + threadIdx.y;
    int increment = gridDim.y * blockDim.y;
    T_ACC mean_reg, mean_reg_tmp;
    T_ACC rstd_reg, rstd_reg_tmp;
    vec_t dY_reg;
    vec_t X_reg;

    //thread reduce
    while (m_offset < M) {
      if (threadIdx.x == 0) {
        mean_reg_tmp = mean[m_offset];
        rstd_reg_tmp = rstd[m_offset];
      }
      #if !defined(USER_ROCM)
        __syncwarp();
      #endif

      mean_reg = WARP_SHFL(mean_reg_tmp, 0, kWarpSize);
      rstd_reg = WARP_SHFL(rstd_reg_tmp, 0, kWarpSize);
      dY_reg = (reinterpret_cast<const vec_t*>(dY + m_offset * N))[j];
      X_reg = (reinterpret_cast<const vec_t*>(X + m_offset * N))[j];

      #pragma unroll
      for (int ii = 0; ii < vec; ii++) {
        dg_sum.val[ii] += dY_reg.val[ii] * (X_reg.val[ii] - mean_reg) * rstd_reg;
        db_sum.val[ii] += dY_reg.val[ii];
      }
      m_offset += increment;
    }

    //warp reduce
    (reinterpret_cast<vec_tcc*>(s_dg + threadIdx.y * padded_bx))[threadIdx.x] = dg_sum;
    (reinterpret_cast<vec_tcc*>(s_db + threadIdx.y * padded_bx))[threadIdx.x] = db_sum;
  }
  __syncthreads();

  j = blockIdx.x * blockDim.x + threadIdx.y;
  vec_tcc reg_dg;
  vec_tcc reg_db;
  if (j * vec < N) {
    reg_dg = (reinterpret_cast<vec_tcc*>(s_dg + threadIdx.x * padded_bx))[threadIdx.y];
    reg_db = (reinterpret_cast<vec_tcc*>(s_db + threadIdx.x * padded_bx))[threadIdx.y];
    #pragma unroll
    for (int ii = 0; ii < vec; ii++) {
      for (int delta = 16; delta >= 1; delta /= 2) {
        reg_dg.val[ii] += WARP_SHFL_XOR(reg_dg.val[ii], delta, kWarpSize);
        reg_db.val[ii] += WARP_SHFL_XOR(reg_db.val[ii], delta, kWarpSize);
      }
    }

    if (threadIdx.x == 0) {
      if(dg) {
        (reinterpret_cast<vec_tcc*>(dg_sum_data + blockIdx.y * N))[j] = reg_dg;
      }
      if(db) {
        (reinterpret_cast<vec_tcc*>(db_sum_data + blockIdx.y * N))[j] = reg_db;
      }
    }
  }
  __threadfence();
  __syncthreads();

  //global reduce
  __shared__ bool is_last_block_done;
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    int old = atomicAdd(&semaphores[blockIdx.x], 1);
    is_last_block_done = (old == (gridDim.y-1));
  }
  __syncthreads();

  if (is_last_block_done) {

    j = (blockIdx.x * blockDim.x + threadIdx.x);
    //global to share
    if (j * vec < N) {
      (reinterpret_cast<vec_tcc*>(s_dg + threadIdx.y * padded_bx))[threadIdx.x]
         = (reinterpret_cast<vec_tcc*>(dg_sum_data + threadIdx.y * N))[j];
      (reinterpret_cast<vec_tcc*>(s_db + threadIdx.y * padded_bx))[threadIdx.x]
         = (reinterpret_cast<vec_tcc*>(db_sum_data + threadIdx.y * N))[j];
    }
    __syncthreads();

    j = blockIdx.x * blockDim.x + threadIdx.y;
    if (j * vec < N) {
      reg_dg = (reinterpret_cast<vec_tcc*>(s_dg + threadIdx.x * padded_bx))[threadIdx.y];
      reg_db = (reinterpret_cast<vec_tcc*>(s_db + threadIdx.x * padded_bx))[threadIdx.y];
      #pragma unroll
      for (int ii = 0; ii < vec; ii++) {
        for (int delta = 16; delta >= 1; delta /= 2) {
          reg_dg.val[ii] += WARP_SHFL_XOR(reg_dg.val[ii], delta, kWarpSize);
          reg_db.val[ii] += WARP_SHFL_XOR(reg_db.val[ii], delta, kWarpSize);
        }
      }

      if (threadIdx.x == 0) {
        vec_t reg_dg_tmp;
        vec_t reg_db_tmp;
        for (int ii = 0; ii < vec; ii++) {
          reg_dg_tmp.val[ii] = reg_dg.val[ii];
          reg_db_tmp.val[ii] = reg_db.val[ii];
        }
        if(dg) {
          (reinterpret_cast<vec_t*>(dg))[j] = reg_dg_tmp;
        }
        if(db) {
          (reinterpret_cast<vec_t*>(db))[j] = reg_db_tmp;
        }
      }
    }

  }

}


template <int vec_maca, typename T, typename T_ACC>
__device__ __inline__ void layer_norm_kernel_opt_pass1_noalign_headAlign(
  const int N,
  const int pow2_y,
  T_ACC eps,
  const  T* __restrict__ X,
  const  T* gamma,
  const  T* beta,
  T_ACC* mean,
  T_ACC* rstd,
  T* Y) {
  const int thrx = threadIdx.x + threadIdx.y * blockDim.x;
  int last_index = thrx * vec_maca;
  int last_vec = N - last_index;
  bool is_last_thread = (last_index + vec_maca) >= N;
  if (last_index >= N) return;

  extern __shared__ float s_data[];
  auto i1 = blockIdx.x;
  const T* block_row = X + i1 * N;

  using vec_t = aligned_vector<T, vec_maca>;
  using acc_t = acc_type<T, true>;
  const vec_t* X_vec = reinterpret_cast<const vec_t*>(block_row);
  WelfordDataLN wd(0.f, 0.f, 0.f);

  vec_t data;
  //thread reduce
  if (is_last_thread) {
      for (int ii = 0; ii < last_vec; ii++) {
        data.val[ii] = block_row[last_index + ii];
        wd = cuWelfordOnlineSum(static_cast<acc_t>(data.val[ii]), wd);
      }
  } else {
      data = X_vec[thrx];
      #pragma unroll
      for (int ii = 0; ii < vec_maca; ii++) {
        wd = cuWelfordOnlineSum(static_cast<acc_t>(data.val[ii]), wd);
      }
  }

  // warp reduce
  for (int offset = (C10_WARP_SIZE >> 1); offset > 0; offset >>= 1) {
    WelfordDataLN wdB{
        WARP_SHFL_DOWN(wd.mean, offset),
        WARP_SHFL_DOWN(wd.sigma2, offset),
        WARP_SHFL_DOWN(wd.count, offset)};
    wd = cuWelfordCombine(wd, wdB);
  }
  __syncthreads();

  //block reduce, blockdim.y >1
  float * meanbuf = s_data;
  float * sigmabuf = s_data + pow2_y;
  float * countbuf = s_data + pow2_y * 2;

  using B = cuda_utils::Block2D;
  const int tid = B::Tid();
  const int lid = tid % C10_WARP_SIZE;
  const int wid = tid / C10_WARP_SIZE;
  if (lid == 0) {
    meanbuf[wid] =  wd.mean;
    sigmabuf[wid] = wd.sigma2;
    countbuf[wid] = wd.count;
  }
  __syncthreads();
  if (wid == 0) {
    wd.mean = T_ACC(0);
    wd.sigma2 = T_ACC(0);
    wd.count = T_ACC(0);
    if (tid < B::Warps()) {
      wd.mean = meanbuf[tid];
      wd.sigma2 = sigmabuf[tid];
      wd.count = countbuf[tid];
    }
    #pragma unroll
    for (int offset = (pow2_y >> 1); offset > 0; offset >>= 1) {
      WelfordDataLN wdB{WARP_SHFL_DOWN(wd.mean, offset),
                        WARP_SHFL_DOWN(wd.sigma2, offset),
                        WARP_SHFL_DOWN(wd.count, offset)};
      wd = cuWelfordCombine(wd, wdB);
    }
  }
  if (tid == 0) {
    meanbuf[0] =  wd.mean;
    sigmabuf[0] = wd.sigma2;
    countbuf[0] = wd.count;
  }
  __syncthreads();
  wd.mean = meanbuf[0];
  wd.sigma2 = sigmabuf[0] * __builtin_mxc_rcpf(float(N));
  wd.count = countbuf[0];

  //compute and save
  T* Y_block_row = Y + i1 * N;
  vec_t* Y_vec = reinterpret_cast<vec_t*>(Y_block_row);
  T_ACC rstd_val = c10::cuda::compat::rsqrt(wd.sigma2 + eps);
  const vec_t * gamma_vec = reinterpret_cast<const vec_t*>(gamma);
  const vec_t * beta_vec = reinterpret_cast<const vec_t*>(beta);

    if (is_last_thread) {
      T out;
      if (gamma != nullptr && beta != nullptr) {
        for (int ii = 0; ii < last_vec; ii++) {
          int64_t index = last_index + ii;
          out = static_cast<T_ACC>(gamma[index]) * (rstd_val * (static_cast<T_ACC>(data.val[ii]) - wd.mean))
                + static_cast<T_ACC>(beta[index]);
          Y_block_row[index] = out;
        }
      } else if (gamma != nullptr) {
        for (int ii = 0; ii < last_vec; ii++) {
          int64_t index = last_index + ii;
          out = static_cast<T_ACC>(gamma[index]) * (rstd_val * (static_cast<T_ACC>(data.val[ii]) - wd.mean));
          Y_block_row[index] = out;
        }
      } else if (beta != nullptr) {
        for (int ii = 0; ii < last_vec; ii++) {
          int64_t index = last_index + ii;
          out = (rstd_val * (static_cast<T_ACC>(data.val[ii]) - wd.mean))
                + static_cast<T_ACC>(beta[index]);
          Y_block_row[index] = out;
        }
      } else {
        for (int ii = 0; ii < last_vec; ii++) {
          int64_t index = last_index + ii;
          out = (rstd_val * (static_cast<T_ACC>(data.val[ii]) - wd.mean));
          Y_block_row[index] = out;
        }
      }

  } else {
      vec_t out;
      if (gamma != nullptr && beta != nullptr) {
        vec_t gamma_data = gamma_vec[thrx];
        vec_t beta_data = beta_vec[thrx];
        #pragma unroll
        for (int ii=0; ii < vec_maca; ii++) {
          out.val[ii] = static_cast<T_ACC>(gamma_data.val[ii]) * (rstd_val * (static_cast<T_ACC>(data.val[ii]) - wd.mean))
          + static_cast<T_ACC>(beta_data.val[ii]);
        }
      } else if (gamma != nullptr) {
        vec_t gamma_data = gamma_vec[thrx];
        #pragma unroll
        for (int ii=0; ii < vec_maca; ii++) {
          out.val[ii] = static_cast<T_ACC>(gamma_data.val[ii]) * (rstd_val * (static_cast<T_ACC>(data.val[ii]) - wd.mean));
        }
      } else if (beta != nullptr) {
        vec_t beta_data = beta_vec[thrx];
        #pragma unroll
        for (int ii=0; ii < vec_maca; ii++) {
          out.val[ii] = (rstd_val * (static_cast<T_ACC>(data.val[ii]) - wd.mean))+ static_cast<T_ACC>(beta_data.val[ii]);
        }
      } else {
        #pragma unroll
        for (int ii=0; ii < vec_maca; ii++) {
          out.val[ii] = rstd_val * (static_cast<T_ACC>(data.val[ii]) - wd.mean);
        }
      }
      Y_vec[thrx] = out;
  }

  if (thrx == 0) {
    mean[i1] = wd.mean;
    rstd[i1] = rstd_val;
  }

}


template <int vec_maca, typename T, typename T_ACC>
__device__ __inline__ void layer_norm_kernel_opt_pass1_noalign_tailAlign(
  const int N,
  const int pow2_y,
  T_ACC eps,
  const  T* __restrict__ X,
  const  T* gamma,
  const  T* beta,
  T_ACC* mean,
  T_ACC* rstd,
  T* Y) {
  const int thrx = threadIdx.x + threadIdx.y * blockDim.x;

  int last_index = thrx * vec_maca;
  int last_vec = N % vec_maca;
  bool is_last_thread = (last_index + vec_maca) >= N;
  if (last_index >= N) return;

  extern __shared__ float s_data[];
  auto i1 = blockIdx.x;
  const T* block_row = X + i1 * N;

  using vec_t = aligned_vector<T, vec_maca>;
  using acc_t = acc_type<T, true>;
  const vec_t* X_vec = reinterpret_cast<const vec_t*>(block_row + last_vec);
  WelfordDataLN wd(0.f, 0.f, 0.f);

  vec_t data;
  //thread reduce
  if (is_last_thread) {
      for (int ii = 0; ii < last_vec; ii++) {
        data.val[ii] = block_row[ii];
        wd = cuWelfordOnlineSum(static_cast<acc_t>(data.val[ii]), wd);
      }
  } else {
      data = X_vec[thrx];
      #pragma unroll
      for (int ii = 0; ii < vec_maca; ii++) {
        wd = cuWelfordOnlineSum(static_cast<acc_t>(data.val[ii]), wd);
      }
  }

  // warp reduce
  for (int offset = (C10_WARP_SIZE >> 1); offset > 0; offset >>= 1) {
    WelfordDataLN wdB{
        WARP_SHFL_DOWN(wd.mean, offset),
        WARP_SHFL_DOWN(wd.sigma2, offset),
        WARP_SHFL_DOWN(wd.count, offset)};
    wd = cuWelfordCombine(wd, wdB);
  }
  __syncthreads();

  //block reduce, blockdim.y >1
  float * meanbuf = s_data;
  float * sigmabuf = s_data + pow2_y;
  float * countbuf = s_data + pow2_y * 2;

  using B = cuda_utils::Block2D;
  const int tid = B::Tid();
  const int lid = tid % C10_WARP_SIZE;
  const int wid = tid / C10_WARP_SIZE;
  if (lid == 0) {
    meanbuf[wid] =  wd.mean;
    sigmabuf[wid] = wd.sigma2;
    countbuf[wid] = wd.count;
  }
  __syncthreads();
  if (wid == 0) {
    wd.mean = T_ACC(0);
    wd.sigma2 = T_ACC(0);
    wd.count = T_ACC(0);
    if (tid < B::Warps()) {
      wd.mean = meanbuf[tid];
      wd.sigma2 = sigmabuf[tid];
      wd.count = countbuf[tid];
    }
    #pragma unroll
    for (int offset = (pow2_y >> 1); offset > 0; offset >>= 1) {
      WelfordDataLN wdB{WARP_SHFL_DOWN(wd.mean, offset),
                        WARP_SHFL_DOWN(wd.sigma2, offset),
                        WARP_SHFL_DOWN(wd.count, offset)};
      wd = cuWelfordCombine(wd, wdB);
    }
  }

  if (tid == 0) {
    meanbuf[0] =  wd.mean;
    sigmabuf[0] = wd.sigma2;
    countbuf[0] = wd.count;
  }
  __syncthreads();
  wd.mean = meanbuf[0];
  wd.sigma2 = sigmabuf[0] * __builtin_mxc_rcpf(float(N));
  wd.count = countbuf[0];

  //compute and save
  T* Y_block_row = Y + i1 * N;
  vec_t* Y_vec = reinterpret_cast<vec_t*>(Y_block_row + last_vec);
  T_ACC rstd_val = c10::cuda::compat::rsqrt(wd.sigma2 + eps);
  // const vec_t * gamma_vec = reinterpret_cast<const vec_t*>(gamma + last_vec);
  // const vec_t * beta_vec = reinterpret_cast<const vec_t*>(beta + last_vec);

  if (is_last_thread) {
    T out;
    if (gamma != nullptr && beta != nullptr) {
      for (int ii = 0; ii < last_vec; ii++) {
        out = static_cast<T_ACC>(gamma[ii]) * (rstd_val * (static_cast<T_ACC>(data.val[ii]) - wd.mean))
              + static_cast<T_ACC>(beta[ii]);
        Y_block_row[ii] = out;
      }
    } else if (gamma != nullptr) {
      for (int ii = 0; ii < last_vec; ii++) {
        out = static_cast<T_ACC>(gamma[ii]) * (rstd_val * (static_cast<T_ACC>(data.val[ii]) - wd.mean));
        Y_block_row[ii] = out;
      }
    } else if (beta != nullptr) {
      for (int ii = 0; ii < last_vec; ii++) {
        out = (rstd_val * (static_cast<T_ACC>(data.val[ii]) - wd.mean))
              + static_cast<T_ACC>(beta[ii]);
        Y_block_row[ii] = out;
      }
    } else {
      for (int ii = 0; ii < last_vec; ii++) {
        out = (rstd_val * (static_cast<T_ACC>(data.val[ii]) - wd.mean));
        Y_block_row[ii] = out;
      }
    }

  } else {
    vec_t out;
    int64_t gamma_start = thrx * vec_maca + last_vec;
    if (gamma != nullptr && beta != nullptr) {
      #pragma unroll
      for (int ii=0; ii < vec_maca; ii++) {
        out.val[ii] = static_cast<T_ACC>(gamma[gamma_start+ii]) * (rstd_val * (static_cast<T_ACC>(data.val[ii]) - wd.mean))
        + static_cast<T_ACC>(beta[gamma_start+ii]);
      }
    } else if (gamma != nullptr) {
      #pragma unroll
      for (int ii=0; ii < vec_maca; ii++) {
        out.val[ii] = static_cast<T_ACC>(gamma[gamma_start+ii]) * (rstd_val * (static_cast<T_ACC>(data.val[ii]) - wd.mean));
      }
    } else if (beta != nullptr) {
      #pragma unroll
      for (int ii=0; ii < vec_maca; ii++) {
        out.val[ii] = (rstd_val * (static_cast<T_ACC>(data.val[ii]) - wd.mean))+ static_cast<T_ACC>(beta[gamma_start+ii]);
      }
    } else {
      #pragma unroll
      for (int ii=0; ii < vec_maca; ii++) {
        out.val[ii] = rstd_val * (static_cast<T_ACC>(data.val[ii]) - wd.mean);
      }
    }
    Y_vec[thrx] = out;
  }

  if (thrx == 0) {
    mean[i1] = wd.mean;
    rstd[i1] = rstd_val;
  }

}



template <int vec_maca, typename T, typename T_ACC,
typename std::enable_if<std::is_same<T, double>::value, int>::type = 0>
__global__ __inline__ void layer_norm_kernel_opt_pass1(
  const int N,
  const int pow2_x,
  T_ACC eps,
  const  T* __restrict__ X,
  const  T* gamma,
  const  T* beta,
  T_ACC* mean,
  T_ACC* rstd,
  T* Y){ CUDA_KERNEL_ASSERT(false && "doesn't work with double"); }



template <int vec_maca, typename T, typename T_ACC,
typename std::enable_if<!std::is_same<T, double>::value, int>::type = 0>
__global__ __inline__ void layer_norm_kernel_opt_pass1(
  const int N,
  const int pow2_y,
  T_ACC eps,
  const  T* __restrict__ X,
  const  T* gamma,
  const  T* beta,
  T_ACC* mean,
  T_ACC* rstd,
  T* Y) {
  layer_norm_kernel_opt_pass1_noalign_headAlign<vec_maca>(N, pow2_y, eps, X, gamma, beta, mean, rstd, Y); 
}


template <int vec_maca, typename T, typename T_ACC,
typename std::enable_if<std::is_same<T, double>::value, int>::type = 0>
__global__ __inline__ void layer_norm_kernel_opt_pass1_noalign(
  const int N,
  const int pow2_x,
  T_ACC eps,
  const  T* __restrict__ X,
  const  T* gamma,
  const  T* beta,
  T_ACC* mean,
  T_ACC* rstd,
  T* Y){ CUDA_KERNEL_ASSERT(false && "doesn't work with double"); }

template <int vec_maca, typename T, typename T_ACC,
typename std::enable_if<!std::is_same<T, double>::value, int>::type = 0>
__global__ __inline__ void layer_norm_kernel_opt_pass1_noalign(
  const int N,
  const int pow2_y,
  T_ACC eps,
  const  T* __restrict__ X,
  const  T* gamma,
  const  T* beta,
  T_ACC* mean,
  T_ACC* rstd,
  T* Y) {
  if (blockIdx.x%2==0) {
    layer_norm_kernel_opt_pass1_noalign_headAlign<vec_maca>(N, pow2_y, eps, X, gamma, beta, mean, rstd, Y);
  } else {
    layer_norm_kernel_opt_pass1_noalign_tailAlign<vec_maca>(N, pow2_y, eps, X, gamma, beta, mean, rstd, Y);
  }
}


template <typename T, typename T_ACC>
bool IsOpt_launch_vectorized_layer_norm_kernel(
    int N,
    int64_t M,
    T_ACC eps,
    const T* X_data,
    const T* gamma_data,
    const T* beta_data,
    T* Y_data,
    T_ACC* mean_data,
    T_ACC* rstd_data
) {
    bool IsOpt = false;
    bool disable_vectorized_layernorm_opt = at::maca::get_maca_disable_vectorized_layernorm_opt() || M == 0 || N == 0;
    if (disable_vectorized_layernorm_opt) return IsOpt;

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    const int warp_size = at::cuda::warp_size();

    bool is_float = std::is_same<T, float>::value;
    bool is_half = std::is_same<T, at::Half>::value || std::is_same<T, at::BFloat16>::value;
    const int vec_f = 4;
    const int vec_h = 8;
    const int max_num_threads = 512;

    const dim3 blocks(M);

    bool cond0 = N <= 2 * warp_size;                      //wave
    bool cond1 = N <= 4 * warp_size;                      //wave
    bool cond2 = is_half && (N <= vec_h * warp_size);     //wave

    bool cond3 = is_float && (N <= vec_f * max_num_threads);  //pass1
    bool cond4 = is_half && (N <= vec_h * max_num_threads);   //pass1

    if (cond0) {
      const int vec_maca = 2;
      const dim3 thread1(N / vec_maca, 1);
      const int pow2_x = min_pow2(thread1.x);
      vectorized_layer_norm_kernel_opt_wave1<vec_maca><<<blocks, thread1, 0, stream>>>(N, pow2_x, eps, X_data,
      gamma_data, beta_data, mean_data, rstd_data, Y_data);
      IsOpt = true;
    } else if (cond1) {
      const int vec_maca = 4;
      const dim3 thread1(N / vec_maca, 1);
      const int pow2_x = min_pow2(thread1.x);
      vectorized_layer_norm_kernel_opt_wave1<vec_maca><<<blocks, thread1, 0, stream>>>(N, pow2_x, eps, X_data,
      gamma_data, beta_data, mean_data, rstd_data, Y_data);
      IsOpt = true;
    } else if (cond2) {
      const int vec_maca = 8;
      const dim3 thread1(N / vec_maca, 1);
      const int pow2_x = min_pow2(thread1.x);
      vectorized_layer_norm_kernel_opt_wave1<vec_maca><<<blocks, thread1, 0, stream>>>(N, pow2_x, eps, X_data,
      gamma_data, beta_data, mean_data, rstd_data, Y_data);
      IsOpt = true;
    } else if (cond3) {
      int thread_dim0 = warp_size;
      int thread_dim1 = ((N / vec_f) + warp_size - 1) / warp_size;
      const dim3 thread1(thread_dim0, thread_dim1);
      const int pow2_y = min_pow2(thread1.y);
      int nshared = thread1.y > 1 ? 3 * pow2_y *sizeof(T_ACC) : 0;
      vectorized_layer_norm_kernel_opt_pass1<vec_f><<<blocks, thread1, nshared, stream>>>(N, pow2_y, eps, X_data,
      gamma_data, beta_data, mean_data, rstd_data, Y_data);
      IsOpt = true;
    } else if (cond4) {
      int thread_dim0 = warp_size;
      int thread_dim1 = ((N / vec_h) + warp_size - 1) / warp_size;
      const dim3 thread1(thread_dim0, thread_dim1);
      const int pow2_y = min_pow2(thread1.y);
      int nshared = thread1.y > 1 ? 3 * pow2_y *sizeof(T_ACC) : 0;
      vectorized_layer_norm_kernel_opt_pass1<vec_h><<<blocks, thread1, nshared, stream>>>(N, pow2_y, eps, X_data,
      gamma_data, beta_data, mean_data, rstd_data, Y_data);
      IsOpt = true;
    } else {
      int vec_maca = 16 / sizeof(T);
      int thread_dim0 = N / vec_maca < warp_size ? N / vec_maca : warp_size;
      int thread_dim1 = N <= warp_size * vec_maca ? 1 : ((N / vec_maca) + warp_size - 1) / warp_size;
      thread_dim1 = std::min(thread_dim1, int(max_num_threads / thread_dim0));
      const dim3 threads(thread_dim0, thread_dim1, 1);
      const dim3 blocks(M);
      const int pow2 = max_num_threads / warp_size;
      int nshared = threads.y > 1 ? 3 * pow2 *sizeof(T_ACC) : 0;

      if (vec_maca == 8) {
        vectorized_layer_norm_kernel_opt<8><<<blocks, threads, nshared, stream>>>(N, pow2, eps, X_data,
        gamma_data, beta_data, mean_data, rstd_data, Y_data);
      } else if(vec_maca == 4) {
        vectorized_layer_norm_kernel_opt<4><<<blocks, threads, nshared, stream>>>(N, pow2, eps, X_data,
        gamma_data, beta_data, mean_data, rstd_data, Y_data);
      } else {
        assert(0);
      }
      IsOpt = true;
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return IsOpt;
}


template <typename T, typename T_ACC>
bool IsOpt_launch_layer_norm_kernel(
    int N,
    int64_t M,
    T_ACC eps,
    const T* X_data,
    const T* gamma_data,
    const T* beta_data,
    T* Y_data,
    T_ACC* mean_data,
    T_ACC* rstd_data
) {
    bool IsOpt = false;
    bool disable_vectorized_layernorm_opt = at::maca::get_maca_disable_vectorized_layernorm_opt() || M == 0 || N == 0;
    if (disable_vectorized_layernorm_opt) return IsOpt;
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    const int warp_size = at::cuda::warp_size();

    bool is_float = std::is_same<T, float>::value;
    bool is_half = std::is_same<T, at::Half>::value || std::is_same<T, at::BFloat16>::value;
    const int vec_f = 4;
    const int vec_h = 8;
    const int max_num_threads = 512;

    bool align0 = reinterpret_cast<uintptr_t>(X_data) % 4 == 0 &&
                  reinterpret_cast<uintptr_t>(gamma_data) % 4 == 0 &&
                  reinterpret_cast<uintptr_t>(beta_data) % 4 == 0 &&
                  reinterpret_cast<uintptr_t>(Y_data) % 4 == 0;

    //float one block once
    bool cond3 = is_float && (N <= vec_f * max_num_threads) && align0 &&
                 (N > vec_f * warp_size) && N % vec_f != 0;
    //half one block once
    bool cond4 = is_half && (N <= vec_h * max_num_threads) && N % 2 == 0 && align0 &&
                 (N > vec_h * warp_size) && N % vec_h != 0;

    //half one block once, address no align
    bool cond5 = is_half && (N <= vec_h * max_num_threads) && N % 2 != 0 && align0 &&
                 (N > vec_h * warp_size) && N % vec_h != 0;

    const dim3 blocks(M);
    if (cond3) {
      int thread_dim0 = warp_size;
      int thread_dim1 = (N + (warp_size * vec_f) -1) / (warp_size * vec_f);
      const dim3 thread1(thread_dim0, thread_dim1);
      const int pow2_y = min_pow2(thread1.y);
      int nshared = thread1.y > 1 ? 3 * pow2_y *sizeof(T_ACC) : 0;

      layer_norm_kernel_opt_pass1<vec_f><<<blocks, thread1, nshared, stream>>>(N, pow2_y, eps, X_data,
      gamma_data, beta_data, mean_data, rstd_data, Y_data);

      IsOpt = true;
    } else if (cond4) {
      int thread_dim0 = warp_size;
      int thread_dim1 = (N + (warp_size * vec_h) -1) / (warp_size * vec_h);
      const dim3 thread1(thread_dim0, thread_dim1);
      const int pow2_y = min_pow2(thread1.y);
      int nshared = thread1.y > 1 ? 3 * pow2_y *sizeof(T_ACC) : 0;

      layer_norm_kernel_opt_pass1<vec_h><<<blocks, thread1, nshared, stream>>>(N, pow2_y, eps, X_data,
      gamma_data, beta_data, mean_data, rstd_data, Y_data);

      IsOpt = true;
    } else if (cond5) {
      int thread_dim0 = warp_size;
      int thread_dim1 = (N + (warp_size * vec_h) -1) / (warp_size * vec_h);
      const dim3 thread1(thread_dim0, thread_dim1);
      const int pow2_y = min_pow2(thread1.y);
      int nshared = thread1.y > 1 ? 3 * pow2_y *sizeof(T_ACC) : 0;

      layer_norm_kernel_opt_pass1_noalign<vec_h><<<blocks, thread1, nshared, stream>>>(N, pow2_y, eps, X_data,
      gamma_data, beta_data, mean_data, rstd_data, Y_data);

      IsOpt = true;
    }

    return IsOpt;
}


}}