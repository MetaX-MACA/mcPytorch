
namespace at::native {
template <typename scalar_t, typename F>
struct DistReduceOp;
namespace distance {
template<typename scalar_t, int vec_size>
struct alignas(sizeof(scalar_t) * vec_size) aligned_vector {
  scalar_t val[vec_size];
};
template <typename scalar_t>
struct dists {
  static __forceinline__ __device__ scalar_t sign(scalar_t val) {
    return (0 < val) - (val < 0);
  }
  // Zero norm
  struct zero {
    static __forceinline__ __device__ void inc(scalar_t& agg, const scalar_t diff, const scalar_t /*p*/) { agg += diff != 0.0; }
    static __forceinline__ __device__ scalar_t finish(const scalar_t agg, const scalar_t /*p*/) { return agg; }
    static __forceinline__ __device__ void agg(scalar_t& update, const scalar_t other) { update += other; }
  };
  // One norm
  struct one {
    static __forceinline__ __device__ void inc(scalar_t& agg, const scalar_t diff, const scalar_t /*p*/) { agg += diff; }
    static __forceinline__ __device__ scalar_t finish(const scalar_t agg, const scalar_t /*p*/) { return agg; }
    static __forceinline__ __device__ void agg(scalar_t& update, const scalar_t other) { update += other; }
    static __forceinline__ __device__ scalar_t backward(const scalar_t diff, const scalar_t grad, const scalar_t /*dist*/, const scalar_t /*p*/) { return grad * sign(diff); }
  };
  // Special case backward when p is less than two
  struct lt_two {
    static __forceinline__ __device__ scalar_t backward(const scalar_t diff, const scalar_t grad, const scalar_t dist, const scalar_t p) {
      return (dist == 0.0 || (diff == 0.0 && p < 1)) ? 0 : (sign(diff) * std::pow(std::abs(diff), p - 1) * grad / std::pow(dist, p - 1));
    }
  };
  // Two norm
  struct two {
    static __forceinline__ __device__ void inc(scalar_t& agg, const scalar_t diff, const scalar_t /*p*/) { agg += diff * diff; }
    static __forceinline__ __device__ scalar_t finish(const scalar_t agg, const scalar_t /*p*/) { return device_sqrt<scalar_t>(agg); }
    static __forceinline__ __device__ void agg(scalar_t& update, const scalar_t other) { update += other; }
    static __forceinline__ __device__ scalar_t backward(const scalar_t diff, const scalar_t grad, const scalar_t dist, const scalar_t /*p*/) { return dist == 0.0 ? 0 : grad * diff / dist; }
  };
  // General p norm
  struct p {
    static __forceinline__ __device__ void inc(scalar_t& agg, const scalar_t diff, const scalar_t p) { agg += std::pow(diff, p); }
    static __forceinline__ __device__ scalar_t finish(const scalar_t agg, const scalar_t p) { return std::pow(agg, static_cast<scalar_t>(1) / p); }
    static __forceinline__ __device__ void agg(scalar_t& update, const scalar_t other) { update += other; }
    static __forceinline__ __device__ scalar_t backward(const scalar_t diff, const scalar_t grad, const scalar_t dist, const scalar_t p) { return dist == 0.0 ? 0 : diff * std::pow(std::abs(diff), p - 2) * grad / std::pow(dist, p - 1); }
  };
  // Inf norm
  struct inf {
    static __forceinline__ __device__ void inc(scalar_t& agg, const scalar_t diff, const scalar_t /*p*/) { if (diff > agg) { agg = diff; } }
    static __forceinline__ __device__ scalar_t finish(const scalar_t agg, const scalar_t /*p*/) { return agg; }
    static __forceinline__ __device__ void agg(scalar_t& update, const scalar_t other) { if (other > update) { update = other; } }
    static __forceinline__ __device__ scalar_t backward(const scalar_t diff, const scalar_t grad, const scalar_t dist, const scalar_t /*p*/) { return grad * sign(diff) * (std::abs(diff) == dist); }
  };
};
template <typename scalar_t, typename F>
struct DistReduceOp {
    __forceinline__ __device__ scalar_t combine(scalar_t a, scalar_t b) const {
        F::agg(a, b);
        return a;
    }
    __forceinline__ __device__ scalar_t warp_shfl_down(scalar_t data, int offset) const {
        return WARP_SHFL_DOWN(data, offset);
    }
};
template <int vec, typename scalar_t, typename F>
__global__ static void cdist_forward_wave_opt(
    scalar_t * result, const scalar_t * x1, const scalar_t * x2,
    const scalar_t p,
    const int64_t r1, const int64_t r2, const int64_t m,
    const int64_t r_size, const int64_t l1_size, const int64_t l2_size,
    const int64_t swarp, const int64_t swarp_pow, const int64_t line_perwarp, const int64_t line_perwarp_pow
) {
    using vec_t = aligned_vector<scalar_t, vec>;
    int64_t l = blockIdx.x;
    int64_t k = blockIdx.y * line_perwarp_pow;
    int64_t tid = threadIdx.x;
    int64_t k_tid = k + (tid >> swarp);
    if (k_tid >= r_size) return;
    int64_t i = k_tid / r2;
    int64_t j = k_tid % r2;
    const scalar_t * const x1_start = x1 + l * l1_size + i * m;
    const scalar_t * const x2_start = x2 + l * l2_size + j * m;
    int tid_in_swarp = tid - (tid >> swarp) * swarp_pow;
    scalar_t agg = 0.0;
    if(tid_in_swarp * vec < m) {
        vec_t vec_x1 =  (reinterpret_cast<const vec_t*>(x1_start))[tid_in_swarp];
        vec_t vec_x2 =  (reinterpret_cast<const vec_t*>(x2_start))[tid_in_swarp];
        #pragma unroll
        for (int ii = 0; ii < vec; ii++) {
            F::inc(agg, std::abs(vec_x1.val[ii] - vec_x2.val[ii]), p);
        }
    }
    auto op = DistReduceOp<scalar_t, F>{};
    #pragma unroll
    for (int offset = (swarp_pow >> 1); offset > 0; offset >>= 1) {
        agg = op.combine(agg, op.warp_shfl_down(agg, offset));
    }
    if (tid_in_swarp == 0) {
        result[l * r_size + k_tid] = F::finish(agg, p);
    }
}
bool is_opt_cdist_forward_wave(Tensor& result,
                          const Tensor& x1,
                          const Tensor& x2,
                          double p,
                          const int64_t r1,
                          const int64_t r2,
                          const int64_t m,
                          const int64_t l1_size,
                          const int64_t l2_size,
                          const int64_t r_size
                          ) {
    bool is_opt = !at::maca::get_maca_disable_cdist_wave_opt() &&
                  x1.numel() > 0 && x2.numel() > 0 && result.numel() > 0 &&
                  (x1.scalar_type() == x2.scalar_type()) && (result.scalar_type() == x2.scalar_type()) &&
                  (x1.scalar_type() == at::kBFloat16 || x1.scalar_type() == at::kHalf || x1.scalar_type() == at::kFloat) &&
                  x1.is_contiguous() && x2.is_contiguous() && result.is_contiguous() &&
                  m <= 256 && m % 4 == 0;
    if (!is_opt) return false;
    int vec, swarp, swarp_pow, line_perwarp, line_perwarp_pow;
    if (m > 0 && m <= 4) {
        if        (r_size >= 53 * 64)                     {  vec = 4; swarp = 0; swarp_pow = 1; line_perwarp = 6; line_perwarp_pow = 64;
        } else if (r_size >= 53 * 32 && r_size < 53 * 64) {  vec = 2; swarp = 1; swarp_pow = 2; line_perwarp = 5; line_perwarp_pow = 32;
        } else if (r_size >= 53 * 16 && r_size < 53 * 32) {  vec = 1; swarp = 2; swarp_pow = 4; line_perwarp = 4; line_perwarp_pow = 16;
        } else if (r_size >= 53 * 8 && r_size < 53 * 16)  {  vec = 1; swarp = 2; swarp_pow = 4; line_perwarp = 3; line_perwarp_pow = 8;
        } else if (r_size >= 53 * 4 && r_size < 53 * 8)   {  vec = 1; swarp = 2; swarp_pow = 4; line_perwarp = 2; line_perwarp_pow = 4;
        } else if (r_size >= 53 * 2 && r_size < 53 * 4)   {  vec = 1; swarp = 2; swarp_pow = 4; line_perwarp = 1; line_perwarp_pow = 2;
        } else if (r_size < 53 * 2 )                      {  vec = 1; swarp = 2; swarp_pow = 4; line_perwarp = 0; line_perwarp_pow = 1;
        } else { assert(0); }
    } else if (m > 4 && m <= 8) {
        if        (r_size >= 53 * 32)                     {  vec = 4; swarp = 1; swarp_pow = 2; line_perwarp = 5; line_perwarp_pow = 32;
        } else if (r_size >= 53 * 16 && r_size < 53 * 32) {  vec = 2; swarp = 2; swarp_pow = 4; line_perwarp = 4; line_perwarp_pow = 16;
        } else if (r_size >= 53 * 8 && r_size < 53 * 16)  {  vec = 1; swarp = 3; swarp_pow = 8; line_perwarp = 3; line_perwarp_pow = 8;
        } else if (r_size >= 53 * 4 && r_size < 53 * 8)   {  vec = 1; swarp = 3; swarp_pow = 8; line_perwarp = 2; line_perwarp_pow = 4;
        } else if (r_size >= 53 * 2 && r_size < 53 * 4)   {  vec = 1; swarp = 3; swarp_pow = 8; line_perwarp = 1; line_perwarp_pow = 2;
        } else if (r_size < 53 * 2 )                      {  vec = 1; swarp = 3; swarp_pow = 8; line_perwarp = 0; line_perwarp_pow = 1;
        } else { assert(0); }
    } else if (m > 8 && m <= 16) {
        if        (r_size >= 53 * 16)                     {  vec = 4; swarp = 2; swarp_pow = 4; line_perwarp = 4; line_perwarp_pow = 16;
        } else if (r_size >= 53 * 8 && r_size < 53 * 16)  {  vec = 2; swarp = 3; swarp_pow = 8; line_perwarp = 3; line_perwarp_pow = 8;
        } else if (r_size >= 53 * 4 && r_size < 53 * 8)   {  vec = 1; swarp = 4; swarp_pow = 16;line_perwarp = 2; line_perwarp_pow = 4;
        } else if (r_size >= 53 * 2 && r_size < 53 * 4)   {  vec = 1; swarp = 4; swarp_pow = 16;line_perwarp = 1; line_perwarp_pow = 2;
        } else if (r_size < 53 * 2 )                      {  vec = 1; swarp = 4; swarp_pow = 16;line_perwarp = 0; line_perwarp_pow = 1;
        } else { assert(0); }
    } else if (m > 16 && m <= 32) {
        if        (r_size >= 53 * 8)                      {  vec = 4; swarp = 3; swarp_pow = 8; line_perwarp = 3; line_perwarp_pow = 8;
        } else if (r_size >= 53 * 4 && r_size < 53 * 8)   {  vec = 2; swarp = 4; swarp_pow = 16;line_perwarp = 2; line_perwarp_pow = 4;
        } else if (r_size >= 53 * 2 && r_size < 53 * 4)   {  vec = 1; swarp = 5; swarp_pow = 32;line_perwarp = 1; line_perwarp_pow = 2;
        } else if (r_size < 53 * 2 )                      {  vec = 1; swarp = 5; swarp_pow = 32;line_perwarp = 0; line_perwarp_pow = 1;
        } else { assert(0); }
    } else if (m > 32 && m <= 64) {
        if        (r_size >= 53 * 4)                      {  vec = 4; swarp = 4; swarp_pow = 16;line_perwarp = 2; line_perwarp_pow = 4;
        } else if (r_size >= 53 * 2 && r_size < 53 * 4)   {  vec = 2; swarp = 5; swarp_pow = 32;line_perwarp = 1; line_perwarp_pow = 2;
        } else if (r_size < 53 * 2 )                      {  vec = 1; swarp = 6; swarp_pow = 64;line_perwarp = 0; line_perwarp_pow = 1;
        } else { assert(0); }
    } else if (m > 64 && m <= 128) {
        if        (r_size >= 53 * 2)                      {  vec = 4; swarp = 5; swarp_pow = 32;line_perwarp = 1; line_perwarp_pow = 2;
        } else if (r_size < 53 * 2 )                      {  vec = 2; swarp = 6; swarp_pow = 64;line_perwarp = 0; line_perwarp_pow = 1;
        } else { assert(0); }
    } else if (m > 128 && m <= 256) {                        vec = 4; swarp = 6; swarp_pow = 64;line_perwarp = 0; line_perwarp_pow = 1;
    } else { assert(0) ;}
    int gridx = result.numel() / r_size;
    int gridy = ((r_size + line_perwarp_pow - 1) / line_perwarp_pow);
    dim3 grid(gridx, gridy);
    int block = C10_WARP_SIZE;
    AT_DISPATCH_FLOATING_TYPES(x1.scalar_type(), "cdist_cuda", [&] {
        if (vec == 4) {
            auto impl_fptr = cdist_forward_wave_opt<4, scalar_t, dists<scalar_t>::p>;
            if (p == 0.0) {
                impl_fptr = cdist_forward_wave_opt<4, scalar_t, dists<scalar_t>::zero>;
            } else if (p == 1.0) {
                impl_fptr = cdist_forward_wave_opt<4, scalar_t, dists<scalar_t>::one>;
            } else if (p == 2.0) {
                impl_fptr = cdist_forward_wave_opt<4, scalar_t, dists<scalar_t>::two>;
            } else if (std::isinf(p)) {
                impl_fptr = cdist_forward_wave_opt<4, scalar_t, dists<scalar_t>::inf>;
            }
            impl_fptr<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
                result.mutable_data_ptr<scalar_t>(),  x1.const_data_ptr<scalar_t>(),  x2.const_data_ptr<scalar_t>(),
                p,
                r1,      r2,       m,
                r_size,  l1_size,  l2_size,
                swarp, swarp_pow, line_perwarp, line_perwarp_pow
            );
        } else if (vec == 2) {
          auto impl_fptr = cdist_forward_wave_opt<2, scalar_t, dists<scalar_t>::p>;
            if (p == 0.0) {
                impl_fptr = cdist_forward_wave_opt<2, scalar_t, dists<scalar_t>::zero>;
            } else if (p == 1.0) {
                impl_fptr = cdist_forward_wave_opt<2, scalar_t, dists<scalar_t>::one>;
            } else if (p == 2.0) {
                impl_fptr = cdist_forward_wave_opt<2, scalar_t, dists<scalar_t>::two>;
            } else if (std::isinf(p)) {
                impl_fptr = cdist_forward_wave_opt<2, scalar_t, dists<scalar_t>::inf>;
            }
            impl_fptr<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
                result.mutable_data_ptr<scalar_t>(),  x1.const_data_ptr<scalar_t>(),  x2.const_data_ptr<scalar_t>(),
                p,
                r1,      r2,       m,
                r_size,  l1_size,  l2_size,
                swarp, swarp_pow, line_perwarp, line_perwarp_pow
            );
        } else if (vec == 1) {
            auto impl_fptr = cdist_forward_wave_opt<1, scalar_t, dists<scalar_t>::p>;
            if (p == 0.0) {
                impl_fptr = cdist_forward_wave_opt<1, scalar_t, dists<scalar_t>::zero>;
            } else if (p == 1.0) {
                impl_fptr = cdist_forward_wave_opt<1, scalar_t, dists<scalar_t>::one>;
            } else if (p == 2.0) {
                impl_fptr = cdist_forward_wave_opt<1, scalar_t, dists<scalar_t>::two>;
            } else if (std::isinf(p)) {
                impl_fptr = cdist_forward_wave_opt<1, scalar_t, dists<scalar_t>::inf>;
            }
            impl_fptr<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
                result.mutable_data_ptr<scalar_t>(),  x1.const_data_ptr<scalar_t>(),  x2.const_data_ptr<scalar_t>(),
                p,
                r1,      r2,       m,
                r_size,  l1_size,  l2_size,
                swarp, swarp_pow, line_perwarp, line_perwarp_pow
            );
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    });
    return true;
}
}
}

