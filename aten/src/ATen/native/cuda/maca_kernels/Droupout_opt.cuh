
namespace at::native {

namespace memory {
template<typename scalar_t>
inline C10_HOST_DEVICE int can_vectorize_up_to_(char *pointer) {
  uint64_t address = reinterpret_cast<uint64_t>(pointer);
  constexpr int vec8_alignment = std::alignment_of<aligned_vector<scalar_t, 8>>::value;
  if (sizeof(scalar_t) == 2 && address % vec8_alignment == 0) {
    return 8;
  } else {
    return can_vectorize_up_to<scalar_t>(pointer);
  }
}
}

namespace dropout {
template <
    typename scalar_t,
    typename accscalar_t,
    typename IndexType,
    int ADims,
    int VEC,
    typename mask_t,
    std::enable_if_t<VEC == 8, int> = 0>
#if __CUDA_ARCH__ >= 350 || defined(USE_ROCM)
C10_LAUNCH_BOUNDS_2(256, 4)
#endif
__global__ void
fused_dropout_kernel_vec(at::cuda::detail::TensorInfo<const scalar_t, IndexType> a,
                         at::cuda::detail::TensorInfo<scalar_t, IndexType> b,
                         at::cuda::detail::TensorInfo<mask_t, IndexType> c,
                         IndexType totalElements, accscalar_t p,
                         PhiloxCudaState philox_args) {
  static_assert(VEC == 8, "Value of VEC must be 8");

  using LoadT = memory::aligned_vector<scalar_t, VEC>;
  using MaskLoadT = memory::aligned_vector<mask_t, VEC>;

  auto seeds = at::cuda::philox::unpack(philox_args);
  IndexType idx = blockIdx.x * blockDim.x + threadIdx.x;
  curandStatePhilox4_32_10_t state;
  curand_init(std::get<0>(seeds),
              idx,
              std::get<1>(seeds),
              &state);
  
  // Helps align the total number of times curand_uniform4 is called by each thread for the same totalElements
  // in the vec=2 and vec=4 cases.
  accscalar_t scale = 1.0 / p;

  float4 rand0;
  float4 rand1;

  // Note: Vectorized loads means we'll stride each thread by an additional VEC factor, as we'll load VEC elements at a time
  for (IndexType linearIndex = idx * VEC;
      linearIndex < totalElements;
      linearIndex += gridDim.x * blockDim.x * VEC) {
    // local storage
    scalar_t src[VEC];
    // We'll use this to actually cause vectorized loads later
    LoadT *value = reinterpret_cast<LoadT*>(&src);

    //curand_uniform_double was pure evil anyway, not doing what it promises, and there's nothing for halfs, so generate float for everything
    // Note: need a new set of random values per 4 elements -- we'll handle VEC elements in this thread, so need ceil(VEC / 4)
    // sets of rand.

    rand0 = curand_uniform4(&state);
    rand0.x = rand0.x < p;
    rand0.y = rand0.y < p;
    rand0.z = rand0.z < p;
    rand0.w = rand0.w < p;
    rand1 = curand_uniform4(&state);
    rand1.x = rand1.x < p;
    rand1.y = rand1.y < p;
    rand1.z = rand1.z < p;
    rand1.w = rand1.w < p;

    // Note: We explicitly check for is_contiguous() before launching the vectorized kernel
    // and replace IndexToOffset call with linearIndex to allow vectorization of NHWC (or other)
    // ordering.
    // Single vectorized load
    *value = *reinterpret_cast<const LoadT*>(&a.data[linearIndex]);

    scalar_t r[VEC];
    mask_t mask[VEC];

    // Perform the actual computation
    #pragma unroll
    for (int ii = 0; ii < 4; ii++) {
      r[ii] = src[ii]*(&rand0.x)[ii]*scale;
      r[ii + 4] = src[ii + 4]*(&rand1.x)[ii]*scale;
      mask[ii] = (mask_t)(&rand0.x)[ii];
      mask[ii + 4] = (mask_t)(&rand1.x)[ii];
    }

    // Vectorized writes for both mask & result
    *(reinterpret_cast<LoadT*>(&b.data[linearIndex])) = *reinterpret_cast<LoadT*>(&r[0]);
    *(reinterpret_cast<MaskLoadT*>(&c.data[linearIndex])) = *reinterpret_cast<MaskLoadT*>(&mask[0]);

    __syncthreads();
  }
}

} //dropout

} //at::native