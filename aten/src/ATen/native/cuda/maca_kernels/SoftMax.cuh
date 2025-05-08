template <template<typename> class ReduceOp, typename T>
__forceinline__ __device__ T WarpReduce(T val) {
  ReduceOp<T> r;
#pragma unroll
  for (int offset = (C10_WARP_SIZE >> 1); offset > 0; offset >>= 1) {
    T t = WARP_SHFL_XOR(val, offset, C10_WARP_SIZE);
    val = r(val, t);
  }
  return val;
}

template <template<typename> class ReduceOp, typename T>
__forceinline__ __device__ T
BlockReduce_opt(T val, T* shared, int warp_num) {
  ReduceOp<T> r;
  const int lid = threadIdx.x % C10_WARP_SIZE;
  const int wid = threadIdx.x / C10_WARP_SIZE;
  val = WarpReduce<ReduceOp, T>(val);
  __syncthreads();
  if (lid == 0) {
    shared[wid] = val;
  }
  __syncthreads();

  val = shared[0];
  #pragma unroll
  for(int i = 1; i < warp_num; ++i){
    val = r(val, shared[i]);
  }

  return val;
}

template <template<typename> class ReduceOp, typename T, int warp_num>
__forceinline__ __device__ T
BlockReduce_opt(T val, T* shared) {
  ReduceOp<T> r;
  const int lid = threadIdx.x % C10_WARP_SIZE;
  const int wid = threadIdx.x / C10_WARP_SIZE;
  val = WarpReduce<ReduceOp, T>(val);
  __syncthreads();
  if (lid == 0) {
    shared[wid] = val;
  }
  __syncthreads();

  val = shared[0];
  #pragma unroll
  for(int i = 1; i < warp_num; ++i){
    val = r(val, shared[i]);
  }

  return val;
}


template <template<typename> class ReduceOp, typename T,
         int warp_num,
         typename std::enable_if<warp_num == 8>::type,
         typename std::enable_if<std::is_same<T, float>::value || std::is_same<T, __half>::value>::type,
         typename std::enable_if<std::is_same<ReduceOp<T>, Add<T>>::value>::type>
__forceinline__ __device__ T
BlockReduce_opt(T val, T* shared) {
  ReduceOp<T> r;
  const int lid = threadIdx.x % C10_WARP_SIZE;
  const int wid = threadIdx.x / C10_WARP_SIZE;
  // val = WarpReduce<ReduceOp, T>(val);
  val = __reduce_add_sync(0xffffffffffffffff, val);
  __syncthreads();
  if (lid == 0) {
    shared[wid] = val;
  }
  __syncthreads();

  float4 vec = (reinterpret_cast<float4*>(shared))[0];
  float4 vec_1 = (reinterpret_cast<float4*>(shared))[1];
  val = r(vec.x, vec.y);
  val = r(val, vec.z);
  val = r(val, vec.w);
  val = r(val, vec_1.x);
  val = r(val, vec_1.y);
  val = r(val, vec_1.z);
  val = r(val, vec_1.w);
  return val;
}

template <template<typename> class ReduceOp, typename T,
         int warp_num,
         typename std::enable_if<warp_num == 8>::type,
         typename std::enable_if<std::is_same<T, float>::value || std::is_same<T, __half>::value>::type,
         typename std::enable_if<std::is_same<ReduceOp<T>, Max<T>>::value>::type>
__forceinline__ __device__ T
BlockReduce_opt(T val, T* shared) {
  ReduceOp<T> r;
  const int lid = threadIdx.x % C10_WARP_SIZE;
  const int wid = threadIdx.x / C10_WARP_SIZE;
  // val = WarpReduce<ReduceOp, T>(val);
  val = __reduce_max_sync(0xffffffffffffffff, val);
  __syncthreads();
  if (lid == 0) {
    shared[wid] = val;
  }
  __syncthreads();

  float4 vec = (reinterpret_cast<float4*>(shared))[0];
  float4 vec_1 = (reinterpret_cast<float4*>(shared))[1];
  val = r(vec.x, vec.y);
  val = r(val, vec.z);
  val = r(val, vec.w);
  val = r(val, vec_1.x);
  val = r(val, vec_1.y);
  val = r(val, vec_1.z);
  val = r(val, vec_1.w);
  return val;
}


template <template<typename> class ReduceOp, typename T,
         int warp_num,
         typename std::enable_if<warp_num == 4>::type,
         typename std::enable_if<std::is_same<T, float>::value || std::is_same<T, __half>::value>::type,
         typename std::enable_if<std::is_same<ReduceOp<T>, Add<T>>::value>::type>
__forceinline__ __device__ T
BlockReduce_opt(T val, T* shared) {
  ReduceOp<T> r;
  const int lid = threadIdx.x % C10_WARP_SIZE;
  const int wid = threadIdx.x / C10_WARP_SIZE;
  // val = WarpReduce<ReduceOp, T>(val);
  val = __reduce_add_sync(0xffffffffffffffff, val);
  __syncthreads();
  if (lid == 0) {
    shared[wid] = val;
  }
  __syncthreads();

  float4 vec = (reinterpret_cast<float4*>(shared))[0];
  val = r(vec.x, vec.y);
  val = r(val, vec.z);
  val = r(val, vec.w);
  return val;
}

template <template<typename> class ReduceOp, typename T,
         int warp_num,
         typename std::enable_if<warp_num == 4>::type,
         typename std::enable_if<std::is_same<T, float>::value || std::is_same<T, __half>::value>::type,
         typename std::enable_if<std::is_same<ReduceOp<T>, Max<T>>::value>::type>
__forceinline__ __device__ T
BlockReduce_opt(T val, T* shared) {
  ReduceOp<T> r;
  const int lid = threadIdx.x % C10_WARP_SIZE;
  const int wid = threadIdx.x / C10_WARP_SIZE;
  // val = WarpReduce<ReduceOp, T>(val);
  val = __reduce_max_sync(0xffffffffffffffff, val);
  __syncthreads();
  if (lid == 0) {
    shared[wid] = val;
  }
  __syncthreads();

  float4 vec = (reinterpret_cast<float4*>(shared))[0];
  val = r(vec.x, vec.y);
  val = r(val, vec.z);
  val = r(val, vec.w);
  return val;
}

template <template<typename> class ReduceOp, typename T,
         int warp_num,
         typename std::enable_if<warp_num == 2>::type,
         typename std::enable_if<std::is_same<T, float>::value || std::is_same<T, __half>::value>::type,
         typename std::enable_if<std::is_same<ReduceOp<T>, Add<T>>::value>::type>
__forceinline__ __device__ T
BlockReduce_opt(T val, T* shared) {
  ReduceOp<T> r;
  const int lid = threadIdx.x % C10_WARP_SIZE;
  const int wid = threadIdx.x / C10_WARP_SIZE;
  // val = WarpReduce<ReduceOp, T>(val);
  val = __reduce_add_sync(0xffffffffffffffff, val);
  __syncthreads();
  if (lid == 0) {
    shared[wid] = val;
  }
  __syncthreads();

  float2 vec = (reinterpret_cast<float2*>(shared))[0];
  val = r(vec.x, vec.y);
  return val;
}

template <template<typename> class ReduceOp, typename T,
         int warp_num,
         typename std::enable_if<warp_num == 2>::type,
         typename std::enable_if<std::is_same<T, float>::value || std::is_same<T, __half>::value>::type,
         typename std::enable_if<std::is_same<ReduceOp<T>, Max<T>>::value>::type>
__forceinline__ __device__ T
BlockReduce_opt(T val, T* shared) {
  ReduceOp<T> r;
  const int lid = threadIdx.x % C10_WARP_SIZE;
  const int wid = threadIdx.x / C10_WARP_SIZE;
  // val = WarpReduce<ReduceOp, T>(val);
  val = __reduce_max_sync(0xffffffffffffffff, val);
  __syncthreads();
  if (lid == 0) {
    shared[wid] = val;
  }
  __syncthreads();

  float2 vec = (reinterpret_cast<float2*>(shared))[0];
  val = r(vec.x, vec.y);
  return val;
}

template <template<typename> class ReduceOp, typename T,
         int warp_num,
         typename std::enable_if<warp_num == 1>::type,
         typename std::enable_if<std::is_same<T, float>::value || std::is_same<T, __half>::value>::type,
         typename std::enable_if<std::is_same<ReduceOp<T>, Add<T>>::value>::type>
__forceinline__ __device__ T
BlockReduce_opt(T val, T* shared) {
  ReduceOp<T> r;
  const int lid = threadIdx.x % C10_WARP_SIZE;
  const int wid = threadIdx.x / C10_WARP_SIZE;
  // val = WarpReduce<ReduceOp, T>(val);
  val = __reduce_add_sync(0xffffffffffffffff, val);
  __syncthreads();
  return val;
}

template <template<typename> class ReduceOp, typename T,
         int warp_num,
         typename std::enable_if<warp_num == 1>::type,
         typename std::enable_if<std::is_same<T, float>::value || std::is_same<T, __half>::value>::type,
         typename std::enable_if<std::is_same<ReduceOp<T>, Max<T>>::value>::type>
__forceinline__ __device__ T
BlockReduce_opt(T val, T* shared) {
  ReduceOp<T> r;
  const int lid = threadIdx.x % C10_WARP_SIZE;
  const int wid = threadIdx.x / C10_WARP_SIZE;
  // val = WarpReduce<ReduceOp, T>(val);
  val = __reduce_max_sync(0xffffffffffffffff, val);
  __syncthreads();
  return val;
}

template <template<typename, typename> class Reduction, int ILP, typename T, typename AccumT>
__device__ __forceinline__ AccumT
ilpReduce(int shift,
          T* data,
          int size,
          const Reduction<T, AccumT>& r,
          AccumT defaultVal)
{
  using LoadT = at::native::memory::aligned_vector<T, ILP>;
  AccumT threadVal = defaultVal;
  int offset = threadIdx.x;

  // shift and do 1
  if(shift > 0){
    data -= shift;
    size += shift;
    if(threadIdx.x >= shift){
      threadVal = r(threadVal, data[offset]);
    }
    size -= blockDim.x;
    data += blockDim.x;
  }
  int last = size % (ILP * blockDim.x);

  T v[ILP];
  LoadT* value = reinterpret_cast<LoadT*>(&v);

  for (; offset * ILP < (size - last); offset += blockDim.x) {
    *value = reinterpret_cast<LoadT*>(data)[offset];

    #pragma unroll
    for (int j = 0; j < ILP; ++j) {
      threadVal = r(threadVal, v[j]);
    }
  }

  offset = size - last + threadIdx.x;
  // Epilogue
  for (; offset < size; offset += blockDim.x)
    threadVal = r(threadVal, data[offset]);

  return threadVal;
}

template <int ILP, typename scalar_t, typename accum_t, typename outscalar_t, template<typename, typename, typename> class Epilogue>
__device__ __forceinline__ void
WriteBpropResultsVectorizedOpt(
  int size,
  const int shift,
  scalar_t *gradInput,
  outscalar_t *output,
  outscalar_t *gradOutput,
  Epilogue<scalar_t, accum_t, outscalar_t> epilogue) {
    using gradInputT = at::native::memory::aligned_vector<scalar_t, ILP>;
    using outputT = at::native::memory::aligned_vector<outscalar_t, ILP>;
    int offset = threadIdx.x;
    scalar_t dX[ILP * 2];
    gradInputT *dX_v = reinterpret_cast<gradInputT*>(&dX);
    outscalar_t Y[ILP * 2];
    outputT *Y_v = reinterpret_cast<outputT*>(&Y);
    outscalar_t dY[ILP * 2];
    outputT *dY_v = reinterpret_cast<outputT*>(&dY);
    *Y_v = reinterpret_cast<outputT*>(output)[offset];
    *dY_v = reinterpret_cast<outputT*>(gradOutput)[offset];
    *(Y_v+1) = reinterpret_cast<outputT*>(output)[offset + blockDim.x];
    *(dY_v+1) = reinterpret_cast<outputT*>(gradOutput)[offset + blockDim.x];
    #pragma unroll
    for (int j = 0; j < 2 * ILP; ++j) {
      dX[j] = epilogue(dY[j], Y[j]);
    }
    reinterpret_cast<gradInputT*>(gradInput)[offset] = *dX_v;
    reinterpret_cast<gradInputT*>(gradInput)[offset + blockDim.x] = *(dX_v+1);
}

template<typename T, typename AccumT>
struct AddFloat
{
  __device__ __forceinline__ AccumT operator()(AccumT sum, T v) const {
    return sum + v;
  }
};

template <typename scalar_t, typename accscalar_t>
__forceinline__ __device__ void
Divide_opt(scalar_t* pack, accscalar_t* buf, const accscalar_t& row_sum, int num_packs, int pack_id) {
  pack[0] = buf[pack_id] / row_sum;
  pack[1] = buf[num_packs + pack_id] / row_sum;
  pack[2] = buf[2 * num_packs + pack_id] / row_sum;
  pack[3] = buf[3 * num_packs + pack_id] / row_sum;
  pack[4] = buf[4 * num_packs + pack_id] / row_sum;
  pack[5] = buf[5 * num_packs + pack_id] / row_sum;
  pack[6] = buf[6 * num_packs + pack_id] / row_sum;
  pack[7] = buf[7 * num_packs + pack_id] / row_sum;
}

template <typename scalar_t, typename accscalar_t,
         typename std::enable_if<std::is_same<accscalar_t, float>::value || std::is_same<accscalar_t, __half>::value>::type>
__forceinline__ __device__ void
Divide_opt(scalar_t* pack, accscalar_t* buf, const accscalar_t& row_sum, int num_packs, int pack_id) {
  pack[0] = buf[pack_id] * __builtin_mxc_rcpf(row_sum);
  pack[1] = buf[num_packs + pack_id] * __builtin_mxc_rcpf(row_sum);
  pack[2] = buf[2 * num_packs + pack_id] * __builtin_mxc_rcpf(row_sum);
  pack[3] = buf[3 * num_packs + pack_id] * __builtin_mxc_rcpf(row_sum);
  pack[4] = buf[4 * num_packs + pack_id] * __builtin_mxc_rcpf(row_sum);
  pack[5] = buf[5 * num_packs + pack_id] * __builtin_mxc_rcpf(row_sum);
  pack[6] = buf[6 * num_packs + pack_id] * __builtin_mxc_rcpf(row_sum);
  pack[7] = buf[7 * num_packs + pack_id] * __builtin_mxc_rcpf(row_sum);
}

template <int ILP, typename scalar_t, typename accscalar_t, typename outscalar_t, template<typename, typename, typename> class Epilogue, int shift>
__global__ void
cunn_SoftMaxBackward_opt2_no_shm(scalar_t *gradInput, outscalar_t *output, outscalar_t *gradOutput, int classes)
{
  extern __shared__ unsigned char smem[];
  auto sdata = reinterpret_cast<accscalar_t*>(smem);
  gradInput += static_cast<int64_t>(blockIdx.x) * classes;
  output += static_cast<int64_t>(blockIdx.x) * classes;
  gradOutput += static_cast<int64_t>(blockIdx.x) * classes;

  accscalar_t threadSum = ilpReduce<AddFloat, ILP, outscalar_t, accscalar_t>(
      shift, gradOutput, classes, AddFloat<outscalar_t, accscalar_t>(), accscalar_t(0));
  accscalar_t sum_k = BlockReduce_opt<Add, accscalar_t, 2>(threadSum, sdata);
  Epilogue<scalar_t, accscalar_t, outscalar_t> epilogue(sum_k);

  WriteBpropResultsVectorizedOpt<ILP, scalar_t, accscalar_t, outscalar_t, Epilogue>(classes, shift, gradInput, output, gradOutput, epilogue);
}


template <typename T, bool is_log_softmax, typename enable = void>
struct ThreadSum {
  __forceinline__  __device__ T operator()(T thread_elem, const T& row_max, T& thread_sum) {
    assert(0);
  }
};

template <typename T, typename enable>
struct ThreadSum<T, false, enable> {
  __forceinline__  __device__ T operator()(T thread_elem, const T& row_max, T& thread_sum) {
    thread_elem = std::exp(thread_elem - row_max);
    thread_sum += thread_elem;
    return thread_elem;
  }
};


template <typename T>
struct ThreadSum<T, false, typename std::enable_if<std::is_same<T, float>::value || std::is_same<T, c10::Half>::value>::type> {
  __forceinline__  __device__ T operator()(T thread_elem, const T& row_max, T& thread_sum) {
    thread_elem = __expf(thread_elem - row_max);
    thread_sum += thread_elem;
    return thread_elem;
  }
};

template <typename T, typename enable>
struct ThreadSum<T, true, enable> {
  __forceinline__  __device__ T operator()(T thread_elem, const T& row_max, T& thread_sum) {
    thread_elem = thread_elem - row_max;
    thread_sum += std::exp(thread_elem);
    return thread_elem;
  }
};

template <typename T>
struct ThreadSum<T, true, typename std::enable_if<std::is_same<T, float>::value || std::is_same<T, c10::Half>::value>::type> {
  __forceinline__  __device__ T operator()(T thread_elem, const T& row_max, T& thread_sum) {
    thread_elem = thread_elem - row_max;
    thread_sum += __expf(thread_elem);
    return thread_elem;
  }
};

template <typename T, bool is_log_softmax, typename enable = void>
struct SumUpdate {
  __forceinline__ __device__ T operator()(const T& sum) {
    assert(0);
  }
};

template <typename T, typename enable>
struct SumUpdate<T, false, enable> {
  __forceinline__ __device__ T operator()(const T& sum) {
    return 1 / sum;
  }
};

template <typename T>
struct SumUpdate<T, false, typename std::enable_if<std::is_same<T, float>::value || std::is_same<T, c10::Half>::value || std::is_same<T, c10::BFloat16>::value>::type> {
  __forceinline__ __device__ T operator()(const T& sum) {
    return __builtin_mxc_rcpf(sum);
  }
};

template <typename T, typename enable>
struct SumUpdate<T, true, enable> {
  __forceinline__ __device__ T operator()(const T& sum) {
    return std::log(sum);
  }
};

template <typename T>
struct SumUpdate<T, true, typename std::enable_if<std::is_same<T, float>::value || std::is_same<T, c10::Half>::value || std::is_same<T, c10::BFloat16>::value>::type> {
  __forceinline__ __device__ T operator()(const T& sum) {
    return __logf(sum);
  }
};

template <typename T, bool is_log_softmax>
struct ThreadUpdate {
  __forceinline__ __device__ T operator()(const T& thread_elem, const T& sum) const {
    return thread_elem * sum;
  }
};

template <typename T>
struct ThreadUpdate<T, true> {
  __forceinline__ __device__ T operator()(const T& thread_elem, const T& sum) const {
    return thread_elem - sum;
  }
};

template <int ILP, int warp_num, int packs_num, typename scalar_t, typename accscalar_t, typename outscalar_t, bool is_log_softmax>
__global__ void
cunn_SoftMaxForward_opt_no_shm_nd(outscalar_t *output, scalar_t *input, int classes, accscalar_t max_default, int packs_rem)
{
  extern __shared__ unsigned char shared_buf[];
  auto sdata = reinterpret_cast<accscalar_t*>(shared_buf);  // warpnum * sizeof(accscalar_t)
  const int tid = threadIdx.x;
  const int64_t rid = blockIdx.x;  // row id
  accscalar_t thread_max = max_default;
  using LoadT = at::native::memory::aligned_vector<scalar_t, ILP>;
  using StoreT = at::native::memory::aligned_vector<outscalar_t, ILP>;
  input += rid * classes;
  output += rid * classes;
  scalar_t pack[packs_num * ILP];
  scalar_t pack_rem[ILP];
  accscalar_t pack_acc[packs_num * ILP];
  accscalar_t pack_acc_rem[ILP];
#pragma unroll
  for (int pack_id = 0; pack_id < packs_num; ++pack_id) {
    LoadT* p_pack = reinterpret_cast<LoadT*>(&pack[pack_id * ILP]);
    *p_pack = reinterpret_cast<LoadT*>(input)[blockDim.x * pack_id + tid];
#pragma unroll
    for (int ilp_id = 0; ilp_id < ILP; ++ilp_id) {
        pack_acc[pack_id * ILP + ilp_id] = static_cast<accscalar_t>(pack[pack_id * ILP + ilp_id]);
        thread_max = max(pack_acc[pack_id * ILP + ilp_id], thread_max);
    }
  }

  // compute packs_rem
  if (packs_rem && tid < packs_rem) {
    LoadT* p_pack = reinterpret_cast<LoadT*>(pack_rem);
    *p_pack = reinterpret_cast<LoadT*>(input)[blockDim.x * packs_num + tid];
#pragma unroll
    for (int ilp_id = 0; ilp_id < ILP; ++ilp_id) {
        pack_acc_rem[ilp_id] = static_cast<accscalar_t>(pack_rem[ilp_id]);
        thread_max = max(pack_acc_rem[ilp_id], thread_max);
    }
  }

  const accscalar_t row_max = BlockReduce_opt<Max, accscalar_t, warp_num>(thread_max, sdata);

  accscalar_t thread_sum = 0;
#pragma unroll
  for (int pack_id = 0; pack_id < packs_num; ++pack_id) {
    #pragma unroll
    for (int ilp_id = 0; ilp_id < ILP; ++ilp_id) {
      int idx_num = pack_id * ILP + ilp_id;
      pack_acc[idx_num] = ThreadSum<accscalar_t, is_log_softmax>()(pack_acc[idx_num], row_max, thread_sum);
    }
  }

  // compute packs_rem
  if (packs_rem && tid < packs_rem) {
    #pragma unroll
    for (int ilp_id = 0; ilp_id < ILP; ++ilp_id) {
      pack_acc_rem[ilp_id] = ThreadSum<accscalar_t, is_log_softmax>()(pack_acc_rem[ilp_id], row_max, thread_sum);
    }
  }

  const accscalar_t row_sum = BlockReduce_opt<Add, accscalar_t, warp_num>(thread_sum, sdata);
  outscalar_t out_pack[packs_num * ILP];
  accscalar_t update_row_sum = SumUpdate<accscalar_t, is_log_softmax>()(row_sum);
#pragma unroll
  for (int pack_id = 0; pack_id < packs_num; ++pack_id) {
    int base_idx = pack_id * ILP;
    StoreT* s_pack = reinterpret_cast<StoreT*>(&out_pack[base_idx]);
    #pragma unroll
    for (int ilp_id = 0; ilp_id < ILP; ++ilp_id) {
        out_pack[base_idx + ilp_id] = static_cast<outscalar_t>(ThreadUpdate<accscalar_t, is_log_softmax>()(pack_acc[base_idx + ilp_id], update_row_sum));
    }
    reinterpret_cast<StoreT*>(output)[blockDim.x * pack_id + tid] = *s_pack;
  }

  // compute packs_rem
  if (packs_rem && tid < packs_rem) {
    StoreT* s_pack = reinterpret_cast<StoreT*>(&out_pack);
    #pragma unroll
    for (int ilp_id = 0; ilp_id < ILP; ++ilp_id) {
        out_pack[ilp_id] = static_cast<outscalar_t>(ThreadUpdate<accscalar_t, is_log_softmax>()(pack_acc_rem[ilp_id], update_row_sum));
    }
    reinterpret_cast<StoreT*>(output)[blockDim.x * packs_num + tid] = *s_pack;
  }
}