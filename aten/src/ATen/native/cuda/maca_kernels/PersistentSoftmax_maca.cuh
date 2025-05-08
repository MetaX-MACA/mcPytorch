template <typename acc_t, int WARP_BATCH, int WARP_SIZE, template<typename> class ReduceOp,
         typename std::enable_if<std::is_same<acc_t, float>::value || std::is_same<acc_t, __half>::value>::type,
         typename std::enable_if<std::is_same<ReduceOp<acc_t>, Max<acc_t>>::value>::type>
__device__ __forceinline__ void warp_reduce(acc_t* sum) {
    #pragma unroll
    for (int i = 0;  i < WARP_BATCH;  ++i) {
      sum[i] = __reduce_max_sync(0xffffffffffffffff, sum[i]);
    }
}

template <typename acc_t, int WARP_BATCH, int WARP_SIZE, template<typename> class ReduceOp,
         typename std::enable_if<std::is_same<acc_t, float>::value || std::is_same<acc_t, __half>::value>::type,
         typename std::enable_if<std::is_same<ReduceOp<acc_t>, Add<acc_t>>::value>::type>
__device__ __forceinline__ void warp_reduce(acc_t* sum) {
    #pragma unroll
    for (int i = 0;  i < WARP_BATCH;  ++i) {
      sum[i] = __reduce_add_sync(0xffffffffffffffff, sum[i]);
    }
}

template <typename acc_t, int WARP_BATCH, int WARP_SIZE, template<typename> class ReduceOp>
__device__ __forceinline__ void warp_reduce(acc_t* sum) {
    ReduceOp<acc_t> r;
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        #pragma unroll
        for (int i = 0;  i < WARP_BATCH;  ++i) {
            acc_t b = WARP_SHFL_XOR(sum[i], offset, WARP_SIZE);
            sum[i] = r(sum[i], b);
        }
    }
}

template <typename acc_t, int WARP_BATCH, int WARP_SIZE, template<typename> class ReduceOp>
__device__ __forceinline__ void warp_reduce_32_threads(acc_t* sum) {
    ReduceOp<acc_t> r;
    #pragma unroll
    for (int offset = WARP_SIZE / 4; offset > 0; offset /= 2) {
        #pragma unroll
        for (int i = 0;  i < WARP_BATCH;  ++i) {
            acc_t b = WARP_SHFL_XOR(sum[i], offset, WARP_SIZE);
            sum[i] = r(sum[i], b);
        }
    }
}

template <typename acc_t, int WARP_BATCH, int WARP_SIZE, template<typename> class ReduceOp>
__device__ __forceinline__ void warp_reduce_4_threads(acc_t* sum) {
    ReduceOp<acc_t> r;
    #pragma unroll
    for (int offset = 2; offset > 0; offset /= 2) {
        #pragma unroll
        for (int i = 0;  i < WARP_BATCH;  ++i) {
            acc_t b = WARP_SHFL_XOR(sum[i], offset, WARP_SIZE);
            sum[i] = r(sum[i], b);
        }
    }
}

template <typename acc_t, int WARP_BATCH, int WARP_SIZE, template<typename> class ReduceOp>
__device__ __forceinline__ void warp_reduce_8_threads(acc_t* sum) {
    ReduceOp<acc_t> r;
    #pragma unroll
    for (int offset = 4; offset > 0; offset /= 2) {
        #pragma unroll
        for (int i = 0;  i < WARP_BATCH;  ++i) {
            acc_t b = WARP_SHFL_XOR(sum[i], offset, WARP_SIZE);
            sum[i] = r(sum[i], b);
        }
    }
}

template <typename input_t, typename output_t, typename acc_t, int log2_elements, int vec, bool is_log_softmax>
__global__ void softmax_warp_forward_opt2_nd(output_t *dst, input_t *src, int batch_size, int stride, int element_count)
{
    // WARP_SIZE and WARP_BATCH must match the return values batches_per_warp and warp_size of method warp_softmax_forward_kernel.
    constexpr int next_power_of_two = 1 << log2_elements;
    constexpr int WARP_SIZE = (next_power_of_two < C10_WARP_SIZE) ? next_power_of_two : C10_WARP_SIZE;
    constexpr int WARP_ITERATIONS = next_power_of_two / WARP_SIZE;
    constexpr int WARP_BATCH = (next_power_of_two <= 128) ? 2 : 1;
    constexpr int vec_num = WARP_ITERATIONS / vec;

    // vec * sizeof(input_t) <= 128
    // for example: half->8 num per thread, float->4 num per thread, double->2 num per thread
    using LoadT = at::native::memory::aligned_vector<input_t, vec>;
    using StoreT = at::native::memory::aligned_vector<output_t, vec>;

    int first_batch = (blockDim.y * blockIdx.x + threadIdx.y) * WARP_BATCH;

    // batch_size might not be a multiple of WARP_BATCH. Check how
    // many batches have to computed within this WARP.
    int local_batches = batch_size - first_batch;
    if (local_batches <= 0){
        return;
    }

    // there might be multiple batches per warp. compute the index within the batch
    int local_idx = threadIdx.x;

    int step = first_batch * stride;

    src += step;
    dst += step;

    // The nested loops over WARP_BATCH and then WARP_ITERATIONS can be simplified to one loop,
    // but I think doing so would obfuscate the logic of the algorithm, thus I chose to keep
    // the nested loops.
    // This should have no impact on performance because the loops are unrolled anyway.

    // load data from global memory
    acc_t elements[WARP_BATCH][WARP_ITERATIONS];
    #pragma unroll
    for (int i = 0;  i < WARP_BATCH;  ++i) {
      if (i >= local_batches){
        break;
      }
      // the next row
      src += i * element_count;
      // cache vec_num * vec = WARP_ITERATIONS data
      input_t pack[vec_num][vec];
      #pragma unroll
      for(int v = 0; v < vec_num; ++v) {
        LoadT* l_pack = reinterpret_cast<LoadT*>(&pack[v]);
        *l_pack = (reinterpret_cast<LoadT*>(src))[local_idx * vec_num + v];
      }
      #pragma unroll
      for (int v = 0; v < vec_num; ++v) {
        #pragma unroll
        for (int j = 0; j < vec; ++j) {
          elements[i][v * vec + j] = pack[v][j];
        }
      }
    }

    // compute max_value
    acc_t max_value[WARP_BATCH];
    #pragma unroll
    for (int i = 0;  i < WARP_BATCH;  ++i) {
        if (i >= local_batches){
            break;
        }
        max_value[i] = elements[i][0];
        #pragma unroll
        for (int it = 0;  it < WARP_ITERATIONS;  ++it) {
            max_value[i] = max_value[i] > elements[i][it] ? max_value[i] : elements[i][it];
        }
    }
    warp_reduce<acc_t, WARP_BATCH, WARP_SIZE, Max>(max_value);
    // for (int i = 0;  i < WARP_BATCH;  ++i) {
    //   max_value[i] = __reduce_max_sync(0xffffffffffffffff, max_value[i]);
    // }

    acc_t sum[WARP_BATCH];
    #pragma unroll
    for (int i = 0;  i < WARP_BATCH;  ++i) {
        if (i >= local_batches){
            break;
        }
        sum[i] = 0.0f;
        if (is_log_softmax) {
            #pragma unroll
            for (int it = 0;  it < WARP_ITERATIONS;  ++it) {
                sum[i] += __expf(elements[i][it] - max_value[i]);
            }
        } else{
            #pragma unroll
            for (int it = 0;  it < WARP_ITERATIONS;  ++it) {
                elements[i][it] = __expf(elements[i][it] - max_value[i]);
                sum[i] += elements[i][it];
            }
        }
    }
    warp_reduce<acc_t, WARP_BATCH, WARP_SIZE, Add>(sum);
    acc_t rcpf_sum[WARP_BATCH];
    #pragma unroll
    for (int i = 0; i < WARP_BATCH; i++){
        rcpf_sum[i] = __builtin_mxc_rcpf(sum[i]);
    }
    // store result
    #pragma unroll
    for (int i = 0;  i < WARP_BATCH;  ++i) {
        if (i >= local_batches)
            break;
        if (is_log_softmax){
            sum[i] = __logf(sum[i]);
        }
        // next row
        dst += i * element_count;
        // cache output data
        output_t pack[vec_num][vec];
        if (is_log_softmax) {
            #pragma unroll
            for (int v = 0; v < vec_num; ++v) {
                StoreT* p_pack = reinterpret_cast<StoreT*>(&pack[v]);
                #pragma unroll
                for (int j = 0; j < vec; ++j) {
                    int element_index = local_idx * WARP_ITERATIONS + v * vec + j;
                    if (element_index < element_count) {
                        pack[v][j] = elements[i][v * vec + j] - max_value[i] - sum[i];
                    } else {
                        break;
                    }
                }
                (reinterpret_cast<StoreT*>(dst))[local_idx * vec_num + v] = *p_pack;
            }
        }
        //else if(sum[i] == 0) branch is deleted because sum[i] always > 1
        else{
            #pragma unroll
            for (int v = 0; v < vec_num; ++v) {
                StoreT* p_pack = reinterpret_cast<StoreT*>(&pack[v]);
                #pragma unroll
                for (int j = 0; j < vec; ++j) {
                    int element_index = local_idx * WARP_ITERATIONS + v * vec + j;
                    if (element_index < element_count) {
                        pack[v][j] = elements[i][v * vec + j] * rcpf_sum[i];
                    } else {
                        break;
                    }
                }
                (reinterpret_cast<StoreT*>(dst))[local_idx * vec_num + v] = *p_pack;
            }
        }
    }
}

template <typename input_t, typename output_t, typename acc_t, int log2_elements, int vec, bool is_log_softmax>
__global__ void softmax_warp_forward_512_half_bhalf(output_t *dst, input_t *src, int batch_size, int stride, int element_count)
{
    // 8 * sizeof(input_t) <= 128
    // for example: half->8 num per thread, float->4 num per thread, double->2 num per thread
    using LoadT = at::native::memory::aligned_vector<input_t, 8>;
    using StoreT = at::native::memory::aligned_vector<output_t, 8>;

    int first_batch = (blockDim.y * blockIdx.x + threadIdx.y) * 1;
    if (first_batch < batch_size) {
        // there might be multiple batches per warp. compute the index within the batch
        int local_idx = threadIdx.x;
        int step = first_batch * stride;
        src += step;

        acc_t elements[8];
        input_t pack0[8];
        LoadT* l_pack0 = reinterpret_cast<LoadT*>(&pack0);
        *l_pack0 = (reinterpret_cast<LoadT*>(src))[local_idx];
        dst += step;
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            elements[j] = pack0[j];
        }

        // compute max_value
        acc_t max_value = elements[0];
        #pragma unroll
        for (int it = 1;  it < 8;  ++it) {
            max_value = max_value > elements[it] ? max_value : elements[it];
        }
        warp_reduce<acc_t, 1, 64, Max>(&max_value);

        acc_t sum { 0.0f };
        if (is_log_softmax) {
            #pragma unroll
            for (int it = 0;  it < 8;  ++it) {
                sum += __expf(elements[it] - max_value);
            }
        } else{
            #pragma unroll
            for (int it = 0;  it < 8;  ++it) {
                elements[it] = __expf(elements[it] - max_value);
                sum += elements[it];
            }
        }
        warp_reduce<acc_t, 1, 64, Add>(&sum);

        // store
        auto rcp_sum = __builtin_mxc_rcpf(sum);
        if (is_log_softmax){
            sum = __logf(sum);
        }
        output_t packo0[8];
        StoreT* p_pack0 = reinterpret_cast<StoreT*>(&packo0);
        if (is_log_softmax) {
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                packo0[j] = (elements[j] - max_value - sum);
            }
        } else {
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                packo0[j] = (elements[j] * rcp_sum);
            }
        }
        (reinterpret_cast<StoreT*>(dst))[local_idx] = *p_pack0;
    }
}

template <typename input_t, typename output_t, typename acc_t, int log2_elements, int vec, bool is_log_softmax>
__global__ void softmax_warp_forward_512_float32(output_t *dst, input_t *src, int batch_size, int stride, int element_count)
{
    using LoadT = at::native::memory::aligned_vector<input_t, 4>;
    using StoreT = at::native::memory::aligned_vector<output_t, 4>;

    int first_batch = (blockDim.y * blockIdx.x + threadIdx.y) * 1;
    if (first_batch < batch_size) {
        // there might be multiple batches per warp. compute the index within the batch
        int local_idx = threadIdx.x;
        int step = first_batch * stride;
        src += step;

        acc_t elements[8];
        input_t pack0[4];
        input_t pack1[4];
        LoadT* l_pack0 = reinterpret_cast<LoadT*>(&pack0);
        *l_pack0 = (reinterpret_cast<LoadT*>(src))[local_idx];
        LoadT* l_pack1 = reinterpret_cast<LoadT*>(&pack1);
        *l_pack1 = (reinterpret_cast<LoadT*>(src))[local_idx + 64];
        dst += step;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
          elements[j] = pack0[j];
        }
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
          elements[j + 4] = pack1[j];
        }

        // compute max_value
        acc_t max_value = elements[0];
        #pragma unroll
        for (int it = 1;  it < 8;  ++it) {
            max_value = max_value > elements[it] ? max_value : elements[it];
        }
        warp_reduce<acc_t, 1, 64, Max>(&max_value);

        acc_t sum { 0.0f };
        if (is_log_softmax) {
            #pragma unroll
            for (int it = 0;  it < 8;  ++it) {
                sum += __expf(elements[it] - max_value);
            }
        } else{
            #pragma unroll
            for (int it = 0;  it < 8;  ++it) {
                elements[it] = __expf(elements[it] - max_value);
                sum += elements[it];
            }
        }
        warp_reduce<acc_t, 1, 64, Add>(&sum);

        // store
        auto rcp_sum = __builtin_mxc_rcpf(sum);
        if (is_log_softmax){
            sum = __logf(sum);
        }
        output_t packo0[4];
        StoreT* p_pack0 = reinterpret_cast<StoreT*>(&packo0);
        if (is_log_softmax) {
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                packo0[j] = (elements[0 * 4 + j] - max_value - sum);
            }
        } else {
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                packo0[j] = (elements[0 * 4 + j] * rcp_sum);
            }
        }
        (reinterpret_cast<StoreT*>(dst))[local_idx + 0 * 64] = *p_pack0;
        output_t packo1[4];
        StoreT* p_pack1 = reinterpret_cast<StoreT*>(&packo1);
        if (is_log_softmax) {
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                packo1[j] = (elements[1 * 4 + j] - max_value - sum);
            }
        } else {
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                packo1[j] = (elements[1 * 4 + j] * rcp_sum);
            }
        }
        (reinterpret_cast<StoreT*>(dst))[local_idx + 1 * 64] = *p_pack1;
    }
}

template <typename input_t, typename output_t, typename acc_t, int log2_elements, int vec, bool is_log_softmax>
__global__ void softmax_warp_forward_1024_float32(output_t *dst, input_t *src, int batch_size, int stride, int element_count)
{
    // 8 * sizeof(input_t) <= 128
    // for example: half->8 num per thread, float->4 num per thread, double->2 num per thread
    using LoadT = at::native::memory::aligned_vector<input_t, 4>;
    using StoreT = at::native::memory::aligned_vector<output_t, 4>;

    int first_batch = (blockDim.y * blockIdx.x + threadIdx.y) * 1;
    if (first_batch < batch_size) {
        // there might be multiple batches per warp. compute the index within the batch
        int local_idx = threadIdx.x;
        int step = first_batch * stride;
        src += step;

        acc_t elements[16];
        input_t pack0[4];
        input_t pack1[4];
        input_t pack2[4];
        input_t pack3[4];
        LoadT* l_pack0 = reinterpret_cast<LoadT*>(&pack0);
        *l_pack0 = (reinterpret_cast<LoadT*>(src))[local_idx];
        LoadT* l_pack1 = reinterpret_cast<LoadT*>(&pack1);
        *l_pack1 = (reinterpret_cast<LoadT*>(src))[local_idx + 64];
        LoadT* l_pack2 = reinterpret_cast<LoadT*>(&pack2);
        *l_pack2 = (reinterpret_cast<LoadT*>(src))[local_idx + 128];
        LoadT* l_pack3 = reinterpret_cast<LoadT*>(&pack3);
        *l_pack3 = (reinterpret_cast<LoadT*>(src))[local_idx + 192];
        dst += step;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
          elements[j] = pack0[j];
        }
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
          elements[j + 4] = pack1[j];
        }
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
          elements[j + 8] = pack2[j];
        }
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
          elements[j + 12] = pack3[j];
        }

        // compute max_value
        acc_t max_value = elements[0];
        #pragma unroll
        for (int it = 1;  it < 16;  ++it) {
            max_value = max_value > elements[it] ? max_value : elements[it];
        }
        warp_reduce<acc_t, 1, 64, Max>(&max_value);

        acc_t sum { 0.0f };
        if (is_log_softmax) {
            #pragma unroll
            for (int it = 0;  it < 16;  ++it) {
                sum += __expf(elements[it] - max_value);
            }   
        } else{
            #pragma unroll
            for (int it = 0;  it < 16;  ++it) {
                elements[it] = __expf(elements[it] - max_value);
                sum += elements[it];
            }
        }
        warp_reduce<acc_t, 1, 64, Add>(&sum);

        // store
        auto rcp_sum = __builtin_mxc_rcpf(sum);
        if (is_log_softmax){
            sum = __logf(sum);
        }
        output_t packo0[4];
        StoreT* p_pack0 = reinterpret_cast<StoreT*>(&packo0);
        if (is_log_softmax) {
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                packo0[j] = (elements[0 * 4 + j] - max_value - sum);
            }
        } else {
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                packo0[j] = (elements[0 * 4 + j] * rcp_sum);
            }
        }
        (reinterpret_cast<StoreT*>(dst))[local_idx + 0 * 64] = *p_pack0;
        output_t packo1[4];
        StoreT* p_pack1 = reinterpret_cast<StoreT*>(&packo1);
        if (is_log_softmax) {
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                packo1[j] = (elements[1 * 4 + j] - max_value - sum);
            }
        } else {
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                packo1[j] = (elements[1 * 4 + j] * rcp_sum);
            }
        }
        (reinterpret_cast<StoreT*>(dst))[local_idx + 1 * 64] = *p_pack1;
        output_t packo2[4];
        StoreT* p_pack2 = reinterpret_cast<StoreT*>(&packo2);
        if (is_log_softmax) {
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                packo2[j] = (elements[2 * 4 + j] - max_value - sum);
            }
        } else {
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                packo2[j] = (elements[2 * 4 + j] * rcp_sum);
            }
        }
        (reinterpret_cast<StoreT*>(dst))[local_idx + 2 * 64] = *p_pack2;
        output_t packo3[4];
        StoreT* p_pack3 = reinterpret_cast<StoreT*>(&packo3);
        if (is_log_softmax) {
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                packo3[j] = (elements[3 * 4 + j] - max_value - sum);
            }
        } else {
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                packo3[j] = (elements[3 * 4 + j] * rcp_sum);
            }
        }
        (reinterpret_cast<StoreT*>(dst))[local_idx + 3 * 64] = *p_pack3;
    }
}

template <typename input_t, typename output_t, typename acc_t, int log2_elements, int vec, bool is_log_softmax>
__global__ void softmax_warp_forward_1024_half_bhalf(output_t *dst, input_t *src, int batch_size, int stride, int element_count)
{
    // 8 * sizeof(input_t) <= 128
    // for example: half->8 num per thread, float->4 num per thread, double->2 num per thread
    using LoadT = at::native::memory::aligned_vector<input_t, 8>;
    using StoreT = at::native::memory::aligned_vector<output_t, 8>;

    int first_batch = (blockDim.y * blockIdx.x + threadIdx.y) * 1;
    if (first_batch < batch_size) {
        // there might be multiple batches per warp. compute the index within the batch
        int local_idx = threadIdx.x;
        int step = first_batch * stride;
        src += step;

        acc_t elements[16];
        input_t pack0[8];
        input_t pack1[8];
        LoadT* l_pack0 = reinterpret_cast<LoadT*>(&pack0);
        *l_pack0 = (reinterpret_cast<LoadT*>(src))[local_idx];
        LoadT* l_pack1 = reinterpret_cast<LoadT*>(&pack1);
        *l_pack1 = (reinterpret_cast<LoadT*>(src))[local_idx + 64];
        dst += step;
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
          elements[j] = pack0[j];
        }
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
          elements[j + 8] = pack1[j];
        }

        // compute max_value
        acc_t max_value = elements[0];
        #pragma unroll
        for (int it = 1;  it < 16;  ++it) {
            max_value = max_value > elements[it] ? max_value : elements[it];
        }
        warp_reduce<acc_t, 1, 64, Max>(&max_value);

        acc_t sum { 0.0f };
        if (is_log_softmax) {
            #pragma unroll
            for (int it = 0;  it < 16;  ++it) {
                sum += __expf(elements[it] - max_value);
            }   
        } else{
            #pragma unroll
            for (int it = 0;  it < 16;  ++it) {
                elements[it] = __expf(elements[it] - max_value);
                sum += elements[it];
            }
        }
        warp_reduce<acc_t, 1, 64, Add>(&sum);

        // store
        auto rcp_sum = __builtin_mxc_rcpf(sum);
        if (is_log_softmax){
            sum = __logf(sum);
        }
        output_t packo0[8];
        StoreT* p_pack0 = reinterpret_cast<StoreT*>(&packo0);
        if (is_log_softmax) {
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                packo0[j] = (elements[0 * 8 + j] - max_value - sum);
            }
        } else {
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                packo0[j] = (elements[0 * 8 + j] * rcp_sum);
            }
        }
        (reinterpret_cast<StoreT*>(dst))[local_idx + 0 * 64] = *p_pack0;
        output_t packo1[8];
        StoreT* p_pack1 = reinterpret_cast<StoreT*>(&packo1);
        if (is_log_softmax) {
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                packo1[j] = (elements[1 * 8 + j] - max_value - sum);
            }
        } else {
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                packo1[j] = (elements[1 * 8 + j] * rcp_sum);
            }
        }
        (reinterpret_cast<StoreT*>(dst))[local_idx + 1 * 64] = *p_pack1;
    }
}

template <typename input_t, typename output_t, typename acc_t, bool is_log_softmax>
__global__ void softmax_warp_forward_combine_load_nd(output_t *dst, input_t *src, int batch_size, int stride, int middle_thread)
{
    // x+y=64 && x * 16 + y * 8 = 720
    // x = 26 && y = 38
    // WARP_SIZE and WARP_BATCH must match the return values batches_per_warp and warp_size of method warp_softmax_forward_kernel.
    constexpr int WARP_SIZE = 64;
    constexpr int WARP_ITERATIONS = 16;
    constexpr int WARP_BATCH = 1;

    int first_batch = (blockDim.y * blockIdx.x + threadIdx.y) * WARP_BATCH;
    // batch_size might not be a multiple of WARP_BATCH. Check how
    // many batches have to computed within this WARP.
    int local_batches = batch_size - first_batch;
    if (local_batches <= 0){
        return;
    }
    int local_idx = threadIdx.x;
    int idx_offset = 0;
    if (local_idx <= middle_thread){
        //for thread 0~middle_thread, load 16 elements per thread
        idx_offset = first_batch * stride + local_idx * 16;
    }
    else{
        //for thread middle_thread+1~63, load 8 elements per thread
        idx_offset = first_batch * stride + (middle_thread+1) * 16 + (local_idx - (middle_thread+1)) * 8;
    }

    src += idx_offset;
    dst += idx_offset;

    // load data from global memory
    acc_t elements[WARP_ITERATIONS];
    input_t tmp_load1[8];
    using LoadT = at::native::memory::aligned_vector<input_t, 8>;
    LoadT* p_input1 = reinterpret_cast<LoadT*>(&tmp_load1);
    *p_input1 = *reinterpret_cast<LoadT*>(src);
    if (local_idx <= middle_thread){
        input_t tmp_load2[8];
        LoadT* p_input2 = reinterpret_cast<LoadT*>(&tmp_load2);
        *p_input2 = *reinterpret_cast<LoadT*>(src + 8);
        #pragma unroll
        for (int i = 0; i < WARP_ITERATIONS;i++){
            if (i < 8){
                elements[i] = tmp_load1[i];
            }
            else{
                elements[i] = tmp_load2[i-8];
            }
        }
    }
    else{
        #pragma unroll
        for (int i = 0; i < WARP_ITERATIONS;i++){
            if (i < 8){
                elements[i] = tmp_load1[i];
            }
            else{
                elements[i] = -std::numeric_limits<acc_t>::infinity();
            }
        }
    }

    // compute max_value
    acc_t max_value = elements[0];
    #pragma unroll
    for (int it = 0;  it < WARP_ITERATIONS;  ++it) {
        max_value = max_value > elements[it] ? max_value : elements[it];
    }

    warp_reduce<acc_t, WARP_BATCH, WARP_SIZE, Max>(&max_value);

    acc_t sum = 0.0f;
    if (is_log_softmax) {
        #pragma unroll
        for (int it = 0;  it < WARP_ITERATIONS;  ++it) {
            sum += __expf(elements[it] - max_value);
        }
    } else{
        #pragma unroll
        for (int it = 0;  it < WARP_ITERATIONS;  ++it) {
            elements[it] = __expf(elements[it] - max_value);
            sum += elements[it];
        }
    }
    
    warp_reduce<acc_t, WARP_BATCH, WARP_SIZE, Add>(&sum);
    acc_t rcpf_sum;
    rcpf_sum = __builtin_mxc_rcpf(sum);
    if (is_log_softmax){
        sum = __logf(sum);
    }
    // store result
    output_t tmp_store1[8];
    using StoreT = at::native::memory::aligned_vector<output_t, 8>;
    StoreT * p_store1 = reinterpret_cast<StoreT*>(&tmp_store1);

    if (is_log_softmax) {
        #pragma unroll
        for (int it = 0;  it < 8;  ++it) {
            tmp_store1[it] = elements[it] - max_value - sum;
        }
    } else{
        #pragma unroll
        for (int it = 0;  it < 8;  ++it) {
            tmp_store1[it] = elements[it] * rcpf_sum;
        }
    }
    
    StoreT* out1 = reinterpret_cast<StoreT*>(dst);
    *out1 = *p_store1;
    if (local_idx <= middle_thread){
        output_t tmp_store2[8];
        StoreT * p_store2 = reinterpret_cast<StoreT*>(&tmp_store2);
        StoreT* out2 = reinterpret_cast<StoreT*>(dst+8);
        if (is_log_softmax) {
            #pragma unroll
            for (int it = 0;  it < 8;  ++it) {
                tmp_store2[it] = elements[it+8] - max_value - sum;
            }
        } else{
            #pragma unroll
            for (int it = 0;  it < 8;  ++it) {
                tmp_store2[it] = elements[it+8] * rcpf_sum;
            }  
        }
        *out2 = *p_store2;
    }
}

template <typename input_t, typename output_t, typename acc_t, bool is_log_softmax>
__global__ void softmax_warp_forward_dim_49_half_bhalf(output_t *dst, input_t *src, int batch_size, int stride)
{
    // x+y=32 && x * 2 + y * 1 = 49
    // x = 17 && y = 15
    // then for threads in [0, 16] or [47, 63], load two elements
    // for threads in [17, 31] or [32, 46], load one element
    constexpr int WARP_SIZE = 64;
    constexpr int WARP_ITERATIONS = 2;
    constexpr int WARP_BATCH = 2;

    int first_batch = (blockDim.y * blockIdx.x + threadIdx.y) * WARP_BATCH;
    // batch_size might not be a multiple of WARP_BATCH. Check how
    // many batches have to computed within this WARP.
    int local_batches = batch_size - first_batch;
    if (local_batches <= 0){
        return;
    }
    int local_idx = threadIdx.x;
    int idx_offset = 0;
    if (local_idx <= 16){
        idx_offset = first_batch * stride + local_idx * 2;
    }
    else if(local_idx <= 31){
        idx_offset = first_batch * stride + 17 * 2 + (local_idx - 17) * 1;
    }
    else if(local_idx <= 46){
        idx_offset = first_batch * stride + 17 * 2 + 15 * 1 + (local_idx - 32) * 1;
    }
    else{
        idx_offset = first_batch * stride + 17 * 2 + 15 * 1 + 15 * 1 + (local_idx - 47) * 2;
    }

    src += idx_offset;
    dst += idx_offset;

    // load data from global memory
    acc_t elements[WARP_ITERATIONS];
    if (local_idx <= 16 || local_idx >= 47){
        input_t tmp_load1[2];
        using LoadT = at::native::memory::aligned_vector<input_t, 2>;
        LoadT* p_input1 = reinterpret_cast<LoadT*>(&tmp_load1);
        *p_input1 = *reinterpret_cast<LoadT*>(src);
        #pragma unroll
        for (int i = 0; i < WARP_ITERATIONS;i++){
            elements[i] = tmp_load1[i];
        }
    }
    else {
        elements[0] = src[0];
        elements[1] = -std::numeric_limits<acc_t>::infinity();
    }
    // compute max_value
    acc_t max_value = elements[0];
    #pragma unroll
    for (int it = 0;  it < WARP_ITERATIONS;  ++it) {
        max_value = max_value > elements[it] ? max_value : elements[it];
    }
    // do reduce between threads [0,31], [32, 63] independently
    warp_reduce_32_threads<acc_t, 1, 64, Max>(&max_value);

    acc_t sum = 0.0f;
    if (is_log_softmax) {
        #pragma unroll
        for (int it = 0;  it < WARP_ITERATIONS;  ++it) {
            sum += __expf(elements[it] - max_value);
        }
    } else{
        #pragma unroll
        for (int it = 0;  it < WARP_ITERATIONS;  ++it) {
            elements[it] = __expf(elements[it] - max_value);
            sum += elements[it];
        }
    }
    // do reduce between threads [0,31], [32, 63] independently
    warp_reduce_32_threads<acc_t, 1, 64, Add>(&sum);
    acc_t rcpf_sum;
    rcpf_sum = __builtin_mxc_rcpf(sum);
    if (is_log_softmax){
        sum = __logf(sum);
    }
    // store result
    if (local_idx <= 16 || local_idx >= 47){
        output_t tmp_store1[2];
        using StoreT = at::native::memory::aligned_vector<output_t, 2>;
        StoreT * p_store1 = reinterpret_cast<StoreT*>(&tmp_store1);
        if (is_log_softmax){
            #pragma unroll
            for (int i = 0; i < WARP_ITERATIONS;i++){
                tmp_store1[i] = elements[i] - max_value - sum;
            }
        } else{
            #pragma unroll
            for (int i = 0; i < WARP_ITERATIONS;i++){
                tmp_store1[i] = elements[i] * rcpf_sum;
            }
        }
        StoreT* out1 = reinterpret_cast<StoreT*>(dst);
        *out1 = *p_store1;
    }
    else {
        if (is_log_softmax){
            dst[0] = elements[0] - max_value - sum;
        } else{
            dst[0] = elements[0] * rcpf_sum;
        }
    }
}

template <typename input_t, typename output_t, typename acc_t, bool is_log_softmax>
__global__ void softmax_warp_forward_dim_32_half_bhalf(output_t *dst, input_t *src, int batch_size, int stride)
{
    // 8 * sizeof(input_t) <= 128
    // for example: half->8 num per thread, float->4 num per thread, double->2 num per thread
    using LoadT = at::native::memory::aligned_vector<input_t, 8>;
    using StoreT = at::native::memory::aligned_vector<output_t, 8>;

    int first_batch = (blockDim.y * blockIdx.x + threadIdx.y) * 1;
    if (first_batch < batch_size) {
        // there might be multiple batches per warp. compute the index within the batch
        int local_idx = threadIdx.x;
        int step = first_batch * stride;
        src += step;

        acc_t elements[8];
        input_t pack0[8];
        LoadT* l_pack0 = reinterpret_cast<LoadT*>(&pack0);
        *l_pack0 = (reinterpret_cast<LoadT*>(src))[local_idx];
        dst += step;
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            elements[j] = pack0[j];
        }

        // compute max_value
        acc_t max_value = elements[0];
        #pragma unroll
        for (int it = 1;  it < 8;  ++it) {
            max_value = max_value > elements[it] ? max_value : elements[it];
        }
        warp_reduce_4_threads<acc_t, 1, 64, Max>(&max_value);

        acc_t sum { 0.0f };
        if (is_log_softmax) {
            #pragma unroll
            for (int it = 0;  it < 8;  ++it) {
                sum += __expf(elements[it] - max_value);
            }
        } else{
            #pragma unroll
            for (int it = 0;  it < 8;  ++it) {
                elements[it] = __expf(elements[it] - max_value);
                sum += elements[it];
            }
        }
        warp_reduce_4_threads<acc_t, 1, 64, Add>(&sum);

        // store
        auto rcp_sum = __builtin_mxc_rcpf(sum);
        if (is_log_softmax){
            sum = __logf(sum);
        }
        output_t packo0[8];
        StoreT* p_pack0 = reinterpret_cast<StoreT*>(&packo0);
        if (is_log_softmax) {
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                packo0[j] = (elements[j] - max_value - sum);
            }
        } else {
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                packo0[j] = (elements[j] * rcp_sum);
            }
        }
        (reinterpret_cast<StoreT*>(dst))[local_idx] = *p_pack0;
    }
}

template <typename input_t, typename output_t, typename acc_t, bool is_log_softmax>
__global__ void softmax_warp_forward_dim_32_float32(output_t *dst, input_t *src, int batch_size, int stride)
{
    // 8 * sizeof(input_t) <= 128
    // for example: half->8 num per thread, float->4 num per thread, double->2 num per thread
    using LoadT = at::native::memory::aligned_vector<input_t, 4>;
    using StoreT = at::native::memory::aligned_vector<output_t, 4>;

    int first_batch = (blockDim.y * blockIdx.x + threadIdx.y) * 1;
    if (first_batch < batch_size) {
        // there might be multiple batches per warp. compute the index within the batch
        int local_idx = threadIdx.x;
        int step = first_batch * stride;
        src += step;

        acc_t elements[4];
        input_t pack0[4];
        LoadT* l_pack0 = reinterpret_cast<LoadT*>(&pack0);
        *l_pack0 = (reinterpret_cast<LoadT*>(src))[local_idx];
        dst += step;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            elements[j] = pack0[j];
        }

        // compute max_value
        acc_t max_value = elements[0];
        #pragma unroll
        for (int it = 1;  it < 4;  ++it) {
            max_value = max_value > elements[it] ? max_value : elements[it];
        }
        warp_reduce_8_threads<acc_t, 1, 64, Max>(&max_value);

        acc_t sum { 0.0f };
        if (is_log_softmax) {
            #pragma unroll
            for (int it = 0;  it < 4;  ++it) {
                sum += __expf(elements[it] - max_value);
            }
        } else{
            #pragma unroll
            for (int it = 0;  it < 4;  ++it) {
                elements[it] = __expf(elements[it] - max_value);
                sum += elements[it];
            }
        }
        warp_reduce_8_threads<acc_t, 1, 64, Add>(&sum);

        // store
        auto rcp_sum = __builtin_mxc_rcpf(sum);
        if (is_log_softmax){
            sum = __logf(sum);
        }
        output_t packo0[4];
        StoreT* p_pack0 = reinterpret_cast<StoreT*>(&packo0);
        if (is_log_softmax) {
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                packo0[j] = (elements[j] - max_value - sum);
            }
        } else {
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                packo0[j] = (elements[j] * rcp_sum);
            }
        }
        (reinterpret_cast<StoreT*>(dst))[local_idx] = *p_pack0;
    }
}