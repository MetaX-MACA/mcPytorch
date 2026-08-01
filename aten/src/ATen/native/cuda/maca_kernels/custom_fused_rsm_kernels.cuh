#pragma once

namespace at::native {


template<typename scalar_t, int vec_size>
struct alignas(sizeof(scalar_t) * vec_size) aligned_vector {
  scalar_t val[vec_size];
};


__global__ void rmsnorm_fused_kernel_base(
    const __maca_bfloat16* __restrict__ input,       // in_ptr0: [batch_size, hidden_size]
    const float* __restrict__ variance,              // in_ptr1: [batch_size]
    const __maca_bfloat16* __restrict__ weight,     // in_ptr2: [hidden_size] - gamma/weight
    __maca_bfloat16* __restrict__ output,           // out_ptr0
    const int hidden_size,                 // 768
    const int num_tokens,                  // batch_size
    float eps = 1e-6,
    float scale = 0.125)                    // 1/8 scaling factor
{
    const int origin_hidden_size = 1024;

    // hidden size per block
    const int token_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;

    if (token_idx >= num_tokens) {
      return;
    }

    // torch.sqrt(variance * scale + eps) per thread
    float var = static_cast<float>(variance[token_idx]);
    float inv_rms = rsqrt(var * scale + eps);  // 1 / sqrt(variance / self.tp_world  + 1e-6)

    // hidden_size * 1  vs  (128 thread per block)
    for (int idx = tid; idx < hidden_size; idx += block_size) {
        // const int global_idx = token_idx * hidden_size + idx;1
        if (idx >= hidden_size) {
            continue;
        }

        const int global_ld_idx = token_idx * origin_hidden_size + idx;
        const int global_st_idx = token_idx * hidden_size + idx;

        // load x && weight
        const float x = __bfloat162float(input[global_ld_idx]);
        const float w = __bfloat162float(weight[idx]);

        // x * torch.rsqrt(variance + eps) * weight
        const float result = x * inv_rms * w;

        // store and to half
        output[global_st_idx] = __float2bfloat16(result);
    }
}


__global__ void rmsnorm_fused_kernel_vec8_rowloop(
    __maca_bfloat16*  input,      // [16384, 1024]
    float*  variance,             // [16384]
    __maca_bfloat16*  weight,     // [1, 768]
    __maca_bfloat16*  output,           // [16384, 768]
    int hidden_size,                         // 768
    int num_tokens)                          // 16384
{
    constexpr int origin_hidden_size = 1024;
    constexpr int VEC_SIZE = 8;
    constexpr int TOKENS_PER_BLOCK = 1; // 1 token(line) per block
    const int base_token_idx = blockIdx.x * TOKENS_PER_BLOCK;
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;

    if (base_token_idx >= num_tokens) {
        return;
    }

    const int stride = block_size * VEC_SIZE;
    using LoadT = aligned_vector<__maca_bfloat16, VEC_SIZE>;

    #pragma unroll 1
    for (int t = 0; t < TOKENS_PER_BLOCK; ++t) {
        int token_idx = base_token_idx + t;

        if (token_idx >= num_tokens) {
            break;
        }

        // calculate inv_rms
        const float eps = 1e-6f;
        const float scale = 0.125f;
        float var = static_cast<float>(variance[token_idx]);
        float inv_rms = rsqrtf(var * scale + eps);

        // baslic offset
        const int token_in_offset = token_idx * origin_hidden_size;
        const int token_out_offset = token_idx * hidden_size;

        // loop for row: hidden_size
        #pragma unroll 2  // 768 / (48*8) = 3
        for (int idx = tid * VEC_SIZE; idx + VEC_SIZE <= hidden_size; idx += stride) {
            const int global_ld_idx = token_in_offset + idx;
            const int global_st_idx = token_out_offset + idx;
            const int weight_idx = idx;

            // vector load intput
            __maca_bfloat16 tmp_load_input[VEC_SIZE];
            LoadT* x_vec = reinterpret_cast<LoadT*>(&tmp_load_input);
            *x_vec = *reinterpret_cast<LoadT*>(input + global_ld_idx);

            // vector load weight
            __maca_bfloat16 tmp_load_weight[VEC_SIZE];
            LoadT* w_vec = reinterpret_cast<LoadT*>(&tmp_load_weight);
            *w_vec = *reinterpret_cast<LoadT*>(weight + weight_idx);

            // calculate
            using StoreT = aligned_vector<__maca_bfloat16, VEC_SIZE>;
            __maca_bfloat16 results[VEC_SIZE];
            StoreT* p_results = reinterpret_cast<StoreT*>(&results);
            StoreT* out = reinterpret_cast<StoreT*>(output + global_st_idx);
            for (int i = 0; i < VEC_SIZE; i++) {
                float x = __bfloat162float(tmp_load_input[i]);
                float w = __bfloat162float(tmp_load_weight[i]);

                float _result = x * inv_rms * w;
                results[i] = __float2bfloat16(_result);
            }
            *out = *p_results;
        }
    }
}


__global__ void rmsnorm_fused_kernel_vec8_regweight(
    __maca_bfloat16* input,          // [16384, 1024]
    float* variance,                 // [16384, ]
    __maca_bfloat16* weight,         // [1, 768]
    __maca_bfloat16* output,         // [16384, 768]
    int hidden_size,                 // 768
    int num_tokens)                  // 16384
{
    constexpr int VEC_SIZE = 8;
    constexpr int TOKENS_PER_BLOCK = 2;
    constexpr int origin_hidden_size = 1024;

    const int base_token_idx = blockIdx.x * TOKENS_PER_BLOCK;
    const int tid = threadIdx.x;

    if (base_token_idx >= num_tokens) {
        return;
    }

    // ========== pre_load  weight to register ============
    // blockDim.x = 96, VEC = 8,  8 * 96 = 768(hidden size)
    // ====================================================
    int weight_start = tid * VEC_SIZE;
    __maca_bfloat16 reg_weight[VEC_SIZE];

    using LoadT = aligned_vector<__maca_bfloat16, VEC_SIZE>;
    *reinterpret_cast<LoadT*>(reg_weight) = *reinterpret_cast<LoadT*>(weight + weight_start);

    // ========== 2 token per block，reuse weight ==========
    #pragma unroll 4
    for (int t = 0; t < TOKENS_PER_BLOCK; ++t) {
        int token_idx = base_token_idx + t;

        if (token_idx >= num_tokens) {
            break;
        }

        // load variance
        float inv_rms = rsqrtf(variance[token_idx] * 0.125f + 1e-6f);

        int token_in_offset = token_idx * origin_hidden_size;
        int token_out_offset = token_idx * hidden_size;

        int global_ld_idx = token_in_offset + weight_start;
        int global_st_idx = token_out_offset + weight_start;

        // load input
        __maca_bfloat16 tmp_input[VEC_SIZE];
        using LoadT = aligned_vector<__maca_bfloat16, VEC_SIZE>;
        *reinterpret_cast<LoadT*>(tmp_input) = *reinterpret_cast<LoadT*>(input + global_ld_idx);

        // calculae reuslt
        __maca_bfloat16 results[VEC_SIZE];
        #pragma unroll 8
        for (int i = 0; i < VEC_SIZE; ++i) {
            float x = __bfloat162float(tmp_input[i]);
            float w = __bfloat162float(reg_weight[i]);  //  reuse weight
            results[i] = __float2bfloat16(x * inv_rms * w);
        }

        // store result
        *reinterpret_cast<LoadT*>(output + global_st_idx) = *reinterpret_cast<LoadT*>(results);
    }
}


void customFuseRsmKernelImpl(
    const Tensor* x,
    const Tensor* variance,
    const Tensor* weight,
    Tensor* out) {
    int batch_size = x->size(0);   // 16384
    int hidden_dim = 768;

    int numel = batch_size * hidden_dim;

    __maca_bfloat16* x_ptr = reinterpret_cast<__maca_bfloat16*>(x->data_ptr());
    float* scale_ptr = reinterpret_cast<float*>(variance->data_ptr());
    __maca_bfloat16* weight_ptr = reinterpret_cast<__maca_bfloat16*>(weight->data_ptr());
    __maca_bfloat16* out_ptr = reinterpret_cast<__maca_bfloat16*>(out->data_ptr());

    /* vec laod & store 8*bfp16 &&  regweight */
    constexpr int TOKENS_PER_BLOCK = 2;
    constexpr int THREADS_PER_BLOCK = 96;

    // Grid：batch_size / 2 = 8192
    int grid_size = (batch_size + TOKENS_PER_BLOCK - 1) / TOKENS_PER_BLOCK;

    dim3 grid(grid_size);
    dim3 block(THREADS_PER_BLOCK);

    rmsnorm_fused_kernel_vec8_regweight<<<grid, block>>>(
        x_ptr, scale_ptr, weight_ptr, out_ptr,
        hidden_dim, batch_size);

    return;
}

} // namespace at::native