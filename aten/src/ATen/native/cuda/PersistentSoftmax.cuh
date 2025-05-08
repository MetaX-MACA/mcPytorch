#pragma once

#include <assert.h>
#include <cfloat>
#include <limits>
#include <stdint.h>
#include <cuda_fp16.h>
#include <c10/macros/Macros.h>
#include <vector>
#include <ATen/cuda/DeviceUtils.cuh>
#include <c10/core/ScalarType.h>
namespace {

int log2_ceil(int value) {
    int log2_value = 0;
    while ((1 << log2_value) < value) ++log2_value;
    return log2_value;
}

template<typename T>
struct Add {
  __device__ __forceinline__ T operator()(T a, T b) const {
    return a + b;
  }
};

template<typename T>
struct Max {
  __device__ __forceinline__ T operator()(T a, T b) const {
    return a < b ? b : a;
  }
};

#include <ATen/native/cuda/maca_kernels/PersistentSoftmax_maca.cuh>

// The softmax_warp_* methods perform softmax forward and backward propagation on samples spanning the fast dimension.
// Each sample contains element_count scalar elements. element_count can be any integer value <= 1024.
// The template arguments have the following meaning:
// One "WARP" works on one "BATCH". One "BATCH" contains "WARP_BATCH" samples.
// WARP_BATCH is equal to 1 when element_count is large, and > 1 when element_count is small.
// A "WARP" contains "C10_WARPS_SIZE" threads, these treads are guaranteed to belong to the same warp.
// This is important because it means only __shfl_ instructions are required for reductions.
// Note that this means WARP_SIZE must be a power of two and <= architecture warp size.
// CUDA warp size is 32 for all existing GPU architectures, but there is no guarantee this will not change for future arch.
// ROCm warp size is 64 for all currently ROCm-supported GPU architectures, but this may change for future archs.
// is_log_softmax is a flag indicating whether SoftMax or LogSoftMax should be computed.
// is_masked is a flag indicating whether SoftMax or MaskedSoftMax should be computed.
// The template can be instantiated with any floating point type for the type arguments input_t, output_t and acc_t.
// This allows SoftMax to be fused with a cast immediately following the SoftMax.
// The mask should have the same shape as input, with a boolean indicate if the value is masked.
// The head_chunk_size is only used for transformer mask softmax, equals to H * D * D.
// For instance:
// input_t=half,  acc_t=float, output_t=half  => read half tensor, float accumulators, write half tensor.
// input_t=half,  acc_t=float, output_t=float => read half tensor, float accumulators, write float tensor.
// input_t_float, acc_t=float, output_t=half  => read float tensor, float accumulators, write half tensor.
template <typename input_t, typename output_t, typename acc_t, int log2_elements, bool is_log_softmax, bool is_masked>
__global__ void softmax_warp_forward(output_t *dst, const input_t *src, int batch_size, int stride, int element_count, const bool *mask = nullptr, const int head_chunk_size = -1, bool is_transformer_mask = false)
{
    // WARP_SIZE and WARP_BATCH must match the return values batches_per_warp and warp_size of method warp_softmax_forward_kernel.
    constexpr int next_power_of_two = 1 << log2_elements;
    constexpr int WARP_SIZE = (next_power_of_two < C10_WARP_SIZE) ? next_power_of_two : C10_WARP_SIZE;
    constexpr int WARP_ITERATIONS = next_power_of_two / WARP_SIZE;
    constexpr int WARP_BATCH = (next_power_of_two <= 128) ? 2 : 1;

    int first_batch = (blockDim.y * blockIdx.x + threadIdx.y) * WARP_BATCH;

    // batch_size might not be a multiple of WARP_BATCH. Check how
    // many batches have to computed within this WARP.
    int local_batches = batch_size - first_batch;
    if (local_batches > WARP_BATCH)
        local_batches = WARP_BATCH;

    // there might be multiple batches per warp. compute the index within the batch
    int local_idx = threadIdx.x;
    int idx_offset = first_batch * stride + local_idx;

    src += idx_offset;
    dst += idx_offset;

    if (is_transformer_mask) {
        mask += ((first_batch * stride) / head_chunk_size) * stride + local_idx;
    } else {
        mask += idx_offset;
    }
    // The nested loops over WARP_BATCH and then WARP_ITERATIONS can be simplified to one loop,
    // but I think doing so would obfuscate the logic of the algorithm, thus I chose to keep
    // the nested loops.
    // This should have no impact on performance because the loops are unrolled anyway.

    // load data from global memory
    acc_t elements[WARP_BATCH][WARP_ITERATIONS];
    for (int i = 0;  i < WARP_BATCH;  ++i) {
        int batch_element_count = (i >= local_batches) ? 0 : element_count;
        for (int it = 0;  it < WARP_ITERATIONS;  ++it) {
            int element_index = local_idx + it * WARP_SIZE;
            if (element_index < batch_element_count) {
                elements[i][it] = src[i*element_count+it*WARP_SIZE];
            } else {
                elements[i][it] = -std::numeric_limits<acc_t>::infinity();
            }
        }
    }

    // compute max_value
    acc_t max_value[WARP_BATCH];
    #pragma unroll
    for (int i = 0;  i < WARP_BATCH;  ++i) {
        int batch_element_count = (i >= local_batches) ? 0 : element_count;
        bool is_meaningful_max = false;
        max_value[i] = elements[i][0];
        #pragma unroll
        for (int it = 0;  it < WARP_ITERATIONS;  ++it) {
            if (is_masked) {
                int idx = it*WARP_SIZE;
                if ((idx + local_idx) < batch_element_count) {
                    if (!is_transformer_mask) {
                        idx += i*element_count;
                    }
                    if (!mask[idx]) {
                        max_value[i] = (is_meaningful_max && max_value[i] > elements[i][it]) ? max_value[i] : elements[i][it];
                        is_meaningful_max = true;
                    }
                }
            } else {
                max_value[i] = max_value[i] > elements[i][it] ? max_value[i] : elements[i][it];
            }
        }
        if (is_masked) {
            if (!is_meaningful_max) {
                max_value[i] = -std::numeric_limits<acc_t>::infinity();
            }
        }
    }
    warp_reduce<acc_t, WARP_BATCH, WARP_SIZE, Max>(max_value);

    acc_t sum[WARP_BATCH];
    #pragma unroll
    for (int i = 0;  i < WARP_BATCH;  ++i) {
        sum[i] = 0.0f;
        int batch_element_count = (i >= local_batches) ? 0 : element_count;
        #pragma unroll
        for (int it = 0;  it < WARP_ITERATIONS;  ++it) {
            if (!is_masked) {
                if (is_log_softmax) {
                    sum[i] += __expf(elements[i][it] - max_value[i]);
                } else {
                    elements[i][it] = __expf(elements[i][it] - max_value[i]);
                    sum[i] += elements[i][it];
                }
            } else {
                int idx = it*WARP_SIZE;
                bool valid = (idx + local_idx) < batch_element_count;
                if (!is_transformer_mask) {
                    idx += i*element_count;
                }
                if (valid) {
                    if (!mask[idx]) {
                        if (is_log_softmax) {
                            sum[i] += std::exp(elements[i][it] - max_value[i]);
                        } else {
                            elements[i][it] = std::exp(elements[i][it] - max_value[i]);
                            sum[i] += elements[i][it];
                        }
                    } else {
                        if (!is_log_softmax) {
                            // Masked values are treated as -infinity, and std::exp(-infinity) is 0.
                            elements[i][it] = 0;
                        }
                    }
                } else {
                    if (!is_log_softmax) {
                        elements[i][it] = 0.;
                    }
                }
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
        if (is_log_softmax) sum[i] = __logf(sum[i]);
        #pragma unroll
        for (int it = 0;  it < WARP_ITERATIONS;  ++it) {
            int element_index = local_idx + it * WARP_SIZE;
            if (element_index < element_count) {
                if (is_log_softmax) {
                    dst[i*element_count+it*WARP_SIZE] = elements[i][it] - max_value[i] - sum[i];
                } else if (sum[i] == 0) {
                    dst[i*element_count+it*WARP_SIZE] = std::numeric_limits<acc_t>::quiet_NaN();
                } else {
                    dst[i*element_count+it*WARP_SIZE] = elements[i][it] * rcpf_sum[i];
                }
            } else {
                break;
            }
        }
    }
}

template <typename input_t, typename output_t, typename acc_t, int log2_elements, bool is_log_softmax, bool is_masked>
__global__ void softmax_warp_forward_double(output_t *dst, const input_t *src, int batch_size, int stride, int element_count, const bool *mask = nullptr, const int head_chunk_size = -1, bool is_transformer_mask = false)
{
    // WARP_SIZE and WARP_BATCH must match the return values batches_per_warp and warp_size of method warp_softmax_forward_kernel.
    constexpr int next_power_of_two = 1 << log2_elements;
    constexpr int WARP_SIZE = (next_power_of_two < C10_WARP_SIZE) ? next_power_of_two : C10_WARP_SIZE;
    constexpr int WARP_ITERATIONS = next_power_of_two / WARP_SIZE;
    constexpr int WARP_BATCH = (next_power_of_two <= 128) ? 2 : 1;

    int first_batch = (blockDim.y * blockIdx.x + threadIdx.y) * WARP_BATCH;

    // batch_size might not be a multiple of WARP_BATCH. Check how
    // many batches have to computed within this WARP.
    int local_batches = batch_size - first_batch;
    if (local_batches > WARP_BATCH)
        local_batches = WARP_BATCH;

    // there might be multiple batches per warp. compute the index within the batch
    int local_idx = threadIdx.x;
    int idx_offset = first_batch * stride + local_idx;

    src += idx_offset;
    dst += idx_offset;

    if (is_transformer_mask) {
        mask += ((first_batch * stride) / head_chunk_size) * stride + local_idx;
    } else {
        mask += idx_offset;
    }
    // The nested loops over WARP_BATCH and then WARP_ITERATIONS can be simplified to one loop,
    // but I think doing so would obfuscate the logic of the algorithm, thus I chose to keep
    // the nested loops.
    // This should have no impact on performance because the loops are unrolled anyway.

    // load data from global memory
    acc_t elements[WARP_BATCH][WARP_ITERATIONS];
    for (int i = 0;  i < WARP_BATCH;  ++i) {
        int batch_element_count = (i >= local_batches) ? 0 : element_count;
        for (int it = 0;  it < WARP_ITERATIONS;  ++it) {
            int element_index = local_idx + it * WARP_SIZE;
            if (element_index < batch_element_count) {
                elements[i][it] = src[i*element_count+it*WARP_SIZE];
            } else {
                elements[i][it] = -std::numeric_limits<acc_t>::infinity();
            }
        }
    }

    // compute max_value
    acc_t max_value[WARP_BATCH];
    #pragma unroll
    for (int i = 0;  i < WARP_BATCH;  ++i) {
        int batch_element_count = (i >= local_batches) ? 0 : element_count;
        bool is_meaningful_max = false;
        max_value[i] = elements[i][0];
        #pragma unroll
        for (int it = 0;  it < WARP_ITERATIONS;  ++it) {
            if (is_masked) {
                int idx = it*WARP_SIZE;
                if ((idx + local_idx) < batch_element_count) {
                    if (!is_transformer_mask) {
                        idx += i*element_count;
                    }
                    if (!mask[idx]) {
                        max_value[i] = (is_meaningful_max && max_value[i] > elements[i][it]) ? max_value[i] : elements[i][it];
                        is_meaningful_max = true;
                    }
                }
            } else {
                max_value[i] = max_value[i] > elements[i][it] ? max_value[i] : elements[i][it];
            }
        }
        if (is_masked) {
            if (!is_meaningful_max) {
                max_value[i] = -std::numeric_limits<acc_t>::infinity();
            }
        }
    }
    warp_reduce<acc_t, WARP_BATCH, WARP_SIZE, Max>(max_value);

    acc_t sum[WARP_BATCH];
    #pragma unroll
    for (int i = 0;  i < WARP_BATCH;  ++i) {
        sum[i] = 0.0f;
        int batch_element_count = (i >= local_batches) ? 0 : element_count;
        #pragma unroll
        for (int it = 0;  it < WARP_ITERATIONS;  ++it) {
            if (!is_masked) {
                if (is_log_softmax) {
                    sum[i] += std::exp(elements[i][it] - max_value[i]);
                } else {
                    elements[i][it] = std::exp(elements[i][it] - max_value[i]);
                    sum[i] += elements[i][it];
                }
            } else {
                int idx = it*WARP_SIZE;
                bool valid = (idx + local_idx) < batch_element_count;
                if (!is_transformer_mask) {
                    idx += i*element_count;
                }
                if (valid) {
                    if (!mask[idx]) {
                        if (is_log_softmax) {
                            sum[i] += std::exp(elements[i][it] - max_value[i]);
                        } else {
                            elements[i][it] = std::exp(elements[i][it] - max_value[i]);
                            sum[i] += elements[i][it];
                        }
                    } else {
                        if (!is_log_softmax) {
                            // Masked values are treated as -infinity, and std::exp(-infinity) is 0.
                            elements[i][it] = 0;
                        }
                    }
                } else {
                    if (!is_log_softmax) {
                        elements[i][it] = 0.;
                    }
                }
            }
        }
    }
    warp_reduce<acc_t, WARP_BATCH, WARP_SIZE, Add>(sum);
    acc_t rcpf_sum[WARP_BATCH];
    #pragma unroll
    for (int i = 0; i < WARP_BATCH; i++){
        rcpf_sum[i] = 1 / sum[i];
    }
    // store result
    #pragma unroll
    for (int i = 0;  i < WARP_BATCH;  ++i) {
        if (i >= local_batches)
            break;
        if (is_log_softmax) sum[i] = std::log(sum[i]);
        #pragma unroll
        for (int it = 0;  it < WARP_ITERATIONS;  ++it) {
            int element_index = local_idx + it * WARP_SIZE;
            if (element_index < element_count) {
                if (is_log_softmax) {
                    dst[i*element_count+it*WARP_SIZE] = elements[i][it] - max_value[i] - sum[i];
                } else if (sum[i] == 0) {
                    dst[i*element_count+it*WARP_SIZE] = std::numeric_limits<acc_t>::quiet_NaN();
                } else {
                    dst[i*element_count+it*WARP_SIZE] = elements[i][it] * rcpf_sum[i];
                }
            } else {
                break;
            }
        }
    }
}

template <typename input_t, typename output_t, typename acc_t, int log2_elements, bool is_log_softmax, bool is_masked>
__global__ void softmax_warp_backward(output_t *gradInput, const input_t *grad, const input_t *output, int batch_size, int stride, int element_count, const bool *mask = nullptr)
{
    // WARP_SIZE and WARP_BATCH must match the return values batches_per_warp and warp_size of method warp_softmax_backward_kernel.
    constexpr int next_power_of_two = 1 << log2_elements;
    constexpr int WARP_SIZE = (next_power_of_two < C10_WARP_SIZE) ? next_power_of_two : C10_WARP_SIZE;
    constexpr int WARP_ITERATIONS = next_power_of_two / WARP_SIZE;
    constexpr int WARP_BATCH = (next_power_of_two <= 128) ? 2 : 1;

    int first_batch = (blockDim.y * blockIdx.x + threadIdx.y) * WARP_BATCH;

    // batch_size might not be a multiple of WARP_BATCH. Check how
    // many batches have to computed within this WARP.
    int local_batches = batch_size - first_batch;
    if (local_batches > WARP_BATCH)
        local_batches = WARP_BATCH;

    // there might be multiple batches per warp. compute the index within the batch
    int local_idx = threadIdx.x % WARP_SIZE;

    // the first element to process by the current thread
    int thread_offset = first_batch * stride + local_idx;
    grad += thread_offset;
    output += thread_offset;
    gradInput += thread_offset;
    if (is_masked) {
        mask += thread_offset;
    }

    // The nested loops over WARP_BATCH and then WARP_ITERATIONS can be simplified to one loop,
    // but I think doing so would obfuscate the logic of the algorithm, thus I chose to keep
    // the nested loops.
    // This should have no impact on performance because the loops are unrolled anyway.

    // load data from global memory
    acc_t grad_reg[WARP_BATCH][WARP_ITERATIONS];
    acc_t output_reg[WARP_BATCH][WARP_ITERATIONS];
    for (int i = 0;  i < WARP_BATCH;  ++i) {
        int batch_element_count = (i >= local_batches) ? 0 : element_count;
        for (int it = 0;  it < WARP_ITERATIONS;  ++it) {
            int element_index = local_idx + it * WARP_SIZE;
            if (element_index < batch_element_count) {
                grad_reg[i][it] = grad[i*element_count+it*WARP_SIZE];
                output_reg[i][it] = output[i*element_count+it*WARP_SIZE];
            } else {
                grad_reg[i][it] = acc_t(0);
                output_reg[i][it] = acc_t(0);
            }
        }
    }

    acc_t sum[WARP_BATCH];
    #pragma unroll
    for (int i = 0;  i < WARP_BATCH;  ++i) {
        sum[i] = 0.0f;
        #pragma unroll
        for (int it = 0;  it < WARP_ITERATIONS;  ++it) {
            if (!is_masked || !mask[i*element_count+it*WARP_SIZE]) {
                sum[i] += grad_reg[i][it];
            }
        }
    }
    warp_reduce<acc_t, WARP_BATCH, WARP_SIZE, Add>(sum);

    // store result
    #pragma unroll
    for (int i = 0;  i < WARP_BATCH;  ++i) {
        if (i >= local_batches)
            break;
        #pragma unroll
        for (int it = 0;  it < WARP_ITERATIONS;  ++it) {
            int element_index = local_idx + it * WARP_SIZE;
            if (element_index < element_count) {
                if (is_masked && mask[i*element_count+it*WARP_SIZE]) {
                    gradInput[i*element_count+it*WARP_SIZE] = 0;
                }
                // compute gradients
                else if (is_log_softmax) {
                    gradInput[i*element_count+it*WARP_SIZE] = (grad_reg[i][it] - std::exp(output_reg[i][it]) * sum[i]);
                } else {
                    gradInput[i*element_count+it*WARP_SIZE] = (grad_reg[i][it] - output_reg[i][it] * sum[i]);
                }
            }
        }
    }
}

} // end of anonymous namespace

template<typename input_t, typename output_t, typename acc_t, bool is_log_softmax, bool is_masked>
void dispatch_softmax_forward(output_t *dst, input_t *src, int softmax_elements, int softmax_elements_stride, int batch_count, const bool *mask = nullptr, int chunk_size = -1, bool is_transformer_mask = false)
{
    TORCH_INTERNAL_ASSERT( softmax_elements >= 0 && softmax_elements <= 1024 );
    if (softmax_elements == 0) {
        return;
    } else {
        int log2_elements = log2_ceil(softmax_elements);
        const int next_power_of_two = 1 << log2_elements;

        // This value must match the WARP_SIZE constexpr value computed inside softmax_warp_forward.
        int warp_size = at::cuda::warp_size();
        warp_size = (next_power_of_two < warp_size) ? next_power_of_two : warp_size;

        // This value must match the WARP_BATCH constexpr value computed inside softmax_warp_forward.
        int batches_per_warp = (next_power_of_two <= 128) ? 2 : 1;

        // use 128 threads per block to maximimize gpu utilization
        constexpr int threads_per_block = 128;

        int warps_per_block = (threads_per_block / warp_size);
        int batches_per_block = warps_per_block * batches_per_warp;
        int blocks = (batch_count + batches_per_block - 1) / batches_per_block;
        dim3 threads(warp_size, warps_per_block, 1);

        // softmax_elements must greater than 64, ensure that the number of per thread greater than 1
        bool check_num = softmax_elements > 64 && softmax_elements % 64 == 0 && sizeof(input_t) < 8 && sizeof(input_t) >= 2;
        // cal the vec number
        int warp_iterations = softmax_elements / warp_size; // the num of per thread handle
        check_num = check_num && ((warp_iterations & (warp_iterations -1)) == 0);
        int vec = (warp_iterations * sizeof(input_t) * 8) > 128 ? 128 / (sizeof(input_t) * 8) : warp_iterations;
        // check addr is aligned
        auto ip = reinterpret_cast<uintptr_t>(src);
        auto op = reinterpret_cast<uintptr_t>(dst);
        bool ip_aligned = false;
        bool op_aligned = false;
        if (vec) {
          ip_aligned = !(ip % 4);
          op_aligned = !(op % 4);
        }
        bool is_aligned = ip_aligned && op_aligned;
        bool disable_softmax_opt = at::maca::get_maca_disable_softmax_opt();
        bool is_opt = !is_masked && !is_transformer_mask && check_num && is_aligned && sizeof(input_t) != 8 && !disable_softmax_opt;
        std::vector<int> combine_list = {
            520, 528, 536, 544, 552, 560, 568, 576, 584, 592, 600, 608, 616, 624, 632, 640, 648, 656, 664, 672, 680, 688, 696, \
            704, 712, 720, 728, 736, 744, 752, 760, 768, 776, 784, 792, 800, 808, 816, 824, 832, 840, 848, 856, 864, 872, 880, \
            888, 896, 904, 912, 920, 928, 936, 944, 952, 960, 968, 976, 984, 992, 1000, 1008, 1016
        };
        bool is_opt_combine_load = !is_masked && !is_transformer_mask && std::find(combine_list.begin(), combine_list.end(), softmax_elements) != combine_list.end() && is_aligned && sizeof(input_t) == 2 && sizeof(output_t) == 2 && !disable_softmax_opt;
        bool is_opt_dim_49 = !is_masked && !is_transformer_mask && softmax_elements == 49 && batch_count % 2 == 0 && !(ip % 4) && !(op % 4) && sizeof(input_t) == 2 && sizeof(output_t) == 2 && !disable_softmax_opt;
        bool is_opt_dim_32 = !is_masked && !is_transformer_mask && softmax_elements == 32 && batch_count % 2 == 0 && !(ip % 4) && !(op % 4) && ((sizeof(input_t) == 2 && sizeof(output_t) == 2) or (sizeof(input_t) == 4 && sizeof(output_t) == 4)) && !disable_softmax_opt;
        if (is_opt) {
          // only f16 and f32
          if (vec == 2) {
            softmax_warp_forward_opt2_nd<input_t, output_t, acc_t, 7, 2, is_log_softmax>
                    <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>
                    (dst, src, batch_count, softmax_elements_stride, softmax_elements); // 128
            C10_CUDA_KERNEL_LAUNCH_CHECK();                                       \
          } else if (vec == 4) {
            if (sizeof(input_t) == 2) {
              softmax_warp_forward_opt2_nd<input_t, output_t, acc_t, 8, 4, is_log_softmax>
                    <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>
                    (dst, src, batch_count, softmax_elements_stride, softmax_elements); // 256
              C10_CUDA_KERNEL_LAUNCH_CHECK();                                       \
            }
            else if (sizeof(input_t) == 4 && sizeof(output_t) == 4 && softmax_elements == 512) {
                softmax_warp_forward_512_float32<input_t, output_t, acc_t, 9, 4, is_log_softmax>
                    <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(dst, src, batch_count, softmax_elements_stride, softmax_elements);
                C10_CUDA_KERNEL_LAUNCH_CHECK();                                       \
            }
            else if (sizeof(input_t) == 4 && sizeof(output_t) == 4 && softmax_elements == 1024) {
                softmax_warp_forward_1024_float32<input_t, output_t, acc_t, 10, 4, is_log_softmax>
                    <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(dst, src, batch_count, softmax_elements_stride, softmax_elements);
                C10_CUDA_KERNEL_LAUNCH_CHECK();                                       \
            }
            else {
                switch (log2_elements) {
                #define LAUNCH_SOFTMAX_WARP_FORWARD(L2E) case L2E:                    \
                softmax_warp_forward_opt2_nd<input_t, output_t, acc_t, L2E, 4, is_log_softmax>   \
                    <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(dst,   \
                        src, batch_count, softmax_elements_stride, softmax_elements); \
                C10_CUDA_KERNEL_LAUNCH_CHECK();                                       \
                break;
                LAUNCH_SOFTMAX_WARP_FORWARD(8);  // 256
                LAUNCH_SOFTMAX_WARP_FORWARD(9);  // 512
                LAUNCH_SOFTMAX_WARP_FORWARD(10); ; // 1024
                default:
                    break;
              }
            }
          } else if (vec == 8) {
            if (sizeof(input_t) == 2 && sizeof(output_t) == 2 && softmax_elements == 1024) {
                softmax_warp_forward_1024_half_bhalf<input_t, output_t, acc_t, 10, 8, is_log_softmax>
                    <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(dst, src, batch_count, softmax_elements_stride, softmax_elements);
                C10_CUDA_KERNEL_LAUNCH_CHECK();                                       \
            } 
            else if(sizeof(input_t) == 2 && sizeof(output_t) == 2 && softmax_elements == 512){
                softmax_warp_forward_512_half_bhalf<input_t, output_t, acc_t, 9, 8, is_log_softmax>
                        <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(dst, src, batch_count, softmax_elements_stride, softmax_elements);
                C10_CUDA_KERNEL_LAUNCH_CHECK();                                       \
            }
            else {
                switch (log2_elements) {
                    #define LAUNCH_SOFTMAX_WARP_FORWARD(L2E) case L2E:                    \
                    softmax_warp_forward_opt2_nd<input_t, output_t, acc_t, L2E, 8, is_log_softmax>   \
                        <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(dst,   \
                            src, batch_count, softmax_elements_stride, softmax_elements); \
                    C10_CUDA_KERNEL_LAUNCH_CHECK();                                       \
                    break;
                    LAUNCH_SOFTMAX_WARP_FORWARD(9);  // 512
                    LAUNCH_SOFTMAX_WARP_FORWARD(10); ; // 1024
                    default:
                        break;
                }
            }
          }
        }
        else if(is_opt_dim_49){
            softmax_warp_forward_dim_49_half_bhalf<input_t, output_t, acc_t, is_log_softmax>
                <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(dst, src, batch_count, softmax_elements_stride);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
        else if(is_opt_dim_32){
            if (sizeof(input_t) == 4){
                batches_per_warp = 1;
                warp_size = 8;
                warps_per_block = (threads_per_block / warp_size);
                batches_per_block = warps_per_block * batches_per_warp;
                blocks = (batch_count + batches_per_block - 1) / batches_per_block;
                threads = dim3(warp_size, warps_per_block, 1);
                softmax_warp_forward_dim_32_float32<input_t, output_t, acc_t, is_log_softmax>
                <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(dst, src, batch_count, softmax_elements_stride);
            }
            else{
                batches_per_warp = 1;
                warp_size = 4;
                warps_per_block = (threads_per_block / warp_size);
                batches_per_block = warps_per_block * batches_per_warp;
                blocks = (batch_count + batches_per_block - 1) / batches_per_block;
                threads = dim3(warp_size, warps_per_block, 1);
                softmax_warp_forward_dim_32_half_bhalf<input_t, output_t, acc_t, is_log_softmax>
                <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(dst, src, batch_count, softmax_elements_stride);
            }
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
        else if(is_opt_combine_load){
            int middle_thread = softmax_elements / 8 - 64 - 1;
            softmax_warp_forward_combine_load_nd<input_t, output_t, acc_t, is_log_softmax>
                <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(dst, src, batch_count, softmax_elements_stride, middle_thread);
            C10_CUDA_KERNEL_LAUNCH_CHECK();                                       \
        }
        else {
          if(sizeof(input_t) != 8){
            switch (log2_elements) {
            #define LAUNCH_SOFTMAX_WARP_FORWARD(L2E) case L2E:                    \
            softmax_warp_forward<input_t, output_t, acc_t, L2E, is_log_softmax, is_masked>   \
                <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(dst,   \
                    src, batch_count, softmax_elements_stride, softmax_elements, mask, chunk_size, is_transformer_mask); \
            C10_CUDA_KERNEL_LAUNCH_CHECK();                                       \
            break;

            LAUNCH_SOFTMAX_WARP_FORWARD(0);  // 1
            LAUNCH_SOFTMAX_WARP_FORWARD(1);  // 2
            LAUNCH_SOFTMAX_WARP_FORWARD(2);  // 4
            LAUNCH_SOFTMAX_WARP_FORWARD(3);  // 8
            LAUNCH_SOFTMAX_WARP_FORWARD(4);  // 16
            LAUNCH_SOFTMAX_WARP_FORWARD(5);  // 32
            LAUNCH_SOFTMAX_WARP_FORWARD(6);  // 64
            LAUNCH_SOFTMAX_WARP_FORWARD(7);  // 128
            LAUNCH_SOFTMAX_WARP_FORWARD(8);  // 256
            LAUNCH_SOFTMAX_WARP_FORWARD(9);  // 512
            LAUNCH_SOFTMAX_WARP_FORWARD(10); ; // 1024
            default:
                break;
            }
          }
          else{
            switch (log2_elements) {
            #define LAUNCH_SOFTMAX_WARP_FORWARD_DOUBLE(L2E) case L2E:                    \
            softmax_warp_forward_double<input_t, output_t, acc_t, L2E, is_log_softmax, is_masked>   \
                <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(dst,   \
                    src, batch_count, softmax_elements_stride, softmax_elements, mask, chunk_size, is_transformer_mask); \
            C10_CUDA_KERNEL_LAUNCH_CHECK();                                       \
            break;

            LAUNCH_SOFTMAX_WARP_FORWARD_DOUBLE(0);  // 1
            LAUNCH_SOFTMAX_WARP_FORWARD_DOUBLE(1);  // 2
            LAUNCH_SOFTMAX_WARP_FORWARD_DOUBLE(2);  // 4
            LAUNCH_SOFTMAX_WARP_FORWARD_DOUBLE(3);  // 8
            LAUNCH_SOFTMAX_WARP_FORWARD_DOUBLE(4);  // 16
            LAUNCH_SOFTMAX_WARP_FORWARD_DOUBLE(5);  // 32
            LAUNCH_SOFTMAX_WARP_FORWARD_DOUBLE(6);  // 64
            LAUNCH_SOFTMAX_WARP_FORWARD_DOUBLE(7);  // 128
            LAUNCH_SOFTMAX_WARP_FORWARD_DOUBLE(8);  // 256
            LAUNCH_SOFTMAX_WARP_FORWARD_DOUBLE(9);  // 512
            LAUNCH_SOFTMAX_WARP_FORWARD_DOUBLE(10); ; // 1024
            default:
                break;
            }
          }
        }
    }
}

template<typename input_t, typename output_t, typename acc_t, bool is_log_softmax, bool is_masked>
void dispatch_softmax_backward(output_t *grad_input, const input_t *grad, const input_t *output, int softmax_elements, int softmax_elements_stride, int batch_count, const bool *mask = nullptr)
{
    TORCH_INTERNAL_ASSERT( softmax_elements >= 0 && softmax_elements <= 1024 );
    if (softmax_elements == 0) {
       return;
    } else {
        int log2_elements = log2_ceil(softmax_elements);
        const int next_power_of_two = 1 << log2_elements;

        // This value must match the WARP_SIZE constexpr value computed inside softmax_warp_backward.
        int warp_size = at::cuda::warp_size();
        warp_size = (next_power_of_two < warp_size) ? next_power_of_two : warp_size;

        // This value must match the WARP_BATCH constexpr value computed inside softmax_warp_backward.
        int batches_per_warp = (next_power_of_two <= 128) ? 2 : 1;

        // use 128 threads per block to maximimize gpu utilization
        constexpr int threads_per_block = 128;

        int warps_per_block = (threads_per_block / warp_size);
        int batches_per_block = warps_per_block * batches_per_warp;
        int blocks = (batch_count + batches_per_block - 1) / batches_per_block;
        dim3 threads(warp_size, warps_per_block, 1);
        // Launch code would be more elegant if C++ supported FOR CONSTEXPR
        switch (log2_elements) {
            #define LAUNCH_SOFTMAX_WARP_BACKWARD(L2E) case L2E:                      \
            softmax_warp_backward<input_t, output_t, acc_t, L2E, is_log_softmax, is_masked> \
                <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>       \
                (grad_input, grad, output, batch_count, softmax_elements_stride, \
                softmax_elements, mask);                                              \
            C10_CUDA_KERNEL_LAUNCH_CHECK();                                      \
            break;

            LAUNCH_SOFTMAX_WARP_BACKWARD(0); // 1
            LAUNCH_SOFTMAX_WARP_BACKWARD(1); // 2
            LAUNCH_SOFTMAX_WARP_BACKWARD(2); // 4
            LAUNCH_SOFTMAX_WARP_BACKWARD(3); // 8
            LAUNCH_SOFTMAX_WARP_BACKWARD(4); // 16
            LAUNCH_SOFTMAX_WARP_BACKWARD(5); // 32
            LAUNCH_SOFTMAX_WARP_BACKWARD(6); // 64
            LAUNCH_SOFTMAX_WARP_BACKWARD(7); // 128
            LAUNCH_SOFTMAX_WARP_BACKWARD(8); // 256
            LAUNCH_SOFTMAX_WARP_BACKWARD(9); // 512
            LAUNCH_SOFTMAX_WARP_BACKWARD(10); // 1024
            default:
                break;
        }
    }
}
