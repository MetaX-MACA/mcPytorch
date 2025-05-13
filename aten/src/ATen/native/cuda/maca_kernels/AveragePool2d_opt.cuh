#pragma once

namespace at::native {
namespace avgpool {

template<typename scalar_t, int vec_size>
struct alignas(sizeof(scalar_t) * vec_size) aligned_vector {
  scalar_t val[vec_size];
};

template <int nt, int vt, typename scalar_t, typename accscalar_t>
__global__ void avg_pool2d_backward_out_cuda_frame_opt(const int nthreads, const scalar_t* const top_diff,
    const int64_t channels, const int64_t height,
    const int64_t width, const int64_t pooled_height, const int64_t pooled_width,
    const int kernel_h, const int kernel_w, const int stride_h,
    const int stride_w, const int pad_h, const int pad_w,
    scalar_t* const bottom_diff, const int divisor_override,
    bool count_include_pad, bool use_divisor, int h_t) {

    int thread_num_w = width/vt;
    int thread_num_h = height/h_t;
    int64_t linear_idx = blockIdx.x * blockDim.x + threadIdx.x;
    using StoreT = aligned_vector<scalar_t, vt>;
    scalar_t grad_out[vt];
    StoreT* p_grad_out = reinterpret_cast<StoreT*>(&grad_out);
    StoreT* out;

    if (linear_idx < channels * thread_num_w * thread_num_h) {

        const int w = linear_idx % thread_num_w * vt;
        const int h = (linear_idx / thread_num_w) % thread_num_h * h_t;
        const int c = linear_idx / thread_num_w / thread_num_h;
        const int n = blockIdx.z;

        // Origin code:
        // const int phstart = (h < kernel_h) ? 0 : (h - kernel_h) / stride_h + 1;
        // const int phend = min(h / stride_h + 1, pooled_height);
        // const int pwstart = (w < kernel_w) ? 0 : (w - kernel_w) / stride_w + 1;
        // const int pwend = min(w / stride_w + 1, pooled_width);
        // const scalar_t* const top_diff_slice =
        //     top_diff + (n * channels + c) * pooled_height * pooled_width;
        // for (int ph = phstart; ph < phend; ++ph) {
        //   for (int pw = pwstart; pw < pwend; ++pw) {
        //     gradient += top_diff_slice[ph * pooled_width + pw] / divide_factor;
        //   }
        // }

        // When kernel_h == stride_h, phend = phstart + 1
        // When kernel_w == stride_w, pwend = pwstart + 1
        // gradient of elements from [n, c, h, w] to [n, c, h + kernel_h - 1, w + kernel_w - 1]
        // only comes from one element of top_diff : [n, c, ph, pw]
        // so we can read & compute gradient once and write multiple times

        // Only comes from one element [n, c, ph, pw]
        // No need to loop
        int ph = (h < kernel_h) ? 0 : (h - kernel_h) / stride_h + 1;
        int pw = (w < kernel_w) ? 0 : (w - kernel_w) / stride_w + 1;

        accscalar_t gradient = accscalar_t(0);
        const scalar_t* const top_diff_slice =
            top_diff + (n * channels + c) * pooled_height * pooled_width;

        // figure out the pooling size
        int hstart = ph * stride_h - pad_h;
        int wstart = pw * stride_w - pad_w;
        int hend = min(hstart + kernel_h, height + pad_h);
        int wend = min(wstart + kernel_w, width + pad_w);
        int pool_size = (hend - hstart) * (wend - wstart);
        hstart = max(hstart, 0);
        wstart = max(wstart, 0);
        hend = min(hend, height);
        wend = min(wend, width);

        // when ph >= pooled height or pw >= pooled width, skip gradient calculation
        if (hstart < hend || wstart < wend) {
            int divide_factor;
            if (use_divisor) {
                divide_factor = divisor_override;
            } else {
                if(count_include_pad) {
                divide_factor = pool_size;
                } else {
                divide_factor = (hend - hstart) * (wend - wstart);
                }
            }
            gradient = top_diff_slice[ph * pooled_width + pw] / divide_factor;
        }
        // vector store
        #pragma unroll
        for (int i = 0; i < vt; i++)  {
            grad_out[i] = static_cast<scalar_t>(gradient);
        }

        int64_t offset = n *channels * height * width + c * height * width + w;

        for (int i=0; i<h_t && h+i<height; i++){
            int64_t h_offset = (h + i) * width;
            out = reinterpret_cast<StoreT*>(bottom_diff + offset + h_offset);
            *out = *p_grad_out;
        }
    }
}

template <int nt, typename scalar_t, typename accscalar_t>
void launch_avg_pool2d_backward_out_cuda_frame_opt(const int nthreads, const scalar_t* const top_diff,
    const int64_t channels, const int64_t height,
    const int64_t width, const int64_t pooled_height, const int64_t pooled_width,
    const int kernel_h, const int kernel_w, const int stride_h,
    const int stride_w, const int pad_h, const int pad_w,
    scalar_t* const bottom_diff, const int divisor_override,
    bool count_include_pad, bool use_divisor) {

    int64_t batch = nthreads / channels / height / width;
    auto stream = at::cuda::getCurrentCUDAStream();

    dim3 block(nt);
    int vec = sizeof(scalar_t) > 2 ? 4 : 8;
    while(kernel_w % vec != 0) {
        vec /= 2;
    }
    while(width % vec != 0) {
        vec /= 2;
    }

    auto ip = reinterpret_cast<uintptr_t>(top_diff);
    if (ip % (sizeof(scalar_t) * vec) != 0) vec=1;

    // kernel_w == stride_w && pad_w == 0
    // write vec per thread
    int thread_num_w = width/vec;

    // kernel_h == stride_h && pad_h == 0
    // write h_t per thread
    int h_t = min(kernel_h, 8);
    while(kernel_h % h_t != 0){
        h_t /= 2;
    }
    while(height % h_t != 0){
        h_t /= 2;
    }
    int thread_num_h = height/h_t;

    int grid_dim_x = (channels * thread_num_h * thread_num_w + block.x - 1) / block.x;
    dim3 grid(grid_dim_x, 1, batch);

    if (vec == 8) {
        avg_pool2d_backward_out_cuda_frame_opt<nt, 8, scalar_t, accscalar_t><<<grid, block, 0, stream>>>(
            nthreads, top_diff, channels, height, width, pooled_height, pooled_width,
            kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w,
            bottom_diff, divisor_override, count_include_pad, use_divisor, h_t);
    } else if (vec == 4) {
        avg_pool2d_backward_out_cuda_frame_opt<nt, 4, scalar_t, accscalar_t><<<grid, block, 0, stream>>>(
            nthreads, top_diff, channels, height, width, pooled_height, pooled_width,
            kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w,
            bottom_diff, divisor_override, count_include_pad, use_divisor, h_t);        
    } else if (vec == 2) {
        avg_pool2d_backward_out_cuda_frame_opt<nt, 2, scalar_t, accscalar_t><<<grid, block, 0, stream>>>(
            nthreads, top_diff, channels, height, width, pooled_height, pooled_width,
            kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w,
            bottom_diff, divisor_override, count_include_pad, use_divisor, h_t); 
    } else {
        avg_pool2d_backward_out_cuda_frame_opt<nt, 1, scalar_t, accscalar_t><<<grid, block, 0, stream>>>(
            nthreads, top_diff, channels, height, width, pooled_height, pooled_width,
            kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w,
            bottom_diff, divisor_override, count_include_pad, use_divisor, h_t); 
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}
}