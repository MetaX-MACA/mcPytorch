#define CUDA_MAX_THREADS 1024

static __device__ inline int p_start(int size, int pad, int kernel, int dilation, int stride) {
  return (size + pad < ((kernel - 1) * dilation + 1)) ? 0 : (size + pad - ((kernel - 1) * dilation + 1)) / stride + 1;
}

static __device__ inline int p_end(int size, int pad, int pooled_size, int stride) {
  return min((size + pad) / stride + 1, pooled_size);
}

template <typename scalar_t, typename accscalar_t>
C10_LAUNCH_BOUNDS_1(CUDA_MAX_THREADS)
__global__ void max_pool_forward_nhwc_opt(const scalar_t* bottom_data, const int nbatch,
                                   const int64_t channels, const int64_t height,
                                   const int64_t width, const int pooled_height, const int pooled_width,
                                   const int kernel_h, const int kernel_w, const int stride_h,
                                   const int stride_w, const int pad_h, const int pad_w,
                                   const int dilation_h, const int dilation_w,
                                   const int in_stride_n, const int in_stride_c,
                                   const int in_stride_h, const int in_stride_w,
                                   const int kernel_stride_C, const int kernel_size_C,
                                   scalar_t* top_data, int64_t* top_mask) {
  
  int c = threadIdx.x;
  int n = blockIdx.z;
  int pw = blockIdx.x;
  int ph = blockIdx.y;
  int64_t index = pooled_height * pooled_width * channels * n + \
          pooled_width * channels * ph + channels * pw + c;
  int hstart = ph * stride_h - pad_h;
  int wstart = pw * stride_w - pad_w;
  int hend = min(hstart + (kernel_h - 1) * dilation_h + 1, height);
  int wend = min(wstart + (kernel_w - 1) * dilation_w + 1, width);
  while(hstart < 0)
    hstart += dilation_h;
  while(wstart < 0)
    wstart += dilation_w;
  accscalar_t maxval = at::numeric_limits<accscalar_t>::lower_bound(); // -Infinity
  int maxidx = hstart * width + wstart;
  // const scalar_t* btm_data = bottom_data + (n * channels + c) * height * width;
  const scalar_t* btm_data = bottom_data + n * channels * height * width + c;
  for (int h = hstart; h < hend; h += dilation_h) {
    for (int w = wstart; w < wend; w += dilation_w) {
      scalar_t val = btm_data[(h * width + w)*channels];
      if ((static_cast<accscalar_t>(val) > maxval) || at::_isnan(val)) {
        maxidx = h * width + w;
        maxval = static_cast<accscalar_t>(val);
      }
    }
  }
  top_data[index] = static_cast<accscalar_t>(maxval);
  top_mask[index] = maxidx;
}


template <typename scalar_t, typename accscalar_t>
C10_LAUNCH_BOUNDS_1(CUDA_MAX_THREADS)
__global__ void max_pool_backward_nhwc_opt(const scalar_t* top_diff,
                                    const int64_t* top_mask, const int nbatch, const int64_t channels,
                                    const int64_t height, const int64_t width, const int pooled_height,
                                    const int pooled_width, const int kernel_h, const int kernel_w,
                                    const int stride_h, const int stride_w, const int pad_h, const int pad_w,
                                    const int dilation_h, const int dilation_w,
                                    const int out_stride_c, const int out_stride_h, const int out_stride_w,
                                    const int kernel_stride_C, const int kernel_size_C,
                                    scalar_t* bottom_diff) {
  int c = threadIdx.x;
  int n = blockIdx.z;
  int h = blockIdx.y;
  int w = blockIdx.x;
  int64_t index = height * width * channels * n + \
          width * channels * h + channels * w + c;
  int phstart = p_start(h, pad_h, kernel_h, dilation_h, stride_h);
  int phend = p_end(h, pad_h, pooled_height, stride_h);
  int pwstart = p_start(w, pad_w, kernel_w, dilation_w, stride_w);
  int pwend = p_end(w, pad_w, pooled_width, stride_w);
  accscalar_t gradient = accscalar_t(0);
  // int offset = (n * channels + c) * pooled_height * pooled_width;
  int offset = n * channels * pooled_height * pooled_width + c;
  #pragma unroll 1
  for (int ph = phstart; ph < phend; ++ph) {
    #pragma unroll 1
    for (int pw = pwstart; pw < pwend; ++pw) {
      if (top_mask[ph*pooled_width*channels + pw*channels + offset] == h * width + w) {
        gradient += static_cast<accscalar_t>(top_diff[ph*pooled_width*channels + pw*channels + offset]);
      }
    }
  }
  bottom_diff[index] = static_cast<scalar_t>(gradient);
}
