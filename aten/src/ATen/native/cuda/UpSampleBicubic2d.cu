#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/AccumulateType.h>
#include <ATen/ceil_div.h>
#include <ATen/Dispatch.h>
#include <ATen/TensorUtils.h>
#include <ATen/Utils.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/native/cuda/UpSample.cuh>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/upsample_bicubic2d_native.h>
#include <ATen/ops/upsample_bicubic2d_backward_native.h>
#endif

#ifdef USE_MACA
#include "maca_kernels/UpSampleBicubic2d_opt.cuh"
#endif

namespace at::native {
namespace {

template <typename scalar_t, typename accscalar_t>
C10_LAUNCH_BOUNDS_1(1024)
__global__ void upsample_bicubic2d_out_frame(
    const int num_elements,
    const accscalar_t height_scale,
    const accscalar_t width_scale,
    const bool align_corners,
    const PackedTensorAccessor64<const scalar_t, 4> idata,
    PackedTensorAccessor64<scalar_t, 4> odata) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;

  const int batchsize = idata.size(0);
  const int channels = idata.size(1);
  const int input_height = idata.size(2);
  const int input_width = idata.size(3);
  const int output_height = odata.size(2);
  const int output_width = odata.size(3);

  if (index >= num_elements) {
    return;
  }

  // Special case: input and output are the same size, just copy
  const int output_x = index % output_width;
  const int output_y = index / output_width;

  if (input_height == output_height && input_width == output_width) {
    for (int n = 0; n < batchsize; n++) {
      for (int c = 0; c < channels; c++) {
        const scalar_t val = idata[n][c][output_y][output_x];
        odata[n][c][output_y][output_x] = val;
      }
    }
    return;
  }

  // Interpolation kernel
  accscalar_t real_x = area_pixel_compute_source_index(
      width_scale, output_x, align_corners, /*cubic=*/true);
  int in_x = floorf(real_x);
  accscalar_t t_x = real_x - in_x;

  accscalar_t real_y = area_pixel_compute_source_index(
      height_scale, output_y, align_corners, /*cubic=*/true);
  int in_y = floorf(real_y);
  accscalar_t t_y = real_y - in_y;

  for (int n = 0; n < batchsize; n++) {
    for (int c = 0; c < channels; c++) {
      accscalar_t coefficients[4];

      for (int k = 0; k < 4; k++) {
        coefficients[k] = cubic_interp1d(
            upsample_get_value_bounded<scalar_t>(
                idata, n, c, input_height, input_width, in_y - 1 + k, in_x - 1),
            upsample_get_value_bounded<scalar_t>(
                idata, n, c, input_height, input_width, in_y - 1 + k, in_x + 0),
            upsample_get_value_bounded<scalar_t>(
                idata, n, c, input_height, input_width, in_y - 1 + k, in_x + 1),
            upsample_get_value_bounded<scalar_t>(
                idata, n, c, input_height, input_width, in_y - 1 + k, in_x + 2),
            t_x);
      }

      odata[n][c][output_y][output_x] = static_cast<scalar_t>(cubic_interp1d(
          coefficients[0],
          coefficients[1],
          coefficients[2],
          coefficients[3],
          t_y));
    }
  }
}

// Backward (adjoint) operation 1 <- 2 (accumulates)
template <typename scalar_t, typename accscalar_t>
C10_LAUNCH_BOUNDS_1(1024)
__global__ void upsample_bicubic2d_backward_out_frame(
    const int num_elements,
    const accscalar_t height_scale,
    const accscalar_t width_scale,
    const bool align_corners,
    PackedTensorAccessor64<scalar_t, 4> idata,
    const PackedTensorAccessor64<const scalar_t, 4> odata) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;

  const int batchsize = idata.size(0);
  const int channels = idata.size(1);
  const int input_height = idata.size(2);
  const int input_width = idata.size(3);
  const int output_height = odata.size(2);
  const int output_width = odata.size(3);

  if (index >= num_elements) {
    return;
  }

  const int output_x = index % output_width;
  const int output_y = index / output_width;
  // special case: output_xust copy
  if (input_height == output_height && input_width == output_width) {
    for (int n = 0; n < batchsize; n++) {
      for (int c = 0; c < channels; ++c) {
        const scalar_t val = odata[n][c][output_y][output_x];
        idata[n][c][output_y][output_x] = val;
      }
    }
    return;
  }

  accscalar_t real_x = area_pixel_compute_source_index(
      width_scale, output_x, align_corners, /*cubic=*/true);
  int input_x = floorf(real_x);
  accscalar_t t_x = real_x - input_x;

  accscalar_t real_y = area_pixel_compute_source_index(
      height_scale, output_y, align_corners, /*cubic=*/true);
  int input_y = floorf(real_y);
  accscalar_t t_y = real_y - input_y;

  accscalar_t x_coeffs[4];
  accscalar_t y_coeffs[4];

  get_cubic_upsampling_coefficients(x_coeffs, t_x);
  get_cubic_upsampling_coefficients(y_coeffs, t_y);

  for (int n = 0; n < batchsize; n++) {
    for (int c = 0; c < channels; ++c) {
      scalar_t out_value = odata[n][c][output_y][output_x];
      for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
          upsample_increment_value_bounded<scalar_t, accscalar_t>(
              idata,
              n,
              c,
              input_height,
              input_width,
              input_y - 1 + i,
              input_x - 1 + j,
              out_value * y_coeffs[i] * x_coeffs[j]);
        }
      }
    }
  }
}

static void upsample_bicubic2d_out_cuda_template(
    const Tensor& output,
    const Tensor& input,
    IntArrayRef output_size,
    bool align_corners,
    std::optional<double> scales_h,
    std::optional<double> scales_w) {
  TensorArg input_arg{input, "input", 1}, output_arg{output, "output", 2};
  checkAllSameGPU(__func__, {input_arg, output_arg});

  int output_height = output_size[0];
  int output_width = output_size[1];

  int input_height = input.size(2);
  int input_width = input.size(3);

  output.zero_();

  const int num_output_elements = output_height * output_width;
  const int max_threads = std::min(
      at::cuda::getCurrentDeviceProperties()->maxThreadsPerBlock, 1024);

  // Launch kernel
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

#ifdef USE_MACA
  if(maca_unlikely(at::maca::get_maca_enable_print_upsample2d_info())){
    printf("Upsample bicubic2d info: InputSize = %d,%d,%d,%d, InputStride = %d,%d,%d,%d,\
OutputSize = %d,%d,%d,%d, OutputStride = %d,%d,%d,%d,\
output_size = %d,%d, align_corners = %d\n",
    input.size(0),input.size(1),input.size(2),input.size(3), input.stride(0),input.stride(1),input.stride(2),input.stride(3),
    output.size(0),output.size(1),output.size(2),output.size(3), output.stride(0),output.stride(1),output.stride(2),output.stride(3),
    output_height, output_width, int(align_corners));
  }
#endif

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16,
      input.scalar_type(), "upsample_bicubic2d_out_frame", [&] {
        using accscalar_t = at::acc_type<scalar_t, true>;

        auto idata = input.packed_accessor64<const scalar_t, 4>();
        auto odata = output.packed_accessor64<scalar_t, 4>();

        // Get scaling factors
        const accscalar_t rheight = area_pixel_compute_scale<accscalar_t>(
            input_height, output_height, align_corners, scales_h);
        const accscalar_t rwidth = area_pixel_compute_scale<accscalar_t>(
            input_width, output_width, align_corners, scales_w);
#ifdef USE_MACA
        if (input.size(0) * input.size(1) * num_output_elements > 0 && input.is_contiguous() && output.is_contiguous() &&
            std::is_same<scalar_t, float>::value &&
            maca_likely(!at::maca::get_maca_disable_upsample_bicubic2d_opt())){
          int batchsize = input.size(0);
          int channels = input.size(1);
          dim3 block(512);
          dim3 grid(ceil_div(num_output_elements, 512), channels, batchsize);
          scalar_t *output_data = output.data_ptr<scalar_t>();
          scalar_t *input_data = input.data_ptr<scalar_t>();
          upsample_bicubic2d_out_frame_opt<scalar_t, accscalar_t>
              <<<grid,
                block,
                0,
                stream>>>(
                  num_output_elements, rheight, rwidth, align_corners,
                  input_data, output_data, batchsize, channels, 
                  input_height, input_width, output_height, output_width
                  );
        } else if (input.size(0) * input.size(1) * num_output_elements > 0 &&
            input.is_contiguous(at::MemoryFormat::ChannelsLast) && output.is_contiguous(at::MemoryFormat::ChannelsLast) &&
            std::is_same<scalar_t, float>::value && input.size(1) % 64 == 0 &&  // channels mod(64) == 0 to avoid partial write
            maca_likely(!at::maca::get_maca_disable_upsample_bicubic2d_opt())){
          int batchsize = input.size(0);
          int channels = input.size(1);
          dim3 block(512);
          // channels_per_thread = 4, threads_per_block * channels_per_thread
          dim3 grid(ceil_div(num_output_elements * channels, 512 * 4), batchsize, 1);
          scalar_t *output_data = output.data_ptr<scalar_t>();
          scalar_t *input_data = input.data_ptr<scalar_t>();          
          upsample_bicubic2d_out_frame_opt_nhwc<scalar_t, accscalar_t, 4>
              <<<grid,
                block,
                0,
                stream>>>(
                  num_output_elements, rheight, rwidth, align_corners,
                  input_data, output_data, batchsize, channels, 
                  input_height, input_width, output_height, output_width
                  );
        } else {
            upsample_bicubic2d_out_frame<scalar_t, accscalar_t>
            <<<ceil_div(num_output_elements, max_threads),
               max_threads,
               0,
               stream>>>(
                num_output_elements,
                rheight,
                rwidth,
                align_corners,
                idata,
                odata);
        }
#else
        upsample_bicubic2d_out_frame<scalar_t, accscalar_t>
            <<<ceil_div(num_output_elements, max_threads),
               max_threads,
               0,
               stream>>>(
                num_output_elements,
                rheight,
                rwidth,
                align_corners,
                idata,
                odata);
#endif
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
}

static void upsample_bicubic2d_backward_out_cuda_template(
    const Tensor& grad_input,
    const Tensor& grad_output_,
    IntArrayRef output_size,
    IntArrayRef input_size,
    bool align_corners,
    std::optional<double> scales_h,
    std::optional<double> scales_w) {
  TensorArg grad_input_arg{grad_input, "grad_input", 1},
      grad_output_arg{grad_output_, "grad_output_", 2};
  checkAllSameGPU(__func__, {grad_output_arg, grad_input_arg});

  int output_height = output_size[0];
  int output_width = output_size[1];

  int input_height = input_size[2];
  int input_width = input_size[3];

  Tensor grad_output = grad_output_.contiguous();

  grad_input.zero_();

  const int num_kernels = output_height * output_width;
  const int num_threads = std::min(
      at::cuda::getCurrentDeviceProperties()->maxThreadsPerBlock, 1024);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16,
      grad_output.scalar_type(), "upsample_bicubic2d_backward_out_frame", [&] {
        using accscalar_t = at::acc_type<scalar_t, true>;

        auto idata = grad_input.packed_accessor64<scalar_t, 4>();
        auto odata = grad_output.packed_accessor64<const scalar_t, 4>();

        const accscalar_t rheight = area_pixel_compute_scale<accscalar_t>(
            input_height, output_height, align_corners, scales_h);
        const accscalar_t rwidth = area_pixel_compute_scale<accscalar_t>(
            input_width, output_width, align_corners, scales_w);

        upsample_bicubic2d_backward_out_frame<scalar_t, accscalar_t>
            <<<ceil_div(num_kernels, num_threads),
               num_threads,
               0,
               stream>>>(
                num_kernels, rheight, rwidth, align_corners, idata, odata);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
}

} // namespace

TORCH_IMPL_FUNC(upsample_bicubic2d_out_cuda) (
    const Tensor& input,
    IntArrayRef output_size,
    bool align_corners,
    std::optional<double> scales_h,
    std::optional<double> scales_w,
    const Tensor& output) {
  upsample_bicubic2d_out_cuda_template(output, input, output_size, align_corners, scales_h, scales_w);
}

TORCH_IMPL_FUNC(upsample_bicubic2d_backward_out_cuda) (
    const Tensor& grad_output,
    IntArrayRef output_size,
    IntArrayRef input_size,
    bool align_corners,
    std::optional<double> scales_h,
    std::optional<double> scales_w,
    const Tensor& grad_input) {
  // See Note [Writing Nondeterministic Operations]
  // Nondeterministic because of atomicAdd usage
  globalContext().alertNotDeterministic("upsample_bicubic2d_backward_out_cuda");
  upsample_bicubic2d_backward_out_cuda_template(
      grad_input, grad_output, output_size, input_size, align_corners, scales_h, scales_w);
}

} // namespace at::native
