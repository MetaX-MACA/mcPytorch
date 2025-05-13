#pragma once
#include <ATen/native/cuda/UpSample.cuh>
#include <ATen/native/cuda/MemoryAccess.cuh>

namespace at::native {
namespace {

template <typename scalar_t, typename accscalar_t>
C10_LAUNCH_BOUNDS_1(1024)
__global__ void upsample_bicubic2d_out_frame_opt(
    const int num_elements, const accscalar_t height_scale, const accscalar_t width_scale, const bool align_corners, 
    const scalar_t* const idata, scalar_t* const odata, const int batchsize, const int channels, 
    const int input_height, const int input_width, const int output_height, const int output_width) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;

    int n = blockIdx.z;
    int c = blockIdx.y;
    int64_t offsets_in = n * channels * input_height * input_width + c * input_height * input_width;
    int64_t offsets_out = n * channels * output_height * output_width + c * output_height * output_width;

    if (index >= num_elements) {
        return;
    }

    // Special case: input and output are the same size, just copy
    const int output_x = index % output_width;
    const int output_y = index / output_width;

    if (input_height == output_height && input_width == output_width) {
        const scalar_t val = idata[offsets_out + output_y * output_width + output_x];
        odata[offsets_out + output_y * output_width + output_x] = val;
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

    accscalar_t coefficients[4];

    using LoadT = at::native::memory::aligned_vector<scalar_t, 4>;
    #pragma unroll
    for (int k = 0; k < 4; k++) {
        scalar_t inp[4];
        LoadT* p_inp = reinterpret_cast<LoadT*>(&inp);
        int access_y = max(min(in_y - 1 + k, input_height - 1), 0);
        if (in_x - 1 >= 0 && in_x + 2 < input_width) {
            int access_x = in_x - 1;
            *p_inp = *reinterpret_cast<const LoadT*>(idata + offsets_in + access_y * input_width + access_x);
        } else {
            #pragma unroll
            for (int i=0; i<4; i++){
                int access_x = max(min(in_x - 1 + i, input_width - 1), 0);
                inp[i] = idata[offsets_in + access_y * input_width + access_x];
            }
        }
        coefficients[k] = cubic_interp1d(inp[0], inp[1], inp[2], inp[3], t_x);
    }

    odata[offsets_out + output_y * output_width + output_x] = static_cast<scalar_t>(cubic_interp1d(
        coefficients[0],
        coefficients[1],
        coefficients[2],
        coefficients[3],
        t_y));
}

}
}