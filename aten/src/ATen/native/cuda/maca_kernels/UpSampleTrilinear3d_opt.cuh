#pragma once
#include <ATen/native/cuda/UpSample.cuh>
#include <ATen/native/cuda/MemoryAccess.cuh>
#include <ATen/native/cuda/UpSample.cuh>

namespace at::native {
namespace {
template <typename scalar_t, typename accscalar_t>
C10_LAUNCH_BOUNDS_1(512)
__global__ void upsample_trilinear3d_out_frame_opt_NDHWC(
    const int n,
    const accscalar_t rdepth,
    const accscalar_t rheight,
    const accscalar_t rwidth,
    const bool align_corners,
    const PackedTensorAccessor64<const scalar_t, 5> idata,
    PackedTensorAccessor64<scalar_t, 5> odata) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  int b = blockIdx.y;

  const int batchsize = idata.size(0);
  const int channels = idata.size(1);
  const int depth1 = idata.size(2);
  const int height1 = idata.size(3);
  const int width1 = idata.size(4);
  const int depth2 = odata.size(2);
  const int height2 = odata.size(3);
  const int width2 = odata.size(4);

  if (index < n) {
    const int c = index % channels;
    index = index / channels;
    const int w2 = index % width2; // 0:width2-1
    index = index / width2;
    const int h2 = index % height2; // 0:height2-1
    const int t2 = index / height2; // 0:depth2-1
    // special case: just copy
    if (depth1 == depth2 && height1 == height2 && width1 == width2) {
      const int t1 = t2;
      const int h1 = h2;
      const int w1 = w2;

      const scalar_t val = idata[b][c][t1][h1][w1];
      odata[b][c][t2][h2][w2] = val;
      return;
    }
    //
    const accscalar_t t1r = area_pixel_compute_source_index<accscalar_t>(
        rdepth, t2, align_corners, /*cubic=*/false);
    const int t1 = t1r;
    const int t1p = (t1 < depth1 - 1) ? 1 : 0;
    const accscalar_t t1lambda = t1r - t1;
    const accscalar_t t0lambda = static_cast<accscalar_t>(1) - t1lambda;
    //
    const accscalar_t h1r = area_pixel_compute_source_index<accscalar_t>(
        rheight, h2, align_corners, /*cubic=*/false);
    const int h1 = h1r;
    const int h1p = (h1 < height1 - 1) ? 1 : 0;
    const accscalar_t h1lambda = h1r - h1;
    const accscalar_t h0lambda = static_cast<accscalar_t>(1) - h1lambda;
    //
    const accscalar_t w1r = area_pixel_compute_source_index<accscalar_t>(
        rwidth, w2, align_corners, /*cubic=*/false);
    const int w1 = w1r;
    const int w1p = (w1 < width1 - 1) ? 1 : 0;
    const accscalar_t w1lambda = w1r - w1;
    const accscalar_t w0lambda = static_cast<accscalar_t>(1) - w1lambda;
    //
    const accscalar_t val = t0lambda *
            (h0lambda *
                  (w0lambda * idata[b][c][t1][h1][w1] +
                  w1lambda * idata[b][c][t1][h1][w1 + w1p]) +
              h1lambda *
                  (w0lambda * idata[b][c][t1][h1 + h1p][w1] +
                  w1lambda * idata[b][c][t1][h1 + h1p][w1 + w1p])) +
        t1lambda *
            (h0lambda *
                  (w0lambda * idata[b][c][t1 + t1p][h1][w1] +
                  w1lambda * idata[b][c][t1 + t1p][h1][w1 + w1p]) +
              h1lambda *
                  (w0lambda * idata[b][c][t1 + t1p][h1 + h1p][w1] +
                  w1lambda * idata[b][c][t1 + t1p][h1 + h1p][w1 + w1p]));
    odata[b][c][t2][h2][w2] = static_cast<scalar_t>(val);
  }
}  
}
}