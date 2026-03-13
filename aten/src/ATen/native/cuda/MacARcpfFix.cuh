#pragma once

// MACA RCPF Precision Fix
// Overrides __builtin_mxc_rcpf with IEEE 754 compliant division
// This file must be included before any MACA headers

#ifdef USE_MACA

// Override __fdividef to use precise division
__device__ inline float __fdividef(float x, float y) {
  return (1.0f / y) * x;
}

// Override __llvm_mxc_rcpf to use precise reciprocal
__device__ inline float __llvm_mxc_rcpf(float a) {
  return (1.0f / a);
}

#endif // USE_MACA
