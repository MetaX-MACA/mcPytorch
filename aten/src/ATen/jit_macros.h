#pragma once
#include <ATen/cuda/CUDAConfig.h>
#include <string>

// AT_USE_JITERATOR(), controls whether we jit some elementwise kernels
#ifdef USE_MACA
#define AT_USE_JITERATOR() false
#else
static_assert(0, "Unexpected branch");
#define AT_USE_JITERATOR() true
#endif // USE_MACA
#define jiterator_stringify(...) std::string(#__VA_ARGS__);
