#pragma once

// MACA RCPF Precision Fix: Include before CUDA headers
#ifdef USE_MACA
#include <ATen/native/cuda/MacARcpfFix.cuh>
#endif

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <c10/macros/Export.h>

// Use TORCH_CUDA_CPP_API or TORCH_CUDA_CU_API for exports from this folder
