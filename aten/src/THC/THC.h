#pragma once

#include <ATen/ceil_div.h>
#include <ATen/cuda/Exceptions.h>
#include <ATen/cuda/CUDAContext.h>

#define THCCeilDiv at::ceil_div
#define THCudaCheck AT_CUDA_CHECK

