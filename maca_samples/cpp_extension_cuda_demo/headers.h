#pragma once

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/ArrayRef.h>
#include <ATen/ScalarType.h>
#include <ATen/Scalar.h>
#include <ATen/Tensor.h>
#include <ATen/Storage.h>
#include <ATen/Generator.h>
#include <ATen/Functions.h>
#include <ATen/AccumulateType.h>
#include <ATen/TensorUtils.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>

#include <c10/macros/Macros.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDACachingAllocator.h>

#include <torch/torch.h>
#include <torch/extension.h>
#include <torch/all.h>
#include <torch/csrc/utils/tensor_flatten.h>

#ifdef __CUDACC__
#include <ATen/cuda/NumericLimits.cuh>
#include <ATen/cuda/DeviceUtils.cuh>
#include <ATen/cuda/CUDAGraphsUtils.cuh>
#include <ATen/cuda/detail/IndexUtils.cuh>
#endif
