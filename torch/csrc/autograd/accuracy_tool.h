#pragma once

#include <ATen/ATen.h>
#include <ATen/record_function.h>

namespace torch {
namespace autograd {

TORCH_API at::CallbackHandle getTLSHandle();
TORCH_API void pushAccuracyCallbacks(const std::unordered_set<at::RecordScope>& scopes);

TORCH_API std::string getDir();

}} // namespace torch::autograd
