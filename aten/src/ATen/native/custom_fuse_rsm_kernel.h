#pragma once

#include <ATen/core/Tensor.h>
#include <ATen/native/DispatchStub.h>
#include <c10/util/accumulate.h>

namespace at::native {

using forward_fn = void (*)(
    const Tensor* /* X */,
    const Tensor* /* variance */,
    const Tensor* /* weight */,
    Tensor* /* out */);

DECLARE_DISPATCH(forward_fn, customFuseRsmKernel)
DEFINE_DISPATCH(customFuseRsmKernel);

} // namespace at::native
