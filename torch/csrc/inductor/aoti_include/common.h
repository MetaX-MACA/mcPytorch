#pragma once

#include <array>
//When compiling pytorch on arm and maca platform, we also use experimental/filesystem header files
#if defined(USE_MACA) && ((defined(__GNUC__) && __GNUC__ < 9 && !defined(__MXCC__)) || defined(__aarch64__))
#include <experimental/filesystem>
#else
#include <filesystem>
#endif
#include <optional>

#include <torch/csrc/inductor/aoti_runtime/interface.h>
#include <torch/csrc/inductor/aoti_runtime/model.h>

#include <c10/util/generic_math.h>
#include <torch/csrc/inductor/aoti_runtime/scalar_to_tensor.h>

// Round up to the nearest multiple of 64
[[maybe_unused]] inline int64_t align(int64_t nbytes) {
  return (nbytes + 64 - 1) & -64;
}
