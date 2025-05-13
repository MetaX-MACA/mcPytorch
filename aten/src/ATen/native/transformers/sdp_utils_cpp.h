#pragma once
#include <cstdint>
namespace sdp {

constexpr int32_t num_backends = 4;
enum class SDPBackend {
  error = -1,
  math = 0,
  flash_attention = 1,
  efficient_attention = 2,
  mha_fusion_attention = 3
};
} // namespace sdp