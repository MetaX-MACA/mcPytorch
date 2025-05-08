#pragma once
#include <algorithm>
#include <ATen/cuda/CUDAContext.h>
#include <c10/macros/Macros.h>
#include <ATen/cuda/detail/OffsetCalculator.cuh>
#include <string>

// 8 * WARP_SIZE is optimized and avoid recompiling
#define MAX_THREADS (8 * C10_WARP_SIZE)

inline int getMaxGridSize(int grid_x_y, int size, int ratio = 1) {
  auto max_sm_size = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  int max_wave_num = max_sm_size * 8 * 4 * ratio;
  int max_grid_size = 1;
  int grid_size = 1;
  while(max_grid_size * grid_x_y <= max_wave_num && max_grid_size <= size) {
    grid_size = max_grid_size;
    max_grid_size <<= 1;
  }
  if ((size / (grid_size * 2) > 1) && (grid_size * 2) <= size && abs((grid_size * 2) * grid_x_y - max_wave_num) < abs(grid_size * grid_x_y - max_wave_num)) {
    grid_size *= 2;
  }
  return grid_size;
}

inline dim3 getMaxBlockSize(size_t dim_size) {
  size_t block_size = 1;
  size_t max_block_size = std::min(dim_size, static_cast<size_t>(MAX_THREADS));
  while (block_size < (max_block_size)) block_size *= 2;
  // Launch at least a single warp - the kernel assumes that.
  block_size = std::max(block_size, static_cast<size_t>(at::cuda::warp_size()));
  return dim3(block_size);
}

inline int log2_floor(int value) {
  int log2_value = 0;
  while ((1 << log2_value) <= value) ++log2_value;
  return log2_value == 0 ? log2_value : log2_value - 1;
}

template<typename dtype>
inline int getSplitGridZ(int s0, int s1, int s2) {
  // ldg128
  auto warp_size = at::cuda::warp_size();
  int max_load_size = warp_size * 128;
  int s0_bytes = s0 * sizeof(dtype) * 8;
  if (max_load_size % s0_bytes != 0) {
    return 1;
  }
  int split_z = max_load_size / s0_bytes;
  // get the max split_z
  while(s2 % split_z != 0) {
    split_z /= 2;
  }
  // adjust split_z ensure can reach the max_wave_num
  auto max_sm_size = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  int max_wave_num = max_sm_size * 8 * 4;
  while((split_z / 2) * s0 >= warp_size && (s2 / (split_z / 2)) * s1 <= max_wave_num) {
    split_z /= 2;
  }
  if (split_z < s2) {
    return split_z;
  } else {
    int log2_elements = log2_floor(s2);
    return 1 << log2_elements;
  }
}

template<typename dtype>
inline int getSplitGridZS(int s0, int s1, int s2) {
  // ldg128
  // s0 is power of 2
  auto warp_size = at::cuda::warp_size();
  int max_load_size = 128;
  int s0_bytes = sizeof(dtype) * 8;
  int vec = max_load_size / s0_bytes;
  vec = std::min(vec, s0);
  if (s0 % 2 != 0) {
    return 1;
  }
  int split_z = warp_size / (s0 / vec);
  // adjust split_z ensure can reach the max_wave_num
  auto max_sm_size = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  int max_wave_num = max_sm_size * 8 * 4;
  while((split_z / 2) * s0 >= warp_size && (s2 / (split_z / 2)) * s1 <= max_wave_num) {
    split_z /= 2;
  }
  // s0 * s2 >= warp_size, if split_z > s2 then s2, vec = 1
  if (split_z < s2) {
    return split_z;
  } else {
    int log2_elements = log2_floor(s2);
    return 1 << log2_elements;
  }
}

template<typename dtype>
inline int getVectorizedAlignment(void* pointer, int alignment_size) {
  if (alignment_size == 0) return 1;
  auto ip = reinterpret_cast<uintptr_t>(pointer);
  int vec_size = 8;
  while(alignment_size % vec_size != 0) {
    vec_size /= 2;
  }
  vec_size = sizeof(dtype) > 2 ? std::min(4, vec_size) : vec_size;
  if (ip % (sizeof(dtype) * vec_size) == 0) {
    return vec_size;
  } else {
    return 1;
  }
}
template<int N, bool signed_strides = false, typename func_t>
inline void get_elementwise_info(int ndim, int narity, OffsetCalculator<N, uint32_t, signed_strides> offset_calc, std::string other_info, func_t f){
  if(maca_unlikely(at::maca::get_maca_enable_elementwise_kernel_info())){
    std::string kernel_info = typeid(f).name();
    std::replace(kernel_info.begin(), kernel_info.end(), ',', '_');
    std::string str = other_info + "_" + kernel_info + "," + std::to_string(ndim) + "," + std::to_string(narity) + ",";
    for (int i = 0;i < ndim; i++){
      str += std::to_string(offset_calc.sizes_[i].divisor);
      str += ",";
    }
    for (int i = 0; i < narity + 1; i++){
      for (int j = 0; j < ndim; j++){
        str += std::to_string(offset_calc.strides_[j][i]);
        str += ",";
      }
    }
    printf("%s\n", str.c_str());
  }
}

template<int N, bool signed_strides = false, typename func_t>
inline void get_elementwise_info(int ndim,
                                 int narity,
                                 OffsetCalculator<N, uint32_t, signed_strides> offset_calc,
                                 at::detail::Array<at::ScalarType, N> dtypes,
                                 std::string other_info,
                                 func_t f){
  if(maca_unlikely(at::maca::get_maca_enable_elementwise_kernel_info())){
    std::string kernel_info = typeid(f).name();
    std::replace(kernel_info.begin(), kernel_info.end(), ',', '_');
    std::string str = other_info + "_" + kernel_info + "," + std::to_string(ndim) + "," + std::to_string(narity) + ",";
    for (int i = 0;i < ndim; i++){
      str += std::to_string(offset_calc.sizes_[i].divisor);
      str += ",";
    }
    for (int i = 0; i < narity + 1; i++){
      for (int j = 0; j < ndim; j++){
        str += std::to_string(offset_calc.strides_[j][i]);
        str += ",";
      }
    }
    for (int i = 0; i < narity + 1; i++){
      str += c10::str(dtypes[i]).c_str();
      str += ",";
    }
    printf("%s\n", str.c_str());
  }
}


template <typename T>
inline __device__ const T func_reinterpret_cast(char* src) {
  return  *reinterpret_cast<T*>(src);
}
template <bool>
inline __device__ const unsigned char func_reinterpret_cast(char* src) {
  static_assert(sizeof(bool) == sizeof(char), "");
  return  *reinterpret_cast<const unsigned char*>(src);
}

template<typename index_t, typename stride_t>
inline bool check_opt_dim_3(index_t size1, stride_t stride10, stride_t stride20) {
  if (stride10 == stride20) {
    return false;
  }
  if (stride10 > stride20) {
    return stride10 / stride20 >= size1;
  } else {
    return stride20 / stride10 >= size1;
  }
}

template<typename index_t, typename stride_t>
inline bool check_opt_dim_4(index_t size1, index_t size2, stride_t stride10, stride_t stride20, stride_t stride30) {
  if (stride10 == stride20 || stride20 == stride30 || stride10 == stride30) {
    return false;
  }
  if (stride30 > stride20) {
    return (stride30 / stride20 >= size2) && check_opt_dim_3<index_t, stride_t>(size1, stride10, stride20);
  } else {
    return stride20 / stride30 >= size2 && check_opt_dim_3<index_t, stride_t>(size1, stride10, stride20);
  }
}

template <typename index_t>
inline int get_block_size(index_t size0) {
  // size0 % 64 == 0, so block_size >= 64
  int block_size = 1024;
  while (size0 % block_size != 0) {
    block_size /= 2;
  }
  return std::max(C10_WARP_SIZE, block_size);
}

template<typename arg_t, typename... Args>
bool check_vec_template(Args&... inps) {
  // if (sizeof(arg_t) >= 4) {
  //   return true;
  // }
    for (const auto& i : {inps...}) {
    if (i % sizeof(float) != 0) {
      return false;
    }
  }
  return true;
}

template<typename stride_t, typename... Args>
bool check_cast_vec_template(at::ScalarType arg_t, stride_t stride, Args&... inps) {
  if (arg_t == at::ScalarType::Float) {
    if (stride != sizeof(float)) {
      return false;
    }
  } else if (arg_t == at::ScalarType::Half) {
    if (stride != (sizeof(float) / 2)) {
      return false;
    }
  } else if (arg_t == at::ScalarType::BFloat16) {
    if (stride != (sizeof(float) / 2)) {
      return false;
    }
  } else if (arg_t == at::ScalarType::Char) {
    if (stride != (sizeof(float) / 4)) {
      return false;
    }
  } else {
    return false;
  }
  for (const auto& i : {inps...}) {
    if (i % sizeof(float) != 0) {
      return false;
    }
  }
  return true;
}

inline int getCastVectorizedAlignment(at::ScalarType dtype, void* pointer, int alignment_size) {
  if (dtype == at::ScalarType::Float) {
    return getVectorizedAlignment<float>(pointer, alignment_size);
  } else if (dtype == at::ScalarType::Half) {
    return getVectorizedAlignment<at::Half>(pointer, alignment_size);
  } else if (dtype == at::ScalarType::Char) {
    return getVectorizedAlignment<int8_t>(pointer, alignment_size);
  } else if (dtype == at::ScalarType::BFloat16) {
    return getVectorizedAlignment<at::BFloat16>(pointer, alignment_size);
  } else if (dtype == at::ScalarType::Long) {
    return getVectorizedAlignment<int64_t>(pointer, alignment_size);
  } else {
    return 1;
  }
}

inline int getMaxWaveNum() {
  auto max_sm_size = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  return max_sm_size * 8 * 4;
}

inline int getScalarTypeBytes(at::ScalarType st) {
  if (st == at::ScalarType::Half || st == at::ScalarType::BFloat16 || st == at::ScalarType::Short)
    return 2;

  if (st == at::ScalarType::Float || st == at::ScalarType::Int || st == at::ScalarType::ComplexHalf)
    return 4;

  if (st == at::ScalarType::Byte || st == at::ScalarType::Char || st == at::ScalarType::Bool)
    return 1;

  if (st == at::ScalarType::Long || st == at::ScalarType::Double || st == at::ScalarType::ComplexFloat)
    return 8;

  if (st == at::ScalarType::ComplexDouble)
     return 16;

  return -1;
}

template<int N, bool signed_strides = false>
bool isArityContiguous(int ndim,
                       int narity,
                       int arity_i,
                       OffsetCalculator<N, uint32_t, signed_strides> offset_calc,
                       at::detail::Array<at::ScalarType, N> dtypes) {
  if (arity_i < 0 || arity_i >= narity) assert(0);

  int DtypeBytes = getScalarTypeBytes(dtypes[arity_i]);
  assert(DtypeBytes > 0);

  int stride = DtypeBytes;
  for (int i = 0; i < ndim; i++) {
    if (offset_calc.strides_[i][arity_i] != stride) {
      return false;
    }
    stride*= offset_calc.sizes_[i].divisor;
  }

  return true;
}

template<int N, bool signed_strides = false>
bool isArityLowContiguous(int ndim,
                       int narity,
                       int arity_i,
                       OffsetCalculator<N, uint32_t, signed_strides> offset_calc,
                       at::detail::Array<at::ScalarType, N> dtypes) {
  if (arity_i < 0 || arity_i >= narity) assert(0);

  int DtypeBytes = getScalarTypeBytes(dtypes[arity_i]);
  assert(DtypeBytes > 0);
  int stride = DtypeBytes;
  for (int i = 0; i < ndim; i++) {
    if (!(offset_calc.strides_[i][arity_i] >= stride &&
        offset_calc.strides_[i][arity_i] % stride == 0)) {
      return false;
    }
  }
  return offset_calc.strides_[0][arity_i] == stride;
}

