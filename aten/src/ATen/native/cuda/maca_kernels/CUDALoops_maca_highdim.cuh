#pragma once

#include <type_traits>
#include <tuple>
#include <iostream>

#include <ATen/cuda/CUDAContext.h>
#include <ATen/core/Array.h>
#include <ATen/native/cuda/MemoryAccess.cuh>
#include <ATen/detail/FunctionTraits.h>
#include <ATen/native/TensorIterator.h>
#include <c10/macros/Macros.h>
#include <c10/core/DynamicCast.h>
#include <c10/core/ScalarType.h>
#include <c10/util/TypeCast.h>
#include <c10/util/C++17.h>
#include "loop_utils.h"
#include <typeinfo>

namespace at { namespace native {

int powExpon(int n) {
  if(n <= 1) return 0;
  int num = 0;
  while (n > 1) {
    if(n % 2 != 0) return 0;
    num += 1;
    n /= 2;
  }
  return num;
}

struct packData {
    uint32_t index[15][3];  //dim,inps,powdim
};

struct packData_out {
    uint32_t index[15][4];   //dim,outs,inps,powdim
};

template<int nt, int vt, typename res_t , typename arg0_t, typename func_t,
typename std::enable_if<!std::is_same<arg0_t, at::Half>::value && !std::is_same<arg0_t, at::BFloat16>::value
&& !std::is_same<arg0_t, float>::value, int>::type = 0>
__global__ void elementwise_kernel_copy_highDim_contiguous(
    int64_t N, int64_t dim,
    char* data0, char* data1,
    packData pd,
    func_t f) {
  assert(0);
}

template<int nt, int vt, typename res_t , typename arg0_t, typename func_t,
typename std::enable_if<std::is_same<arg0_t, at::Half>::value || std::is_same<arg0_t, at::BFloat16>::value
|| std::is_same<arg0_t, float>::value, int>::type = 0>
__global__ void elementwise_kernel_copy_highDim_contiguous(
    int64_t N, int dim,
    char* data0, char* data1,
    packData pd,
    func_t f) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * vt;
    if (idx >= N) return;
    using load_vec = at::native::memory::aligned_vector<arg0_t, vt>;
    using store_vec = at::native::memory::aligned_vector<res_t, vt>;
    load_vec load;

    for (int j = 0; j < vt / 2; j++) {
      uint32_t tmp_idx = idx + j * 2;

      uint32_t offsets = 0;
      uint32_t divmod_div;
      uint32_t divmod_mod;
      for (int i = 0; i < (dim - 1); i++) {
        divmod_div = tmp_idx >> pd.index[i][2];
        divmod_mod = tmp_idx - pd.index[i][0] * divmod_div;
        tmp_idx = divmod_div;
        offsets += divmod_mod * pd.index[i][1];
      }
      offsets += tmp_idx * pd.index[dim-1][1];

      load.val[j * 2] = *(reinterpret_cast<arg0_t*>(data1 + offsets));
      load.val[j * 2 + 1] = *(reinterpret_cast<arg0_t*>(data1 + offsets + pd.index[0][1]));
    }
    *reinterpret_cast<store_vec*>(data0 + idx * sizeof(res_t)) = load;
}


template<int nt, int vt, typename res_t , typename arg0_t, typename func_t,
typename std::enable_if<!std::is_same<arg0_t, at::Half>::value && !std::is_same<arg0_t, at::BFloat16>::value
&& !std::is_same<arg0_t, float>::value, int>::type = 0>
__global__ void elementwise_kernel_copy_highDim_uncontiguous(
    int64_t N, int64_t dim,
    char* data0, char* data1,
    packData_out pd,
    func_t f) {
  assert(0);
}


template<int nt, int vt, typename res_t , typename arg0_t, typename func_t,
typename std::enable_if<std::is_same<arg0_t, at::Half>::value || std::is_same<arg0_t, at::BFloat16>::value
|| std::is_same<arg0_t, float>::value, int>::type = 0>
__global__ void elementwise_kernel_copy_highDim_uncontiguous(
    int64_t N, int dim,
    char* data0, char* data1,
    packData_out pd,
    func_t f) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * vt;
    if (idx >= N) return;
    using load_vec = at::native::memory::aligned_vector<arg0_t, vt>;
    using store_vec = at::native::memory::aligned_vector<res_t, vt>;
    load_vec load;

    uint32_t tmp_idx = idx;

    uint32_t offsets = 0;
    uint32_t offsets_out = 0;
    uint32_t divmod_div;
    uint32_t divmod_mod;
    for (int i = 0; i < (dim - 1); i++) {
      divmod_div = tmp_idx >> pd.index[i][3];
      divmod_mod = tmp_idx - pd.index[i][0] * divmod_div;
      tmp_idx = divmod_div;
      offsets_out += divmod_mod * pd.index[i][1];
      offsets += divmod_mod * pd.index[i][2];
    }
    offsets_out += tmp_idx * pd.index[dim-1][1];
    offsets += tmp_idx * pd.index[dim-1][2];

    load.val[0] = *(reinterpret_cast<arg0_t*>(data1 + offsets));
    load.val[1] = *(reinterpret_cast<arg0_t*>(data1 + offsets + pd.index[0][2]));
    *reinterpret_cast<store_vec*>(data0 + offsets_out) = load;
}

template <typename func_t>
bool gpu_kernel_impl_maca_copy_high_dim(TensorIteratorBase& iter, const func_t& f) {
    using traits = function_traits<func_t>;
    using arg0_t = typename traits::result_type;
    using arg1_t = typename traits::template arg<0>::type;
    constexpr int ntensors = traits::arity + 1;

    TORCH_INTERNAL_ASSERT(iter.can_use_32bit_indexing());
    TORCH_INTERNAL_ASSERT(iter.ninputs() == traits::arity);
    TORCH_INTERNAL_ASSERT(iter.noutputs() == 1);

    at::detail::Array<char*, ntensors> data;
    at::detail::Array<ScalarType, ntensors> dtypes;
    for (int i = 0; i < ntensors; i++) {
        data[i] = (char*)iter.data_ptr(i);
        dtypes[i] = iter.dtype(i);
    }

    int64_t numel = iter.numel();
    bool contiguous = iter.is_contiguous();
    bool dynamic_casting = needs_dynamic_casting<func_t>::check(iter);
    auto offset_calc = ::make_offset_calculator<traits::arity + 1>(iter);    

    constexpr int unroll_factor = sizeof(arg0_t) >= 4 ? 2 : 4;
    constexpr int narity = traits::arity;
    int ndim = iter.ndim();
    assert(narity == 1);

    bool out_contiguous = isArityContiguous(ndim, ntensors, 0, offset_calc, dtypes);
    bool out_dim0contiguous = isArityLowContiguous(ndim, ntensors, 0, offset_calc, dtypes);
    bool is_floating = dtypes[0] == dtypes[1] && (dtypes[0] == ScalarType::BFloat16 || dtypes[0] == ScalarType::Half || dtypes[0] == ScalarType::Float);

    packData pd;
    packData_out pd_out;
    bool is_pow[15];
    for (int i = 0; i < ndim; i++) {
      pd.index[i][0] = offset_calc.sizes_[i].divisor;
      pd.index[i][1] = offset_calc.strides_[i][1];
      pd.index[i][2] = powExpon(offset_calc.sizes_[i].divisor);

      pd_out.index[i][0] = offset_calc.sizes_[i].divisor;
      pd_out.index[i][1] = offset_calc.strides_[i][0];
      pd_out.index[i][2] = offset_calc.strides_[i][1];
      pd_out.index[i][3] = pd.index[i][2];
    }
    is_pow[0] = bool(pd.index[0][2]);
    for (int i = 1; i < ndim; i ++) {
      is_pow[i] = is_pow[i - 1] && bool(pd.index[i][2]);
    }
    //10,1,  2,4,128,4,64,2,64,2,2,4
    bool dis_opt = ndim == 10 &&
                   pd.index[0][0] == 2 && pd.index[1][0] == 4 && pd.index[2][0] == 128 && pd.index[3][0] == 4 && pd.index[4][0] == 64 &&
                   pd.index[5][0] == 2 && pd.index[6][0] == 64 && pd.index[7][0] == 2 && pd.index[8][0] == 2 && pd.index[9][0] == 4;

    bool is_opt = out_contiguous && ndim >= 6 && ndim <=15 && ndim != 7 && is_floating &&
                  is_pow[ndim-2] && numel >= 851968 && !dis_opt &&
                  at::maca::get_maca_enable_elementwise_kernel_highdim();

    bool is_opt1 = out_dim0contiguous && ndim >= 6 && ndim <=15 && ndim != 7 && is_floating &&
                  is_pow[ndim-2] && numel >= 851968 && !dis_opt &&
                  at::maca::get_maca_enable_elementwise_kernel_highdim();
    if (is_opt) {
       auto stream = at::cuda::getCurrentCUDAStream();
       const int unroll = 2;
       dim3 block(128);
       dim3 grid((numel + block.x * unroll - 1) / (block.x * unroll));
       get_elementwise_info<narity + 1>(ndim, narity, offset_calc, dtypes, "p_e_cp_launch_legacy_highDim_contiguous", f);
       elementwise_kernel_copy_highDim_contiguous<128, unroll, arg0_t, arg1_t, func_t><<<grid, block, 0, stream>>> (
        numel, ndim, data[0], data[1],
        pd,
        f
       );
       C10_CUDA_KERNEL_LAUNCH_CHECK();
    } else if(is_opt1) {
      auto stream = at::cuda::getCurrentCUDAStream();
      const int unroll = 2;
      dim3 block(128);
      dim3 grid((numel + block.x * unroll - 1) / (block.x * unroll));
      get_elementwise_info<narity + 1>(ndim, narity, offset_calc, dtypes, "p_e_cp_launch_legacy_highDim_uncontiguous", f);
      elementwise_kernel_copy_highDim_uncontiguous<128, unroll, arg0_t, arg1_t, func_t><<<grid, block, 0, stream>>> (
        numel, ndim, data[0], data[1],
        pd_out,
        f
       );
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    return is_opt || is_opt1;

}

}}
