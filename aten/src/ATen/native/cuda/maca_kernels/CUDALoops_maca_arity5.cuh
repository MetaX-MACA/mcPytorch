#pragma once 

#include <typeinfo>

template<int nt, int vt, typename res_t , typename arg0_t, typename arg1_t, typename arg2_t, typename arg3_t,typename arg4_t, typename func_t, typename index_t, typename stride_t>
C10_LAUNCH_BOUNDS_2(nt, 4)
__global__ void elementwise_kernel_3_5(
    int64_t N,
    char* data0, char* data1, char* data2, char* data3, char* data4, char* data5,
    index_t size0, index_t size1, index_t size2,
    stride_t stride00, stride_t stride01, stride_t stride02, stride_t stride03, stride_t stride04, stride_t stride05, 
    stride_t stride10, stride_t stride11, stride_t stride12, stride_t stride13, stride_t stride14, stride_t stride15, 
    stride_t stride20, stride_t stride21, stride_t stride22, stride_t stride23, stride_t stride24, stride_t stride25, 
    func_t f) {
  // ndim = 3, arity = 5, narg = 6
  int tid = threadIdx.x;
  int nv = nt * vt;
  int64_t idx = nv * blockIdx.x + tid;
  #pragma unroll
  for (int i = 0; i < vt; i++) {
    if (idx < N) {
      // ---------------- f(idx);
      // [=]GPU_LAMBDA(int idx) {
      //   auto offsets = offset_calc.get(idx);
      //   arg0_t* out = (arg0_t*)(data[0] + offsets[0]);
      //   *out = invoke(f, &data.data[1], &offsets.data[1], 1);
      // }
      // -----------------------
      int64_t offsets[6];
      auto linear_idx = idx;
      constexpr int NARGS = 6;
      #pragma unroll
      for (int arg = 0; arg < NARGS; arg++) {
        offsets[arg] = 0;
      }

      auto divmod_div = linear_idx / size0;
      auto divmod_mod = linear_idx % size0;
      linear_idx = divmod_div;
      int arg = 0;
      offsets[arg] += divmod_mod * stride00;
      arg = 1;
      offsets[arg] += divmod_mod * stride01;
      arg = 2;
      offsets[arg] += divmod_mod * stride02;
      arg = 3;
      offsets[arg] += divmod_mod * stride03;
      arg = 4;
      offsets[arg] += divmod_mod * stride04;
      arg = 5;
      offsets[arg] += divmod_mod * stride05;
      // dim = 1
      divmod_div = linear_idx / size1;
      divmod_mod = linear_idx % size1;
      linear_idx = divmod_div;
      arg = 0;
      offsets[arg] += divmod_mod * stride10;
      arg = 1;
      offsets[arg] += divmod_mod * stride11;
      arg = 2;
      offsets[arg] += divmod_mod * stride12;
      arg = 3;
      offsets[arg] += divmod_mod * stride13;
      arg = 4;
      offsets[arg] += divmod_mod * stride14;
      arg = 5;
      offsets[arg] += divmod_mod * stride15;
      // dim = 2
      divmod_div = linear_idx / size2;
      divmod_mod = linear_idx % size2;
      linear_idx = divmod_div;
      arg = 0;
      offsets[arg] += divmod_mod * stride20;
      arg = 1;
      offsets[arg] += divmod_mod * stride21;
      arg = 2;
      offsets[arg] += divmod_mod * stride22;
      arg = 3;
      offsets[arg] += divmod_mod * stride23;
      arg = 4;
      offsets[arg] += divmod_mod * stride24;
      arg = 5;
      offsets[arg] += divmod_mod * stride25;

      res_t* out = (res_t*)(data0 + offsets[0]);
      *out = f(func_reinterpret_cast<arg0_t>(data1 + offsets[1]), \
               func_reinterpret_cast<arg1_t>(data2 + offsets[2]), \
               func_reinterpret_cast<arg2_t>(data3 + offsets[3]), \
               func_reinterpret_cast<arg3_t>(data4 + offsets[4]), \
               func_reinterpret_cast<arg4_t>(data5 + offsets[5]));

      idx += nt;
    }
  }
}

template<int nt, int vt, typename res_t , typename arg0_t, typename arg1_t, typename arg2_t, typename arg3_t,typename arg4_t, typename func_t, typename index_t, typename stride_t>
C10_LAUNCH_BOUNDS_2(nt, 4)
__global__ void elementwise_kernel_3_5_broadcast(
    int64_t N,
    char* data0, char* data1, char* data2, char* data3, char* data4, char* data5,
    index_t size0, index_t size1, index_t size2,
    stride_t stride00, stride_t stride01, stride_t stride02, stride_t stride03, stride_t stride04, stride_t stride05,
    stride_t stride10, stride_t stride11, stride_t stride12, stride_t stride13, stride_t stride14, stride_t stride15,
    stride_t stride20, stride_t stride21, stride_t stride22, stride_t stride23, stride_t stride24, stride_t stride25,
    func_t f, int z_t, int z_remain) {
  // ndim = 3, arity = 5, narg = 6
  int tid = threadIdx.x;
  int nv = blockDim.x * vt;
  int64_t idx = blockIdx.x * nv + tid * vt;
  int y_offset_idx = idx / size0;
  int x_offset_idx = idx % size0;

  using StoreT = at::native::memory::aligned_vector<res_t, vt>;
  using LoadT = at::native::memory::aligned_vector<arg0_t, vt>;
  using LoadT1 = at::native::memory::aligned_vector<arg1_t, vt>;

  if (idx < size0 * size1){
    int64_t offsets[6];

    for(int z_loop = 0; z_loop < z_t; z_loop++){
      int z_offset_idx = gridDim.z * z_loop + blockIdx.z;

      offsets[3] = y_offset_idx * stride13 + z_offset_idx * stride23;
      offsets[4] = z_offset_idx * stride24;
      offsets[5] = z_offset_idx * stride25;

      arg2_t ld_3 = *(arg2_t*)(data3 + offsets[3]);
      arg3_t ld_4 = *(arg3_t*)(data4 + offsets[4]);
      arg4_t ld_5 = *(arg4_t*)(data5 + offsets[5]);

      offsets[0] = x_offset_idx * stride00 + y_offset_idx * stride10 + z_offset_idx * stride20;
      offsets[1] = x_offset_idx * stride01 + y_offset_idx * stride11 + z_offset_idx * stride21;
      offsets[2] = x_offset_idx * stride02 + y_offset_idx * stride12 + z_offset_idx * stride22;

      res_t ld_out[vt];
      arg0_t ld_1[vt];
      arg1_t ld_2[vt];
      LoadT* p_ld_1 = reinterpret_cast<LoadT*>(&ld_1);
      LoadT1* p_ld_2 = reinterpret_cast<LoadT1*>(&ld_2);
      StoreT* p_out = reinterpret_cast<StoreT*>(&ld_out);

      *p_ld_1 = *reinterpret_cast<LoadT*>(data1+offsets[1]);
      *p_ld_2 = *reinterpret_cast<LoadT1*>(data2+offsets[1]);

      // compute
      #pragma unroll
      for(int i=0; i<vt; i++){
        ld_out[i] = f(ld_1[i], ld_2[i], ld_3, ld_4, ld_5);
      }

      // store
      StoreT* out = reinterpret_cast<StoreT*>(data0 + offsets[0]);
      *out = *p_out;
    }

    // deal with z_remain
    if (z_remain != 0 && blockIdx.z < z_remain) {
      int z_offset_idx = gridDim.z * z_t + blockIdx.z;

      offsets[3] = y_offset_idx * stride13 + z_offset_idx * stride23;
      offsets[4] = z_offset_idx * stride24;
      offsets[5] = z_offset_idx * stride25;

      arg2_t ld_3 = *(arg2_t*)(data3 + offsets[3]);
      arg3_t ld_4 = *(arg3_t*)(data4 + offsets[4]);
      arg4_t ld_5 = *(arg4_t*)(data5 + offsets[5]);

      offsets[0] = x_offset_idx * stride00 + y_offset_idx * stride10 + z_offset_idx * stride20;
      offsets[1] = x_offset_idx * stride01 + y_offset_idx * stride11 + z_offset_idx * stride21;
      offsets[2] = x_offset_idx * stride02 + y_offset_idx * stride12 + z_offset_idx * stride22;

      res_t ld_out[vt];
      arg0_t ld_1[vt];
      arg1_t ld_2[vt];
      LoadT* p_ld_1 = reinterpret_cast<LoadT*>(&ld_1);
      LoadT1* p_ld_2 = reinterpret_cast<LoadT1*>(&ld_2);
      StoreT* p_out = reinterpret_cast<StoreT*>(&ld_out);

      *p_ld_1 = *reinterpret_cast<LoadT*>(data1+offsets[1]);
      *p_ld_2 = *reinterpret_cast<LoadT1*>(data2+offsets[1]);

      // compute
      #pragma unroll
      for(int i=0; i<vt; i++){
        ld_out[i] = f(ld_1[i], ld_2[i], ld_3, ld_4, ld_5);
      }

      // store
      StoreT* out = reinterpret_cast<StoreT*>(data0 + offsets[0]);
      *out = *p_out;
    }
  }
}

template<int nt, int vt, typename res_t , typename arg0_t, typename arg1_t, typename arg2_t, typename arg3_t, typename arg4_t, typename func_t, typename index_t, typename stride_t>
static void launch_legacy_kernel_maca_3_5(
    int64_t N,
    char* data0, char* data1, char* data2, char* data3, char* data4, char* data5,
    index_t size0, index_t size1, index_t size2,
    stride_t stride00, stride_t stride01, stride_t stride02, stride_t stride03, stride_t stride04, stride_t stride05,
    stride_t stride10, stride_t stride11, stride_t stride12, stride_t stride13, stride_t stride14, stride_t stride15,
    stride_t stride20, stride_t stride21, stride_t stride22, stride_t stride23, stride_t stride24, stride_t stride25,
    const func_t& f) {
  // ndim = 3, arity = 5
  TORCH_INTERNAL_ASSERT(N >= 0 && N <= std::numeric_limits<int32_t>::max());
  if (N == 0) {
    return;
  }
  dim3 block(nt);
  dim3 grid((N + block.x * vt - 1) / (block.x * vt));
  auto stream = at::cuda::getCurrentCUDAStream();
  elementwise_kernel_3_5<nt, vt, res_t, arg0_t, arg1_t, arg2_t, arg3_t, arg4_t, func_t, index_t, stride_t><<<grid, block, 0, stream>>>(
      N, data0, data1, data2, data3, data4, data5, size0, size1, size2, stride00, stride01, stride02,stride03, stride04, stride05, \
      stride10, stride11, stride12, stride13, stride14, stride15, stride20, stride21, stride22, stride23, stride24, stride25, f);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  }


template<int nt, int vt, typename res_t , typename arg0_t, typename arg1_t, typename arg2_t, typename arg3_t, typename arg4_t, typename func_t, typename index_t, typename stride_t>
static void launch_legacy_kernel_maca_3_5_broadcast(
    int64_t N,
    char* data0, char* data1, char* data2, char* data3, char* data4, char* data5,
    index_t size0, index_t size1, index_t size2,
    stride_t stride00, stride_t stride01, stride_t stride02, stride_t stride03, stride_t stride04, stride_t stride05,
    stride_t stride10, stride_t stride11, stride_t stride12, stride_t stride13, stride_t stride14, stride_t stride15,
    stride_t stride20, stride_t stride21, stride_t stride22, stride_t stride23, stride_t stride24, stride_t stride25,
    const func_t& f) {

  TORCH_INTERNAL_ASSERT(N >= 0 && N <= std::numeric_limits<int32_t>::max());
  if (N == 0) {
    return;
  }
  constexpr int block_dim_x = 64;

  dim3 block(block_dim_x);
  auto stream = at::cuda::getCurrentCUDAStream();
  int vec = getVectorizedAlignment<res_t>((void*)data0, size0);
  int grid_dim_x = (size0 * size1 + block.x * vec - 1) / (block.x * vec);
  int grid_dim_z = getMaxGridSize(grid_dim_x, size2);
  TORCH_INTERNAL_ASSERT(grid_dim_z > 0);
  // adjust z_remain
  int z_t = size2 / grid_dim_z;
  grid_dim_z = std::ceil(float(size2)/float(z_t));
  z_t = size2 / grid_dim_z;
  int z_remain = size2 % grid_dim_z;
  TORCH_INTERNAL_ASSERT(z_t > 0);
  dim3 grid(grid_dim_x, 1, grid_dim_z);

  if (vec==4){
      elementwise_kernel_3_5_broadcast<nt, 4, res_t, arg0_t, arg1_t, arg2_t, arg3_t, arg4_t, func_t, index_t, stride_t><<<grid, block, 0, stream>>>(
          N, data0, data1, data2, data3, data4, data5, size0, size1, size2, stride00, stride01, stride02,stride03, stride04, stride05, \
          stride10, stride11, stride12, stride13, stride14, stride15, stride20, stride21, stride22, stride23, stride24, stride25, f, z_t, z_remain);
  }
  else if (vec==2){
      elementwise_kernel_3_5_broadcast<nt, 2, res_t, arg0_t, arg1_t, arg2_t, arg3_t, arg4_t, func_t, index_t, stride_t><<<grid, block, 0, stream>>>(
          N, data0, data1, data2, data3, data4, data5, size0, size1, size2, stride00, stride01, stride02,stride03, stride04, stride05, \
          stride10, stride11, stride12, stride13, stride14, stride15, stride20, stride21, stride22, stride23, stride24, stride25, f, z_t, z_remain);
  }
  else {
      elementwise_kernel_3_5_broadcast<nt, 1, res_t, arg0_t, arg1_t, arg2_t, arg3_t, arg4_t, func_t, index_t, stride_t><<<grid, block, 0, stream>>>(
          N, data0, data1, data2, data3, data4, data5, size0, size1, size2, stride00, stride01, stride02,stride03, stride04, stride05, \
          stride10, stride11, stride12, stride13, stride14, stride15, stride20, stride21, stride22, stride23, stride24, stride25, f, z_t, z_remain);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template<int nt, int vt, typename res_t , typename arg0_t, typename arg1_t, typename arg2_t, typename arg3_t, typename arg4_t, typename func_t, typename index_t, typename stride_t>
C10_LAUNCH_BOUNDS_2(nt, 4)
__global__ void elementwise_kernel_2_5_cast(
    int64_t N,
    char* data0, char* data1, char* data2, char* data3, char* data4, char* data5,
    ScalarType st0, ScalarType st1, ScalarType st2,ScalarType st3, ScalarType st4, ScalarType st5,
    index_t size0, index_t size1,
    stride_t stride00, stride_t stride01, stride_t stride02, stride_t stride03, stride_t stride04, stride_t stride05,
    stride_t stride10, stride_t stride11, stride_t stride12, stride_t stride13, stride_t stride14, stride_t stride15,
    func_t f) {
  // ndim = 2, arity = 5, narg = 6
  int64_t tid = threadIdx.x;
  int nv = nt * vt;
  int idx = nv * blockIdx.x + tid;
  #pragma unroll
  for (int i = 0; i < vt; i++) {
    if (idx < N) {
      // ---------------- f(idx);
      // [=]GPU_LAMBDA(int idx) {
      //   auto offsets = offset_calc.get(idx);
      //   arg0_t* out = (arg0_t*)(data[0] + offsets[0]);
      //   *out = invoke(f, &data.data[1], &offsets.data[1], 1);
      // }
      // -----------------------
      int64_t offsets[6];
      auto linear_idx = idx;
      constexpr int NARGS = 6;
      #pragma unroll
      for (int arg = 0; arg < NARGS; arg++) {
        offsets[arg] = 0;
      }
      auto divmod_div = linear_idx / size0;
      auto divmod_mod = linear_idx % size0;
      linear_idx = divmod_div;
      int arg = 0;
      offsets[arg] += divmod_mod * stride00;
      arg = 1;
      offsets[arg] += divmod_mod * stride01;
      arg = 2;
      offsets[arg] += divmod_mod * stride02;
      arg = 3;
      offsets[arg] += divmod_mod * stride03;
      arg = 4;
      offsets[arg] += divmod_mod * stride04;
      arg = 5;
      offsets[arg] += divmod_mod * stride05;
      // dim = 1
      divmod_div = linear_idx / size1;
      divmod_mod = linear_idx % size1;
      linear_idx = divmod_div;
      arg = 0;
      offsets[arg] += divmod_mod * stride10;
      arg = 1;
      offsets[arg] += divmod_mod * stride11;
      arg = 2;
      offsets[arg] += divmod_mod * stride12;
      arg = 3;
      offsets[arg] += divmod_mod * stride13;
      arg = 4;
      offsets[arg] += divmod_mod * stride14;
      arg = 5;
      offsets[arg] += divmod_mod * stride15;

      void* out = data0 + offsets[0];
      res_t result = f(c10::fetch_and_cast<arg0_t>(st1, data1 + offsets[1]),
               c10::fetch_and_cast<arg1_t>(st2, data2 + offsets[2]),
               c10::fetch_and_cast<arg2_t>(st3, data3 + offsets[3]),
               c10::fetch_and_cast<arg3_t>(st4, data4 + offsets[4]),
               c10::fetch_and_cast<arg4_t>(st5, data5 + offsets[5])
               );
      c10::cast_and_store<res_t>(st0, out, result);
      idx += nt;
    }
  }
}

template<int nt, int vt, typename res_t , typename arg0_t, typename arg1_t, typename arg2_t, typename arg3_t, typename arg4_t, typename func_t, typename index_t, typename stride_t>
static void launch_legacy_kernel_maca_2_5_cast(
    int64_t N,
    char* data0, char* data1, char* data2, char* data3, char* data4, char* data5,
    ScalarType st0, ScalarType st1, ScalarType st2, ScalarType st3, ScalarType st4, ScalarType st5,
    index_t size0, index_t size1,
    stride_t stride00, stride_t stride01, stride_t stride02, stride_t stride03, stride_t stride04, stride_t stride05,
    stride_t stride10, stride_t stride11, stride_t stride12, stride_t stride13, stride_t stride14, stride_t stride15,
    const func_t& f) {
  // ndim = 2, arity = 5
  TORCH_INTERNAL_ASSERT(N >= 0 && N <= std::numeric_limits<int32_t>::max());
  if (N == 0) {
    return;
  }
  dim3 block(nt);
  dim3 grid((N + block.x * vt - 1) / (block.x * vt));
  auto stream = at::cuda::getCurrentCUDAStream();
  elementwise_kernel_2_5_cast<nt, vt, res_t, arg0_t, arg1_t, arg2_t, arg3_t, arg4_t, func_t, index_t, stride_t><<<grid, block, 0, stream>>>(
      N, data0, data1, data2, data3, data4, data5, st0, st1, st2, st3, st4, st5, size0, size1, stride00, stride01, stride02, stride03, stride04, stride05, stride10, 
      stride11, stride12, stride13, stride14, stride15, f);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename func_t>
void gpu_kernel_impl_maca_arity5(TensorIteratorBase& iter, const func_t& f) {
  using traits = function_traits<func_t>;
  using arg0_t = typename traits::result_type;
  using arg1_t = typename traits::template arg<0>::type;
  using arg2_t = typename traits::template arg<1>::type;
  using arg3_t = typename traits::template arg<2>::type;
  using arg4_t = typename traits::template arg<3>::type;
  using arg5_t = typename traits::template arg<4>::type;
  constexpr int ntensors = traits::arity + 1;

  TORCH_INTERNAL_ASSERT(iter.can_use_32bit_indexing());
  TORCH_INTERNAL_ASSERT(iter.ninputs() == traits::arity);
  TORCH_INTERNAL_ASSERT(iter.noutputs() == 1);

  at::detail::Array<char*, ntensors> data;
  for (int i = 0; i < ntensors; i++) {
    data[i] = (char*)iter.data_ptr(i);
  }

  int64_t numel = iter.numel();

  bool contiguous = iter.is_contiguous();
  bool dynamic_casting = needs_dynamic_casting<func_t>::check(iter);

  if (!dynamic_casting) {
    if (contiguous) {
      launch_vectorized_kernel(numel, f, data);
    } else {
        auto offset_calc = ::make_offset_calculator<traits::arity + 1>(iter);
        constexpr int unroll_factor = sizeof(arg0_t) >= 4 ? 2 : 4;
        constexpr int narity = traits::arity;
        int ndim = iter.ndim();
        assert(narity == 5);
        if (ndim == 3) {
          auto s0 = offset_calc.sizes_[0].divisor;
          auto s1 = offset_calc.sizes_[1].divisor;
          auto s2 = offset_calc.sizes_[2].divisor;
          size_t st00 = offset_calc.strides_[0][0], st10 = offset_calc.strides_[1][0], st20 = offset_calc.strides_[2][0];
          size_t st01 = offset_calc.strides_[0][1], st11 = offset_calc.strides_[1][1], st21 = offset_calc.strides_[2][1];
          size_t st02 = offset_calc.strides_[0][2], st12 = offset_calc.strides_[1][2], st22 = offset_calc.strides_[2][2];
          size_t st03 = offset_calc.strides_[0][3], st13 = offset_calc.strides_[1][3], st23 = offset_calc.strides_[2][3];
          size_t st04 = offset_calc.strides_[0][4], st14 = offset_calc.strides_[1][4], st24 = offset_calc.strides_[2][4];
          size_t st05 = offset_calc.strides_[0][5], st15 = offset_calc.strides_[1][5], st25 = offset_calc.strides_[2][5];

          if ((sizeof(arg0_t)==4 && sizeof(arg1_t)==4 && sizeof(arg2_t)==4 && sizeof(arg3_t)==4 && sizeof(arg4_t)==4 && sizeof(arg5_t)==4) &&
              (st00 == sizeof(arg0_t) && st10 == s0 * sizeof(arg0_t) && st20 == s1 * st10) &&
              (st01 == sizeof(arg1_t) && st11 == s0 * sizeof(arg1_t) && st21 == s1 * st11) &&
              (st02 == sizeof(arg2_t) && st12 == s0 * sizeof(arg2_t) && st22 == s1 * st12) &&
              (st03 == 0 && st13 == sizeof(arg3_t) && st23 == s1 * st13) &&
              (st04 == 0 && st14 == 0 && st24 == sizeof(arg4_t)) &&
              (st05 == 0 && st15 == 0 && st25 == sizeof(arg5_t))
          ){
              get_elementwise_info<narity + 1>(ndim, narity, offset_calc, "p_e_launch_legacy_kernel_maca_3_5_broadcast", f);
              launch_legacy_kernel_maca_3_5_broadcast<64, unroll_factor, arg0_t, typename traits::template arg<0>::type, typename traits::template arg<1>::type, typename traits::template arg<2>::type, typename traits::template arg<3>::type, typename traits::template arg<4>::type>(
                  numel,
                  data[0], data[1], data[2], data[3], data[4], data[5], // data
                  offset_calc.sizes_[0].divisor, offset_calc.sizes_[1].divisor, offset_calc.sizes_[2].divisor,
                  offset_calc.strides_[0][0], offset_calc.strides_[0][1], offset_calc.strides_[0][2], offset_calc.strides_[0][3], offset_calc.strides_[0][4], offset_calc.strides_[0][5],
                  offset_calc.strides_[1][0], offset_calc.strides_[1][1], offset_calc.strides_[1][2], offset_calc.strides_[1][3], offset_calc.strides_[1][4], offset_calc.strides_[1][5],
                  offset_calc.strides_[2][0], offset_calc.strides_[2][1], offset_calc.strides_[2][2], offset_calc.strides_[2][3], offset_calc.strides_[2][4], offset_calc.strides_[2][5],
                  f);
          } else {
              get_elementwise_info<narity + 1>(ndim, narity, offset_calc, "p_e_launch_legacy_kernel_maca_3_5", f);
              launch_legacy_kernel_maca_3_5<128, unroll_factor, arg0_t, typename traits::template arg<0>::type, typename traits::template arg<1>::type, typename traits::template arg<2>::type, typename traits::template arg<3>::type, typename traits::template arg<4>::type>(
                  numel,
                  data[0], data[1], data[2], data[3], data[4], data[5], // data
                  offset_calc.sizes_[0].divisor, offset_calc.sizes_[1].divisor, offset_calc.sizes_[2].divisor,
                  offset_calc.strides_[0][0], offset_calc.strides_[0][1], offset_calc.strides_[0][2], offset_calc.strides_[0][3], offset_calc.strides_[0][4], offset_calc.strides_[0][5],
                  offset_calc.strides_[1][0], offset_calc.strides_[1][1], offset_calc.strides_[1][2], offset_calc.strides_[1][3], offset_calc.strides_[1][4], offset_calc.strides_[1][5],
                  offset_calc.strides_[2][0], offset_calc.strides_[2][1], offset_calc.strides_[2][2], offset_calc.strides_[2][3], offset_calc.strides_[2][4], offset_calc.strides_[2][5],
                  f);
          }
        } else {
          get_elementwise_info<narity + 1>(ndim, narity, offset_calc, "p_e_launch_legacy_kernel_5", f);
          launch_legacy_kernel<128,unroll_factor>(numel, [=]GPU_LAMBDA(int idx) {
            auto offsets = offset_calc.get(idx);
            arg0_t* out = (arg0_t*)(data[0] + offsets[0]);
            *out = invoke(f, &data.data[1], &offsets.data[1], 1);
          });
        }
    }
  } else {
    if (contiguous) {
      auto loader = memory::LoadWithCast<traits::arity>(iter);
      auto storer = memory::StoreWithCast<1>(iter);
      auto input_offset_calculator = TrivialOffsetCalculator<traits::arity>();
      auto output_offset_calculator = TrivialOffsetCalculator<1>();
      launch_unrolled_kernel(numel, f, data, input_offset_calculator, output_offset_calculator, loader, storer);
    } else {
      at::detail::Array<ScalarType, ntensors> dtypes;
      for (int i = 0; i < ntensors; i++) {
        dtypes[i] = iter.dtype(i);
      }
      auto offset_calc = ::make_offset_calculator<traits::arity + 1>(iter);
      constexpr int narity = traits::arity;
      int ndim = iter.ndim();
      constexpr int unroll_factor = sizeof(arg0_t) >= 4 ? 2 : 4;
      if(ndim == 2){
        get_elementwise_info<narity + 1>(ndim, narity, offset_calc, "p_e_launch_legacy_kernel_maca_2_5_cast",f);
        launch_legacy_kernel_maca_2_5_cast<128, unroll_factor, arg0_t,
                                      typename traits::template arg<0>::type,
                                      typename traits::template arg<1>::type,
                                      typename traits::template arg<2>::type,
                                      typename traits::template arg<3>::type,
                                      typename traits::template arg<4>::type>(
          numel,
          data[0], data[1], data[2], data[3], data[4], data[5], // data
          dtypes[0], dtypes[1], dtypes[2], dtypes[3], dtypes[4], dtypes[5],
          offset_calc.sizes_[0].divisor, offset_calc.sizes_[1].divisor,
          offset_calc.strides_[0][0], offset_calc.strides_[0][1], offset_calc.strides_[0][2], offset_calc.strides_[0][3], offset_calc.strides_[0][4], offset_calc.strides_[0][5], 
          offset_calc.strides_[1][0], offset_calc.strides_[1][1], offset_calc.strides_[1][2], offset_calc.strides_[1][3], offset_calc.strides_[1][4], offset_calc.strides_[1][5], 
          f);
      }else{
        assert(narity == 5);
        get_elementwise_info<narity + 1>(ndim, narity, offset_calc, "p_e_launch_legacy_kernel", f);
        launch_legacy_kernel<128, 4>(numel, [=]GPU_LAMBDA(int idx) {
          auto offsets = offset_calc.get(idx);
          void* out = data[0] + offsets[0];
          arg0_t result = invoke(f, &data.data[1], &offsets.data[1], &dtypes.data[1], 1);
          c10::cast_and_store<arg0_t>(dtypes[0], out, result);
        });
      }
    }
  }
}
