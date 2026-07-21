#pragma once

// #include <ATen/core/Tensor.h>
// #include <ATen/Dispatch.h>
// #include <ATen/AccumulateType.h>
// #include <ATen/ceil_div.h>
// #include <ATen/cuda/CUDAContext.h>
// #include <ATen/cuda/DeviceUtils.cuh>
// #include <ATen/native/cuda/block_reduce.cuh>
// #include <ATen/native/cuda/DeviceSqrt.cuh>
// #include <ATen/native/cuda/LaunchUtils.h>
// #include <c10/macros/Macros.h>

namespace at { namespace native {
namespace batchnorm {
constexpr int ELEMENTS_PER_ITER = 4;

template<typename scalar_t, int vec_size>
struct alignas(sizeof(scalar_t) * vec_size) aligned_vector {
  scalar_t val[vec_size];
};  

constexpr int OPTIMAL_TILE_W_OPT = 32;
constexpr int ELEMENTS_PER_THREAD_OPT = 8;
constexpr int MAX_H_BLOCK_OPT = 256;
constexpr int MAX_BLOCK_SIZE_OPT = 512;
__host__ void flexible_launch_configs_opt(
      const int reduction,
      const int stride,
      dim3 &block,
      dim3 &grid) {

  int block_x = std::min(lastPow2(stride / 4), OPTIMAL_TILE_W_OPT);
  int block_y = MAX_BLOCK_SIZE_OPT / block_x;
  
  int grid_x = at::ceil_div(stride, block_x * 4);
  int grid_y = std::min(at::ceil_div(reduction, block_y * ELEMENTS_PER_THREAD_OPT), MAX_H_BLOCK_OPT);
  
  block.x = block_x;
  block.y = block_y;
  block.z = 1;
  grid.x = grid_x;
  grid.y = grid_y;
  grid.z = 1;
}


template <int vec_size, typename input_scalar_t, typename stat_scalar_t, typename stat_accscalar_t, typename index_t>
__global__ void batch_norm_backward_kernel_opt0(
    const input_scalar_t* input,         //type0
    const input_scalar_t* grad_output,   //type0
    input_scalar_t* grad_input,          //type0
    stat_scalar_t* grad_weight,          //type1
    stat_scalar_t* grad_bias,            //type1
    const stat_scalar_t* weight,         //type1
    const stat_scalar_t* running_mean,   //type1
    const stat_scalar_t* running_var,    //type1
    const stat_accscalar_t* save_mean,   //type2
    const stat_accscalar_t* save_invstd, //type2
    bool train,
    stat_accscalar_t epsilon,            //type2
    const index_t stride,
    const index_t reduction_size
    ) {

    using type0 = input_scalar_t;
    using type1 = stat_scalar_t;
    using type2 = stat_accscalar_t;
    using vec_type0 = aligned_vector<input_scalar_t, vec_size>;
    using vec_type1 = aligned_vector<stat_scalar_t, vec_size>;
    using vec_type2 = aligned_vector<stat_accscalar_t, vec_size>;

    const vec_type1* weight_vec = reinterpret_cast<const vec_type1*>(weight);
    const vec_type1* running_mean_vec = reinterpret_cast<const vec_type1*>(running_mean);
    const vec_type1* running_var_vec = reinterpret_cast<const vec_type1*>(running_var);
    const vec_type2* save_mean_vec = reinterpret_cast<const vec_type2*>(save_mean);
    const vec_type2* save_invstd_vec = reinterpret_cast<const vec_type2*>(save_invstd);

    index_t plane = blockIdx.x;
    index_t N = reduction_size;

    vec_type2 mean, invstd;
    if (train) {
      mean = save_mean_vec[plane];
      invstd = save_invstd_vec[plane];
    } else {
      vec_type1 tmp0, tmp1;
      tmp0 = running_mean_vec[plane];
      tmp1 = running_var_vec[plane];
      #pragma unroll
      for (int ii=0; ii < vec_size; ii++) {
        mean.val[ii] = static_cast<type2>(tmp0.val[ii]);
        invstd.val[ii] = static_cast<type2>(1) / device_sqrt(static_cast<type2>(tmp1.val[ii]) + epsilon);
      }
    }

    vec_type2 weight_val;
    vec_type1 tmp0 = weight_vec[plane];
    #pragma unroll
    for (int ii=0; ii < vec_size; ii++) {
      weight_val.val[ii] = static_cast<type2>(tmp0.val[ii]);
    }
    type2 norm = type2(1) / N;

    //reduce
    vec_type2 sum_v0, sum_v1;
    #pragma unroll
    for (int ii=0; ii < vec_size; ii++) {
      sum_v0.val[ii] = 0; sum_v1.val[ii] = 0;
    }

    //thread reduce
    int m_offset = threadIdx.x;
    while (m_offset < reduction_size) {
      const vec_type0 * input_vec = reinterpret_cast<const vec_type0*>(input + m_offset * stride);
      const vec_type0 * grad_output_vec = reinterpret_cast<const vec_type0*>(grad_output+ m_offset * stride);
      const vec_type0 input_item = input_vec[plane];
      const vec_type0 grad_output_item = grad_output_vec[plane];

      #pragma unroll
      for (int ii=0; ii < vec_size; ii++) {
        type2 g = static_cast<type2>(grad_output_item.val[ii]);
        type2 c = static_cast<type2>(input_item.val[ii]) - mean.val[ii];
        sum_v0.val[ii] += g;
        sum_v1.val[ii] += (g*c);
      }
      m_offset += blockDim.x;
    }

    //blockreduce
    __shared__ type2 shared0[C10_WARP_SIZE];
    __shared__ type2 shared1[C10_WARP_SIZE];
    using B = cuda_utils::Block1D;
    const int tid = B::Tid();
    const int lid = tid % C10_WARP_SIZE;  // thread index in warp
    const int wid = tid / C10_WARP_SIZE;  // warp index

    #pragma unroll
    for (int ii=0; ii < vec_size; ii++) {
      #pragma unroll
      for (int offset = (C10_WARP_SIZE >> 1); offset > 0; offset >>= 1) {
        sum_v0.val[ii] += WARP_SHFL_DOWN(sum_v0.val[ii], offset);
        sum_v1.val[ii] += WARP_SHFL_DOWN(sum_v1.val[ii], offset);
      }
      __syncthreads();
      if (lid == 0) {
        shared0[wid] = sum_v0.val[ii];
        shared1[wid] = sum_v1.val[ii];
      }
      __syncthreads();
      sum_v0.val[ii] = (tid < B::Warps()) ? shared0[lid] : 0;
      sum_v1.val[ii] = (tid < B::Warps()) ? shared1[lid] : 0;
      if (wid == 0) {
        #pragma unroll
        for (int offset = (C10_WARP_SIZE >> 1); offset > 0; offset >>= 1) {
          sum_v0.val[ii] += WARP_SHFL_DOWN(sum_v0.val[ii], offset);
          sum_v1.val[ii] += WARP_SHFL_DOWN(sum_v1.val[ii], offset);
        }
      }
    }

    if (threadIdx.x == 0) {
      #pragma unroll
      for (int ii=0; ii < vec_size; ii++) {
        shared0[ii] = sum_v0.val[ii];
        shared1[ii] = sum_v1.val[ii];
      }
    }
    __syncthreads();

    vec_type2 grad_output_sum, dot_p;
    vec_type2 grad_mean, proj_scale, grad_scale;
    #pragma unroll
    for (int ii=0; ii < vec_size; ii++) {
      grad_output_sum.val[ii] = shared0[ii];
      dot_p.val[ii] = shared1[ii];

      grad_mean.val[ii] = grad_output_sum.val[ii] * norm;
      proj_scale.val[ii] = dot_p.val[ii] * norm * invstd.val[ii] * invstd.val[ii];
      grad_scale.val[ii] = invstd.val[ii] * weight_val.val[ii];
    }

    m_offset = threadIdx.x;
    if (grad_input != nullptr) {
      while (m_offset < reduction_size) {
        const vec_type0 * input_vec = reinterpret_cast<const vec_type0*>(input + m_offset * stride);
        const vec_type0 * grad_output_vec = reinterpret_cast<const vec_type0*>(grad_output + m_offset * stride);
        vec_type0 * grad_input_vec = reinterpret_cast<vec_type0*>(grad_input + m_offset * stride);
        vec_type0 go = grad_output_vec[plane];
        if (train) {
          vec_type0 inp = input_vec[plane];
          vec_type0 tmp;
          vec_type2 proj;
          #pragma unroll
          for (int ii=0; ii < vec_size; ii++) {
            proj.val[ii] = (inp.val[ii] - mean.val[ii]) * proj_scale.val[ii];
            tmp.val[ii] = static_cast<input_scalar_t>((go.val[ii] - proj.val[ii] - grad_mean.val[ii]) * grad_scale.val[ii]);
          }
          grad_input_vec[plane] = tmp;
        } else {
          vec_type0 tmp;
          #pragma unroll
          for (int ii=0; ii < vec_size; ii++) {
            tmp.val[ii] = static_cast<input_scalar_t>(go.val[ii] * grad_scale.val[ii]);
          }
          grad_input_vec[plane] = tmp;
        }

        m_offset += blockDim.x;
      }
    }

    if (threadIdx.x == 0) {
      vec_type1* grad_weight_vec = reinterpret_cast<vec_type1*>(grad_weight);
      vec_type1* grad_bias_vec = reinterpret_cast<vec_type1*>(grad_bias);
      vec_type1 tmp0;
      vec_type1 tmp1;
      #pragma unroll
      for (int ii=0; ii < vec_size; ii++) {
        tmp0.val[ii] = static_cast<type1>(dot_p.val[ii] * invstd.val[ii]);
        tmp1.val[ii] = static_cast<type1>(grad_output_sum.val[ii]);
      }
      grad_weight_vec[plane] = tmp0;
      grad_bias_vec[plane] = tmp1;
    }
}


template <int vec_size, typename input_scalar_t, typename stat_scalar_t, typename stat_accscalar_t, typename index_t>
__global__ void batch_norm_backward_kernel_opt1(
    const input_scalar_t* input,         //type0
    const input_scalar_t* grad_output,   //type0
    input_scalar_t* grad_input,          //type0
    stat_scalar_t* grad_weight,          //type1
    stat_scalar_t* grad_bias,            //type1
    const stat_scalar_t* weight,         //type1
    const stat_scalar_t* running_mean,   //type1
    const stat_scalar_t* running_var,    //type1
    const stat_accscalar_t* save_mean,   //type2
    const stat_accscalar_t* save_invstd, //type2
    stat_accscalar_t* staging_sum0,      //type2
    stat_accscalar_t* staging_sum1,      //type2
    int* semaphores,
    bool train,
    stat_accscalar_t epsilon,            //type2
    const index_t stride,
    const index_t reduction_size
    ) {
    using type0 = input_scalar_t;
    using type1 = stat_scalar_t;
    using type2 = stat_accscalar_t;
    using vec_type0 = aligned_vector<input_scalar_t, vec_size>;
    using vec_type1 = aligned_vector<stat_scalar_t, vec_size>;
    using vec_type2 = aligned_vector<stat_accscalar_t, vec_size>;

    const vec_type1* weight_vec = reinterpret_cast<const vec_type1*>(weight);
    const vec_type1* running_mean_vec = reinterpret_cast<const vec_type1*>(running_mean);
    const vec_type1* running_var_vec = reinterpret_cast<const vec_type1*>(running_var);
    const vec_type2* save_mean_vec = reinterpret_cast<const vec_type2*>(save_mean);
    const vec_type2* save_invstd_vec = reinterpret_cast<const vec_type2*>(save_invstd);

    index_t plane = blockIdx.y;
    index_t N = reduction_size;

    vec_type2 mean, invstd;
    if (train) {
      mean = save_mean_vec[plane];
      invstd = save_invstd_vec[plane];
    } else {
      vec_type1 tmp0, tmp1;
      tmp0 = running_mean_vec[plane];
      tmp1 = running_var_vec[plane];
      #pragma unroll
      for (int ii=0; ii < vec_size; ii++) {
        mean.val[ii] = static_cast<type2>(tmp0.val[ii]);
        invstd.val[ii] = static_cast<type2>(1) / device_sqrt(static_cast<type2>(tmp1.val[ii]) + epsilon);
      }
    }

    vec_type2 weight_val;
    vec_type1 tmp0 = weight_vec[plane];
    #pragma unroll
    for (int ii=0; ii < vec_size; ii++) {
      weight_val.val[ii] = static_cast<type2>(tmp0.val[ii]);
    }
    type2 norm = type2(1) / N;

    //reduce
    vec_type2 sum_v0, sum_v1;
    #pragma unroll
    for (int ii=0; ii < vec_size; ii++) {
      sum_v0.val[ii] = 0; sum_v1.val[ii] = 0;
    }

    //thread reduce
    int m_offset = blockIdx.x * blockDim.x + threadIdx.x;
    int increment = gridDim.x * blockDim.x;
    while (m_offset < reduction_size) {
      const vec_type0 * input_vec = reinterpret_cast<const vec_type0*>(input + m_offset * stride);
      const vec_type0 * grad_output_vec = reinterpret_cast<const vec_type0*>(grad_output+ m_offset * stride);
      const vec_type0 input_item = input_vec[plane];
      const vec_type0 grad_output_item = grad_output_vec[plane];

      #pragma unroll
      for (int ii=0; ii < vec_size; ii++) {
        type2 g = static_cast<type2>(grad_output_item.val[ii]);
        type2 c = static_cast<type2>(input_item.val[ii]) - mean.val[ii];
        sum_v0.val[ii] += g;
        sum_v1.val[ii] += (g*c);
      }
      m_offset += increment;
    }

    //blockreduce
    __shared__ type2 shared0[C10_WARP_SIZE * vec_size];
    __shared__ type2 shared1[C10_WARP_SIZE * vec_size];
    using B = cuda_utils::Block1D;
    const int tid = B::Tid();
    const int lid = tid % C10_WARP_SIZE;  // thread index in warp
    const int wid = tid / C10_WARP_SIZE;  // warp index

    #pragma unroll
    for (int ii=0; ii < vec_size; ii++) {
      #pragma unroll
      for (int offset = (C10_WARP_SIZE >> 1); offset > 0; offset >>= 1) {
        sum_v0.val[ii] += WARP_SHFL_DOWN(sum_v0.val[ii], offset);
        sum_v1.val[ii] += WARP_SHFL_DOWN(sum_v1.val[ii], offset);
      }
      __syncthreads(); 
      if (lid == 0) {
        shared0[wid] = sum_v0.val[ii];
        shared1[wid] = sum_v1.val[ii];
      }
      __syncthreads();
      sum_v0.val[ii] = (tid < B::Warps()) ? shared0[lid] : 0;
      sum_v1.val[ii] = (tid < B::Warps()) ? shared1[lid] : 0;
      if (wid == 0) {
        #pragma unroll
        for (int offset = (C10_WARP_SIZE >> 1); offset > 0; offset >>= 1) {
          sum_v0.val[ii] += WARP_SHFL_DOWN(sum_v0.val[ii], offset);
          sum_v1.val[ii] += WARP_SHFL_DOWN(sum_v1.val[ii], offset);
        }
      }
    }

    //globalreduce
    vec_type2 grad_output_sum, dot_p;
    vec_type2* staging_sum0_vec;
    vec_type2* staging_sum1_vec;
    //gridDim.x>1
    staging_sum0_vec = reinterpret_cast<vec_type2*>(staging_sum0 + blockIdx.x * stride);
    staging_sum1_vec = reinterpret_cast<vec_type2*>(staging_sum1 + blockIdx.x * stride);
    if (threadIdx.x == 0) {
    staging_sum0_vec[plane] = sum_v0;
    staging_sum1_vec[plane] = sum_v1;
    }
    __threadfence();
    __syncthreads(); // ensuring writes to staging_ is visible to all blocks

    __shared__ bool is_last_block_done;
    // mark block done
    if (threadIdx.x == 0) {
      int old = atomicAdd(&semaphores[blockIdx.y], 1);
      is_last_block_done = (old == (gridDim.x-1));
    }
    __syncthreads();

    if (is_last_block_done && wid == 0) {
      if (threadIdx.x < gridDim.x) {
        staging_sum0_vec = reinterpret_cast<vec_type2*>(staging_sum0 + threadIdx.x * stride);
        staging_sum1_vec = reinterpret_cast<vec_type2*>(staging_sum1 + threadIdx.x * stride);
        sum_v0 = staging_sum0_vec[plane];
        sum_v1 = staging_sum1_vec[plane];
      } else {
        #pragma unroll
        for (int ii=0; ii < vec_size; ii++) {
          sum_v0.val[ii] = 0;
          sum_v1.val[ii] = 0;
        }
      }

      for (int ii=0; ii < vec_size; ii++) {
        #pragma unroll
        for (int offset = (C10_WARP_SIZE >> 1); offset > 0; offset >>= 1) {
          sum_v0.val[ii] += WARP_SHFL_DOWN(sum_v0.val[ii], offset);
          sum_v1.val[ii] += WARP_SHFL_DOWN(sum_v1.val[ii], offset);
        }
      }
      __syncthreads();

      if (threadIdx.x == 0) {
        reinterpret_cast<vec_type2*>(staging_sum0)[plane] = sum_v0;
        reinterpret_cast<vec_type2*>(staging_sum1)[plane] = sum_v1;
        vec_type1* grad_weight_vec = reinterpret_cast<vec_type1*>(grad_weight);
        vec_type1* grad_bias_vec = reinterpret_cast<vec_type1*>(grad_bias);
        vec_type1 tmp0;
        vec_type1 tmp1;
        #pragma unroll
        for (int ii=0; ii < vec_size; ii++) {
          tmp0.val[ii] = static_cast<type1>(sum_v1.val[ii] * invstd.val[ii]);
          tmp1.val[ii] = static_cast<type1>(sum_v0.val[ii]);
        }
        grad_weight_vec[plane] = tmp0;
        grad_bias_vec[plane] = tmp1;
      }
    }
}


template <int vec_size, typename input_scalar_t, typename stat_scalar_t, typename stat_accscalar_t, typename index_t>
__global__ void batch_norm_backward_kernel_grad_input_opt1(
    const input_scalar_t* input,         //type0
    const input_scalar_t* grad_output,   //type0
    input_scalar_t* grad_input,          //type0
    stat_scalar_t* grad_weight,          //type1
    stat_scalar_t* grad_bias,            //type1
    const stat_scalar_t* weight,         //type1
    const stat_scalar_t* running_mean,   //type1
    const stat_scalar_t* running_var,    //type1
    const stat_accscalar_t* save_mean,   //type2
    const stat_accscalar_t* save_invstd, //type2
    stat_accscalar_t* staging_sum0,      //type2
    stat_accscalar_t* staging_sum1,      //type2
    int* semaphores,
    bool train,
    stat_accscalar_t epsilon,            //type2
    const index_t stride,
    const index_t reduction_size
    ) {
    using type0 = input_scalar_t;
    using type1 = stat_scalar_t;
    using type2 = stat_accscalar_t;
    using vec_type0 = aligned_vector<input_scalar_t, vec_size>;
    using vec_type1 = aligned_vector<stat_scalar_t, vec_size>;
    using vec_type2 = aligned_vector<stat_accscalar_t, vec_size>;

    const vec_type1* weight_vec = reinterpret_cast<const vec_type1*>(weight);
    const vec_type1* running_mean_vec = reinterpret_cast<const vec_type1*>(running_mean);
    const vec_type1* running_var_vec = reinterpret_cast<const vec_type1*>(running_var);
    const vec_type2* save_mean_vec = reinterpret_cast<const vec_type2*>(save_mean);
    const vec_type2* save_invstd_vec = reinterpret_cast<const vec_type2*>(save_invstd);
    vec_type2* staging_sum0_vec = reinterpret_cast<vec_type2*>(staging_sum0);
    vec_type2* staging_sum1_vec = reinterpret_cast<vec_type2*>(staging_sum1);

    index_t plane = blockIdx.y;
    index_t N = reduction_size;

    vec_type2 mean, invstd;
    if (train) {
      mean = save_mean_vec[plane];
      invstd = save_invstd_vec[plane];
    } else {
      vec_type1 tmp0, tmp1;
      tmp0 = running_mean_vec[plane];
      tmp1 = running_var_vec[plane];
      #pragma unroll
      for (int ii=0; ii < vec_size; ii++) {
        mean.val[ii] = static_cast<type2>(tmp0.val[ii]);
        invstd.val[ii] = static_cast<type2>(1) / device_sqrt(static_cast<type2>(tmp1.val[ii]) + epsilon);
      }
    }

    vec_type2 weight_val;
    vec_type1 tmp0 = weight_vec[plane];
    #pragma unroll
    for (int ii=0; ii < vec_size; ii++) {
      weight_val.val[ii] = static_cast<type2>(tmp0.val[ii]);
    }
    type2 norm = type2(1) / N;

    vec_type2 grad_output_sum, dot_p;
    grad_output_sum = staging_sum0_vec[plane];
    dot_p = staging_sum1_vec[plane];

    vec_type2 grad_mean, proj_scale, grad_scale;
    #pragma unroll
    for (int ii=0; ii < vec_size; ii++) {
      grad_mean.val[ii] = grad_output_sum.val[ii] * norm;
      proj_scale.val[ii] = dot_p.val[ii] * norm * invstd.val[ii] * invstd.val[ii];
      grad_scale.val[ii] = invstd.val[ii] * weight_val.val[ii];
    }

    // int m_offset = blockIdx.x * blockDim.x + threadIdx.x;
    // int increment = gridDim.x * blockDim.x;
    int m_offset = threadIdx.x;
    int increment = blockDim.x;
    if (grad_input != nullptr) {
      while (m_offset < reduction_size){
        const vec_type0 * input_vec = reinterpret_cast<const vec_type0*>(input + m_offset * stride);
        const vec_type0 * grad_output_vec = reinterpret_cast<const vec_type0*>(grad_output + m_offset * stride);
        vec_type0 * grad_input_vec = reinterpret_cast<vec_type0*>(grad_input + m_offset * stride);
        vec_type0 go = grad_output_vec[plane];
        if (train) {
          vec_type0 inp = input_vec[plane];
          vec_type0 tmp;
          vec_type2 proj;
          #pragma unroll
          for (int ii=0; ii < vec_size; ii++) {
            proj.val[ii] = (inp.val[ii] - mean.val[ii]) * proj_scale.val[ii];
            tmp.val[ii] = static_cast<input_scalar_t>((go.val[ii] - proj.val[ii] - grad_mean.val[ii]) * grad_scale.val[ii]);
          }
          grad_input_vec[plane] = tmp;
        } else {
          vec_type0 tmp;
          #pragma unroll
          for (int ii=0; ii < vec_size; ii++) {
            tmp.val[ii] = static_cast<input_scalar_t>(go.val[ii] * grad_scale.val[ii]);
          }
          grad_input_vec[plane] = tmp;
        }
        m_offset += increment;
      }
    }
}


template<typename T>
__device__ __forceinline__ void merge_block_vertical_backward_opt_fp32(T& sum_dy,
    T& sum_dy_xmu,
    T* shmem_sum_dy,
    T* shmem_sum_dy_xmu) {
  // write to shared memory
  auto address_base = threadIdx.x + threadIdx.y * blockDim.x;

#pragma unroll
  for (int offset = blockDim.y/2; offset > 0; offset >>= 1) {
    if (threadIdx.y < offset*2) {
      shmem_sum_dy[address_base] = sum_dy;
      shmem_sum_dy_xmu[address_base] = sum_dy_xmu;
    }
    __syncthreads();
    if (threadIdx.y < offset && threadIdx.y + offset < blockDim.y) {
      auto address = address_base + offset * blockDim.x;
      
      (sum_dy).x += (shmem_sum_dy[address]).x;
      (sum_dy).y += (shmem_sum_dy[address]).y;
      (sum_dy).z += (shmem_sum_dy[address]).z;
      (sum_dy).w += (shmem_sum_dy[address]).w;
      (sum_dy_xmu).x += (shmem_sum_dy_xmu[address]).x;
      (sum_dy_xmu).y += (shmem_sum_dy_xmu[address]).y;
      (sum_dy_xmu).z += (shmem_sum_dy_xmu[address]).z;
      (sum_dy_xmu).w += (shmem_sum_dy_xmu[address]).w;
    }
  }
}


template <
    int PARALLEL_LOADS,
    typename scalar_t,
    typename accscalar_t,
    typename layerscalar_t>
__global__ void batch_norm_backward_reduce_channels_last_kernel_opt(
      const scalar_t* __restrict__ input,
      const scalar_t* __restrict__ grad_output,
      const accscalar_t* __restrict__ mean,
      const accscalar_t* __restrict__ inv_std,
      accscalar_t* __restrict__ sum_dy_o,
      accscalar_t* __restrict__ sum_dy_xmu_o,
      layerscalar_t* __restrict__ grad_weight,
      layerscalar_t* __restrict__ grad_bias,
      accscalar_t* staging_data,
      int* semaphores,
      const int reduction_size,
      const int stride) {
  assert(false);
}


// batchnorm backward kernel for c last tensor
// original apex name: reduce_bn_c_last_kernel
template<>
__global__ void batch_norm_backward_reduce_channels_last_kernel_opt<ELEMENTS_PER_ITER, float,float,float>(
      const float* __restrict__ input,
      const float* __restrict__ grad_output,
      const float* __restrict__ mean,
      const float* __restrict__ inv_std,
      float* __restrict__ sum_dy_o,
      float* __restrict__ sum_dy_xmu_o,
      float* __restrict__ grad_weight,
      float* __restrict__ grad_bias,
      float* staging_data,
      int* semaphores,
      const int reduction_size,
      const int stride) {
  // hide latency with concurrency
  #define PARALLEL_LOADS ELEMENTS_PER_ITER
  
  float4 sum_dy0=make_float4(0,0,0,0), sum_dy1=make_float4(0,0,0,0), sum_dy2=make_float4(0,0,0,0), sum_dy3=make_float4(0,0,0,0);
  float4 sum_dy_xmu0=make_float4(0,0,0,0), sum_dy_xmu1=make_float4(0,0,0,0), sum_dy_xmu2=make_float4(0,0,0,0), sum_dy_xmu3=make_float4(0,0,0,0); 

  int inner_loop_stride = blockDim.y * gridDim.y;
  int m_offset = blockIdx.y * blockDim.y + threadIdx.y;
  int c_offset = (blockIdx.x * blockDim.x + threadIdx.x) * 4;

  if (c_offset >= stride || m_offset >= reduction_size) {
    return;
  }

  int loop_count = 1 + (reduction_size - 1) / (inner_loop_stride * PARALLEL_LOADS);
  int address_base = m_offset * stride + c_offset;
  int address_increment = inner_loop_stride * stride;
  
  float4 r_mean = *reinterpret_cast<const float4*>(mean + c_offset);
  float4 factor = *reinterpret_cast<const float4*>(inv_std + c_offset);
  
  float4 x_input0, x_input1, x_input2, x_input3;
  float4 x_grad_output0, x_grad_output1, x_grad_output2, x_grad_output3;
  
  for (int i = 0; i < loop_count; i++) {
    {
      x_input0 = (m_offset < reduction_size)?*reinterpret_cast<const float4*>(input + address_base):make_float4(0,0,0,0);
      x_grad_output0 = (m_offset < reduction_size)?*reinterpret_cast<const float4*>(grad_output + address_base):make_float4(0,0,0,0);
      m_offset += inner_loop_stride;
      address_base += address_increment;

      x_input1 = (m_offset < reduction_size)?*reinterpret_cast<const float4*>(input + address_base):make_float4(0,0,0,0);
      x_grad_output1 = (m_offset < reduction_size)?*reinterpret_cast<const float4*>(grad_output + address_base):make_float4(0,0,0,0);
      m_offset += inner_loop_stride;
      address_base += address_increment;

      x_input2 = (m_offset < reduction_size)?*reinterpret_cast<const float4*>(input + address_base):make_float4(0,0,0,0);
      x_grad_output2 = (m_offset < reduction_size)?*reinterpret_cast<const float4*>(grad_output + address_base):make_float4(0,0,0,0);
      m_offset += inner_loop_stride;
      address_base += address_increment;

      x_input3 = (m_offset < reduction_size)?*reinterpret_cast<const float4*>(input + address_base):make_float4(0,0,0,0);
      x_grad_output3 = (m_offset < reduction_size)?*reinterpret_cast<const float4*>(grad_output + address_base):make_float4(0,0,0,0);
      m_offset += inner_loop_stride;
      address_base += address_increment;
      
    }

    {
      sum_dy0.x += x_grad_output0.x;
      sum_dy0.y += x_grad_output0.y;
      sum_dy0.z += x_grad_output0.z;
      sum_dy0.w += x_grad_output0.w;
      sum_dy_xmu0.x += x_grad_output0.x * (x_input0.x - r_mean.x);
      sum_dy_xmu0.y += x_grad_output0.y * (x_input0.y - r_mean.y);
      sum_dy_xmu0.z += x_grad_output0.z * (x_input0.z - r_mean.z);
      sum_dy_xmu0.w += x_grad_output0.w * (x_input0.w - r_mean.w);
      
      sum_dy1.x += x_grad_output1.x;
      sum_dy1.y += x_grad_output1.y;
      sum_dy1.z += x_grad_output1.z;
      sum_dy1.w += x_grad_output1.w;
      sum_dy_xmu1.x += x_grad_output1.x * (x_input1.x - r_mean.x);
      sum_dy_xmu1.y += x_grad_output1.y * (x_input1.y - r_mean.y);
      sum_dy_xmu1.z += x_grad_output1.z * (x_input1.z - r_mean.z);
      sum_dy_xmu1.w += x_grad_output1.w * (x_input1.w - r_mean.w);
      
      sum_dy2.x += x_grad_output2.x;
      sum_dy2.y += x_grad_output2.y;
      sum_dy2.z += x_grad_output2.z;
      sum_dy2.w += x_grad_output2.w;
      sum_dy_xmu2.x += x_grad_output2.x * (x_input2.x - r_mean.x);
      sum_dy_xmu2.y += x_grad_output2.y * (x_input2.y - r_mean.y);
      sum_dy_xmu2.z += x_grad_output2.z * (x_input2.z - r_mean.z);
      sum_dy_xmu2.w += x_grad_output2.w * (x_input2.w - r_mean.w);

      sum_dy3.x += x_grad_output3.x;
      sum_dy3.y += x_grad_output3.y;
      sum_dy3.z += x_grad_output3.z;
      sum_dy3.w += x_grad_output3.w;
      sum_dy_xmu3.x += x_grad_output3.x * (x_input3.x - r_mean.x);
      sum_dy_xmu3.y += x_grad_output3.y * (x_input3.y - r_mean.y);
      sum_dy_xmu3.z += x_grad_output3.z * (x_input3.z - r_mean.z);
      sum_dy_xmu3.w += x_grad_output3.w * (x_input3.w - r_mean.w);
    }
  }
  
  {
    sum_dy0.x += sum_dy1.x;
    sum_dy0.y += sum_dy1.y;
    sum_dy0.z += sum_dy1.z;
    sum_dy0.w += sum_dy1.w;
    sum_dy_xmu0.x += sum_dy_xmu1.x;
    sum_dy_xmu0.y += sum_dy_xmu1.y;
    sum_dy_xmu0.z += sum_dy_xmu1.z;
    sum_dy_xmu0.w += sum_dy_xmu1.w;

    sum_dy0.x += sum_dy2.x;
    sum_dy0.y += sum_dy2.y;
    sum_dy0.z += sum_dy2.z;
    sum_dy0.w += sum_dy2.w;
    sum_dy_xmu0.x += sum_dy_xmu2.x;
    sum_dy_xmu0.y += sum_dy_xmu2.y;
    sum_dy_xmu0.z += sum_dy_xmu2.z;
    sum_dy_xmu0.w += sum_dy_xmu2.w;

    sum_dy0.x += sum_dy3.x;
    sum_dy0.y += sum_dy3.y;
    sum_dy0.z += sum_dy3.z;
    sum_dy0.w += sum_dy3.w;
    sum_dy_xmu0.x += sum_dy_xmu3.x;
    sum_dy_xmu0.y += sum_dy_xmu3.y;
    sum_dy_xmu0.z += sum_dy_xmu3.z;
    sum_dy_xmu0.w += sum_dy_xmu3.w;
  }
  
  // release array of registers
  auto sum_dy_th = sum_dy0;
  auto sum_dy_xmu_th = sum_dy_xmu0;
  
  static __shared__ float4 shmem_sum_dy[MAX_BLOCK_SIZE_OPT];
  static __shared__ float4 shmem_sum_dy_xmu[MAX_BLOCK_SIZE_OPT];

  merge_block_vertical_backward_opt_fp32(sum_dy_th, sum_dy_xmu_th, shmem_sum_dy, shmem_sum_dy_xmu);

  if (gridDim.y > 1) {
    float* staging_sum_dy = staging_data;
    float* staging_sum_dy_xmu = &staging_data[stride*gridDim.y];

    address_base = c_offset + blockIdx.y * stride;
    // write data to staging_data;
    if (threadIdx.y == 0) {
      *(reinterpret_cast<float4*>(&staging_sum_dy[address_base])) = sum_dy_th;
      *(reinterpret_cast<float4*>(&staging_sum_dy_xmu[address_base])) = sum_dy_xmu_th;
    }

    __threadfence();
    __syncthreads(); // ensuring writes to staging_ is visible to all blocks

    __shared__ bool is_last_block_done;
    // mark block done
    if (threadIdx.x == 0 && threadIdx.y == 0) {
      int old = atomicAdd(&semaphores[blockIdx.x], 1);
      is_last_block_done = (old == (gridDim.y-1));
    }

    __syncthreads();
    float4 tmp;
    // check that all data is now available in global memory
    if (is_last_block_done) {
      sum_dy_th = make_float4(0,0,0,0);
      sum_dy_xmu_th = make_float4(0,0,0,0);

      for (int y = threadIdx.y; y < gridDim.y; y += blockDim.y) {
        address_base = c_offset + y * stride;
        tmp = *(reinterpret_cast<float4*>(&staging_sum_dy[address_base]));

        sum_dy_th.x += tmp.x;
        sum_dy_th.y += tmp.y;
        sum_dy_th.z += tmp.z;
        sum_dy_th.w += tmp.w;
        
        tmp = *(reinterpret_cast<float4*>(&staging_sum_dy_xmu[address_base]));
        sum_dy_xmu_th.x += tmp.x;
        sum_dy_xmu_th.y += tmp.y;
        sum_dy_xmu_th.z += tmp.z;
        sum_dy_xmu_th.w += tmp.w;
      }
      
      merge_block_vertical_backward_opt_fp32(sum_dy_th, sum_dy_xmu_th, shmem_sum_dy, shmem_sum_dy_xmu);
      if (threadIdx.y == 0) {
        if (grad_bias != nullptr) {
          *(reinterpret_cast<float4*>(&grad_bias[c_offset])) = sum_dy_th;
        }
        if (grad_weight != nullptr) {
          tmp.x = sum_dy_xmu_th.x * factor.x;
          tmp.y = sum_dy_xmu_th.y * factor.y;
          tmp.z = sum_dy_xmu_th.z * factor.z;
          tmp.w = sum_dy_xmu_th.w * factor.w;
          *(reinterpret_cast<float4*>(&grad_weight[c_offset])) = tmp;
        }
        *(reinterpret_cast<float4*>(&sum_dy_o[c_offset])) = sum_dy_th;
        *(reinterpret_cast<float4*>(&sum_dy_xmu_o[c_offset])) = sum_dy_xmu_th;
      }
    }
  } else {
    float4 tmp;
    if (blockIdx.y == 0 && threadIdx.y == 0) {
      if (grad_bias != nullptr) {
        *(reinterpret_cast<float4*>(&grad_bias[c_offset])) = sum_dy_th;
      }
      if (grad_weight != nullptr) {
        tmp.x = sum_dy_xmu_th.x * factor.x;
        tmp.y = sum_dy_xmu_th.y * factor.y;
        tmp.z = sum_dy_xmu_th.z * factor.z;
        tmp.w = sum_dy_xmu_th.w * factor.w;
        *(reinterpret_cast<float4*>(&grad_weight[c_offset])) = tmp;
      }
      *(reinterpret_cast<float4*>(&sum_dy_o[c_offset])) = sum_dy_th;
      *(reinterpret_cast<float4*>(&sum_dy_xmu_o[c_offset])) = sum_dy_xmu_th;
    }
  }

}

// batchnorm backward kernel for c last tensor
// original apex name: reduce_bn_c_last_kernel
template<>
__global__ void batch_norm_backward_reduce_channels_last_kernel_opt<ELEMENTS_PER_ITER, at::Half,float,float>(
      const at::Half* __restrict__ input,
      const at::Half* __restrict__ grad_output,
      const float* __restrict__ mean,
      const float* __restrict__ inv_std,
      float* __restrict__ sum_dy_o,
      float* __restrict__ sum_dy_xmu_o,
      float* __restrict__ grad_weight,
      float* __restrict__ grad_bias,
      float* staging_data,
      int* semaphores,
      const int reduction_size,
      const int stride) {
  // hide latency with concurrency
  #define PARALLEL_LOADS ELEMENTS_PER_ITER
  
  float4 sum_dy0=make_float4(0,0,0,0), sum_dy1=make_float4(0,0,0,0), sum_dy2=make_float4(0,0,0,0), sum_dy3=make_float4(0,0,0,0);
  float4 sum_dy_xmu0=make_float4(0,0,0,0), sum_dy_xmu1=make_float4(0,0,0,0), sum_dy_xmu2=make_float4(0,0,0,0), sum_dy_xmu3=make_float4(0,0,0,0); 

  int inner_loop_stride = blockDim.y * gridDim.y;
  int m_offset = blockIdx.y * blockDim.y + threadIdx.y;
  int c_offset = (blockIdx.x * blockDim.x + threadIdx.x) * 4;

  if (c_offset >= stride || m_offset >= reduction_size) {
    return;
  }

  int loop_count = 1 + (reduction_size - 1) / (inner_loop_stride * PARALLEL_LOADS);
  int address_base = m_offset * stride + c_offset;
  int address_increment = inner_loop_stride * stride;
  
  float4 r_mean = *reinterpret_cast<const float4*>(mean + c_offset);
  float4 factor = *reinterpret_cast<const float4*>(inv_std + c_offset);
  
  int64_t x_input0, x_input1, x_input2, x_input3;
  int64_t x_grad_output0, x_grad_output1, x_grad_output2, x_grad_output3;
  
  for (int i = 0; i < loop_count; i++) {
    {
      x_input0 = (m_offset < reduction_size)?*reinterpret_cast<const int64_t*>(input + address_base):0;
      x_grad_output0 = (m_offset < reduction_size)?*reinterpret_cast<const int64_t*>(grad_output + address_base):0;
      m_offset += inner_loop_stride;
      address_base += address_increment;

      x_input1 = (m_offset < reduction_size)?*reinterpret_cast<const int64_t*>(input + address_base):0;
      x_grad_output1 = (m_offset < reduction_size)?*reinterpret_cast<const int64_t*>(grad_output + address_base):0;
      m_offset += inner_loop_stride;
      address_base += address_increment;

      x_input2 = (m_offset < reduction_size)?*reinterpret_cast<const int64_t*>(input + address_base):0;
      x_grad_output2 = (m_offset < reduction_size)?*reinterpret_cast<const int64_t*>(grad_output + address_base):0;
      m_offset += inner_loop_stride;
      address_base += address_increment;

      x_input3 = (m_offset < reduction_size)?*reinterpret_cast<const int64_t*>(input + address_base):0;
      x_grad_output3 = (m_offset < reduction_size)?*reinterpret_cast<const int64_t*>(grad_output + address_base):0;
      m_offset += inner_loop_stride;
      address_base += address_increment;
      
    }

    {
      sum_dy0.x += *(reinterpret_cast<at::Half*>(&x_grad_output0)+0);
      sum_dy0.y += *(reinterpret_cast<at::Half*>(&x_grad_output0)+1);
      sum_dy0.z += *(reinterpret_cast<at::Half*>(&x_grad_output0)+2);
      sum_dy0.w += *(reinterpret_cast<at::Half*>(&x_grad_output0)+3);
      sum_dy_xmu0.x += *(reinterpret_cast<at::Half*>(&x_grad_output0)+0) * (*(reinterpret_cast<at::Half*>(&x_input0)+0) - r_mean.x);
      sum_dy_xmu0.y += *(reinterpret_cast<at::Half*>(&x_grad_output0)+1) * (*(reinterpret_cast<at::Half*>(&x_input0)+1) - r_mean.y);
      sum_dy_xmu0.z += *(reinterpret_cast<at::Half*>(&x_grad_output0)+2) * (*(reinterpret_cast<at::Half*>(&x_input0)+2) - r_mean.z);
      sum_dy_xmu0.w += *(reinterpret_cast<at::Half*>(&x_grad_output0)+3) * (*(reinterpret_cast<at::Half*>(&x_input0)+3) - r_mean.w);
      
      sum_dy1.x += *(reinterpret_cast<at::Half*>(&x_grad_output1)+0);
      sum_dy1.y += *(reinterpret_cast<at::Half*>(&x_grad_output1)+1);
      sum_dy1.z += *(reinterpret_cast<at::Half*>(&x_grad_output1)+2);
      sum_dy1.w += *(reinterpret_cast<at::Half*>(&x_grad_output1)+3);
      sum_dy_xmu1.x += *(reinterpret_cast<at::Half*>(&x_grad_output1)+0) * (*(reinterpret_cast<at::Half*>(&x_input1)+0) - r_mean.x);
      sum_dy_xmu1.y += *(reinterpret_cast<at::Half*>(&x_grad_output1)+1) * (*(reinterpret_cast<at::Half*>(&x_input1)+1) - r_mean.y);
      sum_dy_xmu1.z += *(reinterpret_cast<at::Half*>(&x_grad_output1)+2) * (*(reinterpret_cast<at::Half*>(&x_input1)+2) - r_mean.z);
      sum_dy_xmu1.w += *(reinterpret_cast<at::Half*>(&x_grad_output1)+3) * (*(reinterpret_cast<at::Half*>(&x_input1)+3) - r_mean.w);

      sum_dy2.x += *(reinterpret_cast<at::Half*>(&x_grad_output2)+0);
      sum_dy2.y += *(reinterpret_cast<at::Half*>(&x_grad_output2)+1);
      sum_dy2.z += *(reinterpret_cast<at::Half*>(&x_grad_output2)+2);
      sum_dy2.w += *(reinterpret_cast<at::Half*>(&x_grad_output2)+3);
      sum_dy_xmu2.x += *(reinterpret_cast<at::Half*>(&x_grad_output2)+0) * (*(reinterpret_cast<at::Half*>(&x_input2)+0) - r_mean.x);
      sum_dy_xmu2.y += *(reinterpret_cast<at::Half*>(&x_grad_output2)+1) * (*(reinterpret_cast<at::Half*>(&x_input2)+1) - r_mean.y);
      sum_dy_xmu2.z += *(reinterpret_cast<at::Half*>(&x_grad_output2)+2) * (*(reinterpret_cast<at::Half*>(&x_input2)+2) - r_mean.z);
      sum_dy_xmu2.w += *(reinterpret_cast<at::Half*>(&x_grad_output2)+3) * (*(reinterpret_cast<at::Half*>(&x_input2)+3) - r_mean.w);

      sum_dy3.x += *(reinterpret_cast<at::Half*>(&x_grad_output3)+0);
      sum_dy3.y += *(reinterpret_cast<at::Half*>(&x_grad_output3)+1);
      sum_dy3.z += *(reinterpret_cast<at::Half*>(&x_grad_output3)+2);
      sum_dy3.w += *(reinterpret_cast<at::Half*>(&x_grad_output3)+3);
      sum_dy_xmu3.x += *(reinterpret_cast<at::Half*>(&x_grad_output3)+0) * (*(reinterpret_cast<at::Half*>(&x_input3)+0) - r_mean.x);
      sum_dy_xmu3.y += *(reinterpret_cast<at::Half*>(&x_grad_output3)+1) * (*(reinterpret_cast<at::Half*>(&x_input3)+1) - r_mean.y);
      sum_dy_xmu3.z += *(reinterpret_cast<at::Half*>(&x_grad_output3)+2) * (*(reinterpret_cast<at::Half*>(&x_input3)+2) - r_mean.z);
      sum_dy_xmu3.w += *(reinterpret_cast<at::Half*>(&x_grad_output3)+3) * (*(reinterpret_cast<at::Half*>(&x_input3)+3) - r_mean.w);
    }
  }
  
  {
    sum_dy0.x += sum_dy1.x;
    sum_dy0.y += sum_dy1.y;
    sum_dy0.z += sum_dy1.z;
    sum_dy0.w += sum_dy1.w;
    sum_dy_xmu0.x += sum_dy_xmu1.x;
    sum_dy_xmu0.y += sum_dy_xmu1.y;
    sum_dy_xmu0.z += sum_dy_xmu1.z;
    sum_dy_xmu0.w += sum_dy_xmu1.w;

    sum_dy0.x += sum_dy2.x;
    sum_dy0.y += sum_dy2.y;
    sum_dy0.z += sum_dy2.z;
    sum_dy0.w += sum_dy2.w;
    sum_dy_xmu0.x += sum_dy_xmu2.x;
    sum_dy_xmu0.y += sum_dy_xmu2.y;
    sum_dy_xmu0.z += sum_dy_xmu2.z;
    sum_dy_xmu0.w += sum_dy_xmu2.w;

    sum_dy0.x += sum_dy3.x;
    sum_dy0.y += sum_dy3.y;
    sum_dy0.z += sum_dy3.z;
    sum_dy0.w += sum_dy3.w;
    sum_dy_xmu0.x += sum_dy_xmu3.x;
    sum_dy_xmu0.y += sum_dy_xmu3.y;
    sum_dy_xmu0.z += sum_dy_xmu3.z;
    sum_dy_xmu0.w += sum_dy_xmu3.w;
  }
  
  // release array of registers
  auto sum_dy_th = sum_dy0;
  auto sum_dy_xmu_th = sum_dy_xmu0;
  
  static __shared__ float4 shmem_sum_dy[MAX_BLOCK_SIZE_OPT];
  static __shared__ float4 shmem_sum_dy_xmu[MAX_BLOCK_SIZE_OPT];

  merge_block_vertical_backward_opt_fp32(sum_dy_th, sum_dy_xmu_th, shmem_sum_dy, shmem_sum_dy_xmu);

  if (gridDim.y > 1) {
    float* staging_sum_dy = staging_data;
    float* staging_sum_dy_xmu = &staging_data[stride*gridDim.y];

    address_base = c_offset + blockIdx.y * stride;
    // write data to staging_data;
    if (threadIdx.y == 0) {
      *(reinterpret_cast<float4*>(&staging_sum_dy[address_base])) = sum_dy_th;
      *(reinterpret_cast<float4*>(&staging_sum_dy_xmu[address_base])) = sum_dy_xmu_th;
    }

    __threadfence();
    __syncthreads(); // ensuring writes to staging_ is visible to all blocks

    __shared__ bool is_last_block_done;
    // mark block done
    if (threadIdx.x == 0 && threadIdx.y == 0) {
      int old = atomicAdd(&semaphores[blockIdx.x], 1);
      is_last_block_done = (old == (gridDim.y-1));
    }

    __syncthreads();
    float4 tmp;
    // check that all data is now available in global memory
    if (is_last_block_done) {
      sum_dy_th = make_float4(0,0,0,0);
      sum_dy_xmu_th = make_float4(0,0,0,0);

      for (int y = threadIdx.y; y < gridDim.y; y += blockDim.y) {
        address_base = c_offset + y * stride;
        tmp = *(reinterpret_cast<float4*>(&staging_sum_dy[address_base]));

        sum_dy_th.x += tmp.x;
        sum_dy_th.y += tmp.y;
        sum_dy_th.z += tmp.z;
        sum_dy_th.w += tmp.w;
        
        tmp = *(reinterpret_cast<float4*>(&staging_sum_dy_xmu[address_base]));
        sum_dy_xmu_th.x += tmp.x;
        sum_dy_xmu_th.y += tmp.y;
        sum_dy_xmu_th.z += tmp.z;
        sum_dy_xmu_th.w += tmp.w;
      }
      
      merge_block_vertical_backward_opt_fp32(sum_dy_th, sum_dy_xmu_th, shmem_sum_dy, shmem_sum_dy_xmu);
      if (threadIdx.y == 0) {
        if (grad_bias != nullptr) {
          *(reinterpret_cast<float4*>(&grad_bias[c_offset])) = sum_dy_th;
        }
        if (grad_weight != nullptr) {
          tmp.x = sum_dy_xmu_th.x * factor.x;
          tmp.y = sum_dy_xmu_th.y * factor.y;
          tmp.z = sum_dy_xmu_th.z * factor.z;
          tmp.w = sum_dy_xmu_th.w * factor.w;
          *(reinterpret_cast<float4*>(&grad_weight[c_offset])) = tmp;
        }
        *(reinterpret_cast<float4*>(&sum_dy_o[c_offset])) = sum_dy_th;
        *(reinterpret_cast<float4*>(&sum_dy_xmu_o[c_offset])) = sum_dy_xmu_th;
      }
    }
  } else {
    float4 tmp;
    if (blockIdx.y == 0 && threadIdx.y == 0) {
      if (grad_bias != nullptr) {
        *(reinterpret_cast<float4*>(&grad_bias[c_offset])) = sum_dy_th;
      }
      if (grad_weight != nullptr) {
        tmp.x = sum_dy_xmu_th.x * factor.x;
        tmp.y = sum_dy_xmu_th.y * factor.y;
        tmp.z = sum_dy_xmu_th.z * factor.z;
        tmp.w = sum_dy_xmu_th.w * factor.w;
        *(reinterpret_cast<float4*>(&grad_weight[c_offset])) = tmp;
      }
      *(reinterpret_cast<float4*>(&sum_dy_o[c_offset])) = sum_dy_th;
      *(reinterpret_cast<float4*>(&sum_dy_xmu_o[c_offset])) = sum_dy_xmu_th;
    }
  }

}


}}}