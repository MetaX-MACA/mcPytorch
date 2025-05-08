#pragma once

namespace at::native {

void print_indexing_backward_kernel(const Tensor &src_, const Tensor &expandedValue, 
                                    int64_t num_indices, int64_t sliceSize, int64_t strideBefore, int64_t nElemBefore, bool accumulate) {
  printf("index_put_with_sort_kernel:");
  printf("src_ shape(");
  for (int64_t i = 0; i < src_.dim(); ++i) { printf("%d,",src_.size(i)); }
  printf(") dtype: %s ; ", c10::str(src_.scalar_type()).c_str());

  printf("expandedValue shape(");
  for (int64_t i = 0; i < expandedValue.dim(); ++i) { printf("%d,",expandedValue.size(i)); }
  printf(") dtype: %s ; ", c10::str(expandedValue.scalar_type()).c_str());
  printf("num_indices: %ld; sliceSize: %ld; strideBefore: %ld; nElemBefore: %ld; accumulate: %ld\n",
        num_indices, sliceSize, strideBefore, nElemBefore, accumulate);
}

}

template <typename scalar_t, int SZ>
__global__ void indexing_backward_kernel_opt(
  int64_t* sorted_indices, int64_t* indices, scalar_t* grad_output, scalar_t* grad_weight,
  int64_t numel, int64_t stride, int64_t stride_before, int64_t outer_dim, bool accumulate) {
  // this kernel only for sliceSize == stride == self.size(1): should sliceSize % (warp_size*UNROLL) == 0,
  // we need handle the self all, in another way, we need load and store all the self value.
  // for sliceSize == stride == 1, the original kernel only load some value from the self and src, so in this
  // situation, the load and store instrction maybe only a little
  using opmath_t = at::opmath_type<scalar_t>;
  using MemoryT = at::native::memory::aligned_vector<scalar_t, SZ>;

  for (int64_t z = blockIdx.z; z < outer_dim; z += gridDim.z){
    int64_t idx = blockIdx.x * blockDim.y + threadIdx.y;
    if (idx < numel
        && (idx == 0 || sorted_indices[idx] != sorted_indices[idx - 1])){
      do {
        int64_t start_feature = threadIdx.x * SZ + blockIdx.y * blockDim.x * SZ;
        // if not accumulate, we only keep the last duplicate index so skip those before it
        if (!accumulate && (idx < numel - 1) && sorted_indices[idx] == sorted_indices[idx + 1]) {
          idx++;
          continue;
        }
        const int64_t weight_row = ((int64_t) sorted_indices[idx]) * stride + z * stride_before;
        const int64_t grad_row = ((int64_t) indices[idx]) * stride + z * numel * stride;
        const opmath_t scale = (opmath_t)1.0;

        opmath_t gradient[SZ];
        opmath_t weight[SZ];

        scalar_t gradient_val[SZ];
        scalar_t weight_val[SZ];
        MemoryT* p_ld_grad = reinterpret_cast<MemoryT*>(&gradient_val);
        MemoryT* p_ld_weight = reinterpret_cast<MemoryT*>(&weight_val);

        while (start_feature < stride) {
          int64_t feature_dim = start_feature + SZ - 1;
          if (feature_dim < stride) {
            // load SZ grad value
            *p_ld_grad = *reinterpret_cast<MemoryT*>(grad_output + grad_row + start_feature);
            if (accumulate) {
              // load SZ weight value
              *p_ld_weight = *reinterpret_cast<MemoryT*>(grad_weight + weight_row + start_feature);
            }
            #pragma unroll
            for (int ii = 0; ii < SZ; ii++) {
              gradient[ii] = static_cast<opmath_t>(gradient_val[ii]);
              if (accumulate) {
                weight[ii] = static_cast<opmath_t>(weight_val[ii]);
              }
            }

            #pragma unroll
            for (int ii = 0; ii < SZ; ii++) {
              if (accumulate) {
                weight[ii] += gradient[ii] * scale;
              } else {
                weight[ii] = gradient[ii] * scale;
              }
            }

            #pragma unroll
            for (int ii = 0; ii < SZ; ii++) {
              weight_val[ii] = static_cast<scalar_t>(weight[ii]);
            }
            MemoryT* out = reinterpret_cast<MemoryT*>(grad_weight + weight_row + start_feature);
            *out = *p_ld_weight;
          }

          start_feature += gridDim.y * blockDim.x * SZ;
        }

        idx++;
      } while (idx < numel && sorted_indices[idx] == sorted_indices[idx - 1]);
    }
  }
}


//block:(64,4);
//grid:(num_indices/4, sliceSize/(64*4), nElemBefore)
template <typename scalar_t, int SZ>
__global__ void indexing_backward_kernel_opt1(
  int64_t* sorted_indices,    //sorted_indices
  int64_t* indices,           //orig_indices
  scalar_t* grad_output,      //expandedValue
  scalar_t* grad_weight,      //src_
  int64_t numel,              //num_indices
  int64_t stride,             //sliceSize
  int64_t stride_before,      //strideBefore
  int64_t outer_dim,          //nElemBefore
  bool accumulate
  ) {
  using opmath_t = at::opmath_type<scalar_t>;
  using MemoryT = at::native::memory::aligned_vector<scalar_t, SZ>;

  for (int64_t z = blockIdx.z; z < outer_dim; z += gridDim.z){
    int64_t idx = blockIdx.x * blockDim.y + threadIdx.y;
    if (idx < numel
        && (idx == 0 || sorted_indices[idx] != sorted_indices[idx - 1])){
      do {

        if (!accumulate && (idx < numel - 1) && sorted_indices[idx] == sorted_indices[idx + 1]) {
          idx++;
          continue;
        }
        const int64_t weight_row = ((int64_t) sorted_indices[idx]) * stride + z * stride_before;
        const int64_t grad_row = ((int64_t) indices[idx]) * stride + z * numel * stride;
        const opmath_t scale = (opmath_t)1.0;

        opmath_t gradient[SZ];
        opmath_t weight[SZ];

        MemoryT gradient_val;
        MemoryT weight_val;

        int64_t start_feature = threadIdx.x * SZ + blockIdx.y * blockDim.x * SZ;
        while (start_feature < stride) {
          gradient_val = *reinterpret_cast<MemoryT*>(grad_output + grad_row + start_feature);
          if (accumulate) {
            weight_val = *reinterpret_cast<MemoryT*>(grad_weight + weight_row + start_feature);
          }

          #pragma unroll
          for (int ii = 0; ii < SZ; ii++) {
            if (accumulate) {
              weight[ii] = static_cast<opmath_t>(weight_val.val[ii]) + static_cast<opmath_t>(gradient_val.val[ii]) * scale;
            } else {
              weight[ii] = static_cast<opmath_t>(gradient_val.val[ii]) * scale;
            }
          }

          #pragma unroll
          for (int ii = 0; ii < SZ; ii++) {
            weight_val.val[ii] = static_cast<scalar_t>(weight[ii]);
          }
          *(reinterpret_cast<MemoryT*>(grad_weight + weight_row + start_feature)) = weight_val;

          start_feature += gridDim.y * blockDim.x * SZ;
        }
        idx++;
      } while (idx < numel && sorted_indices[idx] == sorted_indices[idx - 1]);
    }
  }
}