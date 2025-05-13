#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/TensorAdvancedIndexing.h>
#include <ATen/native/IndexingUtils.h>
#include <ATen/native/quantized/IndexKernel.h>
#include <ATen/native/cuda/KernelUtils.cuh>

#include <ATen/core/Tensor.h>
#include <ATen/ceil_div.h>
#include <ATen/Dispatch.h>
#include <ATen/ExpandUtils.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/TensorOperators.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/Resize.h>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/cuda/CUDAUtils.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/arange.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/zeros_like.h>
#include <ATen/ops/ones_like.h>
#include <ATen/ops/empty_quantized.h>
#include <ATen/ops/index_add_native.h>
#include <ATen/ops/index_reduce_native.h>
#include <ATen/ops/index_select_native.h>
#include <ATen/ops/masked_fill_native.h>
#include <ATen/ops/_sparse_coo_tensor_with_dims_and_tensors.h>
#endif

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/cub.h>
#include <c10/util/irange.h>
#include <c10/core/QScheme.h>
#include <ATen/native/quantized/AffineQuantizerBase.h>

#include <limits>

#include <c10/macros/Macros.h>

#include <ATen/native/cuda/maca_kernels/Indexing_maca.cuh>
#include <ATen/native/cuda/maca_kernels/loop_utils.h>

#ifdef USE_MACA
  #include "maca_kernels/indexing_opt.cuh"
#else
  static_assert(0)
#endif

namespace {
template <typename scalar_t, int SZ>
__global__ void indexing_backward_kernel(
  int64_t* sorted_indices, int64_t* indices, scalar_t* grad_output, scalar_t* grad_weight,
  int64_t numel, int64_t stride, int64_t stride_before, int64_t outer_dim, bool accumulate) {
//numel is total number of flattened indices, not expanded to dimensions that are not indexed.
//stride is the cumulative size of the not-indexed last dimensions
//stride_before is the stride of the dimension immediately preceding first indexed dimension
//if indexing starts from the 0th dimension, stride_before does not matter because blockIdx.z will be 0 in this case
//outer_dim is number of elements in the first unindexed dimensions
  using opmath_t = at::opmath_type<scalar_t>;

  // Each warp is responsible for an input into the LookupTable.
  // If the preceding input has the same destination index as this input, then the warp
  // exits immediately. The warp also processes subsequent inputs with the
  // same value.
  //
  // Input Warp
  // 1     <warp 1>
  // 1     <warp 1> (<warp 2> exits without doing any work)
  // 5     <warp 3>
  // 8     <warp 4>

  // Number of values processed by each thread (grain size)
  for (int64_t z = blockIdx.z; z < outer_dim; z += gridDim.z){
    int64_t idx = blockIdx.x * blockDim.y + threadIdx.y;
    if (idx < numel
        && (idx == 0 || sorted_indices[idx] != sorted_indices[idx - 1])){
      do {
        int64_t start_feature = threadIdx.x + blockIdx.y * blockDim.x * SZ;
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

        while (start_feature < stride) {
          #pragma unroll
          for (int ii = 0; ii < SZ; ii++) {
            int64_t feature_dim = start_feature + ii * C10_WARP_SIZE;
            if (feature_dim < stride) {
              gradient[ii] = static_cast<opmath_t>(grad_output[grad_row + feature_dim]);
              if (accumulate) {
                weight[ii] = static_cast<opmath_t>(grad_weight[weight_row + feature_dim]);
              }
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
            int64_t feature_dim = start_feature + ii * C10_WARP_SIZE;
            if (feature_dim < stride) {
                grad_weight[weight_row + feature_dim] = static_cast<scalar_t>(weight[ii]);
            }
          }
          start_feature += gridDim.y * blockDim.x * SZ;
        }

        idx++;
      } while (idx < numel && sorted_indices[idx] == sorted_indices[idx - 1]);
    }
  }
}

template <typename scalar_t, int SZ>
__global__ void indexing_backward_kernel_quantized(
  int64_t* sorted_indices, int64_t* indices, float* grad_output, scalar_t* grad_weight,
  int64_t numel, int64_t stride, int64_t stride_before, int64_t outer_dim,
  float inv_scale, int zero_point, int64_t qmin, int64_t qmax) {

  // This implementation is adopted from indexing_backward_kernel above.
  using opmath_t = at::opmath_type<float>;
  for (int64_t z = blockIdx.z; z < outer_dim; z += gridDim.z){
    int64_t idx = blockIdx.x * blockDim.y + threadIdx.y;
    if (idx < numel
        && (idx == 0 || sorted_indices[idx] != sorted_indices[idx - 1])){
      do {
        int64_t start_feature = threadIdx.x + blockIdx.y * blockDim.x * SZ;
        // we only keep the last duplicate index so skip those before it
        if ((idx < numel - 1) && sorted_indices[idx] == sorted_indices[idx + 1]) {
          idx++;
          continue;
        }
        const int64_t weight_row = ((int64_t) sorted_indices[idx]) * stride + z * stride_before;
        const int64_t grad_row = ((int64_t) indices[idx]) * stride + z * numel * stride;
        const opmath_t scale = (opmath_t)1.0;

        opmath_t gradient[SZ];
        opmath_t weight[SZ];

        while (start_feature < stride) {
          #pragma unroll
          for (int ii = 0; ii < SZ; ii++) {
            int64_t feature_dim = start_feature + ii * C10_WARP_SIZE;
            if (feature_dim < stride) {
              gradient[ii] = static_cast<opmath_t>(grad_output[grad_row + feature_dim]);
            }
          }

          #pragma unroll
          for (int ii = 0; ii < SZ; ii++) {
            weight[ii] = gradient[ii] * scale;
          }

          #pragma unroll
          for (int ii = 0; ii < SZ; ii++) {
            int64_t feature_dim = start_feature + ii * C10_WARP_SIZE;
            if (feature_dim < stride) {
                // we do quantization here
                int64_t qvalue = static_cast<int64_t>(zero_point + nearbyintf(weight[ii]* inv_scale));
                qvalue = min(max(qvalue, qmin), qmax);
                grad_weight[weight_row + feature_dim] = static_cast<scalar_t>(qvalue);
            }
          }
          start_feature += gridDim.y * blockDim.x * SZ;
        }

        idx++;
      } while (idx < numel && sorted_indices[idx] == sorted_indices[idx - 1]);
    }
  }
}


}


namespace at::native {

namespace {

class ReduceMultiply {
public:
  template <typename scalar_t>
  constexpr C10_DEVICE void operator() (scalar_t* self_data_start, int64_t index, int64_t numel, const scalar_t * src_data) const {
    (void)numel; // suppress unused warning
    gpuAtomicMul(self_data_start + index, *src_data);
  }
};
static ReduceMultiply reduce_multiply;

class ReduceAdd {
public:
  template <typename scalar_t>
  constexpr C10_DEVICE void operator() (scalar_t* self_data_start, int64_t index, int64_t numel, const scalar_t * src_data) const {
    fastAtomicAdd(self_data_start, index, numel, *src_data, true);
  }
};
static ReduceAdd reduce_add;

class ReduceMinimum {
public:
  template <typename scalar_t>
  constexpr C10_DEVICE void operator() (scalar_t* self_data_start, int64_t index, int64_t numel, const scalar_t * src_data) const {
    (void)numel; // suppress unused warning
    gpuAtomicMin(self_data_start + index, *src_data);
  }
};
static ReduceMinimum reduce_minimum;

class ReduceMaximum {
public:
  template <typename scalar_t>
  constexpr C10_DEVICE void operator() (scalar_t* self_data_start, int64_t index, int64_t numel, const scalar_t * src_data) const {
    (void)numel; // suppress unused warning
    gpuAtomicMax(self_data_start + index, *src_data);
  }
};
static ReduceMaximum reduce_maximum;

}

static Tensor wrapIndexOnce(const Tensor & index, int64_t dim, int64_t dim_size, bool check_range=true) {
//we don't need to check range in backward - if there were out of bounds indices forward should already have errored out
  if (index.numel() != 0 && check_range) {
    auto max_idx = index.max().item<int64_t>();
    auto min_idx = index.min().item<int64_t>();
    if (max_idx >= dim_size) {
      TORCH_CHECK_INDEX(false, "index ", max_idx, " is out of bounds for dimension ", dim, " with size ", dim_size);
    }
    if (min_idx < -dim_size) {
      TORCH_CHECK_INDEX(false, "index ", min_idx, " is out of bounds for dimension ", dim, " with size ", dim_size);
    }
  }
  return index.remainder(dim_size);
}

static std::vector<int64_t> computeLinearStride(const Tensor & tensor) {
  // computes the stride as if tensor were contiguous
  auto sizes = tensor.sizes();
  std::vector<int64_t> stride(tensor.dim());
  stride[tensor.dim() - 1] = 1;
  std::partial_sum(sizes.rbegin(), sizes.rend() - 1, stride.rbegin() + 1, std::multiplies<int64_t>());
  return stride;
}

static std::tuple<Tensor, int64_t, int64_t, int64_t>
computeLinearIndex(const Tensor & src, TensorList indices, bool check_range) {
  auto strides = computeLinearStride(src);
  const auto& device = src.options().device();

  // Compute the linear index by multiplying the indexing tensors by the
  // stride and summing them. All the indexing tensors have the same shape at
  // this point. We also compute the number of dimensions before and after that
  // are not being index.
  Tensor linearIndex;
  int64_t emptyBefore = 0, emptyAfter = 0, nElemBefore = 1, nElemAfter = 1, strideBefore =0;
  for (const auto i: c10::irange(src.dim())) {
    if (indices[i].defined()) {
      // Cast index to the longType matching src's device
      // This allows us to support ie indexing a cuda tensor with a cpu tensor
      Tensor index = (wrapIndexOnce(indices[i], i, src.size(i), check_range) * strides[i]).to(device);
      if (linearIndex.defined()) {
        linearIndex += index;
      } else {
        linearIndex = index;
        if (i>0) {
           strideBefore = src.stride(i-1); // stride after undefined dimensions
        }
      }
    } else if (linearIndex.defined()) {
      emptyAfter++;
      nElemAfter *= src.size(i);
    } else {
      emptyBefore++;
      nElemBefore *= src.size(i);
    }
  }

  return std::make_tuple(std::move(linearIndex), nElemBefore, strideBefore, nElemAfter);
}


static std::tuple<Tensor, Tensor, int64_t, int64_t, int64_t, std::vector<int64_t>> makeLinearIndex(Tensor self, IOptTensorListRef orig, bool check_range) {
  checkIndexTensorTypes(orig, /*allow_int*/true);
  // first expand BoolTensor (masks) or ByteTensor (masks) into 1 or more LongTensors
  auto indices = expandTensors(self, orig);
  for (auto & i : indices) {
    if (i.defined() && i.dtype() == at::kInt) {
      i = i.to(at::kLong);
    }
  }
  // next broadcast all index tensors together
  indices = expand_outplace(indices);
  // add missing null Tensors so that it matches self.dim()
  while (indices.size() < (size_t)self.dim()) {
    indices.emplace_back();
  }
  // if the non-null indices are not all adjacent, transpose self and indices
  // together so that they're adjacent at the front
  std::vector<int64_t> inversePerm;
  if (!hasContiguousSubspace(indices)) {
    std::tie(self, indices, inversePerm) = transposeToFrontAndInvPerm(self, indices);
  }
  int64_t nElemBefore, strideBefore, nElemAfter;
  Tensor linearIndex;
  std::tie(linearIndex, nElemBefore, strideBefore, nElemAfter) = computeLinearIndex(self, indices, check_range);
  return std::make_tuple(linearIndex, self, nElemBefore, strideBefore, nElemAfter, inversePerm);
}


void index_put_with_sort_kernel_thrust_helper(Tensor &linearIndex, Tensor &orig_indices, Tensor &sorted_indices, int64_t num_indices);

namespace {

int64_t largestIndex(const Tensor &self) {
  int64_t result = 0;
  for (const auto i: c10::irange(self.dim())) {
    result += (self.sizes()[i] - 1) * self.strides()[i];
  }
  return result;
}

void index_put_with_sort_kernel(Tensor & self, const c10::List<c10::optional<Tensor>>& indices, const Tensor & value, bool accumulate, bool unsafe) {
  if (indices.size() > (size_t)self.dim()) {
    TORCH_CHECK_INDEX(false, "too many indices for tensor of dimension ", self.dim(), " (got ", indices.size(), ")");
  }
  bool self_contiguous = self.is_contiguous();
  auto self_ = self_contiguous ? self : self.contiguous();
  Tensor linearIndex, src, expandedValue = value;
  int64_t nElemBefore, strideBefore, sliceSize;
  std::vector<int64_t> inversePerm;
  std::tie(linearIndex, src, nElemBefore, strideBefore, sliceSize, inversePerm) = makeLinearIndex(self_, indices, !unsafe);
  int64_t num_indices = linearIndex.numel();

  if (expandedValue.numel() < num_indices * nElemBefore * sliceSize) {
    auto expanded_size = at::DimVector(expandedValue.sizes());
    auto size1 = expandedValue.sizes();
    auto size2 = linearIndex.sizes();
    if (are_expandable(size1, size2)) {
      expanded_size = infer_size_dimvector(size1, size2);
    }
    if (nElemBefore > 1) {
      expanded_size.insert(expanded_size.begin(), nElemBefore);
    }
    expandedValue = expandedValue.expand(expanded_size);
  }
  expandedValue = expandedValue.contiguous();

  if (num_indices > 0 && sliceSize > 0) {
      const bool permuted = !src.is_contiguous();
      auto src_ = permuted ? src.contiguous() : src;
      linearIndex = linearIndex.reshape(-1);
      auto sorted_indices = at::empty_like(linearIndex, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
      auto orig_indices = at::empty_like(linearIndex, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
      const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

      linearIndex.divide_(sliceSize, "trunc");

      // cub on CUDA <= 11.2 have a bug that for small sizes
      // cub's sort can be much slower than thrust's merge sort
      // this bug is fixed in CUDA 11.3
#if (defined(CUDA_VERSION) && CUDA_VERSION < 11030) || defined(USE_ROCM)
      if (num_indices < 50000) {
        index_put_with_sort_kernel_thrust_helper(linearIndex, orig_indices, sorted_indices, num_indices);
      } else
#endif
      {
      // Sort the inputs into sorted with the corresponding indices
      auto range = at::arange(num_indices, linearIndex.options());
      // linearIndex can not be negative, and we take advantage of this
      // fact to sort on less bits for better performance.
      int64_t nbits = cuda::cub::get_num_bits(largestIndex(self_) / sliceSize);
      cuda::cub::radix_sort_pairs(
        linearIndex.data_ptr<int64_t>(), sorted_indices.data_ptr<int64_t>(),
        range.data_ptr<int64_t>(), orig_indices.data_ptr<int64_t>(),
        num_indices, false, 0, nbits);
      }

      #ifdef USE_MACA
        if(maca_unlikely(at::maca::get_maca_enable_print_indexing_backward_kernel())){
          print_indexing_backward_kernel(src_, expandedValue, num_indices, sliceSize, strideBefore, nElemBefore, accumulate);
        }
      #endif

      TORCH_INTERNAL_ASSERT(
          linearIndex.numel()*sliceSize*nElemBefore == expandedValue.numel(),
          "number of flattened indices did not match number of elements in the value tensor: ",
          linearIndex.numel()*sliceSize*nElemBefore, " vs ", expandedValue.numel());

      bool is_opt = at::maca::get_maca_enable_indexing_backward_kernel_opt() &&
                    src_.dim() == 2 && src_.sizes()[1] == sliceSize &&
                    (expandedValue.scalar_type()==ScalarType::BFloat16 || expandedValue.scalar_type()==ScalarType::Half) &&
                    sliceSize % 4 == 0;

      bool is_opt1 = (!at::maca::get_maca_disable_indexing_backward_kernel_opt1()) &&
                     src_.dim() == 2 && src_.sizes()[1] == sliceSize &&
                     sliceSize % 4 == 0 && sliceSize >= 96 &&
                     num_indices >= 3328;

      if (is_opt) {
        // in some case, adjust indices_per_block and warp_size maybe cause performance fluctuation
        const int indices_per_block = 1;
        const int warp_size = at::cuda::warp_size();
        int load_num = sliceSize / warp_size;
        int UNROLL = getVectorizedAlignment<at::BFloat16>(src_.data_ptr(), load_num);
        dim3 grid(ceil_div(num_indices, (int64_t) indices_per_block),
            std::min<int>(at::cuda::getCurrentDeviceProperties()->maxGridSize[1], ceil_div(sliceSize, (int64_t) (warp_size*UNROLL))),
            std::min(std::max<int>(1,nElemBefore), at::cuda::getCurrentDeviceProperties()->maxGridSize[2]));
        dim3 block(warp_size, indices_per_block);
        AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND4(kComplexHalf, kHalf, kBool, kBFloat16,
        expandedValue.scalar_type(), "indexing_backward", [&] {
          if (UNROLL == 8) {
            indexing_backward_kernel_opt<scalar_t, 8><<<grid, block, 0, stream>>>(
              sorted_indices.data_ptr<int64_t>(),
              orig_indices.data_ptr<int64_t>(),
              expandedValue.data_ptr<scalar_t>(),
              src_.data_ptr<scalar_t>(),
              num_indices,
              sliceSize,
              strideBefore,
              nElemBefore,
              accumulate);
          } else if (UNROLL == 4) {
            indexing_backward_kernel_opt<scalar_t, 4><<<grid, block, 0, stream>>>(
              sorted_indices.data_ptr<int64_t>(),
              orig_indices.data_ptr<int64_t>(),
              expandedValue.data_ptr<scalar_t>(),
              src_.data_ptr<scalar_t>(),
              num_indices,
              sliceSize,
              strideBefore,
              nElemBefore,
              accumulate);
          } else if (UNROLL == 2) {
            indexing_backward_kernel_opt<scalar_t, 2><<<grid, block, 0, stream>>>(
              sorted_indices.data_ptr<int64_t>(),
              orig_indices.data_ptr<int64_t>(),
              expandedValue.data_ptr<scalar_t>(),
              src_.data_ptr<scalar_t>(),
              num_indices,
              sliceSize,
              strideBefore,
              nElemBefore,
              accumulate);
          } else {
            indexing_backward_kernel_opt<scalar_t, 1><<<grid, block, 0, stream>>>(
              sorted_indices.data_ptr<int64_t>(),
              orig_indices.data_ptr<int64_t>(),
              expandedValue.data_ptr<scalar_t>(),
              src_.data_ptr<scalar_t>(),
              num_indices,
              sliceSize,
              strideBefore,
              nElemBefore,
              accumulate);
          }
          C10_CUDA_KERNEL_LAUNCH_CHECK();
        });
      } else if (is_opt1) {

        int unroll = 4;
        if (sliceSize < 192 ) unroll = 2;
        const int indices_per_block = 1;
        const int warp_size = at::cuda::warp_size();

        dim3 block(warp_size, indices_per_block);
        dim3 grid(ceil_div(num_indices, (int64_t) indices_per_block),
                  std::min<int>(at::cuda::getCurrentDeviceProperties()->maxGridSize[1], ceil_div(sliceSize, (int64_t) (warp_size*unroll))),
                  std::min(std::max<int>(1,nElemBefore), at::cuda::getCurrentDeviceProperties()->maxGridSize[2]));

        if (unroll == 2) {
          AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND4(kComplexHalf, kHalf, kBool, kBFloat16,
          expandedValue.scalar_type(), "indexing_backward", [&] {
            indexing_backward_kernel_opt1<scalar_t, 2><<<grid, block, 0, stream>>>(
              sorted_indices.data_ptr<int64_t>(),
              orig_indices.data_ptr<int64_t>(),
              expandedValue.data_ptr<scalar_t>(),
              src_.data_ptr<scalar_t>(),
              num_indices,
              sliceSize,
              strideBefore,
              nElemBefore,
              accumulate);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
          });
        } else if (unroll == 4) {
          AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND4(kComplexHalf, kHalf, kBool, kBFloat16,
          expandedValue.scalar_type(), "indexing_backward", [&] {
            indexing_backward_kernel_opt1<scalar_t, 4><<<grid, block, 0, stream>>>(
              sorted_indices.data_ptr<int64_t>(),
              orig_indices.data_ptr<int64_t>(),
              expandedValue.data_ptr<scalar_t>(),
              src_.data_ptr<scalar_t>(),
              num_indices,
              sliceSize,
              strideBefore,
              nElemBefore,
              accumulate);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
          });
        } else {
          assert(0);
        }
      } else {
        const int UNROLL = 4;
        const int indices_per_block = 4;
        const int warp_size = at::cuda::warp_size();
        dim3 grid(ceil_div(num_indices, (int64_t) indices_per_block),
            std::min<int>(at::cuda::getCurrentDeviceProperties()->maxGridSize[1], ceil_div(sliceSize, (int64_t) (warp_size*UNROLL))),
            std::min(std::max<int>(1,nElemBefore), at::cuda::getCurrentDeviceProperties()->maxGridSize[2]));
        dim3 block(warp_size, indices_per_block);
        AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND4(kComplexHalf, kHalf, kBool, kBFloat16,
        expandedValue.scalar_type(), "indexing_backward", [&] {
          indexing_backward_kernel<scalar_t, UNROLL><<<grid, block, 0, stream>>>(
            sorted_indices.data_ptr<int64_t>(),
            orig_indices.data_ptr<int64_t>(),
            expandedValue.data_ptr<scalar_t>(),
            src_.data_ptr<scalar_t>(),
            num_indices,
            sliceSize,
            strideBefore,
            nElemBefore,
            accumulate);
          C10_CUDA_KERNEL_LAUNCH_CHECK();
        });
      }

      if (permuted) {
        self.copy_(src_.permute(inversePerm));
      } else if (!self_contiguous) {
        self.copy_(self_);
      }
  }
}

REGISTER_CUDA_DISPATCH(index_put_with_sort_stub, &index_put_with_sort_kernel);

void index_put_with_sort_quantized(Tensor & self, const c10::List<c10::optional<Tensor>>& indices, const Tensor & value, double scale, int zero_point, bool unsafe) {
  if (indices.size() > (size_t)self.dim()) {
    TORCH_CHECK_INDEX(false, "too many indices for tensor of dimension ", self.dim(), " (got ", indices.size(), ")");
  }
  bool self_contiguous = self.is_contiguous();
  auto self_ = self_contiguous ? self : self.contiguous();
  Tensor linearIndex, src, expandedValue = value;
  int64_t nElemBefore, strideBefore, sliceSize;
  std::vector<int64_t> inversePerm;
  std::tie(linearIndex, src, nElemBefore, strideBefore, sliceSize, inversePerm) = makeLinearIndex(self_, indices, !unsafe);
  int64_t num_indices = linearIndex.numel();

  if (expandedValue.numel() < num_indices * nElemBefore * sliceSize) {
    auto expanded_size = at::DimVector(expandedValue.sizes());
    auto size1 = expandedValue.sizes();
    auto size2 = linearIndex.sizes();
    if (are_expandable(size1, size2)) {
      expanded_size = infer_size_dimvector(size1, size2);
    }
    if (nElemBefore > 1) {
      expanded_size.insert(expanded_size.begin(), nElemBefore);
    }
    expandedValue = expandedValue.expand(expanded_size);
  }
  expandedValue = expandedValue.contiguous();

  if (num_indices > 0 && sliceSize > 0) {
      const bool permuted = !src.is_contiguous();
      auto src_ = permuted ? src.contiguous() : src;
      linearIndex = linearIndex.reshape(-1);
      auto sorted_indices = at::empty_like(linearIndex, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
      auto orig_indices = at::empty_like(linearIndex, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
      const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

      linearIndex.divide_(sliceSize, "trunc");

      // cub on CUDA <= 11.2 have a bug that for small sizes
      // cub's sort can be much slower than thrust's merge sort
      // this bug is fixed in CUDA 11.3
#if (defined(CUDA_VERSION) && CUDA_VERSION < 11030) || defined(USE_ROCM)
      if (num_indices < 50000) {
        index_put_with_sort_kernel_thrust_helper(linearIndex, orig_indices, sorted_indices, num_indices);
      } else
#endif
      {
      // Sort the inputs into sorted with the corresponding indices
      auto range = at::arange(num_indices, linearIndex.options());
      // linearIndex can not be negative, and we take advantage of this
      // fact to sort on less bits for better performance.
      int64_t nbits = cuda::cub::get_num_bits(largestIndex(self_) / sliceSize);
      cuda::cub::radix_sort_pairs(
        linearIndex.data_ptr<int64_t>(), sorted_indices.data_ptr<int64_t>(),
        range.data_ptr<int64_t>(), orig_indices.data_ptr<int64_t>(),
        num_indices, false, 0, nbits);
      }

      TORCH_INTERNAL_ASSERT(
          linearIndex.numel()*sliceSize*nElemBefore == expandedValue.numel(),
          "number of flattened indices did not match number of elements in the value tensor: ",
          linearIndex.numel()*sliceSize*nElemBefore, " vs ", expandedValue.numel());
      const int UNROLL = 4;
      const int indices_per_block = 4;
      const int warp_size = at::cuda::warp_size();
      dim3 grid(ceil_div(num_indices, (int64_t) indices_per_block),
           std::min<int>(at::cuda::getCurrentDeviceProperties()->maxGridSize[1], ceil_div(sliceSize, (int64_t) (warp_size*UNROLL))),
           std::min(std::max<int>(1,nElemBefore), at::cuda::getCurrentDeviceProperties()->maxGridSize[2]));
      dim3 block(warp_size, indices_per_block);

      AT_DISPATCH_QINT_TYPES(
        src.scalar_type(), "indexing_backward_quantized", [&] {
        constexpr int64_t qmin = std::numeric_limits<typename scalar_t::underlying>::min();
        constexpr int64_t qmax = std::numeric_limits<typename scalar_t::underlying>::max();
        float inv_scale = 1.0f / static_cast<float>(scale);

        indexing_backward_kernel_quantized<scalar_t, UNROLL><<<grid, block, 0, stream>>>(
          sorted_indices.data_ptr<int64_t>(),
          orig_indices.data_ptr<int64_t>(),
          expandedValue.data_ptr<float>(),
          src_.data_ptr<scalar_t>(),
          num_indices,
          sliceSize,
          strideBefore,
          nElemBefore,
          inv_scale,
          zero_point,
          qmin,
          qmax);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });

      if (permuted) {
        self.copy_(src_.permute(inversePerm));
      } else if (!self_contiguous) {
        self.copy_(self_);
      }
  }
}

REGISTER_CUDA_DISPATCH(index_put_with_sort_quantized_stub, &index_put_with_sort_quantized);
} //anonymous


// Check tensor dimensions for index operations, and return the slice size.
static ptrdiff_t getSliceSize(const Tensor & dst,
                              int dim,
                              const Tensor & index,
                              const Tensor & src)
{
  const auto dstDims = dst.dim();
  const auto srcDims = src.dim();

  TORCH_CHECK(index.dim() <= 1, "Index must be vector or scalar");

  ptrdiff_t dstSliceSize = 1;
  TORCH_CHECK(dim >= 0 && dim < dstDims, "Indexing dim ", dim, " is out of bounds");
  for (const auto d: c10::irange(dstDims)) {
    if (d != dim) {
      dstSliceSize *= dst.size(d);
    }
  }

  TORCH_CHECK(dim < srcDims, "Indexing dim ", dim, " is out of bounds");
  TORCH_CHECK(index.numel() == src.size(dim),
             "length of src.size[dim] is not equal to length of indices");

  ptrdiff_t srcSliceSize = 1;
  bool mismatch = false;

  if (dstDims != srcDims) mismatch = true;

  for (const auto d: c10::irange(srcDims)) {
    if (d != dim) {
      srcSliceSize *= src.size(d);
      if (!mismatch && dst.size(d) != src.size(d)) mismatch = true;
    }
  }

  TORCH_CHECK(dstSliceSize == srcSliceSize,
             "Source/destination tensor have different slice sizes (%ld vs %ld)",
             dstSliceSize, srcSliceSize);

  if (mismatch) {
    TORCH_WARN_ONCE(
        "Warning: source/destination slices have same size but different "
        "shape for an index operation.  This behavior is deprecated.\n");
  }

  return dstSliceSize;
}

// We prefer this kernel to avoid reloading index points if the number
// of indices is a small number.
// This kernel in fact works for all choices of problem size, but if
// the number of indices chosen is large, then the
// indexFuncLargeIndex kernel is a better choice to increase
// parallelism.
template <typename T, typename IndicesType, typename IndexType, int DstDim, int SrcDim, int IdxDim,
          typename func_t>
__global__ void indexFuncSmallIndex(cuda::detail::TensorInfo<T, IndexType> dst,
                                    cuda::detail::TensorInfo<T, IndexType> src,
                                    cuda::detail::TensorInfo<IndicesType, IndexType> indices,
                                    int dstAddDim,
                                    int srcAddDim,
                                    IndexType innerSize,
                                    int64_t dstAddDimSize,
                                    int64_t dstNumel,
                                    const func_t& op,
                                    T alpha) {
  // In order to avoid reloading the index that we are copying, load
  // it once to handle all of the points that are being selected, so
  // it can be reused as much as possible. This kernel is chosen when
  // this is a good choice (small number of chosen indices), since
  // re-accessing indices in addition to src elements can be slow.
  for (IndexType srcIndex = 0; srcIndex < indices.sizes[0]; ++srcIndex) {
    // Lua indices begin at 1
    IndexType dstIndex =
        indices.data[cuda::detail::IndexToOffset<IndicesType, IndexType, IdxDim>::get(srcIndex, indices)];
    CUDA_KERNEL_ASSERT(dstIndex < dstAddDimSize);

    // We stride over the output ignoring the indexed dimension
    // (innerSize), whose offset calculation is handled differently
    for (IndexType linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
         linearIndex < innerSize;
         linearIndex += gridDim.x * blockDim.x) {
      IndexType dstOffset =
          cuda::detail::IndexToOffset<T, IndexType, DstDim>::get(linearIndex, dst);
      dstOffset += dstIndex * dst.strides[dstAddDim];

      IndexType srcOffset =
          cuda::detail::IndexToOffset<T, IndexType, SrcDim>::get(linearIndex, src);
      srcOffset += srcIndex * src.strides[srcAddDim];

      T val = src.data[srcOffset] * alpha;
      op(dst.data, dstOffset, dstNumel, &val);
    }

  }
}

template <typename T, typename IndicesType, typename IndexType, int DstDim, int SrcDim, int IdxDim,
          typename func_t>
__global__ void indexFuncSmallIndexWithoutAssert(cuda::detail::TensorInfo<T, IndexType> dst,
                                    cuda::detail::TensorInfo<T, IndexType> src,
                                    cuda::detail::TensorInfo<IndicesType, IndexType> indices,
                                    int dstAddDim,
                                    int srcAddDim,
                                    IndexType innerSize,
                                    int64_t dstAddDimSize,
                                    int64_t dstNumel,
                                    const func_t& op,
                                    T alpha) {
  // In order to avoid reloading the index that we are copying, load
  // it once to handle all of the points that are being selected, so
  // it can be reused as much as possible. This kernel is chosen when
  // this is a good choice (small number of chosen indices), since
  // re-accessing indices in addition to src elements can be slow.
  for (IndexType srcIndex = 0; srcIndex < indices.sizes[0]; ++srcIndex) {
    // Lua indices begin at 1
    IndexType dstIndex =
        indices.data[cuda::detail::IndexToOffset<IndicesType, IndexType, IdxDim>::get(srcIndex, indices)];
    // CUDA_KERNEL_ASSERT(dstIndex < dstAddDimSize);

    // We stride over the output ignoring the indexed dimension
    // (innerSize), whose offset calculation is handled differently
    for (IndexType linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
         linearIndex < innerSize;
         linearIndex += gridDim.x * blockDim.x) {
      IndexType dstOffset =
          cuda::detail::IndexToOffset<T, IndexType, DstDim>::get(linearIndex, dst);
      dstOffset += dstIndex * dst.strides[dstAddDim];

      IndexType srcOffset =
          cuda::detail::IndexToOffset<T, IndexType, SrcDim>::get(linearIndex, src);
      srcOffset += srcIndex * src.strides[srcAddDim];

      T val = src.data[srcOffset] * alpha;
      op(dst.data, dstOffset, dstNumel, &val);
    }

  }
}

// We prefer this kernel to balance parallelism across index points,
// if there are a large number of indices.
// This kernel in fact works for all choices of problem size, but if
// the number of indices chosen is small, then the
// indexFuncSmallIndex kernel is a better choice to reduce memory
// accesses.
template <typename T, typename IndicesType, typename IndexType, int DstDim, int SrcDim, int IdxDim,
          bool IndexIsMajor, typename func_t>
__global__ void indexFuncLargeIndex(cuda::detail::TensorInfo<T, IndexType> dst,
                                    cuda::detail::TensorInfo<T, IndexType> src,
                                    cuda::detail::TensorInfo<IndicesType, IndexType> indices,
                                    int dstAddDim,
                                    int srcAddDim,
                                    IndexType totalSize,
                                    IndexType innerSize,
                                    int64_t dstAddDimSize,
                                    int64_t dstNumel,
                                    const func_t& op,
                                    T alpha) {
  // We stride over the output including the indexed dimension
  // (totalSize), and calculate the destination index point based on that
  for (IndexType linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
       linearIndex < totalSize;
       linearIndex += gridDim.x * blockDim.x) {
    IndexType srcIndex, elementInSlice;
    if (IndexIsMajor) {
      srcIndex = linearIndex / innerSize;
      elementInSlice = linearIndex % innerSize;
    }
    else {
      elementInSlice = linearIndex / innerSize;
      srcIndex = linearIndex % innerSize;
    }

    // Lua indices begin at 1
    IndexType dstIndex =
        indices.data[cuda::detail::IndexToOffset<IndicesType, IndexType, IdxDim>::get(srcIndex, indices)];
    CUDA_KERNEL_ASSERT(dstIndex < dstAddDimSize);

    IndexType dstOffset =
      cuda::detail::IndexToOffset<T, IndexType, DstDim>::get(elementInSlice, dst);
    dstOffset += dstIndex * dst.strides[dstAddDim];

    IndexType srcOffset =
      cuda::detail::IndexToOffset<T, IndexType, SrcDim>::get(elementInSlice, src);
    srcOffset += srcIndex * src.strides[srcAddDim];

    T val = src.data[srcOffset] * alpha;
    op(dst.data, dstOffset, dstNumel, &val);
  }
}

template <typename T, typename IndicesType, typename IndexType, int DstDim, int SrcDim, int IdxDim,
          bool IndexIsMajor, typename func_t>
__global__ void indexFuncLargeIndexWithoutAssert(cuda::detail::TensorInfo<T, IndexType> dst,
                                    cuda::detail::TensorInfo<T, IndexType> src,
                                    cuda::detail::TensorInfo<IndicesType, IndexType> indices,
                                    int dstAddDim,
                                    int srcAddDim,
                                    IndexType totalSize,
                                    IndexType innerSize,
                                    int64_t dstAddDimSize,
                                    int64_t dstNumel,
                                    const func_t& op,
                                    T alpha) {
  // We stride over the output including the indexed dimension
  // (totalSize), and calculate the destination index point based on that
  for (IndexType linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
       linearIndex < totalSize;
       linearIndex += gridDim.x * blockDim.x) {
    IndexType srcIndex, elementInSlice;
    if (IndexIsMajor) {
      srcIndex = linearIndex / innerSize;
      elementInSlice = linearIndex % innerSize;
    }
    else {
      elementInSlice = linearIndex / innerSize;
      srcIndex = linearIndex % innerSize;
    }

    // Lua indices begin at 1
    IndexType dstIndex =
        indices.data[cuda::detail::IndexToOffset<IndicesType, IndexType, IdxDim>::get(srcIndex, indices)];
    // CUDA_KERNEL_ASSERT(dstIndex < dstAddDimSize);

    IndexType dstOffset =
      cuda::detail::IndexToOffset<T, IndexType, DstDim>::get(elementInSlice, dst);
    dstOffset += dstIndex * dst.strides[dstAddDim];

    IndexType srcOffset =
      cuda::detail::IndexToOffset<T, IndexType, SrcDim>::get(elementInSlice, src);
    srcOffset += srcIndex * src.strides[srcAddDim];

    T val = src.data[srcOffset] * alpha;
    op(dst.data, dstOffset, dstNumel, &val);
  }
}

namespace {

// Compare the stride between adjacent slices (sliceStride) with strides in the
// other dimensions (i.e., strides *inside* each slice).
//
// - Returns true if some dimension inside the slice has lower stride than
//   sliceStride.  The simplest example is a 2-D contiguous tensor with sliceDim
//   == 0 (that is, each slice is a row).
//
//   In this case, we choose the CUDA kernel that processes the data in
//   "index-major order".  For example, if thread count equals slice size, then
//   all threads process slice #0 in lockstep, and then slice #1, and so on.
//
// - Otherwise (i.e., sliceStride has the lowest value), this function returns
//   false.  The simplest example is a 2-D contiguous tensor with sliceDim == 1
//   (each slice is a column).
//
//   In this case, we choose the CUDA kernel that processes the data in
//   "elementInSlice-major order".  For example, each thread can process element
//   #0 of every slice, and then element #1 of every slice, and so on.

template <typename scalar_t>
bool indexShouldBeMajor(cuda::detail::TensorInfo<scalar_t, unsigned int> &info,
                                    int sliceDim)
{
  // The stride between adjacent slices (e.g., between element #0 of slice #100
  // and element #0 of slice #101).
  unsigned int sliceStride = info.strides[sliceDim];

  for (const auto i: c10::irange(info.dims)) {
    if (i != sliceDim && info.sizes[i] > 1 && info.strides[i] < sliceStride) {
      return true;
    }
  }

  return false;
}

}

void index_add_cuda_impl(const Tensor& self, int64_t dim, const Tensor& index, const Tensor& source, const Scalar& alpha, const Tensor& result) {
  if (!result.is_same(self)) {
    result.copy_(self);
  }

  // Scalars are treated as 1-d tensor
  const Tensor self_ = (result.dim() == 0) ? result.view(1) : result;
  const Tensor source_ = (source.dim() == 0) ? source.view(1) : source;

  TORCH_CHECK(result.dim() <= MAX_TENSORINFO_DIMS, "tensor has too many (>", MAX_TENSORINFO_DIMS, ") dims");
  TORCH_CHECK(source.dim() <= MAX_TENSORINFO_DIMS, "tensor has too many (>", MAX_TENSORINFO_DIMS, ") dims" );
  TORCH_CHECK(index.dim() <= MAX_TENSORINFO_DIMS, "tensor has too many (>", MAX_TENSORINFO_DIMS, ") dims");

  if (globalContext().deterministicAlgorithms()){
    torch::List<c10::optional<Tensor>> indices;
    indices.reserve(dim + 1);
    for (const auto i: c10::irange(dim)) {
      indices.emplace_back();
    }
    indices.emplace_back(index.to(at::kLong));
    result.index_put_(indices, source * alpha, true);
    return;
  }

  // The `source` is partitioned into two parts:
  // -the size of each slice we are indexing, which is the
  // total size of the tensor ignoring dimension `dim`;
  // -the number of index we are choosing, which is the total size
  // of the tensor `index`.
  const ptrdiff_t sliceSize = getSliceSize(self_, dim, index, source_);
  const ptrdiff_t sourceTotalSize = source.numel();
  const int64_t selfAddDimSize = self_.size(dim);
  const ptrdiff_t numIndex = index.numel();
  const int64_t selfNumel = self_.numel();

  if (sliceSize == 0) {
    return;
  }
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const bool indContig = index.is_contiguous();

  const int mpc = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;

#define SMALL_INDEX(TENSOR_TYPE, INDICES_TYPE, TYPE, SELF_DIM, SOURCE_DIM, IDX_DIM)     \
  indexFuncSmallIndex<TENSOR_TYPE, INDICES_TYPE, TYPE, SELF_DIM, SOURCE_DIM, IDX_DIM>   \
    <<<smallIndexGrid, smallIndexBlock, 0, stream>>>(                                   \
      selfInfo, sourceInfo, indexInfo,                                                  \
      selfAddDim, sourceAddDim, sliceSize, selfAddDimSize,                              \
      selfNumel, reduce_add, alpha_value);                                              \
  C10_CUDA_KERNEL_LAUNCH_CHECK();

#define LARGE_INDEX(TENSOR_TYPE, INDICES_TYPE, TYPE,                        \
                    SELF_DIM, SOURCE_DIM, IDX_DIM, IDX_IS_MAJOR)            \
  indexFuncLargeIndex<TENSOR_TYPE, INDICES_TYPE, TYPE,                      \
                      SELF_DIM, SOURCE_DIM, IDX_DIM, IDX_IS_MAJOR>          \
    <<<largeIndexGrid, largeIndexBlock, 0, stream>>>(                       \
      selfInfo, sourceInfo, indexInfo,                                      \
      selfAddDim, sourceAddDim, sourceTotalSize,                            \
      (IDX_IS_MAJOR) ? sliceSize : numIndex,                                \
      selfAddDimSize, selfNumel, reduce_add, alpha_value);                  \
  C10_CUDA_KERNEL_LAUNCH_CHECK();

#define SMALL_INDEX_WITHOUT_ASSERT(TENSOR_TYPE, INDICES_TYPE, TYPE, SELF_DIM, SOURCE_DIM, IDX_DIM)     \
  indexFuncSmallIndexWithoutAssert<TENSOR_TYPE, INDICES_TYPE, TYPE, SELF_DIM, SOURCE_DIM, IDX_DIM>   \
    <<<smallIndexGrid, smallIndexBlock, 0, stream>>>(                                   \
      selfInfo, sourceInfo, indexInfo,                                                  \
      selfAddDim, sourceAddDim, sliceSize, selfAddDimSize,                              \
      selfNumel, reduce_add, alpha_value);                                              \
  C10_CUDA_KERNEL_LAUNCH_CHECK();

#define LARGE_INDEX_WITHOUT_ASSERT(TENSOR_TYPE, INDICES_TYPE, TYPE,                        \
                    SELF_DIM, SOURCE_DIM, IDX_DIM, IDX_IS_MAJOR)            \
  indexFuncLargeIndexWithoutAssert<TENSOR_TYPE, INDICES_TYPE, TYPE,                      \
                      SELF_DIM, SOURCE_DIM, IDX_DIM, IDX_IS_MAJOR>          \
    <<<largeIndexGrid, largeIndexBlock, 0, stream>>>(                       \
      selfInfo, sourceInfo, indexInfo,                                      \
      selfAddDim, sourceAddDim, sourceTotalSize,                            \
      (IDX_IS_MAJOR) ? sliceSize : numIndex,                                \
      selfAddDimSize, selfNumel, reduce_add, alpha_value);                  \
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  const dim3 smallIndexGrid(std::min(ceil_div(sliceSize, (ptrdiff_t)128), (ptrdiff_t)(mpc * 8 * 4)));
  const dim3 smallIndexBlock(std::min(sliceSize, (ptrdiff_t)128));

  const dim3 largeIndexGrid(std::min(ceil_div(sourceTotalSize, (ptrdiff_t)128), (ptrdiff_t)(mpc * 8 * 4 * 2)));
  const dim3 largeIndexBlock(std::min(sourceTotalSize, (ptrdiff_t)128));

  bool enable_indexing_with_assert = at::maca::get_maca_enable_indexing_assert_kernel();
  if (cuda::detail::canUse32BitIndexMath(result) &&
      cuda::detail::canUse32BitIndexMath(source) &&
      cuda::detail::canUse32BitIndexMath(index)) {
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND4(at::ScalarType::Bool, at::ScalarType::Half, at::ScalarType::BFloat16, at::ScalarType::ComplexHalf, result.scalar_type(), "index_add", [&] {
      cuda::detail::TensorInfo<scalar_t, unsigned int> selfInfo =
          cuda::detail::getTensorInfo<scalar_t, unsigned int>(self_);
      const int selfAddDim = selfInfo.collapseDims(dim);
      selfInfo.reduceDim(selfAddDim);
      const auto alpha_value = alpha.to<scalar_t>();
      AT_DISPATCH_INDEX_TYPES(index.scalar_type(), "index_add_cuda_", [&] () {
        auto sourceInfo =
          cuda::detail::getTensorInfo<scalar_t, unsigned int>(source_);
        const int sourceAddDim = sourceInfo.collapseDims(dim);
        sourceInfo.reduceDim(sourceAddDim);

        auto indexInfo =
        cuda::detail::getTensorInfo<index_t, unsigned int>(index);
        indexInfo.collapseDims();

        // A reasonable choice for when to have each thread iterate over
        // index to choose
#ifdef USE_MACA
        bool disable_opt_indexing = at::maca::get_maca_disable_opt_indexing();
        if (numIndex <= 16) {
          if (selfInfo.dims == 1 && sourceInfo.dims == 1 && indContig) {
            if (maca_unlikely(enable_indexing_with_assert)) {
              SMALL_INDEX(scalar_t, index_t, unsigned int, 1, 1, -2);
            } else {
              SMALL_INDEX_WITHOUT_ASSERT(scalar_t, index_t, unsigned int, 1, 1, -2);
            }
          } else if (selfInfo.dims == 2 && sourceInfo.dims == 2 && indContig) {
            if (maca_unlikely(enable_indexing_with_assert)) {
              SMALL_INDEX(scalar_t, index_t, unsigned int, 2, 2, -2);
            } else {
              SMALL_INDEX_WITHOUT_ASSERT(scalar_t, index_t, unsigned int, 2, 2, -2);
            }
          } else if (selfInfo.dims == 3 && sourceInfo.dims == 3 && indContig) {
            if (maca_unlikely(enable_indexing_with_assert)) {
              SMALL_INDEX(scalar_t, index_t, unsigned int, 3, 3, -2);
            } else {
              SMALL_INDEX_WITHOUT_ASSERT(scalar_t, index_t, unsigned int, 3, 3, -2);
            }
          } else {
            if (maca_unlikely(enable_indexing_with_assert)) {
              SMALL_INDEX(scalar_t, index_t, unsigned int, -1, -1, -1);
            } else {
              SMALL_INDEX_WITHOUT_ASSERT(scalar_t, index_t, unsigned int, -1, -1, -1);
            }
          }
        } else {
          const bool indexIsMajor = indexShouldBeMajor(selfInfo, selfAddDim);
          bool opt_large = self_.dim() >= 2 && dim <= (self_.dim() -2 ) && index.dim() == 1 &&
                             self_.is_contiguous() && source.is_contiguous() && index.is_contiguous() &&
                             (self_.scalar_type() == ScalarType::BFloat16 || self_.scalar_type() == ScalarType::Half) &&
                             (self_.scalar_type() == source.scalar_type()) && 
                             selfInfo.dims == 3 && sourceInfo.dims == 3 && 
                             indContig && indexIsMajor && sourceTotalSize >= 4096 &&
                             selfInfo.sizes[0] == sourceInfo.sizes[0] && selfInfo.strides[0] == sourceInfo.strides[0] &&
                             selfInfo.sizes[1] == sourceInfo.sizes[1] && selfInfo.strides[1] == sourceInfo.strides[1] &&
                             selfInfo.sizes[2] == sourceInfo.sizes[2] && selfInfo.strides[2] == sourceInfo.strides[2];
          if (maca_likely(!disable_opt_indexing) && opt_large) {
              dim3 grid(std::min(ceil_div(sourceTotalSize, (ptrdiff_t)128), (ptrdiff_t)(mpc * 8 * 4 * 2)));
              dim3 block(std::min(sourceTotalSize, (ptrdiff_t)128));
              index::indexFuncLargeIndex_3_3_true_half<
                 scalar_t, index_t, unsigned int, 3, 3, -2, true>
                 <<<grid, block, 0, stream>>>(
                selfInfo.data, sourceInfo.data, indexInfo.data,
                selfInfo.sizes[0],     sourceInfo.sizes[0],     indexInfo.sizes[0],
                selfInfo.sizes[1],     sourceInfo.sizes[1],
                selfInfo.sizes[2],     sourceInfo.sizes[2],
                selfInfo.strides[0],   sourceInfo.strides[0],   indexInfo.strides[0],
                selfInfo.strides[1],   sourceInfo.strides[1],
                selfInfo.strides[2],   sourceInfo.strides[2],
                selfInfo.strides[selfAddDim],
                sourceInfo.strides[sourceAddDim],
                sourceTotalSize, 
                sliceSize,
                selfAddDimSize, 
                selfNumel,    
                reduce_add,      
                alpha_value);
          } else if (selfInfo.dims == 1 && sourceInfo.dims == 1 && indContig) {
            if (maca_unlikely(enable_indexing_with_assert)) {
              LARGE_INDEX(scalar_t, index_t, unsigned int, 1, 1, -2, true);
            } else {
              LARGE_INDEX_WITHOUT_ASSERT(scalar_t, index_t, unsigned int, 1, 1, -2, true);
            }
          } else if (selfInfo.dims == 2 && sourceInfo.dims == 2 && indContig) {
            if (indexIsMajor) {
              if (maca_unlikely(enable_indexing_with_assert)) {
                LARGE_INDEX(scalar_t, index_t, unsigned int, 2, 2, -2, true);
              } else {
                LARGE_INDEX_WITHOUT_ASSERT(scalar_t, index_t, unsigned int, 2, 2, -2, true);
              }
            } else {
              if (maca_unlikely(enable_indexing_with_assert)) {
                LARGE_INDEX(scalar_t, index_t, unsigned int, 2, 2, -2, false);
              } else {
                LARGE_INDEX_WITHOUT_ASSERT(scalar_t, index_t, unsigned int, 2, 2, -2, false);
              }
            }
          } else if (selfInfo.dims == 3 && sourceInfo.dims == 3 && indContig) {
            if (indexIsMajor) {
              if (maca_unlikely(enable_indexing_with_assert)) {
                LARGE_INDEX(scalar_t, index_t, unsigned int, 3, 3, -2, true);
              } else {
                LARGE_INDEX_WITHOUT_ASSERT(scalar_t, index_t, unsigned int, 3, 3, -2, true);
              }
            } else {
              if (maca_unlikely(enable_indexing_with_assert)) {
                LARGE_INDEX(scalar_t, index_t, unsigned int, 3, 3, -2, false);
              } else {
                LARGE_INDEX_WITHOUT_ASSERT(scalar_t, index_t, unsigned int, 3, 3, -2, false);
              }
            }
          } else {
            if (maca_unlikely(enable_indexing_with_assert)) {
              LARGE_INDEX(scalar_t, index_t, unsigned int, -1, -1, -1, true);
            } else {
              LARGE_INDEX_WITHOUT_ASSERT(scalar_t, index_t, unsigned int, -1, -1, -1, true);
            }
          }
        }
#else
  static_assert(0);
#endif
      });
    });
  } else {
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(at::ScalarType::Bool, at::ScalarType::Half, at::ScalarType::BFloat16, self.scalar_type(), "index_add", [&] {
      cuda::detail::TensorInfo<scalar_t, uint64_t> selfInfo =
        cuda::detail::getTensorInfo<scalar_t, uint64_t>(self_);
      const int selfAddDim = selfInfo.collapseDims(dim);
      selfInfo.reduceDim(selfAddDim);
      const auto alpha_value = alpha.to<scalar_t>();

      cuda::detail::TensorInfo<scalar_t, uint64_t> sourceInfo =
        cuda::detail::getTensorInfo<scalar_t, uint64_t>(source_);
      const int sourceAddDim = sourceInfo.collapseDims(dim);
      sourceInfo.reduceDim(sourceAddDim);

      AT_DISPATCH_INDEX_TYPES(index.scalar_type(), "index_add_cuda_", [&] () {
        cuda::detail::TensorInfo<index_t, uint64_t> indexInfo =
          cuda::detail::getTensorInfo<index_t, uint64_t>(index);
        indexInfo.collapseDims();
        if (maca_unlikely(enable_indexing_with_assert)) {
          LARGE_INDEX(scalar_t, index_t, uint64_t, -1, -1, -1, true);
        } else {
          LARGE_INDEX_WITHOUT_ASSERT(scalar_t, index_t, uint64_t, -1, -1, -1, true);
        }
      });
    });
  }

#undef SMALL_INDEX
#undef LARGE_INDEX
#undef SMALL_INDEX_WITHOUT_ASSERT
#undef LARGE_INDEX_WITHOUT_ASSERT
}

template <typename func_t>
void index_reduce_func_cuda_impl(
  const Tensor& self,
  int64_t dim,
  const Tensor& index,
  const Tensor& source,
  bool include_self,
  const ReductionType& reduce,
  const func_t& reduce_func,
  const Tensor& result) {
  globalContext().alertNotDeterministic("index_reduce_cuda");

  if (!result.is_same(self)) result.copy_(self);

  // Scalars are treated as 1-d tensor
  Tensor self_ = (result.dim() == 0) ? result.view(1) : result;
  Tensor source_ = (source.dim() == 0) ? source.view(1) : source;

  TORCH_CHECK(result.dim() <= MAX_TENSORINFO_DIMS, "tensor has too many (>", MAX_TENSORINFO_DIMS, ") dims");
  TORCH_CHECK(source.dim() <= MAX_TENSORINFO_DIMS, "tensor has too many (>", MAX_TENSORINFO_DIMS, ") dims" );
  TORCH_CHECK(index.dim() <= MAX_TENSORINFO_DIMS, "tensor has too many (>", MAX_TENSORINFO_DIMS, ") dims");

  if (!include_self) {
    AT_DISPATCH_ALL_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16,
      self.scalar_type(), "index_reduce_func_cuda_exclude_input_init", [&] {
      scalar_t init_val;
      switch (reduce) {
        case ReductionType::PROD:
          init_val = (scalar_t)1;
          break;
        case ReductionType::MAX:
          init_val = std::numeric_limits<scalar_t>::has_infinity ? -std::numeric_limits<scalar_t>::infinity()
                     : std::numeric_limits<scalar_t>::lowest();
          break;
        case ReductionType::MIN:
          init_val = std::numeric_limits<scalar_t>::has_infinity ? std::numeric_limits<scalar_t>::infinity()
                     : std::numeric_limits<scalar_t>::max();
          break;
        default:
          init_val = (scalar_t)0;
          break;
      }
      // index_fill_ requires index to be a LongTensor
      self_.index_fill_(dim, index.to(at::ScalarType::Long), init_val);
    });
  }

  // The `source` is partitioned into two parts:
  // -the size of each slice we are indexing, which is the
  // total size of the tensor ignoring dimension `dim`;
  // -the number of index we are choosing, which is the total size
  // of the tensor `index`.
  ptrdiff_t sliceSize = getSliceSize(self_, dim, index, source_);
  ptrdiff_t sourceTotalSize = source.numel();
  int64_t selfReduceDimSize = self_.size(dim);
  ptrdiff_t numIndex = index.numel();
  int64_t selfNumel = self_.numel();

  if (sliceSize == 0) {
    return;
  }
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  bool indContig = index.is_contiguous();

  int mpc = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;

#define SMALL_INDEX(TENSOR_TYPE, INDICES_TYPE, TYPE, SELF_DIM, SOURCE_DIM, IDX_DIM)                  \
  indexFuncSmallIndex<TENSOR_TYPE, INDICES_TYPE, TYPE, SELF_DIM, SOURCE_DIM, IDX_DIM>                \
    <<<smallIndexGrid, smallIndexBlock, 0, stream>>>(                                                \
      selfInfo, sourceInfo, indexInfo,                                                               \
      selfReduceDim, sourceReduceDim, sliceSize, selfReduceDimSize,                                  \
      selfNumel, reduce_func, alpha_value);                                                          \
  C10_CUDA_KERNEL_LAUNCH_CHECK();

#define LARGE_INDEX(TENSOR_TYPE, INDICES_TYPE, TYPE,                                     \
                    SELF_DIM, SOURCE_DIM, IDX_DIM, IDX_IS_MAJOR)                         \
  indexFuncLargeIndex<TENSOR_TYPE, INDICES_TYPE, TYPE,                                   \
                     SELF_DIM, SOURCE_DIM, IDX_DIM, IDX_IS_MAJOR>                        \
    <<<largeIndexGrid, largeIndexBlock, 0, stream>>>(                                    \
      selfInfo, sourceInfo, indexInfo,                                                   \
      selfReduceDim, sourceReduceDim, sourceTotalSize,                                   \
      (IDX_IS_MAJOR) ? sliceSize : numIndex,                                             \
      selfReduceDimSize, selfNumel, reduce_func, alpha_value);                           \
  C10_CUDA_KERNEL_LAUNCH_CHECK();

#define SMALL_INDEX_WITHOUT_ASSERT(TENSOR_TYPE, INDICES_TYPE, TYPE, SELF_DIM, SOURCE_DIM, IDX_DIM)                  \
  indexFuncSmallIndexWithoutAssert<TENSOR_TYPE, INDICES_TYPE, TYPE, SELF_DIM, SOURCE_DIM, IDX_DIM>                \
    <<<smallIndexGrid, smallIndexBlock, 0, stream>>>(                                                \
      selfInfo, sourceInfo, indexInfo,                                                               \
      selfReduceDim, sourceReduceDim, sliceSize, selfReduceDimSize,                                  \
      selfNumel, reduce_func, alpha_value);                                                          \
  C10_CUDA_KERNEL_LAUNCH_CHECK();

#define LARGE_INDEX_WITHOUT_ASSERT(TENSOR_TYPE, INDICES_TYPE, TYPE,                                     \
                    SELF_DIM, SOURCE_DIM, IDX_DIM, IDX_IS_MAJOR)                         \
  indexFuncLargeIndexWithoutAssert<TENSOR_TYPE, INDICES_TYPE, TYPE,                                   \
                     SELF_DIM, SOURCE_DIM, IDX_DIM, IDX_IS_MAJOR>                        \
    <<<largeIndexGrid, largeIndexBlock, 0, stream>>>(                                    \
      selfInfo, sourceInfo, indexInfo,                                                   \
      selfReduceDim, sourceReduceDim, sourceTotalSize,                                   \
      (IDX_IS_MAJOR) ? sliceSize : numIndex,                                             \
      selfReduceDimSize, selfNumel, reduce_func, alpha_value);                           \
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  dim3 smallIndexGrid(std::min(ceil_div(sliceSize, (ptrdiff_t)128), (ptrdiff_t)(mpc * 8 * 4)));
  dim3 smallIndexBlock(std::min(sliceSize, (ptrdiff_t)128));

  dim3 largeIndexGrid(std::min(ceil_div(sourceTotalSize, (ptrdiff_t)128), (ptrdiff_t)(mpc * 8 * 4 * 2)));
  dim3 largeIndexBlock(std::min(sourceTotalSize, (ptrdiff_t)128));

  bool enable_indexing_with_assert = at::maca::get_maca_enable_indexing_assert_kernel();
  if (cuda::detail::canUse32BitIndexMath(result) &&
      cuda::detail::canUse32BitIndexMath(source) &&
      cuda::detail::canUse32BitIndexMath(index)) {
    AT_DISPATCH_ALL_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, result.scalar_type(), "index_reduce", [&] {
      cuda::detail::TensorInfo<scalar_t, unsigned int> selfInfo =
          cuda::detail::getTensorInfo<scalar_t, unsigned int>(self_);
      int selfReduceDim = selfInfo.collapseDims(dim);
      selfInfo.reduceDim(selfReduceDim);
      auto alpha_value = (scalar_t) 1;
      AT_DISPATCH_INDEX_TYPES(index.scalar_type(), "index_reduce_cuda", [&] () {
        auto sourceInfo =
          cuda::detail::getTensorInfo<scalar_t, unsigned int>(source_);
        int sourceReduceDim = sourceInfo.collapseDims(dim);
        sourceInfo.reduceDim(sourceReduceDim);

        auto indexInfo =
        cuda::detail::getTensorInfo<index_t, unsigned int>(index);
        indexInfo.collapseDims();

        // A reasonable choice for when to have each thread iterate over
        // index to choose
        if (numIndex <= 16) {
          if (selfInfo.dims == 1 && sourceInfo.dims == 1 && indContig) {
            if (maca_unlikely(enable_indexing_with_assert)) {
              SMALL_INDEX(scalar_t, index_t, unsigned int, 1, 1, -2);
            } else {
              SMALL_INDEX_WITHOUT_ASSERT(scalar_t, index_t, unsigned int, 1, 1, -2);
            }
          } else if (selfInfo.dims == 2 && sourceInfo.dims == 2 && indContig) {
            if (maca_unlikely(enable_indexing_with_assert)) {
              SMALL_INDEX(scalar_t, index_t, unsigned int, 2, 2, -2);
            } else {
              SMALL_INDEX_WITHOUT_ASSERT(scalar_t, index_t, unsigned int, 2, 2, -2);
            }
          } else if (selfInfo.dims == 3 && sourceInfo.dims == 3 && indContig) {
            if (maca_unlikely(enable_indexing_with_assert)) {
              SMALL_INDEX(scalar_t, index_t, unsigned int, 3, 3, -2);
            } else {
              SMALL_INDEX_WITHOUT_ASSERT(scalar_t, index_t, unsigned int, 3, 3, -2);
            }
          } else {
            if (maca_unlikely(enable_indexing_with_assert)) {
              SMALL_INDEX(scalar_t, index_t, unsigned int, -1, -1, -1);
            } else {
              SMALL_INDEX_WITHOUT_ASSERT(scalar_t, index_t, unsigned int, -1, -1, -1);
            }
          }
        } else {
          bool indexIsMajor = indexShouldBeMajor(selfInfo, selfReduceDim);

          if (selfInfo.dims == 1 && sourceInfo.dims == 1 && indContig) {
            if (maca_unlikely(enable_indexing_with_assert)) {
              LARGE_INDEX(scalar_t, index_t, unsigned int, 1, 1, -2, true);
            } else {
              LARGE_INDEX_WITHOUT_ASSERT(scalar_t, index_t, unsigned int, 1, 1, -2, true);
            }
          } else if (selfInfo.dims == 2 && sourceInfo.dims == 2 && indContig) {
            if (indexIsMajor) {
              if (maca_unlikely(enable_indexing_with_assert)) {
                LARGE_INDEX(scalar_t, index_t, unsigned int, 2, 2, -2, true);
              } else {
                LARGE_INDEX_WITHOUT_ASSERT(scalar_t, index_t, unsigned int, 2, 2, -2, true);
              }
            } else {
              if (maca_unlikely(enable_indexing_with_assert)) {
                LARGE_INDEX(scalar_t, index_t, unsigned int, 2, 2, -2, false);
              } else {
                LARGE_INDEX_WITHOUT_ASSERT(scalar_t, index_t, unsigned int, 2, 2, -2, false);
              }
            }
          } else if (selfInfo.dims == 3 && sourceInfo.dims == 3 && indContig) {
            if (indexIsMajor) {
              if (maca_unlikely(enable_indexing_with_assert)) {
                LARGE_INDEX(scalar_t, index_t, unsigned int, 3, 3, -2, true);
              } else {
                LARGE_INDEX_WITHOUT_ASSERT(scalar_t, index_t, unsigned int, 3, 3, -2, true);
              }
            } else {
              if (maca_unlikely(enable_indexing_with_assert)) {
                LARGE_INDEX(scalar_t, index_t, unsigned int, 3, 3, -2, false);
              } else {
                LARGE_INDEX_WITHOUT_ASSERT(scalar_t, index_t, unsigned int, 3, 3, -2, false);
              }
            }
          } else {
            if (maca_unlikely(enable_indexing_with_assert)) {
              LARGE_INDEX(scalar_t, index_t, unsigned int, -1, -1, -1, true);
            } else {
              LARGE_INDEX_WITHOUT_ASSERT(scalar_t, index_t, unsigned int, -1, -1, -1, true);
            }
          }
        }
      });
    });
  } else {
    AT_DISPATCH_ALL_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, self.scalar_type(), "index_reduce", [&] {
      cuda::detail::TensorInfo<scalar_t, uint64_t> selfInfo =
        cuda::detail::getTensorInfo<scalar_t, uint64_t>(self_);
      int selfReduceDim = selfInfo.collapseDims(dim);
      selfInfo.reduceDim(selfReduceDim);
      auto alpha_value = (scalar_t) 1;

      cuda::detail::TensorInfo<scalar_t, uint64_t> sourceInfo =
        cuda::detail::getTensorInfo<scalar_t, uint64_t>(source_);
      int sourceReduceDim = sourceInfo.collapseDims(dim);
      sourceInfo.reduceDim(sourceReduceDim);

      AT_DISPATCH_INDEX_TYPES(index.scalar_type(), "index_reduce_cuda", [&] () {
        cuda::detail::TensorInfo<index_t, uint64_t> indexInfo =
          cuda::detail::getTensorInfo<index_t, uint64_t>(index);
        indexInfo.collapseDims();

        if (maca_unlikely(enable_indexing_with_assert)) {
          LARGE_INDEX(scalar_t, index_t, uint64_t, -1, -1, -1, true);
        } else {
          LARGE_INDEX_WITHOUT_ASSERT(scalar_t, index_t, uint64_t, -1, -1, -1, true);
        }
      });
    });
  }

#undef SMALL_INDEX
#undef LARGE_INDEX
#undef SMALL_INDEX_WITHOUT_ASSERT
#undef LARGE_INDEX_WITHOUT_ASSERT
}

TORCH_IMPL_FUNC(index_add_cuda_out)
(const Tensor& self, int64_t dim, const Tensor& index, const Tensor& source, const Scalar& alpha, const Tensor& result) {
  index_add_cuda_impl(self, dim, index, source, alpha, result);
}

TORCH_IMPL_FUNC(index_reduce_cuda_out)
(const Tensor& self,
 int64_t dim,
 const Tensor& index,
 const Tensor& source,
 const c10::string_view reduce,
 bool include_self,
 const Tensor& result) {
  TORCH_WARN_ONCE("index_reduce() is in beta and the API may change at any time.");

  if (reduce == "prod") {
    index_reduce_func_cuda_impl(self, dim, index, source, include_self, ReductionType::PROD, reduce_multiply, result);
  } else if (reduce == "mean") {
    index_reduce_func_cuda_impl(self, dim, index, source, include_self, ReductionType::MEAN, reduce_add, result);
    auto counts = include_self ? at::ones_like(result) : at::zeros_like(result);
    counts.index_add_(dim, index, at::ones_like(source));
    counts.masked_fill_(counts == 0, 1);
    if (result.is_floating_point() || result.is_complex()) {
      result.div_(counts);
    } else {
      result.div_(counts, "floor");
    }
  } else if (reduce == "amax") {
    index_reduce_func_cuda_impl(self, dim, index, source, include_self, ReductionType::MAX, reduce_maximum, result);
  } else if (reduce == "amin") {
    index_reduce_func_cuda_impl(self, dim, index, source, include_self, ReductionType::MIN, reduce_minimum, result);
  } else {
    TORCH_CHECK(false, "reduce argument must be either prod, mean, amax or amin, got ", reduce, ".");
  }
}

} // at::native
