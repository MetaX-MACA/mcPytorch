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

#ifdef USE_MACA
  #include "maca_kernels/indexing_opt.cuh"
#else
  static_assert(0)
#endif

namespace at::native {

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

namespace {
// We prefer this kernel to avoid reloading index points if the number
// of indices is a small number.
// This kernel in fact works for all choices of problem size, but if
// the number of indices chosen is large, then the
// indexSelectLargeIndex kernel is a better choice to increase
// parallelism.
template <typename T, typename IndicesType, typename IndexType, int DstDim, int SrcDim, int IdxDim>
__global__ void indexSelectSmallIndex(cuda::detail::TensorInfo<T, IndexType> dst,
                                      cuda::detail::TensorInfo<T, IndexType> src,
                                      cuda::detail::TensorInfo<IndicesType, IndexType> indices,
                                      int dstSelectDim,
                                      int srcSelectDim,
                                      IndexType innerSize,
                                      int64_t srcSelectDimSize) {
  // In order to avoid reloading the index that we are copying, load
  // it once to handle all of the points that are being selected, so
  // it can be reused as much as possible. This kernel is chosen when
  // this is a good choice (small number of chosen indices), since
  // re-accessing indices in addition to src elements can be slow.
  for (IndexType dstIndex = 0; dstIndex < indices.sizes[0]; ++dstIndex) {
    IndexType srcIndex =
      indices.data[cuda::detail::IndexToOffset<IndicesType, IndexType, IdxDim>::get(dstIndex, indices)];
    CUDA_KERNEL_ASSERT(srcIndex < srcSelectDimSize);

    // We stride over the output ignoring the indexed dimension
    // (innerSize), whose offset calculation is handled differently
    for (IndexType linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
         linearIndex < innerSize;
         linearIndex += gridDim.x * blockDim.x) {
      IndexType dstOffset =
        cuda::detail::IndexToOffset<T, IndexType, DstDim>::get(linearIndex, dst);
      dstOffset += dstIndex * dst.strides[dstSelectDim];

      IndexType srcOffset =
        cuda::detail::IndexToOffset<T, IndexType, SrcDim>::get(linearIndex, src);
      srcOffset += srcIndex * src.strides[srcSelectDim];

      dst.data[dstOffset] = src.data[srcOffset];
    }
  }
}

// We prefer this kernel to balance parallelism across index points,
// if there are a large number of indices.
// This kernel in fact works for all choices of problem size, but if
// the number of indices chosen is small, then the
// indexSelectSmallIndex kernel is a better choice to reduce memory
// accesses.
template <typename T, typename IndicesType, typename IndexType, int DstDim, int SrcDim, int IdxDim,
          bool IndexIsMajor>
__global__ void indexSelectLargeIndex(cuda::detail::TensorInfo<T, IndexType> dst,
                                      cuda::detail::TensorInfo<T, IndexType> src,
                                      cuda::detail::TensorInfo<IndicesType, IndexType> indices,
                                      int dstSelectDim,
                                      int srcSelectDim,
                                      IndexType totalSize,
                                      IndexType innerSize,
                                      int64_t srcSelectDimSize) {
  // We stride over the output including the indexed dimension
  // (totalSize), and calculate the destination index point based on that
  for (IndexType linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
       linearIndex < totalSize;
       linearIndex += gridDim.x * blockDim.x) {
    IndexType dstIndex, elementInSlice;
    if (IndexIsMajor) {
      dstIndex = linearIndex / innerSize;
      elementInSlice = linearIndex % innerSize;
    }
    else {
      elementInSlice = linearIndex / innerSize;
      dstIndex = linearIndex % innerSize;
    }

    IndexType srcIndex =
      indices.data[cuda::detail::IndexToOffset<IndicesType, IndexType, IdxDim>::get(dstIndex, indices)];
    CUDA_KERNEL_ASSERT(srcIndex < srcSelectDimSize);

    IndexType dstOffset =
      cuda::detail::IndexToOffset<T, IndexType, DstDim>::get(elementInSlice, dst);
    dstOffset += dstIndex * dst.strides[dstSelectDim];

    IndexType srcOffset =
      cuda::detail::IndexToOffset<T, IndexType, SrcDim>::get(elementInSlice, src);
    srcOffset += srcIndex * src.strides[srcSelectDim];

    dst.data[dstOffset] = src.data[srcOffset];
  }
}

namespace {

// When using a 0-dim scalar tensor, we need the legacy (THC) semantics of
// TensorInfo: Pretend that the scalar tensor is in fact a one-element vector.
template <typename T, typename IndexType>
cuda::detail::TensorInfo<T, IndexType>
tensorInfoLegacyIfScalar(cuda::detail::TensorInfo<T, IndexType> ti) {
  if (ti.dims == 0) {
    ti.dims = 1;
    ti.sizes[0] = 1;
    ti.strides[0] = 1;
  }
  return ti;
}

}

template <typename scalar_t>
void index_select_out_cuda_impl(
    Tensor& out,
    const Tensor& self,
    long dim,
    const Tensor& index) {
  ptrdiff_t numIndices = index.numel();
  int selfDims = self.dim() == 0 ? 1 : self.dim();

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  TORCH_CHECK(
      index.dim() <= 1, "Index is supposed to be an empty tensor or a vector");
  TORCH_CHECK(dim < selfDims, "Indexing dim is out of bounds");

  std::vector<int64_t> newSize = self.sizes().vec();
  if (self.dim() > 0) {
    newSize[dim] = numIndices;
  }

  if (self.is_quantized()){
      out = at::empty_quantized(newSize, out);
  } else {
    at::native::resize_output(out, newSize);
  }

  ptrdiff_t outTotalSize = out.numel();
  if (outTotalSize == 0) {
    return;
  }

  bool indContig = index.is_contiguous();

  // The `self` is partitioned into two parts:
  // -the size of each slice we are indexing, which is the
  // total size of the tensor ignoring dimension `dim`;
  // -the number of indices we are choosing, which is the total size
  // of the tensor `indices`.
  int64_t selfSelectDimSize = self.dim() == 0 ? 1 : self.size(dim);
  ptrdiff_t sliceSize = outTotalSize / numIndices;

  int mpc = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;

#define SMALL_INDEX(TENSOR_TYPE, INDICES_TYPE, TYPE, DST_DIM, SRC_DIM, IDX_DIM)         \
  indexSelectSmallIndex<TENSOR_TYPE, INDICES_TYPE, TYPE, DST_DIM, SRC_DIM, IDX_DIM>     \
    <<<smallIndexGrid, smallIndexBlock, 0, stream>>>(                                   \
      outInfo, selfInfo, indicesInfo,                                                   \
      outSelectDim, selfSelectDim, static_cast<TYPE>(sliceSize),                        \
      selfSelectDimSize);                                                               \
  C10_CUDA_KERNEL_LAUNCH_CHECK();

#define LARGE_INDEX(TENSOR_TYPE, INDICES_TYPE, TYPE,                           \
                    DST_DIM, SRC_DIM, IDX_DIM, IDX_IS_MAJOR)                   \
  indexSelectLargeIndex<TENSOR_TYPE, INDICES_TYPE, TYPE,                       \
                        DST_DIM, SRC_DIM, IDX_DIM, IDX_IS_MAJOR>               \
    <<<largeIndexGrid, largeIndexBlock, 0, stream>>>(                          \
      outInfo, selfInfo, indicesInfo,                                          \
      outSelectDim, selfSelectDim, static_cast<TYPE>(outTotalSize),            \
      static_cast<TYPE>((IDX_IS_MAJOR) ? sliceSize : numIndices),              \
      selfSelectDimSize);                                                      \
  C10_CUDA_KERNEL_LAUNCH_CHECK();

#ifdef USE_MACA
#define SMALL_INDEX_WITHOUT_ASSERT(TENSOR_TYPE, INDICES_TYPE, TYPE, DST_DIM, SRC_DIM, IDX_DIM)       \
  index::indexSelectSmallIndexWithoutAssert<TENSOR_TYPE, INDICES_TYPE, TYPE, DST_DIM, SRC_DIM, IDX_DIM>     \
    <<<smallIndexGrid, smallIndexBlock, 0, stream>>>(                                                \
      outInfo, selfInfo, indicesInfo,                                                                \
      outSelectDim, selfSelectDim, static_cast<TYPE>(sliceSize),                                     \
      selfSelectDimSize);                                                                            \
  C10_CUDA_KERNEL_LAUNCH_CHECK();

#define LARGE_INDEX_WITHOUT_ASSERT(TENSOR_TYPE, INDICES_TYPE, TYPE,            \
                    DST_DIM, SRC_DIM, IDX_DIM, IDX_IS_MAJOR)                   \
  index::indexSelectLargeIndexWithoutAssert<TENSOR_TYPE, INDICES_TYPE, TYPE,          \
                        DST_DIM, SRC_DIM, IDX_DIM, IDX_IS_MAJOR>               \
    <<<largeIndexGrid, largeIndexBlock, 0, stream>>>(                          \
      outInfo, selfInfo, indicesInfo,                                          \
      outSelectDim, selfSelectDim, static_cast<TYPE>(outTotalSize),            \
      static_cast<TYPE>((IDX_IS_MAJOR) ? sliceSize : numIndices),              \
      selfSelectDimSize);                                                      \
  C10_CUDA_KERNEL_LAUNCH_CHECK();
#endif

#ifdef USE_MACA
  dim3 smallIndexGrid(std::min(ceil_div(sliceSize, (ptrdiff_t)128), (ptrdiff_t)(mpc * 8 * 4)));
  dim3 smallIndexBlock(std::min(sliceSize, (ptrdiff_t)128));

  dim3 largeIndexGrid(std::min(ceil_div(outTotalSize, (ptrdiff_t)128), (ptrdiff_t)(mpc * 8 * 4 * 2)));
  dim3 largeIndexBlock(std::min(outTotalSize, (ptrdiff_t)128));
  bool enable_indexing_with_assert = at::maca::get_maca_enable_indexing_assert_kernel();
#else
  im3 smallIndexGrid(std::min(ceil_div(sliceSize, (ptrdiff_t)128), (ptrdiff_t)(mpc * 8)));
  dim3 smallIndexBlock(std::min(sliceSize, (ptrdiff_t)128));

  dim3 largeIndexGrid(std::min(ceil_div(outTotalSize, (ptrdiff_t)128), (ptrdiff_t)(mpc * 8)));
  dim3 largeIndexBlock(std::min(outTotalSize, (ptrdiff_t)128));
#endif
  if (cuda::detail::canUse32BitIndexMath(out) &&
      cuda::detail::canUse32BitIndexMath(self) &&
      cuda::detail::canUse32BitIndexMath(index)) {
    auto outInfo = tensorInfoLegacyIfScalar(cuda::detail::getTensorInfo<scalar_t, unsigned int>(out));
    int outSelectDim = outInfo.collapseDims(dim);
    outInfo.reduceDim(outSelectDim);

    auto  selfInfo = tensorInfoLegacyIfScalar(cuda::detail::getTensorInfo<scalar_t, unsigned int>(self));
    int selfSelectDim = selfInfo.collapseDims(dim);
    selfInfo.reduceDim(selfSelectDim);

    AT_DISPATCH_INDEX_TYPES(index.scalar_type(), "index_select_out_cuda_impl", [&] () {
      auto indicesInfo = tensorInfoLegacyIfScalar(cuda::detail::getTensorInfo<index_t, unsigned int>(index));
      indicesInfo.collapseDims();

      // A reasonable choice for when to have each thread iterate over
      // indices to choose
#ifdef USE_MACA
      if (numIndices <= 16) {
        if (outInfo.dims == 1 && selfInfo.dims == 1 && indContig) {
          if (maca_unlikely(enable_indexing_with_assert)) {
            SMALL_INDEX(scalar_t, index_t, unsigned int, 1, 1, -2);
          } else {
            SMALL_INDEX_WITHOUT_ASSERT(scalar_t, index_t, unsigned int, 1, 1, -2);
          }
        } else if (outInfo.dims == 2 && selfInfo.dims == 2 && indContig) {
          if (maca_unlikely(enable_indexing_with_assert)) {
            SMALL_INDEX(scalar_t, index_t, unsigned int, 2, 2, -2);
          } else {
            SMALL_INDEX_WITHOUT_ASSERT(scalar_t, index_t, unsigned int, 2, 2, -2);
          }
        } else if (outInfo.dims == 3 && selfInfo.dims == 3 && indContig) {
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
        bool indexIsMajor = indexShouldBeMajor(outInfo, outSelectDim);

        if (outInfo.dims == 1 && selfInfo.dims == 1 && indContig) {
          if (maca_unlikely(enable_indexing_with_assert)) {
            LARGE_INDEX(scalar_t, index_t, unsigned int, 1, 1, -2, true);
          } else {
            LARGE_INDEX_WITHOUT_ASSERT(scalar_t, index_t, unsigned int, 1, 1, -2, true);
          }
        } else if (outInfo.dims == 2 && selfInfo.dims == 2 && indContig) {
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
        } else if (outInfo.dims == 3 && selfInfo.dims == 3 && indContig) {
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
      if (numIndices <= 16) {
        if (outInfo.dims == 1 && selfInfo.dims == 1 && indContig) {
          SMALL_INDEX(scalar_t, index_t, unsigned int, 1, 1, -2);
        } else if (outInfo.dims == 2 && selfInfo.dims == 2 && indContig) {
          SMALL_INDEX(scalar_t, index_t, unsigned int, 2, 2, -2);
        } else if (outInfo.dims == 3 && selfInfo.dims == 3 && indContig) {
          SMALL_INDEX(scalar_t, index_t, unsigned int, 3, 3, -2);
        } else {
          SMALL_INDEX(scalar_t, index_t, unsigned int, -1, -1, -1);
        }
      } else {
        bool indexIsMajor = indexShouldBeMajor(outInfo, outSelectDim);

        if (outInfo.dims == 1 && selfInfo.dims == 1 && indContig) {
          LARGE_INDEX(scalar_t, index_t, unsigned int, 1, 1, -2, true);
        } else if (outInfo.dims == 2 && selfInfo.dims == 2 && indContig) {
          if (indexIsMajor) {
            LARGE_INDEX(scalar_t, index_t, unsigned int, 2, 2, -2, true);
          } else {
            LARGE_INDEX(scalar_t, index_t, unsigned int, 2, 2, -2, false);
          }
        } else if (outInfo.dims == 3 && selfInfo.dims == 3 && indContig) {
          if (indexIsMajor) {
            LARGE_INDEX(scalar_t, index_t, unsigned int, 3, 3, -2, true);
          } else {
            LARGE_INDEX(scalar_t, index_t, unsigned int, 3, 3, -2, false);
          }
        } else {
          LARGE_INDEX(scalar_t, index_t, unsigned int, -1, -1, -1, true);
        }
      }
#endif
    });
  } else {
    auto outInfo = tensorInfoLegacyIfScalar(cuda::detail::getTensorInfo<scalar_t, uint64_t>(out));
    int outSelectDim = outInfo.collapseDims(dim);
    outInfo.reduceDim(outSelectDim);

    auto selfInfo = tensorInfoLegacyIfScalar(cuda::detail::getTensorInfo<scalar_t, uint64_t>(self));
    int selfSelectDim = selfInfo.collapseDims(dim);
    selfInfo.reduceDim(selfSelectDim);
    AT_DISPATCH_INDEX_TYPES(index.scalar_type(), "index_select_out_cuda_impl", [&] () {
      auto indicesInfo = tensorInfoLegacyIfScalar(cuda::detail::getTensorInfo<index_t, uint64_t>(index));
      indicesInfo.collapseDims();

#ifdef USE_MACA
      if (maca_unlikely(enable_indexing_with_assert)) {
        LARGE_INDEX(scalar_t, index_t, uint64_t, -1, -1, -1, true);
      } else {
        LARGE_INDEX_WITHOUT_ASSERT(scalar_t, index_t, uint64_t, -1, -1, -1, true);
      }
#else
      LARGE_INDEX(scalar_t, index_t, uint64_t, -1, -1, -1, true);
#endif
    });
  }
#undef SMALL_INDEX
#undef LARGE_INDEX
#ifdef USE_MACA
#undef SMALL_INDEX_WITHOUT_ASSERT
#undef LARGE_INDEX_WITHOUT_ASSERT
#endif
}
} // anonymous namespace

Tensor& index_select_out_cuda(
    const Tensor& self,
    int64_t dim,
    const Tensor& index,
    Tensor& out) {
  static constexpr string_view DIM_WARNING =
      "Tensor too large or too many (> 25) dimensions";
  TORCH_CHECK(
      at::cuda::check_device({out, self, index}),
      "Input, output and indices must be on the current device");
  at::assert_no_internal_overlap(out);
  at::assert_no_overlap(out, self);
  at::assert_no_overlap(out, index);

  dim = at::maybe_wrap_dim(dim, self);
  TORCH_CHECK(self.dim() <= MAX_TENSORINFO_DIMS, DIM_WARNING);
  TORCH_CHECK(index.dim() <= MAX_TENSORINFO_DIMS, DIM_WARNING);
  if (self.is_quantized()){
    TORCH_CHECK(
      self.qscheme() == kPerTensorAffine,
      "Only per_tensor quantized quantized tensors are supported by index_select.")
    AT_DISPATCH_QINT_TYPES(out.scalar_type(), "index_select_quant_cuda", [&] {
      index_select_out_cuda_impl<scalar_t>(out, self, dim, index);
    });
  } else {
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND4(
        at::ScalarType::ComplexHalf,
        at::ScalarType::Half,
        at::ScalarType::Bool,
        at::ScalarType::BFloat16,
        out.scalar_type(),
        "index_select_cuda",
        [&] { index_select_out_cuda_impl<scalar_t>(out, self, dim, index); });
  }

  return out;
}

Tensor index_select_cuda(const Tensor& self, int64_t dim, const Tensor& index) {
  Tensor out = at::empty({0}, self.options());
  at::native::index_select_out_cuda(self, dim, index, out);
  return out;
}

Tensor index_select_quantized_cuda(const Tensor& self, int64_t dim, const Tensor& index) {
  TORCH_CHECK(
    self.qscheme() == kPerTensorAffine,
    "Only per_tensor quantized quantized tensors are supported by index_select.")
  Tensor out = at::empty_quantized({0}, self);
  at::native::index_select_out_cuda(self, dim, index, out);
  return out;
}

namespace {

template <typename mask_t>
void masked_fill_kernel(TensorIterator& iter, const Scalar& value) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND4_OPT_TYPE(
      kBool, kHalf, kBFloat16, kComplexHalf, iter.common_dtype(), "masked_fill_", 
      [&]() {
        const auto value_ = value.to<scalar_t>();
        gpu_kernel(
            iter, [value_] GPU_LAMBDA(scalar_t self, mask_t mask) -> scalar_t {
              if (mask) {
                return value_;
              }
              return self;
            });
      },
      [&]() {
        const auto value_ = value.to<scalar_t>();
        gpu_kernel_maca_arity2(
            iter, [value_] GPU_LAMBDA(scalar_t self, mask_t mask) -> scalar_t {
              if (mask) {
                return value_;
              }
              return self;
            });
      });
}

template <typename scalar_t, typename mask_t>
void cuda_masked_fill_kernel_quantized(TensorIterator& iter, scalar_t quantized_val) {
    gpu_kernel(
        iter, [quantized_val] GPU_LAMBDA(scalar_t self, mask_t mask) -> scalar_t {
          if (mask) {
            return quantized_val;
          }
          return self;
    });
}

void masked_fill_kernel_quantized(TensorIterator& iter, const Scalar& value, double scale, int zero_point) {
  AT_DISPATCH_QINT_TYPES(
      iter.common_dtype(), "masked_fill_", [&]() {
        float float_val = value.to<float>();
        const auto quantized_val = quantize_val<scalar_t>(scale, zero_point, float_val);
        auto mask_dtype = iter.input_dtype(0);

        if (mask_dtype == at::ScalarType::Bool) {
            cuda_masked_fill_kernel_quantized<scalar_t, bool>(iter, quantized_val);
        }
        else {
            cuda_masked_fill_kernel_quantized<scalar_t, uint8_t>(iter, quantized_val);
        }
    });
}

REGISTER_CUDA_DISPATCH(masked_fill_kernel_quantized_stub, &masked_fill_kernel_quantized);

} // anonymous namespace

Tensor & masked_fill__cuda(Tensor& self, const Tensor & mask, const Scalar& value) {
  TORCH_CHECK(self.device() == mask.device(), "expected self and mask to be on the same device, but got mask on ",
    mask.device(), " and self on ", self.device());
  TORCH_CHECK(mask.scalar_type() == kByte || mask.scalar_type() == kBool,
    "expected mask dtype to be Bool but got ", mask.scalar_type());
  auto maybe_outnames = namedinference::broadcast_to_outnames(self, mask, "masked_fill_");
  if (at::has_internal_overlap(self) == MemOverlap::Yes) {
    TORCH_WARN(
      "Use of masked_fill_ on expanded tensors is deprecated. "
      "Please clone() the tensor before performing this operation. "
      "This also applies to advanced indexing e.g. tensor[mask] = scalar");
  }
  at::assert_no_partial_overlap(self, mask);

  c10::MaybeOwned<Tensor> b_mask = expand_inplace(self, mask, "masked_fill_");

  auto iter = TensorIteratorConfig()
      .set_check_mem_overlap(false)
      .check_all_same_dtype(false)
      .resize_outputs(false)
      .add_output(self)
      .add_input(self)
      .add_input(*b_mask)
      .build();

  if (b_mask->dtype() == at::ScalarType::Byte) {
    TORCH_WARN("masked_fill_ received a mask with dtype torch.uint8, this behavior is now deprecated," \
            "please use a mask with dtype torch.bool instead.");
    masked_fill_kernel<uint8_t>(iter, value);
  } else {
    masked_fill_kernel<bool>(iter, value);
  }
  namedinference::propagate_names_if_nonempty(self, maybe_outnames);
  return self;
}

Tensor & masked_fill__cuda(Tensor& self, const Tensor & mask, const Tensor & value) {
  TORCH_CHECK(value.dim() == 0, "masked_fill_ only supports a 0-dimensional value tensor, but got tensor "
      "with ", value.dim(), " dimension(s).");
  // We hit this function if either of the input tensor lives on CUDA.
  // It is ok, if `value` is `CPU` tensor but we should not allow `self` or
  // `mask` to be CPU tensor. Check for `self` and `mask` being on same device
  // exists in `masked_fill__cuda` (Scalar version).
  TORCH_CHECK(!self.device().is_cpu(), "masked_fill_: Expected inputs to be on same device")
  return masked_fill__cuda(self, mask, value.item());
}

namespace {

// ForwardIt: only legacy random access iterator is supported.
template<class ForwardIt, class T, bool is_lower = true>
static __host__ __device__ __forceinline__
ForwardIt find_bound(ForwardIt first, ForwardIt last, const T& value) {
    ForwardIt it;
    typename std::iterator_traits<ForwardIt>::difference_type count, step;
    // NOTE: std::distance(first, last) compiles but produces wrong results here,
    // so only legacy random access iterators are safe in this code.
    count = last - first;

    while (count > 0) {
      it = first;
      step = count / 2;
      // avoiding std::advance(it, step),
      // although it does work unlike std::distance
      it += step;
      if (is_lower ? *it < value : value >= *it) {
        first = ++it;
        count -= step + 1;
      }
      else {
        count = step;
      }
    }
    return first;
}

}

Tensor index_select_sparse_cuda(const Tensor& self, int64_t dim, const Tensor& index) {
  const auto ndim = self.dim();
  TORCH_CHECK_INDEX(ndim, "index_select() cannot be applied to a 0-dim tensor.");
  TORCH_CHECK_INDEX(
      index.dim() == 1 && index.dtype() == at::kLong && index.options().layout() == at::kStrided,
      "index_select() argument index must be 1-D strided (non-sparse) long-tensor.");
  dim = maybe_wrap_dim(dim, ndim);
  const auto size = self.size(dim);
  const auto sparse_dim = self.sparse_dim();
  const auto dense_dim = self.dense_dim();
  const auto indices = self._indices();
  const auto values = self._values();
  const auto nnz = values.size(0);
  const auto index_len = index.size(0);
  auto res_sizes = self.sizes().vec();
  res_sizes[dim] = index_len;

  // If indexing into sparse dimensions
  if (dim < sparse_dim) {
    const auto make_output = [
      dim, sparse_dim, dense_dim, res_sizes, &self, &indices, &values
    ](
        const Tensor& selected_dim_indices,
        const Tensor& res_dim_indices
    ) -> Tensor {
      auto res_indices = indices.index_select(1, selected_dim_indices);
      res_indices[dim] = res_dim_indices;
      const auto res_values = values.index_select(0, selected_dim_indices);

      return at::_sparse_coo_tensor_with_dims_and_tensors(
          sparse_dim, dense_dim, res_sizes, res_indices, res_values, self.options());
    };

    // short-circuit if index is empty
    if (!index_len) {
      return make_output(index, index);
    }

    const auto nneg_index = [&index, size]() -> Tensor {
      auto nneg_index = at::empty_like(index, at::MemoryFormat::Contiguous);

      auto iter = TensorIteratorConfig()
        .add_output(nneg_index)
        .add_input(index)
        .build();

      AT_DISPATCH_INDEX_TYPES(index.scalar_type(), "index_select_sparse_cuda", [&]() {
          gpu_kernel(iter, [size] GPU_LAMBDA (index_t idx) -> index_t {
              CUDA_KERNEL_ASSERT(idx >= -size && idx < size
                  && "index_select(): index out of bounds");
              return idx < 0 ? idx + size : idx;
          });
      });
      return nneg_index;
    }();

    const auto dim_indices = indices[dim].contiguous();
    const auto idx_nneg_index = at::arange(index_len, nneg_index.options());
    const auto idx_dim_indices = at::arange(nnz, dim_indices.options());

    Tensor sorted_dim_indices, argsort_dim_indices;
    std::tie(sorted_dim_indices, argsort_dim_indices) = [&]() -> std::tuple<Tensor, Tensor> {
      if (dim == 0 && self.is_coalesced()) {
        return std::make_tuple(dim_indices, idx_dim_indices);
      }
      else {
        return dim_indices.sort();
      }
    }();

    Tensor intrsc_counts_nneg_index;
    Tensor intrsc_first_match_nneg_index;
    std::tie(intrsc_counts_nneg_index, intrsc_first_match_nneg_index) = [&]() -> std::tuple<Tensor, Tensor> {
      auto intrsc_counts_nneg_index = at::zeros_like(nneg_index);
      auto intrsc_first_match_nneg_index = at::zeros_like(nneg_index);

      auto iter = TensorIteratorConfig()
        .add_output(intrsc_first_match_nneg_index)
        .add_input(nneg_index)
        .add_input(idx_nneg_index)
        .build();

      AT_DISPATCH_INDEX_TYPES(nneg_index.scalar_type(), "index_select_sparse_cuda", [&]() {
          index_t* ptr_intrsc_counts_nneg_index = intrsc_counts_nneg_index.data_ptr<index_t>();
          index_t* ptr_sorted_dim_indices = sorted_dim_indices.data_ptr<index_t>();
          gpu_kernel(
              iter,
              [ptr_intrsc_counts_nneg_index, ptr_sorted_dim_indices, nnz] GPU_LAMBDA (
                index_t idx_val, index_t idx_idx
              ) -> index_t {
                auto* lb = find_bound<index_t*, index_t, true>(
                  ptr_sorted_dim_indices,
                  ptr_sorted_dim_indices + nnz,
                  idx_val
                );
                auto* ub = find_bound<index_t*, index_t, false>(
                  ptr_sorted_dim_indices,
                  ptr_sorted_dim_indices + nnz,
                  idx_val
                );
                const auto idx_count = ub - lb;
                ptr_intrsc_counts_nneg_index[idx_idx] = idx_count;

                return lb - ptr_sorted_dim_indices;
              }
          );
      });

      return std::make_tuple(intrsc_counts_nneg_index, intrsc_first_match_nneg_index);
    }();

    // Unavoidable sync since the shape of the result is not known in advance
    auto res_len = intrsc_counts_nneg_index.sum().item<int64_t>();
    // Short-circuit if empty intersection
    if (!res_len) {
      auto empty_idx = at::empty({0}, nneg_index.options());
      return make_output(empty_idx, empty_idx);
    }

    Tensor selected_dim_indices, res_dim_indices;
    std::tie(selected_dim_indices, res_dim_indices) = [&]() -> std::tuple<Tensor, Tensor> {
      auto res_dim_indices = at::empty({res_len}, nneg_index.options());
      auto selected_dim_indices = at::empty_like(res_dim_indices);
      auto selected_dim_indices_offsets = intrsc_counts_nneg_index.cumsum(0)
        .sub_(intrsc_counts_nneg_index);

      // Need to have output as TensorIterator does not allow having void lambdas.
      auto dummy_output = at::empty({1}, dim_indices.options()).expand(IntArrayRef({index_len}));
      auto iter = TensorIteratorConfig()
        .add_output(dummy_output)
        // All iterations map to a single element in dummy_output by design,
        // hence removed output memory overlap check.
        .set_check_mem_overlap(false)
        .add_input(idx_nneg_index)
        .add_input(intrsc_counts_nneg_index)
        .add_input(selected_dim_indices_offsets)
        .add_input(intrsc_first_match_nneg_index)
        .build();

      AT_DISPATCH_INDEX_TYPES(nneg_index.scalar_type(), "index_select_sparse_cuda", [&]() {
          index_t* ptr_res_dim_indices = res_dim_indices.data_ptr<index_t>();
          index_t* ptr_selected_dim_indices = selected_dim_indices.data_ptr<index_t>();
          index_t* ptr_argsort_dim_indices = argsort_dim_indices.data_ptr<index_t>();
          gpu_kernel(
              iter,
              [ptr_res_dim_indices, ptr_selected_dim_indices, ptr_argsort_dim_indices] GPU_LAMBDA (
                index_t idx_idx, index_t count, index_t offset, index_t first_match
              ) -> index_t {
                index_t* __restrict__ ptr_res_dim_indices_out = ptr_res_dim_indices + offset;
                index_t* __restrict__ ptr_argsort_dim_indices_in = ptr_argsort_dim_indices + first_match;
                index_t* __restrict__ ptr_selected_dim_indices_out = ptr_selected_dim_indices + offset;
                for (index_t i = 0; i < count; ++i) {
                  *ptr_res_dim_indices_out++ = idx_idx;
                  *ptr_selected_dim_indices_out++ = *ptr_argsort_dim_indices_in++;
                }

                // A dummy return scalar for a dummy output
                return static_cast<index_t>(1);
              }
          );
      });

      return std::make_tuple(selected_dim_indices, res_dim_indices);
    }();

    return make_output(selected_dim_indices, res_dim_indices);
  }
  // If indexing into dense dimensions
  else {
    // It is sufficient to just perform `index_select` on values
    // if `dim` refers to dense dimensions.
    const auto res_values = values.index_select(dim - sparse_dim + 1, index);

    return _sparse_coo_tensor_with_dims_and_tensors(
        sparse_dim, dense_dim, res_sizes, indices, res_values, self.options());
  }
}


} // at::native
