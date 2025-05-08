#pragma once

namespace at::native {
namespace index {

template<typename scalar_t, int vec_size>
struct alignas(sizeof(scalar_t) * vec_size) aligned_vector {
  scalar_t val[vec_size];
};

template <typename T, typename IndicesType, typename IndexType, int DstDim, int SrcDim, int IdxDim,
          bool IndexIsMajor, typename func_t>
__global__ void indexFuncLargeIndex_3_3_true_half(
    T* self_data, const T* source_data, const IndicesType* index_data,
    IndexType self_size0,       IndexType source_size0,        IndexType index_size0,
    IndexType self_size1,       IndexType source_size1,
    IndexType self_size2,       IndexType source_size2,
    IndexType self_stride0,     IndexType source_stride0,      IndexType index_stride0,
    IndexType self_stride1,     IndexType source_stride1,
    IndexType self_stride2,     IndexType source_stride2,
    // int dstAddDim,
    // int srcAddDim,
    IndexType dstAddStride,
    IndexType srcAddStride,
    IndexType totalSize,
    IndexType innerSize,
    int64_t dstAddDimSize,
    int64_t dstNumel,
    const func_t& op,
    T alpha
) {
  for (IndexType linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
      linearIndex < totalSize;
      linearIndex += gridDim.x * blockDim.x) {
    IndexType srcIndex = linearIndex / innerSize;
    IndexType elementInSlice = linearIndex % innerSize;

    IndexType offset_index = srcIndex * index_stride0;
    IndexType dstIndex = index_data[offset_index];

    IndexType curDimIndex;
    IndexType curDimOffset;
    IndexType base_offset = 0;

    IndexType linearId = elementInSlice;
    curDimIndex = linearId % self_size2;   //dim2
    curDimOffset = curDimIndex * self_stride2;
    base_offset += curDimOffset;
    linearId /= self_size2;
    curDimIndex = linearId % self_size1;   //dim1
    curDimOffset = curDimIndex * self_stride1;
    base_offset += curDimOffset;
    linearId /= self_size1;
    base_offset += linearId * self_stride0;  //dim0

    IndexType dstOffset = base_offset + dstIndex * dstAddStride; 
    IndexType srcOffset = base_offset + srcIndex * srcAddStride;

    T val = source_data[srcOffset] * alpha;
    op(self_data, dstOffset, dstNumel, &val);
   }
}

template <typename T, typename IndicesType, typename IndexType, int DstDim, int SrcDim, int IdxDim,
          bool IndexIsMajor, typename func_t,
typename std::enable_if<!std::is_same<T, at::BFloat16>::value && !std::is_same<T, at::Half>::value, int>::type = 0>
__global__ void indexFuncLargeIndexAdd_2_2_true(
    cuda::detail::TensorInfo<T, IndexType> dst,
    cuda::detail::TensorInfo<const T, IndexType> src,
    cuda::detail::TensorInfo<const IndicesType, IndexType> indices,
    int dstAddDim,        //0
    int srcAddDim,        //0
    IndexType totalSize,  //24576*2048
    IndexType innerSize,  //2048
    int64_t dstAddDimSize,//4096
    int64_t dstNumel,     //4096*2048
    const func_t& op,
    T alpha
) {
  assert(false);
}

template <typename T, typename IndicesType, typename IndexType, int DstDim, int SrcDim, int IdxDim,
          bool IndexIsMajor, typename func_t,
typename std::enable_if<std::is_same<T, at::Half>::value, int>::type = 0>
__global__ void indexFuncLargeIndexAdd_2_2_true(
    cuda::detail::TensorInfo<T, IndexType> dst,
    cuda::detail::TensorInfo<const T, IndexType> src,
    cuda::detail::TensorInfo<const IndicesType, IndexType> indices,
    int dstAddDim,        //0
    int srcAddDim,        //0
    IndexType totalSize,  //24576*2048
    IndexType innerSize,  //2048
    int64_t dstAddDimSize,//4096
    int64_t dstNumel,     //4096*2048
    const func_t& op,
    T alpha
) {
  const int vec_size = 2;
  using vec_t = aligned_vector<T, vec_size>;

  IndexType start = (blockIdx.x * blockDim.x + threadIdx.x) * vec_size;
  IndexType step = gridDim.x * blockDim.x * vec_size;

  for (IndexType linearIndex = start;
       linearIndex < totalSize;
       linearIndex += step) {
    IndexType srcIndex, elementInSlice;
    srcIndex = linearIndex / innerSize;
    elementInSlice = linearIndex % innerSize;

    IndexType dstIndex = indices.data[srcIndex];

    IndexType Offset =
      cuda::detail::IndexToOffset<T, IndexType, DstDim>::get(elementInSlice, dst);

    IndexType dstOffset = Offset + dstIndex * dst.strides[dstAddDim];
    IndexType srcOffset = Offset + srcIndex * src.strides[srcAddDim];

    vec_t SrcVal = *(reinterpret_cast<const vec_t*>(src.data + srcOffset));
    #pragma unroll
    for (int ii=0; ii < vec_size; ii++){
       SrcVal.val[ii] = SrcVal.val[ii] * alpha;
    }

    __half2 value2;
    value2.x = *reinterpret_cast<__half*>(&SrcVal.val[0]);
    value2.y = *reinterpret_cast<__half*>(&SrcVal.val[1]);
    atomicAdd(reinterpret_cast<__half2*>(dst.data + dstOffset), value2);
  }

}

template <typename T, typename IndicesType, typename IndexType, int DstDim, int SrcDim, int IdxDim,
          bool IndexIsMajor, typename func_t,
typename std::enable_if<std::is_same<T, at::BFloat16>::value, int>::type = 0>
__global__ void indexFuncLargeIndexAdd_2_2_true(
    cuda::detail::TensorInfo<T, IndexType> dst,
    cuda::detail::TensorInfo<const T, IndexType> src,
    cuda::detail::TensorInfo<const IndicesType, IndexType> indices,
    int dstAddDim,        //0
    int srcAddDim,        //0
    IndexType totalSize,  //24576*2048
    IndexType innerSize,  //2048
    int64_t dstAddDimSize,//4096
    int64_t dstNumel,     //4096*2048
    const func_t& op,
    T alpha
) {
  const int vec_size = 2;
  using vec_t = aligned_vector<T, vec_size>;

  IndexType start = (blockIdx.x * blockDim.x + threadIdx.x) * vec_size;
  IndexType step = gridDim.x * blockDim.x * vec_size;

  for (IndexType linearIndex = start;
       linearIndex < totalSize;
       linearIndex += step) {
    IndexType srcIndex, elementInSlice;
    srcIndex = linearIndex / innerSize;
    elementInSlice = linearIndex % innerSize;

    IndexType dstIndex = indices.data[srcIndex];

    IndexType Offset =
      cuda::detail::IndexToOffset<T, IndexType, DstDim>::get(elementInSlice, dst);

    IndexType dstOffset = Offset + dstIndex * dst.strides[dstAddDim];
    IndexType srcOffset = Offset + srcIndex * src.strides[srcAddDim];

    vec_t SrcVal = *(reinterpret_cast<const vec_t*>(src.data + srcOffset));
    #pragma unroll
    for (int ii=0; ii < vec_size; ii++){
       SrcVal.val[ii] = SrcVal.val[ii] * alpha;
    }

    __nv_bfloat162 value2;
    value2.x = *reinterpret_cast<__nv_bfloat16*>(&SrcVal.val[0]);
    value2.y = *reinterpret_cast<__nv_bfloat16*>(&SrcVal.val[1]);
    atomicAdd(reinterpret_cast<__nv_bfloat162*>(dst.data + dstOffset), value2);
  }

}


template <typename T, typename IndicesType, typename IndexType, int DstDim, int SrcDim, int IdxDim,
          typename func_t>
__global__ void indexFuncSmallIndexWithoutAssert(cuda::detail::TensorInfo<T, IndexType> dst,
                                    cuda::detail::TensorInfo<const T, IndexType> src,
                                    cuda::detail::TensorInfo<const IndicesType, IndexType> indices,
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
        indices.data[cuda::detail::IndexToOffset<const IndicesType, IndexType, IdxDim>::get(srcIndex, indices)];
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
          cuda::detail::IndexToOffset<const T, IndexType, SrcDim>::get(linearIndex, src);
      srcOffset += srcIndex * src.strides[srcAddDim];

      T val = src.data[srcOffset] * alpha;
      op(dst.data, dstOffset, dstNumel, &val);
    }

  }
}


template <typename T, typename IndicesType, typename IndexType, int DstDim, int SrcDim, int IdxDim,
          bool IndexIsMajor, typename func_t>
__global__ void indexFuncLargeIndexWithoutAssert(cuda::detail::TensorInfo<T, IndexType> dst,
                                    cuda::detail::TensorInfo<const T, IndexType> src,
                                    cuda::detail::TensorInfo<const IndicesType, IndexType> indices,
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
        indices.data[cuda::detail::IndexToOffset<const IndicesType, IndexType, IdxDim>::get(srcIndex, indices)];
    // CUDA_KERNEL_ASSERT(dstIndex < dstAddDimSize);

    IndexType dstOffset =
      cuda::detail::IndexToOffset<T, IndexType, DstDim>::get(elementInSlice, dst);
    dstOffset += dstIndex * dst.strides[dstAddDim];

    IndexType srcOffset =
      cuda::detail::IndexToOffset<const T, IndexType, SrcDim>::get(elementInSlice, src);
    srcOffset += srcIndex * src.strides[srcAddDim];

    T val = src.data[srcOffset] * alpha;
    op(dst.data, dstOffset, dstNumel, &val);
  }
}

// func in Indexing2.cu
template <typename T, typename IndicesType, typename IndexType, int DstDim, int SrcDim, int IdxDim>
__global__ void indexSelectSmallIndexWithoutAssert(cuda::detail::TensorInfo<T, IndexType> dst,
                                      cuda::detail::TensorInfo<const T, IndexType> src,
                                      cuda::detail::TensorInfo<const IndicesType, IndexType> indices,
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
      indices.data[cuda::detail::IndexToOffset<const IndicesType, IndexType, IdxDim>::get(dstIndex, indices)];
    // CUDA_KERNEL_ASSERT(srcIndex < srcSelectDimSize);

    // We stride over the output ignoring the indexed dimension
    // (innerSize), whose offset calculation is handled differently
    for (IndexType linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
         linearIndex < innerSize;
         linearIndex += gridDim.x * blockDim.x) {
      IndexType dstOffset =
        cuda::detail::IndexToOffset<T, IndexType, DstDim>::get(linearIndex, dst);
      dstOffset += dstIndex * dst.strides[dstSelectDim];

      IndexType srcOffset =
        cuda::detail::IndexToOffset<const T, IndexType, SrcDim>::get(linearIndex, src);
      srcOffset += srcIndex * src.strides[srcSelectDim];

      dst.data[dstOffset] = src.data[srcOffset];
    }
  }
}

template <typename T, typename IndicesType, typename IndexType, int DstDim, int SrcDim, int IdxDim,
          bool IndexIsMajor>
__global__ void indexSelectLargeIndexWithoutAssert(cuda::detail::TensorInfo<T, IndexType> dst,
                                      cuda::detail::TensorInfo<const T, IndexType> src,
                                      cuda::detail::TensorInfo<const IndicesType, IndexType> indices,
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
      indices.data[cuda::detail::IndexToOffset<const IndicesType, IndexType, IdxDim>::get(dstIndex, indices)];
    // CUDA_KERNEL_ASSERT(srcIndex < srcSelectDimSize);

    IndexType dstOffset =
      cuda::detail::IndexToOffset<T, IndexType, DstDim>::get(elementInSlice, dst);
    dstOffset += dstIndex * dst.strides[dstSelectDim];

    IndexType srcOffset =
      cuda::detail::IndexToOffset<const T, IndexType, SrcDim>::get(elementInSlice, src);
    srcOffset += srcIndex * src.strides[srcSelectDim];

    dst.data[dstOffset] = src.data[srcOffset];
  }
}

}   //index

}  //at::native