#pragma once

namespace at::native {
namespace index {

template <int vec_size, typename T, typename IndicesType, typename IndexType, int DstDim, int SrcDim, int IdxDim,
          bool IndexIsMajor, typename func_t>
__global__ void vectorized_indexFuncLargeIndex_3_3_true_half(
    T* self_data, T* source_data, IndicesType* index_data,
    IndexType self_size0,       IndexType source_size0,        IndexType index_size0,
    IndexType self_size1,       IndexType source_size1,
    IndexType self_size2,       IndexType source_size2,
    IndexType self_stride0,     IndexType source_stride0,      IndexType index_stride0,
    IndexType self_stride1,     IndexType source_stride1,
    IndexType self_stride2,     IndexType source_stride2,
    IndexType dstAddStride,
    IndexType srcAddStride,
    IndexType totalSize,
    IndexType innerSize,
    int64_t dstAddDimSize,
    int64_t dstNumel,
    const func_t& op,
    T alpha
) {
  IndexType linearIndex = (blockIdx.x * blockDim.x + threadIdx.x) * vec_size;
  if (linearIndex >= totalSize) return;

  IndexType srcIndex = linearIndex / innerSize;
  IndexType elementInSlice = linearIndex % innerSize;

  IndexType offset_index = srcIndex * index_stride0;
  IndexType dstIndex = index_data[offset_index];

  IndexType curDimIndex;
  IndexType curDimOffset;

  IndexType linearId = elementInSlice;
  IndexType dstOffset = 0;
  curDimIndex = linearId % self_size2;   //dim2
  curDimOffset = curDimIndex * self_stride2;
  dstOffset += curDimOffset;
  linearId /= self_size2;
  curDimIndex = linearId % self_size1;   //dim1
  curDimOffset = curDimIndex * self_stride1;
  dstOffset += curDimOffset;
  linearId /= self_size1;
  dstOffset += linearId * self_stride0;  //dim0
  dstOffset += dstIndex * dstAddStride;  //add

  linearId = elementInSlice;
  IndexType srcOffset = 0;
  curDimIndex = linearId % source_size2;   //dim2
  curDimOffset = curDimIndex * source_stride2;
  srcOffset += curDimOffset;
  linearId /= source_size2;
  curDimIndex = linearId % source_size1;   //dim1
  curDimOffset = curDimIndex * source_stride1;
  srcOffset += curDimOffset;
  linearId /= source_size1;
  srcOffset += linearId * source_stride0;  //dim0
  srcOffset += srcIndex * srcAddStride;

  using LoadT = at::native::memory::aligned_vector<T, vec_size>; 
  LoadT val =  *(reinterpret_cast<LoadT*>(&source_data[srcOffset]));
  #pragma unroll
  for (int i = 0; i < vec_size; i++) {
    val.val[i] *= alpha;
  };

  #pragma unroll
  for (int i = 0; i < vec_size; i++) {
    op(self_data, dstOffset + i, dstNumel, &val.val[i]);
  };
}
    

template <typename T, typename IndicesType, typename IndexType, int DstDim, int SrcDim, int IdxDim,
          bool IndexIsMajor, typename func_t>
__global__ void indexFuncLargeIndex_3_3_true_half(
    T* self_data, T* source_data, IndicesType* index_data,
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

    T val = source_data[srcOffset];
    op(self_data, dstOffset, dstNumel, &val);
   }
}


template <typename T, typename IndicesType, typename IndexType, int DstDim, int SrcDim, int IdxDim>
__global__ void indexSelectSmallIndexWithoutAssert(cuda::detail::TensorInfo<T, IndexType> dst,
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
        cuda::detail::IndexToOffset<T, IndexType, SrcDim>::get(linearIndex, src);
      srcOffset += srcIndex * src.strides[srcSelectDim];

      dst.data[dstOffset] = src.data[srcOffset];
    }
  }
}

template <typename T, typename IndicesType, typename IndexType, int DstDim, int SrcDim, int IdxDim,
          bool IndexIsMajor>
__global__ void indexSelectLargeIndexWithoutAssert(cuda::detail::TensorInfo<T, IndexType> dst,
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
    // CUDA_KERNEL_ASSERT(srcIndex < srcSelectDimSize);

    IndexType dstOffset =
      cuda::detail::IndexToOffset<T, IndexType, DstDim>::get(elementInSlice, dst);
    dstOffset += dstIndex * dst.strides[dstSelectDim];

    IndexType srcOffset =
      cuda::detail::IndexToOffset<T, IndexType, SrcDim>::get(elementInSlice, src);
    srcOffset += srcIndex * src.strides[srcSelectDim];

    dst.data[dstOffset] = src.data[srcOffset];
  }
}

}   //index

}  //at::native