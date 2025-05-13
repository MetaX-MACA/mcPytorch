#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/native/Resize.h>
#include <ATen/native/TypeProperties.h>
#include <ATen/native/TensorShape.h>
#include <ATen/Dispatch.h>
#include <c10/core/MemoryFormat.h>
#include <c10/util/Optional.h>
#include <ATen/native/cuda/MemoryAccess.cuh>
#include <c10/core/ScalarType.h>
#include <ATen/cuda/CUDAGraphsUtils.cuh>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/cat_native.h>
#include <ATen/ops/copy_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/narrow.h>
#endif

namespace at::native {

#ifdef USE_MACA
constexpr int CAT_ARRAY_BATCH_SIZE = 1024;
#else
constexpr int CAT_ARRAY_BATCH_SIZE = 128;
#endif
constexpr int CAT_ARRAY_MAX_INPUT_DIMS = 4;

constexpr int num_threads_per_block = 512;

constexpr int num_threads_per_block_no_partial = 128;
namespace {

inline bool getCatGrid(ptrdiff_t nTensors, dim3& grid) {
  const int numSM = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;

  //X dim of grid for cat array cooperates on a single tensor in the cat.
  //Given half of the GPU, full utilization will always occur.
  grid = dim3( 2LL * numSM, (long long) nTensors );

  return true;
}

inline bool getAdaptCatGrid(ptrdiff_t nTensors, dim3& grid, int32_t max_input_numel, const int vec) {
  grid = dim3((max_input_numel + num_threads_per_block * vec - 1) / (num_threads_per_block * vec), (long long) nTensors);

  return true;
}

inline bool getNoPartialWriteCatGrid(ptrdiff_t nTensors, dim3& grid, int32_t output_numel, const int vec) {
  grid = dim3((output_numel + num_threads_per_block_no_partial * vec - 1) / (num_threads_per_block_no_partial * vec));

  return true;
}

// Similar to any other IndexToOffset calculation for copying along a given
// dimension.
template <typename IndexType, int Dims>
struct CatArrIndexToOffset {
  static inline __device__ IndexType compute(
      const IndexType tensorSize[Dims],
      const IndexType tensorStride[Dims],
      const IndexType dimSize,
      const unsigned int concatDim,
      IndexType linearIndex) {
    // linearIndex is not really linear index, but instead the offset in
    // input tensor. If the input tensor is contiguous, then this offset
    // is the linear index, but if the input tensor is channels last, then
    // it is the linear index of the permuted contiguous tensor
    IndexType offset = 0;

#pragma unroll
    for (int i = Dims - 1; i >= 1; --i) {
      IndexType curDimSize = i == concatDim ? dimSize : tensorSize[i];
      IndexType nextDimIndex = linearIndex / curDimSize;
      IndexType curDimIndex = linearIndex - curDimSize * nextDimIndex;
      IndexType curDimOffset = curDimIndex * tensorStride[i];
      offset += curDimOffset;
      linearIndex = nextDimIndex;
    }

    return offset + linearIndex * tensorStride[0];
  }
};

template<typename IndexType, unsigned int MaxDims>
struct TensorSizeStride {
  IndexType tensorSize[MaxDims];
  IndexType tensorStride[MaxDims];
};

/**
  * Kernel used to concatenated grimDim.y tensors into an output tensor. Uses a
  * grid-stride loop based off of the blockIdx.x, threadIdx.x for each input to
  * copy each element from each input tensor into the output.
  *
  * output: base pointer to the storage associated with the output tensor
  * inputs: GPU-allocated array of input metadata for each input to concatenate
  *         in the kernel
  * os: the size/stride vectors for the output tensor
  * concatDim: dimension along which we are concatenating
  * dimStride: the stride of the output tensor at the concatDim
  *
  * The most important assumption made is that the input tensors are contiguous.
  */


// pass meta data directly through kernel argument instead of pin memory
// In contiguous case, we will not need stride_size, setting it as 1 as placeholder
// to pass compile.
template <typename T, typename IndexType, int n, int stride_size>
struct CatArrInputTensorMetadata {
  T* input[n];
  IndexType offset[n];
  IndexType dimSize[n];
  IndexType nElements[n];
  bool isContiguous[n];
  TensorSizeStride<IndexType, CAT_ARRAY_MAX_INPUT_DIMS> tensorStride[stride_size];
};

template <typename T, typename IndexType, int Dims, int batch_size, int stride_size>
__global__ void CatArrayBatchedCopy(
    T* output,
    CatArrInputTensorMetadata<T, IndexType, batch_size, stride_size> inputs,
    TensorSizeStride<IndexType, CAT_ARRAY_MAX_INPUT_DIMS> os,
    const int concatDim,
    IndexType dimStride) {

    IndexType tid = blockIdx.x * blockDim.x + threadIdx.x;
    IndexType nElements = inputs.nElements[blockIdx.y];
    TensorSizeStride<IndexType, CAT_ARRAY_MAX_INPUT_DIMS> ins = stride_size > 1 ? inputs.tensorStride[blockIdx.y] : inputs.tensorStride[0];
    bool isContig = inputs.isContiguous[blockIdx.y];

    if(tid >= nElements) return;

    T* data = inputs.input[blockIdx.y];
    IndexType offset = inputs.offset[blockIdx.y];
    IndexType dimSize = inputs.dimSize[blockIdx.y];
    IndexType dataOffset = offset * dimStride;

    IndexType stride = gridDim.x * blockDim.x;

    while( tid < nElements){
      IndexType elementOffset = CatArrIndexToOffset<IndexType, Dims>::compute(
                    os.tensorSize, os.tensorStride, dimSize, concatDim, tid);
      if (isContig) {
        output[dataOffset + elementOffset] = data[tid];
      } else {
        IndexType inElementOffset = CatArrIndexToOffset<IndexType, Dims>::compute(
                    ins.tensorSize, ins.tensorStride, dimSize, concatDim, tid);
        output[dataOffset + elementOffset] = data[inElementOffset];
      }
    tid += stride;
    }
}

// Use pinned memory and and pass the struct by pointer on ROCm
template <typename T, typename IndexType>
struct CatArrInputTensor {
  T* input;
  IndexType offset;
  IndexType dimSize;
  IndexType nElements;
};

template <typename T, typename IndexType, int Dims>
C10_LAUNCH_BOUNDS_1(512)
__global__ void MACA_CatArrayBatchedCopy(
    T* output,
    CatArrInputTensor<T, IndexType>* inputs,
    TensorSizeStride<IndexType, CAT_ARRAY_MAX_INPUT_DIMS> os,
    const int concatDim,
    IndexType dimStride) {

    IndexType tid = blockIdx.x * blockDim.x + threadIdx.x;
    IndexType nElements = inputs[blockIdx.y].nElements;

    if(tid >= nElements) return;

    T* data = inputs[blockIdx.y].input;
    IndexType offset = inputs[blockIdx.y].offset;
    IndexType dimSize = inputs[blockIdx.y].dimSize;
    IndexType dataOffset = offset * dimStride;

    IndexType stride = gridDim.x * blockDim.x;

    while( tid < nElements){
    IndexType elementOffset = CatArrIndexToOffset<IndexType, Dims>::compute(
                  os.tensorSize, os.tensorStride, dimSize, concatDim, tid);
    output[dataOffset + elementOffset] = data[tid];
    tid += stride;

    }
}

template <typename T, typename IndexType, int Dims, int vec>
C10_LAUNCH_BOUNDS_1(512)
__global__ void MACA_CatArrayBatchedCopyAdaptGridSpe(
    T* output,
    CatArrInputTensor<T, IndexType>* inputs,
    TensorSizeStride<IndexType, CAT_ARRAY_MAX_INPUT_DIMS> os,
    const int concatDim,
    IndexType dimStride) {

    IndexType tid = (blockIdx.x * blockDim.x + threadIdx.x) * vec;
    IndexType nElements = inputs[blockIdx.y].nElements;

    if(tid >= nElements) return;

    T* data = inputs[blockIdx.y].input;
    IndexType offset = inputs[blockIdx.y].offset;
    IndexType dimSize = inputs[blockIdx.y].dimSize;
    IndexType dataOffset = offset * dimStride;
    using LoadT = at::native::memory::aligned_vector<T, vec>;
    T vec_input[vec];
    LoadT* p_input = reinterpret_cast<LoadT*>(&vec_input);
    *p_input = *reinterpret_cast<LoadT*>(data + tid);
    if (blockIdx.y == 1){
      #pragma unroll
      for(int i = 0;i < vec;i++){
        IndexType elementOffset = CatArrIndexToOffset<IndexType, Dims>::compute(
                    os.tensorSize, os.tensorStride, dimSize, concatDim, tid+i);
        if (i == 0 && (dataOffset + elementOffset) % 2 == 0){
          LoadT* out = reinterpret_cast<LoadT*>(output + dataOffset + elementOffset);
          *out = *p_input;
          break;
        }
        output[dataOffset + elementOffset] = vec_input[i];
      }
    }
    else{
      #pragma unroll
      for(int i = 0;i < vec;i++){
        IndexType elementOffset = CatArrIndexToOffset<IndexType, Dims>::compute(
                    os.tensorSize, os.tensorStride, dimSize, concatDim, tid+i);
        output[dataOffset + elementOffset] = vec_input[i];
      }
    }
}

template <typename T, typename IndexType, int Dims, int vec>
C10_LAUNCH_BOUNDS_1(512)
__global__ void MACA_CatArrayBatchedCopyNoPartialWrite(
    T* output,
    CatArrInputTensor<T, IndexType>* inputs,
    int64_t out_stride0,int64_t out_stride1,int64_t out_stride2,int64_t out_stride3,
    const int concatDim,
    int64_t input0_size0, int64_t input0_size1, int64_t input0_size2, int64_t input0_size3,\
    int64_t input0_stride0, int64_t input0_stride1, int64_t input0_stride2, int64_t input0_stride3,\
    int64_t input1_size0, int64_t input1_size1, int64_t input1_size2, int64_t input1_size3,\
    int64_t input1_stride0, int64_t input1_stride1, int64_t input1_stride2, int64_t input1_stride3){
    using StoreT = at::native::memory::aligned_vector<T, vec>;
    using LoadT = at::native::memory::aligned_vector<T, vec>;

    IndexType tid = blockIdx.x * blockDim.x + threadIdx.x;
    IndexType linearIndex = tid * vec;

    T output_data[vec];
    int output_coord[4];
    output_coord[3] = linearIndex / out_stride3;
    linearIndex = linearIndex % out_stride3;
    output_coord[2] = linearIndex / out_stride2;
    linearIndex = linearIndex % out_stride2;
    output_coord[1] = linearIndex / out_stride1;
    linearIndex = linearIndex % out_stride1;
    output_coord[0] = linearIndex / out_stride0;
    if (output_coord[1] >= input0_size1){
      output_coord[1] = output_coord[1] - input0_size1;
      int64_t input1_offset = output_coord[0] * input1_stride0 + output_coord[1] * input1_stride1 + output_coord[2] * input1_stride2 + output_coord[3] * input1_stride3; 
      LoadT* p_input = reinterpret_cast<LoadT*>(&output_data);
      *p_input = *reinterpret_cast<LoadT*>(inputs[1].input + input1_offset);
    } else{
      int64_t input0_offset = output_coord[0] * input0_stride0 + output_coord[1] * input0_stride1 + output_coord[2] * input0_stride2 + output_coord[3] * input0_stride3; 
      LoadT* p_input = reinterpret_cast<LoadT*>(&output_data);
      *p_input = *reinterpret_cast<LoadT*>(inputs[0].input + input0_offset);
    }

    StoreT* p_pack0 = reinterpret_cast<StoreT*>(&output_data);
    (reinterpret_cast<StoreT*>(output))[tid] = *p_pack0;
}

//MACA_CatArrayBatchedCopyNoPartialWrite_{batch_counter}_{nDims}_{concat_dim}_{memory_format}
template <typename T, typename IndexType, int Dims, int vec>
C10_LAUNCH_BOUNDS_1(512)
__global__ void MACA_CatArrayBatchedCopyNoPartialWrite_2_3_2_0(
    T* output,
    CatArrInputTensor<T, IndexType>* inputs,
    int64_t out_stride0,int64_t out_stride1,int64_t out_stride2,
    const int concatDim,
    int64_t input0_size0, int64_t input0_size1, int64_t input0_size2,
    int64_t input0_stride0, int64_t input0_stride1, int64_t input0_stride2,
    int64_t input1_size0, int64_t input1_size1, int64_t input1_size2,
    int64_t input1_stride0, int64_t input1_stride1, int64_t input1_stride2){
    using StoreT = at::native::memory::aligned_vector<T, vec>;
    using LoadT = at::native::memory::aligned_vector<T, vec>;

    IndexType tid = blockIdx.x * blockDim.x + threadIdx.x;
    IndexType linearIndex = tid * vec;

    T output_data[vec];
    int output_coord[3];
    output_coord[0] = linearIndex / out_stride0;
    linearIndex = linearIndex % out_stride0;
    output_coord[1] = linearIndex / out_stride1;
    linearIndex = linearIndex % out_stride1;
    output_coord[2] = linearIndex / out_stride2;
    linearIndex = linearIndex % out_stride2;
    if (output_coord[2] >= input0_size2){
      output_coord[2] = output_coord[2] - input0_size2;
      int64_t input1_offset = output_coord[0] * input1_stride0 + output_coord[1] * input1_stride1 + output_coord[2] * input1_stride2; 
      LoadT* p_input = reinterpret_cast<LoadT*>(&output_data);
      *p_input = *reinterpret_cast<LoadT*>(inputs[1].input + input1_offset);
    } else{
      int64_t input0_offset = output_coord[0] * input0_stride0 + output_coord[1] * input0_stride1 + output_coord[2] * input0_stride2; 
      LoadT* p_input = reinterpret_cast<LoadT*>(&output_data);
      *p_input = *reinterpret_cast<LoadT*>(inputs[0].input + input0_offset);
    }

    StoreT* p_pack0 = reinterpret_cast<StoreT*>(&output_data);
    (reinterpret_cast<StoreT*>(output))[tid] = *p_pack0;
}

//MACA_CatArrayBatchedCopyNoPartialWrite_{batch_counter}_{nDims}_{concat_dim}_{memory_format}
template <typename T, typename IndexType, int Dims, int vec>
C10_LAUNCH_BOUNDS_1(512)
__global__ void MACA_CatArrayBatchedCopyNoPartialWrite_3_4_3_0(
    T* output,
    CatArrInputTensor<T, IndexType>* inputs,
    int64_t out_stride0,int64_t out_stride1,int64_t out_stride2,int64_t out_stride3,
    const int concatDim,
    int64_t input0_size0, int64_t input0_size1, int64_t input0_size2, int64_t input0_size3,\
    int64_t input0_stride0, int64_t input0_stride1, int64_t input0_stride2, int64_t input0_stride3,\
    int64_t input1_size0, int64_t input1_size1, int64_t input1_size2, int64_t input1_size3,\
    int64_t input1_stride0, int64_t input1_stride1, int64_t input1_stride2, int64_t input1_stride3,\
    int64_t input2_size0, int64_t input2_size1, int64_t input2_size2, int64_t input2_size3,\
    int64_t input2_stride0, int64_t input2_stride1, int64_t input2_stride2, int64_t input2_stride3){
    using StoreT = at::native::memory::aligned_vector<T, vec>;
    using LoadT = at::native::memory::aligned_vector<T, vec>;

    IndexType tid = blockIdx.x * blockDim.x + threadIdx.x;
    IndexType linearIndex = tid * vec;

    T output_data[vec];
    int output_coord[4];
    output_coord[0] = linearIndex / out_stride0;
    linearIndex = linearIndex % out_stride0;
    output_coord[1] = linearIndex / out_stride1;
    linearIndex = linearIndex % out_stride1;
    output_coord[2] = linearIndex / out_stride2;
    linearIndex = linearIndex % out_stride2;
    output_coord[3] = linearIndex / out_stride3;
    linearIndex = linearIndex % out_stride3;
    LoadT* p_input = reinterpret_cast<LoadT*>(&output_data);
    if (output_coord[3] >= (input0_size3 + input1_size3)){
      output_coord[3] = output_coord[3] - (input0_size3 + input1_size3);
      int64_t input2_offset = output_coord[0] * input2_stride0 + output_coord[1] * input2_stride1 + output_coord[2] * input2_stride2 + output_coord[3] * input2_stride3;
      *p_input = *reinterpret_cast<LoadT*>(inputs[2].input + input2_offset);
    } 
    else if (output_coord[3] >= (input0_size3)){
      output_coord[3] = output_coord[3] - input0_size3;
      int64_t input1_offset = output_coord[0] * input1_stride0 + output_coord[1] * input1_stride1 + output_coord[2] * input1_stride2 + output_coord[3] * input1_stride3;
      *p_input = *reinterpret_cast<LoadT*>(inputs[1].input + input1_offset);
    }
    else{
      int64_t input0_offset = output_coord[0] * input0_stride0 + output_coord[1] * input0_stride1 + output_coord[2] * input0_stride2 + output_coord[3] * input0_stride3; 
      *p_input = *reinterpret_cast<LoadT*>(inputs[0].input + input0_offset);
    }

    StoreT* p_pack0 = reinterpret_cast<StoreT*>(&output_data);
    (reinterpret_cast<StoreT*>(output))[tid] = *p_pack0;
}



template <typename T, typename IndexType, int Dims, int vec>
C10_LAUNCH_BOUNDS_1(512)
__global__ void MACA_CatArrayBatchedCopyAdaptGrid(
    T* output,
    CatArrInputTensor<T, IndexType>* inputs,
    TensorSizeStride<IndexType, CAT_ARRAY_MAX_INPUT_DIMS> os,
    const int concatDim,
    IndexType dimStride) {

    IndexType tid = (blockIdx.x * blockDim.x + threadIdx.x) * vec;
    IndexType nElements = inputs[blockIdx.y].nElements;

    if(tid >= nElements) return;

    T* data = inputs[blockIdx.y].input;
    IndexType offset = inputs[blockIdx.y].offset;
    IndexType dimSize = inputs[blockIdx.y].dimSize;
    IndexType dataOffset = offset * dimStride;
    using LoadT = at::native::memory::aligned_vector<T, vec>;
    T vec_input[vec];
    LoadT* p_input = reinterpret_cast<LoadT*>(&vec_input);
    *p_input = *reinterpret_cast<LoadT*>(data + tid);
    #pragma unroll
    for(int i = 0;i < vec;i++){
      IndexType elementOffset = CatArrIndexToOffset<IndexType, Dims>::compute(
                  os.tensorSize, os.tensorStride, dimSize, concatDim, tid+i);
      output[dataOffset + elementOffset] = vec_input[i];
    }
}

template <typename scalar_t>
void maca_parallel_cat(const Tensor &out, const MaterializedITensorListRef &inputs, int64_t dimension,
                  int nDims, c10::MemoryFormat memory_format) {
  // First, let's set up our kernel parameters. We start with a raw pointer to
  // the storage for the output Tensor.
  scalar_t *data = out.data_ptr<scalar_t>();

  // Kernel Parameter
  long tensorMetadataSize =
    sizeof(CatArrInputTensor<scalar_t, unsigned int>) * CAT_ARRAY_BATCH_SIZE;
  auto d_inputs_storage = at::empty(
    {tensorMetadataSize}, out.options().dtype(at::kByte));
  auto d_inputs = static_cast<CatArrInputTensor<scalar_t, unsigned int> *>(
    d_inputs_storage.data_ptr());

  TensorSizeStride<unsigned int, CAT_ARRAY_MAX_INPUT_DIMS> outputParam;

  // Next, let's initialize the size, stride arrays for the output Tensor.
  if (memory_format == c10::MemoryFormat::Contiguous) {
    for (int i = 0; i < nDims; ++i) {
      outputParam.tensorSize[i] = out.size(i);
      outputParam.tensorStride[i] = out.stride(i);
    }
  } else if (memory_format == c10::MemoryFormat::ChannelsLast || memory_format == c10::MemoryFormat::ChannelsLast3d) {
    // permute the semantics of dims from NCHW to NHWC so that the input
    // tensor is now contiguous
    outputParam.tensorSize[0] = out.size(0);
    outputParam.tensorStride[0] = out.stride(0);
    for (int i = 1; i < nDims - 1; ++i) {
      outputParam.tensorSize[i] = out.size(i + 1);
      outputParam.tensorStride[i] = out.stride(i + 1);
    }
    outputParam.tensorSize[nDims - 1] = out.size(1);
    outputParam.tensorStride[nDims - 1] = out.stride(1);
  } else {
    TORCH_CHECK(false, "unsupported memory format");
  }

  at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();

  // Now we loop
  int batchCounter = 0;
  int64_t offset = 0;
  for (int i = 0; i < inputs.size() ; i += CAT_ARRAY_BATCH_SIZE) {
    int32_t max_input_numel = 0;
    // Re-allocate stackInputs every iteration to avoid read-after-write hazard
    {
      auto stackInputs_storage = at::empty({tensorMetadataSize},
          out.options().dtype(at::kByte).device(at::kCPU).pinned_memory(true));
      auto stackInputs =
        static_cast<CatArrInputTensor<scalar_t, unsigned int> *>(
          stackInputs_storage.data_ptr());
      for (batchCounter = 0;
           batchCounter < CAT_ARRAY_BATCH_SIZE &&
             (i+batchCounter) < inputs.size();
           ++batchCounter) {
        int64_t dimSize = 0;
        // There is a legacy case where a 1-D empty tensor can be concat with
        // high-dimensional tensor
        if (inputs[i+batchCounter].get().numel() > 0) {
          if (inputs[i+batchCounter].get().numel() > max_input_numel){
            max_input_numel = inputs[i+batchCounter].get().numel();
          }
          dimSize = inputs[i+batchCounter].get().size(dimension);
        }

        stackInputs[batchCounter].input =
          inputs[i+batchCounter].get().data_ptr<scalar_t>();
        stackInputs[batchCounter].offset = offset;
        stackInputs[batchCounter].dimSize = dimSize;
        stackInputs[batchCounter].nElements = inputs[i+batchCounter].get().numel();

        // update offset
        offset += dimSize;
      }
      at::native::copy_(d_inputs_storage, stackInputs_storage,
                        /* non_blocking= */ true);
    }

    // Next, let's consider how we set our kernel launch parameters.
    // We borrow from THCApply, which the kernel's internal indexing
    // is based on.
    dim3 applyBlock = dim3(num_threads_per_block);

    //Get grid where x dim fills half gpu and y dim is number of tensors.
    //This will have cating two tensors fill the entire grid, but prevent
    //many threads from needlessly load meta data if their sizes is small.
    dim3 catGrid;
    if (memory_format != c10::MemoryFormat::Contiguous) {
      switch (dimension) {
      case 0:
        break;
      case 1:
        dimension = nDims - dimension;
        break;
      default:
        dimension--;
      }
    }
    bool aligned = true;
    bool can_vectorize4 = true;
    bool can_vectorize8 = true;
    const int vec8 = 8;
    const int vec4 = 4;
    const int vec2 = 2;
    for (int i = 0; i < batchCounter; i++){
      uint64_t addr = reinterpret_cast<uint64_t>(inputs[i].get().data_ptr<scalar_t>());
      aligned = aligned && (addr % 4 == 0);
      can_vectorize4 = can_vectorize4 && (inputs[i].get().numel() % vec4 == 0);
      can_vectorize8 = can_vectorize8 && (inputs[i].get().numel() % vec8 == 0);
    }
    bool disable_cat = at::maca::get_maca_disable_cat();
    if (!disable_cat && dimension == 3 && batchCounter == 2 && memory_format == c10::MemoryFormat::ChannelsLast && nDims == 4 && out.numel() % (num_threads_per_block_no_partial * vec8) == 0 && inputs[0].get().size(1) % vec8 == 0 && inputs[1].get().size(1) % vec8 == 0){
      std::vector<int64_t> input0_size{inputs[0].get().size(0), inputs[0].get().size(1),inputs[0].get().size(2),inputs[0].get().size(3)};
      std::vector<int64_t> input0_stride{inputs[0].get().stride(0),inputs[0].get().stride(1),inputs[0].get().stride(2),inputs[0].get().stride(3)};
      std::vector<int64_t> input1_size{inputs[1].get().size(0), inputs[1].get().size(1),inputs[1].get().size(2),inputs[1].get().size(3)};
      std::vector<int64_t> input1_stride{inputs[1].get().stride(0),inputs[1].get().stride(1),inputs[1].get().stride(2),inputs[1].get().stride(3)};
      std::vector<int64_t> out_stride{out.stride(0), out.stride(1),out.stride(2),out.stride(3)};
      applyBlock = dim3(num_threads_per_block_no_partial);
      getNoPartialWriteCatGrid(batchCounter, catGrid, out.numel(), vec8);
      MACA_CatArrayBatchedCopyNoPartialWrite<scalar_t, unsigned int, 4, vec8><<<catGrid, applyBlock, 0, stream.stream()>>>(data, d_inputs, out_stride[0],out_stride[1],out_stride[2],out_stride[3], 1, \
      input0_size[0], input0_size[1], input0_size[2], input0_size[3],\
      input0_stride[0], input0_stride[1], input0_stride[2], input0_stride[3],\
      input1_size[0], input1_size[1], input1_size[2], input1_size[3],\
      input1_stride[0], input1_stride[1], input1_stride[2], input1_stride[3]);
    }
    else if (!disable_cat && dimension == 3 && batchCounter == 2 && memory_format == c10::MemoryFormat::ChannelsLast && nDims == 4 && out.numel() % (num_threads_per_block_no_partial * vec4) == 0 && inputs[0].get().size(1) % vec4 == 0 && inputs[1].get().size(1) % vec4 == 0){
      std::vector<int64_t> input0_size{inputs[0].get().size(0), inputs[0].get().size(1),inputs[0].get().size(2),inputs[0].get().size(3)};
      std::vector<int64_t> input0_stride{inputs[0].get().stride(0),inputs[0].get().stride(1),inputs[0].get().stride(2),inputs[0].get().stride(3)};
      std::vector<int64_t> input1_size{inputs[1].get().size(0), inputs[1].get().size(1),inputs[1].get().size(2),inputs[1].get().size(3)};
      std::vector<int64_t> input1_stride{inputs[1].get().stride(0),inputs[1].get().stride(1),inputs[1].get().stride(2),inputs[1].get().stride(3)};
      std::vector<int64_t> out_stride{out.stride(0), out.stride(1),out.stride(2),out.stride(3)};
      applyBlock = dim3(num_threads_per_block_no_partial);
      getNoPartialWriteCatGrid(batchCounter, catGrid, out.numel(), vec4);
      MACA_CatArrayBatchedCopyNoPartialWrite<scalar_t, unsigned int, 4, vec4><<<catGrid, applyBlock, 0, stream.stream()>>>(data, d_inputs, out_stride[0],out_stride[1],out_stride[2],out_stride[3], 1, \
      input0_size[0], input0_size[1], input0_size[2], input0_size[3],\
      input0_stride[0], input0_stride[1], input0_stride[2], input0_stride[3],\
      input1_size[0], input1_size[1], input1_size[2], input1_size[3],\
      input1_stride[0], input1_stride[1], input1_stride[2], input1_stride[3]);
    }
    else if (!disable_cat && dimension == 3 && batchCounter == 2 && memory_format == c10::MemoryFormat::ChannelsLast && nDims == 4 && out.numel() % (num_threads_per_block_no_partial * vec2) == 0 && inputs[0].get().size(1) % vec2 == 0 && inputs[1].get().size(1) % vec2 == 0){
      std::vector<int64_t> input0_size{inputs[0].get().size(0), inputs[0].get().size(1),inputs[0].get().size(2),inputs[0].get().size(3)};
      std::vector<int64_t> input0_stride{inputs[0].get().stride(0),inputs[0].get().stride(1),inputs[0].get().stride(2),inputs[0].get().stride(3)};
      std::vector<int64_t> input1_size{inputs[1].get().size(0), inputs[1].get().size(1),inputs[1].get().size(2),inputs[1].get().size(3)};
      std::vector<int64_t> input1_stride{inputs[1].get().stride(0),inputs[1].get().stride(1),inputs[1].get().stride(2),inputs[1].get().stride(3)};
      std::vector<int64_t> out_stride{out.stride(0), out.stride(1),out.stride(2),out.stride(3)};
      applyBlock = dim3(num_threads_per_block_no_partial);
      getNoPartialWriteCatGrid(batchCounter, catGrid, out.numel(), vec2);
      MACA_CatArrayBatchedCopyNoPartialWrite<scalar_t, unsigned int, 4, vec2><<<catGrid, applyBlock, 0, stream.stream()>>>(data, d_inputs, out_stride[0],out_stride[1],out_stride[2],out_stride[3], 1, \
      input0_size[0], input0_size[1], input0_size[2], input0_size[3],\
      input0_stride[0], input0_stride[1], input0_stride[2], input0_stride[3],\
      input1_size[0], input1_size[1], input1_size[2], input1_size[3],\
      input1_stride[0], input1_stride[1], input1_stride[2], input1_stride[3]);
    }
    else if (!disable_cat && dimension == 2 && batchCounter == 2 && memory_format == c10::MemoryFormat::Contiguous && nDims == 3 && out.numel() % (num_threads_per_block_no_partial * vec8) == 0 && inputs[0].get().size(2) % vec8 == 0 && inputs[1].get().size(2) % vec8 == 0){
      std::vector<int64_t> input0_size{inputs[0].get().size(0), inputs[0].get().size(1),inputs[0].get().size(2)};
      std::vector<int64_t> input0_stride{inputs[0].get().stride(0),inputs[0].get().stride(1),inputs[0].get().stride(2)};
      std::vector<int64_t> input1_size{inputs[1].get().size(0), inputs[1].get().size(1),inputs[1].get().size(2)};
      std::vector<int64_t> input1_stride{inputs[1].get().stride(0),inputs[1].get().stride(1),inputs[1].get().stride(2)};
      std::vector<int64_t> out_stride{out.stride(0), out.stride(1),out.stride(2)};
      applyBlock = dim3(num_threads_per_block_no_partial);
      getNoPartialWriteCatGrid(batchCounter, catGrid, out.numel(), vec8);
      MACA_CatArrayBatchedCopyNoPartialWrite_2_3_2_0<scalar_t, unsigned int, 3, vec8><<<catGrid, applyBlock, 0, stream.stream()>>>(data, d_inputs, out_stride[0],out_stride[1],out_stride[2], 2, \
      input0_size[0], input0_size[1], input0_size[2],\
      input0_stride[0], input0_stride[1], input0_stride[2],\
      input1_size[0], input1_size[1], input1_size[2],\
      input1_stride[0], input1_stride[1], input1_stride[2]);
    }
    else if (!disable_cat && dimension == 3 && batchCounter == 3 && memory_format == c10::MemoryFormat::Contiguous && nDims == 4 && out.numel() % (num_threads_per_block_no_partial * vec8) == 0 && inputs[0].get().size(3) % vec8 == 0 && inputs[1].get().size(3) % vec8 == 0 && inputs[2].get().size(3) % vec8 == 0){
      std::vector<int64_t> input0_size{inputs[0].get().size(0), inputs[0].get().size(1),inputs[0].get().size(2),inputs[0].get().size(3)};
      std::vector<int64_t> input0_stride{inputs[0].get().stride(0),inputs[0].get().stride(1),inputs[0].get().stride(2),inputs[0].get().stride(3)};
      std::vector<int64_t> input1_size{inputs[1].get().size(0), inputs[1].get().size(1),inputs[1].get().size(2),inputs[1].get().size(3)};
      std::vector<int64_t> input1_stride{inputs[1].get().stride(0),inputs[1].get().stride(1),inputs[1].get().stride(2),inputs[1].get().stride(3)};
      std::vector<int64_t> input2_size{inputs[2].get().size(0), inputs[2].get().size(1),inputs[2].get().size(2),inputs[2].get().size(3)};
      std::vector<int64_t> input2_stride{inputs[2].get().stride(0),inputs[2].get().stride(1),inputs[2].get().stride(2),inputs[2].get().stride(3)};
      std::vector<int64_t> out_stride{out.stride(0), out.stride(1),out.stride(2),out.stride(3)};
      applyBlock = dim3(num_threads_per_block_no_partial);
      getNoPartialWriteCatGrid(batchCounter, catGrid, out.numel(), vec8);
      MACA_CatArrayBatchedCopyNoPartialWrite_3_4_3_0<scalar_t, unsigned int, 4, vec8><<<catGrid, applyBlock, 0, stream.stream()>>>(data, d_inputs, out_stride[0],out_stride[1],out_stride[2],out_stride[3], 3, \
      input0_size[0], input0_size[1], input0_size[2], input0_size[3],\
      input0_stride[0], input0_stride[1], input0_stride[2], input0_stride[3],\
      input1_size[0], input1_size[1], input1_size[2], input1_size[3],\
      input1_stride[0], input1_stride[1], input1_stride[2], input1_stride[3],\
      input2_size[0], input2_size[1], input2_size[2], input2_size[3],\
      input2_stride[0], input2_stride[1], input2_stride[2], input2_stride[3]);
    }
    else if (sizeof(scalar_t) == 2 && inputs.size() == 2 && dimension == 3 && inputs[0].get().size(3) == 1 && inputs[1].get().size(3) == 720 && !disable_cat && aligned && can_vectorize8){
      getAdaptCatGrid(batchCounter, catGrid, max_input_numel, vec8);
      #define HANDLE_CASE_OPT(DIMS) \
      MACA_CatArrayBatchedCopyAdaptGridSpe<scalar_t, unsigned int, DIMS, vec8><<<\
          catGrid, applyBlock, 0, stream.stream()>>>(\
              data, d_inputs, outputParam, dimension, outputParam.tensorStride[dimension]); \
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      switch (nDims) {
        case 1:
          HANDLE_CASE_OPT(1);
          break;
        case 2:
          HANDLE_CASE_OPT(2);
          break;
        case 3:
          HANDLE_CASE_OPT(3);
          break;
        case 4:
          HANDLE_CASE_OPT(4);
          break;
      }
      #undef HANDLE_CASE_OPT
    }
    else if (!disable_cat && aligned && can_vectorize8){
      getAdaptCatGrid(batchCounter, catGrid, max_input_numel, vec8);
      #define HANDLE_CASE_OPT(DIMS) \
      MACA_CatArrayBatchedCopyAdaptGrid<scalar_t, unsigned int, DIMS, vec8><<<\
          catGrid, applyBlock, 0, stream.stream()>>>(\
              data, d_inputs, outputParam, dimension, outputParam.tensorStride[dimension]); \
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      switch (nDims) {
        case 1:
          HANDLE_CASE_OPT(1);
          break;
        case 2:
          HANDLE_CASE_OPT(2);
          break;
        case 3:
          HANDLE_CASE_OPT(3);
          break;
        case 4:
          HANDLE_CASE_OPT(4);
          break;
      }
      #undef HANDLE_CASE_OPT
    }
    else if (!disable_cat && aligned && can_vectorize4){
      getAdaptCatGrid(batchCounter, catGrid, max_input_numel, vec4);
      #define HANDLE_CASE_OPT(DIMS) \
      MACA_CatArrayBatchedCopyAdaptGrid<scalar_t, unsigned int, DIMS, vec4><<<\
          catGrid, applyBlock, 0, stream.stream()>>>(\
              data, d_inputs, outputParam, dimension, outputParam.tensorStride[dimension]); \
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      switch (nDims) {
        case 1:
          HANDLE_CASE_OPT(1);
          break;
        case 2:
          HANDLE_CASE_OPT(2);
          break;
        case 3:
          HANDLE_CASE_OPT(3);
          break;
        case 4:
          HANDLE_CASE_OPT(4);
          break;
      }
      #undef HANDLE_CASE_OPT
    }
    else{
      getCatGrid(batchCounter, catGrid);
      // Template Declarations for dim = 1, 2, 3, 4
      #define HANDLE_CASE(DIMS) \
      MACA_CatArrayBatchedCopy<scalar_t, unsigned int, DIMS><<<\
          catGrid, applyBlock, 0, stream.stream()>>>(\
              data, d_inputs, outputParam, dimension, outputParam.tensorStride[dimension]); \
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      switch (nDims) {
        case 1:
          HANDLE_CASE(1);
          break;
        case 2:
          HANDLE_CASE(2);
          break;
        case 3:
          HANDLE_CASE(3);
          break;
        case 4:
          HANDLE_CASE(4);
          break;
      }
      #undef HANDLE_CASE
    }
  }
}

template <typename scalar_t>
__global__ void cat_for_dim5_copy_contiguous(scalar_t* dst, scalar_t* src0, scalar_t* src1, int64_t N){
  using LoadT = at::native::memory::aligned_vector<scalar_t, 2>;
  using StoreT = at::native::memory::aligned_vector<scalar_t, 4>;
  int offset = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
  if (offset >= N){
    return;
  }
  src0 += offset;
  src1 += offset;
  dst += offset * 2;
  scalar_t src0_arr[2];
  scalar_t src1_arr[2];
  LoadT* src_pack0 = reinterpret_cast<LoadT*>(src0_arr);
  *src_pack0 = (reinterpret_cast<LoadT*>(src0))[0];
  LoadT* src_pack1 = reinterpret_cast<LoadT*>(src1_arr);
  *src_pack1 = (reinterpret_cast<LoadT*>(src1))[0];
  scalar_t dst_arr[4];
  StoreT* dst_pack = reinterpret_cast<StoreT*>(&dst_arr);
  dst_arr[0] = src0_arr[0];
  dst_arr[1] = src1_arr[0];
  dst_arr[2] = src0_arr[1];
  dst_arr[3] = src1_arr[1];
  (reinterpret_cast<StoreT*>(dst))[0] = *dst_pack;
  return;
}

template <typename scalar_t>
__global__ void cat_for_dim5_copy_non_contiguous_same_inputs_stride(scalar_t* dst, scalar_t* src0, scalar_t* src1, int size0, int size1, int size2, int size3, int size4, int stride00, int stride10, int stride20, int stride30, int stride40, int64_t N){
  using StoreT = at::native::memory::aligned_vector<scalar_t, 2>;
  int linear_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (linear_idx >= N){
    return;
  }
  dst += linear_idx * 2;
  int offsets = 0;
  auto divmod_div = linear_idx / size4;
  auto divmod_mod = linear_idx % size4;
  linear_idx = divmod_div;
  offsets += divmod_mod * stride40;

  divmod_div = linear_idx / size3;
  divmod_mod = linear_idx % size3;
  linear_idx = divmod_div;
  offsets += divmod_mod * stride30;

  divmod_div = linear_idx / size2;
  divmod_mod = linear_idx % size2;
  linear_idx = divmod_div;
  offsets += divmod_mod * stride20;

  divmod_div = linear_idx / size1;
  divmod_mod = linear_idx % size1;
  linear_idx = divmod_div;
  offsets += divmod_mod * stride10;

  divmod_div = linear_idx / size0;
  divmod_mod = linear_idx % size0;
  linear_idx = divmod_div;
  offsets += divmod_mod * stride00;
  scalar_t dst_arr[2];
  StoreT* dst_pack = reinterpret_cast<StoreT*>(&dst_arr);
  dst_arr[0] = src0[offsets];
  dst_arr[1] = src1[offsets];
  (reinterpret_cast<StoreT*>(dst))[0] = *dst_pack;
  return;
}

template <typename scalar_t>
__global__ void cat_for_dim5_copy_non_contiguous_not_same_inputs_stride(scalar_t* dst, scalar_t* src0, scalar_t* src1, int size0, int size1, int size2, int size3, int size4, int stride00, int stride10, int stride20, int stride30, int stride40, int stride01, int stride11, int stride21, int stride31, int stride41, int64_t N){
  using StoreT = at::native::memory::aligned_vector<scalar_t, 2>;
  int linear_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (linear_idx >= N){
    return;
  }
  dst += linear_idx * 2;
  int offsets0 = 0;
  int offsets1 = 0;
  auto divmod_div = linear_idx / size4;
  auto divmod_mod = linear_idx % size4;
  linear_idx = divmod_div;
  offsets0 += divmod_mod * stride40;
  offsets1 += divmod_mod * stride41;

  divmod_div = linear_idx / size3;
  divmod_mod = linear_idx % size3;
  linear_idx = divmod_div;
  offsets0 += divmod_mod * stride30;
  offsets1 += divmod_mod * stride31;

  divmod_div = linear_idx / size2;
  divmod_mod = linear_idx % size2;
  linear_idx = divmod_div;
  offsets0 += divmod_mod * stride20;
  offsets1 += divmod_mod * stride21;

  divmod_div = linear_idx / size1;
  divmod_mod = linear_idx % size1;
  linear_idx = divmod_div;
  offsets0 += divmod_mod * stride10;
  offsets1 += divmod_mod * stride11;

  divmod_div = linear_idx / size0;
  divmod_mod = linear_idx % size0;
  linear_idx = divmod_div;
  offsets0 += divmod_mod * stride00;
  offsets1 += divmod_mod * stride01;
  scalar_t dst_arr[2];
  StoreT* dst_pack = reinterpret_cast<StoreT*>(&dst_arr);
  dst_arr[0] = src0[offsets0];
  dst_arr[1] = src1[offsets1];
  (reinterpret_cast<StoreT*>(dst))[0] = *dst_pack;
  return;
}

template <typename scalar_t>
void maca_parallel_cat_coalesce_write_copy_contiguous(const Tensor &out, const MaterializedITensorListRef &inputs, int64_t dimension,
                  int nDims, c10::MemoryFormat memory_format){
    dim3 threads(512);
    int block_work_size = threads.x * 2;
    dim3 blocks = (inputs[0].get().numel() + block_work_size - 1) / block_work_size;
    auto output_ptr = out.data_ptr<scalar_t>();
    auto input0_ptr = inputs[0].get().data_ptr<scalar_t>();
    auto input1_ptr = inputs[1].get().data_ptr<scalar_t>();
    cat_for_dim5_copy_contiguous<scalar_t><<<blocks, threads>>>(output_ptr, input0_ptr, input1_ptr, inputs[0].get().numel());
    return;
}

template <typename scalar_t>
void maca_parallel_cat_coalesce_write_copy_non_contiguous(const Tensor &out, const MaterializedITensorListRef &inputs, int64_t dimension,
                  int nDims, c10::MemoryFormat memory_format){
    dim3 threads(512);
    int block_work_size = threads.x;
    dim3 blocks = (inputs[0].get().numel() + block_work_size - 1) / block_work_size;
    auto output_ptr = out.data_ptr<scalar_t>();
    auto input0_ptr = inputs[0].get().data_ptr<scalar_t>();
    auto input1_ptr = inputs[1].get().data_ptr<scalar_t>();
    int size0 = inputs[0].get().size(0);
    int size1 = inputs[0].get().size(1);
    int size2 = inputs[0].get().size(2);
    int size3 = inputs[0].get().size(3);
    int size4 = inputs[0].get().size(4);
    int stride00 = inputs[0].get().stride(0);
    int stride10 = inputs[0].get().stride(1);
    int stride20 = inputs[0].get().stride(2);
    int stride30 = inputs[0].get().stride(3);
    int stride40 = inputs[0].get().stride(4);
    int stride01 = inputs[1].get().stride(0);
    int stride11 = inputs[1].get().stride(1);
    int stride21 = inputs[1].get().stride(2);
    int stride31 = inputs[1].get().stride(3);
    int stride41 = inputs[1].get().stride(4);
    if (stride00 == stride01 && stride10 == stride11 && stride20 == stride21 && stride30 == stride31 && stride40 == stride41){
      cat_for_dim5_copy_non_contiguous_same_inputs_stride<<<blocks, threads>>>(output_ptr, input0_ptr, input1_ptr, size0, size1, size2, size3, size4, stride00, stride10, stride20, stride30, stride40, inputs[0].get().numel());
    } else{
      cat_for_dim5_copy_non_contiguous_not_same_inputs_stride<<<blocks, threads>>>(output_ptr, input0_ptr, input1_ptr, size0, size1, size2, size3, size4, stride00, stride10, stride20, stride30, stride40, stride01, stride11, stride21, stride31, stride41, inputs[0].get().numel());
    }
    return;
}

template <typename scalar_t, int batch_size, int stride_size>
void parallel_cat(const Tensor &out, const MaterializedITensorListRef& inputs, int64_t dimension,
                  int nDims, c10::MemoryFormat memory_format) {
  // First, let's set up our kernel parameters. We start with a raw pointer to
  // the storage for the output Tensor.
  scalar_t *data = out.data_ptr<scalar_t>();
  CatArrInputTensorMetadata<scalar_t, unsigned int, batch_size, stride_size> catMetaData;
  TensorSizeStride<unsigned int, CAT_ARRAY_MAX_INPUT_DIMS> outputParam;

  // Next, let's initialize the size, stride arrays for the output Tensor.
  if (memory_format == c10::MemoryFormat::Contiguous) {
    for (int i = 0; i < nDims; ++i) {
      outputParam.tensorSize[i] = out.size(i);
      outputParam.tensorStride[i] = out.stride(i);
    }
  } else if (memory_format == c10::MemoryFormat::ChannelsLast || memory_format == c10::MemoryFormat::ChannelsLast3d) {
    // permute the semantics of dims from NCHW to NHWC so that the input
    // tensor is now contiguous
    outputParam.tensorSize[0] = out.size(0);
    outputParam.tensorStride[0] = out.stride(0);
    for (int i = 1; i < nDims - 1; ++i) {
      outputParam.tensorSize[i] = out.size(i + 1);
      outputParam.tensorStride[i] = out.stride(i + 1);
    }
    outputParam.tensorSize[nDims - 1] = out.size(1);
    outputParam.tensorStride[nDims - 1] = out.stride(1);
  } else {
    TORCH_CHECK(false, "unsupported memory format");
  }

  at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();

  // Now we loop
  int batchCounter = 0;
  int64_t offset = 0;
  for (int i = 0; i < inputs.size() ; i += batch_size) {
    for (batchCounter = 0;
          batchCounter < batch_size &&
            (i+batchCounter) < inputs.size();
          ++batchCounter) {
      int64_t dimSize = 0;
      // There is a legacy case where a 1-D empty tensor can be concat with
      // high-dimensional tensor
      if (inputs[i+batchCounter].get().numel() > 0) {
        dimSize = inputs[i+batchCounter].get().size(dimension);
      }
      catMetaData.input[batchCounter] = inputs[i+batchCounter].get().data_ptr<scalar_t>();
      catMetaData.offset[batchCounter] = offset;
      catMetaData.dimSize[batchCounter] = dimSize;
      catMetaData.nElements[batchCounter] = inputs[i+batchCounter].get().numel();
      if (stride_size > 1) {
        auto strides = inputs[i+batchCounter].get().strides();
        auto sizes = inputs[i+batchCounter].get().sizes();
        for(int j = 0; j < nDims; j++){
          catMetaData.tensorStride[batchCounter].tensorSize[j] = sizes[j];
          catMetaData.tensorStride[batchCounter].tensorStride[j] = strides[j];
        }
        catMetaData.isContiguous[batchCounter] = false;
      } else {
        catMetaData.isContiguous[batchCounter] = true;
      }
      // update offset
      offset += dimSize;
    }
    // Next, let's consider how we set our kernel launch parameters.
    // We borrow from THCApply, which the kernel's internal indexing
    // is based on.
    dim3 applyBlock = dim3(num_threads_per_block);

    //Get grid where x dim fills half gpu and y dim is number of tensors.
    //This will have cating two tensors fill the entire grid, but prevent
    //many threads from needlessly load meta data if their sizes is small.
    dim3 catGrid;
    getCatGrid(batchCounter, catGrid);

    if (memory_format != c10::MemoryFormat::Contiguous) {
      switch (dimension) {
      case 0:
        break;
      case 1:
        dimension = nDims - dimension;
        break;
      default:
        dimension--;
      }
    }
    // Template Declarations for dim = 1, 2, 3, 4
#define HANDLE_CASE(DIMS) \
    CatArrayBatchedCopy<scalar_t, unsigned int, DIMS, batch_size, stride_size><<<\
        catGrid, applyBlock, 0, stream.stream()>>>(\
            data, catMetaData, outputParam, dimension, outputParam.tensorStride[dimension]); \
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    switch (nDims) {
      case 1:
        HANDLE_CASE(1);
        break;
      case 2:
        HANDLE_CASE(2);
        break;
      case 3:
        HANDLE_CASE(3);
        break;
      case 4:
        HANDLE_CASE(4);
        break;
    }
#undef HANDLE_CASE
  }
}
} // namespace

TORCH_IMPL_FUNC(cat_out_cuda)
(const ITensorListRef& tensors,
 int64_t dim,
 int64_t valid,
 bool all_contiguous,
 bool all_same_dtype,
 bool all_same_sizes_and_stride,
 MemoryFormat memory_format,
 const Tensor& result) {
  if (result.numel() == 0) {
    return;
  }

  auto materialized = tensors.materialize();
  //remove 0 numel tensor
  for (auto it = materialized.begin(); it != materialized.end();){
    if (it->get().numel() == 0){
      it = materialized.erase(it);
    } else{
      ++it;
    }
  }
  // We parallelize the copy if all 6 conditions pass:
  //
  // 1. There is more than one input tensor
  // 2. The out tensor is 32-bit indexable
  // 3. The number of dimensions is <= 4
  // 4. All input tensors are contiguous (output tensor may be non-contig)
  // 5. All input tensors can use 32-bit indexing

  const bool all32BitIndexable = std::all_of(materialized.begin(), materialized.end(),
    [] (const Tensor& t) {
      return at::cuda::detail::canUse32BitIndexMath(t);
    });

  int nDims = materialized[valid].get().dim();
  bool print_cat_shape = at::maca::get_maca_print_cat_shape();
  if(print_cat_shape){
    printf("cat info start.\n");
    printf("materialized.size() > 1:%d, result.dim():%d, at::cuda::detail::canUse32BitIndexMath(result):%d, all_contiguous:%d, all32BitIndexable:%d, all_same_dtype:%d, concat_dim:%d \n", materialized.size() > 1,  result.dim(), at::cuda::detail::canUse32BitIndexMath(result), all_contiguous, all32BitIndexable, all_same_dtype, dim);
    int num_inputs = materialized.size();
    for (int j = 0; j < num_inputs; j++){
      printf("shape(");
      for (int64_t i = 0; i < materialized[j].get().dim(); ++i){
        printf("%d,",materialized[j].get().size(i));
      }
      printf(") stride(");
      for (int64_t i = 0; i < materialized[j].get().dim(); ++i){
        printf("%d,",materialized[j].get().stride(i));
      }
      printf(") \n");
    }
    printf("cat info end.\n");
  }
  bool disable_opt_cat_for_dim5 = at::maca::get_maca_disable_opt_cat_for_dim5();
  bool no_cudagraph_trace = at::cuda::currentStreamCaptureStatus() == at::cuda::CaptureStatus::None;
  
#ifdef USE_MACA
  if (materialized.size() > 1 &&
      result.dim() <= CAT_ARRAY_MAX_INPUT_DIMS &&
      at::cuda::detail::canUse32BitIndexMath(result) &&
      all_contiguous &&
      all32BitIndexable &&
      all_same_dtype &&
      no_cudagraph_trace) {
      AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND4(
          kComplexHalf, kHalf, kBool, kBFloat16,
          result.scalar_type(), "cat_cuda", [&]() {
        maca_parallel_cat<scalar_t>(result, materialized, dim, nDims, memory_format);
      });
    } else if (!disable_opt_cat_for_dim5 &&
        materialized.size() == 2 &&
        materialized[0].get().numel() == materialized[1].get().numel() &&
        (result.scalar_type() == at::ScalarType::Float || result.scalar_type() == at::ScalarType::Half || result.scalar_type() == at::ScalarType::BFloat16) &&
        result.dim() == 5 && dim == 4 && materialized[0].get().size(dim) == 1 &&
        (reinterpret_cast<uint64_t>(materialized[0].get().data_ptr()) % 4 == 0) && (reinterpret_cast<uint64_t>(materialized[1].get().data_ptr()) % 4 == 0) && (reinterpret_cast<uint64_t>(result.data_ptr()) % 4 == 0) &&
        at::cuda::detail::canUse32BitIndexMath(result) &&
        all_contiguous &&
        all32BitIndexable &&
        all_same_dtype
      ){
        if (result.scalar_type() == at::ScalarType::Float){
          maca_parallel_cat_coalesce_write_copy_contiguous<float>(result, materialized, dim, nDims, memory_format);
        } else if(result.scalar_type() == at::ScalarType::Half){
          maca_parallel_cat_coalesce_write_copy_contiguous<at::Half>(result, materialized, dim, nDims, memory_format);
        } else{
          maca_parallel_cat_coalesce_write_copy_contiguous<at::BFloat16>(result, materialized, dim, nDims, memory_format);
        }
      } else if(!disable_opt_cat_for_dim5 &&
        materialized.size() == 2 &&
        materialized[0].get().numel() == materialized[1].get().numel() &&
        (result.scalar_type() == at::ScalarType::Float || result.scalar_type() == at::ScalarType::Half || result.scalar_type() == at::ScalarType::BFloat16) &&
        result.dim() == 5 && dim ==4 && materialized[0].get().size(dim) == 1 &&
        reinterpret_cast<uint64_t>(result.data_ptr()) % 4 == 0 &&
        at::cuda::detail::canUse32BitIndexMath(result) &&
        !all_contiguous &&
        all32BitIndexable &&
        all_same_dtype
      ){
        if (result.scalar_type() == at::ScalarType::Float){
          maca_parallel_cat_coalesce_write_copy_non_contiguous<float>(result, materialized, dim, nDims, memory_format);
        } else if(result.scalar_type() == at::ScalarType::Half){
          maca_parallel_cat_coalesce_write_copy_non_contiguous<at::Half>(result, materialized, dim, nDims, memory_format);
        } else{
          maca_parallel_cat_coalesce_write_copy_non_contiguous<at::BFloat16>(result, materialized, dim, nDims, memory_format);
        }
#else
  // We support the contiguous inputs and non-contiguous input (<=4 dims) in different ways
  // For contiguous input, we don't need to pass stride meta data to cuda kernel through constant
  // memory. Therefore, we could pass more inputs to cuda threads.
  // For non-contiguous, we reduce the number of inputs passed to cuda kernel due to the limitation
  // of constant memory.
  if (materialized.size() > 1 &&
      result.dim() <= CAT_ARRAY_MAX_INPUT_DIMS &&
      at::cuda::detail::canUse32BitIndexMath(result) &&
      all_contiguous &&
      all32BitIndexable &&
      all_same_dtype) {
      AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND4(
          kComplexHalf, kHalf, kBool, kBFloat16,
          result.scalar_type(), "cat_cuda", [&]() {
        parallel_cat<scalar_t, CAT_ARRAY_BATCH_SIZE, 1>(result, materialized, dim, nDims, memory_format);
      });
  } else if (materialized.size() > 1 &&
      result.dim() <= CAT_ARRAY_MAX_INPUT_DIMS &&
      at::cuda::detail::canUse32BitIndexMath(result) &&
      nDims <= CAT_ARRAY_MAX_INPUT_DIMS &&
      all32BitIndexable &&
      all_same_dtype &&
      memory_format == c10::MemoryFormat::Contiguous) {
      AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND4(
          kComplexHalf, kHalf, kBool, kBFloat16,
          result.scalar_type(), "cat_cuda", [&]() {
        parallel_cat<scalar_t, CAT_ARRAY_BATCH_SIZE/2, CAT_ARRAY_BATCH_SIZE/2>(result, materialized, dim, nDims, memory_format);
      });
#endif
  } else {
    int64_t offset = 0;
    for (const Tensor& t : materialized) {
      if (cat_should_skip_tensor(t)) continue;
      int64_t dimSize = t.size(dim);
      Tensor nt = at::narrow(result, dim, offset, dimSize);
      copy_(nt, t);
      offset += dimSize;
    }
  }
}

} // namespace at::native
