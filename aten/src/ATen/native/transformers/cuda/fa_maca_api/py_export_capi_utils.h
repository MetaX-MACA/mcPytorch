#ifdef USE_MACA
#pragma once

#include <flash_attn/flash_attn.h>
#include <c10/core/ScalarType.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <ATen/cuda/CUDAGraphsUtils.cuh>

#define CHECK_DEVICE(x) TORCH_CHECK(x.is_cuda(), #x " must be on CUDA")
#define CHECK_SHAPE(x, ...) TORCH_CHECK(x.sizes() == at::IntArrayRef({__VA_ARGS__}), #x " must have shape (" #__VA_ARGS__ ")")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

inline void get_philox_state(const std::optional<at::Generator> &gen_, at::Tensor &rng_state, int64_t counter_offset) {
    TORCH_CHECK(rng_state.dtype() == at::kLong, "rng_state tensor must have dtype Int64");
    TORCH_CHECK(rng_state.stride(-1) == 1, "rng_state tensor must have contiguous last dimension");
    CHECK_SHAPE(rng_state, 2);

    auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(
        gen_, at::cuda::detail::getDefaultCUDAGenerator());
    // auto gen = at::cuda::detail::getDefaultCUDAGenerator().get<at::CUDAGeneratorImpl>();

    // See Note [Acquire lock when using random generators]
    std::lock_guard<std::mutex> lock(gen->mutex_);

    auto philox_args = gen->philox_cuda_state(counter_offset);
    auto seeds = at::cuda::philox::unpack(philox_args);

    uint64_t *rng_state_p;
    rng_state_p = reinterpret_cast<uint64_t*>(rng_state.data_ptr());
    rng_state_p[0] = std::get<0>(seeds);
    rng_state_p[1] = std::get<1>(seeds);
}

inline Tensor_t convert_mcfa_tensor(const at::Tensor &at_tensor){

    auto sizes = at_tensor.sizes();
    auto dims = sizes.size();

    if(dims <= 0
        || (dims == 1 && sizes[0] == 0)) {
        return nullptr;
    }

    auto dtype = at_tensor.dtype();
    mcflashattnDataType_t mcft_dtype = MCFLASHATTN_DATATYPE_NONE;

    if(dtype == at::ScalarType::Half)        mcft_dtype = MCFLASHATTN_DATATYPE_FP16;
    else if(dtype == at::ScalarType::BFloat16)  mcft_dtype = MCFLASHATTN_DATATYPE_BF16;
    else if(dtype == at::ScalarType::Long)     mcft_dtype = MCFLASHATTN_DATATYPE_INT64;
    else if(dtype == at::ScalarType::Float)     mcft_dtype = MCFLASHATTN_DATATYPE_FP32; // param softmax_lse is float32
    else AT_ERROR(str(at_tensor.scalar_type()), " is not handle correctly!");

    auto strides = at_tensor.strides();
    if(dims == 1){
        return make_tensor1d(at_tensor.data_ptr(),mcft_dtype,sizes[0],strides[0]);
    }else if(dims == 2){
        return make_tensor2d(at_tensor.data_ptr(),mcft_dtype,sizes[0],sizes[1],strides[0],strides[1]);
    }else if(dims == 3){
        return make_tensor3d(at_tensor.data_ptr(),mcft_dtype,sizes[0],sizes[1],sizes[2],strides[0],strides[1],strides[2]);
    }else if(dims == 4){
        return make_tensor4d(at_tensor.data_ptr(),mcft_dtype,sizes[0],sizes[1],sizes[2],sizes[3],strides[0],strides[1],strides[2],strides[3]);
    }else if(dims == 5){
        return make_tensor5d(at_tensor.data_ptr(),mcft_dtype,sizes[0],sizes[1],sizes[2],sizes[3],sizes[4],
            strides[0],strides[1],strides[2],strides[3],strides[4]);
    }else {
        return nullptr;
    }
}

inline Tensor_t convert_mcfa_tensor(const std::optional<at::Tensor> &at_tensor){
    if(at_tensor.has_value()){
        return convert_mcfa_tensor(at_tensor.value());
    }
    return nullptr;
}

inline void release_mcfa_tensor(const std::initializer_list<Tensor_t> &list){
    for(Tensor_t tensor : list){
        release_tensor(tensor);
    }
}

#endif