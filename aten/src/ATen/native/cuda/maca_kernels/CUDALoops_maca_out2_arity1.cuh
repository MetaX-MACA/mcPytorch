#pragma once

template <typename func_t, typename array_t, 
          typename rets_t, typename ret0_t, typename ret1_t, typename arg0_t,
          typename index_t, typename stride_t>
C10_LAUNCH_BOUNDS_1(num_threads())
__global__ void legacy_optim_unrolled_kernel_for_multi_outputs_out2_arity1_dim1(
  int64_t N, func_t f, array_t data,
  index_t size0,
  stride_t ostride00,
  stride_t ostride01,
  stride_t istride02
){
    int64_t remaining = N - block_work_size() * blockIdx.x;
    int64_t idx = blockIdx.x;
    int thread_idx = threadIdx.x;
    int64_t block_start_index = block_work_size() * idx;

    #pragma unroll
    for(int i = 0; i < thread_work_size(); i++){
        if(thread_idx >= remaining) return;
        int64_t linear_idx = thread_idx + block_start_index;

        int64_t offsets[3];
        constexpr int NARGS = 3;
        #pragma unroll
        for (int arg = 0; arg < NARGS; arg++) {
            offsets[arg] = 0;
        }
        
        //dim0
        auto divmod_div = linear_idx / size0;
        auto divmod_mod = linear_idx % size0;
        linear_idx = divmod_div;
        offsets[0] += divmod_mod * ostride00;
        offsets[1] += divmod_mod * ostride01;
        offsets[2] += divmod_mod * istride02;
        
        //compute and save
        arg0_t arg0 = *(reinterpret_cast<arg0_t*>(data[2]) + offsets[2]);
        rets_t res = f(arg0);
        ret0_t *ret0 = reinterpret_cast<ret0_t *>(data[0]) + offsets[0];
        ret1_t *ret1 = reinterpret_cast<ret1_t *>(data[1]) + offsets[1];
        *ret0 = thrust::get<0>(res);
        *ret1 = thrust::get<1>(res);

        thread_idx += num_threads();
    }

}



template <typename func_t, typename array_t, 
          typename rets_t, typename ret0_t, typename ret1_t, typename arg0_t,
          typename index_t, typename stride_t>
C10_LAUNCH_BOUNDS_1(num_threads())
__global__ void legacy_optim_unrolled_kernel_for_multi_outputs_out2_arity1_dim2(
  int64_t N, func_t f, array_t data,
  index_t size0, index_t size1,
  stride_t ostride00, stride_t ostride10,
  stride_t ostride01, stride_t ostride11,
  stride_t istride02, stride_t istride12
){
    int64_t remaining = N - block_work_size() * blockIdx.x;
    int64_t idx = blockIdx.x;
    int thread_idx = threadIdx.x;
    int64_t block_start_index = block_work_size() * idx;

    #pragma unroll
    for(int i = 0; i < thread_work_size(); i++){
        if(thread_idx >= remaining) return;
        int64_t linear_idx = thread_idx + block_start_index;

        int64_t offsets[3];
        constexpr int NARGS = 3;
        #pragma unroll
        for (int arg = 0; arg < NARGS; arg++) {
            offsets[arg] = 0;
        }
        
        //dim0
        auto divmod_div = linear_idx / size0;
        auto divmod_mod = linear_idx % size0;
        linear_idx = divmod_div;
        offsets[0] += divmod_mod * ostride00;
        offsets[1] += divmod_mod * ostride01;
        offsets[2] += divmod_mod * istride02;
        
        //dim1
        divmod_div = linear_idx / size1;
        divmod_mod = linear_idx % size1;
        linear_idx = divmod_div;
        offsets[0] += divmod_mod * ostride10;
        offsets[1] += divmod_mod * ostride11;
        offsets[2] += divmod_mod * istride12;
        
        //compute and save
        arg0_t arg0 = *(reinterpret_cast<arg0_t*>(data[2]) + offsets[2]);
        rets_t res = f(arg0);
        ret0_t *ret0 = reinterpret_cast<ret0_t *>(data[0]) + offsets[0];
        ret1_t *ret1 = reinterpret_cast<ret1_t *>(data[1]) + offsets[1];
        *ret0 = thrust::get<0>(res);
        *ret1 = thrust::get<1>(res);

        thread_idx += num_threads();
    }

}



template <typename func_t, typename array_t, 
          typename rets_t, typename ret0_t, typename ret1_t, typename arg0_t,
          typename index_t, typename stride_t>
C10_LAUNCH_BOUNDS_1(num_threads())
__global__ void legacy_optim_unrolled_kernel_for_multi_outputs_out2_arity1_dim3(
  int64_t N, func_t f, array_t data,
  index_t size0, index_t size1, index_t size2,
  stride_t ostride00, stride_t ostride10, stride_t ostride20,
  stride_t ostride01, stride_t ostride11, stride_t ostride21,
  stride_t istride02, stride_t istride12, stride_t istride22
){
    int64_t remaining = N - block_work_size() * blockIdx.x;
    int64_t idx = blockIdx.x;
    int thread_idx = threadIdx.x;
    int64_t block_start_index = block_work_size() * idx;

    #pragma unroll
    for(int i = 0; i < thread_work_size(); i++){
        if(thread_idx >= remaining) return;
        int64_t linear_idx = thread_idx + block_start_index;

        int64_t offsets[3];
        constexpr int NARGS = 3;
        #pragma unroll
        for (int arg = 0; arg < NARGS; arg++) {
            offsets[arg] = 0;
        }
        
        //dim0
        auto divmod_div = linear_idx / size0;
        auto divmod_mod = linear_idx % size0;
        linear_idx = divmod_div;
        offsets[0] += divmod_mod * ostride00;
        offsets[1] += divmod_mod * ostride01;
        offsets[2] += divmod_mod * istride02;
        
        //dim1
        divmod_div = linear_idx / size1;
        divmod_mod = linear_idx % size1;
        linear_idx = divmod_div;
        offsets[0] += divmod_mod * ostride10;
        offsets[1] += divmod_mod * ostride11;
        offsets[2] += divmod_mod * istride12;

        //dim2
        divmod_div = linear_idx / size2;
        divmod_mod = linear_idx % size2;
        linear_idx = divmod_div;
        offsets[0] += divmod_mod * ostride20;
        offsets[1] += divmod_mod * ostride21;
        offsets[2] += divmod_mod * istride22;
        
        //compute and save
        arg0_t arg0 = *(reinterpret_cast<arg0_t*>(data[2]) + offsets[2]);
        rets_t res = f(arg0);
        ret0_t *ret0 = reinterpret_cast<ret0_t *>(data[0]) + offsets[0];
        ret1_t *ret1 = reinterpret_cast<ret1_t *>(data[1]) + offsets[1];
        *ret0 = thrust::get<0>(res);
        *ret1 = thrust::get<1>(res);

        thread_idx += num_threads();
    }

}


template <typename func_t, typename array_t, 
          typename rets_t, typename ret0_t, typename ret1_t, typename arg0_t,
          typename index_t, typename stride_t>
C10_LAUNCH_BOUNDS_1(num_threads())
__global__ void legacy_optim_unrolled_kernel_for_multi_outputs_out2_arity1_dim4(
  int64_t N, func_t f, array_t data,
  index_t size0, index_t size1, index_t size2, index_t size3,
  stride_t ostride00, stride_t ostride10, stride_t ostride20, stride_t ostride30,
  stride_t ostride01, stride_t ostride11, stride_t ostride21, stride_t ostride31,
  stride_t istride02, stride_t istride12, stride_t istride22, stride_t istride32
){
    int64_t remaining = N - block_work_size() * blockIdx.x;
    int64_t idx = blockIdx.x;
    int thread_idx = threadIdx.x;
    int64_t block_start_index = block_work_size() * idx;

    #pragma unroll
    for(int i = 0; i < thread_work_size(); i++){
        if(thread_idx >= remaining) return;
        int64_t linear_idx = thread_idx + block_start_index;

        int64_t offsets[3];
        constexpr int NARGS = 3;
        #pragma unroll
        for (int arg = 0; arg < NARGS; arg++) {
            offsets[arg] = 0;
        }
        
        //dim0
        auto divmod_div = linear_idx / size0;
        auto divmod_mod = linear_idx % size0;
        linear_idx = divmod_div;
        offsets[0] += divmod_mod * ostride00;
        offsets[1] += divmod_mod * ostride01;
        offsets[2] += divmod_mod * istride02;
        
        //dim1
        divmod_div = linear_idx / size1;
        divmod_mod = linear_idx % size1;
        linear_idx = divmod_div;
        offsets[0] += divmod_mod * ostride10;
        offsets[1] += divmod_mod * ostride11;
        offsets[2] += divmod_mod * istride12;

        //dim2
        divmod_div = linear_idx / size2;
        divmod_mod = linear_idx % size2;
        linear_idx = divmod_div;
        offsets[0] += divmod_mod * ostride20;
        offsets[1] += divmod_mod * ostride21;
        offsets[2] += divmod_mod * istride22;

        //dim3
        divmod_div = linear_idx / size3;
        divmod_mod = linear_idx % size3;
        linear_idx = divmod_div;
        offsets[0] += divmod_mod * ostride30;
        offsets[1] += divmod_mod * ostride31;
        offsets[2] += divmod_mod * istride32;
        
        //compute and save
        arg0_t arg0 = *(reinterpret_cast<arg0_t*>(data[2]) + offsets[2]);
        rets_t res = f(arg0);
        ret0_t *ret0 = reinterpret_cast<ret0_t *>(data[0]) + offsets[0];
        ret1_t *ret1 = reinterpret_cast<ret1_t *>(data[1]) + offsets[1];
        *ret0 = thrust::get<0>(res);
        *ret1 = thrust::get<1>(res);

        thread_idx += num_threads();
    }

}
