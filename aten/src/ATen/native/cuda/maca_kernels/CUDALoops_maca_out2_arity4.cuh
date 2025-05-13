#pragma once

template <typename func_t, typename array_t, 
          typename rets_t, typename ret0_t, typename ret1_t,
          typename arg0_t, typename arg1_t, typename arg2_t,  typename arg3_t,
          typename index_t, typename stride_t>
C10_LAUNCH_BOUNDS_1(num_threads())
__global__ void legacy_optim_unrolled_kernel_for_multi_outputs_out2_arity4_dim1(
  int64_t N, const func_t& f, array_t data,
  index_t size0, 
  stride_t ostride00, 
  stride_t ostride01, 
  stride_t istride02, 
  stride_t istride03, 
  stride_t istride04, 
  stride_t istride05
){
    int64_t remaining = N - block_work_size() * blockIdx.x;
    int64_t idx = blockIdx.x;
    int thread_idx = threadIdx.x;
    int64_t block_start_index = block_work_size() * idx;

    #pragma unroll
    for(int i = 0; i < thread_work_size(); i++){
        if(thread_idx >= remaining) return;
        int64_t linear_idx = thread_idx + block_start_index;

        int64_t offsets[6];
        constexpr int NARGS = 6;
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
        offsets[3] += divmod_mod * istride03;
        offsets[4] += divmod_mod * istride04;
        offsets[5] += divmod_mod * istride05;
        
        //compute and save
        arg0_t arg0 = *(reinterpret_cast<arg0_t*>(data[2]) + offsets[2]);
        arg1_t arg1 = *(reinterpret_cast<arg1_t*>(data[3]) + offsets[3]);
        arg2_t arg2 = *(reinterpret_cast<arg2_t*>(data[4]) + offsets[4]);
        arg3_t arg3 = *(reinterpret_cast<arg3_t*>(data[5]) + offsets[5]);
        rets_t res = f(arg0, arg1, arg2, arg3);
        ret0_t *ret0 = reinterpret_cast<ret0_t *>(data[0]) + offsets[0];
        ret1_t *ret1 = reinterpret_cast<ret1_t *>(data[1]) + offsets[1];
        *ret0 = thrust::get<0>(res);
        *ret1 = thrust::get<1>(res);

        thread_idx += num_threads();
    }
}


template <typename func_t, typename array_t, 
          typename rets_t, typename ret0_t, typename ret1_t,
          typename arg0_t, typename arg1_t, typename arg2_t,  typename arg3_t,
          typename index_t, typename stride_t>
C10_LAUNCH_BOUNDS_1(num_threads())
__global__ void legacy_optim_unrolled_kernel_for_multi_outputs_out2_arity4_dim2(
  int64_t N, const func_t& f, array_t data,
  index_t size0, index_t size1, 
  stride_t ostride00, stride_t ostride10, 
  stride_t ostride01, stride_t ostride11, 
  stride_t istride02, stride_t istride12, 
  stride_t istride03, stride_t istride13, 
  stride_t istride04, stride_t istride14, 
  stride_t istride05, stride_t istride15
){
    int64_t remaining = N - block_work_size() * blockIdx.x;
    int64_t idx = blockIdx.x;
    int thread_idx = threadIdx.x;
    int64_t block_start_index = block_work_size() * idx;

    #pragma unroll
    for(int i = 0; i < thread_work_size(); i++){
        if(thread_idx >= remaining) return;
        int64_t linear_idx = thread_idx + block_start_index;

        int64_t offsets[6];
        constexpr int NARGS = 6;
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
        offsets[3] += divmod_mod * istride03;
        offsets[4] += divmod_mod * istride04;
        offsets[5] += divmod_mod * istride05;
        
        //dim1
        divmod_div = linear_idx / size1;
        divmod_mod = linear_idx % size1;
        linear_idx = divmod_div;
        offsets[0] += divmod_mod * ostride10;
        offsets[1] += divmod_mod * ostride11;
        offsets[2] += divmod_mod * istride12;
        offsets[3] += divmod_mod * istride13;
        offsets[4] += divmod_mod * istride14;
        offsets[5] += divmod_mod * istride15;
        
        //compute and save
        arg0_t arg0 = *(reinterpret_cast<arg0_t*>(data[2]) + offsets[2]);
        arg1_t arg1 = *(reinterpret_cast<arg1_t*>(data[3]) + offsets[3]);
        arg2_t arg2 = *(reinterpret_cast<arg2_t*>(data[4]) + offsets[4]);
        arg3_t arg3 = *(reinterpret_cast<arg3_t*>(data[5]) + offsets[5]);
        rets_t res = f(arg0, arg1, arg2, arg3);
        ret0_t *ret0 = reinterpret_cast<ret0_t *>(data[0]) + offsets[0];
        ret1_t *ret1 = reinterpret_cast<ret1_t *>(data[1]) + offsets[1];
        *ret0 = thrust::get<0>(res);
        *ret1 = thrust::get<1>(res);

        thread_idx += num_threads();
    }

}


template <typename func_t, typename array_t, 
          typename rets_t, typename ret0_t, typename ret1_t,
          typename arg0_t, typename arg1_t, typename arg2_t,  typename arg3_t,
          typename index_t, typename stride_t>
C10_LAUNCH_BOUNDS_1(num_threads())
__global__ void legacy_optim_unrolled_kernel_for_multi_outputs_out2_arity4_dim3(
  int64_t N, const func_t& f, array_t data,
  index_t size0, index_t size1, index_t size2,
  stride_t ostride00, stride_t ostride10, stride_t ostride20,
  stride_t ostride01, stride_t ostride11, stride_t ostride21, 
  stride_t istride02, stride_t istride12, stride_t istride22,
  stride_t istride03, stride_t istride13, stride_t istride23, 
  stride_t istride04, stride_t istride14, stride_t istride24, 
  stride_t istride05, stride_t istride15, stride_t istride25
){
    int64_t remaining = N - block_work_size() * blockIdx.x;
    int idx = blockIdx.x;
    int64_t thread_idx = threadIdx.x;
    int64_t block_start_index = block_work_size() * idx;

    #pragma unroll
    for(int i = 0; i < thread_work_size(); i++){
        if(thread_idx >= remaining) return;
        int linear_idx = thread_idx + block_start_index;

        int64_t offsets[6];
        constexpr int NARGS = 6;
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
        offsets[3] += divmod_mod * istride03;
        offsets[4] += divmod_mod * istride04;
        offsets[5] += divmod_mod * istride05;
        
        //dim1
        divmod_div = linear_idx / size1;
        divmod_mod = linear_idx % size1;
        linear_idx = divmod_div;
        offsets[0] += divmod_mod * ostride10;
        offsets[1] += divmod_mod * ostride11;
        offsets[2] += divmod_mod * istride12;
        offsets[3] += divmod_mod * istride13;
        offsets[4] += divmod_mod * istride14;
        offsets[5] += divmod_mod * istride15;

        //dim2
        divmod_div = linear_idx / size2;
        divmod_mod = linear_idx % size2;
        linear_idx = divmod_div;
        offsets[0] += divmod_mod * ostride20;
        offsets[1] += divmod_mod * ostride21;
        offsets[2] += divmod_mod * istride22;
        offsets[3] += divmod_mod * istride23;
        offsets[4] += divmod_mod * istride24;
        offsets[5] += divmod_mod * istride25;
        
        //compute and save
        arg0_t arg0 = *(reinterpret_cast<arg0_t*>(data[2]) + offsets[2]);
        arg1_t arg1 = *(reinterpret_cast<arg1_t*>(data[3]) + offsets[3]);
        arg2_t arg2 = *(reinterpret_cast<arg2_t*>(data[4]) + offsets[4]);
        arg3_t arg3 = *(reinterpret_cast<arg3_t*>(data[5]) + offsets[5]);
        rets_t res = f(arg0, arg1, arg2, arg3);
        ret0_t *ret0 = reinterpret_cast<ret0_t *>(data[0]) + offsets[0];
        ret1_t *ret1 = reinterpret_cast<ret1_t *>(data[1]) + offsets[1];
        *ret0 = thrust::get<0>(res);
        *ret1 = thrust::get<1>(res);

        thread_idx += num_threads();
    }

}



template <typename func_t, typename array_t, 
          typename rets_t, typename ret0_t, typename ret1_t,
          typename arg0_t, typename arg1_t, typename arg2_t,  typename arg3_t,
          typename index_t, typename stride_t>
C10_LAUNCH_BOUNDS_1(num_threads())
__global__ void legacy_optim_unrolled_kernel_for_multi_outputs_out2_arity4_dim4(
  int64_t N, const func_t& f, array_t data,
  index_t size0, index_t size1, index_t size2, index_t size3,
  stride_t ostride00, stride_t ostride10, stride_t ostride20, stride_t ostride30,
  stride_t ostride01, stride_t ostride11, stride_t ostride21, stride_t ostride31,
  stride_t istride02, stride_t istride12, stride_t istride22, stride_t istride32,
  stride_t istride03, stride_t istride13, stride_t istride23, stride_t istride33,
  stride_t istride04, stride_t istride14, stride_t istride24, stride_t istride34,
  stride_t istride05, stride_t istride15, stride_t istride25, stride_t istride35
){
    int64_t remaining = N - block_work_size() * blockIdx.x;
    int64_t idx = blockIdx.x;
    int thread_idx = threadIdx.x;
    int64_t block_start_index = block_work_size() * idx;

    #pragma unroll
    for(int i = 0; i < thread_work_size(); i++){
        if(thread_idx >= remaining) return;
        int64_t linear_idx = thread_idx + block_start_index;

        int64_t offsets[6];
        constexpr int NARGS = 6;
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
        offsets[3] += divmod_mod * istride03;
        offsets[4] += divmod_mod * istride04;
        offsets[5] += divmod_mod * istride05;
        
        //dim1
        divmod_div = linear_idx / size1;
        divmod_mod = linear_idx % size1;
        linear_idx = divmod_div;
        offsets[0] += divmod_mod * ostride10;
        offsets[1] += divmod_mod * ostride11;
        offsets[2] += divmod_mod * istride12;
        offsets[3] += divmod_mod * istride13;
        offsets[4] += divmod_mod * istride14;
        offsets[5] += divmod_mod * istride15;

        //dim2
        divmod_div = linear_idx / size2;
        divmod_mod = linear_idx % size2;
        linear_idx = divmod_div;
        offsets[0] += divmod_mod * ostride20;
        offsets[1] += divmod_mod * ostride21;
        offsets[2] += divmod_mod * istride22;
        offsets[3] += divmod_mod * istride23;
        offsets[4] += divmod_mod * istride24;
        offsets[5] += divmod_mod * istride25;

        //dim3
        divmod_div = linear_idx / size3;
        divmod_mod = linear_idx % size3;
        linear_idx = divmod_div;
        offsets[0] += divmod_mod * ostride30;
        offsets[1] += divmod_mod * ostride31;
        offsets[2] += divmod_mod * istride32;
        offsets[3] += divmod_mod * istride33;
        offsets[4] += divmod_mod * istride34;
        offsets[5] += divmod_mod * istride35;
        
        //compute and save
        arg0_t arg0 = *(reinterpret_cast<arg0_t*>(data[2]) + offsets[2]);
        arg1_t arg1 = *(reinterpret_cast<arg1_t*>(data[3]) + offsets[3]);
        arg2_t arg2 = *(reinterpret_cast<arg2_t*>(data[4]) + offsets[4]);
        arg3_t arg3 = *(reinterpret_cast<arg3_t*>(data[5]) + offsets[5]);
        rets_t res = f(arg0, arg1, arg2, arg3);
        ret0_t *ret0 = reinterpret_cast<ret0_t *>(data[0]) + offsets[0];
        ret1_t *ret1 = reinterpret_cast<ret1_t *>(data[1]) + offsets[1];
        *ret0 = thrust::get<0>(res);
        *ret1 = thrust::get<1>(res);

        thread_idx += num_threads();
    }

}


template <typename func_t, typename array_t, 
          typename rets_t, typename ret0_t, typename ret1_t,
          typename arg0_t, typename arg1_t, typename arg2_t,  typename arg3_t,
          typename index_t, typename stride_t>
C10_LAUNCH_BOUNDS_1(num_threads())
__global__ void legacy_optim_unrolled_kernel_for_multi_outputs_out2_arity4_dim5(
  int64_t N, const func_t& f, array_t data,
  index_t size0, index_t size1, index_t size2, index_t size3, index_t size4,
  stride_t ostride00, stride_t ostride10, stride_t ostride20, stride_t ostride30, stride_t ostride40,
  stride_t ostride01, stride_t ostride11, stride_t ostride21, stride_t ostride31, stride_t ostride41,
  stride_t istride02, stride_t istride12, stride_t istride22, stride_t istride32, stride_t istride42,
  stride_t istride03, stride_t istride13, stride_t istride23, stride_t istride33, stride_t istride43,
  stride_t istride04, stride_t istride14, stride_t istride24, stride_t istride34, stride_t istride44,
  stride_t istride05, stride_t istride15, stride_t istride25, stride_t istride35, stride_t istride45
){
    int64_t remaining = N - block_work_size() * blockIdx.x;
    int64_t idx = blockIdx.x;
    int thread_idx = threadIdx.x;
    int64_t block_start_index = block_work_size() * idx;

    #pragma unroll
    for(int i = 0; i < thread_work_size(); i++){
        if(thread_idx >= remaining) return;
        int64_t linear_idx = thread_idx + block_start_index;

        int64_t offsets[6];
        constexpr int NARGS = 6;
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
        offsets[3] += divmod_mod * istride03;
        offsets[4] += divmod_mod * istride04;
        offsets[5] += divmod_mod * istride05;
        
        //dim1
        divmod_div = linear_idx / size1;
        divmod_mod = linear_idx % size1;
        linear_idx = divmod_div;
        offsets[0] += divmod_mod * ostride10;
        offsets[1] += divmod_mod * ostride11;
        offsets[2] += divmod_mod * istride12;
        offsets[3] += divmod_mod * istride13;
        offsets[4] += divmod_mod * istride14;
        offsets[5] += divmod_mod * istride15;

        //dim2
        divmod_div = linear_idx / size2;
        divmod_mod = linear_idx % size2;
        linear_idx = divmod_div;
        offsets[0] += divmod_mod * ostride20;
        offsets[1] += divmod_mod * ostride21;
        offsets[2] += divmod_mod * istride22;
        offsets[3] += divmod_mod * istride23;
        offsets[4] += divmod_mod * istride24;
        offsets[5] += divmod_mod * istride25;

        //dim3
        divmod_div = linear_idx / size3;
        divmod_mod = linear_idx % size3;
        linear_idx = divmod_div;
        offsets[0] += divmod_mod * ostride30;
        offsets[1] += divmod_mod * ostride31;
        offsets[2] += divmod_mod * istride32;
        offsets[3] += divmod_mod * istride33;
        offsets[4] += divmod_mod * istride34;
        offsets[5] += divmod_mod * istride35;

        //dim4
        divmod_div = linear_idx / size4;
        divmod_mod = linear_idx % size4;
        linear_idx = divmod_div;
        offsets[0] += divmod_mod * ostride40;
        offsets[1] += divmod_mod * ostride41;
        offsets[2] += divmod_mod * istride42;
        offsets[3] += divmod_mod * istride43;
        offsets[4] += divmod_mod * istride44;
        offsets[5] += divmod_mod * istride45;
        
        //compute and save
        arg0_t arg0 = *(reinterpret_cast<arg0_t*>(data[2]) + offsets[2]);
        arg1_t arg1 = *(reinterpret_cast<arg1_t*>(data[3]) + offsets[3]);
        arg2_t arg2 = *(reinterpret_cast<arg2_t*>(data[4]) + offsets[4]);
        arg3_t arg3 = *(reinterpret_cast<arg3_t*>(data[5]) + offsets[5]);
        rets_t res = f(arg0, arg1, arg2, arg3);
        ret0_t *ret0 = reinterpret_cast<ret0_t *>(data[0]) + offsets[0];
        ret1_t *ret1 = reinterpret_cast<ret1_t *>(data[1]) + offsets[1];
        *ret0 = thrust::get<0>(res);
        *ret1 = thrust::get<1>(res);

        thread_idx += num_threads();
    }

}


template <typename func_t, typename array_t, 
          typename rets_t, typename ret0_t, typename ret1_t,
          typename arg0_t, typename arg1_t, typename arg2_t,  typename arg3_t,
          typename index_t, typename stride_t>
C10_LAUNCH_BOUNDS_1(num_threads())
__global__ void legacy_optim_unrolled_kernel_for_multi_outputs_out2_arity4_dim6(
  int64_t N, const func_t& f, array_t data,
  index_t size0, index_t size1, index_t size2, index_t size3, index_t size4, index_t size5,
  stride_t ostride00, stride_t ostride10, stride_t ostride20, stride_t ostride30, stride_t ostride40, stride_t ostride50,
  stride_t ostride01, stride_t ostride11, stride_t ostride21, stride_t ostride31, stride_t ostride41, stride_t ostride51,
  stride_t istride02, stride_t istride12, stride_t istride22, stride_t istride32, stride_t istride42, stride_t istride52,
  stride_t istride03, stride_t istride13, stride_t istride23, stride_t istride33, stride_t istride43, stride_t istride53,
  stride_t istride04, stride_t istride14, stride_t istride24, stride_t istride34, stride_t istride44, stride_t istride54,
  stride_t istride05, stride_t istride15, stride_t istride25, stride_t istride35, stride_t istride45, stride_t istride55
){
    int64_t remaining = N - block_work_size() * blockIdx.x;
    int64_t idx = blockIdx.x;
    int thread_idx = threadIdx.x;
    int64_t block_start_index = block_work_size() * idx;

    #pragma unroll
    for(int i = 0; i < thread_work_size(); i++){
        if(thread_idx >= remaining) return;
        int64_t linear_idx = thread_idx + block_start_index;

        int64_t offsets[6];
        constexpr int NARGS = 6;
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
        offsets[3] += divmod_mod * istride03;
        offsets[4] += divmod_mod * istride04;
        offsets[5] += divmod_mod * istride05;
        
        //dim1
        divmod_div = linear_idx / size1;
        divmod_mod = linear_idx % size1;
        linear_idx = divmod_div;
        offsets[0] += divmod_mod * ostride10;
        offsets[1] += divmod_mod * ostride11;
        offsets[2] += divmod_mod * istride12;
        offsets[3] += divmod_mod * istride13;
        offsets[4] += divmod_mod * istride14;
        offsets[5] += divmod_mod * istride15;

        //dim2
        divmod_div = linear_idx / size2;
        divmod_mod = linear_idx % size2;
        linear_idx = divmod_div;
        offsets[0] += divmod_mod * ostride20;
        offsets[1] += divmod_mod * ostride21;
        offsets[2] += divmod_mod * istride22;
        offsets[3] += divmod_mod * istride23;
        offsets[4] += divmod_mod * istride24;
        offsets[5] += divmod_mod * istride25;

        //dim3
        divmod_div = linear_idx / size3;
        divmod_mod = linear_idx % size3;
        linear_idx = divmod_div;
        offsets[0] += divmod_mod * ostride30;
        offsets[1] += divmod_mod * ostride31;
        offsets[2] += divmod_mod * istride32;
        offsets[3] += divmod_mod * istride33;
        offsets[4] += divmod_mod * istride34;
        offsets[5] += divmod_mod * istride35;

        //dim4
        divmod_div = linear_idx / size4;
        divmod_mod = linear_idx % size4;
        linear_idx = divmod_div;
        offsets[0] += divmod_mod * ostride40;
        offsets[1] += divmod_mod * ostride41;
        offsets[2] += divmod_mod * istride42;
        offsets[3] += divmod_mod * istride43;
        offsets[4] += divmod_mod * istride44;
        offsets[5] += divmod_mod * istride45;

        //dim5
        divmod_div = linear_idx / size5;
        divmod_mod = linear_idx % size5;
        linear_idx = divmod_div;
        offsets[0] += divmod_mod * ostride50;
        offsets[1] += divmod_mod * ostride51;
        offsets[2] += divmod_mod * istride52;
        offsets[3] += divmod_mod * istride53;
        offsets[4] += divmod_mod * istride54;
        offsets[5] += divmod_mod * istride55;
        
        //compute and save
        arg0_t arg0 = *(reinterpret_cast<arg0_t*>(data[2]) + offsets[2]);
        arg1_t arg1 = *(reinterpret_cast<arg1_t*>(data[3]) + offsets[3]);
        arg2_t arg2 = *(reinterpret_cast<arg2_t*>(data[4]) + offsets[4]);
        arg3_t arg3 = *(reinterpret_cast<arg3_t*>(data[5]) + offsets[5]);
        rets_t res = f(arg0, arg1, arg2, arg3);
        ret0_t *ret0 = reinterpret_cast<ret0_t *>(data[0]) + offsets[0];
        ret1_t *ret1 = reinterpret_cast<ret1_t *>(data[1]) + offsets[1];
        *ret0 = thrust::get<0>(res);
        *ret1 = thrust::get<1>(res);

        thread_idx += num_threads();
    }

}
