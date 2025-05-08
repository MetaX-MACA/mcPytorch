#pragma once

#include <ATen/native/cuda/maca_kernels/CUDALoops_maca_out2_arity4.cuh>
#include <ATen/native/cuda/maca_kernels/CUDALoops_maca_out2_arity1.cuh>


template <typename inp_calc_t, typename out_calc_t>
void print_multi_outputs(int64_t num_outputs, int64_t num_inputs, int64_t ndim, inp_calc_t input_calc, out_calc_t output_calc){
  std::cout<<"---------------multiple_outputs_nocontiguous----------------"<<std::endl;
  if(num_inputs==4 && num_outputs==2){
      if(ndim==0){
        std::cout<<"multiple_outputs(num_inputs=4,num_outputs=2,ndim=0)"<<std::endl;
      }
      else if(ndim==1){
        std::cout<<"multiple_outputs(num_inputs=4,num_outputs=2,ndim=1)"<<std::endl;
        std::cout<<output_calc.sizes_[0].divisor<<std::endl;
        std::cout<<input_calc.sizes_[0].divisor<<std::endl;
        std::cout<<output_calc.strides_[0][0]<<std::endl;
        std::cout<<output_calc.strides_[0][1]<<std::endl;
        std::cout<<input_calc.strides_[0][0]<<std::endl;
        std::cout<<input_calc.strides_[0][1]<<std::endl;
        std::cout<<input_calc.strides_[0][2]<<std::endl;
        std::cout<<input_calc.strides_[0][3]<<std::endl;
      }
      else if(ndim==2){
        std::cout<<"multiple_outputs(num_inputs=4,num_outputs=2,ndim=2)"<<std::endl;
        std::cout<<output_calc.sizes_[0].divisor<<","<<output_calc.sizes_[1].divisor<<std::endl;
        std::cout<<input_calc.sizes_[0].divisor<<","<<input_calc.sizes_[1].divisor<<std::endl;
        std::cout<<output_calc.strides_[0][0]<<","<<output_calc.strides_[1][0]<<std::endl;
        std::cout<<output_calc.strides_[0][1]<<","<<output_calc.strides_[1][1]<<std::endl;
        std::cout<<input_calc.strides_[0][0]<<","<<input_calc.strides_[1][0]<<std::endl;
        std::cout<<input_calc.strides_[0][1]<<","<<input_calc.strides_[1][1]<<std::endl;
        std::cout<<input_calc.strides_[0][2]<<","<<input_calc.strides_[1][2]<<std::endl;
        std::cout<<input_calc.strides_[0][3]<<","<<input_calc.strides_[1][3]<<std::endl;
      }
      else if(ndim==3){
        std::cout<<"multiple_outputs(num_inputs=4,num_outputs=2,ndim=3)"<<std::endl;
        std::cout<<output_calc.sizes_[0].divisor<<","<<output_calc.sizes_[1].divisor<<","<<output_calc.sizes_[2].divisor<<std::endl;
        std::cout<<input_calc.sizes_[0].divisor<<","<<input_calc.sizes_[1].divisor<<","<<input_calc.sizes_[2].divisor<<std::endl;
        std::cout<<output_calc.strides_[0][0]<<","<<output_calc.strides_[1][0]<<","<<output_calc.strides_[2][0]<<std::endl;
        std::cout<<output_calc.strides_[0][1]<<","<<output_calc.strides_[1][1]<<","<<output_calc.strides_[2][1]<<std::endl;
        std::cout<<input_calc.strides_[0][0]<<","<<input_calc.strides_[1][0]<<","<<input_calc.strides_[2][0]<<std::endl;
        std::cout<<input_calc.strides_[0][1]<<","<<input_calc.strides_[1][1]<<","<<input_calc.strides_[2][1]<<std::endl;
        std::cout<<input_calc.strides_[0][2]<<","<<input_calc.strides_[1][2]<<","<<input_calc.strides_[2][2]<<std::endl;
        std::cout<<input_calc.strides_[0][3]<<","<<input_calc.strides_[1][3]<<","<<input_calc.strides_[2][3]<<std::endl;
      }
      else if(ndim==4){
        std::cout<<"multiple_outputs(num_inputs=4,num_outputs=2,ndim=4)"<<std::endl;
        std::cout<<output_calc.sizes_[0].divisor<<","<<output_calc.sizes_[1].divisor<<","<<output_calc.sizes_[2].divisor<<","<<output_calc.sizes_[3].divisor<<std::endl;
        std::cout<<input_calc.sizes_[0].divisor<<","<<input_calc.sizes_[1].divisor<<","<<input_calc.sizes_[2].divisor<<","<<input_calc.sizes_[3].divisor<<std::endl;
        std::cout<<output_calc.strides_[0][0]<<","<<output_calc.strides_[1][0]<<","<<output_calc.strides_[2][0]<<","<<output_calc.strides_[3][0]<<std::endl;
        std::cout<<output_calc.strides_[0][1]<<","<<output_calc.strides_[1][1]<<","<<output_calc.strides_[2][1]<<","<<output_calc.strides_[3][1]<<std::endl;
        std::cout<<input_calc.strides_[0][0]<<","<<input_calc.strides_[1][0]<<","<<input_calc.strides_[2][0]<<","<<input_calc.strides_[3][0]<<std::endl;
        std::cout<<input_calc.strides_[0][1]<<","<<input_calc.strides_[1][1]<<","<<input_calc.strides_[2][1]<<","<<input_calc.strides_[3][1]<<std::endl;
        std::cout<<input_calc.strides_[0][2]<<","<<input_calc.strides_[1][2]<<","<<input_calc.strides_[2][2]<<","<<input_calc.strides_[3][2]<<std::endl;
        std::cout<<input_calc.strides_[0][3]<<","<<input_calc.strides_[1][3]<<","<<input_calc.strides_[2][3]<<","<<input_calc.strides_[3][3]<<std::endl;
      }
      else {std::cout<<"multiple_outputs(num_inputs=4,num_outputs=2,ndim="<<ndim<<")"<<std::endl;}
      return;
    }


    if(num_inputs==1 && num_outputs==2){
      if(ndim==0){
        std::cout<<"multiple_outputs(num_inputs=1,num_outputs=2,ndim=0)"<<std::endl;
      }
      else if(ndim==1){
        std::cout<<"multiple_outputs(num_inputs=1,num_outputs=2,ndim=1)"<<std::endl;
        std::cout<<output_calc.sizes_[0].divisor<<std::endl;
        std::cout<<input_calc.sizes_[0].divisor<<std::endl;
        std::cout<<output_calc.strides_[0][0]<<std::endl;
        std::cout<<output_calc.strides_[0][1]<<std::endl;
        std::cout<<input_calc.strides_[0][0]<<std::endl;
      }
      else if(ndim==2){
        std::cout<<"multiple_outputs(num_inputs=1,num_outputs=2,ndim=2)"<<std::endl;
        std::cout<<output_calc.sizes_[0].divisor<<","<<output_calc.sizes_[1].divisor<<std::endl;
        std::cout<<input_calc.sizes_[0].divisor<<","<<input_calc.sizes_[1].divisor<<std::endl;
        std::cout<<output_calc.strides_[0][0]<<","<<output_calc.strides_[1][0]<<std::endl;
        std::cout<<output_calc.strides_[0][1]<<","<<output_calc.strides_[1][1]<<std::endl;
        std::cout<<input_calc.strides_[0][0]<<","<<input_calc.strides_[1][0]<<std::endl;
      }
      else if(ndim==3){
        std::cout<<"multiple_outputs(num_inputs=1,num_outputs=2,ndim=3)"<<std::endl;
        std::cout<<output_calc.sizes_[0].divisor<<","<<output_calc.sizes_[1].divisor<<","<<output_calc.sizes_[2].divisor<<std::endl;
        std::cout<<input_calc.sizes_[0].divisor<<","<<input_calc.sizes_[1].divisor<<","<<input_calc.sizes_[2].divisor<<std::endl;
        std::cout<<output_calc.strides_[0][0]<<","<<output_calc.strides_[1][0]<<","<<output_calc.strides_[2][0]<<std::endl;
        std::cout<<output_calc.strides_[0][1]<<","<<output_calc.strides_[1][1]<<","<<output_calc.strides_[2][1]<<std::endl;
        std::cout<<input_calc.strides_[0][0]<<","<<input_calc.strides_[1][0]<<","<<input_calc.strides_[2][0]<<std::endl;
      }
      else if(ndim==4){
        std::cout<<"multiple_outputs(num_inputs=1,num_outputs=2,ndim=4)"<<std::endl;
        std::cout<<output_calc.sizes_[0].divisor<<","<<output_calc.sizes_[1].divisor<<","<<output_calc.sizes_[2].divisor<<","<<output_calc.sizes_[3].divisor<<std::endl;
        std::cout<<input_calc.sizes_[0].divisor<<","<<input_calc.sizes_[1].divisor<<","<<input_calc.sizes_[2].divisor<<","<<input_calc.sizes_[3].divisor<<std::endl;
        std::cout<<output_calc.strides_[0][0]<<","<<output_calc.strides_[1][0]<<","<<output_calc.strides_[2][0]<<","<<output_calc.strides_[3][0]<<std::endl;
        std::cout<<output_calc.strides_[0][1]<<","<<output_calc.strides_[1][1]<<","<<output_calc.strides_[2][1]<<","<<output_calc.strides_[3][1]<<std::endl;
        std::cout<<input_calc.strides_[0][0]<<","<<input_calc.strides_[1][0]<<","<<input_calc.strides_[2][0]<<","<<input_calc.strides_[3][0]<<std::endl;
      }
      else {std::cout<<"multiple_outputs(num_inputs=1,num_outputs=2,ndim="<<ndim<<")"<<std::endl;}
      return;
    }

    std::cout<<"multiple_outputs(num_inputs=other,num_outputs=other,ndim=other)"<<std::endl;
}



template <typename func_t, typename array_t, 
          typename rets_t, typename ret0_t, typename ret1_t,
          typename arg0_t, typename arg1_t, typename arg2_t,  typename arg3_t>
C10_LAUNCH_BOUNDS_1(num_threads())
__global__ void contiguous_optim_unrolled_kernel_for_multi_outputs_out2_arity4(
  int64_t N, func_t f, array_t data
){
  int64_t remaining = N - block_work_size() * blockIdx.x;
  int64_t idx = blockIdx.x;
  int thread_idx = threadIdx.x;
  int64_t block_start_index = block_work_size() * idx;
  
  #pragma unroll
  for(int i = 0; i < thread_work_size(); i++){
    if(thread_idx >= remaining) return;
    int64_t linear_idx = thread_idx + block_start_index;
    arg0_t arg0 = *(reinterpret_cast<arg0_t*>(data[2]) + linear_idx);
    arg1_t arg1 = *(reinterpret_cast<arg1_t*>(data[3]) + linear_idx);
    arg2_t arg2 = *(reinterpret_cast<arg2_t*>(data[4]) + linear_idx);
    arg3_t arg3 = *(reinterpret_cast<arg3_t*>(data[5]) + linear_idx);

    rets_t res = f(arg0, arg1, arg2, arg3);

    ret0_t *ret0 = reinterpret_cast<ret0_t *>(data[0]) + linear_idx;
    ret1_t *ret1 = reinterpret_cast<ret1_t *>(data[1]) + linear_idx;
    *ret0 = thrust::get<0>(res);
    *ret1 = thrust::get<1>(res);

    thread_idx += num_threads();
  }
}


template <typename func_t, typename array_t, 
          typename rets_t, typename ret0_t, typename ret1_t,
          typename arg0_t>
C10_LAUNCH_BOUNDS_1(num_threads())
__global__ void contiguous_optim_unrolled_kernel_for_multi_outputs_out2_arity1(
  int64_t N, func_t f, array_t data
){
  int64_t remaining = N - block_work_size() * blockIdx.x;
  int64_t idx = blockIdx.x;
  int thread_idx = threadIdx.x;
  int64_t block_start_index = block_work_size() * idx;
  
  #pragma unroll
  for(int i = 0; i < thread_work_size(); i++){
    if(thread_idx >= remaining) return;
    int64_t linear_idx = thread_idx + block_start_index;
    arg0_t arg0 = *(reinterpret_cast<arg0_t*>(data[2]) + linear_idx);

    rets_t res = f(arg0);

    ret0_t *ret0 = reinterpret_cast<ret0_t *>(data[0]) + linear_idx;
    ret1_t *ret1 = reinterpret_cast<ret1_t *>(data[1]) + linear_idx;
    *ret0 = thrust::get<0>(res);
    *ret1 = thrust::get<1>(res);

    thread_idx += num_threads();
  }
}

template <int num_outputs, int num_inputs>
class launch_contiguous_optim_unrolled_kernel_for_multi_outputs{
public:
   launch_contiguous_optim_unrolled_kernel_for_multi_outputs() {}
  
  template<typename func_t, typename array_t>
  void operator() (int64_t N, const func_t& f, array_t data){
    assert(false);
  }
};


template <>
class launch_contiguous_optim_unrolled_kernel_for_multi_outputs<2,4>{
public:
  launch_contiguous_optim_unrolled_kernel_for_multi_outputs() {}
  
  template<typename func_t, typename array_t>
  void operator() (int64_t N,const func_t& f, array_t data){

    TORCH_INTERNAL_ASSERT(N > 0 && N <= std::numeric_limits<int32_t>::max());
    using traits = function_traits<func_t>;

    using args_t = typename traits::ArgsTuple;
    using arg0_t = typename traits::template arg<0>::type;
    using arg1_t = typename traits::template arg<1>::type;
    using arg2_t = typename traits::template arg<2>::type;
    using arg3_t = typename traits::template arg<3>::type;

    using rets_t = typename traits::result_type;
    using ret0_t = typename thrust::tuple_element<0, rets_t>::type;
    using ret1_t = typename thrust::tuple_element<1, rets_t>::type;

    int64_t grid = (N + block_work_size() - 1) / block_work_size();
    auto stream = at::cuda::getCurrentCUDAStream();
    
    contiguous_optim_unrolled_kernel_for_multi_outputs_out2_arity4<
                        func_t, array_t, 
                        rets_t, ret0_t, ret1_t,
                        arg0_t, arg1_t, arg2_t, arg3_t>
      <<<grid, num_threads(), 0, stream>>>(N, f, data);

    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
};


template <>
class launch_contiguous_optim_unrolled_kernel_for_multi_outputs<2,1>{
public:
  launch_contiguous_optim_unrolled_kernel_for_multi_outputs() {}
  
  template<typename func_t, typename array_t>
  void operator() (int64_t N,const func_t& f, array_t data){

    TORCH_INTERNAL_ASSERT(N > 0 && N <= std::numeric_limits<int32_t>::max());
    using traits = function_traits<func_t>;

    using args_t = typename traits::ArgsTuple;
    using arg0_t = typename traits::template arg<0>::type;

    using rets_t = typename traits::result_type;
    using ret0_t = typename thrust::tuple_element<0, rets_t>::type;
    using ret1_t = typename thrust::tuple_element<1, rets_t>::type;

    int64_t grid = (N + block_work_size() - 1) / block_work_size();
    auto stream = at::cuda::getCurrentCUDAStream();
    
    contiguous_optim_unrolled_kernel_for_multi_outputs_out2_arity1<
                        func_t, array_t, 
                        rets_t, ret0_t, ret1_t,
                        arg0_t>
      <<<grid, num_threads(), 0, stream>>>(N, f, data);

    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
};


template <int num_outputs, int num_inputs>
class launch_legacy_optim_unrolled_kernel_for_multi_outputs{
public:
   launch_legacy_optim_unrolled_kernel_for_multi_outputs() {}
  
  template<typename func_t, typename array_t, typename inp_calc_t, typename out_calc_t>
  void operator() (int64_t N, const func_t& f, array_t data, inp_calc_t ic, out_calc_t oc, int ndim){
    assert(false);
  }
};


template <>
class launch_legacy_optim_unrolled_kernel_for_multi_outputs<2,4>{
public:
  launch_legacy_optim_unrolled_kernel_for_multi_outputs() {}
  
  template<typename func_t, typename array_t, typename inp_calc_t, typename out_calc_t>
  void operator() (int64_t N, const func_t& f, array_t data, inp_calc_t ic, out_calc_t oc, int ndim){

    TORCH_INTERNAL_ASSERT(N > 0 && N <= std::numeric_limits<int32_t>::max());
    using traits = function_traits<func_t>;

    using args_t = typename traits::ArgsTuple;
    using arg0_t = typename traits::template arg<0>::type;
    using arg1_t = typename traits::template arg<1>::type;
    using arg2_t = typename traits::template arg<2>::type;
    using arg3_t = typename traits::template arg<3>::type;

    using rets_t = typename traits::result_type;
    using ret0_t = typename thrust::tuple_element<0, rets_t>::type;
    using ret1_t = typename thrust::tuple_element<1, rets_t>::type;

    if(ndim==1){
      auto osize0 = oc.sizes_[0].divisor;
      auto isize0 = ic.sizes_[0].divisor;
      assert(isize0==osize0);

      auto ostride00 = oc.strides_[0][0];
      auto ostride01 = oc.strides_[0][1];

      auto istride02 = ic.strides_[0][0];
      auto istride03 = ic.strides_[0][1];
      auto istride04 = ic.strides_[0][2];
      auto istride05 = ic.strides_[0][3];
      
      int64_t grid = (N + block_work_size() - 1) / block_work_size();
      auto stream = at::cuda::getCurrentCUDAStream();
      
      legacy_optim_unrolled_kernel_for_multi_outputs_out2_arity4_dim1<func_t, array_t, rets_t, ret0_t, ret1_t,
                                                                      arg0_t, arg1_t, arg2_t, arg3_t><<<
                                                                      grid, num_threads(), 0, stream>>>(
                                                                      N, f, data,
                                                                      isize0,
                                                                      ostride00,
                                                                      ostride01,
                                                                      istride02,
                                                                      istride03,
                                                                      istride04,
                                                                      istride05
                                                                     );
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      
    }else if(ndim==2){
      auto osize0 = oc.sizes_[0].divisor, osize1 = oc.sizes_[1].divisor;
      auto isize0 = ic.sizes_[0].divisor, isize1 = ic.sizes_[1].divisor;
      assert(isize0==osize0 && isize1==osize1);

      auto ostride00 = oc.strides_[0][0], ostride10 = oc.strides_[1][0];
      auto ostride01 = oc.strides_[0][1], ostride11 = oc.strides_[1][1];

      auto istride02 = ic.strides_[0][0], istride12 = ic.strides_[1][0];
      auto istride03 = ic.strides_[0][1], istride13 = ic.strides_[1][1];
      auto istride04 = ic.strides_[0][2], istride14 = ic.strides_[1][2];
      auto istride05 = ic.strides_[0][3], istride15 = ic.strides_[1][3];
      
      int64_t grid = (N + block_work_size() - 1) / block_work_size();
      auto stream = at::cuda::getCurrentCUDAStream();
      
      legacy_optim_unrolled_kernel_for_multi_outputs_out2_arity4_dim2<func_t, array_t, rets_t, ret0_t, ret1_t,
                                                                      arg0_t, arg1_t, arg2_t, arg3_t><<<
                                                                      grid, num_threads(), 0, stream>>>(
                                                                      N, f, data,
                                                                      isize0, isize1,
                                                                      ostride00, ostride10,
                                                                      ostride01, ostride11,
                                                                      istride02, istride12,
                                                                      istride03, istride13,
                                                                      istride04, istride14,
                                                                      istride05, istride15
                                                                     );
      C10_CUDA_KERNEL_LAUNCH_CHECK();

    }else if(ndim==3){
      auto osize0 = oc.sizes_[0].divisor, osize1 = oc.sizes_[1].divisor, osize2 = oc.sizes_[2].divisor;
      auto isize0 = ic.sizes_[0].divisor, isize1 = ic.sizes_[1].divisor, isize2 = ic.sizes_[2].divisor;
      assert(isize0==osize0 && isize1==osize1 && isize2==osize2);

      auto ostride00 = oc.strides_[0][0], ostride10 = oc.strides_[1][0], ostride20 = oc.strides_[2][0];
      auto ostride01 = oc.strides_[0][1], ostride11 = oc.strides_[1][1], ostride21 = oc.strides_[2][1];

      auto istride02 = ic.strides_[0][0], istride12 = ic.strides_[1][0], istride22 = ic.strides_[2][0];
      auto istride03 = ic.strides_[0][1], istride13 = ic.strides_[1][1], istride23 = ic.strides_[2][1];
      auto istride04 = ic.strides_[0][2], istride14 = ic.strides_[1][2], istride24 = ic.strides_[2][2];
      auto istride05 = ic.strides_[0][3], istride15 = ic.strides_[1][3], istride25 = ic.strides_[2][3];
      
      int64_t grid = (N + block_work_size() - 1) / block_work_size();
      auto stream = at::cuda::getCurrentCUDAStream();
      
      legacy_optim_unrolled_kernel_for_multi_outputs_out2_arity4_dim3<func_t, array_t, rets_t, ret0_t, ret1_t,
                                                                      arg0_t, arg1_t, arg2_t, arg3_t><<<
                                                                      grid, num_threads(), 0, stream>>>(
                                                                      N, f, data,
                                                                      isize0, isize1, isize2,
                                                                      ostride00, ostride10, ostride20,
                                                                      ostride01, ostride11, ostride21,
                                                                      istride02, istride12, istride22,
                                                                      istride03, istride13, istride23,
                                                                      istride04, istride14, istride24,
                                                                      istride05, istride15, istride25
                                                                     );
      C10_CUDA_KERNEL_LAUNCH_CHECK();

    }else if(ndim==4){
      auto osize0 = oc.sizes_[0].divisor, osize1 = oc.sizes_[1].divisor, osize2 = oc.sizes_[2].divisor, osize3 = oc.sizes_[3].divisor;
      auto isize0 = ic.sizes_[0].divisor, isize1 = ic.sizes_[1].divisor, isize2 = ic.sizes_[2].divisor, isize3 = ic.sizes_[3].divisor;
      assert(isize0==osize0 && isize1==osize1 && isize2==osize2 && isize3==osize3);

      auto ostride00 = oc.strides_[0][0], ostride10 = oc.strides_[1][0], ostride20 = oc.strides_[2][0], ostride30 = oc.strides_[3][0];
      auto ostride01 = oc.strides_[0][1], ostride11 = oc.strides_[1][1], ostride21 = oc.strides_[2][1], ostride31 = oc.strides_[3][1];

      auto istride02 = ic.strides_[0][0], istride12 = ic.strides_[1][0], istride22 = ic.strides_[2][0], istride32 = ic.strides_[3][0];
      auto istride03 = ic.strides_[0][1], istride13 = ic.strides_[1][1], istride23 = ic.strides_[2][1], istride33 = ic.strides_[3][1];
      auto istride04 = ic.strides_[0][2], istride14 = ic.strides_[1][2], istride24 = ic.strides_[2][2], istride34 = ic.strides_[3][2];
      auto istride05 = ic.strides_[0][3], istride15 = ic.strides_[1][3], istride25 = ic.strides_[2][3], istride35 = ic.strides_[3][3];
      
      int64_t grid = (N + block_work_size() - 1) / block_work_size();
      auto stream = at::cuda::getCurrentCUDAStream();
      
      legacy_optim_unrolled_kernel_for_multi_outputs_out2_arity4_dim4<func_t, array_t, rets_t, ret0_t, ret1_t,
                                                                      arg0_t, arg1_t, arg2_t, arg3_t><<<
                                                                      grid, num_threads(), 0, stream>>>(
                                                                      N, f, data,
                                                                      isize0, isize1, isize2, isize3,
                                                                      ostride00, ostride10, ostride20, ostride30,
                                                                      ostride01, ostride11, ostride21, ostride31,
                                                                      istride02, istride12, istride22, istride32,
                                                                      istride03, istride13, istride23, istride33,
                                                                      istride04, istride14, istride24, istride34,
                                                                      istride05, istride15, istride25, istride35
                                                                     );
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }else if(ndim==5){
      auto osize0 = oc.sizes_[0].divisor, osize1 = oc.sizes_[1].divisor, osize2 = oc.sizes_[2].divisor, osize3 = oc.sizes_[3].divisor, osize4 = oc.sizes_[4].divisor;
      auto isize0 = ic.sizes_[0].divisor, isize1 = ic.sizes_[1].divisor, isize2 = ic.sizes_[2].divisor, isize3 = ic.sizes_[3].divisor, isize4 = ic.sizes_[4].divisor;
      assert(isize0==osize0 && isize1==osize1 && isize2==osize2 && isize3==osize3 && isize4==osize4);

      auto ostride00 = oc.strides_[0][0], ostride10 = oc.strides_[1][0], ostride20 = oc.strides_[2][0], ostride30 = oc.strides_[3][0], ostride40 = oc.strides_[4][0];
      auto ostride01 = oc.strides_[0][1], ostride11 = oc.strides_[1][1], ostride21 = oc.strides_[2][1], ostride31 = oc.strides_[3][1], ostride41 = oc.strides_[4][1];

      auto istride02 = ic.strides_[0][0], istride12 = ic.strides_[1][0], istride22 = ic.strides_[2][0], istride32 = ic.strides_[3][0], istride42 = ic.strides_[4][0];
      auto istride03 = ic.strides_[0][1], istride13 = ic.strides_[1][1], istride23 = ic.strides_[2][1], istride33 = ic.strides_[3][1], istride43 = ic.strides_[4][1];
      auto istride04 = ic.strides_[0][2], istride14 = ic.strides_[1][2], istride24 = ic.strides_[2][2], istride34 = ic.strides_[3][2], istride44 = ic.strides_[4][2];
      auto istride05 = ic.strides_[0][3], istride15 = ic.strides_[1][3], istride25 = ic.strides_[2][3], istride35 = ic.strides_[3][3], istride45 = ic.strides_[4][3];
      
      int64_t grid = (N + block_work_size() - 1) / block_work_size();
      auto stream = at::cuda::getCurrentCUDAStream();
      
      legacy_optim_unrolled_kernel_for_multi_outputs_out2_arity4_dim5<func_t, array_t, rets_t, ret0_t, ret1_t,
                                                                      arg0_t, arg1_t, arg2_t, arg3_t><<<
                                                                      grid, num_threads(), 0, stream>>>(
                                                                      N, f, data,
                                                                      isize0, isize1, isize2, isize3, isize4,
                                                                      ostride00, ostride10, ostride20, ostride30, ostride40,
                                                                      ostride01, ostride11, ostride21, ostride31, ostride41,
                                                                      istride02, istride12, istride22, istride32, istride42,
                                                                      istride03, istride13, istride23, istride33, istride43,
                                                                      istride04, istride14, istride24, istride34, istride44,
                                                                      istride05, istride15, istride25, istride35, istride45
                                                                     );
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }else if(ndim==6){
      auto osize0 = oc.sizes_[0].divisor, osize1 = oc.sizes_[1].divisor, osize2 = oc.sizes_[2].divisor, osize3 = oc.sizes_[3].divisor, osize4 = oc.sizes_[4].divisor, osize5 = oc.sizes_[5].divisor;
      auto isize0 = ic.sizes_[0].divisor, isize1 = ic.sizes_[1].divisor, isize2 = ic.sizes_[2].divisor, isize3 = ic.sizes_[3].divisor, isize4 = ic.sizes_[4].divisor, isize5 = ic.sizes_[5].divisor;
      assert(isize0==osize0 && isize1==osize1 && isize2==osize2 && isize3==osize3 && isize4==osize4 && osize5==isize5);

      auto ostride00 = oc.strides_[0][0], ostride10 = oc.strides_[1][0], ostride20 = oc.strides_[2][0], ostride30 = oc.strides_[3][0], ostride40 = oc.strides_[4][0], ostride50 = oc.strides_[5][0];
      auto ostride01 = oc.strides_[0][1], ostride11 = oc.strides_[1][1], ostride21 = oc.strides_[2][1], ostride31 = oc.strides_[3][1], ostride41 = oc.strides_[4][1], ostride51 = oc.strides_[5][1];

      auto istride02 = ic.strides_[0][0], istride12 = ic.strides_[1][0], istride22 = ic.strides_[2][0], istride32 = ic.strides_[3][0], istride42 = ic.strides_[4][0], istride52 = ic.strides_[5][0];
      auto istride03 = ic.strides_[0][1], istride13 = ic.strides_[1][1], istride23 = ic.strides_[2][1], istride33 = ic.strides_[3][1], istride43 = ic.strides_[4][1], istride53 = ic.strides_[5][1];
      auto istride04 = ic.strides_[0][2], istride14 = ic.strides_[1][2], istride24 = ic.strides_[2][2], istride34 = ic.strides_[3][2], istride44 = ic.strides_[4][2], istride54 = ic.strides_[5][2];
      auto istride05 = ic.strides_[0][3], istride15 = ic.strides_[1][3], istride25 = ic.strides_[2][3], istride35 = ic.strides_[3][3], istride45 = ic.strides_[4][3], istride55 = ic.strides_[5][3];
      
      int64_t grid = (N + block_work_size() - 1) / block_work_size();
      auto stream = at::cuda::getCurrentCUDAStream();
      
      legacy_optim_unrolled_kernel_for_multi_outputs_out2_arity4_dim6<func_t, array_t, rets_t, ret0_t, ret1_t,
                                                                      arg0_t, arg1_t, arg2_t, arg3_t><<<
                                                                      grid, num_threads(), 0, stream>>>(
                                                                      N, f, data,
                                                                      isize0, isize1, isize2, isize3, isize4, isize5,
                                                                      ostride00, ostride10, ostride20, ostride30, ostride40, ostride50,
                                                                      ostride01, ostride11, ostride21, ostride31, ostride41, ostride51,
                                                                      istride02, istride12, istride22, istride32, istride42, istride52,
                                                                      istride03, istride13, istride23, istride33, istride43, istride53,
                                                                      istride04, istride14, istride24, istride34, istride44, istride54,
                                                                      istride05, istride15, istride25, istride35, istride45, istride55
                                                                     );
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

  }
};


template <>
class launch_legacy_optim_unrolled_kernel_for_multi_outputs<2,1>{
public:
  launch_legacy_optim_unrolled_kernel_for_multi_outputs() {}
  
  template<typename func_t, typename array_t, typename inp_calc_t, typename out_calc_t>
  void operator() (int64_t N, const func_t& f, array_t data, inp_calc_t ic, out_calc_t oc, int ndim){

    TORCH_INTERNAL_ASSERT(N > 0 && N <= std::numeric_limits<int32_t>::max());
    using traits = function_traits<func_t>;

    using args_t = typename traits::ArgsTuple;
    using arg0_t = typename traits::template arg<0>::type;

    using rets_t = typename traits::result_type;
    using ret0_t = typename thrust::tuple_element<0, rets_t>::type;
    using ret1_t = typename thrust::tuple_element<1, rets_t>::type;

    if(ndim==1){
      auto osize0 = oc.sizes_[0].divisor;
      auto isize0 = ic.sizes_[0].divisor;
      assert(isize0==osize0);

      auto ostride00 = oc.strides_[0][0];
      auto ostride01 = oc.strides_[0][1];

      auto istride02 = ic.strides_[0][0];

      int64_t grid = (N + block_work_size() - 1) / block_work_size();
      auto stream = at::cuda::getCurrentCUDAStream();
      
      legacy_optim_unrolled_kernel_for_multi_outputs_out2_arity1_dim1<func_t, array_t, rets_t, ret0_t, ret1_t, arg0_t><<<
                                                                      grid, num_threads(), 0, stream>>>(
                                                                      N, f, data,
                                                                      isize0, 
                                                                      ostride00, 
                                                                      ostride01, 
                                                                      istride02
                                                                     );
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      
    }else if(ndim==2){
      auto osize0 = oc.sizes_[0].divisor, osize1 = oc.sizes_[1].divisor;
      auto isize0 = ic.sizes_[0].divisor, isize1 = ic.sizes_[1].divisor;
      assert(isize0==osize0 && isize1==osize1);

      auto ostride00 = oc.strides_[0][0], ostride10 = oc.strides_[1][0];
      auto ostride01 = oc.strides_[0][1], ostride11 = oc.strides_[1][1];

      auto istride02 = ic.strides_[0][0], istride12 = ic.strides_[1][0];

      int64_t grid = (N + block_work_size() - 1) / block_work_size();
      auto stream = at::cuda::getCurrentCUDAStream();
      
      legacy_optim_unrolled_kernel_for_multi_outputs_out2_arity1_dim2<func_t, array_t, rets_t, ret0_t, ret1_t, arg0_t><<<
                                                                      grid, num_threads(), 0, stream>>>(
                                                                      N, f, data,
                                                                      isize0, isize1,
                                                                      ostride00, ostride10,
                                                                      ostride01, ostride11,
                                                                      istride02, istride12
                                                                     );
      C10_CUDA_KERNEL_LAUNCH_CHECK();

    }else if(ndim==3){
      auto osize0 = oc.sizes_[0].divisor, osize1 = oc.sizes_[1].divisor, osize2 = oc.sizes_[2].divisor;
      auto isize0 = ic.sizes_[0].divisor, isize1 = ic.sizes_[1].divisor, isize2 = ic.sizes_[2].divisor;
      assert(isize0==osize0 && isize1==osize1 && isize2==osize2);

      auto ostride00 = oc.strides_[0][0], ostride10 = oc.strides_[1][0], ostride20 = oc.strides_[2][0];
      auto ostride01 = oc.strides_[0][1], ostride11 = oc.strides_[1][1], ostride21 = oc.strides_[2][1];

      auto istride02 = ic.strides_[0][0], istride12 = ic.strides_[1][0], istride22 = ic.strides_[2][0];

      int64_t grid = (N + block_work_size() - 1) / block_work_size();
      auto stream = at::cuda::getCurrentCUDAStream();
      
      legacy_optim_unrolled_kernel_for_multi_outputs_out2_arity1_dim3<func_t, array_t, rets_t, ret0_t, ret1_t, arg0_t><<<
                                                                      grid, num_threads(), 0, stream>>>(
                                                                      N, f, data,
                                                                      isize0, isize1, isize2,
                                                                      ostride00, ostride10, ostride20,
                                                                      ostride01, ostride11, ostride21,
                                                                      istride02, istride12, istride22
                                                                     );
      C10_CUDA_KERNEL_LAUNCH_CHECK();

    }else if(ndim==4){
      auto osize0 = oc.sizes_[0].divisor, osize1 = oc.sizes_[1].divisor, osize2 = oc.sizes_[2].divisor, osize3 = oc.sizes_[3].divisor;
      auto isize0 = ic.sizes_[0].divisor, isize1 = ic.sizes_[1].divisor, isize2 = ic.sizes_[2].divisor, isize3 = ic.sizes_[3].divisor;
      assert(isize0==osize0 && isize1==osize1 && isize2==osize2 && isize3==osize3);

      auto ostride00 = oc.strides_[0][0], ostride10 = oc.strides_[1][0], ostride20 = oc.strides_[2][0], ostride30 = oc.strides_[3][0];
      auto ostride01 = oc.strides_[0][1], ostride11 = oc.strides_[1][1], ostride21 = oc.strides_[2][1], ostride31 = oc.strides_[3][1];

      auto istride02 = ic.strides_[0][0], istride12 = ic.strides_[1][0], istride22 = ic.strides_[2][0], istride32 = ic.strides_[3][0];

      int64_t grid = (N + block_work_size() - 1) / block_work_size();
      auto stream = at::cuda::getCurrentCUDAStream();
      
      legacy_optim_unrolled_kernel_for_multi_outputs_out2_arity1_dim4<func_t, array_t, rets_t, ret0_t, ret1_t, arg0_t><<<
                                                                      grid, num_threads(), 0, stream>>>(
                                                                      N, f, data,
                                                                      isize0, isize1, isize2, isize3,
                                                                      ostride00, ostride10, ostride20, ostride30,
                                                                      ostride01, ostride11, ostride21, ostride31,
                                                                      istride02, istride12, istride22, istride32
                                                                     );
      C10_CUDA_KERNEL_LAUNCH_CHECK();

    }

  }
};

