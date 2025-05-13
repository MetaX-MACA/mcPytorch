#include <mcblas/mcblas.h>
#include <ATen/cuda/CUDAContext.h>
#include <common/maca_fp16.hpp>
#include <ATen/TensorIndexing.h>
#include <ATen/ATen.h>
#include <ATen/core/TensorBody.h>
#include <ATen/core/TensorBase.h>
typedef enum { MCBLAS_COL_MAJOR = 0, MCBLAS_ROW_MAJOR = 1 } mcblasLayOut_t;
MCBLASAPIENTRY mcblasStatus_t flashattention(mcblasHandle_t handle,
                                              mcblas_half *Q,
                                              mcblas_half *K,
                                              mcblas_half *V,
                                              int batch_count0,
                                              int batch_count1,
                                              int mQ,
                                              int nQ,
                                              int ldbc0Q,
                                              int ldbc1Q,
                                              int ldmQ,
                                              int mK,
                                              int nK,
                                              int ldbc0K,
                                              int ldbc1K,
                                              int ldmK,
                                              int mV,
                                              int nV,
                                              int ldbc0V,
                                              int ldbc1V,
                                              int ldmV,
                                              mcblas_half *O,
                                              int mO,
                                              int nO,
                                              int ldbc0O,
                                              int ldbc1O,
                                              int ldmO,
                                              mcblasLayOut_t layout = MCBLAS_ROW_MAJOR);

namespace at{
namespace native {
    Tensor _scaled_dot_product_mha_blas_cuda(const Tensor& query_, const Tensor& key, const Tensor& value, double dropout_p, bool is_causal){
        Tensor result = at::empty({query_.size(0), query_.size(1), query_.size(2), value.size(3)}, query_.options());
        auto handle = at::cuda::getCurrentCUDABlasHandle();
        flashattention(handle, (__half*)query_.data_ptr(), (__half*)key.data_ptr(), (__half*)value.data_ptr(), \
        query_.size(0), query_.size(1), \
        query_.size(2), query_.size(3), query_.strides()[0], query_.strides()[1], query_.strides()[2], \
        key.size(2), key.size(3), key.strides()[0], key.strides()[1], key.strides()[2], \
        value.size(2), value.size(3), value.strides()[0], value.strides()[1], value.strides()[2], \
        (__half*)result.data_ptr(), \
        result.size(2), result.size(3), result.strides()[0], result.strides()[1], result.strides()[2] \
        );
        return result;
    }
}}
