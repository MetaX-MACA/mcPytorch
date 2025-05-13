#include <ATen/native/cuda/fused_adamw_impl.cuh>

#include <ATen/Dispatch.h>
#include <ATen/native/ForeachUtils.h>
#include <ATen/native/cuda/fused_adam_utils.cuh>
#include <ATen/native/cuda/MultiTensorApply.cuh>
#include <vector>
#include <ATen/cuda/CUDAGraphsUtils.cuh>

namespace at { namespace native {

void _fused_adamw_cuda_impl_(
    at::TensorList params,
    at::TensorList grads,
    at::TensorList exp_avgs,
    at::TensorList exp_avg_sqs,
    at::TensorList state_steps,
    const double lr,
    const double beta1,
    const double beta2,
    const double weight_decay,
    const double eps,
    const bool maximize,
    const c10::optional<at::Tensor>& grad_scale,
    const c10::optional<at::Tensor>& found_inf
) {
  std::vector<std::vector<at::Tensor>> tensor_lists{
    params.vec(), grads.vec(), exp_avgs.vec(), exp_avg_sqs.vec() };

  float* grad_scale_ptr = grad_scale.has_value() ? grad_scale->data_ptr<float>() : nullptr;
  float* found_inf_ptr = found_inf.has_value() ? found_inf->data_ptr<float>() : nullptr;

  AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16, params[0].scalar_type(),
      "fused_adamw_kernel_cuda", [&]() {
        bool is_opt = at::maca::get_maca_enable_fast_multitensorapply_fused() && at::cuda::currentStreamCaptureStatus() == at::cuda::CaptureStatus::None;
        if (maca_likely(is_opt)) {
          multi_tensor_apply_for_fused_optimizer_opt<4>(
            tensor_lists,
            state_steps,
            FusedAdamMathFunctorOpt<scalar_t, 4>(),
            lr,
            beta1,
            beta2,
            weight_decay,
            eps,
            maximize,
            /* amsgrad */false,
            grad_scale_ptr,
            found_inf_ptr,
            ADAM_MODE::ADAMW);
        } else {
          multi_tensor_apply_for_fused_optimizer<4>(
            tensor_lists,
            state_steps,
            FusedAdamMathFunctor<scalar_t, 4>(),
            lr,
            beta1,
            beta2,
            weight_decay,
            eps,
            maximize,
            /* amsgrad */false,
            grad_scale_ptr,
            found_inf_ptr,
            ADAM_MODE::ADAMW);
        }
        });
}

} } // namespace at::native
