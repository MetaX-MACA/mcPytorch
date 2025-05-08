import torch
import torch.nn as nn


#支持的数据类型
Dtype_bytes = {"Byte":1, "Char":1, "Short":2, "Int":4,    "BFloat16":2,
               "Long":8, "Half":2, "Float":4, "Double":8, "Bool":1}

arity1_tensor_map = {
    "direct_copy_kernel_cuda" :     {"func":"copy_",  "extra_param":{}},
}

arity1_torch_map = {
     "GeluCUDAKernelImpl":                    {"func": nn.GELU(),          "extra_param":{}},
     "silu_kernel":                           {"func": nn.SiLU(),          "extra_param":{}},
     "BUnaryFunctor&&MulFunctor":             {"func": torch.div,          "extra_param":{"other":4}},
     "AUnaryFunctor&&MulFunctor":             {"func": torch.mul,          "extra_param":{"other":4}},
     "launch_clamp_scalar":                   {"func": torch.clamp,        "extra_param":{"min":-0.5, "max":0.5}},
     "cos_kernel_cuda":                       {"func": torch.cos,          "extra_param":{}},
     "sin_kernel_cuda":                       {"func": torch.sin,          "extra_param":{}},
     "tanh_kernel_cuda":                      {"func": torch.tanh,         "extra_param":{}},
     "neg_kernel_cuda":                       {"func": torch.neg,          "extra_param":{}},
     "sigmoid_kernel_cuda":                   {"func": torch.sigmoid,      "extra_param":{}},
     "leaky_relu_kernel":                     {"func": nn.LeakyReLU(0.1),  "extra_param":{}},
     "compare_scalar_kernel":                 {"func": torch.ge,           "extra_param":{"other":0.5}},
     "pow_tensor_scalar_kernel_impl":         {"func": torch.pow,          "extra_param":{"exponent":2}},
     "rsqrt_kernel_cuda":                     {"func": torch.rsqrt,        "extra_param":{}},
     "exp_kernel_cuda":                       {"func": torch.exp,          "extra_param":{}},
     "log_kernel_cuda":                       {"func": torch.log,          "extra_param":{}},
     "div_trunc_kernel_cuda":                 {"func": torch.div,          "extra_param":{"other" : 0.1, "rounding_mode" : "trunc"}},
     "div_floor_kernel_cuda":                 {"func": torch.div,          "extra_param":{"other" : 0.1, "rounding_mode" : "floor"}},
     "BUnaryFunctor&&remainder_kernel_cuda":  {"func": torch.remainder,    "extra_param":{"other" : 0.1}},
     "BUnaryFunctor&&fmod_kernel_cuda":       {"func": torch.fmod,         "extra_param":{"other" : 0.1}},
     "AUnaryFunctor&&CompareEqFunctor":       {"func": torch.eq,           "extra_param":{"other" : 0.1}},
     "sqrt_kernel_cuda":                      {"func": torch.sqrt,         "extra_param":{}},
     "logical_not_kernel_cuda":               {"func": torch.logical_not,  "extra_param":{}},
     "reciprocal_kernel_cuda":                {"func": torch.reciprocal,   "extra_param":{}},
     "CUDAFunctorOnSelf_add":                 {"func": torch.add,          "extra_param":{"other" : 0.1}},
     "CUDAFunctorOnOther_add":                {"func": torch.add,          "extra_param":{"other" : 0.1}, "substitue": "CUDAFunctorOnSelf_add"},
}

arity2_tensor_map = {
    "masked_fill_kernel":           {"func":"masked_fill",  "extra_param":{"value":1}},
    "GeluBackwardCUDAKernelImpl":   {"func":"backward",  "forward_func":   nn.GELU(),           "extra_param":{}},
    "leaky_relu_backward_kernel":   {"func":"backward",  "forward_func":   nn.LeakyReLU(0.1),   "extra_param":{}},
    "silu_backward_kernel":         {"func":"backward",  "forward_func":   nn.SiLU(),           "extra_param":{}},
    "sigmoid_backward_kernel_cuda": {"func":"backward",  "forward_func":   torch.sigmoid,       "extra_param":{}},
    "tanh_backward_kernel_cuda":    {"func":"backward",  "forward_func":   torch.tanh,          "extra_param":{}},
    "masked_scale_kernel":          {"func":"masked_fill","extra_param":{"value":1}, "substitue": "masked_fill_kernel"},
}

arity2_torch_map = {
    "CompareFunctor":                          {"func": torch.lt,  "extra_param":{}},
    "CompareEqFunctor":                        {"func": torch.eq,  "extra_param":{}},
    "BinaryFunctor&&MulFunctor":               {"func": torch.mul, "extra_param":{}},
    "BinaryFunctor&&DivFunctor":               {"func": torch.div, "extra_param":{}},
    "GroupNormBackwardKernelImplInternal" :    {"func": torch.mul, "extra_param":{}, "substitue": "MulFunctor"},
    "pow_tensor_tensor_kernel" :               {"func": torch.pow, "extra_param":{}},
    "CUDAFunctor_add" :                        {"func": torch.add, "extra_param":{}},
    "glu_kernel":                              {"func": torch.add, "extra_param":{}, "substitue": "CUDAFunctor_add"},
    "batch_norm_elementwise_backward_eval":    {"func": torch.mul, "extra_param":{}, "substitue": "MulFunctor"},
    "div_trunc_kernel_cuda":                   {"func": torch.div, "extra_param":{"rounding_mode" : "trunc"}},
    "div_floor_kernel_cuda":                   {"func": torch.div, "extra_param":{"rounding_mode" : "floor"}},
    "remainder_kernel_cuda" :                  {"func": torch.remainder, "extra_param":{}},
    "fmod_kernel_cuda" :                       {"func": torch.fmod, "extra_param":{}},
    "BitwiseOrFunctor" :                       {"func": torch.bitwise_or, "extra_param":{}},
    "BitwiseXorFunctor" :                      {"func": torch.bitwise_xor, "extra_param":{}},
    "BitwiseAndFunctor" :                      {"func": torch.bitwise_and, "extra_param":{}},
    "logical_and_kernel_cuda" :                {"func": torch.logical_and, "extra_param":{}},
    "logical_or_kernel_cuda":                  {"func": torch.logical_or, "extra_param":{}},
    "logical_xor_kernel_cuda":                 {"func": torch.logical_xor, "extra_param":{}},
    "mse_kernel_cuda":                         {"func": nn.MSELoss(), "extra_param":{}},
}

arity3_tensor_map = {}

arity3_torch_map = {
   "addcmul_cuda_kernel" :                     {"func": torch.addcmul, "extra_param":{}},
   "GroupNormKernelImplInternal":              {"func": torch.addcmul, "extra_param":{}, "substitue": "addcmul_cuda_kernel"},
   "batch_norm_elementwise_backward_eval":     {"func": torch.addcmul, "extra_param":{}, "substitue": "addcmul_cuda_kernel"},
   "where_kernel_impl" :                       {"func": torch.where,   "extra_param":{}},
   "mse_backward_cuda_kernel":                 {"func": torch.addcmul, "extra_param":{}, "substitue": "addcmul_cuda_kernel"},
}

