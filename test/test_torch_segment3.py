import torch
import unittest
from itertools import product
from common_util import *

class TestTorchFunctionSegment3(unittest.TestCase):
    activation_func_lst = [
        ('round', 1e-4, 1e-4),
        ('sigmoid', 1e-1, 1e-3),
        ('sign', 1e-4, 1e-4),
        ('rsqrt', 1e-1, 1e-3),
        ('sgn', 1e-4, 1e-4),
        ('sin', 1e-3, 1e-3),
        ('sinh', 1e-3, 1e-3),
        ('sqrt', 1e-3, 1e-3),
        ('square', 1e-3, 1e-3),
        ('tan', 1e-3, 1e-3),
        ('tanh', 1e-3, 1e-3),
        ('trunc', 1e-4, 1e-4),
        ('signbit', 1e-4, 1e-4)
    ]

    arthmetic_func_lst = [
        ('remainder', 1e-4),
        ('sub', 1e-4),
        ('subtract', 1e-4),
        ('xlogy', 1e-3),
        ('true_divide', 1e-2)
    ]

    reduce_func_lst = [
        ('argmax', 1e-4),
        ('argmin', 1e-4),
        ('aminmax', 1e-4),
        ('all', 1e-4),
        ('any', 1e-4),
        ('max', 1e-4),
        ('min', 1e-4),
        ('mean', 1e-3),
        ('median', 1e-4),
        ('nansum', 1e-1),
        ('prod', 1e-4),
        ('sum', 1e-2),
        ('count_nonzero', 1e-4)
    ]

    reduce_dim_func_lst = [
        ('amax', 1e-4),
        ('amin', 1e-4),
        ('all', 1e-4),
        ('any', 1e-4),
        ('max', 1e-4),
        ('min', 1e-4),
        ('logsumexp', 3e-3),
        ('mean', 1e-3),
        ('nanmean', 1e-3),
        ('median', 1e-4),
        ('nanmedian', 1e-4),
        ('mode', 1e-4),
        ('nansum', 1e-1),
        ('prod', 1e-4),
        ('sum', 1e-2)
    ]

    quantile_func_lst = [
        ('quantile', 1e-3, 3e-3),
        ('nanquantile', 1e-3, 3e-3)
    ]

    std_func_lst = [
        ('std', 1e-2, 1e-1),
        ('std_mean', 1e-3, 1e-2)
    ]

    var_func_lst = [
        ('var', 1e-2, 1e-1),
        ('var_mean', 1e-3, 1e-2)
    ]

    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_fp_dtypes(include_bfloat16=False)))
    @parametrize("func_type,fp16_fwd_tol,fp16_bwd_tol", activation_func_lst, name_fn=lambda func_type, fp16_fwd_tol, fp16_bwd_tol: '{}'.format(func_type))
    def test_activation(self, func_type, fp16_fwd_tol, fp16_bwd_tol, device, dtype):
        torch.manual_seed(1)
        def helper(shape):
            func = getattr(torch, func_type)
            shape_list = [(shape)]
            forward_inputs = gendata(shape_list, type_list=[dtype], rand_algo="rand")
            if dtype == torch.float16:
                fwd_tol = fp16_fwd_tol
                bwd_tol = fp16_bwd_tol
            else:
                fwd_tol = 1e-4
                bwd_tol = 1e-4
            if func_type == 'signbit':
                runtestapi(func=func, type_dict={torch.float:{torch.float16}}, fwd_input_list=forward_inputs, enable_backward=False, fwd_tol=fwd_tol, bwd_tol=bwd_tol)
            else:
                runtestapi(func=func, type_dict={torch.float:{torch.float16}}, fwd_input_list=forward_inputs, enable_backward=True, fwd_tol=fwd_tol, bwd_tol=bwd_tol)
        helper((1,10))

    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_fp_dtypes(include_bfloat16=False)))
    @parametrize("func_type,fp16_fwd_tol", arthmetic_func_lst, name_fn=lambda func_type, fp16_fwd_tol: '{}'.format(func_type))
    def test_arthmetic(self, func_type, fp16_fwd_tol, device, dtype):
        def helper(shape_list):
            func = getattr(torch, func_type)
            if dtype == torch.float16:
                fwd_tol = fp16_fwd_tol
            else:
                fwd_tol = 1e-4
            forward_inputs = gendata(shape_list, type_list=[dtype], rand_algo="rand")
            # TODO(yuliu): derivative for the other value.
            runtestapi(func=func, type_dict={torch.float:{torch.float16}}, fwd_input_list=forward_inputs, enable_backward=False, fwd_tol=fwd_tol)
        helper([(1,10), (1,10)])

    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_fp_dtypes(include_bfloat16=False)))
    @parametrize("func_type,fp16_fwd_tol", reduce_func_lst, name_fn=lambda func_type, fp16_fwd_tol: '{}'.format(func_type))
    def test_reduce(self, func_type, fp16_fwd_tol, device, dtype):
        def helper(shape_list):
            func = getattr(torch, func_type)
            if dtype == torch.float16:
                fwd_tol = fp16_fwd_tol
            else:
                fwd_tol = 1e-4
            forward_inputs = gendata(shape_list, type_list=[dtype], rand_algo="rand")
            # TODO(yuliu): derivative for the other value.
            runtestapi(func=func, type_dict={torch.float:{torch.float16}}, fwd_input_list=forward_inputs, enable_backward=False, fwd_tol=fwd_tol)
        helper([(10,10)])

    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_fp_dtypes(include_bfloat16=False)))
    @parametrize("func_type,fp16_fwd_tol", reduce_dim_func_lst, name_fn=lambda func_type, fp16_fwd_tol: '{}'.format(func_type))
    def test_reduce_dim(self, func_type, fp16_fwd_tol, device, dtype):
        def helper(shape_list, dim, keepdim):
            func = getattr(torch, func_type)
            forward_inputs = gendata(shape_list, type_list=[dtype], rand_algo="rand")
            if dtype == torch.float16:
                fwd_tol = fp16_fwd_tol
            else:
                fwd_tol = 1e-4
            forward_inputs.append(dim)
            forward_inputs.append(keepdim)
            # TODO(yuliu): derivative for the other value.
            runtestapi(func=func, type_dict={torch.float:{torch.float16}}, fwd_input_list=forward_inputs, enable_backward=False, fwd_tol=fwd_tol)
        helper([(10,10)], 0, False)
        helper([(10,10)], 1, True)

    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_fp_dtypes(include_bfloat16=False)))
    def test_norm(self, device, dtype):
        def helper(shape_list, p, dim, keepdim):
            forward_inputs = gendata(shape_list, type_list=[dtype], rand_algo="rand")
            forward_inputs.append(p)
            forward_inputs.append(dim)
            forward_inputs.append(keepdim)
            if dtype == torch.float16:
                fwd_tol = 1e-3
                bwd_tol = 1e-2
            else:
                fwd_tol = 1e-4
                bwd_tol = 1e-4
            runtestapi(func=torch.norm, type_dict={torch.float:{torch.float16}}, fwd_input_list=forward_inputs, enable_backward=True, fwd_tol=fwd_tol, bwd_tol=bwd_tol)
        helper([(10,10)], 'fro', 0, False)
        helper([(10,10)], 1, 1, True)

    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_fp_dtypes(include_half=False, include_bfloat16=False)))
    @parametrize("func_type,fp16_fwd_tol,fp16_bwd_tol", quantile_func_lst, name_fn=lambda func_type, fp16_fwd_tol, fp16_bwd_tol: '{}'.format(func_type))
    def test_quantile(self, func_type, fp16_fwd_tol, fp16_bwd_tol, device, dtype):
        def helper(shape_list, q, dim, keepdim, interpolation):
            func = getattr(torch, func_type)
            if dtype == torch.float16:
                fwd_tol = fp16_fwd_tol
                bwd_tol = fp16_bwd_tol
            else:
                fwd_tol = 1e-4
                bwd_tol = 1e-4
            forward_inputs = gendata(shape_list, type_list=[dtype], rand_algo="rand")
            inputs_dict = {'input': forward_inputs[0], 'q': q, 'dim': dim, 'keepdim': keepdim,
                           'interpolation': interpolation}
            runtestapi(func=func, type_dict={torch.float:{torch.float16}}, fwd_input_list=inputs_dict, enable_backward=True, fwd_tol=fwd_tol, bwd_tol=bwd_tol)
        helper([(10,10)], 0.6, 0, False, 'linear')
        helper([(10,10)], 0.6, 1, True, 'nearest')

    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_fp_dtypes(include_bfloat16=False)))
    @parametrize("func_type,fp16_fwd_tol,fp16_bwd_tol", std_func_lst, name_fn=lambda func_type, fp16_fwd_tol, fp16_bwd_tol: '{}'.format(func_type))
    def test_std(self, func_type, fp16_fwd_tol, fp16_bwd_tol, device, dtype):
        def helper(shape_list, dim, unbiased, keepdim):
            func = getattr(torch, func_type)
            if dtype == torch.float16:
                fwd_tol = fp16_fwd_tol
                bwd_tol = fp16_bwd_tol
            else:
                fwd_tol = 1e-4
                bwd_tol = 1e-4
            forward_inputs = gendata(shape_list, type_list=[dtype], rand_algo="rand")
            forward_inputs.append(dim)
            forward_inputs.append(unbiased)
            forward_inputs.append(keepdim)
            runtestapi(func=func, type_dict={torch.float:{torch.float16}}, fwd_input_list=forward_inputs, enable_backward=True, fwd_tol=fwd_tol, bwd_tol=bwd_tol)
        helper([(10,10)], 0, False, False)
        helper([(10,10)], 1, True, True)

    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_fp_dtypes(include_bfloat16=False)))
    @parametrize("func_type,fp16_fwd_tol,fp16_bwd_tol", var_func_lst, name_fn=lambda func_type, fp16_fwd_tol, fp16_bwd_tol: '{}'.format(func_type))
    def test_var(self, func_type, fp16_fwd_tol, fp16_bwd_tol, device, dtype):
        def helper(shape_list, dim, unbiased, keepdim):
            func = getattr(torch, func_type)
            if dtype == torch.float16:
                fwd_tol = fp16_fwd_tol
                bwd_tol = fp16_bwd_tol
            else:
                fwd_tol = 1e-4
                bwd_tol = 1e-4
            forward_inputs = gendata(shape_list, type_list=[dtype], rand_algo="rand")
            forward_inputs.append(dim)
            forward_inputs.append(unbiased)
            forward_inputs.append(keepdim)
            runtestapi(func=func, type_dict={torch.float:{torch.float16}}, fwd_input_list=forward_inputs, enable_backward=True, fwd_tol=fwd_tol, bwd_tol=bwd_tol)
        helper([(10,10)], 0, False, False)
        helper([(10,10)], 1, True, True)

    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_fp_dtypes(include_bfloat16=False)))
    def test_unique(self, device, dtype):
        def helper(shape_list, sort, return_inverse, return_counts):
            forward_inputs = gendata(shape_list, type_list=[dtype], rand_algo="rand")
            forward_inputs.append(sort)
            forward_inputs.append(return_inverse)
            forward_inputs.append(return_counts)
            # TODO(yuliu): derivative for the other value.
            runtestapi(func=torch.unique, type_dict={torch.float:{torch.float16}}, fwd_input_list=forward_inputs, enable_backward=False)
        helper([(10,10)], True, False, False)

    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_fp_dtypes(include_bfloat16=False)))
    def test_unique_consecutive(self, device, dtype):
        def helper(shape_list, return_inverse, return_counts):
            forward_inputs = gendata(shape_list, type_list=[dtype], rand_algo="rand")
            forward_inputs.append(return_inverse)
            forward_inputs.append(return_counts)
            # TODO(yuliu): derivative for the other value.
            runtestapi(func=torch.unique_consecutive, type_dict={torch.float:{torch.float16}}, fwd_input_list=forward_inputs, enable_backward=False)
        helper([(10,10)], False, False)
        helper([(10,10)], True, False)

    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_fp_dtypes(include_bfloat16=False)))
    def test_dist(self, device, dtype):
        def helper(shape_list, p):
            if dtype == torch.float16:
                fwd_tol = 3e-4
                bwd_tol = 3e-1
            else:
                fwd_tol = 1e-4
                bwd_tol = 1e-4
            forward_inputs = gendata(shape_list, type_list=[dtype], rand_algo="rand")
            forward_inputs.append(p)
            runtestapi(func=torch.dist, type_dict={torch.float:{torch.float16}}, fwd_input_list=forward_inputs, enable_backward=True, fwd_tol=fwd_tol, bwd_tol=bwd_tol)
        helper([(10,10), (10,10)], 3.5)
        helper([(10,10), (10,10)], 0)

    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_fp_dtypes(include_half=False, include_bfloat16=False)))
    def test_argsort(self, device, dtype):
        def helper(shape_list, dim, descending):
            forward_inputs = gendata(shape_list, type_list=[dtype], rand_algo="rand")
            forward_inputs.append(dim)
            forward_inputs.append(descending)
            # TODO(yuliu): derivative for the other value.
            runtestapi(func=torch.argsort, type_dict={torch.float:{torch.float16}}, fwd_input_list=forward_inputs, enable_backward=False)
        helper([(10,10)], 1, True)
        helper([(10,10)], 1, False)

instantiate_device_type_tests(TestTorchFunctionSegment3, globals())

if __name__ == "__main__":
    unittest.main()
