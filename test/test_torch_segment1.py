import torch
import unittest
from numpy import inf
from common_util import *


class TestErf(unittest.TestCase):
    '''
    test erf, erfc, erfinv
    '''
    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_fp_dtypes()))
    def test_erf_erfc(self, device, dtype):
        shape_list = [(10,), (1, 2), (3, 4, 5)]

        fn_list = [{"fn": torch.erf, "tol": {"fwd_tol": {torch.float16: 1e-2, torch.bfloat16: 1e-2},
                                            "bwd_tol": {torch.float16: 1e-2, torch.bfloat16: 1e-2}}},
                    {"fn": torch.erfc, "tol": {"fwd_tol": {torch.float16: 1e-2, torch.bfloat16: 1e-2},
                                            "bwd_tol": {torch.float16: 1e-2, torch.bfloat16: 1e-2}}}]

        forward_inputs = gendata(shape_list, type_list=[dtype])
        for input in forward_inputs:
            for fn_item in fn_list:
                func = fn_item["fn"]
                fwd_tol = gettol(fn_item, dtype, "fwd_tol")
                bwd_tol = gettol(fn_item, dtype, "bwd_tol")
                runtestapi(func=func, fwd_input_list=[input], enable_backward=True, 
                        type_dict={torch.float:{torch.float16, torch.bfloat16}}, fwd_tol=fwd_tol, bwd_tol=bwd_tol)

    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_fp_dtypes(include_bfloat16=False)))
    def test_erfinv(self, device, dtype):
        shape_list = [(10,), (1, 2), (3, 4, 5)]

        fn_list = [{"fn": torch.erfinv, "tol": {"fwd_tol": {torch.float16: 1e-2, torch.bfloat16: 1e-2},
                                            "bwd_tol": {torch.float16: 1e-2, torch.bfloat16: 1e-2}}}]
        torch.manual_seed(0)
        forward_inputs = gendata(shape_list, type_list=[dtype], rand_algo="uniform", lower=-1.0, upper=1.0)
        for input in forward_inputs:
            for fn_item in fn_list:
                func = fn_item["fn"]
                fwd_tol = gettol(fn_item, dtype, "fwd_tol")
                bwd_tol = gettol(fn_item, dtype, "bwd_tol")
                runtestapi(func=func, fwd_input_list=[input], enable_backward=True, 
                        type_dict={torch.float:{torch.float16, torch.bfloat16}}, fwd_tol=fwd_tol, bwd_tol=bwd_tol)

class TestExp(unittest.TestCase):
    '''
    test exp, exp2, expm1
    '''
    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_fp_dtypes()))
    def test_exp(self, device, dtype):
        shape_list = [(10,), (1, 2), (3, 4, 5)]

        fn_list = [{"fn": torch.exp, "tol": {"fwd_tol": {torch.float16: 1e-2, torch.bfloat16: 1e-2},
                                            "bwd_tol": {torch.float16: 1e-2, torch.bfloat16: 1e-2}}},
                    {"fn": torch.exp2, "tol": {"fwd_tol": {torch.float16: 1e-2, torch.bfloat16: 1e-2},
                                            "bwd_tol": {torch.float16: 1e-2, torch.bfloat16: 1e-2}}},
                    {"fn": torch.expm1, "tol": {"fwd_tol": {torch.float16: 1e-2, torch.bfloat16: 1e-2},
                                            "bwd_tol": {torch.float16: 1e-2, torch.bfloat16: 1e-2}}}]
        torch.manual_seed(0)
        forward_inputs = gendata(shape_list, type_list=[dtype])
        for input in forward_inputs:
            for fn_item in fn_list:
                func = fn_item["fn"]
                fwd_tol = gettol(fn_item, dtype, "fwd_tol")
                bwd_tol = gettol(fn_item, dtype, "bwd_tol")
                runtestapi(func=func, fwd_input_list=[input], enable_backward=True, 
                        type_dict={torch.float:{torch.float16, torch.bfloat16}}, fwd_tol=fwd_tol, bwd_tol=bwd_tol)


class TestFakeQuantize(unittest.TestCase):
    '''
    test fake_quantize_per_channel_affine, fake_quantize_per_tensor_affine
    '''
    @onlyCUDA
    @dtypesIfCUDA(*set([torch.float]))
    def test_fake_quantize_per_channel_affine(self, device, dtype):
        input_shape_list = [(2, 2, 2)]
        input = gendata(input_shape_list, type_list=[dtype], )
        scale = [(torch.randn(2) + 1) * 0.05]
        zero_points = [torch.zeros(2).to(torch.int32)]
        axis_all = [0, 1, 2]
        quant_min = [0]
        quant_max = [255]

        func = torch.fake_quantize_per_channel_affine
        for axis in axis_all:
            fwd_input_list = input + scale + zero_points + [axis] + quant_min + quant_max
            runtestapi(func=func, fwd_input_list=fwd_input_list, enable_backward=True, 
                        input_grad_skip_idx=[1], type_dict={torch.float:{torch.float16, torch.bfloat16}})
        
    @onlyCUDA
    @dtypesIfCUDA(*set([torch.float]))
    def test_fake_quantize_per_tensor_affine(self, device, dtype):
        input_shape_list = [(2, 2, 2)]
        input = gendata(input_shape_list, type_list=[dtype])
        scale_all = [torch.tensor(0.1), 0.1]
        zero_points_all = [torch.tensor(0).to(torch.int32), 0]
        quant_min = [0]
        quant_max = [255]

        func = torch.fake_quantize_per_tensor_affine
        for scale in scale_all:
            for zero_points in zero_points_all:
                fwd_input_list = input + [scale] + [zero_points] + quant_min + quant_max
                runtestapi(func=func, fwd_input_list=fwd_input_list, enable_backward=True, 
                            input_grad_skip_idx=[1], type_dict={torch.float:{torch.float16, torch.bfloat16}})


class TestLog(unittest.TestCase):
    '''
    test log, log10, log1p, log2
    '''
    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_fp_dtypes()))
    def test_log(self, device, dtype):
        input_shape_list = [(5)]
        fn_list = [{"fn": torch.log, "tol": {"fwd_tol": {torch.float16: 1e-2, torch.bfloat16: 1e-2},
                                            "bwd_tol": {torch.float16: 1e-2, torch.bfloat16: 1e-2}}},
                    {"fn": torch.log10, "tol": {"fwd_tol": {torch.float16: 1e-2, torch.bfloat16: 1e-2},
                                            "bwd_tol": {torch.float16: 1e-2, torch.bfloat16: 1e-2}}},
                    {"fn": torch.log1p, "tol": {"fwd_tol": {torch.float16: 1e-2, torch.bfloat16: 1e-2},
                                            "bwd_tol": {torch.float16: 1e-2, torch.bfloat16: 1e-2}}},
                    {"fn": torch.log2, "tol": {"fwd_tol": {torch.float16: 1e-2, torch.bfloat16: 1e-2},
                                            "bwd_tol": {torch.float16: 1e-2, torch.bfloat16: 1e-2}}}]

        forward_inputs = gendata(input_shape_list, type_list=[dtype], rand_algo="uniform", lower=0, upper=100)
        for input in forward_inputs:
            for fn_item in fn_list:
                func = fn_item["fn"]
                fwd_tol = gettol(fn_item, dtype, "fwd_tol")
                bwd_tol = gettol(fn_item, dtype, "bwd_tol")
                runtestapi(func=func, fwd_input_list=[input], enable_backward=True, 
                            type_dict={torch.float:{torch.float16, torch.bfloat16}}, fwd_tol=fwd_tol, bwd_tol=bwd_tol)


class TestLogadd(unittest.TestCase):
    '''
    test loggaddexp, loggaddexp2
    '''
    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_fp_dtypes(include_half=False)))
    def test_logadd(self, device, dtype):
        input_shape_list = [(5)]
        other_shape_list = [(5)]

        fn_list = [{"fn": torch.logaddexp, "tol": {"fwd_tol": {torch.float16: 1e-2, torch.bfloat16: 1e-2}, "bwd_tol": {torch.float16: 1e-2, torch.bfloat16: 1e-2}}}, 
                    {"fn": torch.logaddexp2, "tol": {"fwd_tol": {torch.float16: 1e-2, torch.bfloat16: 1e-2}, "bwd_tol": {torch.float16: 1e-2, torch.bfloat16: 1e-2}}}]

        inputs = gendata(input_shape_list, type_list=[dtype])   # [(info, data)]
        others = gendata(other_shape_list, type_list=[dtype])

        for input in inputs:    # input is a tuple, ((info, data), (info, data))
            for other in others:
                input_list = [input] + [other]

                for fn_item in fn_list:
                    func = fn_item["fn"]
                    fwd_tol = gettol(fn_item, dtype, "fwd_tol")
                    bwd_tol = gettol(fn_item, dtype, "bwd_tol")
                    runtestapi(func=func, fwd_input_list=input_list, enable_backward=True, 
                                type_dict={torch.float:{torch.float16, torch.bfloat16}}, fwd_tol=fwd_tol, bwd_tol=bwd_tol)


class TestLogical(unittest.TestCase):
    '''
    test logical_and, logical_or, logical_xor, logical_not
    '''
    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_math_dtypes("cuda")))
    def test_logical(self, device, dtype):
        input_shape_list = [(5)]
        other_shape_list = [(5)]

        fn_list = [{"fn": torch.logical_and}, {"fn": torch.logical_or}, {"fn": torch.logical_xor}]

        inputs = gendata(input_shape_list, type_list=[dtype])   # [(info, data)]
        others = gendata(other_shape_list, type_list=[dtype])

        for input in inputs:    # input is a tuple, ((info, data), (info, data))
            for other in others:
                input_list = [input] + [other]

                for fn_item in fn_list:
                    func = fn_item["fn"]
                    fwd_tol = gettol(fn_item, dtype, "fwd_tol")
                    bwd_tol = gettol(fn_item, dtype, "bwd_tol")
                    runtestapi(func=func, fwd_input_list=input_list, enable_backward=False, 
                               fwd_tol=fwd_tol, bwd_tol=bwd_tol)

    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_math_dtypes("cuda")))
    def test_logical_not(self, device, dtype):
        input_shape_list = [(5)]
        fn_list = [{"fn": torch.logical_not}]

        forward_inputs = gendata(input_shape_list, type_list=[dtype])   # [(info, data)]
        for input in forward_inputs:
            for fn_item in fn_list:
                func = fn_item["fn"]
                fwd_tol = gettol(fn_item, dtype, "fwd_tol")
                bwd_tol = gettol(fn_item, dtype, "bwd_tol")

                runtestapi(func=func, fwd_input_list=[input], enable_backward=False, 
                            fwd_tol=fwd_tol, bwd_tol=bwd_tol)


class TestLogit(unittest.TestCase):
    '''
    test logit
    '''
    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_fp_dtypes()))
    def test_logit(self, device, dtype):
        shape_list = [(10,), (1, 2), (3, 4, 5)]

        fn_list = [{"fn": torch.logit, "tol": {"fwd_tol": {torch.float16: 1e-2, torch.bfloat16: 1e-2},
                                            "bwd_tol": {torch.float16: 1e-2, torch.bfloat16: 1e-2}}}]

        forward_inputs = gendata(shape_list, type_list=[dtype], rand_algo="uniform", lower=0.0, upper=1.0)
        eps_inputs = [1e-6]
        for input in forward_inputs:
            for eps in eps_inputs:
                input_list = [input, eps]
                for fn_item in fn_list:
                    func = fn_item["fn"]
                    fwd_tol = gettol(fn_item, dtype, "fwd_tol")
                    bwd_tol = gettol(fn_item, dtype, "bwd_tol")

                    runtestapi(func=func, fwd_input_list=input_list, enable_backward=True, 
                                type_dict={torch.float:{torch.float16}}, fwd_tol=fwd_tol, bwd_tol=bwd_tol)

class TestHypot(unittest.TestCase):
    '''
    test hypot
    '''
    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_fp_dtypes()))
    def test_hypot(self, device, dtype):
        shape_list = [(10,), (1, 2), (3, 4, 5)]

        fn_list = [{"fn": torch.hypot, "tol": {"fwd_tol": {torch.float16: 1e-2, torch.bfloat16: 1e-2},
                                            "bwd_tol": {torch.float16: 1e-2, torch.bfloat16: 1e-2}}}]

        forward_inputs = gendata(shape_list, type_list=[dtype])
        forward_others = gendata(shape_list, type_list=[dtype])

        for i, input in enumerate(forward_inputs):
            other = forward_others[i]
            input_list = [input, other]

            for fn_item in fn_list:
                func = fn_item["fn"]
                fwd_tol = gettol(fn_item, dtype, "fwd_tol")
                bwd_tol = gettol(fn_item, dtype, "bwd_tol")
                runtestapi(func=func, fwd_input_list=input_list, type_dict={torch.float:{torch.float16}}, 
                            enable_backward=False, fwd_tol=fwd_tol, bwd_tol=bwd_tol)


class TestPow(unittest.TestCase):
    '''
    test pow, float_power
    '''
    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_math_dtypes("cuda")))
    def test_pow(self, device, dtype):
        shape_list = [(10,)]

        fn_list = [{"fn": torch.pow, "tol": {"fwd_tol": {torch.float16: 1e-2, torch.bfloat16: 1e-2},
                                            "bwd_tol": {torch.float16: 1e-2, torch.bfloat16: 1e-2}}},
                    {"fn": torch.float_power, "tol": {"fwd_tol": {torch.float16: 1e-2, torch.bfloat16: 1e-2},
                                            "bwd_tol": {torch.float16: 1e-2, torch.bfloat16: 1e-2}}}]

        forward_inputs = [make_tensor(shape=[10], dtype=dtype, device="cpu", low=0.0, high=2.0)]
        forward_exponent = [-1.0, -0.5, 0.5, torch.arange(1., 11.)]

        for input in forward_inputs:
            for exponet in forward_exponent:
                input_list = [input, exponet]
                for fn_item in fn_list:
                    func = fn_item["fn"]
                    fwd_tol = gettol(fn_item, dtype, "fwd_tol")
                    bwd_tol = gettol(fn_item, dtype, "bwd_tol")
                    runtestapi(func=func, fwd_input_list=input_list, enable_backward=True, input_grad_skip_idx=[1], fwd_tol=fwd_tol, 
                                bwd_tol=bwd_tol)


class TestFloor(unittest.TestCase):
    '''
    test fix, floor, floor_divide, 
    '''
    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_fp_dtypes()))
    @parametrize("shape", [(4,), (2, 2)], name_fn=NORM_NAME)
    @parametrize("func", ["fix", "floor"])
    def test_floor(self, device, dtype, shape, func):
        func = getattr(torch, func)
        forward_input = [make_tensor(shape=shape, dtype=dtype, device="cpu")]
        runtestapi(func=func, fwd_input_list=forward_input, enable_backward=False, type_dict={torch.float:{torch.float16}})

    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_fp_dtypes()))
    @parametrize("shape", [(4,), (2, 2)], name_fn=NORM_NAME)
    @parametrize("func", ["floor_divide"])
    def test_floor_divide(self, device, dtype, shape, func):
        func = getattr(torch, func)
        forward_inputs = [[make_tensor(shape=shape, dtype=dtype, device="cpu"),
                            make_tensor(shape=shape, dtype=dtype, device="cpu")],
                            [make_tensor(shape=shape, dtype=dtype, device="cpu"),
                            2.0]]
        for forward_input in forward_inputs:
            runtestapi(func=func, fwd_input_list=forward_input, enable_backward=False)


class TestFmod(unittest.TestCase):
    '''
    test fmod
    '''
    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_fp_dtypes(include_bfloat16=False)))
    @parametrize("shape", [(4,), (2, 2)], name_fn=NORM_NAME)
    @parametrize("func", ["fmod"])
    def test_fmod(self, device, dtype, shape, func):
        func = getattr(torch, func)
        forward_inputs = [[make_tensor(shape=shape, dtype=dtype, device="cpu"),
                            make_tensor(shape=shape, dtype=dtype, device="cpu")],
                            [make_tensor(shape=shape, dtype=dtype, device="cpu"),
                            2.0]]
        for forward_input in forward_inputs:
            runtestapi(func=func, fwd_input_list=forward_input, enable_backward=False)


class TestFrac(unittest.TestCase):
    '''
    test frac
    '''
    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_fp_dtypes()))
    @parametrize("shape", [(4,), (2, 2)], name_fn=NORM_NAME)
    @parametrize("func", ["frac"])
    def test_fmod(self, device, dtype, shape, func):
        func = getattr(torch, func)
        forward_input = [make_tensor(shape=shape, dtype=dtype, device="cpu")]
        runtestapi(func=func, fwd_input_list=forward_input, enable_backward=False)


class TestFrexp(unittest.TestCase):
    '''
    test fmod
    '''
    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_fp_dtypes(include_bfloat16=False)))
    @parametrize("shape", [(4,), (2, 2)], name_fn=NORM_NAME)
    @parametrize("func", ["frexp"])
    def test_frexp(self, device, dtype, shape, func):
        func = getattr(torch, func)
        forward_input = make_tensor(shape=shape, dtype=dtype, device="cpu")
        fwd_golden = func(forward_input)
        fwd_output = func(forward_input.to("cuda"))
        for i in range(len(fwd_golden)):
            checkclose(fwd_output[i], fwd_golden[i].cpu())


class TestImag(unittest.TestCase):
    '''
    test imag
    '''
    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_complex_dtypes()))
    @parametrize("shape", [(4,), (2, 2)], name_fn=NORM_NAME)
    @parametrize("func", ["imag"])
    def test_imag(self, device, dtype, shape, func):
        func = getattr(torch, func)
        forward_input = [make_tensor(shape=shape, dtype=dtype, device="cpu")]
        runtestapi(func=func, fwd_input_list=forward_input, enable_backward=True)


class TestLdexp(unittest.TestCase):
    '''
    test ldexp
    '''
    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_fp_dtypes()))
    @parametrize("shape", [(4,)], name_fn=NORM_NAME)
    @parametrize("func", ["ldexp"])
    def test_ldexp(self, device, dtype, shape, func):
        func = getattr(torch, func)
        forward_inputs = [[make_tensor(shape=shape, dtype=dtype, device="cpu"),
                            make_tensor(shape=[1], dtype=torch.int32, device="cpu", low=1, high=10)],
                            [make_tensor(shape=shape, dtype=dtype, device="cpu"),
                            make_tensor(shape=shape, dtype=torch.int32, device="cpu")]]
        for forward_input in forward_inputs:
            runtestapi(func=func, fwd_input_list=forward_input, enable_backward=True)


class TestLerp(unittest.TestCase):
    '''
    test lerp
    '''
    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_fp_dtypes()))
    @parametrize("shape", [(4,)], name_fn=NORM_NAME)
    def test_lerp(self, device, dtype, shape):
        fn_list = [{"fn": torch.lerp, "tol": {"fwd_tol": {torch.float16: 1e-2, torch.bfloat16: 1e-1},
                                            "bwd_tol": {torch.float16: 1e-2, torch.bfloat16: 1e-2}}}]

        forward_inputs = [[make_tensor(shape=shape, dtype=dtype, device="cpu"),
                            make_tensor(shape=shape, dtype=dtype, device="cpu"),
                            torch.ones(shape).to(dtype)],
                            [make_tensor(shape=shape, dtype=dtype, device="cpu"),
                            make_tensor(shape=shape, dtype=dtype, device="cpu"),
                            0.5]]
        for forward_input in forward_inputs:
            for fn_item in fn_list:
                    func = fn_item["fn"]
                    fwd_tol = gettol(fn_item, dtype, "fwd_tol")
                    bwd_tol = gettol(fn_item, dtype, "bwd_tol")
                    runtestapi(func=func, fwd_input_list=forward_input, enable_backward=True, 
                                type_dict={torch.float:{torch.float16, torch.bfloat16}}, fwd_tol=fwd_tol, bwd_tol=bwd_tol)


class TestLgamma(unittest.TestCase):
    '''
    test lgamma
    '''
    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_fp_dtypes(include_bfloat16=False)))
    @parametrize("shape", [(4,), (2, 2)], name_fn=NORM_NAME)
    def test_lgamma(self, device, dtype, shape):
        fn_list = [{"fn": torch.lgamma, "tol": {"fwd_tol": {torch.float16: 1e-2},"bwd_tol": {torch.float16: 1e-2}}}]
        forward_input = [make_tensor(shape=shape, dtype=dtype, device="cpu")]
        for fn_item in fn_list:
                    func = fn_item["fn"]
                    fwd_tol = gettol(fn_item, dtype, "fwd_tol")
                    bwd_tol = gettol(fn_item, dtype, "bwd_tol")
                    runtestapi(func=func, fwd_input_list=forward_input, enable_backward=True, 
                                type_dict={torch.float:{torch.float16}}, fwd_tol=fwd_tol, bwd_tol=bwd_tol)


class TestIgamma(unittest.TestCase):
    '''
    test igamma, igammac
    '''
    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_fp_dtypes(include_bfloat16=False, include_half=False)))
    @parametrize("shape", [(4,)], name_fn=NORM_NAME)
    @parametrize("func", ["igamma", "igammac"])
    def test_igamma(self, device, dtype, shape, func):
        func = getattr(torch, func)
        forward_inputs = [[make_tensor(shape=shape, dtype=dtype, device="cpu", low=1, high=10),
                            make_tensor(shape=shape, dtype=dtype, device="cpu", low=1, high=10)],]
        for forward_input in forward_inputs:
            runtestapi(func=func, fwd_input_list=forward_input, enable_backward=False)


class TestMul(unittest.TestCase):
    '''
    test mul, multiply
    '''
    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_math_dtypes("cuda")))
    @parametrize("shape1,shape2", [((4,1), (1,4)), ((4,1), (4,4)), ((1,4), (4,1)), ((1,4), (4,4)), ((128,256), (128,256))], name_fn=NORM_NAME)
    @parametrize("func", ["mul", "multiply"])
    def test_mul(self, device, dtype, shape1, shape2, func):
        func = getattr(torch, func)
        forward_inputs = [[make_tensor(shape=shape1, dtype=dtype, device="cpu"),
                            make_tensor(shape=shape2, dtype=dtype, device="cpu")],
                            [make_tensor(shape=shape1, dtype=dtype, device="cpu"),
                            0.5]]
        for forward_input in forward_inputs:
            runtestapi(func=func, fwd_input_list=forward_input, enable_backward=True)


class TestMvlgamma(unittest.TestCase):
    '''
    test Mvlgamma
    '''
    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_fp_dtypes(include_bfloat16=False)))
    @parametrize("shape", [(2, 4)], name_fn=NORM_NAME)
    @parametrize("p", [1, 2])
    def test_mvlgamma(self, device, dtype, shape, p):
        fn_list = [{"fn": torch.mvlgamma, "tol": {"fwd_tol": {torch.float16: 1e-2},"bwd_tol": {torch.float16: 1e-2}}}]
        forward_inputs = [make_tensor(shape=shape, dtype=dtype, device="cpu", low=1, high=2), p]
        for fn_item in fn_list:
            func = fn_item["fn"]
            fwd_tol = gettol(fn_item, dtype, "fwd_tol")
            bwd_tol = gettol(fn_item, dtype, "bwd_tol")
            runtestapi(func=func, fwd_input_list=forward_inputs, enable_backward=True, 
                        type_dict={torch.float:{torch.float16}}, fwd_tol=fwd_tol, bwd_tol=bwd_tol)


class TestNantonum(unittest.TestCase):
    '''
    test nan_to_num
    '''
    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_fp_dtypes()))
    @parametrize("nan", [0.0])
    @parametrize("posinf", [None, 0])
    @parametrize("neginf", [None, 0])
    @parametrize("func", ["nan_to_num"])
    def test_nantonum(self, device, dtype, nan, posinf, neginf, func):
        func = getattr(torch, func)
        forward_inputs = [torch.tensor([nan, inf, -inf, 1], dtype=dtype), nan, posinf, neginf]
        runtestapi(func=func, fwd_input_list=forward_inputs, enable_backward=False)


class TestNeg(unittest.TestCase):
    '''
    test neg, negative, positive
    '''
    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_math_dtypes("cuda")))
    @parametrize("shape", [(4,)], name_fn=NORM_NAME)
    @parametrize("func", ["neg", "negative", "positive"])
    def test_neg(self, device, dtype, shape, func):
        func = getattr(torch, func)
        forward_inputs = [make_tensor(shape=shape, dtype=dtype, device="cpu")]
        runtestapi(func=func, fwd_input_list=forward_inputs, enable_backward=False)


class TestPolygamma(unittest.TestCase):
    '''
    test polygamma
    '''
    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_fp_dtypes(include_bfloat16=False, include_half=False)))
    @parametrize("shape", [4,])
    @parametrize("n", [0, 1, 2])
    @parametrize("func", ["polygamma"])
    def test_polygamma(self, device, dtype, shape, n, func):
        func = getattr(torch, func)
        forward_inputs = [n, make_tensor(shape=shape, dtype=dtype, device="cpu", low=1, high=2)]
        runtestapi(func=func, fwd_input_list=forward_inputs, enable_backward=True)


class TestNextafter(unittest.TestCase):
    '''
    test nextafter
    '''
    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_fp_dtypes(include_bfloat16=True, include_half=False)))
    @parametrize("shape", [(4,5)], name_fn=NORM_NAME)
    @parametrize("func", ["nextafter"])
    def test_nextafter(self, device, dtype, shape, func):
        func = getattr(torch, func)
        forward_inputs = [make_tensor(shape=shape, dtype=dtype, device="cpu", low=1, high=2),
                            make_tensor(shape=shape, dtype=dtype, device="cpu", low=1, high=2)]
        runtestapi(func=func, fwd_input_list=forward_inputs, enable_backward=False)


class TestGradient(unittest.TestCase):
    '''
    test gradient
    '''
    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_fp_dtypes(include_bfloat16=True, include_half=True)))
    @parametrize("shape", [(4,)],name_fn=NORM_NAME)
    @parametrize("spacing", [2, (2,), (torch.tensor([2,1,1,1]),)], name_fn=NORM_NAME)
    @parametrize("dim", [0, (-1,)], name_fn=NORM_NAME)
    @parametrize("edge_order", [1, 2])
    @parametrize("func", ["gradient"])
    def test_gradient(self, device, dtype, shape, spacing, dim, edge_order, func):
        func = getattr(torch, func)
        input_c = make_tensor(shape=shape, dtype=dtype, device="cpu", low=1, high=2)
        input_g = input_c.detach().clone().cuda()
        kwargs_c = {"input":input_c, "spacing":spacing, "dim":dim, "edge_order":edge_order}
        kwargs_g = {"input":input_g, "spacing":spacing, "dim":dim, "edge_order":edge_order}
        if isinstance(kwargs_g["spacing"], tuple) and torch.is_tensor(kwargs_g["spacing"][0]):
            kwargs_g["spacing"] = (kwargs_g["spacing"][0].cuda(),)
        golden = func(**kwargs_c)
        out = func(**kwargs_g)
        checkclose(out[0], golden[0])


instantiate_device_type_tests(TestErf, globals())
instantiate_device_type_tests(TestExp, globals())
instantiate_device_type_tests(TestFakeQuantize, globals())
instantiate_device_type_tests(TestLog, globals())
instantiate_device_type_tests(TestLogadd, globals())
instantiate_device_type_tests(TestLogical, globals())
instantiate_device_type_tests(TestLogit, globals())
instantiate_device_type_tests(TestHypot, globals())
instantiate_device_type_tests(TestPow, globals())
instantiate_device_type_tests(TestFloor, globals())
instantiate_device_type_tests(TestFmod, globals())
instantiate_device_type_tests(TestFrac, globals())
instantiate_device_type_tests(TestFrexp, globals())
instantiate_device_type_tests(TestImag, globals())
instantiate_device_type_tests(TestLdexp, globals())
instantiate_device_type_tests(TestLerp, globals())
instantiate_device_type_tests(TestIgamma, globals())
instantiate_device_type_tests(TestLgamma, globals())
instantiate_device_type_tests(TestMul, globals())
instantiate_device_type_tests(TestMvlgamma, globals())
instantiate_device_type_tests(TestNantonum, globals())
instantiate_device_type_tests(TestNeg, globals())
instantiate_device_type_tests(TestPolygamma, globals())
instantiate_device_type_tests(TestNextafter, globals())
instantiate_device_type_tests(TestGradient, globals())


if __name__ == "__main__":
    unittest.main()
