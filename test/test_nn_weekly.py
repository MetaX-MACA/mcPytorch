import random
import unittest
import copy
from copy import deepcopy
import numpy
from scipy import stats
from functools import reduce
from operator import mul
import itertools
import tempfile
from itertools import product, combinations, combinations_with_replacement
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.nn.utils.prune as prune
import torch.nn.utils.rnn as rnn_utils
from torch import inf, nan
from torch.testing._internal.common_utils import run_tests, TestCase
from common_util import *

class TestNnActivation(unittest.TestCase):

    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_fp_dtypes(include_half=False, include_bfloat16=False)))
    def test_activation_with_only_inplace_parameter_list(self, device, dtype, inplace=True):
        shape_list = [(2, 2, 4, 8)]
        forward_inputs = gendata(shape_list, type_list=[dtype], rand_algo="rand")
        inplace_options = [True, False]
        activation_only_with_inplace_list = [nn.ReLU6, nn.SELU, nn.SiLU, nn.Mish]
        for inplace, input, activation in product(inplace_options, forward_inputs, activation_only_with_inplace_list):
            func = activation(inplace)
            runtestapi(func=func, fwd_input_list=[input], enable_backward=not inplace)

    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_fp_dtypes(include_half=False, include_bfloat16=False)))
    def test_activation_without_parameter_list(self, device, dtype):
        shape_list = [(2, 2, 4, 8)]
        forward_inputs = gendata(shape_list, type_list=[dtype], rand_algo="rand")
        activation_without_parameter_list = [nn.GELU, nn.Sigmoid, nn.Softsign, nn.Tanh, nn.Tanhshrink]
        for input, activation in product(forward_inputs, activation_without_parameter_list):
            func = activation()
            runtestapi(func=func, fwd_input_list=[input], enable_backward=True)

    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_fp_dtypes(include_half=False, include_bfloat16=False)))
    def test_nn_rrelu(self, device, dtype):
        shape_list = [(2, 2, 4, 8)]
        forward_inputs = gendata(shape_list, type_list=[dtype], rand_algo="rand")
        inplace_options = [True, False]
        bounds = [-0.1, 0, 0.1]
        for inplace, bound, input in product(inplace_options, bounds, forward_inputs):
            func = nn.RReLU(lower=bound, upper=bound, inplace=inplace)
            runtestapi(func=func, fwd_input_list=[input], enable_backward=not inplace)


    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_fp_dtypes(include_half=False, include_bfloat16=False)))
    def test_nn_celu(self, device, dtype):
        shape_list = [(2, 2, 4, 8)]
        forward_inputs = gendata(shape_list, type_list=[dtype], rand_algo="rand")
        inplace_options = [True, False]
        alpha_options = [-1.0, 1.0]
        for inplace, alpha, input in product(inplace_options, alpha_options, forward_inputs):
            func = nn.CELU(alpha=alpha, inplace=inplace)
            runtestapi(func=func, fwd_input_list=[input], enable_backward=not inplace)

    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_fp_dtypes(include_half=False, include_bfloat16=False)))
    def test_nn_softplus(self, device, dtype):
        shape_list = [(2, 2, 4, 8)]
        forward_inputs = gendata(shape_list, type_list=[dtype], rand_algo="rand")
        beta_options = [-1, 1]
        threshold_options = [-10, 0, 10]
        for beta, threshold, input in product(beta_options, threshold_options, forward_inputs):
            func = nn.Softplus(beta, threshold)
            runtestapi(func=func, fwd_input_list=[input], enable_backward=True)

    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_fp_dtypes(include_half=False, include_bfloat16=False)))
    def test_nn_softshrink(self, device, dtype):
        shape_list = [(2, 2, 4, 8)]
        forward_inputs = gendata(shape_list, type_list=[dtype], rand_algo="rand")
        lambd_options = [0, 1]
        for lambd, input in product(lambd_options, forward_inputs):
            func = nn.Softshrink(lambd)
            runtestapi(func=func, fwd_input_list=[input], enable_backward=True)

    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_fp_dtypes(include_half=False, include_bfloat16=False)))
    def test_nn_threshold(self, device, dtype):
        shape_list = [(2, 2, 4, 8)]
        forward_inputs = gendata(shape_list, type_list=[dtype], rand_algo="rand")
        thresholds = [-10, 0, 10]
        values = [-5, 0, 5]
        inplace_options = [True, False]
        for inplace, threshold, value, input in product(inplace_options, thresholds, values, forward_inputs):
            func = nn.Threshold(threshold, value, inplace)
            runtestapi(func=func, fwd_input_list=[input], enable_backward=not inplace)

    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_fp_dtypes(include_half=False, include_bfloat16=False)))
    def test_nn_glu(self, device, dtype):
        shape_list = [(2, 2, 4, 8)]
        forward_inputs = gendata(shape_list, type_list=[dtype], rand_algo="rand")
        dim_options = [0, 1, 2, 3, -1]
        for dim, input in product(dim_options, forward_inputs):
            func = nn.GLU(dim)
            runtestapi(func=func, fwd_input_list=[input], enable_backward=True)

class TestNnUpsample(unittest.TestCase):

    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_fp_dtypes(include_half=False, include_bfloat16=False)))
    def test_nn_upsample(self, device, dtype):
        '''
        Cover all upsample related kernel functions' tests.
        '''
        shape_lists_3d = [(2, 2, 4), (2, 20, 4)]
        shape_lists_4d = [(2, 2, 4, 8), (2, 20, 4, 8)]
        shape_lists_5d = [(2, 2, 2, 4, 8), (2, 2, 20, 4, 8)]
        large_shape = (16, 20, 20, 18)
        small_shape = (2, 2, 2, 2)
        mode_to_shape_list = {
            "nearest":shape_lists_3d+shape_lists_4d+shape_lists_5d,
            "bicubic":shape_lists_4d,
            "linear":shape_lists_3d,
            "bilinear":shape_lists_4d,
            "trilinear":shape_lists_5d
        }
        mode_with_align_corners_list = ["linear", "bilinear", "trilinear"]
        all_modes = ["linear", "bilinear", "trilinear", "nearest", "bicubic"]
        scale_factors = [0.5, 1, 2]
        align_corners = [True, False]
        for align_corner, mode, scale_factor in product(align_corners, all_modes, scale_factors):
            if mode in mode_with_align_corners_list:
                func = nn.Upsample(mode=mode, scale_factor=scale_factor, align_corners=align_corner)
            else:
                func = nn.Upsample(mode=mode, scale_factor=scale_factor)
            shape_list = mode_to_shape_list[mode]
            forward_inputs = gendata(shape_list, type_list=[dtype], rand_algo="rand")
            for index, input in enumerate(forward_inputs):
                if (mode == "bilinear" and index % 2 == 0) or \
                    (mode == "nearest" and len(input[1].shape) == 4 and index % 2 == 0):
                    input = (torch.rand(*large_shape, dtype=dtype).to(memory_format=torch.channels_last))
                if (mode == "bilinear" and index % 2 == 1) or \
                    (mode == "nearest" and len(input[1].shape) == 4 and index % 2 == 1):
                    input = (torch.rand(*small_shape, dtype=dtype).to(memory_format=torch.channels_last))
                fwd_golden, bwd_input, bwd_golden, _ = runtest(
                        func=func, fwd_input_list=[input], device="cpu", enable_backward=True)
                if mode == "bilinear" or (mode == "nearest" and len(input[1].shape) == 4):
                    # bwd_input = [(item[0], item[1].to(memory_format=torch.channels_last)) for item in bwd_input]
                    bwd_input = [item.to(memory_format=torch.channels_last) for item in bwd_input]
                runtest(func=func, fwd_input_list=[
                        input], fwd_golden=fwd_golden, bwd_input_list=bwd_input, bwd_golden_list=bwd_golden, device=device, enable_backward=True)

class TestNnParametrize(unittest.TestCase):
    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_fp_dtypes(include_half=False, include_bfloat16=False)))
    def test_nn_parametrize(self, device, dtype):
        r"""Test that it is possible to add a parametrization
        on a parameter and that removing it restores the initial state
        """
        class Resize(nn.Module):
            def forward(self, X):
                return X[[0]]

        model = nn.Linear(8, 8, device=device, dtype=dtype)
        initial_weight_id = id(model.weight)
        initial_model = deepcopy(model)

        # Test unsafe flag
        with self.assertRaisesRegex(ValueError, "Registering a parametrization may not change the shape of the tensor"):
            torch.nn.utils.parametrize.register_parametrization(model, "weight", Resize())  # default unsafe = False
            model(torch.ones(8, 8))
        torch.nn.utils.parametrize.register_parametrization(model, "weight", Resize(), unsafe=True)
        param_mod = model.parametrizations.weight
        self.assertEqual(param_mod.__class__, torch.nn.utils.parametrize.ParametrizationList)
        self.assertTrue(hasattr(model, "parametrizations"))
        self.assertTrue(torch.nn.utils.parametrize.is_parametrized(model))
        self.assertTrue(torch.nn.utils.parametrize.is_parametrized(model, "weight"))
        self.assertFalse(torch.nn.utils.parametrize.is_parametrized(model, "bias"))
        self.assertNotIn("weight", model._parameters)
        A = model.weight
        self.assertTrue(A.shape[0] == 1)

        # Test that the caching system works
        with torch.nn.utils.parametrize.cached():
            X = model.weight
            Y = model.weight
            self.assertEqual(id(X), id(Y))

        torch.nn.utils.parametrize.remove_parametrizations(model, "weight", leave_parametrized=False)
        self.assertFalse(hasattr(model, "parametrizations"))
        assert torch.equal(model.weight, initial_model.weight)
        self.assertEqual(id(model.weight), initial_weight_id)
        self.assertEqual(model.__class__, nn.Linear)

class TestNnElementwise(unittest.TestCase):
    @onlyCUDA
    def test_nn_elementwise_dynamic_casting_false_with(self, device):
        shape_list = [[(2, 2, 4, 8), (2, 1, 1, 8)], [(2, 2, 4, 8), (2, 2, 4, 8)]]
        type_list = [(torch.float32, torch.half), (torch.float32, torch.float32)]
        for shapes, types in product(shape_list, type_list):
            forward_inputs1 = gendata([shapes[0]], type_list=[types[0]], rand_algo="rand")
            forward_inputs2 = gendata([shapes[1]], type_list=[types[1]], rand_algo="rand")
            for input1, input2 in zip(forward_inputs1, forward_inputs2):
                func = torch.add
                runtestapi(func=func, fwd_input_list=[input1, input2], enable_backward=True)

class TestSoftxx(unittest.TestCase):
    '''
    test Softmax, Softmin, LogSoftmax, Softmax2d, AdaptiveLogSoftmaxWithLoss
    '''
    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_fp_dtypes()))
    @parametrize("shape", [(4,), (4,5), (4,5,6)], name_fn=NORM_NAME)
    def test_softmax(self, device, dtype, shape):
        fn_list = [{"fn": nn.Softmax(dim=0), "tol": {"fwd_tol": {torch.float16: 1e-2, torch.bfloat16: 1e-2},
                        "bwd_tol": {torch.float16: 1e-2, torch.bfloat16: 1e-1}}},
                    {"fn": nn.Softmax(dim=len(shape)-1), "tol": {"fwd_tol": {torch.float16: 1e-2, torch.bfloat16: 1e-2},
                        "bwd_tol": {torch.float16: 1e-2, torch.bfloat16: 1e-1}}},
                    {"fn": nn.Softmin(dim=0), "tol": {"fwd_tol": {torch.float16: 1e-2, torch.bfloat16: 1e-2},
                        "bwd_tol": {torch.float16: 1e-2, torch.bfloat16: 1e-1}}},
                    {"fn": nn.Softmin(dim=len(shape)-1), "tol": {"fwd_tol": {torch.float16: 1e-2, torch.bfloat16: 1e-2},
                        "bwd_tol": {torch.float16: 1e-2, torch.bfloat16: 1e-1}}},
                    {"fn": nn.LogSoftmax(dim=0), "tol": {"fwd_tol": {torch.float16: 1e-2, torch.bfloat16: 1e-2},
                        "bwd_tol": {torch.float16: 1e-2, torch.bfloat16: 1e-1}}},
                    {"fn": nn.LogSoftmax(dim=len(shape)-1), "tol": {"fwd_tol": {torch.float16: 1e-2, torch.bfloat16: 1e-2},
                        "bwd_tol": {torch.float16: 1e-2, torch.bfloat16: 1e-1}}}]
        forward_inputs = [make_tensor(shape=shape, dtype=dtype, device="cpu")]
        for fn_item in fn_list:
            func = fn_item["fn"]
            fwd_tol = gettol(fn_item, dtype, "fwd_tol")
            bwd_tol = gettol(fn_item, dtype, "bwd_tol")
            runtestapi(func=func, fwd_input_list=forward_inputs, enable_backward=True, 
                        type_dict={torch.float:{torch.float16}}, fwd_tol=fwd_tol, bwd_tol=bwd_tol)

    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_fp_dtypes()))
    @parametrize("shape", [(4,5,6), (4,5,6,7)], name_fn=NORM_NAME)
    def test_softmax2d(self, device, dtype, shape):
        fn_list = [{"fn": nn.Softmax2d(), "tol": {"fwd_tol": {torch.float16: 1e-2, torch.bfloat16: 1e-2},
                        "bwd_tol": {torch.float16: 1e-2, torch.bfloat16: 1e-1}}}]
        forward_inputs = [make_tensor(shape=shape, dtype=dtype, device="cpu")]
        for fn_item in fn_list:
            func = fn_item["fn"]
            fwd_tol = gettol(fn_item, dtype, "fwd_tol")
            bwd_tol = gettol(fn_item, dtype, "bwd_tol")
            runtestapi(func=func, fwd_input_list=forward_inputs, enable_backward=True, 
                        type_dict={torch.float:{torch.float16}}, fwd_tol=fwd_tol, bwd_tol=bwd_tol)

    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_fp_dtypes()))
    @parametrize("in_features", [16])
    @parametrize("n_classes", [20])
    @parametrize("cutoffs", [[5, 10, 15], [5, 15]], name_fn=NORM_NAME)
    @parametrize("div_value", [4.0, 2.0])
    @parametrize("head_bias", [False, True])
    @parametrize("N", [1, 2])
    def test_adaptivelogsoftmaxwithloss(self, dtype, in_features, n_classes, cutoffs, div_value, head_bias, N):
        torch.manual_seed(0)
        fn_list = [{"fn": nn.AdaptiveLogSoftmaxWithLoss(in_features=in_features,n_classes=n_classes, cutoffs=cutoffs,
                    div_value=div_value, head_bias=head_bias), "tol": {"fwd_tol": {torch.float16: 1e-2, torch.bfloat16: 1e-2},
                    "bwd_tol": {torch.float16: 1e-2, torch.bfloat16: 1e-1}}}]
        input = make_tensor(shape=(N, in_features), dtype=dtype, device="cpu", low=0.0, high=1.0)

        for fn_item in fn_list:
            func = fn_item["fn"]
            fwd_tol = gettol(fn_item, dtype, "fwd_tol")
            bwd_tol = gettol(fn_item, dtype, "bwd_tol")
            c_type = dtype 
            if dtype == torch.float16:
                c_type = torch.float
            func_c = copy.deepcopy(func).to(c_type)
            input_c = input.clone().to(c_type)
            func_g = copy.deepcopy(func).to(dtype).cuda()
            input_g = input.clone().cuda()
            pred_golden = func_c.predict(input_c)
            pred_output = func_g.predict(input_g)
            checkclose(pred_output.cpu().to(pred_golden.dtype), pred_golden)
            prob_golden = func_c.log_prob(input_c)
            prob_output = func_g.log_prob(input_g)
            if dtype == torch.bfloat16:
                eps = 1e-1
            else:
                eps = 1e-2
            checkclose(prob_output.cpu().to(prob_golden.dtype), prob_golden, eps)
            
class TestLinear(unittest.TestCase):
    '''
    test Linear, bilinear
    '''
    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_fp_dtypes(include_half=False)))  # close half as cpu impl not support
    @parametrize("in_features", [5, 32])   # k
    @parametrize("out_features", [1, 32, 33])  # n
    @parametrize("batch", [1, 32, 33])  # n
    @parametrize("bias", [True, False])
    def test_linear(self, device, dtype, in_features, out_features, batch, bias):
        torch.manual_seed(0)
        fn_list = [{"fn": nn.Linear(in_features=in_features, out_features=out_features, bias=bias),
                    "tol": {"fwd_tol": {torch.float16: 1e-2, torch.bfloat16: 2e-1},
                    "bwd_tol": {torch.float16: 1e-1, torch.bfloat16: 1e-1}}}]
        forward_inputs = [make_tensor(shape=(batch, in_features), dtype=dtype, device="cpu", low=0.0, high=1.0)]
        fwd_tol= 1e-2
        bwd_tol = 1e-2
        for fn_item in fn_list:
            func = fn_item["fn"]
            fwd_tol = max(gettol(fn_item, dtype, "fwd_tol"), fwd_tol)
            bwd_tol = max(gettol(fn_item, dtype, "bwd_tol"), bwd_tol)
            runtestapi(func=func, fwd_input_list=forward_inputs, enable_backward=True, 
                        fwd_tol=fwd_tol, bwd_tol=bwd_tol)
    
    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_complex_dtypes()))
    @parametrize("in_features", [5, 32])   # k
    @parametrize("out_features", [1, 32, 33])  # n
    @parametrize("batch", [1, 32, 33])  # n
    @parametrize("bias", [True, False])
    def test_linear_complex(self, device, dtype, in_features, out_features, batch, bias):
        fn_list = [{"fn": nn.Linear(in_features=in_features, out_features=out_features, bias=bias)}]
        forward_inputs = gendata(shape_list=[(batch, in_features)], type_list=[dtype])
        for fn_item in fn_list:
            func = fn_item["fn"]
            runtestapi(func=func, fwd_input_list=forward_inputs, enable_backward=True, 
                        fwd_tol=1e-2, bwd_tol=1e-2)

    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_fp_dtypes(include_half=False)))  # close half as cpu impl not support
    @parametrize("in1_features", [5, 32])   # k
    @parametrize("in2_features", [5, 32])   # k
    @parametrize("out_features", [1, 32, 33])  # n
    @parametrize("batch", [1, 32, 33])  # n
    @parametrize("bias", [True, False])
    def test_bilinear(self, device, dtype, in1_features, in2_features, out_features, batch, bias):
        torch.manual_seed(0)
        fn_list = [{"fn": nn.Bilinear(in1_features=in1_features, in2_features=in2_features, out_features=out_features, bias=bias),
                    "tol": {"fwd_tol": {torch.float16: 1e-1, torch.bfloat16: 2e-1},
                            "bwd_tol": {torch.float16: 1e-1, torch.bfloat16: 1}}}]
        forward_inputs1 = [make_tensor(shape=(batch, in1_features), dtype=dtype, device="cpu", low=0.0, high=1.0)]
        forward_inputs2 = [make_tensor(shape=(batch, in2_features), dtype=dtype, device="cpu", low=0.0, high=1.0)]
        forward_inputs = forward_inputs1 + forward_inputs2
        fwd_tol= 1e-2
        bwd_tol = 1e-2
        for fn_item in fn_list:
            func = fn_item["fn"]
            fwd_tol = max(gettol(fn_item, dtype, "fwd_tol"), fwd_tol)
            bwd_tol = max(gettol(fn_item, dtype, "bwd_tol"), bwd_tol)
            runtestapi(func=func, fwd_input_list=forward_inputs, enable_backward=True, 
                        fwd_tol=fwd_tol, bwd_tol=bwd_tol)

class TestDropout(unittest.TestCase):
    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_fp_dtypes(include_half=False)))
    def test_dropout(self, device, dtype):
        def helper(cls, device, input, memory_format=torch.contiguous_format):
            p = 0.2
            input = input.to(device).fill_(1 - p)

            module = cls(p)
            input_var = input.clone(memory_format=memory_format).requires_grad_()
            output = module(input_var)
            assert output.is_contiguous(memory_format=memory_format)
            assert abs(output.data.mean() - (1 - p)) < 0.05
            output.backward(input)
            assert input_var.grad.is_contiguous(memory_format=memory_format)
            assert abs(input_var.grad.data.mean() - (1 - p)) < 0.05

            module = cls(p, True)
            input_var = input.clone(memory_format=memory_format).requires_grad_()
            output = module(input_var + 0)
            assert output.is_contiguous(memory_format=memory_format)
            assert abs(output.data.mean() - (1 - p)) < 0.05
            output.backward(input)
            assert input_var.grad.is_contiguous(memory_format=memory_format)
            assert abs(input_var.grad.data.mean() - (1 - p)) < 0.05

            # check eval mode doesn't change anything
            for inplace in [True, False]:
                module = cls(p, inplace).eval()
                torch.equal(input, module(input))

            # Check that these don't raise errors
            module.__repr__()
            str(module)

        def discontiguous_helper(cls, device, memory_format=torch.contiguous_format):
            # In this test, we verify that dropout preserves the layout and data for different memory formats.
            # We check whether, we get same values for the output of dropout, when the probability
            # of dropout is 0 or very close to 0.
            # Reference: https://github.com/pytorch/pytorch/issues/47176
            close_to_zero_p = 1e-10  # Should be almost zero but not zero, as for p=0 different path is taken
            for p in [0, close_to_zero_p]:
                inp = torch.ones(2, 3, 3, 3, device=device)
                inp_discontiguous = torch.empty(2, 3, 3, 6, device=device, memory_format=memory_format)[..., ::2]
                inp_discontiguous.copy_(inp)
                mod = cls(p=p)
                out = mod(inp_discontiguous)
                if p != 0:  # Zero will keep strides as is based on input.
                    # When prob == 0, input stride (54, 18, 6, 2) -> output stride (54, 18, 6, 2)
                    # When prob != 0, input stride (54, 18, 6, 2) -> output stride (27, 9, 3, 1)
                    assert out.is_contiguous(memory_format=memory_format)
                torch.equal(inp_discontiguous, out)
        
        # test dropout
        input = torch.empty(1000).to(dtype)
        helper(nn.Dropout, device, input)
        discontiguous_helper(nn.Dropout, device)
        discontiguous_helper(nn.Dropout, device, memory_format=torch.channels_last)

        # test dropout2d
        b = random.randint(1, 5)
        w = random.randint(1, 5)
        h = random.randint(1, 5)
        num_features = 1000
        input = torch.empty(num_features, b, w, h).to(dtype)
        helper(nn.Dropout2d, device, input)
        helper(nn.Dropout2d, device, input, memory_format=torch.channels_last)
        discontiguous_helper(nn.Dropout2d, device)
        discontiguous_helper(nn.Dropout2d, device, memory_format=torch.channels_last)
        # no batch dims
        input = torch.empty(20, 64, 64).to(dtype)
        helper(nn.Dropout2d, device, input)

        # test dropout3d
        b = random.randint(1, 5)
        w = random.randint(1, 5)
        h = random.randint(1, 5)
        d = random.randint(1, 2)
        num_features = 1000
        input = torch.empty(num_features, b, d, w, h).to(dtype)
        helper(nn.Dropout3d, device, input)
        discontiguous_helper(nn.Dropout3d, device)
        discontiguous_helper(nn.Dropout3d, device, memory_format=torch.channels_last)
        # no batch dims
        input = torch.empty(50, 20, 64, 64).to(dtype)
        helper(nn.Dropout3d, device, input)
    
    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_fp_dtypes(include_half=False)))
    def test_alpha_dropout(self, device, dtype):
        def helper(cls, input):
            mean = input.mean()
            std = input.std()

            for p in [0.2, 0.5, 0.8]:
                module = cls(p)
                input_var = input.detach().clone().requires_grad_()
                output = module(input_var)
                # output mean should be close to input mean
                self.assertLess(abs(output.data.mean() - mean), 0.1)
                # output std should be close to input std
                self.assertLess(abs(output.data.std() - std), 0.1)
                output.backward(input)

        # test AlphaDropout
        input = torch.randn(5000).to(dtype)
        helper(nn.AlphaDropout, input)

        # test FeatureAlphaDropout
        b = random.randint(1, 5)
        w = random.randint(1, 5)
        h = random.randint(1, 5)
        d = random.randint(1, 2)
        num_features = 1000
        input = torch.randn(num_features, b, d, w, h).to(dtype)
        helper(nn.FeatureAlphaDropout, input)
        # no batch dims
        input = torch.randn(50, 20, 64, 64).to(dtype)
        helper(nn.FeatureAlphaDropout, input)

class TestInit(unittest.TestCase):
    def setUp(self):
        random.seed(123)

    def _is_normal(self, tensor, mean, std):
        tensor = tensor.cpu()
        samples = tensor.view(-1).tolist()
        p_value = stats.kstest(samples, 'norm', args=(mean, std))[1]
        return p_value > 0.0001

    def _is_trunc_normal(self, tensor, mean, std, a, b):
        # scipy's trunc norm is suited for data drawn from N(0, 1),
        # so we need to transform our data to test it using scipy.
        z_samples = (tensor.view(-1) - mean) / std
        z_samples = z_samples.tolist()
        a0 = (a - mean) / std
        b0 = (b - mean) / std
        p_value = stats.kstest(z_samples, 'truncnorm', args=(a0, b0))[1]
        return p_value > 0.0001

    def _is_uniform(self, tensor, a, b):
        samples = tensor.view(-1).tolist()
        p_value = stats.kstest(samples, 'uniform', args=(a, (b - a)))[1]
        return p_value > 0.0001
    
    def _create_random_nd_tensor(self, dims, size_min, size_max):
        size = [random.randint(size_min, size_max) for _ in range(dims)]
        tensor = torch.zeros(size)
        return tensor

    def _random_float(self, a, b):
        return (b - a) * random.random() + a


    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_fp_dtypes(include_bfloat16=False)))
    def test_uniform(self, device, dtype):
        for dims in [1, 2, 4]:
            input_tensor = self._create_random_nd_tensor(dims, size_min=30, size_max=50).to(dtype).to(device)
            a = self._random_float(-3, 3)
            b = a + self._random_float(1, 5)
            init.uniform_(input_tensor, a=a, b=b)
            assert self._is_uniform(input_tensor, a, b)

    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_fp_dtypes(include_bfloat16=False)))
    def test_normal(self, device, dtype):
        for dims in [1, 2, 4]:
            input_tensor = self._create_random_nd_tensor(dims, size_min=30, size_max=50).to(dtype).to(device)
            mean = self._random_float(-3, 3)
            std = self._random_float(1, 5)
            init.normal_(input_tensor, mean=mean, std=std)
            assert self._is_normal(input_tensor, mean, std)

    @onlyCUDA
    # half type test also failed under nvidia
    @dtypesIfCUDA(*set(get_all_fp_dtypes(include_bfloat16=False, include_half=False)))
    def test_trunc_normal(self, device, dtype):
        for dims in [1, 2, 4]:
            input_tensor = self._create_random_nd_tensor(dims, size_min=30, size_max=50).to(dtype).to(device)
            mean = self._random_float(-3, 3)
            std = self._random_float(.01, 1)
            a = self._random_float(mean - 2 * std, mean)
            b = self._random_float(mean, mean + 2 * std)
            init.trunc_normal_(input_tensor, mean=mean, std=std, a=a, b=b)
            assert self._is_trunc_normal(input_tensor, mean, std, a, b)

    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_fp_dtypes(include_bfloat16=False)))
    def test_constant(self, device, dtype):
        for dims in [1, 2, 4]:
            input_tensor = self._create_random_nd_tensor(dims, size_min=1, size_max=5).to(dtype).to(device)
            val = self._random_float(1, 10)
            init.constant_(input_tensor, val)
            torch.equal(input_tensor, input_tensor.clone().fill_(val))

    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_fp_dtypes(include_bfloat16=False)))
    def test_ones_and_zeros(self, device, dtype):
        for init_fn_, val in zip([init.ones_, init.zeros_], [1, 0]):
            for dims in [1, 2, 4]:
                input_tensor = self._create_random_nd_tensor(dims, size_min=1, size_max=5).to(dtype).to(device)
                init_fn_(input_tensor) 
                torch.equal(input_tensor, input_tensor.clone().fill_(val))

    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_fp_dtypes(include_bfloat16=False)))
    def test_eye(self, device, dtype):
        input_tensor = self._create_random_nd_tensor(2, size_min=1, size_max=5).to(dtype).to(device)
        init.eye_(input_tensor)

        # Check every single element
        for i in range(input_tensor.size(0)):
            for j in range(input_tensor.size(1)):
                if i == j:
                    assert input_tensor[i][j] == 1
                else:
                    assert input_tensor[i][j] == 0

    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_fp_dtypes(include_bfloat16=False)))
    def test_eye_only_works_on_2d_inputs(self, device, dtype):
        for dims in [1, 3]:
            with self.assertRaises(ValueError):
                tensor = self._create_random_nd_tensor(dims, size_min=1, size_max=3).to(dtype).to(device)
                init.eye_(tensor)

    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_fp_dtypes(include_bfloat16=False)))
    def test_dirac_properties(self, device, dtype):
        for dims in [3, 4, 5]:
            for groups in [1, 2, 3]:
                # prepare random tensor with random sizes, but fits groups
                a, c, d, e = (random.randint(1, 5) for _ in range(4))
                b = random.randint(1, 5 * groups)  # same range as a*groups but all range allowed
                # make sure first dim divides by groups
                input_tensor = torch.randn((a * groups, b, c, d, e)[:dims]).to(dtype).to(device)

                init.dirac_(input_tensor, groups)

                c_out, c_in = input_tensor.size(0) // groups, input_tensor.size(1)
                min_d = min(c_out, c_in)
                # Check number of nonzeros is equivalent to smallest dim (for each group)
                assert torch.nonzero(input_tensor).size(0) == min_d * groups
                # Check sum of values (can have precision issues, hence assertEqual) is also equivalent
                assert input_tensor.sum() == min_d * groups

    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_fp_dtypes(include_bfloat16=False))) 
    def test_dirac_identity(self, device, dtype):
        for groups in [1, 3]:
            batch, in_c, out_c, size, kernel_size = 8, 3, 9, 5, 3  # in_c, out_c must divide by groups
            eff_out_c = out_c // groups

            # Test 1D
            input_var = torch.randn(batch, in_c, size).to(dtype).to(device)
            filter_var = torch.zeros(eff_out_c, in_c, kernel_size).to(dtype).to(device)
            filter_var = torch.cat([filter_var] * groups)
            init.dirac_(filter_var, groups)
            # m1 = F.conv1d.to(dtype).to(device)
            output_var = F.conv1d(input_var, filter_var)
            input_tensor, output_tensor = input_var.data, output_var.data  # Variables do not support nonzero
            for g in range(groups):
                # Assert in_c outputs are preserved (per each group)
                torch.equal(input_tensor[:, :, 1:-1],
                                 output_tensor[:, eff_out_c * g:eff_out_c * g + in_c, :])
                # Assert extra outputs are 0
                assert torch.nonzero(output_tensor[:, eff_out_c * g + in_c:eff_out_c * (g + 1), :]).numel() == 0

            # Test 2D
            input_var = torch.randn(batch, in_c, size, size).to(dtype).to(device)
            filter_var = torch.zeros(eff_out_c, in_c, kernel_size, kernel_size).to(dtype).to(device)
            filter_var = torch.cat([filter_var] * groups)
            init.dirac_(filter_var, groups)
            # m2 = F.conv2d.to(dtype).to(device)
            output_var = F.conv2d(input_var, filter_var)
            input_tensor, output_tensor = input_var.data, output_var.data  # Variables do not support nonzero
            for g in range(groups):
                # Assert in_c outputs are preserved (per each group)
                torch.equal(input_tensor[:, :, 1:-1, 1:-1],
                                 output_tensor[:, eff_out_c * g:eff_out_c * g + in_c, :, :])
                # Assert extra outputs are 0
                assert torch.nonzero(output_tensor[:, eff_out_c * g + in_c:eff_out_c * (g + 1), :, :]).numel() == 0

            # Test 3D
            input_var = torch.randn(batch, in_c, size, size, size).to(dtype).to(device)
            filter_var = torch.zeros(eff_out_c, in_c, kernel_size, kernel_size, kernel_size).to(dtype).to(device)
            filter_var = torch.cat([filter_var] * groups)
            init.dirac_(filter_var, groups)
            # m3 = F.conv3d.to(dtype).to(device)
            output_var = F.conv3d(input_var, filter_var)
            input_tensor, output_tensor = input_var.data, output_var.data
            for g in range(groups):
                # Assert in_c outputs are preserved (per each group)
                torch.equal(input_tensor[:, :, 1:-1, 1:-1, 1:-1],
                                 output_tensor[:, eff_out_c * g:eff_out_c * g + in_c, :, :, :])
                # Assert extra outputs are 0
                assert torch.nonzero(output_tensor[:, eff_out_c * g + in_c:eff_out_c * (g + 1), :, :, :]).numel() == 0
    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_fp_dtypes(include_bfloat16=False)))
    def test_dirac_only_works_on_3_4_5d_inputs(self, device, dtype):
        for dims in [1, 2, 6]:
            with self.assertRaises(ValueError):
                tensor = self._create_random_nd_tensor(dims, size_min=1, size_max=3).to(dtype).to(device)
                init.dirac_(tensor)
    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_fp_dtypes(include_bfloat16=False)))
    def test_xavier_uniform_errors_on_inputs_smaller_than_2d(self, device, dtype):
        for dims in [0, 1]:
            tensor = self._create_random_nd_tensor(dims, size_min=1, size_max=1).to(dtype).to(device)
            with self.assertRaises(ValueError):
                init.xavier_uniform_(tensor)
    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_fp_dtypes(include_bfloat16=False)))
    def test_xavier_normal_errors_on_inputs_smaller_than_2d(self, device, dtype):
        for dims in [0, 1]:
            tensor = self._create_random_nd_tensor(dims, size_min=1, size_max=1).to(dtype).to(device)
            with self.assertRaises(ValueError):
                init.xavier_normal_(tensor)

    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_fp_dtypes(include_bfloat16=False)))
    def test_xavier_uniform(self, dtype, device):
        for use_gain in [True, False]:
            for dims in [2, 4]:
                input_tensor = self._create_random_nd_tensor(dims, size_min=20, size_max=25).to(dtype).to(device)
                gain = 1

                if use_gain:
                    gain = self._random_float(0.1, 2)
                    init.xavier_uniform_(input_tensor, gain=gain)
                else:
                    init.xavier_uniform_(input_tensor)

                fan_in = input_tensor.size(1)
                fan_out = input_tensor.size(0)
                if input_tensor.dim() > 2:
                    fan_in *= input_tensor[0, 0].numel()
                    fan_out *= input_tensor[0, 0].numel()

                expected_std = gain * math.sqrt(2.0 / (fan_in + fan_out))
                bounds = expected_std * math.sqrt(3)
                assert self._is_uniform(input_tensor, -bounds, bounds)

    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_fp_dtypes(include_bfloat16=False)))
    def test_xavier_normal(self, dtype, device):
        for use_gain in [True, False]:
            for dims in [2, 4]:
                input_tensor = self._create_random_nd_tensor(dims, size_min=20, size_max=25).to(dtype).to(device)
                gain = 1

                if use_gain:
                    gain = self._random_float(0.1, 2)
                    init.xavier_normal_(input_tensor, gain=gain)
                else:
                    init.xavier_normal_(input_tensor)

                fan_in = input_tensor.size(1)
                fan_out = input_tensor.size(0)
                if input_tensor.dim() > 2:
                    fan_in *= input_tensor[0, 0].numel()
                    fan_out *= input_tensor[0, 0].numel()

                expected_std = gain * math.sqrt(2.0 / (fan_in + fan_out))
                assert self._is_normal(input_tensor, 0, expected_std)

    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_fp_dtypes(include_bfloat16=False)))
    def test_kaiming_uniform_errors_on_inputs_smaller_than_2d(self, dtype, device):
        for dims in [0, 1]:
            with self.assertRaises(ValueError):
                tensor = self._create_random_nd_tensor(dims, size_min=1, size_max=1).to(dtype).to(device)
                init.kaiming_uniform_(tensor)

    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_fp_dtypes(include_bfloat16=False)))
    def test_kaiming_normal_errors_on_inputs_smaller_than_2d(self, dtype, device):
        for dims in [0, 1]:
            with self.assertRaises(ValueError):
                tensor = self._create_random_nd_tensor(dims, size_min=1, size_max=1).to(dtype).to(device)
                init.kaiming_normal_(tensor)

    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_fp_dtypes(include_bfloat16=False)))
    def test_kaiming_uniform_warning_on_0element_tensor(self, dtype, device):
        tensor = torch.empty(0, 1).to(dtype).to(device)
        with self.assertWarnsRegex(UserWarning, "Initializing zero-element tensors is a no-op"):
            _ = init.kaiming_uniform_(tensor)

    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_fp_dtypes(include_bfloat16=False)))
    def test_kaiming_normal_warning_on_0element_tensor(self, dtype, device):
        tensor = torch.empty(0, 1).to(dtype).to(device)
        with self.assertWarnsRegex(UserWarning, "Initializing zero-element tensors is a no-op"):
            _ = init.kaiming_normal_(tensor)

    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_fp_dtypes(include_bfloat16=False)))
    def test_kaiming_uniform(self, dtype, device):
        for use_a in [True, False]:
            for dims in [2, 4]:
                for mode in ['fan_in', 'fan_out']:
                    input_tensor = self._create_random_nd_tensor(dims, size_min=20, size_max=25).to(dtype).to(device)
                    if use_a:
                        a = self._random_float(0.1, 2)
                        init.kaiming_uniform_(input_tensor, a=a, mode=mode)
                    else:
                        a = 0
                        init.kaiming_uniform_(input_tensor, mode=mode)

                    fan_in = input_tensor.size(1)
                    fan_out = input_tensor.size(0)
                    if input_tensor.dim() > 2:
                        fan_in *= input_tensor[0, 0].numel()
                        fan_out *= input_tensor[0, 0].numel()

                    if mode == 'fan_in':
                        n = fan_in
                    else:
                        n = fan_out

                    expected_std = math.sqrt(2.0 / ((1 + a**2) * n))
                    bounds = expected_std * math.sqrt(3.0)
                    assert self._is_uniform(input_tensor, -bounds, bounds)

    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_fp_dtypes(include_bfloat16=False)))
    def test_kaiming_normal(self, dtype, device):
        for use_a in [True, False]:
            for dims in [2, 4]:
                for mode in ['fan_in', 'fan_out']:
                    input_tensor = self._create_random_nd_tensor(dims, size_min=20, size_max=25).to(dtype).to(device)
                    if use_a:
                        a = self._random_float(0.1, 2)
                        init.kaiming_normal_(input_tensor, a=a, mode=mode)
                    else:
                        a = 0
                        init.kaiming_normal_(input_tensor, mode=mode)

                    fan_in = input_tensor.size(1)
                    fan_out = input_tensor.size(0)
                    if input_tensor.dim() > 2:
                        fan_in *= input_tensor[0, 0].numel()
                        fan_out *= input_tensor[0, 0].numel()

                    if mode == 'fan_in':
                        n = fan_in
                    else:
                        n = fan_out

                    expected_std = math.sqrt(2.0 / ((1 + a**2) * n))
                    assert self._is_normal(input_tensor, 0, expected_std)

    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_fp_dtypes(include_bfloat16=False)))
    def test_sparse_only_works_on_2d_inputs(self, dtype, device):
        for dims in [1, 3]:
            with self.assertRaises(ValueError):
                sparsity = self._random_float(0.1, 0.9)
                tensor = self._create_random_nd_tensor(dims, size_min=1, size_max=3).to(dtype).to(device)
                init.sparse_(tensor, sparsity)

    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_fp_dtypes(include_bfloat16=False)))
    def test_sparse_default_std(self, dtype, device):
        for use_random_std in [True, False]:
            input_tensor = self._create_random_nd_tensor(2, size_min=30, size_max=35).to(dtype).to(device)
            rows, cols = input_tensor.size(0), input_tensor.size(1)
            sparsity = self._random_float(0.1, 0.2)

            std = 0.01  # default std
            if use_random_std:
                std = self._random_float(0.01, 0.2)
                init.sparse_(input_tensor, sparsity=sparsity, std=std)
            else:
                init.sparse_(input_tensor, sparsity=sparsity)

            for col_idx in range(input_tensor.size(1)):
                column = input_tensor[:, col_idx]
                assert column[column == 0].nelement() >= math.ceil(sparsity * rows)

            assert self._is_normal(input_tensor[input_tensor != 0], 0, std)

    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_fp_dtypes(include_bfloat16=False, include_half=False)))
    def test_orthogonal(self, dtype, device):
        for use_gain in [True, False]:
            for tensor_size in [[3, 4], [4, 3], [20, 2, 3, 4], [2, 3, 4, 5]]:
                input_tensor = torch.zeros(tensor_size).to(dtype).to(device)
                gain = 1.0

                if use_gain:
                    gain = self._random_float(0.1, 2)
                    init.orthogonal_(input_tensor, gain=gain)
                else:
                    init.orthogonal_(input_tensor)

                rows, cols = tensor_size[0], reduce(mul, tensor_size[1:])
                flattened_tensor = input_tensor.view(rows, cols)
                if rows > cols:
                    torch.allclose(torch.mm(flattened_tensor.t(), flattened_tensor).cpu(),
                                     torch.eye(cols).to(dtype) * gain ** 2, atol=1e-6, rtol=0)
                else:
                    torch.allclose(torch.mm(flattened_tensor, flattened_tensor.t()).cpu(),
                                     torch.eye(rows).to(dtype) * gain ** 2, atol=1e-6, rtol=0)

class TestDistanceOps(unittest.TestCase):
    def __distance_ops_verify_run(self, input1, input2, func):
        runtestapi(func=func, fwd_input_list=[input1, input2], fwd_golden=[],  enable_backward=True, fwd_tol=1e-2, bwd_tol=1e-2, bww_tol=1e-2)
    
    @dtypes(*set(get_all_fp_dtypes()))
    @parametrize("shape", [(2), (2, 2), (4, 2)])
    @parametrize("p", [-1, 1, 2, 3])
    @parametrize("keep_dim", [True, False])
    def test_pairwise_distance(self, dtype, shape, p, keep_dim):
        input1 = gendata([shape], type_list=[dtype], rand_algo="uniform", lower=-1.0, upper=1.0)[0]
        input2 = gendata([shape], type_list=[dtype], rand_algo="uniform", lower=-1.0, upper=1.0)[0]
        func = torch.nn.PairwiseDistance()
        self.__distance_ops_verify_run(input1, input2, func)

    
    @dtypes(*set(get_all_fp_dtypes(include_half=False)))
    @parametrize("shape", [(2, 2), (4, 2)])
    def test_consine_similarity(self, dtype, shape):
        input1 = gendata([shape], type_list=[dtype], rand_algo="uniform", lower=-1.0, upper=1.0)[0]
        input2 = gendata([shape], type_list=[dtype], rand_algo="uniform", lower=-1.0, upper=1.0)[0]
        func = torch.nn.CosineSimilarity()
        self.__distance_ops_verify_run(input1, input2, func)

class TestLossOps(unittest.TestCase):
    @onlyCUDA
    @dtypes(*set(get_all_fp_dtypes(include_half=False, include_bfloat16=False)))
    @parametrize("shape", [(6), (2, 2), (4, 2)])
    @parametrize("reduction", ["mean", "sum", "none"])
    @parametrize("func", [torch.nn.L1Loss, torch.nn.MSELoss])
    def test_mse_loss(self, dtype, shape, reduction, func):
        input1 = gendata([shape], type_list=[dtype], rand_algo="uniform", lower=-1.0, upper=1.0)[0]
        input2 = gendata([shape], type_list=[dtype], rand_algo="uniform", lower=-1.0, upper=1.0)[0]
        run_func = func(reduction=reduction)
        runtestapi(func=run_func, fwd_input_list=[input1, input2], fwd_golden=[],  enable_backward=True)
    
    @onlyCUDA
    @dtypes(*set(get_all_fp_dtypes(include_half=False, include_bfloat16=False)))
    @parametrize("shape", [[6, 10], [4, 8, 100, 100]])
    @parametrize("use_weight", [True, False])
    @parametrize("ignore_index", [-1, 1, 3, 5, 11])
    @parametrize("reduction", ["mean", "sum", "none"])
    @parametrize("label_smoothing", [0.0, 0.2, 0.5, 0.8])
    @parametrize("class_indices", [True, False])
    def test_cross_entropy_loss(self, dtype, device, shape, use_weight, ignore_index, reduction, label_smoothing, class_indices):
        C = shape[0]
        class_indices_target_shape = ()
        if len(shape) > 1:
            C = shape[1]
            class_indices_target_shape = [shape[0]]
            if len(shape) > 2:
                class_indices_target_shape += shape[2:]
        weight = None
        if use_weight:
            weight = make_tensor(shape=[C], device=device, dtype=dtype, low=-1., high=1.)
        input = gendata([shape], type_list=[dtype], rand_algo="uniform", lower=-1.0, upper=1.0)[0]
        target = gendata([shape], type_list=[dtype], rand_algo="uniform", lower=0.0, upper=1.0)[0]
        func = torch.nn.CrossEntropyLoss(weight, reduction=reduction, label_smoothing=label_smoothing)
        if class_indices:
            target = gendata([class_indices_target_shape], type_list=[torch.long], rand_algo="randint", lower=0, upper=C)[0]
            func = torch.nn.CrossEntropyLoss(weight, ignore_index=ignore_index, reduction=reduction, label_smoothing=label_smoothing)
        runtestapi(func=func, fwd_input_list=[input, target], fwd_golden=[],  enable_backward=True)
    
    @onlyCUDA
    @dtypes(*set(get_all_fp_dtypes(include_half=False, include_bfloat16=False)))
    @parametrize("shape", [[6, 10], [4, 8, 100, 100]])
    @parametrize("use_weight", [True, False])
    @parametrize("reduction", ["mean", "sum", "none"])
    def test_bce_loss(self, dtype, device, shape, use_weight, reduction):
        weight = None
        if use_weight:
            weight = make_tensor(shape=shape, device=device, dtype=dtype, low=-1., high=1.)
        input = gendata([shape], type_list=[dtype], rand_algo="uniform", lower=0.0, upper=1.0)[0]
        target = gendata([shape], type_list=[dtype], rand_algo="uniform", lower=0.0, upper=1.0)[0]
        func = torch.nn.BCELoss(weight, reduction=reduction)
        runtestapi(func=func, fwd_input_list=[input, target], fwd_golden=[],  enable_backward=True)

    @onlyCUDA
    @dtypes(*set(get_all_fp_dtypes(include_half=False, include_bfloat16=False)))
    @parametrize("shape", [[6, 10], [4, 8, 100, 100]])
    @parametrize("use_weight", [True, False])
    @parametrize("use_pos_weight", [True, False])
    @parametrize("reduction", ["mean", "sum", "none"])
    def test_bce_with_logits_loss(self, dtype, device, shape, use_weight, reduction, use_pos_weight):
        weight = None
        if use_weight:
            weight = make_tensor(shape=shape, device=device, dtype=dtype, low=-1., high=1.)
        pos_weight = None
        if use_pos_weight:
            pos_weight = make_tensor(shape=shape[1:], device=device, dtype=dtype, low=0., high=2.)
        input = gendata([shape], type_list=[dtype], rand_algo="uniform", lower=0.0, upper=1.0)[0]
        target = gendata([shape], type_list=[dtype], rand_algo="uniform", lower=0.0, upper=1.0)[0]
        func = torch.nn.BCEWithLogitsLoss(weight, reduction=reduction, pos_weight=pos_weight)
        runtestapi(func=func, fwd_input_list=[input, target], fwd_golden=[],  enable_backward=True)
    
    @onlyCUDA
    @dtypes(*set(get_all_fp_dtypes(include_half=False, include_bfloat16=False)))
    @parametrize("blank", [0, 1])
    @parametrize("reduction", ["mean", "sum", "none"])
    @parametrize("zero_infinity", [True, False])
    @parametrize("T", [50])
    @parametrize("C", [20])
    @parametrize("N", [16])
    @parametrize("S", [30])
    @parametrize("S_min", [10])
    def test_ctc_loss(self, dtype, blank, reduction, zero_infinity, T, C, N, S, S_min):
        input = torch.randn(T, N, C).log_softmax(2).detach().requires_grad_()
        target = torch.randint(low=1, high=C, size=(N, S), dtype=torch.long)
        input_lengths = torch.full(size=(N,), fill_value=T, dtype=torch.long)
        target_lengths = torch.randint(low=S_min, high=S, size=(N,), dtype=torch.long)
        func = torch.nn.CTCLoss(blank, reduction, zero_infinity)
        # runtestapi(func=func, fwd_input_list=[input, target, input_lengths, target_lengths], fwd_golden=[],  enable_backward=True)


    @onlyCUDA
    @dtypes(*set(get_all_fp_dtypes(include_half=False, include_bfloat16=False)))
    @parametrize("shape", [[6, 10], [4, 8, 100, 100]], name_fn=NORM_NAME)
    @parametrize("use_weight", [True, False])
    @parametrize("ignore_index", [-1, 1, 3, 5, 11])
    @parametrize("reduction", ["mean", "sum", "none"])
    def test_nll_loss(self, dtype, device, shape, use_weight, ignore_index, reduction):
        C = shape[0]
        target_shape = ()
        if len(shape) > 1:
            C = shape[1]
            target_shape = [shape[0]]
            if len(shape) > 2:
                target_shape += shape[2:]
        weight = None
        if use_weight:
            weight = make_tensor(shape=[C], device=device, dtype=dtype, low=-1., high=1.)
        target = gendata([target_shape], type_list=[torch.long], rand_algo="randint", lower=0, upper=C)[0]
        input = gendata([shape], type_list=[dtype], rand_algo="uniform", lower=-1.0, upper=1.0)[0]
        func = torch.nn.NLLLoss(weight, ignore_index=ignore_index, reduction=reduction)
        runtestapi(func=func, fwd_input_list=[input, target], fwd_golden=[],  enable_backward=True)  

    @onlyCUDA
    @dtypes(*set(get_all_fp_dtypes(include_half=False, include_bfloat16=False)))
    @parametrize("shape", [[6], [6, 10], [4, 8, 100, 100]], name_fn=NORM_NAME)
    @parametrize("log_input", [True, False])
    @parametrize("full", [True, False])
    @parametrize("eps", [1e-5])
    @parametrize("reduction", ["mean", "sum", "none"])
    def test_poisson_nll_loss(self, dtype, device, shape, log_input, full, eps, reduction): 
        input1 = gendata([shape], type_list=[dtype], rand_algo="uniform", lower=-1.0, upper=1.0)[0]
        input2 = gendata([shape], type_list=[dtype], rand_algo="uniform", lower=-1.0, upper=1.0)[0]
        func = torch.nn.PoissonNLLLoss(log_input, full, eps=eps, reduction=reduction)
        runtestapi(func=func, fwd_input_list=[input1, input2], fwd_golden=[],  enable_backward=True)   

    @onlyCUDA
    @dtypes(*set(get_all_fp_dtypes(include_half=False, include_bfloat16=False)))
    @parametrize("shape", [[6], [6, 10], [4, 8, 100, 100]], name_fn=NORM_NAME)
    @parametrize("full", [True, False])
    @parametrize("eps", [1e-5])
    @parametrize("reduction", ["mean", "sum", "none"])
    def test_gaussian_nll_loss(self, dtype, device, shape, full, eps, reduction): 
        input = gendata([shape], type_list=[dtype], rand_algo="uniform", lower=-1.0, upper=1.0)[0]
        target = gendata([shape], type_list=[dtype], rand_algo="uniform", lower=-1.0, upper=1.0)[0]
        var = torch.ones(shape, requires_grad=True)
        func = torch.nn.GaussianNLLLoss(full=full, eps=eps, reduction=reduction)
        runtestapi(func=func, fwd_input_list=[input, target, var], fwd_golden=[],  enable_backward=True)   

    @onlyCUDA
    @dtypes(*set(get_all_fp_dtypes(include_half=False, include_bfloat16=False)))
    @parametrize("shape", [[6], [6, 10], [4, 8, 100, 100]], name_fn=NORM_NAME)
    @parametrize("reduction", ["mean", "sum", "none"])
    @parametrize("log_target", [True, False])
    def test_kl_div_loss(self, dtype, device, shape, reduction, log_target): 
        input = gendata([shape], type_list=[dtype], rand_algo="uniform", lower=-1.0, upper=1.0)[0]
        target = gendata([shape], type_list=[dtype], rand_algo="uniform", lower=-1.0, upper=1.0)[0]
        func = torch.nn.KLDivLoss(reduction=reduction, log_target=log_target)
        runtestapi(func=func, fwd_input_list=[input, target], fwd_golden=[],  enable_backward=True)  

    @onlyCUDA
    @dtypes(*set(get_all_fp_dtypes(include_half=False, include_bfloat16=False)))
    @parametrize("shape", [[6], [6, 10], [4, 8, 100, 100]], name_fn=NORM_NAME)
    @parametrize("reduction", ["mean", "sum", "none"])
    @parametrize("margin", [0.0, 0.5, 0.8])
    def test_margin_ranking_loss(self, dtype, device, shape, margin, reduction): 
        input1 = gendata([shape], type_list=[dtype], rand_algo="uniform", lower=-1.0, upper=1.0)[0]
        input2 = gendata([shape], type_list=[dtype], rand_algo="uniform", lower=-1.0, upper=1.0)[0]
        target = gendata([shape], type_list=[dtype], rand_algo="uniform", lower=-1.0, upper=1.0)[0]
        func = torch.nn.MarginRankingLoss(reduction=reduction, margin=margin)
        runtestapi(func=func, fwd_input_list=[input1, input2, target], fwd_golden=[],  enable_backward=True)      

    @onlyCUDA
    @dtypes(*set(get_all_fp_dtypes(include_half=False, include_bfloat16=False)))
    @parametrize("shape", [[6], [6, 10], [4, 8, 100, 100]], name_fn=NORM_NAME)
    @parametrize("reduction", ["mean", "sum", "none"])
    @parametrize("margin", [0.0, 0.5, 0.8])
    def test_hinge_embedding_loss(self, dtype, device, shape, margin, reduction): 
        input = gendata([shape], type_list=[dtype], rand_algo="uniform", lower=-1.0, upper=1.0)[0]
        target = torch.ones(shape, requires_grad=False)
        func = torch.nn.HingeEmbeddingLoss(reduction=reduction, margin=margin)
        runtestapi(func=func, fwd_input_list=[input, target], fwd_golden=[],  enable_backward=True, input_grad_skip_idx=[1]) 
    
    @onlyCUDA
    @dtypes(*set(get_all_fp_dtypes(include_half=False, include_bfloat16=False)))
    @parametrize("shape", [[6, 10], [4, 8]], name_fn=NORM_NAME)
    @parametrize("reduction", ["mean", "sum", "none"])
    def test_multi_label_margin_loss(self, dtype, device, shape, reduction): 
        input = gendata([shape], type_list=[dtype], rand_algo="uniform", lower=-1.0, upper=1.0)[0]
        target = gendata([shape], type_list=[torch.long], rand_algo="randint", lower=0, upper=shape[1])[0]
        func = torch.nn.MultiLabelMarginLoss(reduction=reduction)
        runtestapi(func=func, fwd_input_list=[input, target], fwd_golden=[],  enable_backward=True) 

    @onlyCUDA
    @dtypes(*set(get_all_fp_dtypes(include_half=False, include_bfloat16=False)))
    @parametrize("shape", [[6, 10], [4, 8]], name_fn=NORM_NAME)
    @parametrize("reduction", ["mean", "sum", "none"])
    @parametrize("delta", [0.5, 1.0, 2.0])
    def test_huber_loss(self, dtype, device, shape, reduction, delta): 
        input = gendata([shape], type_list=[dtype], rand_algo="uniform", lower=-1.0, upper=1.0)[0]
        target = gendata([shape], type_list=[dtype], rand_algo="uniform", lower=-1.0, upper=1.0)[0]
        func = torch.nn.HuberLoss(reduction=reduction, delta=delta)
        runtestapi(func=func, fwd_input_list=[input, target], fwd_golden=[],  enable_backward=True) 
    
    @onlyCUDA
    @dtypes(*set(get_all_fp_dtypes(include_half=False, include_bfloat16=False)))
    @parametrize("shape", [[6, 10], [4, 8]], name_fn=NORM_NAME)
    @parametrize("reduction", ["mean", "sum", "none"])
    @parametrize("beta", [0.0, 0.5, 1.0, 2.0])
    def test_smoothl1_loss(self, dtype, device, shape, reduction, beta): 
        input = gendata([shape], type_list=[dtype], rand_algo="uniform", lower=-1.0, upper=1.0)[0]
        target = gendata([shape], type_list=[dtype], rand_algo="uniform", lower=-1.0, upper=1.0)[0]
        func = torch.nn.SmoothL1Loss(reduction=reduction, beta=beta)
        runtestapi(func=func, fwd_input_list=[input, target], fwd_golden=[],  enable_backward=True)

    @onlyCUDA
    @dtypes(*set(get_all_fp_dtypes(include_half=False, include_bfloat16=False)))
    @parametrize("shape", [[6, 10], [4, 8, 10, 10]], name_fn=NORM_NAME)
    @parametrize("reduction", ["mean", "sum", "none"])
    def test_soft_margin_loss(self, dtype, device, shape, reduction): 
        input = gendata([shape], type_list=[dtype], rand_algo="uniform", lower=-1.0, upper=1.0)[0]
        target = gendata([shape], type_list=[dtype], rand_algo="uniform", lower=-1.0, upper=1.0)[0]
        func = torch.nn.SoftMarginLoss(reduction=reduction)
        runtestapi(func=func, fwd_input_list=[input, target], fwd_golden=[],  enable_backward=True, input_grad_skip_idx=[1])  

    @onlyCUDA
    @dtypes(*set(get_all_fp_dtypes(include_half=False, include_bfloat16=False)))
    @parametrize("shape", [[6, 10], [4, 8, 10, 10]], name_fn=NORM_NAME)
    @parametrize("reduction", ["mean", "sum", "none"])
    @parametrize("use_weight", [True, False])
    def test_multi_label_soft_margin_loss(self, dtype, device, shape, reduction, use_weight): 
        input = gendata([shape], type_list=[dtype], rand_algo="uniform", lower=-1.0, upper=1.0)[0]
        target = gendata([shape], type_list=[dtype], rand_algo="uniform", lower=-1.0, upper=1.0)[0]
        weight = None
        if use_weight:
            weight = make_tensor(shape=shape, device=device, dtype=dtype, low=-1., high=1.)
        func = torch.nn.MultiLabelSoftMarginLoss(weight=weight, reduction=reduction)
        runtestapi(func=func, fwd_input_list=[input, target], fwd_golden=[],  enable_backward=True)  
    
    @onlyCUDA
    @dtypes(*set(get_all_fp_dtypes(include_half=False, include_bfloat16=False)))
    @parametrize("shape", [[6, 10]], name_fn=NORM_NAME)
    @parametrize("margin", [-1, 0.0, 0.5, 1.0])
    @parametrize("reduction", ["mean", "sum", "none"])
    def test_cosine_embedding_loss(self, dtype, device, shape, margin, reduction): 
        input1 = gendata([shape], type_list=[dtype], rand_algo="uniform", lower=-1.0, upper=1.0)[0]
        input2 = gendata([shape], type_list=[dtype], rand_algo="uniform", lower=-1.0, upper=1.0)[0]
        target = torch.ones([shape[0]])
        func = torch.nn.CosineEmbeddingLoss(margin=margin, reduction=reduction)
        runtestapi(func=func, fwd_input_list=[input1, input2, target], fwd_golden=[],  enable_backward=True, input_grad_skip_idx=[2]) 

    @onlyCUDA
    @dtypes(*set(get_all_fp_dtypes(include_half=False, include_bfloat16=False)))
    @parametrize("shape", [[6, 10]], name_fn=NORM_NAME)
    @parametrize("margin", [-1, 0.0, 0.5, 1.0])
    @parametrize("reduction", ["mean", "sum", "none"])
    @parametrize("use_weight", [True, False])
    @parametrize("p", [1, 2])
    def test_multi_margin_loss(self, dtype, device, shape, p, margin, use_weight, reduction): 
        input = gendata([shape], type_list=[dtype], rand_algo="uniform", lower=-1.0, upper=1.0)[0]
        target = gendata([[shape[0]]], type_list=[torch.long], rand_algo="randint", lower=0, upper=shape[1]-1)[0]
        weight = None
        if use_weight:
            weight = make_tensor(shape=[(shape[1])], device=device, dtype=dtype, low=-1., high=1.)
        func = torch.nn.MultiMarginLoss(p=p, margin=margin, weight=weight, reduction=reduction)
        runtestapi(func=func, fwd_input_list=[input, target], fwd_golden=[],  enable_backward=True, input_grad_skip_idx=[1])

    @onlyCUDA
    @dtypes(*set(get_all_fp_dtypes(include_half=False, include_bfloat16=False)))
    @parametrize("shape", [[6, 10]], name_fn=NORM_NAME)
    @parametrize("margin", [-1, 0.0, 0.5, 1.0])
    @parametrize("reduction", ["mean", "sum", "none"])
    @parametrize("swap", [True, False])
    @parametrize("p", [1, 2])
    def test_triple_margin_loss(self, dtype, device, shape, p, margin, swap, reduction): 
        anchor = gendata([shape], type_list=[dtype], rand_algo="uniform", lower=-1.0, upper=1.0)[0]
        positive = gendata([shape], type_list=[dtype], rand_algo="uniform", lower=-1.0, upper=1.0)[0]
        negative = gendata([shape], type_list=[dtype], rand_algo="uniform", lower=-1.0, upper=1.0)[0]
        func = torch.nn.TripletMarginLoss(p=p, margin=margin, swap=swap, reduction=reduction)
        runtestapi(func=func, fwd_input_list=[anchor, positive, negative], fwd_golden=[],  enable_backward=True)


    @onlyCUDA
    @dtypes(*set(get_all_fp_dtypes(include_half=False, include_bfloat16=False)))
    @parametrize("shape", [[6, 10]], name_fn=NORM_NAME)
    @parametrize("margin", [-1, 0.0, 0.5, 1.0])
    @parametrize("reduction", ["mean", "sum", "none"])
    @parametrize("swap", [True, False])
    def test_triple_margin_with_distance_loss(self, dtype, device, shape, margin, swap, reduction): 
        anchor = gendata([shape], type_list=[dtype], rand_algo="uniform", lower=-1.0, upper=1.0)[0]
        positive = gendata([shape], type_list=[dtype], rand_algo="uniform", lower=-1.0, upper=1.0)[0]
        negative = gendata([shape], type_list=[dtype], rand_algo="uniform", lower=-1.0, upper=1.0)[0]
        distance_func = torch.nn.PairwiseDistance()
        func = torch.nn.TripletMarginWithDistanceLoss(distance_function=distance_func, margin=margin, swap=swap, reduction=reduction)
        runtestapi(func=func, fwd_input_list=[anchor, positive, negative], fwd_golden=[],  enable_backward=True)
    
class TestNNZmw(unittest.TestCase):
    
    def assert_equal(self, a, b):
        assert torch.allclose(a, b)

    # basicly copy from test/test_nn.py, supplement
    # 1. unshuffle backward test
    # 2. more dtypes test
    # 3. non-contiguous input
    def test_pixel_shuffle_unshuffle(self):
        def _test_pixel_shuffle_unshuffle_helper(num_input_dims, valid_channels_dim=True,
                                                 upscale_factor=None, dtype=torch.float32,
                                                 is_contig=True):
                        
            # Function to imperatively ensure pixels are shuffled to the correct locations.
            # Used to validate the batch operations in pixel_shuffle.
            def _verify_pixel_shuffle(input, output, upscale_factor):
                if input.is_cuda is True:
                    input = input.cpu()
                if output.is_cuda is True:
                    output = output.cpu()
                for c in range(output.size(-3)):
                    for h in range(output.size(-2)):
                        for w in range(output.size(-1)):
                            height_idx = h // upscale_factor
                            weight_idx = w // upscale_factor
                            channel_idx = (upscale_factor * (h % upscale_factor)) + (w % upscale_factor) + \
                                          (c * upscale_factor ** 2)
                            self.assert_equal(output[..., c, h, w], input[..., channel_idx, height_idx, weight_idx])
            
            def is_integer(dtype):
                return dtype in [torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8, torch.bool]

            upscale_factor = random.randint(2, 5) if upscale_factor is None else upscale_factor
            # If valid_channels_dim=False, add 1 to make channels dim indivisible by upscale_factor ** 2.
            channels = random.randint(1, 4) * upscale_factor ** 2 + (0 if valid_channels_dim else 1)
            height = random.randint(5, 10)
            width = random.randint(5, 10)

            gen_dtype = dtype
            if is_integer(dtype):
                gen_dtype = torch.float32

            if num_input_dims == 1:
                input = torch.rand(channels, requires_grad=True, device=torch.device("cuda:0"), dtype=gen_dtype)
            elif num_input_dims == 2:
                input = torch.rand(height, width, requires_grad=True, device=torch.device("cuda:0"), dtype=gen_dtype)
            else:
                batch_sizes = [random.randint(1, 3) for _ in range(num_input_dims - 3)]
                if is_contig:
                    input = torch.rand(*batch_sizes, channels, height, width, requires_grad=True, device=torch.device("cuda:0"), dtype=gen_dtype)
                else:
                    input = torch.rand(*batch_sizes, channels, width, height, requires_grad=True, device=torch.device("cuda:0"), dtype=gen_dtype).transpose(-1, -2)
                    input.retain_grad()

            if dtype == torch.uint8:
                input = (input * 255).to(dtype)
            elif dtype in [torch.int8, torch.int16, torch.int32, torch.int64]:
                input = (input * 255 - 128).to(dtype)
            elif dtype == torch.bool:
                input = (input > 0.5).to(dtype)

            ps = nn.PixelShuffle(upscale_factor)
            pus = nn.PixelUnshuffle(downscale_factor=upscale_factor)

            if num_input_dims >= 3 and valid_channels_dim and upscale_factor > 0:
                if is_integer(dtype):
                    output = ps(input)
                    _verify_pixel_shuffle(input, output, upscale_factor)

                    # Ensure unshuffle properly inverts shuffle.
                    unshuffle_output = pus(output)
                    self.assert_equal(input, unshuffle_output)
                else:
                    output = ps(input)
                    _verify_pixel_shuffle(input, output, upscale_factor)
                    output.backward(output)
                    self.assert_equal(input, input.grad)

                    # Ensure unshuffle properly inverts shuffle.
                    output.retain_grad()
                    unshuffle_output = pus(output)
                    unshuffle_output.backward(unshuffle_output)
                    self.assert_equal(input, unshuffle_output)
                    self.assert_equal(output, output.grad)
            else:
                self.assertRaises(RuntimeError, lambda: ps(input))

        def _test_pixel_unshuffle_error_case_helper(num_input_dims, valid_height_dim=True, valid_width_dim=True,
                                                    downscale_factor=None):
            downscale_factor = random.randint(2, 5) if downscale_factor is None else downscale_factor
            channels = random.randint(1, 4)
            # If valid_height_dim=False, add 1 to make height dim indivisible by downscale_factor.
            height = random.randint(3, 5) * abs(downscale_factor) + (0 if valid_height_dim else 1)
            # If valid_width_dim=False, add 1 to make width dim indivisible by downscale_factor.
            width = random.randint(3, 5) * abs(downscale_factor) + (0 if valid_width_dim else 1)

            if num_input_dims == 1:
                input = torch.rand(channels, requires_grad=True, device=torch.device("cuda:0"))
            elif num_input_dims == 2:
                input = torch.rand(height, width, requires_grad=True, device=torch.device("cuda:0"))
            else:
                batch_sizes = [random.randint(1, 3) for _ in range(num_input_dims - 3)]
                input = torch.rand(*batch_sizes, channels, height, width, requires_grad=True, device=torch.device("cuda:0"))

            pus = nn.PixelUnshuffle(downscale_factor)
            self.assertRaises(RuntimeError, lambda: pus(input))

        def _test_pixel_shuffle_unshuffle_for_input_dims(num_input_dims):
            # For 1D - 2D, this is an error case.
            # For 3D - 5D, this is a success case for pixel_shuffle + pixel_unshuffle.
            for dtype in ALL_FLOATING_TYPES:
                _test_pixel_shuffle_unshuffle_helper(num_input_dims=num_input_dims, dtype=dtype)
                _test_pixel_shuffle_unshuffle_helper(num_input_dims=num_input_dims, dtype=dtype, is_contig=False)
            for dtype in [torch.complex64, torch.complex128]:
                _test_pixel_shuffle_unshuffle_helper(num_input_dims=num_input_dims, dtype=dtype)
            for dtype in ALL_INTEGER_TYPES:
                _test_pixel_shuffle_unshuffle_helper(num_input_dims=num_input_dims, dtype=dtype)

            # Error cases for pixel_shuffle.
            _test_pixel_shuffle_unshuffle_helper(num_input_dims=num_input_dims, valid_channels_dim=False)
            _test_pixel_shuffle_unshuffle_helper(num_input_dims=num_input_dims, upscale_factor=0)
            _test_pixel_shuffle_unshuffle_helper(num_input_dims=num_input_dims, upscale_factor=-2)

            # Error cases for pixel_unshuffle.
            _test_pixel_unshuffle_error_case_helper(num_input_dims=num_input_dims, valid_height_dim=False)
            _test_pixel_unshuffle_error_case_helper(num_input_dims=num_input_dims, valid_width_dim=False)
            _test_pixel_unshuffle_error_case_helper(num_input_dims=num_input_dims, downscale_factor=0)
            _test_pixel_unshuffle_error_case_helper(num_input_dims=num_input_dims, downscale_factor=-2)

        def test_pixel_shuffle_unshuffle_1D():
            _test_pixel_shuffle_unshuffle_for_input_dims(num_input_dims=1)

        def test_pixel_shuffle_unshuffle_2D():
            _test_pixel_shuffle_unshuffle_for_input_dims(num_input_dims=2)

        def test_pixel_shuffle_unshuffle_3D():
            _test_pixel_shuffle_unshuffle_for_input_dims(num_input_dims=3)

        def test_pixel_shuffle_unshuffle_4D():
            _test_pixel_shuffle_unshuffle_for_input_dims(num_input_dims=4)

        def test_pixel_shuffle_unshuffle_5D():
            _test_pixel_shuffle_unshuffle_for_input_dims(num_input_dims=5)

        test_pixel_shuffle_unshuffle_1D()
        test_pixel_shuffle_unshuffle_2D()
        test_pixel_shuffle_unshuffle_3D()
        test_pixel_shuffle_unshuffle_4D()
        test_pixel_shuffle_unshuffle_5D()

    # in pytorch 1.10.0-rc3, ChannelShuffle is not implemented for gpu, and problematic for cpu
    def test_channel_shuffle(self):
        pass

    def test_clip_grad_value(self):
        l = nn.Linear(10, 10).cuda()
        clip_value = 2.5

        grad_w, grad_b = torch.arange(-50., 50).view(10, 10).div_(5), torch.ones(10).mul_(2)
        for grad_list in [[grad_w, grad_b], [grad_w, None]]:
            for p, g in zip(l.parameters(), grad_list):
                p.grad = g.clone().view_as(p.data).cuda() if g is not None else g

            torch.nn.utils.clip_grad_value_(l.parameters(), clip_value)
            for p in filter(lambda p: p.grad is not None, l.parameters()):
                self.assertLessEqual(p.grad.data.max(), clip_value)
                self.assertGreaterEqual(p.grad.data.min(), -clip_value)

        # Should accept a single Tensor as input
        p1, p2 = torch.randn(10, 10).cuda(), torch.randn(10, 10).cuda()
        g = torch.arange(-50., 50).view(10, 10).div_(5).cuda()
        p1._grad = g.clone()
        p2._grad = g.clone()
        torch.nn.utils.clip_grad_value_(p1, clip_value)
        torch.nn.utils.clip_grad_value_([p2], clip_value)
        self.assert_equal(p1.grad, p2.grad)


    def convertion_between_parameters_and_vecotr(self):
        # parameter to vector
        conv1 = nn.Conv2d(3, 10, 5)
        fc1 = nn.Linear(10, 20)
        model = nn.Sequential(conv1, fc1).cuda()
        vec = torch.nn.utils.parameters_to_vector(model.parameters())
        self.assertEqual(vec.size(0), 980)

        # vector to parameter
        vec = torch.arange(0., 980).cuda()
        torch.nn.utils.vector_to_parameters(vec, model.parameters())

        sample = next(model.parameters())[0, 0, 0]
        self.assertTrue(torch.equal(sample.data, vec.data[:5]))
        
    def test_prune(self):
        # create a new pruning method
        p = prune.L1Unstructured(amount=2)
        # create tensor to be pruned
        t = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]]).to(dtype=torch.float32).cuda()
        # create prior mask by hand
        default_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 1]]).cuda()
        # since we are pruning the two lowest magnitude units, the outcome of
        # the calculation should be this:
        expected_mask = torch.tensor([[0, 0, 1, 0], [1, 1, 0, 1]]).cuda()
        pruned_tensor = p.prune(t, default_mask)
        self.assert_equal(t * expected_mask, pruned_tensor)

    def test_prune_importance_scores(self):
        # create a new pruning method
        p = prune.L1Unstructured(amount=2)
        # create tensor to be pruned
        t = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]]).to(dtype=torch.float32).cuda()
        importance_scores = torch.tensor(
            [[1, 2, 3, 4], [1.5, 1.6, 1.7, 1.8]]
        ).to(dtype=torch.float32).cuda()
        # create prior mask by hand
        default_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 1]]).cuda()
        # since we are pruning the two lowest magnitude units, the outcome of
        # the calculation should be this:
        expected_mask = torch.tensor([[0, 1, 1, 0], [0, 1, 0, 1]]).cuda()
        pruned_tensor = p.prune(t, default_mask, importance_scores=importance_scores)
        self.assert_equal(t * expected_mask, pruned_tensor)

    def test_prune_importance_scores_mimic_default(self):
        # create a new pruning method
        p = prune.L1Unstructured(amount=2)
        # create tensor to be pruned
        t = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]]).to(dtype=torch.float32).cuda()
        # create prior mask by hand
        default_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 1]]).cuda()
        # since we are pruning the two lowest magnitude units, the outcome of
        # the calculation should be this:
        expected_mask = torch.tensor([[0, 0, 1, 0], [1, 1, 0, 1]]).cuda()
        pruned_tensor_without_importance_scores = p.prune(t, default_mask)
        pruned_tensor_with_importance_scores = p.prune(t, default_mask, importance_scores=t)
        self.assert_equal(pruned_tensor_without_importance_scores, pruned_tensor_with_importance_scores)
        self.assert_equal(t * expected_mask, pruned_tensor_without_importance_scores)

"""
TestPooling includes all pooling python comparsion tests.
test_pool: Tests for MaxPool2d, AvgPool2d, AvgPool1d,
MaxPool1d, AvgPool3d, MaxPool3d.
test_adaptive_pool: Tests for AdaptiveMaxPool2d, AdaptiveAvgPool2d, AdaptiveAvgPool1d,
AdaptiveMaxPool1d, AdaptiveAvgPool3d, AdaptiveMaxPool3d.
test_fractional_pool: Tests for FractionalMaxpool2d, FractionalMaxpool3d.
"""
class TestPooling(unittest.TestCase):
    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_fp_dtypes(include_bfloat16=False)))
    @parametrize("layout", [torch.contiguous_format, torch.channels_last],
                 name_fn=lambda layout: 'NCHW' if layout==torch.contiguous_format else 'NHWC')
    @parametrize("pool_type,has_indices", [("Max",False),("Avg",False),("Max",True)],
                 name_fn=lambda pool_type, has_indices: '{}PoolWithIndices'.format(pool_type)
                 if has_indices else '{}Pool'.format(pool_type))
    def test_pool(self, device, dtype, layout, pool_type, has_indices):
        def helper(shape, kernel_size, stride=None,
                   count_include_pad=True, divisor_override=None, padding=0):
            if layout == torch.channels_last and len(shape) != 4:
                return
            if layout == torch.channels_last and has_indices:
                print("CUDA computes maxpool indices differently in NHWC!")
                return
            if dtype == torch.float16:
                fwd_tol = 1e-2
                bwd_tol = 1e-2
            else:
                fwd_tol = 1e-4
                bwd_tol = 1e-4
            if stride is None:
                stride = kernel_size
            pool_dim = len(shape)-2
            cls_name = "{}Pool{}d"
            kwargs = {'kernel_size': kernel_size, 'stride': stride}
            cls = getattr(torch.nn, cls_name.format(pool_type, pool_dim))
            shape_list = [(shape)]
            forward_inputs = gendata(shape_list, type_list=[dtype], rand_algo="rand", memory_formats=[layout])
            if pool_type == "Avg" and pool_dim == 1:
                kwargs['count_include_pad'] = count_include_pad
            elif pool_type == "Avg":
                kwargs['count_include_pad'] = count_include_pad
                kwargs['divisor_override'] = divisor_override
            elif pool_type == "Max":
                kwargs['return_indices'] = has_indices
            else:
                raise Exception("Invalid pool type!")
            pool = cls(**kwargs)

            runtestapi(func=pool, type_dict={torch.float:{torch.float16}}, fwd_input_list=forward_inputs, enable_backward=True, fwd_tol=fwd_tol, bwd_tol=bwd_tol)

        # Pool1d
        helper((1, 10, 10), 2, stride=2)
        helper((1, 16, 50), 3, stride=2)
        # Pool2d
        helper((4, 8, 8, 8), 3)
        helper((4, 8, 8, 8), 3, count_include_pad=False, padding=1)
        helper((4, 8, 8, 8), 3, count_include_pad=False, padding=2, stride=2)
        helper((4, 8, 8, 8), 3, divisor_override=42)
        helper((4, 8, 8, 8), 7)
        # Clear caching allocator prior to running large subtest.
        if 'cuda' in device:
            torch.cuda.empty_cache()
        #helper((200, 512, 28, 28), 2)
        helper((4, 8, 7, 7), 3, stride=1)
        helper((4, 8, 7, 7), 3, padding=2, stride=1)
        #helper((10, 512, 31, 31), 3, stride=2)
        helper((1, 129, 8, 8), 3, stride=2)
        # Pool3d
        helper((2, 3, 6, 6, 6), 1)
        helper((1, 3, 4, 4, 4), 3, stride=2)

    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_fp_dtypes(include_bfloat16=False)))
    @parametrize("layout", [torch.contiguous_format, torch.channels_last],
                 name_fn=lambda layout: 'NCHW' if layout==torch.contiguous_format else 'NHWC')
    @parametrize("pool_type,has_indices", [("Max",False),("Avg",False),("Max",True)],
                 name_fn=lambda pool_type, has_indices: '{}PoolWithIndices'.format(pool_type)
                 if has_indices else '{}Pool'.format(pool_type))
    def test_adaptive_pool(self, device, dtype, layout, pool_type, has_indices):
        def helper(shape, output_size):
            if layout == torch.channels_last and len(shape) != 4:
                return
            pool_dim = len(shape)-2
            cls_name = "Adaptive{}Pool{}d"
            kwargs = {'output_size': output_size}
            cls = getattr(torch.nn, cls_name.format(pool_type, pool_dim))
            shape_list = [(shape)]
            forward_inputs = gendata(shape_list, type_list=[dtype], rand_algo="rand", memory_formats=[layout])
            if pool_type == "Max":
                kwargs['return_indices'] = has_indices
            pool = cls(**kwargs)

            runtestapi(func=pool, type_dict={torch.float:{torch.float16}}, fwd_input_list=forward_inputs, enable_backward=True)

        # AdaptivePool1d
        helper((0, 8, 5), 3)
        # AdaptivePool2d
        helper((0, 8, 5, 4), 3)
        # AdaptivePool3d
        helper((0, 8, 5, 4, 3), 3)

    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_fp_dtypes(include_bfloat16=False)))
    @parametrize("layout", [torch.contiguous_format, torch.channels_last],
                 name_fn=lambda layout: 'NCHW' if layout==torch.contiguous_format else 'NHWC')
    @parametrize("pool_type,has_indices", [("Max",False), ("Max",True)],
                 name_fn=lambda pool_type, has_indices: '{}PoolWithIndices'.format(pool_type)
                 if has_indices else '{}Pool'.format(pool_type))
    def test_fractional_pool(self, device, dtype, layout, pool_type, has_indices):
        def helper(shape, kernel_size, output_size=None, output_ratio=None):
            if layout == torch.channels_last and len(shape) != 4:
                return
            pool_dim = len(shape)-2
            if pool_dim <= 1:
                raise Exception("Fractional pooling not supportd 1d!")
            cls_name = "Fractional{}Pool{}d"
            kwargs = {'kernel_size': kernel_size, 'output_size': output_size, 'output_ratio': output_ratio}
            cls = getattr(torch.nn, cls_name.format(pool_type, pool_dim))
            shape_list = [(shape)]
            forward_inputs = gendata(shape_list, type_list=[dtype], rand_algo="rand", memory_formats=[layout])
            if pool_type == "Max":
                kwargs['return_indices'] = has_indices
            else:
                raise Exception("Invalid pool type for fractional pooling!")
            pool = cls(**kwargs)

            runtestapi(func=pool, type_dict={torch.float:{torch.float16}}, fwd_input_list=forward_inputs, enable_backward=True)

        ## FIXME(yuliu): CUDA computes differently in fractional pooling cases!
        ## FractionalPool2d
        #helper((1, 16, 50, 32), 3, output_ratio=(0.5, 0.5))
        ## FractionalPool3d
        #helper((1, 16, 50, 32, 32), 3, output_ratio=(0.5, 0.5, 0.5))

class TestNorm(unittest.TestCase):
    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_fp_dtypes(include_bfloat16=False)))
    @parametrize("norm_type", ['weight']) #TODO(yuliu): spectral norm fails GPU/CPU accuracy comparsion!
    def test_norm(self, device, dtype, norm_type):
        def helper(layer, shape, dim=None):
            shape_list = [(shape)]
            forward_inputs = gendata(shape_list, type_list=[dtype], rand_algo="rand")
            cls_name = "{}_norm".format(norm_type)
            rm_cls_name = "remove_{}_norm".format(norm_type)
            cls = getattr(torch.nn.utils, cls_name)
            rm_cls = getattr(torch.nn.utils, rm_cls_name)
            kwargs = {'module': layer, 'dim': dim}
            norm_func = cls(**kwargs)

            runtestapi(func=norm_func, type_dict={torch.float:{torch.float16}}, fwd_input_list=forward_inputs, enable_backward=False, fwd_tol=0.003)

            rm_norm_func = rm_cls(norm_func)

            runtestapi(func=rm_norm_func, type_dict={torch.float:{torch.float16}}, fwd_input_list=forward_inputs, enable_backward=False, fwd_tol=0.003)

        helper(torch.nn.Linear(5, 7), (3, 5))
        helper(torch.nn.Linear(5, 7), (3, 5), dim=1)

class TestFlatten(unittest.TestCase):
    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_fp_dtypes(include_bfloat16=False)))
    def test_flatten(self, device, dtype):
        def helper(shape, start_dim, end_dim):
            shape_list = [(shape)]
            forward_inputs = gendata(shape_list, type_list=[dtype], rand_algo="rand")
            flatten = nn.Flatten(start_dim=start_dim, end_dim=end_dim)

            runtestapi(func=flatten, type_dict={torch.float:{torch.float16}}, fwd_input_list=forward_inputs, enable_backward=False)

        helper((2, 1, 2, 3), start_dim=1, end_dim=-1)


    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_fp_dtypes(include_bfloat16=False)))
    def test_unflatten(self, device, dtype):
        def helper(shape, dim, unflattened_size):
            shape_list = [(shape)]
            forward_inputs = gendata(shape_list, type_list=[dtype], rand_algo="rand")
            unflatten = nn.Unflatten(dim=1, unflattened_size=unflattened_size)

            runtestapi(func=unflatten, type_dict={torch.float:{torch.float16}}, fwd_input_list=forward_inputs, enable_backward=False)

        helper((2, 50), dim=1, unflattened_size=(2, 5, 5))

class TestNNSegment0(unittest.TestCase):

    ##### nn.BatchNorm
    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_fp_dtypes(include_half=False, include_bfloat16=False)))
    def test_nn_BatchNorm(self, device, dtype):
        
        def help(shape_list, op, is_lazy=False):
            forward_inputs = gendata(shape_list, type_list=[dtype], rand_algo="rand")
            eps_list = [1e-5, 5e-5]
            momentum_list = [0.1, 0.2]
            affine_list = [True, False]
            track_running_stats_list = [True, False]
            for input, eps, momentum, affine, track_running_stats in product(forward_inputs, eps_list, momentum_list, affine_list, track_running_stats_list):
                if is_lazy:
                    func = op(eps, momentum, affine, track_running_stats)
                else:
                    num_features = input.shape[1]
                    func = op(num_features, eps, momentum, affine, track_running_stats)
                runtestapi(func=func, fwd_input_list=[input], enable_backward=True)

        help([(4, 512), (4, 15, 256)], nn.BatchNorm1d)
        help([(4, 32, 32, 32), (4, 128, 32, 32)], nn.BatchNorm2d)
        help([(4, 128, 8, 8, 8)], nn.BatchNorm3d)

        help([(4, 512), (4, 15, 512)], nn.LazyBatchNorm1d, True)
        help([(4, 32, 32, 32), (4, 256, 32, 32)], nn.LazyBatchNorm2d, True)
        help([(4, 128, 8, 8, 8)], nn.LazyBatchNorm3d, True)

    ##### nn.InstanceNorm
    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_fp_dtypes(include_half=False, include_bfloat16=False)))
    def test_nn_InstanceNorm(self, device, dtype):
        
        def help(shape_list, norm_level, is_lazy=False):
            forward_inputs = gendata(shape_list, type_list=[dtype], rand_algo="rand")
            eps_list = [1e-5, 5e-5]
            momentum_list = [0.1, 0.2]
            affine_list = [True, False]
            track_running_stats_list = [True, False]
            if is_lazy:
                op_name = "LazyInstanceNorm{}d".format(norm_level)
            else:
                op_name = "InstanceNorm{}d".format(norm_level)
            op = getattr(nn, op_name)
            for input, eps, momentum, affine, track_running_stats in product(forward_inputs, eps_list, momentum_list, affine_list, track_running_stats_list):
                if is_lazy:
                    func = op(eps, momentum, affine, track_running_stats)
                else:
                    num_features = input.shape[1]
                    func = op(num_features, eps, momentum, affine, track_running_stats)
                runtestapi(func=func, fwd_input_list=[input], enable_backward=True)

        help([(4, 33, 256)], 1)
        help([(4, 32, 32, 32)], 2)
        help([(4, 128, 8, 8, 8)], 3)

        help([(4, 33, 256)], 1, True)
        help([(4, 32, 32, 32)], 2, True)
        help([(4, 128, 8, 8, 8)], 3, True)

class TestNNSegment2(unittest.TestCase):
    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_fp_dtypes(include_half=True, include_bfloat16=False)))
    @with_cudnn_off_helper
    def test_conv1d(self, device, dtype):

        def test_help(N, Cin, Cout, L, kwargs):
            shapes=(N, Cin, L)
            input = make_tensor(shapes, dtype=dtype, device='cpu',low=0, high=1)
            input_list = [input]
            model = nn.Conv1d(Cin, Cout, **kwargs)
            tol = 1e-2

            runtestapi(func=model, fwd_input_list=input_list, enable_backward=True, 
                    type_dict={torch.float:{torch.float16, torch.bfloat16}}, fwd_tol=tol, bwd_tol=tol,
                    low=0, high=1, compare_type="torch")

        Ns = [1, 7]
        Cins = [4]
        Couts = [4, 9]
        Ls = [11]
        kwargss = [{"kernel_size":2,"stride":1,"padding":0,"dilation":1,"groups":1,"bias":True,"padding_mode":"zeros"},
                  {"kernel_size":2,"stride":2,"padding":2,"dilation":1,"groups":2,"bias":False,"padding_mode":"reflect"},
                  {"kernel_size":2,"stride":2,"padding":2,"dilation":2,"groups":1,"bias":True,"padding_mode":"replicate"},
                  {"kernel_size":2,"stride":1,"padding":3,"dilation":2,"groups":2,"bias":False,"padding_mode":"circular"},
                  {"kernel_size":3,"stride":1,"padding":3,"dilation":2,"groups":2,"bias":False,"padding_mode":"zeros"},
                  {"kernel_size":2,"stride":2,"padding":2,"dilation":1,"groups":1,"bias":False,"padding_mode":"reflect"},
                  {"kernel_size":2,"stride":1,"padding":3,"dilation":1,"groups":1,"bias":True,"padding_mode":"replicate"},
                  {"kernel_size":2,"stride":1,"padding":3,"dilation":2,"groups":4,"bias":False,"padding_mode":"circular"}
                 ]
        for N in Ns:
            for Cin in Cins:
                for Cout in Couts:
                    for L in Ls:
                        for kwargs in kwargss:
                            if Cin%kwargs["groups"]==0 and Cout%kwargs["groups"]==0:
                                test_help(N, Cin, Cout, L, kwargs)
  

    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_fp_dtypes(include_half=True, include_bfloat16=False)))
    @with_cudnn_off_helper
    def test_conv2d(self, device, dtype):

        def test_help(N, Cin, Cout, h, w, kwargs):
            shapes=(N, Cin, h, w)
            input = make_tensor(shapes, dtype=dtype, device='cpu',low=0, high=1)
            input_list = [input]
            model = nn.Conv2d(Cin, Cout, **kwargs)
            tol = 1e-2

            runtestapi(func=model, fwd_input_list=input_list, enable_backward=True, 
                    type_dict={torch.float:{torch.float16, torch.bfloat16}}, fwd_tol=tol, bwd_tol=tol,
                    low=0, high=1, compare_type="torch")

        Ns = [1]
        Cins = [4, 9]
        Couts = [4]
        hws = [(11, 12), (8,16)]
        kwargss = [{"kernel_size":2,"stride":1,"padding":0,"dilation":1,"groups":1,"bias":True,"padding_mode":"zeros"},
                  {"kernel_size":2,"stride":2,"padding":2,"dilation":1,"groups":2,"bias":False,"padding_mode":"reflect"},
                  {"kernel_size":3,"stride":2,"padding":4,"dilation":2,"groups":1,"bias":True,"padding_mode":"replicate"},
                  {"kernel_size":2,"stride":1,"padding":3,"dilation":2,"groups":2,"bias":False,"padding_mode":"circular"},
                  {"kernel_size":3,"stride":1,"padding":3,"dilation":2,"groups":2,"bias":False,"padding_mode":"zeros"},
                  {"kernel_size":4,"stride":2,"padding":2,"dilation":1,"groups":1,"bias":False,"padding_mode":"reflect"},
                  {"kernel_size":2,"stride":3,"padding":3,"dilation":1,"groups":1,"bias":True,"padding_mode":"replicate"},
                  {"kernel_size":2,"stride":1,"padding":3,"dilation":2,"groups":4,"bias":False,"padding_mode":"circular"}
                 ]
        for N in Ns:
            for Cin in Cins:
                for Cout in Couts:
                    for (h,w) in hws:
                        for kwargs in kwargss:
                            if Cin%kwargs["groups"]==0 and Cout%kwargs["groups"]==0:
                                test_help(N, Cin, Cout, h, w, kwargs)
  

    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_fp_dtypes(include_half=True, include_bfloat16=False)))
    @with_cudnn_off_helper
    def test_conv3d(self, device, dtype):
        

        def test_help(N, Cin, Cout, d, h, w, kwargs):
            shapes=(N, Cin, d, h, w)
            input = make_tensor(shapes, dtype=dtype, device='cpu',low=0, high=1)
            input_list = [input]
            model = nn.Conv3d(Cin, Cout, **kwargs)
            tol = 1e-2

            runtestapi(func=model, fwd_input_list=input_list, enable_backward=True, 
                    type_dict={torch.float:{torch.float16, torch.bfloat16}}, fwd_tol=tol, bwd_tol=tol,
                    low=0, high=1, compare_type="torch")


        Ns = [1, 5]
        Cins = [4]
        Couts = [4, 9]
        dhws = [(5, 11, 12)]
        kwargss = [{"kernel_size":2,"stride":1,"padding":0,"dilation":1,"groups":1,"bias":True,"padding_mode":"zeros"},
                  {"kernel_size":2,"stride":2,"padding":2,"dilation":1,"groups":2,"bias":False,"padding_mode":"reflect"},
                  {"kernel_size":3,"stride":2,"padding":4,"dilation":2,"groups":1,"bias":True,"padding_mode":"replicate"},
                  {"kernel_size":2,"stride":1,"padding":3,"dilation":2,"groups":2,"bias":False,"padding_mode":"circular"},
                  {"kernel_size":3,"stride":2,"padding":3,"dilation":2,"groups":1,"bias":False,"padding_mode":"zeros"},
                  {"kernel_size":4,"stride":2,"padding":2,"dilation":2,"groups":2,"bias":True,"padding_mode":"reflect"},
                  {"kernel_size":2,"stride":3,"padding":3,"dilation":1,"groups":1,"bias":True,"padding_mode":"replicate"},
                  {"kernel_size":2,"stride":1,"padding":1,"dilation":2,"groups":4,"bias":False,"padding_mode":"circular"}
                 ]
        for N in Ns:
            for Cin in Cins:
                for Cout in Couts:
                    for (d,h,w) in dhws:
                        for kwargs in kwargss:
                            if Cin%kwargs["groups"]==0 and Cout%kwargs["groups"]==0:
                                test_help(N, Cin, Cout, d, h, w, kwargs)


    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_fp_dtypes(include_half=True, include_bfloat16=False)))
    @with_cudnn_off_helper
    def test_convtranspose1d(self, device, dtype):
        

        def test_help(N, Cin, Cout, L, kwargs):
            shapes=(N, Cin, L)
            input = make_tensor(shapes, dtype=dtype, device='cpu',low=0, high=1)
            input_list = [input]
            model = nn.ConvTranspose1d(Cin, Cout, **kwargs)
            tol = 1e-2

            runtestapi(func=model, fwd_input_list=input_list, enable_backward=True, 
                    type_dict={torch.float:{torch.float16, torch.bfloat16}}, fwd_tol=tol, bwd_tol=tol,
                    low=0, high=1, compare_type="torch")

        Ns = [7]
        Cins = [4]
        Couts = [4, 7]
        Ls = [11]
        kwargss = [{"kernel_size":2,"stride":1,"padding":0,"dilation":1,"groups":1,"bias":True,"padding_mode":"zeros"},
                  {"kernel_size":2,"stride":2,"padding":2,"dilation":1,"groups":2,"bias":False,"padding_mode":"zeros"},
                  {"kernel_size":2,"stride":2,"padding":2,"dilation":2,"groups":1,"bias":True,"padding_mode":"zeros"},
                  {"kernel_size":2,"stride":1,"padding":3,"dilation":2,"groups":2,"bias":False,"padding_mode":"zeros"},
                  {"kernel_size":3,"stride":1,"padding":3,"dilation":2,"groups":2,"bias":False,"padding_mode":"zeros"},
                  {"kernel_size":2,"stride":2,"padding":2,"dilation":1,"groups":1,"bias":False,"padding_mode":"zeros"},
                  {"kernel_size":2,"stride":1,"padding":3,"dilation":1,"groups":1,"bias":True,"padding_mode":"zeros"},
                  {"kernel_size":2,"stride":1,"padding":3,"dilation":2,"groups":4,"bias":False,"padding_mode":"zeros"}
                 ]
        for N in Ns:
            for Cin in Cins:
                for Cout in Couts:
                    for L in Ls:
                        for kwargs in kwargss:
                            if Cin%kwargs["groups"]==0 and Cout%kwargs["groups"]==0:
                                test_help(N, Cin, Cout, L, kwargs)
  

    @onlyCUDA   
    @dtypesIfCUDA(*set(get_all_fp_dtypes(include_half=True, include_bfloat16=False)))
    @with_cudnn_off_helper
    def test_convtranspose2d(self, device, dtype):
        

        def test_help(N, Cin, Cout, h, w, kwargs):
            shapes=(N, Cin, h, w)
            input = make_tensor(shapes, dtype=dtype, device='cpu',low=0, high=1)
            input_list = [input]
            model = nn.ConvTranspose2d(Cin, Cout, **kwargs)
            tol = 1e-2

            runtestapi(func=model, fwd_input_list=input_list, enable_backward=True, 
                    type_dict={torch.float:{torch.float16, torch.bfloat16}}, fwd_tol=tol, bwd_tol=tol,
                    low=0, high=1, compare_type="torch")


        Ns = [1, 5]
        Cins = [4,8]
        Couts = [4]
        hws = [(11, 12)]
        kwargss = [{"kernel_size":2,"stride":1,"padding":0,"dilation":1,"groups":1,"bias":True,"padding_mode":"zeros"},
                  {"kernel_size":2,"stride":2,"padding":2,"dilation":1,"groups":2,"bias":False,"padding_mode":"zeros"},
                  {"kernel_size":3,"stride":2,"padding":4,"dilation":2,"groups":1,"bias":True,"padding_mode":"zeros"},
                  {"kernel_size":2,"stride":1,"padding":3,"dilation":2,"groups":2,"bias":False,"padding_mode":"zeros"},
                  {"kernel_size":3,"stride":1,"padding":3,"dilation":2,"groups":2,"bias":False,"padding_mode":"zeros"},
                  {"kernel_size":4,"stride":2,"padding":2,"dilation":1,"groups":1,"bias":False,"padding_mode":"zeros"},
                  {"kernel_size":2,"stride":3,"padding":3,"dilation":1,"groups":1,"bias":True,"padding_mode":"zeros"},
                  {"kernel_size":2,"stride":1,"padding":3,"dilation":2,"groups":4,"bias":False,"padding_mode":"zeros"}
                 ]
        for N in Ns:
            for Cin in Cins:
                for Cout in Couts:
                    for (h,w) in hws:
                        for kwargs in kwargss:
                            if Cin%kwargs["groups"]==0 and Cout%kwargs["groups"]==0:
                                test_help(N, Cin, Cout, h, w, kwargs)


    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_fp_dtypes(include_half=True, include_bfloat16=False)))
    @with_cudnn_off_helper
    def test_convtranspose3d(self, device, dtype):
        

        def test_help(N, Cin, Cout, d, h, w, kwargs):
            shapes=(N, Cin, d, h, w)
            input = make_tensor(shapes, dtype=dtype, device='cpu',low=0, high=1)
            input_list = [input]
            model = nn.ConvTranspose3d(Cin, Cout, **kwargs)
            tol = 1e-2

            runtestapi(func=model, fwd_input_list=input_list, enable_backward=True, 
                    type_dict={torch.float:{torch.float16, torch.bfloat16}}, fwd_tol=tol, bwd_tol=tol,
                    low=0, high=1, compare_type="torch")

        Ns = [1, 5]
        Cins = [8]
        Couts = [4, 8]
        dhws = [(9, 15,7)]
        kwargss = [{"kernel_size":2,"stride":1,"padding":0,"dilation":1,"groups":1,"bias":True,"padding_mode":"zeros"},
                  {"kernel_size":2,"stride":2,"padding":2,"dilation":1,"groups":2,"bias":False,"padding_mode":"zeros"},
                  {"kernel_size":3,"stride":2,"padding":4,"dilation":2,"groups":1,"bias":True,"padding_mode":"zeros"},
                  {"kernel_size":2,"stride":1,"padding":3,"dilation":2,"groups":2,"bias":False,"padding_mode":"zeros"},
                  {"kernel_size":3,"stride":2,"padding":3,"dilation":2,"groups":1,"bias":False,"padding_mode":"zeros"},
                  {"kernel_size":4,"stride":2,"padding":2,"dilation":2,"groups":2,"bias":True,"padding_mode":"zeros"},
                  {"kernel_size":2,"stride":3,"padding":3,"dilation":1,"groups":1,"bias":True,"padding_mode":"zeros"},
                  {"kernel_size":2,"stride":1,"padding":1,"dilation":2,"groups":4,"bias":False,"padding_mode":"zeros"}
                 ]
        for N in Ns:
            for Cin in Cins:
                for Cout in Couts:
                    for (d,h,w) in dhws:
                        for kwargs in kwargss:
                            if Cin%kwargs["groups"]==0 and Cout%kwargs["groups"]==0:
                                test_help(N, Cin, Cout, d, h, w, kwargs)


    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_fp_dtypes(include_half=True, include_bfloat16=False)))
    @with_cudnn_off_helper
    def test_lazyconv1d(self, device, dtype):
        

        def test_help(N, Cin, Cout, L, kwargs):
            shapes=(N, Cin, L)
            input = make_tensor(shapes, dtype=dtype, device='cpu',low=0, high=1)
            input_list = [input]
            model = nn.LazyConv1d(Cout, **kwargs)
            tol = 1e-2

            runtestapi(func=model, fwd_input_list=input_list, enable_backward=True, 
                    type_dict={torch.float:{torch.float16, torch.bfloat16}}, fwd_tol=tol, bwd_tol=tol,
                    low=0, high=1, compare_type="torch")
        
        Ns = [7, 16]
        Cins = [9]
        Couts = [4, 7]
        Ls = [11]
        kwargss = [{"kernel_size":2,"stride":1,"padding":0,"dilation":1,"groups":1,"bias":True,"padding_mode":"zeros"},
                  {"kernel_size":2,"stride":2,"padding":2,"dilation":1,"groups":2,"bias":False,"padding_mode":"reflect"},
                  {"kernel_size":2,"stride":2,"padding":2,"dilation":2,"groups":1,"bias":True,"padding_mode":"replicate"},
                  {"kernel_size":2,"stride":1,"padding":3,"dilation":2,"groups":2,"bias":False,"padding_mode":"circular"},
                  {"kernel_size":3,"stride":1,"padding":3,"dilation":2,"groups":2,"bias":False,"padding_mode":"zeros"},
                  {"kernel_size":2,"stride":2,"padding":2,"dilation":1,"groups":1,"bias":False,"padding_mode":"reflect"},
                  {"kernel_size":2,"stride":1,"padding":3,"dilation":1,"groups":1,"bias":True,"padding_mode":"replicate"},
                  {"kernel_size":2,"stride":1,"padding":3,"dilation":2,"groups":4,"bias":False,"padding_mode":"circular"}
                 ]
        for N in Ns:
            for Cin in Cins:
                for Cout in Couts:
                    for L in Ls:
                        for kwargs in kwargss:
                            if Cin%kwargs["groups"]==0 and Cout%kwargs["groups"]==0:
                                test_help(N, Cin, Cout, L, kwargs)
  

    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_fp_dtypes(include_half=True, include_bfloat16=False)))
    @with_cudnn_off_helper
    def test_lazyconv2d(self, device, dtype):
        

        def test_help(N, Cin, Cout, h, w, kwargs):
            shapes=(N, Cin, h, w)
            input = make_tensor(shapes, dtype=dtype, device='cpu',low=0, high=1)
            input_list = [input]
            model = nn.LazyConv2d(Cout, **kwargs)
            tol = 1e-2

            runtestapi(func=model, fwd_input_list=input_list, enable_backward=True, 
                    type_dict={torch.float:{torch.float16, torch.bfloat16}}, fwd_tol=tol, bwd_tol=tol,
                    low=0, high=1, compare_type="torch")

        Ns = [1,]
        Cins = [4, 7]
        Couts = [4, 9]
        hws = [(8,16)]
        kwargss = [{"kernel_size":2,"stride":1,"padding":0,"dilation":1,"groups":1,"bias":True,"padding_mode":"zeros"},
                  {"kernel_size":2,"stride":2,"padding":2,"dilation":1,"groups":2,"bias":False,"padding_mode":"reflect"},
                  {"kernel_size":3,"stride":2,"padding":4,"dilation":2,"groups":1,"bias":True,"padding_mode":"replicate"},
                  {"kernel_size":2,"stride":1,"padding":3,"dilation":2,"groups":2,"bias":False,"padding_mode":"circular"},
                  {"kernel_size":3,"stride":1,"padding":3,"dilation":2,"groups":2,"bias":False,"padding_mode":"zeros"},
                  {"kernel_size":4,"stride":2,"padding":2,"dilation":1,"groups":1,"bias":False,"padding_mode":"reflect"},
                  {"kernel_size":2,"stride":3,"padding":3,"dilation":1,"groups":1,"bias":True,"padding_mode":"replicate"},
                  {"kernel_size":2,"stride":1,"padding":3,"dilation":2,"groups":4,"bias":False,"padding_mode":"circular"}
                 ]
        for N in Ns:
            for Cin in Cins:
                for Cout in Couts:
                    for (h,w) in hws:
                        for kwargs in kwargss:
                            if Cin%kwargs["groups"]==0 and Cout%kwargs["groups"]==0:
                                test_help(N, Cin, Cout, h, w, kwargs)


    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_fp_dtypes(include_half=True, include_bfloat16=False)))
    @with_cudnn_off_helper
    def test_lazyconv3d(self, device, dtype):
        

        def test_help(N, Cin, Cout, d, h, w, kwargs):
            shapes=(N, Cin, d, h, w)
            input = make_tensor(shapes, dtype=dtype, device='cpu',low=0, high=1)
            input_list = [input]
            model = nn.LazyConv3d(Cout, **kwargs)
            tol = 1e-2

            runtestapi(func=model, fwd_input_list=input_list, enable_backward=True, 
                    type_dict={torch.float:{torch.float16, torch.bfloat16}}, fwd_tol=tol, bwd_tol=tol,
                    low=0, high=1, compare_type="torch")

        Ns = [1, 5]
        Cins = [4, 9]
        Couts = [8]
        dhws = [(7, 8,16)]
        kwargss = [{"kernel_size":2,"stride":1,"padding":0,"dilation":1,"groups":1,"bias":True,"padding_mode":"zeros"},
                  {"kernel_size":2,"stride":2,"padding":2,"dilation":1,"groups":2,"bias":False,"padding_mode":"reflect"},
                  {"kernel_size":3,"stride":2,"padding":4,"dilation":2,"groups":1,"bias":True,"padding_mode":"replicate"},
                  {"kernel_size":2,"stride":1,"padding":3,"dilation":2,"groups":2,"bias":False,"padding_mode":"circular"},
                  {"kernel_size":3,"stride":2,"padding":3,"dilation":2,"groups":1,"bias":False,"padding_mode":"zeros"},
                  {"kernel_size":4,"stride":2,"padding":2,"dilation":2,"groups":2,"bias":True,"padding_mode":"reflect"},
                  {"kernel_size":2,"stride":3,"padding":3,"dilation":1,"groups":1,"bias":True,"padding_mode":"replicate"},
                  {"kernel_size":2,"stride":1,"padding":1,"dilation":2,"groups":4,"bias":False,"padding_mode":"circular"}
                 ]
        for N in Ns:
            for Cin in Cins:
                for Cout in Couts:
                    for (d,h,w) in dhws:
                        for kwargs in kwargss:
                            if Cin%kwargs["groups"]==0 and Cout%kwargs["groups"]==0:
                                test_help(N, Cin, Cout, d, h, w, kwargs)


    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_fp_dtypes(include_half=True, include_bfloat16=False)))
    @with_cudnn_off_helper
    def test_lazyconvtranspose1d(self, device, dtype):
        

        def test_help(N, Cin, Cout, L, kwargs):
            shapes=(N, Cin, L)
            input = make_tensor(shapes, dtype=dtype, device='cpu',low=0, high=1)
            input_list = [input]
            model = nn.LazyConvTranspose1d(Cout, **kwargs)
            tol = 1e-2

            runtestapi(func=model, fwd_input_list=input_list, enable_backward=True, 
                    type_dict={torch.float:{torch.float16, torch.bfloat16}}, fwd_tol=tol, bwd_tol=tol,
                    low=0, high=1, compare_type="torch")

        Ns = [1]
        Cins = [4]
        Couts = [4, 9]
        Ls = [11, 12]
        kwargss = [{"kernel_size":2,"stride":1,"padding":0,"dilation":1,"groups":1,"bias":True,"padding_mode":"zeros"},
                  {"kernel_size":2,"stride":2,"padding":2,"dilation":1,"groups":2,"bias":False,"padding_mode":"zeros"},
                  {"kernel_size":2,"stride":2,"padding":2,"dilation":2,"groups":1,"bias":True,"padding_mode":"zeros"},
                  {"kernel_size":2,"stride":1,"padding":3,"dilation":2,"groups":2,"bias":False,"padding_mode":"zeros"},
                  {"kernel_size":3,"stride":1,"padding":3,"dilation":2,"groups":2,"bias":False,"padding_mode":"zeros"},
                  {"kernel_size":2,"stride":2,"padding":2,"dilation":1,"groups":1,"bias":False,"padding_mode":"zeros"},
                  {"kernel_size":2,"stride":1,"padding":3,"dilation":1,"groups":1,"bias":True,"padding_mode":"zeros"},
                  {"kernel_size":2,"stride":1,"padding":3,"dilation":2,"groups":4,"bias":False,"padding_mode":"zeros"}
                 ]
        for N in Ns:
            for Cin in Cins:
                for Cout in Couts:
                    for L in Ls:
                        for kwargs in kwargss:
                            if Cin%kwargs["groups"]==0 and Cout%kwargs["groups"]==0:
                                test_help(N, Cin, Cout, L, kwargs)


    @onlyCUDA   
    @dtypesIfCUDA(*set(get_all_fp_dtypes(include_half=True, include_bfloat16=False)))
    @with_cudnn_off_helper
    def test_lazyconvtranspose2d(self, device, dtype):
        

        def test_help(N, Cin, Cout, h, w, kwargs):
            shapes=(N, Cin, h, w)
            input = make_tensor(shapes, dtype=dtype, device='cpu',low=0, high=1)
            input_list = [input]
            model = nn.LazyConvTranspose2d(Cout, **kwargs)
            tol = 1e-2

            runtestapi(func=model, fwd_input_list=input_list, enable_backward=True, 
                    type_dict={torch.float:{torch.float16, torch.bfloat16}}, fwd_tol=tol, bwd_tol=tol,
                    low=0, high=1, compare_type="torch")

        Ns = [7, 16]
        Cins = [9]
        Couts = [4, 7]
        hws = [(11, 12)]
        kwargss = [{"kernel_size":2,"stride":1,"padding":0,"dilation":1,"groups":1,"bias":True,"padding_mode":"zeros"},
                  {"kernel_size":2,"stride":2,"padding":2,"dilation":1,"groups":2,"bias":False,"padding_mode":"zeros"},
                  {"kernel_size":3,"stride":2,"padding":4,"dilation":2,"groups":1,"bias":True,"padding_mode":"zeros"},
                  {"kernel_size":2,"stride":1,"padding":3,"dilation":2,"groups":2,"bias":False,"padding_mode":"zeros"},
                  {"kernel_size":3,"stride":1,"padding":3,"dilation":2,"groups":2,"bias":False,"padding_mode":"zeros"},
                  {"kernel_size":4,"stride":2,"padding":2,"dilation":1,"groups":1,"bias":False,"padding_mode":"zeros"},
                  {"kernel_size":2,"stride":3,"padding":3,"dilation":1,"groups":1,"bias":True,"padding_mode":"zeros"},
                  {"kernel_size":2,"stride":1,"padding":3,"dilation":2,"groups":4,"bias":False,"padding_mode":"zeros"}
                 ]
        for N in Ns:
            for Cin in Cins:
                for Cout in Couts:
                    for (h,w) in hws:
                        for kwargs in kwargss:
                            if Cin%kwargs["groups"]==0 and Cout%kwargs["groups"]==0:
                                test_help(N, Cin, Cout, h, w, kwargs)


    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_fp_dtypes(include_half=True, include_bfloat16=False)))
    @with_cudnn_off_helper
    def test_lazyconvtranspose3d(self, device, dtype):
        

        def test_help(N, Cin, Cout, d, h, w, kwargs):
            shapes=(N, Cin, d, h, w)
            input = make_tensor(shapes, dtype=dtype, device='cpu',low=0, high=1)
            input_list = [input]
            model = nn.LazyConvTranspose3d(Cout, **kwargs)
            tol = 1e-2

            runtestapi(func=model, fwd_input_list=input_list, enable_backward=True, 
                    type_dict={torch.float:{torch.float16, torch.bfloat16}}, fwd_tol=tol, bwd_tol=tol,
                    low=0, high=1, compare_type="torch")

        Ns = [5, 7]
        Cins = [4]
        Couts = [4, 7]
        dhws = [(7, 8,16)]
        kwargss = [{"kernel_size":2,"stride":1,"padding":0,"dilation":1,"groups":1,"bias":True,"padding_mode":"zeros"},
                  {"kernel_size":2,"stride":2,"padding":2,"dilation":1,"groups":2,"bias":False,"padding_mode":"zeros"},
                  {"kernel_size":3,"stride":2,"padding":4,"dilation":2,"groups":1,"bias":True,"padding_mode":"zeros"},
                  {"kernel_size":2,"stride":1,"padding":3,"dilation":2,"groups":2,"bias":False,"padding_mode":"zeros"},
                  {"kernel_size":3,"stride":2,"padding":3,"dilation":2,"groups":1,"bias":False,"padding_mode":"zeros"},
                  {"kernel_size":4,"stride":2,"padding":2,"dilation":2,"groups":2,"bias":True,"padding_mode":"zeros"},
                  {"kernel_size":2,"stride":3,"padding":3,"dilation":1,"groups":1,"bias":True,"padding_mode":"zeros"},
                  {"kernel_size":2,"stride":1,"padding":1,"dilation":2,"groups":4,"bias":False,"padding_mode":"zeros"}
                 ]
        for N in Ns:
            for Cin in Cins:
                for Cout in Couts:
                    for (d,h,w) in dhws:
                        for kwargs in kwargss:
                            if Cin%kwargs["groups"]==0 and Cout%kwargs["groups"]==0:
                                test_help(N, Cin, Cout, d, h, w, kwargs)


    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_fp_dtypes(include_half=True, include_bfloat16=False)))
    @with_cudnn_off_helper
    def test_unfold(self, device, dtype):

        def test_help(N, C, H, W, kwargs):
            shapes=(N, C, H, W)
            input = make_tensor(shapes, dtype=dtype, device='cpu',low=0, high=1)
            input_list = [input]
            model = nn.Unfold(**kwargs)
            tol = 1e-2

            runtestapi(func=model, fwd_input_list=input_list, enable_backward=True, 
                    type_dict={torch.float:{torch.float16, torch.bfloat16}}, fwd_tol=tol, bwd_tol=tol,
                    low=0, high=1, compare_type="torch")

        Ns = [3,6]
        Cs = [9,]
        Hs = [11,14]
        Ws = [17]
        kwargss = [{"kernel_size":2,"stride":1,"padding":0,"dilation":1},
                  {"kernel_size":2,"stride":2,"padding":2,"dilation":1},
                  {"kernel_size":3,"stride":2,"padding":4,"dilation":2},
                  {"kernel_size":2,"stride":1,"padding":3,"dilation":2},
                  {"kernel_size":3,"stride":2,"padding":3,"dilation":2},
                  {"kernel_size":4,"stride":2,"padding":2,"dilation":2},
                  {"kernel_size":2,"stride":3,"padding":3,"dilation":1},
                  {"kernel_size":2,"stride":1,"padding":1,"dilation":2}
                 ]

        for N in Ns:
            for C in Cs:
                for H in Hs:
                    for W in Ws:
                        for kwargs in kwargss:
                            test_help(N,C,H,W,kwargs)


    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_fp_dtypes(include_half=True, include_bfloat16=False)))
    @with_cudnn_off_helper
    def test_elu(self, device, dtype):
        def test_help(shape,alpha):
            input = make_tensor(shape, dtype=dtype, device='cpu',low=0, high=1)
            input_list = [input]
            model = nn.ELU(alpha)
            tol = 1e-2

            runtestapi(func=model, fwd_input_list=input_list, enable_backward=True, 
                    type_dict={torch.float:{torch.float16, torch.bfloat16}}, fwd_tol=tol, bwd_tol=tol,
                    low=0, high=1, compare_type="torch")

        shapes = [(13,),(22,24),(5,6,7,8,9,10)]
        alphas = [1.0, 1.1, 2.0, 2.1]
        for shape in shapes:
            for alpha in alphas:
                test_help(shape,alpha)


    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_fp_dtypes(include_half=False, include_bfloat16=False)))
    @with_cudnn_off_helper
    def test_hardshrink(self, device, dtype):
        def test_help(shape,lamb):
            input = make_tensor(shape, dtype=dtype, device='cpu',low=0, high=1)
            input_list = [input]
            model = nn.Hardshrink(lamb)
            tol = 1e-2

            runtestapi(func=model, fwd_input_list=input_list, enable_backward=True, 
                    type_dict={torch.float:{torch.float16, torch.bfloat16}}, fwd_tol=tol, bwd_tol=tol,
                    low=0, high=1, compare_type="torch")
        
        shapes = [(13,),(22,24),(5,6,7,8,9,10)]
        lambs = [-0.1, -0.2, 0.2, 0.3]
        for shape in shapes:
            for lamb in lambs:
                test_help(shape,lamb)

        
    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_fp_dtypes(include_half=True, include_bfloat16=False)))
    @with_cudnn_off_helper
    def test_hardsigmoid(self, device, dtype):
        def test_help(shape):
            input = make_tensor(shape, dtype=dtype, device='cpu',low=0, high=1)
            input_list = [input]
            model = nn.Hardsigmoid()
            tol = 1e-2

            runtestapi(func=model, fwd_input_list=input_list, enable_backward=True, 
                    type_dict={torch.float:{torch.float16, torch.bfloat16}}, fwd_tol=tol, bwd_tol=tol,
                    low=0, high=1, compare_type="torch")
        
        shapes = [(13,),(22,24),(5,6,7),(11,12,12,14),(5,6,7,8,9,10)]
        for shape in shapes:
            test_help(shape)


    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_fp_dtypes(include_half=True, include_bfloat16=False)))
    @with_cudnn_off_helper
    def test_hardtanh(self, device, dtype):
        def test_help(shape,min_v,max_v):
            input = make_tensor(shape, dtype=dtype, device='cpu',low=0, high=1)
            input_list = [input]
            model = nn.Hardtanh(min_v,max_v)
            tol = 1e-2

            runtestapi(func=model, fwd_input_list=input_list, enable_backward=True, 
                    type_dict={torch.float:{torch.float16, torch.bfloat16}}, fwd_tol=tol, bwd_tol=tol,
                    low=0, high=1, compare_type="torch")

        shapes = [(13,),(5,6,7),(5,6,7,8,9,10)]
        min_maxs = [(-1.1,1.1),(-2.2,2.2),(-3.3,3.3)]
        for shape in shapes:
            for (min_v,max_v) in min_maxs:
                test_help(shape,min_v,max_v)


    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_fp_dtypes(include_half=True, include_bfloat16=False)))
    @with_cudnn_off_helper
    def test_hardswish(self, device, dtype):
        def test_help(shape):
            input = make_tensor(shape, dtype=dtype, device='cpu',low=0, high=1)
            input_list = [input]
            model = nn.Hardswish()
            tol = 1e-2

            runtestapi(func=model, fwd_input_list=input_list, enable_backward=True, 
                    type_dict={torch.float:{torch.float16, torch.bfloat16}}, fwd_tol=tol, bwd_tol=tol,
                    low=0, high=1, compare_type="torch")

        shapes = [(13,),(22,24),(5,6,7),(11,12,12,14),(5,6,7,8,9,10)]
        for shape in shapes:
            test_help(shape)


    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_fp_dtypes(include_half=True, include_bfloat16=False)))
    @with_cudnn_off_helper
    def test_leakyrelu(self, device, dtype):
        def test_help(shape,slop):
            input = make_tensor(shape, dtype=dtype, device='cpu',low=0, high=1)
            input_list = [input]
            model = nn.LeakyReLU(slop)
            tol = 1e-2

            runtestapi(func=model, fwd_input_list=input_list, enable_backward=True, 
                    type_dict={torch.float:{torch.float16, torch.bfloat16}}, fwd_tol=tol, bwd_tol=tol,
                    low=0, high=1, compare_type="torch")
        
        shapes = [(13,),(11,12,12,14),(5,6,7,8,9,10)]
        slops = [-1., 2, -2]
        for shape in shapes:
            for slop in slops:
                test_help(shape,slop)    
    

    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_fp_dtypes(include_half=True, include_bfloat16=False)))
    @with_cudnn_off_helper
    def test_logsigmoid(self, device, dtype):
        def test_help(shape):
            input = make_tensor(shape, dtype=dtype, device='cpu',low=0, high=1)
            input_list = [input]
            model = nn.LogSigmoid()
            tol = 1e-2

            runtestapi(func=model, fwd_input_list=input_list, enable_backward=True, 
                    type_dict={torch.float:{torch.float16, torch.bfloat16}}, fwd_tol=tol, bwd_tol=tol,
                    low=0, high=1, compare_type="torch")
        
        shapes = [(13,),(22,24),(5,6,7),(11,12,12,14),(5,6,7,8,9,10)]
        for shape in shapes:
            test_help(shape)   


    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_fp_dtypes(include_half=True, include_bfloat16=False)))
    @with_cudnn_off_helper
    def test_prelu(self, device, dtype):
        def test_help(shape,alp,init):
            input = make_tensor(shape, dtype=dtype, device='cpu',low=0, high=1)
            input_list = [input]
            model = nn.PReLU(alp,init)
            tol = 1e-2

            runtestapi(func=model, fwd_input_list=input_list, enable_backward=True, 
                    type_dict={torch.float:{torch.float16, torch.bfloat16}}, fwd_tol=tol, bwd_tol=tol,
                    low=0, high=1, compare_type="torch")
        
        shapes = [(13,11),(11,12,12,14),(5,6,7,8,9,10)]
        inits = [0.45,0.55]
        for shape in shapes:
            for init in inits:
                test_help(shape,1,init)  
                test_help(shape,shape[1],init)


    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_fp_dtypes(include_half=False, include_bfloat16=False)))
    @with_cudnn_off_helper
    def test_embedding(self, device, dtype):

        def test_help(shape,kwarg):
            rol = 1e-5
            num_emb = kwarg["num_embeddings"]
            input_c = torch.randint(num_emb, size=shape)
            model_c = nn.Embedding(**kwarg).to(dtype=dtype)
            out_c = model_c(input_c)

            backward_input_c = torch.rand(out_c.shape).to(dtype)
            out_c.backward(backward_input_c)

            backward_input_g = backward_input_c.detach().clone().cuda()
            input_g = input_c.detach().clone().cuda()
            model_g = copy.deepcopy(model_c).cuda()
            out_g = model_g(input_g)
            out_g.backward(backward_input_g)

            assert torch.allclose(out_g.cpu(), out_c, rtol=rol, atol=rol)
            assert torch.allclose(model_g.weight.grad.cpu(), model_c.weight.grad, rtol=rol, atol=rol)


        shapes = [(2,3),(5,6,7)]
        kwargs = [{"num_embeddings":10,"embedding_dim":15,"padding_idx":0,"max_norm":0.5, "norm_type":2.0, "scale_grad_by_freq":False},
                    {"num_embeddings":20,"embedding_dim":25,"padding_idx":1,"max_norm":0.6, "norm_type":3.0, "scale_grad_by_freq":True},
                    {"num_embeddings":30,"embedding_dim":35,"padding_idx":2,"max_norm":0.7, "norm_type":1.0, "scale_grad_by_freq":False},
                    {"num_embeddings":40,"embedding_dim":45,"padding_idx":None,"max_norm":0.8, "norm_type":4.0, "scale_grad_by_freq":True},
                    {"num_embeddings":50,"embedding_dim":55,"padding_idx":3,"max_norm":0.4, "norm_type":2.0, "scale_grad_by_freq":False},
                    {"num_embeddings":60,"embedding_dim":65,"padding_idx":4,"max_norm":0.3, "norm_type":3.0, "scale_grad_by_freq":True},
                    {"num_embeddings":70,"embedding_dim":75,"padding_idx":5,"max_norm":0.3, "norm_type":4.0, "scale_grad_by_freq":False},
                    {"num_embeddings":80,"embedding_dim":85,"padding_idx":6,"max_norm":1, "norm_type":1.0, "scale_grad_by_freq":True},     
                 ]

        for shape in shapes:
            for kwarg in kwargs:
                test_help(shape,kwarg)

class TestNNSegment1(unittest.TestCase):

    ##### nn.BatchNorm
    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_fp_dtypes(include_half=False, include_bfloat16=False)))
    @with_cudnn_off_helper
    def test_nn_BatchNorm(self, device, dtype):
        
        def help(shape_list, op, is_lazy=False):
            forward_inputs = gendata(shape_list, type_list=[dtype], rand_algo="rand")
            eps_list = [1e-5, 5e-5]
            momentum_list = [0.1, 0.2]
            affine_list = [True, False]
            track_running_stats_list = [True, False]
            for input, eps, momentum, affine, track_running_stats in product(forward_inputs, eps_list, momentum_list, affine_list, track_running_stats_list):
                if is_lazy:
                    func = op(eps, momentum, affine, track_running_stats)
                else:
                    num_features = input.shape[1]
                    func = op(num_features, eps, momentum, affine, track_running_stats)
                runtestapi(func=func, fwd_input_list=[input], enable_backward=True)

        help([(4, 512), (4, 15, 256)], nn.BatchNorm1d)
        help([(4, 32, 32, 32), (4, 128, 32, 32)], nn.BatchNorm2d)
        help([(4, 128, 8, 8, 8)], nn.BatchNorm3d)

        help([(4, 512), (4, 15, 512)], nn.LazyBatchNorm1d, True)
        help([(4, 32, 32, 32), (4, 256, 32, 32)], nn.LazyBatchNorm2d, True)
        help([(4, 128, 8, 8, 8)], nn.LazyBatchNorm3d, True)

    ##### nn.GroupNorm
    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_fp_dtypes(include_half=False, include_bfloat16=False)))
    @with_cudnn_off_helper
    def test_nn_GroupNorm(self, device, dtype):
        shape_list = [(4, 256), (4, 512, 16), (4, 256, 16, 16), (4, 128, 8, 8, 8)]
        forward_inputs = gendata(shape_list, type_list=[dtype], rand_algo="randn")
        eps_list = [1e-5, 5e-5]
        affine_list = [True, False]
        for input, eps, affine in product(forward_inputs, eps_list, affine_list):
            num_channels = input.shape[1]
            func = nn.GroupNorm(num_channels//2, num_channels, eps, affine)
            runtestapi(func=func, fwd_input_list=[input], enable_backward=True, bwd_tol=1e-3)

    ##### nn.InstanceNorm
    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_fp_dtypes(include_half=False, include_bfloat16=False)))
    @with_cudnn_off_helper
    def test_nn_InstanceNorm(self, device, dtype):
        
        def help(shape_list, norm_level, is_lazy=False):
            forward_inputs = gendata(shape_list, type_list=[dtype], rand_algo="rand")
            eps_list = [1e-5, 5e-5]
            momentum_list = [0.1, 0.2]
            affine_list = [True, False]
            track_running_stats_list = [True, False]
            if is_lazy:
                op_name = "LazyInstanceNorm{}d".format(norm_level)
            else:
                op_name = "InstanceNorm{}d".format(norm_level)
            op = getattr(nn, op_name)
            for input, eps, momentum, affine, track_running_stats in product(forward_inputs, eps_list, momentum_list, affine_list, track_running_stats_list):
                if is_lazy:
                    func = op(eps, momentum, affine, track_running_stats)
                else:
                    num_features = input.shape[1]
                    func = op(num_features, eps, momentum, affine, track_running_stats)
                runtestapi(func=func, fwd_input_list=[input], enable_backward=True)

        help([(4, 33, 256)], 1)
        help([(4, 32, 32, 32)], 2)
        help([(4, 128, 8, 8, 8)], 3)

        help([(4, 33, 256)], 1, True)
        help([(4, 32, 32, 32)], 2, True)
        help([(4, 128, 8, 8, 8)], 3, True)

    ##### nn.LayerNorm
    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_fp_dtypes(include_half=False, include_bfloat16=False)))
    @with_cudnn_off_helper
    def test_nn_LayerNorm(self, device, dtype):
        shape_list = [(4, 128, 100), (4, 128, 32, 32)]
        forward_inputs = gendata(shape_list, type_list=[dtype], rand_algo="rand")
        eps_list = [1e-5, 5e-5]
        elementwise_affine_list = [True, False]
        for input, eps, affine in product(forward_inputs, eps_list, elementwise_affine_list):
            shape = input.shape
            if len(shape) == 3:
                func = nn.LayerNorm(shape[-1], eps, affine)
            else:
                func = nn.LayerNorm(shape[1:], eps, affine)
            runtestapi(func=func, fwd_input_list=[input], enable_backward=True)

    ##### nn.LocalResponseNorm
    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_fp_dtypes(include_half=False, include_bfloat16=False)))
    @with_cudnn_off_helper
    def test_nn_LocalResponseNorm(self, device, dtype):
        shape_list = [(4, 128, 100), (4, 128, 32, 32)]
        forward_inputs = gendata(shape_list, type_list=[dtype], rand_algo="rand")
        size_list = [2, 8]
        alpha_list = [1e-4, 2e-4]
        beta_list = [0.75, 0.8]
        k_list = [1., 0.9]
        for input, size, alpha, beta, k in product(forward_inputs, size_list, alpha_list, beta_list, k_list):
            func = nn.LocalResponseNorm(size, alpha, beta, k)
            runtestapi(func=func, fwd_input_list=[input], enable_backward=True)

class TestNNSegment3(TestCase):

    @onlyCUDA
    @dtypesIfCUDA(*set([torch.float, torch.double, torch.half, torch.complex64, torch.complex128]))
    @with_tf32_off_helper
    def test_nn_ReflectionPad1d(self, device, dtype):
        torch.manual_seed(0)
        for contig in [True, False]:
            # ReflectionPad1d no-empty
            for mod, inp in [
                    (torch.nn.ReflectionPad1d(16), torch.rand(5, 10, 64, device='cuda', dtype=dtype, requires_grad=True)),
                    (torch.nn.ReflectionPad1d(4), torch.rand(5, 16, 2048, device='cuda', dtype=dtype, requires_grad=True)),
                    (torch.nn.ReflectionPad1d((16, 5)), torch.rand(5, 10, 64, device='cuda', dtype=dtype, requires_grad=True)),
                    (torch.nn.ReflectionPad1d((8, 16)), torch.rand(5, 16, 2048, device='cuda', dtype=dtype, requires_grad=True)),
                    ]:
                if (dtype == (torch.half or torch.bfloat16)):
                    inp_ref = inp.detach().to(torch.float32).cpu().requires_grad_()
                else:
                    inp_ref = inp.detach().cpu().requires_grad_()
                if contig == False:
                    inp = make_noncontig(inp).requires_grad_()
                    inp_ref = make_noncontig(inp_ref).requires_grad_()
                out = mod(inp)
                out_ref = mod(inp_ref)
                if (dtype != (torch.half or torch.bfloat16)):
                    self.assertEqual(out, out_ref)
                g = torch.rand_like(out)
                g_ref = g.cpu()
                out.backward(g)
                out_ref.backward(g_ref)
                if (dtype != (torch.half or torch.bfloat16)):
                    self.assertEqual(inp.retain_grad(), inp_ref.retain_grad())
            # ReflectionPad1d empty
            for mod, inp in [
                    (torch.nn.ReflectionPad1d(16), torch.rand(0, 10, 64, device='cuda', dtype=dtype, requires_grad=True)),
                    (torch.nn.ReflectionPad1d(4), torch.rand(0, 16, 2048, device='cuda', dtype=dtype, requires_grad=True)),
                    (torch.nn.ReflectionPad1d((16, 5)), torch.rand(0, 10, 64, device='cuda', dtype=dtype, requires_grad=True)),
                    (torch.nn.ReflectionPad1d((8, 16)), torch.rand(0, 16, 2048, device='cuda', dtype=dtype, requires_grad=True))
                    ]:
                inp = inp.requires_grad_(True)
                if contig == False:
                    inp = make_noncontig(inp)
                out = mod(inp)
                g = torch.rand_like(out)
                out.backward(g)
                for p in mod.parameters():
                    if p.requires_grad:
                        self.assertEqual(p.retain_grad(), torch.zeros_like(p.retain_grad()))
                if inp.retain_grad() == None or inp_ref.retain_grad() == None:
                    self.assertEqual(inp.retain_grad(), inp_ref.retain_grad())
                else:
                    self.assertEqual(inp.retain_grad(), torch.zeros_like(inp))


    @onlyCUDA
    @dtypesIfCUDA(*set([torch.float, torch.double, torch.half, torch.complex64, torch.complex128]))
    @with_tf32_off_helper
    def test_nn_ReflectionPad2d(self, device, dtype):
        torch.manual_seed(0)
        for contig in [True, False]:
            # ReflectionPad2d no-empty
            for mod, inp in [
                    (torch.nn.ReflectionPad2d(3), torch.rand(3, 5, 10, 2048, device='cuda', dtype=dtype, requires_grad=True)),
                    (torch.nn.ReflectionPad2d(16), torch.rand(5, 10, 64, 64, device='cuda', dtype=dtype, requires_grad=True)),
                    (torch.nn.ReflectionPad2d((16, 5, 0, 2)), torch.rand(5, 10, 16, 32, device='cuda', dtype=dtype, requires_grad=True)),
                    (torch.nn.ReflectionPad2d((8, 4, 1, 1)), torch.rand(16, 8, 8, 2048, device='cuda', dtype=dtype, requires_grad=True))
                    ]:
                if (dtype == (torch.half or torch.bfloat16)):
                    inp_ref = inp.detach().to(torch.float32).cpu().requires_grad_()
                else:
                    inp_ref = inp.detach().cpu().requires_grad_()
                if contig == False:
                    inp = make_noncontig(inp)
                    inp_ref = make_noncontig(inp_ref)
                out = mod(inp)
                out_ref = mod(inp_ref)
                if (dtype != (torch.half or torch.bfloat16)):
                    self.assertEqual(out, out_ref)
                g = torch.rand_like(out)
                g_ref = g.cpu()
                out.backward(g)
                out_ref.backward(g_ref)
                if (dtype != (torch.half or torch.bfloat16)):
                    self.assertEqual(inp.retain_grad(), inp_ref.retain_grad())
            # ReflectionPad2d empty
            for mod, inp in [
                    (torch.nn.ReflectionPad2d(3), torch.rand(0, 5, 10, 65736, device='cuda', dtype=dtype)),
                    (torch.nn.ReflectionPad2d(15), torch.rand(0, 10, 64, 64, device='cuda', dtype=dtype)),
                    (torch.nn.ReflectionPad2d((3, 2, 3, 0)), torch.rand(0, 5, 10, 65736, device='cuda', dtype=dtype)),
                    (torch.nn.ReflectionPad2d((32, 5, 0, 2)), torch.rand(0, 3, 64, 128, device='cuda', dtype=dtype)),
                    (torch.nn.ReflectionPad2d((16, 64, 1, 1)), torch.rand(0, 16, 8, 2048, device='cuda', dtype=dtype))
                    ]:
                inp = inp.requires_grad_(True)
                if contig == False:
                    inp = make_noncontig(inp)
                out = mod(inp)
                g = torch.rand_like(out)
                out.backward(g)
                for p in mod.parameters():
                    if p.requires_grad:
                        self.assertEqual(p.retain_grad(), torch.zeros_like(p.retain_grad()))
                if inp.retain_grad() == None or inp_ref.retain_grad() == None:
                    self.assertEqual(inp.retain_grad(), inp_ref.retain_grad())
                else:
                    self.assertEqual(inp.retain_grad(), torch.zeros_like(inp))


    @onlyCUDA
    @dtypesIfCUDA(*set([torch.float, torch.double, torch.half, torch.complex64, torch.complex128]))
    @with_tf32_off_helper
    def test_nn_ReflectionPad3d(self, device, dtype):
        torch.manual_seed(0)
        for contig in [True, False]:
            # ReflectionPad3d no-empty
            for mod, inp in [
                    (torch.nn.ReflectionPad3d(2), torch.rand(2, 2, 3, 3, 1024, device='cuda', dtype=dtype, requires_grad=True)),
                    (torch.nn.ReflectionPad3d(2), torch.rand(2, 1, 4, 16, 4, device='cuda', dtype=dtype, requires_grad=True)),
                    (torch.nn.ReflectionPad3d((0, 1, 2, 3, 2, 2)), torch.rand(3, 2, 5, 4, 1024, device='cuda', dtype=dtype, requires_grad=True)),
                    (torch.nn.ReflectionPad3d((2, 4, 5, 2, 2, 3)), torch.rand(5, 3, 64, 6, 128, device='cuda', dtype=dtype, requires_grad=True)),
                    (torch.nn.ReflectionPad3d((1, 1, 1, 1, 1, 1)), torch.rand(64, 4, 4, 8, 1024, device='cuda', dtype=dtype, requires_grad=True))
                    ]:
                if (dtype == (torch.half or torch.bfloat16)):
                    inp_ref = inp.detach().to(torch.float32).cpu().requires_grad_()
                else:
                    inp_ref = inp.detach().cpu().requires_grad_()
                if contig == False:
                    inp = make_noncontig(inp)
                    inp_ref = make_noncontig(inp_ref)
                out = mod(inp)
                out_ref = mod(inp_ref)
                if (dtype != (torch.half or torch.bfloat16)):
                    self.assertEqual(out, out_ref)
                g = torch.rand_like(out)
                g_ref = g.cpu()
                out.backward(g)
                out_ref.backward(g_ref)
                if (dtype != (torch.half or torch.bfloat16)):
                    self.assertEqual(inp.retain_grad(), inp_ref.retain_grad())
            # ReflectionPad3d empty
            for mod, inp in [
                    (torch.nn.ReflectionPad3d(2), torch.rand(0, 2, 3, 3, 1024, device='cuda', dtype=dtype, requires_grad=True)),
                    (torch.nn.ReflectionPad3d(2), torch.rand(0, 1, 4, 16, 4, device='cuda', dtype=dtype, requires_grad=True)),
                    (torch.nn.ReflectionPad3d((0, 1, 2, 3, 2, 2)), torch.rand(0, 2, 5, 4, 1024, device='cuda', dtype=dtype, requires_grad=True)),
                    (torch.nn.ReflectionPad3d((2, 4, 5, 2, 2, 3)), torch.rand(0, 3, 64, 6, 128, device='cuda', dtype=dtype, requires_grad=True)),
                    (torch.nn.ReflectionPad3d((1, 1, 1, 1, 1, 1)), torch.rand(0, 4, 4, 8, 1024, device='cuda', dtype=dtype, requires_grad=True))
                    ]:
                inp = inp.requires_grad_(True)
                if contig == False:
                    inp = make_noncontig(inp)
                out = mod(inp)
                g = torch.rand_like(out)
                out.backward(g)
                for p in mod.parameters():
                    if p.requires_grad:
                        self.assertEqual(p.retain_grad(), torch.zeros_like(p.retain_grad()))
                if inp.retain_grad() == None or inp_ref.retain_grad() == None:
                    self.assertEqual(inp.retain_grad(), inp_ref.retain_grad())
                else:
                    self.assertEqual(inp.retain_grad(), torch.zeros_like(inp))


    @onlyCUDA
    @dtypesIfCUDA(*set([torch.float, torch.double, torch.half, torch.complex64, torch.complex128]))
    @with_tf32_off_helper
    def test_nn_ReplicationPad1d(self, device, dtype):
        torch.manual_seed(0)
        for contig in [True, False]:
            # ReplicationPad1d no-empty
            for mod, inp in [
                    (torch.nn.ReplicationPad1d(16), torch.rand(5, 10, 64, device='cuda', dtype=dtype, requires_grad=True)),
                    (torch.nn.ReplicationPad1d(4), torch.rand(5, 16, 2048, device='cuda', dtype=dtype, requires_grad=True)),
                    (torch.nn.ReplicationPad1d((16, 5)), torch.rand(5, 10, 64, device='cuda', dtype=dtype, requires_grad=True)),
                    (torch.nn.ReplicationPad1d((8, 16)), torch.rand(5, 16, 2048, device='cuda', dtype=dtype, requires_grad=True))
                    ]:
                if (dtype == (torch.half or torch.bfloat16)):
                    inp_ref = inp.detach().to(torch.float32).cpu().requires_grad_()
                else:
                    inp_ref = inp.detach().cpu().requires_grad_()
                if contig == False:
                    inp = make_noncontig(inp)
                    inp_ref = make_noncontig(inp_ref)
                out = mod(inp)
                out_ref = mod(inp_ref)
                if (dtype != (torch.half or torch.bfloat16)):
                    self.assertEqual(out, out_ref)
                g = torch.rand_like(out)
                g_ref = g.cpu()
                out.backward(g)
                out_ref.backward(g_ref)
                if (dtype != (torch.half or torch.bfloat16)):
                    self.assertEqual(inp.retain_grad(), inp_ref.retain_grad())
            # ReplicationPad1d empty
            for mod, inp in [
                    (torch.nn.ReplicationPad1d(16), torch.rand(0, 10, 64, device='cuda', dtype=dtype, requires_grad=True)),
                    (torch.nn.ReplicationPad1d(4), torch.rand(0, 16, 2048, device='cuda', dtype=dtype, requires_grad=True)),
                    (torch.nn.ReplicationPad1d((16, 5)), torch.rand(0, 10, 64, device='cuda', dtype=dtype, requires_grad=True)),
                    (torch.nn.ReplicationPad1d((8, 16)), torch.rand(0, 16, 2048, device='cuda', dtype=dtype, requires_grad=True))
                    ]:
                inp = inp.requires_grad_(True)
                if contig == False:
                    inp = make_noncontig(inp)
                out = mod(inp)
                g = torch.rand_like(out)
                out.backward(g)
                for p in mod.parameters():
                    if p.requires_grad:
                        self.assertEqual(p.retain_grad(), torch.zeros_like(p.retain_grad()))
                if inp.retain_grad() == None or inp_ref.retain_grad() == None:
                    self.assertEqual(inp.retain_grad(), inp_ref.retain_grad())
                else:
                    self.assertEqual(inp.retain_grad(), torch.zeros_like(inp))


    @onlyCUDA
    @dtypesIfCUDA(*set([torch.float, torch.double, torch.half, torch.complex64, torch.complex128]))
    @with_tf32_off_helper
    def test_nn_ReplicationPad2d(self, device, dtype):
        torch.manual_seed(0)
        for contig in [True, False]:
            # ReplicationPad2d no-empty
            for mod, inp in [
                    (torch.nn.ReplicationPad2d(3), torch.rand(3, 5, 10, 2048, device='cuda', dtype=dtype, requires_grad=True)),
                    (torch.nn.ReplicationPad2d(16), torch.rand(5, 10, 64, 64, device='cuda', dtype=dtype, requires_grad=True)),
                    (torch.nn.ReplicationPad2d((4, 5, 0, 2)), torch.rand(5, 10, 16, 32, device='cuda', dtype=dtype, requires_grad=True)),
                    (torch.nn.ReplicationPad2d((16, 64, 1, 1)), torch.rand(2, 16, 8, 2048, device='cuda', dtype=dtype, requires_grad=True))
                    ]:
                if (dtype == (torch.half or torch.bfloat16)):
                    inp_ref = inp.detach().to(torch.float32).cpu().requires_grad_()
                else:
                    inp_ref = inp.detach().cpu().requires_grad_()
                if contig == False:
                    inp = make_noncontig(inp)
                    inp_ref = make_noncontig(inp_ref)
                out = mod(inp)
                out_ref = mod(inp_ref)
                if (dtype != (torch.half or torch.bfloat16)):
                    self.assertEqual(out, out_ref)
                g = torch.rand_like(out)
                g_ref = g.cpu()
                out.backward(g)
                out_ref.backward(g_ref)
                if (dtype != (torch.half or torch.bfloat16)):
                    self.assertEqual(inp.retain_grad(), inp_ref.retain_grad())
            # ReplicationPad2d empty
            for mod, inp in [
                    (torch.nn.ReplicationPad2d(3), torch.rand(0, 5, 10, 65736, device='cuda', dtype=dtype)),
                    (torch.nn.ReplicationPad2d(15), torch.rand(0, 10, 64, 64, device='cuda', dtype=dtype)),
                    (torch.nn.ReplicationPad2d((3, 2, 3, 0)), torch.rand(0, 5, 10, 65736, device='cuda', dtype=dtype)),
                    (torch.nn.ReplicationPad2d((32, 5, 0, 2)), torch.rand(0, 3, 64, 128, device='cuda', dtype=dtype)),
                    (torch.nn.ReplicationPad2d((16, 64, 1, 1)), torch.rand(0, 16, 8, 2048, device='cuda', dtype=dtype))
                    ]:
                inp = inp.requires_grad_(True)
                if contig == False:
                    inp = make_noncontig(inp)
                out = mod(inp)
                g = torch.rand_like(out)
                out.backward(g)
                for p in mod.parameters():
                    if p.requires_grad:
                        self.assertEqual(p.retain_grad(), torch.zeros_like(p.retain_grad()))
                if inp.retain_grad() == None or inp_ref.retain_grad() == None:
                    self.assertEqual(inp.retain_grad(), inp_ref.retain_grad())
                else:
                    self.assertEqual(inp.retain_grad(), torch.zeros_like(inp))


    @onlyCUDA
    @dtypesIfCUDA(*set([torch.float, torch.double, torch.half, torch.complex64, torch.complex128]))
    @with_tf32_off_helper
    def test_nn_ReplicationPad3d(self, device, dtype):
        torch.manual_seed(0)
        for contig in [True, False]:
            # ReplicationPad3d no-empty
            for mod, inp in [
                    (torch.nn.ReplicationPad3d(2), torch.rand(2, 2, 3, 3, 1024, device='cuda', dtype=dtype, requires_grad=True)),
                    (torch.nn.ReplicationPad3d(2), torch.rand(2, 1, 4, 16, 4, device='cuda', dtype=dtype, requires_grad=True)),
                    (torch.nn.ReplicationPad3d((0, 1, 2, 3, 2, 2)), torch.rand(3, 2, 5, 4, 1024, device='cuda', dtype=dtype, requires_grad=True)),
                    (torch.nn.ReplicationPad3d((2, 4, 5, 2, 2, 3)), torch.rand(5, 3, 64, 6, 128, device='cuda', dtype=dtype, requires_grad=True)),
                    (torch.nn.ReplicationPad3d((1, 1, 1, 1, 1, 1)), torch.rand(16, 4, 4, 8, 1024, device='cuda', dtype=dtype, requires_grad=True))
                    ]:
                if (dtype == (torch.half or torch.bfloat16)):
                    inp_ref = inp.detach().to(torch.float32).cpu().requires_grad_()
                else:
                    inp_ref = inp.detach().cpu().requires_grad_()
                if contig == False:
                    inp = make_noncontig(inp)
                    inp_ref = make_noncontig(inp_ref)
                out = mod(inp)
                out_ref = mod(inp_ref)
                if (dtype != (torch.half or torch.bfloat16)):
                    self.assertEqual(out, out_ref)
                g = torch.rand_like(out)
                g_ref = g.cpu()
                out.backward(g)
                out_ref.backward(g_ref)
                if (dtype != (torch.half or torch.bfloat16)):
                    self.assertEqual(inp.retain_grad(), inp_ref.retain_grad())
            # ReplicationPad3d empty
            for mod, inp in [
                    (torch.nn.ReplicationPad3d(2), torch.rand(0, 2, 3, 3, 1024, device='cuda', dtype=dtype, requires_grad=True)),
                    (torch.nn.ReplicationPad3d(2), torch.rand(0, 1, 4, 16, 4, device='cuda', dtype=dtype, requires_grad=True)),
                    (torch.nn.ReplicationPad3d((0, 1, 2, 3, 2, 2)), torch.rand(0, 2, 5, 4, 1024, device='cuda', dtype=dtype, requires_grad=True)),
                    (torch.nn.ReplicationPad3d((2, 4, 5, 2, 2, 3)), torch.rand(0, 3, 64, 6, 128, device='cuda', dtype=dtype, requires_grad=True)),
                    (torch.nn.ReplicationPad3d((1, 1, 1, 1, 1, 1)), torch.rand(0, 4, 4, 8, 1024, device='cuda', dtype=dtype, requires_grad=True))
                    ]:
                inp = inp.requires_grad_(True)
                if contig == False:
                    inp = make_noncontig(inp)
                out = mod(inp)
                g = torch.rand_like(out)
                out.backward(g)
                for p in mod.parameters():
                    if p.requires_grad:
                        self.assertEqual(p.retain_grad(), torch.zeros_like(p.retain_grad()))
                if inp.retain_grad() == None or inp_ref.retain_grad() == None:
                    self.assertEqual(inp.retain_grad(), inp_ref.retain_grad())
                else:
                    self.assertEqual(inp.retain_grad(), torch.zeros_like(inp))


    @onlyCUDA
    @dtypesIfCUDA(*set([torch.float, torch.double, torch.half, torch.bfloat16, torch.complex64, torch.complex128]))
    @with_tf32_off_helper
    def test_nn_ZeroPad2d(self, device, dtype):
        torch.manual_seed(0)
        for contig in [True, False]:
            # ZeroPad2d no-empty
            for mod, inp in [
                    (torch.nn.ZeroPad2d(3), torch.rand(3, 5, 10, 2048, device='cuda', dtype=dtype, requires_grad=True)),
                    (torch.nn.ZeroPad2d(16), torch.rand(5, 10, 64, 64, device='cuda', dtype=dtype, requires_grad=True)),
                    (torch.nn.ZeroPad2d((16, 5, 0, 2)), torch.rand(5, 10, 16, 32, device='cuda', dtype=dtype, requires_grad=True)),
                    (torch.nn.ZeroPad2d((8, 4, 1, 1)), torch.rand(16, 8, 8, 2048, device='cuda', dtype=dtype, requires_grad=True))
                    ]:
                if (dtype == (torch.half or torch.bfloat16)):
                    inp_ref = inp.detach().to(torch.float32).cpu().requires_grad_()
                else:
                    inp_ref = inp.detach().cpu().requires_grad_()
                if contig == False:
                    inp = make_noncontig(inp)
                    inp_ref = make_noncontig(inp_ref)
                out = mod(inp)
                out_ref = mod(inp_ref)
                if (dtype != (torch.half or torch.bfloat16)):
                    self.assertEqual(out, out_ref)
                g = torch.rand_like(out)
                g_ref = g.cpu()
                out.backward(g)
                out_ref.backward(g_ref)
                if (dtype != (torch.half or torch.bfloat16)):
                    self.assertEqual(inp.retain_grad(), inp_ref.retain_grad())
            # ZeroPad2d empty
            for mod, inp in [
                    (torch.nn.ZeroPad2d(3), torch.rand(0, 5, 10, 65736, device='cuda', dtype=dtype)),
                    (torch.nn.ZeroPad2d(15), torch.rand(0, 10, 64, 64, device='cuda', dtype=dtype)),
                    (torch.nn.ZeroPad2d((3, 2, 3, 0)), torch.rand(0, 5, 10, 65736, device='cuda', dtype=dtype)),
                    (torch.nn.ZeroPad2d((32, 5, 0, 2)), torch.rand(0, 3, 64, 128, device='cuda', dtype=dtype)),
                    (torch.nn.ZeroPad2d((16, 64, 1, 1)), torch.rand(0, 16, 8, 2048, device='cuda', dtype=dtype))
                    ]:
                inp = inp.requires_grad_(True)
                if contig == False:
                    inp = make_noncontig(inp)
                out = mod(inp)
                g = torch.rand_like(out)
                out.backward(g)
                for p in mod.parameters():
                    if p.requires_grad:
                        self.assertEqual(p.retain_grad(), torch.zeros_like(p.retain_grad()))
                if inp.retain_grad() == None or inp_ref.retain_grad() == None:
                    self.assertEqual(inp.retain_grad(), inp_ref.retain_grad())
                else:
                    self.assertEqual(inp.retain_grad(), torch.zeros_like(inp))


    @onlyCUDA
    @dtypesIfCUDA(*set([torch.float, torch.double, torch.half, torch.bfloat16, torch.complex64, torch.complex128]))
    @with_tf32_off_helper
    def test_nn_ConstantPad1d(self, device, dtype):
        torch.manual_seed(0)
        for contig in [True, False]:
            # ConstantPad1d no-empty
            for mod, inp in [
                    (torch.nn.ConstantPad1d(16, 3.5), torch.rand(5, 10, 64, device='cuda', dtype=dtype, requires_grad=True)),
                    (torch.nn.ConstantPad1d(4, -3.5), torch.rand(5, 16, 2048, device='cuda', dtype=dtype, requires_grad=True)),
                    (torch.nn.ConstantPad1d((16, 5), 3.5), torch.rand(5, 10, 64, device='cuda', dtype=dtype, requires_grad=True)),
                    (torch.nn.ConstantPad1d((8, 16), -3.5), torch.rand(5, 16, 2048, device='cuda', dtype=dtype, requires_grad=True))
                    ]:
                if (dtype == (torch.half or torch.bfloat16)):
                    inp_ref = inp.detach().to(torch.float32).cpu().requires_grad_()
                else:
                    inp_ref = inp.detach().cpu().requires_grad_()
                if contig == False:
                    inp = make_noncontig(inp)
                    inp_ref = make_noncontig(inp_ref)
                out = mod(inp)
                out_ref = mod(inp_ref)
                if (dtype != (torch.half or torch.bfloat16)):
                    self.assertEqual(out, out_ref)
                g = torch.rand_like(out)
                g_ref = g.cpu()
                out.backward(g)
                out_ref.backward(g_ref)
                if (dtype != (torch.half or torch.bfloat16)):
                    self.assertEqual(inp.retain_grad(), inp_ref.retain_grad())
            # ConstantPad1d empty
            for mod, inp in [
                    (torch.nn.ConstantPad1d(16, 3.5), torch.rand(0, 10, 64, device='cuda', dtype=dtype, requires_grad=True)),
                    (torch.nn.ConstantPad1d(4, -3.5), torch.rand(0, 16, 2048, device='cuda', dtype=dtype, requires_grad=True)),
                    (torch.nn.ConstantPad1d((16, 5), 3.5), torch.rand(0, 10, 64, device='cuda', dtype=dtype, requires_grad=True)),
                    (torch.nn.ConstantPad1d((8, 16), -3.5), torch.rand(0, 16, 2048, device='cuda', dtype=dtype, requires_grad=True))
                    ]:
                inp = inp.requires_grad_(True)
                if contig == False:
                    inp = make_noncontig(inp)
                out = mod(inp)
                g = torch.rand_like(out)
                out.backward(g)
                for p in mod.parameters():
                    if p.requires_grad:
                        self.assertEqual(p.retain_grad(), torch.zeros_like(p.retain_grad()))
                if inp.retain_grad() == None or inp_ref.retain_grad() == None:
                    self.assertEqual(inp.retain_grad(), inp_ref.retain_grad())
                else:
                    self.assertEqual(inp.retain_grad(), torch.zeros_like(inp))


    @onlyCUDA
    @dtypesIfCUDA(*set([torch.float, torch.double, torch.half, torch.bfloat16, torch.complex64, torch.complex128]))
    @with_tf32_off_helper
    def test_nn_ConstantPad2d(self, device, dtype):
        torch.manual_seed(0)
        for contig in [True, False]:
            # ConstantPad2d no-empty
            for mod, inp in [
                    (torch.nn.ConstantPad2d(3, 3.5), torch.rand(3, 5, 10, 2048, device='cuda', dtype=dtype, requires_grad=True)),
                    (torch.nn.ConstantPad2d(16, -3.5), torch.rand(5, 10, 64, 64, device='cuda', dtype=dtype, requires_grad=True)),
                    (torch.nn.ConstantPad2d((16, 5, 0, 2), 3.5), torch.rand(5, 10, 16, 32, device='cuda', dtype=dtype, requires_grad=True)),
                    (torch.nn.ConstantPad2d((8, 4, 1, 1), -3.5), torch.rand(16, 8, 8, 2048, device='cuda', dtype=dtype, requires_grad=True))
                    ]:
                if (dtype == (torch.half or torch.bfloat16)):
                    inp_ref = inp.detach().to(torch.float32).cpu().requires_grad_()
                else:
                    inp_ref = inp.detach().cpu().requires_grad_()
                if contig == False:
                    inp = make_noncontig(inp)
                    inp_ref = make_noncontig(inp_ref)
                out = mod(inp)
                out_ref = mod(inp_ref)
                if (dtype != (torch.half or torch.bfloat16)):
                    self.assertEqual(out, out_ref)
                g = torch.rand_like(out)
                g_ref = g.cpu()
                out.backward(g)
                out_ref.backward(g_ref)
                if (dtype != (torch.half or torch.bfloat16)):
                    self.assertEqual(inp.retain_grad(), inp_ref.retain_grad())
            # ConstantPad2d empty
            for mod, inp in [
                    (torch.nn.ConstantPad2d(3, 3.5), torch.rand(0, 5, 10, 65736, device='cuda', dtype=dtype)),
                    (torch.nn.ConstantPad2d(15, -3.5), torch.rand(0, 10, 64, 64, device='cuda', dtype=dtype)),
                    (torch.nn.ConstantPad2d((3, 2, 3, 0), 3.5), torch.rand(0, 5, 10, 65736, device='cuda', dtype=dtype)),
                    (torch.nn.ConstantPad2d((32, 5, 0, 2), -3.5), torch.rand(0, 3, 64, 128, device='cuda', dtype=dtype)),
                    (torch.nn.ConstantPad2d((16, 64, 1, 1), 3.5), torch.rand(0, 16, 8, 2048, device='cuda', dtype=dtype))
                    ]:
                inp = inp.requires_grad_(True)
                if contig == False:
                    inp = make_noncontig(inp)
                out = mod(inp)
                g = torch.rand_like(out)
                out.backward(g)
                for p in mod.parameters():
                    if p.requires_grad:
                        self.assertEqual(p.retain_grad(), torch.zeros_like(p.retain_grad()))
                if inp.retain_grad() == None or inp_ref.retain_grad() == None:
                    self.assertEqual(inp.retain_grad(), inp_ref.retain_grad())
                else:
                    self.assertEqual(inp.retain_grad(), torch.zeros_like(inp))


    @onlyCUDA
    @dtypesIfCUDA(*set([torch.float, torch.double, torch.half, torch.bfloat16, torch.complex64, torch.complex128]))
    @with_tf32_off_helper
    def test_nn_ConstantPad3d(self, device, dtype):
        torch.manual_seed(0)
        for contig in [True, False]:
            # ConstantPad3d no-empty
            for mod, inp in [
                    (torch.nn.ConstantPad3d(2, 3.5), torch.rand(2, 2, 3, 3, 1024, device='cuda', dtype=dtype, requires_grad=True)),
                    (torch.nn.ConstantPad3d(2, -3.5), torch.rand(2, 1, 4, 16, 4, device='cuda', dtype=dtype, requires_grad=True)),
                    (torch.nn.ConstantPad3d((0, 1, 2, 3, 2, 2), 3.5), torch.rand(3, 2, 5, 4, 1024, device='cuda', dtype=dtype, requires_grad=True)),
                    (torch.nn.ConstantPad3d((2, 4, 5, 2, 2, 3), -3.5), torch.rand(5, 3, 64, 6, 128, device='cuda', dtype=dtype, requires_grad=True)),
                    (torch.nn.ConstantPad3d((1, 1, 1, 1, 1, 1), 3.5), torch.rand(4, 4, 4, 8, 1024, device='cuda', dtype=dtype, requires_grad=True))
                    ]:
                if (dtype == (torch.half or torch.bfloat16)):
                    inp_ref = inp.detach().to(torch.float32).cpu().requires_grad_()
                else:
                    inp_ref = inp.detach().cpu().requires_grad_()
                if contig == False:
                    inp = make_noncontig(inp)
                    inp_ref = make_noncontig(inp_ref)
                out = mod(inp)
                out_ref = mod(inp_ref)
                if (dtype != (torch.half or torch.bfloat16)):
                    self.assertEqual(out, out_ref)
                g = torch.rand_like(out)
                g_ref = g.cpu()
                out.backward(g)
                out_ref.backward(g_ref)
                if (dtype != (torch.half or torch.bfloat16)):
                    self.assertEqual(inp.retain_grad(), inp_ref.retain_grad())
            # ReflectionPad3d empty
            for mod, inp in [
                    (torch.nn.ConstantPad3d(2, 3.5), torch.rand(0, 2, 3, 3, 1024, device='cuda', dtype=dtype, requires_grad=True)),
                    (torch.nn.ConstantPad3d(2, -3.5), torch.rand(0, 1, 4, 16, 4, device='cuda', dtype=dtype, requires_grad=True)),
                    (torch.nn.ConstantPad3d((0, 1, 2, 3, 2, 2), 3.5), torch.rand(0, 2, 5, 4, 1024, device='cuda', dtype=dtype, requires_grad=True)),
                    (torch.nn.ConstantPad3d((2, 4, 5, 2, 2, 3), -3.5), torch.rand(0, 3, 64, 6, 128, device='cuda', dtype=dtype, requires_grad=True)),
                    (torch.nn.ConstantPad3d((1, 1, 1, 1, 1, 1), 3.5), torch.rand(0, 4, 4, 8, 1024, device='cuda', dtype=dtype, requires_grad=True))
                    ]:
                inp = inp.requires_grad_(True)
                if contig == False:
                    inp = make_noncontig(inp)
                out = mod(inp)
                g = torch.rand_like(out)
                out.backward(g)
                for p in mod.parameters():
                    if p.requires_grad:
                        self.assertEqual(p.retain_grad(), torch.zeros_like(p.retain_grad()))
                if inp.retain_grad() == None or inp_ref.retain_grad() == None:
                    self.assertEqual(inp.retain_grad(), inp_ref.retain_grad())
                else:
                    self.assertEqual(inp.retain_grad(), torch.zeros_like(inp))


    @onlyCUDA
    @dtypesIfCUDA(*set([torch.float, torch.double, torch.half, torch.bfloat16]))
    @with_tf32_off_helper
    def test_nn_RNN_LSTM_GRU(self, device, dtype):
        def forward_backward(cuda, rnn, input_val, grad_output, weights_val, hx_val, grad_hy,
                             cx_val=None, grad_cy=None):
            is_lstm = isinstance(rnn, nn.LSTM)
            for x_layer, y_layer in zip(rnn.all_weights, weights_val):
                for x, y in zip(x_layer, y_layer):
                    x.data.copy_(y.data)
            if isinstance(input_val, rnn_utils.PackedSequence):
                input = rnn_utils.PackedSequence(
                    input_val.data.data.requires_grad_(True), input_val.batch_sizes)
                input_var = input.data
            else:
                input = input_val.clone().requires_grad_(True)
                input_var = input
            if is_lstm:
                if cx_val is None:
                    hx = (hx_val.clone().requires_grad_(True), hx_val.add(1).requires_grad_(True))
                else:
                    hx = (hx_val.clone().requires_grad_(True), cx_val.add(1).requires_grad_(True))
            else:
                hx = hx_val.clone().requires_grad_(True)
            if cuda:
                rnn.cuda()
                input_var.data = input_var.data.cuda()
                if is_lstm:
                    hx[0].data = hx[0].data.cuda()
                    hx[1].data = hx[1].data.cuda()
                else:
                    hx.data = hx.data.cuda()
                grad_hy = grad_hy.cuda()
                if grad_cy is not None:
                    grad_cy = grad_cy.cuda()
                grad_output = grad_output.cuda()
            output, hy = rnn(input, hx)
            if isinstance(output, rnn_utils.PackedSequence):
                output = output.data
            if is_lstm:
                if grad_cy is None:
                    torch.autograd.backward([output, hy[0], hy[1]], [grad_output, grad_hy, grad_hy + 1])
                else:
                    torch.autograd.backward([output, hy[0], hy[1]], [grad_output, grad_hy, grad_cy + 1])
            else:
                torch.autograd.backward([output, hy], [grad_output, grad_hy])
            return {'output': output.data,
                    'hy': hy[0].data if is_lstm else hy.data,
                    'weights': rnn.all_weights,
                    'grad_input': input_var.grad.data,
                    'grad_hx': hx[0].grad.data if is_lstm else hx.grad.data,
                    'cy': hy[1].data if is_lstm else None,
                    'grad_cx': hx[1].grad.data if is_lstm else None}

        def compare_cpu_gpu(outputs_cpu, outputs_gpu):
            self.assertEqual(list(outputs_cpu.keys()), list(outputs_gpu.keys()))
            for key in outputs_cpu.keys():
                if key != 'weights':
                    self.assertEqual(outputs_cpu[key], outputs_gpu[key], atol=5e-5, rtol=0, msg=key)
            # check grad weights separately, as nested dict
            for cpu_layer_weight, gpu_layer_weight in zip(outputs_cpu['weights'], outputs_gpu['weights']):
                for (cpu_weight, gpu_weight) in zip(cpu_layer_weight, gpu_layer_weight):
                    self.assertEqual(cpu_weight.grad.data, gpu_weight.grad.data, atol=5e-4, rtol=0)

        torch.manual_seed(0)
        para_list = [
            #input_size, hidden_size, proj_size, num_layers, seq_len, batch, length
            [1, 2, 1, 1, 3, 2, [3, 2]],
            [3, 6, 4, 5, 32, 4, [30, 25, 13, 7]],
        ]
        # test rnn
        for input_size, hidden_size, proj_size, num_layers, seq_len, batch_size, length in para_list:
            for bias, bidirectional, batch_first, contig, variable_len, lens_as_tensor \
                    in product((True, False), repeat=6):
                num_directions = 2 if bidirectional else 1
                for module in (nn.RNN, nn.LSTM, nn.GRU):
                    if (module == nn.LSTM or nn.GRU) and dtype == torch.bfloat16:
                        continue
                    rnn = module(input_size, hidden_size, num_layers, bias=bias,
                                bidirectional=bidirectional, batch_first=batch_first).to(dtype)
                    rnn_gpu = module(input_size, hidden_size, num_layers, bias=bias,
                                    bidirectional=bidirectional, batch_first=batch_first).to(dtype)

                    if batch_first:
                        input_val = torch.rand(batch_size, seq_len, input_size, dtype=dtype)
                        grad_output = torch.rand(batch_size, seq_len, hidden_size * num_directions, dtype=dtype)
                    else:
                        input_val = torch.rand(seq_len, batch_size, input_size, dtype=dtype)
                        grad_output = torch.rand(seq_len, batch_size, hidden_size * num_directions, dtype=dtype)
                    hx_val = torch.rand(num_layers * num_directions, batch_size, hidden_size, dtype=dtype)
                    grad_hy = torch.rand(num_layers * num_directions, batch_size, hidden_size, dtype=dtype)
                    if not contig:
                        grad_output = make_noncontig(grad_output)
                        grad_hy = make_noncontig(grad_hy)
                        input_var = make_noncontig(input_val)
                        hx_val = make_noncontig(hx_val)
                    if variable_len:
                        if lens_as_tensor:
                            lengths = torch.tensor(length, dtype=torch.long)
                        input_val = rnn_utils.pack_padded_sequence(input_val, lengths, batch_first=batch_first)
                        grad_output = rnn_utils.pack_padded_sequence(grad_output, lengths, batch_first=batch_first).data
                    if dtype == torch.float or dtype == torch.double:
                        outputs_cpu = forward_backward(False, rnn, input_val, grad_output, rnn.all_weights, hx_val, grad_hy)
                    outputs_gpu = forward_backward(True, rnn_gpu, input_val, grad_output, rnn.all_weights, hx_val, grad_hy)
                    if dtype == torch.float or dtype == torch.double:
                        compare_cpu_gpu(outputs_cpu, outputs_gpu)
                for nonlinearity in ('tanh', 'relu'):
                    hx_val = torch.rand(num_layers * num_directions, batch_size, hidden_size, dtype=dtype)
                    input_val = torch.rand(seq_len, batch_size, input_size, dtype=dtype)
                    grad_output = torch.rand(seq_len, batch_size, hidden_size * num_directions, dtype=dtype)
                    grad_hy = torch.rand(num_layers * num_directions, batch_size, hidden_size, dtype=dtype)
                    rnn = nn.RNN(input_size, hidden_size, num_layers, bias=bias, nonlinearity=nonlinearity, \
                                    bidirectional=bidirectional).to(dtype)
                    if dtype == torch.float or dtype == torch.double:
                        outputs_cpu = forward_backward(False, rnn, input_val, grad_output, rnn.all_weights, hx_val, grad_hy)
                    rnn_gpu = nn.RNN(input_size, hidden_size, num_layers, bias=bias, nonlinearity=nonlinearity, \
                                    bidirectional=bidirectional).to(dtype)
                    outputs_gpu = forward_backward(True, rnn_gpu, input_val, grad_output, rnn.all_weights, hx_val, grad_hy)
                    if dtype == torch.float or dtype == torch.double:
                        compare_cpu_gpu(outputs_cpu, outputs_gpu)
                # LSTM projection
                if dtype == torch.bfloat16:
                        continue
                num_directions = 2 if bidirectional else 1
                if batch_first:
                    input_val = torch.rand(batch_size, seq_len, input_size, dtype=dtype)
                    grad_output = torch.rand(batch_size, seq_len, proj_size * num_directions, dtype=dtype)
                else:
                    input_val = torch.rand(seq_len, batch_size, input_size, dtype=dtype)
                    grad_output = torch.rand(seq_len, batch_size, proj_size * num_directions, dtype=dtype)
                hx_val = torch.rand(num_layers * num_directions, batch_size, proj_size, dtype=dtype)
                cx_val = torch.rand(num_layers * num_directions, batch_size, hidden_size, dtype=dtype)
                grad_hy = torch.rand(num_layers * num_directions, batch_size, proj_size, dtype=dtype)
                grad_cy = torch.rand(num_layers * num_directions, batch_size, hidden_size, dtype=dtype)
                if not contig:
                    grad_output = make_noncontig(grad_output)
                    grad_hy = make_noncontig(grad_hy)
                    grad_cy = make_noncontig(grad_cy)
                    input_var = make_noncontig(input_val)
                    hx_val = make_noncontig(hx_val)
                    cx_val = make_noncontig(cx_val)
                if variable_len:
                    if lens_as_tensor:
                        lengths = torch.tensor(length, dtype=torch.long)
                    input_val = rnn_utils.pack_padded_sequence(input_val, lengths, batch_first=batch_first)
                    grad_output = rnn_utils.pack_padded_sequence(grad_output, lengths, batch_first=batch_first).data
                rnn = nn.LSTM(input_size, hidden_size, num_layers, bias=bias, bidirectional=bidirectional,
                                batch_first=batch_first,proj_size=proj_size).to(dtype)
                if dtype == torch.float or dtype == torch.double:
                    outputs_cpu = forward_backward(False, rnn, input_val, grad_output, rnn.all_weights,
                        hx_val, grad_hy, cx_val, grad_cy)
                rnn_gpu = nn.LSTM(input_size, hidden_size, num_layers, bias=bias, bidirectional=bidirectional,
                                    batch_first=batch_first, proj_size=proj_size).to(dtype)
                outputs_gpu = forward_backward(True, rnn_gpu, input_val, grad_output, rnn.all_weights,
                                                hx_val, grad_hy, cx_val, grad_cy)
                if dtype == torch.float or dtype == torch.double:
                    compare_cpu_gpu(outputs_cpu, outputs_gpu)


    @onlyCUDA
    @dtypesIfCUDA(*set([torch.float, torch.double, torch.half, torch.bfloat16]))
    @with_tf32_off_helper
    def test_nn_RNNCell(self, device, dtype):
        torch.manual_seed(0)
        rnn_type_list = ['tanh', 'relu']
        para_list = [
            # batch_size, input_size, output_size
            [3, 11, 19], [7, 15, 7], [12, 128, 64],
            [32, 512, 512], [32, 768, 768], [32, 512, 1024],
        ]
        for rnn_type in rnn_type_list:
            for bias, contig in product((True, False), repeat=2):
                for batch_size, input_size, output_size in para_list:
                    inp = torch.rand(batch_size, input_size, device='cuda', dtype=dtype, requires_grad=True)
                    if dtype == torch.float or dtype == torch.double:
                        inpu_cpu = inp.clone().detach().requires_grad_(True).cpu()
                        inp = inpu_cpu.clone().detach().requires_grad_(True).cuda()
                    hx = torch.rand(batch_size, output_size, device='cuda', dtype=dtype, requires_grad=True)
                    if dtype == torch.float or dtype == torch.double:
                        hx_cpu = hx.clone().detach().requires_grad_(True).cpu()
                    torch.manual_seed(0)
                    mod = nn.RNNCell(input_size, output_size, bias=bias, nonlinearity=rnn_type, device='cuda', dtype=dtype)
                    if dtype == torch.float or dtype == torch.double:
                        torch.manual_seed(0)
                        mod_cpu = nn.RNNCell(input_size, output_size, bias=bias, nonlinearity=rnn_type, device='cuda', dtype=dtype).cpu()
                    if contig == False:
                        inp = make_noncontig(inp)
                        hx = make_noncontig(hx)
                        if dtype == torch.float or dtype == torch.double:
                            inpu_cpu = make_noncontig(inpu_cpu)
                            hx_cpu = make_noncontig(hx_cpu)
                    hy = mod(inp, hx)
                    hy.sum().backward()
                    if dtype == torch.float or dtype == torch.double:
                        hy_cpu = mod_cpu(inpu_cpu, hx_cpu)
                        hy_cpu.sum().backward()
                        self.assertEqual(hy, hy_cpu)
                        self.assertEqual(inp.retain_grad(), inpu_cpu.retain_grad())
                        for para, para_cpu in zip(mod.parameters(), mod_cpu.parameters()):
                            self.assertEqual(para.retain_grad(), para_cpu.retain_grad())


    @onlyCUDA
    @dtypesIfCUDA(*set([torch.float]))
    @with_tf32_off_helper
    def test_nn_LSTMCell(self, device, dtype):
        torch.manual_seed(0)
        para_list = [
            # batch_size, input_size, output_size
            [3, 11, 19], [7, 15, 7], [12, 128, 64],
            [32, 512, 512], [32, 768, 768], [32, 512, 1024],
        ]
        for bias, contig in product((True, False), repeat=2):
            for batch_size, input_size, output_size in para_list:
                inp = torch.rand(batch_size, input_size, device='cuda', dtype=dtype, requires_grad=True)
                if dtype == torch.float or dtype == torch.double:
                    inpu_cpu = inp.clone().detach().requires_grad_(True).cpu()
                    inp = inpu_cpu.clone().detach().requires_grad_(True).cuda()
                hx = torch.rand(batch_size, output_size, device='cuda', dtype=dtype, requires_grad=True)
                if dtype == torch.float or dtype == torch.double:
                    hx_cpu = hx.clone().detach().requires_grad_(True).cpu()
                cx = torch.rand(batch_size, output_size, device='cuda', dtype=dtype, requires_grad=True)
                if dtype == torch.float or dtype == torch.double:
                    cx_cpu = cx.clone().detach().requires_grad_(True).cpu()
                torch.manual_seed(0)
                mod = nn.LSTMCell(input_size, output_size, bias=bias, device='cuda', dtype=dtype)
                if dtype == torch.float or dtype == torch.double:
                    torch.manual_seed(0)
                    mod_cpu = nn.LSTMCell(input_size, output_size, bias=bias, device='cuda', dtype=dtype).cpu()
                if contig == False:
                        inp = make_noncontig(inp)
                        hx = make_noncontig(hx)
                        cx = make_noncontig(cx)
                        inpu_cpu = make_noncontig(inpu_cpu)
                        hx_cpu = make_noncontig(hx_cpu)
                        cx_cpu = make_noncontig(cx_cpu)
                hy, cy = mod(inp, (hx, cx))
                (hy + cy).sum().backward()
                if dtype == torch.float or dtype == torch.double:
                    hy_cpu, cy_cpu = mod_cpu(inpu_cpu, (hx_cpu, cx_cpu))
                    (hy_cpu + cy_cpu).sum().backward()
                    self.assertEqual(hy, hy_cpu)
                    self.assertEqual(cy, cy_cpu)
                    self.assertEqual(inp.retain_grad(), inpu_cpu.retain_grad())
                    for para, para_cpu in zip(mod.parameters(), mod_cpu.parameters()):
                        self.assertEqual(para.retain_grad(), para_cpu.retain_grad())


    @onlyCUDA
    @dtypesIfCUDA(*set([torch.float, torch.double, torch.half]))
    @with_tf32_off_helper
    def test_nn_GRUCell(self, device, dtype):
        torch.manual_seed(0)
        para_list = [
            # batch_size, input_size, output_size
            [3, 11, 19], [7, 15, 7], [12, 128, 64],
            [32, 512, 512], [32, 768, 768], [32, 512, 1024],
        ]
        for bias, contig in product((True, False), repeat=2):
            for batch_size, input_size, output_size in para_list:
                inp = torch.rand(batch_size, input_size, device='cuda', dtype=dtype, requires_grad=True)
                if dtype == torch.float or dtype == torch.double:
                    input_cpu = inp.clone().detach().requires_grad_(True).cpu()
                    inp = input_cpu.clone().detach().requires_grad_(True).cuda()
                hx = torch.rand(batch_size, output_size, device='cuda', dtype=dtype, requires_grad=True)
                if dtype == torch.float or dtype == torch.double:
                    hx_cpu = hx.clone().detach().requires_grad_(True).cpu()
                torch.manual_seed(0)
                mod = nn.GRUCell(input_size, output_size, bias=bias, device='cuda', dtype=dtype)
                if dtype == torch.float or dtype == torch.double:
                    torch.manual_seed(0)
                    mod_cpu = nn.GRUCell(input_size, output_size, bias=bias, device='cuda', dtype=dtype).cpu()
                if contig == False:
                        inp = make_noncontig(inp)
                        hx = make_noncontig(hx)
                        if dtype == torch.float or dtype == torch.double:
                            input_cpu = make_noncontig(input_cpu)
                            hx_cpu = make_noncontig(hx_cpu)
                hy = mod(inp, hx)
                hy.sum().backward()
                if dtype == torch.float or dtype == torch.double:
                    hy_cpu = mod_cpu(input_cpu, hx_cpu)
                    hy_cpu.sum().backward()
                    self.assertEqual(hy, hy_cpu)
                    self.assertEqual(inp.retain_grad(), input_cpu.retain_grad())
                    for para, para_cpu in zip(mod.parameters(), mod_cpu.parameters()):
                        self.assertEqual(para.retain_grad(), para_cpu.retain_grad())


    @onlyCUDA
    @dtypesIfCUDA(*set([torch.float, torch.double, torch.half, torch.bfloat16]))
    @with_tf32_off_helper
    def test_nn_Transformer(self, device, dtype):
        torch.manual_seed(0)
        d_model = 512
        nhead = 16
        num_encoder_layers = 4
        num_decoder_layers = 3
        dim_feedforward = 256
        dropout = 0
        bsz = 8
        seq_length = 35
        tgt_length = 15
        for contig in [True, False]:
            for batch_first, src_size, tgt_size in zip((True, False),
                                                    [(bsz, seq_length, d_model), (seq_length, bsz, d_model)],
                                                    [(bsz, tgt_length, d_model), (tgt_length, bsz, d_model)]):
                torch.manual_seed(0)
                transformer = nn.Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward,
                                            dropout, batch_first=batch_first, device='cuda', dtype=dtype)
                src = torch.rand(src_size, device='cuda', dtype=dtype).requires_grad_(True)
                src_mask = transformer.generate_square_subsequent_mask(seq_length).to(dtype).requires_grad_(True).cuda()
                tgt = torch.rand(tgt_size, device='cuda', dtype=dtype).requires_grad_(True)
                tgt_mask = transformer.generate_square_subsequent_mask(tgt_length).to(dtype).requires_grad_(True).cuda()
                memory_mask = torch.rand(tgt_length, seq_length).to(dtype).requires_grad_(True).cuda()
                src_key_padding_mask = torch.rand(bsz, seq_length, device='cuda', dtype=dtype) >= 0.5
                tgt_key_padding_mask = torch.rand(bsz, tgt_length, device='cuda', dtype=dtype) >= 0.5
                memory_key_padding_mask = torch.rand(bsz, seq_length, device='cuda', dtype=dtype) >= 0.5
                if dtype == torch.float or dtype == torch.double:
                    torch.manual_seed(0)
                    transformer_cpu = nn.Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward,
                                                dropout, batch_first=batch_first, device='cuda', dtype=dtype).cpu()
                    src_cpu = src.clone().detach().requires_grad_(True).cpu()
                    src_mask_cpu = src_mask.clone().detach().requires_grad_(True).cpu()
                    tgt_cpu = tgt.clone().detach().requires_grad_(True).cpu()
                    tgt_mask_cpu = tgt_mask.clone().detach().requires_grad_(True).cpu()
                    memory_mask_cpu = memory_mask.clone().detach().requires_grad_(True).cpu()
                    src_key_padding_mask_cpu = src_key_padding_mask.clone().detach().cpu()
                    tgt_key_padding_mask_cpu = tgt_key_padding_mask.clone().detach().cpu()
                    memory_key_padding_mask_cpu = memory_key_padding_mask.clone().detach().cpu()
                if contig == False:
                    src = make_noncontig(src)
                    src_mask = make_noncontig(src_mask)
                    tgt = make_noncontig(tgt)
                    tgt_mask = make_noncontig(tgt_mask)
                    memory_mask = make_noncontig(memory_mask)
                    src_key_padding_mask = make_noncontig(src_key_padding_mask)
                    tgt_key_padding_mask = make_noncontig(tgt_key_padding_mask)
                    memory_key_padding_mask = make_noncontig(memory_key_padding_mask)
                    if dtype == torch.float or dtype == torch.double:
                        src_cpu = make_noncontig(src_cpu)
                        src_mask_cpu = make_noncontig(src_mask_cpu)
                        tgt_cpu = make_noncontig(tgt_cpu)
                        tgt_mask_cpu = make_noncontig(tgt_mask_cpu)
                        memory_mask_cpu = make_noncontig(memory_mask_cpu)
                        src_key_padding_mask_cpu = make_noncontig(src_key_padding_mask_cpu)
                        tgt_key_padding_mask_cpu = make_noncontig(tgt_key_padding_mask_cpu)
                        memory_key_padding_mask_cpu = make_noncontig(memory_key_padding_mask_cpu)
                output = transformer(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask, memory_mask=memory_mask,
                                    src_key_padding_mask=src_key_padding_mask, tgt_key_padding_mask=tgt_key_padding_mask,
                                    memory_key_padding_mask=memory_key_padding_mask)
                output.sum().backward()
                if dtype == torch.float or dtype == torch.double:
                    output_cpu = transformer_cpu(src_cpu, tgt_cpu, src_mask=src_mask_cpu, tgt_mask=tgt_mask_cpu,
                                        memory_mask=memory_mask_cpu, src_key_padding_mask=src_key_padding_mask_cpu,
                                        tgt_key_padding_mask=tgt_key_padding_mask_cpu,
                                        memory_key_padding_mask=memory_key_padding_mask_cpu)
                    output_cpu.sum().backward()
                    self.assertEqual(src, src_cpu)
                    self.assertEqual(tgt, tgt_cpu)
                    self.assertEqual(src_mask, src_mask_cpu)
                    self.assertEqual(tgt_mask, tgt_mask_cpu)
                    self.assertEqual(memory_mask, memory_mask_cpu)
                    self.assertEqual(src_key_padding_mask, src_key_padding_mask_cpu)
                    self.assertEqual(tgt_key_padding_mask, tgt_key_padding_mask_cpu)
                    self.assertEqual(memory_key_padding_mask, memory_key_padding_mask_cpu)


    @onlyCUDA
    @dtypesIfCUDA(*set([torch.float, torch.double, torch.half, torch.bfloat16, torch.half]))
    @with_tf32_off_helper
    def test_nn_TransformerEncoder_TransformerEncoderLayer(self, device, dtype):
        torch.manual_seed(0)
        list_para = [
            [8, 4, 4, 0.0, 'relu', 4],
            [32, 16, 32, 0.0, 'gelu', 8],
            [512, 8, 8, 0.0, 'relu', 5],
            [512, 4, 5, 0.0, 'gelu', 3],
        ]
        for d_model, nhead, dim_feedforward, dropout, activation, num_layers in list_para:
            for batch_first, norm_first, contig in product((True, False), repeat=3):
                if dtype == torch.float or dtype == torch.double:
                    torch.manual_seed(0)
                    encoder_layer_cpu = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout,
                                        activation, batch_first=batch_first, norm_first=norm_first).cpu()
                torch.manual_seed(0)
                encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout,
                                    activation, batch_first=batch_first, norm_first=norm_first).cuda()
                for norm_cpu, norm in [[nn.LayerNorm(d_model, device='cpu'), nn.LayerNorm(d_model, device='cuda')], [None, None]]:
                    if dtype == torch.float or dtype == torch.double:
                        torch.manual_seed(0)
                        transformer_encoder_cpu = nn.TransformerEncoder(encoder_layer_cpu, num_layers, norm_cpu).to(dtype)
                    torch.manual_seed(0)
                    transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers, norm).to(dtype).cuda()
                    for a, b in [[8, 32], [3, 7]]:
                        if dtype == torch.float or dtype == torch.double:
                            inp_cpu = torch.rand(a, b, d_model).to(dtype).requires_grad_(True).cpu()
                            inp = inp_cpu.clone().detach().requires_grad_(True).cuda()
                        else:
                            inp = torch.rand(a, b, d_model, device='cuda', dtype=dtype).requires_grad_(True)
                        if contig == False:
                            inp = make_noncontig(inp)
                            if dtype == torch.float or dtype == torch.double:
                                inp_cpu = make_noncontig(inp_cpu)
                        if dtype == torch.float or dtype == torch.double:
                            out_cpu = transformer_encoder_cpu(inp_cpu)
                        out = transformer_encoder(inp)
                        if dtype == torch.float or dtype == torch.double:
                            self.assertEqual(out_cpu, out)
                            self.assertEqual(inp_cpu.retain_grad(), inp.retain_grad())
                            for para, para_cpu in zip(transformer_encoder.parameters(), transformer_encoder_cpu.parameters()):
                                self.assertEqual(para.retain_grad(), para_cpu.retain_grad())


    @onlyCUDA
    @dtypesIfCUDA(*set([torch.float, torch.double, torch.half, torch.half]))
    @with_tf32_off_helper
    def test_nn_TransformerDecoder_TransformerDecoderLayer(self, device, dtype):
        torch.manual_seed(0)
        list_para = [
            [8, 4, 4, 0.0, 'relu', 4],
            [32, 16, 32, 0.0, 'gelu', 8],
            [512, 8, 8, 0.0, 'relu', 5],
            [512, 4, 5, 0.0, 'gelu', 3],
        ]
        for d_model, nhead, dim_feedforward, dropout, activation, num_layers in list_para:
            for batch_first, norm_first, contig in product((True, False), repeat=3):
                if dtype == torch.float or dtype == torch.double:
                    torch.manual_seed(0)
                    decoder_layer_cpu = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout,
                                        activation, batch_first=batch_first, norm_first=norm_first).cpu()
                torch.manual_seed(0)
                decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout,
                                    activation, batch_first=batch_first, norm_first=norm_first).cuda()
                for norm_cpu, norm in [[nn.LayerNorm(d_model, device='cpu'), nn.LayerNorm(d_model, device='cuda')], [None, None]]:
                    if dtype == torch.float or dtype == torch.double:
                        torch.manual_seed(0)
                        transformer_decoder_cpu = nn.TransformerDecoder(decoder_layer_cpu, num_layers, norm_cpu).to(dtype)
                    torch.manual_seed(0)
                    transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers, norm).to(dtype).cuda()
                    for a, b in [[15, 32], [13, 7]]:
                        if dtype == torch.float or dtype == torch.double:
                            inp_cpu = torch.rand(a, b, d_model).to(dtype).requires_grad_(True).cpu()
                            inp = inp_cpu.clone().detach().requires_grad_(True).cuda()
                            tgt_cpu = torch.rand(a, b, d_model).to(dtype).cpu()
                            tgt = tgt_cpu.clone().detach().requires_grad_(True).cuda()
                        else:
                            inp = torch.rand(a, b, d_model, device='cuda', dtype=dtype).requires_grad_(True)
                            tgt = torch.rand(a, b, d_model, device='cuda', dtype=dtype)
                        if contig == False:
                            inp = make_noncontig(inp)
                            tgt = make_noncontig(tgt)
                            if dtype == torch.float or dtype == torch.double:
                                inp_cpu = make_noncontig(inp_cpu)
                                tgt_cpu = make_noncontig(tgt_cpu)
                        if dtype == torch.float or dtype == torch.double:
                            out_cpu = transformer_decoder_cpu(inp_cpu, tgt_cpu)
                        out = transformer_decoder(inp, tgt)
                        if dtype == torch.float or dtype == torch.double:
                            self.assertEqual(out_cpu, out)
                            self.assertEqual(inp_cpu.retain_grad(), inp.retain_grad())
                            for para, para_cpu in zip(transformer_decoder.parameters(), transformer_decoder_cpu.parameters()):
                                self.assertEqual(para.retain_grad(), para_cpu.retain_grad())


    @onlyCUDA
    @dtypesIfCUDA(*set([torch.float, torch.double, torch.half, torch.bfloat16,
            torch.complex64, torch.complex128, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]))
    @with_tf32_off_helper
    def test_nn_utils_rnn_PackedSequence(self, device, dtype):
        torch.manual_seed(0)
        list_size = [
            [[5, 7], [64, 5, 7]],
            [[512, 768], [4, 512, 768]],
            [[2048, 2048], [4, 1024, 1024]]
        ]
        for sz in list_size:
            for contig in [True, False]:
                if dtype in ALL_INTEGER_TYPES:
                    inp1_temp = numpy.random.randint(-2048, 2048, size=sz[0])
                    inp1_cpu = torch.tensor(inp1_temp, dtype=dtype)
                    inp2_temp = numpy.random.randint(-2048, 2048, size=sz[1])
                    inp2_cpu = torch.tensor(inp2_temp, dtype=dtype)
                else:
                    inp1_cpu = torch.randn(sz[0]).to(dtype)
                    inp2_cpu = torch.randn(sz[1]).to(dtype)
                inp1_cuda = inp1_cpu.clone().detach().cuda()
                inp2_cuda = inp2_cpu.clone().detach().cuda()
                if contig == False:
                    inp1_cpu = make_noncontig(inp1_cpu)
                    inp2_cpu = make_noncontig(inp2_cpu)
                    inp1_cuda = make_noncontig(inp1_cuda)
                    inp2_cuda = make_noncontig(inp2_cuda)
                out_cpu = torch.nn.utils.rnn.PackedSequence([inp1_cpu, inp2_cpu])
                out_cuda = torch.nn.utils.rnn.PackedSequence([inp1_cuda, inp2_cuda])
                self.assertEqual(out_cpu, out_cuda)


    @onlyCUDA
    @dtypesIfCUDA(*set([torch.float, torch.double, torch.half, torch.bfloat16,
            torch.complex64, torch.complex128, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]))
    @with_tf32_off_helper
    def test_nn_utils_rnn_pack_padded_sequence(self, device, dtype):
        torch.manual_seed(0)
        list_size = [
            [5, 7], [512, 768], [2048, 2048],
            [64, 5, 7], [4, 512, 768], [4, 1024, 1024]
        ]
        for sz in list_size:
            for contig in [True, False]:
                if dtype in ALL_INTEGER_TYPES:
                    inp_temp = numpy.random.randint(-2048, 2048, size=sz)
                    inp_cpu = torch.tensor(inp_temp, dtype=dtype)
                else:
                    inp_cpu = torch.randn(sz).to(dtype)
                inp_cuda = inp_cpu.clone().detach().cuda()
                if contig == False:
                    inp_cpu = make_noncontig(inp_cpu)
                    inp_cuda = make_noncontig(inp_cuda)
                lengths = numpy.random.randint(1, 32, size=sz[1])
                out_cpu = torch.nn.utils.rnn.pack_padded_sequence(inp_cpu, lengths, enforce_sorted=False)
                out_cuda = torch.nn.utils.rnn.pack_padded_sequence(inp_cuda, lengths, enforce_sorted=False)
                self.assertEqual(out_cpu, out_cuda)


    @onlyCUDA
    @dtypesIfCUDA(*set([torch.float, torch.double, torch.half, torch.bfloat16,
            torch.complex64, torch.complex128, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]))
    @with_tf32_off_helper
    def test_nn_utils_rnn_pad_packed_sequence(self, device, dtype):
        torch.manual_seed(0)
        list_size = [
            [5, 7], [512, 768], [2048, 2048],
            [64, 5, 7], [4, 512, 768], [4, 1024, 1024]
        ]
        for sz in list_size:
            for contig in [True, False]:
                if dtype in ALL_INTEGER_TYPES:
                    inp_temp = numpy.random.randint(-2048, 2048, size=sz)
                    inp_cpu = torch.tensor(inp_temp, dtype=dtype)
                else:
                    inp_cpu = torch.randn(sz).to(dtype)
                inp_cuda = inp_cpu.clone().detach().cuda()
                if contig == False:
                    inp_cpu = make_noncontig(inp_cpu)
                    inp_cuda = make_noncontig(inp_cuda)
                lengths = numpy.random.randint(1, sz[0], size=sz[1])
                packed_cpu = torch.nn.utils.rnn.pack_padded_sequence(inp_cpu, lengths, enforce_sorted=False)
                packed_cuda = torch.nn.utils.rnn.pack_padded_sequence(inp_cuda, lengths, enforce_sorted=False)
                self.assertEqual(packed_cpu, packed_cuda)
                seq_unpacked_cpu, len_unpacked_cpu = torch.nn.utils.rnn.pad_packed_sequence(packed_cpu)
                seq_unpacked_cuda, len_unpacked_cuda = torch.nn.utils.rnn.pad_packed_sequence(packed_cuda)
                self.assertEqual(seq_unpacked_cpu, seq_unpacked_cuda)
                self.assertEqual(len_unpacked_cpu, len_unpacked_cuda)


    @onlyCUDA
    @dtypesIfCUDA(*set([torch.float, torch.double, torch.half, torch.bfloat16,
            torch.complex64, torch.complex128, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]))
    @with_tf32_off_helper
    def test_nn_utils_rnn_pad_sequence(self, device, dtype):
        torch.manual_seed(0)
        list_size = [64, 128, 256, 512, 1024]
        for sz in list_size:
            for contig in [True, False]:
                if dtype in ALL_INTEGER_TYPES:
                    inp1_temp = numpy.random.randint(-2048, 2048, size=(numpy.random.randint(1, 2048), sz))
                    inp1_cpu = torch.tensor(inp1_temp, dtype=dtype)
                    inp2_temp = numpy.random.randint(-2048, 2048, size=(numpy.random.randint(1, 2048), sz))
                    inp2_cpu = torch.tensor(inp2_temp, dtype=dtype)
                else:
                    inp1_cpu = torch.randn(numpy.random.randint(1, 2048), sz).to(dtype)
                    inp2_cpu = torch.randn(numpy.random.randint(1, 2048), sz).to(dtype)
                inp1_cuda = inp1_cpu.clone().detach().cuda()
                inp2_cuda = inp2_cpu.clone().detach().cuda()
                if contig == False:
                    inp1_cpu = make_noncontig(inp1_cpu)
                    inp1_cuda = make_noncontig(inp1_cuda)
                    inp2_cpu = make_noncontig(inp2_cpu)
                    inp2_cuda = make_noncontig(inp2_cuda)
                out_cpu = torch.nn.utils.rnn.pad_sequence([inp1_cpu, inp2_cpu])
                out_cuda = torch.nn.utils.rnn.pad_sequence([inp1_cuda, inp2_cuda])
                self.assertEqual(out_cpu, out_cuda)


    @onlyCUDA
    @dtypesIfCUDA(*set([torch.float, torch.double, torch.half, torch.bfloat16,
            torch.complex64, torch.complex128, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]))
    @with_tf32_off_helper
    def test_nn_utils_rnn_pack_sequence(self, device, dtype):
        torch.manual_seed(0)
        list_size = [
            [5, 7], [512, 768], [2048, 2048],
            [64, 5, 7], [4, 512, 768], [4, 1024, 1024]
        ]
        for sz in list_size:
            for contig in [True, False]:
                if dtype in ALL_INTEGER_TYPES:
                    inp1_temp = numpy.random.randint(-2048, 2048, size=sz)
                    inp1_cpu = torch.tensor(inp1_temp, dtype=dtype)
                    inp2_temp = numpy.random.randint(-2048, 2048, size=sz)
                    inp2_cpu = torch.tensor(inp2_temp, dtype=dtype)
                else:
                    inp1_cpu = torch.randn(sz).to(dtype)
                    inp2_cpu = torch.randn(sz).to(dtype)
                inp1_cuda = inp1_cpu.clone().detach().cuda()
                inp2_cuda = inp2_cpu.clone().detach().cuda()
                if contig == False:
                    inp1_cpu = make_noncontig(inp1_cpu)
                    inp2_cpu = make_noncontig(inp2_cpu)
                    inp1_cuda = make_noncontig(inp1_cuda)
                    inp2_cuda = make_noncontig(inp2_cuda)
                out_cpu = torch.nn.utils.rnn.pack_sequence([inp1_cpu, inp2_cpu])
                out_cuda = torch.nn.utils.rnn.pack_sequence([inp1_cuda, inp2_cuda])
                self.assertEqual(out_cpu, out_cuda)

instantiate_device_type_tests(TestNnActivation, globals())
instantiate_device_type_tests(TestNnUpsample, globals())
instantiate_device_type_tests(TestNnParametrize, globals())
instantiate_device_type_tests(TestNnElementwise, globals())
instantiate_device_type_tests(TestSoftxx, globals())
instantiate_device_type_tests(TestLinear, globals())
instantiate_device_type_tests(TestDropout, globals())
instantiate_device_type_tests(TestInit, globals())
instantiate_device_type_tests(TestDistanceOps, globals())
instantiate_device_type_tests(TestLossOps, globals())
instantiate_device_type_tests(TestPooling, globals())
instantiate_device_type_tests(TestNorm, globals())
instantiate_device_type_tests(TestFlatten, globals())
instantiate_device_type_tests(TestNNSegment0, globals())
instantiate_device_type_tests(TestNNSegment1, globals())
instantiate_device_type_tests(TestNNSegment2, globals())
instantiate_device_type_tests(TestNNSegment3, globals())

if __name__ == "__main__":
    unittest.main()
