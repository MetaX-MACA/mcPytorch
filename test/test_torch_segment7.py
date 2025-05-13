import torch
import unittest
import copy
import numpy as np
import random
from itertools import product, combinations, combinations_with_replacement
from torch import inf, nan
from common_util import runtestapi, runtest, onlyCUDA, dtypesIfCUDA, get_all_dtypes, instantiate_device_type_tests, \
    make_tensor, get_all_fp_dtypes, get_all_int_dtypes, get_all_complex_dtypes, parametrize, gendata, checkclose, \
    precisionOverride, dtypes, NORM_NAME
from torch.testing._internal.common_utils import torch_to_numpy_dtype_dict, TestCase

class TestComparisionOps(unittest.TestCase):
    def __comparision_ops_verify_run(self, input1, input2, func, device):
        runtestapi(func=func, fwd_input_list=[input1, input2], fwd_golden=[],  enable_backward=False)

    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_dtypes()))
    def test_eq_or_ne(self, device, dtype):
        '''
        same shape
        '''
        func_list = [torch.eq, torch.ne, torch.not_equal]
        for func in func_list:
            input1 = torch.Tensor([1, 2]).to(dtype)
            input2 = torch.Tensor([1, 1]).to(dtype)
            self.__comparision_ops_verify_run(input1, input2, func, device)

            input1 = torch.Tensor([[1, 2], [3, 4]]).to(dtype)
            input2 = torch.Tensor([[1, 1], [4, 4]]).to(dtype)
            self.__comparision_ops_verify_run(input1, input2, func, device)

            input1 = torch.Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]).to(dtype)
            input2 = torch.Tensor([[[1, 2], [4, 3]], [[6, 6], [7, 8]]]).to(dtype)
            self.__comparision_ops_verify_run(input1, input2, func, device)

            '''
            broadcast
            '''
            input1 = torch.Tensor([1,2]).to(dtype)
            input2 = torch.Tensor([[1,2], [1,3], [1,2]]).to(dtype)
            self.__comparision_ops_verify_run(input1, input2, func, device)

    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_fp_dtypes(include_half=True, include_bfloat16=False)))
    def test_equal(self, device, dtype):
        # Contiguous, 1D
        t1 = torch.tensor((3., 4., 9., 10.))
        t2 = t1.contiguous()
        t3 = torch.tensor((1., 9., 3., 10.))
        t4 = torch.tensor((3., 4., 9.))
        t5 = torch.tensor([])
        self.assertTrue(t1.equal(t2))
        self.assertFalse(t1.equal(t3))
        self.assertFalse(t1.equal(t4))
        self.assertFalse(t1.equal(t5))
        self.assertTrue(torch.equal(t1, t2))
        self.assertFalse(torch.equal(t1, t3))
        self.assertFalse(torch.equal(t1, t4))
        self.assertFalse(torch.equal(t1, t5))

        # Non contiguous, 2D
        s = torch.tensor(((1, 2, 3, 4), (5, 6, 7, 8)))
        s1 = s[:, 1:3]
        s2 = s1.clone()
        s3 = torch.tensor(((2, 3), (6, 7)))
        s4 = torch.tensor(((0, 0), (0, 0)))

        self.assertFalse(s1.is_contiguous())
        self.assertTrue(s1.equal(s2))
        self.assertTrue(s1.equal(s3))
        self.assertFalse(s1.equal(s4))
        self.assertTrue(torch.equal(s1, s2))
        self.assertTrue(torch.equal(s1, s3))
        self.assertFalse(torch.equal(s1, s4))

    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_fp_dtypes()))
    @parametrize("shape", [(2), (2, 2), (4, 2)], name_fn=NORM_NAME)
    def test_ge(self, device, dtype, shape):
        func_list =  [torch.ge, torch.gt, torch.le, torch.lt]
        for func in func_list:
            input1 = gendata([shape], type_list=[dtype], rand_algo="uniform", lower=-1.0, upper=1.0)[0]
            input2 = gendata([shape], type_list=[dtype], rand_algo="uniform", lower=-1.0, upper=1.0)[0]
            self.__comparision_ops_verify_run(input1, input2, func, device)

    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_fp_dtypes()))
    @parametrize("shape1,shape2", [((2), (3,2)), ((1, 2), (4, 2)), ((5, 3, 4, 1), (3, 1, 1))], name_fn=NORM_NAME)
    def test_ge_broadcast(self, device, dtype, shape1, shape2):
        func_list = [torch.ge, torch.gt, torch.le, torch.lt]
        for func in func_list:
            input1 = gendata([shape1], type_list=[dtype], rand_algo="uniform", lower=-1.0, upper=1.0)[0]
            input2 = gendata([shape2], type_list=[dtype], rand_algo="uniform", lower=-1.0, upper=1.0)[0]
            self.__comparision_ops_verify_run(input1, input2, func, device)

class TestIsOps(unittest.TestCase):
    def __is_ops_verify_run(self, input, expected, func, device):
        real, _, _, _ = runtest(func=func, fwd_input_list=[input], 
            device=device, enable_backward=False)
        print("real:", real)
        checkclose(real[0].to(dtype=torch.bool), expected)

    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_fp_dtypes())) 
    def test_is_finite(self, device, dtype):
        input = torch.tensor([1, float('inf'), 2, float('-inf'), float('nan')]).to(dtype)
        expected = torch.Tensor([True, False, True, False, False]).to(torch.bool)
        func = torch.isfinite
        self.__is_ops_verify_run(input, expected, func, device)
    
    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_fp_dtypes())) 
    def test_is_inf(self, device, dtype):
        input = torch.tensor([1, float('inf'), 2, float('-inf'), float('nan')]).to(dtype)
        expected = torch.Tensor([False, True, False, True, False]).to(torch.bool)
        func = torch.isinf
        self.__is_ops_verify_run(input, expected, func, device)

    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_fp_dtypes())) 
    def test_is_posinf(self, device, dtype):
        input = torch.tensor([-float('inf'), float('inf'), 1.2]).to(dtype)
        expected = torch.Tensor([False, True, False]).to(torch.bool)
        func = torch.isposinf
        self.__is_ops_verify_run(input, expected, func, device)

    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_fp_dtypes())) 
    def test_is_neginf(self, device, dtype):
        input = torch.tensor([-float('inf'), float('inf'), 1.2]).to(dtype)
        expected = torch.Tensor([True, False, False]).to(torch.bool)
        func = torch.isneginf
        self.__is_ops_verify_run(input, expected, func, device)

    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_fp_dtypes())) 
    def test_is_nan(self, device, dtype):
        input = torch.tensor([-float('inf'), float('-inf'), 1.2, float('nan')]).to(dtype)
        expected = torch.Tensor([False, False, False, True]).to(torch.bool)
        func = torch.isnan
        self.__is_ops_verify_run(input, expected, func, device)

    @onlyCUDA
    def test_is_real(self, device):
        input = torch.tensor([1, 1+1j, 2+0j])
        expected = torch.Tensor([True, False, True]).to(torch.bool)
        func = torch.isreal
        self.__is_ops_verify_run(input, expected, func, device)

class TestWindowOps(unittest.TestCase):
    @onlyCUDA 
    @dtypesIfCUDA(*set(get_all_fp_dtypes(include_half=False, include_bfloat16=False)))
    @parametrize("window_length", [1, 3, 5, 10])
    @parametrize("periodic", [True, False])
    @parametrize("layout", [torch.strided])
    def test_bartlett_window(self, device, window_length, periodic, layout, dtype):
        cpu_result = torch.bartlett_window(window_length, periodic=periodic, dtype=dtype, layout=layout, device="cpu")
        target_result = torch.bartlett_window(window_length, periodic=periodic, dtype=dtype, layout=layout, device=device)
        checkclose(target_result.cpu(), cpu_result)

    @onlyCUDA 
    @dtypesIfCUDA(*set(get_all_fp_dtypes(include_half=False, include_bfloat16=False)))  
    @parametrize("window_length", [1, 3, 5, 10])
    @parametrize("periodic", [True, False])
    @parametrize("layout", [torch.strided])
    def test_blackman_window(self, device, window_length, periodic, layout, dtype):
        cpu_result = torch.blackman_window(window_length, periodic=periodic, dtype=dtype, layout=layout, device="cpu")
        target_result = torch.blackman_window(window_length, periodic=periodic, dtype=dtype, layout=layout, device=device)
        checkclose(target_result.cpu(), cpu_result)

    @onlyCUDA 
    @dtypesIfCUDA(*set(get_all_fp_dtypes(include_half=False, include_bfloat16=False)))  
    @parametrize("window_length", [1, 3, 5, 10])
    @parametrize("periodic", [True, False])
    @parametrize("alpha,beta", [(0.54, 0.46), (0.3, 0.7), (0.4, 0.6)], name_fn=NORM_NAME)
    @parametrize("layout", [torch.strided])
    def test_hamming_window(self, device, window_length, periodic, alpha, beta, layout, dtype):
        cpu_result = torch.hamming_window(window_length, periodic=periodic, alpha=alpha, beta=beta, dtype=dtype, layout=layout, device="cpu")
        target_result = torch.hamming_window(window_length, periodic=periodic, alpha=alpha, beta=beta, dtype=dtype, layout=layout, device=device)
        checkclose(target_result.cpu(), cpu_result)

    @onlyCUDA 
    @dtypesIfCUDA(*set(get_all_fp_dtypes(include_half=False, include_bfloat16=False)))  
    @parametrize("window_length", [1, 3, 5, 10])
    @parametrize("periodic", [True, False])
    @parametrize("layout", [torch.strided])
    def test_hann_window(self, device, window_length, periodic, layout, dtype):
        cpu_result = torch.hann_window(window_length, periodic=periodic, dtype=dtype, layout=layout, device="cpu")
        target_result = torch.hann_window(window_length, periodic=periodic, dtype=dtype, layout=layout, device=device)
        checkclose(target_result.cpu(), cpu_result)

    @onlyCUDA 
    @dtypesIfCUDA(*set(get_all_fp_dtypes(include_half=False, include_bfloat16=False)))  
    @parametrize("window_length", [1, 3, 5, 10])
    @parametrize("periodic", [True, False])
    @parametrize("beta", [3.0, 5.0, 10.0, 12.0, 15.0])
    @parametrize("layout", [torch.strided])
    def test_kaiser_window(self, device, window_length, periodic, beta, layout, dtype):
        cpu_result = torch.kaiser_window(window_length, periodic=periodic, beta=beta, dtype=dtype, layout=layout, device="cpu")
        target_result = torch.kaiser_window(window_length, periodic=periodic, beta=beta, dtype=dtype, layout=layout, device=device)
        checkclose(target_result.cpu(), cpu_result)

class TestReductionOps(TestCase):
    # TODO: consider refactoring with bincount test
    @onlyCUDA
    def test_bucketization(self, device):
        values_1d = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9], device=device)
        values_3d = torch.tensor([[[1, 3, 5], [2, 4, 6]], [[1, 2, 3], [4, 5, 6]]], device=device)

        # regular case 3d boundary and 3d input value
        boundaries = torch.tensor([[[1, 2, 3, 4], [3, 4, 5, 6]], [[1, 3, 5, 7], [2, 4, 6, 8]]], device=device)
        expected_result = torch.tensor([[[0, 2, 4], [0, 1, 3]], [[0, 1, 1], [1, 2, 2]]], device=device)
        output = torch.empty(2, 2, 3, device=device, dtype=torch.int64)
        self.assertEqual(torch.searchsorted(boundaries, values_3d), expected_result)
        self.assertEqual(torch.searchsorted(boundaries, values_3d, out=output), expected_result)
        expected_result = torch.tensor([[[1, 3, 4], [0, 2, 4]], [[1, 1, 2], [2, 2, 3]]], device=device)
        self.assertEqual(torch.searchsorted(boundaries, values_3d, right=True), expected_result)
        self.assertEqual(torch.searchsorted(boundaries, values_3d, right=True, out=output), expected_result)

        # simple 1d boundary and 3d input value
        boundaries = torch.tensor([1, 2, 3, 4, 5, 6], device=device)
        expected_result = torch.tensor([[[0, 2, 4], [1, 3, 5]], [[0, 1, 2], [3, 4, 5]]], device=device)
        output = torch.empty(2, 2, 3, device=device, dtype=torch.int64)
        self.assertEqual(torch.searchsorted(boundaries, values_3d), expected_result)
        self.assertEqual(torch.bucketize(values_3d, boundaries), expected_result)
        self.assertEqual(torch.bucketize(values_3d, boundaries, out=output), expected_result)
        expected_result = torch.tensor([[[1, 3, 5], [2, 4, 6]], [[1, 2, 3], [4, 5, 6]]], device=device)
        self.assertEqual(torch.searchsorted(boundaries, values_3d, right=True), expected_result)
        self.assertEqual(torch.bucketize(values_3d, boundaries, right=True), expected_result)
        self.assertEqual(torch.bucketize(values_3d, boundaries, out=output, right=True), expected_result)

        # simple float 1d boundary and 1d input with output int32 type
        values_1d_float = values_1d.to(torch.float32)
        boundaries = torch.tensor([0.9, 1, 2, 2, 3, 3, 4, 4.1, 9, 9], device=device, dtype=torch.float32)
        expected_result = torch.tensor([1, 2, 4, 6, 8, 8, 8, 8, 8], device=device, dtype=torch.int32)
        self.assertEqual(torch.searchsorted(boundaries, values_1d_float, out_int32=True), expected_result)
        self.assertEqual(torch.bucketize(values_1d_float, boundaries, out_int32=True), expected_result)

        # multiple dimension input with 0 elements
        boundaries = torch.tensor([1, 2, 3, 4, 5, 6], device=device, dtype=torch.int64)
        values_0_el = torch.tensor([[[]]], device=device, dtype=torch.int64)
        expected_result = values_0_el.to(torch.int64)
        self.assertEqual(torch.searchsorted(boundaries, values_0_el), expected_result)
        self.assertEqual(torch.bucketize(values_0_el, boundaries), expected_result)

        # nan input
        values_nan = torch.tensor([1.0, float('nan'), 2.0, float('nan')], device=device, dtype=torch.float64)
        boundaries = torch.tensor([0.0, 1.0, 2.0, 3.0], device=device, dtype=torch.float64)
        expected_result = torch.tensor([1, 4, 2, 4], device=device)
        self.assertEqual(torch.searchsorted(boundaries, values_nan), expected_result)
        expected_result = torch.tensor([2, 4, 3, 4], device=device)
        self.assertEqual(torch.searchsorted(boundaries, values_nan, right=True), expected_result)

        # type promotion and non contiguous tensors
        values_3d_permute = values_3d.permute(2, 1, 0).to(torch.int32)
        boundaries_permute = values_3d.permute(2, 1, 0).to(torch.float64)
        expected_result = torch.tensor([[[0, 0], [0, 1]], [[2, 0], [0, 1]], [[2, 0], [0, 0]]], device=device)
        if self.device_type != 'xla':
            self.assertWarnsRegex(
                UserWarning, "tensor is non-contiguous",
                lambda: self.assertEqual(torch.searchsorted(boundaries_permute, values_3d_permute), expected_result))
        else:
            # All tensors in XLA is contiguous even doing permute, no warning msg will be generate in XLA
            self.assertEqual(torch.searchsorted(boundaries_permute, values_3d_permute), expected_result)

        # scalar type
        boundaries = torch.tensor([1.5, 2.5, 3.5], device=device)
        expected_result = torch.tensor(1, device=device)
        self.assertEqual(torch.searchsorted(boundaries, 2), expected_result)
        self.assertEqual(torch.bucketize(torch.tensor(2, device=device), boundaries), expected_result)
        expected_result = torch.tensor(3, device=device)
        scalar_tensor_nan = torch.tensor(float('nan'), device=device)
        self.assertEqual(torch.searchsorted(boundaries, scalar_tensor_nan), expected_result)
        self.assertEqual(torch.bucketize(float('nan'), boundaries, right=True), expected_result)

        # invalid input dimensions
        boundaries = torch.tensor([[1, 2, 3], [4, 5, 6]], device=device)
        with self.assertRaisesRegex(
                RuntimeError, "first N-1 dimensions of boundaries tensor and input value tensor must match"):
            torch.searchsorted(boundaries, values_3d)
        with self.assertRaisesRegex(
                RuntimeError, "boundaries tensor must be 1 dimension"):
            torch.bucketize(values_3d, boundaries)
        with self.assertRaisesRegex(
                RuntimeError, "only when boundaries tensor dimension is 1"):
            torch.searchsorted(boundaries, 1)

        # incompatiable output tensor's dtype
        def test_output_dtype(dtype, is_int32):
            output = values_1d.to(dtype)
            with self.assertRaisesRegex(
                    RuntimeError, "output tensor's dtype is wrong"):
                torch.searchsorted(values_1d, values_1d, out=output, out_int32=is_int32)

        test_output_dtype(torch.float32, False)
        test_output_dtype(torch.int32, False)
        test_output_dtype(torch.int64, True)

        # scalar type bfloat16
        if self.device_type == 'cpu':
            def test_dtype_bfloat16(values_bf16=False, boundaries_bf16=False):
                values_1d_float = values_1d.to(torch.float32)
                boundaries = torch.tensor([0.9, 1, 2, 2, 3, 3, 4, 4.1, 9, 9], device=device, dtype=torch.float32)
                if values_bf16:
                    values_1d_float = values_1d_float.to(torch.bfloat16)
                if boundaries_bf16:
                    boundaries = boundaries.to(torch.bfloat16)
                expected_result = torch.tensor([1, 2, 4, 6, 8, 8, 8, 8, 8], device=device, dtype=torch.int32)
                self.assertEqual(torch.searchsorted(boundaries, values_1d_float, out_int32=True), expected_result)
                self.assertEqual(torch.bucketize(values_1d_float, boundaries, out_int32=True), expected_result)

            test_dtype_bfloat16(True, False)
            test_dtype_bfloat16(False, True)
            test_dtype_bfloat16(True, True)

    @onlyCUDA  
    def test_cartesian_prod(self, device):
        a = torch.tensor([1], device=device)
        b = torch.tensor([1, 2, 3], device=device)
        c = torch.tensor([1, 2], device=device)
        prod = torch.cartesian_prod(a, b, c)
        expected = torch.tensor(list(product([a], b, c)), device=device)
        self.assertEqual(expected, prod)

        # test 0 size input
        d = torch.empty(0, dtype=b.dtype, device=device)
        prod = torch.cartesian_prod(a, b, c, d)
        expected = torch.empty(0, 4, dtype=b.dtype, device=device)
        self.assertEqual(expected, prod)

        # test single input
        prod = torch.cartesian_prod(b)
        self.assertEqual(b, prod)

    @onlyCUDA  
    def test_combinations(self, device):
        a = torch.tensor([1, 2, 3], device=device)

        c = torch.combinations(a, r=1)
        expected = torch.tensor(list(combinations(a, r=1)), device=device)
        self.assertEqual(c, expected)

        c = torch.combinations(a, r=1, with_replacement=True)
        expected = torch.tensor(list(combinations_with_replacement(a, r=1)), device=device)
        self.assertEqual(c, expected)

        c = torch.combinations(a)
        expected = torch.tensor(list(combinations(a, r=2)), device=device)
        self.assertEqual(c, expected)

        c = torch.combinations(a, with_replacement=True)
        expected = torch.tensor(list(combinations_with_replacement(a, r=2)), device=device)
        self.assertEqual(c, expected)

        c = torch.combinations(a, r=3)
        expected = torch.tensor(list(combinations(a, r=3)), device=device)
        self.assertEqual(c, expected)

        c = torch.combinations(a, r=4)
        expected = torch.empty(0, 4, dtype=a.dtype, device=device)
        self.assertEqual(c, expected)

        c = torch.combinations(a, r=5)
        expected = torch.empty(0, 5, dtype=a.dtype, device=device)
        self.assertEqual(c, expected)

        # test empty imput
        a = torch.empty(0, device=device)
        c1 = torch.combinations(a)
        c2 = torch.combinations(a, with_replacement=True)
        expected = torch.empty(0, 2, dtype=a.dtype, device=device)
        self.assertEqual(c1, expected)
        self.assertEqual(c2, expected)
    
    @onlyCUDA
    @dtypesIfCUDA(*product(get_all_dtypes(include_complex=False), get_all_dtypes(include_complex=False)))
    def test_maximum_minimum_type_promotion(self, device, dtypes):
        a = torch.tensor((0, 1), device=device, dtype=dtypes[0])
        b = torch.tensor((1, 0), device=device, dtype=dtypes[1])
        for op in (torch.maximum, torch.max, torch.fmax, torch.minimum, torch.min, torch.fmin):
            result = op(a, b)
            self.assertEqual(result.dtype, torch.result_type(a, b))

    @onlyCUDA
    @dtypesIfCUDA(*(get_all_int_dtypes() + [torch.bool]))
    def test_maximum_minimum_int_and_bool(self, device, dtype):
        ops = ((torch.maximum, torch.max, np.maximum), (torch.minimum, torch.min, np.minimum),
               (torch.fmax, None, np.fmax), (torch.fmin, None, np.fmin))
        rng = np.random.default_rng()
        a_np = np.array(rng.integers(-100, 100, size=10), dtype=torch_to_numpy_dtype_dict[dtype])
        b_np = np.array(rng.integers(-100, 100, size=10), dtype=torch_to_numpy_dtype_dict[dtype])

        for torch_op, alias, numpy_op in ops:
            a_tensor = torch.from_numpy(a_np).to(device=device, dtype=dtype)
            b_tensor = torch.from_numpy(b_np).to(device=device, dtype=dtype)
            tensor_result = torch_op(a_tensor, b_tensor)

            out = torch.empty_like(a_tensor)
            torch_op(a_tensor, b_tensor, out=out)

            numpy_result = numpy_op(a_np, b_np)

            if alias is not None:
                alias_result = alias(a_tensor, b_tensor)
                self.assertEqual(alias_result, tensor_result)

            self.assertEqual(tensor_result, numpy_result)
            self.assertEqual(out, numpy_result)

    @precisionOverride({torch.bfloat16: 1e-2})
    @onlyCUDA
    @dtypesIfCUDA(*(get_all_fp_dtypes()))
    def test_maximum_minimum_float(self, device, dtype):
        ops = ((torch.maximum, torch.max, np.maximum), (torch.minimum, torch.min, np.minimum),
               (torch.fmax, None, np.fmax), (torch.fmin, None, np.fmin))

        if dtype == torch.bfloat16:
            a_np = np.random.randn(10).astype(np.float64)
            b_np = np.random.randn(10).astype(np.float64)
        else:
            a_np = np.random.randn(10).astype(torch_to_numpy_dtype_dict[dtype])
            b_np = np.random.randn(10).astype(torch_to_numpy_dtype_dict[dtype])

        for torch_op, alias, numpy_op in ops:
            numpy_result = numpy_op(a_np, b_np)

            a_tensor = torch.from_numpy(a_np).to(device=device, dtype=dtype)
            b_tensor = torch.from_numpy(b_np).to(device=device, dtype=dtype)
            tensor_result = torch_op(a_tensor, b_tensor)
            out = torch.empty_like(a_tensor)
            torch_op(a_tensor, b_tensor, out=out)

            if alias is not None:
                alias_result = alias(a_tensor, b_tensor)
                self.assertEqual(alias_result, tensor_result, exact_dtype=False)
            self.assertEqual(tensor_result, numpy_result, exact_dtype=False)
            self.assertEqual(out, numpy_result, exact_dtype=False)

    @dtypesIfCUDA(*(get_all_fp_dtypes()))
    def test_maximum_minimum_float_nan_and_inf(self, device, dtype):
        # np.maximum and np.minimum functions compare input arrays element-wisely.
        # if one of the elements being compared is a NaN, then that element is returned.
        ops = ((torch.maximum, torch.max, np.maximum), (torch.minimum, torch.min, np.minimum),
               (torch.fmax, None, np.fmax), (torch.fmin, None, np.fmin))
        a_vals = (float('inf'), -float('inf'), float('nan'), float('inf'), float('nan'), float('nan'), 1, float('nan'))
        b_vals = (-float('inf'), float('inf'), float('inf'), float('nan'), float('nan'), 0, float('nan'), -5)
        if dtype == torch.bfloat16:
            a_np = np.array(a_vals, dtype=np.float64)
            b_np = np.array(b_vals, dtype=np.float64)
        else:
            a_np = np.array(a_vals, dtype=torch_to_numpy_dtype_dict[dtype])
            b_np = np.array(b_vals, dtype=torch_to_numpy_dtype_dict[dtype])

        for torch_op, alias, numpy_op in ops:
            numpy_result = numpy_op(a_np, b_np)

            a_tensor = torch.from_numpy(a_np).to(device=device, dtype=dtype)
            b_tensor = torch.from_numpy(b_np).to(device=device, dtype=dtype)
            tensor_result = torch_op(a_tensor, b_tensor)

            out = torch.empty_like(a_tensor)
            torch_op(a_tensor, b_tensor, out=out)

            if alias is not None:
                alias_result = alias(a_tensor, b_tensor)
                self.assertEqual(alias_result, tensor_result)

            if dtype == torch.bfloat16:
                self.assertEqual(tensor_result, numpy_result, exact_dtype=False)
                self.assertEqual(out, numpy_result, exact_dtype=False)
            else:
                self.assertEqual(tensor_result, numpy_result)
                self.assertEqual(out, numpy_result)

    @dtypesIfCUDA(*product(get_all_complex_dtypes(), get_all_dtypes()))
    def test_maximum_minimum_complex(self, device, dtypes):
        for torch_op in (torch.maximum, torch.minimum, torch.max, torch.min, torch.fmax, torch.fmin):
            with self.assertRaisesRegex(RuntimeError, '.+not implemented for.+'):
                torch_op(torch.ones(1, device=device, dtype=dtypes[0]),
                         torch.ones(1, device=device, dtype=dtypes[1]))

            with self.assertRaisesRegex(RuntimeError, '.+not implemented for.+'):
                torch_op(torch.ones(1, device=device, dtype=dtypes[1]),
                         torch.ones(1, device=device, dtype=dtypes[0]))

    @onlyCUDA
    def test_maximum_minimum_cross_device(self, device):
        a = torch.tensor((1, 2, -1))
        b = torch.tensor((3, 0, 4), device=device)
        ops = (torch.maximum, torch.minimum)

        for torch_op in ops:
            with self.assertRaisesRegex(RuntimeError,
                                        "Expected all tensors to be on the same device"):
                torch_op(a, b)

            with self.assertRaisesRegex(RuntimeError,
                                        "Expected all tensors to be on the same device"):
                torch_op(b, a)

        # test cuda tensor and cpu scalar
        ops = ((torch.maximum, np.maximum), (torch.minimum, np.minimum))
        a_np = np.array(1)
        b_np = np.array([3, 0, 4])

        for torch_op, numpy_op in ops:
            a_tensor = torch.from_numpy(a_np)
            b_tensor = torch.from_numpy(b_np).to(device=device)
            tensor_result_1 = torch_op(a_tensor, b_tensor)
            numpy_result_1 = numpy_op(a_np, b_np)
            tensor_result_2 = torch_op(b_tensor, a_tensor)
            numpy_result_2 = numpy_op(b_np, a_np)

            self.assertEqual(tensor_result_1, numpy_result_1)
            self.assertEqual(tensor_result_2, numpy_result_2)
    
    # TODO: bincount isn't a classic reduction -- maybe this test suite is
    #   reductions and summary ops?
    @onlyCUDA
    def test_bincount(self, device):
        # negative input throws
        with self.assertRaisesRegex(RuntimeError, '1-d non-negative integral'):
            torch.bincount(torch.tensor([1, -1], device=device))
        # n-d input, with n > 1 throws
        with self.assertRaisesRegex(RuntimeError, '1-d non-negative integral'):
            torch.bincount(torch.tensor([[1, 2], [3, 4]], device=device))
        # floating input type throws
        with self.assertRaisesRegex(RuntimeError, 'not implemented'):
            torch.bincount(torch.tensor([1., 0.3], device=device))
        # minlength < 0 throws
        with self.assertRaisesRegex(RuntimeError, 'minlength should be >= 0'):
            torch.bincount(torch.tensor([1, 3], device=device),
                           torch.tensor([.2, .2], device=device),
                           minlength=-1)
        # input and weights dim mismatch
        with self.assertRaisesRegex(RuntimeError, 'same length'):
            torch.bincount(torch.tensor([1, 0], device=device),
                           torch.tensor([1., 0.3, 0.5], device=device))
        # 1-d input with no elements and default minlength
        self.assertEqual(torch.bincount(torch.tensor([], device=device, dtype=torch.long)),
                         torch.zeros(0, dtype=torch.long, device=device))
        # 1-d input with no elements and specified minlength
        self.assertEqual(torch.bincount(torch.tensor([], device=device, dtype=torch.long), minlength=10),
                         torch.zeros(10, dtype=torch.long, device=device))

        # test tensor method without weights
        long_counts = torch.tensor(
            [0, 3, 2, 1, 3], dtype=torch.uint8, device=device).bincount()
        self.assertEqual(
            torch.tensor([1, 1, 1, 2], dtype=torch.int64, device=device),
            long_counts)
        # test minlength functionality
        int_counts = torch.bincount(
            torch.tensor([1, 1, 1, 1], device=device), minlength=5)
        self.assertEqual(
            torch.tensor([0, 4, 0, 0, 0], dtype=torch.int64, device=device),
            int_counts)
        # test weights
        byte_counts = torch.bincount(
            torch.tensor([0, 1, 1, 1, 4], device=device),
            torch.tensor([.1, .2, .3, .4, .5], device=device))
        self.assertEqual(
            torch.tensor([0.1, 0.9, 0, 0, 0.5], device=device), byte_counts)
        byte_counts = torch.bincount(
            torch.tensor([0, 1, 1, 1, 4], device=device),
            torch.tensor([1, 2, 3, 4, 5], dtype=torch.int8, device=device))
        self.assertEqual(
            torch.tensor([1, 9, 0, 0, 5], device=device, dtype=torch.float64), byte_counts)
        # test non-contiguous inputs and weights
        inputs = torch.tensor([[0, 0], [3, 1], [2, 1], [1, 1], [3, 4]], device=device)
        weights = torch.tensor([[.1, 1], [.2, 2], [.3, 3], [.4, 4], [.5, 5]], device=device)
        for i in [0, 1]:
            assert not inputs[:, i].is_contiguous(), "Inputs are supposed to be non-contiguous"
            assert not weights[:, i].is_contiguous(), "Weights are supposed to be non-contiguous"
        # inputs are non-contiguous but weights are contiguous
        self.assertEqual(inputs[:, 0].bincount(), torch.tensor([1, 1, 1, 2]))
        # inputs and weights are non-contiguous
        self.assertEqual(
            inputs[:, 1].bincount(weights[:, 1]),
            torch.tensor([1, 9, 0, 0, 5], dtype=torch.float32))
        # weights are non-contiguous but inputs are contiguous
        self.assertEqual(inputs[:, 1].contiguous().bincount(weights[:, 1]),
                         torch.tensor([1, 9, 0, 0, 5], dtype=torch.float32))

        # test bincount on non-contiguous slices
        all0s = torch.zeros((32, 2), dtype=torch.int64, device=device)
        self.assertEqual(all0s[:, 0].bincount(), torch.tensor([32]))

        all1s = torch.ones((32, 2), dtype=torch.int64, device=device)
        self.assertEqual(all1s[:, 0].bincount(), torch.tensor([0, 32]))

        # test large number of bins - global memory use
        big_exp = torch.zeros(10000000, device=device)
        big_exp[-1] = 50.0
        big_w = torch.tensor([.5] * 100, device=device)
        big_out = torch.tensor([9999999] * 100, device=device).bincount(big_w)
        self.assertEqual(big_exp, big_out)
        # test large input size
        big_exp = torch.zeros(2, device=device, dtype=torch.int64)
        big_exp[1] = 1000000
        big_out = torch.ones(1000000, dtype=torch.int8, device=device).bincount()
        self.assertEqual(big_exp, big_out)

class TestAtleastNdOps(TestCase):
    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_fp_dtypes())) 
    def test_atleast_1d(self, device, dtype):
        input = torch.arange(2, dtype=dtype, device=device)
        self.assertEqual(torch.atleast_1d(input).cpu(), torch.tensor([0, 1]).to(dtype))

        input = torch.tensor(1., dtype=dtype, device=device)
        self.assertEqual(torch.atleast_1d(input).cpu(), torch.tensor([1.]).to(dtype))

        input1 = torch.tensor(1., dtype=dtype, device=device)
        input2 = torch.tensor(0.5, dtype=dtype, device=device)
        result = torch.atleast_1d((input1, input2))
        self.assertTrue(isinstance(result, tuple) and len(result) == 2)
        self.assertEqual(result[0].cpu(), torch.tensor([1.]).to(dtype))
        self.assertEqual(result[1].cpu(), torch.tensor([0.5]).to(dtype))

    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_fp_dtypes())) 
    def test_atleast_2d(self, device, dtype):
        input = torch.arange(2, dtype=dtype, device=device)
        self.assertEqual(torch.atleast_2d(input).cpu(), torch.tensor([[0, 1]]).to(dtype))

        input = torch.tensor(1., dtype=dtype, device=device)
        self.assertEqual(torch.atleast_2d(input).cpu(), torch.tensor([[1.]]).to(dtype))

        input1 = torch.tensor(1., dtype=dtype, device=device)
        input2 = torch.tensor(0.5, dtype=dtype, device=device)
        result = torch.atleast_2d((input1, input2))
        self.assertTrue(isinstance(result, tuple) and len(result) == 2)
        self.assertEqual(result[0].cpu(), torch.tensor([[1.]]).to(dtype))
        self.assertEqual(result[1].cpu(), torch.tensor([[0.5]]).to(dtype))

    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_fp_dtypes())) 
    def test_atleast_3d(self, device, dtype):
        input = torch.arange(4, dtype=dtype, device=device).view(2, 2)
        self.assertEqual(torch.atleast_3d(input).cpu(), torch.tensor([[[0], [1]], [[2], [3]]]).to(dtype))

        input = torch.tensor(1., dtype=dtype, device=device)
        self.assertEqual(torch.atleast_3d(input).cpu(), torch.tensor([[[1.]]]).to(dtype))

        input1 = torch.tensor(1., dtype=dtype, device=device)
        input2 = torch.tensor(0.5, dtype=dtype, device=device)
        result = torch.atleast_3d((input1, input2))
        self.assertTrue(isinstance(result, tuple) and len(result) == 2)
        self.assertEqual(result[0].cpu(), torch.tensor([[[1.]]]).to(dtype))
        self.assertEqual(result[1].cpu(), torch.tensor([[[0.5]]]).to(dtype))


class TestSortAndSelect(TestCase):
    def assertIsOrdered(self, order, x, mxx, ixx, task):
        SIZE = x.size(1)
        if order == 'descending':
            def check_order(a, b):
                # `a != a` because we put NaNs
                # at the end of ascending sorted lists,
                # and the beginning of descending ones.
                return ((a != a) | (a >= b)).all().item()
        elif order == 'ascending':
            def check_order(a, b):
                # see above
                return ((b != b) | (a <= b)).all().item()
        else:
            error('unknown order "{}", must be "ascending" or "descending"'.format(order))

        are_ordered = True
        for k in range(1, SIZE):
            self.assertTrue(check_order(mxx[:, k - 1], mxx[:, k]),
                            'torch.sort ({}) values unordered for {}'.format(order, task))

        seen = set()
        indicesCorrect = True
        size0 = x.size(0)
        size = x.size(x.dim() - 1)
        x = x.tolist()
        mxx = mxx.tolist()
        ixx = ixx.tolist()
        for k in range(size0):
            seen.clear()
            for j in range(size):
                self.assertEqual(x[k][ixx[k][j]], mxx[k][j],
                                 msg='torch.sort ({}) indices wrong for {}'.format(order, task))
                seen.add(ixx[k][j])
            self.assertEqual(len(seen), size)

    def test_sort(self, device):
        # on CUDA 2048 vs >2048 have different code path for the dim being sorted
        for SIZE in (4, 2049):
            x = torch.rand(4, SIZE, device=device)
            res1val, res1ind = torch.sort(x)

            # Test inplace
            y = x.clone()
            y_inds = torch.tensor((), dtype=torch.int64, device=device)
            torch.sort(y, out=(y, y_inds))
            x_vals, x_inds = torch.sort(x)
            self.assertEqual(x_vals, y)
            self.assertEqual(x_inds, y_inds)

            # Test use of result tensor
            res2val = torch.tensor((), device=device)
            res2ind = torch.tensor((), device=device, dtype=torch.long)
            torch.sort(x, out=(res2val, res2ind))
            self.assertEqual(res1val, res2val, atol=0, rtol=0)
            self.assertEqual(res1ind, res2ind, atol=0, rtol=0)
            self.assertEqual(torch.argsort(x), res1ind)
            self.assertEqual(x.argsort(), res1ind)

            # Test sorting of random numbers
            self.assertIsOrdered('ascending', x, res2val, res2ind, 'random')

            # Test simple sort
            self.assertEqual(
                torch.sort(torch.tensor((50, 40, 30, 20, 10), device=device))[0],
                torch.tensor((10, 20, 30, 40, 50), device=device),
                atol=0, rtol=0
            )

            # Test that we still have proper sorting with duplicate keys
            x = torch.floor(torch.rand(4, SIZE, device=device) * 10)
            torch.sort(x, out=(res2val, res2ind))
            self.assertIsOrdered('ascending', x, res2val, res2ind, 'random with duplicate keys')

            # DESCENDING SORT
            x = torch.rand(4, SIZE, device=device)
            res1val, res1ind = torch.sort(x, x.dim() - 1, True)

            # Test use of result tensor
            res2val = torch.tensor((), device=device)
            res2ind = torch.tensor((), device=device, dtype=torch.long)
            torch.sort(x, x.dim() - 1, True, out=(res2val, res2ind))
            self.assertEqual(res1val, res2val, atol=0, rtol=0)
            self.assertEqual(res1ind, res2ind, atol=0, rtol=0)
            self.assertEqual(torch.argsort(x, x.dim() - 1, True), res1ind)
            self.assertEqual(x.argsort(x.dim() - 1, True), res1ind)

            # Test sorting of random numbers
            self.assertIsOrdered('descending', x, res2val, res2ind, 'random')

            # Test simple sort task
            self.assertEqual(
                torch.sort(torch.tensor((10, 20, 30, 40, 50), device=device), 0, True)[0],
                torch.tensor((50, 40, 30, 20, 10), device=device),
                atol=0, rtol=0
            )

            # Test that we still have proper sorting with duplicate keys
            self.assertIsOrdered('descending', x, res2val, res2ind, 'random with duplicate keys')

            # Test sorting with NaNs
            x = torch.rand(4, SIZE, device=device)
            x[1][2] = float('NaN')
            x[3][0] = float('NaN')
            torch.sort(x, out=(res2val, res2ind))
            self.assertIsOrdered('ascending', x, res2val, res2ind,
                                 'random with NaNs')
            torch.sort(x, out=(res2val, res2ind), descending=True)
            self.assertIsOrdered('descending', x, res2val, res2ind,
                                 'random with NaNs')

    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_dtypes(include_complex=False, include_bool=False, include_bfloat16=False)))
    def test_isin(self, device, dtype):
        def assert_isin_equal(a, b):
            # Compare to the numpy reference implementation.
            x = torch.isin(a, b)
            a = a.cpu().numpy() if torch.is_tensor(a) else np.array(a)
            b = b.cpu().numpy() if torch.is_tensor(b) else np.array(b)
            y = np.isin(a, b)
            self.assertEqual(x, y)

        # multi-dim tensor, multi-dim tensor
        a = torch.arange(24, device=device, dtype=dtype).reshape([2, 3, 4])
        b = torch.tensor([[10, 20, 30], [0, 1, 3], [11, 22, 33]], device=device, dtype=dtype)
        assert_isin_equal(a, b)

        # zero-dim tensor
        zero_d = torch.tensor(3, device=device, dtype=dtype)
        assert_isin_equal(zero_d, b)
        assert_isin_equal(a, zero_d)
        assert_isin_equal(zero_d, zero_d)

        # empty tensor
        empty = torch.tensor([], device=device, dtype=dtype)
        assert_isin_equal(empty, b)
        assert_isin_equal(a, empty)
        assert_isin_equal(empty, empty)

        # scalar
        assert_isin_equal(a, 6)
        assert_isin_equal(5, b)

        def define_expected(lst, invert=False):
            expected = torch.tensor(lst, device=device)
            if invert:
                expected = expected.logical_not()
            return expected

        # Adapted from numpy's in1d tests
        for mult in [1, 10]:
            for invert in [False, True]:
                a = torch.tensor([5, 7, 1, 2], device=device, dtype=dtype)
                b = torch.tensor([2, 4, 3, 1, 5] * mult, device=device, dtype=dtype)
                ec = define_expected([True, False, True, True], invert=invert)
                c = torch.isin(a, b, assume_unique=True, invert=invert)
                self.assertEqual(c, ec)

                a[0] = 8
                ec = define_expected([False, False, True, True], invert=invert)
                c = torch.isin(a, b, assume_unique=True, invert=invert)
                self.assertEqual(c, ec)

                a[0], a[3] = 4, 8
                ec = define_expected([True, False, True, False], invert=invert)
                c = torch.isin(a, b, assume_unique=True, invert=invert)
                self.assertEqual(c, ec)

                a = torch.tensor([5, 4, 5, 3, 4, 4, 3, 4, 3, 5, 2, 1, 5, 5], device=device, dtype=dtype)
                b = torch.tensor([2, 3, 4] * mult, device=device, dtype=dtype)
                ec = define_expected([False, True, False, True, True, True, True, True, True,
                                      False, True, False, False, False], invert=invert)
                c = torch.isin(a, b, invert=invert)
                self.assertEqual(c, ec)

                b = torch.tensor([2, 3, 4] * mult + [5, 5, 4] * mult, device=device, dtype=dtype)
                ec = define_expected([True, True, True, True, True, True, True, True, True, True,
                                      True, False, True, True], invert=invert)
                c = torch.isin(a, b, invert=invert)
                self.assertEqual(c, ec)

                a = torch.tensor([5, 7, 1, 2], device=device, dtype=dtype)
                b = torch.tensor([2, 4, 3, 1, 5] * mult, device=device, dtype=dtype)
                ec = define_expected([True, False, True, True], invert=invert)
                c = torch.isin(a, b, invert=invert)
                self.assertEqual(c, ec)

                a = torch.tensor([5, 7, 1, 1, 2], device=device, dtype=dtype)
                b = torch.tensor([2, 4, 3, 3, 1, 5] * mult, device=device, dtype=dtype)
                ec = define_expected([True, False, True, True, True], invert=invert)
                c = torch.isin(a, b, invert=invert)
                self.assertEqual(c, ec)

                a = torch.tensor([5, 5], device=device, dtype=dtype)
                b = torch.tensor([2, 2] * mult, device=device, dtype=dtype)
                ec = define_expected([False, False], invert=invert)
                c = torch.isin(a, b, invert=invert)
                self.assertEqual(c, ec)

                # multi-dimensional input case using sort-based algo
                for assume_unique in [False, True]:
                    a = torch.arange(6, device=device, dtype=dtype).reshape([2, 3])
                    b = torch.arange(3, 30, device=device, dtype=dtype)
                    ec = define_expected([[False, False, False], [True, True, True]], invert=invert)
                    c = torch.isin(a, b, invert=invert, assume_unique=assume_unique)
                    self.assertEqual(c, ec)
        
    @dtypes(torch.double)
    @onlyCUDA
    def test_kthvalue(self, device, dtype):
        SIZE = 50
        x = torch.rand(SIZE, SIZE, SIZE, dtype=dtype, device=device)
        x0 = x.clone()

        k = random.randint(1, SIZE)
        res1val, res1ind = torch.kthvalue(x, k, keepdim=False)
        res2val, res2ind = torch.sort(x)

        self.assertEqual(res1val[:, :], res2val[:, :, k - 1], atol=0, rtol=0)
        self.assertEqual(res1ind[:, :], res2ind[:, :, k - 1], atol=0, rtol=0)
        # test use of result tensors
        k = random.randint(1, SIZE)
        res1val = torch.tensor([], dtype=dtype, device=device)
        res1ind = torch.tensor([], dtype=torch.long, device=device)
        torch.kthvalue(x, k, keepdim=False, out=(res1val, res1ind))
        res2val, res2ind = torch.sort(x)
        self.assertEqual(res1val[:, :], res2val[:, :, k - 1], atol=0, rtol=0)
        self.assertEqual(res1ind[:, :], res2ind[:, :, k - 1], atol=0, rtol=0)

        # test non-default dim
        k = random.randint(1, SIZE)
        res1val, res1ind = torch.kthvalue(x, k, 0, keepdim=False)
        res2val, res2ind = torch.sort(x, 0)
        self.assertEqual(res1val, res2val[k - 1], atol=0, rtol=0)
        self.assertEqual(res1ind, res2ind[k - 1], atol=0, rtol=0)

        # non-contiguous
        y = x.narrow(1, 0, 1)
        y0 = y.contiguous()
        k = random.randint(1, SIZE)
        res1val, res1ind = torch.kthvalue(y, k)
        res2val, res2ind = torch.kthvalue(y0, k)
        self.assertEqual(res1val, res2val, atol=0, rtol=0)
        self.assertEqual(res1ind, res2ind, atol=0, rtol=0)

        # non-contiguous [Reference: https://github.com/pytorch/pytorch/issues/45721]
        non_contig_t = torch.tensor([0, -1, 1, -2, 2], dtype=dtype, device=device)[::2]
        expected_val, expected_ind = non_contig_t.contiguous().kthvalue(2)
        non_contig_cpu_t = non_contig_t.cpu()
        expected_val_cpu, expected_ind_cpu = non_contig_cpu_t.kthvalue(2)

        out_val, out_ind = non_contig_t.kthvalue(2)
        self.assertEqual(expected_val, out_val, atol=0, rtol=0)
        self.assertEqual(expected_ind, out_ind, atol=0, rtol=0)
        self.assertEqual(expected_val_cpu, out_val, atol=0, rtol=0)
        self.assertEqual(expected_ind_cpu, out_ind, atol=0, rtol=0)

        # check that the input wasn't modified
        self.assertEqual(x, x0, atol=0, rtol=0)

        # simple test case (with repetitions)
        y = torch.tensor((3., 5, 4, 1, 1, 5), dtype=dtype, device=device)
        self.assertEqual(torch.kthvalue(y, 3)[0], 3, atol=0, rtol=0)
        self.assertEqual(torch.kthvalue(y, 2)[0], 1, atol=0, rtol=0)

        # simple test case (with NaN)
        SIZE = 50
        x = torch.rand(SIZE, SIZE, SIZE, dtype=dtype, device=device)
        x[torch.arange(SIZE), :, torch.randint(50, (50,))] = float('nan')
        ks = [random.randint(1, SIZE), 1, SIZE, SIZE - 1]
        res2val, res2ind = torch.sort(x)
        for k in ks:
            res1val, res1ind = torch.kthvalue(x, k, keepdim=False)
            self.assertEqual(res1val[:, :], res2val[:, :, k - 1], atol=0, rtol=0)
            self.assertEqual(res1ind[:, :], res2ind[:, :, k - 1], atol=0, rtol=0)

    # test overlapping output
    @dtypes(torch.double)
    @onlyCUDA   # Fails on XLA
    def test_kthvalue_overlap(self, device, dtype):
        S = 10
        k = 5
        a = torch.randn(S, device=device)
        indices = torch.empty((), device=device, dtype=torch.long)
        with self.assertRaisesRegex(RuntimeError, "unsupported operation:"):
            torch.kthvalue(a, k, out=(a, indices))

    @dtypes(torch.float)
    @onlyCUDA   # Fails on XLA
    def test_kthvalue_scalar(self, device, dtype):
        # Test scalar input (test case from https://github.com/pytorch/pytorch/issues/30818)
        # Tests that passing a scalar tensor or 1D tensor with 1 element work either way
        res = torch.tensor(2, device=device, dtype=dtype).kthvalue(1)
        ref = torch.tensor([2], device=device, dtype=dtype).kthvalue(1)
        self.assertEqual(res[0], ref[0].squeeze())
        self.assertEqual(res[1], ref[1].squeeze())

    @dtypes(*(get_all_int_dtypes() + get_all_fp_dtypes()))
    @onlyCUDA
    @parametrize("shape", [[], (0, ), (20, ), (1, 20), (30, 30), (10, 20, 30)], name_fn=NORM_NAME)
    def test_msort(self, dtype, device, shape):
        tensor = make_tensor(shape, device, dtype, low=-9, high=9)
        if tensor.size() != torch.Size([]):
            if dtype is torch.bfloat16:
                expected = torch.from_numpy(np.msort(tensor.float().cpu().numpy())).bfloat16()
            else:
                expected = torch.from_numpy(np.msort(tensor.cpu().numpy()))
        else:
            expected = tensor  # numpy.msort() does not support empty shapes tensor

        result = torch.msort(tensor)
        self.assertEqual(result, expected)

        out = torch.empty_like(result)
        torch.msort(tensor, out=out)
        self.assertEqual(out, expected)

class TestTensorCreation(TestCase):
    @onlyCUDA
    def test_block_diag(self, device):
        def block_diag_workaround(*arrs):
            arrs_expanded = []
            for a in arrs:
                if a.dim() == 2:
                    arrs_expanded.append(a)
                elif a.dim() == 1:
                    arrs_expanded.append(a.expand(1, a.size(0)))
                elif a.dim() == 0:
                    arrs_expanded.append(a.expand(1, 1))
            shapes = torch.tensor([a.shape for a in arrs_expanded], device=device)
            out = torch.zeros(
                torch.sum(shapes, dim=0).tolist(),
                dtype=arrs_expanded[0].dtype,
                device=device
            )
            r, c = 0, 0
            for i, (rr, cc) in enumerate(shapes):
                out[r:r + rr, c:c + cc] = arrs_expanded[i]
                r += rr
                c += cc
            return out

        tensors = [
            torch.rand((2, 2), device=device),
            torch.rand((2, 3), device=device),
            torch.rand(10, device=device),
            torch.rand((8, 1), device=device),
            torch.rand(1, device=device)[0]
        ]
        result = torch.block_diag(*tensors)
        result_check = block_diag_workaround(*tensors)
        self.assertEqual(result, result_check)

        tensor = torch.rand(1, device=device)[0]
        result = torch.block_diag(tensor)
        result_check = tensor.expand(1, 1)
        self.assertEqual(result, result_check)

        tensor = torch.rand(10, device=device)
        result = torch.block_diag(tensor)
        result_check = tensor.expand(1, tensor.size(0))
        self.assertEqual(result, result_check)

        result = torch.block_diag()
        result_check = torch.empty(1, 0, device=device)
        self.assertEqual(result, result_check)
        self.assertEqual(result.device.type, 'cpu')

        test_dtypes = [
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.float32,
            torch.float64,
            torch.complex64,
            torch.complex128
        ]
        # Test pairs of different dtypes
        for dtype1 in test_dtypes:
            for dtype2 in test_dtypes:
                a = torch.tensor(1, device=device, dtype=dtype1)
                b = torch.tensor(2, device=device, dtype=dtype2)
                result = torch.block_diag(a, b)
                result_dtype = torch.result_type(a, b)
                result_check = torch.tensor([[1, 0], [0, 2]], device=device, dtype=result_dtype)
                self.assertEqual(result, result_check)

        with self.assertRaisesRegex(
            RuntimeError,
            "torch.block_diag: Input tensors must have 2 or fewer dimensions. Input 1 has 3 dimensions"
        ):
            torch.block_diag(torch.tensor(5), torch.tensor([[[6]]]))

        with self.assertRaisesRegex(
            RuntimeError,
            "torch.block_diag: Input tensors must have 2 or fewer dimensions. Input 0 has 4 dimensions"
        ):
            torch.block_diag(torch.tensor([[[[6]]]]))

        if device != 'cpu':
            with self.assertRaisesRegex(
                RuntimeError,
                (
                    "torch.block_diag: input tensors must all be on the same device."
                    " Input 0 is on device cpu and input 1 is on device "
                )
            ):
                torch.block_diag(torch.ones(2, 2).cpu(), torch.ones(2, 2, device=device))


class TestBroadcast(TestCase):
    @onlyCUDA
    @dtypes(torch.float)
    def test_broadcast_tensors(self, device, dtype):
        x0 = torch.randn(2, 1, 3, dtype=dtype, device=device)
        x1 = torch.randn(3, dtype=dtype, device=device)
        x2 = torch.randn(3, 1, dtype=dtype, device=device)
        expected_size = (2, 3, 3)

        y0, y1, y2 = torch.broadcast_tensors(x0, x1, x2)
        self.assertTrue(y0.size() == expected_size)
        self.assertTrue(y1.size() == expected_size)
        self.assertTrue(y2.size() == expected_size)

    @onlyCUDA
    def test_broadcast_shapes(self, device):
        examples = [(), (1,), (2,), (1, 1), (3, 1), (3, 2), (4, 1, 1), (4, 3, 2)]
        for s0 in examples:
            x0 = torch.randn(s0)
            expected = torch.broadcast_tensors(x0)[0].shape
            actual = torch.broadcast_shapes(s0)
            self.assertEqual(expected, actual)

            for s1 in examples:
                x1 = torch.randn(s1)
                expected = torch.broadcast_tensors(x0, x1)[0].shape
                actual = torch.broadcast_shapes(s0, s1)
                self.assertEqual(expected, actual)

    # Skip BFloat16 since numpy does not support it
    @onlyCUDA
    @dtypes(*get_all_dtypes(include_bfloat16=False))
    def test_broadcast_to(self, device, dtype):
        def can_broadcast(s0, s1):
            # s0.dim() <= s1.dim(), reverse s0 and s1 to compare trailing dimension
            s0 = tuple(reversed(s0))
            s1 = tuple(reversed(s1))
            for i in range(len(s0)):
                if s0[i] != 1 and s0[i] != s1[i]:
                    return False
            return True

        sizes = (
            (), (1,), (2,), (1, 1), (3, 1), (3, 2), (4, 1, 1), (4, 3, 2)
        )
        for s0, s1 in combinations(sizes, r=2):
            t = make_tensor(s0, device, dtype, low=-9, high=9)
            t_np = t.cpu().numpy()

            if can_broadcast(s0, s1):
                res = torch.broadcast_to(t, s1)
                np_res = np.broadcast_to(t_np, s1)
                self.assertEqual(res, np_res)
            else:
                with self.assertRaisesRegex(RuntimeError,
                                            r"The expanded size of the tensor \(\d\) "
                                            r"must match the existing size \(\d\)"):
                    torch.broadcast_to(t, s1)

class TestCloneOp(TestCase):
    @onlyCUDA
    def test_copy_all_dtypes_and_devices(self, device):
        from copy import copy
        for dt in get_all_dtypes():
            x = torch.tensor([1, 2, 3, 4], dtype=dt, device=device)
            x_clone = x.clone()
            y = copy(x)
            y.fill_(1)
            # copy is a shallow copy, only copies the tensor view,
            # not the data
            self.assertEqual(x, y)

    @onlyCUDA
    def test_clone_all_dtypes_and_devices(self, device):
        for dt in get_all_dtypes():
            x = torch.tensor((1, 1), dtype=dt, device=device)
            y = x.clone()
            self.assertEqual(x, y)

    @onlyCUDA
    def test_clone_zero_stride_dim(self, device):
        # stride zero, size 1 axis, not contiguous
        x = torch.randn(10)
        y = x.as_strided([2, 1, 5], [1, 0, 2])
        self.assertEqual(y, y.clone())

    @onlyCUDA
    def test_clone_not_memory_dense(self):
        # github issue: https://github.com/pytorch/pytorch/issues/64176
        x = torch.randn(10, 8).t()[::2, ::2]
        y = x.clone()
        # should retain permutation after densification
        self.assertTrue(y.stride() == (1, 4))
    
class TestCoef(TestCase):
    def _generate_correlation_tensors(self, device, dtype):
        yield make_tensor((0, 0), device, dtype)
        yield make_tensor((1, 0), device, dtype)
        yield make_tensor((0, 1), device, dtype)
        yield make_tensor((2,), device, dtype)
        yield make_tensor((2, 1), device, dtype)
        yield make_tensor((2, 2), device, dtype)
        yield make_tensor((2, 3), device, dtype)
        yield make_tensor((5, 10), device, dtype)
        yield make_tensor((5, 10), device, dtype, noncontiguous=True)
        if dtype != torch.int:
            yield torch.tensor([0, -2, nan, 10.2, inf], dtype=dtype, device=device)

    @precisionOverride({torch.int: 1e-3, torch.float: 1e-3})
    @onlyCUDA
    @dtypes(torch.int, torch.float, torch.cfloat)
    def test_corrcoef(self, device, dtype):
        for x in self._generate_correlation_tensors(device, dtype):
            res = torch.corrcoef(x)
            ref = np.corrcoef(x.cpu().numpy())
            self.assertEqual(res, ref, exact_dtype=False, atol=1e-3, rtol=1e-3)

    @onlyCUDA
    @dtypes(torch.int, torch.float, torch.cfloat)
    def test_cov(self, device, dtype):
        def check(t, correction=1, fweights=None, aweights=None):
            res = torch.cov(t, correction=correction, fweights=fweights, aweights=aweights)
            t = t.cpu().numpy()
            fweights = fweights.cpu().numpy() if fweights is not None else None
            aweights = aweights.cpu().numpy() if aweights is not None else None
            ref = np.cov(t, ddof=correction, fweights=fweights, aweights=aweights)
            self.assertEqual(res, ref, atol=1e-02, rtol=1e-02, exact_dtype=False)

        for x in self._generate_correlation_tensors(device, dtype):
            check(x)
            num_observations = x.numel() if x.ndim < 2 else x.size(1)
            if num_observations > 0:
                fweights = torch.randint(1, 10, (num_observations,), device=device)
                aweights = make_tensor((num_observations,), device, torch.float, low=1)
                for correction, fw, aw in product([0, 1, 2], [None, fweights], [None, aweights]):
                    check(x, correction, fweights, aweights)

    @onlyCUDA
    def test_cov_error(self, device):
        def check(msg, *args, **kwargs):
            with self.assertRaisesRegex(RuntimeError, r'cov\(\):.*' + msg + r'.*'):
                torch.cov(*args, **kwargs)

        a = torch.rand(2)
        check(r'expected input to have two or fewer dimensions', torch.rand(2, 2, 2))
        check(r'expected fweights to have one or fewer dimensions', a, fweights=torch.rand(2, 2))
        check(r'expected aweights to have one or fewer dimensions', a, aweights=torch.rand(2, 2))
        check(r'expected fweights to have integral dtype', a, fweights=torch.rand(2))
        check(r'expected aweights to have floating point dtype', a, aweights=torch.tensor([1, 1]))
        check(r'expected fweights to have the same numel', a, fweights=torch.tensor([1]))
        check(r'expected aweights to have the same numel', a, aweights=torch.rand(1))
        check(r'fweights cannot be negative', a, fweights=torch.tensor([-1, -2]))
        check(r'aweights cannot be negative', a, aweights=torch.tensor([-1., -2.]))


instantiate_device_type_tests(TestComparisionOps, globals())
instantiate_device_type_tests(TestIsOps, globals())
instantiate_device_type_tests(TestWindowOps, globals())
instantiate_device_type_tests(TestReductionOps, globals())
instantiate_device_type_tests(TestAtleastNdOps, globals())
instantiate_device_type_tests(TestSortAndSelect, globals())
instantiate_device_type_tests(TestTensorCreation, globals())
instantiate_device_type_tests(TestBroadcast, globals())
instantiate_device_type_tests(TestCloneOp, globals())
instantiate_device_type_tests(TestCoef, globals())

if __name__ == "__main__":
    unittest.main()


