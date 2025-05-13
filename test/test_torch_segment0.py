import torch
from torch.testing._internal.common_utils import TestCase, iter_indices, random_fullrank_matrix_distinct_singular_value
from torch.testing._internal.common_device_type import precisionOverride, skipCUDAIf
from torch.testing._internal.common_cuda import tf32_on_and_off
from torch.testing import make_tensor
import unittest
from itertools import product
import scipy.integrate
import numpy as np
from common_util import *

# Skips a test on GPU if LAPACK is not available.
def skipCUDAIfNoLapack(fn):
    return skipCUDAIf(not torch._C.has_lapack, "PyTorch compiled without Lapack")(fn)

class TestTorchFunctionSegment0(TestCase):

    def assertRelativeEqual(self, x, y, eps=1e-3, exact_dtype=True):
        assert check_close_relative(x, y, eps, exact_dtype)

    # @dtypes(torch.float32, torch.cfloat, torch.half, torch.bfloat16)
    @dtypes(torch.float32,  torch.cfloat, torch.half)
    @precisionOverride({torch.cfloat: 1e-4, torch.float32: 5e-5})
    def test_dot(self, device, dtype):
        self._test_dot_vdot_vs_numpy(device, dtype, torch.dot, np.dot)


    @onlyCUDA
    @skipCUDAIfNoLapack
    @dtypes(torch.float64, torch.complex128)
    def test_eig(self, device, dtype):
        def run_test(shape, *, symmetric=False):
            from torch.testing._internal.common_utils import random_symmetric_matrix

            if not dtype.is_complex and symmetric:
                # for symmetric real-valued inputs eigenvalues and eigenvectors have imaginary part equal to zero
                # unlike NumPy the result is not cast to float32 or float64 dtype in this case
                a = random_symmetric_matrix(shape[-1], *shape[:-2], dtype=dtype, device=device)
            else:
                a = make_tensor(shape, dtype=dtype, device=device)
            actual = torch.linalg.eig(a)

            # compare with NumPy
            # the eigenvalues are not necessarily ordered
            # so order of NumPy and PyTorch can be different
            expected = np.linalg.eig(a.cpu().numpy())

            # sort NumPy output
            ind = np.argsort(expected[0], axis=-1)[::-1]
            expected = (np.take_along_axis(expected[0], ind, axis=-1), np.take_along_axis(expected[1], ind[:, None], axis=-1))

            # sort PyTorch output
            # torch.argsort doesn't work with complex inputs, NumPy sorting on CPU is used instead
            # RuntimeError: _th_sort not supported on CUDAType for ComplexDouble
            # RuntimeError: "sorting_kernel_method_name" not implemented for 'ComplexDouble'
            ind = np.argsort(actual[0].cpu().numpy(), axis=-1)[::-1]
            actual_np = [x.cpu().numpy() for x in actual]
            sorted_actual = (
                np.take_along_axis(actual_np[0], ind, axis=-1),
                np.take_along_axis(actual_np[1], ind[:, None], axis=-1))

            self.assertRelativeEqual(expected[0], sorted_actual[0], exact_dtype=False)
            self.assertRelativeEqual(abs(expected[1]), abs(sorted_actual[1]), exact_dtype=False)

        shapes = [(0, 0),  # Empty matrix
                  (5, 5),  # Single matrix
                  (0, 0, 0), (0, 5, 5),  # Zero batch dimension tensors
                  (2, 5, 5),  # 3-dim tensors
                  (2, 1, 5, 5)]  # 4-dim tensors
        for shape in shapes:
            run_test(shape)
            run_test(shape, symmetric=True)

    @onlyCUDA
    @dtypes(torch.double, torch.float32, torch.complex64, torch.complex128)
    def test_geqrf(self, device, dtype):
        def run_test(shape):
            # numpy.linalg.qr with mode = 'raw' computes the same operation as torch.geqrf
            # so this test compares against that function
            A = make_tensor(shape, dtype=dtype, device=device)

            # numpy.linalg.qr doesn't work with batched input
            m, n = A.shape[-2:]
            tau_size = "n" if m > n else "m"
            np_dtype = A.cpu().numpy().dtype
            ot = [np_dtype, np_dtype]
            numpy_geqrf_batched = np.vectorize(
                lambda x: np.linalg.qr(x, mode='raw'),
                otypes=ot,
                signature=f'(m,n)->(n,m),({tau_size})')

            expected = numpy_geqrf_batched(A.cpu())
            actual = torch.geqrf(A)

            # numpy.linalg.qr returns transposed result
            self.assertRelativeEqual(expected[0].swapaxes(-2, -1), actual[0])
            self.assertRelativeEqual(expected[1], actual[1])

        batches = [(), (0, ), (2, ), (2, 1)]
        ns = [5, 2, 0]
        for batch, (m, n) in product(batches, product(ns, ns)):
            run_test((*batch, m, n))

    @onlyCUDA
    @dtypes(*(get_all_dtypes()))
    def test_ger(self, device, dtype):
        def run_test_case(a, b):
            if dtype == torch.bfloat16:
                a_np = a.to(torch.double).cpu().numpy()
                b_np = b.to(torch.double).cpu().numpy()
                exact_dtype = False
            else:
                a_np = a.cpu().numpy()
                b_np = b.cpu().numpy()
                exact_dtype = True
            expected = np.outer(a_np, b_np)

            self.assertEqual(torch.outer(a, b), expected, exact_dtype=False)
            self.assertEqual(torch.Tensor.outer(a, b), expected, exact_dtype=False)

            self.assertEqual(torch.ger(a, b), expected, exact_dtype=False)
            self.assertEqual(torch.Tensor.ger(a, b), expected, exact_dtype=False)

            # test out variant
            out = torch.empty(a.size(0), b.size(0), device=device, dtype=dtype)
            torch.outer(a, b, out=out)
            self.assertEqual(out, expected, exact_dtype=False)

            out = torch.empty(a.size(0), b.size(0), device=device, dtype=dtype)
            torch.ger(a, b, out=out)
            self.assertEqual(out, expected, exact_dtype=False)

        a = torch.randn(50).to(device=device, dtype=dtype)
        b = torch.randn(50).to(device=device, dtype=dtype)
        run_test_case(a, b)

        # test 0 strided tensor
        zero_strided = torch.randn(1).to(device=device, dtype=dtype).expand(50)
        run_test_case(zero_strided, b)
        run_test_case(a, zero_strided)

    @onlyCUDA
    @dtypes(torch.float32, torch.cfloat, torch.half, torch.bfloat16)
    @precisionOverride({torch.float32: 1e-02, torch.cfloat: 1e-02, torch.bfloat16:1e-02})
    def test_inner(self, device, dtype):
        def check(a_sizes_, b_sizes_):
            for a_sizes, b_sizes in ((a_sizes_, b_sizes_), (b_sizes_, a_sizes_)):
                a = torch.randn(a_sizes, dtype=dtype, device=device)
                b = torch.randn(b_sizes, dtype=dtype, device=device)
                res = torch.inner(a, b)
                if dtype == torch.bfloat16:
                    ref = np.inner(a.to(torch.double).cpu().numpy(), b.to(torch.double).cpu().numpy())
                else:
                    ref = np.inner(a.cpu().numpy(), b.cpu().numpy())
                self.assertEqual(res.cpu(), torch.from_numpy(np.array(ref)), exact_dtype=False)
                out = torch.zeros_like(res)
                torch.inner(a, b, out=out)
                self.assertEqual(res, out, exact_dtype=False)

        check([], [])                       # scalar x scalar
        check([], [0])                      # scalar x empty
        check([], [3])                      # scalar x 1D
        check([], [2, 3, 4])                # scalar x 3D

        check([0], [0])                     # empty x empty
        check([0], [2, 0])                  # empty x 2D

        check([2], [2])                     # 1D x 1D
        check([2], [3, 1, 2])               # 1D x 3D
        check([2], [3, 0, 2])               # 1D x 3D empty

        check([1, 2], [3, 2])               # 2D x 2D
        check([1, 2], [3, 4, 2])            # 2D x 3D
        check([2, 1, 3, 2], [1, 3, 2, 2])   # 4D x 4D

        # Test noncontiguous input
        a = torch.randn(3, 2, device=device, dtype=dtype).transpose_(0, 1)
        b = torch.randn(4, 3, device=device, dtype=dtype)[::2, :]
        self.assertFalse(a.is_contiguous() or b.is_contiguous())
        if dtype == torch.bfloat16:
            self.assertEqual(a.inner(b).to(torch.double), np.inner(a.to(torch.double).cpu().numpy(), b.to(torch.double).cpu().numpy()))
        else:
            self.assertEqual(a.inner(b), np.inner(a.cpu().numpy(), b.cpu().numpy()))

        # Test error message
        with self.assertRaisesRegex(RuntimeError,
                                    r"inner\(\) the last dimension must match on both "
                                    r"input tensors but got shapes \[2, 3\] and \[2, 2\]"):
            torch.randn(2, 3, device=device, dtype=dtype).inner(torch.randn(2, 2, device=device, dtype=dtype))

    @onlyCUDA
    @dtypes(torch.float32, torch.float64, torch.complex64, torch.complex128)
    @precisionOverride({torch.float32: 2e-3, torch.complex64: 2e-3,
                        torch.float64: 1e-8, torch.complex128: 1e-8})
    def test_inverse(self, device, dtype):

        def run_test(torch_inverse, matrix, batches, n):
            matrix_inverse = torch_inverse(matrix)

            # Compare against NumPy output
            # NumPy uses 'gesv' LAPACK routine solving the equation A A_inv = I
            # But in PyTorch 'gertf' + 'getri' is used causing element-wise differences
            expected = np.linalg.inv(matrix.cpu().numpy())
            self.assertRelativeEqual(matrix_inverse, expected, atol=self.precision, rtol=self.precision)

            # Additional correctness tests, check matrix*matrix_inverse == identity
            identity = torch.eye(n, dtype=dtype, device=device)
            self.assertRelativeEqual(identity.expand_as(matrix), np.matmul(matrix.cpu(), matrix_inverse.cpu()))
            self.assertRelativeEqual(identity.expand_as(matrix), np.matmul(matrix_inverse.cpu(), matrix.cpu()))

            # check the out= variant
            # prepare the expected out tensor
            matrix_inverse_out = torch.empty(*batches, n, n, dtype=dtype, device=device)
            matrix_inverse_out_t = matrix_inverse_out.transpose(-2, -1).clone(memory_format=torch.contiguous_format)
            matrix_inverse_out = matrix_inverse_out_t.transpose(-2, -1)
            ans = torch_inverse(matrix, out=matrix_inverse_out)
            self.assertRelativeEqual(matrix_inverse_out, ans, atol=0, rtol=0)
            self.assertRelativeEqual(matrix_inverse_out, matrix_inverse, atol=0, rtol=0)

            # batched matrices: 3+ dimensional tensors, check matrix_inverse same as single-inverse for each matrix
            if matrix.ndim > 2 and batches[0] != 0:
                expected_inv_list = []
                p = int(np.prod(batches))  # use `p` instead of -1, so that the test works for empty input as well
                for mat in matrix.contiguous().view(p, n, n):
                    expected_inv_list.append(torch_inverse(mat))
                expected_inv = torch.stack(expected_inv_list).view(*batches, n, n)
                if self.device_type == 'cuda' and dtype in [torch.float32, torch.complex64]:
                    # single-inverse is done using cuSOLVER, while batched inverse is done using MAGMA
                    # individual values can be significantly different for fp32, hence rather high rtol is used
                    # the important thing is that torch_inverse passes above checks with identity
                    self.assertRelativeEqual(matrix_inverse, expected_inv, atol=1e-1, rtol=1e-2)
                else:
                    self.assertRelativeEqual(matrix_inverse, expected_inv)

        # helper function for testing torch.linalg.inv_ex
        def test_inv_ex(input, out=None):
            if out is not None:
                info = torch.empty(0, dtype=torch.int32, device=device)
                return torch.linalg.inv_ex(input, out=(out, info)).inverse
            return torch.linalg.inv_ex(input).inverse

        for torch_inverse in [torch.inverse, torch.linalg.inv, test_inv_ex]:
            for batches, n in product(
                [[], [0], [2], [2, 1]],
                [0, 5]
            ):
                matrices = random_fullrank_matrix_distinct_singular_value(n, *batches, dtype=dtype, device=device).to(device)
                run_test(torch_inverse, matrices, batches, n)

                # test non-contiguous input
                run_test(torch_inverse, matrices.transpose(-2, -1), batches, n)
                if n > 0:
                    run_test(
                        torch_inverse,
                        random_fullrank_matrix_distinct_singular_value(n * 2, *batches, dtype=dtype, device=device).to(device)
                        .view(-1, n * 2, n * 2)[:, ::2, ::2].view(*batches, n, n),
                        batches, n
                    )
    @onlyCUDA
    @dtypes(torch.double, torch.cdouble)
    def test_det(self, device, dtype):
        tensors = (
            torch.randn((2, 2), device=device, dtype=dtype),
            torch.randn((129, 129), device=device, dtype=dtype),
            torch.randn((3, 52, 52), device=device, dtype=dtype),
            torch.randn((4, 2, 26, 26), device=device, dtype=dtype))


        ops = (torch.det, torch.Tensor.det,
               torch.linalg.det)
        for t in tensors:
            expected = np.linalg.det(t.cpu().numpy())
            for op in ops:
                actual = op(t)
                self.assertRelativeEqual(actual, expected)
                self.compare_with_numpy(op, np.linalg.det, t)

                # NOTE: det requires a 2D+ tensor
                x = torch.randn(1, device=device, dtype=dtype)
                with self.assertRaises(RuntimeError):
                    op(x)

    @onlyCUDA
    @dtypes(torch.double)
    def test_logdet(self, device, dtype):
        tensors = (
            torch.randn((2, 2), device=device, dtype=dtype),
            torch.randn((129, 129), device=device, dtype=dtype),
            torch.randn((3, 52, 52), device=device, dtype=dtype),
            torch.randn((4, 2, 26, 26), device=device, dtype=dtype))


        ops = (torch.logdet, torch.Tensor.logdet)
        for t in tensors:
            expected = np.log(np.linalg.det(t.cpu().numpy()))
            for op in ops:
                actual = op(t)
                self.assertRelativeEqual(actual, expected)
                # self.compare_with_numpy(op, np.linalg.det, t)

                # NOTE: det requires a 2D+ tensor
                x = torch.randn(1, device=device, dtype=dtype)
                with self.assertRaises(RuntimeError):
                    op(x)

    @dtypes(torch.float32, torch.float64, torch.complex64, torch.complex128)
    @precisionOverride({torch.float32: 1e-3, torch.complex64: 1e-3,
                        torch.float64: 1e-8, torch.complex128: 1e-8})
    def test_slogdet(self, device, dtype):
        from torch.testing._internal.common_utils import (random_hermitian_matrix, random_hermitian_psd_matrix,
                                                          random_hermitian_pd_matrix, random_square_matrix_of_rank)

        # mat_chars denotes matrix characteristics
        # possible values are: hermitian, hermitian_psd, hermitian_pd, singular, non_singular
        def run_test(matsize, batchdims, mat_chars):
            num_matrices = np.prod(batchdims)
            list_of_matrices = []
            if num_matrices != 0:
                for idx in range(num_matrices):
                    mat_type = idx % len(mat_chars)
                    if mat_chars[mat_type] == 'hermitian':
                        list_of_matrices.append(random_hermitian_matrix(matsize, dtype=dtype, device=device))
                    elif mat_chars[mat_type] == 'hermitian_psd':
                        list_of_matrices.append(random_hermitian_psd_matrix(matsize, dtype=dtype, device=device))
                    elif mat_chars[mat_type] == 'hermitian_pd':
                        list_of_matrices.append(random_hermitian_pd_matrix(matsize, dtype=dtype, device=device))
                    elif mat_chars[mat_type] == 'singular':
                        list_of_matrices.append(torch.ones(matsize, matsize, dtype=dtype, device=device))
                    elif mat_chars[mat_type] == 'non_singular':
                        list_of_matrices.append(random_square_matrix_of_rank(matsize, matsize, dtype=dtype, device=device))
                full_tensor = torch.stack(list_of_matrices, dim=0).reshape(batchdims + (matsize, matsize))
            else:
                full_tensor = torch.randn(*batchdims, matsize, matsize, dtype=dtype, device=device)

            actual_value = torch.linalg.slogdet(full_tensor)
            expected_value = np.linalg.slogdet(full_tensor.cpu().numpy())
            self.assertRelativeEqual(expected_value[0], actual_value[0], atol=self.precision, rtol=self.precision)
            self.assertRelativeEqual(expected_value[1], actual_value[1], atol=self.precision, rtol=self.precision)

            # test out=variant
            sign_out = torch.empty_like(actual_value[0])
            logabsdet_out = torch.empty_like(actual_value[1])
            ans = torch.linalg.slogdet(full_tensor, out=(sign_out, logabsdet_out))
            self.assertRelativeEqual(ans[0], sign_out)
            self.assertRelativeEqual(ans[1], logabsdet_out)
            self.assertRelativeEqual(sign_out, actual_value[0])
            self.assertRelativeEqual(logabsdet_out, actual_value[1])

        for matsize, batchdims in product([0, 3, 5], [(0,), (3,), (5, 3)]):
            run_test(matsize, batchdims, mat_chars=['hermitian_pd'])
            run_test(matsize, batchdims, mat_chars=['singular'])
            run_test(matsize, batchdims, mat_chars=['non_singular'])
            run_test(matsize, batchdims, mat_chars=['hermitian', 'hermitian_pd', 'hermitian_psd'])
            run_test(matsize, batchdims, mat_chars=['singular', 'non_singular'])

    @onlyCUDA
    @dtypes(torch.double, torch.float32)
    @precisionOverride({torch.float32: 1e-2})
    def test_lstsq(self, device, dtype):
        def _test_underdetermined(a, b, expectedNorm):
            # underdetermined systems are only supported on CPU
            if self.device_type != 'cpu':
                return

            m = a.size()[0]
            n = a.size()[1]
            assert(m <= n)

            a_copy = a.clone()
            b_copy = b.clone()
            res1 = torch.lstsq(b, a)[0]
            self.assertRelativeEqual(a, a_copy, atol=0, rtol=0)
            self.assertRelativeEqual(b, b_copy, atol=0, rtol=0)
            self.assertRelativeEqual((torch.mm(a, res1) - b).norm(), expectedNorm, atol=1e-8, rtol=0)

            ta = torch.tensor((), dtype=dtype, device=device)
            tb = torch.tensor((), dtype=dtype, device=device)
            res2 = torch.lstsq(b, a, out=(tb, ta))[0]
            self.assertRelativeEqual(a, a_copy, atol=0, rtol=0)
            self.assertRelativeEqual(b, b_copy, atol=0, rtol=0)
            self.assertRelativeEqual((torch.mm(a, res1) - b).norm(), expectedNorm, atol=1e-8, rtol=0)

            res3 = torch.lstsq(b, a, out=(b, a))[0]
            self.assertRelativeEqual((torch.mm(a_copy, b) - b_copy).norm(), expectedNorm, atol=1e-8, rtol=0)
            self.assertRelativeEqual(res1, tb, atol=0, rtol=0)
            self.assertRelativeEqual(res1, b, atol=0, rtol=0)
            self.assertRelativeEqual(res1, res2, atol=0, rtol=0)
            self.assertRelativeEqual(res1, res3, atol=0, rtol=0)

        def _test_overdetermined(a, b, expectedNorm):
            m = a.size()[0]
            n = a.size()[1]
            assert(m > n)

            def check_norm(a, b, expected_norm, gels_result):
                # Checks |ax - b| and the residual info from the result

                # The first n rows is the least square solution.
                # Rows n to m-1 contain residual information.
                x = gels_result[:n]
                resid_info = gels_result[n:]

                resid_norm = (torch.mm(a, x) - b).norm()
                self.assertRelativeEqual(resid_norm, expectedNorm, atol=1e-8, rtol=0)
                self.assertRelativeEqual(resid_info.norm(), resid_norm, atol=1e-8, rtol=0)

            a_copy = a.clone()
            b_copy = b.clone()
            res1 = torch.lstsq(b, a)[0]
            self.assertRelativeEqual(a, a_copy, atol=0, rtol=0)
            self.assertRelativeEqual(b, b_copy, atol=0, rtol=0)
            check_norm(a, b, expectedNorm, res1)

            ta = torch.tensor((), dtype=dtype, device=device)
            tb = torch.tensor((), dtype=dtype, device=device)
            res2 = torch.lstsq(b, a, out=(tb, ta))[0]
            self.assertRelativeEqual(a, a_copy, atol=0, rtol=0)
            self.assertRelativeEqual(b, b_copy, atol=0, rtol=0)
            check_norm(a, b, expectedNorm, res2)

            res3 = torch.lstsq(b, a, out=(b, a))[0]
            check_norm(a_copy, b_copy, expectedNorm, res3)

            self.assertRelativeEqual(res1, tb, atol=0, rtol=0)
            self.assertRelativeEqual(res1, b, atol=0, rtol=0)
            self.assertRelativeEqual(res1, res2, atol=0, rtol=0)
            self.assertRelativeEqual(res1, res3, atol=0, rtol=0)

        # basic test
        expectedNorm = 0
        a = torch.tensor(((1.44, -9.96, -7.55, 8.34),
                          (-7.84, -0.28, 3.24, 8.09),
                          (-4.39, -3.24, 6.27, 5.28),
                          (4.53, 3.83, -6.64, 2.06)), dtype=dtype, device=device).t()
        b = torch.tensor(((8.58, 8.26, 8.48, -5.28),
                          (9.35, -4.43, -0.70, -0.26)), dtype=dtype, device=device).t()
        _test_underdetermined(a, b, expectedNorm)

        # test overdetermined
        expectedNorm = 17.390200628863
        a = torch.tensor(((1.44, -9.96, -7.55, 8.34, 7.08, -5.45),
                          (-7.84, -0.28, 3.24, 8.09, 2.52, -5.70),
                          (-4.39, -3.24, 6.27, 5.28, 0.74, -1.19),
                          (4.53, 3.83, -6.64, 2.06, -2.47, 4.70)), dtype=dtype, device=device).t()
        b = torch.tensor(((8.58, 8.26, 8.48, -5.28, 5.72, 8.93),
                          (9.35, -4.43, -0.70, -0.26, -7.36, -2.52)), dtype=dtype, device=device).t()
        _test_overdetermined(a, b, expectedNorm)

        # test underdetermined
        expectedNorm = 0
        a = torch.tensor(((1.44, -9.96, -7.55),
                          (-7.84, -0.28, 3.24),
                          (-4.39, -3.24, 6.27),
                          (4.53, 3.83, -6.64)), dtype=dtype, device=device).t()
        b = torch.tensor(((8.58, 8.26, 8.48),
                          (9.35, -4.43, -0.70)), dtype=dtype, device=device).t()
        _test_underdetermined(a, b, expectedNorm)

        # test reuse
        expectedNorm = 0
        a = torch.tensor(((1.44, -9.96, -7.55, 8.34),
                          (-7.84, -0.28, 3.24, 8.09),
                          (-4.39, -3.24, 6.27, 5.28),
                          (4.53, 3.83, -6.64, 2.06)), dtype=dtype, device=device).t()
        b = torch.tensor(((8.58, 8.26, 8.48, -5.28),
                          (9.35, -4.43, -0.70, -0.26)), dtype=dtype, device=device).t()
        ta = torch.tensor((), dtype=dtype, device=device)
        tb = torch.tensor((), dtype=dtype, device=device)
        torch.lstsq(b, a, out=(tb, ta))
        self.assertRelativeEqual((torch.mm(a, tb) - b).norm(), expectedNorm, atol=1e-8, rtol=0)
        torch.lstsq(b, a, out=(tb, ta))
        self.assertRelativeEqual((torch.mm(a, tb) - b).norm(), expectedNorm, atol=1e-8, rtol=0)
        torch.lstsq(b, a, out=(tb, ta))
        self.assertRelativeEqual((torch.mm(a, tb) - b).norm(), expectedNorm, atol=1e-8, rtol=0)

    @onlyCUDA
    @precisionOverride({torch.complex64: 5e-6, torch.float32:1e-4})
    @dtypes(torch.double, torch.cfloat, torch.cdouble, torch.float32)
    def test_lu(self, device, dtype):
        from torch.testing._internal.common_utils import random_matrix

        def run_test(device, pivot):
            def run_subtest(matrix_size, batches, device, pivot, singular=False, a=None):
                if isinstance(matrix_size, int):
                    rows = columns = matrix_size
                else:
                    rows, columns = matrix_size
                if a is None:
                    a = random_matrix(rows, columns, *batches, **dict(singular=singular, dtype=dtype, device=device)).to(device)
                a_LU_info, pivots_info, info_ = torch.lu(a, pivot=pivot, get_infos=True)
                self.assertRelativeEqual(a_LU_info.size(), torch.Size(batches + (rows, columns)))
                self.assertRelativeEqual(pivots_info.size(), torch.Size(batches + (min(rows, columns),)))
                self.assertRelativeEqual(info_.size(), torch.Size(batches))
                # If a randomly generated input matrix is singular,
                # then info_ contains indices i such that U[i, i] ==
                # 0. This however conveys that the factorization was
                # successful albeit with a singular input. Therefore,
                # we require info.min() >= 0
                self.assertGreaterEqual(info_.min(), 0)
                a_LU, pivots = torch.lu(a, pivot=pivot)
                self.assertRelativeEqual(a_LU, a_LU_info)
                self.assertRelativeEqual(pivots_info, pivots)


                P, L, U = torch.lu_unpack(a_LU, pivots)
                P_ = P.cpu().numpy()
                L_ = L.cpu().numpy()
                U_ = U.cpu().numpy()

                self.assertRelativeEqual(np.matmul(P_, np.matmul(L_, U_)), a)

                if self.device_type == 'cuda':
                    # lu without pivoting is implemented only for cuda device
                    a_LU_info_nopiv, nopiv, info_nopiv = torch.lu(a, pivot=False, get_infos=True)
                    P_nopiv, L_nopiv, U_nopiv = torch.lu_unpack(a_LU_info_nopiv, nopiv)
                    P_nopiv_ = P_nopiv.cpu().numpy()
                    L_nopiv_ = L_nopiv.cpu().numpy()
                    U_nopiv_ = U_nopiv.cpu().numpy()

                    self.assertRelativeEqual(np.matmul(P_nopiv_, np.matmul(L_nopiv_, U_nopiv_)), a)

                    k = min(rows, columns)
                    self.assertRelativeEqual(nopiv, torch.arange(1, 1 + k, device=device, dtype=torch.int32).expand(a.shape[:-2] + (k, )))
                    if not singular:
                        # It is not guaranteed that LU factorization
                        # without pivoting is able to determine if a
                        # matrix is singular while LU factorization
                        # with pivoting is. Therefore, we require the
                        # equality of info-s only for non-singular
                        # matrices.
                        # NOTE: infor_ is reshaped because info_nopiv might have
                        # squashed batch dimensions for complex types on CUDA,
                        # see the TODOs above.
                        self.assertRelativeEqual(info_.reshape(info_nopiv.shape), info_nopiv)

            for ms, batch in product([3, 5, 7, (4, 2), (3, 4)], [(), (2,), (3,), (3, 5)]):
                run_subtest(ms, batch, device, pivot)
                run_subtest(ms, batch, device, pivot, singular=True)

                # Reproducer of a magma bug, see https://bitbucket.org/icl/magma/issues/13/getrf_batched-kernel-produces-nans-on
                a = torch.ones(batch + (ms if isinstance(ms, tuple) else (ms, ms)), dtype=torch.double, device=device)
                run_subtest(ms, batch, device, pivot, singular=True, a=a)

            # Info should be positive for rank deficient matrices
            a = torch.ones(5, 3, 3, device=device)
            self.assertGreater(a.lu(pivot=pivot, get_infos=True)[2][0], 0)

        run_test(device, True)

        if self.device_type == 'cpu':
            # Error checking, no pivoting variant on CPU
            with self.assertRaisesRegex(RuntimeError, 'lu without pivoting is not implemented on the CPU'):
                torch.lu(torch.empty(1, 2, 2), pivot=False)
        else:
            run_test(device, False)

    def lu_solve_test_helper(self, A_dims, b_dims, pivot, device, dtype):
        from torch.testing._internal.common_utils import random_fullrank_matrix_distinct_singular_value

        b = torch.randn(*b_dims, dtype=dtype, device=device)
        A = random_fullrank_matrix_distinct_singular_value(*A_dims, dtype=dtype, device=device).to(device)
        LU_data, LU_pivots, info = torch.lu(A, get_infos=True, pivot=pivot)
        self.assertRelativeEqual(info, torch.zeros_like(info))
        return b, A, LU_data, LU_pivots

    @onlyCUDA
    @dtypes(torch.float32, torch.float64, torch.complex64, torch.complex128)
    @precisionOverride({torch.float32: 1e-3, torch.complex64: 1e-3,
                        torch.float64: 1e-8, torch.complex128: 1e-8})
    def test_lu_solve(self, device, dtype):
        def sub_test(pivot):
            for k, n in zip([2, 3, 5], [3, 5, 7]):
                b, A, LU_data, LU_pivots = self.lu_solve_test_helper((n,), (n, k), pivot, device, dtype)
                x = torch.lu_solve(b, LU_data, LU_pivots)
                self.assertRelativeEqual(b, np.matmul(A.cpu(), x.cpu()))

        sub_test(True)
        sub_test(False)

    @onlyCUDA
    @dtypes(torch.float32, torch.double, torch.cfloat, torch.cdouble)
    @precisionOverride({torch.float32: 1e-3, torch.complex64: 3e-2})
    def test_lu_unpack(self, device, dtype):
        def run_test(pivot):
            for shape in ((3, 3), (5, 3, 3), (7, 3, 5, 5), (7, 5, 3, 3, 3)):
                a = torch.randn(*shape, dtype=dtype, device=device)
                a_lu, p = torch.lu(a, pivot=pivot)
                p_ref, l_ref, u_ref = torch.lu_unpack(a_lu, p)
                self.assertRelativeEqual(p_ref.matmul(l_ref.matmul(u_ref)), a)
            for shape in ((3, 3), (5, 3, 3), (7, 3, 5, 5), (7, 5, 3, 3, 3),
                          (3, 5), (5, 3), (3, 3, 5), (3, 5, 3),
                          (7, 5, 3, 5, 3), (7, 5, 3, 3, 5),
                          # empty tensors
                          (0, 0), (0, 0, 0), (0, 3, 3)
                          ):
                a = make_tensor(shape, dtype=dtype, device=device, low=-0.1, high=+0.1)
                a_lu, p = torch.lu(a, pivot=pivot)
                p_ref, l_ref, u_ref = torch.lu_unpack(a_lu, p)
                self.assertRelativeEqual(p_ref.matmul(l_ref.matmul(u_ref)), a)

        run_test(True)
        run_test(False)

    @onlyCUDA
    @dtypes(torch.float32, torch.double, torch.half, torch.bfloat16)
    @precisionOverride({torch.float64: 1e-3, torch.float32: 1e-2})
    def test_matmul(self, device, dtype):
        a = torch.rand(655, 22, 64, device=device, dtype=dtype)
        b = torch.rand(655, 64, 22, device=device, dtype=dtype)
        c = torch.full((655, 22, 22), math.nan, dtype=dtype, device=device)
        cpu_result = torch.matmul(a.cpu().float(), b.cpu().float()).cuda().to(dtype=dtype)
        torch.matmul(a, b, out=c)
        self.assertRelativeEqual(c, cpu_result)

    @onlyCUDA
    @dtypes(torch.double, torch.cdouble, torch.half)
    def test_matrix_power_non_negative(self, device, dtype):
        def check(*size, noncontiguous=False):
            t = make_tensor(size, device, dtype, noncontiguous=noncontiguous)
            for n in range(8):
                res = torch.matrix_power(t, n)
                ref = np.linalg.matrix_power(t.cpu().numpy(), n)
                self.assertRelativeEqual(res.cpu(), torch.from_numpy(ref), exact_dtype=False)

        check(0, 0)
        check(1, 1)
        check(5, 5)
        check(5, 5, noncontiguous=True)
        check(0, 3, 3)
        check(2, 3, 3)
        check(2, 3, 4, 4, noncontiguous=True)

    @onlyCUDA
    @dtypes(torch.double, torch.cdouble)
    def test_matrix_power_negative(self, device, dtype):
        from torch.testing._internal.common_utils import random_fullrank_matrix_distinct_singular_value

        def check(*size):
            t = random_fullrank_matrix_distinct_singular_value(*size, dtype=dtype, device=device)
            for n in range(-7, 0):
                res = torch.matrix_power(t, n)
                ref = np.linalg.matrix_power(t.cpu().numpy(), n)
                self.assertRelativeEqual(res.cpu(), torch.from_numpy(ref))

        check(0)
        check(5)
        check(0, 2)
        check(3, 0)
        check(3, 2)
        check(5, 2, 3)

    @onlyCUDA
    @dtypes(torch.float32, torch.float64, torch.complex64, torch.complex128)
    def test_matrix_rank(self, device, dtype):
        matrix_rank = torch.linalg.matrix_rank

        def run_test(shape0, shape1, batch):
            a = torch.randn(*batch, shape0, shape1, dtype=dtype, device=device)
            rank_a = matrix_rank(a)

            self.assertRelativeEqual(rank_a, matrix_rank(a.conj().transpose(-2, -1)))
            aaH = torch.matmul(a, a.conj().transpose(-2, -1))
            rank_aaH = matrix_rank(aaH)
            rank_aaH_hermitian = matrix_rank(aaH, hermitian=True)
            self.assertRelativeEqual(rank_aaH, rank_aaH_hermitian)
            aHa = torch.matmul(a.conj().transpose(-2, -1), a)
            self.assertRelativeEqual(matrix_rank(aHa), matrix_rank(aHa, hermitian=True))

            # check against NumPy
            self.assertRelativeEqual(rank_a, np.linalg.matrix_rank(a.cpu().numpy()))
            self.assertRelativeEqual(matrix_rank(a, 0.01), np.linalg.matrix_rank(a.cpu().numpy(), 0.01))

            self.assertRelativeEqual(rank_aaH, np.linalg.matrix_rank(aaH.cpu().numpy()))
            self.assertRelativeEqual(matrix_rank(aaH, 0.01), np.linalg.matrix_rank(aaH.cpu().numpy(), 0.01))

            # hermitian flag for NumPy was added in 1.14.0
            if np.lib.NumpyVersion(np.__version__) >= '1.14.0':
                self.assertRelativeEqual(rank_aaH_hermitian,
                                 np.linalg.matrix_rank(aaH.cpu().numpy(), hermitian=True))
                self.assertRelativeEqual(matrix_rank(aaH, 0.01, True),
                                 np.linalg.matrix_rank(aaH.cpu().numpy(), 0.01, True))

            # check out= variant
            out = torch.empty(a.shape[:-2], dtype=torch.int64, device=device)
            ans = matrix_rank(a, out=out)
            self.assertRelativeEqual(ans, out)
            self.assertRelativeEqual(ans, rank_a)

        shapes = (3, 13)
        batches = ((), (0, ), (4, ), (3, 5, ))
        for (shape0, shape1), batch in zip(product(shapes, reversed(shapes)), batches):
            run_test(shape0, shape1, batch)

    @onlyCUDA
    @dtypes(torch.float32, torch.double, torch.complex64, torch.complex128, torch.half, torch.bfloat16)
    def test_matrix_exp(self, device, dtype):
        x = torch.randn(3, 3, 1, 1, device=device, dtype=dtype)
        mexp = torch.matrix_exp(x)
        self.assertRelativeEqual(mexp, x.exp())

    @onlyCUDA
    @dtypes(torch.float32, torch.float64, torch.cfloat, torch.cdouble, torch.half, torch.bfloat16)
    @precisionOverride({torch.half: 1e-2, torch.bfloat16: 1e-3})
    @tf32_on_and_off(0.01)
    def test_mm(self, device, dtype):
        def _test_mm(n, m, p, dtype, genf):
            # helper function
            def matrixmultiply(mat1, mat2):
                n = mat1.size(0)
                m = mat1.size(1)
                p = mat2.size(1)
                res = torch.zeros(n, p, dtype=dtype, device=device)
                for i, j in iter_indices(res):
                    res[i, j] = sum(mat1[i, k] * mat2[k, j] for k in range(m))
                return res

            # contiguous case
            mat1 = genf(n, m)
            mat2 = genf(m, p)
            res = torch.mm(mat1, mat2)

            res2 = matrixmultiply(mat1, mat2)
            self.assertRelativeEqual(res, res2)

            # non contiguous case 1
            mat1 = genf(n, m)
            mat2 = genf(p, m).t()
            res = torch.mm(mat1, mat2)

            res2 = matrixmultiply(mat1, mat2)
            self.assertRelativeEqual(res, res2)

            # non contiguous case 2
            mat1 = genf(m, n).t()
            mat2 = genf(m, p)
            res = torch.mm(mat1, mat2)

            res2 = matrixmultiply(mat1, mat2)
            self.assertRelativeEqual(res, res2)

            # non contiguous case 3
            mat1 = genf(m, n).t()
            mat2 = genf(p, m).t()
            res = torch.mm(mat1, mat2)

            res2 = matrixmultiply(mat1, mat2)
            self.assertRelativeEqual(res, res2)

            # test with zero stride
            mat1 = genf(n, m)
            mat2 = genf(m, 1).expand(m, p)
            res = torch.mm(mat1, mat2)

            res2 = matrixmultiply(mat1, mat2)
            self.assertRelativeEqual(res, res2)

            # explicitly exercise the _out variant in torch.mm().
            # contiguous case
            mat1 = genf(n, m)
            mat2 = genf(m, p)
            res = genf(n, p)
            torch.mm(mat1, mat2, out=res)

            res2 = matrixmultiply(mat1, mat2)
            self.assertRelativeEqual(res, res2)

            # explicitly exercise the _out variant in torch.mm().
            # non contiguous case 3
            mat1 = genf(m, n).t()
            mat2 = genf(p, m).t()
            res = genf(n, p)
            torch.mm(mat1, mat2, out=res)

            res2 = matrixmultiply(mat1, mat2)
            self.assertRelativeEqual(res, res2)

        def genf_int(x, y):
            return torch.randint(0, 100, (x, y), dtype=dtype, device=device)

        def genf_bfloat(x, y):
            return torch.randn(x, y, dtype=torch.float32, device=device).to(dtype) * 0.1

        def genf_float(x, y):
            return torch.randn(x, y, dtype=dtype, device=device)

        for (n, m, p) in [(20, 10, 15), (15, 20, 10), (25, 18, 10)]:
            if (dtype == torch.int32) or (dtype == torch.int64):
                genf = genf_int
            elif (dtype == torch.bfloat16):
                genf = genf_bfloat
            else:
                genf = genf_float

            _test_mm(n, m, p, dtype, genf)

    @onlyCUDA
    @dtypes(torch.float32, torch.float64, torch.cfloat, torch.cdouble, torch.half, torch.bfloat16)
    def test_mv(self, device, dtype):
        with self.assertRaises(RuntimeError):
            x = torch.rand((2, 3), device=device, dtype=dtype)
            y = torch.rand((2, 3), device=device, dtype=dtype)
            res = torch.mv(x, y)
        x = torch.rand((2, 3), device=device, dtype=dtype)
        y = torch.rand((3,), device=device, dtype=dtype)
        actual = torch.mv(x, y)
        if dtype == torch.bfloat16:
            expected = np.matmul(x.to(torch.double).cpu().numpy(), y.to(torch.double).cpu().numpy())
        else:
            expected = np.matmul(x.cpu().numpy(), y.cpu().numpy())
        self.assertRelativeEqual(actual, expected, exact_dtype=False)

    @dtypes(torch.float32, torch.float64, torch.complex64, torch.complex128)
    def test_orgqr(self, device, dtype):
        def generate_reflectors_and_tau(A):
            """
            This function uses numpy.linalg.qr with mode "raw" to extract output of LAPACK's geqrf.
            There is torch.geqrf function but it doesn't work with complex-valued input.
            """
            if A.numel() > 0:
                A_cpu = A.cpu()
                flattened_batch_shape = [-1, *A_cpu.shape[-2:]]
                reflectors = torch.empty_like(A_cpu).view(*flattened_batch_shape)
                tau_shape = [*A_cpu.shape[:-2], A_cpu.shape[-1]]
                tau = torch.empty(tau_shape, dtype=dtype).view(-1, A_cpu.shape[-1])
                for A_i, reflectors_i, tau_i in zip(A_cpu.contiguous().view(*flattened_batch_shape), reflectors, tau):
                    reflectors_tmp, tau_i[:] = map(torch.from_numpy, np.linalg.qr(A_i, mode='raw'))
                    reflectors_i[:] = reflectors_tmp.T
                reflectors = reflectors.view(*A_cpu.shape)
                tau = tau.view(tau_shape)
                return reflectors.to(A.device), tau.to(A.device)

            reflectors = torch.empty_like(A)
            tau = torch.empty(*A.shape[:-2], A.shape[-1], dtype=dtype, device=device)
            return reflectors, tau

        def run_test(shape):
            A = torch.randn(*shape, dtype=dtype, device=device)
            reflectors, tau = generate_reflectors_and_tau(A)
            expected, _ = torch.linalg.qr(A)
            actual = torch.orgqr(reflectors, tau)
            # torch.linalg.qr does not work correctly for zero batch dimension tensors
            # see https://github.com/pytorch/pytorch/issues/50576
            if (A.numel() > 0):
                self.assertRelativeEqual(expected, actual)
            else:
                self.assertTrue(actual.shape == shape)

            # if tau is empty and A is not the result should be a matrix with ones on the diagonal
            if (A.numel() > 0):
                tau_empty = torch.empty(*shape[:-2], 0, dtype=dtype, device=device)
                identity_mat = torch.zeros_like(reflectors)
                identity_mat.diagonal(dim1=-1, dim2=-2)[:] = 1
                actual = torch.linalg.householder_product(reflectors, tau_empty)
                self.assertRelativeEqual(actual, identity_mat)

            out = torch.empty_like(A)
            ans = torch.linalg.householder_product(reflectors, tau, out=out)
            self.assertRelativeEqual(ans, out)
            if (A.numel() > 0):
                self.assertRelativeEqual(expected, out)

        shapes = [(0, 0), (5, 0),  # Empty matrix
                  (5, 5), (5, 3),  # Single matrix
                  (0, 0, 0), (0, 5, 5), (0, 5, 3),  # Zero batch dimension tensors
                  (2, 5, 5), (2, 5, 3),  # 3-dim tensors
                  (2, 1, 5, 5), (2, 1, 5, 3)]  # 4-dim tensors
        for shape in shapes:
            run_test(shape)

    @onlyCUDA
    @dtypes(torch.float32, torch.float64, torch.complex64, torch.complex128)
    @tf32_on_and_off(0.01)
    def test_ormqr(self, device, dtype):

        def run_test(batch, m, n, fortran_contiguous):
            A = make_tensor((*batch, m, n), dtype=dtype, device=device)
            reflectors, tau = torch.geqrf(A)
            if not fortran_contiguous:
                self.assertTrue(reflectors.transpose(-2, -1).is_contiguous())
                reflectors = reflectors.contiguous()

            # Q is of size m x m
            Q, _ = torch.linalg.qr(A, mode='complete')
            C_right = make_tensor((*batch, m, n), dtype=dtype, device=device)
            C_left = make_tensor((*batch, n, m), dtype=dtype, device=device)

            expected = Q @ C_right
            actual = torch.ormqr(reflectors, tau, C_right, left=True, transpose=False)
            self.assertRelativeEqual(expected, actual)

            expected = C_left @ Q
            actual = torch.ormqr(reflectors, tau, C_left, left=False, transpose=False)
            self.assertRelativeEqual(expected, actual)

            expected = Q.transpose(-2, -1).conj() @ C_right
            actual = torch.ormqr(reflectors, tau, C_right, left=True, transpose=True)
            self.assertRelativeEqual(expected, actual)

            expected = C_left @ Q.transpose(-2, -1).conj()
            actual = torch.ormqr(reflectors, tau, C_left, left=False, transpose=True)
            self.assertRelativeEqual(expected, actual)

            # if tau is all zeros then the implicit matrix Q is the identity matrix
            # so the actual result should be C_right in this case
            zero_tau = torch.zeros_like(tau)
            actual = torch.ormqr(reflectors, zero_tau, C_right, left=True, transpose=False)
            self.assertRelativeEqual(C_right, actual)

        batches = [(), (0, ), (2, ), (2, 1)]
        ns = [5, 2, 0]
        for batch, (m, n), fortran_contiguous in product(batches, product(ns, ns), [True, False]):
            run_test(batch, m, n, fortran_contiguous)

    @precisionOverride({torch.bfloat16: 1e-1})
    @dtypes(*(get_all_dtypes()))
    def test_outer(self, device, dtype):
        def run_test_case(a, b):
            if dtype == torch.bfloat16:
                a_np = a.to(torch.double).cpu().numpy()
                b_np = b.to(torch.double).cpu().numpy()
                exact_dtype = False
            else:
                a_np = a.cpu().numpy()
                b_np = b.cpu().numpy()
                exact_dtype = True
            expected = np.outer(a_np, b_np)

            self.assertEqual(torch.outer(a, b), expected, exact_dtype=False)
            self.assertEqual(torch.Tensor.outer(a, b), expected, exact_dtype=False)

            self.assertEqual(torch.ger(a, b), expected, exact_dtype=False)
            self.assertEqual(torch.Tensor.ger(a, b), expected, exact_dtype=False)

            # test out variant
            out = torch.empty(a.size(0), b.size(0), device=device, dtype=dtype)
            torch.outer(a, b, out=out)
            self.assertEqual(out, expected, exact_dtype=False)

            out = torch.empty(a.size(0), b.size(0), device=device, dtype=dtype)
            torch.ger(a, b, out=out)
            self.assertEqual(out, expected, exact_dtype=False)

        a = torch.randn(50).to(device=device, dtype=dtype)
        b = torch.randn(50).to(device=device, dtype=dtype)
        run_test_case(a, b)

        # test 0 strided tensor
        zero_strided = torch.randn(1).to(device=device, dtype=dtype).expand(50)
        run_test_case(zero_strided, b)
        run_test_case(a, zero_strided)

    @precisionOverride({torch.float32: 5e-3, torch.complex64: 1e-3})
    @dtypes(torch.float32, torch.float64, torch.complex64, torch.complex128)
    def test_pinverse(self, device, dtype):
        from torch.testing._internal.common_utils import random_fullrank_matrix_distinct_singular_value as fullrank

        def run_test(M):
            # Testing against definition for pseudo-inverses
            MPI = torch.pinverse(M)
            MPI_ = MPI.cpu().numpy()
            M_ = M.cpu().numpy()
            if M.numel() > 0:
                self.assertRelativeEqual(M_, np.matmul(np.matmul(M_, MPI_), M_))
                self.assertRelativeEqual(MPI_, np.matmul(np.matmul(MPI_, M_), MPI_))
                self.assertRelativeEqual(np.matmul(M_, MPI_), np.matmul(M_, MPI_).swapaxes(-2, -1).conj())
                self.assertRelativeEqual(np.matmul(MPI_, M_), np.matmul(MPI_, M_).swapaxes(-2, -1).conj())
            else:
                self.assertRelativeEqual(M.shape, MPI.shape[:-2] + (MPI.shape[-1], MPI.shape[-2]))
        for sizes in [(5, 5), (3, 5, 5), (3, 7, 5, 5),  # square matrices
                      (3, 2), (5, 3, 2), (7, 5, 3, 2),  # fat matrices
                      (2, 3), (5, 2, 3), (7, 5, 2, 3),  # thin matrices
                      (0, 0), (0, 2), (2, 0), (3, 0, 0), (0, 3, 0), (0, 0, 3)]:  # zero numel matrices
            M = torch.randn(*sizes, dtype=dtype, device=device)
            run_test(M)

        # Test inverse and pseudo-inverse for invertible matrix
        for sizes in [(5, 5), (3, 5, 5), (3, 7, 5, 5)]:
            matsize = sizes[-1]
            batchdims = sizes[:-2]
            M = fullrank(matsize, *batchdims, dtype=dtype, device=device)
            self.assertRelativeEqual(torch.eye(matsize, dtype=dtype, device=device).expand(sizes), M.pinverse().matmul(M),
                             atol=1e-7, rtol=0, msg='pseudo-inverse for invertible matrix')

    @onlyCUDA
    @precisionOverride({torch.float32: 5e-6, torch.complex64: 5e-6})
    @dtypes(torch.float32, torch.float64, torch.complex64, torch.complex128)
    def test_qr(self, device, dtype):
        def run_test(tensor_dims, some):
            A = torch.randn(*tensor_dims, dtype=dtype, device=device)
            Q, R = torch.qr(A, some=some)

            # Check0: Q[-2:] = (m, n_columns), R[-2:] = (n_columns, n)
            m, n = tensor_dims[-2:]
            n_columns = m if (not some) and m > n else min(m, n)
            self.assertRelativeEqual(Q.size(-2), m)
            self.assertRelativeEqual(R.size(-1), n)
            self.assertRelativeEqual(Q.size(-1), n_columns)

            A_ = A.cpu().numpy()
            Q_ = Q.cpu().numpy()
            R_ = R.cpu().numpy()

            # Check1: A = QR
            self.assertRelativeEqual(A_, np.matmul(Q_, R_))

            # Check2: A = QR (with out)
            Q_out, R_out = torch.full_like(Q, math.nan), torch.full_like(R, math.nan)
            torch.qr(A, some=some, out=(Q_out, R_out))
            Q_out_ = Q_out.cpu().numpy()
            R_out_ = R_out.cpu().numpy()
            self.assertRelativeEqual(A_, np.matmul(Q_out_, R_out_))

            # Check3: Q == Q_out, R == R_out
            self.assertRelativeEqual(Q_, Q_out_)
            self.assertRelativeEqual(R_, R_out_)

            # Check4: Q^{T}Q = I, triu(R) = R
            eye = torch.eye(n_columns, device=device, dtype=dtype).expand(Q.shape[:-2] + (n_columns, n_columns)).cpu().numpy()
            self.assertRelativeEqual(np.matmul(Q_.swapaxes(-1, -2).conj(), Q_), eye)
            self.assertRelativeEqual(R.triu(), R)

        tensor_dims_list = [(0, 5), (0, 0), (5, 0),  # Empty Tensors
                            (2, 1, 0, 5), (2, 1, 0, 0), (2, 1, 5, 0), (2, 0, 5, 5),  # Batched empty Tensors
                            (3, 5), (5, 5), (5, 3),  # Single matrix
                            (7, 3, 5), (7, 5, 5), (7, 5, 3),  # 3-dim Tensors
                            (7, 5, 3, 5), (7, 5, 5, 5), (7, 5, 5, 3)]  # 4-dim Tensors
        for tensor_dims, some in product(tensor_dims_list, [True, False]):
            run_test(tensor_dims, some)

    def solve_test_helper(self, A_dims, b_dims, device, dtype):
        from torch.testing._internal.common_utils import random_fullrank_matrix_distinct_singular_value

        b = torch.randn(*b_dims, dtype=dtype, device=device)
        A = random_fullrank_matrix_distinct_singular_value(*A_dims, dtype=dtype, device=device).to(device)
        return b, A

    @onlyCUDA
    @dtypes(torch.float32, torch.float64, torch.complex64, torch.complex128)
    @precisionOverride({torch.float32: 1e-3, torch.complex64: 1e-3})
    def test_solve(self, device, dtype):
        def run_test(n, batch, rhs):
            A_dims = (n, *batch)
            b_dims = (*batch, n, *rhs)
            b, A = self.solve_test_helper(A_dims, b_dims, device, dtype)

            # Correctness test
            x = torch.linalg.solve(A, b)
            if rhs == ():
                Ax = np.matmul(A.cpu(), x.unsqueeze(-1).cpu())
                Ax.squeeze_(-1)
            else:
                Ax = np.matmul(A.cpu(), x.cpu())
            self.assertRelativeEqual(b.expand_as(Ax), Ax)

            # Check against NumPy
            expected = np.linalg.solve(A.cpu().numpy(), b.expand_as(x).cpu().numpy())
            self.assertRelativeEqual(x, expected)

            # Check out= variant
            out = torch.empty_like(x)
            ans = torch.linalg.solve(A, b, out=out)
            self.assertRelativeEqual(ans, out)
            self.assertRelativeEqual(x, out)

            # Check out= variant with complex128 out tensor
            out = torch.empty_like(x).to(torch.complex128)
            ans = torch.linalg.solve(A, b, out=out)
            self.assertRelativeEqual(ans, out)
            self.assertRelativeEqual(x.to(torch.complex128), out)

            # Check empty out
            out = torch.empty(0, dtype=dtype, device=device)
            ans = torch.linalg.solve(A, b, out=out)
            self.assertRelativeEqual(ans, out)
            self.assertRelativeEqual(x, out)

        batches = [(), (0, ), (3, ), (2, 3)]
        ns = [0, 5, 32]
        nrhs = [(), (1, ), (5, )]
        for n, batch, rhs in product(ns, batches, nrhs):
            run_test(n, batch, rhs)

    @onlyCUDA
    @dtypes(torch.double, torch.float32)
    @precisionOverride({torch.float32: 1e-2})
    def test_svd(self, device, dtype):
        def run_test(dims, some, compute_uv):
            x = torch.randn(*dims, dtype=dtype, device=device)
            outu = torch.empty(0, dtype=dtype, device=device)
            outs = torch.empty(0, dtype=dtype, device=device)
            outv = torch.empty(0, dtype=dtype, device=device)
            torch.svd(x, some=some, compute_uv=compute_uv, out=(outu, outs, outv))

            if compute_uv:
                if some:
                    x_recon = torch.matmul(outu, torch.matmul(outs.diag_embed(), outv.transpose(-2, -1)))
                    self.assertRelativeEqual(x, x_recon, atol=1e-8, rtol=0, msg='Incorrect reconstruction using U @ diag(S) @ V.T')
                else:
                    narrow_u = outu[..., :min(*dims[-2:])]
                    narrow_v = outv[..., :min(*dims[-2:])]
                    x_recon = torch.matmul(narrow_u, torch.matmul(outs.diag_embed(), narrow_v.transpose(-2, -1)))
                    self.assertRelativeEqual(x, x_recon, atol=1e-8, rtol=0, msg='Incorrect reconstruction using U @ diag(S) @ V.T')
            else:
                _, singvals, _ = torch.svd(x, compute_uv=True)
                self.assertRelativeEqual(singvals, outs, msg='Singular values mismatch')
                self.assertRelativeEqual(outu, torch.zeros_like(outu), msg='U not zero')
                self.assertRelativeEqual(outv, torch.zeros_like(outv), msg='V not zero')

            resu, ress, resv = torch.svd(x, some=some, compute_uv=compute_uv)
            self.assertRelativeEqual(resu, outu, msg='outputs of svd and svd with out differ')
            self.assertRelativeEqual(ress, outs, msg='outputs of svd and svd with out differ')
            self.assertRelativeEqual(resv, outv, msg='outputs of svd and svd with out differ')

            # test non-contiguous
            x = torch.randn(*dims, dtype=dtype, device=device)
            if x.numel() > 0:
                n_dim = len(dims)
                # Reverse the batch dimensions and the matrix dimensions and then concat them
                x = x.permute(tuple(range(n_dim - 3, -1, -1)) + (n_dim - 1, n_dim - 2))
                assert not x.is_contiguous(), "x is intentionally non-contiguous"
                resu, ress, resv = torch.svd(x, some=some, compute_uv=compute_uv)
                if compute_uv:
                    if some:
                        x_recon = torch.matmul(resu, torch.matmul(ress.diag_embed(), resv.transpose(-2, -1)))
                        self.assertRelativeEqual(x, x_recon, atol=1e-8, rtol=0, msg='Incorrect reconstruction using U @ diag(S) @ V.T')
                    else:
                        narrow_u = resu[..., :min(*dims[-2:])]
                        narrow_v = resv[..., :min(*dims[-2:])]
                        x_recon = torch.matmul(narrow_u, torch.matmul(ress.diag_embed(), narrow_v.transpose(-2, -1)))
                        self.assertRelativeEqual(x, x_recon, atol=1e-8, rtol=0, msg='Incorrect reconstruction using U @ diag(S) @ V.T')
                else:
                    _, singvals, _ = torch.svd(x, compute_uv=True)
                    self.assertRelativeEqual(singvals, ress, msg='Singular values mismatch')
                    self.assertRelativeEqual(resu, torch.zeros_like(resu), msg='U not zero')
                    self.assertRelativeEqual(resv, torch.zeros_like(resv), msg='V not zero')

        shapes = [(0, 0), (5, 0), (0, 5),  # empty matrices
                  (0, 0, 0), (0, 5, 5), (0, 5, 3),  # zero batch dimension
                  (3, 3), (5, 3, 3), (7, 5, 3, 3),  # square matrices
                  (7, 3), (5, 7, 3), (7, 5, 7, 3),  # fat matrices
                  (3, 7), (5, 3, 7), (7, 5, 3, 7)]  # thin matrices
        for dims, some, compute_uv in product(shapes, [True, False], [True, False]):
            run_test(dims, some, compute_uv)

    @onlyCUDA
    @dtypes(torch.double)
    def test_svd_lowrank(self, device, dtype):
        from torch.testing._internal.common_utils import random_lowrank_matrix, random_sparse_matrix

        def run_subtest(actual_rank, matrix_size, batches, device, svd_lowrank, **options):
            density = options.pop('density', 1)
            if isinstance(matrix_size, int):
                rows = columns = matrix_size
            else:
                rows, columns = matrix_size
            if density == 1:
                a_input = random_lowrank_matrix(actual_rank, rows, columns, *batches, device=device, dtype=dtype)
                a = a_input
            else:
                assert batches == ()
                a_input = random_sparse_matrix(rows, columns, density, device=device, dtype=dtype)
                a = a_input.to_dense()

            q = min(*size)
            u, s, v = svd_lowrank(a_input, q=q, **options)

            # check if u, s, v is a SVD
            u, s, v = u[..., :q], s[..., :q], v[..., :q]
            A = u.matmul(s.diag_embed()).matmul(v.transpose(-2, -1))
            self.assertRelativeEqual(A, a, rtol=1e-7, atol=2e-7)

            # check if svd_lowrank produces same singular values as torch.svd
            U, S, V = torch.svd(a)
            self.assertRelativeEqual(s.shape, S.shape)
            self.assertRelativeEqual(u.shape, U.shape)
            self.assertRelativeEqual(v.shape, V.shape)
            self.assertRelativeEqual(s, S)

            if density == 1:
                # actual_rank is known only for dense inputs
                #
                # check if pairs (u, U) and (v, V) span the same
                # subspaces, respectively
                u, s, v = u[..., :actual_rank], s[..., :actual_rank], v[..., :actual_rank]
                U, S, V = U[..., :actual_rank], S[..., :actual_rank], V[..., :actual_rank]
                self.assertRelativeEqual(u.transpose(-2, -1).matmul(U).det().abs(), torch.ones(batches, device=device, dtype=dtype))
                self.assertRelativeEqual(v.transpose(-2, -1).matmul(V).det().abs(), torch.ones(batches, device=device, dtype=dtype))

        all_batches = [(), (1,), (3,), (2, 3)]
        for actual_rank, size, all_batches in [
                (2, (17, 4), all_batches),
                (4, (17, 4), all_batches),
                (4, (17, 17), all_batches),
                (10, (100, 40), all_batches),
                (7, (1000, 1000), [()]),
        ]:
            # dense input
            for batches in all_batches:
                run_subtest(actual_rank, size, batches, device, torch.svd_lowrank)
                if size != size[::-1]:
                    run_subtest(actual_rank, size[::-1], batches, device, torch.svd_lowrank)

        # sparse input
        for size in [(17, 4), (4, 17), (17, 17), (100, 40), (40, 100), (1000, 1000)]:
            for density in [0.005, 0.1]:
                run_subtest(None, size, (), device, torch.svd_lowrank, density=density)

        # jitting support
        jitted = torch.jit.script(torch.svd_lowrank)
        actual_rank, size, batches = 2, (17, 4), ()
        run_subtest(actual_rank, size, batches, device, jitted)

    @onlyCUDA
    @dtypes(torch.double)
    def test_pca_lowrank(self, device, dtype):
        from torch.testing._internal.common_utils import random_lowrank_matrix, random_sparse_matrix


        def run_subtest(guess_rank, actual_rank, matrix_size, batches, device, pca, **options):
            density = options.pop('density', 1)
            if isinstance(matrix_size, int):
                rows = columns = matrix_size
            else:
                rows, columns = matrix_size
            if density == 1:
                a_input = random_lowrank_matrix(actual_rank, rows, columns, *batches, device=device, dtype=dtype)
                a = a_input
            else:
                a_input = random_sparse_matrix(rows, columns, density, device=device, dtype=dtype)
                a = a_input.to_dense()

            u, s, v = pca(a_input, q=guess_rank, **options)

            self.assertRelativeEqual(s.shape[-1], guess_rank)
            self.assertRelativeEqual(u.shape[-2], rows)
            self.assertRelativeEqual(u.shape[-1], guess_rank)
            self.assertRelativeEqual(v.shape[-1], guess_rank)
            self.assertRelativeEqual(v.shape[-2], columns)

            A1 = u.matmul(s.diag_embed()).matmul(v.transpose(-2, -1))
            ones_m1 = torch.ones(batches + (rows, 1), dtype=a.dtype, device=device)
            c = a.sum(axis=-2) / rows
            c = c.reshape(batches + (1, columns))
            A2 = a - ones_m1.matmul(c)
            self.assertRelativeEqual(A1, A2)

            if density == 1:
                # actual rank is known only for dense input
                detect_rank = (s.abs() > 1e-5).sum(axis=-1)
                self.assertRelativeEqual(actual_rank * torch.ones(batches, device=device, dtype=torch.int64), detect_rank)
                S = torch.linalg.svdvals(A2)
                self.assertRelativeEqual(s[..., :actual_rank], S[..., :actual_rank])

        all_batches = [(), (1,), (3,), (2, 3)]
        for actual_rank, size, all_batches in [
                (2, (17, 4), all_batches),
                (2, (100, 4), all_batches),
                (6, (100, 40), all_batches),
                (12, (1000, 1000), [()]),
        ]:
            for batches in all_batches:
                for guess_rank in [
                        actual_rank,
                        actual_rank + 2,
                        actual_rank + 6,
                ]:
                    if guess_rank <= min(*size):
                        run_subtest(guess_rank, actual_rank, size, batches, device, torch.pca_lowrank)
                        run_subtest(guess_rank, actual_rank, size[::-1], batches, device, torch.pca_lowrank)

        # sparse input
        for guess_rank, size in [
                (4, (17, 4)), (4, (4, 17)), (16, (17, 17)),
                (21, (100, 40)), (20, (40, 100)), (600, (1000, 1000))]:
            for density in [0.005, 0.1]:
                run_subtest(guess_rank, None, size, (), device, torch.pca_lowrank, density=density)

        # jitting support
        jitted = torch.jit.script(torch.pca_lowrank)
        guess_rank, actual_rank, size, batches = 2, 2, (17, 4), ()
        run_subtest(guess_rank, actual_rank, size, batches, device, jitted)

    @onlyCUDA
    @dtypes(torch.double, torch.float32, torch.half, torch.bfloat16)
    @precisionOverride({torch.half: 1e-3})
    def test_trapezoid(self, device, dtype):
        def test_dx(sizes, dim, dx, device):
            t = torch.randn(sizes, device=device, dtype=dtype)
            actual = torch.trapz(t, dx=dx, dim=dim)
            if dtype == torch.bfloat16:
                expected = np.trapz(t.to(torch.double).cpu().numpy(), dx=dx, axis=dim)
            else:
                expected = np.trapz(t.cpu().numpy(), dx=dx, axis=dim)
            self.assertEqual(expected.shape, actual.shape)
            self.assertEqual(expected, actual, exact_dtype=False)

        def test_x(sizes, dim, x, device):
            t = torch.randn(sizes, device=device)
            actual = torch.trapz(t, x=torch.tensor(x, device=device), dim=dim)
            expected = np.trapz(t.cpu().numpy(), x=x, axis=dim)
            self.assertEqual(expected.shape, actual.shape)
            self.assertEqual(expected, actual.cpu(), exact_dtype=False)

        test_dx((2, 3, 4), 1, 1, device)
        test_dx((10, 2), 0, 0.1, device)
        test_dx((1, 10), 0, 2.3, device)
        test_dx((0, 2), 0, 1.0, device)
        test_dx((0, 2), 1, 1.0, device)
        test_x((2, 3, 4), 1, [1.0, 2.0, 3.0], device)
        test_x((10, 2), 0, [2.0, 3.0, 4.0, 7.0, 11.0, 14.0, 22.0, 26.0, 26.1, 30.3], device)
        test_x((1, 10), 0, [1.0], device)
        test_x((0, 2), 0, [], device)
        test_x((0, 2), 1, [1.0, 2.0], device)
        test_x((2, 3, 4), -1, [1.0, 2.0, 3.0, 4.0], device)
        test_x((2, 3, 4), 0, [1.0, 2.0], device)
        test_x((2, 3, 4), 1, [1.0, 2.0, 3.0], device)
        test_x((2, 3, 4), 2, [1.0, 2.0, 3.0, 4.0], device)
        test_x((2, 2, 4), -1, [[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]], device)
        with self.assertRaisesRegex(
                IndexError,
                'Dimension out of range'):
            test_x((2, 3), 2, [], device)
            test_dx((2, 3), 2, 1.0, device)
        with self.assertRaisesRegex(
                RuntimeError,
                'There must be one `x` value for each sample point'):
            test_x((2, 3), 1, [1.0, 2.0], device)
            test_x((2, 3), 1, [1.0, 2.0, 3.0, 4.0], device)

    @onlyCUDA
    @dtypes(torch.double, torch.float32)
    def test_cumulative_trapezoid(self, device, dtype):

        if hasattr(scipy.integrate, 'cumulative_trapezoid'):
            scipy_cumulative_trapezoid = scipy.integrate.cumulative_trapezoid
        else:  # Older version of SciPy uses a different name
            scipy_cumulative_trapezoid = scipy.integrate.cumtrapz

        def test_dx(sizes, dim, dx, device):
            t = torch.randn(sizes, device=device, dtype=dtype)
            y = t.cpu().numpy()
            actual = torch.cumulative_trapezoid(t, dx=dx, dim=dim)
            expected = scipy_cumulative_trapezoid(t.cpu().numpy(), dx=dx, axis=dim)
            self.assertRelativeEqual(expected.shape, actual.shape)
            self.assertRelativeEqual(expected, actual, exact_dtype=False)

        def test_x(sizes, dim, x, device):
            t = torch.randn(sizes, device=device)
            actual = torch.cumulative_trapezoid(t, x=torch.tensor(x, device=device), dim=dim)
            expected = scipy_cumulative_trapezoid(t.cpu().numpy(), x=x, axis=dim)
            self.assertRelativeEqual(expected.shape, actual.shape)
            self.assertRelativeEqual(expected, actual.cpu(), exact_dtype=False)

        def test_empty_x(sizes, dim, x, device):
            t = torch.randn(sizes, device=device)
            actual = torch.cumulative_trapezoid(t, x=torch.tensor(x, device=device), dim=dim)
            self.assertRelativeEqual(torch.empty(actual.shape), actual)

        test_dx((2,), -1, 1, device)
        test_dx((3, 3), -1, 1, device)
        test_dx((4, 2), 0, 1, device)
        test_dx((2, 3, 4), 1, 1, device)
        test_dx((10, 2), 0, 0.1, device)
        test_dx((1, 10), 0, 2.3, device)
        test_dx((0, 2), 0, 1.0, device)
        test_dx((0, 2), 1, 1.0, device)
        test_dx((512, 512), 1, 1.0, device)
        test_dx((100, 100, 100), 1, 1.0, device)

        test_x((2,), -1, [100, 50], device)
        test_x((4, 2), 0, [2, 3, 4, 5], device)
        test_x((2, 3, 4), 1, [1.0, 2.0, 3.0], device)
        test_x((10, 2), 0, [2.0, 3.0, 4.0, 7.0, 11.0, 14.0, 22.0, 26.0, 26.1, 30.3], device)
        test_x((1, 10), 0, [1.0], device)
        test_x((0, 2), 1, [1, 2], device)
        test_x((2, 3, 4), -1, [1.0, 2.0, 3.0, 4.0], device)
        test_x((2, 3, 4), 0, [1.0, 2.0], device)
        test_x((2, 3, 4), 1, [1.0, 2.0, 3.0], device)
        test_x((2, 3, 4), 2, [1.0, 2.0, 3.0, 4.0], device)

        test_empty_x((0, 2), 0, [], device)  # SciPy failing when x == [], but our version returns empty

        with self.assertRaisesRegex(
                IndexError,
                'Dimension out of range'):
            test_x((2, 3), 2, [], device)
            test_dx((2, 3), 2, 1.0, device)
        with self.assertRaisesRegex(
                RuntimeError,
                'There must be one `x` value for each sample point'):
            test_x((2, 3), 1, [1.0, 2.0], device)
            test_x((0, 2), 0, [1.0, 2.0], device)
            test_x((2, 3), 1, [1.0, 2.0, 3.0, 4.0], device)
        with self.assertRaisesRegex(
                RuntimeError,
                'Currently, we only support dx as a real number'):
            test_dx((2, 2), -1, complex(1, 1) , device)
        with self.assertRaisesRegex(
                TypeError, 'received an invalid combination of arguments'):
            actual = torch.cumulative_trapezoid(torch.randn((3, 3)), x=torch.randn((3, 3)), dx=3)

    def triangular_solve_test_helper(self, A_dims, b_dims, upper, unitriangular,
                                     device, dtype):
        triangle_function = torch.triu if upper else torch.tril
        b = torch.randn(*b_dims, dtype=dtype, device=device)
        A = torch.randn(*A_dims, dtype=dtype, device=device)
        # create positive definite matrix
        A = torch.matmul(A, A.transpose(-2, -1))
        A_triangular = triangle_function(A)
        if unitriangular:
            A_triangular.diagonal(dim1=-2, dim2=-1).fill_(1.)
        return b, A_triangular

    @onlyCUDA
    @dtypes(torch.float32, torch.float64, torch.complex64, torch.complex128)
    @precisionOverride({torch.float32: 1e-3, torch.complex64: 1e-3,
                        torch.float64: 1e-8, torch.complex128: 1e-8})
    def test_triangular_solve(self, device, dtype):
        ks = [0, 1, 3]
        ns = [0, 5]
        for (k, n), (upper, unitriangular, transpose) in product(zip(ks, ns), product([True, False], repeat=3)):
            b, A = self.triangular_solve_test_helper((n, n), (n, k), upper,
                                                     unitriangular, device, dtype)
            x = torch.triangular_solve(b, A, upper=upper, unitriangular=unitriangular, transpose=transpose)[0]
            if transpose:
                self.assertRelativeEqual(b, np.matmul(A.t().cpu(), x.cpu()))
            else:
                self.assertRelativeEqual(b, np.matmul(A.cpu(), x.cpu()))

    def _test_dot_vdot_vs_numpy(self, device, dtype, torch_fn, np_fn):
        def check(x, y):
            # Compare with numpy
            res = torch_fn(x, y)
            ("res:", res)
            if x.dtype == torch.bfloat16:
                ref = torch.from_numpy(np.array(np_fn(x.cpu().float().numpy(), y.cpu().float().numpy())))
            else:
                ref = torch.from_numpy(np.array(np_fn(x.cpu().numpy(), y.cpu().numpy())))
            if res.dtype == torch.bfloat16:
                self.assertEqual(res.cpu(), ref.bfloat16())
            else:
                self.assertEqual(res.cpu(), ref)

            # Test out variant
            out = torch.empty_like(res)
            torch_fn(x, y, out=out)
            self.assertEqual(out, res)

        # Empty
        x = torch.tensor([], dtype=dtype, device=device)
        y = torch.tensor([], dtype=dtype, device=device)
        check(x, y)

        # Contiguous
        x = 0.1 * torch.randn(5000, dtype=dtype, device=device)
        y = 0.1 * torch.randn(5000, dtype=dtype, device=device)
        check(x, y)

        # 0 strided
        y = 0.1 * torch.randn(1, dtype=dtype, device=device).expand(5000)
        check(x, y)

        # 2 strided
        check(x[::2], y[::2])

    @onlyCUDA
    @dtypes(torch.double, torch.float32, torch.cfloat, torch.half)
    @precisionOverride({torch.cfloat: 1e-4, torch.float32: 5e-5})
    def test_vdot_vs_numpy(self, device, dtype):
        self._test_dot_vdot_vs_numpy(device, dtype, torch.vdot, np.vdot)

    @onlyCUDA
    def test_compiled_with_cxx11_abi(self):
        self.assertRelativeEqual(torch.compiled_with_cxx11_abi(), True)

    @onlyCUDA
    @dtypes(*product(get_all_dtypes(), get_all_dtypes()))
    def test_result_type(self, device, dtypes):
        "Test result_type for tensor vs tensor and scalar vs scalar."

        def _get_dtype(x):
            "Get the dtype of x if x is a tensor. If x is a scalar, get its corresponding dtype if it were a tensor."
            if torch.is_tensor(x):
                return x.dtype
            elif isinstance(x, bool):
                return torch.bool
            elif isinstance(x, int):
                return torch.int64
            elif isinstance(x, float):
                return torch.float32
            elif isinstance(x, complex):
                return torch.complex64
            else:
                raise AssertionError(f"Unkonwn type {x}")

        # tensor against tensor
        a_tensor = torch.tensor((0, 1), device=device, dtype=dtypes[0])
        a_single_tensor = torch.tensor(1, device=device, dtype=dtypes[0])
        a_scalar = a_single_tensor.item()
        b_tensor = torch.tensor((1, 0), device=device, dtype=dtypes[1])
        b_single_tensor = torch.tensor(1, device=device, dtype=dtypes[1])
        b_scalar = b_single_tensor.item()
        combo = ((a_tensor, a_single_tensor, a_scalar), (b_tensor, b_single_tensor, b_scalar))
        for a, b in product(*combo):
            dtype_a = _get_dtype(a)
            dtype_b = _get_dtype(b)
            try:
                result = a + b
            except RuntimeError:
                with self.assertRaises(RuntimeError):
                    torch.promote_types(dtype_a, dtype_b)
                with self.assertRaises(RuntimeError):
                    torch.result_type(a, b)
            else:
                dtype_res = _get_dtype(result)
                if a is a_scalar and b is b_scalar and dtype_a == torch.bool and dtype_b == torch.bool:
                    # special case: in Python, True + True is an integer
                    self.assertRelativeEqual(dtype_res, torch.int64)
                else:
                    self.assertRelativeEqual(dtype_res, torch.result_type(a, b))
                if a is a_scalar and b is b_scalar:  # Python internal type determination is good enough in this case
                    continue
                if any(a is a0 and b is b0 for a0, b0 in zip(*combo)):  # a and b belong to the same class
                    self.assertRelativeEqual(dtype_res, torch.promote_types(dtype_a, dtype_b))

    def test_can_cast(self):
        self.assertTrue(torch.can_cast(torch.double, torch.float32))
        self.assertFalse(torch.can_cast(torch.float32, torch.int))

    @onlyCUDA
    def test_promote_types(self):
        self.assertRelativeEqual(torch.promote_types(torch.float32, torch.int), torch.float32)
        self.assertRelativeEqual(torch.promote_types(torch.float32, torch.double), torch.double)
        self.assertRelativeEqual(torch.promote_types(torch.int, torch.uint8), torch.int)

    def test_use_deterministic_algorithms(self):
        for deterministic in [True, False]:
            torch.use_deterministic_algorithms(deterministic)
            self.assertRelativeEqual(deterministic, torch.are_deterministic_algorithms_enabled())

        with self.assertRaisesRegex(TypeError, r"_set_deterministic_algorithms():*"):
            torch.use_deterministic_algorithms(1)

    def test_warn_always(self):
        prev = torch.is_warn_always_enabled()
        torch.set_warn_always(True)
        ret = torch.is_warn_always_enabled()
        self.assertRelativeEqual(ret, True)
        torch.set_warn_always(False)
        ret = torch.is_warn_always_enabled()
        self.assertRelativeEqual(ret, False)
        torch.set_warn_always(prev)

    def test_assert(self):
        # verify assertions work as expected
        # bool argument
        torch._assert(True, "foo")
        with self.assertRaisesRegex(AssertionError, "bar"):
            torch._assert(False, "bar")
        # tensor argument
        torch._assert(torch.tensor([True], dtype=torch.bool), "foo")
        with self.assertRaisesRegex(AssertionError, "bar"):
            torch._assert(torch.tensor([False], dtype=torch.bool), "bar")



instantiate_device_type_tests(TestTorchFunctionSegment0, globals())

if __name__ == "__main__":
    unittest.main()
