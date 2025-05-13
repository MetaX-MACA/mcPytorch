import torch
import copy
import unittest
import torch.nn as nn
from itertools import product

from common_util import *

class TestTorchFunctionSegment2(unittest.TestCase):

    def none_kernel_help(self, input, op, dtype, device, kwargs={}, need_to_cuda=[], is_backward=True):
        input_list, input_g_list = [], []
        if type(input) == list:
            for input_value in input:
                input_g_list.append(copy.deepcopy(input_value).to(device))
            input_list = tuple(input)
            input_g_list = tuple(input_g_list)
        else:
            input_list = input
            input_g_list = copy.deepcopy(input).to(device)
        cpu_forward = op(input_list, **kwargs)
        cuda_kwargs = {}
        for key, value in kwargs.items():
            if key in need_to_cuda and  hasattr(value, "to"):
                cuda_kwargs[key] = value.to(device) if hasattr(value, "to") else value
            else:
                cuda_kwargs[key] = value
        cuda_forward = op(input_g_list, **cuda_kwargs)
        if (dtype in ALL_FLOATING_TYPES or dtype in ALL_COMPLEX_TYPES) and is_backward:
            if type(cpu_forward) == List or type(cpu_forward) == tuple:
                for idx in range(len(cpu_forward)):
                    cpu_res = cpu_forward[idx]
                    cpu_res.retain_grad()
                    grad = make_tensor(cpu_res.shape, "cpu", dtype)
                    cpu_res.backward(grad)

                    cuda_res = cuda_forward[idx]
                    grad_g = grad.to(device)
                    cuda_res.retain_grad()
                    cuda_res.backward(grad_g)
                    assert torch.equal(cpu_res, cuda_res.cpu())
                    assert torch.equal(cpu_res.grad, cuda_res.grad.cpu())
            else:
                cpu_forward.retain_grad()
                grad = make_tensor(cpu_forward.shape, "cpu", dtype)
                cpu_forward.backward(grad)

                grad_g = grad.to(device)
                cuda_forward.retain_grad()
                cuda_forward.backward(grad_g)

                assert torch.equal(cpu_forward, cuda_forward.cpu())
                assert torch.equal(cpu_forward.grad, cuda_forward.grad.cpu())
        else:
            if type(cpu_forward) == List or type(cpu_forward) == tuple:
                for idx in range(len(cpu_forward)):
                    cpu_res = cpu_forward[idx]
                    cuda_res = cuda_forward[idx]
                    assert torch.equal(cpu_res, cuda_res.cpu())
            else:
                assert torch.equal(cpu_forward, cuda_forward.cpu())

    ##### index_add
    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_dtypes(include_bool=False)))
    def test_index_add(self, device, dtype):

        IDX_DTYPES = [torch.int, torch.long]

        def help(size, dim, indexIsMajor=True):
            for idx_dtype in IDX_DTYPES:
                tensor = make_tensor(size, device=device, dtype=dtype)
                zeros = torch.zeros(size, dtype=dtype, device=device)
                if not indexIsMajor:
                    for i in range(0, len(size)-1):
                        zeros = zeros.transpose(i+1, i)
                added = zeros.index_add(dim, torch.arange(0, size[0], dtype=idx_dtype, device=device), tensor)
                assert torch.equal(added, tensor)
                # alpha = -1
                added = zeros.index_add(dim, torch.arange(0, size[0], dtype=idx_dtype, device=device), tensor, alpha=-1)
                assert torch.equal(added, -tensor)

        # numIndex <= 16
        help([16], 0)
        help([16, 16], 0)
        help([16, 16, 16], 1)
        help([16, 16, 16, 16], 1, False)

        # numIndex > 16
        help([32], 0)
        help([32, 32], 0)
        help([32, 32], 0, False)
        help([32, 32, 32], 1)
        help([32, 32, 32], 1, False)
        help([32, 32, 32, 32], 1, False)

    ##### index_select
    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_dtypes()))
    def test_index_select(self, device, dtype):
        shape_list = [(16, 65), (16, 65, 64), (16, 65, 64, 32)]
        for shape in shape_list:
            input = make_tensor(shape, "cpu", dtype, requires_grad=True)
            index = torch.randint(0, 16, (16,), dtype=torch.int64)
            for i in range(len(shape)):
                self.none_kernel_help(input, torch.index_select, dtype, device, {"dim":i, "index":index}, need_to_cuda=["index"])

    ##### masked_select 
    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_dtypes(include_complex=False)))
    def test_masked_select(self, device, dtype):
        shape_list = [(16, 65), (4, 65, 63), (4, 65, 13, 32)]
        for shape in shape_list:
            input = make_tensor(shape, "cpu", dtype, low=0, high=1, requires_grad=True)
            mask = input.ge(0.5)
            self.none_kernel_help(input, torch.masked_select, dtype, device, {"mask":mask}, need_to_cuda=["mask"])

    ##### movedim use the permute op
    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_dtypes()))
    def test_movedim(self, device, dtype):
        shape_list = [(16, 65), (4, 65, 63), (4, 65, 13, 32)]
        for shape in shape_list:
            input = make_tensor(shape, "cpu", dtype, requires_grad=True)
            for i in range(1, len(shape)):
                self.none_kernel_help(input, torch.movedim, dtype, device, {"source":i, "destination":i-1})
                self.none_kernel_help(input, torch.movedim, dtype, device, {"source":[i], "destination":[i-1]})

    ##### moveaxis use the permute op
    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_dtypes()))
    def test_moveaxis(self, device, dtype):
        shape_list = [(16, 65), (4, 65, 63), (4, 65, 13, 32)]
        for shape in shape_list:
            input = make_tensor(shape, "cpu", dtype, requires_grad=True)
            for i in range(1, len(shape)):
                self.none_kernel_help(input, torch.moveaxis, dtype, device, {"source":i, "destination":i-1})
                self.none_kernel_help(input, torch.moveaxis, dtype, device, {"source":[i], "destination":[i-1]})

    ##### narrow
    @onlyCUDA
    @dtypesIfCUDA(*get_all_dtypes())
    def test_narrow(self, device, dtype):
        x = make_tensor((3, 3), "cpu", dtype, requires_grad=True)
        self.none_kernel_help(x, torch.narrow, dtype, device, {"dim":0, "start":0,"length":1})
        self.none_kernel_help(x, torch.narrow, dtype, device, {"dim":0, "start":0,"length":2})
        self.none_kernel_help(x, torch.narrow, dtype, device, {"dim":0, "start":1,"length":1})
        self.none_kernel_help(x, torch.narrow, dtype, device, {"dim":0, "start":-1,"length":1})
        self.none_kernel_help(x, torch.narrow, dtype, device, {"dim":0, "start":-2,"length":2})
        self.none_kernel_help(x, torch.narrow, dtype, device, {"dim":0, "start":-3,"length":3})
        self.none_kernel_help(x, torch.narrow, dtype, device, {"dim":-1, "start":-1,"length":1})
        self.none_kernel_help(x, torch.narrow, dtype, device, {"dim":-2, "start":-1,"length":1})

    ##### narrow_tensor
    @onlyCUDA
    @dtypesIfCUDA(*get_all_dtypes())
    def test_narrow_tensor(self, device, dtype):
        x = torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]], device=device, dtype=dtype)
        assert torch.equal(x.narrow(0, torch.tensor(0), 1), torch.tensor([[0, 1, 2]], device=device, dtype=dtype))
        with self.assertRaises(Exception):
            x.narrow(0, torch.tensor(0.), 1)
        with self.assertRaises(Exception):
            x.narrow(0, torch.tensor([0]), 1)
        with self.assertRaises(Exception):
            x.narrow(0, torch.tensor([0, 1]), 1)

    ##### nonzero
    @onlyCUDA
    @dtypesIfCUDA(*get_all_dtypes())
    def test_nonzero(self, device, dtype):
        # torch.nonzero will raise an error in maca env when dim1>=256 before
        shape = [32, 2 ** 8]
        a = torch.zeros(shape, device=device, dtype=dtype)
        a[0:shape[0], shape[1]-1] = 1
        out = torch.nonzero(a)
        ref = torch.tensor([(x, shape[1]-1) for x in range(0, shape[0])], device=device)
        assert torch.equal(out, ref)

        # dim1 = 2 ** 10
        shape = [32, 2 ** 10]
        a = torch.zeros(shape, device=device, dtype=dtype)
        a[0:shape[0], shape[1]-1] = 1
        out = torch.nonzero(a)
        ref = torch.tensor([(x, shape[1]-1) for x in range(0, shape[0])], device=device)
        assert torch.equal(out, ref)

    @onlyCUDA
    def test_nonzero_empty(self, device):
        def assert_tuple_empty(tup, dim):
            self.assertEqual(dim, len(tup))
            for t in tup:
                self.assertEqual(torch.Size([0]), t.shape)

        x = torch.randn(0, 2, 0, 5, 0, device=device)
        y = torch.nonzero(x)
        z = torch.nonzero(x, as_tuple=True)

        self.assertEqual(0, y.numel())
        self.assertEqual(torch.Size([0, 5]), y.shape)
        assert_tuple_empty(z, 5)

        x = torch.tensor(0.5, device=device)
        y = torch.nonzero(x)
        # nonzero with as_tuple returns a
        # tuple of len 1 for a zero-dim tensor.
        # This is done to match Numpy behavior.
        z = torch.nonzero(x, as_tuple=True)
        self.assertEqual(1, len(z))
        self.assertEqual(torch.zeros(1, dtype=torch.long), z[0].cpu())

        x = torch.zeros((), device=device)
        y = torch.nonzero(x)
        z = torch.nonzero(x, as_tuple=True)
        self.assertEqual(torch.Size([0, 0]), y.shape)
        self.assertEqual(1, len(z))
        assert torch.equal(torch.empty(0, dtype=torch.long), z[0].cpu())

    @onlyCUDA
    @dtypesIfCUDA(*get_all_dtypes())
    def test_nonzero_noncontiguous(self, device, dtype):
        x = make_tensor((10, 10, 10), dtype=dtype, device=device,
                        low=1, noncontiguous=False)
        mask = make_tensor((10, 10, 10), dtype=torch.bool, device=device)
        x[mask] = 0

        def permute_storage(tensor, dims):
            dest_dims = tuple(range(len(dims)))
            return tensor.permute(dims).contiguous().movedim(dims, dest_dims)

        # Assume contiguous case is correct
        expect = x.nonzero()

        # Dense, permuted
        assert torch.equal(permute_storage(x, [0, 2, 1]).nonzero(), expect)
        assert torch.equal(permute_storage(x, [2, 1, 0]).nonzero(), expect)

        # Non-dense
        nondense = torch.empty((40, 10, 20), dtype=dtype, device=device)[::4, :, ::2]
        nondense[:] = x
        assert torch.equal(nondense.nonzero(), expect)

        # Non-dense, permuted
        nondense = nondense.permute([0, 2, 1])
        nondense[:] = x
        assert torch.equal(nondense.nonzero(), expect)

    ##### permute
    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_dtypes()))
    def test_permute(self, device, dtype):
        orig = [1, 2, 3, 4, 5, 6, 7]
        perm = torch.randperm(7, device=device).tolist()
        x = torch.empty(*orig, dtype=dtype, device=device).fill_(0)
        new = [i - 1 for i in x.permute(*perm).size()]
        self.assertEqual(perm, new)
        self.assertEqual(list(x.size()), orig)

        input = make_tensor((32, 65), "cpu", dtype, requires_grad=True)
        self.none_kernel_help(input, torch.permute, dtype, device, {"dims":(1, 0)})

        input = make_tensor((32, 65, 64), "cpu", dtype, requires_grad=True)
        self.none_kernel_help(input, torch.permute, dtype, device, {"dims":(2, 0, 1)})

    ##### reshape
    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_fp_dtypes()))
    def test_reshape(self, device, dtype):
        input = make_tensor((8), "cpu", dtype, requires_grad=True)
        self.none_kernel_help(input, torch.reshape, dtype, device, {"shape":(2, 4)})

    #### row_stack same as vstack
    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_dtypes()))
    def test_row_stack(self, device, dtype):
        shape_list = [(16, 65), (4, 65, 63), (4, 65, 13, 32)]
        for shape in shape_list:
            a = make_tensor(shape, "cpu", dtype, requires_grad=True)
            b = make_tensor(shape, "cpu", dtype, requires_grad=True)
            cpu_forward = torch.row_stack((a, b))
            a_g = copy.deepcopy(a).to(device)
            b_g = copy.deepcopy(b).to(device)
            cuda_forward = torch.row_stack((a_g, b_g))
            if dtype in ALL_FLOATING_TYPES or dtype in ALL_COMPLEX_TYPES:
                cpu_forward.retain_grad()
                grad = make_tensor(cpu_forward.shape, "cpu", dtype)
                cpu_forward.backward(grad)

                grad_g = grad.to(device)
                cuda_forward.retain_grad()
                cuda_forward.backward(grad_g)

                assert torch.equal(cpu_forward.grad, cuda_forward.grad.cpu())
            assert torch.equal(cpu_forward, cuda_forward.cpu())

    ##### select call as_stride op
    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_dtypes()))
    def test_select(self, device, dtype):
        shape_list = [(16, 65), (4, 65, 63), (4, 65, 13, 32)]
        for shape in shape_list:
            for i in range(len(shape)):
                input = make_tensor(shape, "cpu", dtype, requires_grad=True)
                self.none_kernel_help(input, torch.select, dtype, device, {"dim":i, "index":i}, is_backward=True)

    ##### scatter see test_torch.py-test_scatter func, index should unique other the behavior is non-deterministic
    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_dtypes()))
    def test_scatter(self, device, dtype):
        shape_list = [(32, 65), (32, 65, 128)]
        for shape in shape_list:
            input = make_tensor(shape, "cpu", dtype, requires_grad=True)
            index = torch.tensor([0, 3, 1, 4], dtype=torch.int64)
            for i in range(1, len(shape)):
                index = torch.unsqueeze(index, i)
            for i in range(len(shape)):
                self.none_kernel_help(input, torch.scatter, dtype, device, {"dim":i, "index":index, "value":3}, need_to_cuda=["index"], is_backward=True)

    ##### scatter_add see test_torch.py-test_scatterAdd func
    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_dtypes(include_bool=False)))
    def test_scatter_add(self, device, dtype):
        shape_list = [(32, 65), (32, 65, 128)]
        for shape in shape_list:
            input = make_tensor(shape, "cpu", dtype, requires_grad=True)
            index = torch.tensor([0, 3, 1, 4], dtype=torch.int64)
            for i in range(1, len(shape)):
                index = torch.unsqueeze(index, i)
            src = make_tensor(shape, "cpu", dtype)
            for i in range(len(shape)):
                self.none_kernel_help(input, torch.scatter_add, dtype, device, {"dim":i, "index":index, "src":src}, need_to_cuda=["index", "src"], is_backward=True)

    ##### split see test_torch.py-test_split func
    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_dtypes()))
    def test_split(self, device, dtype):
        shape_list = [(32, 65), (32, 65, 128)]
        for shape in shape_list:
            for i in range(len(shape)):
                input = make_tensor(shape, "cpu", dtype, requires_grad=True)
                self.none_kernel_help(input, torch.split, dtype, device, {"split_size_or_sections":2, "dim":i}, is_backward=False)
            self.none_kernel_help(input, torch.split, dtype, device, {"split_size_or_sections":[8, 24], "dim":0}, is_backward=False)

    ##### squeeze see test_torch.py-test_squeeze func
    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_dtypes()))
    def test_squeeze(self, device, dtype):
        shape_list = [(31, 1), (32, 1, 65), (32, 1, 65, 128)]
        for shape in shape_list:
            for i in range(len(shape)):
                input = make_tensor(shape, "cpu", dtype, requires_grad=True)
                self.none_kernel_help(input, torch.squeeze, dtype, device, {"dim":i}, is_backward=True)

    #### stack call the cat op
    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_dtypes()))
    def test_stack(self, device, dtype):
        shape_list = [(32, 65), (32, 65, 128), (32, 16, 8, 16), (4, 32, 16, 8, 16)]
        for shape in shape_list:
            for i in range(len(shape)):
                input1 = make_tensor(shape, "cpu", dtype, requires_grad=True)
                input2 = make_tensor(shape, "cpu", dtype, requires_grad=True)
                self.none_kernel_help([input1, input2], torch.stack, dtype, device, {"dim":i}, is_backward=True)

    ##### swapaxes call the transpose op
    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_dtypes()))
    def test_swapaxes(self, device, dtype):
        shape_list = [(32, 65), (32, 65, 128), (32, 16, 8, 16), (4, 32, 16, 8, 16)]
        for shape in shape_list:
            for i in range(1, len(shape)):
                input = make_tensor(shape, "cpu", dtype, requires_grad=True)
                self.none_kernel_help(input, torch.swapaxes, dtype, device, {"axis0": i, "axis1": i-1}, is_backward=True)
                if i >= 2:
                    self.none_kernel_help(input, torch.swapaxes, dtype, device, {"axis0": i, "axis1": i-2}, is_backward=True)

    ##### swapdims call the transpose op
    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_dtypes()))
    def test_swapdims(self, device, dtype):
        shape_list = [(32, 65), (32, 65, 128), (32, 16, 8, 16), (4, 32, 16, 8, 16)]
        for shape in shape_list:
            for i in range(1, len(shape)):
                input = make_tensor(shape, "cpu", dtype, requires_grad=True)
                self.none_kernel_help(input, torch.swapdims, dtype, device, {"dim0": i, "dim1": i-1}, is_backward=True)
                if i >= 2:
                    self.none_kernel_help(input, torch.swapdims, dtype, device, {"dim0": i, "dim1": i-1}, is_backward=True)

    ##### take 
    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_dtypes()))
    def test_take(self, device, dtype):
        import random
        shape_list = [(10,), (32, 65), (32, 65, 128)]
        for shape in shape_list:
            input = make_tensor(shape, "cpu", dtype, requires_grad=True)
            random_num_count = random.randint(1, input.numel())
            index_tensor = torch.randint(0, input.numel(), (random_num_count,))
            self.none_kernel_help(input, torch.take, dtype, device, {"index": index_tensor}, need_to_cuda=["index"], is_backward=True)

    ##### take_along_dim
    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_dtypes()))
    def test_take_along_dim(self, device, dtype):
        import random
        shape_list = [(10,), (65, 65), (33, 32, 32)]
        for shape in shape_list:
            input = make_tensor(shape, "cpu", dtype, requires_grad=True)
            num = shape[0]
            max_idx = torch.tensor(num//2, dtype=torch.int64)
            self.none_kernel_help(input, torch.take_along_dim, dtype, device, {"indices": max_idx}, need_to_cuda=["indices"], is_backward=True)

    ##### tensor_split
    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_dtypes()))
    def test_tensor_split(self, device, dtype):
        import random
        shape_list = [(10,), (32, 65), (32, 65, 128)]
        for shape in shape_list:
            for dim in range(0, len(shape)):
                input = make_tensor(shape, "cpu", dtype, requires_grad=True)

                # sections
                self.none_kernel_help(input, torch.tensor_split, dtype, device, {"sections":1, "dim":dim})

                # tensor_indices_or_sections
                tensor_indices_or_sections = torch.tensor(1, dtype=torch.int64)
                self.none_kernel_help(input, torch.tensor_split, dtype, device, {"tensor_indices_or_sections":tensor_indices_or_sections, "dim":dim})

                # sizes
                self.none_kernel_help(input, torch.tensor_split, dtype, device, {"indices":[1], "dim":dim})

    ##### tile
    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_dtypes()))
    def test_tile(self, device, dtype):
        shape_list = [(10,), (32, 65), (32, 65, 128)]
        for shape in shape_list:
            input = make_tensor(shape, "cpu", dtype, requires_grad=True)
            self.none_kernel_help(input, torch.tile, dtype, device, {"dims":(3, )})

    ##### transpose
    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_dtypes()))
    def test_transpose(self, device, dtype):
        shape_list = [(16, 65), (4, 65, 63), (4, 65, 13, 32)]
        for shape in shape_list:
            input = make_tensor(shape, "cpu", dtype, requires_grad=True)
            for i in range(1, len(shape)):
                self.none_kernel_help(input, torch.transpose, dtype, device, {"dim0":i, "dim1":i-1})

    ##### unbind
    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_dtypes()))
    def test_unbind(self, device, dtype):
        shape_list = [(16, 65), (4, 65, 63), (4, 65, 13, 32)]
        for shape in shape_list:
            for i in range(len(shape)):
                input = make_tensor(shape, "cpu", dtype, requires_grad=True)
                self.none_kernel_help(input, torch.unbind, dtype, device, {"dim":i}, is_backward=False)

    ##### unsqueeze
    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_dtypes()))
    def test_unsqueeze(self, device, dtype):
        shape_list = [(16, 65), (4, 65, 63), (4, 65, 13, 32)]
        for shape in shape_list:
            for i in range(len(shape)):
                input = make_tensor(shape, "cpu", dtype, requires_grad=True)
                self.none_kernel_help(input, torch.unsqueeze, dtype, device, {"dim":i}, is_backward=True)

    ##### vsplit
    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_dtypes()))
    def test_vsplit(self, device, dtype):
        shape_list = [(16, 65), (4, 65, 63), (4, 65, 13, 32)]
        for shape in shape_list:
            input = make_tensor(shape, "cpu", dtype, requires_grad=True)
            self.none_kernel_help(input, torch.vsplit, dtype, device, {"sections":2}, is_backward=True)
            self.none_kernel_help(input, torch.vsplit, dtype, device, {"indices":[2, 3]}, is_backward=True)

    ##### vstack
    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_dtypes()))
    def test_vstack(self, device, dtype):
        shape_list = [(16, 65), (4, 65, 63), (4, 65, 13, 32)]
        for shape in shape_list:
            a = make_tensor(shape, "cpu", dtype, requires_grad=True)
            b = make_tensor(shape, "cpu", dtype, requires_grad=True)
            cpu_forward = torch.vstack((a, b))
            a_g = copy.deepcopy(a).to(device)
            b_g = copy.deepcopy(b).to(device)
            cuda_forward = torch.vstack((a_g, b_g))
            if dtype in ALL_FLOATING_TYPES or dtype in ALL_COMPLEX_TYPES:
                cpu_forward.retain_grad()
                grad = make_tensor(cpu_forward.shape, "cpu", dtype)
                cpu_forward.backward(grad)

                grad_g = grad.to(device)
                cuda_forward.retain_grad()
                cuda_forward.backward(grad_g)

                assert torch.equal(cpu_forward.grad, cuda_forward.grad.cpu())
            assert torch.equal(cpu_forward, cuda_forward.cpu())

    ##### where
    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_dtypes(include_complex=False)))
    def test_where(self, device, dtype):
        shape_list = [(16, 65), (4, 65, 63), (4, 65, 13, 32)]
        for shape in shape_list:
            a = make_tensor(shape, "cpu", dtype, requires_grad=True)
            b = make_tensor(shape, "cpu", dtype, requires_grad=True)
            cpu_forward = torch.where(a > 0, a, b)
            a_g = copy.deepcopy(a).to(device)
            b_g = copy.deepcopy(b).to(device)
            cuda_forward = torch.where(a_g > 0, a_g, b_g)
            if dtype in ALL_FLOATING_TYPES or dtype in ALL_COMPLEX_TYPES:
                cpu_forward.retain_grad()
                grad = make_tensor(cpu_forward.shape, "cpu", dtype)
                cpu_forward.backward(grad)

                grad_g = grad.to(device)
                cuda_forward.retain_grad()
                cuda_forward.backward(grad_g)

                assert torch.equal(cpu_forward.grad, cuda_forward.grad.cpu())
            assert torch.equal(cpu_forward, cuda_forward.cpu())

    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_fp_dtypes(include_half=False)))
    def test_statistics_op(self, device, dtype):
        def help(op, kwargs={}):
            torch.cuda.manual_seed(0)
            out1 = op(**kwargs)
            torch.cuda.manual_seed(0)
            out2 = op(**kwargs)
            assert torch.equal(out1, out2)

        # bernoulli
        shape = (32, 65)
        a = make_tensor(shape, device, dtype, low=0, high=1)
        help(torch.bernoulli, {"input": a})
        
        # multinomial
        help(torch.multinomial, {"input":a, "num_samples":10})

    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_dtypes()))
    def test_rand_op(self, device, dtype):
        def help(op, kwargs={}):
            torch.cuda.manual_seed(0)
            out1 = op(**kwargs)
            torch.cuda.manual_seed(0)
            out2 = op(**kwargs)
            assert torch.equal(out1, out2)
        
        # rand
        if dtype in ALL_FLOATING_TYPES or dtype in ALL_COMPLEX_TYPES:
            help(torch.rand, {"size":(16, 32), "dtype":dtype, "device":device})

        # rand_like
        if dtype in ALL_FLOATING_TYPES or dtype in ALL_COMPLEX_TYPES:
            a = make_tensor((32, 16), "cuda", dtype)
            help(torch.rand_like, {"input":a, "dtype":dtype, "device": device})

        # randint
        if dtype in ALL_INTEGER_TYPES:
            help(torch.randint, {"high":100, "size":(32, 16), "dtype":dtype, "device": device})

        # randint_like
        if dtype in ALL_INTEGER_TYPES:
            a = make_tensor((32, 16), "cuda", dtype)
            help(torch.randint_like, {"input":a, "low":0, "high":100, "dtype":dtype, "device": device})

        # randn
        if dtype in ALL_FLOATING_TYPES or dtype in ALL_COMPLEX_TYPES:
            help(torch.randn, {"size":(16, 32), "dtype":dtype, "device":device})

        # randn_like
        if dtype in ALL_FLOATING_TYPES or dtype in ALL_COMPLEX_TYPES:
            a = make_tensor((32, 16), "cuda", dtype)
            help(torch.randn_like, {"input":a, "dtype":dtype, "device":device})

        # randperm
        if dtype in ALL_INTEGER_TYPES or (dtype in ALL_FLOATING_TYPES and dtype != torch.bfloat16):
            help(torch.randperm, {"n":32, "dtype":dtype, "device":device})

instantiate_device_type_tests(TestTorchFunctionSegment2, globals())

if __name__ == "__main__":
    unittest.main()
