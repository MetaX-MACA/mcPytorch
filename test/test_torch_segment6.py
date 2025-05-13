import unittest
import random
from common_util import *
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import random

class TestTorchFunctionSegment6(unittest.TestCase):
    def assert_equal(self, a, b, rtol=1e-5, atol=1e-8):
        assert torch.allclose(a, b, rtol=rtol, atol=atol)

    def test_is_tensor(self):
        x = torch.tensor([1,2])
        assert torch.is_tensor(x) is True

        x = torch.tensor([1,2], device="cuda:0")
        assert torch.is_tensor(x) is True

        x = torch.tensor([1], device="cuda:0")
        assert torch.is_tensor(x) is True

        x = torch.tensor([], device="cuda:0")
        assert torch.is_tensor(x) is True

        assert torch.is_tensor(3) is False

    def test_is_storage(self):
        assert torch.is_storage(torch.DoubleStorage())
        assert torch.is_storage(torch.FloatStorage())
        assert torch.is_storage(torch.BFloat16Storage())
        assert torch.is_storage(torch.HalfStorage())
        assert not torch.is_storage(torch.tensor([2,3], device="cuda"))

    def test_is_complex(self):
        x = torch.randn(2, 3, dtype=torch.complex64, device="cuda:0")
        y = torch.randn(2, 3, dtype=torch.complex128, device="cuda:0")
        z = torch.randn(2, 3, dtype=torch.float, device="cuda:0")
        assert torch.is_complex(x) is True
        assert torch.is_complex(y) is True
        assert torch.is_complex(z) is False

    def test_is_conj(self):
        y = torch.randn(2, 3, dtype=torch.complex128, device="cuda:0")
        assert not y.is_conj()
        y_conj = y.conj()
        assert y_conj.is_conj()
        z = y_conj.resolve_conj()
        assert not z.is_conj()
        y = torch.randn(2, 3, dtype=torch.float, device="cuda:0")
        assert not y.is_conj()
        y_conj = y.conj()
        assert not y_conj.is_conj()
        z = y_conj.resolve_conj()
        assert not z.is_conj()

    def test_is_floating_point(self):
        assert torch.is_floating_point(torch.tensor([1], dtype=torch.float32, device="cuda:0"))
        assert torch.is_floating_point(torch.tensor([], dtype=torch.float64, device="cuda:0"))
        assert torch.is_floating_point(torch.tensor([1,2], dtype=torch.float16, device="cuda:0"))
        assert torch.is_floating_point(torch.tensor([[1,2],[3,4]], dtype=torch.bfloat16, device="cuda:0"))
        assert not torch.is_floating_point(torch.tensor([1], dtype=torch.complex128, device="cuda:0"))

    def test_is_nonzero(self):
        assert not torch.is_nonzero(torch.tensor([0.0], device="cuda:0"))
        assert torch.is_nonzero(torch.tensor([1.5], device="cuda:0"))
        assert not torch.is_nonzero(torch.tensor([False], device="cuda:0"))
        self.assertRaises(RuntimeError, lambda: torch.is_nonzero(torch.tensor([1,2,3], device="cuda")))
        self.assertRaises(RuntimeError, lambda: torch.is_nonzero(torch.tensor([], device="cuda")))

    def test_default_type(self):
        x = torch.tensor([2., 4], device="cuda")
        assert x.dtype == torch.float32
        x = torch.tensor([2., 4j], device="cuda")
        assert x.dtype == torch.complex64
        torch.set_default_dtype(torch.float64)
        assert torch.get_default_dtype() == torch.float64
        assert torch.tensor([2., 4], device="cuda").dtype == torch.float64
        assert torch.tensor([2.j, 4], device="cuda").dtype == torch.complex128
        # TODO(mingwei.zhang): check device
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
        assert torch.tensor([2., 4]).device == torch.device("cuda:0")
        torch.set_default_tensor_type(torch.FloatTensor)
        assert torch.get_default_dtype() == torch.float32
        assert torch.tensor([2., 4], device="cuda").dtype == torch.float32
        assert torch.tensor([2.j, 4], device="cuda").dtype == torch.complex64

    def test_numel(self):
        x = torch.tensor([], dtype=torch.int, device="cuda")
        assert x.numel() == 0
        y = torch.tensor([1, 2.], device="cuda")
        assert torch.numel(y) == 2
        z = y.expand(6, 2)
        assert torch.numel(z) == 12

    def test_set_printoptions(self):
        pass 

    def test_set_flush_denormal(self):
        # only affect x86 CPU supporting sse3
        pass

    def test_tensor(self):
        pass 

    def test_sparse_coo_tensor(self):
        pass

    def test_asarray(self):
        # check autograd history's behaviour 
        pass 

    def test_as_tensor(self):
        # check autograd history's behaviour 
        pass 
    
    def test_as_strided(self):
        x = torch.tensor([[1,2,3],[4,5,6]], device="cuda:0", dtype=torch.float32)
        y = torch.as_strided(x, (2,2), (1,2))
        assert torch.allclose(y, torch.tensor([[1,3],[2,4]], device="cuda:0", dtype=torch.float32))
        y = torch.as_strided(x, (2,2), (1,2), 2)
        assert torch.allclose(y, torch.tensor([[3,5],[4,6]], device="cuda:0", dtype=torch.float32))
        self.assertRaises(RuntimeError, lambda: torch.as_strided(x, (3,3), (3,1)))

    def test_from_numpy(self):
        # no cuda related option
        pass

    def test_frombuffer(self):
        # no cuda related option
        pass

    def test_zeros(self):
        # see test_tensor_creation_ops.py:test_zeros(self, device)
        pass

    def test_zeros_like(self):
        # see test_tensor_creation_ops.py:test_zeros_like(self, device)
        pass

    def test_ones(self):
        # see also test_tensor_creation_ops.py:test_zeros_like(self, device)
        device = "cuda:0"
        res1 = torch.ones(100, 100, device=device)
        res2 = torch.tensor((), device=device)
        torch.ones(100, 100, device=device, out=res2)
        self.assert_equal(res1, res2)

        boolTensor = torch.ones(2, 2, device=device, dtype=torch.bool)
        expected = torch.tensor([[True, True], [True, True]],
                                device=device, dtype=torch.bool)
        self.assert_equal(boolTensor, expected)

        halfTensor = torch.ones(1, 1, device=device, dtype=torch.half)
        expected = torch.tensor([[1.]], device=device, dtype=torch.float16)
        self.assert_equal(halfTensor, expected)

        bfloat16Tensor = torch.ones(1, 1, device=device, dtype=torch.bfloat16)
        expected = torch.tensor([[1.]], device=device, dtype=torch.bfloat16)
        self.assert_equal(bfloat16Tensor, expected)

        complexTensor = torch.ones(2, 2, device=device, dtype=torch.complex64)
        expected = torch.tensor([[1., 1.], [1., 1.]], device=device, dtype=torch.complex64)
        self.assert_equal(complexTensor, expected)

    def test_ones_like(self):
        # see test_tensor_creation_ops.py:test_zeros_like(self, device)
        pass

    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_dtypes()))
    def test_zeros_ones_with_shape(self, device, dtype):
        seed = random.randint(0,1000)
        print("seed: {}".format(seed))
        random.seed(seed)
        iter = 2
        for i in range(0, iter):
            shape = (random.randint(0,1024), random.randint(0,128))
            ones_y = torch.ones(shape, device=device, dtype=dtype)
            ones_golden = torch.ones(shape, dtype=dtype)
            self.assert_equal(ones_golden, ones_y.to("cpu"))
            zeros_y = torch.zeros(shape, device=device, dtype=dtype)
            zeros_golden = torch.zeros(shape, dtype=dtype)
            self.assert_equal(zeros_golden, zeros_y.to("cpu"))

    @onlyCUDA
    @dtypesIfCUDA(torch.float32, torch.float64)
    def test_arange(self, device, dtype):
        seed = random.randint(0,1000)
        print("seed: {}".format(seed))
        start = random.randint(-100, 100)
        num = random.randint(0, 100)
        step = random.randint(1, 10)
        end = start + num * step + 1e-2
        y_cuda = torch.arange(start, end, step, dtype=dtype, device=device)
        y_cpu = torch.arange(start, end, step, dtype=torch.float64, device="cpu")
        self.assert_equal(y_cuda.cpu().to(torch.float64), y_cpu)

    @onlyCUDA
    @dtypesIfCUDA(torch.float16, torch.bfloat16)
    def test_arange_bit16(self, device, dtype):
        start = -100
        num = 100
        step = 3
        end = start + num * step + 1e-2
        y_cuda = torch.arange(start, end, step, dtype=dtype, device=device)
        y_cpu = torch.arange(start, end, step, dtype=torch.float64, device="cpu")
        self.assert_equal(y_cuda.cpu().to(torch.float64), y_cpu)

    @onlyCUDA
    @dtypesIfCUDA(torch.float32, torch.float64)
    def test_range(self, device, dtype):
        seed = random.randint(0,1000)
        print("seed: {}".format(seed))
        start = random.randint(-100, 100)
        num = random.randint(0, 100)
        step = random.randint(1, 10)
        end = start + num * step + 1e-2
        y_cuda = torch.range(start, end, step, dtype=dtype, device=device)
        y_cpu = torch.range(start, end, step, dtype=torch.float64, device="cpu")
        self.assert_equal(y_cuda.cpu().to(torch.float64), y_cpu)

    @onlyCUDA
    @dtypesIfCUDA(torch.float16)
    def test_range_bit16(self, device, dtype):
        start = -100
        num = 100
        step = 3
        end = start + num * step + 1e-2
        y_cuda = torch.range(start, end, step, dtype=dtype, device=device)
        y_cpu = torch.range(start, end, step, dtype=torch.float64, device="cpu")
        self.assert_equal(y_cuda.cpu().to(torch.float64), y_cpu)

    @onlyCUDA
    @dtypesIfCUDA(torch.float32, torch.float64)
    def test_linspace(self, device, dtype):
        seed = random.randint(0,1000)
        print("seed: {}".format(seed))
        start = random.randint(-100, 100)
        end = random.randint(200, 300)
        step = random.randint(1, 100)
        y_cuda = torch.linspace(start, end, step, dtype=dtype, device=device)
        y_cpu = torch.linspace(start, end, step, dtype=torch.float64, device="cpu")
        self.assert_equal(y_cuda.cpu().to(torch.float64), y_cpu)

    @onlyCUDA
    @dtypesIfCUDA(torch.float16, torch.bfloat16)
    def test_linspace(self, device, dtype):
        start = -100
        end = 100
        step = 51
        y_cuda = torch.linspace(start, end, step, dtype=dtype, device=device)
        y_cpu = torch.linspace(start, end, step, dtype=torch.float64, device="cpu")
        self.assert_equal(y_cuda.cpu().to(torch.float64), y_cpu)

    @onlyCUDA
    @dtypesIfCUDA(torch.bfloat16, torch.float16, torch.float32, torch.float64)
    def test_logspace(self, device, dtype):
        start = 0
        end = 3
        step = 4
        y_cuda = torch.logspace(start, end, step, dtype=dtype, device=device)
        y_cpu = torch.logspace(start, end, step, dtype=torch.float64, device="cpu")
        self.assert_equal(y_cuda.cpu().to(torch.float64), y_cpu)

    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_dtypes()))
    def test_eye(self, dtype, device):
        seed = random.randint(0,1000)
        print("seed: {}".format(seed))
        m = random.randint(1, 100)
        n = random.randint(1, 50)
        y_cuda = torch.eye(m, n, dtype=dtype, device=device)
        if dtype in [torch.complex64, torch.complex128]:
            y_cpu = torch.eye(m, n, dtype=torch.complex128, device="cpu")
            self.assert_equal(y_cuda.cpu().to(torch.complex128), y_cpu)
        else:
            y_cpu = torch.eye(m, n, dtype=torch.float64, device="cpu")
            self.assert_equal(y_cuda.cpu().to(torch.float64), y_cpu)

    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_dtypes()))
    def test_empty_and_empty_like(self, device, dtype):
        seed = random.randint(0,1000)
        print("seed: {}".format(seed))
        m = random.randint(1, 100)
        n = random.randint(1, 50)
        input = torch.empty(m, n, dtype=dtype, device=device)
        assert input.size() == torch.Size((m, n))
        input2 = torch.empty_like(input)
        assert input.size() == input2.size()

    def test_empty_strided(self):
        # test_tensor_creation_ops.py::test_torch_complex
        pass

    def test_full(self):
        # test_tensor_creation_ops.py
        pass

    def test_full_like(self):
        # test_tensor_creation_ops.py
        pass

    def test_quantize_per_tensor(self):
        pass

    def test_quantize_per_channel(self):
        pass

    def test_dequantize(self):
        pass

    def test_complex(self):
        # test_tensor_creation_ops.py::test_torch_complex
        pass

    def test_polar(self):
        # test_tensor_creation_ops.py::test_torch_polar
        pass

    def test_heaviside(self):
        # test_binary_ufuncs.py::test_heaviside_xx
        pass

    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_dtypes()))
    def test_adjoint(self, dtype, device):
        # supported start from v1.11
        pass
#        seed = random.randint(0,1000)
#        print("seed: {}".format(seed))
#        m = random.randint(1, 100)
#        n = random.randint(1, 50)
#        input = torch.randn(m, n).to(dtype)
#        y_cuda = torch.adjoint(input.cuda())
#        y_cpu = torch.adjoint(input)
#        self.assert_equal(y_cuda.cpu(), y_cpu)

    @onlyCUDA
    @dtypesIfCUDA(*set(get_all_dtypes()))
    def test_argwhere(self, dtype, device):
        # supported started from torch1.11
        pass
    #    seed = random.randint(0,1000)
    #    print("seed: {}".format(seed))
    #    m = random.randint(1, 30)
    #    n = random.randint(1, 20)
    #    k = random.randint(1, 3)
    #    input = torch.randint(0, 3, (m, n, k))
    #    input_dtype = input.to(dtype)
    #    y_cuda = torch.argwhere(input_dtype.cuda())
    #    y_cpu = torch.argwhere(input_dtype)
    #    self.assert_equal(y_cuda.cpu(), y_cpu)

    def test_cat(self):
        # test_tensor_creation_ops.py::test_cat_xxx
        pass

    def test_concat(self):
        # alias of cat
        pass

    def test_conj(self):
        # test_view_ops.py::test_conj_xxx
        pass

    def test_chunk(self):
        # test_view_ops.py+test_vmap.py::test_chunk
        pass

    def test_dsplit(self):
        # test_tensor_creation_ops.py::test_dsplit
        pass

    def test_column_stack(self):
        # test_tensor_creation_ops.py::test_hstack_column_stack
        pass

    def test_dstack(self):
        # test_tensor_creation_ops.py::test_dstack
        pass

    def test_gather(self):
        # test_torch,test_gather_backward_deterministic_path_cuda
        # test_torch,test_gather_mem_overlap_cuda
        pass

    def test_hsplit(self):
        # test_tensor_creation_ops.py::test_hsplit
        pass

    def test_torch_device(self):
        a = torch.device('cuda:0')
        assert a.type == "cuda" and a.index == 0

        b = torch.device('cuda')
        assert b.type == "cuda" and b.index is None

        c = torch.device('cpu')
        assert c.type == "cpu" and c.index is None

        c = torch.device('cpu:0')
        assert c.type == "cpu" and c.index == 0

        c = torch.device('cpu:1')
        assert c.type == "cpu" and c.index == 1

        d = torch.device('cuda', 0)
        assert d.type == "cuda" and d.index == 0

        e = torch.device('cpu', 0)
        assert e.type == "cpu" and e.index == 0

        f = torch.device('cpu', 1)
        assert f.type == "cpu" and f.index == 1

        assert torch.device("cuda:0") != torch.device("cuda")

        x = torch.tensor([1,2,3], device="cuda:0")
        assert x.device == torch.device("cuda:0")

        torch.cuda.set_device(0)
        x = torch.tensor([1,2,3], device="cuda")
        assert x.device == torch.device("cuda:0")


    @unittest.skipIf(not TEST_MULTIGPU, TEST_MULTIGPU_HINT)
    def test_torch_multi_device(self):
        a = torch.device('cuda:1')
        assert a.type == "cuda" and a.index == 1

        b = torch.device('cuda')
        assert b.type == "cuda" and b.index is None

        d = torch.device('cuda', 1)
        assert d.type == "cuda" and d.index == 1

        assert torch.device("cuda:1") != torch.device("cuda")

        x = torch.tensor([1,2,3], device="cuda:1")
        assert x.device == torch.device("cuda:1")

        old_device = torch.cuda.current_device()
        torch.cuda.set_device(1)
        x = torch.tensor([1,2,3], device="cuda")
        assert x.device == torch.device("cuda:1")
        torch.cuda.set_device(old_device)

        a_scalar = torch.tensor(1, device="cpu")
        a = torch.tensor([0.,1], device="cpu")
        a_scalar_d0 = a_scalar.to("cuda:0")
        a_scalar_d1 = a_scalar.to("cuda:1")
        a_d0 = a.to("cuda:0")
        a_d1 = a.to("cuda:1")
        b_scalar = a_scalar + a_scalar_d0
        assert b_scalar.device == a_scalar_d0.device
        self.assertRaises(RuntimeError, lambda: a_scalar_d0 + a_scalar_d1)
        self.assertRaises(RuntimeError, lambda: a + a_d0)
        self.assertRaises(RuntimeError, lambda: a_d0 + a_d1)


    def test_basic(self):
        x = torch.tensor([[1],[2]], device="cuda:0")
        assert x.dtype == torch.int64
        y = torch.tensor([[1.0],[2.0]], device="cuda:0")
        assert y.dtype == torch.float

        a = torch.ones((3,4,5), device="cuda:0", dtype=torch.float32)
        b = a
        assert b[0,0,0] == 1
        a[0,0,0] = 10.0
        assert b[0,0,0] == 10


    def test_casting_rules(self):
        float_tensor = torch.ones(1, dtype=torch.float, device="cuda:0")
        double_tensor = torch.ones(1, dtype=torch.double, device="cuda:0")
        complex_float_tensor = torch.ones(1, dtype=torch.complex64, device="cuda:0")
        complex_double_tensor = torch.ones(1, dtype=torch.complex128, device="cuda:0")
        int_tensor = torch.ones(1, dtype=torch.int, device="cuda:0")
        long_tensor = torch.ones(1, dtype=torch.long, device="cuda:0")
        uint8_tensor = torch.ones(1, dtype=torch.uint8, device="cuda:0")
        int8_tensor = torch.ones(1, dtype=torch.int8, device="cuda:0")
        bool_tensor = torch.ones(1, dtype=torch.bool, device="cuda:0")
        long_zerodim = torch.tensor(1, dtype=torch.long, device="cuda:0")
        int_zerodim = torch.tensor(1, dtype=torch.int, device="cuda:0")

        assert torch.add(5, 5).dtype == torch.int64
        assert torch.add(5.0, 5.0).dtype == torch.float64  # interesting
        assert torch.add(5.0, 5).dtype == torch.float32    # interesting
        assert (int_tensor + 5).dtype == torch.int32
        assert (int_tensor + long_zerodim).dtype == torch.int32
        assert (int_tensor + long_tensor).dtype == torch.int64
        assert (bool_tensor + long_tensor).dtype == torch.int64
        assert (uint8_tensor + bool_tensor).dtype == torch.uint8
        assert (float_tensor + double_tensor).dtype == torch.float64
        assert (complex_float_tensor + complex_double_tensor).dtype == torch.complex128
        assert (long_tensor + float_tensor).dtype == torch.float32

        float_tensor *= float_tensor
        float_tensor *= int_tensor
        float_tensor *= uint8_tensor
        float_tensor *= bool_tensor
        float_tensor *= double_tensor
        int_tensor *= long_tensor
        int_tensor *= uint8_tensor
        uint8_tensor *= int_tensor

        def test_internal(dst, src):
            dst *= src
        self.assertRaises(RuntimeError, test_internal, int_tensor, float_tensor)
        self.assertRaises(RuntimeError, test_internal, bool_tensor, int_tensor)
        self.assertRaises(RuntimeError, test_internal, bool_tensor, uint8_tensor)
        self.assertRaises(RuntimeError, test_internal, bool_tensor, int8_tensor)
        self.assertRaises(RuntimeError, test_internal, bool_tensor, float_tensor)
        self.assertRaises(RuntimeError, test_internal, float_tensor, complex_float_tensor)

        
    def test_is_same_tensor(self):
        # expand
        x = torch.tensor([[1],[2],[3]], device="cuda:0", dtype=torch.float32)
        y = x.expand(3,2)
        assert id(y) != id(x)

        # add
        old_id = id(x)
        x += 1
        assert old_id == id(x)
        x = x + 1
        assert old_id != id(x)


    def test_expand(self):
        x = torch.tensor([[1],[2],[3]], device="cuda:0", dtype=torch.float32)
        assert x.size() == torch.Size([3,1])
        y = x.expand(3,2)
        assert torch.allclose(y, torch.tensor([[1,1],[2,2],[3,3]], device="cuda:0", dtype=torch.float32))
        assert y.shape == y.size() == torch.Size([3,2])
        assert y.stride() == (1, 0)
        z = x.expand(-1,2)
        assert torch.allclose(y, torch.tensor([[1,1],[2,2],[3,3]], device="cuda:0", dtype=torch.float32))
        assert y.shape == y.size() == torch.Size([3,2])
        assert y.stride() == (1, 0)

        x = torch.tensor([[1,2],[3,4],[5,6]], device="cuda:0", dtype=torch.float32)
        assert x.size() == torch.Size([3,2])
        self.assertRaises(RuntimeError, lambda: x.expand(3, 4))
        self.assertRaises(RuntimeError, lambda: x.expand(3, 2, 4))

        r1 = torch.rand(2,1,3, device="cuda:0", dtype=torch.float32)
        r2 = torch.rand((2,1,3), device="cuda:0", dtype=torch.float32)
        r3 = torch.rand((((2,1,3))), device="cuda:0", dtype=torch.float32)
        r4 = torch.rand([2,1,3], device="cuda:0", dtype=torch.float32)
        self.assertRaises(TypeError, lambda: torch.rand([[[2,1,3]]], device="cuda:0", dtype=torch.float32))
        assert r1.size() == r2.size() == r3.size() == r4.size() == torch.Size([2,1,3])
        r5 = r1.expand(5,2,4,3)
        assert r5.size() == torch.Size([5,2,4,3])

    
    def test_view(self):
        x = torch.randn(4, 4, device="cuda:0", dtype=torch.float32)
        y = x.view(16)
        assert y.size() == torch.Size([16])
        z = x.view(-1, 8)
        assert z.size() == torch.Size([2,8])

        a = torch.rand(1,2,3,4)
        b = a.transpose(1,2)
        c = a.view(1,3,2,4)
        assert c.size() == torch.Size([1,3,2,4])
        assert torch.equal(b, c) is False
        self.assertRaises(RuntimeError, lambda: a.view(6,5))


    def test_reshape(self):
        a = torch.arange(4,  device="cuda:0")
        b = torch.reshape(a, (2,2))
        assert b.size() == torch.Size([2,2])


    @unittest.skipIf(not TEST_EMPIRICAL, TEST_EMPIRICAL_HINT)
    def test_reshape_empirical(self):
        # view
        a = torch.ones((3,4,5), device="cuda:0", dtype=torch.float32)
        b = torch.reshape(a, (3,10,2))
        assert b.size() == torch.Size([3,10,2])
        assert b[0,0,0] == 1.0
        a[0,0,0] = 10.0
        assert b[0,0,0] == 10.0

        a = torch.ones((3,4,5), device="cuda:0", dtype=torch.float32)
        a = a.transpose(1,2)
        b = torch.reshape(a, (3,5,4))
        assert b.stride() == (20, 1, 5)
        
        # copy
        a = torch.ones((2,5), device="cuda:0", dtype=torch.float32)
        b = a.expand(3,2,5)
        c = b.reshape((6,5))
        assert b[0,0,0] == 1.0
        assert c[0,0] == 1.0
        a[0,0] = 10.0
        assert b[0,0,0] == 10.0
        assert c[0,0] == 1.0


    def test_transpose(self):
        a = torch.ones((3,4,5), device="cuda:0", dtype=torch.float32)
        b = a.transpose(1,2)
        assert b[0,0,0] == 1
        a[0,0,0] = 10.0
        assert b[0,0,0] == 10




    def test_resize(self):
        x = torch.tensor([[1,2,3],[4,5,6]], device="cuda:0", dtype=torch.float32)
        x.resize_(2,2)
        assert torch.allclose(x, torch.tensor([[1,2],[3,4]], device="cuda:0", dtype=torch.float32))
        x = torch.tensor([[1],[2],[3]], device="cuda:0", dtype=torch.float32)
        y = x.expand(3,2)
        assert torch.allclose(x, torch.tensor([[1,1],[2,2],[3,3]], device="cuda:0", dtype=torch.float32))
        y.resize_(1,3)
        assert torch.allclose(y, torch.tensor([[1,2,3]], device="cuda:0", dtype=torch.float32))

instantiate_device_type_tests(TestTorchFunctionSegment6, globals())

if __name__ == "__main__":
    unittest.main()