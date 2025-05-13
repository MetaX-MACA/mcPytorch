import torch
import numpy
import tempfile
import unittest
import itertools
import torch.nn as nn
from itertools import product
from common_util import *
import torch.nn.utils.rnn as rnn_utils
from torch.testing._internal.common_utils import run_tests, TestCase

class TestTorchFunctionSegment4(TestCase):

    @with_tf32_off_helper
    def test_quasirandom_SobolEngine(self):
        # no cuda version
        pass


    @onlyCUDA
    @dtypesIfCUDA(*set([torch.float, torch.double, torch.half, torch.bfloat16,
            torch.complex64, torch.complex128, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]))
    @with_tf32_off_helper
    def test_save_load(self, device, dtype):
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
                with tempfile.NamedTemporaryFile() as f1:
                    torch.save(inp_cuda, f1)
                    f1.seek(0)
                    tgt_cuda = torch.load(f1)
                    self.assertEqual(inp_cuda, tgt_cuda)
                    self.assertEqual(inp_cpu, tgt_cuda)


    @with_tf32_off_helper
    def test_get_num_threads(self):
        # no cuda version
        pass


    @with_tf32_off_helper
    def test_set_num_threads(self):
        # no cuda version
        pass


    @with_tf32_off_helper
    def test_get_num_interop_threads(self):
        # no cuda version
        pass


    @with_tf32_off_helper
    def test_set_num_interop_threads(self):
        # no cuda version
        pass


    @onlyCUDA
    @dtypesIfCUDA(*set([torch.float, torch.double, torch.half, torch.bfloat16, torch.complex64, torch.complex128]))
    @with_tf32_off_helper
    def test_no_grad(self, device, dtype):
        torch.manual_seed(0)
        list_size = [
            [5, 7], [512, 768], [2048, 2048],
            [64, 5, 7], [4, 512, 768], [4, 1024, 1024]
        ]
        for sz in list_size:
            for contig in [True, False]:
                inp_cpu = torch.randn(sz, requires_grad=True).to(dtype)
                inp_cuda = inp_cpu.clone().detach().requires_grad_(True).cuda()
                if contig == False:
                    inp_cpu = make_noncontig(inp_cpu)
                    inp_cuda = make_noncontig(inp_cuda)
                assert inp_cpu.requires_grad == True
                assert inp_cuda.requires_grad == True
                with torch.no_grad():
                    out_cpu = inp_cpu * 2
                    out_cuda = inp_cuda * 2
                    assert out_cpu.requires_grad == False
                    assert out_cuda.requires_grad == False


    @onlyCUDA
    @dtypesIfCUDA(*set([torch.float, torch.double, torch.half, torch.bfloat16, torch.complex64, torch.complex128,]))
    @with_tf32_off_helper
    def test_enable_grad(self, device, dtype):
        torch.manual_seed(0)
        list_size = [
            [5, 7], [512, 768], [2048, 2048],
            [64, 5, 7], [4, 512, 768], [4, 1024, 1024]
        ]
        for sz in list_size:
            for contig in [True, False]:
                inp_cpu = torch.randn(sz, requires_grad=True).to(dtype)
                inp_cuda = inp_cpu.clone().detach().requires_grad_(True).cuda()
                if contig == False:
                    inp_cpu = make_noncontig(inp_cpu)
                    inp_cuda = make_noncontig(inp_cuda)
                assert inp_cpu.requires_grad == True
                assert inp_cuda.requires_grad == True
                with torch.no_grad():
                    with torch.enable_grad():
                        out_cpu = inp_cpu * 2
                        out_cuda = inp_cuda * 2
                        assert out_cpu.requires_grad == True
                        assert out_cuda.requires_grad == True


    @onlyCUDA
    @dtypesIfCUDA(*set([torch.float, torch.double, torch.half, torch.bfloat16, torch.complex64, torch.complex128,]))
    @with_tf32_off_helper
    def test_set_grad_enabled(self, device, dtype):
        torch.manual_seed(0)
        list_size = [
            [5, 7], [512, 768], [2048, 2048],
            [64, 5, 7], [4, 512, 768], [4, 1024, 1024]
        ]
        for sz in list_size:
            for contig in [True, False]:
                inp_cpu = torch.randn(sz, requires_grad=True).to(dtype)
                inp_cuda = inp_cpu.clone().detach().requires_grad_(True).cuda()
                if contig == False:
                    inp_cpu = make_noncontig(inp_cpu)
                    inp_cuda = make_noncontig(inp_cuda)
                assert inp_cpu.requires_grad == True
                assert inp_cuda.requires_grad == True
                for flag in [False, True]:
                    with torch.set_grad_enabled(flag):
                        out_cpu = inp_cpu * 2
                        out_cuda = inp_cuda * 2
                        assert out_cpu.requires_grad == flag
                        assert out_cuda.requires_grad == flag


    @with_tf32_off_helper
    def test_is_grad_enabled(self):
        # no cuda version
        pass


    @onlyCUDA
    @dtypesIfCUDA(*set([torch.float, torch.double, torch.half, torch.bfloat16, torch.complex64, torch.complex128]))
    @with_tf32_off_helper
    def test_inference_mode(self, device, dtype):
        torch.manual_seed(0)
        list_size = [
            [5, 7], [512, 768], [2048, 2048],
            [64, 5, 7], [4, 512, 768], [4, 1024, 1024]
        ]
        for sz in list_size:
            for contig in [True, False]:
                inp_cpu = torch.randn(sz, requires_grad=True).to(dtype)
                inp_cuda = inp_cpu.clone().detach().requires_grad_(True).cuda()
                if contig == False:
                    inp_cpu = make_noncontig(inp_cpu)
                    inp_cuda = make_noncontig(inp_cuda)
                assert inp_cpu.requires_grad == True
                assert inp_cuda.requires_grad == True
                with torch.inference_mode():
                    out_cpu = inp_cpu * 2
                    out_cuda = inp_cuda * 2
                    assert out_cpu.requires_grad == False
                    assert out_cuda.requires_grad == False


    @with_tf32_off_helper
    def test_is_inference_mode_enabled(self):
        # no cuda version
        pass


    @onlyCUDA
    @dtypesIfCUDA(*set([torch.float, torch.double, torch.half, torch.bfloat16,
            torch.complex64, torch.complex128, torch.int8, torch.int16, torch.int32, torch.int64]))
    @with_tf32_off_helper
    def test_abs(self, device, dtype):
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
                    inp_cuda = inp_cpu.clone().detach().cuda()
                else:
                    inp = torch.randn(sz, requires_grad=True).to(dtype)
                    inp_cuda = inp.clone().detach().requires_grad_(True).cuda()
                    inp_cpu = inp_cuda.clone().detach().requires_grad_(True).cpu()
                if contig == False:
                    inp_cpu = make_noncontig(inp_cpu)
                    inp_cuda = make_noncontig(inp_cuda)
                out_cpu = torch.abs(inp_cpu)
                out_cuda = torch.abs(inp_cuda)
                self.assertEqual(out_cpu, out_cuda)
                if dtype not in ALL_INTEGER_TYPES:
                    out_cpu.sum().backward()
                    out_cuda.sum().backward()
                    self.assertEqual(inp_cpu.retain_grad(), inp_cuda.retain_grad())


    @onlyCUDA
    @dtypesIfCUDA(*set([torch.float, torch.double, torch.half, torch.bfloat16,
            torch.complex64, torch.complex128, torch.int8, torch.int16, torch.int32, torch.int64]))
    @with_tf32_off_helper
    def test_absolute(self, device, dtype):
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
                    inp_cuda = inp_cpu.clone().detach().cuda()
                else:
                    inp = torch.randn(sz, requires_grad=True).to(dtype)
                    inp_cuda = inp.clone().detach().requires_grad_(True).cuda()
                    inp_cpu = inp_cuda.clone().detach().requires_grad_(True).cpu()
                if contig == False:
                    inp_cpu = make_noncontig(inp_cpu)
                    inp_cuda = make_noncontig(inp_cuda)
                out_cpu = torch.absolute(inp_cpu)
                out_cuda = torch.absolute(inp_cuda)
                self.assertEqual(out_cpu, out_cuda)
                if dtype not in ALL_INTEGER_TYPES:
                    out_cpu.sum().backward()
                    out_cuda.sum().backward()
                    self.assertEqual(inp_cpu.retain_grad(), inp_cuda.retain_grad())


    @onlyCUDA
    @dtypesIfCUDA(*set([torch.float, torch.double, torch.half, torch.bfloat16,
            torch.complex64, torch.complex128, torch.int8, torch.int16, torch.int32, torch.int64]))
    @with_tf32_off_helper
    def test_acos(self, device, dtype):
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
                    inp_cuda = inp_cpu.clone().detach().cuda()
                else:
                    inp = torch.randn(sz, requires_grad=True).to(dtype)
                    inp_cuda = inp.clone().detach().requires_grad_(True).cuda()
                    inp_cpu = inp_cuda.clone().detach().requires_grad_(True).cpu()
                if contig == False:
                    inp_cpu = make_noncontig(inp_cpu)
                    inp_cuda = make_noncontig(inp_cuda)
                out_cuda = torch.acos(inp_cuda)
                if dtype != torch.half:
                    out_cpu = torch.acos(inp_cpu)
                    self.assertEqual(out_cpu, out_cuda)
                if dtype not in ALL_INTEGER_TYPES:
                    out_cuda.sum().backward()
                    if dtype != torch.half:
                        out_cpu.sum().backward()
                        self.assertEqual(inp_cpu.retain_grad(), inp_cuda.retain_grad())


    @onlyCUDA
    @dtypesIfCUDA(*set([torch.float, torch.double, torch.half, torch.bfloat16,
            torch.complex64, torch.complex128, torch.int8, torch.int16, torch.int32, torch.int64]))
    @with_tf32_off_helper
    def test_arccos(self, device, dtype):
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
                    inp_cuda = inp_cpu.clone().detach().cuda()
                else:
                    inp = torch.randn(sz, requires_grad=True).to(dtype)
                    inp_cuda = inp.clone().detach().requires_grad_(True).cuda()
                    inp_cpu = inp_cuda.clone().detach().requires_grad_(True).cpu()
                if contig == False:
                    inp_cpu = make_noncontig(inp_cpu)
                    inp_cuda = make_noncontig(inp_cuda)
                out_cuda = torch.arccos(inp_cuda)
                if dtype != torch.half:
                    out_cpu = torch.arccos(inp_cpu)
                    self.assertEqual(out_cpu, out_cuda)
                if dtype not in ALL_INTEGER_TYPES:
                    out_cuda.sum().backward()
                    if dtype != torch.half:
                        out_cpu.sum().backward()
                        self.assertEqual(inp_cpu.retain_grad(), inp_cuda.retain_grad())


    @onlyCUDA
    @dtypesIfCUDA(*set([torch.float, torch.double, torch.half, torch.bfloat16,
            torch.complex64, torch.complex128, torch.int8, torch.int16, torch.int32, torch.int64]))
    @with_tf32_off_helper
    def test_acosh(self, device, dtype):
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
                    inp_cuda = inp_cpu.clone().detach().cuda()
                else:
                    inp = torch.randn(sz, requires_grad=True).to(dtype)
                    inp_cuda = inp.clone().detach().requires_grad_(True).cuda()
                    inp_cpu = inp_cuda.clone().detach().requires_grad_(True).cpu()
                if contig == False:
                    inp_cpu = make_noncontig(inp_cpu)
                    inp_cuda = make_noncontig(inp_cuda)
                out_cuda = torch.acosh(inp_cuda)
                if dtype != torch.half:
                    out_cpu = torch.acosh(inp_cpu)
                    self.assertEqual(out_cpu, out_cuda)
                if dtype not in ALL_INTEGER_TYPES:
                    out_cuda.sum().backward()
                    if dtype != torch.half:
                        out_cpu.sum().backward()
                        self.assertEqual(inp_cpu.retain_grad(), inp_cuda.retain_grad())


    @onlyCUDA
    @dtypesIfCUDA(*set([torch.float, torch.double, torch.half, torch.bfloat16,
            torch.complex64, torch.complex128, torch.int8, torch.int16, torch.int32, torch.int64]))
    @with_tf32_off_helper
    def test_arccosh(self, device, dtype):
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
                    inp_cuda = inp_cpu.clone().detach().cuda()
                else:
                    inp = torch.randn(sz, requires_grad=True).to(dtype)
                    inp_cuda = inp.clone().detach().requires_grad_(True).cuda()
                    inp_cpu = inp_cuda.clone().detach().requires_grad_(True).cpu()
                if contig == False:
                    inp_cpu = make_noncontig(inp_cpu)
                    inp_cuda = make_noncontig(inp_cuda)
                out_cuda = torch.arccosh(inp_cuda)
                if dtype != torch.half:
                    out_cpu = torch.arccosh(inp_cpu)
                    self.assertEqual(out_cpu, out_cuda)
                if dtype not in ALL_INTEGER_TYPES:
                    out_cuda.sum().backward()
                    if dtype != torch.half:
                        out_cpu.sum().backward()
                        self.assertEqual(inp_cpu.retain_grad(), inp_cuda.retain_grad())


    @onlyCUDA
    @dtypesIfCUDA(*set([torch.float, torch.double, torch.half, torch.bfloat16,
            torch.complex64, torch.complex128, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]))
    @with_tf32_off_helper
    def test_add(self, device, dtype):
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
                    inp1_cuda = inp1_cpu.clone().detach().cuda()
                    inp2_temp = numpy.random.randint(-2048, 2048, size=sz)
                    inp2_cpu = torch.tensor(inp2_temp, dtype=dtype)
                    inp2_cuda = inp2_cpu.clone().detach().cuda()
                else:
                    inp1 = torch.randn(sz, dtype=dtype, requires_grad=True)
                    inp1_cuda = inp1.clone().detach().requires_grad_(True).cuda()
                    inp1_cpu = inp1_cuda.clone().detach().requires_grad_(True).cpu()
                    inp2 = torch.randn(sz, dtype=dtype, requires_grad=True)
                    inp2_cuda = inp2.clone().detach().requires_grad_(True).cuda()
                    inp2_cpu = inp2_cuda.clone().detach().requires_grad_(True).cpu()
                if contig == False:
                    inp1_cpu = make_noncontig(inp1_cpu)
                    inp1_cuda = make_noncontig(inp1_cuda)
                    inp2_cpu = make_noncontig(inp2_cpu)
                    inp2_cuda = make_noncontig(inp2_cuda)
                out_cpu = torch.add(inp1_cpu, inp2_cpu)
                out_cuda = torch.add(inp1_cuda, inp2_cuda)
                self.assertEqual(out_cpu, out_cuda)
                if dtype not in ALL_INTEGER_TYPES:
                    out_cpu.sum().backward()
                    out_cuda.sum().backward()
                    self.assertEqual(inp1_cpu.retain_grad(), inp1_cuda.retain_grad())
                    self.assertEqual(inp2_cpu.retain_grad(), inp2_cuda.retain_grad())


    @onlyCUDA
    @dtypesIfCUDA(*set([torch.float, torch.double, torch.half, torch.bfloat16, torch.complex64, torch.complex128,]))
    @with_tf32_off_helper
    def test_addcdiv(self, device, dtype):
        torch.manual_seed(0)
        list_size = [
            [[5, 7], 0.1], [[512, 768], 0.2], [[2048, 2048], 0.4],
            [[64, 5, 7], 0.5], [[4, 512, 768], 0.7], [[4, 1024, 1024], 0.8]
        ]
        for sz, value in list_size:
            for contig in [True, False]:
                if dtype in ALL_INTEGER_TYPES:
                    inp1_temp = numpy.random.randint(-2048, 2048, size=sz)
                    inp1_cpu = torch.tensor(inp1_temp, dtype=dtype)
                    inp1_cuda = inp1_cpu.clone().detach().cuda()
                    inp2_temp = numpy.random.randint(-2048, 2048, size=sz)
                    inp2_cpu = torch.tensor(inp2_temp, dtype=dtype)
                    inp2_cuda = inp2_cpu.clone().detach().cuda()
                    inp3_temp = numpy.random.randint(-2048, 2048, size=sz)
                    inp3_cpu = torch.tensor(inp3_temp, dtype=dtype)
                    inp3_cuda = inp3_cpu.clone().detach().cuda()
                else:
                    inp1 = torch.randn(sz, dtype=dtype, requires_grad=True)
                    inp1_cuda = inp1.clone().detach().requires_grad_(True).cuda()
                    inp1_cpu = inp1_cuda.clone().detach().requires_grad_(True).cpu()
                    inp2 = torch.randn(sz, dtype=dtype, requires_grad=True)
                    inp2_cuda = inp2.clone().detach().requires_grad_(True).cuda()
                    inp2_cpu = inp2_cuda.clone().detach().requires_grad_(True).cpu()
                    inp3 = torch.randn(sz, dtype=dtype, requires_grad=True)
                    inp3_cuda = inp3.clone().detach().requires_grad_(True).cuda()
                    inp3_cpu = inp3_cuda.clone().detach().requires_grad_(True).cpu()
                if contig == False:
                    inp1_cpu = make_noncontig(inp1_cpu)
                    inp1_cuda = make_noncontig(inp1_cuda)
                    inp2_cpu = make_noncontig(inp2_cpu)
                    inp2_cuda = make_noncontig(inp2_cuda)
                    inp3_cpu = make_noncontig(inp3_cpu)
                    inp3_cuda = make_noncontig(inp3_cuda)
                out_cuda = torch.addcdiv(inp1_cuda, inp2_cuda, inp3_cuda, value=value)
                if dtype != torch.half:
                    out_cpu = torch.addcdiv(inp1_cpu, inp2_cpu, inp3_cpu, value=value)
                    self.assertEqual(out_cpu, out_cuda)
                if dtype not in ALL_INTEGER_TYPES:
                    out_cuda.sum().backward()
                    if dtype != torch.half:
                        out_cpu.sum().backward()
                        self.assertEqual(inp1_cpu.retain_grad(), inp1_cuda.retain_grad())
                        self.assertEqual(inp2_cpu.retain_grad(), inp2_cuda.retain_grad())
                        self.assertEqual(inp3_cpu.retain_grad(), inp3_cuda.retain_grad())


    @onlyCUDA
    @dtypesIfCUDA(*set([torch.float, torch.double, torch.half, torch.bfloat16,
            torch.complex64, torch.complex128, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]))
    @with_tf32_off_helper
    def test_addcmul(self, device, dtype):
        torch.manual_seed(0)
        list_size = [
            [[5, 7], 0.1], [[512, 768], 0.2], [[2048, 2048], 0.4],
            [[64, 5, 7], 0.5], [[4, 512, 768], 0.7], [[4, 1024, 1024], 0.8]
        ]
        for sz, value in list_size:
            for contig in [True, False]:
                if dtype in ALL_INTEGER_TYPES:
                    inp1_temp = numpy.random.randint(-2048, 2048, size=sz)
                    inp1_cpu = torch.tensor(inp1_temp, dtype=dtype)
                    inp1_cuda = inp1_cpu.clone().detach().cuda()
                    inp2_temp = numpy.random.randint(-2048, 2048, size=sz)
                    inp2_cpu = torch.tensor(inp2_temp, dtype=dtype)
                    inp2_cuda = inp2_cpu.clone().detach().cuda()
                    inp3_temp = numpy.random.randint(-2048, 2048, size=sz)
                    inp3_cpu = torch.tensor(inp3_temp, dtype=dtype)
                    inp3_cuda = inp3_cpu.clone().detach().cuda()
                else:
                    inp1 = torch.randn(sz, dtype=dtype, requires_grad=True)
                    inp1_cuda = inp1.clone().detach().requires_grad_(True).cuda()
                    inp1_cpu = inp1_cuda.clone().detach().requires_grad_(True).cpu()
                    inp2 = torch.randn(sz, dtype=dtype, requires_grad=True)
                    inp2_cuda = inp2.clone().detach().requires_grad_(True).cuda()
                    inp2_cpu = inp2_cuda.clone().detach().requires_grad_(True).cpu()
                    inp3 = torch.randn(sz, dtype=dtype, requires_grad=True)
                    inp3_cuda = inp3.clone().detach().requires_grad_(True).cuda()
                    inp3_cpu = inp3_cuda.clone().detach().requires_grad_(True).cpu()
                if contig == False:
                    inp1_cpu = make_noncontig(inp1_cpu)
                    inp1_cuda = make_noncontig(inp1_cuda)
                    inp2_cpu = make_noncontig(inp2_cpu)
                    inp2_cuda = make_noncontig(inp2_cuda)
                    inp3_cpu = make_noncontig(inp3_cpu)
                    inp3_cuda = make_noncontig(inp3_cuda)
                out_cuda = torch.addcmul(inp1_cuda, inp2_cuda, inp3_cuda, value=value)
                if dtype != torch.half:
                    out_cpu = torch.addcmul(inp1_cpu, inp2_cpu, inp3_cpu, value=value)
                    self.assertEqual(out_cpu, out_cuda)
                if dtype not in ALL_INTEGER_TYPES:
                    out_cuda.sum().backward()
                    if dtype != torch.half:
                        out_cpu.sum().backward()
                        self.assertEqual(inp1_cpu.retain_grad(), inp1_cuda.retain_grad())
                        self.assertEqual(inp2_cpu.retain_grad(), inp2_cuda.retain_grad())
                        self.assertEqual(inp3_cpu.retain_grad(), inp3_cuda.retain_grad())


    @onlyCUDA
    @dtypesIfCUDA(*set([torch.float, torch.double, torch.complex64, torch.complex128,
            torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]))
    @with_tf32_off_helper
    def test_angle(self, device, dtype):
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
                    inp_cuda = inp_cpu.clone().detach().cuda()
                else:
                    inp = torch.randn(sz, dtype=dtype, requires_grad=True)
                    inp_cuda = inp.clone().detach().requires_grad_(True).cuda()
                    inp_cpu = inp_cuda.clone().detach().requires_grad_(True).cpu()
                if contig == False:
                    inp_cpu = make_noncontig(inp_cpu)
                    inp_cuda = make_noncontig(inp_cuda)
                out_cpu = torch.angle(inp_cpu)
                out_cuda = torch.angle(inp_cuda)
                self.assertEqual(out_cpu, out_cuda)
                if dtype not in ALL_INTEGER_TYPES:
                    out_cpu.sum().backward()
                    out_cuda.sum().backward()
                    self.assertEqual(inp_cpu.retain_grad(), inp_cuda.retain_grad())


    @onlyCUDA
    @dtypesIfCUDA(*set([torch.float, torch.double, torch.half, torch.bfloat16,
            torch.complex64, torch.complex128, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]))
    @with_tf32_off_helper
    def test_asin(self, device, dtype):
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
                    inp_cuda = inp_cpu.clone().detach().cuda()
                else:
                    inp = torch.randn(sz, dtype=dtype, requires_grad=True)
                    inp_cuda = inp.clone().detach().requires_grad_(True).cuda()
                    inp_cpu = inp_cuda.clone().detach().requires_grad_(True).cpu()
                if contig == False:
                    inp_cpu = make_noncontig(inp_cpu)
                    inp_cuda = make_noncontig(inp_cuda)
                out_cuda = torch.asin(inp_cuda)
                if dtype != torch.half:
                    out_cpu = torch.asin(inp_cpu)
                    self.assertEqual(out_cpu, out_cuda)
                if dtype not in ALL_INTEGER_TYPES:
                    out_cuda.sum().backward()
                    if dtype != torch.half:
                        out_cpu.sum().backward()
                        self.assertEqual(inp_cpu.retain_grad(), inp_cuda.retain_grad())


    @onlyCUDA
    @dtypesIfCUDA(*set([torch.float, torch.double, torch.half, torch.bfloat16,
            torch.complex64, torch.complex128, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]))
    @with_tf32_off_helper
    def test_arcsin(self, device, dtype):
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
                    inp_cuda = inp_cpu.clone().detach().cuda()
                else:
                    inp = torch.randn(sz, dtype=dtype, requires_grad=True)
                    inp_cuda = inp.clone().detach().requires_grad_(True).cuda()
                    inp_cpu = inp_cuda.clone().detach().requires_grad_(True).cpu()
                if contig == False:
                    inp_cpu = make_noncontig(inp_cpu)
                    inp_cuda = make_noncontig(inp_cuda)
                out_cuda = torch.arcsin(inp_cuda)
                if dtype != torch.half:
                    out_cpu = torch.arcsin(inp_cpu)
                    self.assertEqual(out_cpu, out_cuda)
                if dtype not in ALL_INTEGER_TYPES:
                    out_cuda.sum().backward()
                    if dtype != torch.half:
                        out_cpu.sum().backward()
                        self.assertEqual(inp_cpu.retain_grad(), inp_cuda.retain_grad())


    @onlyCUDA
    @dtypesIfCUDA(*set([torch.float, torch.double, torch.half, torch.bfloat16,
            torch.complex64, torch.complex128, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]))
    @with_tf32_off_helper
    def test_asinh(self, device, dtype):
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
                    inp_cuda = inp_cpu.clone().detach().cuda()
                else:
                    inp = torch.randn(sz, dtype=dtype, requires_grad=True)
                    inp_cuda = inp.clone().detach().requires_grad_(True).cuda()
                    inp_cpu = inp_cuda.clone().detach().requires_grad_(True).cpu()
                if contig == False:
                    inp_cpu = make_noncontig(inp_cpu)
                    inp_cuda = make_noncontig(inp_cuda)
                out_cuda = torch.asinh(inp_cuda)
                if dtype != torch.half:
                    out_cpu = torch.asinh(inp_cpu)
                    self.assertEqual(out_cpu, out_cuda)
                if dtype not in ALL_INTEGER_TYPES:
                    out_cuda.sum().backward()
                    if dtype != torch.half:
                        out_cpu.sum().backward()
                        self.assertEqual(inp_cpu.retain_grad(), inp_cuda.retain_grad())


    @onlyCUDA
    @dtypesIfCUDA(*set([torch.float, torch.double, torch.half, torch.bfloat16,
            torch.complex64, torch.complex128, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]))
    @with_tf32_off_helper
    def test_arcsinh(self, device, dtype):
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
                    inp_cuda = inp_cpu.clone().detach().cuda()
                else:
                    inp = torch.randn(sz, dtype=dtype, requires_grad=True)
                    inp_cuda = inp.clone().detach().requires_grad_(True).cuda()
                    inp_cpu = inp_cuda.clone().detach().requires_grad_(True).cpu()
                if contig == False:
                    inp_cpu = make_noncontig(inp_cpu)
                    inp_cuda = make_noncontig(inp_cuda)
                out_cuda = torch.arcsinh(inp_cuda)
                if dtype != torch.half:
                    out_cpu = torch.arcsinh(inp_cpu)
                    self.assertEqual(out_cpu, out_cuda)
                if dtype not in ALL_INTEGER_TYPES:
                    out_cuda.sum().backward()
                    if dtype != torch.half:
                        out_cpu.sum().backward()
                        self.assertEqual(inp_cpu.retain_grad(), inp_cuda.retain_grad())


    @onlyCUDA
    @dtypesIfCUDA(*set([torch.float, torch.double, torch.half, torch.bfloat16,
            torch.complex64, torch.complex128, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]))
    @with_tf32_off_helper
    def test_atan(self, device, dtype):
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
                    inp_cuda = inp_cpu.clone().detach().cuda()
                else:
                    inp = torch.rand(sz, dtype=dtype, requires_grad=True)
                    inp_cuda = inp.clone().detach().requires_grad_(True).cuda()
                    inp_cpu = inp_cuda.clone().detach().requires_grad_(True).cpu()
                if contig == False:
                    inp_cpu = make_noncontig(inp_cpu)
                    inp_cuda = make_noncontig(inp_cuda)
                out_cuda = torch.atan(inp_cuda)
                if dtype != torch.half:
                    out_cpu = torch.atan(inp_cpu)
                    self.assertEqual(out_cpu, out_cuda)
                if dtype not in ALL_INTEGER_TYPES:
                    out_cuda.sum().backward()
                    if dtype != torch.half:
                        out_cpu.sum().backward()
                        self.assertEqual(inp_cpu.retain_grad(), inp_cuda.retain_grad())


    @onlyCUDA
    @dtypesIfCUDA(*set([torch.float, torch.double, torch.half, torch.bfloat16,
            torch.complex64, torch.complex128, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]))
    @with_tf32_off_helper
    def test_arctan(self, device, dtype):
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
                    inp_cuda = inp_cpu.clone().detach().cuda()
                else:
                    inp = torch.rand(sz, dtype=dtype, requires_grad=True)
                    inp_cuda = inp.clone().detach().requires_grad_(True).cuda()
                    inp_cpu = inp_cuda.clone().detach().requires_grad_(True).cpu()
                if contig == False:
                    inp_cpu = make_noncontig(inp_cpu)
                    inp_cuda = make_noncontig(inp_cuda)
                out_cuda = torch.arctan(inp_cuda)
                if dtype != torch.half:
                    out_cpu = torch.arctan(inp_cpu)
                    self.assertEqual(out_cpu, out_cuda)
                if dtype not in ALL_INTEGER_TYPES:
                    out_cuda.sum().backward()
                    if dtype != torch.half:
                        out_cpu.sum().backward()
                        self.assertEqual(inp_cpu.retain_grad(), inp_cuda.retain_grad())


    @onlyCUDA
    @dtypesIfCUDA(*set([torch.float, torch.double, torch.half, torch.bfloat16,
            torch.complex64, torch.complex128, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]))
    @with_tf32_off_helper
    def test_atanh(self, device, dtype):
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
                    inp_cuda = inp_cpu.clone().detach().cuda()
                else:
                    inp = torch.randn(sz, dtype=dtype, requires_grad=True)
                    inp_cuda = inp.clone().detach().requires_grad_(True).cuda()
                    inp_cpu = inp_cuda.clone().detach().requires_grad_(True).cpu()
                if contig == False:
                    inp_cpu = make_noncontig(inp_cpu)
                    inp_cuda = make_noncontig(inp_cuda)
                out_cuda = torch.atanh(inp_cuda)
                if dtype != torch.half:
                    out_cpu = torch.atanh(inp_cpu)
                    self.assertEqual(out_cpu, out_cuda)
                if dtype not in ALL_INTEGER_TYPES:
                    out_cuda.sum().backward()
                    if dtype != torch.half:
                        out_cpu.sum().backward()
                        self.assertEqual(inp_cpu.retain_grad(), inp_cuda.retain_grad())


    @onlyCUDA
    @dtypesIfCUDA(*set([torch.float, torch.double, torch.half, torch.bfloat16,
            torch.complex64, torch.complex128, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]))
    @with_tf32_off_helper
    def test_arctanh(self, device, dtype):
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
                    inp_cuda = inp_cpu.clone().detach().cuda()
                else:
                    inp = torch.randn(sz, dtype=dtype, requires_grad=True)
                    inp_cuda = inp.clone().detach().requires_grad_(True).cuda()
                    inp_cpu = inp_cuda.clone().detach().requires_grad_(True).cpu()
                if contig == False:
                    inp_cpu = make_noncontig(inp_cpu)
                    inp_cuda = make_noncontig(inp_cuda)
                out_cuda = torch.arctanh(inp_cuda)
                if dtype != torch.half:
                    out_cpu = torch.arctanh(inp_cpu)
                    self.assertEqual(out_cpu, out_cuda)
                if dtype not in ALL_INTEGER_TYPES:
                    out_cuda.sum().backward()
                    if dtype != torch.half:
                        out_cpu.sum().backward()
                        self.assertEqual(inp_cpu.retain_grad(), inp_cuda.retain_grad())


    @onlyCUDA
    @dtypesIfCUDA(*set([torch.float, torch.double, torch.half, torch.bfloat16,
            torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]))
    @with_tf32_off_helper
    def test_atan2(self, device, dtype):
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
                    inp1_cuda = inp1_cpu.clone().detach().cuda()
                    inp2_temp = numpy.random.randint(-2048, 2048, size=sz)
                    inp2_cpu = torch.tensor(inp2_temp, dtype=dtype)
                    inp2_cuda = inp2_cpu.clone().detach().cuda()
                else:
                    inp1 = torch.randn(sz, dtype=dtype, requires_grad=True)
                    inp1_cuda = inp1.clone().detach().requires_grad_(True).cuda()
                    inp1_cpu = inp1_cuda.clone().detach().requires_grad_(True).cpu()
                    inp2 = torch.randn(sz, dtype=dtype, requires_grad=True)
                    inp2_cuda = inp2.clone().detach().requires_grad_(True).cuda()
                    inp2_cpu = inp2_cuda.clone().detach().requires_grad_(True).cpu()
                if contig == False:
                    inp1_cpu = make_noncontig(inp1_cpu)
                    inp1_cuda = make_noncontig(inp1_cuda)
                    inp2_cpu = make_noncontig(inp2_cpu)
                    inp2_cuda = make_noncontig(inp2_cuda)
                out_cuda = torch.atan2(inp1_cuda, inp2_cuda)
                if dtype != torch.half and dtype != torch.bfloat16:
                    out_cpu = torch.atan2(inp1_cpu, inp2_cpu)
                    self.assertEqual(out_cpu, out_cuda)
                if dtype not in ALL_INTEGER_TYPES:
                    out_cuda.sum().backward()
                    if dtype != torch.half and dtype != torch.bfloat16:
                        out_cpu.sum().backward()
                        self.assertEqual(inp1_cpu.retain_grad(), inp1_cuda.retain_grad())
                        self.assertEqual(inp2_cpu.retain_grad(), inp2_cuda.retain_grad())


    @with_tf32_off_helper
    def test_arctan2(self):
        # module 'torch' has no attribute 'arctan2' in pytorch version 1.10
        pass


    @onlyCUDA
    @dtypesIfCUDA(*set([torch.int8, torch.int16, torch.int32, torch.int64]))
    @with_tf32_off_helper
    def test_bitwise_not(self, device, dtype):
        torch.manual_seed(0)
        list_size = [
            [5, 7], [512, 768], [2048, 2048],
            [64, 5, 7], [4, 512, 768], [4, 1024, 1024]
        ]
        for sz in list_size:
            for contig in [True, False]:
                inp_temp = numpy.random.randint(-2048, 2048, size=sz)
                inp_cpu = torch.tensor(inp_temp, dtype=dtype)
                inp_cuda = inp_cpu.clone().detach().cuda()
                if contig == False:
                    inp_cpu = make_noncontig(inp_cpu)
                    inp_cuda = make_noncontig(inp_cuda)
                out_cpu = torch.bitwise_not(inp_cpu)
                out_cuda = torch.bitwise_not(inp_cuda)
                self.assertEqual(out_cpu, out_cuda)


    @onlyCUDA
    @dtypesIfCUDA(*set([torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]))
    @with_tf32_off_helper
    def test_bitwise_and(self, device, dtype):
        torch.manual_seed(0)
        list_size = [
            [5, 7], [512, 768], [2048, 2048],
            [64, 5, 7], [4, 512, 768], [4, 1024, 1024]
        ]
        for sz in list_size:
            for contig in [True, False]:
                inp1_temp = numpy.random.randint(-2048, 2048, size=sz)
                inp1_cpu = torch.tensor(inp1_temp, dtype=dtype)
                inp1_cuda = inp1_cpu.clone().detach().cuda()
                inp2_temp = numpy.random.randint(-2048, 2048, size=sz)
                inp2_cpu = torch.tensor(inp2_temp, dtype=dtype)
                inp2_cuda = inp2_cpu.clone().detach().cuda()
                if contig == False:
                    inp1_cpu = make_noncontig(inp1_cpu)
                    inp1_cuda = make_noncontig(inp1_cuda)
                    inp2_cpu = make_noncontig(inp2_cpu)
                    inp2_cuda = make_noncontig(inp2_cuda)
                out_cpu = torch.bitwise_and(inp1_cpu, inp2_cpu)
                out_cuda = torch.bitwise_and(inp1_cuda, inp2_cuda)
                self.assertEqual(out_cpu, out_cuda)


    @onlyCUDA
    @dtypesIfCUDA(*set([torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]))
    @with_tf32_off_helper
    def test_bitwise_or(self, device, dtype):
        torch.manual_seed(0)
        list_size = [
            [5, 7], [512, 768], [2048, 2048],
            [64, 5, 7], [4, 512, 768], [4, 1024, 1024]
        ]
        for sz in list_size:
            for contig in [True, False]:
                inp1_temp = numpy.random.randint(-2048, 2048, size=sz)
                inp1_cpu = torch.tensor(inp1_temp, dtype=dtype)
                inp1_cuda = inp1_cpu.clone().detach().cuda()
                inp2_temp = numpy.random.randint(-2048, 2048, size=sz)
                inp2_cpu = torch.tensor(inp2_temp, dtype=dtype)
                inp2_cuda = inp2_cpu.clone().detach().cuda()
                if contig == False:
                    inp1_cpu = make_noncontig(inp1_cpu)
                    inp1_cuda = make_noncontig(inp1_cuda)
                    inp2_cpu = make_noncontig(inp2_cpu)
                    inp2_cuda = make_noncontig(inp2_cuda)
                out_cpu = torch.bitwise_or(inp1_cpu, inp2_cpu)
                out_cuda = torch.bitwise_or(inp1_cuda, inp2_cuda)
                self.assertEqual(out_cpu, out_cuda)


    @onlyCUDA
    @dtypesIfCUDA(*set([torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]))
    @with_tf32_off_helper
    def test_bitwise_xor(self, device, dtype):
        torch.manual_seed(0)
        list_size = [
            [5, 7], [512, 768], [2048, 2048],
            [64, 5, 7], [4, 512, 768], [4, 1024, 1024]
        ]
        for sz in list_size:
            for contig in [True, False]:
                inp1_temp = numpy.random.randint(-2048, 2048, size=sz)
                inp1_cpu = torch.tensor(inp1_temp, dtype=dtype)
                inp1_cuda = inp1_cpu.clone().detach().cuda()
                inp2_temp = numpy.random.randint(-2048, 2048, size=sz)
                inp2_cpu = torch.tensor(inp2_temp, dtype=dtype)
                inp2_cuda = inp2_cpu.clone().detach().cuda()
                if contig == False:
                    inp1_cpu = make_noncontig(inp1_cpu)
                    inp1_cuda = make_noncontig(inp1_cuda)
                    inp2_cpu = make_noncontig(inp2_cpu)
                    inp2_cuda = make_noncontig(inp2_cuda)
                out_cpu = torch.bitwise_xor(inp1_cpu, inp2_cpu)
                out_cuda = torch.bitwise_xor(inp1_cuda, inp2_cuda)
                self.assertEqual(out_cpu, out_cuda)


    @onlyCUDA
    @dtypesIfCUDA(*set([torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]))
    @with_tf32_off_helper
    def test_bitwise_left_shift(self, device, dtype):
        torch.manual_seed(0)
        list_size = [
            [5, 7], [512, 768], [2048, 2048],
            [64, 5, 7], [4, 512, 768], [4, 1024, 1024]
        ]
        for sz in list_size:
            for contig in [True, False]:
                inp1_temp = numpy.random.randint(-2048, 2048, size=sz)
                inp1_cpu = torch.tensor(inp1_temp, dtype=dtype)
                inp1_cuda = inp1_cpu.clone().detach().cuda()
                inp2_temp = numpy.random.randint(-2048, 2048, size=sz)
                inp2_cpu = torch.tensor(inp2_temp, dtype=dtype)
                inp2_cuda = inp2_cpu.clone().detach().cuda()
                if contig == False:
                    inp1_cpu = make_noncontig(inp1_cpu)
                    inp1_cuda = make_noncontig(inp1_cuda)
                    inp2_cpu = make_noncontig(inp2_cpu)
                    inp2_cuda = make_noncontig(inp2_cuda)
                out_cpu = torch.bitwise_left_shift(inp1_cpu, inp2_cpu)
                out_cuda = torch.bitwise_left_shift(inp1_cuda, inp2_cuda)


    @onlyCUDA
    @dtypesIfCUDA(*set([torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]))
    @with_tf32_off_helper
    def test_bitwise_right_shift(self, device, dtype):
        torch.manual_seed(0)
        list_size = [
            [5, 7], [512, 768], [2048, 2048],
            [64, 5, 7], [4, 512, 768], [4, 1024, 1024]
        ]
        for sz in list_size:
            for contig in [True, False]:
                inp1_temp = numpy.random.randint(-2048, 2048, size=sz)
                inp1_cpu = torch.tensor(inp1_temp, dtype=dtype)
                inp1_cuda = inp1_cpu.clone().detach().cuda()
                inp2_temp = numpy.random.randint(-2048, 2048, size=sz)
                inp2_cpu = torch.tensor(inp2_temp, dtype=dtype)
                inp2_cuda = inp2_cpu.clone().detach().cuda()
                if contig == False:
                    inp1_cpu = make_noncontig(inp1_cpu)
                    inp1_cuda = make_noncontig(inp1_cuda)
                    inp2_cpu = make_noncontig(inp2_cpu)
                    inp2_cuda = make_noncontig(inp2_cuda)
                out_cpu = torch.bitwise_right_shift(inp1_cpu, inp2_cpu)
                out_cuda = torch.bitwise_right_shift(inp1_cuda, inp2_cuda)


    @onlyCUDA
    @dtypesIfCUDA(*set([torch.float, torch.double, torch.half, torch.bfloat16]))
    @with_tf32_off_helper
    def test_ceil(self, device, dtype):
        torch.manual_seed(0)
        list_size = [
            [5, 7], [512, 768], [2048, 2048],
            [64, 5, 7], [4, 512, 768], [4, 1024, 1024]
        ]
        for sz in list_size:
            for contig in [True, False]:
                inp = torch.randn(sz, dtype=dtype, requires_grad=True)
                inp_cuda = inp.clone().detach().requires_grad_(True).cuda()
                inp_cpu = inp_cuda.clone().detach().requires_grad_(True).cpu()
                if contig == False:
                    inp_cpu = make_noncontig(inp_cpu)
                    inp_cuda = make_noncontig(inp_cuda)
                out_cuda = torch.ceil(inp_cuda)
                out_cuda.sum().backward()
                if dtype != torch.half:
                    out_cpu = torch.ceil(inp_cpu)
                    out_cpu.sum().backward()
                    self.assertEqual(out_cpu, out_cuda)
                    self.assertEqual(inp_cpu.retain_grad(), inp_cuda.retain_grad())


    @onlyCUDA
    @dtypesIfCUDA(*set([torch.float, torch.double, torch.half, torch.bfloat16,
            torch.int8, torch.int16, torch.int32, torch.int64]))
    @with_tf32_off_helper
    def test_clamp(self, device, dtype):
        torch.manual_seed(0)
        list_size = [
            [[5, 7], [-0.1, 0.1]], [[512, 768], [-0.2, 0.2]], [[2048, 2048], [-0.4, 0.4]],
            [[64, 5, 7], [-0.5, 0.5]], [[32, 512, 768], [-0.7, 0.7]], [[64, 1024, 1024], [-0.8, 0.8]]
        ]
        for sz, value in list_size:
            for contig in [True, False]:
                if dtype in ALL_INTEGER_TYPES:
                    inp_temp = numpy.random.randint(-2048, 2048, size=sz)
                    inp_cpu = torch.tensor(inp_temp, dtype=dtype)
                    inp_cuda = inp_cpu.clone().detach().cuda()
                else:
                    inp = torch.randn(sz, dtype=dtype, requires_grad=True)
                    inp_cuda = inp.clone().detach().requires_grad_(True).cuda()
                    inp_cpu = inp_cuda.clone().detach().requires_grad_(True).cpu()
                if contig == False:
                    inp_cpu = make_noncontig(inp_cpu)
                    inp_cuda = make_noncontig(inp_cuda)
                out_cuda = torch.clamp(inp_cuda, min=value[0], max=value[1])
                if dtype != torch.half:
                    out_cpu = torch.clamp(inp_cpu, min=value[0], max=value[1])
                    self.assertEqual(out_cpu, out_cuda)
                if dtype not in ALL_INTEGER_TYPES:
                    out_cuda.sum().backward()
                    if dtype != torch.half:
                        out_cpu.sum().backward()
                        self.assertEqual(inp_cpu.retain_grad(), inp_cuda.retain_grad())


    @onlyCUDA
    @dtypesIfCUDA(*set([torch.float, torch.double, torch.half, torch.bfloat16,
            torch.int8, torch.int16, torch.int32, torch.int64]))
    @with_tf32_off_helper
    def test_clip(self, device, dtype):
        torch.manual_seed(0)
        list_size = [
            [[5, 7], [-0.1, 0.1]], [[512, 768], [-0.2, 0.2]], [[2048, 2048], [-0.4, 0.4]],
            [[64, 5, 7], [-0.5, 0.5]], [[32, 512, 768], [-0.7, 0.7]], [[64, 1024, 1024], [-0.8, 0.8]]
        ]
        for sz, value in list_size:
            for contig in [True, False]:
                if dtype in ALL_INTEGER_TYPES:
                    inp_temp = numpy.random.randint(-2048, 2048, size=sz)
                    inp_cpu = torch.tensor(inp_temp, dtype=dtype)
                    inp_cuda = inp_cpu.clone().detach().cuda()
                else:
                    inp = torch.randn(sz, dtype=dtype, requires_grad=True)
                    inp_cuda = inp.clone().detach().requires_grad_(True).cuda()
                    inp_cpu = inp_cuda.clone().detach().requires_grad_(True).cpu()
                if contig == False:
                    inp_cpu = make_noncontig(inp_cpu)
                    inp_cuda = make_noncontig(inp_cuda)
                out_cuda = torch.clip(inp_cuda, min=value[0], max=value[1])
                if dtype != torch.half:
                    out_cpu = torch.clip(inp_cpu, min=value[0], max=value[1])
                    self.assertEqual(out_cpu, out_cuda)
                if dtype not in ALL_INTEGER_TYPES:
                    out_cuda.sum().backward()
                    if dtype != torch.half:
                        out_cpu.sum().backward()
                        self.assertEqual(inp_cpu.retain_grad(), inp_cuda.retain_grad())


    @onlyCUDA
    @dtypesIfCUDA(*set([torch.float, torch.double, torch.half, torch.bfloat16,
            torch.complex64, torch.complex128, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]))
    @with_tf32_off_helper
    def test_conj_physical(self, device, dtype):
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
                    inp_cuda = inp_cpu.clone().detach().cuda()
                else:
                    inp = torch.randn(sz, dtype=dtype, requires_grad=True)
                    inp_cuda = inp.clone().detach().requires_grad_(True).cuda()
                    inp_cpu = inp_cuda.clone().detach().requires_grad_(True).cpu()
                if contig == False:
                    inp_cpu = make_noncontig(inp_cpu)
                    inp_cuda = make_noncontig(inp_cuda)
                out_cpu = torch.conj_physical(inp_cpu)
                out_cuda = torch.conj_physical(inp_cuda)
                self.assertEqual(out_cpu, out_cuda)
                if dtype not in ALL_INTEGER_TYPES:
                    out_cpu.sum().backward()
                    out_cuda.sum().backward()
                    self.assertEqual(inp_cpu.retain_grad(), inp_cuda.retain_grad())


    @onlyCUDA
    @dtypesIfCUDA(*set([torch.float, torch.double, torch.half, torch.bfloat16,
            torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]))
    @with_tf32_off_helper
    def test_copysign(self, device, dtype):
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
                    inp1_cuda = inp1_cpu.clone().detach().cuda()
                    inp2_temp = numpy.random.randint(-2048, 2048, size=sz)
                    inp2_cpu = torch.tensor(inp2_temp, dtype=dtype)
                    inp2_cuda = inp2_cpu.clone().detach().cuda()
                else:
                    inp1 = torch.randn(sz, dtype=dtype, requires_grad=True)
                    inp1_cuda = inp1.clone().detach().requires_grad_(True).cuda()
                    inp1_cpu = inp1_cuda.clone().detach().requires_grad_(True).cpu()
                    inp2 = torch.randn(sz, dtype=dtype, requires_grad=True)
                    inp2_cuda = inp2.clone().detach().requires_grad_(True).cuda()
                    inp2_cpu = inp2_cuda.clone().detach().requires_grad_(True).cpu()
                if contig == False:
                    inp1_cpu = make_noncontig(inp1_cpu)
                    inp1_cuda = make_noncontig(inp1_cuda)
                    inp2_cpu = make_noncontig(inp2_cpu)
                    inp2_cuda = make_noncontig(inp2_cuda)
                out_cpu = torch.copysign(inp1_cpu, inp2_cpu)
                out_cuda = torch.copysign(inp1_cuda, inp2_cuda)
                self.assertEqual(out_cpu, out_cuda)
                if dtype not in ALL_INTEGER_TYPES:
                    out_cpu.sum().backward()
                    out_cuda.sum().backward()
                    self.assertEqual(inp1_cpu.retain_grad(), inp1_cuda.retain_grad())
                    self.assertEqual(inp2_cpu.retain_grad(), inp2_cuda.retain_grad())


    @onlyCUDA
    @dtypesIfCUDA(*set([torch.float, torch.double, torch.half, torch.bfloat16,
            torch.complex64, torch.complex128, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]))
    @with_tf32_off_helper
    def test_cos(self, device, dtype):
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
                    inp_cuda = inp_cpu.clone().detach().cuda()
                else:
                    inp = torch.randn(sz, dtype=dtype, requires_grad=True)
                    inp_cuda = inp.clone().detach().requires_grad_(True).cuda()
                    inp_cpu = inp_cuda.clone().detach().requires_grad_(True).cpu()
                if contig == False:
                    inp_cpu = make_noncontig(inp_cpu)
                    inp_cuda = make_noncontig(inp_cuda)
                out_cuda = torch.cos(inp_cuda)
                if dtype != torch.half:
                    out_cpu = torch.cos(inp_cpu)
                    self.assertEqual(out_cpu, out_cuda)
                if dtype not in ALL_INTEGER_TYPES:
                    out_cuda.sum().backward()
                    if dtype != torch.half:
                        out_cpu.sum().backward()
                        self.assertEqual(inp_cpu.retain_grad(), inp_cuda.retain_grad())


    @onlyCUDA
    @dtypesIfCUDA(*set([torch.float, torch.double, torch.half, torch.bfloat16,
            torch.complex64, torch.complex128, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]))
    @with_tf32_off_helper
    def test_cosh(self, device, dtype):
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
                    inp_cuda = inp_cpu.clone().detach().cuda()
                else:
                    inp = torch.randn(sz, dtype=dtype, requires_grad=True)
                    inp_cuda = inp.clone().detach().requires_grad_(True).cuda()
                    inp_cpu = inp_cuda.clone().detach().requires_grad_(True).cpu()
                if contig == False:
                    inp_cpu = make_noncontig(inp_cpu)
                    inp_cuda = make_noncontig(inp_cuda)
                out_cuda = torch.cosh(inp_cuda)
                if dtype != torch.half:
                    out_cpu = torch.cosh(inp_cpu)
                    if dtype not in ALL_INTEGER_TYPES:
                        self.assertEqual(out_cpu, out_cuda)
                if dtype not in ALL_INTEGER_TYPES:
                    out_cuda.sum().backward()
                    if dtype != torch.half:
                        out_cpu.sum().backward()
                        self.assertEqual(inp_cpu.retain_grad(), inp_cuda.retain_grad())


    @onlyCUDA
    @dtypesIfCUDA(*set([torch.float, torch.double, torch.half, torch.bfloat16,
            torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]))
    @with_tf32_off_helper
    def test_deg2rad(self, device, dtype):
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
                    inp_cuda = inp_cpu.clone().detach().cuda()
                else:
                    inp = torch.randn(sz, dtype=dtype, requires_grad=True)
                    inp_cuda = inp.clone().detach().requires_grad_(True).cuda()
                    inp_cpu = inp_cuda.clone().detach().requires_grad_(True).cpu()
                if contig == False:
                    inp_cpu = make_noncontig(inp_cpu)
                    inp_cuda = make_noncontig(inp_cuda)
                out_cpu = torch.deg2rad(inp_cpu)
                out_cuda = torch.deg2rad(inp_cuda)
                self.assertEqual(out_cpu, out_cuda)
                if dtype not in ALL_INTEGER_TYPES:
                    out_cpu.sum().backward()
                    out_cuda.sum().backward()
                    self.assertEqual(inp_cpu.retain_grad(), inp_cuda.retain_grad())


    @onlyCUDA
    @dtypesIfCUDA(*set([torch.float, torch.double, torch.half, torch.bfloat16,
            torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]))
    @with_tf32_off_helper
    def test_div(self, device, dtype):
        torch.manual_seed(0)
        list_size = [
            [5, 7], [512, 768], [2048, 2048],
            [64, 5, 7], [4, 512, 768], [4, 1024, 1024]
        ]
        for sz in list_size:
            for contig in [True, False]:
                for mod in [None, 'trunc', 'floor']:
                    if dtype in ALL_INTEGER_TYPES:
                        inp1_temp = numpy.random.randint(-2048, 2048, size=sz)
                        inp1_cpu = torch.tensor(inp1_temp, dtype=dtype)
                        inp1_cuda = inp1_cpu.clone().detach().cuda()
                        inp2_temp = numpy.random.randint(-2048, 2048, size=sz)
                        inp2_cpu = torch.tensor(inp2_temp, dtype=dtype)
                        inp2_cuda = inp2_cpu.clone().detach().cuda()
                    else:
                        inp1 = torch.randn(sz, dtype=dtype, requires_grad=True)
                        inp1_cuda = inp1.clone().detach().requires_grad_(True).cuda()
                        inp1_cpu = inp1_cuda.clone().detach().requires_grad_(True).cpu()
                        inp2 = torch.randn(sz, dtype=dtype, requires_grad=True)
                        inp2_cuda = inp2.clone().detach().requires_grad_(True).cuda()
                        inp2_cpu = inp2_cuda.clone().detach().requires_grad_(True).cpu()
                    if contig == False:
                        inp1_cpu = make_noncontig(inp1_cpu)
                        inp1_cuda = make_noncontig(inp1_cuda)
                        inp2_cpu = make_noncontig(inp2_cpu)
                        inp2_cuda = make_noncontig(inp2_cuda)
                    out_cuda = torch.div(inp1_cuda, inp2_cuda, rounding_mode=mod)
                    if (dtype != torch.half) and (dtype not in ALL_INTEGER_TYPES):
                        out_cpu = torch.div(inp1_cpu, inp2_cpu, rounding_mode=mod)
                        self.assertEqual(out_cpu, out_cuda)
                        out_cuda.sum().backward()
                        out_cpu.sum().backward()
                        self.assertEqual(inp1_cpu.retain_grad(), inp1_cuda.retain_grad())
                        self.assertEqual(inp2_cpu.retain_grad(), inp2_cuda.retain_grad())


    @onlyCUDA
    @dtypesIfCUDA(*set([torch.float, torch.double, torch.half, torch.bfloat16,
            torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]))
    @with_tf32_off_helper
    def test_divide(self, device, dtype):
        torch.manual_seed(0)
        list_size = [
            [5, 7], [512, 768], [2048, 2048],
            [64, 5, 7], [4, 512, 768], [4, 1024, 1024]
        ]
        for sz in list_size:
            for contig in [True, False]:
                for mod in [None, 'trunc', 'floor']:
                    if dtype in ALL_INTEGER_TYPES:
                        inp1_temp = numpy.random.randint(-2048, 2048, size=sz)
                        inp1_cpu = torch.tensor(inp1_temp, dtype=dtype)
                        inp1_cuda = inp1_cpu.clone().detach().cuda()
                        inp2_temp = numpy.random.randint(-2048, 2048, size=sz)
                        inp2_cpu = torch.tensor(inp2_temp, dtype=dtype)
                        inp2_cuda = inp2_cpu.clone().detach().cuda()
                    else:
                        inp1 = torch.randn(sz, dtype=dtype, requires_grad=True)
                        inp1_cuda = inp1.clone().detach().requires_grad_(True).cuda()
                        inp1_cpu = inp1_cuda.clone().detach().requires_grad_(True).cpu()
                        inp2 = torch.randn(sz, dtype=dtype, requires_grad=True)
                        inp2_cuda = inp2.clone().detach().requires_grad_(True).cuda()
                        inp2_cpu = inp2_cuda.clone().detach().requires_grad_(True).cpu()
                    if contig == False:
                        inp1_cpu = make_noncontig(inp1_cpu)
                        inp1_cuda = make_noncontig(inp1_cuda)
                        inp2_cpu = make_noncontig(inp2_cpu)
                        inp2_cuda = make_noncontig(inp2_cuda)
                    out_cuda = torch.divide(inp1_cuda, inp2_cuda, rounding_mode=mod)
                    if (dtype != torch.half) and (dtype not in ALL_INTEGER_TYPES):
                        out_cpu = torch.divide(inp1_cpu, inp2_cpu, rounding_mode=mod)
                        self.assertEqual(out_cpu, out_cuda)
                        out_cuda.sum().backward()
                        out_cpu.sum().backward()
                        self.assertEqual(inp1_cpu.retain_grad(), inp1_cuda.retain_grad())
                        self.assertEqual(inp2_cpu.retain_grad(), inp2_cuda.retain_grad())


    @onlyCUDA
    @dtypesIfCUDA(*set([torch.float, torch.double, torch.half,
            torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]))
    @with_tf32_off_helper
    def test_digamma(self, device, dtype):
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
                    inp_cuda = inp_cpu.clone().detach().cuda()
                else:
                    inp = torch.randn(sz, dtype=dtype, requires_grad=True)
                    inp_cuda = inp.clone().detach().requires_grad_(True).cuda()
                    inp_cpu = inp_cuda.clone().detach().requires_grad_(True).cpu()
                if contig == False:
                    inp_cpu = make_noncontig(inp_cpu)
                    inp_cuda = make_noncontig(inp_cuda)
                out_cuda = torch.digamma(inp_cuda)
                if dtype != torch.half:
                    out_cpu = torch.digamma(inp_cpu)
                    self.assertEqual(out_cpu, out_cuda)
                if dtype not in ALL_INTEGER_TYPES:
                    out_cuda.sum().backward()
                    if dtype != torch.half:
                        out_cpu.sum().backward()
                        self.assertEqual(inp_cpu.retain_grad(), inp_cuda.retain_grad())


instantiate_device_type_tests(TestTorchFunctionSegment4, globals())


if __name__ == '__main__':
    run_tests()
