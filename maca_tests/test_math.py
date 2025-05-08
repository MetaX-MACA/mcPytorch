import torch
import numpy
from torch.testing._internal.common_utils import TestCase


numpy.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.cuda.empty_cache()


def test_log(shape, dtype, rtol, save_tensor):
    np_num = numpy.random.uniform(low=0.1, high=10, size=shape)
    inp = torch.tensor(np_num, dtype=dtype)
    inp_cuda = inp.clone().detach().cuda()
    out_cuda = torch.log(inp_cuda)
    if save_tensor:
        torch.save(out_cuda, "./test_log_" + str(dtype) + ".pth")
    else:
        golden = torch.load("/netapp/pytorch/golden/test_math_golden/test_math/test_log_" + str(dtype) + ".pth")
        TestCase().assertEqual(golden, out_cuda, atol=1e-5, rtol=1e-5)


def test_erfinv(shape, dtype, rtol, save_tensor):
    np_num = numpy.random.uniform(low=-0.99, high=0.99, size=shape)
    inp = torch.tensor(np_num, dtype=dtype)
    inp_cuda = inp.clone().detach().cuda()
    out_cuda = torch.special.erfinv(inp_cuda)
    if save_tensor:
        torch.save(out_cuda, "./test_erfinv_" + str(dtype) + ".pth")
    else:
        golden = torch.load("/netapp/pytorch/golden/test_math_golden/test_math/test_erfinv_" + str(dtype) + ".pth")
        TestCase().assertEqual(golden, out_cuda, atol=1e-5, rtol=1e-5)


def test_tanh(shape, dtype, rtol, save_tensor):
    np_num = numpy.random.uniform(low=-0.99, high=0.99, size=shape)
    inp = torch.tensor(np_num, dtype=dtype)
    inp_cuda = inp.clone().detach().cuda()
    out_cuda = torch.tanh(inp_cuda)
    if save_tensor:
        torch.save(out_cuda, "./test_tanh_" + str(dtype) + ".pth")
    else:
        golden = torch.load("/netapp/pytorch/golden/test_math_golden/test_math/test_tanh_" + str(dtype) + ".pth")
        TestCase().assertEqual(golden, out_cuda, atol=1e-5, rtol=1e-5)


def test_sqrt(shape, dtype, rtol, save_tensor):
    np_num = numpy.random.uniform(low=0, high=1, size=shape)
    inp = torch.tensor(np_num, dtype=dtype)
    inp_cuda = inp.clone().detach().cuda()
    out_cuda = torch.sqrt(inp_cuda)
    if save_tensor:
        torch.save(out_cuda, "./test_sqrt_" + str(dtype) + ".pth")
    else:
        golden = torch.load("/netapp/pytorch/golden/test_math_golden/test_math/test_sqrt_" + str(dtype) + ".pth")
        TestCase().assertEqual(golden, out_cuda, atol=1e-5, rtol=1e-5)


def test_rsqrt(shape, dtype, rtol, save_tensor):
    np_num = numpy.random.uniform(low=0, high=1, size=shape)
    inp = torch.tensor(np_num, dtype=dtype)
    inp_cuda = inp.clone().detach().cuda()
    out_cuda = torch.rsqrt(inp_cuda)
    if save_tensor:
        torch.save(out_cuda, "./test_rsqrt_" + str(dtype) + ".pth")
    else:
        golden = torch.load("/netapp/pytorch/golden/test_math_golden/test_math/test_rsqrt_" + str(dtype) + ".pth")
        TestCase().assertEqual(golden, out_cuda, atol=1e-5, rtol=1e-5)


def test_pow(shape, dtype, rtol, save_tensor):
    np_num = numpy.random.uniform(low=-1, high=1, size=shape)
    inp = torch.tensor(np_num, dtype=dtype)
    inp_cuda = inp.clone().detach().cuda()
    out_cuda = torch.pow(inp_cuda, 2.4)
    if save_tensor:
        torch.save(out_cuda, "./test_pow_" + str(dtype) + ".pth")
    else:
        golden = torch.load("/netapp/pytorch/golden/test_math_golden/test_math/test_pow_" + str(dtype) + ".pth")
        TestCase().assertEqual(golden, out_cuda, atol=1e-5, rtol=1e-5)


def test_exp(shape, dtype, rtol, save_tensor):
    np_num = numpy.random.uniform(low=-1, high=1, size=shape)
    inp = torch.tensor(np_num, dtype=dtype)
    inp_cuda = inp.clone().detach().cuda()
    out_cuda = torch.exp(inp_cuda)
    if save_tensor:
        torch.save(out_cuda, "./test_exp_" + str(dtype) + ".pth")
    else:
        golden = torch.load("/netapp/pytorch/golden/test_math_golden/test_math/test_exp_" + str(dtype) + ".pth")
        TestCase().assertEqual(golden, out_cuda, atol=1e-5, rtol=1e-5)


shape = (8196 * 2, 8196 * 2)

save_tensor = False

# test float
test_log(shape, torch.float, 0, save_tensor)
test_erfinv(shape, torch.float, 2e-7, save_tensor)
test_tanh(shape, torch.float, 3e-7, save_tensor)
test_sqrt(shape, torch.float, 0, save_tensor)
test_rsqrt(shape, torch.float, 2e-7, save_tensor)
test_pow(shape, torch.float, 3e-7, save_tensor)
test_exp(shape, torch.float, 2e-7, save_tensor)

# test half
test_log(shape, torch.half, 0, save_tensor)
test_erfinv(shape, torch.half, 7e-4, save_tensor)
test_tanh(shape, torch.half, 0, save_tensor)
test_sqrt(shape, torch.half, 0, save_tensor)
test_rsqrt(shape, torch.half, 0, save_tensor)
test_pow(shape, torch.half, 0, save_tensor)
test_exp(shape, torch.half, 0, save_tensor)

# test bfloat16
test_log(shape, torch.bfloat16, 0, save_tensor)
# RuntimeError: "erfinv_cuda" not implemented for 'BFloat16'
# test_erfinv(shape, torch.bfloat16, 0, save_tensor)
test_tanh(shape, torch.bfloat16, 0, save_tensor)
test_sqrt(shape, torch.bfloat16, 0, save_tensor)
test_rsqrt(shape, torch.bfloat16, 0, save_tensor)
test_pow(shape, torch.bfloat16, 0, save_tensor)
test_exp(shape, torch.bfloat16, 0, save_tensor)