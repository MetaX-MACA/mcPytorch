import torch
from binascii import unhexlify
from torch.testing._internal.common_utils import TestCase

is_gen_inp = False
save_tensor = False


def gen_inp(dtype):
    list_16 = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "a", "b", "c", "d", "e", "f"]
    if dtype == torch.half or dtype == torch.bfloat16:
        inp = torch.zeros((16, 16, 16, 16), dtype=dtype)
        for i1, a1 in enumerate(list_16):
            for i2, a2 in enumerate(list_16):
                for i3, a3 in enumerate(list_16):
                    for i4, a4 in enumerate(list_16):
                        inp[i1, i2, i3, i4] = torch.frombuffer(unhexlify(bytes(a1+a2+a3+a4, 'utf-8')), dtype=dtype)
    elif dtype == torch.float:
        inp = torch.zeros((16, 16, 16, 16), dtype=dtype)
        for i1, a1 in enumerate(list_16):
            for i2, a2 in enumerate(list_16):
                for i3, a3 in enumerate(list_16):
                    for i4, a4 in enumerate(list_16):
                        for i5, a5 in enumerate(list_16):
                            for i6, a6 in enumerate(list_16):
                                for i7, a7 in enumerate(list_16):
                                    for i8, a8 in enumerate(list_16):
                                        inp[i1, i2, i3, i4, i5, i6, i7, i8] = \
                                            torch.frombuffer(unhexlify(bytes(a1+a2+a3+a4+a5+a6+a7+a8, 'utf-8')), dtype=dtype)
    inp_cuda = inp.clone().detach().cuda()
    return inp_cuda


def test_log(dtype, rtol, low=None, high=None):
    if is_gen_inp:
        inp_cuda = gen_inp(dtype)
    else:
        inp_cuda = torch.load("/netapp/pytorch/golden/test_math_golden/all_" + str(dtype) + ".pth")
    if (low != None and high != None):
        inp_cuda = inp_cuda.clamp(min=low, max=high)
    out_cuda = torch.log(inp_cuda)
    if save_tensor:
        torch.save(out_cuda, "./test_log_" + str(dtype) + ".pth")
    else:
        golden = torch.load("/netapp/pytorch/golden/test_math_golden/test_math_2/test_log_" + str(dtype) + ".pth")
        TestCase().assertEqual(golden, out_cuda, atol=0, rtol=rtol)


def test_erfinv(dtype, rtol, low=None, high=None):
    if is_gen_inp:
        inp_cuda = gen_inp(dtype)
    else:
        inp_cuda = torch.load("/netapp/pytorch/golden/test_math_golden/all_" + str(dtype) + ".pth")
    if (low != None and high != None):
        inp_cuda = inp_cuda.clamp(min=low, max=high)
    out_cuda = torch.special.erfinv(inp_cuda)
    if save_tensor:
        torch.save(out_cuda, "./test_erfinv_" + str(dtype) + ".pth")
    else:
        golden = torch.load("/netapp/pytorch/golden/test_math_golden/test_math_2/test_erfinv_" + str(dtype) + ".pth") 
        TestCase().assertEqual(golden, out_cuda, atol=0, rtol=rtol)


def test_tanh(dtype, rtol, low=None, high=None):
    if is_gen_inp:
        inp_cuda = gen_inp(dtype)
    else:
        inp_cuda = torch.load("/netapp/pytorch/golden/test_math_golden/all_" + str(dtype) + ".pth")
    if (low != None and high != None):
        inp_cuda = inp_cuda.clamp(min=low, max=high)
    out_cuda = torch.tanh(inp_cuda)
    if save_tensor:
        torch.save(out_cuda, "./test_tanh_" + str(dtype) + ".pth")
    else:
        golden = torch.load("/netapp/pytorch/golden/test_math_golden/test_math_2/test_tanh_" + str(dtype) + ".pth")
        TestCase().assertEqual(golden, out_cuda, atol=0, rtol=rtol)


def test_sqrt(dtype, rtol, low=None, high=None):
    if is_gen_inp:
        inp_cuda = gen_inp(dtype)
    else:
        inp_cuda = torch.load("/netapp/pytorch/golden/test_math_golden/all_" + str(dtype) + ".pth")
    if (low != None and high != None):
        inp_cuda = inp_cuda.clamp(min=low, max=high)
    out_cuda = torch.sqrt(inp_cuda)
    if save_tensor:
        torch.save(out_cuda, "./test_sqrt_" + str(dtype) + ".pth")
    else:
        golden = torch.load("/netapp/pytorch/golden/test_math_golden/test_math_2/test_sqrt_" + str(dtype) + ".pth")
        TestCase().assertEqual(golden, out_cuda, atol=0, rtol=rtol)


def test_rsqrt(dtype, rtol, low=None, high=None):
    if is_gen_inp:
        inp_cuda = gen_inp(dtype)
    else:
        inp_cuda = torch.load("/netapp/pytorch/golden/test_math_golden/all_" + str(dtype) + ".pth")
    if (low != None and high != None):
        inp_cuda = inp_cuda.clamp(min=low, max=high)
    out_cuda = torch.rsqrt(inp_cuda)
    if save_tensor:
        torch.save(out_cuda, "./test_rsqrt_" + str(dtype) + ".pth")
    else:
        golden = torch.load("/netapp/pytorch/golden/test_math_golden/test_math_2/test_rsqrt_" + str(dtype) + ".pth")
        TestCase().assertEqual(golden, out_cuda, atol=0, rtol=rtol)


def test_pow(dtype, rtol, low=None, high=None):
    if is_gen_inp:
        inp_cuda = gen_inp(dtype)
    else:
        inp_cuda = torch.load("/netapp/pytorch/golden/test_math_golden/all_" + str(dtype) + ".pth")
    if (low != None and high != None):
        inp_cuda = inp_cuda.clamp(min=low, max=high)
    out_cuda = torch.pow(inp_cuda, 2.4)
    if save_tensor:
        torch.save(out_cuda, "./test_pow_" + str(dtype) + ".pth")
    else:
        golden = torch.load("/netapp/pytorch/golden/test_math_golden/test_math_2/test_pow_" + str(dtype) + ".pth")
        TestCase().assertEqual(golden, out_cuda, atol=0, rtol=rtol)


def test_exp(dtype, rtol, low=None, high=None):
    if is_gen_inp:
        inp_cuda = gen_inp(dtype)
    else:
        inp_cuda = torch.load("/netapp/pytorch/golden/test_math_golden/all_" + str(dtype) + ".pth")
    if (low != None and high != None):
        inp_cuda = inp_cuda.clamp(min=low, max=high)
    out_cuda = torch.exp(inp_cuda)
    if save_tensor:
        torch.save(out_cuda, "./test_exp_" + str(dtype) + ".pth")
    else:
        golden = torch.load("/netapp/pytorch/golden/test_math_golden/test_math_2/test_exp_" + str(dtype) + ".pth")
        TestCase().assertEqual(golden, out_cuda, atol=0, rtol=rtol)


# test bfloat16
test_log(torch.bfloat16, 0, 0, None)
test_tanh(torch.bfloat16, 0, -1, 1)
test_sqrt(torch.bfloat16, 0, 0, None)
test_rsqrt(torch.bfloat16, 0, 0, None)
test_pow(torch.bfloat16, 0, None, None)
test_exp(torch.bfloat16, 0, None, None)


# test half
test_log(torch.half, 0, 0, None)
test_erfinv(torch.half, 7e-4, -1, 1)
test_tanh(torch.half, 0, -1, 1)
test_sqrt(torch.half, 0, 0, None)
test_rsqrt(torch.half, 0, 0, None)
test_pow(torch.half, 0, None, None)
test_exp(torch.half, 0, None, None)


# test float
# test_log(torch.float, 0, 0, None)
# test_erfinv(torch.float, 0, -1, 1)
# test_tanh(torch.float, 0, -1, 1)
# test_sqrt(torch.float, 0, 0, None)
# test_rsqrt(torch.float, 0, 0, None)
# test_pow(torch.float, 0, None, None)
# test_exp(torch.float, 0, None, None)
