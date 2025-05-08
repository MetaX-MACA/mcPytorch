import torch


def test_half_to_float():
    input_cpu = torch.ones(512, 512, device="cpu", dtype=torch.half) * 0.1
    output_cpu = input_cpu.to(torch.float)
    input_cuda = torch.ones(512, 512, device="cuda", dtype=torch.half) * 0.1
    output_cuda = input_cuda.to(torch.float)
    return output_cpu.sum() == output_cuda.sum()

def test_float_to_half():
    input_cpu = torch.ones(512, 512, device="cpu", dtype=torch.float) * 0.1
    output_cpu = input_cpu.to(torch.half)
    input_cuda = torch.ones(512, 512, device="cuda", dtype=torch.float) * 0.1
    output_cuda = input_cuda.to(torch.half)
    return output_cpu.sum() == output_cuda.sum()

def test_bfloat_to_float():
    input_cpu = torch.ones(512, 512, device="cpu", dtype=torch.bfloat16) * 0.1
    output_cpu = input_cpu.to(torch.float)
    input_cuda = torch.ones(512, 512, device="cuda", dtype=torch.bfloat16) * 0.1
    output_cuda = input_cuda.to(torch.float)
    return output_cpu.sum() == output_cuda.sum()

def test_float_to_bfloat():
    input_cpu = torch.ones(512, 512, device="cpu", dtype=torch.float) * 0.1
    output_cpu = input_cpu.to(torch.bfloat16)
    input_cuda = torch.ones(512, 512, device="cpu", dtype=torch.float) * 0.1
    output_cuda = input_cuda.to(torch.bfloat16)
    return output_cpu.sum() == output_cuda.sum()

def test_half_to_float_case0():
    input_cpu = torch.ones(2048, 99999, device="cpu", dtype=torch.float) * 0.1
    output_cpu = input_cpu[1:2, 0:1024].to(torch.float)
    input_cuda = torch.ones(2048, 99999, device="cpu", dtype=torch.float) * 0.1
    output_cuda = input_cuda[1:2, 0:1024].to(torch.float)
    return output_cpu.sum() == output_cuda.sum()

def test_half_to_float_case1():
    input_cpu = torch.ones(2147483645, device="cpu", dtype=torch.float) * 0.1
    output_cpu = input_cpu.to(torch.float)
    input_cuda = torch.ones(2147483645, device="cpu", dtype=torch.float) * 0.1
    output_cuda = input_cuda.to(torch.float)
    return output_cpu.sum() == output_cuda.sum()

ret = True
ret = ret and test_half_to_float()
ret = ret and test_float_to_half()
ret = ret and test_bfloat_to_float()
ret = ret and test_float_to_bfloat()
ret = ret and test_half_to_float_case0()
ret = ret and test_half_to_float_case1()


print("###test result: ", ret)
if ret:
    print("test passed!")
    exit(0)
else:
    print("test failed!")
    exit(1)