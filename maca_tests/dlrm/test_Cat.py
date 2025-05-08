import sys
import torch
import torch.nn as nn
import os
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append("{}/../../maca_samples/dlrm".format(cur_dir))
import configs.config_dlrm as config_dlrm

result = []
DLRM_TEST_BATCHSIZE = [1, 2, 4]
accept_eps = 1e-5


def test_cat(input1_cpu, input2_cpu):
    ret = True
    (batch_size, d) = input1_cpu.shape
    input1_gpu = input1_cpu.detach().clone().cuda().requires_grad_(True)
    input2_gpu = input2_cpu.detach().clone().cuda().requires_grad_(True)

    input_cpu = [input1_cpu] + [input2_cpu]
    input_gpu = [input1_gpu] + [input2_gpu]

    output_cpu = torch.cat(input_cpu, dim=1)
    output_gpu = torch.cat(input_gpu, dim=1)

    (output_cpu.sum()).backward()
    (output_gpu.sum()).backward()

    if abs(output_cpu - output_gpu.cpu()).sum() / output_cpu.numel() < accept_eps:
        ret = ret and True
    else:
        ret = ret and False

    if abs(input1_cpu.grad - input1_gpu.grad.cpu()).sum() / input1_cpu.grad.numel() < accept_eps:
        ret = ret and True
    else:
        ret = ret and False
    return ret

    if abs(input2_cpu.grad - input2_gpu.grad.cpu()).sum() / input2_cpu.grad.numel() < accept_eps:
        ret = ret and True
    else:
        ret = ret and False
    return ret


for bs in DLRM_TEST_BATCHSIZE:
    n = len(config_dlrm.embedding_num)
    input1_cpu = torch.rand(bs, config_dlrm.embedding_dim).requires_grad_(True)
    input2_cpu = torch.rand(bs, int((n + 1) * n / 2)).requires_grad_(True)
    result.append(test_cat(input1_cpu, input2_cpu))
    input1_cpu = torch.rand(bs, config_dlrm.embedding_dim).requires_grad_(True)
    input2_cpu = torch.rand(bs, config_dlrm.embedding_dim).requires_grad_(True)
    result.append(test_cat(input1_cpu, input2_cpu))


sum_result = sum(result)
print("###test result: ", result)
if sum_result < len(result):
    print("Failed: {}".format(__file__))
    exit(1)
else:
    print("Passed: {}".format(__file__))
    exit(0)
