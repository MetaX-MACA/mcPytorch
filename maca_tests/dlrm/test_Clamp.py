import sys
import torch
import torch.nn as nn
import os
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append("{}/../../maca_samples/dlrm".format(cur_dir))
import configs.config_dlrm as config_dlrm


result = []
DLRM_TEST_BATCHSIZE = [1, 2, 4]
threshold_list = [0.1, 0.2, 0.3, 0.4, 0.5]
accept_eps = 1e-5


def test_clamp(input_cpu, threshold):
    ret = True
    input_gpu = input_cpu.detach().clone().cuda().requires_grad_(True)

    output_cpu = torch.clamp(input_cpu, min=threshold, max=(1.0 - threshold))
    output_gpu = torch.clamp(input_gpu, min=threshold, max=(1.0 - threshold))

    (output_cpu.sum()).backward()
    (output_gpu.sum()).backward()

    if abs(output_cpu - output_gpu.cpu()).sum() / output_cpu.numel() < accept_eps:
        ret = ret and True
    else:
        ret = ret and False

    if abs(input_cpu.grad - input_gpu.grad.cpu()).sum() / input_cpu.grad.numel() < accept_eps:
        ret = ret and True
    else:
        ret = ret and False
    return ret


for bs in DLRM_TEST_BATCHSIZE:
    for threshold in threshold_list:
        input_cpu = torch.rand(bs, config_dlrm.layer_top[-1]).requires_grad_(True)
        result.append(test_clamp(input_cpu, threshold))


sum_result = sum(result)
print("###test result: ", result)
if sum_result < len(result):
    print("Failed: {}".format(__file__))
    exit(1)
else:
    print("Passed: {}".format(__file__))
    exit(0)
