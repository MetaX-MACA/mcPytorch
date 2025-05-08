import sys
import torch
import torch.nn as nn
import os
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append("{}/../../maca_samples/dlrm".format(cur_dir))
import configs.config_dlrm as config_dlrm


DLRM_TEST_BATCHSIZE = [1, 2, 4]
accept_eps = 1e-5
input_fwd_cpu = 0
output_fwd_cpu = 0
input_fwd_gpu = 0
output_fwd_gpu = 0
input_bwd_cpu = 0
output_bwd_cpu = 0
input_bwd_gpu = 0
output_bwd_gpu = 0


def hook_forward_cpu(module, input, output):
    global input_fwd_cpu
    input_fwd_cpu = input
    global output_fwd_cpu
    output_fwd_cpu = output


def hook_backward_cpu(module, input, output):
    global input_bwd_cpu
    input_bwd_cpu = input
    global output_bwd_cpu
    output_bwd_cpu = output


def hook_forward_gpu(module, input, output):
    global input_fwd_gpu
    input_fwd_gpu = input
    global output_fwd_gpu
    output_fwd_gpu = output


def hook_backward_gpu(module, input, output):
    global input_bwd_gpu
    input_bwd_gpu = input
    global output_bwd_gpu
    output_bwd_gpu = output


def checkout_error(data_cpu, data_gpu):
    ret = True
    if isinstance(data_cpu, torch.Tensor):
        real_eps = (abs(data_cpu.cpu() - data_gpu.cpu()).sum()) / (data_cpu.cpu().numel())
        if real_eps < accept_eps:
            ret = ret and True
        else:
            ret = ret and False
    else:
        for idx in range(len(data_cpu)):
            if data_cpu[idx] is None and data_gpu[idx] is None:
                ret = ret and True
            elif data_cpu[idx] is None and data_gpu[idx] is not None:
                ret = ret and False
            elif data_cpu[idx] is not None and data_gpu[idx] is None:
                ret = ret and False
            else:
                real_eps = (abs(data_cpu[idx].cpu() - data_gpu[idx].cpu()).sum()) / \
                    (data_cpu[idx].cpu().numel())
                if real_eps < accept_eps:
                    ret = ret and True
                else:
                    ret = ret and False
    return ret


if __name__ == "__main__":
    ret = True
    for bs in DLRM_TEST_BATCHSIZE:
        input1_cpu = torch.rand(bs, config_dlrm.layer_top[-1]).requires_grad_(True)
        input2_cpu = torch.rand(bs, config_dlrm.layer_top[-1]).requires_grad_(False)
        func_cpu = nn.BCELoss(reduction="mean")
        func_cpu.register_forward_hook(hook_forward_cpu)
        func_cpu.register_backward_hook(hook_backward_cpu)

        input1_gpu = input1_cpu.detach().clone().cuda().requires_grad_(True)
        input2_gpu = input2_cpu.detach().clone().cuda().requires_grad_(False)
        func_gpu = nn.BCELoss(reduction="mean").cuda()
        func_gpu.register_forward_hook(hook_forward_gpu)
        func_gpu.register_backward_hook(hook_backward_gpu)

        output_cpu = func_cpu(input1_cpu, input2_cpu)
        output_gpu = func_gpu(input1_gpu, input2_gpu)

        output_cpu.backward()
        output_gpu.backward()

        ret = ret and checkout_error(input_fwd_cpu, input_fwd_gpu)
        ret = ret and checkout_error(output_fwd_cpu, output_fwd_gpu)
        ret = ret and checkout_error(input_bwd_cpu, input_bwd_gpu)
        ret = ret and checkout_error(output_bwd_cpu, output_bwd_gpu)

    if ret:
        print("Passed: {}".format(__file__))
        exit(0)
    else:
        print("Failed: {}".format(__file__))
        exit(1)
