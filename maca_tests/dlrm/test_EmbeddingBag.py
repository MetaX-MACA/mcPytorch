import sys
import torch
import numpy as np
import torch.nn as nn
import os
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append("{}/../../maca_samples/dlrm".format(cur_dir))
import dlrm_module
import configs.config_dlrm as config_dlrm


DLRM_TEST_BATCHSIZE = [1, 2, 4]
eps = 1e-5
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
        if data_cpu.is_sparse:
            real_eps = (abs(data_cpu.cpu().to_dense() - data_gpu.cpu().to_dense()).sum()) / \
                (data_cpu.cpu().to_dense().numel())
        else:
            real_eps = (abs(data_cpu.cpu() - data_gpu.cpu()).sum()) / (data_cpu.cpu().numel())
        if real_eps < eps:
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
                if data_cpu[idx].is_sparse:
                    real_eps = (abs(data_cpu[idx].cpu().to_dense() - data_gpu[idx].cpu().to_dense()).sum()) / \
                        (data_cpu[idx].cpu().to_dense().numel())
                else:
                    real_eps = (abs(data_cpu[idx].cpu() - data_gpu[idx].cpu()).sum()) / \
                        (data_cpu[idx].cpu().numel())
                if real_eps < eps:
                    ret = ret and True
                else:
                    ret = ret and False
    return ret


if __name__ == "__main__":
    ret = True
    for bs in DLRM_TEST_BATCHSIZE:
        emb_l_cpu = nn.ModuleList()
        m = config_dlrm.embedding_dim
        ln = config_dlrm.embedding_num
        np.random.seed(0)
        for i in range(0, len(ln)):
            n = ln[i]
            EE = nn.EmbeddingBag(n, m, mode="sum", sparse=True)
            W = np.random.uniform(
                low=-np.sqrt(1 / n), high=np.sqrt(1 / n), size=(n, m)
            ).astype(np.float32)
            EE.weight.data = torch.tensor(W, requires_grad=True)
            EE.register_forward_hook(hook_forward_cpu)
            EE.register_backward_hook(hook_backward_cpu)
            emb_l_cpu.append(EE)

        emb_l_gpu = nn.ModuleList().cuda()
        m = config_dlrm.embedding_dim
        ln = config_dlrm.embedding_num
        np.random.seed(0)
        for i in range(0, len(ln)):
            n = ln[i]
            EE = nn.EmbeddingBag(n, m, mode="sum", sparse=True).cuda()
            W = np.random.uniform(
                low=-np.sqrt(1 / n), high=np.sqrt(1 / n), size=(n, m)
            ).astype(np.float32)
            EE.weight.data = torch.tensor(W, requires_grad=True).cuda()
            EE.register_forward_hook(hook_forward_gpu)
            EE.register_backward_hook(hook_backward_gpu)
            emb_l_gpu.append(EE)

        _, input_sparse_offset_cpu, input_sparse_cpu, _ = dlrm_module.create_input_data(bs)
        input_sparse_gpu = [element.cuda() for element in input_sparse_cpu]
        input_sparse_offset_gpu = input_sparse_offset_cpu.detach().clone().cuda()

        for k, sparse_index_group_batch in enumerate(input_sparse_cpu):
            E_cpu = emb_l_cpu[k]
            V_cpu = E_cpu(
                input_sparse_cpu[k],
                input_sparse_offset_cpu[k],
                per_sample_weights=None,
            )
            E_gpu = emb_l_gpu[k]
            V_gpu = E_gpu(
                input_sparse_gpu[k],
                input_sparse_offset_gpu[k],
                per_sample_weights=None,
            )

            (V_cpu.sum()).backward()
            (V_gpu.sum()).backward()

            ret = ret and checkout_error(input_fwd_cpu, input_fwd_gpu)
            ret = ret and checkout_error(output_fwd_cpu, output_fwd_gpu)
            ret = ret and checkout_error(input_bwd_cpu, input_bwd_gpu)
            ret = ret and checkout_error(output_bwd_cpu, output_bwd_gpu)

    print(input_fwd_gpu)
    print(input_fwd_cpu)
    print(output_fwd_gpu)
    print(output_fwd_cpu)
    print(input_bwd_gpu)
    print(input_bwd_cpu)
    print(output_bwd_gpu)
    print(output_bwd_cpu)
    if ret:
        print("Passed: {}".format(__file__))
        exit(0)
    else:
        print("Failed: {}".format(__file__))
        exit(1)
