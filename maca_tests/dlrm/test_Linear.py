import sys
import torch
import torch.nn as nn
import os
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append("{}/../".format(cur_dir))
sys.path.append("{}/../maca_tests".format(cur_dir))
from utils import RunCpuAndGpuTest
sys.path.append("{}/../../maca_samples/dlrm".format(cur_dir))
import configs.config_dlrm as config_dlrm


result = []
DLRM_TEST_BATCHSIZE = [1, 2, 4]
DLRM_LINEAR_SIZE = []
accept_eps = 1e-3


for bs in DLRM_TEST_BATCHSIZE:
    for i in range(len(config_dlrm.layer_bot)):
        if i is not len(config_dlrm.layer_bot) - 1:
            func = nn.Linear(config_dlrm.layer_bot[i], config_dlrm.layer_bot[i + 1])
        else:
            func = nn.Linear(config_dlrm.layer_bot[i], config_dlrm.embedding_dim)
        input = torch.rand(bs, config_dlrm.layer_bot[i]).requires_grad_(True)
        result.append(RunCpuAndGpuTest(func, input, backward=True, loop=True, ftol=accept_eps, btol=accept_eps))

    for i in range(len(config_dlrm.layer_top)):
        if i == 0:
            func = nn.Linear((len(config_dlrm.embedding_num) + 1) *
                             config_dlrm.embedding_dim, config_dlrm.layer_top[i])
            input = torch.rand(bs, (len(config_dlrm.embedding_num) + 1) *
                               config_dlrm.embedding_dim).requires_grad_(True)
            result.append(RunCpuAndGpuTest(func, input, backward=True, loop=True, ftol=accept_eps, btol=accept_eps))

            func = nn.Linear(int(len(config_dlrm.embedding_num) * (len(config_dlrm.embedding_num) +
                                                                   1) / 2) + config_dlrm.embedding_dim, config_dlrm.layer_top[i])
            input = torch.rand(bs, int(len(config_dlrm.embedding_num) * (len(config_dlrm.embedding_num) +
                                                                         1) / 2) + config_dlrm.embedding_dim).requires_grad_(True)
            print(input.shape)
            result.append(RunCpuAndGpuTest(func, input, backward=True, loop=True, ftol=accept_eps, btol=accept_eps))
        else:
            func = nn.Linear(config_dlrm.layer_top[i - 1], config_dlrm.layer_top[i])
            input = torch.rand(bs, config_dlrm.layer_top[i - 1]).requires_grad_(True)
            result.append(RunCpuAndGpuTest(func, input, backward=True, loop=True, ftol=accept_eps, btol=accept_eps))


sum_result = sum(result)
print("###test result: ", result)
if sum_result < len(result):
    print("Failed: {}".format(__file__))
    exit(1)
else:
    print("Passed: {}".format(__file__))
    exit(0)
