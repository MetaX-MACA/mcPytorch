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
for bs in DLRM_TEST_BATCHSIZE:
    input = torch.rand(bs, config_dlrm.layer_top[-1]).requires_grad_(True)
    func = nn.Sigmoid()
    result.append(RunCpuAndGpuTest(func, input, backward=True, loop=True))


sum_result = sum(result)
print("###test result: ", result)
if sum_result < len(result):
    print("Failed: {}".format(__file__))
    exit(1)
else:
    print("Passed: {}".format(__file__))
    exit(0)
