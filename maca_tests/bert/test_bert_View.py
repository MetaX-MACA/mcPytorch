#!/usr/bin/env python
import torch
import sys
import os
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "{}/..".format(cur_dir))
from utils import RunCpuAndGpuTest, BERT_TEST_BATCHSIZE

result = []
for bs in BERT_TEST_BATCHSIZE:
    input = torch.rand(bs, 512, 12, 64)
    input1 = torch.rand(bs, 512, 768)

    result.append(RunCpuAndGpuTest("view", input, bs, 512, 768, loop=True))
    result.append(RunCpuAndGpuTest("view", input1, bs, 512, 12, 64, loop=True))

sum_result = sum(result)
print("###", result)
if sum_result < len(result):
    print("Failed: {}".format(__file__))
    exit(1)
else:
    print("Passed: {}".format(__file__))
    exit(0)
