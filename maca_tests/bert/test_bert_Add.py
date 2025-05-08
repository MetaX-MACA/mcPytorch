#!/usr/bin/env python
import torch
import sys
import os
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "{}/..".format(cur_dir))
from utils import RunCpuAndGpuTest, BERT_TEST_BATCHSIZE

torch.set_printoptions(precision=5)

result = []
for bs in BERT_TEST_BATCHSIZE:
    input11 = torch.rand(bs, 12, 512, 512)
    input12 = torch.rand(bs, 1, 1, 512)
    input21 = torch.rand(bs, 512, 768)
    input22 = torch.rand(bs, 512, 768)

    m = torch.add

    result.append(RunCpuAndGpuTest(m, input11, input12, backward=True, loop=True))
    result.append(RunCpuAndGpuTest(m, input21, input22, backward=True, loop=True))

sum_result = sum(result)
print("###test result: ", result)
if sum_result < len(result):
    print("Failed: {}".format(__file__))
    exit(1)
else:
    print("Passed: {}".format(__file__))
    exit(0)
