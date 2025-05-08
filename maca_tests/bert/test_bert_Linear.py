#!/usr/bin/env python
import torch
import torch.nn as nn
import sys
import os
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "{}/..".format(cur_dir))
from utils import RunCpuAndGpuTest, BERT_TEST_BATCHSIZE

torch.set_printoptions(precision=5)

result = []
for bs in BERT_TEST_BATCHSIZE:
    input1 = torch.rand(bs, 512, 768)
    input2 = torch.rand(bs, 512, 768)
    input3 = torch.rand(bs, 512, 3072)
    input4 = torch.rand(bs, 768)

    m1 = nn.Linear(768, 768)
    m2 = nn.Linear(768, 3072)
    m3 = nn.Linear(3072, 768)
    m4 = nn.Linear(768, 768)

    result.append(RunCpuAndGpuTest(m1, input1, backward=True, loop=True))
    result.append(RunCpuAndGpuTest(m2, input2, backward=True, loop=True))
    result.append(RunCpuAndGpuTest(m3, input3, backward=True, loop=True))
    result.append(RunCpuAndGpuTest(m4, input4, backward=True, loop=True))

sum_result = sum(result)
print("###test result: ", result)
if sum_result < len(result):
    exit(1)
else:
    exit(0)
