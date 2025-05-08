#!/usr/bin/env python
import torch
import torch.nn as nn
import sys
import os
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "{}/..".format(cur_dir))
from utils import RunCpuAndGpuTest
from utils import BERT_TEST_BATCHSIZE

result = []
for bs in BERT_TEST_BATCHSIZE:
    input = torch.rand(bs, 12, 512, 512)
    m = nn.Softmax(dim=-1)
    result.append(RunCpuAndGpuTest(m, input, backward=True, loop=True))

sum_result = sum(result)
print("###", result)
if sum_result < len(result):
    exit(1)
else:
    exit(0)
