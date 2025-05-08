#!/usr/bin/env python
import torch
import torch.nn as nn
from utils import RunCpuAndGpuTest

m = nn.AdaptiveAvgPool2d(4)
input = torch.rand(2, 8, 8)

result = []
result.append(RunCpuAndGpuTest(m, input, loop=True))
sum_result = sum(result)
if sum_result < len(result):
    print("Failed: {}".format(__file__))
    exit(1)
else:
    print("Passed: {}".format(__file__))
    exit(0)
