#!/usr/bin/env python
import torch
import sys
from utils import RunCpuAndGpuTest

input1 = torch.rand(64)
input2 = torch.rand(64)

m = torch.dot

result = RunCpuAndGpuTest(m, input1, input2, backward=True, loop=True)

print("###test result: ", result)
if result:
    exit(0)
else:
    exit(1)
