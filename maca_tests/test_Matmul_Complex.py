#!/usr/bin/env python
import torch
import sys
from utils import RunCpuAndGpuTest

in1 = [1 +2j, 3 - 1j, 1 +2j, 3 - 1j] * 8 * 32
in2 = [2 +1j, 4 - 0j, 2 +1j, 4 - 0j] * 8 * 32
input1 = torch.tensor(in1).reshape(32, 32)
input2 = torch.tensor(in2).reshape(32, 32)

m = torch.matmul
result = m(input1.cuda(), input2.cuda()).cpu()
result = result.reshape(32*32)
result_g = m(input1, input2)
result_g = result_g.reshape(32*32)

status = True
for i in range(result.shape.numel()):
    if result[i] != result_g[i]:
        status = False
        break

if status:
    exit(0)
else:
    exit(1)
