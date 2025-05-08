#!/usr/bin/env python
import torch
import sys
from utils import RunCpuAndGpuTest

input1 = torch.tensor((1 +2j, 3 - 1j))
input2 = torch.tensor((2 +1j, 4 - 0j))

m = torch.vdot
result = m(input1.cuda(), input2.cuda())
result_g = m(input1, input2)

print("###test result: ", result)
if result == result_g:
    exit(0)
else:
    exit(1)
