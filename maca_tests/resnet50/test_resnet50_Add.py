#!/usr/bin/env python
import torch
import sys
import os
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "{}/..".format(cur_dir))
from utils import RunCpuAndGpuTest, resnet_50_shapes, RESNET50_TEST_BATCHSIZE

dtype = torch.float32

m = torch.add

for b in RESNET50_TEST_BATCHSIZE:
    for shape in resnet_50_shapes:
        shape1 = (b, ) + shape
        shape2 = (b, ) + shape
        input1 = torch.rand(shape1, dtype=dtype, requires_grad=True)
        input2 = torch.rand(shape2, dtype=dtype, requires_grad=True)

        if not RunCpuAndGpuTest(m, input1, input2, backward=True, loop=True):
            print("--------------------Error raise with shape", shape1)
            print("Failed: resnet50 {}".format(__file__))
            exit(1)

print("Passed: resnet50 {}".format(__file__))
