#!/usr/bin/env python
import torch
import torch.nn as nn
import sys
import os
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "{}/..".format(cur_dir))
from utils import RunCpuAndGpuTest, resnet_50_shapes, RESNET50_TEST_BATCHSIZE

dtype = torch.float32

m = nn.AdaptiveAvgPool2d(output_size=(1, 1))

for b in RESNET50_TEST_BATCHSIZE:
    for shape in resnet_50_shapes:
        shape = (b, ) + shape
        input = torch.rand(shape, dtype=dtype, requires_grad=True)

        if not RunCpuAndGpuTest(m, input, backward=True, loop=True):
            print("--------------------Error raise with shape:", shape)
            print("Failed: resnet50 {}".format(__file__))
            exit(1)

print("Passed: resnet50 {}".format(__file__))
exit(0)
