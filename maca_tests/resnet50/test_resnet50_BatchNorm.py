#!/usr/bin/env python
import torch
import torch.nn as nn
import sys
import os
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "{}/..".format(cur_dir))
from utils import resnet_50_shapes, RESNET50_TEST_BATCHSIZE, RunCpuAndGpuTest

dtype = torch.float32

SHAPE_1 = [1]
SHAPE_2 = [(4,3,3)]
RESNET50_TEST_BATCHSIZE.extend(SHAPE_1)
resnet_50_shapes.extend(SHAPE_2)
def test_bn(perf_mode):
    if perf_mode:
        os.environ["MACA_TORCH_PERF_MODE"]="batch_norm"
    else:
        if os.getenv("MACA_TORCH_PERF_MODE"):
            del os.environ["MACA_TORCH_PERF_MODE"]
    for b in RESNET50_TEST_BATCHSIZE:
        for shape in resnet_50_shapes:
            c = shape[0]
            m = nn.BatchNorm2d(c)
            shape = (b, ) + shape
            input = torch.rand(shape, dtype=dtype) * 10
            input.requires_grad_(True)

            if not RunCpuAndGpuTest(m, input, backward=True, loop=True):
                print("--------------------Error raise with shape:", shape)
                print("Failed: resnet50 {}".format(__file__))
                exit(1)

if __name__ == "__main__":
    test_bn(False)
    test_bn(True)
    print("Passed: resnet50 {}".format(__file__))
