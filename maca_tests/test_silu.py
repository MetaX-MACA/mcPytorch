#!/usr/bin/env python
import torch
import torch.nn as nn
import sys
import os
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "{}/..".format(cur_dir))
from utils import RunCpuAndGpuTest

dtype = torch.float32
# inplace=False
m = nn.SiLU(inplace=False)

yolov5_test_bs = [1, 2]
yolov5_shapes = [(64, 160, 160), (128, 80, 80), (256, 40, 40), (512, 20, 20), (1024, 10, 10)]

for b in yolov5_test_bs:
    for shape in yolov5_shapes:
        shape = (b, ) + shape
        input = torch.rand(shape, dtype=dtype, requires_grad=True)

        if not RunCpuAndGpuTest(m, input, backward=True, loop=True):
            print("--------------------SiLU test error raise with shape:", shape)
            exit(1)
