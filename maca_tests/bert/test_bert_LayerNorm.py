#!/usr/bin/env python
import torch
import torch.nn as nn
import sys
import os
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "{}/..".format(cur_dir))
from utils import RunCpuAndGpuTest, BERT_TEST_BATCHSIZE


HWC = [[512, 768]]

result = []
for bs in BERT_TEST_BATCHSIZE:
    for hwc in HWC:
        shape = [bs, *hwc]
        input = torch.rand(bs, 512, 768)

        m = nn.LayerNorm(normalized_shape=768, elementwise_affine=True)
        result.append(RunCpuAndGpuTest(m, input, backward=True, loop=True))

sum_result = sum(result)
print("###", result)
if sum_result < len(result):
    exit(1)
else:
    exit(0)
