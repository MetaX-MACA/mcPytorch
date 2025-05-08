import torch
import torch.nn as nn
import os
import itertools
from utils import RunCpuAndGpuTest, perfModeEnvGuard

def test(perf_mode):
    if perf_mode:
        temp_env = "upsample_nearest2d"
    else:
        temp_env = ""
    with perfModeEnvGuard(temp_env):
        shapes = [(2, 2, 3, 4)]
        scale_factors = [2, 4.5]
        result = []

        for shape, scale_factor in itertools.product(shapes, scale_factors):
            input = torch.randn(shape, dtype=torch.float32)
            m = nn.Upsample(scale_factor=scale_factor, mode='nearest')
            result.append(RunCpuAndGpuTest(m, input, backward=True, loop=True))

        sum_result = sum(result)
        print("###", result)
        if sum_result < len(result):
            exit(1)

if __name__ == '__main__':
    test(True)
    test(False)