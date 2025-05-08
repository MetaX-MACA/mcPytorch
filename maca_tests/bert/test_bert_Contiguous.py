import torch
import sys
import os
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "{}/..".format(cur_dir))
from utils import RunCpuAndGpuTest, BERT_TEST_BATCHSIZE

result = []
for bs in BERT_TEST_BATCHSIZE:
    input = torch.rand(bs, 512, 12, 64)
    result.append(RunCpuAndGpuTest("contiguous", input, loop=True))

sum_result = sum(result)
print("###", result)
if sum_result < len(result):
    exit(1)
else:
    exit(0)
