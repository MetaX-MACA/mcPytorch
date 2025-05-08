import torch
import sys
import os
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "{}/..".format(cur_dir))
from utils import RunCpuAndGpuTest, BERT_TEST_BATCHSIZE

result = []
for bs in BERT_TEST_BATCHSIZE:
    input = torch.rand(bs, 12, 512, 64)
    result.append(RunCpuAndGpuTest("transpose", input, -1, -2, loop=True))

sum_result = sum(result)
print("###", result)
if sum_result < len(result):
    print("Failed: {}".format(__file__))
    exit(1)
else:
    print("Passed: {}".format(__file__))
    exit(0)
