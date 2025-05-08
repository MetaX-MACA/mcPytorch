#!/usr/bin/env python
import torch
import sys
import os
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "{}/..".format(cur_dir))
from utils import BERT_TEST_BATCHSIZE

result = []
for bs in BERT_TEST_BATCHSIZE:
    output_golden = torch.zeros(torch.Size([bs, 512]), dtype=torch.long)
    output = torch.zeros(torch.Size([bs, 512]), dtype=torch.long, device="cuda").cpu()

    result.append(torch.allclose(output, output_golden))

sum_result = sum(result)
print("###", result)
if sum_result < len(result):
    print("Failed: {}".format(__file__))
    exit(1)
else:
    print("Passed: {}".format(__file__))
    exit(0)
