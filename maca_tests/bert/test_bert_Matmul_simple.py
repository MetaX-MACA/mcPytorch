#!/usr/bin/env python
import torch
import sys
import os
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "{}/..".format(cur_dir))
from utils import MMTest, BERT_TEST_BATCHSIZE

torch.set_printoptions(precision=5)

result = []
for bs in BERT_TEST_BATCHSIZE:
    result.append(MMTest(512, 512, 64, op=torch.matmul, batch1=2*bs, batch2=bs))

sum_result = sum(result)
print("###test result: ", result)
if sum_result < len(result):
    exit(1)
else:
    exit(0)
