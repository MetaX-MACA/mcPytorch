#!/usr/bin/env python
import torch
import torch.nn as nn
import sys
import os
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "{}/..".format(cur_dir))
from utils import MMTest, BERT_TEST_BATCHSIZE

result = []
for bs in BERT_TEST_BATCHSIZE:
    result.append(MMTest(bs, 768, 768, op=torch.nn.Linear, batch1=bs))

sum_result = sum(result)
print("###test result: ", result)
if sum_result < len(result):
    exit(1)
else:
    exit(0)
