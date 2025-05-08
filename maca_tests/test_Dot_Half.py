#!/usr/bin/env python
import torch
import sys
import os
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "{}/..".format(cur_dir))
from utils import MMTest, BERT_TEST_BATCHSIZE, check_close

torch.set_printoptions(precision=5, threshold=10000)
torch.manual_seed(0)

a = torch.rand(100, dtype=torch.float16)
a_g = a.cuda()
b = torch.rand(100, dtype=torch.float16)
b_g = b.cuda()

m = torch.dot

out = m(a.float(), b.float()).half()
out_g = m(a_g, b_g)

status = check_close(out_g.cpu().float(), out.float(), eps=1e-3)

# assert status, "### fail"
# print("### pass")

if status:
    exit(0)
else:
    exit(1)
