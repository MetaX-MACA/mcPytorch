#!/usr/bin/env python
import torch
import torch.nn as nn
import sys
import os
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "{}/..".format(cur_dir))

gold_path = r"/netapp/pytorch/golden/golden_silu/a100/"

# inplace=False
m = nn.SiLU(inplace=False)
dtypes = [torch.float32, torch.float16, torch.bfloat16]

for dtype in dtypes:
    inp = torch.load(gold_path+str(dtype)+"_input.pt")
    golden = torch.load(gold_path+str(dtype)+"_output.pt")
    out = m(inp)
    if not torch.allclose(golden, out):
        exit(1)

print("### pass")
exit(0)