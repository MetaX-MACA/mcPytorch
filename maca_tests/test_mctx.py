#!/usr/bin/env python
import torch
import os

ret = False

torch.cuda.nvtx.range_push("foo")
torch.cuda.nvtx.mark("bar")
torch.cuda.nvtx.range_pop()
range_handle = torch.cuda.nvtx.range_start("range_start")
torch.cuda.nvtx.range_end(range_handle)

ret = True

if ret:
    exit(0)
else:
    exit(1)