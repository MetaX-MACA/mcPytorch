#!/usr/bin/env python
import torch
import torch.nn as nn

dtype = torch.float32

m = nn.Linear(20, 30).cuda()

input = torch.randn(128, 20).cuda()

output = m(input)
