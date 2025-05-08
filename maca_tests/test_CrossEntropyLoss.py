import torch
import torch.nn as nn
import numpy as np
from utils import RunCpuAndGpuTest

device = "cuda"

model = nn.CrossEntropyLoss()

input_g = torch.randn(1,2).requires_grad_(True)
input = input_g.detach().clone().cuda().requires_grad_(True)

label = torch.ones((1), dtype= torch.int64)

out = model(input, label.to(device))
out_g = model(input_g, label)

bw_input = torch.randn(out_g.shape, dtype=out_g.dtype)

out.backward(bw_input.to(device))
out_g.backward(bw_input)

f_status =  torch.allclose(out_g, out.cpu())
b_status = torch.allclose(input_g.grad, input.grad.cpu())

if f_status and b_status:
    exit(0)
else:
    exit(1)


