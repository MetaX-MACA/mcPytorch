#!/usr/bin/env python
import torch
import torch.nn as nn
import sys
import copy
import os
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "{}/..".format(cur_dir))
from utils import check_close

torch.manual_seed(0)

MN = [[64, 32], [512, 768], [513, 769], [768, 512], [1024, 512]]

status = []
for mn in MN:
    input_g = torch.randn(1, *mn).requires_grad_(True)
    input = input_g.detach().clone().cuda().requires_grad_(True)
    b_input_g = torch.randn(1, *mn)
    b_input = b_input_g.detach().clone().cuda()

    m_g = nn.LayerNorm(normalized_shape=mn[-1], eps=1e-12, elementwise_affine=True)
    m = copy.deepcopy(m_g).cuda()

    out_g = m_g(input_g)
    out = m(input)

    x_status = check_close(out_g, out.cpu())

    out_g.backward(b_input_g)
    out.backward(b_input)

    dx_status = check_close(input_g.grad, input.grad.cpu())

    wg = m.weight.grad.cpu()
    w_status = check_close(m_g.weight.grad, wg)

    bg = m.bias.grad.cpu()
    b_status = check_close(m_g.bias.grad, bg)

    print(f"### x, dx, w, b: {x_status}, {dx_status}, {w_status}, {b_status}")
    if x_status and dx_status and w_status and b_status:
        status.append(True)
    else:
        status.append(False)

if sum(status) == len(MN):
    exit(0)
else:
    exit(1)
