#!/usr/bin/env python
import torch
import sys
import os
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "{}/..".format(cur_dir))
from utils import BERT_TEST_BATCHSIZE, check_close

torch.set_printoptions(precision=5)

result = []
for bs in BERT_TEST_BATCHSIZE:
    input11_g = torch.randn(bs, 12, 512, 512).requires_grad_(True)
    input12_g = torch.randn(bs, 1, 512, 64).requires_grad_(True)

    input11 = input11_g.detach().clone().cuda().requires_grad_(True)
    input12 = input12_g.detach().clone().cuda().requires_grad_(True)

    backward_input_g = torch.ones(1, 12, 512, 64)
    backward_input = torch.ones(1, 12, 512, 64).cuda()

    m = torch.matmul

    output_g = m(input11_g, input12_g)
    output = m(input11, input12)

    assert check_close(output_g, output.cpu()), "forward fail"

    output_g.backward(backward_input_g)
    output.backward(backward_input)

    b_g = input11_g.grad
    b = input11.grad

    print(f"backward golden:\n {b_g}")
    print(f"backward:\n {b.cpu()}")

    status = check_close(b.cpu(), b_g)
    print(f"### backward {status}")

if status:
    exit(0)
else:
    exit(1)
