import torch
import torch.nn as nn
from torch.testing._internal.common_utils import TestCase


for case_size in ([64, 128, 150, 150], [64, 256, 75, 75], [64, 512, 38, 38]):
    m = torch.nn.MaxPool2d(2, 2, 0, 1, True, False)
    input = torch.rand(case_size).requires_grad_(True)
    input_d = input.detach().clone().cuda().requires_grad_(True)
    output_golden = m(input)
    (output_golden[0] + output_golden[1]).sum().backward()
    output_d = m(input_d)
    (output_d[0] + output_d[1]).sum().backward()
    TestCase().assertEqual(output_golden, output_d)
    TestCase().assertEqual(input.grad, input_d.grad)


m = torch.nn.MaxPool2d(3, 1, 1, 1, False, False)
input = torch.rand(64, 512, 19, 19).requires_grad_(True)
input_d = input.detach().clone().cuda().requires_grad_(True)
output_golden = m(input)
(output_golden[0] + output_golden[1]).sum().backward()
output_d = m(input_d)
(output_d[0] + output_d[1]).sum().backward()
TestCase().assertEqual(output_golden, output_d)
TestCase().assertEqual(input.grad, input_d.grad)