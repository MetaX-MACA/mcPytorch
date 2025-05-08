import torch
import torch.nn as nn
from torch.testing._internal.common_utils import TestCase

m = nn.MaxPool3d(3, stride=2)

input = torch.randn(20, 16, 50, 44).requires_grad_(True)
input_d = input.detach().clone().cuda().requires_grad_(True)

loss = nn.CrossEntropyLoss()
label = torch.ones((20, 24, 21), dtype= torch.int64)

output_golden = loss(m(input), label)
output_d = loss(m(input_d), label.cuda())
output = output_d.cpu()

bw_input = torch.randn(output.shape, dtype=output.dtype)

output_golden.backward()
output_d.backward()

TestCase().assertEqual(output_golden, output_d)
TestCase().assertEqual(input.grad, input_d.grad)
