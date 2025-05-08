import torch
import torch.nn as nn
from torch.testing._internal.common_utils import TestCase

m = nn.AvgPool2d(4)

input = torch.rand(20, 16, 50, 44).requires_grad_(True)
input_d = input.detach().clone().cuda().requires_grad_(True)

loss = nn.CrossEntropyLoss()
label = torch.ones((20, 16, 12, 11))

output_golden = loss(m(input), label)
output_d = loss(m(input_d), label.cuda())
output = output_d.cpu()

output_golden.backward()
output_d.backward()

TestCase().assertEqual(output_golden, output_d)
TestCase().assertEqual(input.grad, input_d.grad)
