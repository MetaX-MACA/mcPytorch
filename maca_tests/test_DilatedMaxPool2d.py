import torch
import torch.nn as nn

m = nn.MaxPool2d(3, 2, dilation=2)

input = torch.rand(1, 2, 8, 8)
output_golden = m(input)

input_d = input.cuda()
output_d = m(input_d)
output = output_d.cpu()

if torch.allclose(output, output_golden):
    print("Passed: {}".format(__file__))
    exit(0)
else:
    print("Failed: {}".format(__file__))
    exit(1)