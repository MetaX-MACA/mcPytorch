import torch
import torch.nn as nn

m = nn.Upsample(scale_factor=2, mode='nearest')

input = torch.arange(1, 5, dtype=torch.float32).view(1, 1, 1, 2, 2)
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