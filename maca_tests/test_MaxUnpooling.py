import torch
import torch.nn as nn

pool = nn.MaxPool2d(2, stride=2, return_indices=True)
unpool = nn.MaxUnpool2d(2, stride=2)

input = torch.rand(1, 2, 8, 8)
output, indices = pool(input)
output_cpu = unpool(output, indices)

input_d = input.cuda()
output_d, indices_d = pool(input_d)
output_gpu = unpool(output_d, indices_d)
output_gpu = output_gpu.cpu()

if torch.allclose(output_cpu, output_gpu):
    print("Passed: {}".format(__file__))
    exit(0)
else:
    print("Failed: {}".format(__file__))
    exit(1)