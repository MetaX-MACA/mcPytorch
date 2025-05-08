import torch
import torch.nn as nn

input = torch.rand(2, 8, 8)
output_golden = input.resize(2, 4, 16)

input_d = input.cuda()
output_d = input_d.resize(2, 4, 16)
output = output_d.cpu()

if torch.allclose(output, output_golden):
    print("Passed: {}".format(__file__))
    exit(0)
else:
    print("Failed: {}".format(__file__))
    exit(1)