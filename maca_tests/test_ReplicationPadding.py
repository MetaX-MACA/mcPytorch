import torch
import torch.nn as nn

# ReplicationPad1d
m = nn.ReplicationPad1d(2)

input = torch.rand(1, 2, 8)
output_golden = m(input)

input_d = input.cuda()
output_d = m(input_d)
output = output_d.cpu()

if not torch.allclose(output, output_golden):
    print("Failed: {}".format(__file__))
    exit(1)

# ReplicationPad2d
m = nn.ReplicationPad2d(2)

input = torch.rand(1, 2, 8, 8)
output_golden = m(input)

input_d = input.cuda()
output_d = m(input_d)
output = output_d.cpu()

if not torch.allclose(output, output_golden):
    print("Failed: {}".format(__file__))
    exit(1)

# ReplicationPad3d
m = nn.ReplicationPad3d(2)

input = torch.rand(1, 2, 8, 8, 8)
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