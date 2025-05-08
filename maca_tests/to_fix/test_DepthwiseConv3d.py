import torch
import torch.nn as nn

m = nn.Conv3d(2, 4, 2, 2, groups=2)

input = torch.rand(1, 2, 8, 8, 8)
output_golden = m(input)

input_d = input.cuda()
m.cuda()
output_d = m(input_d)
output = output_d.cpu()

if torch.allclose(output, output_golden):
    print("Passed: {}".format(__file__))
    exit(0)
else:
    # 50% random fail in maca daily building environment
    print("Failed: {}".format(__file__))
    exit(1)