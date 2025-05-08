import torch
import torch.nn as nn

# pdist
input = torch.rand(2, 8)
output_golden = nn.functional.pdist(input, p=2)

input_d = input.cuda()
output_d = nn.functional.pdist(input_d, p=2)
output = output_d.cpu()

if not torch.allclose(output, output_golden):
    exit(1)

# cdist
a = torch.tensor([[0.9041,  0.0196], [-0.3108, -2.4423], [-0.4821,  1.059]])
b = torch.tensor([[-2.1763, -0.4713], [-0.6986,  1.3702]])
output_golden = torch.cdist(a, b, p=2)

a_d = a.cuda()
b_d = b.cuda()
output_d = torch.cdist(a_d, b_d, p=2)
output = output_d.cpu()

if torch.allclose(output, output_golden):
    print("Passed: {}".format(__file__))
    exit(0)
else:
    print("Failed: {}".format(__file__))
    exit(1)

