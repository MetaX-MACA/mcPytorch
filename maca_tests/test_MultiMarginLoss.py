import torch
import torch.nn as nn

m = nn.MultiMarginLoss()

x = torch.tensor([[0.1, 0.2, 0.4, 0.8]])
y = torch.tensor([3])
output_golden = m(x, y)

x_d = x.cuda()
y_d = y.cuda()
output_d = m(x_d, y_d)
output = output_d.cpu()

if torch.allclose(output, output_golden):
    exit(0)
else:
    exit(1)