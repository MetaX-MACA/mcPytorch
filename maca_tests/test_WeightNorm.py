import torch
import torch.nn as nn

input = torch.rand(2, 8)
linear = nn.Linear(8, 16)
m = nn.utils.weight_norm(linear)
output_golden = m(input)

input_d = input.cuda()
m_d = m.cuda()
output_d = m_d(input_d)
output = output_d.cpu()

# keep the accuracy as CUDA
if torch.allclose(output, output_golden, rtol=1e-02, atol=1e-03):
    exit(0)
else:
    exit(1)
