import torch

input = torch.rand(2, 3, dtype=torch.float32)

output_golden = torch.cat((input, input), 0)

input_d = input.cuda()
output_d = torch.cat((input_d, input_d), 0)
output = output_d.cpu()

if torch.allclose(output, output_golden):
    exit(0)
else:
    exit(1)
