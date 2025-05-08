import torch

# histc
input = torch.tensor([1, 2, 1], dtype=torch.float32)
output_golden = torch.histc(input, bins=4, min=0, max=3)

input_d = input.cuda()
output_d = torch.histc(input_d, bins=4, min=0, max=3)
output = output_d.cpu()

if not torch.allclose(output, output_golden):
    exit(1)

# bincount
input = torch.linspace(0, 1, steps=5, dtype=torch.int64)
output_golden = torch.bincount(input)

input_d = input.cuda()
output_d = torch.bincount(input_d)
output = output_d.cpu()

if torch.allclose(output, output_golden):
    exit(0)
else:
    exit(1)
