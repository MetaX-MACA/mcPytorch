import torch

input = torch.tensor([1, 2, 3])
output_golden = torch.repeat_interleave(input, 2)

input_d = input.cuda()
output_d = torch.repeat_interleave(input_d, 2)
output = output_d.cpu()

if torch.allclose(output, output_golden):
    exit(0)
else:
    exit(1)