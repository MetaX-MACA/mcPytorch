import torch

# median
input = torch.randn(1, 3)

output_golden = torch.median(input)

input_d = input.cuda()
output_d = torch.median(input_d)
output = output_d.cpu()

if not torch.allclose(output, output_golden):
    exit(1)

# nanmedian
input = torch.tensor([1, float('nan'), 3, 2])

output_golden = torch.nanmedian(input)

input_d = input.cuda()
output_d = torch.nanmedian(input_d)
output = output_d.cpu()

if torch.allclose(output, output_golden):
    exit(0)
else:
    exit(1)