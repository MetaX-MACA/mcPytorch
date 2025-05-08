import torch
import torch.nn as nn

N, C, D, H, W = 1, 1, 2, 2, 8
# if _random_samples=None  "_random_samples = torch.rand(input.size(0), input.size(1), 3, dtype=input.dtype, 
# device=input.device)" which defined in functional.py will raise an error
kernel_size = (1, 1, 2)
output_size = (1, 1, 4)

random_samples = torch.rand(N, C, 3, dtype=torch.float32)
m = nn.FractionalMaxPool3d(kernel_size, output_size, return_indices=False, _random_samples=random_samples)

input = torch.rand(N, C, D, H, W)
output_golden = m(input)

random_samples_d = random_samples.cuda()
m_d = nn.FractionalMaxPool3d(kernel_size, output_size, return_indices=False, _random_samples=random_samples_d)
input_d = input.cuda()
output_d = m_d(input_d)
output = output_d.cpu()

if torch.allclose(output, output_golden):
    print("Passed: {}".format(__file__))
    exit(0)
else:
    print("Failed: {}".format(__file__))
    exit(1)