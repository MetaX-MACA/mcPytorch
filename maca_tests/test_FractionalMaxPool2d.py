import torch
import torch.nn as nn

N, C, H, W = 1, 2, 8, 8
# if _random_samples=None  "_random_samples = torch.rand(input.size(0), input.size(-2), 2, dtype=input.dtype, 
# device=input.device)" which defined in functional.py will raise an error
random_samples = torch.rand(N, C, 2, dtype=torch.float32)
m = nn.FractionalMaxPool2d(3, 4, return_indices=False, _random_samples=random_samples)

input = torch.rand(N, C, H, W)
output_golden = m(input)

random_samples_d = random_samples.cuda()
m_d = nn.FractionalMaxPool2d(3, 4, return_indices=False, _random_samples=random_samples_d)
input_d = input.cuda()
output_d = m_d(input_d)
output = output_d.cpu()

if torch.allclose(output, output_golden):
    print("Passed: {}".format(__file__))
    exit(0)
else:
    print("Failed: {}".format(__file__))
    exit(1)