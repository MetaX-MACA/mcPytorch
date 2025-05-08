#!/usr/bin/env python
import torch

output_golden = torch.arange(512, device="cpu")

output = torch.arange(512, device="cuda").cpu()

print(output_golden)
print(output)


if torch.allclose(output, output_golden):
    exit(0)
else:
    exit(1)
