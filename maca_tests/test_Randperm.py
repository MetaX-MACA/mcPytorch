#!/usr/bin/env python
import torch

seed = 0
torch.manual_seed(seed)

idx = torch.randperm(257, dtype=torch.float, device="cuda")

print("\n###### ", idx.cpu())
