import torch
import numpy as np
import random
import os

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True


seed = 31
print(f"seed: {seed}")

sizes = [2*8192*96*128]
dtypes = [torch.float, torch.half, torch.bfloat16]
iter = 10

golden1 = torch.tensor([31,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0], dtype=torch.uint8)
golden2 = torch.tensor([31,   0,   0,   0,   0,   0,   0,   0, 144,   3,   0,   0,   0,   0, 0,   0], dtype=torch.uint8)

for size in sizes:
    for dtype in dtypes:
        for _ in range(iter):
            # check rand
            setup_seed(seed)
            rs1 = torch.cuda.get_rng_state()
            golden = torch.rand(size, dtype=dtype, device="cuda")
            rs2 = torch.cuda.get_rng_state()
            
            if torch.sum(golden1-rs1[-16:]).item() != 0:
                exit(1)

print("### pass")
exit(0)