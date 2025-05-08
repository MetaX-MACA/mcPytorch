import torch
import numpy as np
import random


def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True


# seed = random.uniform(0, 100)
seed = 31
print(f"seed: {seed}")

sizes = [8192*96*128, 2*8192*96*128, 4*8192*96*128]
dtypes = [torch.float, torch.half, torch.bfloat16]
iter = 10

for size in sizes:
    for dtype in dtypes:
        for _ in range(iter):
             # check rand
            setup_seed(seed)
            golden = torch.rand(size, dtype=dtype, device="cuda")
            torch.save(golden, "a100_rand_"+str(size)+"_"+str(dtype)+".pt")

            setup_seed(seed)
            out1 = torch.rand(size, dtype=dtype, device="cuda")
            if torch.sum(torch.abs(out1-golden)).item() != 0:
                exit(1)

            # check randn
            setup_seed(seed)
            golden = torch.randn(size, dtype=dtype, device="cuda")*0.02
            torch.save(golden, "a100_randn_"+str(size)+"_"+str(dtype)+".pt")

            setup_seed(seed)
            out1 = torch.randn(size, dtype=dtype, device="cuda")*0.02
            if torch.sum(torch.abs(out1-golden)).item() != 0:
                exit(1)

            # check normal_
            setup_seed(seed)
            golden = torch.empty(size, dtype=dtype, device="cuda")
            torch.nn.init.normal_(golden, mean=0.0, std=0.02)
            setup_seed(seed)
            out1 = torch.empty(size, dtype=dtype, device="cuda")
            torch.nn.init.normal_(out1, mean=0.0, std=0.02)
            if torch.sum(torch.abs(out1-golden)).item() != 0:
                exit(1)

print("### test pass")
exit(0)