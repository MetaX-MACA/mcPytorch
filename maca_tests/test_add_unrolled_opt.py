import torch
from itertools import product
dtypes_input0 = [torch.float, torch.half, torch.bfloat16]
dtypes_input1 = [torch.float, torch.half, torch.bfloat16]

for dtype_input0, dtype_input1 in product(dtypes_input0, dtypes_input1):
    for i in range(1024):
        input0 = torch.rand(i, dtype = dtype_input0, device = "cuda:0")
        input1 = torch.rand(i, dtype = dtype_input1, device = "cuda:0")
        result = input0 + input1
        result_cpu = input0.cpu() + input1.cpu()
        if not torch.allclose(result.cpu().float(), result_cpu.float()):
            exit(1)