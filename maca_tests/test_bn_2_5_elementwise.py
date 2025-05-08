import torch
import torch.nn as nn
import os
os.environ["PYTORCH_DEFAULT_NLC"]="1"
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)


tensor_size = (256, 512, 720)
dtypes = [torch.float16, torch.bfloat16, torch.float32]
for dtype in dtypes:
    input = torch.randn(*tensor_size).to(dtype).cuda()
    batch_norm = torch.nn.BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    output_cpu = batch_norm(input.float().cpu())
    module = batch_norm.cuda()
    output = module(input)
    if dtype in [torch.float16, torch.bfloat16]:
        result = torch.allclose(output.cpu().float(), output_cpu, rtol=5e-3, atol=5e-3)
    else:
        result = torch.allclose(output.cpu().float(), output_cpu)
    if not result:
        exit(1)
